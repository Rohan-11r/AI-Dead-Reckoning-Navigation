#!/usr/bin/env python3
"""Phase 4 real-data baseline: classical INS + EKF on IO-VNBD validation drives.

Phone-only inputs (CAN/VBOX are NEVER read by the filter -- they score it):
* IMU: synced phone accelerometer + vertical gyro via ``IOVNBD_AXIS_MAP``
  (horizontal gyro columns are not angular rates: reports/phase4/gyro_axis_resolution.json)
* GNSS: phone fixes, applied at their EPOCH and only after they were received
* Stillness (phone-only: low accel variance, |f| ~ g, no yaw rate, last fix < 1 m/s)
  -> ZUPT + accelerometer levelling
* Recovery: 3 consecutive rejected fixes -> reset to GNSS (navcore.recovery)
* Heading: mount yaw from phone GNSS motion where observable, else an 8-hypothesis
  coarse alignment over the first 120 s (lowest mean NIS wins)
* Initial attitude: gravity levelling + mount yaw from phone GNSS motion
  (navcore.alignment); initial position/velocity from the first moving fix

Scored against VBOX (evaluation only): horizontal error while aided, horizontal drift at
the end of simulated GNSS outages (30 s / 60 s, fixes withheld), and whether the filter's
own 1-sigma covers its error.

Scoring is only meaningful where the phone clock was VERIFIED against VBOX in Phase 2
(``alignment_verified``): in unverified sessions a ~1 s residual clock error at 15 m/s
puts the truth lookup ~15 m off (Vtb2: phone fixes 18.6 m "wrong" vs 2.8 m in verified
S3a). Default set = validation split AND verified. ``--all-verified`` adds verified
TRAIN sessions (legitimate here: nothing in this classical filter is fitted to them;
noise comes from the stationary Vw1 Allan analysis). The held-out TEST drivers are never
touched in Phase 4.

    .venv/Scripts/python.exe scripts/evaluate/phase4_baseline.py [--sessions S3a,Vtb2]
    -> reports/phase4/baseline_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from navcore.alignment.mounting import estimate_mounting  # noqa: E402
from navcore.filtering.ekf import CovarianceError, ErrorStateEKF, NoiseParams  # noqa: E402
from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius  # noqa: E402
from navcore.geometry.quaternion import dcm_to_q, q_to_euler  # noqa: E402
from navcore.geometry.rotation import rot_z  # noqa: E402
from navcore.ins.mechanization import NavState  # noqa: E402
from navcore.recovery.gnss_reset import ConsecutiveRejectionReset  # noqa: E402
from navcore.sensors.samples import IOVNBD_AXIS_MAP, GnssSample  # noqa: E402
from training.preprocessing.columns import assert_model_inputs  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
OUT = CFG.reports_dir / "phase4" / "baseline_evaluation.json"
NOISE_REPORT = CFG.reports_dir / "phase4" / "imu_noise.json"
INPUTS = ["acc_x_mps2", "acc_y_mps2", "acc_z_mps2", "gyro_x_radps", "gyro_y_radps", "gyro_z_radps",
          "ph_gnss_lat_deg", "ph_gnss_lon_deg", "ph_gnss_alt_m", "ph_gnss_speed_mps",
          "ph_gnss_accuracy_m", "ph_gnss_bearing_deg", "ph_gnss_new_fix", "ph_gnss_epoch_utc"]
OUTAGES_S = (30.0, 60.0)
OUTAGE_EVERY_S = 240.0  # one outage start every 4 min, alternating lengths
STILL_WINDOW = 10  # 1 s
STILL_ACC_STD = 0.08  # m/s^2
STILL_GYRO_ABS = 0.02  # rad/s, vertical gyro
STILL_NORM_TOL = 0.1  # | |f| - g | [m/s^2]: rejects steady turns (centripetal accel)
STILL_GNSS_SPEED = 1.0  # m/s, latest received phone fix
MH_WINDOW_S = 120.0  # multi-hypothesis coarse-alignment window


def horiz_err_m(lat, lon, lat_t, lon_t) -> float:
    return math.hypot(*err_en_m(lat, lon, lat_t, lon_t))


def err_en_m(lat, lon, lat_t, lon_t) -> tuple[float, float]:
    rm = float(meridian_radius(lat_t))
    rn = float(prime_vertical_radius(lat_t))
    return (lon - lon_t) * rn * math.cos(lat_t), (lat - lat_t) * rm


def along_cross(de: float, dn: float, heading_deg: float) -> tuple[float, float]:
    """Split an E/N error into along-track and cross-track (right of travel +)."""
    h = math.radians(heading_deg)
    return de * math.sin(h) + dn * math.cos(h), de * math.cos(h) - dn * math.sin(h)


MIN_SEGMENT_S = 300.0  # segments shorter than this are not scored


def run_session(sid: str, noise: NoiseParams) -> dict:
    """Run every continuous segment (Phase 3 ``segment_id``: real dt > 1 s breaks) with its
    own initialisation, and pool the per-segment errors into one session result."""
    assert_model_inputs(INPUTS)  # CAN rule, mechanically
    full = pd.read_parquet(CFG.synced_root / f"{sid}.parquet")
    segs, raw = [], {"aided_err": [], "aided_sig": [], "outage_end": {}, "outage_ac": {}}
    for seg_id, d in full.groupby("segment_id", sort=True):
        dur = (d["t_utc"].iloc[-1] - d["t_utc"].iloc[0]).total_seconds()
        if dur < MIN_SEGMENT_S:
            segs.append({"segment_id": int(seg_id), "skipped": f"shorter than {MIN_SEGMENT_S:.0f} s"})
            continue
        r = run_segment(sid, d.reset_index(drop=True), noise)
        r["segment_id"] = int(seg_id)
        rr = r.pop("_raw", None)
        segs.append(r)
        if rr:
            raw["aided_err"] += rr["aided_err"]
            raw["aided_sig"] += rr["aided_sig"]
            for k in ("outage_end", "outage_ac"):
                for o, v in rr[k].items():
                    raw[k].setdefault(o, []).extend(v)
    scored = [x for x in segs if x.get("aided_horizontal_err_m")]
    if not scored:
        return {"session_id": sid, "skipped": "no scoreable segment", "segments": segs}
    ae, sg = np.array(raw["aided_err"]), np.array(raw["aided_sig"])
    return {
        "session_id": sid, "segments": segs, "n_segments_scored": len(scored),
        "aided_horizontal_err_m": {"p50": float(np.median(ae)), "p95": float(np.percentile(ae, 95)), "n": len(ae)},
        "aided_err_within_2sigma_fraction": float(np.mean(ae <= 2 * sg)),
        "outage_end_horizontal_err_m": {
            str(o): {"p50": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": len(v)} if v else None
            for o, v in raw["outage_end"].items()},
        "outage_end_along_cross_abs_median_m": {
            str(o): [float(np.median(np.abs([x[0] for x in v]))), float(np.median(np.abs([x[1] for x in v])))]
            if v else None for o, v in raw["outage_ac"].items()},
        "gnss_lockout_resets": sum(x.get("gnss_lockout_resets", 0) for x in scored),
        "runtime_s": round(sum(x.get("runtime_s") or 0 for x in scored), 1),
    }


def run_segment(sid: str, d: pd.DataFrame, noise: NoiseParams) -> dict:
    truth = d[["gt_lat_deg", "gt_lon_deg", "gt_heading_deg", "gt_valid"]]  # scoring only
    d = d[[*INPUTS, "t_utc", "dt_s", "segment_id"]]
    t0 = d["t_utc"].iloc[0]
    t = (d["t_utc"] - t0).dt.total_seconds().to_numpy()
    acc = d[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy()
    gyr = d[["gyro_x_radps", "gyro_y_radps", "gyro_z_radps"]].to_numpy()

    fx = d[d["ph_gnss_new_fix"] & d["ph_gnss_epoch_utc"].notna() & d["ph_gnss_accuracy_m"].gt(0)]
    fixes, n_bad_velocity = [], 0
    for r in fx.itertuples():
        # 6 of 14,208 phone fixes (all Vtb3) log a negative speed: keep the position, drop
        # the impossible velocity, and count it
        vel_ok = (np.isfinite(r.ph_gnss_speed_mps) and np.isfinite(r.ph_gnss_bearing_deg)
                  and r.ph_gnss_speed_mps >= 0)
        n_bad_velocity += not vel_ok
        fixes.append(GnssSample(
            t_s=(r.ph_gnss_epoch_utc - t0).total_seconds(), t_received_s=(r.t_utc - t0).total_seconds(),
            lat_rad=math.radians(r.ph_gnss_lat_deg), lon_rad=math.radians(r.ph_gnss_lon_deg),
            h_m=r.ph_gnss_alt_m, horizontal_accuracy_m=r.ph_gnss_accuracy_m,
            speed_mps=r.ph_gnss_speed_mps if vel_ok else None,
            bearing_rad=math.radians(r.ph_gnss_bearing_deg) if vel_ok else None))
    if len(fixes) < 20:
        return {"session_id": sid, "skipped": "fewer than 20 usable phone fixes"}

    # --- phone-only alignment for the initial attitude
    vf = [f for f in fixes if f.speed_mps is not None]
    ft = np.array([f.t_s for f in vf])
    interval = float(np.median(np.diff(ft)))
    mount = estimate_mounting(acc, t, ft, np.array([f.speed_mps for f in vf]),
                              np.array([f.bearing_rad for f in vf]),
                              max_fix_gap_s=2.5 if interval <= 2 else 12.0)
    k0 = next((i for i, f in enumerate(fixes) if f.speed_mps is not None and f.speed_mps > 5.0), None)
    if k0 is None:
        return {"session_id": sid, "skipped": "vehicle never above 5 m/s on a phone fix"}
    f0 = fixes[k0]
    i0 = int(np.searchsorted(t, f0.t_received_s))
    # ---- initial state: GNSS position/velocity, gravity-levelled tilt, heading from the
    # GNSS course through the mount. Where the mount yaw is NOT observable (most 9 s-fix
    # sessions) a multi-hypothesis coarse alignment runs 8 yaw offsets for the first
    # MH_WINDOW_S and keeps the one with the lowest mean (capped) NIS -- phone data only.
    veh_yaw = math.pi / 2 - f0.bearing_rad
    v0 = f0.velocity_enu()
    mount_ok = mount.acceptable()
    offsets = [0.0] if mount_ok else [k * math.pi / 4 for k in range(8)]

    def make_filter(offset: float) -> ErrorStateEKF:
        R_nb = rot_z(veh_yaw + offset) @ mount.R_vb
        nominal = NavState(t_s=t[i0], lat_rad=f0.lat_rad, lon_rad=f0.lon_rad, h_m=f0.h_m,
                           v_enu_mps=np.array([v0[0], v0[1], 0.0]), q_nb=dcm_to_q(R_nb))
        yaw_sigma = math.radians(5.0 if mount_ok else 25.0)
        P0 = np.diag([f0.horizontal_accuracy_m ** 2] * 2 + [(2.5 * f0.horizontal_accuracy_m) ** 2]
                     + [0.5 ** 2] * 3 + [math.radians(2.0) ** 2] * 2 + [yaw_sigma ** 2]
                     + [noise.accel_bias_sigma_mps2 ** 2] * 3 + [noise.gyro_bias_sigma_radps ** 2] * 3)
        return ErrorStateEKF(nominal, P0, noise)

    filters = [(make_filter(o), ConsecutiveRejectionReset(), o) for o in offsets]

    # outage schedule (fixes withheld), starting after the coarse-alignment window
    outages = []
    s, k = t[i0] + max(120.0, MH_WINDOW_S + 30.0), 0
    while s + max(OUTAGES_S) < t[-1]:
        outages.append((s, s + OUTAGES_S[k % len(OUTAGES_S)]))
        s += OUTAGE_EVERY_S
        k += 1

    def in_outage(x):
        return any(a <= x < b for a, b in outages)

    end_rows = {int(np.searchsorted(t, b)) - 1: round(b - a) for a, b in outages}
    gamma = 9.81
    acc_norm = np.linalg.norm(acc, axis=1)
    acc_std = pd.Series(acc_norm).rolling(STILL_WINDOW, min_periods=STILL_WINDOW).std().to_numpy()
    fix_i = next(i for i, f in enumerate(fixes) if f.t_received_s > t[i0])
    last_speed = f0.speed_mps
    aided_err, aided_sig, outage_end = [], [], {round(o): [] for o in OUTAGES_S}
    outage_ac = {round(o): [] for o in OUTAGES_S}
    started = time.time()
    chosen_offset = None
    try:
        for i in range(i0 + 1, len(t)):
            dt = t[i] - t[i - 1]
            if dt <= 0:
                continue  # one of the 86 kept duplicate timestamps (Phase 3): no interval
            if dt > 1.0:
                ekf, rec, _ = filters[0]
                return {"session_id": sid, "stopped": f"segment break at t={t[i]:.1f} s",
                        **_summary(aided_err, aided_sig, outage_end, ekf, mount, interval, outages, rec,
                                   n_bad_velocity, chosen_offset, outage_ac)}
            imu = IOVNBD_AXIS_MAP.to_body(t[i - 1], acc[i - 1], gyr[i - 1])
            new_fixes = []
            while fix_i < len(fixes) and fixes[fix_i].t_received_s <= t[i] + 1e-9:
                new_fixes.append(fixes[fix_i])
                # only a fix the filter actually receives may inform the stillness detector:
                # a withheld (outage) fix's speed leaked GNSS into outages (found in Phase 7)
                if fixes[fix_i].speed_mps is not None and not in_outage(fixes[fix_i].t_s):
                    last_speed = fixes[fix_i].speed_mps
                fix_i += 1
            still = (acc_std[i] < STILL_ACC_STD and abs(acc_norm[i] - gamma) < STILL_NORM_TOL
                     and abs(gyr[i, 1]) < STILL_GYRO_ABS and last_speed is not None
                     and last_speed < STILL_GNSS_SPEED)
            for ekf, rec, _ in filters:
                ekf.predict(imu, dt)
                for fx_ in new_fixes:
                    if not in_outage(fx_.t_s):
                        rec.after_update(ekf, fx_, ekf.update_gnss(fx_))
                if still:
                    ekf.update_zupt(t[i])
                    ekf.update_levelling(IOVNBD_AXIS_MAP.to_body(t[i], acc[i], gyr[i]))
            if len(filters) > 1 and t[i] >= t[i0] + MH_WINDOW_S:
                def score(item):
                    g = [min(r.nis, 50.0) if r.accepted else 50.0 for r in item[0].log if r.kind == "gnss"]
                    return (np.mean(g) if g else np.inf) + 50.0 * item[1].n_resets
                filters = [min(filters, key=score)]
                chosen_offset = math.degrees(filters[0][2])
            if chosen_offset is None and len(filters) == 1:
                chosen_offset = math.degrees(filters[0][2])
            ekf = filters[0][0]
            if truth["gt_valid"].iat[i] and len(filters) == 1:
                e = horiz_err_m(ekf.nominal.lat_rad, ekf.nominal.lon_rad,
                                math.radians(truth["gt_lat_deg"].iat[i]), math.radians(truth["gt_lon_deg"].iat[i]))
                if i in end_rows:
                    outage_end[end_rows[i]].append(e)
                    de, dn = err_en_m(ekf.nominal.lat_rad, ekf.nominal.lon_rad,
                                      math.radians(truth["gt_lat_deg"].iat[i]), math.radians(truth["gt_lon_deg"].iat[i]))
                    outage_ac[end_rows[i]].append(along_cross(de, dn, truth["gt_heading_deg"].iat[i]))
                elif not in_outage(t[i]) and t[i] > t[i0] + 60:
                    aided_err.append(e)
                    aided_sig.append(math.hypot(*ekf.std()[:2]))
    except CovarianceError as exc:
        return {"session_id": sid, "diverged": str(exc)}
    ekf, rec, _ = filters[0]
    out = _summary(aided_err, aided_sig, outage_end, ekf, mount, interval, outages, rec,
                   n_bad_velocity, chosen_offset, outage_ac)
    out.update(session_id=sid, runtime_s=round(time.time() - started, 1))
    out["_raw"] = {"aided_err": aided_err, "aided_sig": aided_sig,
                   "outage_end": outage_end, "outage_ac": outage_ac}
    return out


def _summary(aided_err, aided_sig, outage_end, ekf, mount, interval, outages, rec,
             n_bad_velocity, chosen_offset, outage_ac) -> dict:
    ae, sg = np.array(aided_err), np.array(aided_sig)
    upd = [r for r in ekf.log if r.kind == "gnss"]
    return {
        "phone_fix_interval_s": interval,
        "mount": {"yaw_deg": math.degrees(mount.yaw_rad), "acceptable": mount.acceptable(),
                  "r2": mount.fit.r2},
        "aided_horizontal_err_m": {"p50": float(np.median(ae)), "p95": float(np.percentile(ae, 95)),
                                   "n": len(ae)} if len(ae) else None,
        "aided_err_within_2sigma_fraction": float(np.mean(ae <= 2 * sg)) if len(ae) else None,
        "outage_end_horizontal_err_m": {
            str(int(o)): {"p50": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": len(v)}
            if v else None for o, v in outage_end.items()},
        "gnss_updates": {"accepted": sum(r.accepted for r in upd), "rejected": sum(not r.accepted for r in upd),
                         "mean_nis_accepted": float(np.mean([r.nis for r in upd if r.accepted])) if upd else None},
        "zupt_updates": sum(r.kind == "zupt" for r in ekf.log),
        "final_yaw_deg": math.degrees(q_to_euler(ekf.nominal.q_nb)[0]),
        "n_outages": len(outages),
        "gnss_lockout_resets": rec.n_resets,
        "outage_end_along_cross_abs_median_m": {
            str(o): [float(np.median(np.abs([x[0] for x in v]))), float(np.median(np.abs([x[1] for x in v])))]
            if v else None for o, v in outage_ac.items()},
        "fixes_with_invalid_velocity": n_bad_velocity,
        "coarse_alignment_yaw_offset_deg": chosen_offset,
        "n_gnss_updates_attempted": sum(r.kind == "gnss" for r in ekf.log),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=None, help="comma list; default = val AND verified")
    ap.add_argument("--all-verified", action="store_true", help="train + val, verified only")
    args = ap.parse_args()
    split = json.loads(CFG.split_manifest.read_text(encoding="utf-8"))
    p3 = json.loads((CFG.reports_dir / "phase3" / "preprocessing_report.json").read_text(encoding="utf-8"))
    verified = {s["session_id"] for s in p3["sessions"] if s["alignment_verified"]}
    pool = split["splits"]["val"] + (split["splits"]["train"] if args.all_verified else [])
    sessions = args.sessions.split(",") if args.sessions else sorted(s for s in pool if s in verified)
    leaked = [s for s in sessions if split["sessions"][s]["split"] == "test"]
    if leaked:
        raise SystemExit(f"refusing to evaluate held-out TEST sessions in Phase 4: {leaked}")
    noise = NoiseParams.from_report(json.loads(NOISE_REPORT.read_text(encoding="utf-8"))["filter_parameters"])
    results = []
    for sid in sessions:
        r = run_session(sid, noise)
        results.append(r)
        print(json.dumps({k: r.get(k) for k in ("session_id", "skipped", "stopped", "diverged",
                                                 "aided_horizontal_err_m", "outage_end_horizontal_err_m",
                                                 "aided_err_within_2sigma_fraction", "runtime_s")}, default=str))
    pooled = {}
    for o in OUTAGES_S:
        vals = [r["outage_end_horizontal_err_m"][str(int(o))] for r in results
                if r.get("outage_end_horizontal_err_m") and r["outage_end_horizontal_err_m"].get(str(int(o)))]
        pooled[f"outage_{int(o)}s_session_p50_median_m"] = float(np.median([v["p50"] for v in vals])) if vals else None
        pooled[f"outage_{int(o)}s_n_outages"] = int(sum(v["n"] for v in vals))
    aided = [r["aided_horizontal_err_m"]["p50"] for r in results if r.get("aided_horizontal_err_m")]
    pooled["aided_p50_session_median_m"] = float(np.median(aided)) if aided else None
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "script": "scripts/evaluate/phase4_baseline.py",
              "session_set": "custom" if args.sessions else ("train+val verified" if args.all_verified else "val verified"),
              "sessions_used": sessions,
              "noise": noise.__dict__, "outages_s": list(OUTAGES_S), "outage_every_s": OUTAGE_EVERY_S,
              "pooled": pooled, "sessions": results}
    out = OUT
    if args.sessions:
        out = OUT.with_name("baseline_evaluation_custom.json")
    elif args.all_verified:
        out = OUT.with_name("baseline_evaluation_all_verified.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(pooled, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
