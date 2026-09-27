#!/usr/bin/env python3
"""Phase 7 real-data ablation: AI-assisted fusion + NHC vs the classical baseline.

    .venv/Scripts/python.exe scripts/evaluate/phase7_fusion_eval.py [--sessions S3a,...]

Every arm runs the SAME navigator (navcore.fusion.navigator.FusedNavigator); only switches
differ, so differences are attributable:
    baseline      no AI, no NHC, no state machine   (= Phase 4 behaviour, tilt fix included)
    nhc           + NHC (vertical always; lateral only with an accepted mount)
    sm_only       + GNSS state machine alone (noise inflation, no AI): isolates its effect
    ai_speed      + GNSS state machine + Model A speed updates when GNSS is not GOOD
    ai_speed_nhc  both
    full          + Model B acceleration correction before predict()
    default       FusionConfig() as shipped (= ai_speed_nhc; Model B off, see navigator.py)

Initialisation, multi-hypothesis heading alignment, outage schedule (30 s / 60 s GNSS
withheld every 4 min) and scoring follow scripts/evaluate/phase4_baseline.py. Scored
against VBOX on phone-clock-VERIFIED sessions only. Default set: VALIDATION split
(verified). Caveat: the Phase 6 sweep selected models on the same split, so these
numbers lean optimistic; the held-out TEST drivers are kept for a final evaluation.
Phone-only inputs; CAN/VBOX never reach the navigator.
Writes reports/phase7/fusion_evaluation.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from navcore.alignment.mounting import estimate_mounting  # noqa: E402
from navcore.filtering.ekf import CovarianceError, ErrorStateEKF, NoiseParams  # noqa: E402
from navcore.fusion.ai_models import OnnxSequenceModel  # noqa: E402
from navcore.fusion.navigator import FusedNavigator, FusionConfig  # noqa: E402
from navcore.geometry.quaternion import dcm_to_q  # noqa: E402
from navcore.geometry.rotation import rot_z  # noqa: E402
from navcore.ins.mechanization import NavState  # noqa: E402
from navcore.sensors.samples import IOVNBD_AXIS_MAP, GnssSample  # noqa: E402
from scripts.evaluate.phase4_baseline import along_cross, err_en_m  # noqa: E402
from training.preprocessing.columns import assert_model_inputs  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
OUT = CFG.reports_dir / "phase7" / "fusion_evaluation.json"
INPUTS = ["acc_x_mps2", "acc_y_mps2", "acc_z_mps2", "grav_x_mps2", "grav_y_mps2", "grav_z_mps2",
          "gyro_x_radps", "gyro_y_radps", "gyro_z_radps", "ph_gnss_lat_deg", "ph_gnss_lon_deg",
          "ph_gnss_alt_m", "ph_gnss_speed_mps", "ph_gnss_accuracy_m", "ph_gnss_bearing_deg",
          "ph_gnss_new_fix", "ph_gnss_epoch_utc"]
ARMS = {
    "baseline": FusionConfig(use_ai_speed=False, use_ai_accel=False, use_nhc=False, use_state_machine=False),
    "nhc": FusionConfig(use_ai_speed=False, use_ai_accel=False, use_nhc=True, use_state_machine=False,
                        nhc_requires_ai_speed=False),
    "sm_only": FusionConfig(use_ai_speed=False, use_ai_accel=False, use_nhc=False),
    "ai_speed": FusionConfig(use_ai_speed=True, use_ai_accel=False, use_nhc=False),
    "ai_speed_nhc": FusionConfig(use_ai_speed=True, use_ai_accel=False, use_nhc=True),
    "full": FusionConfig(use_ai_speed=True, use_ai_accel=True, use_nhc=True),
    "default": FusionConfig(),  # the shipped defaults: AI speed + NHC, Model B off
}
OUTAGES_S = (30.0, 60.0)
OUTAGE_EVERY_S = 240.0
MIN_SEGMENT_S = 300.0
MH_WINDOW_S = 120.0


def build_fixes(d: pd.DataFrame, t0) -> list[GnssSample]:
    fx = d[d["ph_gnss_new_fix"] & d["ph_gnss_epoch_utc"].notna() & d["ph_gnss_accuracy_m"].gt(0)]
    out = []
    for r in fx.itertuples():
        ok = np.isfinite(r.ph_gnss_speed_mps) and np.isfinite(r.ph_gnss_bearing_deg) and r.ph_gnss_speed_mps >= 0
        out.append(GnssSample(
            t_s=(r.ph_gnss_epoch_utc - t0).total_seconds(), t_received_s=(r.t_utc - t0).total_seconds(),
            lat_rad=math.radians(r.ph_gnss_lat_deg), lon_rad=math.radians(r.ph_gnss_lon_deg),
            h_m=r.ph_gnss_alt_m, horizontal_accuracy_m=r.ph_gnss_accuracy_m,
            speed_mps=r.ph_gnss_speed_mps if ok else None,
            bearing_rad=math.radians(r.ph_gnss_bearing_deg) if ok else None))
    return out


def run_arm(arm: str, d: pd.DataFrame, truth: pd.DataFrame, fixes: list[GnssSample], mount, noise,
            models: dict, observer=None, cfg: FusionConfig | None = None) -> dict:
    """``observer(i, nav, in_outage)`` (optional, read-only) is called after every step once a
    single heading hypothesis remains -- Phase 8 uses it to feed the map matcher."""
    t0 = d["t_utc"].iloc[0]
    t = (d["t_utc"] - t0).dt.total_seconds().to_numpy()
    acc = d[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy()
    grav = d[["grav_x_mps2", "grav_y_mps2", "grav_z_mps2"]].to_numpy()
    gyr = d[["gyro_x_radps", "gyro_y_radps", "gyro_z_radps"]].to_numpy()
    k0 = next((i for i, f in enumerate(fixes) if f.speed_mps is not None and f.speed_mps > 5.0), None)
    if k0 is None:
        return {"skipped": "never above 5 m/s on a phone fix"}
    f0 = fixes[k0]
    i0 = int(np.searchsorted(t, f0.t_received_s))
    veh_yaw = math.pi / 2 - f0.bearing_rad
    v0 = f0.velocity_enu()
    mount_ok = mount.acceptable()
    cfg = cfg or ARMS[arm]
    offsets = [0.0] if mount_ok else [k * math.pi / 4 for k in range(8)]

    def make(offset):
        nominal = NavState(t_s=t[i0], lat_rad=f0.lat_rad, lon_rad=f0.lon_rad, h_m=f0.h_m,
                           v_enu_mps=np.array([v0[0], v0[1], 0.0]),
                           q_nb=dcm_to_q(rot_z(veh_yaw + offset) @ mount.R_vb))
        ys = math.radians(5.0 if mount_ok else 25.0)
        P0 = np.diag([f0.horizontal_accuracy_m ** 2] * 2 + [(2.5 * f0.horizontal_accuracy_m) ** 2]
                     + [0.25] * 3 + [math.radians(2.0) ** 2] * 2 + [ys ** 2]
                     + [noise.accel_bias_sigma_mps2 ** 2] * 3 + [noise.gyro_bias_sigma_radps ** 2] * 3)
        use = {k: v for k, v in models.items() if (k == "model_speed" and cfg.use_ai_speed)
               or (k == "model_accel" and cfg.use_ai_accel)}
        # vertical NHC needs only levelling (R_vb's third row); lateral only with an accepted mount
        return FusedNavigator(ErrorStateEKF(nominal, P0, noise), IOVNBD_AXIS_MAP, cfg,
                              R_vb=mount.R_vb, lateral_nhc_allowed=mount_ok, **use), offset

    navs = [make(o) for o in offsets]
    outages, s, k = [], t[i0] + MH_WINDOW_S + 30.0, 0
    while s + max(OUTAGES_S) < t[-1]:
        outages.append((s, s + OUTAGES_S[k % 2]))
        s += OUTAGE_EVERY_S
        k += 1
    end_rows = {int(np.searchsorted(t, b)) - 1: round(b - a) for a, b in outages}
    in_out = np.zeros(len(t), bool)
    for a, b in outages:
        in_out |= (t >= a) & (t < b)
    fix_i = next((i for i, f in enumerate(fixes) if f.t_received_s > t[i0]), len(fixes))
    aided, sig, oe, oac = [], [], {round(o): [] for o in OUTAGES_S}, {round(o): [] for o in OUTAGES_S}
    started = time.time()
    try:
        for i in range(i0, len(t)):
            new = []
            while fix_i < len(fixes) and fixes[fix_i].t_received_s <= t[i] + 1e-9:
                if not any(a <= fixes[fix_i].t_s < b for a, b in outages):
                    new.append(fixes[fix_i])
                fix_i += 1
            for nav, _ in navs:
                nav.step(t[i], acc[i], grav[i], gyr[i], new)
            if len(navs) > 1 and t[i] >= t[i0] + MH_WINDOW_S:
                def score(item):
                    g = [min(r.nis, 50.0) if r.accepted else 50.0 for r in item[0].ekf.log if r.kind == "gnss"]
                    return (np.mean(g) if g else np.inf) + 50.0 * item[0].reset.n_resets
                navs = [min(navs, key=score)]
            if len(navs) == 1 and observer is not None:
                observer(i, navs[0][0], bool(in_out[i]))
            if len(navs) == 1 and truth["gt_valid"].iat[i]:
                nav = navs[0][0]
                de, dn = err_en_m(nav.ekf.nominal.lat_rad, nav.ekf.nominal.lon_rad,
                                  math.radians(truth["gt_lat_deg"].iat[i]), math.radians(truth["gt_lon_deg"].iat[i]))
                e = math.hypot(de, dn)
                if i in end_rows:
                    oe[end_rows[i]].append(e)
                    oac[end_rows[i]].append(along_cross(de, dn, truth["gt_heading_deg"].iat[i]))
                elif not in_out[i] and t[i] > t[i0] + 60:
                    aided.append(e)
                    sig.append(math.hypot(*nav.ekf.std()[:2]))
    except CovarianceError as exc:
        return {"diverged": str(exc)}
    nav = navs[0][0]
    return {"aided_err": aided, "aided_sig": sig, "outage_end": oe, "outage_ac": oac,
            "counts": dict(nav.counts), "chosen_yaw_offset_deg": math.degrees(navs[0][1]),
            "mount_acceptable": mount_ok, "runtime_s": round(time.time() - started, 1)}


def summarise(raws: list[dict]) -> dict:
    ae = np.array([x for r in raws for x in r.get("aided_err", [])])
    sg = np.array([x for r in raws for x in r.get("aided_sig", [])])
    out = {"aided_p50_m": float(np.median(ae)) if ae.size else None,
           "aided_p95_m": float(np.percentile(ae, 95)) if ae.size else None,
           "aided_within_2sigma": float(np.mean(ae <= 2 * sg)) if ae.size else None}
    for o in (30, 60):
        v = [x for r in raws for x in r.get("outage_end", {}).get(o, [])]
        ac = [x for r in raws for x in r.get("outage_ac", {}).get(o, [])]
        out[f"outage_{o}s"] = {"n": len(v), "p50_m": float(np.median(v)) if v else None,
                               "p90_m": float(np.percentile(v, 90)) if v else None,
                               "along_p50_m": float(np.median(np.abs([a for a, _ in ac]))) if ac else None,
                               "cross_p50_m": float(np.median(np.abs([c for _, c in ac]))) if ac else None}
    # Phase 4's headline statistic: median over sessions of each session's median
    per = [r for r in raws if r.get("aided_err")]
    out["session_median_of_medians"] = {
        "aided_m": float(np.median([np.median(r["aided_err"]) for r in per])) if per else None,
        **{f"outage_{o}s_m": (float(np.median(m)) if (m := [np.median(r["outage_end"][o]) for r in per
                                                            if r["outage_end"].get(o)]) else None)
           for o in (30, 60)}}
    out["n_failed"] = sum(1 for r in raws if "diverged" in r)
    counts = {}
    for r in raws:
        for k, v in r.get("counts", {}).items():
            counts[k] = counts.get(k, 0) + v
    out["counts"] = counts
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=None)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    assert_model_inputs(INPUTS)  # CAN rule, mechanically
    split = json.loads(CFG.split_manifest.read_text(encoding="utf-8"))
    p3 = json.loads((CFG.reports_dir / "phase3" / "preprocessing_report.json").read_text(encoding="utf-8"))
    verified = {s["session_id"] for s in p3["sessions"] if s["alignment_verified"]}
    sessions = args.sessions.split(",") if args.sessions else sorted(s for s in split["splits"]["val"] if s in verified)
    if any(split["sessions"][s]["split"] == "test" for s in sessions):
        raise SystemExit("refusing to evaluate held-out TEST sessions in Phase 7")
    sel = json.loads((REPO_ROOT / "reports" / "phase6" / "selection.json").read_text(encoding="utf-8"))
    exp = REPO_ROOT / "models" / "exported"
    models = {"model_speed": OnnxSequenceModel(exp / f"{sel['model_a']['selected']}.onnx"),
              "model_accel": OnnxSequenceModel(exp / f"{sel['model_b']['selected']}.onnx")}
    noise = NoiseParams.from_report(json.loads((CFG.reports_dir / "phase4" / "imu_noise.json")
                                               .read_text(encoding="utf-8"))["filter_parameters"])
    arms = args.arms.split(",")
    raw = {a: [] for a in arms}
    per_session = []
    for sid in sessions:
        full = pd.read_parquet(CFG.synced_root / f"{sid}.parquet")
        for seg_id, d in full.groupby("segment_id"):
            d = d.reset_index(drop=True)
            if (d["t_utc"].iloc[-1] - d["t_utc"].iloc[0]).total_seconds() < MIN_SEGMENT_S:
                continue
            truth = d[["gt_lat_deg", "gt_lon_deg", "gt_heading_deg", "gt_valid"]]  # scoring only
            d = d[[*INPUTS, "t_utc"]]
            t0 = d["t_utc"].iloc[0]
            fixes = build_fixes(d, t0)
            vf = [f for f in fixes if f.speed_mps is not None]
            if len(vf) < 20:
                continue
            tt = (d["t_utc"] - t0).dt.total_seconds().to_numpy()
            ft = np.array([f.t_s for f in vf])
            iv = float(np.median(np.diff(ft)))
            mount = estimate_mounting(d[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy(), tt, ft,
                                      np.array([f.speed_mps for f in vf]), np.array([f.bearing_rad for f in vf]),
                                      max_fix_gap_s=2.5 if iv <= 2 else 12.0)
            row = {"session_id": sid, "segment_id": int(seg_id), "mount_acceptable": mount.acceptable(),
                   "fix_interval_s": iv, "arms": {}}
            for arm in arms:
                r = run_arm(arm, d, truth, fixes, mount, noise, models)
                raw[arm].append(r)
                row["arms"][arm] = summarise([r]) | {k: r[k] for k in ("diverged", "skipped") if k in r}
                o30, o60 = row["arms"][arm]["outage_30s"], row["arms"][arm]["outage_60s"]
                print(f"{sid}/{seg_id} {arm:13} aided p50 {row['arms'][arm]['aided_p50_m']}  30s {o30['p50_m']}  "
                      f"60s {o60['p50_m']}  ({r.get('runtime_s')} s)", flush=True)
            per_session.append(row)
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "script": "scripts/evaluate/phase7_fusion_eval.py", "sessions": sessions,
              "models": {k: v["selected"] for k, v in sel.items()},
              "arms": {a: {"config": ARMS[a].__dict__ | {"nhc": ARMS[a].nhc.__dict__}, **summarise(raw[a])}
                       for a in arms},
              "per_segment": per_session,
              "caveat": "VAL split; models were selected on VAL in Phase 6 -> optimistic; TEST reserved"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    for a in arms:
        s = report["arms"][a]
        print(f"== {a:13} aided p50 {s['aided_p50_m']}  30s p50 {s['outage_30s']['p50_m']} (n{s['outage_30s']['n']})  "
              f"60s p50 {s['outage_60s']['p50_m']} (n{s['outage_60s']['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
