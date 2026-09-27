#!/usr/bin/env python3
"""Phone-to-vehicle mounting alignment on the IO-VNBD sessions (Phase 4).

Per session:
* PHONE-ONLY estimate: phone accelerometer + phone GNSS speed/course at the fix EPOCHS
  (latency-tagged in Phase 3) -> navcore.alignment.mounting.estimate_mounting.
* REFERENCE (evaluation only): the same algorithm with VBOX speed/heading resampled at
  1 Hz on true time -- a check on the phone-only answer, never an input to it.
* STABILITY: phone-only estimate on the first vs second half of the session.

    .venv/Scripts/python.exe scripts/analysis/mounting_alignment.py
    -> reports/phase4/mounting_alignment.json
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from navcore.alignment.mounting import estimate_mounting  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
OUT = CFG.reports_dir / "phase4" / "mounting_alignment.json"


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def phone_fixes(d: pd.DataFrame, t0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = d[d["ph_gnss_new_fix"] & d["ph_gnss_speed_mps"].notna() & d["ph_gnss_bearing_deg"].notna()
          & d["ph_gnss_epoch_utc"].notna()]
    t = (f["ph_gnss_epoch_utc"] - t0).dt.total_seconds().to_numpy()
    order = np.argsort(t, kind="stable")
    return (t[order], f["ph_gnss_speed_mps"].to_numpy()[order],
            np.radians(f["ph_gnss_bearing_deg"].to_numpy()[order]))


def summarise(est) -> dict:
    return {"yaw_deg": math.degrees(est.yaw_rad), "tilt_deg": math.degrees(est.tilt_rad),
            "r2": est.fit.r2, "scale": est.fit.scale, "n": est.fit.n, "acceptable": est.acceptable()}


def main() -> int:
    rows = []
    for f in sorted(CFG.synced_root.glob("*.parquet")):
        d = pd.read_parquet(f)
        gts = d["gt_speed_mps"].to_numpy()
        if not np.isfinite(gts).any() or np.nanmax(gts) < 3.0:
            continue  # stationary recording: yaw unobservable by construction
        t0 = d["t_utc"].iloc[0]
        t_imu = (d["t_utc"] - t0).dt.total_seconds().to_numpy()
        acc = d[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy()
        r = {"session_id": f.stem}
        try:
            tf, v, b = phone_fixes(d, t0)
            interval = float(np.median(np.diff(tf))) if len(tf) > 1 else float("nan")
            r["phone_fix_interval_s"] = interval
            gap = 2.5 if interval <= 2.0 else 12.0
            est = estimate_mounting(acc, t_imu, tf, v, b, max_fix_gap_s=gap)
            r["phone"] = summarise(est)
            half = t_imu[-1] / 2
            h = [estimate_mounting(acc[m], t_imu[m], tf[k], v[k], b[k], max_fix_gap_s=gap) for m, k in (
                (t_imu < half, tf < half), (t_imu >= half, tf >= half))]
            r["halves_yaw_deg"] = [math.degrees(x.yaw_rad) for x in h]
            r["halves_acceptable"] = [x.acceptable() for x in h]
        except ValueError as exc:
            r["phone"] = {"error": str(exc), "acceptable": False}
        try:  # evaluation reference: VBOX at 1 Hz on true time (no latency)
            g = d[d["gt_valid"]].iloc[::10]
            tg = (g["t_utc"] - t0).dt.total_seconds().to_numpy()
            ref = estimate_mounting(acc, t_imu, tg, g["gt_speed_mps"].to_numpy(),
                                    np.radians(g["gt_heading_deg"].to_numpy()))
            r["reference"] = summarise(ref)
        except ValueError as exc:
            r["reference"] = {"error": str(exc), "acceptable": False}
        if "yaw_deg" in r["phone"] and "yaw_deg" in r["reference"]:
            r["phone_minus_reference_deg"] = wrap_deg(r["phone"]["yaw_deg"] - r["reference"]["yaw_deg"])
        if "halves_yaw_deg" in r:
            r["half_difference_deg"] = wrap_deg(r["halves_yaw_deg"][0] - r["halves_yaw_deg"][1])
        rows.append(r)

    ok = [r for r in rows if r["phone"].get("acceptable") and r["reference"].get("acceptable")]
    diff = np.abs([r["phone_minus_reference_deg"] for r in ok])
    halves = np.abs([r["half_difference_deg"] for r in ok if all(r.get("halves_acceptable", []))])
    tilts = [r["phone"]["tilt_deg"] for r in rows if "tilt_deg" in r["phone"]]
    summary = {
        "sessions_evaluated": len(rows),
        "phone_only_acceptable": sum(bool(r["phone"].get("acceptable")) for r in rows),
        "reference_acceptable": sum(bool(r["reference"].get("acceptable")) for r in rows),
        "both_acceptable": len(ok),
        "yaw_phone_vs_reference_abs_deg": {
            "median": float(np.median(diff)) if len(diff) else None,
            "p90": float(np.percentile(diff, 90)) if len(diff) else None,
            "max": float(diff.max()) if len(diff) else None},
        "yaw_first_vs_second_half_abs_deg": {
            "n": len(halves),
            "median": float(np.median(halves)) if len(halves) else None,
            "p90": float(np.percentile(halves, 90)) if len(halves) else None},
        "tilt_deg": {"median": float(np.median(tilts)), "max": float(np.max(tilts))},
        "acceptance_rule": "n >= 30 fix epochs, r2 >= 0.3, scale in [0.5, 1.5]",
    }
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "script": "scripts/analysis/mounting_alignment.py", "summary": summary, "sessions": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    for r in rows:
        p, q = r["phone"], r["reference"]
        print(f"{r['session_id']:7} phone yaw {p.get('yaw_deg', float('nan')):8.1f} r2 {p.get('r2', float('nan')):5.2f} "
              f"sc {p.get('scale', float('nan')):5.2f} ok {p.get('acceptable')!s:5} | ref yaw {q.get('yaw_deg', float('nan')):8.1f} "
              f"r2 {q.get('r2', float('nan')):5.2f} | diff {r.get('phone_minus_reference_deg', float('nan')):6.1f} "
              f"halves {r.get('half_difference_deg', float('nan')):6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
