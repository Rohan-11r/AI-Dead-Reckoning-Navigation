#!/usr/bin/env python3
"""Measure IO-VNBD phone IMU noise for the filter's process noise (docs sec 8.4).

* AT REST: overlapping Allan deviation on the stationary-only recordings (Vw1: 34 min,
  Vw15) -> white-noise density and bias instability per channel.
* MOVING: the same white-noise read-out on moving stretches (VBOX speed > 3 m/s,
  contiguous >= 120 s, first-differenced to strip vehicle dynamics). Road vibration makes
  this much larger than the at-rest value; the filter uses the MOVING density for white
  noise (it is what the filter experiences) and the at-rest bias instability for biases.

Only the phone channels the navigation uses are characterised: the accelerometer and the
vertical gyro (logged column 2). Columns 1/3 are reported for the record but are not
angular rates (reports/phase4/gyro_axis_resolution.json).

    .venv/Scripts/python.exe scripts/analysis/imu_noise.py   -> reports/phase4/imu_noise.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from navcore.calibration.allan import noise_parameters  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
OUT = CFG.reports_dir / "phase4" / "imu_noise.json"
DT = 0.1
STATIONARY_SESSIONS = ("Vw1", "Vw15")
CHANNELS = {"acc_x_mps2": "m/s^2", "acc_y_mps2": "m/s^2", "acc_z_mps2": "m/s^2",
            "gyro_y_radps": "rad/s (logged col 2 = vertical)",
            "gyro_x_radps": "rad/s (col 1, NOT an angular rate -- record only)",
            "gyro_z_radps": "rad/s (col 3, NOT an angular rate -- record only)"}


def runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(a, b) for a, b in zip(e[::2], e[1::2], strict=True) if b - a >= min_len]


def main() -> int:
    rest: dict[str, list[dict]] = {c: [] for c in CHANNELS}
    for sid in STATIONARY_SESSIONS:
        d = pd.read_parquet(CFG.synced_root / f"{sid}.parquet")
        for a, b in runs(d["gt_stationary"].to_numpy() & (d["segment_id"] == 0).to_numpy(), 3000):
            for c in CHANNELS:
                rest[c].append({"session": sid, "n": b - a,
                                **noise_parameters(d[c].to_numpy()[a:b], DT)})
    moving: dict[str, list[float]] = {c: [] for c in CHANNELS}
    for f in sorted(CFG.synced_root.glob("*.parquet")):
        d = pd.read_parquet(f)
        mv = (np.nan_to_num(d["gt_speed_mps"].to_numpy()) > 3.0) & d["dt_s"].notna().to_numpy()
        for a, b in runs(mv, 1200):
            for c in CHANNELS:
                y = d[c].to_numpy()[a:b]
                # first difference removes slow vehicle dynamics; for white noise
                # var(diff) = 2 var -> rescale by 1/sqrt(2)
                sigma = np.std(np.diff(y)) / np.sqrt(2.0)
                moving[c].append(float(sigma * np.sqrt(DT)))  # density = sigma * sqrt(dt)

    summary = {}
    for c, unit in CHANNELS.items():
        r = rest[c]
        summary[c] = {
            "unit": unit,
            "rest_white_noise_density": float(np.median([x["white_noise_density"] for x in r])),
            "rest_bias_instability": float(np.median([x["bias_instability"] for x in r])),
            "rest_bias_instability_tau_s": float(np.median([x["bias_instability_tau_s"] for x in r])),
            "moving_white_noise_density_median": float(np.median(moving[c])),
            "moving_white_noise_density_p90": float(np.percentile(moving[c], 90)),
            "n_rest_windows": len(r), "n_moving_windows": len(moving[c]),
        }
    acc = ["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]
    filt = {
        "accel_white_noise_density": float(np.max([summary[c]["moving_white_noise_density_median"] for c in acc])),
        "gyro_white_noise_density": summary["gyro_y_radps"]["moving_white_noise_density_median"],
        "accel_bias_sigma": float(np.max([summary[c]["rest_bias_instability"] for c in acc])),
        "accel_bias_tau_s": float(np.median([summary[c]["rest_bias_instability_tau_s"] for c in acc])),
        "gyro_bias_sigma": summary["gyro_y_radps"]["rest_bias_instability"],
        "gyro_bias_tau_s": summary["gyro_y_radps"]["rest_bias_instability_tau_s"],
        "units": "density: [unit]*sqrt(s); bias sigma: [unit]; tau: s",
        "rule": "white noise = MOVING density (vibration the filter actually sees; max over accel "
                "axes); bias = AT-REST bias instability, Gauss-Markov with tau at the Allan floor",
    }
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "script": "scripts/analysis/imu_noise.py", "dt_s": DT,
              "stationary_sessions": list(STATIONARY_SESSIONS), "channels": summary,
              "filter_parameters": filt}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for c, s in summary.items():
        print(f"{c:13} rest N {s['rest_white_noise_density']:.2e}  B {s['rest_bias_instability']:.2e} "
              f"@{s['rest_bias_instability_tau_s']:.0f}s | moving N {s['moving_white_noise_density_median']:.2e} "
              f"(p90 {s['moving_white_noise_density_p90']:.2e}) windows {s['n_rest_windows']}/{s['n_moving_windows']}")
    print(json.dumps(filt, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
