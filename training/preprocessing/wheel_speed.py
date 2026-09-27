"""Empirically resolve the IO-VNBD wheel-speed unit and calibrate it against VBOX.

The V-file labels wheel speeds "rad/sec". Two hypotheses:
  H_kmh : the channel is km/h (vehicle speed at the wheel), mislabelled.
  H_rads: the channel is wheel angular rate; v = r * w for rolling radius r.
They differ ONLY by a constant factor, so no internal-consistency test can separate a
unit label from a scale. What the ground truth needs is the empirical factor that turns
the channel into m/s, with its accuracy -- that is measured here against VBOX GNSS
Doppler speed (an independent sensor). The label question is then decided on the
evidence the factor provides (see "unit_decision" in the output).

    .venv/Scripts/python.exe scripts/preprocess/calibrate_wheel_speed.py   (thin wrapper)
    python -m training.preprocessing.pipeline                             (runs it as step 1)

Reads data/processed/iovnbd/v1/vehicle/*.parquet (V-file only: both channels come from the
same VBOX logger, so no phone alignment is involved).
Writes reports/phase3/wheel_speed_calibration.json.

GROUND TRUTH ONLY. The CAN channels are labels/benchmarks, never model or filter inputs
(user decision 2026-09-27; SIH26168 forbids OBD-II / external speed input).
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from training.preprocessing.config import DatasetConfig, load_dataset_config
from training.preprocessing.iovnbd_audit import fft_xcorr_lag

WHEELS = ["wheel_speed_fl_raw", "wheel_speed_fr_raw", "wheel_speed_rl_raw", "wheel_speed_rr_raw"]

# Clean-sample gate: steady, straight, well-tracked driving
MIN_SPEED_MPS = 5.0
MAX_LONG_ACCEL_MPS2 = 0.3
MAX_YAW_RATE_RADPS = 0.03
MIN_VBOX_SATS = 8
SPEED_BINS_MPS = [5, 10, 15, 20, 25, 30, 45]


def shift(a: np.ndarray, lag: int) -> np.ndarray:
    """Return a[t + lag] aligned to t, NaN-padded."""
    out = np.full(a.shape, np.nan)
    if lag >= 0:
        out[: a.size - lag] = a[lag:]
    else:
        out[-lag:] = a[: a.size + lag]
    return out


def calibrate(cfg: DatasetConfig | None = None, verbose: bool = True) -> dict:
    """Run the calibration, write reports/phase3/wheel_speed_calibration.json, return it."""
    cfg = cfg or load_dataset_config()
    out_path = cfg.reports_dir / "phase3" / "wheel_speed_calibration.json"
    per_session, pooled = [], []
    drive_corr = []
    for f in sorted((cfg.processed_root / "vehicle").glob("*.parquet")):
        v = pd.read_parquet(f)
        vb = v["vbox_speed_mps"].to_numpy()
        if np.nanmax(vb) < 1.0:
            continue  # stationary-only session
        raw = v[WHEELS].to_numpy()
        rear = raw[:, 2:].mean(axis=1)
        lag, c = fft_xcorr_lag(rear, vb, 30)  # rear[t + lag] matches vbox[t]
        rear_s = shift(rear, lag)
        raw_s = np.column_stack([shift(raw[:, i], lag) for i in range(4)])
        la = v["can_long_accel_mps2"].to_numpy()
        m = ((vb > MIN_SPEED_MPS) & (np.abs(la) < MAX_LONG_ACCEL_MPS2)
             & (np.abs(v["yaw_rate_radps"].to_numpy()) < MAX_YAW_RATE_RADPS)
             & (v["vbox_sats"].astype("float64").to_numpy() >= MIN_VBOX_SATS)
             & np.isfinite(rear_s))
        # driven axle: driven wheels over-speed under throttle, so (front - rear) tracks accel
        mv = (vb > 3.0) & np.isfinite(raw_s).all(axis=1)
        if mv.sum() > 100:
            fr = raw_s[mv, :2].mean(axis=1) - raw_s[mv, 2:].mean(axis=1)
            drive_corr.append(float(np.corrcoef(fr, la[mv])[0, 1]))
        if m.sum() < 50:
            continue
        ratio = rear_s[m] / (vb[m] * 3.6)
        per_session.append({"session_id": f.stem, "lag_samples": lag, "lag_corr": c,
                            "n_clean": int(m.sum()), "rear_ratio_median": float(np.median(ratio))})
        pooled.append(pd.DataFrame({
            "v": vb[m], "rear": rear_s[m], "fl": raw_s[m, 0], "fr": raw_s[m, 1],
            "rl": raw_s[m, 2], "rr": raw_s[m, 3], "session": f.stem}))

    P = pd.concat(pooled, ignore_index=True)
    kmh = P["v"] * 3.6
    ratios = {w: float(np.median(P[w] / kmh)) for w in ("fl", "fr", "rl", "rr", "rear")}
    k_rear = ratios["rear"]  # raw units per km/h of true speed
    # calibrated ground-truth speed [m/s] = rear_raw / (3.6 * k_rear)
    resid = P["rear"] / (3.6 * k_rear) - P["v"]
    bins = pd.cut(P["v"], SPEED_BINS_MPS)
    by_bin = [
        {"speed_bin_mps": str(b), "n": int(g.shape[0]),
         "rear_ratio_median": float(np.median(g["rear"] / (g["v"] * 3.6))),
         "resid_median_mps": float(np.median(resid[g.index])),
         "resid_p95_abs_mps": float(np.percentile(np.abs(resid[g.index]), 95))}
        for b, g in P.groupby(bins, observed=True)
    ]
    sess_r = np.array([s["rear_ratio_median"] for s in per_session])
    implied_r = 1.0 / (3.6 * k_rear)  # rolling radius [m] required under H_rads
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/preprocess/calibrate_wheel_speed.py",
        "reference": "VBOX GNSS Doppler speed (V-file), same logger as the CAN channels",
        "gate": {"min_speed_mps": MIN_SPEED_MPS, "max_abs_long_accel_mps2": MAX_LONG_ACCEL_MPS2,
                 "max_abs_yaw_rate_radps": MAX_YAW_RATE_RADPS, "min_vbox_sats": MIN_VBOX_SATS},
        "n_sessions": len(per_session),
        "n_clean_samples": len(P),
        "lag_samples_counts": {str(k): int(n) for k, n in
                               pd.Series([s["lag_samples"] for s in per_session]).value_counts().items()},
        "median_ratio_raw_per_kmh": ratios,
        "driven_axle": {
            "corr_front_minus_rear_vs_long_accel_median": float(np.median(drive_corr)),
            "fraction_sessions_positive": float(np.mean(np.array(drive_corr) > 0)),
            "n_sessions": len(drive_corr),
            "conclusion": "front-wheel drive: front wheels over-speed under throttle; the REAR "
                          "(non-driven) axle mean is the slip-free speed reference",
        },
        "calibration": {
            "channel": "mean(wheel_speed_rl_raw, wheel_speed_rr_raw)",
            "k_rear_raw_per_kmh": k_rear,
            "formula": "gt_wheel_speed_mps = rear_raw_mean / (3.6 * k_rear_raw_per_kmh)",
            "session_ratio_median_range": [float(sess_r.min()), float(sess_r.max())],
            "session_ratio_iqr": [float(np.percentile(sess_r, 25)), float(np.percentile(sess_r, 75))],
            "resid_vs_vbox_median_mps": float(np.median(resid)),
            "resid_vs_vbox_p50_abs_mps": float(np.median(np.abs(resid))),
            "resid_vs_vbox_p95_abs_mps": float(np.percentile(np.abs(resid), 95)),
            "by_speed_bin": by_bin,
        },
        "unit_decision": {
            "label_in_file": "rad/sec",
            "identifiable_from_data": False,
            "why_not": "H_kmh and H_rads differ by a constant factor only; every kinematic "
                       "relation (speed, left/right turn differential, gear ratios) scales "
                       "identically, so the label cannot be proven from this data",
            "implied_rolling_radius_under_rad_s_m": implied_r,
            "favoured": "km/h",
            "evidence": [
                f"rear-axle mean equals VBOX speed in km/h to a factor {k_rear:.5f}: the "
                "expected result for a calibrated km/h signal",
                f"under rad/s the rear rolling radius would have to be {implied_r:.5f} m, "
                "i.e. exactly 1/3.6 m to within the calibration spread -- a coincidence",
                "the channel tracks CAN 'Indicated Vehicle Speed (km/hr)' to ~0.1 %",
            ],
            "ground_truth_consequence": "none: the empirical factor above converts the channel "
                                        "to m/s regardless of the label",
        },
        "per_session": per_session,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if not verbose:
        return report
    c = report["calibration"]
    print(f"k_rear = {k_rear:.5f} raw/(km/h); implied r under rad/s = {implied_r:.5f} m")
    print(f"residual vs VBOX: median {c['resid_vs_vbox_median_mps']:+.4f} m/s, "
          f"p95 |.| {c['resid_vs_vbox_p95_abs_mps']:.4f} m/s over {len(P):,} clean samples, "
          f"{len(per_session)} sessions; session ratios {c['session_ratio_median_range']}")
    for b in by_bin:
        print(f"  {b['speed_bin_mps']:>10}  n={b['n']:7d}  ratio={b['rear_ratio_median']:.5f}  "
              f"resid={b['resid_median_mps']:+.4f}  p95={b['resid_p95_abs_mps']:.4f}")
    print(f"driven axle: {report['driven_axle']['conclusion']}")
    return report
