#!/usr/bin/env python3
"""Empirical resolution of the IO-VNBD gyroscope axis mapping (Phase 4).

Known from Phase 2: logged column 2 (``gyro_y``) is rotation about the vertical, sign +,
gain 0.997 vs CAN yaw rate. Open: which device axes columns 1 and 3 are.

Decisive test -- stop-to-stop gravity prediction. While the car is stopped the
accelerometer measures pure gravity, i.e. the exact device tilt, and gyro bias is
observable as the mean rate. Between two consecutive stops (3-60 s apart, same segment)
the three gyro rates are bias-corrected and integrated under each of the 8 signed
permutations of columns 1/3 onto device x/y (column 2 -> device z). The correct mapping
must rotate stop-1 gravity into stop-2 gravity. The NULL hypothesis (horizontal rates = 0)
is the baseline any real mapping has to beat.

Stops are taken from ``gt_stationary`` (VBOX + CAN + wheels all zero): ground truth used
to SELECT analysis windows, never as a model/filter input.

    .venv/Scripts/python.exe scripts/analysis/gyro_axis_resolution.py

Writes reports/phase4/gyro_axis_resolution.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from navcore.geometry import quaternion as Q  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
OUT = CFG.reports_dir / "phase4" / "gyro_axis_resolution.json"
MIN_STOP_SAMPLES = 30  # 3 s
DRIVE_BETWEEN_S = (3.0, 60.0)
# (column for device x, sign, column for device y, sign); None = horizontal rates zero
HYPOTHESES: dict[str, tuple[int, int, int, int] | None] = {
    "x=+c1,y=+c3": (1, 1, 3, 1), "x=+c1,y=-c3": (1, 1, 3, -1),
    "x=-c1,y=+c3": (1, -1, 3, 1), "x=-c1,y=-c3": (1, -1, 3, -1),
    "x=+c3,y=+c1": (3, 1, 1, 1), "x=+c3,y=-c1": (3, 1, 1, -1),
    "x=-c3,y=+c1": (3, -1, 1, 1), "x=-c3,y=-c1": (3, -1, 1, -1),
    "null (horizontal = 0)": None,
}


def stop_runs(stationary: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, stationary.astype(int), 0]))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= MIN_STOP_SAMPLES]


def main() -> int:
    err: dict[str, list[float]] = {h: [] for h in HYPOTHESES}
    tilt_change, pairs_by_session = [], {}
    for f in sorted(CFG.synced_root.glob("*.parquet")):
        d = pd.read_parquet(f)
        cols = {1: d["gyro_x_radps"].to_numpy(), 2: d["gyro_y_radps"].to_numpy(),
                3: d["gyro_z_radps"].to_numpy()}
        acc = d[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy()
        dt, seg = d["dt_s"].to_numpy(), d["segment_id"].to_numpy()
        runs = stop_runs(d["gt_stationary"].to_numpy())
        n_pairs = 0
        for (a1, b1), (a2, b2) in pairwise(runs):
            gap_s = (a2 - b1) / 10.0
            if seg[a1] != seg[b2 - 1] or not DRIVE_BETWEEN_S[0] <= gap_s <= DRIVE_BETWEEN_S[1]:
                continue
            g1, g2 = acc[a1:b1].mean(0), acc[a2:b2].mean(0)
            g1, g2 = g1 / np.linalg.norm(g1), g2 / np.linalg.norm(g2)
            bias = {k: 0.5 * (c[a1:b1].mean() + c[a2:b2].mean()) for k, c in cols.items()}
            tilt_change.append(float(np.degrees(np.arccos(np.clip(g1 @ g2, -1, 1)))))
            for h, spec in HYPOTHESES.items():
                q = Q.Q_IDENTITY.copy()
                for i in range(b1 - 1, a2):
                    if not np.isfinite(dt[i + 1]):
                        continue
                    wz = cols[2][i] - bias[2]
                    if spec is None:
                        w = np.array([0.0, 0.0, wz])
                    else:
                        ix, sx, iy, sy = spec
                        w = np.array([sx * (cols[ix][i] - bias[ix]), sy * (cols[iy][i] - bias[iy]), wz])
                    q = Q.q_integrate(q, w, dt[i + 1])
                g2_pred = Q.q_rotate(Q.q_conj(q), g1)  # q = R_{b2}^{b1}
                err[h].append(float(np.degrees(np.arccos(np.clip(g2_pred @ g2, -1, 1)))))
            n_pairs += 1
        if n_pairs:
            pairs_by_session[f.stem] = n_pairs

    null = np.array(err["null (horizontal = 0)"])
    table = {h: {"median_err_deg": float(np.median(e)), "mean_err_deg": float(np.mean(e)),
                 "p90_err_deg": float(np.percentile(e, 90)),
                 "fraction_pairs_beating_null": float(np.mean(np.array(e) < null))}
             for h, e in err.items()}
    best = min((h for h in HYPOTHESES if HYPOTHESES[h] is not None), key=lambda h: table[h]["median_err_deg"])
    resolved = table[best]["median_err_deg"] < table["null (horizontal = 0)"]["median_err_deg"]
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/analysis/gyro_axis_resolution.py",
        "known": "column 2 (gyro_y) = device z (vertical), sign +, gain 0.997 vs CAN yaw rate (Phase 2)",
        "test": "stop-to-stop gravity prediction, bias-corrected 3-axis integration",
        "n_stop_pairs": len(null), "sessions_with_pairs": len(pairs_by_session),
        "measured_tilt_change_deg": {"median": float(np.median(tilt_change)),
                                     "p90": float(np.percentile(tilt_change, 90))},
        "hypotheses": table,
        "best_permutation": best,
        "any_permutation_beats_null": bool(resolved),
        "conclusion": (
            f"{best} resolves the horizontal axes" if resolved else
            "UNRESOLVABLE AS ANGULAR RATES: every signed permutation of the logged horizontal "
            "columns predicts the stop-to-stop tilt change WORSE than ignoring them "
            f"(best {table[best]['median_err_deg']:.2f} deg vs null "
            f"{table['null (horizontal = 0)']['median_err_deg']:.2f} deg median). Integrating them "
            "injects several degrees of spurious tilt within a minute. Consistent with aliased "
            "vibration at 10 Hz sampling. Decision: IO-VNBD uses the vertical gyro only; pitch "
            "and roll come from gravity/levelling (reduced-IMU mechanization)."),
        "screening_references_also_negative": [
            "GRAVITY-vector rotation vs columns 1/3 at 1/2/5/10 s: median |r| < 0.04",
            "Android ORIENTATION pitch/roll derivatives: median |r| < 0.09, signs inconsistent",
            "integrated lateral-axis rate vs VBOX road grade, 8 hypotheses: best median r 0.11",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"{len(null)} stop pairs in {len(pairs_by_session)} sessions; tilt change median "
          f"{report['measured_tilt_change_deg']['median']:.2f} deg")
    for h, r in sorted(table.items(), key=lambda kv: kv[1]["median_err_deg"]):
        print(f"  {h:22} median {r['median_err_deg']:.3f} deg  beats null {r['fraction_pairs_beating_null']:.0%}")
    print(report["conclusion"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
