#!/usr/bin/env python3
"""Evaluation report for the SELECTED Phase 6 models (reports/phase6/selection.json).

    .venv/Scripts/python.exe scripts/evaluate/evaluate_models.py

Protocol: selection was made on VAL only (scripts/train/sweep.py). This script reports VAL
and then evaluates the held-out TEST split (drivers never seen in training) ONCE -- the
only number that supports a deployment claim (docs/ml_pipeline.md sec 3).

Per model and split: MAE, RMSE, bias, |error| percentiles, errors by target range, and
uncertainty calibration -- 1/2/3-sigma coverage (Gaussian: 68.3 / 95.4 / 99.7 %), the std of
z = err / sigma (ideal 1.0), and a reliability table: RMS error vs mean predicted sigma in
10 bins of predicted sigma, summarised by ENCE = mean_b |RMSE_b - sigma_b| / sigma_b
(expected normalised calibration error; 0 = perfectly calibrated).
Writes reports/phase6/evaluation.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import pairwise
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.export.export_models import load_checkpoint  # noqa: E402
from training.configs.loader import load_training_config  # noqa: E402
from training.datasets.windows import IovnbdWindows  # noqa: E402

OUT = REPO_ROOT / "reports" / "phase6" / "evaluation.json"
BINS = {"model_a": [0, 2, 5, 10, 20, 30, 60], "model_b": [-10, -1.5, -0.5, 0.5, 1.5, 10]}
LOGVAR_CLAMP = (-8.0, 6.0)


@torch.no_grad()
def predict(model, ds, device, batch: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.to(device).eval()
    ms, ls, ys = [], [], []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False)
    for x, y in loader:
        m, lv = model(x.to(device))
        ms.append(m.cpu().numpy())
        ls.append(lv.clamp(*LOGVAR_CLAMP).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ms), np.concatenate(ls), np.concatenate(ys)


def metrics(mu: np.ndarray, lv: np.ndarray, y: np.ndarray, key: str, const: float) -> dict:
    err = mu - y
    ae = np.abs(err)
    sig = np.exp(0.5 * lv)
    z = err / sig
    q = np.quantile(sig, np.linspace(0, 1, 11))
    rel = []
    for lo, hi in pairwise(q):
        m = (sig >= lo) & (sig <= hi)
        if m.sum() >= 10:
            rel.append({"sigma_mean": float(sig[m].mean()), "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                        "n": int(m.sum())})
    ence = float(np.mean([abs(r["rmse"] - r["sigma_mean"]) / r["sigma_mean"] for r in rel])) if rel else None
    by_range = []
    edges = BINS[key]
    for lo, hi in pairwise(edges):
        m = (y >= lo) & (y < hi)
        if m.sum():
            by_range.append({"range": [lo, hi], "n": int(m.sum()), "mae": float(ae[m].mean()),
                             "bias": float(err[m].mean())})
    return {
        "n": len(y), "mae": float(ae.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
        "abs_err_percentiles": {p: float(np.percentile(ae, int(p[1:]))) for p in ("p50", "p90", "p95", "p99")},
        "constant_train_mean_mae": float(np.mean(np.abs(const - y))),
        "calibration": {"coverage_1sigma": float(np.mean(ae <= sig)), "coverage_2sigma": float(np.mean(ae <= 2 * sig)),
                        "coverage_3sigma": float(np.mean(ae <= 3 * sig)), "ideal": [0.683, 0.954, 0.997],
                        "z_std": float(np.std(z)), "z_mean": float(np.mean(z)), "ence": ence,
                        "reliability": rel},
        "by_target_range": by_range,
    }


def main() -> int:
    sel = json.loads((REPO_ROOT / "reports" / "phase6" / "selection.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "device": device.type,
              "protocol": "selection on VAL (sweep); TEST evaluated once here", "models": {}}
    for key, s in sel.items():
        name = s["selected"]
        model, rec, _ = load_checkpoint(name)
        cfg = load_training_config()
        W = rec["window_samples"]
        cfg["data"]["window_samples"] = W
        cfg["data"]["stride_eval"] = W
        verified = cfg["data"]["verified_only"][key]
        const = rec["baseline_constant_train_mean"]["value"]
        out = {"name": name, "arch": rec["arch"], "window_s": rec["window_s"], "n_params": rec["n_params"],
               "output": rec["output"], "selection_reason": s["reason"]}
        for split in ("val", "test"):
            ds = IovnbdWindows(split, rec["target"], cfg, augment=False, verified_only=verified,
                               allow_test=(split == "test"))
            mu, lv, y = predict(model, ds, device)
            out[split] = {"sessions": ds.sessions, "excluded": dict(ds.exclusions),
                          **metrics(mu, lv, y, key, const)}
            m = out[split]
            print(f"[{name}] {split}: n {m['n']} MAE {m['mae']:.4f} RMSE {m['rmse']:.4f} bias {m['bias']:+.4f} "
                  f"p95 {m['abs_err_percentiles']['p95']:.3f} | const {m['constant_train_mean_mae']:.4f} | "
                  f"cov {m['calibration']['coverage_1sigma']:.2f}/{m['calibration']['coverage_2sigma']:.2f}/"
                  f"{m['calibration']['coverage_3sigma']:.2f} z_std {m['calibration']['z_std']:.2f} "
                  f"ENCE {m['calibration']['ence']:.3f}")
        report["models"][key] = out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
