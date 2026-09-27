#!/usr/bin/env python3
"""Train the Phase 5 models (A: forward speed, B: longitudinal acceleration).

    .venv/Scripts/python.exe scripts/train/train_all.py --smoke          # 1 epoch, few batches
    .venv/Scripts/python.exe scripts/train/train_all.py                  # full (configs/training.yaml)
    .venv/Scripts/python.exe scripts/train/train_all.py --models a --device cpu

MODEL A  forward speed [m/s] + log-variance       target gt_speed_mps
MODEL B  longitudinal acceleration [m/s^2] + log-variance, target d(gt_speed)/dt: a learned,
         mount-independent replacement for the raw accelerometer's along-track component --
         the channel whose errors dominate the Phase 4 outage drift.

Inputs are phone channels only; gt_* are targets only (checked in the dataset). Train on
the TRAIN split with augmentation, validate on VAL without; the TEST split is not loaded.
Device: configs/training.yaml `train.device` (auto -> CUDA if available, else CPU); CUDA
out-of-memory halves the batch, then falls back to CPU (training.trainers.trainer).
Writes reports/phase5/train_<smoke|limited|full>.json and models/metadata/<model>.json
("limited" = any run capped by --epochs / --max-batches that is not --smoke).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from training.configs.loader import load_training_config  # noqa: E402
from training.datasets.windows import FEATURE_NAMES, IovnbdWindows  # noqa: E402
from training.models.registry import build, n_params  # noqa: E402
from training.preprocessing.columns import assert_model_targets  # noqa: E402
from training.trainers.trainer import fit, model_card, seed_everything  # noqa: E402

OUTPUTS = {
    "model_a": {"name": "forward_speed", "unit": "m/s", "range": [0.0, 60.0], "positive": True},
    "model_b": {"name": "longitudinal_acceleration", "unit": "m/s^2", "range": [-10.0, 10.0],
                "positive": False},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default="a,b")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, --max-batches batches")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None, help="override train.device (auto|cuda|cpu)")
    args = ap.parse_args()

    cfg = load_training_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    epochs = 1 if args.smoke else args.epochs
    max_batches = args.max_batches if args.max_batches is not None else (20 if args.smoke else None)
    seed_everything(cfg["seed"])
    mode = "smoke" if args.smoke else ("limited" if (args.max_batches or args.epochs) else "full")
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "mode": mode,
              "epochs_cap": epochs,
              "config": cfg["_source"], "python": platform.python_version(), "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(),
              "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "max_batches": max_batches, "models": {}}

    for key in [f"model_{m.strip()}" for m in args.models.split(",")]:
        mc = dict(cfg["models"][key])
        out = OUTPUTS[key]
        assert_model_targets([mc["target"], out["name"]])  # never a coordinate
        verified = cfg["data"]["verified_only"][key]
        t0 = time.time()
        train_ds = IovnbdWindows("train", mc["target"], cfg, augment=True, verified_only=verified,
                                 seed=cfg["seed"])
        val_ds = IovnbdWindows("val", mc["target"], cfg, augment=False, verified_only=verified)
        load_s = round(time.time() - t0, 1)
        kw = {k: v for k, v in mc.items() if k not in ("arch", "target")}
        model = build(mc["arch"], in_channels=train_ds.n_channels, positive=out["positive"], **kw)
        meta = {"model": key, "arch": mc["arch"], "hparams": kw, "target": mc["target"],
                "output": out, "inputs": FEATURE_NAMES, "input_rule": "phone channels only (CAN rule)",
                "n_params": n_params(model), "seed": cfg["seed"],
                "train_data": train_ds.summary(), "val_data": val_ds.summary(), "data_load_s": load_s}
        print(f"[{key}] {mc['arch']} {meta['n_params']:,} params | train {len(train_ds):,} windows "
              f"({len(train_ds.sessions)} sessions) | val {len(val_ds):,} ({len(val_ds.sessions)}) | load {load_s}s")
        # trivial reference: always predict the TRAIN target mean -- a model must beat this
        import numpy as np
        y_val = np.array([val_ds.arrays[si]["y"][e] for si, e in val_ds.index])
        meta["baseline_constant_train_mean"] = {
            "value": train_ds.summary()["target_mean"],
            "val_mae": float(np.mean(np.abs(y_val - train_ds.summary()["target_mean"]))) if y_val.size else None}
        res = fit(model, train_ds, val_ds, cfg["train"], f"{key}_{mc['arch']}", meta,
                  epochs=epochs, max_batches=max_batches)
        last = res["history"][-1] if res["history"] else {}
        print(f"[{key}] device {res['device_final']} batch {res['batch_size_final']} epochs {res['epochs_run']} "
              f"train_loss {last.get('train_loss')} val {json.dumps(last.get('val'))} fallbacks {res['fallbacks']}")
        print(f"[{key}] reference (constant train mean) val MAE {meta['baseline_constant_train_mean']['val_mae']}")
        card = model_card(f"{key}_{mc['arch']}", {k: v for k, v in res.items() if k != "history"},
                          REPO_ROOT / "models" / "metadata")
        report["models"][key] = {**res, "model_card": str(card.relative_to(REPO_ROOT))}

    out_dir = REPO_ROOT / "reports" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"train_{report['mode']}.json"
    path.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"report -> {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
