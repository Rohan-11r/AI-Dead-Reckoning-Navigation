"""One training run of Model A or B -- shared by scripts/train/train_all.py and the sweep."""

from __future__ import annotations

import copy
import time

import numpy as np

from training.datasets.windows import FEATURE_NAMES, IovnbdWindows
from training.models.registry import build, n_params
from training.preprocessing.columns import assert_model_targets
from training.trainers.trainer import fit

OUTPUTS = {
    "model_a": {"name": "forward_speed", "unit": "m/s", "range": [0.0, 60.0], "positive": True},
    "model_b": {"name": "longitudinal_acceleration", "unit": "m/s^2", "range": [-10.0, 10.0],
                "positive": False},
}


def train_one(cfg: dict, key: str, arch: str | None = None, window: int | None = None,
              hparams: dict | None = None, epochs: int | None = None,
              max_batches: int | None = None, val_max_batches: int | None = -1,
              name: str | None = None, log=print) -> dict:
    """Train ``key`` (model_a | model_b) with optional arch / window / hparam overrides.

    Validation always uses the eval stride (non-overlapping windows) on the VAL split.
    Returns the trainer result (history, best val metrics, checkpoint path, provenance).
    """
    cfg = copy.deepcopy(cfg)
    if window is not None:
        cfg["data"]["window_samples"] = int(window)
        cfg["data"]["stride_eval"] = int(window)  # eval stride = window (ml_pipeline sec 4)
    mc = dict(cfg["models"][key])
    if arch is not None:
        mc["arch"] = arch
    out = OUTPUTS[key]
    assert_model_targets([mc["target"], out["name"]])  # never a coordinate
    verified = cfg["data"]["verified_only"][key]
    t0 = time.time()
    train_ds = IovnbdWindows("train", mc["target"], cfg, augment=True, verified_only=verified,
                             seed=cfg["seed"])
    val_ds = IovnbdWindows("val", mc["target"], cfg, augment=False, verified_only=verified)
    load_s = round(time.time() - t0, 1)
    # ``hparams`` REPLACES the model's configured hyperparameters (so e.g. model_b's GRU
    # "layers" never leaks into a CNN of the sweep); without it the config is used as-is
    kw = dict(hparams) if hparams is not None else {k: v for k, v in mc.items() if k not in ("arch", "target")}
    model = build(mc["arch"], in_channels=train_ds.n_channels, positive=out["positive"], **kw)
    W = cfg["data"]["window_samples"]
    meta = {"model": key, "arch": mc["arch"], "hparams": kw, "target": mc["target"],
            "window_samples": W, "window_s": W * 0.1,
            "output": out, "inputs": FEATURE_NAMES, "input_shape": [len(FEATURE_NAMES), W],
            "feature_scaling": {"accel_scale_mps2": cfg["features"]["accel_scale_mps2"],
                                "gyro_scale_radps": cfg["features"]["gyro_scale_radps"]},
            "input_rule": "phone channels only (CAN rule)", "n_params": n_params(model),
            "seed": cfg["seed"], "train_data": train_ds.summary(), "val_data": val_ds.summary(),
            "data_load_s": load_s}
    y_val = np.array([val_ds.arrays[si]["y"][e] for si, e in val_ds.index])
    mu = train_ds.summary()["target_mean"]
    meta["baseline_constant_train_mean"] = {
        "value": mu, "val_mae": float(np.mean(np.abs(y_val - mu))) if y_val.size else None}
    name = name or f"{key}_{mc['arch']}"
    log(f"[{name}] {meta['n_params']:,} params | W={W} | train {len(train_ds):,} windows "
        f"({len(train_ds.sessions)} sessions) | val {len(val_ds):,} ({len(val_ds.sessions)}) | load {load_s}s")
    res = fit(model, train_ds, val_ds, cfg["train"], name, meta, epochs=epochs,
              max_batches=max_batches, val_max_batches=val_max_batches)
    res["_model"] = model
    return res
