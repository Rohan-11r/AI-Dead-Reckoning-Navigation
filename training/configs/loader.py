"""Load configs/training.yaml. Every key is required -- no hyperparameter has a code default."""

from __future__ import annotations

from pathlib import Path

import yaml

from training.preprocessing.config import REPO_ROOT

DEFAULT_TRAINING_CONFIG = REPO_ROOT / "configs" / "training.yaml"

REQUIRED = {
    "schema_version": None, "seed": None,
    "data": ["window_samples", "stride_train", "stride_eval", "max_window_dt_s",
             "min_gt_valid_fraction", "min_vbox_sats", "max_speed_mps", "verified_only"],
    "features": ["raw_channels", "gyro_vertical_channel", "invariant", "accel_scale_mps2",
                 "gyro_scale_radps"],
    "augmentation": ["enabled", "accel_noise_mps2", "gyro_noise_radps", "accel_bias_mps2",
                     "time_jitter_s", "rotation", "max_tilt_deg"],
    "models": ["model_a", "model_b"],
    "train": ["batch_size", "min_batch_size", "lr", "weight_decay", "epochs", "grad_clip",
              "huber_delta", "nll_weight", "logvar_clamp", "early_stopping_patience",
              "num_workers", "device"],
}


class ConfigError(KeyError):
    pass


def load_training_config(path: Path | None = None) -> dict:
    src = Path(path) if path else DEFAULT_TRAINING_CONFIG
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != 1:
        raise ConfigError(f"{src}: unsupported schema_version {cfg.get('schema_version')!r}")
    missing = []
    for section, keys in REQUIRED.items():
        if section not in cfg:
            missing.append(section)
            continue
        for k in keys or []:
            if k not in cfg[section]:
                missing.append(f"{section}.{k}")
    for name in ("model_a", "model_b"):
        for k in ("arch", "target", "hidden", "dropout"):
            if k not in cfg.get("models", {}).get(name, {}):
                missing.append(f"models.{name}.{k}")
    if missing:
        raise ConfigError(f"{src}: missing required keys {missing}")
    if cfg["augmentation"]["rotation"] not in ("yaw_tilt", "full", "none"):
        raise ConfigError("augmentation.rotation must be yaw_tilt | full | none")
    cfg["_source"] = str(src)
    return cfg
