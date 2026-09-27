"""Load configs/dataset.yaml -- the one place dataset paths are defined.

    from training.preprocessing.config import load_dataset_config
    cfg = load_dataset_config()
    cfg.raw_root  # -> absolute Path

Missing keys raise; nothing silently falls back to a default path (AGENTS.md 5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "dataset.yaml"
ENV_RAW_ROOT = "SIH26168_DATASET_ROOT"
_KEYS = ("raw_root", "processed_root", "synced_root", "splits_dir", "split_manifest",
         "normalization_dir", "reports_dir")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_root: Path
    processed_root: Path
    synced_root: Path
    splits_dir: Path
    split_manifest: Path
    normalization_dir: Path
    reports_dir: Path
    source: Path


def _resolve(p: str | os.PathLike) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_dataset_config(path: Path | None = None) -> DatasetConfig:
    src = Path(path) if path else DEFAULT_CONFIG
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1:
        raise ValueError(f"{src}: unsupported schema_version {doc.get('schema_version')!r}")
    ds = doc["dataset"]
    missing = [k for k in (*_KEYS, "name") if k not in ds]
    if missing:
        raise KeyError(f"{src}: dataset config missing keys {missing}")
    paths = {k: _resolve(ds[k]) for k in _KEYS}
    if os.environ.get(ENV_RAW_ROOT):
        paths["raw_root"] = _resolve(os.environ[ENV_RAW_ROOT])
    return DatasetConfig(name=ds["name"], source=src, **paths)
