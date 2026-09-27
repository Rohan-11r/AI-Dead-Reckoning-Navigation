"""IMU normalisation statistics -- computed from the TRAIN split only.

docs/ml_pipeline.md section 3: statistics fitted on val/test leak their distribution into
training. ``compute_imu_stats`` therefore takes the split manifest, uses ONLY
``splits["train"]``, and refuses any session that the manifest places elsewhere. The
output records the exact sessions used and the manifest's SHA-256, so the file is
traceable to one split and can be re-derived by anyone (AGENTS.md 2.2).

Statistic: per-channel mean and POPULATION standard deviation (ddof = 0) over all valid
samples of the train sessions, accumulated in float64 with a two-pass algorithm.
Shipped to Android with the model (AGENTS.md 4), hence plain JSON.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from training.preprocessing.columns import IMU_COLUMNS, assert_model_inputs
from training.preprocessing.sync import SYNCED_UNITS

NORM_VERSION = 1


class SplitLeakError(RuntimeError):
    """A non-train session was offered to a train-only computation."""


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_imu_stats(manifest_path: Path, synced_root: Path, sessions: list[str] | None = None,
                      channels: list[str] = IMU_COLUMNS) -> dict:
    """Mean/std of ``channels`` over the manifest's train sessions (or the given subset of
    them). Raises SplitLeakError if any requested session is not in train."""
    assert_model_inputs(channels)  # never normalise ground truth into a model input
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    train = set(manifest["splits"]["train"])
    use = sorted(train) if sessions is None else sorted(sessions)
    leaked = [s for s in use if s not in train]
    if leaked:
        where = {s: manifest["sessions"].get(s, {}).get("split", "unknown") for s in leaked}
        raise SplitLeakError(f"refusing to fit normalisation on non-train sessions: {where}")

    frames = [pd.read_parquet(Path(synced_root) / f"{s}.parquet", columns=channels) for s in use]
    X = np.concatenate([f.to_numpy(dtype=np.float64) for f in frames], axis=0)
    finite = np.isfinite(X)
    n = finite.sum(axis=0)
    mean = np.where(finite, X, 0.0).sum(axis=0) / n
    var = np.where(finite, (X - mean) ** 2, 0.0).sum(axis=0) / n  # second pass
    std = np.sqrt(var)
    if np.any(std <= 0) or np.any(~np.isfinite(std)):
        raise ValueError(f"degenerate channel std: {dict(zip(channels, std, strict=True))}")
    return {
        "channels": channels,
        "units": {c: SYNCED_UNITS[c] for c in channels},
        "mean": dict(zip(channels, map(float, mean), strict=True)),
        "std": dict(zip(channels, map(float, std), strict=True)),
        "n_samples": dict(zip(channels, map(int, n), strict=True)),
        "n_nonfinite_excluded": dict(zip(channels, map(int, len(X) - n), strict=True)),
        "sessions_used": use,
        "split": "train",
        "split_manifest": str(Path(manifest_path).as_posix()),
        "split_manifest_sha256": manifest_sha256(manifest_path),
        "statistic": "mean and population std (ddof=0), float64 two-pass",
    }


def write_stats(stats: dict, out_dir: Path, provenance: dict) -> tuple[Path, Path]:
    """Write imu_mean.json and imu_std.json; each is self-describing."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "norm_version": NORM_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        **provenance,
        **{k: stats[k] for k in ("channels", "units", "n_samples", "n_nonfinite_excluded",
                                 "sessions_used", "split", "split_manifest",
                                 "split_manifest_sha256", "statistic")},
    }
    paths = []
    for name, key in (("imu_mean.json", "mean"), ("imu_std.json", "std")):
        p = out_dir / name
        p.write_text(json.dumps({"quantity": key, "values": stats[key], **common}, indent=1) + "\n",
                     encoding="utf-8")
        paths.append(p)
    return paths[0], paths[1]
