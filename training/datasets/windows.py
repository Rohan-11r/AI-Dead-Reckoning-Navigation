"""Windowed IO-VNBD datasets for the Phase 5 models.

Rules enforced here (docs/ml_pipeline.md sec 3-6, AGENTS.md 2):
* INPUTS are phone channels only -- ``assert_model_inputs`` (the CAN rule); ``gt_*``
  columns are read as TARGETS and never enter the feature tensor.
* TARGETS are physical quantities -- ``assert_model_targets`` (no coordinates).
* A window never spans a data gap (any real dt > ``max_window_dt_s``) or a segment
  boundary, and every sample must have matched ground truth.
* A target-quality gate (VBOX satellites, speed range, finite value); every excluded
  window is COUNTED by reason in ``self.exclusions``.
* The split comes from the leak-free drive-level manifest (v2); the TEST split is refused
  unless explicitly requested.

Features (the phone mount is NOT fixed -- Phase 4): device-frame acc, gravity and the
gyro vector [0, 0, w_z] (only the vertical column is an angular rate, Phase 4), plus
mount-INVARIANT features |f|, f.g_hat, |f x g_hat|, w.g_hat. Scaled by fixed PHYSICAL
constants from the config (not data-fitted, so no statistic can leak across splits).
The Phase 3 per-axis statistics in models/normalization/ are deliberately NOT applied:
random-rotation augmentation makes per-device-axis statistics meaningless.

Augmentation (train only): random device rotation (yaw about gravity + tilt, or full
SO(3)), sensor noise, per-window accel bias, timestamp jitter -- applied BEFORE the
invariant features are computed, so they stay physically consistent.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from training.preprocessing.columns import assert_model_inputs, assert_model_targets
from training.preprocessing.config import load_dataset_config

DERIVED_TARGETS = {"gt_long_accel_from_speed_mps2"}
FEATURE_NAMES = ["acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z", "gyro_x", "gyro_y",
                 "gyro_z", "f_norm", "f_vertical", "f_horizontal", "w_vertical"]
ACCEL_DIFF_HALF_SAMPLES = 5  # central difference over +-0.5 s for the acceleration target


def random_rotation(rng: np.random.Generator, mode: str, max_tilt_deg: float) -> np.ndarray:
    """A random device-frame rotation matrix."""
    if mode == "none":
        return np.eye(3)
    if mode == "full":
        q = rng.normal(size=4)
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    yaw = rng.uniform(0, 2 * math.pi)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    axis_ang = rng.uniform(0, 2 * math.pi)
    k = np.array([math.cos(axis_ang), math.sin(axis_ang), 0.0])
    tilt = math.radians(rng.uniform(0, max_tilt_deg))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Rt = np.eye(3) + math.sin(tilt) * K + (1 - math.cos(tilt)) * K @ K
    return Rt @ Rz


def make_features(acc: np.ndarray, grav: np.ndarray, gyro_vec: np.ndarray, fcfg: dict) -> np.ndarray:
    """(W, 3) x3 -> (C, W) float32 feature tensor."""
    g_s, w_s = fcfg["accel_scale_mps2"], fcfg["gyro_scale_radps"]
    parts = [acc / g_s, grav / g_s, gyro_vec / w_s]
    if fcfg["invariant"]:
        gn = np.linalg.norm(grav, axis=1, keepdims=True)
        g_hat = grav / np.where(gn > 0, gn, 1.0)
        f_norm = np.linalg.norm(acc, axis=1, keepdims=True)
        f_vert = np.sum(acc * g_hat, axis=1, keepdims=True)
        f_horiz = np.linalg.norm(np.cross(acc, g_hat), axis=1, keepdims=True)
        w_vert = np.sum(gyro_vec * g_hat, axis=1, keepdims=True)
        parts += [f_norm / g_s - 1.0, f_vert / g_s - 1.0, f_horiz / g_s, w_vert / w_s]
    return np.concatenate(parts, axis=1).T.astype(np.float32)


class IovnbdWindows(Dataset):
    """Windows ending at row ``end`` (inclusive) of a session; target taken at ``end``."""

    def __init__(self, split: str, target: str, cfg: dict, augment: bool, verified_only: bool,
                 allow_test: bool = False, sessions: list[str] | None = None,
                 stride: int | None = None, seed: int = 0) -> None:
        if split == "test" and not allow_test:
            raise PermissionError("the held-out TEST split is refused unless allow_test=True")
        assert_model_targets([target])
        fc, dc = cfg["features"], cfg["data"]
        self.raw_channels = list(fc["raw_channels"])
        self.gyro_channel = fc["gyro_vertical_channel"]
        assert_model_inputs([*self.raw_channels, self.gyro_channel])
        self.cfg, self.target, self.augment = cfg, target, augment
        self.W = int(dc["window_samples"])
        self.stride = int(stride or (dc["stride_train"] if split == "train" else dc["stride_eval"]))
        self.rng = np.random.default_rng(seed)
        ds_cfg = load_dataset_config()
        manifest = json.loads(ds_cfg.split_manifest.read_text(encoding="utf-8"))
        chosen = sessions if sessions is not None else manifest["splits"][split]
        wrong = [s for s in chosen if manifest["sessions"][s]["split"] != split]
        if wrong:
            raise ValueError(f"sessions not in split {split!r}: {wrong}")
        if verified_only:
            rep = json.loads((ds_cfg.reports_dir / "phase3" / "preprocessing_report.json")
                             .read_text(encoding="utf-8"))
            ok = {s["session_id"] for s in rep["sessions"] if s["alignment_verified"]}
            chosen = [s for s in chosen if s in ok]
        self.sessions = list(chosen)
        self.exclusions: Counter = Counter()
        self.arrays: list[dict] = []
        self.index: list[tuple[int, int]] = []
        gt_col = "gt_speed_mps" if target in DERIVED_TARGETS else target
        for si, sid in enumerate(self.sessions):
            d = pd.read_parquet(ds_cfg.synced_root / f"{sid}.parquet",
                                columns=[*self.raw_channels, self.gyro_channel, "t_utc", "dt_s",
                                         "segment_id", "gt_valid", "gt_sats", gt_col])
            y = self._target(d, gt_col)
            arr = {"acc": d[self.raw_channels[:3]].to_numpy(np.float64),
                   "grav": d[self.raw_channels[3:6]].to_numpy(np.float64),
                   "wz": d[self.gyro_channel].to_numpy(np.float64), "y": y}
            self.arrays.append(arr)
            self._index_session(si, d, y)

    def _target(self, d: pd.DataFrame, gt_col: str) -> np.ndarray:
        v = d[gt_col].to_numpy(np.float64)
        if self.target not in DERIVED_TARGETS:
            return v
        # longitudinal acceleration = d(speed)/dt, central difference over +-0.5 s using
        # REAL timestamps, never crossing a segment boundary
        t = (d["t_utc"] - d["t_utc"].iloc[0]).dt.total_seconds().to_numpy()
        seg = d["segment_id"].to_numpy()
        k = ACCEL_DIFF_HALF_SAMPLES
        out = np.full(len(v), np.nan)
        if len(v) > 2 * k:
            same = seg[2 * k:] == seg[:-2 * k]
            dt = t[2 * k:] - t[:-2 * k]
            out[k:-k] = np.where(same & (dt > 0), (v[2 * k:] - v[:-2 * k]) / np.where(dt > 0, dt, 1), np.nan)
        return out

    def _index_session(self, si: int, d: pd.DataFrame, y: np.ndarray) -> None:
        dc, W = self.cfg["data"], self.W
        dt = d["dt_s"].to_numpy()
        seg = d["segment_id"].to_numpy()
        valid = d["gt_valid"].to_numpy(bool)
        sats = d["gt_sats"].astype("float64").to_numpy()
        # rows whose dt breaks contiguity (NaN = segment start, or dt too large)
        bad_dt = ~(dt <= dc["max_window_dt_s"])
        bad_dt_cum = np.concatenate([[0], np.cumsum(bad_dt)])
        invalid_cum = np.concatenate([[0], np.cumsum(~valid)])
        for end in range(W - 1, len(d), self.stride):
            start = end - W + 1
            if seg[start] != seg[end] or bad_dt_cum[end + 1] - bad_dt_cum[start + 1] > 0:
                self.exclusions["spans_gap_or_segment"] += 1
                continue
            if invalid_cum[end + 1] - invalid_cum[start] > (1 - dc["min_gt_valid_fraction"]) * W:
                self.exclusions["ground_truth_unmatched"] += 1
                continue
            if not np.isfinite(y[end]):
                self.exclusions["target_not_finite"] += 1
                continue
            if not (sats[end] >= dc["min_vbox_sats"]):
                self.exclusions["target_low_vbox_sats"] += 1
                continue
            if self.target == "gt_speed_mps" and not (0 <= y[end] <= dc["max_speed_mps"]):
                self.exclusions["target_out_of_range"] += 1
                continue
            self.index.append((si, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        si, end = self.index[i]
        a = self.arrays[si]
        sl = slice(end - self.W + 1, end + 1)
        acc, grav = a["acc"][sl].copy(), a["grav"][sl].copy()
        gyro = np.zeros((self.W, 3))
        gyro[:, 2] = a["wz"][sl]
        if self.augment and self.cfg["augmentation"]["enabled"]:
            acc, grav, gyro = self._augment(acc, grav, gyro)
        x = make_features(acc, grav, gyro, self.cfg["features"])
        return torch.from_numpy(x), torch.tensor(a["y"][end], dtype=torch.float32)

    def _augment(self, acc, grav, gyro):
        ac, rng, W = self.cfg["augmentation"], self.rng, self.W
        if ac["time_jitter_s"] > 0:
            grid = np.arange(W, dtype=np.float64)
            tj = np.clip(grid + rng.normal(0, ac["time_jitter_s"] / 0.1, W), 0, W - 1)
            acc = np.stack([np.interp(tj, grid, acc[:, k]) for k in range(3)], axis=1)
            grav = np.stack([np.interp(tj, grid, grav[:, k]) for k in range(3)], axis=1)
            gyro = np.stack([np.interp(tj, grid, gyro[:, k]) for k in range(3)], axis=1)
        R = random_rotation(rng, ac["rotation"], ac["max_tilt_deg"])
        acc, grav, gyro = acc @ R.T, grav @ R.T, gyro @ R.T
        acc = acc + rng.uniform(-ac["accel_bias_mps2"], ac["accel_bias_mps2"], 3)
        acc = acc + rng.normal(0, ac["accel_noise_mps2"], acc.shape)
        gyro = gyro + rng.normal(0, ac["gyro_noise_radps"], gyro.shape)
        return acc, grav, gyro

    def summary(self) -> dict:
        y = np.array([self.arrays[si]["y"][e] for si, e in self.index]) if self.index else np.array([])
        return {"sessions": len(self.sessions), "windows": len(self.index),
                "excluded": dict(self.exclusions), "target": self.target,
                "target_mean": float(y.mean()) if y.size else None,
                "target_std": float(y.std()) if y.size else None,
                "window_samples": self.W, "stride": self.stride, "augment": self.augment}

    @property
    def n_channels(self) -> int:
        return len(FEATURE_NAMES) if self.cfg["features"]["invariant"] else 9
