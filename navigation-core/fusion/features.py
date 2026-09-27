"""Model input features -- the SINGLE implementation shared by training and runtime.

training.datasets.windows imports this module, and navcore.fusion.ai_models builds the
ONNX input with it, so the model sees bit-identical features in training and on-device
(AGENTS.md 4: one specification, two implementations; the Android port must reproduce
exactly this function, as documented in each exported model card).

Channels (C = 13), each a time series over the window, oldest sample first:
    acc_x..z   device specific force / accel_scale
    grav_x..z  Android GRAVITY vector / accel_scale
    gyro_x..z  [0, 0, w_z] / gyro_scale   (only the vertical gyro is a real rate, Phase 4)
    f_norm       |f| / accel_scale - 1
    f_vertical   (f . g_hat) / accel_scale - 1,   g_hat = grav / |grav|
    f_horizontal |f x g_hat| / accel_scale
    w_vertical   (w . g_hat) / gyro_scale
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z", "gyro_x", "gyro_y",
                 "gyro_z", "f_norm", "f_vertical", "f_horizontal", "w_vertical"]


def make_features(acc: np.ndarray, grav: np.ndarray, gyro_vec: np.ndarray, accel_scale: float,
                  gyro_scale: float, invariant: bool = True) -> np.ndarray:
    """(W, 3) acc, grav, gyro vector -> (C, W) float32 model input."""
    parts = [acc / accel_scale, grav / accel_scale, gyro_vec / gyro_scale]
    if invariant:
        gn = np.linalg.norm(grav, axis=1, keepdims=True)
        g_hat = grav / np.where(gn > 0, gn, 1.0)
        f_norm = np.linalg.norm(acc, axis=1, keepdims=True)
        f_vert = np.sum(acc * g_hat, axis=1, keepdims=True)
        f_horiz = np.linalg.norm(np.cross(acc, g_hat), axis=1, keepdims=True)
        w_vert = np.sum(gyro_vec * g_hat, axis=1, keepdims=True)
        parts += [f_norm / accel_scale - 1.0, f_vert / accel_scale - 1.0, f_horiz / accel_scale,
                  w_vert / gyro_scale]
    return np.concatenate(parts, axis=1).T.astype(np.float32)
