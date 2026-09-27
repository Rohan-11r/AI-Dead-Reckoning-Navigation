"""Runtime use of the exported AI models inside the navigation filter.

* ``OnnxSequenceModel`` runs an exported model (models/exported/<name>.onnx) exactly as
  validated in Phase 6: the model card's SHA-256 must match the file, and inputs are built
  by navcore.fusion.features (the single feature implementation shared with training).
* ``ImuWindow`` is the rolling input buffer; it refuses to produce a window that spans a
  data gap (real dt > max_dt), mirroring the training data rule.
* ``speed_innovation`` -- Model A as a measurement of horizontal ground speed
  (docs/navigation_math.md sec 8.5):  nu = s_AI - |v_h|,  H = [0, v_h^T/|v_h|, 0 ...].
  Singular at |v_h| = 0 -> skipped below ``min_speed`` (ZUPT covers rest).
* ``correct_specific_force`` -- Model B applied to the raw IMU BEFORE predict(): the
  along-track linear acceleration in the navigation frame (direction = the estimated
  horizontal velocity, so no mount yaw is needed) is replaced by the inverse-variance
  blend of the IMU value and Model B's estimate; the result is rotated back to the body.

AI outputs are measurements with covariance and physical units only -- never positions
(AGENTS.md 2.1). Their inputs are phone channels only (SIH26168: no CAN/OBD).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from navcore.filtering.ekf import N_STATES, ErrorStateEKF
from navcore.fusion.features import FEATURE_NAMES, make_features
from navcore.geometry.quaternion import q_conj, q_rotate

LOGVAR_CLAMP = (-8.0, 6.0)


class ModelIntegrityError(RuntimeError):
    """The ONNX file does not match the model card it was validated with."""


class OnnxSequenceModel:
    def __init__(self, onnx_path: Path, card_path: Path | None = None) -> None:
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        card_path = Path(card_path) if card_path else onnx_path.with_suffix("").with_suffix(".model_card.json")
        self.card = json.loads(card_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        if digest != self.card["files"]["onnx_sha256"]:
            raise ModelIntegrityError(f"{onnx_path.name}: SHA-256 differs from its model card")
        if self.card["input"]["features"] != FEATURE_NAMES:
            raise ModelIntegrityError(f"{onnx_path.name}: feature order differs from navcore.fusion.features")
        self.window = int(self.card["input"]["shape"][2])
        sc = self.card["input"]["scaling"]
        self.accel_scale, self.gyro_scale = sc["accel_scale_mps2"], sc["gyro_scale_radps"]
        self.quantity = self.card["outputs"]["mean"]["quantity"]
        self.unit = self.card["outputs"]["mean"]["unit"]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def predict(self, acc: np.ndarray, grav: np.ndarray, wz: np.ndarray) -> tuple[float, float]:
        """One window (W, 3), (W, 3), (W,) -> (mean, variance) in the card's unit."""
        gyro = np.zeros((len(wz), 3))
        gyro[:, 2] = wz
        x = make_features(acc, grav, gyro, self.accel_scale, self.gyro_scale)[None]
        mean, logvar = self.sess.run(["mean", "logvar"], {"window": x})
        lv = float(np.clip(logvar[0], *LOGVAR_CLAMP))
        return float(mean[0]), math.exp(lv)


@dataclass
class ImuWindow:
    size: int
    max_dt_s: float = 0.15

    def __post_init__(self) -> None:
        self._t: deque = deque(maxlen=self.size)
        self._acc: deque = deque(maxlen=self.size)
        self._grav: deque = deque(maxlen=self.size)
        self._wz: deque = deque(maxlen=self.size)

    def push(self, t_s: float, acc, grav, wz: float) -> None:
        if self._t and not (0 < t_s - self._t[-1] <= self.max_dt_s):
            self.clear()  # a gap (or a non-increasing time): start a fresh window
        self._t.append(t_s)
        self._acc.append(np.asarray(acc, dtype=np.float64))
        self._grav.append(np.asarray(grav, dtype=np.float64))
        self._wz.append(float(wz))

    def clear(self) -> None:
        for d in (self._t, self._acc, self._grav, self._wz):
            d.clear()

    def ready(self) -> bool:
        return len(self._t) == self.size

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array(self._acc), np.array(self._grav), np.array(self._wz)


def speed_innovation(ekf: ErrorStateEKF, speed_mps: float, min_speed_mps: float = 1.0
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """(nu, H) for an AI horizontal-speed measurement, or None when |v_h| is too small."""
    v = ekf.nominal.v_enu_mps
    sh = math.hypot(v[0], v[1])
    if sh < min_speed_mps:
        return None
    H = np.zeros((1, N_STATES))
    H[0, 3], H[0, 4] = v[0] / sh, v[1] / sh
    return np.array([speed_mps - sh]), H


def correct_specific_force(f_b: np.ndarray, q_nb: np.ndarray, v_enu: np.ndarray, g_n: np.ndarray,
                           a_long_ai: float, var_ai: float, var_imu: float,
                           min_speed_mps: float = 3.0) -> tuple[np.ndarray, bool]:
    """Model B correction of the body specific force before predict().

    Along-track unit vector u = v_h / |v_h| (navigation frame). The IMU's along-track
    LINEAR acceleration a_imu = u . (R f_b + g_n) is replaced by the inverse-variance blend
    a = (a_imu / var_imu + a_ai / var_ai) / (1/var_imu + 1/var_ai); the other components of
    f are untouched. Returns (corrected f_b, applied?).
    """
    sh = math.hypot(v_enu[0], v_enu[1])
    if sh < min_speed_mps or not (np.isfinite(a_long_ai) and var_ai > 0 and var_imu > 0):
        return np.asarray(f_b, dtype=np.float64), False
    u = np.array([v_enu[0] / sh, v_enu[1] / sh, 0.0])
    f_n = q_rotate(q_nb, f_b)
    a_imu = float(u @ (f_n + g_n))
    a_blend = (a_imu / var_imu + a_long_ai / var_ai) / (1.0 / var_imu + 1.0 / var_ai)
    f_n_corr = f_n + (a_blend - a_imu) * u
    return q_rotate(q_conj(q_nb), f_n_corr), True
