"""Non-holonomic constraints: a road vehicle does not slide sideways or fly.

docs/navigation_math.md sec 9. Pseudo-measurements on the VEHICLE-frame velocity
(x forward, y left, z up):  v^v_y = 0 (lateral),  v^v_z = 0 (vertical).

Under the filter's truth-minus-estimate convention (navcore.filtering.ekf):
    v^b_true = R_true^T v_true ~= v_hat^b + R_hat^T dv + R_hat^T [v_hat]x psi
so for rows S of R_vb:
    nu = -S R_vb v_hat^b ,   H = S R_vb [ 0  R_hat^T  R_hat^T [v_hat]x  0  0 ]
(sec 9 prints -R_n^b [v^n]x psi for the opposite convention; tests/navigation/test_nhc.py
checks this sign by finite differences.)

Mount dependence -- the Phase 4 finding that drives this design:
* the VERTICAL row of R_vb is the third row of the gravity-levelling rotation; the mount
  YAW (a rotation about that axis) does not change it. Vertical NHC therefore needs only
  levelling, which is always available.
* the LATERAL row needs the mount yaw, observable in only 8/68 IO-VNBD sessions and not
  rigid. Lateral NHC runs ONLY with an accepted mount estimate (sec 9: "never with an
  assumed mounting"); otherwise it is skipped and counted.

Relaxation (sec 9): skipped above a lateral-acceleration threshold (hard cornering) and
below a minimum speed. Every application is NIS-gated by the filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from navcore.filtering.ekf import A_, N_STATES, V_, ErrorStateEKF
from navcore.geometry.quaternion import q_to_dcm
from navcore.geometry.rotation import skew


@dataclass(frozen=True)
class NhcConfig:
    sigma_lateral_mps: float = 0.3
    sigma_vertical_mps: float = 0.3
    min_speed_mps: float = 2.0
    max_lateral_accel_mps2: float = 3.0  # hard cornering: constraint relaxed


def nhc_innovation(ekf: ErrorStateEKF, R_vb: np.ndarray, rows: tuple[int, ...]
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(nu, H) for the selected vehicle-frame axes (1 = lateral, 2 = vertical)."""
    s = ekf.nominal
    R_nb = q_to_dcm(s.q_nb)
    v = s.v_enu_mps
    v_b = R_nb.T @ v
    S = np.asarray(R_vb)[list(rows), :]  # (k, 3): selected vehicle axes expressed in body
    nu = -(S @ v_b)
    H = np.zeros((len(rows), N_STATES))
    H[:, V_] = S @ R_nb.T
    H[:, A_] = S @ R_nb.T @ skew(v)
    return nu, H


def apply_nhc(ekf: ErrorStateEKF, t_s: float, R_vb: np.ndarray, lateral_allowed: bool,
              lateral_accel_mps2: float, cfg: NhcConfig) -> dict:
    """Apply vertical (+ lateral when allowed) NHC. Returns what happened, for counting."""
    speed = float(np.linalg.norm(ekf.nominal.v_enu_mps[:2]))
    if speed < cfg.min_speed_mps:
        return {"applied": False, "reason": "low_speed"}
    if abs(lateral_accel_mps2) > cfg.max_lateral_accel_mps2:
        return {"applied": False, "reason": "hard_cornering"}
    rows, sig = [2], [cfg.sigma_vertical_mps]
    if lateral_allowed:
        rows, sig = [1, 2], [cfg.sigma_lateral_mps, cfg.sigma_vertical_mps]
    nu, H = nhc_innovation(ekf, R_vb, tuple(rows))
    ok = ekf.update("nhc", t_s, nu, H, np.diag(np.square(sig)))
    return {"applied": ok, "reason": "accepted" if ok else "gated", "lateral": lateral_allowed}
