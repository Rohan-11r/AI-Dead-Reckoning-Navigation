"""Unit quaternions -- the internal attitude representation.

Convention (docs/navigation_math.md section 4.1 -- declared once, asserted by tests):

* Storage order ``q = [w, x, y, z]``, scalar FIRST.
* Hamilton product. ``q_mul(q1, q2)`` composes rotations applying ``q2`` FIRST.
* A quaternion ``q_ab`` represents the rotation matrix ``R_b^a``: it takes a vector
  expressed in frame ``b`` and returns it expressed in frame ``a``:
  ``v_a = q_ab (x) [0, v_b] (x) q_ab*`` == ``q_rotate(q_ab, v_b)``.
* Composition follows the frames: ``q_ac = q_mul(q_ab, q_bc)``.
* Canonical sign: ``q`` and ``-q`` are the same rotation; functions that CREATE
  quaternions from other representations return ``w >= 0``.

All functions accept a single quaternion ``(4,)`` or a stack ``(..., 4)`` and broadcast
like numpy. float64 throughout (AGENTS.md 4: accumulators are double).
"""

from __future__ import annotations

import numpy as np

from navcore.common.constants import SMALL_ANGLE_THRESHOLD_RAD, UNIT_QUATERNION_TOL

Q_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _as_q(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must have last dimension 4, got shape {q.shape}")
    return q


def _as_v(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"vector must have last dimension 3, got shape {v.shape}")
    return v


def q_norm(q) -> np.ndarray:
    return np.linalg.norm(_as_q(q), axis=-1)


def q_normalize(q) -> np.ndarray:
    """Scale to unit norm. Raises on a zero quaternion -- never returns NaN silently."""
    q = _as_q(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n == 0.0):
        raise ValueError("cannot normalise a zero quaternion")
    return q / n


def is_unit(q, tol: float = UNIT_QUATERNION_TOL) -> bool:
    return bool(np.all(np.abs(q_norm(q) - 1.0) <= tol))


def q_canonical(q) -> np.ndarray:
    """Return the representative with w >= 0 (q and -q are the same rotation)."""
    q = _as_q(q)
    return np.where(q[..., :1] < 0.0, -q, q)


def q_conj(q) -> np.ndarray:
    q = _as_q(q)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def q_inv(q) -> np.ndarray:
    """General inverse; equals the conjugate for unit quaternions."""
    q = _as_q(q)
    return q_conj(q) / np.sum(q * q, axis=-1, keepdims=True)


def q_mul(p, q) -> np.ndarray:
    """Hamilton product p (x) q. As rotations: apply q first, then p."""
    p, q = _as_q(p), _as_q(q)
    pw, px, py, pz = np.moveaxis(p, -1, 0)
    qw, qx, qy, qz = np.moveaxis(q, -1, 0)
    return np.stack([
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ], axis=-1)


def q_rotate(q, v) -> np.ndarray:
    """Rotate vector(s) ``v`` by unit quaternion(s) ``q``: returns R(q) v.

    Uses the expanded form v' = v + 2w (u x v) + 2 u x (u x v), u = vector part, which
    equals q (x) [0, v] (x) q* for unit q.
    """
    q, v = _as_q(q), _as_v(v)
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def q_to_dcm(q) -> np.ndarray:
    """Rotation matrix R(q), section 4.4: R = (w^2 - v.v) I + 2 v v^T + 2 w [v]x."""
    q = q_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def dcm_to_q(R) -> np.ndarray:
    """Rotation matrix -> unit quaternion (Shepperd's method; numerically stable for all
    rotations including 180 deg). Returns the canonical w >= 0 representative."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must be (...,3,3), got {R.shape}")
    flat = R.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4))
    for i, m in enumerate(flat):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        # pick the largest of (w, x, y, z)^2 * 4 to divide by
        cand = np.array([tr, m[0, 0], m[1, 1], m[2, 2]])
        k = int(np.argmax(cand))
        if k == 0:
            s = 2.0 * np.sqrt(1.0 + tr)
            q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif k == 1:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif k == 2:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        out[i] = q
    out = q_canonical(q_normalize(out))
    return out.reshape((*R.shape[:-2], 4))


def q_from_rotvec(rv) -> np.ndarray:
    """Exponential map: rotation vector (axis * angle, rad) -> unit quaternion.

    Section 4.3: for |rv| below SMALL_ANGLE_THRESHOLD_RAD the (sin(t/2)/t) factor is
    evaluated by its Taylor series instead of 0/0.
    """
    rv = _as_v(rv)
    theta = np.linalg.norm(rv, axis=-1, keepdims=True)
    small = theta < SMALL_ANGLE_THRESHOLD_RAD
    safe = np.where(small, 1.0, theta)
    half_sinc = np.where(small, 0.5 - theta * theta / 48.0, np.sin(0.5 * safe) / safe)
    w = np.where(small, 1.0 - theta * theta / 8.0, np.cos(0.5 * theta))
    return q_normalize(np.concatenate([w, half_sinc * rv], axis=-1))


def q_to_rotvec(q) -> np.ndarray:
    """Logarithm map: unit quaternion -> rotation vector with angle in [0, pi]."""
    q = q_canonical(q_normalize(q))
    w = np.clip(q[..., :1], -1.0, 1.0)
    u = q[..., 1:]
    s = np.linalg.norm(u, axis=-1, keepdims=True)
    theta = 2.0 * np.arctan2(s, w)
    small = s < 0.5 * SMALL_ANGLE_THRESHOLD_RAD
    factor = np.where(small, 2.0 / np.where(small, w, 1.0), theta / np.where(small, 1.0, s))
    return factor * u


def q_from_axis_angle(axis, angle_rad) -> np.ndarray:
    axis = _as_v(axis)
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    if np.any(n == 0.0):
        raise ValueError("rotation axis must be non-zero")
    return q_from_rotvec(axis / n * np.asarray(angle_rad, dtype=np.float64)[..., None])


def q_integrate(q, omega_radps, dt_s) -> np.ndarray:
    """One attitude step, section 4.3: q_{k+1} = q_k (x) exp(omega * dt), renormalised.

    ``omega`` is the body rate relative to the target frame, expressed in the BODY frame
    (right-multiplication). ``dt`` is the real sample interval -- never a nominal rate.
    """
    dt = np.asarray(dt_s, dtype=np.float64)
    if np.any(dt < 0):
        raise ValueError("dt must be non-negative")
    dq = q_from_rotvec(_as_v(omega_radps) * dt[..., None] if dt.ndim else _as_v(omega_radps) * dt)
    return q_normalize(q_mul(q, dq))


def q_slerp(q0, q1, t) -> np.ndarray:
    """Spherical linear interpolation along the SHORTER arc, t in [0, 1]."""
    q0, q1 = q_normalize(q0), q_normalize(q1)
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0.0, -q1, q1)
    d = np.abs(d)
    t = np.asarray(t, dtype=np.float64)[..., None] if np.ndim(t) else np.float64(t)
    near = d > 1.0 - 1e-12
    omega = np.arccos(np.clip(d, -1.0, 1.0))
    so = np.where(near, 1.0, np.sin(omega))
    a = np.where(near, 1.0 - t, np.sin((1.0 - t) * omega) / so)
    b = np.where(near, t, np.sin(t * omega) / so)
    return q_normalize(a * q0 + b * q1)


def q_angle(q0, q1) -> np.ndarray:
    """Smallest rotation angle [rad, 0..pi] taking q0 to q1."""
    d = np.abs(np.sum(q_normalize(q0) * q_normalize(q1), axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


# ---- Euler angles: Tait-Bryan z-y'-x'' (yaw, pitch, roll) ------------------------------
# R_b^n = Rz(yaw) @ Ry(pitch) @ Rx(roll). In the ENU navigation frame yaw is measured
# counter-clockwise from EAST about UP; compass heading (clockwise from North) is
# heading = pi/2 - yaw and is converted only at I/O boundaries (see heading helpers).

def q_from_euler(yaw_rad, pitch_rad, roll_rad) -> np.ndarray:
    cy, sy = np.cos(0.5 * np.asarray(yaw_rad)), np.sin(0.5 * np.asarray(yaw_rad))
    cp, sp = np.cos(0.5 * np.asarray(pitch_rad)), np.sin(0.5 * np.asarray(pitch_rad))
    cr, sr = np.cos(0.5 * np.asarray(roll_rad)), np.sin(0.5 * np.asarray(roll_rad))
    q = np.stack([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ], axis=-1)
    return q_canonical(q_normalize(q))


def q_to_euler(q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of q_from_euler. Returns (yaw, pitch, roll) with pitch in [-pi/2, pi/2].

    At gimbal lock (|pitch| = pi/2) yaw and roll are not separately defined; roll is set
    to 0 and the whole rotation about the vertical is reported as yaw.
    """
    R = q_to_dcm(q)
    sp = np.clip(-R[..., 2, 0], -1.0, 1.0)
    pitch = np.arcsin(sp)
    lock = np.abs(sp) > 1.0 - 1e-12
    yaw = np.where(lock, np.arctan2(-R[..., 0, 1], R[..., 1, 1]), np.arctan2(R[..., 1, 0], R[..., 0, 0]))
    roll = np.where(lock, 0.0, np.arctan2(R[..., 2, 1], R[..., 2, 2]))
    return yaw, pitch, roll


def heading_from_enu_yaw(yaw_rad):
    """ENU yaw (CCW from East) -> compass heading (CW from North), wrapped to [0, 2pi)."""
    return np.mod(np.pi / 2.0 - np.asarray(yaw_rad), 2.0 * np.pi)


def enu_yaw_from_heading(heading_rad):
    """Compass heading (CW from North) -> ENU yaw (CCW from East), wrapped to (-pi, pi]."""
    y = np.mod(np.pi / 2.0 - np.asarray(heading_rad) + np.pi, 2.0 * np.pi) - np.pi
    return np.where(y == -np.pi, np.pi, y)
