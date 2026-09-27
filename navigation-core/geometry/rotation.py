"""Rotation-matrix utilities. Quaternions (geometry.quaternion) are the internal state;
matrices exist for frame algebra, Jacobians and I/O.

Convention: ``R_b^a`` maps a vector expressed in ``b`` to the same vector expressed in
``a``: ``x_a = R_b^a @ x_b`` (docs/navigation_math.md section 1). ``rot_x/y/z(angle)``
are the ACTIVE right-handed rotations by ``angle`` about the axis; with the section 4
Euler sequence, ``R_b^n = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)``.
"""

from __future__ import annotations

import numpy as np

from navcore.geometry.quaternion import dcm_to_q, q_to_euler


def skew(a) -> np.ndarray:
    """[a]x such that skew(a) @ u == cross(a, u)."""
    a = np.asarray(a, dtype=np.float64)
    if a.shape[-1] != 3:
        raise ValueError(f"vector must have last dimension 3, got {a.shape}")
    x, y, z = np.moveaxis(a, -1, 0)
    o = np.zeros_like(x)
    return np.stack([np.stack([o, -z, y], -1), np.stack([z, o, -x], -1),
                     np.stack([-y, x, o], -1)], -2)


def unskew(S) -> np.ndarray:
    """Inverse of skew for a skew-symmetric matrix."""
    S = np.asarray(S, dtype=np.float64)
    return np.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], -1)


def rot_x(angle_rad) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    o, i = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([i, o, o], -1), np.stack([o, c, -s], -1), np.stack([o, s, c], -1)], -2)


def rot_y(angle_rad) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    o, i = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, o, s], -1), np.stack([o, i, o], -1), np.stack([-s, o, c], -1)], -2)


def rot_z(angle_rad) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    o, i = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, -s, o], -1), np.stack([s, c, o], -1), np.stack([o, o, i], -1)], -2)


def euler_to_dcm(yaw_rad, pitch_rad, roll_rad) -> np.ndarray:
    """R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)."""
    return rot_z(yaw_rad) @ rot_y(pitch_rad) @ rot_x(roll_rad)


def dcm_to_euler(R) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return q_to_euler(dcm_to_q(R))


def is_rotation_matrix(R, tol: float = 1e-9) -> bool:
    """Orthonormal with determinant +1 (a proper rotation, not a reflection)."""
    R = np.asarray(R, dtype=np.float64)
    eye = np.broadcast_to(np.eye(3), R.shape)
    orth = np.max(np.abs(np.swapaxes(R, -1, -2) @ R - eye)) <= tol
    return bool(orth and np.all(np.abs(np.linalg.det(R) - 1.0) <= tol))


def orthonormalize(R) -> np.ndarray:
    """Nearest proper rotation in the Frobenius norm (SVD / polar decomposition)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    D = np.ones(U.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


def rotation_angle(R) -> np.ndarray:
    """Angle [rad, 0..pi] of the rotation R."""
    tr = np.trace(np.asarray(R, dtype=np.float64), axis1=-2, axis2=-1)
    return np.arccos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0))
