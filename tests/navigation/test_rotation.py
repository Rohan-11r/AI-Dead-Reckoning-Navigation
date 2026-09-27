"""Rotation-matrix utilities."""

from __future__ import annotations

import numpy as np
import pytest

from navcore.geometry import quaternion as Q
from navcore.geometry import rotation as R

RNG = np.random.default_rng(7)


def test_skew_implements_cross_product_and_is_antisymmetric():
    a, u = RNG.standard_normal((200, 3)), RNG.standard_normal((200, 3))
    S = R.skew(a)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", S, u), np.cross(a, u), atol=1e-14)
    np.testing.assert_array_equal(S, -np.swapaxes(S, 1, 2))
    np.testing.assert_array_equal(R.unskew(S), a)


def test_skew_squared_identity():
    a = RNG.standard_normal(3)
    np.testing.assert_allclose(R.skew(a) @ R.skew(a), np.outer(a, a) - (a @ a) * np.eye(3), atol=1e-13)


def test_rotation_conjugation_of_skew():
    # R [a]x R^T = [R a]x  -- the identity every EKF Jacobian relies on
    Rm = Q.q_to_dcm(Q.q_normalize(RNG.standard_normal(4)))
    a = RNG.standard_normal(3)
    np.testing.assert_allclose(Rm @ R.skew(a) @ Rm.T, R.skew(Rm @ a), atol=1e-13)


@pytest.mark.parametrize("fn,axis", [(R.rot_x, 0), (R.rot_y, 1), (R.rot_z, 2)])
def test_elementary_rotations_fix_their_axis_and_are_proper(fn, axis):
    for ang in np.linspace(-np.pi, np.pi, 13):
        M = fn(ang)
        assert R.is_rotation_matrix(M)
        e = np.eye(3)[axis]
        np.testing.assert_allclose(M @ e, e, atol=1e-15)
        assert R.rotation_angle(M) == pytest.approx(abs(ang), abs=1e-7)


def test_rot_z_is_counter_clockwise():
    np.testing.assert_allclose(R.rot_z(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-15)


def test_euler_dcm_round_trip():
    y, p, r = 2.1, 0.3, -0.8
    M = R.euler_to_dcm(y, p, r)
    np.testing.assert_allclose(M, Q.q_to_dcm(Q.q_from_euler(y, p, r)), atol=1e-14)
    np.testing.assert_allclose(R.dcm_to_euler(M), (y, p, r), atol=1e-12)


def test_is_rotation_matrix_rejects_reflections_and_scaling():
    assert R.is_rotation_matrix(np.eye(3))
    assert not R.is_rotation_matrix(np.diag([1.0, 1.0, -1.0]))  # reflection, det -1
    assert not R.is_rotation_matrix(1.001 * np.eye(3))
    assert not R.is_rotation_matrix(np.array([[1, 0.01, 0], [0, 1, 0], [0, 0, 1.0]]))


def test_orthonormalize_recovers_nearest_rotation():
    M = Q.q_to_dcm(Q.q_normalize(RNG.standard_normal(4)))
    noisy = M + 1e-4 * RNG.standard_normal((3, 3))
    fixed = R.orthonormalize(noisy)
    assert R.is_rotation_matrix(fixed, tol=1e-12)
    assert np.max(np.abs(fixed - M)) < 1e-3
    # never returns a reflection, even for a reflected input
    assert np.linalg.det(R.orthonormalize(np.diag([1.0, 1.0, -1.0]) @ M)) == pytest.approx(1.0)
