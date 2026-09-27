"""Quaternion math: convention, algebra, conversions, integration.

The convention tests (order [w,x,y,z], Hamilton, q represents R_b^n) are the ones
docs/navigation_math.md section 11 requires: a JPL or [x,y,z,w] implementation fails them.
Random tests use a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from navcore.common.constants import SMALL_ANGLE_THRESHOLD_RAD
from navcore.geometry import quaternion as Q
from navcore.geometry.rotation import rot_x, rot_y, rot_z

RNG = np.random.default_rng(26168)
TOL = 1e-12


def rand_q(n: int) -> np.ndarray:
    return Q.q_normalize(RNG.standard_normal((n, 4)))


def rand_v(n: int) -> np.ndarray:
    return RNG.standard_normal((n, 3))


# ---- convention -------------------------------------------------------------------------

def test_storage_order_is_scalar_first():
    np.testing.assert_array_equal(Q.Q_IDENTITY, [1.0, 0.0, 0.0, 0.0])
    q = Q.q_from_axis_angle([0, 0, 1], np.pi / 2)
    assert q[0] == pytest.approx(np.cos(np.pi / 4), abs=TOL)
    assert q[3] == pytest.approx(np.sin(np.pi / 4), abs=TOL)


def test_rotation_is_active_and_right_handed():
    # +90 deg about z takes x -> y (right-hand rule): a JPL/passive implementation gives -y
    q = Q.q_from_axis_angle([0, 0, 1], np.pi / 2)
    np.testing.assert_allclose(Q.q_rotate(q, [1, 0, 0]), [0, 1, 0], atol=TOL)
    q = Q.q_from_axis_angle([1, 0, 0], np.pi / 2)
    np.testing.assert_allclose(Q.q_rotate(q, [0, 1, 0]), [0, 0, 1], atol=TOL)
    q = Q.q_from_axis_angle([0, 1, 0], np.pi / 2)
    np.testing.assert_allclose(Q.q_rotate(q, [0, 0, 1]), [1, 0, 0], atol=TOL)


def test_hamilton_basis_products():
    i, j, k = np.eye(4)[1], np.eye(4)[2], np.eye(4)[3]
    np.testing.assert_array_equal(Q.q_mul(i, j), k)  # Hamilton: ij = k (JPL: ij = -k)
    np.testing.assert_array_equal(Q.q_mul(j, k), i)
    np.testing.assert_array_equal(Q.q_mul(k, i), j)
    np.testing.assert_array_equal(Q.q_mul(i, i), [-1, 0, 0, 0])


def test_composition_applies_right_operand_first():
    qa = Q.q_from_axis_angle([0, 0, 1], np.pi / 2)  # z 90
    qb = Q.q_from_axis_angle([1, 0, 0], np.pi / 2)  # x 90
    v = np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(Q.q_rotate(Q.q_mul(qa, qb), v), Q.q_rotate(qa, Q.q_rotate(qb, v)), atol=TOL)
    # x90 takes y->z; z90 leaves z. Opposite order would give -x.
    np.testing.assert_allclose(Q.q_rotate(Q.q_mul(qa, qb), v), [0, 0, 1], atol=TOL)


def test_rotate_equals_sandwich_product():
    q, v = rand_q(500), rand_v(500)
    vq = np.concatenate([np.zeros((500, 1)), v], axis=1)
    sandwich = Q.q_mul(Q.q_mul(q, vq), Q.q_conj(q))
    np.testing.assert_allclose(sandwich[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(sandwich[:, 1:], Q.q_rotate(q, v), atol=1e-12)


# ---- algebra ----------------------------------------------------------------------------

def test_multiplication_is_associative_not_commutative():
    a, b, c = rand_q(200), rand_q(200), rand_q(200)
    np.testing.assert_allclose(Q.q_mul(Q.q_mul(a, b), c), Q.q_mul(a, Q.q_mul(b, c)), atol=1e-12)
    assert np.max(np.abs(Q.q_mul(a, b) - Q.q_mul(b, a))) > 1e-3


def test_product_of_unit_quaternions_is_unit():
    a, b = rand_q(1000), rand_q(1000)
    np.testing.assert_allclose(Q.q_norm(Q.q_mul(a, b)), 1.0, atol=1e-14)


def test_inverse_and_conjugate():
    q = rand_q(300)
    np.testing.assert_allclose(Q.q_mul(q, Q.q_inv(q)), np.tile(Q.Q_IDENTITY, (300, 1)), atol=1e-14)
    np.testing.assert_allclose(Q.q_inv(q), Q.q_conj(q), atol=1e-15)
    nonunit = np.array([2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(Q.q_mul(nonunit, Q.q_inv(nonunit)), Q.Q_IDENTITY, atol=1e-15)


def test_rotation_preserves_norm_and_inner_products():
    q, u, v = rand_q(500), rand_v(500), rand_v(500)
    ru, rv = Q.q_rotate(q, u), Q.q_rotate(q, v)
    np.testing.assert_allclose(np.linalg.norm(ru, axis=1), np.linalg.norm(u, axis=1), rtol=1e-13)
    np.testing.assert_allclose(np.sum(ru * rv, 1), np.sum(u * v, 1), atol=1e-12)
    np.testing.assert_allclose(np.cross(ru, rv), Q.q_rotate(q, np.cross(u, v)), atol=1e-12)


def test_q_and_minus_q_are_the_same_rotation():
    q, v = rand_q(100), rand_v(100)
    np.testing.assert_allclose(Q.q_rotate(q, v), Q.q_rotate(-q, v), atol=1e-13)
    assert np.all(Q.q_canonical(-q)[:, 0] >= 0)


def test_normalize_rejects_zero_and_bad_shapes():
    with pytest.raises(ValueError):
        Q.q_normalize([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        Q.q_mul([1, 0, 0], [1, 0, 0, 0])
    with pytest.raises(ValueError):
        Q.q_rotate(Q.Q_IDENTITY, [1, 0])


# ---- DCM conversion ---------------------------------------------------------------------

def test_dcm_matches_elementary_rotations():
    for axis, rot in (([1, 0, 0], rot_x), ([0, 1, 0], rot_y), ([0, 0, 1], rot_z)):
        for ang in (-2.5, -0.3, 0.0, 0.7, np.pi / 2, 3.0):
            np.testing.assert_allclose(Q.q_to_dcm(Q.q_from_axis_angle(axis, ang)), rot(ang), atol=TOL)


def test_dcm_is_proper_rotation_and_matches_q_rotate():
    q, v = rand_q(500), rand_v(500)
    R = Q.q_to_dcm(q)
    np.testing.assert_allclose(R @ np.swapaxes(R, 1, 2), np.broadcast_to(np.eye(3), R.shape), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", R, v), Q.q_rotate(q, v), atol=1e-13)


def test_dcm_composition_matches_quaternion_composition():
    a, b = rand_q(300), rand_q(300)
    np.testing.assert_allclose(Q.q_to_dcm(Q.q_mul(a, b)), Q.q_to_dcm(a) @ Q.q_to_dcm(b), atol=1e-13)


def test_dcm_round_trip_including_180_degree_rotations():
    q = Q.q_canonical(rand_q(2000))
    np.testing.assert_allclose(Q.dcm_to_q(Q.q_to_dcm(q)), q, atol=1e-12)
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -2, 3]):
        qp = Q.q_from_axis_angle(axis, np.pi)
        back = Q.dcm_to_q(Q.q_to_dcm(qp))
        assert Q.q_angle(back, qp) < 1e-7  # q and -q are equivalent at exactly pi
        np.testing.assert_allclose(Q.q_to_dcm(back), Q.q_to_dcm(qp), atol=1e-12)


# ---- exponential / logarithm ------------------------------------------------------------

def test_rotvec_round_trip_and_angle():
    rv = RNG.uniform(-1, 1, (2000, 3))
    rv *= (RNG.uniform(0, np.pi - 1e-6, 2000) / np.linalg.norm(rv, axis=1))[:, None]
    q = Q.q_from_rotvec(rv)
    np.testing.assert_allclose(Q.q_to_rotvec(q), rv, atol=1e-11)
    np.testing.assert_allclose(Q.q_angle(Q.Q_IDENTITY, q), np.linalg.norm(rv, axis=1), atol=1e-11)


@pytest.mark.parametrize("theta", [0.0, 1e-12, 1e-9, 0.5 * SMALL_ANGLE_THRESHOLD_RAD,
                                   SMALL_ANGLE_THRESHOLD_RAD, 2 * SMALL_ANGLE_THRESHOLD_RAD, 1e-4])
def test_small_angle_branch_is_continuous_and_exact(theta):
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    q = Q.q_from_rotvec(axis * theta)
    exact = np.concatenate([[np.cos(theta / 2)], np.sin(theta / 2) * axis])
    np.testing.assert_allclose(q, exact, atol=1e-16, rtol=1e-12)
    assert np.all(np.isfinite(q))
    np.testing.assert_allclose(Q.q_to_rotvec(q), axis * theta, atol=1e-17, rtol=1e-9)


# ---- integration ------------------------------------------------------------------------

def test_integration_of_constant_rate_equals_exact_rotation():
    omega = np.array([0.02, -0.01, 0.3])  # rad/s, body frame
    q = Q.Q_IDENTITY.copy()
    for _ in range(1000):
        q = Q.q_integrate(q, omega, 0.01)
    exact = Q.q_from_rotvec(omega * 10.0)
    assert Q.q_angle(q, exact) < 1e-11
    assert Q.is_unit(q)


def test_integration_uses_real_non_uniform_dt():
    omega = np.array([0.0, 0.0, 0.5])
    dts = RNG.uniform(0.05, 0.15, 400)
    q = Q.Q_IDENTITY.copy()
    for dt in dts:
        q = Q.q_integrate(q, omega, dt)
    assert Q.q_angle(q, Q.q_from_rotvec(omega * dts.sum())) < 1e-11


def test_integration_body_rate_is_right_multiplied():
    # Body pointing North (yaw 90 deg in ENU); a body-x rate must roll about NORTH, not East.
    q0 = Q.q_from_euler(np.pi / 2, 0.0, 0.0)
    q1 = Q.q_integrate(q0, [0.1, 0.0, 0.0], 1.0)
    body_z_in_nav = Q.q_rotate(q1, [0, 0, 1])
    assert abs(body_z_in_nav[1]) < 1e-12  # rotation axis is North, so no North component
    assert body_z_in_nav[0] == pytest.approx(np.sin(0.1), abs=1e-12)


def test_integration_rejects_negative_dt():
    with pytest.raises(ValueError):
        Q.q_integrate(Q.Q_IDENTITY, [0, 0, 1], -0.1)


# ---- slerp ------------------------------------------------------------------------------

def test_slerp_endpoints_midpoint_and_short_arc():
    q0 = Q.Q_IDENTITY
    q1 = Q.q_from_axis_angle([0, 0, 1], 1.2)
    np.testing.assert_allclose(Q.q_slerp(q0, q1, 0.0), q0, atol=TOL)
    np.testing.assert_allclose(Q.q_slerp(q0, q1, 1.0), q1, atol=TOL)
    np.testing.assert_allclose(Q.q_slerp(q0, q1, 0.5), Q.q_from_axis_angle([0, 0, 1], 0.6), atol=TOL)
    # -q1 is the same rotation; slerp must still take the 0.6 rad short path
    assert Q.q_angle(Q.q_slerp(q0, -q1, 0.5), Q.q_from_axis_angle([0, 0, 1], 0.6)) < 1e-12
    np.testing.assert_allclose(Q.q_slerp(q1, q1, 0.3), q1, atol=TOL)  # degenerate: identical


def test_slerp_has_constant_angular_velocity():
    q0, q1 = rand_q(1)[0], rand_q(1)[0]
    ts = np.linspace(0, 1, 11)
    qs = np.array([Q.q_slerp(q0, q1, t) for t in ts])
    steps = Q.q_angle(qs[:-1], qs[1:])
    np.testing.assert_allclose(steps, steps[0], atol=1e-12)


# ---- Euler ------------------------------------------------------------------------------

def test_euler_sequence_is_zyx_intrinsic():
    y, p, r = 0.7, -0.4, 1.1
    np.testing.assert_allclose(Q.q_to_dcm(Q.q_from_euler(y, p, r)), rot_z(y) @ rot_y(p) @ rot_x(r), atol=TOL)


def test_euler_round_trip_away_from_gimbal_lock():
    yaw = RNG.uniform(-np.pi, np.pi, 3000)
    pitch = RNG.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 3000)
    roll = RNG.uniform(-np.pi, np.pi, 3000)
    y2, p2, r2 = Q.q_to_euler(Q.q_from_euler(yaw, pitch, roll))
    wrap = lambda a: np.angle(np.exp(1j * a))  # noqa: E731
    np.testing.assert_allclose(wrap(y2 - yaw), 0, atol=1e-9)
    np.testing.assert_allclose(p2 - pitch, 0, atol=1e-9)
    np.testing.assert_allclose(wrap(r2 - roll), 0, atol=1e-9)


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
def test_euler_at_gimbal_lock_reproduces_the_rotation(pitch):
    q = Q.q_from_euler(0.4, pitch, 0.9)
    y, p, r = Q.q_to_euler(q)
    assert r == 0.0 and p == pytest.approx(pitch, abs=1e-7)
    assert Q.q_angle(Q.q_from_euler(y, p, r), q) < 1e-7


def test_heading_yaw_conversion_enu():
    # ENU yaw 0 = facing East = compass 90 deg; yaw +90 deg = North = compass 0
    assert Q.heading_from_enu_yaw(0.0) == pytest.approx(np.pi / 2)
    assert Q.heading_from_enu_yaw(np.pi / 2) == pytest.approx(0.0, abs=1e-15)
    assert Q.heading_from_enu_yaw(-np.pi / 2) == pytest.approx(np.pi)
    h = RNG.uniform(0, 2 * np.pi, 1000)
    np.testing.assert_allclose(Q.heading_from_enu_yaw(Q.enu_yaw_from_heading(h)), h, atol=1e-12)
