"""Explicit frame management: tagged vectors, tagged rotations, mismatch detection."""

from __future__ import annotations

import numpy as np
import pytest

from navcore.geometry import frames as F
from navcore.geometry import geodesy as G
from navcore.geometry import quaternion as Q

LAT0, LON0 = np.radians(52.402565), np.radians(-1.503471)
RNG = np.random.default_rng(3)


def test_device_is_an_alias_of_body():
    assert F.Frame.DEVICE is F.Frame.BODY


def test_apply_checks_source_frame():
    r = F.nav_from_body(Q.q_from_euler(0.3, 0.1, -0.2))
    v_b = F.FramedVector([1.0, 2.0, 3.0], F.Frame.BODY)
    out = r.apply(v_b)
    assert out.frame is F.Frame.ENU
    np.testing.assert_allclose(out.v, Q.q_rotate(r.q, v_b.v), atol=1e-15)
    with pytest.raises(F.FrameMismatchError):
        r.apply(F.FramedVector([1.0, 2.0, 3.0], F.Frame.VEHICLE))
    with pytest.raises(TypeError):
        r.apply(np.array([1.0, 2.0, 3.0]))  # untagged arrays are refused


def test_composition_chains_frames_and_matches_matrices():
    q_vb = Q.q_from_euler(0.05, -0.02, 0.01)  # mounting (would come from Phase 4)
    q_nb = Q.q_from_euler(1.2, 0.03, -0.04)
    body_to_vehicle = F.vehicle_from_body(q_vb)
    body_to_nav = F.nav_from_body(q_nb)
    vehicle_to_nav = body_to_nav @ body_to_vehicle.inverse()
    assert (vehicle_to_nav.src, vehicle_to_nav.dst) == (F.Frame.VEHICLE, F.Frame.ENU)
    np.testing.assert_allclose(vehicle_to_nav.as_matrix(),
                               body_to_nav.as_matrix() @ body_to_vehicle.as_matrix().T, atol=1e-14)
    v_b = F.FramedVector(RNG.standard_normal(3), F.Frame.BODY)
    np.testing.assert_allclose(vehicle_to_nav.apply(body_to_vehicle.apply(v_b)).v,
                               body_to_nav.apply(v_b).v, atol=1e-14)
    with pytest.raises(F.FrameMismatchError):
        _ = body_to_vehicle @ body_to_nav  # (v<-b) @ (n<-b): does not chain


def test_inverse_round_trip_and_identity():
    r = F.Rotation(Q.q_normalize(RNG.standard_normal(4)), F.Frame.BODY, F.Frame.VEHICLE)
    ident = r.inverse() @ r
    assert ident.src is ident.dst is F.Frame.BODY
    assert Q.q_angle(ident.q, Q.Q_IDENTITY) < 1e-7
    np.testing.assert_allclose((r @ F.Rotation.identity(F.Frame.BODY)).q, r.q)


def test_ecef_enu_ned_chain():
    e2n = F.enu_from_ecef(LAT0, LON0)
    n2ned = F.ned_from_enu()
    np.testing.assert_allclose(e2n.as_matrix(), G.R_ecef_to_enu(LAT0, LON0), atol=1e-14)
    up_ecef = G.R_ecef_to_enu(LAT0, LON0)[2]
    down = (n2ned @ e2n).apply(F.FramedVector(up_ecef, F.Frame.ECEF))
    assert down.frame is F.Frame.NED
    np.testing.assert_allclose(down.v, [0, 0, -1], atol=1e-14)


def test_vector_arithmetic_requires_same_frame():
    a = F.FramedVector([1, 0, 0], F.Frame.ENU)
    b = F.FramedVector([0, 1, 0], F.Frame.ENU)
    np.testing.assert_array_equal((a + b).v, [1, 1, 0])
    np.testing.assert_array_equal((a - b).v, [1, -1, 0])
    with pytest.raises(F.FrameMismatchError):
        _ = a + F.FramedVector([0, 1, 0], F.Frame.NED)


def test_geodetic_is_not_a_vector_frame():
    with pytest.raises(F.FrameMismatchError):
        F.FramedVector([0, 0, 0], F.Frame.GEODETIC)
    with pytest.raises(F.FrameMismatchError):
        F.Rotation(Q.Q_IDENTITY, F.Frame.GEODETIC, F.Frame.ENU)


def test_rotation_rejects_non_unit_quaternion_and_values_are_immutable():
    with pytest.raises(ValueError):
        F.Rotation([2.0, 0, 0, 0], F.Frame.BODY, F.Frame.ENU)
    r = F.Rotation.identity(F.Frame.ENU)
    with pytest.raises(ValueError):
        r.q[0] = 0.5
    v = F.FramedVector([1.0, 2.0, 3.0], F.Frame.ENU)
    with pytest.raises(ValueError):
        v.v[0] = 9.0


def test_stacked_vectors_and_rotations():
    q = Q.q_normalize(RNG.standard_normal((50, 4)))
    r = F.Rotation(q, F.Frame.BODY, F.Frame.ENU)
    v = F.FramedVector(RNG.standard_normal((50, 3)), F.Frame.BODY)
    np.testing.assert_allclose(r.apply(v).v, np.einsum("nij,nj->ni", Q.q_to_dcm(q), v.v), atol=1e-13)
