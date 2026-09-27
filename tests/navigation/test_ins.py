"""Strapdown INS: physically correct propagation, checked against analytic motions.

All IMU inputs here are SYNTHETIC, generated from closed-form physics (a vehicle at rest
on the rotating Earth, constant velocity along a parallel, ...), so the correct answer
is known exactly. Nothing here is, or is used as, measured data.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navcore.common.constants import WGS84_OMEGA_IE_RADPS as OMEGA
from navcore.geometry.geodesy import (
    LocalTangentPlane,
    meridian_radius,
    normal_gravity,
    prime_vertical_radius,
)
from navcore.geometry.quaternion import (
    q_angle,
    q_conj,
    q_from_euler,
    q_rotate,
    q_to_euler,
)
from navcore.ins import mechanization as M

LAT, LON, H = math.radians(52.402565), math.radians(-1.503471), 144.59


def at_rest_imu(q_nb, lat=LAT, h=H):
    """Exact IMU output of a body at rest on the rotating Earth: f = -g, w = w_ie."""
    f_n = np.array([0.0, 0.0, float(normal_gravity(lat, h))])
    return q_rotate(q_conj(q_nb), f_n), q_rotate(q_conj(q_nb), M.earth_rate_enu(lat))


def run(state, f_b, w_b, dts):
    for dt in dts:
        state = M.propagate(state, f_b, w_b, dt)
    return state


# ---- rest on the rotating Earth ---------------------------------------------------------

@pytest.mark.parametrize("euler", [(0, 0, 0), (1.1, 0.05, -0.08), (-2.5, 0.3, 0.6)])
def test_stationary_for_one_hour_stays_put(euler):
    q = q_from_euler(*euler)
    f_b, w_b = at_rest_imu(q)
    s0 = M.NavState(0.0, LAT, LON, H, np.zeros(3), q)
    s = run(s0, f_b, w_b, [0.1] * 36000)
    rn = float(prime_vertical_radius(LAT))
    assert abs(s.lat_rad - LAT) * rn < 1e-3
    assert abs(s.lon_rad - LON) * rn * math.cos(LAT) < 1e-3
    assert abs(s.h_m - H) < 1e-3
    assert np.max(np.abs(s.v_enu_mps)) < 1e-6
    assert q_angle(s.q_nb, q) < 1e-9
    assert s.t_s == pytest.approx(3600.0)


def test_stationary_with_irregular_real_dt():
    q = q_from_euler(0.4, 0.0, 0.0)
    f_b, w_b = at_rest_imu(q)
    dts = np.random.default_rng(0).uniform(0.05, 0.15, 6000)
    s = run(M.NavState(0.0, LAT, LON, H, np.zeros(3), q), f_b, w_b, dts)
    assert np.max(np.abs(s.v_enu_mps)) < 1e-6
    assert s.t_s == pytest.approx(dts.sum())


def test_ignoring_earth_rate_drifts_heading_at_the_predicted_rate():
    # docs/navigation_math.md sec 6: omega_ie sin(phi) = 0.199 deg/min at this latitude
    q = q_from_euler(0.0, 0.0, 0.0)
    f_b, _ = at_rest_imu(q)
    s = run(M.NavState(0.0, LAT, LON, H, np.zeros(3), q), f_b, np.zeros(3), [0.1] * 600)
    yaw, pitch, roll = q_to_euler(s.q_nb)
    expected = -OMEGA * math.sin(LAT) * 60.0  # the nav frame turns under a non-rotating body
    # first-order prediction: the simultaneous tilt (cos phi component, below) couples in
    # at the 1e-3 level through the Euler extraction and leaked gravity
    assert yaw == pytest.approx(expected, rel=1e-2)
    assert abs(math.degrees(expected)) == pytest.approx(0.199, abs=5e-4)
    tilt = math.hypot(pitch, roll)
    assert tilt == pytest.approx(OMEGA * math.cos(LAT) * 60.0, rel=1e-2)


# ---- constant velocity along a parallel: Coriolis + transport rate --------------------

def eastward_imu(v_east, lat=LAT, h=H):
    """Body aligned with ENU moving East at constant speed along the parallel."""
    v = np.array([v_east, 0.0, 0.0])
    w_ie, w_en = M.earth_rate_enu(lat), M.transport_rate_enu(lat, h, v)
    f_n = np.cross(2 * w_ie + w_en, v) - M.gravity_enu(lat, h)  # makes dv/dt = 0
    return f_n, w_ie + w_en, v


def test_constant_eastward_velocity_along_a_parallel():
    f_b, w_b, v = eastward_imu(30.0)
    s0 = M.NavState(0.0, LAT, LON, H, v, np.array([1.0, 0, 0, 0]))
    s = run(s0, f_b, w_b, [0.1] * 6000)  # 10 min, 18 km
    rn = float(prime_vertical_radius(LAT))
    expected_dlon = 30.0 * 600.0 / ((rn + H) * math.cos(LAT))
    assert (s.lon_rad - LON) == pytest.approx(expected_dlon, rel=1e-9)
    assert abs(s.lat_rad - LAT) * rn < 0.01  # stays on the parallel to 1 cm over 18 km
    assert abs(s.h_m - H) < 0.01
    np.testing.assert_allclose(s.v_enu_mps, v, atol=1e-6)
    assert q_angle(s.q_nb, np.array([1.0, 0, 0, 0])) < 1e-9


def test_coriolis_deflects_to_the_right_with_the_predicted_magnitude():
    # Same motion but the specific force omits the Coriolis/transport compensation: the
    # INS must then deflect the trajectory SOUTH (right of travel, northern hemisphere)
    # at dv_N/dt = -2 Omega sin(phi) v_E (+ the much smaller transport term).
    v = np.array([30.0, 0.0, 0.0])
    f_b = -M.gravity_enu(LAT, H)
    w_b = M.earth_rate_enu(LAT) + M.transport_rate_enu(LAT, H, v)
    s = run(M.NavState(0.0, LAT, LON, H, v, np.array([1.0, 0, 0, 0])), f_b, w_b, [0.1] * 600)
    rm = float(meridian_radius(LAT))
    north_m = (s.lat_rad - LAT) * (rm + H)
    w_en_z = 30.0 * math.tan(LAT) / (float(prime_vertical_radius(LAT)) + H)
    a_n = -(2 * OMEGA * math.sin(LAT) + w_en_z) * 30.0
    assert north_m < 0
    assert north_m == pytest.approx(0.5 * a_n * 60.0 ** 2, rel=0.02)
    assert s.v_enu_mps[1] == pytest.approx(a_n * 60.0, rel=0.02)


# ---- accelerations and rotations --------------------------------------------------------

def test_constant_northward_acceleration_from_rest():
    q = np.array([1.0, 0, 0, 0])
    f_rest, w_b = at_rest_imu(q)
    a = 2.0
    f_b = f_rest + np.array([0.0, a, 0.0])
    s = run(M.NavState(0.0, LAT, LON, H, np.zeros(3), q), f_b, w_b, [0.1] * 200)
    rm = float(meridian_radius(LAT))
    assert (s.lat_rad - LAT) * (rm + H) == pytest.approx(0.5 * a * 20.0 ** 2, rel=2e-3)
    assert s.v_enu_mps[1] == pytest.approx(a * 20.0, rel=2e-3)
    assert abs(s.v_enu_mps[0]) < 0.05  # small Coriolis-driven East component only


def test_constant_yaw_rate_turns_heading_exactly():
    q = np.array([1.0, 0, 0, 0])
    f_b, w_rest = at_rest_imu(q)
    r = math.radians(10.0)
    s = M.NavState(0.0, LAT, LON, H, np.zeros(3), q)
    for _ in range(90):  # 9 s at 10 deg/s = 90 deg
        f_b, w_rest = at_rest_imu(s.q_nb)
        s = M.propagate(s, f_b, w_rest + np.array([0.0, 0.0, r]), 0.1)
    assert q_to_euler(s.q_nb)[0] == pytest.approx(math.pi / 2, abs=1e-9)
    # the synthetic input holds f_b / w_ie^b fixed over each 0.1 s while the body turns at
    # 10 deg/s (zero-order hold), leaving ~1e-6 rad of tilt -> ~1e-6 m/s leaked velocity
    assert np.max(np.abs(s.v_enu_mps)) < 1e-5


def test_body_frame_specific_force_is_rotated_into_nav():
    # Body yawed 90 deg (x points North): a forward (body-x) push must accelerate NORTH.
    q = q_from_euler(math.pi / 2, 0.0, 0.0)
    f_rest, w_b = at_rest_imu(q)
    s = run(M.NavState(0.0, LAT, LON, H, np.zeros(3), q), f_rest + np.array([1.0, 0, 0]), w_b,
            [0.1] * 50)
    assert s.v_enu_mps[1] == pytest.approx(5.0, rel=1e-3)
    assert abs(s.v_enu_mps[0]) < 1e-2


def test_gravity_is_local_normal_gravity_not_g0():
    # A body at rest fed the specific force of g0 instead of local gamma must FALL, at
    # (gamma - g0) -- section 5.2's ~10 m per minute.
    q = np.array([1.0, 0, 0, 0])
    _, w_b = at_rest_imu(q)
    s = run(M.NavState(0.0, LAT, LON, H, np.zeros(3), q), np.array([0, 0, 9.80665]), w_b, [0.1] * 600)
    dg = float(normal_gravity(LAT, H)) - 9.80665
    assert s.h_m - H == pytest.approx(-0.5 * dg * 60.0 ** 2, rel=0.01)


# ---- section 7 cross-check: curvilinear vs fixed tangent plane --------------------------

def test_curvilinear_and_tangent_plane_positions_agree_over_an_outage():
    f_b, w_b, v = eastward_imu(30.0)
    states = [M.NavState(0.0, LAT, LON, H, v, np.array([1.0, 0, 0, 0]))]
    for _ in range(600):  # 60 s, 1.8 km
        states.append(M.propagate(states[-1], f_b, w_b, 0.1))
    ltp = LocalTangentPlane(LAT, LON, H)
    p_tan = M.integrate_tangent_plane(ltp, states)[-1]
    s = states[-1]
    p_cur = ltp.geodetic_to_enu(s.lat_rad, s.lon_rad, s.h_m)
    assert np.linalg.norm(p_tan[:2] - p_cur[:2]) < 0.01  # horizontal: 1 cm over 1.8 km
    assert abs(p_tan[2] - p_cur[2]) < 0.3  # vertical: Earth curvature d^2/2R ~ 0.25 m


# ---- input guards ------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_dt_is_refused(dt):
    with pytest.raises(ValueError):
        M.propagate(M.NavState(0.0, LAT, LON, H), np.zeros(3), np.zeros(3), dt)


def test_nan_state_is_refused():
    with pytest.raises(ValueError):
        M.NavState(0.0, LAT, LON, H, np.array([np.nan, 0, 0]))
