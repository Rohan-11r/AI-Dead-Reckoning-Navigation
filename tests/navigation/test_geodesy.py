"""WGS84 geodetic <-> ECEF <-> ENU/NED.

pyproj is the INDEPENDENT ORACLE here (docs/navigation_math.md section 3.4): it is never in
the runtime position path, only in tests.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from pyproj import Transformer

from navcore.common.constants import WGS84_A_M, WGS84_B_M
from navcore.geometry import geodesy as G

RNG = np.random.default_rng(1982)
TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
TO_GEO = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

# First IO-VNBD fix (Coventry), docs/navigation_math.md section 5.2
LAT0, LON0, H0 = np.radians(52.402565), np.radians(-1.503471), 144.59


def grid(n: int, hmin: float = -500.0, hmax: float = 20_000.0):
    lat = np.arcsin(RNG.uniform(-1, 1, n))  # uniform on the sphere, poles included
    lon = RNG.uniform(-np.pi, np.pi, n)
    h = RNG.uniform(hmin, hmax, n)
    return lat, lon, h


# ---- analytically known points ----------------------------------------------------------

def test_known_points():
    np.testing.assert_allclose(G.geodetic_to_ecef(0.0, 0.0, 0.0), [WGS84_A_M, 0, 0], atol=1e-9)
    np.testing.assert_allclose(G.geodetic_to_ecef(0.0, np.pi / 2, 0.0), [0, WGS84_A_M, 0], atol=1e-9)
    np.testing.assert_allclose(G.geodetic_to_ecef(np.pi / 2, 0.0, 0.0), [0, 0, WGS84_B_M], atol=1e-9)
    np.testing.assert_allclose(G.geodetic_to_ecef(-np.pi / 2, 1.0, 100.0), [0, 0, -WGS84_B_M - 100], atol=1e-8)
    np.testing.assert_allclose(G.geodetic_to_ecef(0.0, np.pi, 10.0), [-WGS84_A_M - 10, 0, 0], atol=1e-8)


def test_radii_of_curvature_known_values():
    assert G.prime_vertical_radius(0.0) == pytest.approx(WGS84_A_M, abs=1e-9)
    assert G.meridian_radius(0.0) == pytest.approx(WGS84_B_M ** 2 / WGS84_A_M, abs=1e-6)
    # at the pole both radii equal a^2/b
    assert G.prime_vertical_radius(np.pi / 2) == pytest.approx(WGS84_A_M ** 2 / WGS84_B_M, abs=1e-6)
    assert G.meridian_radius(np.pi / 2) == pytest.approx(WGS84_A_M ** 2 / WGS84_B_M, abs=1e-6)


# ---- oracle cross-check -----------------------------------------------------------------

def test_geodetic_to_ecef_matches_pyproj():
    lat, lon, h = grid(5000)
    ours = G.geodetic_to_ecef(lat, lon, h)
    ref = np.column_stack(TO_ECEF.transform(np.degrees(lon), np.degrees(lat), h))
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_ecef_to_geodetic_matches_pyproj():
    lat, lon, h = grid(5000)
    p = G.geodetic_to_ecef(lat, lon, h)
    la, lo, hh = G.ecef_to_geodetic(p)
    rlon, rlat, rh = TO_GEO.transform(p[:, 0], p[:, 1], p[:, 2])
    # compare as ground distance: 1e-10 rad of latitude ~ 0.6 mm
    np.testing.assert_allclose(la, np.radians(rlat), atol=1e-10)
    np.testing.assert_allclose(np.cos(la) * np.angle(np.exp(1j * (lo - np.radians(rlon)))), 0, atol=1e-10)
    np.testing.assert_allclose(hh, rh, atol=1e-4)


# ---- round trips: section 3.2 requires closure < 1e-4 m --------------------------------

def test_round_trip_closure_sub_0p1_mm_globally():
    lat, lon, h = grid(20000, -1000.0, 100_000.0)
    p = G.geodetic_to_ecef(lat, lon, h)
    p2 = G.geodetic_to_ecef(*G.ecef_to_geodetic(p))
    assert np.max(np.linalg.norm(p2 - p, axis=1)) < 1e-4


@pytest.mark.parametrize("lat_deg", [90.0, -90.0, 89.999999, -89.999999, 0.0, 1e-9])
def test_round_trip_at_poles_and_equator(lat_deg):
    for h in (-100.0, 0.0, 144.59, 10_000.0):
        lat, lon = np.radians(lat_deg), np.radians(37.0)
        la, lo, hh = G.ecef_to_geodetic(G.geodetic_to_ecef(lat, lon, h))
        assert la == pytest.approx(lat, abs=1e-12)
        assert hh == pytest.approx(h, abs=1e-6)
        if abs(lat_deg) < 90:
            assert lo == pytest.approx(lon, abs=1e-12)


def test_ecef_to_geodetic_refuses_earth_centre():
    with pytest.raises(ValueError):
        G.ecef_to_geodetic([1000.0, 0.0, 0.0])


# ---- ENU / NED --------------------------------------------------------------------------

def test_R_ecef_to_enu_is_orthonormal_and_points_the_right_way():
    Rm = G.R_ecef_to_enu(LAT0, LON0)
    np.testing.assert_allclose(Rm @ Rm.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(Rm) == pytest.approx(1.0, abs=1e-15)
    # "Up" row is the ellipsoid normal: parallel to the direction of increasing height
    d = G.geodetic_to_ecef(LAT0, LON0, H0 + 1.0) - G.geodetic_to_ecef(LAT0, LON0, H0)
    np.testing.assert_allclose(Rm @ d, [0, 0, 1], atol=1e-9)


def test_ltp_origin_maps_to_zero_and_axes_follow_enu():
    ltp = G.LocalTangentPlane(LAT0, LON0, H0)
    np.testing.assert_allclose(ltp.geodetic_to_enu(LAT0, LON0, H0), 0.0, atol=1e-9)
    # section 11: a pure +East displacement is component 0 and moves longitude, not latitude
    dlon = 1.0 / ((G.prime_vertical_radius(LAT0) + H0) * np.cos(LAT0))  # 1 m east at h0
    e = ltp.geodetic_to_enu(LAT0, LON0 + dlon, H0)
    assert e[0] == pytest.approx(1.0, abs=1e-6) and abs(e[1]) < 1e-6
    dlat = 1.0 / (G.meridian_radius(LAT0) + H0)  # 1 m north at h0
    n = ltp.geodetic_to_enu(LAT0 + dlat, LON0, H0)
    assert n[1] == pytest.approx(1.0, abs=1e-6) and abs(n[0]) < 1e-6
    u = ltp.geodetic_to_enu(LAT0, LON0, H0 + 1.0)
    np.testing.assert_allclose(u, [0, 0, 1], atol=1e-9)


def test_ltp_round_trip_over_the_operating_region():
    ltp = G.LocalTangentPlane(LAT0, LON0, H0)
    enu = RNG.uniform([-50_000, -50_000, -200], [50_000, 50_000, 500], (5000, 3))
    back = ltp.geodetic_to_enu(*ltp.enu_to_geodetic(enu))
    assert np.max(np.linalg.norm(back - enu, axis=1)) < 1e-4
    ecef = ltp.enu_to_ecef(enu)
    np.testing.assert_allclose(ltp.ecef_to_enu(ecef), enu, atol=1e-7)


def test_ltp_enu_vs_pyproj_topocentric():
    # independent check: pyproj geodetic->ECEF, then OUR rotation must reproduce the
    # distance that pyproj's own geodesic implies over short baselines
    ltp = G.LocalTangentPlane(LAT0, LON0, H0)
    lat = LAT0 + np.radians(RNG.uniform(-0.01, 0.01, 200))
    lon = LON0 + np.radians(RNG.uniform(-0.01, 0.01, 200))
    h = H0 + RNG.uniform(-20, 20, 200)
    ref_ecef = np.column_stack(TO_ECEF.transform(np.degrees(lon), np.degrees(lat), h))
    enu = ltp.geodetic_to_enu(lat, lon, h)
    np.testing.assert_allclose(np.linalg.norm(enu, axis=1),
                               np.linalg.norm(ref_ecef - ltp.origin_ecef, axis=1), atol=1e-6)


def test_ltp_origin_is_immutable():
    ltp = G.LocalTangentPlane(LAT0, LON0, H0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ltp.lat0_rad = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError):
        ltp.origin_ecef[0] = 0.0
    with pytest.raises(ValueError):
        ltp.R_e2n[0, 0] = 0.0


def test_ltp_rejects_invalid_origin():
    with pytest.raises(ValueError):
        G.LocalTangentPlane(np.nan, 0.0, 0.0)
    with pytest.raises(ValueError):
        G.LocalTangentPlane(2.0, 0.0, 0.0)


def test_enu_ned_conversion():
    v = RNG.standard_normal((100, 3))
    ned = G.enu_to_ned(v)
    np.testing.assert_array_equal(ned[:, 0], v[:, 1])
    np.testing.assert_array_equal(ned[:, 1], v[:, 0])
    np.testing.assert_array_equal(ned[:, 2], -v[:, 2])
    np.testing.assert_array_equal(G.ned_to_enu(ned), v)
    assert np.linalg.det(G.R_ENU_TO_NED) == pytest.approx(1.0)
    ltp = G.LocalTangentPlane(LAT0, LON0, H0)
    la, lo, h = ltp.ned_to_geodetic([10.0, -5.0, -3.0])
    np.testing.assert_allclose(ltp.geodetic_to_ned(la, lo, h), [10.0, -5.0, -3.0], atol=1e-8)


# ---- gravity ----------------------------------------------------------------------------

def test_normal_gravity_reproduces_spec_and_differs_from_g0():
    # section 5.2 (corrected in Phase 2) and section 11: local gamma != g0
    assert G.normal_gravity(LAT0, H0) == pytest.approx(9.812381, abs=1e-6)
    assert G.normal_gravity(0.0, 0.0) == pytest.approx(9.7803253359, abs=1e-10)
    assert abs(G.normal_gravity(LAT0, H0) - 9.80665) > 5e-3
    assert G.normal_gravity(np.pi / 2, 0.0) > G.normal_gravity(0.0, 0.0)  # stronger at poles
