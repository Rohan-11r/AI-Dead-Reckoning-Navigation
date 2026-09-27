"""Pin every constant to docs/navigation_math.md section 2 / 5 (AGENTS.md 4)."""

from __future__ import annotations

import math

import pytest

from navcore.common import constants as C


def test_wgs84_defining_parameters():
    assert C.WGS84_A_M == 6378137.0
    assert C.WGS84_F == 1.0 / 298.257223563
    assert C.WGS84_OMEGA_IE_RADPS == 7.292115e-5
    assert C.WGS84_GM_M3PS2 == 3.986004418e14


def test_wgs84_derived_match_spec_table():
    assert C.WGS84_B_M == pytest.approx(6356752.314245, abs=1e-6)
    assert C.WGS84_E2 == pytest.approx(6.69437999014e-3, rel=1e-11)
    assert C.WGS84_EP2 == pytest.approx(6.73949674228e-3, rel=1e-11)


def test_gravity_constants_and_g0_is_not_local_gravity():
    assert C.WGS84_GAMMA_E_MPS2 == 9.7803253359
    assert C.WGS84_SOMIGLIANA_K == 1.93185265241e-3
    assert C.FREE_AIR_GRADIENT_PER_S2 == 3.086e-6
    assert C.STANDARD_GRAVITY_MPS2 == 9.80665


def test_unit_conversions():
    assert C.DEG_TO_RAD * 180.0 == pytest.approx(math.pi, rel=1e-16)
    assert C.RAD_TO_DEG * C.DEG_TO_RAD == pytest.approx(1.0, rel=1e-16)
    assert C.KMH_TO_MPS * 3.6 == pytest.approx(1.0, rel=1e-16)


def test_small_angle_threshold_is_the_shared_value():
    # Changing this breaks Python/Android parity (section 4.3); change both or neither.
    assert C.SMALL_ANGLE_THRESHOLD_RAD == 1e-6
