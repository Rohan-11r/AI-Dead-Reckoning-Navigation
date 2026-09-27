"""Physical and numerical constants -- the single Python source of truth.

Values and symbols follow docs/navigation_math.md section 2. Every navcore module imports
from here; nothing re-types a constant. The Kotlin/C++ copy is to be GENERATED from this
file in Phase 12 (AGENTS.md 4), never hand-copied; tests/navigation/test_constants.py
pins each value to the spec.

All quantities SI: metres, seconds, radians.
"""

from __future__ import annotations

import math

# ---- WGS84 ellipsoid (defining parameters + derived) ---------------------------------
WGS84_A_M = 6378137.0  # semi-major axis (exact, defining)
WGS84_F = 1.0 / 298.257223563  # flattening (defining)
WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)  # semi-minor axis
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)  # second eccentricity squared
WGS84_OMEGA_IE_RADPS = 7.292115e-5  # Earth rotation rate (defining)
WGS84_GM_M3PS2 = 3.986004418e14  # geocentric gravitational constant (defining)

# ---- Gravity (docs/navigation_math.md section 5.1) -----------------------------------
WGS84_GAMMA_E_MPS2 = 9.7803253359  # normal gravity at the equator
WGS84_SOMIGLIANA_K = 1.93185265241e-3  # Somigliana coefficient
FREE_AIR_GRADIENT_PER_S2 = 3.086e-6  # d(gamma)/dh [1/s^2], i.e. m/s^2 per metre
STANDARD_GRAVITY_MPS2 = 9.80665  # g0: a DEFINED constant, NOT local gravity (section 5.2)

# ---- Numerical thresholds shared by both implementations (section 4.3) ---------------
# Below this rotation angle the quaternion exponential switches to its Taylor series.
# A different threshold in the Kotlin port is a parity failure.
SMALL_ANGLE_THRESHOLD_RAD = 1e-6
# Tolerance used when asserting a quaternion is unit-norm.
UNIT_QUATERNION_TOL = 1e-9

# ---- Unit conversions (I/O boundaries only) ------------------------------------------
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
KMH_TO_MPS = 1.0 / 3.6
