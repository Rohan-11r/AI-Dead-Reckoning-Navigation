"""Geodesy: WGS84 geodetic <-> ECEF <-> local tangent ENU / NED.

THE ONLY code path allowed to produce latitude/longitude (AGENTS.md 2.1,
docs/navigation_math.md section 3). Angles in RADIANS, heights in metres (ellipsoidal).

* ``geodetic_to_ecef``  -- closed form, section 3.1.
* ``ecef_to_geodetic``  -- Heikkinen (1982) closed form: exact, non-iterative,
  deterministic (one of the two methods section 3.2 admits). Round-trip closure is
  asserted < 1e-4 m in tests and cross-checked against pyproj (test oracle only).
* ``R_ecef_to_enu``     -- R_e^n in ENU row order, section 3.3.
* ``LocalTangentPlane`` -- ENU frame with an IMMUTABLE origin, fixed per run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from navcore.common.constants import (
    FREE_AIR_GRADIENT_PER_S2,
    WGS84_A_M,
    WGS84_B_M,
    WGS84_E2,
    WGS84_EP2,
    WGS84_GAMMA_E_MPS2,
    WGS84_SOMIGLIANA_K,
)

# R_ned_from_enu: [N, E, D] = M @ [E, N, U]. A proper rotation (det +1), its own inverse.
R_ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def meridian_radius(lat_rad) -> np.ndarray:
    """M(phi): radius of curvature in the meridian [m]."""
    s2 = np.sin(lat_rad) ** 2
    return WGS84_A_M * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s2) ** 1.5


def prime_vertical_radius(lat_rad) -> np.ndarray:
    """N(phi): radius of curvature in the prime vertical [m]."""
    return WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat_rad) ** 2)


def geodetic_to_ecef(lat_rad, lon_rad, h_m) -> np.ndarray:
    """(phi, lambda, h) -> ECEF [x, y, z] in metres, stacked on the last axis."""
    lat, lon, h = (np.asarray(a, dtype=np.float64) for a in (lat_rad, lon_rad, h_m))
    n = prime_vertical_radius(lat)
    cl = np.cos(lat)
    return np.stack([(n + h) * cl * np.cos(lon), (n + h) * cl * np.sin(lon),
                     (n * (1.0 - WGS84_E2) + h) * np.sin(lat)], axis=-1)


def ecef_to_geodetic(p_ecef) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF -> (lat_rad, lon_rad, h_m), Heikkinen closed form.

    Valid everywhere except within ~43 km of the Earth's centre (irrelevant here);
    such inputs raise rather than return garbage.
    """
    p_ecef = np.asarray(p_ecef, dtype=np.float64)
    x, y, z = np.moveaxis(p_ecef, -1, 0)
    a, b, e2, ep2 = WGS84_A_M, WGS84_B_M, WGS84_E2, WGS84_EP2
    p = np.hypot(x, y)
    if np.any(np.hypot(p, z) < 50_000.0):
        raise ValueError("ecef_to_geodetic: point too close to the Earth's centre")
    F = 54.0 * b * b * z * z
    G = p * p + (1.0 - e2) * z * z - e2 * (a * a - b * b)
    c = e2 * e2 * F * p * p / (G ** 3)
    s = np.cbrt(1.0 + c + np.sqrt(c * c + 2.0 * c))
    k = s + 1.0 + 1.0 / s
    P = F / (3.0 * k * k * G * G)
    Q = np.sqrt(1.0 + 2.0 * e2 * e2 * P)
    r0 = (-(P * e2 * p) / (1.0 + Q)
          + np.sqrt(np.maximum(0.5 * a * a * (1.0 + 1.0 / Q)
                               - P * (1.0 - e2) * z * z / (Q * (1.0 + Q)) - 0.5 * P * p * p, 0.0)))
    U = np.hypot(p - e2 * r0, z)
    V = np.sqrt((p - e2 * r0) ** 2 + (1.0 - e2) * z * z)
    z0 = b * b * z / (a * V)
    h = U * (1.0 - b * b / (a * V))
    lat = np.arctan2(z + ep2 * z0, p)
    lon = np.arctan2(y, x)
    return lat, lon, h


def R_ecef_to_enu(lat_rad, lon_rad) -> np.ndarray:
    """R_e^n (ENU row order), section 3.3. Orthonormal; inverse is the transpose."""
    sl, cl = np.sin(lat_rad), np.cos(lat_rad)
    so, co = np.sin(lon_rad), np.cos(lon_rad)
    z = np.zeros_like(sl)
    return np.stack([
        np.stack([-so, co, z], -1),
        np.stack([-sl * co, -sl * so, cl], -1),
        np.stack([cl * co, cl * so, sl], -1),
    ], -2)


def enu_to_ned(v_enu) -> np.ndarray:
    return np.asarray(v_enu, dtype=np.float64) @ R_ENU_TO_NED.T


def ned_to_enu(v_ned) -> np.ndarray:
    return np.asarray(v_ned, dtype=np.float64) @ R_ENU_TO_NED  # R is symmetric & involutory


@dataclass(frozen=True)
class LocalTangentPlane:
    """ENU frame anchored at a fixed geodetic origin (section 3.3).

    The origin is immutable once constructed (frozen dataclass; arrays are made
    read-only), so a drifting origin cannot silently corrupt positions.
    """

    lat0_rad: float
    lon0_rad: float
    h0_m: float
    origin_ecef: np.ndarray = field(init=False, repr=False)
    R_e2n: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("lat0_rad", "lon0_rad", "h0_m"):
            val = float(getattr(self, name))
            if not np.isfinite(val):
                raise ValueError(f"LocalTangentPlane: {name} must be finite")
            object.__setattr__(self, name, val)
        if abs(self.lat0_rad) > np.pi / 2:
            raise ValueError("LocalTangentPlane: |lat0| > pi/2")
        o = geodetic_to_ecef(self.lat0_rad, self.lon0_rad, self.h0_m)
        R = R_ecef_to_enu(self.lat0_rad, self.lon0_rad)
        o.setflags(write=False)
        R.setflags(write=False)
        object.__setattr__(self, "origin_ecef", o)
        object.__setattr__(self, "R_e2n", R)

    def ecef_to_enu(self, p_ecef) -> np.ndarray:
        return (np.asarray(p_ecef, dtype=np.float64) - self.origin_ecef) @ self.R_e2n.T

    def enu_to_ecef(self, p_enu) -> np.ndarray:
        return np.asarray(p_enu, dtype=np.float64) @ self.R_e2n + self.origin_ecef

    def geodetic_to_enu(self, lat_rad, lon_rad, h_m) -> np.ndarray:
        return self.ecef_to_enu(geodetic_to_ecef(lat_rad, lon_rad, h_m))

    def enu_to_geodetic(self, p_enu) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return ecef_to_geodetic(self.enu_to_ecef(p_enu))

    def geodetic_to_ned(self, lat_rad, lon_rad, h_m) -> np.ndarray:
        return enu_to_ned(self.geodetic_to_enu(lat_rad, lon_rad, h_m))

    def ned_to_geodetic(self, p_ned) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.enu_to_geodetic(ned_to_enu(p_ned))


def normal_gravity(lat_rad, h_m) -> np.ndarray:
    """Somigliana normal gravity with free-air correction [m/s^2], section 5.1."""
    s2 = np.sin(lat_rad) ** 2
    return (WGS84_GAMMA_E_MPS2 * (1.0 + WGS84_SOMIGLIANA_K * s2) / np.sqrt(1.0 - WGS84_E2 * s2)
            - FREE_AIR_GRADIENT_PER_S2 * np.asarray(h_m, dtype=np.float64))
