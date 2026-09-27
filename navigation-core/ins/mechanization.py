"""Strapdown INS mechanization in the local-level ENU frame (docs/navigation_math.md sec 6-7).

    Attitude:   q_b^n' = exp(-omega_in^n dt) (x) q_b^n (x) exp(omega_ib^b dt)
    Velocity:   dv^n/dt = R_b^n f^b - (2 omega_ie^n + omega_en^n) x v^n + g^n
    Position:   dphi/dt = v_N/(M+h),  dlambda/dt = v_E/((N+h) cos phi),  dh/dt = v_U

* Earth rate, transport rate and Coriolis are RETAINED (section 6 quantifies why).
* Gravity is Somigliana normal gravity at the current position -- never g0 (section 5).
* dt is the real interval between samples (section 7.1); velocity and position use a
  trapezoidal (Heun) step, second order in dt.
* NO GNSS FIELD IS READ IN THIS MODULE (section 7). It consumes calibrated specific force
  and angular rate only; aiding lives in navcore.filtering.

The state is geodetic (lat, lon, h) + ENU velocity + attitude quaternion q_nb = R_b^n in
the quaternion convention of navcore.geometry.quaternion (sec 4.1). Everything float64.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np

from navcore.common.constants import WGS84_OMEGA_IE_RADPS
from navcore.geometry.geodesy import (
    LocalTangentPlane,
    R_ecef_to_enu,
    meridian_radius,
    normal_gravity,
    prime_vertical_radius,
)
from navcore.geometry.quaternion import q_from_rotvec, q_mul, q_normalize, q_rotate, q_slerp


@dataclass(frozen=True)
class NavState:
    """Navigation state at time ``t_s``: geodetic position (rad, rad, m), ENU velocity
    [m/s], attitude q_nb (R_b^n)."""

    t_s: float
    lat_rad: float
    lon_rad: float
    h_m: float
    v_enu_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_nb: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        v = np.asarray(self.v_enu_mps, dtype=np.float64).copy()
        q = q_normalize(np.asarray(self.q_nb, dtype=np.float64)).copy()
        if v.shape != (3,) or q.shape != (4,):
            raise ValueError("NavState needs v_enu (3,) and q_nb (4,)")
        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(q))
                and all(math.isfinite(x) for x in (self.t_s, self.lat_rad, self.lon_rad, self.h_m))):
            raise ValueError("NavState must be finite -- the INS has diverged or was fed NaN")
        v.setflags(write=False)
        q.setflags(write=False)
        object.__setattr__(self, "v_enu_mps", v)
        object.__setattr__(self, "q_nb", q)


def earth_rate_enu(lat_rad: float) -> np.ndarray:
    """omega_ie^n = Omega [0, cos phi, sin phi] (ENU: no East component)."""
    return WGS84_OMEGA_IE_RADPS * np.array([0.0, math.cos(lat_rad), math.sin(lat_rad)])


def transport_rate_enu(lat_rad: float, h_m: float, v_enu) -> np.ndarray:
    """omega_en^n = [-v_N/(M+h), v_E/(N+h), v_E tan(phi)/(N+h)]."""
    ve, vn, _ = v_enu
    rm = float(meridian_radius(lat_rad)) + h_m
    rn = float(prime_vertical_radius(lat_rad)) + h_m
    return np.array([-vn / rm, ve / rn, ve * math.tan(lat_rad) / rn])


def gravity_enu(lat_rad: float, h_m: float) -> np.ndarray:
    return np.array([0.0, 0.0, -float(normal_gravity(lat_rad, h_m))])


def _vdot(lat: float, h: float, v: np.ndarray, f_n: np.ndarray) -> np.ndarray:
    w_ie = earth_rate_enu(lat)
    w_en = transport_rate_enu(lat, h, v)
    return f_n - np.cross(2.0 * w_ie + w_en, v) + gravity_enu(lat, h)


def _position_rates(lat: float, h: float, v: np.ndarray) -> tuple[float, float, float]:
    rm = float(meridian_radius(lat)) + h
    rn = float(prime_vertical_radius(lat)) + h
    return v[1] / rm, v[0] / (rn * math.cos(lat)), v[2]


def propagate(state: NavState, f_b, w_ib_b, dt_s: float) -> NavState:
    """Advance ``state`` by one IMU interval.

    ``f_b`` [m/s^2] and ``w_ib_b`` [rad/s] are the CALIBRATED (bias-corrected) specific
    force and inertial angular rate in the body frame, treated as constant over the
    interval (zero-order hold on the sample at the START of the interval).
    """
    dt = float(dt_s)
    if not (dt > 0 and math.isfinite(dt)):
        raise ValueError(f"dt must be finite and > 0, got {dt_s!r}")
    f_b = np.asarray(f_b, dtype=np.float64)
    w_ib = np.asarray(w_ib_b, dtype=np.float64)
    lat, lon, h, v, q = state.lat_rad, state.lon_rad, state.h_m, state.v_enu_mps, state.q_nb

    # --- attitude: body rotation (right) and navigation-frame rotation (left)
    w_in = earth_rate_enu(lat) + transport_rate_enu(lat, h, v)
    q_new = q_normalize(q_mul(q_from_rotvec(-w_in * dt), q_mul(q, q_from_rotvec(w_ib * dt))))

    # --- velocity: Heun (trapezoidal) with the mid-interval attitude
    f_n = q_rotate(q_slerp(q, q_new, 0.5), f_b)
    a0 = _vdot(lat, h, v, f_n)
    v_pred = v + a0 * dt
    dlat0, _, dh0 = _position_rates(lat, h, 0.5 * (v + v_pred))
    lat_p, h_p = lat + dlat0 * dt, h + dh0 * dt
    a1 = _vdot(lat_p, h_p, v_pred, f_n)
    v_new = v + 0.5 * (a0 + a1) * dt

    # --- position: trapezoidal velocity, radii at the mid-interval latitude
    v_avg = 0.5 * (v + v_new)
    lat_mid = lat + 0.5 * _position_rates(lat, h, v_avg)[0] * dt
    h_mid = h + 0.5 * v_avg[2] * dt
    dlat, dlon, dh = _position_rates(lat_mid, h_mid, v_avg)
    lon_new = math.remainder(lon + dlon * dt, 2.0 * math.pi)
    return NavState(t_s=state.t_s + dt, lat_rad=lat + dlat * dt, lon_rad=lon_new,
                    h_m=h + dh * dt, v_enu_mps=v_new, q_nb=q_new)


def with_state(state: NavState, **changes) -> NavState:
    return replace(state, **changes)


# ---- Second position form, for the section 7 cross-check -------------------------------

def integrate_tangent_plane(ltp: LocalTangentPlane, states: list[NavState]) -> np.ndarray:
    """Integrate the SAME velocity history as p^n in a FIXED tangent plane (trapezoidal).

    Section 7 requires both position forms to exist and agree over outage-length windows;
    they differ only by Earth curvature (the local-level frame rotates, the fixed plane
    does not), which is O(d^2 / 2R). Returns ENU positions relative to ``ltp``'s origin.
    """
    p = [ltp.geodetic_to_enu(states[0].lat_rad, states[0].lon_rad, states[0].h_m)]
    for a, b in pairwise(states):
        # rotate each local-level velocity into the fixed plane before integrating
        va = _local_to_fixed(ltp, a)
        vb = _local_to_fixed(ltp, b)
        p.append(p[-1] + 0.5 * (va + vb) * (b.t_s - a.t_s))
    return np.asarray(p)


def _local_to_fixed(ltp: LocalTangentPlane, s: NavState) -> np.ndarray:
    R_local = R_ecef_to_enu(s.lat_rad, s.lon_rad)  # R_e^{n(local)}
    return ltp.R_e2n @ (R_local.T @ s.v_enu_mps)
