"""Automatic phone-to-vehicle alignment: R_b^v (device/body -> vehicle FLU).

Two steps, using only what the phone itself measures (AGENTS.md: CAN never an input):

1. LEVEL (pitch/roll of the mount) from gravity: over a drive the mean specific force
   points "up" in the device frame (linear accelerations and road grades average out).
   ``tilt_rotation`` is the minimal rotation taking that direction to +z.
2. YAW (heading of the mount) from GNSS-derived motion: at each fix epoch the vehicle's
   along-track acceleration a_long = dv/dt and centripetal acceleration a_lat = v * dpsi/dt
   are computed from GNSS speed and course; the levelled device-frame horizontal specific
   force must equal the same vector rotated by the mount yaw:
        a_level_xy = Rz(yaw) [a_long, a_lat]
   ``fit_mounting_yaw`` solves this in closed form (2-D orthogonal Procrustes).

Result: R_b^v = Rz(-yaw) @ R_tilt, with fit quality attached so a caller can refuse a
poorly observed alignment instead of trusting it. Frames: vehicle x forward, y left,
z up (docs/navigation_math.md sec 1).

GNSS latency: pass fix EPOCH times (``GnssSample.t_s`` / ``ph_gnss_epoch_utc``) so the
GNSS-derived accelerations line up with the IMU window they describe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from navcore.geometry.frames import Frame, Rotation
from navcore.geometry.quaternion import dcm_to_q
from navcore.geometry.rotation import rot_z


@dataclass(frozen=True)
class YawFit:
    yaw_rad: float  # rotation of the vehicle frame into the levelled device frame
    scale: float  # least-squares |a_level| / |a_vehicle| -- ~1 if the model holds
    r2: float  # fraction of levelled-accel variance explained by the rotated vehicle accel
    n: int


@dataclass(frozen=True)
class MountingEstimate:
    R_vb: np.ndarray  # R_b^v: body vectors -> vehicle frame
    tilt_rad: float
    yaw_rad: float
    fit: YawFit

    def rotation(self) -> Rotation:
        return Rotation(dcm_to_q(self.R_vb), Frame.BODY, Frame.VEHICLE)

    def acceptable(self, min_r2: float = 0.3, min_n: int = 30, scale_band=(0.5, 1.5)) -> bool:
        f = self.fit
        return f.n >= min_n and f.r2 >= min_r2 and scale_band[0] <= f.scale <= scale_band[1]


def up_direction(accel_b: np.ndarray) -> np.ndarray:
    """Unit 'up' direction in the body frame from specific-force samples (N, 3)."""
    a = np.asarray(accel_b, dtype=np.float64)
    a = a[np.all(np.isfinite(a), axis=1)]
    if len(a) < 10:
        raise ValueError("need >= 10 finite accelerometer samples to level")
    m = a.mean(axis=0)
    n = np.linalg.norm(m)
    if n < 5.0:
        raise ValueError(f"mean specific force {n:.2f} m/s^2 is not gravity-dominated")
    return m / n


def tilt_rotation(up_b: np.ndarray) -> np.ndarray:
    """Minimal rotation R_lb with R_lb @ up_b = [0, 0, 1] (Rodrigues)."""
    u = np.asarray(up_b, dtype=np.float64)
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u, z)
    s, c = np.linalg.norm(axis), float(u @ z)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])  # upside down: 180 deg about x
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def gnss_vehicle_acceleration(t_s: np.ndarray, speed_mps: np.ndarray, bearing_rad: np.ndarray,
                              min_speed_mps: float = 3.0, max_gap_s: float = 2.5
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Vehicle-frame acceleration [a_long, a_lat] at interior fix epochs (central
    differences over the neighbouring fixes). Returns (t_mid, a (N, 2)).

    Course over ground is unreliable at low speed, so epochs below ``min_speed_mps`` are
    dropped; so are differences spanning a gap longer than ``max_gap_s``.
    """
    t = np.asarray(t_s, dtype=np.float64)
    v = np.asarray(speed_mps, dtype=np.float64)
    b = np.unwrap(np.asarray(bearing_rad, dtype=np.float64))
    if len(t) < 3:
        return np.empty(0), np.empty((0, 2))
    dt = t[2:] - t[:-2]
    a_long = (v[2:] - v[:-2]) / dt
    yaw_rate = -(b[2:] - b[:-2]) / dt  # ENU yaw (CCW) = -bearing (CW)
    a_lat = v[1:-1] * yaw_rate  # centripetal, positive to the LEFT in FLU
    ok = ((dt > 0) & (dt <= 2 * max_gap_s) & (np.minimum(np.minimum(v[:-2], v[1:-1]), v[2:]) >= min_speed_mps)
          & np.isfinite(a_long) & np.isfinite(a_lat))
    return t[1:-1][ok], np.column_stack([a_long, a_lat])[ok]


def fit_mounting_yaw(a_level_xy: np.ndarray, a_vehicle: np.ndarray) -> YawFit:
    """Closed-form 2-D Procrustes: yaw minimising sum |a_level - Rz(yaw) a_vehicle|^2."""
    A = np.asarray(a_level_xy, dtype=np.float64)
    B = np.asarray(a_vehicle, dtype=np.float64)
    ok = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1)
    A, B = A[ok], B[ok]
    if len(A) < 3:
        raise ValueError("need >= 3 paired acceleration samples to fit the mount yaw")
    dot = float(np.sum(B[:, 0] * A[:, 0] + B[:, 1] * A[:, 1]))
    cross = float(np.sum(B[:, 0] * A[:, 1] - B[:, 1] * A[:, 0]))
    yaw = math.atan2(cross, dot)
    Rb = B @ rot_z(yaw)[:2, :2].T
    scale = float(np.sum(Rb * A) / np.sum(Rb * Rb))
    resid = A - scale * Rb
    r2 = 1.0 - float(np.sum(resid ** 2) / np.sum((A - A.mean(axis=0)) ** 2))
    return YawFit(yaw_rad=yaw, scale=scale, r2=r2, n=len(A))


def estimate_mounting(accel_b: np.ndarray, imu_t_s: np.ndarray, fix_t_s: np.ndarray,
                      fix_speed_mps: np.ndarray, fix_bearing_rad: np.ndarray,
                      window_s: float | None = None, max_fix_gap_s: float = 2.5) -> MountingEstimate:
    """Full alignment from phone data only.

    ``accel_b`` (N, 3) body specific force at ``imu_t_s``; fixes at EPOCH times
    ``fix_t_s``. The levelled horizontal accel is averaged over the same window the GNSS
    central difference spans (default: between the neighbouring fixes).
    ``max_fix_gap_s``: longest fix spacing accepted. With 1 Hz fixes the differences resolve
    individual manoeuvres; with the ~9 s fixes most IO-VNBD sessions have, they average
    over ~18 s and the fit is weaker -- which the returned r2/scale report honestly.
    """
    accel_b = np.asarray(accel_b, dtype=np.float64)
    R_lb = tilt_rotation(up_direction(accel_b))
    a_lvl = accel_b @ R_lb.T
    t_mid, a_veh = gnss_vehicle_acceleration(fix_t_s, fix_speed_mps, fix_bearing_rad,
                                             max_gap_s=max_fix_gap_s)
    imu_t = np.asarray(imu_t_s, dtype=np.float64)
    fix_t = np.asarray(fix_t_s, dtype=np.float64)
    cs = np.vstack([np.zeros((1, 2)), np.cumsum(np.nan_to_num(a_lvl[:, :2]), axis=0)])
    rows = []
    for tm in t_mid:
        if window_s is None:
            k = int(np.searchsorted(fix_t, tm))
            lo_t, hi_t = fix_t[max(k - 1, 0)], fix_t[min(k + 1, len(fix_t) - 1)]
        else:
            lo_t, hi_t = tm - window_s / 2, tm + window_s / 2
        i0, i1 = np.searchsorted(imu_t, [lo_t, hi_t])
        rows.append((cs[i1] - cs[i0]) / (i1 - i0) if i1 - i0 >= 5 else [np.nan, np.nan])
    fit = fit_mounting_yaw(np.asarray(rows).reshape(-1, 2), a_veh)
    R_vb = rot_z(-fit.yaw_rad) @ R_lb
    tilt = math.acos(max(-1.0, min(1.0, float(R_lb[2, 2]))))
    return MountingEstimate(R_vb=R_vb, tilt_rad=tilt, yaw_rad=fit.yaw_rad, fit=fit)
