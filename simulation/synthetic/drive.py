"""SYNTHETIC drive for tests and demos (AGENTS.md 2.2: clearly synthetic, never data).

A vehicle with sinusoidal acceleration/braking and turning, integrated by the INS itself so
the generated IMU is exactly consistent with the truth trajectory; body = vehicle frame.
IMU in the IO-VNBD column layout (vertical gyro in column 2) with biases and white noise;
GNSS at ``fix_hz`` with 3 m position noise, withheld inside ``dropouts``. Same generator as
tests/integration/test_fused_navigation.py, packaged for reuse.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius
from navcore.geometry.quaternion import q_conj, q_from_euler, q_rotate
from navcore.ins.mechanization import (
    NavState,
    earth_rate_enu,
    gravity_enu,
    propagate,
    transport_rate_enu,
)
from navcore.sensors.samples import GnssSample

LAT, LON, H = math.radians(52.4), math.radians(-1.5), 100.0


@dataclass
class DriveStep:
    t: float
    acc: np.ndarray
    grav: np.ndarray
    gyro_raw: np.ndarray
    fixes: list[GnssSample]
    truth: NavState            # truth at time t (before this step's propagation)
    speed: float
    a_long: float


def synthetic_drive(seed: int = 7, t_end_s: float = 260.0, dt: float = 0.1, dropouts=((100.0, 200.0),),
                    fix_hz: float = 1.0, speed0: float = 15.0) -> Iterator[DriveStep]:
    rng = np.random.default_rng(seed)
    q0 = q_from_euler(0.3, 0, 0)
    truth = NavState(0.0, LAT, LON, H, q_rotate(q0, [speed0, 0, 0]), q0)
    b_a = rng.normal(0, 0.02, 3)
    b_g = np.array([0.0, 0.0, rng.normal(0, 5e-4)])
    rm = float(meridian_radius(LAT)) + H
    rn = (float(prime_vertical_radius(LAT)) + H) * math.cos(LAT)
    period = round(1.0 / fix_hz / dt)
    k, t = 0, 0.0
    while t < t_end_s - 1e-9:
        a_long = 0.8 * math.sin(2 * math.pi * t / 60.0)
        yaw_rate = 0.08 * math.sin(2 * math.pi * t / 45.0)
        speed = float(np.linalg.norm(truth.v_enu_mps[:2]))
        v = truth.v_enu_mps
        dv_n = q_rotate(truth.q_nb, [a_long, speed * yaw_rate, 0.0])
        w_ie, w_en = earth_rate_enu(truth.lat_rad), transport_rate_enu(truth.lat_rad, truth.h_m, v)
        f_n = dv_n + np.cross(2 * w_ie + w_en, v) - gravity_enu(truth.lat_rad, truth.h_m)
        f_b = q_rotate(q_conj(truth.q_nb), f_n)
        w_b = np.array([0, 0, yaw_rate]) + q_rotate(q_conj(truth.q_nb), w_ie + w_en)
        acc = f_b + b_a + rng.normal(0, 0.05 / math.sqrt(dt), 3)
        w = w_b + b_g + rng.normal(0, 0.005 / math.sqrt(dt), 3)
        gyro_raw = np.array([w[0], w[2], w[1]])  # IO-VNBD: vertical in column 2
        grav = q_rotate(q_conj(truth.q_nb), -gravity_enu(truth.lat_rad, truth.h_m))
        fixes = []
        if k % period == 0 and t > 0 and not any(a <= t < b for a, b in dropouts):
            ve, vn = truth.v_enu_mps[:2] + rng.normal(0, 0.2, 2)
            fixes.append(GnssSample(t_s=t, lat_rad=truth.lat_rad + rng.normal(0, 3) / rm,
                                    lon_rad=truth.lon_rad + rng.normal(0, 3) / rn, h_m=truth.h_m + rng.normal(0, 7.5),
                                    horizontal_accuracy_m=3.0, speed_mps=math.hypot(ve, vn),
                                    bearing_rad=math.atan2(ve, vn)))
        yield DriveStep(t, acc, grav, gyro_raw, fixes, truth, speed, a_long)
        truth = propagate(truth, f_b, w_b, dt)
        k += 1
        t = round(t + dt, 10)


__all__ = ["LAT", "LON", "DriveStep", "H", "synthetic_drive"]
