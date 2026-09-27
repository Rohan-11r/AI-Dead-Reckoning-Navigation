"""Integration: raw IMU + AI outputs + GNSS dropouts through the fused navigator.

SYNTHETIC SCENARIO (AGENTS.md 2.2 -- clearly synthetic, never used as data):
* truth: a vehicle driving with sinusoidal acceleration/braking and turns, integrated by
  the INS itself so the generated IMU is exactly consistent with it; body = vehicle frame
* raw IMU in the IO-VNBD column layout (vertical gyro in column 2, horizontal columns
  carry the true small horizontal rates), with biases and white noise
* GNSS at 1 Hz, 3 m noise, with a 100 s DROPOUT
* AI outputs from NOISY-ORACLE STUBS (truth + noise at a stated sigma) -- the trained
  IO-VNBD models are meaningless on synthetic motion; real-model fusion is evaluated on
  real validation drives by scripts/evaluate/phase7_fusion_eval.py
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navcore.filtering.ekf import N_STATES, ErrorStateEKF, NoiseParams
from navcore.fusion.navigator import FusedNavigator, FusionConfig
from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius
from navcore.geometry.quaternion import q_conj, q_from_euler, q_rotate
from navcore.ins.mechanization import (
    NavState,
    earth_rate_enu,
    gravity_enu,
    propagate,
    transport_rate_enu,
)
from navcore.sensors.samples import IOVNBD_AXIS_MAP, GnssSample
from navcore.state.gnss_state import GnssState

LAT, LON, H = math.radians(52.4), math.radians(-1.5), 100.0
DT, T_END = 0.1, 260.0
DROPOUT = (100.0, 200.0)
NOISE = NoiseParams(0.05, 0.005, 0.02, 1e4, 1e-3, 1e4)


class NoisyOracle:
    """SYNTHETIC stand-in for a trained model: returns truth + N(0, sigma), with sigma^2."""

    def __init__(self, truth_fn, sigma: float, rng: np.random.Generator, window: int = 20) -> None:
        self.truth_fn, self.sigma, self.rng, self.window = truth_fn, sigma, rng, window

    def predict(self, acc, grav, wz):
        return self.truth_fn() + self.rng.normal(0, self.sigma), self.sigma ** 2


def horizontal_err(a: NavState, b: NavState) -> float:
    rm = float(meridian_radius(b.lat_rad)) + b.h_m
    rn = (float(prime_vertical_radius(b.lat_rad)) + b.h_m) * math.cos(b.lat_rad)
    return math.hypot((a.lat_rad - b.lat_rad) * rm, (a.lon_rad - b.lon_rad) * rn)


def run(cfg: FusionConfig, seed: int = 7, dropout=DROPOUT, with_models: bool = True):
    rng = np.random.default_rng(seed)
    truth = NavState(0.0, LAT, LON, H, q_rotate(q_from_euler(0.3, 0, 0), [15.0, 0, 0]), q_from_euler(0.3, 0, 0))
    state = {"a_long": 0.0, "speed": 15.0}
    b_a = rng.normal(0, 0.02, 3)
    b_g = np.array([0.0, 0.0, rng.normal(0, 5e-4)])
    P0 = np.diag([9, 9, 25, 0.04, 0.04, 0.04, 1e-4, 1e-4, 1e-3, 4e-4, 4e-4, 4e-4, 1e-6, 1e-6, 1e-6])
    ekf = ErrorStateEKF(truth, P0, NOISE)
    models = {}
    if with_models:
        models = {"model_speed": NoisyOracle(lambda: state["speed"], 0.5, rng),
                  "model_accel": NoisyOracle(lambda: state["a_long"], 0.3, rng)}
    nav = FusedNavigator(ekf, IOVNBD_AXIS_MAP, cfg, R_vb=np.eye(3), lateral_nhc_allowed=True, **models)
    rm = float(meridian_radius(LAT)) + H
    rn = (float(prime_vertical_radius(LAT)) + H) * math.cos(LAT)
    errs, states = {}, []
    t = 0.0
    while t < T_END - 1e-9:
        a_long = 0.8 * math.sin(2 * math.pi * t / 60.0)
        yaw_rate = 0.08 * math.sin(2 * math.pi * t / 45.0)
        state["a_long"], state["speed"] = a_long, float(np.linalg.norm(truth.v_enu_mps[:2]))
        v = truth.v_enu_mps
        dv_n = q_rotate(truth.q_nb, [a_long, state["speed"] * yaw_rate, 0.0])
        w_ie, w_en = earth_rate_enu(truth.lat_rad), transport_rate_enu(truth.lat_rad, truth.h_m, v)
        f_n = dv_n + np.cross(2 * w_ie + w_en, v) - gravity_enu(truth.lat_rad, truth.h_m)
        f_b = q_rotate(q_conj(truth.q_nb), f_n)
        w_b = np.array([0, 0, yaw_rate]) + q_rotate(q_conj(truth.q_nb), w_ie + w_en)
        acc_meas = f_b + b_a + rng.normal(0, 0.05 / math.sqrt(DT), 3)
        w_meas = w_b + b_g + rng.normal(0, 0.005 / math.sqrt(DT), 3)
        gyro_raw = np.array([w_meas[0], w_meas[2], w_meas[1]])  # IO-VNBD: vertical in column 2
        grav = q_rotate(q_conj(truth.q_nb), -gravity_enu(truth.lat_rad, truth.h_m))
        fixes = []
        if abs(t - round(t)) < 1e-9 and t > 0 and not (dropout[0] <= t < dropout[1]):
            ve, vn = truth.v_enu_mps[:2] + rng.normal(0, 0.2, 2)
            fixes.append(GnssSample(t_s=t, lat_rad=truth.lat_rad + rng.normal(0, 3) / rm,
                                    lon_rad=truth.lon_rad + rng.normal(0, 3) / rn, h_m=truth.h_m + rng.normal(0, 7.5),
                                    horizontal_accuracy_m=3.0, speed_mps=math.hypot(ve, vn),
                                    bearing_rad=math.atan2(ve, vn)))
        nav.step(t, acc_meas, grav, gyro_raw, fixes)
        truth = propagate(truth, f_b, w_b, DT)
        t = round(t + DT, 10)
        states.append((t, nav.sm.state))
        for probe in (dropout[1] - DT,):
            if abs(t - probe) < 1e-6:
                errs["dropout_end"] = horizontal_err(nav.ekf.nominal, truth)
    errs["final"] = horizontal_err(nav.ekf.nominal, truth)
    return errs, states, nav


INS_ONLY = FusionConfig(use_ai_speed=False, use_ai_accel=False, use_nhc=False)
FULL = FusionConfig(use_ai_accel=True)  # every component, incl. Model B (off by default)


@pytest.mark.slow
def test_ai_and_nhc_bound_the_dropout_drift():
    """Over 8 seeds with IDENTICAL noise per seed in both arms (the stubs are always attached,
    so they consume the same random stream; INS-only just ignores them). One seed is not
    evidence: per seed the full/INS ratio ranges 0.02-0.48."""
    ins, full = [], []
    for seed in range(1, 9):
        ins.append(run(INS_ONLY, seed=seed)[0]["dropout_end"])
        e, _, nav = run(FULL, seed=seed)
        full.append(e["dropout_end"])
        assert e["final"] < 15.0, (seed, e)  # re-converged after GNSS returns
        assert nav.counts["ai_speed_accepted"] > 80 and nav.counts["nhc_accepted"] > 1000
        assert nav.counts["ai_accel_corrections"] > 1000
    assert np.median(full) < 0.25 * np.median(ins), (full, ins)
    assert np.median(full) < 60.0  # 100 s without GNSS on a 15 m/s drive


@pytest.mark.slow
def test_state_machine_walks_through_every_state_across_the_dropout():
    _, states, nav = run(FULL)
    seq = [s for i, (_, s) in enumerate(states) if i == 0 or states[i - 1][1] is not s]
    assert seq[:1] == [GnssState.RECOVERING] or GnssState.RECOVERING in seq
    names = [s.value for s in seq]
    i_deg = names.index("DEGRADED", 1)
    assert names[i_deg + 1] == "LOST" and names[i_deg + 2] == "RECOVERING" and names[i_deg + 3] == "GOOD"
    lost = [t for t, s in states if s is GnssState.LOST and t > 50]
    assert DROPOUT[0] + 12 <= min(lost) <= DROPOUT[0] + 26  # lost after ~2 missed fix windows
    assert nav.sm.state is GnssState.GOOD


@pytest.mark.slow
def test_filter_covariance_stays_positive_definite_throughout():
    _, _, nav = run(FULL)
    np.linalg.cholesky(nav.ekf.P)  # the filter asserts PD after every step; final check
    assert nav.ekf.P.shape == (N_STATES, N_STATES)
