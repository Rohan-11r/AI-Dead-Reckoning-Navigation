"""Error-state EKF: Jacobians by finite differences, update mechanics, consistency.

docs/navigation_math.md sec 11 requires F and every H to be validated against finite
differences of the ACTUAL propagation / measurement functions -- the test, not the
document, is the guarantee on signs. All IMU/GNSS data here are SYNTHETIC, generated
from known truth with seeded noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import chi2

from navcore.filtering.ekf import (
    A_,
    BA_,
    BG_,
    N_STATES,
    CovarianceError,
    ErrorStateEKF,
    NoiseParams,
    error_between,
)
from navcore.geometry.geodesy import meridian_radius, normal_gravity, prime_vertical_radius
from navcore.geometry.quaternion import (
    q_conj,
    q_from_euler,
    q_from_rotvec,
    q_mul,
    q_normalize,
    q_rotate,
)
from navcore.geometry.rotation import skew
from navcore.ins.mechanization import (
    NavState,
    earth_rate_enu,
    gravity_enu,
    propagate,
    transport_rate_enu,
)
from navcore.sensors.samples import GnssSample, ImuSample

LAT, LON, H = math.radians(52.402565), math.radians(-1.503471), 144.59
NOISE = NoiseParams(accel_white_mps2_rtHz=0.05, gyro_white_radps_rtHz=0.005,
                    accel_bias_sigma_mps2=0.02, accel_bias_tau_s=100.0,
                    gyro_bias_sigma_radps=1e-3, gyro_bias_tau_s=100.0)


def inject(s: NavState, dx9) -> NavState:
    """Independent re-implementation of the filter's correction (truth = nominal (+) dx)."""
    rm = float(meridian_radius(s.lat_rad)) + s.h_m
    rn = float(prime_vertical_radius(s.lat_rad)) + s.h_m
    return NavState(s.t_s, s.lat_rad + dx9[1] / rm, s.lon_rad + dx9[0] / (rn * math.cos(s.lat_rad)),
                    s.h_m + dx9[2], s.v_enu_mps + dx9[3:6], q_normalize(q_mul(q_from_rotvec(dx9[6:9]), s.q_nb)))


def moving_state() -> NavState:
    return NavState(0.0, LAT, LON, H, np.array([12.0, 5.0, 0.3]), q_from_euler(0.7, 0.05, -0.03))


def fd_phi(ekf: ErrorStateEKF, f_m, w_m, dt, eps) -> np.ndarray:
    """Finite-difference state-transition matrix of the real propagation."""
    nom1 = propagate(ekf.nominal, f_m - ekf.b_a, w_m - ekf.b_g, dt)
    Phi = np.zeros((N_STATES, N_STATES))
    for j in range(N_STATES):
        dx = np.zeros(N_STATES)
        dx[j] = eps[j]
        truth0 = inject(ekf.nominal, dx[:9])
        truth1 = propagate(truth0, f_m - (ekf.b_a + dx[BA_]), w_m - (ekf.b_g + dx[BG_]), dt)
        Phi[:, j] = np.concatenate([error_between(truth1, nom1), dx[9:]]) / eps[j]
    return Phi


EPS = np.array([1.0] * 3 + [1e-2] * 3 + [1e-5] * 3 + [1e-3] * 3 + [1e-5] * 3)


def model_phi(ekf, f_b, dt):
    F = ekf.jacobian_F(f_b)
    Fd = F * dt
    return np.eye(N_STATES) + Fd + 0.5 * Fd @ Fd


# ---- F against finite differences (sec 11) ------------------------------------------

def test_F_matches_finite_difference_jacobian():
    ekf = ErrorStateEKF(moving_state(), np.eye(N_STATES), NOISE,
                        b_a=[0.05, -0.03, 0.02], b_g=[1e-3, -2e-3, 5e-4])
    f_m, w_m, dt = np.array([0.8, -0.4, 9.9]), np.array([0.02, -0.01, 0.15]), 0.01
    fd = fd_phi(ekf, f_m, w_m, dt, EPS)
    mod = model_phi(ekf, f_m - ekf.b_a, dt)
    # bias rows: the truth's biases are constant, the model decays them by dt/tau (1e-4)
    np.testing.assert_allclose(fd, mod, atol=5e-4)


def test_the_documented_plus_sign_on_f_cross_psi_is_wrong_for_this_convention():
    # sec 8.3 prints +[f^n]x psi. Under the truth-minus-estimate convention of sec 8.5/8.6
    # the finite-difference Jacobian must reject it -- this pins the correction.
    ekf = ErrorStateEKF(moving_state(), np.eye(N_STATES), NOISE)
    f_m, w_m, dt = np.array([0.8, -0.4, 9.9]), np.array([0.02, -0.01, 0.15]), 0.01
    fd = fd_phi(ekf, f_m, w_m, dt, EPS)
    F = ekf.jacobian_F(f_m)
    F_doc = F.copy()
    F_doc[3:6, A_] = +skew(q_rotate(ekf.nominal.q_nb, f_m))
    Fd = F_doc * dt
    wrong = np.eye(N_STATES) + Fd + 0.5 * Fd @ Fd
    assert np.max(np.abs(fd - wrong)) > 0.1  # ~ 2 * g * dt


def test_unmeasured_gyro_axes_do_not_couple_bias_into_attitude():
    ekf = ErrorStateEKF(moving_state(), np.eye(N_STATES), NOISE)
    F = ekf.jacobian_F(np.array([0, 0, 9.8]), gyro_valid=(False, False, True))
    R = ekf.jacobian_F(np.array([0, 0, 9.8]))[A_, BG_]
    np.testing.assert_array_equal(F[A_, BG_][:, :2], 0.0)
    np.testing.assert_array_equal(F[A_, BG_][:, 2], R[:, 2])


# ---- H against finite differences ---------------------------------------------------------

def test_gnss_H_matches_finite_difference():
    nom = moving_state()
    ekf = ErrorStateEKF(nom, np.eye(N_STATES), NOISE)
    rng = np.random.default_rng(3)
    for _ in range(20):
        dx = np.concatenate([rng.normal(0, 5, 3), rng.normal(0, 0.3, 3), np.zeros(9)])
        truth = inject(nom, dx[:9])
        ve, vn = truth.v_enu_mps[:2]
        fix = GnssSample(t_s=0.0, lat_rad=truth.lat_rad, lon_rad=truth.lon_rad, h_m=truth.h_m,
                         horizontal_accuracy_m=3.0, speed_mps=math.hypot(ve, vn),
                         bearing_rad=math.atan2(ve, vn))
        nu, H, _ = ekf.gnss_innovation(fix)
        np.testing.assert_allclose(nu, H @ dx, atol=1e-6)


def test_levelling_H_matches_finite_difference():
    nom = NavState(0.0, LAT, LON, H, np.zeros(3), q_from_euler(1.0, 0.02, -0.01))
    ekf = ErrorStateEKF(nom, np.eye(N_STATES), NOISE, b_a=[0.01, -0.02, 0.03])
    rng = np.random.default_rng(4)
    g = float(normal_gravity(LAT, H))

    def residual(dx):
        truth = inject(nom, dx[:9])
        f_true_b = q_rotate(q_conj(truth.q_nb), np.array([0, 0, g]))  # at rest: f = -g
        imu = ImuSample(t_s=0.0, accel_mps2=f_true_b + ekf.b_a + dx[BA_])
        nu, Hm = ekf.levelling_innovation(imu)
        return np.linalg.norm(nu - Hm @ dx), np.linalg.norm(Hm @ dx)

    for _ in range(20):
        dx = np.zeros(N_STATES)
        dx[A_] = rng.normal(0, 2e-3, 3)
        dx[BA_] = rng.normal(0, 5e-3, 3)
        r1, lin = residual(dx)
        r01, _ = residual(0.1 * dx)
        assert r1 < 0.01 * lin  # first order dominates ...
        assert r01 / r1 < 0.02  # ... and the remainder is QUADRATIC: H is the exact Jacobian


# ---- update mechanics ---------------------------------------------------------------------

def static_ekf(P0_diag=None):
    q = q_from_euler(0.3, 0.0, 0.0)
    P0 = np.diag(P0_diag if P0_diag is not None else
                 [9, 9, 25, 0.01, 0.01, 0.01, 4e-4, 4e-4, 0.25, 4e-4, 4e-4, 4e-4, 1e-6, 1e-6, 1e-6])
    return ErrorStateEKF(NavState(0.0, LAT, LON, H, np.zeros(3), q), P0, NOISE)


def fix_at(s: NavState, t: float, acc: float = 3.0, d_north: float = 0.0) -> GnssSample:
    return GnssSample(t_s=t, lat_rad=s.lat_rad + d_north / (float(meridian_radius(s.lat_rad)) + s.h_m),
                      lon_rad=s.lon_rad, h_m=s.h_m, horizontal_accuracy_m=acc)


def test_gnss_update_shrinks_covariance_and_moves_state_toward_fix():
    ekf = static_ekf()
    tr0 = np.trace(ekf.P[:3, :3])
    assert ekf.update_gnss(fix_at(ekf.nominal, 0.0, d_north=4.0))
    assert np.trace(ekf.P[:3, :3]) < tr0
    moved = (ekf.nominal.lat_rad - LAT) * (float(meridian_radius(LAT)) + H)
    assert 0.0 < moved < 4.0  # partially toward the 4 m-north fix, weighted by P vs R
    assert np.allclose(ekf.P, ekf.P.T)
    np.linalg.cholesky(ekf.P)


def test_gating_rejects_a_1km_outlier_and_logs_it():
    ekf = static_ekf()
    assert not ekf.update_gnss(fix_at(ekf.nominal, 0.0, d_north=1000.0))
    rec = ekf.log[-1]
    assert rec.kind == "gnss" and not rec.accepted and rec.nis > 100
    assert ekf.nominal.lat_rad == LAT  # rejected: state untouched


def test_future_fix_is_refused():
    ekf = static_ekf()
    with pytest.raises(ValueError, match="non-causal"):
        ekf.update_gnss(GnssSample(t_s=0.0, t_received_s=5.0, lat_rad=LAT, lon_rad=LON, h_m=H,
                                   horizontal_accuracy_m=3.0))


def test_delayed_fix_is_compared_with_the_state_at_its_epoch():
    # Move north at 10 m/s for 5 s; a fix of the t=1 s position, received at t=5 s, must
    # produce ~zero innovation (it matches the epoch state), not -40 m.
    s0 = NavState(0.0, LAT, LON, H, np.array([0.0, 10.0, 0.0]), q_from_euler(math.pi / 2, 0, 0))
    ekf = ErrorStateEKF(s0, np.eye(N_STATES), NOISE)
    f_b = q_rotate(q_conj(s0.q_nb), -gravity_enu(LAT, H))
    w_b = q_rotate(q_conj(s0.q_nb), earth_rate_enu(LAT))
    states = {}
    for k in range(50):
        ekf.predict(ImuSample(t_s=k * 0.1, accel_mps2=f_b, gyro_radps=w_b), 0.1)
        states[round((k + 1) * 0.1, 1)] = ekf.nominal
    s1 = states[1.0]
    fix = GnssSample(t_s=1.0, t_received_s=5.0, lat_rad=s1.lat_rad, lon_rad=s1.lon_rad, h_m=s1.h_m,
                     horizontal_accuracy_m=3.0)
    nu, _, _ = ekf.gnss_innovation(fix)
    assert np.linalg.norm(nu[:3]) < 1e-6


def test_zupt_drives_velocity_to_zero():
    ekf = static_ekf()
    ekf.nominal = NavState(0.0, LAT, LON, H, np.array([0.3, -0.2, 0.05]), ekf.nominal.q_nb)
    for _ in range(20):
        ekf.update_zupt(0.0, sigma_mps=0.01)
    assert np.linalg.norm(ekf.nominal.v_enu_mps) < 0.01


def test_broken_covariance_raises_instead_of_continuing():
    ekf = static_ekf()
    ekf.P[0, 0] = -1.0
    with pytest.raises(CovarianceError):
        ekf.predict(ImuSample(t_s=0.0, accel_mps2=[0, 0, 9.81]), 0.1)


def test_covariance_grows_without_aiding():
    ekf = static_ekf()
    f_b = q_rotate(q_conj(ekf.nominal.q_nb), -gravity_enu(LAT, H))
    traces = []
    for k in range(100):
        ekf.predict(ImuSample(t_s=k * 0.1, accel_mps2=f_b), 0.1)
        traces.append(np.trace(ekf.P[:6, :6]))
    assert np.all(np.diff(traces) > 0)


# ---- consistency: Monte Carlo NEES / NIS (sec 8.7) ------------------------------------
# The truth must satisfy the filter's model for NEES to mean anything: exact IMU signal
# (constant 20 m/s along a parallel) + the DECLARED white noise + constant biases drawn
# from the prior (tau long enough that "constant" is the model), GNSS simulated with the
# DECLARED noise, small-angle initial errors. (A 29 deg yaw prior or a ZUPT with a
# conservative sigma on an exactly-stationary truth break those premises -- found and
# excluded while writing this test, see PROJECT_STATUS.md Phase 4.)
MC_NOISE = NoiseParams(0.05, 0.005, 0.02, 1e5, 1e-3, 1e5)


def run_moving_trial(seed: int, T: float = 60.0, dt: float = 0.1, v: float = 20.0):
    rng = np.random.default_rng(seed)
    vv = np.array([v, 0.0, 0.0])
    w_ie, w_en = earth_rate_enu(LAT), transport_rate_enu(LAT, H, vv)
    f_exact, w_exact = np.cross(2 * w_ie + w_en, vv) - gravity_enu(LAT, H), w_ie + w_en
    truth = NavState(0.0, LAT, LON, H, vv, np.array([1.0, 0, 0, 0]))
    P0 = np.diag([9, 9, 25, 0.04, 0.04, 0.04, 1e-4, 1e-4, 2.5e-3, 4e-4, 4e-4, 4e-4, 1e-6, 1e-6, 1e-6])
    dx0 = rng.multivariate_normal(np.zeros(N_STATES), P0)
    ekf = ErrorStateEKF(inject(truth, -dx0[:9]), P0, MC_NOISE)
    b_a, b_g = dx0[BA_], dx0[BG_]
    sa = MC_NOISE.accel_white_mps2_rtHz / math.sqrt(dt)
    sg = MC_NOISE.gyro_white_radps_rtHz / math.sqrt(dt)
    rm = float(meridian_radius(LAT)) + H
    rn = (float(prime_vertical_radius(LAT)) + H) * math.cos(LAT)
    for k in range(int(T / dt)):
        t = k * dt
        ekf.predict(ImuSample(t_s=t, accel_mps2=f_exact + b_a + rng.normal(0, sa, 3),
                              gyro_radps=w_exact + b_g + rng.normal(0, sg, 3)), dt)
        truth = propagate(truth, f_exact, w_exact, dt)
        if (k + 1) % 10 == 0:  # 1 Hz GNSS position + velocity
            ve, vn = truth.v_enu_mps[:2] + rng.normal(0, 0.5, 2)
            ekf.update_gnss(GnssSample(
                t_s=t + dt, horizontal_accuracy_m=3.0, lat_rad=truth.lat_rad + rng.normal(0, 3) / rm,
                lon_rad=truth.lon_rad + rng.normal(0, 3) / rn, h_m=truth.h_m + rng.normal(0, 7.5),
                speed_mps=math.hypot(ve, vn), bearing_rad=math.atan2(ve, vn)), velocity_sigma_mps=0.5)
    err = np.concatenate([error_between(truth, ekf.nominal), b_a - ekf.b_a, b_g - ekf.b_g])
    nis = [r.nis for r in ekf.log if r.accepted]
    return float(err @ np.linalg.solve(ekf.P, err)), float(np.sum(nis)), len(nis)


@pytest.mark.slow
def test_filter_is_consistent_monte_carlo_nees_and_nis():
    n = 20
    runs = [run_moving_trial(s) for s in range(n)]
    nees = np.array([r[0] for r in runs])
    lo, hi = chi2.ppf([0.005, 0.995], N_STATES * n) / n
    assert lo <= nees.mean() <= hi, f"mean NEES {nees.mean():.2f} outside [{lo:.2f}, {hi:.2f}]"
    nis_sum, n_upd = sum(r[1] for r in runs), sum(r[2] for r in runs)
    lo, hi = chi2.ppf([0.005, 0.995], 5 * n_upd) / n_upd  # 5-dim GNSS pos + horiz vel
    assert lo <= nis_sum / n_upd <= hi, f"mean NIS {nis_sum / n_upd:.3f} outside [{lo:.3f}, {hi:.3f}]"
