"""15-state error-state EKF: INS + GNSS (+ ZUPT, accelerometer levelling).

State (docs/navigation_math.md sec 8.1):  dx = [dp_ENU(3), dv_ENU(3), psi(3), db_a(3), db_g(3)]

Conventions -- declared ONCE here, enforced by tests/navigation/test_ekf.py:
* Every error is TRUTH MINUS ESTIMATE:  p = p_hat + dp,  v = v_hat + dv,
  b = b_hat + db,  and attitude  R_b^n = (I + [psi]x) R_hat_b^n
  (sec 8.2 writes R_hat = (I - [psi]x) R -- the same thing to first order). Hence the
  innovation is z - h(x_hat) = H dx with H_pos = [I 0 0 0 0] (sec 8.5), and the correction
  R <- (I + [psi]x) R_hat (sec 8.6).
* Continuous error dynamics under THIS convention:

      dp'  = dv
      dv'  = -[f^n]x psi - R db_a - [2 w_ie + w_en]x dv + (d g/d h) dh e_U
      psi' = -[w_in]x psi - R db_g
      db'  = -db / tau   (first-order Gauss-Markov)

  Note the sign of the [f^n]x psi term: sec 8.3 prints ``+[f^n]x psi``, which belongs
  to the opposite (estimate-minus-truth) convention; with the truth-minus-estimate
  convention that sec 8.5/8.6 use, it is MINUS. The finite-difference Jacobian test
  (sec 11) confirms the sign used here.

GNSS latency: a fix describes its epoch ``t_s``, received ``latency`` later. The
innovation is formed against the nominal state AT THE EPOCH (kept in a short history
buffer) and applied to the current error state -- the standard approximation for a
delay much shorter than the time constant of the error growth. Documented, not hidden.

Numerical guards (sec 8.6): Joseph-form update, symmetrisation, and a Cholesky
positive-definiteness assertion after every predict and update -- a broken P RAISES.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from navcore.common.constants import FREE_AIR_GRADIENT_PER_S2
from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius
from navcore.geometry.quaternion import (
    q_conj,
    q_from_rotvec,
    q_mul,
    q_normalize,
    q_to_dcm,
    q_to_rotvec,
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

N_STATES = 15
P_, V_, A_, BA_, BG_ = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)

# chi-square 99.9 % quantiles for NIS gating, by measurement dimension
CHI2_999 = {1: 10.828, 2: 13.816, 3: 16.266, 4: 18.467, 5: 20.515, 6: 22.458}


class CovarianceError(RuntimeError):
    """P lost symmetry or positive-definiteness. The filter must stop, not continue."""


@dataclass(frozen=True)
class NoiseParams:
    """Process noise (sec 8.4). Defaults are NOT tuned values -- construct from the
    measured reports/phase4/imu_noise.json (see ``from_report``)."""

    accel_white_mps2_rtHz: float
    gyro_white_radps_rtHz: float
    accel_bias_sigma_mps2: float
    accel_bias_tau_s: float
    gyro_bias_sigma_radps: float
    gyro_bias_tau_s: float
    # angular-rate PSD standing in for axes the IMU does NOT measure (gyro_valid False):
    # the true rate there is unobserved, so its variance enters as process noise.
    unmeasured_rate_radps_rtHz: float = 0.02

    @classmethod
    def from_report(cls, filter_parameters: dict, **overrides) -> NoiseParams:
        p = filter_parameters
        return cls(accel_white_mps2_rtHz=p["accel_white_noise_density"],
                   gyro_white_radps_rtHz=p["gyro_white_noise_density"],
                   accel_bias_sigma_mps2=p["accel_bias_sigma"], accel_bias_tau_s=p["accel_bias_tau_s"],
                   gyro_bias_sigma_radps=p["gyro_bias_sigma"], gyro_bias_tau_s=p["gyro_bias_tau_s"],
                   **overrides)


@dataclass
class UpdateRecord:
    kind: str
    t_s: float
    accepted: bool
    nis: float
    dim: int


@dataclass
class ErrorStateEKF:
    nominal: NavState
    P: np.ndarray
    noise: NoiseParams
    b_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    history_s: float = 15.0
    log: list[UpdateRecord] = field(default_factory=list)
    _hist: deque = field(default_factory=deque, repr=False)
    _last_imu: ImuSample | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.P = np.array(self.P, dtype=np.float64)
        if self.P.shape != (N_STATES, N_STATES):
            raise ValueError(f"P must be {N_STATES}x{N_STATES}")
        self.b_a = np.array(self.b_a, dtype=np.float64)
        self.b_g = np.array(self.b_g, dtype=np.float64)
        self._check_P("init")
        self._hist.append(self.nominal)

    # ---- guards ---------------------------------------------------------------------
    def _check_P(self, where: str) -> None:
        if not np.all(np.isfinite(self.P)):
            raise CovarianceError(f"P non-finite after {where}")
        self.P = 0.5 * (self.P + self.P.T)
        try:
            np.linalg.cholesky(self.P)
        except np.linalg.LinAlgError as exc:
            raise CovarianceError(f"P not positive-definite after {where}") from exc

    # ---- model --------------------------------------------------------------------------
    def jacobian_F(self, f_b: np.ndarray, gyro_valid=(True, True, True)) -> np.ndarray:
        """Continuous-time error-dynamics matrix F at the current nominal state."""
        s = self.nominal
        R = q_to_dcm(s.q_nb)
        f_n = R @ f_b
        w_ie = earth_rate_enu(s.lat_rad)
        w_en = transport_rate_enu(s.lat_rad, s.h_m, s.v_enu_mps)
        F = np.zeros((N_STATES, N_STATES))
        F[P_, V_] = np.eye(3)
        F[V_, V_] = -skew(2.0 * w_ie + w_en)
        F[V_, A_] = -skew(f_n)
        F[V_, BA_] = -R
        F[5, 2] = FREE_AIR_GRADIENT_PER_S2  # g_U = -gamma(h): d(g_U)/dh = +3.086e-6
        F[A_, A_] = -skew(w_ie + w_en)
        # gyro bias only drives attitude through axes the IMU actually measures
        F[A_, BG_] = -R * np.asarray(gyro_valid, dtype=np.float64)[None, :]
        F[BA_, BA_] = -np.eye(3) / self.noise.accel_bias_tau_s
        F[BG_, BG_] = -np.eye(3) / self.noise.gyro_bias_tau_s
        return F

    def _Q(self, dt: float, gyro_valid) -> np.ndarray:
        n = self.noise
        R = q_to_dcm(self.nominal.q_nb)
        g_psd = np.array([n.gyro_white_radps_rtHz if ok else n.unmeasured_rate_radps_rtHz
                          for ok in gyro_valid]) ** 2
        Q = np.zeros((N_STATES, N_STATES))
        Q[V_, V_] = n.accel_white_mps2_rtHz ** 2 * dt * np.eye(3)
        Q[A_, A_] = R @ np.diag(g_psd) @ R.T * dt
        Q[BA_, BA_] = 2.0 * n.accel_bias_sigma_mps2 ** 2 / n.accel_bias_tau_s * dt * np.eye(3)
        qbg = 2.0 * n.gyro_bias_sigma_radps ** 2 / n.gyro_bias_tau_s * dt
        Q[BG_, BG_] = np.diag([qbg if ok else 0.0 for ok in gyro_valid])
        return Q

    # ---- predict ----------------------------------------------------------------------
    def predict(self, imu: ImuSample, dt_s: float) -> None:
        """Propagate nominal state and covariance over ``dt_s`` using ``imu`` (the sample
        at the START of the interval), bias-corrected on the measured axes only."""
        valid = np.asarray(imu.gyro_valid, dtype=bool)
        f_b = imu.accel_mps2 - self.b_a
        # Unmeasured axes (reduced IMU): assume the body does not rotate RELATIVE TO THE
        # LOCAL-LEVEL FRAME about them, i.e. use R^T w_in (Earth rate + transport rate), not
        # zero. Zero would mean "no rotation relative to inertial space" and tilt the INS at
        # Omega cos(phi) ~ 4.4e-5 rad/s (Phase 7 finding; ~350 m in 100 s of leaked gravity).
        # The vehicle's real, unmodelled pitch/roll rate stays in Q (unmeasured_rate PSD).
        s = self.nominal
        w_in_b = q_to_dcm(s.q_nb).T @ (earth_rate_enu(s.lat_rad)
                                       + transport_rate_enu(s.lat_rad, s.h_m, s.v_enu_mps))
        w_b = np.where(valid, imu.gyro_radps - self.b_g, w_in_b)
        F = self.jacobian_F(f_b, imu.gyro_valid)
        Fd = F * dt_s
        Phi = np.eye(N_STATES) + Fd + 0.5 * Fd @ Fd
        self.P = Phi @ self.P @ Phi.T + self._Q(dt_s, imu.gyro_valid)
        # unmeasured gyro-bias states are not part of the model: DECOUPLE them (no rows or
        # columns of correlation) at their prior variance. (Pinning them at ~1e-12 while
        # position variance grows past 1e8 in an outage made P numerically indefinite --
        # found on IO-VNBD S3c in Phase 4.)
        for i, ok in enumerate(imu.gyro_valid):
            if not ok:
                k = BG_.start + i
                self.P[k, :] = 0.0
                self.P[:, k] = 0.0
                self.P[k, k] = self.noise.gyro_bias_sigma_radps ** 2
        self.nominal = propagate(self.nominal, f_b, w_b, dt_s)
        self._last_imu = imu
        self._hist.append(self.nominal)
        while self._hist and self._hist[0].t_s < self.nominal.t_s - self.history_s:
            self._hist.popleft()
        self._check_P("predict")

    # ---- generic correction ---------------------------------------------------------
    def update(self, kind: str, t_s: float, nu: np.ndarray, H: np.ndarray, Rm: np.ndarray,
               gate: bool = True) -> bool:
        """Public generic measurement update (used by navcore.nhc and navcore.fusion):
        innovation ``nu`` = z - h(x_hat) = H dx under the truth-minus-estimate convention.
        NIS-gated, Joseph form, PD-checked, logged -- exactly like the built-in updates."""
        nu = np.atleast_1d(np.asarray(nu, dtype=np.float64))
        H = np.atleast_2d(np.asarray(H, dtype=np.float64))
        Rm = np.atleast_2d(np.asarray(Rm, dtype=np.float64))
        if H.shape != (nu.size, N_STATES) or Rm.shape != (nu.size, nu.size):
            raise ValueError(f"update {kind!r}: inconsistent shapes nu {nu.shape}, H {H.shape}, R {Rm.shape}")
        return self._update(kind, t_s, nu, H, Rm, gate)

    def _update(self, kind: str, t_s: float, nu: np.ndarray, H: np.ndarray, Rm: np.ndarray,
                gate: bool = True) -> bool:
        S = H @ self.P @ H.T + Rm
        nis = float(nu @ np.linalg.solve(S, nu))
        dim = nu.size
        if gate and nis > CHI2_999[dim]:
            self.log.append(UpdateRecord(kind, t_s, False, nis, dim))
            return False
        K = np.linalg.solve(S, H @ self.P).T  # P H^T S^-1 (S symmetric)
        dx = K @ nu
        IKH = np.eye(N_STATES) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T  # Joseph form (sec 8.6)
        self._inject(dx)
        self._check_P(f"update:{kind}")
        self.log.append(UpdateRecord(kind, t_s, True, nis, dim))
        return True

    def _inject(self, dx: np.ndarray) -> None:
        """Absorb dx into the nominal state (sec 8.6) and reset the error to zero."""
        s = self.nominal
        rm = float(meridian_radius(s.lat_rad)) + s.h_m
        rn = float(prime_vertical_radius(s.lat_rad)) + s.h_m
        q = q_normalize(q_mul(q_from_rotvec(dx[A_]), s.q_nb))  # R <- (I + [psi]x) R_hat
        self.nominal = NavState(t_s=s.t_s, lat_rad=s.lat_rad + dx[1] / rm,
                                lon_rad=s.lon_rad + dx[0] / (rn * math.cos(s.lat_rad)),
                                h_m=s.h_m + dx[2], v_enu_mps=s.v_enu_mps + dx[V_], q_nb=q)
        self.b_a = self.b_a + dx[BA_]
        self.b_g = self.b_g + dx[BG_]
        if self._hist:
            self._hist[-1] = self.nominal

    def _state_at(self, t_s: float) -> NavState:
        if not self._hist:
            return self.nominal
        return min(self._hist, key=lambda s: abs(s.t_s - t_s))

    # ---- measurements -------------------------------------------------------------------
    def update_gnss(self, fix: GnssSample, vertical_sigma_factor: float = 2.5,
                    velocity_sigma_mps: float | None = 0.5, min_speed_for_velocity_mps: float = 2.0
                    ) -> bool:
        """GNSS position (+ horizontal velocity when moving) update at the fix EPOCH.

        Position noise from the fix's reported horizontal accuracy (vertical = factor x
        horizontal: phones report no vertical accuracy). Course over ground is meaningless
        at low speed, so velocity is used only above ``min_speed_for_velocity_mps``.
        """
        nu, H, Rm = self.gnss_innovation(fix, vertical_sigma_factor, velocity_sigma_mps,
                                         min_speed_for_velocity_mps)
        return self._update("gnss", fix.t_s, nu, H, Rm)

    def gnss_innovation(self, fix: GnssSample, vertical_sigma_factor: float = 2.5,
                        velocity_sigma_mps: float | None = 0.5,
                        min_speed_for_velocity_mps: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(nu, H, R) for a GNSS fix, formed against the nominal state at the fix epoch."""
        if fix.t_received_s > self.nominal.t_s + 1e-9:
            raise ValueError("fix used before it was received (non-causal)")
        ref = self._state_at(fix.t_s)
        rm = float(meridian_radius(ref.lat_rad)) + ref.h_m
        rn = float(prime_vertical_radius(ref.lat_rad)) + ref.h_m
        dlon = math.remainder(fix.lon_rad - ref.lon_rad, 2.0 * math.pi)
        nu = [dlon * rn * math.cos(ref.lat_rad), (fix.lat_rad - ref.lat_rad) * rm, fix.h_m - ref.h_m]
        sig = fix.horizontal_accuracy_m
        var = [sig ** 2, sig ** 2, (vertical_sigma_factor * sig) ** 2]
        H = [np.hstack([np.eye(3), np.zeros((3, 12))])]
        v_en = fix.velocity_enu()
        if velocity_sigma_mps is not None and v_en is not None and fix.speed_mps >= min_speed_for_velocity_mps:
            nu += list(v_en - ref.v_enu_mps[:2])
            var += [velocity_sigma_mps ** 2] * 2
            Hv = np.zeros((2, N_STATES))
            Hv[0, 3] = Hv[1, 4] = 1.0
            H.append(Hv)
        return np.asarray(nu), np.vstack(H), np.diag(var)

    def update_zupt(self, t_s: float, sigma_mps: float = 0.05) -> bool:
        """Zero-velocity update: z = 0 - v_hat, H = [0 I 0 0 0] (sec 8.5)."""
        H = np.zeros((3, N_STATES))
        H[:, V_] = np.eye(3)
        return self._update("zupt", t_s, -self.nominal.v_enu_mps, H, sigma_mps ** 2 * np.eye(3))

    def update_levelling(self, imu: ImuSample, sigma_mps2: float = 0.2) -> bool:
        """Accelerometer levelling when quasi-static: true R f + g = 0, so

            nu = -(R_hat f_hat + g) = -[f^n]x psi - R db_a  ->  H = [0 0 -[f^n]x -R 0].

        Only valid when linear acceleration is small; the caller decides when (and
        ``sigma`` must cover the residual linear acceleration).
        """
        nu, H = self.levelling_innovation(imu)
        return self._update("level", imu.t_s, nu, H, sigma_mps2 ** 2 * np.eye(3))

    def levelling_innovation(self, imu: ImuSample) -> tuple[np.ndarray, np.ndarray]:
        s = self.nominal
        R = q_to_dcm(s.q_nb)
        f_n = R @ (imu.accel_mps2 - self.b_a)
        nu = -(f_n + gravity_enu(s.lat_rad, s.h_m))
        H = np.zeros((3, N_STATES))
        H[:, A_] = -skew(f_n)
        H[:, BA_] = -R
        return nu, H

    # ---- diagnostics --------------------------------------------------------------------
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.P))


def error_between(truth: NavState, est: NavState) -> np.ndarray:
    """dx[0:9] = truth - estimate in the filter's convention (ENU metres, m/s, psi rad).
    Used by tests and evaluation (NEES), never by the filter itself."""
    rm = float(meridian_radius(est.lat_rad)) + est.h_m
    rn = float(prime_vertical_radius(est.lat_rad)) + est.h_m
    dp = np.array([math.remainder(truth.lon_rad - est.lon_rad, 2 * math.pi) * rn * math.cos(est.lat_rad),
                   (truth.lat_rad - est.lat_rad) * rm, truth.h_m - est.h_m])
    psi = q_to_rotvec(q_mul(truth.q_nb, q_conj(est.q_nb)))  # R = exp(psi) R_hat
    return np.concatenate([dp, truth.v_enu_mps - est.v_enu_mps, psi])
