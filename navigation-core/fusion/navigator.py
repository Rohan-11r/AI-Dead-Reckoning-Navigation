"""AI-assisted fused navigator: INS/EKF + GNSS state machine + AI measurements + NHC.

One step per IMU sample (phone channels only -- SIH26168 forbids CAN/OBD inputs):

  1. raw logged columns -> BODY via the evidence-based axis map (reduced IMU, Phase 4)
  2. rolling window; when a gap-free window is ready, run Model A (speed) and Model B
     (longitudinal acceleration) -- every ``ai_every`` samples
  3. Model B corrects the raw specific force BEFORE predict()        [use_ai_accel]
  4. EKF predict with the real dt
  5. GNSS state machine; each received fix is applied at its epoch with the state's noise
     inflation (a fix arriving in LOST is treated as the start of RECOVERING); gate
     lock-out recovery (navcore.recovery) stays active
  6. Model A speed update when the state policy asks for it (DEGRADED/LOST/RECOVERING)
     -- only on a FRESH prediction, at most every ``ai_speed_interval_s``, with its variance
     inflated by (model window / interval): consecutive windows overlap almost entirely, so
     their errors are nearly identical and must not be counted as independent evidence.
     (Applied at every 10 Hz sample, on real drives it made the filter over-confident, the
     GNSS gate then rejected 146/204 fixes and aided error went 7 m -> 42 m.)
                                                                  [use_ai_speed]
  7. NHC: vertical always (levelling suffices), lateral only with an ACCEPTED mount;
     relaxed in hard cornering                                    [use_nhc]
     -- by default ONLY when AI speed is enabled (``nhc_requires_ai_speed``)

DEFAULTS (Phase 7 real-data ablation, reports/phase7/fusion_evaluation.json):
  AI speed ON; NHC ON but only paired with AI speed (alone it hurt and diverged once;
  paired it gave the best outage results); Model B OFF -- it was fully trained, exported,
  validated and evaluated, and it DEGRADED real-world accuracy (aided p50 12.5 -> 37.9 m,
  60 s outage 364 -> 633 m). It stays available behind ``use_ai_accel`` for research.
  8. phone-only stillness -> ZUPT + accelerometer levelling
  9. OUTPUT: ``output()`` is the filter position plus an offset that absorbs every
     correction jump and glides back to zero within 10 s (navcore.recovery.manager.OutputSmoother) -- the
     reported trajectory never teleports; the filter estimate itself is unaffected.
  Optional GNSS recovery manager (``recovery``): after LOST, fixes are held until
  mutually consistent along the DR path, then fused with graded noise, with covariance
  inflation instead of reject-then-reset (navcore.recovery.manager.RecoveryManager).

Every switch exists so that the real-data ablation (scripts/evaluate/phase7_fusion_eval.py)
can measure each component's contribution with ONE code path. Everything is counted.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from navcore.filtering.ekf import ErrorStateEKF
from navcore.fusion.ai_models import ImuWindow, correct_specific_force, speed_innovation
from navcore.ins.mechanization import gravity_enu
from navcore.nhc.constraints import NhcConfig, apply_nhc
from navcore.recovery.gnss_reset import ConsecutiveRejectionReset
from navcore.recovery.manager import OutputSmoother, RecoveryConfig, RecoveryManager
from navcore.sensors.samples import GnssSample, ImuAxisMap, ImuSample
from navcore.state.gnss_state import POLICY, FilterPolicy, GnssState, GnssStateMachine


@dataclass(frozen=True)
class FusionConfig:
    use_ai_speed: bool = True
    use_ai_accel: bool = False          # Model B: measured to hurt on real drives (Phase 7)
    use_nhc: bool = True
    nhc_requires_ai_speed: bool = True  # NHC alone hurt on real drives; paired, it helped
    use_state_machine: bool = True
    ai_every: int = 1                 # run the models every N samples
    ai_speed_min_mps: float = 1.0
    ai_speed_sigma_floor_mps: float = 0.5
    ai_speed_interval_s: float = 1.0    # at most one speed update per interval
    ai_sample_dt_s: float = 0.1         # model input rate (10 Hz): window length = W * dt
    accel_imu_sigma_mps2: float = 1.04  # per-sample along-track IMU noise (0.33 m/s^2*sqrt(s) at 10 Hz)
    still_acc_std_mps2: float = 0.08
    still_norm_tol_mps2: float = 0.1
    still_gyro_radps: float = 0.02
    still_speed_mps: float = 1.0
    nhc: NhcConfig = field(default_factory=NhcConfig)
    recovery: RecoveryConfig | None = None   # GNSS recovery manager (None = Phase 7 behaviour)
    output_max_rate_mps: float = 15.0        # output smoother: correction speed for small offsets
    output_max_blend_s: float = 10.0         # ... and any offset is absorbed at >= |offset| / this


class FusedNavigator:
    def __init__(self, ekf: ErrorStateEKF, axis_map: ImuAxisMap, cfg: FusionConfig,
                 model_speed=None, model_accel=None, R_vb: np.ndarray | None = None,
                 lateral_nhc_allowed: bool = False, gamma_mps2: float = 9.81) -> None:
        self.ekf, self.axis_map, self.cfg = ekf, axis_map, cfg
        self.model_speed, self.model_accel = model_speed, model_accel
        self.R_vb = np.eye(3) if R_vb is None else np.asarray(R_vb, dtype=np.float64)
        self.lateral_ok = bool(lateral_nhc_allowed and R_vb is not None)
        self.sm = GnssStateMachine()
        self.reset = ConsecutiveRejectionReset()
        self.recovery = RecoveryManager(cfg.recovery) if cfg.recovery is not None else None
        self._tracked = False  # has GNSS ever been out of LOST? (start-up is not a recovery)
        self.smoother = OutputSmoother(cfg.output_max_rate_mps, cfg.output_max_blend_s)
        self.counts: Counter = Counter()
        sizes = [m.window for m in (model_speed, model_accel) if m is not None]
        self.win = ImuWindow(max(sizes) if sizes else 1)
        self._acc_hist: list[float] = []
        self._prev: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._n = 0
        self.ai_speed: tuple[float, float] | None = None
        self._ai_speed_fresh = False
        self._ai_speed_last_t = -math.inf
        W = getattr(model_speed, "window", 1)
        self._ai_speed_inflation = max(1.0, W * cfg.ai_sample_dt_s / cfg.ai_speed_interval_s)
        self.ai_accel: tuple[float, float] | None = None
        self.last_fix_speed: float | None = None
        self.gamma = gamma_mps2

    # ---- helpers ---------------------------------------------------------------------
    def _run_models(self) -> None:
        acc, grav, wz = self.win.arrays()
        for model, attr in ((self.model_speed, "ai_speed"), (self.model_accel, "ai_accel")):
            if model is not None:
                W = model.window
                setattr(self, attr, model.predict(acc[-W:], grav[-W:], wz[-W:]))
                self.counts[f"{attr}_inferences"] += 1
        self._ai_speed_fresh = self.model_speed is not None

    def _policy(self) -> FilterPolicy:
        if not self.cfg.use_state_machine:
            return FilterPolicy(use_gnss=True, gnss_noise_inflation=1.0, use_ai_speed=False, use_nhc=True)
        return self.sm.policy

    def _still(self, acc_norm: float, wz: float) -> bool:
        self._acc_hist.append(acc_norm)
        if len(self._acc_hist) > 10:
            self._acc_hist.pop(0)
        if len(self._acc_hist) < 10:
            return False
        speed_hint = self.ai_speed[0] if self.ai_speed is not None else self.last_fix_speed
        return (float(np.std(self._acc_hist)) < self.cfg.still_acc_std_mps2
                and abs(acc_norm - self.gamma) < self.cfg.still_norm_tol_mps2
                and abs(wz) < self.cfg.still_gyro_radps
                and speed_hint is not None and speed_hint < self.cfg.still_speed_mps)

    # ---- one IMU sample ---------------------------------------------------------------
    def step(self, t_s: float, acc, grav, gyro_raw, fixes: list[GnssSample] = ()) -> None:
        acc = np.asarray(acc, dtype=np.float64)
        grav = np.asarray(grav, dtype=np.float64)
        gyro_raw = np.asarray(gyro_raw, dtype=np.float64)
        body = self.axis_map.to_body(t_s, acc, gyro_raw)
        wz = float(body.gyro_radps[2])
        prev, self._prev = self._prev, (t_s, acc, grav, gyro_raw)
        self.win.push(t_s, body.accel_mps2, grav, wz)
        self._n += 1
        if self.win.ready() and self._n % self.cfg.ai_every == 0:
            self._run_models()
        if prev is None:
            return
        dt = t_s - prev[0]
        if dt <= 0:
            self.counts["skipped_nonpositive_dt"] += 1
            return
        imu = self.axis_map.to_body(prev[0], prev[1], prev[3])
        f_b = imu.accel_mps2
        if self.cfg.use_ai_accel and self.ai_accel is not None:
            s = self.ekf.nominal
            f_b, applied = correct_specific_force(
                f_b, s.q_nb, s.v_enu_mps, gravity_enu(s.lat_rad, s.h_m), self.ai_accel[0],
                self.ai_accel[1], self.cfg.accel_imu_sigma_mps2 ** 2)
            self.counts["ai_accel_corrections"] += applied
        self.ekf.predict(ImuSample(t_s=imu.t_s, accel_mps2=f_b, gyro_radps=imu.gyro_radps,
                                   frame=imu.frame, gyro_valid=imu.gyro_valid), dt)
        self.smoother.mark(self.ekf)
        self._updates(t_s, fixes, body, wz, float(np.linalg.norm(acc)))
        self.smoother.absorb(self.ekf, dt)

    def output(self) -> tuple[float, float, float]:
        """The reported position (lat_rad, lon_rad, h_m): continuous across corrections."""
        return self.smoother.output(self.ekf)

    def _fuse_gnss(self, t_s: float, fix: GnssSample, confirmed: bool) -> None:
        pol = self._policy()
        if self.cfg.use_state_machine and self.sm.state is GnssState.LOST:
            pol = POLICY[GnssState.RECOVERING]
        if not pol.use_gnss:
            return
        use = self.recovery.grade(fix) if self.recovery is not None else fix
        f = replace(use, horizontal_accuracy_m=use.horizontal_accuracy_m * pol.gnss_noise_inflation)
        if self.recovery is not None:
            self.recovery.admit(self.ekf, f, confirmed)
        ok = self.ekf.update_gnss(f)
        self.counts["gnss_accepted" if ok else "gnss_rejected"] += 1
        if self.reset.after_update(self.ekf, fix, ok):
            self.counts["gnss_lockout_resets"] += 1
        if self.cfg.use_state_machine:
            self.sm.on_fix(t_s, fix.horizontal_accuracy_m, ok)

    def _updates(self, t_s: float, fixes, body, wz: float, acc_norm: float) -> None:
        # ---- GNSS through the state machine (and the recovery manager, if enabled)
        if self.cfg.use_state_machine:
            self.sm.on_time(t_s)
            if self.sm.state is not GnssState.LOST:
                self._tracked = True
            elif self.recovery is not None and self._tracked:
                self.recovery.on_lost()  # a LOSS after tracking -- not the start-up LOST
        for fix in fixes:
            if fix.speed_mps is not None:
                self.last_fix_speed = fix.speed_mps
            if self.recovery is not None and self.cfg.use_state_machine:
                released, confirmed = self.recovery.submit(fix, self.ekf)
            else:
                released, confirmed = [fix], False
            for f in released:
                self._fuse_gnss(t_s, f, confirmed)
        pol = self._policy()

        # ---- AI speed as a measurement
        if (self.cfg.use_ai_speed and pol.use_ai_speed and self.ai_speed is not None and self._ai_speed_fresh
                and t_s - self._ai_speed_last_t >= self.cfg.ai_speed_interval_s - 1e-9):
            mean, var = self.ai_speed
            self._ai_speed_fresh, self._ai_speed_last_t = False, t_s
            inn = speed_innovation(self.ekf, mean, self.cfg.ai_speed_min_mps)
            if inn is None:
                self.counts["ai_speed_skipped_low_speed"] += 1
            else:
                sig2 = max(var, self.cfg.ai_speed_sigma_floor_mps ** 2) * self._ai_speed_inflation
                ok = self.ekf.update("ai_speed", t_s, inn[0], inn[1], np.array([[sig2]]))
                self.counts["ai_speed_accepted" if ok else "ai_speed_rejected"] += 1

        # ---- non-holonomic constraints
        if self.cfg.use_nhc and pol.use_nhc and (self.cfg.use_ai_speed or not self.cfg.nhc_requires_ai_speed):
            v = self.ekf.nominal.v_enu_mps
            lat_acc = math.hypot(v[0], v[1]) * wz  # centripetal ~ v * yaw rate
            r = apply_nhc(self.ekf, t_s, self.R_vb, self.lateral_ok, lat_acc, self.cfg.nhc)
            self.counts[f"nhc_{r['reason']}"] += 1

        # ---- stillness
        if self._still(acc_norm, wz):
            self.ekf.update_zupt(t_s)
            self.ekf.update_levelling(body)
            self.counts["zupt"] += 1
