"""Streaming navigation engine: initialisation + heading hypotheses + fused navigator.

Feed it one phone sample at a time (``on_sample``), with the GNSS fixes RECEIVED since the
previous sample, exactly as a phone would deliver them. It is causal: nothing it does
depends on data after the current sample -- with one explicitly-labelled exception, a
``mount`` passed in by the caller (``mount_mode="given"``), which is how the Phase 4/7
evaluations worked (their mount was estimated from the WHOLE drive in advance, a
lookahead). ``mount_mode="causal"`` estimates the tilt from the accelerometer seen before
initialisation only; the heading is resolved by the hypotheses below in either mode.

Lifecycle (the same protocol as scripts/evaluate/phase4_baseline.py and phase7):
  WAITING    until a received fix shows speed > ``init_min_speed_mps`` with a bearing
  ALIGNING   from the first sample at/after that fix's receipt time: if the mount yaw is
             not trusted, 8 navigators with vehicle-heading offsets k*45 deg run in
             parallel for ``mh_window_s``; the one with the lowest mean (capped) GNSS NIS
             plus 50 per lock-out reset is kept
  NAVIGATING a single FusedNavigator
Output (``EngineOutput``): the navigator's continuous output position (see OutputSmoother),
the raw filter position, 1-sigma horizontal, GNSS state, mode and counters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from navcore.alignment.mounting import MountingEstimate, YawFit, tilt_rotation, up_direction
from navcore.filtering.ekf import ErrorStateEKF, NoiseParams
from navcore.fusion.navigator import FusedNavigator, FusionConfig
from navcore.geometry.quaternion import dcm_to_q
from navcore.geometry.rotation import rot_z
from navcore.ins.mechanization import NavState
from navcore.sensors.samples import IOVNBD_AXIS_MAP, GnssSample, ImuAxisMap


@dataclass(frozen=True)
class EngineConfig:
    fusion: FusionConfig = field(default_factory=FusionConfig)
    init_min_speed_mps: float = 5.0
    mh_window_s: float = 120.0
    n_hypotheses: int = 8
    mount_mode: str = "causal"   # "causal" | "given"


@dataclass
class EngineOutput:
    t_s: float
    mode: str                 # ALIGNING | NAVIGATING
    lat_rad: float            # continuous OUTPUT position (never teleports)
    lon_rad: float
    h_m: float
    filter_lat_rad: float     # the filter's own estimate (jumps when evidence arrives)
    filter_lon_rad: float
    sigma_h_m: float          # filter horizontal 1-sigma (root of the EN covariance trace)
    gnss_state: str
    v_enu_mps: np.ndarray


def causal_mount(accel_b: np.ndarray, grav_b: np.ndarray | None = None, min_accel_samples: int = 50
                 ) -> tuple[MountingEstimate, str]:
    """Tilt from what has been seen so far; yaw is left to the heading hypotheses.

    With >= ``min_accel_samples`` (5 s) of accelerometer data, the accelerometer mean; with
    less -- a drive can start moving within its first second -- the phone's own GRAVITY
    channel, which is available from the first sample. Returns (mount, source)."""
    if len(accel_b) >= min_accel_samples or grav_b is None or not len(grav_b):
        src, data = "accelerometer", accel_b
    else:
        src, data = "gravity_channel", grav_b
    R = tilt_rotation(up_direction(np.asarray(data)) if len(data) >= 10 else
                      np.asarray(data, dtype=np.float64).mean(axis=0) / np.linalg.norm(np.asarray(data).mean(axis=0)))
    return MountingEstimate(R_vb=R, tilt_rad=math.acos(max(-1.0, min(1.0, float(R[2, 2])))), yaw_rad=0.0,
                            fit=YawFit(yaw_rad=0.0, scale=0.0, r2=0.0, n=0)), src


class NavigationEngine:
    def __init__(self, noise: NoiseParams, cfg: EngineConfig | None = None, models: dict | None = None,
                 mount: MountingEstimate | None = None, axis_map: ImuAxisMap = IOVNBD_AXIS_MAP) -> None:
        self.cfg = cfg or EngineConfig()
        if self.cfg.mount_mode not in ("causal", "given"):
            raise ValueError(f"mount_mode {self.cfg.mount_mode!r}")
        if self.cfg.mount_mode == "given" and mount is None:
            raise ValueError("mount_mode='given' needs a mount")
        self.noise, self.axis_map, self.mount = noise, axis_map, mount
        f = self.cfg.fusion
        self.models = {k: v for k, v in (models or {}).items()
                       if (k == "model_speed" and f.use_ai_speed) or (k == "model_accel" and f.use_ai_accel)}
        self.mode = "WAITING"
        self.navs: list[tuple[FusedNavigator, float]] = []
        self._acc_buf: list[np.ndarray] = []
        self._grav_buf: list[np.ndarray] = []
        self.tilt_source = "given" if self.cfg.mount_mode == "given" else None
        self._f0: GnssSample | None = None
        self.t_init: float | None = None
        self.chosen_offset_rad: float | None = None

    @property
    def nav(self) -> FusedNavigator | None:
        return self.navs[0][0] if len(self.navs) == 1 else None

    def _start(self, t: float) -> None:
        f0, cfg = self._f0, self.cfg
        if cfg.mount_mode == "given":
            mount = self.mount
        else:
            mount, self.tilt_source = causal_mount(np.array(self._acc_buf), np.array(self._grav_buf))
        self.mount = mount
        ok = mount.acceptable()
        offsets = [0.0] if ok else [k * 2 * math.pi / cfg.n_hypotheses for k in range(cfg.n_hypotheses)]
        veh_yaw = math.pi / 2 - f0.bearing_rad
        v0 = f0.velocity_enu()
        ys = math.radians(5.0 if ok else 25.0)
        n = self.noise
        for off in offsets:
            nominal = NavState(t_s=t, lat_rad=f0.lat_rad, lon_rad=f0.lon_rad, h_m=f0.h_m,
                               v_enu_mps=np.array([v0[0], v0[1], 0.0]),
                               q_nb=dcm_to_q(rot_z(veh_yaw + off) @ mount.R_vb))
            P0 = np.diag([f0.horizontal_accuracy_m ** 2] * 2 + [(2.5 * f0.horizontal_accuracy_m) ** 2]
                         + [0.25] * 3 + [math.radians(2.0) ** 2] * 2 + [ys ** 2]
                         + [n.accel_bias_sigma_mps2 ** 2] * 3 + [n.gyro_bias_sigma_radps ** 2] * 3)
            nav = FusedNavigator(ErrorStateEKF(nominal, P0, n), self.axis_map, cfg.fusion, R_vb=mount.R_vb,
                                 lateral_nhc_allowed=ok, **self.models)
            self.navs.append((nav, off))
        self.t_init, self.mode = t, "ALIGNING"

    @staticmethod
    def _score(item: tuple[FusedNavigator, float]) -> float:
        nav = item[0]
        g = [min(r.nis, 50.0) if r.accepted else 50.0 for r in nav.ekf.log if r.kind == "gnss"]
        return (float(np.mean(g)) if g else math.inf) + 50.0 * nav.reset.n_resets

    def on_sample(self, t: float, acc, grav, gyro, fixes: list[GnssSample] = ()) -> EngineOutput | None:
        """One phone sample plus the fixes received since the previous one."""
        if self.mode == "WAITING":
            self._acc_buf.append(np.asarray(acc, dtype=np.float64))
            self._grav_buf.append(np.asarray(grav, dtype=np.float64))
            for f in fixes:
                if (self._f0 is None and f.speed_mps is not None and f.speed_mps > self.cfg.init_min_speed_mps
                        and f.bearing_rad is not None):
                    self._f0 = f
            if self._f0 is None or t < self._f0.t_received_s - 1e-9:
                return None
            self._start(t)
            fixes = [f for f in fixes if f.t_received_s > t + 1e-9]  # f0 initialises; it is not fused
        for nav, _ in self.navs:
            nav.step(t, acc, grav, gyro, list(fixes))
        if self.mode == "ALIGNING" and len(self.navs) > 1 and t >= self.t_init + self.cfg.mh_window_s:
            self.navs = [min(self.navs, key=self._score)]
        if len(self.navs) == 1 and self.mode == "ALIGNING" and t >= self.t_init + self.cfg.mh_window_s:
            self.mode = "NAVIGATING"
            self.chosen_offset_rad = self.navs[0][1]
        nav = self.navs[0][0]  # while ALIGNING: hypothesis 0, flagged by ``mode`` (scoring all 8
        # at every sample would rescan every filter's whole update log)
        lat, lon, h = nav.output()
        s = nav.ekf.nominal
        return EngineOutput(t_s=t, mode=self.mode, lat_rad=lat, lon_rad=lon, h_m=h, filter_lat_rad=s.lat_rad,
                            filter_lon_rad=s.lon_rad, sigma_h_m=float(math.sqrt(nav.ekf.P[0, 0] + nav.ekf.P[1, 1])),
                            gnss_state=nav.sm.state.value, v_enu_mps=s.v_enu_mps.copy())


__all__ = ["EngineConfig", "EngineOutput", "NavigationEngine", "causal_mount"]
