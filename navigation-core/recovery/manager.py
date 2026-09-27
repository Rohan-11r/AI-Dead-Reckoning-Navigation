"""GNSS recovery after an outage: confirm, then fuse gradually; and a continuous output.

Two separate problems, two separate mechanisms:

1. WHICH fixes to trust when GNSS returns (``RecoveryManager``, changes the FILTER)
   * CONFIRM: after LOST, no fix is used until ``confirm_fixes`` consecutive fixes agree
     with EACH OTHER along the dead-reckoned path: the fix-to-fix displacement must match
     the dead-reckoned displacement between the same epochs within
       k * sqrt(acc1^2 + acc2^2 + (rel * |d_dr| + abs)^2
                + trace(P_v) dt^2 + (|d_dr| * sigma_yaw)^2)
     -- the last two terms are the FILTER'S OWN velocity and heading uncertainty. A drifted
     absolute position is not trusted, only the short-term displacement, and only as far as
     the filter itself claims to know it. (Without those terms, the Phase 9 real-drive
     benchmark rejected 1,390 good fixes: a reduced-IMU phone's DR heading is off by tens of
     degrees after an outage, so its 9 s displacement is not metre-accurate.) The held fixes
     are then released in order.
   * FAIL OPEN: if nothing is confirmed within ``confirm_timeout_s`` of the first fix after
     the loss, the latest fix is released unconfirmed and normal fusion resumes -- the
     manager must never keep GNSS out for long (counted).
   * GRADUAL: the released and following fixes are fused with their variance multiplied
     by ``r_schedule`` (e.g. 9, 4, 2), so the DR state is pulled in over several updates
     instead of snapped in one.
   * NO REJECT-THEN-RESET LOOP: a confirmed fix far from an over-confident DR state would
     fail the NIS gate, and three failures trigger a hard reset. Instead, when a CONFIRMED
     fix fails the gate because of its POSITION, the position covariance is inflated
     (rows and columns scaled, correlations kept; at most x100) until the fix is admissible;
     if that is not enough, or the mismatch is in velocity, P is left untouched (the
     first real-drive run showed that leaving a failed inflation in place compounds to
     kilometre-scale divergence). Velocity is
     never inflated: a huge velocity variance lets one position fix yank the velocity, and
     the position then runs away (seen on real drives in the first benchmark).
     This states the truth -- the DR covariance was too small -- rather than discarding
     the evidence. Counted.

2. HOW the reported position moves (``OutputSmoother``, never touches the filter)
   * the filter's estimate should jump when evidence arrives (that is the best estimate);
     a displayed/downstream trajectory must not teleport. The smoother reports
       output = filter position + offset
     and on every correction jump D it sets offset -= D (output unchanged at that instant),
     then glides the offset linearly to zero at max(``max_rate_mps``, |offset| /
     ``max_blend_s``), fixed at each correction. Output motion is therefore the
     filter's own propagated motion plus a bounded, continuous correction -- never a
     teleport -- and even a kilometre-scale correction is absorbed within ~max_blend_s
     (a fixed 15 m/s cap alone made the output lag the filter by minutes after real
     recoveries: median output error 375 m vs filter 22 m on S3a).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from navcore.filtering.ekf import CHI2_999, P_, ErrorStateEKF
from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius
from navcore.sensors.samples import GnssSample


@dataclass(frozen=True)
class RecoveryConfig:
    confirm_fixes: int = 2
    consistency_k: float = 3.0
    dr_rel_error: float = 0.05        # DR displacement error: 5 % of the distance ...
    dr_abs_error_m: float = 3.0       # ... plus 3 m
    confirm_max_gap_s: float = 14.0   # further apart cannot confirm each other (< EKF history_s 15 s,
                                      # so a held fix can still be fused at its own epoch)
    r_schedule: tuple[float, ...] = (9.0, 4.0, 2.0)  # variance multipliers, successive recovery updates
    inflate_to_admit: bool = True
    max_inflation: float = 100.0      # position variance x100 (sigma x10) at most
    confirm_timeout_s: float = 30.0   # fail open: never hold GNSS back longer than this


@dataclass
class _Held:
    fix: GnssSample
    dr_en: np.ndarray  # dead-reckoned horizontal position at the fix epoch [m, local EN]
    var_v: float = 0.0     # trace of the filter's horizontal velocity covariance [m^2/s^2]
    var_yaw: float = 0.0   # filter heading (yaw error) variance [rad^2]


def _en(lat0: float, lon0: float, lat: float, lon: float, h_m: float = 0.0) -> np.ndarray:
    """(lat, lon) minus (lat0, lon0) in local metres [E, N], radii at lat0 and height h_m."""
    rm = float(meridian_radius(lat0)) + h_m
    rn = (float(prime_vertical_radius(lat0)) + h_m) * math.cos(lat0)
    return np.array([math.remainder(lon - lon0, 2 * math.pi) * rn, (lat - lat0) * rm])


@dataclass
class RecoveryManager:
    cfg: RecoveryConfig = field(default_factory=RecoveryConfig)
    phase: str = "NORMAL"            # NORMAL | CONFIRMING | GRADUAL
    held: list[_Held] = field(default_factory=list)
    n_graded: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    _ref: tuple[float, float] | None = None
    _t_first: float | None = None

    def _count(self, k: str, n: int = 1) -> None:
        self.counts[k] = self.counts.get(k, 0) + n

    def on_lost(self) -> None:
        """GNSS declared LOST: every later fix must be confirmed before use."""
        if self.phase != "CONFIRMING":
            self.phase, self.held, self.n_graded, self._ref, self._t_first = "CONFIRMING", [], 0, None, None
            self._count("outages")

    def submit(self, fix: GnssSample, ekf: ErrorStateEKF) -> tuple[list[GnssSample], bool]:
        """Offer a received fix. Returns (fixes to fuse NOW, in time order -- possibly none,
        possibly several held ones; whether they were just CONFIRMED after an outage).
        Pass each through ``grade`` before fusing."""
        if self.phase != "CONFIRMING":
            return [fix], False
        if self._t_first is None:
            self._t_first = fix.t_received_s
        elif fix.t_received_s - self._t_first >= self.cfg.confirm_timeout_s:
            self._count("recovery_timeouts")  # fail open: resume normal fusion
            self.phase, self.held = "GRADUAL", []
            return [fix], False
        s = ekf._state_at(fix.t_s)
        if self._ref is None:
            self._ref = (s.lat_rad, s.lon_rad)
        h = _Held(fix, _en(*self._ref, s.lat_rad, s.lon_rad),
                  var_v=float(ekf.P[3, 3] + ekf.P[4, 4]), var_yaw=float(ekf.P[8, 8]))
        if self.held:
            prev = self.held[-1]
            if not self._consistent(prev, h):
                self._count("recovery_fixes_inconsistent")
                self.held = [h]  # the older one is the suspect only if the next agrees with the new one
                return [], False
        self.held.append(h)
        if len(self.held) < self.cfg.confirm_fixes:
            self._count("recovery_fixes_held")
            return [], False
        held, self.held = self.held, []
        self.phase = "GRADUAL"
        self._count("recoveries_confirmed")
        return [x.fix for x in held], True

    def _consistent(self, a: _Held, b: _Held) -> bool:
        c = self.cfg
        if b.fix.t_s - a.fix.t_s > c.confirm_max_gap_s:
            return False
        d_fix = _en(*self._ref, b.fix.lat_rad, b.fix.lon_rad) - _en(*self._ref, a.fix.lat_rad, a.fix.lon_rad)
        d_dr = b.dr_en - a.dr_en
        dist, dt = float(np.linalg.norm(d_dr)), b.fix.t_s - a.fix.t_s
        tol = c.consistency_k * math.sqrt(a.fix.horizontal_accuracy_m ** 2 + b.fix.horizontal_accuracy_m ** 2
                                          + (c.dr_rel_error * dist + c.dr_abs_error_m) ** 2
                                          + b.var_v * dt * dt + dist * dist * b.var_yaw)
        return float(np.linalg.norm(d_fix - d_dr)) <= tol

    def grade(self, fix: GnssSample) -> GnssSample:
        """The fix to fuse: accuracy inflated by the recovery schedule while GRADUAL."""
        if self.phase != "GRADUAL":
            return fix
        sched = self.cfg.r_schedule
        k = self.n_graded
        self.n_graded += 1
        if k >= len(sched):
            self.phase = "NORMAL"
            return fix
        self._count("recovery_graded_updates")
        return replace(fix, horizontal_accuracy_m=fix.horizontal_accuracy_m * math.sqrt(sched[k]))

    def admit(self, ekf: ErrorStateEKF, fix: GnssSample, confirmed: bool) -> None:
        """Before fusing a CONFIRMED recovery fix that would fail the NIS gate BECAUSE OF ITS
        POSITION: inflate the position covariance (at most ``max_inflation``) just enough
        that it passes. If the position part is fine (the mismatch is in velocity), or the
        capped inflation still does not admit the fix, the covariance is left UNCHANGED
        and the fix takes its normal course (counted either way).
        """
        if not (confirmed and self.cfg.inflate_to_admit):
            return
        nu, H, Rm = ekf.gnss_innovation(fix)
        P0 = ekf.P

        def nis(P, rows=slice(None)):
            h, r, n = H[rows], Rm[rows, rows], nu[rows]
            return float(n @ np.linalg.solve(h @ P @ h.T + r, n))

        if nis(P0) <= CHI2_999[nu.size]:
            return
        pos = slice(0, 3)
        if nis(P0, pos) <= CHI2_999[3]:
            self._count("recovery_inflation_not_position")
            return
        f = min(self.cfg.max_inflation, 1.05 * nis(P0, pos) / CHI2_999[3])
        while True:  # scale POSITION rows & columns by sqrt(f): P' = D P D, stays PD
            d = np.ones(P0.shape[0])
            d[P_] = math.sqrt(f)
            P = P0 * np.outer(d, d)
            if nis(P) <= CHI2_999[nu.size] or f >= self.cfg.max_inflation:
                break
            f = min(self.cfg.max_inflation, f * 2.0)
        if nis(P) > CHI2_999[nu.size]:
            self._count("recovery_inflation_refused")  # would not help: leave P alone
            return
        ekf.P = P
        ekf._check_P("recovery_inflation")
        self._count("recovery_cov_inflations")

    @property
    def confirming(self) -> bool:
        return self.phase == "CONFIRMING"


class OutputSmoother:
    """Continuous reported position: filter position + an offset that GLIDES to zero.

    Each correction jump D of the filter is absorbed into the offset (output unchanged at that
    instant); the offset then shrinks linearly at the glide speed
        max(max_rate_mps, |offset right after the jump| / max_blend_s),
    so a small correction is absorbed at max_rate (15 m/s), and ANY correction within
    max_blend_s (10 s). Per step the output moves by the filter's own propagated motion plus
    at most glide * dt: continuous, never a teleport, and never minutes of lag.
    """

    def __init__(self, max_rate_mps: float = 15.0, max_blend_s: float = 10.0) -> None:
        self.max_rate, self.max_blend_s = max_rate_mps, max_blend_s
        self.offset = np.zeros(3)  # E, N, U [m]
        self.glide_mps = max_rate_mps
        self._before: tuple[float, float, float] | None = None
        self.max_offset_m = 0.0
        self.max_glide_mps = max_rate_mps

    def mark(self, ekf: ErrorStateEKF) -> None:
        """Call after predict(), before any measurement update."""
        s = ekf.nominal
        self._before = (s.lat_rad, s.lon_rad, s.h_m)

    def absorb(self, ekf: ErrorStateEKF, dt: float) -> None:
        """Call after all updates of the step: absorb this step's correction, then glide."""
        if self._before is not None:
            s = ekf.nominal
            lat0, lon0, h0 = self._before
            jump = np.append(_en(lat0, lon0, s.lat_rad, s.lon_rad, h0), s.h_m - h0)
            self._before = None
            if np.any(jump):
                self.offset = self.offset - jump
                self.glide_mps = max(self.max_rate, float(np.linalg.norm(self.offset)) / self.max_blend_s)
                self.max_glide_mps = max(self.max_glide_mps, self.glide_mps)
        mag = float(np.linalg.norm(self.offset))
        self.max_offset_m = max(self.max_offset_m, mag)
        if mag > 0.0:
            red = min(mag, self.glide_mps * dt)
            self.offset *= (mag - red) / mag

    def output(self, ekf: ErrorStateEKF) -> tuple[float, float, float]:
        """(lat_rad, lon_rad, h_m) of the smoothed output."""
        s = ekf.nominal
        rm = float(meridian_radius(s.lat_rad)) + s.h_m
        rn = (float(prime_vertical_radius(s.lat_rad)) + s.h_m) * math.cos(s.lat_rad)
        return (s.lat_rad + self.offset[1] / rm, s.lon_rad + self.offset[0] / rn, s.h_m + self.offset[2])


__all__ = ["OutputSmoother", "RecoveryConfig", "RecoveryManager"]
