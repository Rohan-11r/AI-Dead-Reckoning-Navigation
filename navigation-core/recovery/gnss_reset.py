"""Recovery from GNSS gate lock-out.

Failure mode (observed on IO-VNBD in Phase 4): after one legitimately rejected fix the
INS keeps drifting, every later fix fails the NIS gate as well, and the filter never
recovers -- position error grows to kilometres while it keeps "rejecting outliers".

Policy: after ``n_consecutive`` rejected fixes, trust GNSS again -- re-initialise position
(and horizontal velocity when the fix carries a usable one) from the fix, with covariance
from the fix accuracy, and inflate the attitude/bias covariance so the filter can
re-converge instead of fighting its own stale state. Every reset is counted and logged;
it is a recovery event, never silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from navcore.filtering.ekf import A_, P_, V_, ErrorStateEKF, UpdateRecord
from navcore.ins.mechanization import NavState
from navcore.sensors.samples import GnssSample

MAX_ATTITUDE_SIGMA_RAD = 0.5


def reset_to_fix(ekf: ErrorStateEKF, fix: GnssSample, vertical_sigma_factor: float = 2.5,
                 velocity_sigma_mps: float = 1.0, attitude_inflation: float = 4.0) -> None:
    """Re-initialise the nominal position (+ horizontal velocity) at ``fix``.

    The fix describes its epoch; the displacement accumulated since then is carried over
    from the current nominal state (position at epoch -> fix, same drift-free offset
    thereafter), so a delayed fix does not teleport the state backwards in time.
    """
    s = ekf.nominal
    ref = ekf._state_at(fix.t_s)
    lat = fix.lat_rad + (s.lat_rad - ref.lat_rad)
    lon = fix.lon_rad + (s.lon_rad - ref.lon_rad)
    h = fix.h_m + (s.h_m - ref.h_m)
    v = s.v_enu_mps.copy()
    v_en = fix.velocity_enu()
    if v_en is not None:
        v[:2] = v_en
    ekf.nominal = NavState(t_s=s.t_s, lat_rad=lat, lon_rad=lon, h_m=h, v_enu_mps=v, q_nb=s.q_nb)
    P = ekf.P.copy()
    P[P_, :] = 0.0
    P[:, P_] = 0.0
    P[V_, :] = 0.0
    P[:, V_] = 0.0
    sig = fix.horizontal_accuracy_m
    P[P_, P_] = np.diag([sig ** 2, sig ** 2, (vertical_sigma_factor * sig) ** 2])
    vv = velocity_sigma_mps ** 2 if v_en is not None else 25.0
    P[V_, V_] = np.diag([vv, vv, 1.0])
    # inflate attitude uncertainty, capped at MAX_ATTITUDE_SIGMA_RAD per axis (repeated
    # resets must not grow it without bound); correlations are preserved
    std = np.sqrt(np.diag(P[A_, A_]))
    new_std = np.minimum(std * np.sqrt(attitude_inflation), MAX_ATTITUDE_SIGMA_RAD)
    P[A_, A_] = P[A_, A_] * np.outer(new_std / std, new_std / std)
    ekf.P = P
    ekf._check_P("reset_to_fix")
    ekf._hist.clear()
    ekf._hist.append(ekf.nominal)
    ekf.log.append(UpdateRecord("reset", fix.t_s, True, float("nan"), 0))


@dataclass
class ConsecutiveRejectionReset:
    """Apply ``reset_to_fix`` after ``n_consecutive`` rejected GNSS updates."""

    n_consecutive: int = 3
    streak: int = 0
    n_resets: int = 0
    reset_times_s: list[float] = field(default_factory=list)

    def after_update(self, ekf: ErrorStateEKF, fix: GnssSample, accepted: bool) -> bool:
        """Call after every GNSS update attempt. Returns True if a reset happened."""
        if accepted:
            self.streak = 0
            return False
        self.streak += 1
        if self.streak < self.n_consecutive:
            return False
        reset_to_fix(ekf, fix)
        self.streak = 0
        self.n_resets += 1
        self.reset_times_s.append(fix.t_s)
        return True


__all__ = ["ConsecutiveRejectionReset", "reset_to_fix"]
