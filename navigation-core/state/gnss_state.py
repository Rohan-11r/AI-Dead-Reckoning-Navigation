"""GNSS quality state machine: GOOD / DEGRADED / LOST / RECOVERING.

The fused filter's behaviour depends on how trustworthy GNSS currently is. The state is a
PURE function of observable GNSS facts, so every transition is unit-testable:

* fix age        -- time since the last fix was received (IO-VNBD phones: ~9 s cadence)
* accuracy       -- the fix's reported horizontal accuracy
* gate outcome   -- whether the fix passed the filter's NIS gate

Transitions (thresholds in ``GnssStateConfig``; defaults sized for the 9 s phone cadence):

    GOOD       -> DEGRADED    accuracy > degraded_accuracy_m, OR a fix rejected by the gate,
                              OR fix age > degraded_age_s (a missed fix)
    GOOD/DEG.  -> LOST        fix age > lost_age_s
    DEGRADED   -> GOOD        good_streak consecutive accepted, accurate fixes
    LOST       -> RECOVERING  first fix after LOST
    RECOVERING -> GOOD        recover_streak consecutive accepted, accurate fixes
    RECOVERING -> LOST        fix age > lost_age_s again
    RECOVERING -> DEGRADED    an inaccurate or rejected fix

Filter policy per state (``policy``): whether GNSS is used and with what noise inflation,
whether the AI speed measurement is applied, and whether NHC applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GnssState(Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class GnssStateConfig:
    degraded_accuracy_m: float = 10.0
    degraded_age_s: float = 12.0   # > one missed 9 s fix
    lost_age_s: float = 25.0       # ~ two missed fixes
    good_streak: int = 2
    recover_streak: int = 3


@dataclass(frozen=True)
class FilterPolicy:
    use_gnss: bool
    gnss_noise_inflation: float  # multiplies the reported accuracy
    use_ai_speed: bool
    use_nhc: bool


POLICY = {
    GnssState.GOOD: FilterPolicy(use_gnss=True, gnss_noise_inflation=1.0, use_ai_speed=False, use_nhc=True),
    GnssState.DEGRADED: FilterPolicy(use_gnss=True, gnss_noise_inflation=2.0, use_ai_speed=True, use_nhc=True),
    GnssState.LOST: FilterPolicy(use_gnss=False, gnss_noise_inflation=1.0, use_ai_speed=True, use_nhc=True),
    GnssState.RECOVERING: FilterPolicy(use_gnss=True, gnss_noise_inflation=3.0, use_ai_speed=True, use_nhc=True),
}


@dataclass
class GnssStateMachine:
    cfg: GnssStateConfig = field(default_factory=GnssStateConfig)
    state: GnssState = GnssState.LOST  # nothing is trusted before the first fix
    last_fix_t: float | None = None
    streak: int = 0
    transitions: list[tuple[float, str, str, str]] = field(default_factory=list)

    def _go(self, t: float, new: GnssState, why: str) -> None:
        if new is not self.state:
            self.transitions.append((t, self.state.value, new.value, why))
            self.state = new
            self.streak = 0

    def on_time(self, t: float) -> GnssState:
        """Advance with no fix: age-based downgrades only."""
        age = float("inf") if self.last_fix_t is None else t - self.last_fix_t
        if self.state is not GnssState.LOST and age > self.cfg.lost_age_s:
            self._go(t, GnssState.LOST, f"no fix for {age:.1f} s")
        elif self.state is GnssState.GOOD and age > self.cfg.degraded_age_s:
            self._go(t, GnssState.DEGRADED, f"fix age {age:.1f} s")
        return self.state

    def on_fix(self, t: float, accuracy_m: float, accepted: bool) -> GnssState:
        """Update with a received fix and the filter's gate outcome for it."""
        self.last_fix_t = t
        good = accepted and accuracy_m <= self.cfg.degraded_accuracy_m
        why = "accurate fix accepted" if good else (
            "fix rejected by gate" if not accepted else f"accuracy {accuracy_m:.1f} m")
        s = self.state
        if s is GnssState.LOST:
            self._go(t, GnssState.RECOVERING, "first fix after loss")
            self.streak = 1 if good else 0
        elif s is GnssState.GOOD:
            if not good:
                self._go(t, GnssState.DEGRADED, why)
        elif s is GnssState.DEGRADED:
            self.streak = self.streak + 1 if good else 0
            if self.streak >= self.cfg.good_streak:
                self._go(t, GnssState.GOOD, f"{self.cfg.good_streak} good fixes")
        elif s is GnssState.RECOVERING:
            if not good:
                self._go(t, GnssState.DEGRADED, why)
            else:
                self.streak += 1
                if self.streak >= self.cfg.recover_streak:
                    self._go(t, GnssState.GOOD, f"{self.cfg.recover_streak} good fixes after loss")
        return self.state

    @property
    def policy(self) -> FilterPolicy:
        return POLICY[self.state]
