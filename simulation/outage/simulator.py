"""GNSS outage simulator over RECORDED fixes (benchmarking and demonstration).

The input is the real fix sequence of a recording; the output is a NEW list (the input is
never mutated) in which fixes are withheld and, for some kinds, perturbed. Perturbations
are SYNTHETIC (AGENTS.md 2.2): every one is logged in ``SimulationLog`` with its reason, and
nothing produced here is ever written back to a dataset.

Event kinds (``OutageEvent.kind``), all over [start_s, end_s) of the fix EPOCH time:
  full          every fix withheld (a plain dropout: 10 s, 30 s, 60 s, 120 s, 300 s ...)
  tunnel        full loss inside; on the approach (``ramp_s`` before start) accuracy is
                reported worse and the position gets matching noise; on exit, the first
                ``exit_outliers`` fixes are MULTIPATH OUTLIERS -- displaced by
                ``outlier_m`` in a random direction while reporting a GOOD accuracy (the
                case that fools a naive filter) -- then accuracy ramps back over ``ramp_s``
  intermittent  alternating OFF (uniform ``off_s``) / ON (uniform ``on_s``) periods;
                fixes that get through are degraded x ``degrade_factor``
  degraded      fixes kept, reported accuracy x ``degrade_factor`` and position noise of
                that size (urban canyon)
Randomness comes from one seeded generator: the same (fixes, schedule, seed) always gives
the same output.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

import numpy as np

from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius
from navcore.sensors.samples import GnssSample

KINDS = ("full", "tunnel", "intermittent", "degraded")


@dataclass(frozen=True)
class OutageEvent:
    start_s: float
    end_s: float
    kind: str = "full"
    ramp_s: float = 6.0
    exit_outliers: int = 1
    outlier_m: float = 150.0
    off_s: tuple[float, float] = (5.0, 15.0)
    on_s: tuple[float, float] = (1.0, 4.0)
    degrade_factor: float = 3.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown outage kind {self.kind!r}")
        if not self.end_s > self.start_s:
            raise ValueError("outage must have end_s > start_s")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def covers(self, t: float) -> bool:
        return self.start_s <= t < self.end_s


@dataclass
class SimulationLog:
    withheld: list[tuple[float, str]] = field(default_factory=list)   # (fix epoch, event label)
    perturbed: list[tuple[float, str, str]] = field(default_factory=list)  # (epoch, label, what)


@dataclass(frozen=True)
class OutageSchedule:
    events: tuple[OutageEvent, ...]

    def __post_init__(self) -> None:
        ev = sorted(self.events, key=lambda e: e.start_s)
        for a, b in itertools.pairwise(ev):
            if b.start_s < a.end_s + (a.ramp_s if a.kind == "tunnel" else 0.0):
                raise ValueError(f"overlapping outages {a.label or a.start_s} / {b.label or b.start_s}")
        object.__setattr__(self, "events", tuple(ev))

    def in_outage(self, t: float) -> OutageEvent | None:
        return next((e for e in self.events if e.covers(t)), None)

    # ---- schedule builders ----------------------------------------------------------------
    @classmethod
    def none(cls) -> OutageSchedule:
        return cls(())

    @classmethod
    def periodic(cls, first_start_s: float, every_s: float, durations_s, t_end_s: float,
                 kind: str = "full") -> OutageSchedule:
        """Outages every ``every_s`` from ``first_start_s``, cycling through ``durations_s``,
        while the whole outage fits before ``t_end_s`` (the Phase 4/7 protocol:
        periodic(t_init + 150, 240, (30, 60), t_end))."""
        ev, s, k = [], first_start_s, 0
        durs = list(durations_s)
        while s + max(durs) < t_end_s:
            d = durs[k % len(durs)]
            ev.append(OutageEvent(s, s + d, kind, label=f"{kind}_{d:g}s#{k}"))
            s += every_s
            k += 1
        return cls(tuple(ev))

    @classmethod
    def benchmark(cls, first_start_s: float, t_end_s: float, gap_s: float = 150.0,
                  profile=(("full", 10), ("full", 30), ("full", 60), ("full", 120), ("full", 300),
                           ("tunnel", 45), ("intermittent", 120), ("degraded", 60))) -> OutageSchedule:
        """The Phase 9 battery: each (kind, duration) in turn, ``gap_s`` of normal GNSS between
        events (time to re-converge), cycling until the recording ends."""
        ev, s, k = [], first_start_s, 0
        while True:
            kind, d = profile[k % len(profile)]
            if s + d + gap_s > t_end_s:
                # skip a profile entry too long for the remaining time; stop when none fits
                rest = [(kk, dd) for kk, dd in profile if s + dd + gap_s <= t_end_s]
                if not rest:
                    break
                kind, d = min(rest, key=lambda x: x[1])
            ev.append(OutageEvent(s, s + d, kind, label=f"{kind}_{d:g}s#{k}"))
            s += d + gap_s
            k += 1
        return cls(tuple(ev))


def _displace(fix: GnssSample, de: float, dn: float) -> GnssSample:
    rm = float(meridian_radius(fix.lat_rad)) + fix.h_m
    rn = (float(prime_vertical_radius(fix.lat_rad)) + fix.h_m) * math.cos(fix.lat_rad)
    return replace(fix, lat_rad=fix.lat_rad + dn / rm, lon_rad=fix.lon_rad + de / rn)


def apply(fixes: list[GnssSample], schedule: OutageSchedule, seed: int = 0
          ) -> tuple[list[GnssSample], SimulationLog]:
    """Return (new fix list, log). ``fixes`` must be in epoch order; it is not modified."""
    rng = np.random.default_rng(seed)
    log = SimulationLog()
    out: list[GnssSample] = []
    # intermittent ON windows, drawn once per event in schedule order (deterministic)
    on_windows: dict[int, list[tuple[float, float]]] = {}
    for k, e in enumerate(schedule.events):
        if e.kind == "intermittent":
            w, t = [], e.start_s
            while t < e.end_s:
                t += rng.uniform(*e.off_s)
                on = rng.uniform(*e.on_s)
                if t < e.end_s:
                    w.append((t, min(t + on, e.end_s)))
                t += on
            on_windows[k] = w
    exits_left = {k: e.exit_outliers for k, e in enumerate(schedule.events) if e.kind == "tunnel"}
    for f in fixes:
        t = f.t_s
        g = f
        for k, e in enumerate(schedule.events):
            lab = e.label or f"{e.kind}@{e.start_s:g}"
            if e.covers(t):
                if e.kind == "full" or e.kind == "tunnel" or (
                        e.kind == "intermittent" and not any(a <= t < b for a, b in on_windows[k])):
                    log.withheld.append((t, lab))
                    g = None
                else:  # intermittent-ON or degraded: degrade
                    sig = f.horizontal_accuracy_m * (e.degrade_factor - 1.0)
                    g = _displace(f, *rng.normal(0.0, sig, 2))
                    g = replace(g, horizontal_accuracy_m=f.horizontal_accuracy_m * e.degrade_factor)
                    log.perturbed.append((t, lab, f"degraded x{e.degrade_factor:g}"))
                break
            if e.kind == "tunnel" and e.start_s - e.ramp_s <= t < e.start_s:
                x = (t - (e.start_s - e.ramp_s)) / e.ramp_s  # 0 -> 1 on the approach
                fac = 1.0 + 3.0 * x
                g = _displace(f, *rng.normal(0.0, f.horizontal_accuracy_m * (fac - 1.0), 2))
                g = replace(g, horizontal_accuracy_m=f.horizontal_accuracy_m * fac)
                log.perturbed.append((t, lab, f"tunnel approach x{fac:.2f}"))
                break
            if e.kind == "tunnel" and e.end_s <= t < e.end_s + e.ramp_s + 30.0:
                if exits_left[k] > 0:
                    exits_left[k] -= 1
                    ang = rng.uniform(0.0, 2 * math.pi)
                    g = _displace(f, e.outlier_m * math.cos(ang), e.outlier_m * math.sin(ang))
                    log.perturbed.append((t, lab, f"multipath outlier {e.outlier_m:g} m, accuracy kept"))
                elif t < e.end_s + e.ramp_s:
                    x = (t - e.end_s) / e.ramp_s  # 0 -> 1 leaving the portal
                    fac = 4.0 - 3.0 * x
                    g = _displace(f, *rng.normal(0.0, f.horizontal_accuracy_m * (fac - 1.0), 2))
                    g = replace(g, horizontal_accuracy_m=f.horizontal_accuracy_m * fac)
                    log.perturbed.append((t, lab, f"tunnel exit x{fac:.2f}"))
                break
        if g is not None:
            out.append(g)
    return out, log


__all__ = ["KINDS", "OutageEvent", "OutageSchedule", "SimulationLog", "apply"]
