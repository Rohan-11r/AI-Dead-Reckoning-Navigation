"""GNSS outage simulator. Input fixes are SYNTHETIC (1 Hz, straight line; AGENTS.md 2.2)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius
from navcore.sensors.samples import GnssSample
from simulation.outage.simulator import OutageEvent, OutageSchedule, apply

LAT, LON, H = math.radians(52.4), math.radians(-1.5), 100.0
RM = float(meridian_radius(LAT)) + H
RN = (float(prime_vertical_radius(LAT)) + H) * math.cos(LAT)


def fixes(n: int = 1200, acc: float = 4.0) -> list[GnssSample]:
    return [GnssSample(t_s=float(k), t_received_s=float(k), lat_rad=LAT + 15.0 * k / RM, lon_rad=LON, h_m=H,
                       horizontal_accuracy_m=acc, speed_mps=15.0, bearing_rad=0.0) for k in range(n)]


def offset_m(a: GnssSample, b: GnssSample) -> float:
    return math.hypot((a.lat_rad - b.lat_rad) * RM, (a.lon_rad - b.lon_rad) * RN)


@pytest.mark.parametrize("dur", [10, 30, 60, 120, 300])
def test_full_dropout_withholds_exactly_the_window(dur):
    src = fixes()
    out, log = apply(src, OutageSchedule((OutageEvent(100.0, 100.0 + dur, "full"),)))
    kept = {f.t_s for f in out}
    assert kept == {f.t_s for f in src if not 100.0 <= f.t_s < 100.0 + dur}
    assert len(log.withheld) == dur and not log.perturbed
    assert all(f is g for f, g in zip(out, [f for f in src if f.t_s in kept], strict=True))  # untouched


def test_input_is_never_mutated_and_runs_are_deterministic():
    src = fixes(3000)
    snap = copy.deepcopy(src)
    sched = OutageSchedule.benchmark(60.0, 3000.0)  # long enough to reach the random kinds
    a, la = apply(src, sched, seed=3)
    b, lb = apply(src, sched, seed=3)
    c, _ = apply(src, sched, seed=4)
    assert src == snap
    assert a == b and la == lb
    assert a != c  # the seed matters where randomness is used


def test_benchmark_schedule_covers_the_battery_without_overlap():
    sched = OutageSchedule.benchmark(100.0, 3000.0, gap_s=150.0)
    kinds = [(e.kind, round(e.duration_s)) for e in sched.events]
    assert kinds[:8] == [("full", 10), ("full", 30), ("full", 60), ("full", 120), ("full", 300),
                         ("tunnel", 45), ("intermittent", 120), ("degraded", 60)]
    for a, b in zip(sched.events[:-1], sched.events[1:], strict=True):
        assert b.start_s - a.end_s == pytest.approx(150.0)
    assert sched.events[-1].end_s + 150.0 <= 3000.0
    with pytest.raises(ValueError):
        OutageSchedule((OutageEvent(0, 50), OutageEvent(40, 80)))


def test_periodic_schedule_reproduces_the_phase7_protocol():
    s = OutageSchedule.periodic(150.0, 240.0, (30.0, 60.0), 1000.0)
    # a start is used while start + the LONGEST duration fits before the end (Phase 4/7 rule)
    assert [(e.start_s, e.duration_s) for e in s.events] == [(150.0, 30.0), (390.0, 60.0), (630.0, 30.0), (870.0, 60.0)]


def test_tunnel_degrades_the_approach_and_exits_with_a_confident_multipath_outlier():
    src = fixes()
    ev = OutageEvent(300.0, 345.0, "tunnel", ramp_s=6.0, exit_outliers=1, outlier_m=150.0)
    out, log = apply(src, OutageSchedule((ev,)), seed=1)
    by_t = {f.t_s: f for f in out}
    approach = [by_t[t].horizontal_accuracy_m for t in range(294, 300)]
    assert approach == sorted(approach) and approach[0] >= 4.0 and approach[-1] > 12.0
    assert not any(300 <= t < 345 for t in by_t)
    first = by_t[345.0]
    assert first.horizontal_accuracy_m == 4.0  # reports a GOOD accuracy ...
    assert offset_m(first, src[345]) == pytest.approx(150.0, rel=1e-3)  # ... while 150 m off
    after = [by_t[t].horizontal_accuracy_m for t in range(346, 351)]
    assert after == sorted(after, reverse=True) and by_t[360.0] is src[360]
    assert any("multipath" in w for _, _, w in log.perturbed)


def test_intermittent_lets_some_degraded_fixes_through():
    src = fixes()
    out, log = apply(src, OutageSchedule((OutageEvent(200.0, 320.0, "intermittent", degrade_factor=3.0),)), seed=2)
    inside = [f for f in out if 200.0 <= f.t_s < 320.0]
    assert 0 < len(inside) < 60
    assert all(f.horizontal_accuracy_m == pytest.approx(12.0) for f in inside)
    assert len(log.withheld) + len(inside) == 120


def test_degraded_keeps_every_fix_with_worse_accuracy_and_matching_noise():
    src = fixes()
    out, _ = apply(src, OutageSchedule((OutageEvent(100.0, 700.0, "degraded", degrade_factor=3.0),)), seed=5)
    inside = [(f, src[int(f.t_s)]) for f in out if 100.0 <= f.t_s < 700.0]
    assert len(inside) == 600
    d = np.array([offset_m(a, b) for a, b in inside])
    # per-axis noise sigma = 4 m x (3 - 1) = 8 m -> Rayleigh mean 8 * sqrt(pi/2) = 10.0 m
    assert np.mean(d) == pytest.approx(8.0 * math.sqrt(math.pi / 2), rel=0.1)
