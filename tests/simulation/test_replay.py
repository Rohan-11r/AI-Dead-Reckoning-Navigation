"""Deterministic replay engine. Recordings here are SYNTHETIC (simulation/synthetic/drive.py)
or a tiny synthetic parquet in the IO-VNBD synced layout (AGENTS.md 2.2)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from navcore.filtering.ekf import NoiseParams
from navcore.fusion.engine import EngineConfig, NavigationEngine
from navcore.fusion.navigator import FusionConfig
from navcore.sensors.samples import GnssSample
from simulation.outage.simulator import OutageEvent, OutageSchedule
from simulation.replay.replay import ReplayEngine, ReplaySegment, load_segments
from simulation.synthetic.drive import synthetic_drive

NOISE = NoiseParams(0.05, 0.005, 0.02, 1e4, 1e-3, 1e4)
RX_DELAY = 0.25  # fixes arrive 0.25 s after their epoch


def segment(t_end: float = 260.0, dropouts=()) -> ReplaySegment:
    steps = list(synthetic_drive(seed=11, t_end_s=t_end, dropouts=dropouts))
    fixes = [replace(f, t_received_s=f.t_s + RX_DELAY) for s in steps for f in s.fixes]
    truth = pd.DataFrame({"gt_lat_deg": [math.degrees(s.truth.lat_rad) for s in steps],
                          "gt_lon_deg": [math.degrees(s.truth.lon_rad) for s in steps]})
    return ReplaySegment("SYN", 0, np.array([s.t for s in steps]), np.array([s.acc for s in steps]),
                         np.array([s.grav for s in steps]), np.array([s.gyro_raw for s in steps]), fixes, truth)


class Recorder:
    """Stub engine: records exactly what it is given."""

    def __init__(self) -> None:
        self.seen: list[tuple[float, list]] = []

    def on_sample(self, t, acc, grav, gyro, fixes):
        self.seen.append((t, list(fixes)))


def test_stream_is_causal_and_delivers_every_fix_once_at_receipt():
    seg = segment(60.0)
    rec = Recorder()
    res = ReplayEngine(seg).run(rec)
    delivered = [(t, f) for t, fs in rec.seen for f in fs]
    assert [f for _, f in delivered] == sorted(seg.fixes, key=lambda f: f.t_received_s)  # each once, in order
    for t, f in delivered:
        assert f.t_received_s <= t + 1e-9  # never from the future ...
        assert t - f.t_received_s < 0.1 + 1e-9  # ... and at the first sample after receipt
    assert res.fixes_delivered == len(seg.fixes)


def test_simulated_outage_fixes_never_reach_the_engine():
    seg = segment(120.0)
    rec = Recorder()
    sched = OutageSchedule((OutageEvent(30.0, 90.0, "full"),))
    res = ReplayEngine(seg, sched).run(rec)
    got = [f.t_s for _, fs in rec.seen for f in fs]
    assert got and not any(30.0 <= t < 90.0 for t in got)
    assert len(res.sim_log.withheld) == 60


@pytest.mark.slow
def test_replay_through_the_engine_is_bit_identical_and_follows_the_lifecycle():
    seg = segment()
    cfg = EngineConfig(fusion=FusionConfig(use_ai_speed=False, use_nhc=False), mh_window_s=60.0)
    sched = OutageSchedule((OutageEvent(100.0, 160.0, "tunnel"),))
    r1 = ReplayEngine(seg, sched, seed=1).run(NavigationEngine(NOISE, cfg))
    r2 = ReplayEngine(seg, sched, seed=1).run(NavigationEngine(NOISE, cfg))
    assert r1.digest == r2.digest  # deterministic, bit for bit
    r3 = ReplayEngine(seg, OutageSchedule.none()).run(NavigationEngine(NOISE, cfg))
    assert r3.digest != r1.digest
    # WAITING produces no output; the first output is at the first fix's receipt (15 m/s > 5)
    assert r1.t[0] == pytest.approx(1.0 + RX_DELAY, abs=0.1 + 1e-9)
    modes = r1.mode
    assert modes[0] == "ALIGNING" and modes[-1] == "NAVIGATING"
    assert r1.t[modes.index("NAVIGATING")] == pytest.approx(r1.t[0] + 60.0, abs=0.11)
    eng = NavigationEngine(NOISE, cfg)
    ReplayEngine(seg, sched, seed=1).run(eng)
    assert eng.chosen_offset_rad == pytest.approx(0.0)  # the right heading hypothesis wins
    lat, lon = np.radians(seg.truth["gt_lat_deg"].to_numpy()), np.radians(seg.truth["gt_lon_deg"].to_numpy())
    i = r1.rows[-1]
    e = math.hypot((r1.lat[-1] - lat[i]) * 6.37e6, (r1.lon[-1] - lon[i]) * 6.37e6 * math.cos(lat[i]))
    assert e < 10.0


def test_realtime_pacing_uses_the_injected_clock():
    seg = segment(20.0)
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s

    ReplayEngine(seg, realtime_factor=4.0, clock=lambda: now[0], sleep=sleep).run(Recorder())
    assert now[0] == pytest.approx((seg.t[-1] - seg.t[0]) / 4.0, abs=1e-9)  # 4x faster than live
    assert len(slept) == len(seg.t) - 1


def test_time_going_backwards_is_refused():
    seg = segment(5.0)
    seg.t[10] = seg.t[8]
    with pytest.raises(ValueError, match="backwards"):
        list(ReplayEngine(seg).stream())


def test_load_segments_exposes_phone_channels_only(tmp_path):
    n = 3500  # 350 s at 10 Hz
    t_utc = pd.Timestamp("2020-01-01T10:00:00Z") + pd.to_timedelta(np.arange(n) * 0.1, unit="s")
    rng = np.random.default_rng(0)
    new_fix = np.zeros(n, bool)
    new_fix[::10] = True
    df = pd.DataFrame({
        "t_utc": t_utc, "segment_id": 0,
        **{c: rng.normal(0, 1, n) for c in ("acc_x_mps2", "acc_y_mps2", "grav_x_mps2", "grav_y_mps2",
                                             "gyro_x_radps", "gyro_y_radps", "gyro_z_radps")},
        "acc_z_mps2": 9.8 + rng.normal(0, 0.1, n), "grav_z_mps2": np.full(n, 9.8),
        "ph_gnss_lat_deg": 52.4, "ph_gnss_lon_deg": -1.5, "ph_gnss_alt_m": 100.0, "ph_gnss_speed_mps": 10.0,
        "ph_gnss_accuracy_m": 4.0, "ph_gnss_bearing_deg": 90.0, "ph_gnss_new_fix": new_fix,
        "ph_gnss_epoch_utc": t_utc,
        "gt_lat_deg": 52.4, "gt_lon_deg": -1.5, "gt_heading_deg": 90.0, "gt_speed_mps": 10.0, "gt_valid": True,
        "can_wheel_speed_fl_kmh": 36.0, "can_yaw_rate_dps": 0.0})  # CAN: must never be exposed
    p = tmp_path / "SYN.parquet"
    df.to_parquet(p)
    (seg,) = load_segments(p)
    assert seg.session_id == "SYN" and len(seg.t) == n and len(seg.fixes) == n // 10
    assert list(seg.truth.columns) == ["gt_lat_deg", "gt_lon_deg", "gt_heading_deg", "gt_speed_mps", "gt_valid"]
    arrays = np.concatenate([seg.acc.ravel(), seg.grav.ravel(), seg.gyro.ravel()])
    assert not np.any(arrays == 36.0)  # the wheel-speed column went nowhere near the inputs
    assert load_segments(p, min_duration_s=400.0) == []


def test_engine_starting_on_the_first_sample_levels_from_the_gravity_channel():
    """A real drive can be moving at its first fix, received within the first second (found
    on IO-VNBD by the Phase 9 benchmark): too little accelerometer data to level from."""
    st = next(iter(synthetic_drive(seed=3, t_end_s=1.0)))
    fix = GnssSample(t_s=0.0, t_received_s=0.0, lat_rad=st.truth.lat_rad, lon_rad=st.truth.lon_rad,
                     h_m=st.truth.h_m, horizontal_accuracy_m=3.0, speed_mps=15.0, bearing_rad=math.radians(73.0))
    eng = NavigationEngine(NOISE, EngineConfig(fusion=FusionConfig(use_ai_speed=False, use_nhc=False)))
    out = eng.on_sample(0.0, st.acc, st.grav, st.gyro_raw, [fix])
    assert out is not None and out.mode == "ALIGNING" and eng.tilt_source == "gravity_channel"
    assert eng.mount.tilt_rad < math.radians(1.0)  # the synthetic phone is level
