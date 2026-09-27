"""Deterministic replay of a recorded IO-VNBD drive through the navigation engine.

``load_segments`` turns a synced ``.parquet`` recording into ``ReplaySegment`` objects: phone
IMU arrays, the phone's GNSS fixes (tagged with their epoch and RECEIPT time), and --
SEPARATELY -- the VBOX ground truth, which the replay never hands to the engine (it is for
scoring only; the CAN/VBOX rule is checked mechanically on the input column list).

``ReplayEngine`` then streams a segment sample by sample, exactly as a phone would deliver
it: at each IMU sample the engine receives only the fixes whose receipt time has passed
since the previous sample -- never anything from the future -- optionally after an
``OutageSchedule`` has withheld or perturbed fixes. Streaming is as fast as possible by
default; ``realtime_factor`` paces it against a clock (1.0 = live), with injectable
``clock``/``sleep`` so tests stay fast and deterministic. The run is deterministic: the
same inputs give bit-identical outputs, summarised by ``ReplayResult.digest``.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from navcore.sensors.samples import GnssSample
from simulation.outage.simulator import OutageSchedule, SimulationLog, apply
from training.preprocessing.columns import assert_model_inputs

PHONE_INPUTS = ["acc_x_mps2", "acc_y_mps2", "acc_z_mps2", "grav_x_mps2", "grav_y_mps2", "grav_z_mps2",
                "gyro_x_radps", "gyro_y_radps", "gyro_z_radps", "ph_gnss_lat_deg", "ph_gnss_lon_deg",
                "ph_gnss_alt_m", "ph_gnss_speed_mps", "ph_gnss_accuracy_m", "ph_gnss_bearing_deg",
                "ph_gnss_new_fix", "ph_gnss_epoch_utc"]
TRUTH = ["gt_lat_deg", "gt_lon_deg", "gt_heading_deg", "gt_speed_mps", "gt_valid"]


@dataclass
class ReplaySegment:
    session_id: str
    segment_id: int
    t: np.ndarray                   # [s] since the segment's first sample (phone clock, UTC-joined)
    acc: np.ndarray                 # (N, 3) phone accelerometer, logged axes [m/s^2]
    grav: np.ndarray                # (N, 3) phone gravity channel [m/s^2]
    gyro: np.ndarray                # (N, 3) phone gyroscope, logged columns [rad/s]
    fixes: list[GnssSample]         # phone GNSS, epoch + receipt time
    truth: pd.DataFrame = field(repr=False)  # VBOX, scoring only -- never given to the engine

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0])


def build_fixes(d: pd.DataFrame, t0) -> list[GnssSample]:
    """Phone fixes at their epochs (sample-and-hold rows collapsed to the new-fix rows)."""
    fx = d[d["ph_gnss_new_fix"] & d["ph_gnss_epoch_utc"].notna() & d["ph_gnss_accuracy_m"].gt(0)]
    out = []
    for r in fx.itertuples():
        ok = np.isfinite(r.ph_gnss_speed_mps) and np.isfinite(r.ph_gnss_bearing_deg) and r.ph_gnss_speed_mps >= 0
        out.append(GnssSample(
            t_s=(r.ph_gnss_epoch_utc - t0).total_seconds(), t_received_s=(r.t_utc - t0).total_seconds(),
            lat_rad=math.radians(r.ph_gnss_lat_deg), lon_rad=math.radians(r.ph_gnss_lon_deg),
            h_m=r.ph_gnss_alt_m, horizontal_accuracy_m=r.ph_gnss_accuracy_m,
            speed_mps=r.ph_gnss_speed_mps if ok else None,
            bearing_rad=math.radians(r.ph_gnss_bearing_deg) if ok else None))
    return out


def load_segments(path: str | Path, min_duration_s: float = 300.0, session_id: str | None = None
                  ) -> list[ReplaySegment]:
    """Every segment of a synced recording lasting at least ``min_duration_s``."""
    assert_model_inputs(PHONE_INPUTS)  # the CAN rule, mechanically: no vehicle channel is an input
    path = Path(path)
    full = pd.read_parquet(path)
    out = []
    for seg_id, d in full.groupby("segment_id"):
        d = d.reset_index(drop=True)
        t0 = d["t_utc"].iloc[0]
        t = (d["t_utc"] - t0).dt.total_seconds().to_numpy()
        if t[-1] - t[0] < min_duration_s:
            continue
        inp = d[[*PHONE_INPUTS, "t_utc"]]
        out.append(ReplaySegment(
            session_id=session_id or path.stem, segment_id=int(seg_id), t=t,
            acc=inp[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy(),
            grav=inp[["grav_x_mps2", "grav_y_mps2", "grav_z_mps2"]].to_numpy(),
            gyro=inp[["gyro_x_radps", "gyro_y_radps", "gyro_z_radps"]].to_numpy(),
            fixes=build_fixes(inp, t0), truth=d[TRUTH].copy()))
    return out


@dataclass
class ReplayResult:
    t: np.ndarray
    rows: np.ndarray                # sample index of each output
    lat: np.ndarray                 # OUTPUT position (continuous)
    lon: np.ndarray
    filter_lat: np.ndarray          # filter estimate
    filter_lon: np.ndarray
    sigma_h: np.ndarray
    v_en: np.ndarray
    mode: list[str]
    gnss_state: list[str]
    sim_log: SimulationLog
    fixes_delivered: int
    wall_s: float

    @property
    def digest(self) -> str:
        """SHA-256 over every numeric output: equal digests = bit-identical runs."""
        h = hashlib.sha256()
        for a in (self.t, self.rows, self.lat, self.lon, self.filter_lat, self.filter_lon, self.sigma_h, self.v_en):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update("|".join(self.mode + self.gnss_state).encode())
        return h.hexdigest()


class ReplayEngine:
    def __init__(self, segment: ReplaySegment, schedule: OutageSchedule | None = None, seed: int = 0,
                 realtime_factor: float | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.seg = segment
        self.schedule = schedule or OutageSchedule.none()
        self.fixes, self.sim_log = apply(segment.fixes, self.schedule, seed)
        self.realtime_factor, self.clock, self.sleep = realtime_factor, clock, sleep

    def stream(self) -> Iterator[tuple[int, float, np.ndarray, np.ndarray, np.ndarray, list[GnssSample]]]:
        """Yield (row, t, acc, grav, gyro, fixes received since the previous row), in order."""
        seg, fixes = self.seg, sorted(self.fixes, key=lambda f: (f.t_received_s, f.t_s))
        k, t_prev = 0, -math.inf
        wall0 = self.clock()
        for i in range(len(seg.t)):
            t = float(seg.t[i])
            if t < t_prev:
                raise ValueError(f"time goes backwards at row {i}")
            new = []
            while k < len(fixes) and fixes[k].t_received_s <= t + 1e-9:
                new.append(fixes[k])
                k += 1
            if self.realtime_factor:
                due = wall0 + (t - float(seg.t[0])) / self.realtime_factor
                lag = due - self.clock()
                if lag > 0:
                    self.sleep(lag)
            t_prev = t
            yield i, t, seg.acc[i], seg.grav[i], seg.gyro[i], new

    def run(self, engine, on_output: Callable | None = None) -> ReplayResult:
        """Stream the whole segment through ``engine.on_sample``; collect every output."""
        w0 = time.perf_counter()
        cols: dict[str, list] = {k: [] for k in ("t", "rows", "lat", "lon", "flat", "flon", "sig", "v", "mode", "gs")}
        n_fix = 0
        for i, t, acc, grav, gyro, fixes in self.stream():
            n_fix += len(fixes)
            out = engine.on_sample(t, acc, grav, gyro, fixes)
            if out is None:
                continue
            for k, v in (("t", t), ("rows", i), ("lat", out.lat_rad), ("lon", out.lon_rad), ("flat", out.filter_lat_rad),
                         ("flon", out.filter_lon_rad), ("sig", out.sigma_h_m), ("v", out.v_enu_mps[:2]),
                         ("mode", out.mode), ("gs", out.gnss_state)):
                cols[k].append(v)
            if on_output is not None:
                on_output(i, out)
        return ReplayResult(t=np.array(cols["t"]), rows=np.array(cols["rows"], dtype=np.int64), lat=np.array(cols["lat"]),
                            lon=np.array(cols["lon"]), filter_lat=np.array(cols["flat"]), filter_lon=np.array(cols["flon"]),
                            sigma_h=np.array(cols["sig"]), v_en=np.array(cols["v"]).reshape(-1, 2), mode=cols["mode"],
                            gnss_state=cols["gs"], sim_log=self.sim_log, fixes_delivered=n_fix,
                            wall_s=time.perf_counter() - w0)


__all__ = ["PHONE_INPUTS", "TRUTH", "ReplayEngine", "ReplayResult", "ReplaySegment", "build_fixes", "load_segments"]
