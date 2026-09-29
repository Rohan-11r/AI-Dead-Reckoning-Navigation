#!/usr/bin/env python3
"""Edge-engine live demo: a recorded IO-VNBD drive streamed through navcore in real time.

    .venv/Scripts/python.exe scripts/demo_edge_live.py                     # S3a, 200 samples/s
    .venv/Scripts/python.exe scripts/demo_edge_live.py --session Vta2 --outage-s 120
    .venv/Scripts/python.exe scripts/demo_edge_live.py --no-plot --max-samples 3000

A pure-Python, laptop-side simulation of the phone/edge deployment. It does NOT touch the
Android app and does NOT modify navcore: it only drives the existing, unmodified
``navcore.fusion.engine.NavigationEngine`` (shipped defaults: AI speed + NHC, Model B off,
causal mount) one sample at a time, exactly as scripts/evaluate/phase9_outage_benchmark.py
does, and paces the stream against a wall clock.

HONEST RATE LABELLING (AGENTS.md 2.2). IO-VNBD phone data is logged at 10 Hz (median
effective rate 10.0 Hz, reports/phase2/dataset_report.json). There is no 200 Hz IO-VNBD
data and this script does not invent any (no upsampling, no interpolation). Instead the
REAL 10 Hz samples are delivered to the engine at ``--stream-hz`` samples per wall-clock
second (default 200, one sample every 1/200 s), i.e. a 20x time-compressed replay. That
demonstrates what matters for an edge claim -- the engine keeps up with a 200 samples/s
input stream -- and the readout states both rates so nobody can mistake it for a 200 Hz
sensor. Use ``--stream-hz 10`` for true 1x live pace.

GNSS OUTAGE. So that the plot shows dead reckoning rather than GNSS tracking, one full GNSS
dropout (``--outage-s``, default 60 s) is injected ``--outage-after-s`` after initialisation
with the Phase 9 simulator (simulation.outage): fixes inside it are WITHHELD from the engine.
No GNSS field reaches the engine during the outage (enforced by the simulator, not here).

TRUTH. The VBOX reference (``gt_*``) is loaded separately by simulation.replay and is NEVER
passed to the engine; this script reads it only to draw the reference track and to score
error / drift. Drift % = horizontal error vs VBOX / VBOX distance travelled since the outage
began (shown only during and after the outage, once >= 50 m has been travelled).

LATENCY is the wall time of ``engine.on_sample`` alone (time.perf_counter), per sample. While
ALIGNING the engine runs 8 heading hypotheses, so latency is higher for the first 120 s after
initialisation; the readout shows the mode so this is visible, not hidden.

Default session S3a is a VAL drive (not seen by Model A in training) with a verified phone
clock (reports/phase3). Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from navcore.filtering.ekf import NoiseParams  # noqa: E402
from navcore.fusion.ai_models import OnnxSequenceModel  # noqa: E402
from navcore.fusion.engine import EngineConfig, NavigationEngine  # noqa: E402
from navcore.fusion.navigator import FusionConfig  # noqa: E402
from navcore.geometry.geodesy import LocalTangentPlane  # noqa: E402
from simulation.outage.simulator import OutageEvent, OutageSchedule  # noqa: E402
from simulation.replay.replay import ReplayEngine, load_segments  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
DRIFT_MIN_DIST_M = 50.0     # below this, a percentage of distance is noise, not drift

# dataviz reference palette (light): truth = neutral reference, engine = categorical 1 / 2
C_TRUTH, C_AIDED, C_OUTAGE, C_FIX = "#9a9994", "#2a78d6", "#eb6834", "#52514e"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--session", default="S3a", help="IO-VNBD session id (default: S3a, a VAL drive)")
    p.add_argument("--segment", type=int, default=None, help="segment id (default: first >= 5 min)")
    p.add_argument("--stream-hz", type=float, default=200.0, help="samples delivered per wall-clock second")
    p.add_argument("--outage-after-s", type=float, default=150.0, help="outage start, seconds after init")
    p.add_argument("--outage-s", type=float, default=60.0, help="GNSS outage length [s]; 0 = none")
    p.add_argument("--plot-every", type=int, default=20, help="redraw the plot every N samples")
    p.add_argument("--print-every", type=int, default=10, help="refresh the terminal readout every N samples")
    p.add_argument("--max-samples", type=int, default=None, help="stop after N samples")
    p.add_argument("--view-m", type=float, default=1500.0,
                   help="plot half-width following the vehicle [m]; 0 = whole drive")
    p.add_argument("--no-plot", action="store_true", help="terminal readout only")
    return p.parse_args()


def load_engine() -> NavigationEngine:
    """The shipped engine configuration, built exactly as the Phase 9 benchmark builds it."""
    sel = json.loads((REPO_ROOT / "reports" / "phase6" / "selection.json").read_text(encoding="utf-8"))
    model = REPO_ROOT / "models" / "exported" / f"{sel['model_a']['selected']}.onnx"
    if not model.exists():
        raise FileNotFoundError(f"selected speed model missing: {model}")
    noise = NoiseParams.from_report(json.loads((CFG.reports_dir / "phase4" / "imu_noise.json")
                                               .read_text(encoding="utf-8"))["filter_parameters"])
    return NavigationEngine(noise, EngineConfig(fusion=FusionConfig()), {"model_speed": OnnxSequenceModel(model)})


def pick_segment(session: str, segment: int | None):
    path = CFG.synced_root / f"{session}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no synced recording for session {session!r}: {path}")
    p3 = json.loads((CFG.reports_dir / "phase3" / "preprocessing_report.json").read_text(encoding="utf-8"))
    if not any(s["session_id"] == session and s["alignment_verified"] for s in p3["sessions"]):
        print(f"WARNING: {session} phone clock is NOT alignment-verified; VBOX error/drift will be "
              "off by ~15 m (reports/phase3). Trajectory is still genuine.", file=sys.stderr)
    segs = load_segments(path)
    if not segs:
        raise RuntimeError(f"{session}: no segment >= 5 min")
    if segment is None:
        return segs[0]
    for s in segs:
        if s.segment_id == segment:
            return s
    raise ValueError(f"{session}: segment {segment} not found (have {[s.segment_id for s in segs]})")


def init_time(seg) -> float | None:
    """When the engine will initialise (same rule as NavigationEngine: first fix > 5 m/s with bearing)."""
    f0 = next((f for f in seg.fixes if f.speed_mps is not None and f.speed_mps > 5.0 and f.bearing_rad is not None), None)
    if f0 is None:
        return None
    i = int(np.searchsorted(seg.t, f0.t_received_s - 1e-9))
    return float(seg.t[i]) if i < len(seg.t) else None


def truth_track(seg, ltp: LocalTangentPlane) -> tuple[np.ndarray, np.ndarray]:
    """VBOX EN [m] per row (nan where invalid) and cumulative VBOX distance [m] (scoring only)."""
    tr = seg.truth
    valid = tr["gt_valid"].to_numpy().astype(bool)
    lat = np.radians(tr["gt_lat_deg"].to_numpy(dtype=np.float64))
    lon = np.radians(tr["gt_lon_deg"].to_numpy(dtype=np.float64))
    en = np.full((len(tr), 2), np.nan)
    en[valid] = ltp.geodetic_to_enu(lat[valid], lon[valid], np.zeros(valid.sum()))[:, :2]
    step = np.zeros(len(tr))
    idx = np.flatnonzero(valid)
    step[idx[1:]] = np.linalg.norm(np.diff(en[idx], axis=0), axis=1)
    return en, np.cumsum(step)


class LivePlot:
    def __init__(self, truth_en: np.ndarray, title: str, view_m: float) -> None:
        import matplotlib.pyplot as plt
        self.plt, self.view_m = plt, view_m
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        self.fig.canvas.manager.set_window_title("Edge Engine - live dead reckoning")
        ax = self.ax
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.plot(truth_en[:, 0], truth_en[:, 1], color=C_TRUTH, lw=1.2, alpha=0.6,
                label="VBOX reference (scoring only)")
        self.fix_pts = ax.scatter([], [], s=10, color=C_FIX, alpha=0.7, label="phone GNSS fixes delivered", zorder=3)
        (self.aided,) = ax.plot([], [], color=C_AIDED, lw=2.0, label="engine - GNSS-aided")
        (self.dr,) = ax.plot([], [], color=C_OUTAGE, lw=2.0, label="engine - GNSS outage (pure DR)")
        (self.head,) = ax.plot([], [], "o", ms=9, color=C_AIDED, mec="#fcfcfb", mew=2, zorder=5)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        pad = 50.0
        ok = np.isfinite(truth_en[:, 0])
        ax.set_xlim(np.nanmin(truth_en[ok, 0]) - pad, np.nanmax(truth_en[ok, 0]) + pad)
        ax.set_ylim(np.nanmin(truth_en[ok, 1]) - pad, np.nanmax(truth_en[ok, 1]) + pad)
        self.fig.suptitle(title, fontsize=11, x=0.02, ha="left")
        self.status = ax.text(0.99, 0.01, "", transform=ax.transAxes, ha="right", va="bottom",
                              fontsize=9, family="monospace", color="#0b0b0b")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def alive(self) -> bool:
        return self.plt.fignum_exists(self.fig.number)

    def update(self, aided: np.ndarray, dr: np.ndarray, fixes: np.ndarray, head, in_outage: bool, status: str) -> None:
        self.aided.set_data(aided[:, 0], aided[:, 1])
        self.dr.set_data(dr[:, 0], dr[:, 1])
        if len(fixes):
            self.fix_pts.set_offsets(fixes)
        self.head.set_data([head[0]], [head[1]])
        if self.view_m > 0:
            self.ax.set_xlim(head[0] - self.view_m, head[0] + self.view_m)
            self.ax.set_ylim(head[1] - self.view_m, head[1] + self.view_m)
        self.head.set_color(C_OUTAGE if in_outage else C_AIDED)
        self.status.set_text(status)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def hold(self) -> None:
        self.plt.ioff()
        if self.alive():
            self.plt.show()


def fmt_ms(x: float) -> str:
    return f"{x * 1e3:7.3f} ms"


def main() -> int:
    a = parse_args()
    if a.stream_hz <= 0:
        raise ValueError("--stream-hz must be > 0")
    seg = pick_segment(a.session, a.segment)
    t_init = init_time(seg)
    if t_init is None:
        raise RuntimeError(f"{a.session}/{seg.segment_id}: no GNSS fix > 5 m/s with bearing; engine cannot initialise")
    src_hz = 1.0 / float(np.median(np.diff(seg.t)))
    speedup = a.stream_hz / src_hz

    if a.outage_s > 0:
        o0 = t_init + a.outage_after_s
        if o0 + a.outage_s > seg.t[-1]:
            raise ValueError(f"outage [{o0:.0f}, {o0 + a.outage_s:.0f}] s does not fit the {seg.t[-1]:.0f} s segment")
        sched = OutageSchedule((OutageEvent(o0, o0 + a.outage_s, "full", label=f"full_{a.outage_s:g}s"),))
    else:
        sched = OutageSchedule.none()
    ev = sched.events[0] if sched.events else None

    first_fix = seg.fixes[0]
    ltp = LocalTangentPlane(first_fix.lat_rad, first_fix.lon_rad, 0.0)
    truth_en, truth_s = truth_track(seg, ltp)
    replay = ReplayEngine(seg, sched, seed=26168)   # outage applied here; pacing is ours
    engine = load_engine()

    n_total = len(seg.t) if a.max_samples is None else min(len(seg.t), a.max_samples)
    header = (f"EDGE ENGINE - {a.stream_hz:g} Hz STREAM | IO-VNBD {seg.session_id}/seg {seg.segment_id} | "
              f"source {src_hz:.1f} Hz, replayed {speedup:.1f}x real time")
    print("=" * len(header))
    print(header)
    print(f"samples {n_total}  drive {seg.duration_s:.0f} s  init at t={t_init:.1f} s  "
          + (f"GNSS OUTAGE t=[{ev.start_s:.1f}, {ev.end_s:.1f}) s ({ev.duration_s:g} s, fixes withheld)" if ev
             else "no outage"))
    print("engine: NavigationEngine, FusionConfig() shipped defaults, causal mount | truth: VBOX, scoring only")
    print("=" * len(header))

    plot = None
    if not a.no_plot:
        plot = LivePlot(truth_en, header, a.view_m)

    period = 1.0 / a.stream_hz
    lat_hist: deque[float] = deque(maxlen=2000)
    lat_all: list[tuple[float, str]] = []   # (latency [s], engine mode)
    aided, dr, fused = [], [], []
    missed = 0
    err_m = drift_pct = None
    last_outage = None     # (err_m, dist_m, pct) at the end of the outage
    out = None
    head = None
    was_outage = False
    wall0 = time.perf_counter()
    next_due = wall0
    for i, t, acc, grav, gyro, fixes in replay.stream():
        if i >= n_total:
            break
        # ---- pace: one sample every 1/stream_hz wall seconds (deadline-based, no drift)
        now = time.perf_counter()
        if next_due > now:
            time.sleep(next_due - now)
        elif now - next_due > period:
            missed += 1
        next_due += period

        t0 = time.perf_counter()
        o = engine.on_sample(t, acc, grav, gyro, fixes)
        dt_proc = time.perf_counter() - t0
        lat_hist.append(dt_proc)
        lat_all.append((dt_proc, o.mode if o is not None else "WAITING"))
        fused.extend(fixes)

        in_outage = ev is not None and ev.covers(t)
        if o is not None:
            out = o
            p = ltp.geodetic_to_enu(o.lat_rad, o.lon_rad, 0.0)[:2]
            # separate polylines: DR starts where aided ended; aided resumes after a NaN break
            if in_outage and not was_outage and aided:
                dr.append(aided[-1])
            if was_outage and not in_outage:
                aided.append(np.array([np.nan, np.nan]))
            was_outage = in_outage
            (dr if in_outage else aided).append(p)
            head = p
            if o.mode != "WAITING" and np.isfinite(truth_en[i, 0]):
                err_m = float(np.hypot(*(p - truth_en[i])))
                if ev is not None and t >= ev.start_s:
                    j0 = int(np.searchsorted(seg.t, ev.start_s))
                    dist = float(truth_s[i] - truth_s[j0])
                    if in_outage:
                        drift_pct = 100.0 * err_m / dist if dist >= DRIFT_MIN_DIST_M else None
                        if drift_pct is not None:
                            last_outage = (err_m, dist, drift_pct)

        if i % a.print_every == 0 or i == n_total - 1:
            lh = np.fromiter(lat_hist, float)
            mode = out.mode if out else "WAITING"
            gs = "OUTAGE" if in_outage else (out.gnss_state if out else "-")
            drift = (f"{drift_pct:5.2f} %" if in_outage and drift_pct is not None
                     else f"{last_outage[2]:5.2f} % (end of outage)" if last_outage and not in_outage
                     else "  n/a")
            line = (f"\r[EDGE ENGINE - {a.stream_hz:g}Hz] t={t:7.1f}s  {mode:10} GNSS {gs:10} "
                    f"lat {fmt_ms(lh[-1])} p50 {fmt_ms(np.median(lh))} p99 {fmt_ms(np.percentile(lh, 99))}  "
                    f"err {err_m if err_m is not None else float('nan'):7.1f} m  drift {drift:<22}  missed {missed}")
            sys.stdout.write(line)
            sys.stdout.flush()

        if plot is not None and (i % a.plot_every == 0 or i == n_total - 1) and out is not None:
            if not plot.alive():
                print("\nplot window closed - stopping")
                break
            lh = np.fromiter(lat_hist, float)
            fx = ltp.geodetic_to_enu(np.array([f.lat_rad for f in fused]), np.array([f.lon_rad for f in fused]),
                                     np.zeros(len(fused)))[:, :2] if fused else np.empty((0, 2))
            status = (f"t {t:6.1f} s   {out.mode}   GNSS {'OUTAGE' if in_outage else out.gnss_state}\n"
                      f"latency p50 {np.median(lh) * 1e3:.3f} ms  p99 {np.percentile(lh, 99) * 1e3:.3f} ms\n"
                      f"error {err_m if err_m is not None else float('nan'):.1f} m   "
                      + (f"drift {drift_pct:.2f} % of distance" if in_outage and drift_pct is not None
                         else f"outage drift {last_outage[2]:.2f} %" if last_outage else "drift n/a"))
            plot.update(np.array(aided).reshape(-1, 2), np.array(dr).reshape(-1, 2), fx, head, in_outage, status)

    wall = time.perf_counter() - wall0
    n = len(lat_all)
    print("\n" + "=" * len(header))
    print(f"SUMMARY  {n} samples in {wall:.1f} s wall = {n / wall:.1f} samples/s achieved "
          f"(target {a.stream_hz:g}); deadlines missed by > 1 period: {missed}")
    print("engine latency per sample (engine.on_sample only; this host, CPython):")
    for mode in ("WAITING", "ALIGNING", "NAVIGATING"):
        la = np.array([x for x, m in lat_all if m == mode])
        if la.size:
            p99 = float(np.percentile(la, 99))
            print(f"  {mode:10} n {la.size:6d}  p50 {np.median(la) * 1e3:7.3f} ms  p99 {p99 * 1e3:7.3f} ms  "
                  f"max {la.max() * 1e3:7.3f} ms  -> sustainable ~{1.0 / p99:,.0f} samples/s at p99")
    if last_outage:
        e, d, pct = last_outage
        print(f"GNSS outage ({ev.duration_s:g} s): error {e:.1f} m after {d:.0f} m travelled = drift {pct:.2f} % "
              "of distance (vs VBOX)")
    elif ev is not None:
        print("GNSS outage: not reached / < 50 m travelled - no drift figure")
    if out is not None:
        print(f"final: mode {out.mode}, horizontal error {err_m if err_m is not None else float('nan'):.1f} m, "
              f"filter 1-sigma {out.sigma_h_m:.1f} m")
    print("Note: latency is measured on this host in CPython, not on a phone; the stream is a time-compressed "
          f"replay of {src_hz:.0f} Hz data, not a {a.stream_hz:g} Hz sensor.")
    if plot is not None:
        print("Close the plot window to exit.")
        plot.hold()
    return 0


if __name__ == "__main__":
    sys.exit(main())
