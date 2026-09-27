#!/usr/bin/env python3
"""Phase 9 benchmark: the GNSS outage battery replayed through the live-style engine.

    .venv/Scripts/python.exe scripts/evaluate/phase9_outage_benchmark.py

Every clock-verified, non-TEST drive with a >= 5 min segment is replayed sample by sample
(simulation.replay) through navcore.fusion.engine.NavigationEngine with the shipped fusion
defaults (AI speed + NHC, Model B off) and CAUSAL mount estimation. After initialisation +
150 s, the Phase 9 battery (simulation.outage OutageSchedule.benchmark) runs: full dropouts
of 10 / 30 / 60 / 120 / 300 s, a 45 s tunnel (degraded approach, multipath outlier at the
exit), 120 s intermittent, 60 s degraded -- 150 s of normal GNSS between events, cycling.

Arms:
  phase7      FusionConfig()                              (no recovery manager)
  recovery    FusionConfig(recovery=RecoveryConfig())     (Phase 9 recovery manager)
  given_mount phase7 with the whole-drive mount (the Phase 4/7 lookahead) -- VAL only,
              to measure what that lookahead was worth
Both arms report the continuous OUTPUT (OutputSmoother) and the raw FILTER estimate.

Per event: error at the event end; then over the following gap: the largest per-step
position change not explained by the velocity ("jump"), error 10 s / 30 s after, time until
the output is back within 20 m, and whether the filter jumped >= 50 m AWAY from the truth
within 5 s (a wrong jump). VBOX scores only. The recovery manager has no fitted
parameters, so TRAIN drives are legitimate for comparing arms; absolute outage errors on
TRAIN drives are optimistic (Model A was trained on them) and VAL is reported separately.
Writes reports/phase9/outage_benchmark.json.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from navcore.alignment.mounting import estimate_mounting  # noqa: E402
from navcore.filtering.ekf import CovarianceError, NoiseParams  # noqa: E402
from navcore.fusion.ai_models import OnnxSequenceModel  # noqa: E402
from navcore.fusion.engine import EngineConfig, NavigationEngine  # noqa: E402
from navcore.fusion.navigator import FusionConfig  # noqa: E402
from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius  # noqa: E402
from navcore.recovery.manager import RecoveryConfig  # noqa: E402
from simulation.outage.simulator import OutageSchedule  # noqa: E402
from simulation.replay.replay import ReplayEngine, load_segments  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
OUT = CFG.reports_dir / "phase9" / "outage_benchmark.json"
ARMS = {"phase7": FusionConfig(), "recovery": FusionConfig(recovery=RecoveryConfig())}
RECONVERGED_M = 20.0
WRONG_JUMP_M = 50.0


def init_time(seg) -> float | None:
    f0 = next((f for f in seg.fixes if f.speed_mps is not None and f.speed_mps > 5.0 and f.bearing_rad is not None), None)
    if f0 is None:
        return None
    i = int(np.searchsorted(seg.t, f0.t_received_s - 1e-9))
    return float(seg.t[i]) if i < len(seg.t) else None


def whole_drive_mount(seg):
    vf = [f for f in seg.fixes if f.speed_mps is not None]
    ft = np.array([f.t_s for f in vf])
    iv = float(np.median(np.diff(ft)))
    return estimate_mounting(seg.acc, seg.t, ft, np.array([f.speed_mps for f in vf]),
                             np.array([f.bearing_rad for f in vf]), max_fix_gap_s=2.5 if iv <= 2 else 12.0)


def errors(res, seg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal error of OUTPUT and FILTER vs VBOX per output row (nan where invalid)."""
    tr = seg.truth
    i = res.rows
    glat, glon = np.radians(tr["gt_lat_deg"].to_numpy()[i]), np.radians(tr["gt_lon_deg"].to_numpy()[i])
    ok = tr["gt_valid"].to_numpy()[i].astype(bool) & np.array([m == "NAVIGATING" for m in res.mode])
    rm = meridian_radius(glat)
    rn = prime_vertical_radius(glat) * np.cos(glat)

    def e(lat, lon):
        d = np.hypot((lat - glat) * rm, (lon - glon) * rn)
        return np.where(ok, d, np.nan)

    return e(res.lat, res.lon), e(res.filter_lat, res.filter_lon), ok


def step_anomaly(res, lat, lon) -> np.ndarray:
    """|dp - v dt| between consecutive outputs [m]: position change the velocity does not explain."""
    rm = meridian_radius(lat[1:])
    rn = prime_vertical_radius(lat[1:]) * np.cos(lat[1:])
    d = np.column_stack([np.diff(lon) * rn, np.diff(lat) * rm])
    dt = np.diff(res.t)[:, None]
    return np.r_[0.0, np.linalg.norm(d - res.v_en[1:] * dt, axis=1)]


def score_events(res, seg, sched, t_init) -> list[dict]:
    e_out, e_fil, ok = errors(res, seg)
    a_out, a_fil = step_anomaly(res, res.lat, res.lon), step_anomaly(res, res.filter_lat, res.filter_lon)
    rows = []
    for k, ev in enumerate(sched.events):
        nxt = sched.events[k + 1].start_s if k + 1 < len(sched.events) else res.t[-1] + 1.0
        inside = np.where((res.t >= ev.start_s) & (res.t < ev.end_s))[0]
        after = np.where((res.t >= ev.end_s) & (res.t < nxt))[0]
        if not inside.size or not after.size:
            continue
        j = inside[-1]

        def at(dt, arr, t0=ev.end_s, t_next=nxt):
            w = np.where((res.t >= t0 + dt) & ok)[0]
            return float(arr[w[0]]) if w.size and res.t[w[0]] < t_next else None

        good = after[ok[after] & (e_out[after] < RECONVERGED_M)]
        base = e_fil[j] if ok[j] else np.nan
        early = after[(res.t[after] < ev.end_s + 5.0) & ok[after]]
        rows.append({
            "kind": ev.kind, "duration_s": round(ev.duration_s), "label": ev.label,
            "end_err_output_m": float(e_out[j]) if ok[j] else None, "end_err_filter_m": float(base) if ok[j] else None,
            "max_jump_output_m": float(np.max(a_out[after])), "max_jump_filter_m": float(np.max(a_fil[after])),
            "err_output_10s_m": at(10.0, e_out), "err_filter_10s_m": at(10.0, e_fil),
            "err_output_30s_m": at(30.0, e_out), "err_filter_30s_m": at(30.0, e_fil),
            "reconverge_s": float(res.t[good[0]] - ev.end_s) if good.size else None,
            "wrong_jump": bool(early.size and np.isfinite(base) and np.nanmax(e_fil[early]) > base + WRONG_JUMP_M)})
    return rows


def aggregate(rows: list[dict]) -> dict:
    def med(key, rs):
        v = [r[key] for r in rs if r[key] is not None]
        return (float(np.median(v)), float(np.percentile(v, 90)), len(v)) if v else (None, None, 0)

    out = {}
    for key in sorted({(r["kind"], r["duration_s"]) for r in rows}):
        rs = [r for r in rows if (r["kind"], r["duration_s"]) == key]
        out[f"{key[0]}_{key[1]}s"] = {
            "n": len(rs),
            **{f"{m}_p50_p90_n": med(m, rs) for m in ("end_err_output_m", "end_err_filter_m", "err_output_10s_m",
                                                      "err_filter_10s_m", "err_output_30s_m", "err_filter_30s_m",
                                                      "reconverge_s")},
            "max_jump_output_m": max(r["max_jump_output_m"] for r in rs),
            "max_jump_filter_m": max(r["max_jump_filter_m"] for r in rs),
            "wrong_jumps": sum(r["wrong_jump"] for r in rs),
            "never_reconverged": sum(r["reconverge_s"] is None for r in rs)}
    return out


def main() -> int:
    t_start = time.time()
    split = json.loads(CFG.split_manifest.read_text(encoding="utf-8"))
    p3 = json.loads((CFG.reports_dir / "phase3" / "preprocessing_report.json").read_text(encoding="utf-8"))
    verified = {s["session_id"] for s in p3["sessions"] if s["alignment_verified"]}
    sessions = sorted(s for s in verified if split["sessions"][s]["split"] in ("train", "val"))
    sel = json.loads((REPO_ROOT / "reports" / "phase6" / "selection.json").read_text(encoding="utf-8"))
    models = {"model_speed": OnnxSequenceModel(REPO_ROOT / "models" / "exported" / f"{sel['model_a']['selected']}.onnx")}
    noise = NoiseParams.from_report(json.loads((CFG.reports_dir / "phase4" / "imu_noise.json")
                                               .read_text(encoding="utf-8"))["filter_parameters"])
    rows = {a: [] for a in [*ARMS, "given_mount"]}
    runs, counts = [], {a: {} for a in rows}
    for sid in sessions:
        sp = split["sessions"][sid]["split"]
        for seg in load_segments(CFG.synced_root / f"{sid}.parquet"):
            t_init = init_time(seg)
            if t_init is None or len([f for f in seg.fixes if f.speed_mps is not None]) < 20:
                continue
            sched = OutageSchedule.benchmark(t_init + 150.0, float(seg.t[-1]))
            if not sched.events:
                continue
            arms = list(ARMS) + (["given_mount"] if sp == "val" else [])
            for arm in arms:
                if arm == "given_mount":
                    eng = NavigationEngine(noise, EngineConfig(fusion=ARMS["phase7"], mount_mode="given"), models,
                                           mount=whole_drive_mount(seg))
                else:
                    eng = NavigationEngine(noise, EngineConfig(fusion=ARMS[arm]), models)
                try:
                    res = ReplayEngine(seg, sched, seed=26168).run(eng)
                except CovarianceError as exc:
                    runs.append({"session": sid, "segment": seg.segment_id, "arm": arm, "diverged": str(exc)})
                    print(f"{sid}/{seg.segment_id} {arm}: DIVERGED {exc}", flush=True)
                    continue
                ev = score_events(res, seg, sched, t_init)
                for r in ev:
                    r.update(session=sid, segment=seg.segment_id, split=sp)
                rows[arm].extend(ev)
                c = dict(eng.nav.counts) | (eng.nav.recovery.counts if eng.nav.recovery else {})
                for k, v in c.items():
                    counts[arm][k] = counts[arm].get(k, 0) + v
                e_out, e_fil, _ = errors(res, seg)
                runs.append({"session": sid, "segment": seg.segment_id, "split": sp, "arm": arm,
                             "events": len(ev), "digest": res.digest, "wall_s": round(res.wall_s, 1),
                             "output_err_p50_m": float(np.nanmedian(e_out)), "filter_err_p50_m": float(np.nanmedian(e_fil))})
                print(f"{sid}/{seg.segment_id} [{sp}] {arm:11} {len(ev)} events, {res.wall_s:.0f} s", flush=True)
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "script": "scripts/evaluate/phase9_outage_benchmark.py",
              "fusion": "FusionConfig() shipped defaults; engine mount_mode='causal' unless given_mount",
              "battery": "OutageSchedule.benchmark(t_init + 150 s): full 10/30/60/120/300 s, tunnel 45 s, "
                         "intermittent 120 s, degraded 60 s; 150 s gaps; seed 26168",
              "sessions": sessions, "runs": runs,
              "val": {a: aggregate([r for r in rows[a] if r["split"] == "val"]) for a in rows},
              "train_and_val": {a: aggregate(rows[a]) for a in ARMS},
              "counts": counts, "events": rows,
              "caveat": "TRAIN absolute errors optimistic (Model A trained there); arms compared on identical inputs",
              "runtime_s": round(time.time() - t_start)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    for a in ARMS:
        print(f"== {a} (train+val)")
        for k, v in report["train_and_val"][a].items():
            print(f"   {k:18} n{v['n']:3}  end out/filt p50 {v['end_err_output_m_p50_p90_n'][0]} / "
                  f"{v['end_err_filter_m_p50_p90_n'][0]}  +30s out {v['err_output_30s_m_p50_p90_n'][0]}  "
                  f"jump out/filt {v['max_jump_output_m']:.1f}/{v['max_jump_filter_m']:.1f}  "
                  f"reconv {v['reconverge_s_p50_p90_n'][0]}  wrong {v['wrong_jumps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
