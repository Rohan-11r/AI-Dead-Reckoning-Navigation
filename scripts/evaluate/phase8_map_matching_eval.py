#!/usr/bin/env python3
"""Phase 8 real-data evaluation: HMM map matching of the fused dead-reckoned trajectory.

    .venv/Scripts/python.exe scripts/evaluate/phase8_map_matching_eval.py

1. RECORD: for each clock-verified drive inside the cached OSM area, run the Phase 7 fused
   navigator with the SHIPPED defaults (FusionConfig(): AI speed + NHC, Model B off) through
   the exact Phase 7 protocol (initialisation, heading hypotheses, 30 s / 60 s GNSS outages
   every 4 min). The fused state is recorded at every step -- read-only; the matcher never
   feeds back (severability).
2. TUNE on TRAIN drives only (S1, S2, S4): transition scale beta and a covariance inflation
   (Phase 7 showed the fused covariance is over-confident: 2-sigma covers 75-79 %). The
   criterion is the median output error inside outages.
3. REPORT on VAL drives (S3a, S3b, S3c) with the tuned setting. TEST drives stay untouched.

Output policy scored here: use the matched position when matched and confidence >= 0.5,
otherwise the fused position (always valid). Road correctness uses a REFERENCE labelling:
the VBOX truth track itself matched with the same HMM (3 m sigma) -- evaluation only.
Writes reports/phase8/map_matching_evaluation.json.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scripts.evaluate.phase7_fusion_eval as P7  # noqa: E402
from navcore.alignment.mounting import estimate_mounting  # noqa: E402
from navcore.filtering.ekf import P_, V_, NoiseParams  # noqa: E402
from navcore.fusion.ai_models import OnnxSequenceModel  # noqa: E402
from navcore.fusion.navigator import FusionConfig  # noqa: E402
from navcore.map_matching import (  # noqa: E402
    DeadReckoningMapMatcher,
    HmmMapMatcher,
    MatcherConfig,
    RoadNetwork,
)
from scripts.evaluate.phase4_baseline import along_cross, err_en_m  # noqa: E402

OSM = REPO_ROOT / "data" / "raw" / "osm" / "coventry_drivable.osm"
CACHE = REPO_ROOT / "data" / "processed" / "osm" / "coventry_drivable.npz"
REC_DIR = REPO_ROOT / "data" / "interim" / "phase8"
OUT = REPO_ROOT / "reports" / "phase8" / "map_matching_evaluation.json"
TRAIN = ["S1", "S2", "S4"]
VAL = ["S3a", "S3b", "S3c"]
GRID = {"beta_m": [5.0, 10.0, 20.0, 40.0], "cov_scale": [1.0, 4.0, 9.0]}
CONF_MIN = 0.5
CORRECT_M = 20.0


def load_network() -> RoadNetwork:
    if CACHE.exists():
        return RoadNetwork.load(CACHE)
    net = RoadNetwork.from_osm_xml(OSM)
    net.save(CACHE)
    return net


def record(sid: str, models: dict, noise: NoiseParams) -> list[dict]:
    """Fused trajectory (every step) for each scoreable segment of ``sid``; cached."""
    f = REC_DIR / f"{sid}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return [{k[len(f"s{i}_"):]: z[k] for k in z.files if k.startswith(f"s{i}_")}
                for i in range(int(z["n_segments"]))]
    full = pd.read_parquet(P7.CFG.synced_root / f"{sid}.parquet")
    segs = []
    for _, d in full.groupby("segment_id"):
        d = d.reset_index(drop=True)
        if (d["t_utc"].iloc[-1] - d["t_utc"].iloc[0]).total_seconds() < P7.MIN_SEGMENT_S:
            continue
        truth = d[["gt_lat_deg", "gt_lon_deg", "gt_heading_deg", "gt_speed_mps", "gt_valid"]]
        inp = d[[*P7.INPUTS, "t_utc"]]
        t0 = inp["t_utc"].iloc[0]
        fixes = P7.build_fixes(inp, t0)
        vf = [x for x in fixes if x.speed_mps is not None]
        if len(vf) < 20:
            continue
        tt = (inp["t_utc"] - t0).dt.total_seconds().to_numpy()
        ft = np.array([x.t_s for x in vf])
        iv = float(np.median(np.diff(ft)))
        mount = estimate_mounting(inp[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy(), tt, ft,
                                  np.array([x.speed_mps for x in vf]), np.array([x.bearing_rad for x in vf]),
                                  max_fix_gap_s=2.5 if iv <= 2 else 12.0)
        rows = []

        def obs(i, nav, in_outage, rows=rows):
            s = nav.ekf.nominal
            P = nav.ekf.P
            rows.append([i, s.t_s, s.lat_rad, s.lon_rad, *P[P_, P_][:2, :2].ravel(), *s.v_enu_mps[:2],
                         *P[V_, V_][:2, :2].ravel(), float(in_outage)])

        r = P7.run_arm("default", inp, truth, fixes, mount, noise, models, observer=obs, cfg=FusionConfig())
        if "diverged" in r or not rows:
            print(f"  {sid}: {r.get('diverged') or r.get('skipped')}")
            continue
        a = np.array(rows)
        idx = a[:, 0].astype(int)
        segs.append({"t": a[:, 1], "lat": a[:, 2], "lon": a[:, 3], "P": a[:, 4:8].reshape(-1, 2, 2),
                     "v": a[:, 8:10], "Pv": a[:, 10:14].reshape(-1, 2, 2), "outage": a[:, 14] > 0.5,
                     "gt_lat": truth["gt_lat_deg"].to_numpy()[idx], "gt_lon": truth["gt_lon_deg"].to_numpy()[idx],
                     "gt_heading": truth["gt_heading_deg"].to_numpy()[idx],
                     "gt_speed": truth["gt_speed_mps"].to_numpy()[idx],
                     "gt_valid": truth["gt_valid"].to_numpy()[idx].astype(bool)})
    REC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, n_segments=np.array(len(segs)),
                        **{f"s{i}_{k}": v for i, s in enumerate(segs) for k, v in s.items()})
    return segs


def truth_reference(net: RoadNetwork, s: dict) -> dict[float, tuple[float, float]]:
    """VBOX truth track matched with the same HMM (3 m sigma): the road reference (scoring only)."""
    m = HmmMapMatcher(net, MatcherConfig(sigma_map_m=3.0))
    ref, last = {}, -math.inf
    for k in range(len(s["t"])):
        if not s["gt_valid"][k] or s["t"][k] - last < 1.0 - 1e-9:
            continue
        last = s["t"][k]
        x, y, _ = net.to_xy(math.radians(s["gt_lat"][k]), math.radians(s["gt_lon"][k]))
        h = math.pi / 2 - math.radians(s["gt_heading"][k])
        r = m.step(s["t"][k], x, y, np.eye(2) * 9.0, h, float(s["gt_speed"][k]), math.radians(5.0))
        if r.matched and r.map_match_confidence >= 0.9:
            ref[round(s["t"][k], 1)] = (r.x_m, r.y_m)
    return ref


def score(net: RoadNetwork, segs: list[dict], refs: list[dict], cfg: MatcherConfig, cov_scale: float) -> dict:
    ep = []  # one row per matcher epoch
    for s, ref in zip(segs, refs, strict=True):
        dr = DeadReckoningMapMatcher(net, cfg)
        n = len(s["t"])
        in_out = s["outage"]
        win, t_start, rows = -1, {}, []
        for k in range(n):
            if in_out[k] and (k == 0 or not in_out[k - 1]):
                win += 1
                t_start[win] = s["t"][k]
            r = dr.update(s["t"][k], s["lat"][k], s["lon"][k], s["P"][k] * cov_scale, s["v"][k], s["Pv"][k] * cov_scale)
            if r is None or not s["gt_valid"][k]:
                continue
            gl, go = math.radians(s["gt_lat"][k]), math.radians(s["gt_lon"][k])
            de, dn = err_en_m(s["lat"][k], s["lon"][k], gl, go)
            use = r.matched and r.map_match_confidence >= CONF_MIN
            me, mn = err_en_m(r.lat_rad, r.lon_rad, gl, go) if use else (de, dn)
            rp = ref.get(round(s["t"][k], 1))
            correct = (math.hypot(r.x_m - rp[0], r.y_m - rp[1]) <= CORRECT_M) if (use and rp) else None
            rows.append({"outage": bool(in_out[k]), "win": win if in_out[k] else None,
                         "dur": round(s["t"][k] - t_start[win]) if in_out[k] else None, "end": None,
                         "fused": math.hypot(de, dn), "out": math.hypot(me, mn),
                         "fused_ac": along_cross(de, dn, s["gt_heading"][k]),
                         "out_ac": along_cross(me, mn, s["gt_heading"][k]), "matched": r.matched, "used": use,
                         "conf": r.map_match_confidence, "correct": correct, "mode": r.mode})
        # outage-end score = the last matcher epoch inside each window (as Phase 4/7: its end)
        last = {}
        for i, r in enumerate(rows):
            if r["win"] is not None:
                last[r["win"]] = i
        for i in last.values():
            d = rows[i]["dur"]
            rows[i]["end"] = 30 if abs(d - 30) <= 2 else 60 if abs(d - 60) <= 2 else None
        ep.extend(rows)
    return summarise(ep)


def summarise(ep: list[dict]) -> dict:
    def stats(rows, key):
        v = np.array([r[key] for r in rows])
        return {"n": int(v.size), "p50_m": float(np.median(v)) if v.size else None,
                "p90_m": float(np.percentile(v, 90)) if v.size else None}

    def ac(rows, key):
        a = np.abs(np.array([r[key] for r in rows])) if rows else np.zeros((0, 2))
        return {"along_p50_m": float(np.median(a[:, 0])) if len(a) else None,
                "cross_p50_m": float(np.median(a[:, 1])) if len(a) else None}

    out = {}
    for name, rows in (("outage_epochs", [r for r in ep if r["outage"]]),
                       ("aided_epochs", [r for r in ep if not r["outage"]]),
                       ("outage_end_30s", [r for r in ep if r["end"] == 30]),
                       ("outage_end_60s", [r for r in ep if r["end"] == 60])):
        used = [r for r in rows if r["used"]]
        judged = [r for r in used if r["correct"] is not None]
        out[name] = {"fused": stats(rows, "fused") | ac(rows, "fused_ac"),
                     "output": stats(rows, "out") | ac(rows, "out_ac"),
                     "match_rate": len([r for r in rows if r["matched"]]) / len(rows) if rows else None,
                     "used_rate": len(used) / len(rows) if rows else None,
                     "correct_road_rate": (sum(r["correct"] for r in judged) / len(judged)) if judged else None,
                     "n_judged": len(judged),
                     "modes": {m: sum(r["mode"] == m for r in rows) for m in {r["mode"] for r in rows}}}
    # does it help when the dead reckoning is still close? outage epochs by fused-error band
    bands = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 1e9)]
    oe = [r for r in ep if r["outage"]]
    out["outage_by_fused_error"] = []
    for lo, hi in bands:
        rows = [r for r in oe if lo <= r["fused"] < hi]
        used = [r for r in rows if r["used"]]
        judged = [r for r in used if r["correct"] is not None]
        out["outage_by_fused_error"].append({
            "fused_error_m": [lo, hi if hi < 1e9 else None], "n": len(rows),
            "fused_p50_m": float(np.median([r["fused"] for r in rows])) if rows else None,
            "output_p50_m": float(np.median([r["out"] for r in rows])) if rows else None,
            "fused_cross_p50_m": float(np.median([abs(r["fused_ac"][1]) for r in rows])) if rows else None,
            "output_cross_p50_m": float(np.median([abs(r["out_ac"][1]) for r in rows])) if rows else None,
            "used_rate": len(used) / len(rows) if rows else None,
            "correct_road_rate": (sum(r["correct"] for r in judged) / len(judged)) if judged else None})
    bins = [0.5, 0.8, 0.95, 1.01]
    judged = [r for r in ep if r["used"] and r["correct"] is not None]
    out["confidence_calibration"] = [
        {"conf": [lo, min(hi, 1.0)], "n": len(b), "correct_rate": (sum(r["correct"] for r in b) / len(b)) if b else None}
        for lo, hi in itertools.pairwise(bins)
        for b in [[r for r in judged if lo <= r["conf"] < hi]]]
    return out


def main() -> int:
    t_start = time.time()
    net = load_network()
    sel = json.loads((REPO_ROOT / "reports" / "phase6" / "selection.json").read_text(encoding="utf-8"))
    exp = REPO_ROOT / "models" / "exported"
    models = {"model_speed": OnnxSequenceModel(exp / f"{sel['model_a']['selected']}.onnx")}
    noise = NoiseParams.from_report(json.loads((P7.CFG.reports_dir / "phase4" / "imu_noise.json")
                                               .read_text(encoding="utf-8"))["filter_parameters"])
    split = json.loads(P7.CFG.split_manifest.read_text(encoding="utf-8"))
    assert all(split["sessions"][s]["split"] == "train" for s in TRAIN)
    assert all(split["sessions"][s]["split"] == "val" for s in VAL)
    data = {}
    for sid in TRAIN + VAL:
        t0 = time.time()
        segs = record(sid, models, noise)
        data[sid] = (segs, [truth_reference(net, s) for s in segs])
        print(f"recorded {sid}: {len(segs)} segment(s), {sum(len(s['t']) for s in segs)} steps "
              f"({time.time() - t0:.0f} s)", flush=True)

    def pooled(sids):
        return [s for sid in sids for s in data[sid][0]], [r for sid in sids for r in data[sid][1]]

    tr_segs, tr_refs = pooled(TRAIN)
    tuning = []
    for beta in GRID["beta_m"]:
        for cs in GRID["cov_scale"]:
            res = score(net, tr_segs, tr_refs, replace(MatcherConfig(), beta_m=beta), cs)
            o = res["outage_epochs"]
            tuning.append({"beta_m": beta, "cov_scale": cs, "outage_output_p50_m": o["output"]["p50_m"],
                           "outage_fused_p50_m": o["fused"]["p50_m"], "correct_road_rate": o["correct_road_rate"],
                           "used_rate": o["used_rate"]})
            print(f"  train beta {beta:4.0f} cov x{cs:.0f}: outage p50 fused {o['fused']['p50_m']:.1f} -> "
                  f"output {o['output']['p50_m']:.1f} m | used {o['used_rate']:.2f} correct {o['correct_road_rate']}",
                  flush=True)
    best = min(tuning, key=lambda r: (r["outage_output_p50_m"], r["beta_m"], r["cov_scale"]))
    cfg = replace(MatcherConfig(), beta_m=best["beta_m"])
    va_segs, va_refs = pooled(VAL)
    val = score(net, va_segs, va_refs, cfg, best["cov_scale"])
    per_session = {sid: score(net, *data[sid], cfg, best["cov_scale"]) for sid in VAL}
    report = {"generated": datetime.now().isoformat(timespec="seconds"),
              "script": "scripts/evaluate/phase8_map_matching_eval.py",
              "network": {"source": "reports/phase8/osm_manifest.json", "segments": net.n_segments,
                          "nodes": len(net.node_xy), "ways": len(net.way_tags)},
              "fusion": "FusionConfig() shipped defaults (AI speed + NHC, Model B off)",
              "output_policy": f"matched position if matched and confidence >= {CONF_MIN}, else fused",
              "correct_road": f"matched point within {CORRECT_M} m of the VBOX-truth-track match (reference)",
              "train_sessions": TRAIN, "val_sessions": VAL,
              "tuning_on_train": tuning, "selected": best,
              "val": val, "val_per_session": per_session,
              "caveat": "3 VAL drives; Phase 6/7 choices were also made on VAL -> optimistic; TEST reserved",
              "runtime_s": round(time.time() - t_start)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    for k in ("outage_epochs", "outage_end_30s", "outage_end_60s", "aided_epochs"):
        v = val[k]
        print(f"VAL {k:15} fused p50 {v['fused']['p50_m']:.1f} / p90 {v['fused']['p90_m']:.1f} -> output p50 "
              f"{v['output']['p50_m']:.1f} / p90 {v['output']['p90_m']:.1f} (n {v['fused']['n']}) | cross "
              f"{v['fused']['cross_p50_m']:.1f} -> {v['output']['cross_p50_m']:.1f} | along "
              f"{v['fused']['along_p50_m']:.1f} -> {v['output']['along_p50_m']:.1f} | used {v['used_rate']:.2f} "
              f"correct {v['correct_road_rate']}")
    print("calibration", val["confidence_calibration"])
    for b in val["outage_by_fused_error"]:
        print("band", b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
