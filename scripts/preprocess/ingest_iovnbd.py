#!/usr/bin/env python3
"""Phase 2 ingestion: IO-VNBD raw CSV -> deduplicated, audited, SI-unit Parquet.

    .venv/Scripts/python.exe scripts/preprocess/ingest_iovnbd.py [--workers N]

Steps
  1. Discover all 288 CSVs under data/raw/IO-VNBD (read-only).
  2. Map every header onto the canonical schema (two schemas: S = phone, V = vehicle).
  3. Deduplicate Categorised vs Uncategorised copies by SHA-256 of the bytes AND by a
     content digest of the parsed values (the S copies differ only by float formatting
     and header labels -- see training/preprocessing/iovnbd.content_digest).
  4. Convert the canonical copy of each unique session to SI and audit it.
  5. Write data/processed/iovnbd/v1/{phone,vehicle}/<session>.parquet + manifest.json.
  6. Derive the route/driver split from real durations -> data/splits/iovnbd_split_v1.json.
  7. Write reports/phase2/dataset_report.json and dataset_report.html.

Deterministic: no randomness except the split's seeded shuffle (SEED below).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.preprocessing import iovnbd as io  # noqa: E402
from training.preprocessing import iovnbd_audit as audit  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
RAW_ROOT = CFG.raw_root
OUT_ROOT = CFG.processed_root
# Phase 2 manifest; superseded by the drive-aware v2 from training/preprocessing/splits.py
SPLIT_PATH = CFG.splits_dir / "iovnbd_split_v1.json"
REPORT_DIR = CFG.reports_dir / "phase2"
SCHEMA_VERSION = "iovnbd-v1"
SEED = 26168
TEST_FRACTION_TARGET = 0.20
TEST_FRACTION_BOUNDS = (0.10, 0.35)
VAL_FRACTION_TARGET = 0.15


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_parquet(df, path: Path, meta: dict) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    units = {c: io.COLUMN_UNITS.get(c, "?") for c in df.columns}
    md = dict(table.schema.metadata or {})
    md[b"sih26168"] = json.dumps({"schema_version": SCHEMA_VERSION, "units": units, **meta}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(md), path, compression="zstd")


def process_session(session_key: str, copies: dict[str, list[dict]]) -> dict:
    """Worker: dedupe the copies of one session, convert, audit, write Parquet."""
    res: dict = {"session_key": session_key, "files": [], "errors": []}
    frames: dict[str, object] = {}
    for kind in ("S", "V"):
        infos = copies.get(kind, [])
        loaded = []
        for info in infos:
            path = Path(info["path"])
            entry = {k: info[k] for k in ("rel", "collection", "kind")}
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = io.sha256_file(path)
            try:
                raw, rep = io.load_raw(path, kind)
            except Exception as exc:
                entry["status"] = "excluded"
                entry["reason"] = f"{type(exc).__name__}: {exc}"
                res["errors"].append(entry["reason"])
                res["files"].append(entry)
                continue
            entry["content_digest"] = io.content_digest(raw)
            entry["header_variant"] = rep.header_variant
            entry["n_rows"] = rep.n_rows
            entry["coerced_non_numeric"] = rep.coerced_non_numeric
            loaded.append((info, entry, raw, rep))
            res["files"].append(entry)

        if not loaded:
            res["errors"].append(f"no loadable {kind} copy")
            continue
        # canonical copy: Categorised (carries driver/category; float text un-mangled)
        loaded.sort(key=lambda x: x[0]["collection"] != "categorised")
        canon_info, canon_entry, canon_raw, canon_rep = loaded[0]
        dedupe = {"kind": kind, "n_copies": len(loaded), "canonical": canon_entry["rel"]}
        if len(loaded) == 2:
            other = loaded[1]
            dedupe["byte_identical"] = canon_entry["sha256"] == other[1]["sha256"]
            dedupe["content_identical"] = canon_entry["content_digest"] == other[1]["content_digest"]
            dedupe["max_abs_numeric_diff"] = io.max_abs_numeric_diff(canon_raw, other[2]) if (
                len(canon_raw) == len(other[2])) else None
            dedupe["duplicate_of_canonical"] = other[1]["rel"]
            other[1]["status"] = "duplicate" if dedupe["content_identical"] else "CONFLICT"
            if not dedupe["content_identical"]:
                res["errors"].append(f"{kind} copies differ in content")
        canon_entry["status"] = "canonical"
        res[f"dedupe_{kind}"] = dedupe
        res["category"] = res.get("category") or canon_info["category"]
        res["driver"] = res.get("driver") or canon_info["driver"]
        if kind == "S":
            res["session_id"] = canon_info["stem"]
            si = io.to_si_phone(canon_raw, canon_rep)
            res["parse_S"] = {
                "sats_malformed": canon_rep.sats_malformed,
                "sats_excel_recovered": canon_rep.sats_excel_recovered,
                "date_unparsed": canon_rep.date_unparsed,
                "utc_lost_at_dst": canon_rep.utc_lost_at_dst,
                "coerced_non_numeric": canon_rep.coerced_non_numeric,
            }
        else:
            si = io.to_si_vehicle(canon_raw)
            res["parse_V"] = {"coerced_non_numeric": canon_rep.coerced_non_numeric}
        frames[kind] = (si, canon_entry)

    sid = res.get("session_id", session_key)
    res["session_id"] = sid
    res["route_group"] = io.route_group(sid)
    if "S" in frames:
        p, e = frames["S"]
        res["phone"] = audit.audit_phone(p)
        write_parquet(p, OUT_ROOT / "phone" / f"{sid}.parquet",
                      {"session_id": sid, "source": e["rel"], "source_sha256": e["sha256"]})
    if "V" in frames:
        v, e = frames["V"]
        res["vehicle"] = audit.audit_vehicle(v)
        if "S" in frames:
            res["pair"] = audit.audit_pair(frames["S"][0], v)
            # absolute UTC for the vehicle log: join S and V on t_utc, NEVER on row_idx
            v = v.assign(t_utc=audit.vehicle_utc(frames["S"][0], v))
        write_parquet(v, OUT_ROOT / "vehicle" / f"{sid}.parquet",
                      {"session_id": sid, "source": e["rel"], "source_sha256": e["sha256"],
                       "join_rule": "join with phone on t_utc (nearest, <= 60 ms); "
                                    "row_idx pairing is misaligned by the phone clock error"})
    return res


# --------------------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------------------

def make_split(sessions: list[dict]) -> dict:
    """Route/driver split per docs/ml_pipeline.md section 3, from REAL durations.

    test  = held-out drivers, entirely (answers: new person + vehicle + roads)
    val   = whole route groups from the remaining drivers (answers: new roads)
    train = the rest
    """
    dur = {s["session_id"]: s["phone"]["duration_s"] for s in sessions}
    total = sum(dur.values())
    by_driver: dict[str, float] = defaultdict(float)
    for s in sessions:
        by_driver[s["driver"]] += dur[s["session_id"]]
    drivers = sorted(by_driver)

    candidates = []
    for r in range(1, len(drivers)):
        for combo in combinations(drivers, r):
            frac = sum(by_driver[d] for d in combo) / total
            if TEST_FRACTION_BOUNDS[0] <= frac <= TEST_FRACTION_BOUNDS[1]:
                candidates.append((abs(frac - TEST_FRACTION_TARGET), r, combo, frac))
    if not candidates:
        raise RuntimeError(f"no driver subset gives a test fraction in {TEST_FRACTION_BOUNDS}")
    _, _, test_drivers, test_frac = min(candidates)

    rest = [s for s in sessions if s["driver"] not in test_drivers]
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for s in rest:
        groups[(s["driver"], s["route_group"])].append(s["session_id"])
    keys = sorted(groups)
    rng = np.random.default_rng(SEED)
    order = [keys[i] for i in rng.permutation(len(keys))]
    rest_total = sum(dur[s["session_id"]] for s in rest)
    val_keys: list[tuple[str, str]] = []
    val_dur = 0.0
    groups_per_driver = defaultdict(int)
    for k in keys:
        groups_per_driver[k[0]] += 1
    taken = defaultdict(int)
    for k in order:
        if val_dur >= VAL_FRACTION_TARGET * rest_total:
            break
        gd = sum(dur[x] for x in groups[k])
        # never empty a driver's train set; never let one group overshoot val by > 2x
        if taken[k[0]] + 1 >= groups_per_driver[k[0]] or val_dur + gd > 2 * VAL_FRACTION_TARGET * rest_total:
            continue
        val_keys.append(k)
        taken[k[0]] += 1
        val_dur += gd

    assign = {}
    for s in sessions:
        sid = s["session_id"]
        if s["driver"] in test_drivers:
            assign[sid] = "test"
        elif (s["driver"], s["route_group"]) in val_keys:
            assign[sid] = "val"
        else:
            assign[sid] = "train"
    splits = {name: sorted(k for k, v in assign.items() if v == name)
              for name in ("train", "val", "test")}
    return {
        "schema_version": SCHEMA_VERSION,
        "rule": "test = held-out drivers entirely; val = whole route groups of remaining "
                "drivers; train = rest. docs/ml_pipeline.md section 3.",
        "seed": SEED,
        "test_fraction_target": TEST_FRACTION_TARGET,
        "test_fraction_bounds": list(TEST_FRACTION_BOUNDS),
        "val_fraction_target_of_remaining": VAL_FRACTION_TARGET,
        "test_drivers": list(test_drivers),
        "driver_duration_h": {d: round(by_driver[d] / 3600, 3) for d in drivers},
        "splits": splits,
        "sessions": {
            s["session_id"]: {"split": assign[s["session_id"]], "driver": s["driver"],
                              "category": s["category"], "route_group": s["route_group"],
                              "duration_s": round(dur[s["session_id"]], 3)}
            for s in sorted(sessions, key=lambda s: s["session_id"])
        },
        "split_duration_h": {
            name: round(sum(dur[x] for x in ids) / 3600, 3) for name, ids in splits.items()
        },
        "split_fraction": {
            name: round(sum(dur[x] for x in ids) / total, 4) for name, ids in splits.items()
        },
        "test_fraction_actual": round(test_frac, 4),
    }


# --------------------------------------------------------------------------------------
# Dataset-level aggregation
# --------------------------------------------------------------------------------------

def aggregate(sessions: list[dict], file_entries: list[dict]) -> dict:
    ph = [s["phone"] for s in sessions if "phone" in s]
    ve = [s["vehicle"] for s in sessions if "vehicle" in s]
    pr = [s["pair"] for s in sessions if "pair" in s]

    def sum_dict(ds):
        out: dict[str, int] = defaultdict(int)
        for d in ds:
            for k, v in d.items():
                out[k] += v
        return dict(out)

    g_means = np.array([p["gravity_channel"]["norm_mean_mps2"] for p in ph])
    g_std = np.array([p["gravity_channel"]["norm_std_mps2"] for p in ph])
    g_local = np.array([p["gravity_channel"]["local_normal_gravity_mps2"] for p in ph])
    n_rows = np.array([p["n_rows"] for p in ph])
    pooled_g = float(np.sum(g_means * n_rows) / n_rows.sum())
    stat = [(x["stationary"]["n_rows"], x["stationary"]["acc_norm_mean_mps2"],
             x["stationary"]["linacc_norm_mean_mps2"]) for x in pr
            if x["stationary"]["n_rows"] > 0]
    st_n = np.array([s[0] for s in stat])
    st_acc = np.array([s[1] for s in stat])
    st_lin = np.array([s[2] for s in stat])

    gyro_votes: dict[str, int] = defaultdict(int)
    gyro_signs: dict[str, int] = defaultdict(int)
    strong = [x for x in pr if x["gyro_vs_can_yaw_rate"]["best_axis"] and abs(
        x["gyro_vs_can_yaw_rate"]["corr"][x["gyro_vs_can_yaw_rate"]["best_axis"]]) >= 0.5]
    for x in strong:
        g = x["gyro_vs_can_yaw_rate"]
        if g["best_axis"]:
            gyro_votes[g["best_axis"]] += 1
            gyro_signs[f'{g["best_axis"]}{"+" if g["best_sign"] > 0 else "-"}'] += 1
    corr_by_axis = {ax: [x["gyro_vs_can_yaw_rate"]["corr"][ax] for x in strong
                         if x["gyro_vs_can_yaw_rate"]["corr"][ax] is not None] for ax in "xyz"}
    gain_by_axis = {ax: [x["gyro_vs_can_yaw_rate"]["gain"][ax] for x in strong
                         if x["gyro_vs_can_yaw_rate"]["gain"][ax] is not None] for ax in "xyz"}
    long_votes: dict[str, int] = defaultdict(int)
    lat_votes: dict[str, int] = defaultdict(int)
    for x in pr:
        if x["linacc_vs_can_long"]["best_axis"]:
            long_votes[x["linacc_vs_can_long"]["best_axis"]] += 1
        if x["linacc_vs_can_lat"]["best_axis"]:
            lat_votes[x["linacc_vs_can_lat"]["best_axis"]] += 1

    def med(vals):
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    dedupe_rows = [s.get(f"dedupe_{k}") for s in sessions for k in "SV" if s.get(f"dedupe_{k}")]
    status_counts: dict[str, int] = defaultdict(int)
    for f in file_entries:
        status_counts[f.get("status", "?")] += 1

    return {
        "files": {
            "total_csv": len(file_entries),
            "by_status": dict(status_counts),
            "unique_sha256": len({f["sha256"] for f in file_entries}),
            "unique_content_digest": len({f.get("content_digest") for f in file_entries
                                          if f.get("content_digest")}),
            "header_variants": dict(sum_dict([{f.get("header_variant", "?"): 1}
                                              for f in file_entries])),
        },
        "dedupe": {
            "pairs": len([d for d in dedupe_rows if d["n_copies"] == 2]),
            "byte_identical_pairs": sum(1 for d in dedupe_rows if d.get("byte_identical")),
            "content_identical_pairs": sum(1 for d in dedupe_rows if d.get("content_identical")),
            "byte_identical_by_kind": {k: sum(1 for d in dedupe_rows if d["kind"] == k
                                              and d.get("byte_identical")) for k in "SV"},
            "content_identical_by_kind": {k: sum(1 for d in dedupe_rows if d["kind"] == k
                                                 and d.get("content_identical")) for k in "SV"},
            "max_abs_numeric_diff_overall": max(
                (d["max_abs_numeric_diff"] for d in dedupe_rows
                 if d.get("max_abs_numeric_diff") is not None), default=None),
            "conflicts": [s["session_id"] for s in sessions
                          if any(d.get("content_identical") is False
                                 for d in (s.get("dedupe_S", {}), s.get("dedupe_V", {})))],
        },
        "sessions": {
            "unique": len(sessions),
            "with_both_logs": len(pr),
            "rows_equal_S_V": sum(1 for x in pr if x["rows_equal"]),
            "drivers": sorted({s["driver"] for s in sessions}),
            "categories": sorted({s["category"] for s in sessions}),
            "route_groups": len({(s["driver"], s["route_group"]) for s in sessions}),
            "phone_rows_total": int(n_rows.sum()),
            "phone_hours_total": round(sum(p["duration_s"] for p in ph) / 3600, 3),
            "vbox_distance_km_total": round(sum(v["distance_vbox_km"] for v in ve), 2),
            "continuous_segments_total": sum(p["timing"]["n_continuous_segments"] for p in ph),
        },
        "timing_phone": {
            "dt_histogram": sum_dict([p["timing"]["dt_histogram"] for p in ph]),
            "n_backwards": sum(p["timing"]["n_backwards"] for p in ph),
            "n_duplicate_t": sum(p["timing"]["n_duplicate_t"] for p in ph),
            "n_gaps_gt_150ms": sum(p["timing"]["n_gaps_gt_150ms"] for p in ph),
            "n_gaps_gt_1s": sum(p["timing"]["n_gaps_gt_1s"] for p in ph),
            "effective_rate_hz_median": med([p["timing"]["effective_rate_hz"] for p in ph]),
            "date_vs_trel_max_abs_s": max(p["date_vs_trel_max_abs_s"] for p in ph),
        },
        "timing_vehicle": {
            "dt_histogram": sum_dict([v["timing"]["dt_histogram"] for v in ve]),
            "n_backwards": sum(v["timing"]["n_backwards"] for v in ve),
            "n_gaps_gt_1s": sum(v["timing"]["n_gaps_gt_1s"] for v in ve),
        },
        "nan_phone": sum_dict([p["nan"] for p in ph]),
        "nan_vehicle": sum_dict([v["nan"] for v in ve]),
        "parse": {
            "sats_malformed": sum(s["parse_S"]["sats_malformed"] for s in sessions if "parse_S" in s),
            "sats_excel_recovered": sum(s["parse_S"]["sats_excel_recovered"]
                                        for s in sessions if "parse_S" in s),
            "sats_excel_recovered_sessions": sorted(
                s["session_id"] for s in sessions
                if s.get("parse_S", {}).get("sats_excel_recovered")),
            "sats_used_gt_visible": sum(p["gnss"]["n_sats_used_gt_visible"] for p in ph),
            "date_unparsed": sum(s["parse_S"]["date_unparsed"] for s in sessions if "parse_S" in s),
            "utc_lost_at_dst": sum(s["parse_S"]["utc_lost_at_dst"] for s in sessions if "parse_S" in s),
            "coerced_non_numeric_cells": sum(
                sum(s.get(f"parse_{k}", {}).get("coerced_non_numeric", {}).values())
                for s in sessions for k in "SV"),
            "exact_duplicate_rows_phone": sum(p["n_exact_duplicate_rows"] for p in ph),
            "zero_fix_rows_phone": sum(p["gnss"]["n_zero_fix_rows"] for p in ph),
        },
        "gravity": {
            "standard_g0_mps2": io.STANDARD_GRAVITY_MPS2,
            "local_normal_gravity_mps2_range": [float(g_local.min()), float(g_local.max())],
            "channel_norm_pooled_mean_mps2": pooled_g,
            "channel_norm_session_mean_range_mps2": [float(g_means.min()), float(g_means.max())],
            "channel_norm_session_std_max_mps2": float(g_std.max()),
            "sessions_closer_to_g0_than_local": int(np.sum(
                np.abs(g_means - io.STANDARD_GRAVITY_MPS2) < np.abs(g_means - g_local))),
            "sessions_total": len(ph),
            "pooled_minus_g0_mps2": pooled_g - io.STANDARD_GRAVITY_MPS2,
            "pooled_minus_local_mean_mps2": pooled_g - float(np.mean(g_local)),
            "stationary_rows": int(st_n.sum()) if st_n.size else 0,
            "stationary_sessions": int(st_n.size),
            "stationary_acc_norm_weighted_mean_mps2": float(np.sum(st_acc * st_n) / st_n.sum())
            if st_n.size else None,
            "stationary_acc_norm_session_range_mps2": [float(st_acc.min()), float(st_acc.max())]
            if st_n.size else None,
            "stationary_linacc_norm_weighted_mean_mps2": float(np.sum(st_lin * st_n) / st_n.sum())
            if st_n.size else None,
            "dominant_gravity_axis_votes": dict(sum_dict(
                [{p["gravity_channel"]["dominant_axis"]: 1} for p in ph])),
        },
        "gyro_axis_mapping": {
            "reference": "vehicle CAN yaw rate (V-file), TIME-ALIGNED rows with CAN speed > "
                         f"{audit.MOVING_MPS} m/s; votes/medians over sessions with best |corr| >= 0.5",
            "sessions_total": len(pr),
            "sessions_strong": len(strong),
            "best_axis_votes": dict(gyro_votes),
            "best_axis_sign_votes": dict(gyro_signs),
            "median_corr_by_axis": {ax: med(v) for ax, v in corr_by_axis.items()},
            "median_gain_by_axis": {ax: med(v) for ax, v in gain_by_axis.items()},
            "residual_lag_s_median_strong": med([x["gyro_vs_can_yaw_rate"].get("residual_lag_s")
                                                 for x in strong]),
            "residual_lag_s_range_strong": [
                min(x["gyro_vs_can_yaw_rate"]["residual_lag_s"] for x in strong),
                max(x["gyro_vs_can_yaw_rate"]["residual_lag_s"] for x in strong)] if strong else None,
            "gyro_y_vs_yaw_corr_median_row_paired": med(
                [x["row_paired_gyro_y_vs_yaw_corr"] for x in pr]),
            "gyro_y_vs_yaw_corr_median_time_aligned": med(
                [x["gyro_vs_can_yaw_rate"]["corr"]["y"] for x in pr]),
            "linacc_best_axis_votes_vs_can_long": dict(long_votes),
            "linacc_best_axis_votes_vs_can_lat": dict(lat_votes),
        },
        "alignment": {
            "rule": "join S and V on UTC time (phone DATE -> Europe/London -> UTC vs VBOX "
                    "UTC), nearest within 60 ms; verified per session by gyro/yaw-rate xcorr",
            "status_counts": dict(sum_dict([{s["pair"]["alignment"]["status"]: 1}
                                            for s in sessions if "pair" in s])),
            "verified_sessions": sorted(
                s["session_id"] for s in sessions if "pair" in s
                and s["pair"]["alignment"]["status"] == "time_join_verified"),
            "contradicted_sessions": sorted(
                s["session_id"] for s in sessions if "pair" in s
                and s["pair"]["alignment"]["status"] == "time_join_contradicted"),
            "verified_correction_s_range": [
                min((s["pair"]["alignment"]["phone_clock_correction_s"] for s in sessions
                     if s.get("pair", {}).get("alignment", {}).get("status")
                     == "time_join_verified"), default=None),
                max((s["pair"]["alignment"]["phone_clock_correction_s"] for s in sessions
                     if s.get("pair", {}).get("alignment", {}).get("status")
                     == "time_join_verified"), default=None)],
            "stationary_only_sessions": sorted(
                s["session_id"] for s in sessions if "vehicle" in s
                and s["vehicle"]["vbox_speed_mps_max"] < 1.0),
            "sessions_time_match_fraction_min": min(
                x["time_aligned"]["match_fraction"] or 0 for x in pr),
            "sessions_with_abs_row_offset_gt_1s": sorted(
                s["session_id"] for s in sessions if "pair" in s
                and abs(s["pair"]["clock_offset_s"]["median"]) > 1.0),
            "sessions_with_row_offset_spread_gt_1s": sorted(
                s["session_id"] for s in sessions if "pair" in s
                and s["pair"]["clock_offset_s"]["p99"] - s["pair"]["clock_offset_s"]["p01"] > 1.0),
            "sessions_rows_unequal": sorted(
                s["session_id"] for s in sessions if "pair" in s and not s["pair"]["rows_equal"]),
        },
        "clock": {
            "phone_utc_minus_vbox_median_s_range": [
                min(x["clock_offset_s"]["median"] for x in pr),
                max(x["clock_offset_s"]["median"] for x in pr)],
            "max_within_session_spread_s": max(
                x["clock_offset_s"]["p99"] - x["clock_offset_s"]["p01"] for x in pr),
        },
        "unit_evidence": {
            "wheel_raw_over_vbox_kmh_median": med(
                [x["unit_evidence"]["wheel_raw_over_vbox_kmh_median"] for x in pr]),
            "wheel_raw_over_can_speed_kmh_median": med(
                [x["unit_evidence"]["wheel_raw_over_can_speed_kmh_median"] for x in pr]),
            "height_raw_minus_phone_alt_median_m": med(
                [x["unit_evidence"]["height_raw_minus_phone_alt_median"] for x in pr]),
            "dhdt_over_vvel_raw_slope_median": med(
                [x["unit_evidence"]["dhdt_mps_over_vvel_raw_slope"] for x in pr]),
            "phone_gnss_speed_lag_s_median": med(
                [x["unit_evidence"]["phone_gnss_speed_lag_s_vs_vbox"] for x in pr
                 if (x["unit_evidence"]["phone_gnss_speed_corr_at_lag"] or 0) >= 0.8]),
            "phone_gnss_speed_lag_s_range": [
                min(x["unit_evidence"]["phone_gnss_speed_lag_s_vs_vbox"] for x in pr
                    if (x["unit_evidence"]["phone_gnss_speed_corr_at_lag"] or 0) >= 0.8),
                max(x["unit_evidence"]["phone_gnss_speed_lag_s_vs_vbox"] for x in pr
                    if (x["unit_evidence"]["phone_gnss_speed_corr_at_lag"] or 0) >= 0.8)],
            "vbox_height_raw_range": [min(v["channels"]["vbox_height_raw"]["min"] for v in ve),
                                      max(v["channels"]["vbox_height_raw"]["max"] for v in ve)],
            "accel_pedal_raw_range": [min(v["channels"]["accel_pedal_raw"]["min"] for v in ve),
                                      max(v["channels"]["accel_pedal_raw"]["max"] for v in ve)],
            "brake_pressure_pa_min": min(v["channels"]["brake_pressure_pa"]["min"] for v in ve),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()

    if not RAW_ROOT.is_dir():
        print(f"ERROR: {RAW_ROOT} missing; run scripts/preprocess/normalize_raw_layout.py",
              file=sys.stderr)
        return 1
    csvs = sorted(RAW_ROOT.rglob("*.csv"))
    by_session: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    undescribed = []
    for path in csvs:
        try:
            info = io.describe_raw_file(path, RAW_ROOT)
        except io.SchemaError as exc:
            undescribed.append({"rel": path.relative_to(RAW_ROOT).as_posix(),
                                "status": "excluded", "reason": str(exc)})
            continue
        by_session[info.session_key][info.kind].append({
            "path": str(info.path), "rel": info.rel, "collection": info.collection,
            "kind": info.kind, "stem": info.stem, "category": info.category,
            "driver": info.driver,
        })
    print(f"{len(csvs)} CSV files -> {len(by_session)} session keys; "
          f"processing with {args.workers} workers")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {k: ex.submit(process_session, k, {kk: vv for kk, vv in v.items()})
                for k, v in sorted(by_session.items())}
        sessions = [f.result() for f in futs.values()]
    sessions.sort(key=lambda s: s["session_id"])

    file_entries = [f for s in sessions for f in s["files"]] + undescribed
    errors = {s["session_id"]: s["errors"] for s in sessions if s["errors"]}
    complete = [s for s in sessions if "phone" in s and s.get("driver")]

    split = make_split(complete)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "sessions": {
            s["session_id"]: {
                "phone": f"phone/{s['session_id']}.parquet" if "phone" in s else None,
                "vehicle": f"vehicle/{s['session_id']}.parquet" if "vehicle" in s else None,
                "driver": s.get("driver"), "category": s.get("category"),
                "route_group": s["route_group"],
                "phone_rows": s.get("phone", {}).get("n_rows"),
                "alignment_status": s.get("pair", {}).get("alignment", {}).get("status"),
                "phone_clock_correction_s": s.get("pair", {}).get("alignment", {}).get(
                    "phone_clock_correction_s"),
                "vehicle_rows": s.get("vehicle", {}).get("n_rows"),
            } for s in sessions
        },
        "units": io.COLUMN_UNITS,
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = {
        "meta": {
            "title": "IO-VNBD dataset integrity report (Phase 2)",
            "generated": datetime.now().isoformat(timespec="seconds"),
            "script": "scripts/preprocess/ingest_iovnbd.py",
            "commit": git_commit(),
            "python": platform.python_version(),
            "raw_root": RAW_ROOT.relative_to(REPO_ROOT).as_posix(),
            "processed_root": OUT_ROOT.relative_to(REPO_ROOT).as_posix(),
            "schema_version": SCHEMA_VERSION,
            "runtime_s": None,
            "thresholds": {"gap_s": audit.GAP_S, "break_s": audit.BREAK_S,
                           "moving_mps": audit.MOVING_MPS,
                           "stationary_mps": audit.STATIONARY_MPS},
        },
        "summary": aggregate(sessions, file_entries),
        "split": {k: split[k] for k in ("test_drivers", "driver_duration_h", "split_duration_h",
                                        "split_fraction", "splits")},
        "errors": errors,
        "excluded_files": [f for f in file_entries if f.get("status") == "excluded"],
        "sessions": sessions,
    }
    report["meta"]["runtime_s"] = round(time.time() - t0, 1)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "dataset_report.json").write_text(
        json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")

    from scripts.preprocess.dataset_report_html import render  # local import: optional dep
    (REPORT_DIR / "dataset_report.html").write_text(render(report), encoding="utf-8")

    s = report["summary"]
    print(json.dumps({"files": s["files"], "dedupe": s["dedupe"], "sessions": s["sessions"]},
                     indent=1))
    print(f"errors: {errors or 'none'}; runtime {report['meta']['runtime_s']} s")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
