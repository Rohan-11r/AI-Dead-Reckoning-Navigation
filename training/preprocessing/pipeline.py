"""Phase 3 preprocessing pipeline -- one reproducible command.

    .venv/Scripts/python.exe -m training.preprocessing.pipeline [--workers 8]

Steps (inputs: the Phase 2 Parquet in data/processed/iovnbd/v1):
  1. Wheel-speed calibration      -> reports/phase3/wheel_speed_calibration.json
  2. Phone GNSS fix delay, pass 1 (new-fix rows only; pooled median of reliable sessions)
  3. UTC-time sync, pass 2        -> data/processed/iovnbd/v1/synced/<session>.parquet
  4. Drive-level split v2         -> data/splits/iovnbd_split_v2.json (+ leak audit)
  5. Train-only IMU normalisation -> models/normalization/imu_{mean,std}.json
  6. Report                       -> reports/phase3/preprocessing_report.json

Deterministic: the only randomness is the split's seeded shuffle.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from training.preprocessing import normalization, splits, sync
from training.preprocessing.columns import GT_COLUMNS, INPUT_COLUMNS, META_COLUMNS
from training.preprocessing.config import REPO_ROOT, DatasetConfig, load_dataset_config
from training.preprocessing.wheel_speed import calibrate

SCRIPT = "training/preprocessing/pipeline.py"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load(cfg: DatasetConfig, sid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(cfg.processed_root / "phone" / f"{sid}.parquet"),
            pd.read_parquet(cfg.processed_root / "vehicle" / f"{sid}.parquet"))


def _measure_worker(args) -> tuple[str, float | None, float | None, float | None]:
    cfg, sid, align = args
    p, v = _load(cfg, sid)
    info = sync.SyncInfo(session_id=sid)
    p = sync.prepare_phone(p, info)
    verified = align.get("alignment_status") == "time_join_verified"
    c = float(align["phone_clock_correction_s"]) if verified else 0.0
    delay, err, interval, _ = sync.measure_fix_delay(p, p["t_utc"] + pd.Timedelta(seconds=c), v)
    return sid, delay, err, interval


def _sync_worker(args) -> dict:
    cfg, sid, align, k, default_lat, meta = args
    p, v = _load(cfg, sid)
    df, info = sync.sync_session(sid, p, v, align, k, default_lat)
    cols = META_COLUMNS + INPUT_COLUMNS + GT_COLUMNS
    missing = set(cols) ^ set(df.columns)
    if missing:
        raise RuntimeError(f"{sid}: synced columns disagree with the declared roles: {missing}")
    df = df[cols]
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md[b"sih26168"] = json.dumps({
        "schema_version": sync.SCHEMA_VERSION, "session_id": sid,
        "units": {c: sync.SYNCED_UNITS[c] for c in df.columns},
        "input_columns": INPUT_COLUMNS, "ground_truth_columns": GT_COLUMNS,
        "rule": "gt_* columns are labels/evaluation only -- never model or filter inputs",
        "sync": asdict(info), **meta,
    }, default=str).encode()
    out = cfg.synced_root / f"{sid}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(md), out, compression="zstd")
    t = df["t_utc"]
    return {**asdict(info), "start_utc": str(t.min()), "end_utc": str(t.max()),
            "duration_s": float(np.nansum(df["dt_s"].to_numpy())),
            "gt_stationary_rows": int(df["gt_stationary"].sum())}


def v1_leaks_under_drive_rule(v1_path: Path, table: pd.DataFrame) -> list[str]:
    """Re-audit the Phase 2 v1 manifest with the Phase 3 drive-adjacency rule."""
    if not v1_path.exists():
        return ["v1 manifest not found"]
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    tab = table.set_index("session_id")
    sessions = {sid: {"split": m["split"], "driver": m["driver"], "route_group": m["route_group"],
                      "drive_id": m["route_group"], "start_utc": tab.loc[sid, "start_utc"],
                      "end_utc": tab.loc[sid, "end_utc"]} for sid, m in v1["sessions"].items()}
    return splits.find_leaks({"splits": v1["splits"], "sessions": sessions})


def verify_phone_gps_speed_unit(cfg: DatasetConfig, infos: list[dict]) -> dict:
    """Evidence for the phone GPS speed unit: the header says "Kmh", Android reports m/s.

    Compares the synced phone speed (value as logged, NOT divided by 3.6) with VBOX
    Doppler speed at the fix's own epoch. Ratio ~1 => m/s; the km/h label would give ~3.6.
    """
    ratios = []
    for i in infos:
        if not i["gnss_fix_delay_source"].startswith("measured"):
            continue
        d = pd.read_parquet(cfg.synced_root / f"{i['session_id']}.parquet",
                            columns=["t_utc", "ph_gnss_new_fix", "ph_gnss_epoch_utc",
                                     "ph_gnss_speed_mps", "gt_speed_mps"])
        t0 = d["t_utc"].iloc[0]
        tt = (d["t_utc"] - t0).dt.total_seconds().to_numpy()
        gt = d["gt_speed_mps"].to_numpy()
        ok_gt = np.isfinite(gt)
        f = d[d["ph_gnss_new_fix"]]
        te = (f["ph_gnss_epoch_utc"] - t0).dt.total_seconds().to_numpy()
        a = f["ph_gnss_speed_mps"].to_numpy()
        b = np.interp(te, tt[ok_gt], gt[ok_gt])
        m = np.isfinite(a) & np.isfinite(b) & (b > 5.0)
        if m.sum() > 20:
            ratios.append(float(np.median(a[m] / b[m])))
    r = np.array(ratios)
    out = {
        "generated": datetime.now().isoformat(timespec="seconds"), "script": SCRIPT,
        "question": "unit of the phone 'GPS SPEED (Kmh)' column",
        "method": "median(logged phone speed / VBOX Doppler speed [m/s]) at each NEW fix's "
                  "epoch, VBOX speed > 5 m/s, sessions with a measured fix delay",
        "n_sessions": int(r.size),
        "ratio_median": float(np.median(r)),
        "ratio_iqr": [float(np.percentile(r, 25)), float(np.percentile(r, 75))],
        "expected_if_mps": 1.0, "expected_if_kmh": 3.6,
        "conclusion": "m/s -- the 'Kmh' header is wrong (Android Location.getSpeed returns m/s). "
                      "Phases 0-2 divided the value by 3.6; fixed in Phase 3.",
    }
    p = cfg.reports_dir / "phase3" / "phone_gps_speed_unit.json"
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def run(workers: int = 8) -> dict:
    t0 = time.time()
    cfg = load_dataset_config()
    p2 = json.loads((cfg.processed_root / "manifest.json").read_text(encoding="utf-8"))["sessions"]
    sids = sorted(p2)

    print("[1/6] wheel-speed calibration")
    wheel = calibrate(cfg, verbose=False)
    k = wheel["calibration"]["k_rear_raw_per_kmh"]

    print("[2/6] phone GNSS fix delay (pass 1)")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        lat = list(ex.map(_measure_worker, [(cfg, s, p2[s]) for s in sids]))
    reliable = [x[1] for x in lat if x[1] is not None]
    pooled = float(np.median(reliable)) if reliable else None

    print(f"[3/6] sync {len(sids)} sessions (pooled fix delay {pooled} s from "
          f"{len(reliable)} sessions)")
    meta = {"wheel_k_raw_per_kmh": k, "pooled_gnss_fix_delay_s": pooled,
            "script": SCRIPT, "commit": git_commit()}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        infos = list(ex.map(_sync_worker, [(cfg, s, p2[s], k, pooled, meta) for s in sids]))

    gps_unit = verify_phone_gps_speed_unit(cfg, infos)
    if not 0.95 < gps_unit["ratio_median"] < 1.05:
        raise RuntimeError(f"phone GPS speed unit check failed: {gps_unit['ratio_median']}")

    print("[4/6] drive-level split v2")
    table = pd.DataFrame([{
        "session_id": i["session_id"], "driver": p2[i["session_id"]]["driver"],
        "route_group": p2[i["session_id"]]["route_group"],
        "start_utc": pd.Timestamp(i["start_utc"]), "end_utc": pd.Timestamp(i["end_utc"]),
        "duration_s": i["duration_s"]} for i in infos])
    manifest = splits.make_split(table)
    manifest.update(generated=datetime.now().isoformat(timespec="seconds"), script=SCRIPT,
                    supersedes="iovnbd_split_v1.json (Phase 2; violates the drive-adjacency rule)")
    leaks = splits.find_leaks(manifest)
    if leaks:
        raise RuntimeError(f"split v2 failed its own leak audit: {leaks[:5]}")
    cfg.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    cfg.split_manifest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    v1_leaks = v1_leaks_under_drive_rule(cfg.splits_dir / "iovnbd_split_v1.json", table)

    print("[5/6] train-only IMU normalisation")
    stats = normalization.compute_imu_stats(cfg.split_manifest, cfg.synced_root)
    stats["split_manifest"] = cfg.split_manifest.relative_to(REPO_ROOT).as_posix()
    mean_p, std_p = normalization.write_stats(
        stats, cfg.normalization_dir, {"script": SCRIPT, "commit": git_commit()})

    print("[6/6] report")
    sess_df = pd.DataFrame(infos)
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"), "script": SCRIPT,
        "commit": git_commit(), "python": platform.python_version(),
        "runtime_s": round(time.time() - t0, 1),
        "wheel_speed": {k2: wheel[k2] for k2 in ("n_sessions", "n_clean_samples", "driven_axle",
                                                  "calibration", "unit_decision")},
        "gnss_fix_timing": {
            "reliable_sessions": len(reliable), "pooled_median_delay_s": pooled,
            "measured_delay_range_s": [min(reliable), max(reliable)] if reliable else None,
            "measured_delay_iqr_s": [float(np.percentile(reliable, 25)),
                                     float(np.percentile(reliable, 75))] if reliable else None,
            "source_counts": sess_df["gnss_fix_delay_source"].value_counts().to_dict(),
            "fix_interval_median_s_by_session": {
                r.session_id: r.gnss_fix_interval_median_s for r in sess_df.itertuples()},
            "sessions_fix_interval_le_2s": int((sess_df["gnss_fix_interval_median_s"] <= 2).sum()),
            "rule": "delay of a NEW fix vs VBOX speed, grid "
                    f"{list(sync.FIX_DELAY_GRID_S)} s, accepted if median |speed err| <= "
                    f"{sync.FIX_DELAY_MAX_MEDIAN_ERR_MPS} m/s over >= {sync.FIX_DELAY_MIN_FIXES} fixes",
        },
        "phone_gps_speed_unit": gps_unit,
        "sync_totals": {
            "sessions": len(sess_df), "rows_in": int(sess_df.rows_in.sum()), "rows_out": int(sess_df.rows_out.sum()),
            "dropped_no_time": int(sess_df.dropped_no_time.sum()),
            "dropped_exact_duplicates": int(sess_df.dropped_exact_duplicates.sum()),
            "reordered_rows": int(sess_df.reordered_rows.sum()),
            "duplicate_timestamps_kept": int(sess_df.duplicate_timestamps_kept.sum()),
            "gt_matched": int(sess_df.gt_matched.sum()),
            "gt_match_fraction": float(sess_df.gt_matched.sum() / sess_df.rows_out.sum()),
            "segments": int(sess_df.n_segments.sum()),
            "alignment_verified_sessions": int(sess_df.alignment_verified.sum()),
            "clock_correction_applied_range_s": [float(sess_df.clock_correction_s.min()),
                                                 float(sess_df.clock_correction_s.max())],
            "gt_stationary_rows": int(sess_df.gt_stationary_rows.sum()),
            "hours": round(float(sess_df.duration_s.sum()) / 3600, 3),
        },
        "split": {k2: manifest[k2] for k2 in ("split_version", "unit", "test_drivers",
                                               "split_duration_h", "split_fraction")},
        "split_n_sessions": {n: len(v) for n, v in manifest["splits"].items()},
        "split_n_drives": {n: sum(1 for d in manifest["drives"].values() if d["split"] == n)
                           for n in ("train", "val", "test")},
        "split_v2_leaks": leaks,
        "split_v1_leaks_under_drive_rule": v1_leaks,
        "normalization": {"files": [mean_p.relative_to(REPO_ROOT).as_posix(),
                                    std_p.relative_to(REPO_ROOT).as_posix()],
                          "sessions_used": len(stats["sessions_used"]),
                          "n_samples": stats["n_samples"], "mean": stats["mean"], "std": stats["std"]},
        "sessions": infos,
    }
    out = cfg.reports_dir / "phase3" / "preprocessing_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k2: report[k2] for k2 in ("gnss_fix_timing", "sync_totals", "split",
                                                  "split_n_drives")}, indent=1, default=str))
    print(f"v1 leaks under the drive rule: {len(v1_leaks)}; v2 leaks: {len(leaks)}; "
          f"runtime {report['runtime_s']} s")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 preprocessing pipeline")
    ap.add_argument("--workers", type=int, default=8)
    run(ap.parse_args().workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
