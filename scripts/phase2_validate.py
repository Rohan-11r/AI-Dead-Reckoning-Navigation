#!/usr/bin/env python3
"""Phase 2 validation for SIH26168: dataset ingestion, schema, integrity, split.

Re-derives the Phase 2 claims from the artefacts instead of trusting the report
(AGENTS.md 2.2). Run AFTER the pipeline:

    .venv/Scripts/python.exe scripts/preprocess/normalize_raw_layout.py
    .venv/Scripts/python.exe scripts/preprocess/validate_schema.py
    .venv/Scripts/python.exe scripts/preprocess/ingest_iovnbd.py
    .venv/Scripts/python.exe scripts/phase2_validate.py

Writes reports/phase2_validation.txt. Exit 0 iff all CRITICAL checks pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.preprocessing import iovnbd as io  # noqa: E402

RAW = REPO_ROOT / "data" / "raw" / "IO-VNBD"
PROC = REPO_ROOT / "data" / "processed" / "iovnbd" / "v1"
REPORT = REPO_ROOT / "reports" / "phase2" / "dataset_report.json"
SCHEMA_REPORT = REPO_ROOT / "reports" / "phase2" / "schema_validation.json"
HTML = REPO_ROOT / "reports" / "phase2" / "dataset_report.html"
SPLIT = REPO_ROOT / "data" / "splits" / "iovnbd_split_v1.json"
OUT = REPO_ROOT / "reports" / "phase2_validation.txt"

CRITICAL, INFO = "CRITICAL", "INFO"
results: list[tuple[str, str, str, str]] = []


def record(sev: str, name: str, ok: bool | None, detail: str) -> None:
    results.append((sev, name, "OBSERVED" if ok is None else ("PASS" if ok else "FAIL"), detail))


def safe(sev: str, name: str):
    def wrap(fn):
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:  # noqa: BLE001 - report, never hide
                record(sev, name, False, f"raised {type(exc).__name__}: {exc}")
                return None
        return inner
    return wrap


@safe(CRITICAL, "raw layout")
def check_layout():
    legacy = [p for p in (REPO_ROOT / "data" / "raw").iterdir()
              if p.is_dir() and p.name != "IO-VNBD"]
    record(CRITICAL, "raw root normalised to data/raw/IO-VNBD", RAW.is_dir() and not legacy,
           f"exists={RAW.is_dir()}, other dirs={[p.name for p in legacy]}")
    n_csv = len(list(RAW.rglob("*.csv")))
    n_jpg = len([p for p in RAW.rglob("*") if p.suffix.lower() == ".jpg"])
    record(CRITICAL, "raw file inventory 288 CSV + 72 JPG", n_csv == 288 and n_jpg == 72,
           f"{n_csv} CSV, {n_jpg} JPG")


@safe(CRITICAL, "schema validation")
def check_schema():
    sv = json.loads(SCHEMA_REPORT.read_text(encoding="utf-8"))
    record(CRITICAL, "every raw file passes the schema validator",
           sv["n_ok"] == sv["n_files"] == 288, f"{sv['n_ok']}/{sv['n_files']} ok, "
           f"{sv['total_data_rows']:,} data rows")
    return {f["rel"]: f["n_data_rows"] for f in sv["files"]}


@safe(CRITICAL, "dataset report")
def check_report():
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    s = r["summary"]
    st = s["files"]["by_status"]
    record(CRITICAL, "288 files accounted for: 144 canonical + 144 duplicate, 0 excluded",
           st.get("canonical") == 144 and st.get("duplicate") == 144 and not st.get("excluded"),
           json.dumps(st))
    d = s["dedupe"]
    record(CRITICAL, "V pairs byte-identical (SHA-256)", d["byte_identical_by_kind"]["V"] == 72,
           f"{d['byte_identical_by_kind']['V']}/72")
    record(CRITICAL, "S pairs content-identical (differ only in float text + labels)",
           d["content_identical_by_kind"]["S"] == 72 and d["max_abs_numeric_diff_overall"] < 1e-12,
           f"{d['content_identical_by_kind']['S']}/72, max |diff| {d['max_abs_numeric_diff_overall']:.2e}, "
           f"byte-identical S pairs: {d['byte_identical_by_kind']['S']}")
    record(CRITICAL, "no content conflicts between copies", not d["conflicts"], str(d["conflicts"]))
    record(CRITICAL, "unique sessions == 72, all with phone + vehicle log",
           s["sessions"]["unique"] == 72 and s["sessions"]["with_both_logs"] == 72,
           f"{s['sessions']['unique']} sessions")
    p = s["parse"]
    record(CRITICAL, "DATE parsed with explicit format: 0 failures", p["date_unparsed"] == 0,
           f"unparsed={p['date_unparsed']}, lost at DST={p['utc_lost_at_dst']}")
    record(CRITICAL, "satellites: 0 malformed, used <= visible everywhere",
           p["sats_malformed"] == 0 and p["sats_used_gt_visible"] == 0,
           f"excel-repaired {p['sats_excel_recovered']}, malformed {p['sats_malformed']}, "
           f"used>visible {p['sats_used_gt_visible']}")
    record(CRITICAL, "both S header variants seen and mapped",
           s["files"]["header_variants"].get("Yaw/Pitch/Roll") == 72
           and s["files"]["header_variants"].get("XYZ/Azimuth") == 72,
           json.dumps(s["files"]["header_variants"]))
    g = s["gravity"]
    record(CRITICAL, "GRAVITY channel normalised to g0 in every session",
           g["sessions_closer_to_g0_than_local"] == g["sessions_total"] == 72
           and abs(g["pooled_minus_g0_mps2"]) < 1e-3,
           f"pooled |g| {g['channel_norm_pooled_mean_mps2']:.6f}; closer to g0 in "
           f"{g['sessions_closer_to_g0_than_local']}/{g['sessions_total']}")
    record(CRITICAL, "accelerometer is gravity-inclusive (stationary |a| ~ g)",
           g["stationary_rows"] > 1000 and abs(g["stationary_acc_norm_weighted_mean_mps2"] - 9.81) < 0.5,
           f"{g['stationary_acc_norm_weighted_mean_mps2']:.4f} m/s^2 over {g['stationary_rows']:,} rows")
    al = s["alignment"]
    record(CRITICAL, "UTC-time join never contradicted by gyro/yaw-rate evidence",
           not al["contradicted_sessions"], f"status {json.dumps(al['status_counts'])}")
    ga = s["gyro_axis_mapping"]
    record(INFO, "gyro vertical axis vs CAN yaw rate", None,
           f"votes {ga['best_axis_sign_votes']} over {ga['sessions_strong']} strong sessions; "
           f"gain(y) {ga['median_gain_by_axis']['y']:.4f}")
    record(INFO, "row-pairing offset > 1 s", None,
           f"{len(al['sessions_with_abs_row_offset_gt_1s'])} sessions")
    ue = s["unit_evidence"]
    record(INFO, "unit evidence", None,
           f"height-alt {ue['height_raw_minus_phone_alt_median_m']:.1f} m; dh/dt/vvel "
           f"{ue['dhdt_over_vvel_raw_slope_median']:.3f}; wheel/vbox_kmh "
           f"{ue['wheel_raw_over_vbox_kmh_median']:.4f}; GNSS lag "
           f"{ue['phone_gnss_speed_lag_s_median']} s")
    return r


@safe(CRITICAL, "processed parquet")
def check_parquet(raw_rows: dict[str, int]):
    phone_cols = ["row_idx", "t_rel_s", "t_local", "t_utc", "gps_lat_deg", "gps_lon_deg",
                  "gps_alt_m", "gps_speed_mps", "gps_accuracy_m", "gps_bearing_deg",
                  "gps_sats_used", "gps_sats_visible", "gps_sats_excel_recovered"] + [
        c for c, _ in io.PHONE_SCHEMA if c.startswith(("acc_", "grav_", "gyro_", "mag_", "orient_"))]
    ph = sorted((PROC / "phone").glob("*.parquet"))
    ve = sorted((PROC / "vehicle").glob("*.parquet"))
    record(CRITICAL, "72 phone + 72 vehicle parquet files", len(ph) == 72 and len(ve) == 72,
           f"{len(ph)} phone, {len(ve)} vehicle")
    bad_schema, bad_rows, bad_units, bad_hash, bad_range = [], [], [], [], []
    for f in ph + ve:
        pf = pq.ParquetFile(f)
        meta = json.loads(pf.schema_arrow.metadata[b"sih26168"])
        cols = pf.schema_arrow.names
        if f.parent.name == "phone" and cols != phone_cols:
            bad_schema.append(f.name)
        if f.parent.name == "vehicle" and "t_utc" not in cols:
            bad_schema.append(f.name)
        if set(meta["units"]) != set(cols) or "?" in meta["units"].values():
            bad_units.append(f.name)
        src = RAW / meta["source"]
        if pf.metadata.num_rows != raw_rows.get(meta["source"]):
            bad_rows.append(f"{f.name}: {pf.metadata.num_rows} vs raw {raw_rows.get(meta['source'])}")
        if io.sha256_file(src) != meta["source_sha256"]:
            bad_hash.append(f.name)
        if f.parent.name == "phone":
            t = pf.read(columns=["gps_speed_mps", "grav_x_mps2", "grav_y_mps2", "grav_z_mps2"]).to_pandas()
            gn = np.sqrt(t.grav_x_mps2 ** 2 + t.grav_y_mps2 ** 2 + t.grav_z_mps2 ** 2)
            if t.gps_speed_mps.max() > 70 or abs(gn.mean() - 9.80665) > 0.01:
                bad_range.append(f.name)
    record(CRITICAL, "phone schema exact; vehicle has t_utc", not bad_schema, str(bad_schema[:5]))
    record(CRITICAL, "every column carries a unit in parquet metadata", not bad_units, str(bad_units[:5]))
    record(CRITICAL, "parquet row count == raw data-row count (independent line count)",
           not bad_rows, str(bad_rows[:5]))
    record(CRITICAL, "raw sources unchanged since ingest (SHA-256 in parquet metadata)",
           not bad_hash, f"{len(ph) + len(ve) - len(bad_hash)}/{len(ph) + len(ve)} match")
    record(CRITICAL, "SI sanity: GPS speed < 70 m/s, |gravity| ~ 9.807", not bad_range, str(bad_range[:5]))


@safe(CRITICAL, "split")
def check_split(report: dict):
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    all_ids = {s["session_id"] for s in report["sessions"]}
    seen = [x for ids in sp["splits"].values() for x in ids]
    record(CRITICAL, "every session in exactly one split", len(seen) == len(set(seen)) and set(seen) == all_ids,
           f"{len(seen)} assignments, {len(set(seen))} unique, {len(all_ids)} sessions")
    grp: dict[tuple, set] = {}
    drv: dict[str, set] = {}
    for sid, m in sp["sessions"].items():
        grp.setdefault((m["driver"], m["route_group"]), set()).add(m["split"])
        drv.setdefault(m["driver"], set()).add(m["split"])
    split_groups = [k for k, v in grp.items() if len(v) > 1]
    record(CRITICAL, "no route group spans two splits", not split_groups, str(split_groups))
    leak = [d for d, v in drv.items() if "test" in v and len(v) > 1]
    record(CRITICAL, "test drivers are held out entirely", not leak and bool(sp["test_drivers"]),
           f"test drivers {sp['test_drivers']}, leaking {leak}")
    lo, hi = sp["test_fraction_bounds"]
    record(CRITICAL, "test fraction within bounds", lo <= sp["split_fraction"]["test"] <= hi,
           f"fractions {sp['split_fraction']}, hours {sp['split_duration_h']}")


@safe(CRITICAL, "unit tests")
def check_tests():
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/unit/test_iovnbd_loader.py", "-q"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["?"])[-1]
    record(CRITICAL, "loader unit tests (encoding, headers, sats, datetime, dedupe)", r.returncode == 0, tail)


@safe(CRITICAL, "html report")
def check_html():
    t = HTML.read_text(encoding="utf-8")
    record(CRITICAL, "dataset_report.html rendered", "<h1>IO-VNBD dataset integrity report" in t
           and t.rstrip().endswith("</html>"), f"{len(t):,} bytes")


def main() -> int:
    check_layout()
    raw_rows = check_schema() or {}
    report = check_report()
    check_parquet(raw_rows)
    if report:
        check_split(report)
    check_tests()
    check_html()
    crit = [r for r in results if r[0] == CRITICAL]
    passed = sum(1 for r in crit if r[2] == "PASS")
    lines = ["=" * 100, "SIH26168 -- PHASE 2 DATASET INGESTION & INTEGRITY VALIDATION",
             f"generated: {datetime.now().isoformat(timespec='seconds')}",
             "script:    scripts/phase2_validate.py", "=" * 100, ""]
    lines += [f"  {sev:<9} {st:<9} {name:<66} {detail}" for sev, name, st, detail in results]
    lines += ["", "-" * 100, f"CRITICAL checks: {passed}/{len(crit)} passed",
              f"RESULT: {'PASS' if passed == len(crit) else 'FAIL'}", "-" * 100]
    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0 if passed == len(crit) else 1


if __name__ == "__main__":
    sys.exit(main())
