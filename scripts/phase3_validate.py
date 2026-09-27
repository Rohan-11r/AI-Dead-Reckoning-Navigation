#!/usr/bin/env python3
"""Phase 3 validation: preprocessing pipeline, coordinate frames, splits, normalisation.

Re-derives the Phase 3 claims from the artefacts rather than trusting the pipeline's own
report (AGENTS.md 2.2). Run after:

    .venv/Scripts/python.exe -m training.preprocessing.pipeline
    .venv/Scripts/python.exe scripts/phase3_validate.py

Writes reports/phase3_validation.txt. Exit 0 iff every CRITICAL check passes.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.preprocessing import columns, normalization, splits  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
PY = sys.executable
OUT = CFG.reports_dir / "phase3_validation.txt"
REPORT = CFG.reports_dir / "phase3" / "preprocessing_report.json"
WHEEL = CFG.reports_dir / "phase3" / "wheel_speed_calibration.json"
# the Phase 3 geometry suite (the INS/EKF suites added in Phase 4 are slow and gated there)
GEOMETRY_TESTS = [f"tests/navigation/test_{m}.py" for m in ("constants", "quaternion", "rotation", "geodesy", "frames")]

CRITICAL, INFO = "CRITICAL", "INFO"
results: list[tuple[str, str, str, str]] = []


def record(sev: str, name: str, ok: bool | None, detail: str) -> None:
    results.append((sev, name, "OBSERVED" if ok is None else ("PASS" if ok else "FAIL"), detail))


def safe(sev: str, name: str):
    def wrap(fn):
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:  # report, never hide
                record(sev, name, False, f"raised {type(exc).__name__}: {exc}")
                return None
        return inner
    return wrap


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONUTF8": "1"})


# ---- housekeeping -------------------------------------------------------------------------

@safe(CRITICAL, "tooling")
def check_tooling():
    pin = re.search(r"^ruff==(\S+)$", (REPO_ROOT / "requirements-dev.txt").read_text(), re.M)
    installed = md.version("ruff")
    record(CRITICAL, "ruff pinned in requirements-dev.txt and installed at that version",
           bool(pin) and pin.group(1) == installed, f"pinned {pin and pin.group(1)}, installed {installed}")
    r = run([PY, "-m", "ruff", "check", "."])
    record(CRITICAL, "ruff check . is clean", r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1])
    r = run([PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"])
    record(CRITICAL, "full test suite passes", r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1])


@safe(CRITICAL, "config")
def check_config():
    record(CRITICAL, "configs/dataset.yaml loads; raw root exists",
           CFG.raw_root.is_dir(), f"raw_root={CFG.raw_root.relative_to(REPO_ROOT).as_posix()}")
    # path CONSTRUCTION in code; docstrings/comments that merely mention a path are fine
    pat = re.compile(r'"data"\s*/\s*"(raw|processed|splits)"'
                     r"""|Path\(\s*r?["'][^"']*(data[/\\]raw|SIH26168)""")
    offenders = [str(p.relative_to(REPO_ROOT)) for d in ("scripts", "training", "navigation-core")
                 for p in (REPO_ROOT / d).rglob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    record(CRITICAL, "no hardcoded dataset paths in scripts/, training/, navigation-core/",
           not offenders, str(offenders))


# ---- geometry -----------------------------------------------------------------------------

@safe(CRITICAL, "geometry mutation check")
def check_mutations():
    """The navigation math suite must FAIL when a classic convention bug is injected."""
    plugin = (
        "import os\nimport numpy as np\nfrom navcore.geometry import quaternion as Q, geodesy as G\n"
        "M = os.environ.get('MUT')\n_mul, _rot, _R, _g2e = Q.q_mul, Q.q_rotate, G.R_ecef_to_enu, G.geodetic_to_ecef\n"
        "if M == 'jpl_product': Q.q_mul = lambda p, q: _mul(q, p)\n"
        "if M == 'passive_rotation': Q.q_rotate = lambda q, v: _rot(Q.q_conj(q), v)\n"
        "if M == 'enu_transposed': G.R_ecef_to_enu = lambda a, b: np.swapaxes(_R(a, b), -1, -2)\n"
        "if M == 'no_small_angle_branch': Q.SMALL_ANGLE_THRESHOLD_RAD = -1.0\n"
        "if M == 'height_ignored': G.geodetic_to_ecef = lambda la, lo, h: _g2e(la, lo, 0.0 * np.asarray(h))\n")
    caught = {}
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "sih_mutplug.py").write_text(plugin)
        import os
        for m in ("jpl_product", "passive_rotation", "enu_transposed", "no_small_angle_branch",
                  "height_ignored"):
            r = subprocess.run([PY, "-m", "pytest", *GEOMETRY_TESTS, "-q", "-p", "sih_mutplug",
                                "-p", "no:cacheprovider"], cwd=REPO_ROOT, capture_output=True, text=True,
                               env={**os.environ, "MUT": m, "PYTHONPATH": td, "PYTHONUTF8": "1"})
            caught[m] = r.returncode != 0
    record(CRITICAL, "navigation tests catch 5/5 injected convention bugs", all(caught.values()),
           json.dumps(caught))


# ---- wheel speed ----------------------------------------------------------------------------

@safe(CRITICAL, "wheel speed")
def check_wheel():
    w = json.loads(WHEEL.read_text(encoding="utf-8"))
    c = w["calibration"]
    ok = (0.98 < c["k_rear_raw_per_kmh"] < 1.02 and abs(c["resid_vs_vbox_median_mps"]) < 0.05
          and w["driven_axle"]["fraction_sessions_positive"] > 0.9)
    record(CRITICAL, "wheel-speed calibration: rear-axle factor, residual, driven axle", ok,
           f"k={c['k_rear_raw_per_kmh']:.5f} raw/(km/h); resid median {c['resid_vs_vbox_median_mps']:+.4f} "
           f"p95|.| {c['resid_vs_vbox_p95_abs_mps']:.3f} m/s over {w['n_clean_samples']:,} samples; "
           f"FWD evidence {w['driven_axle']['fraction_sessions_positive']:.2f}")
    record(INFO, "wheel-speed unit decision", None,
           f"favoured {w['unit_decision']['favoured']}; identifiable={w['unit_decision']['identifiable_from_data']}; "
           f"implied r under rad/s {w['unit_decision']['implied_rolling_radius_under_rad_s_m']:.5f} m")


# ---- synced data -----------------------------------------------------------------------------

def smooth(a: np.ndarray, k: int = 10) -> np.ndarray:
    return pd.Series(a).rolling(k, center=True, min_periods=1).mean().to_numpy()


@safe(CRITICAL, "synced data")
def check_synced(rep: dict):
    files = sorted(CFG.synced_root.glob("*.parquet"))
    record(CRITICAL, "72 synced session files", len(files) == 72, f"{len(files)} files")
    expected = columns.META_COLUMNS + columns.INPUT_COLUMNS + columns.GT_COLUMNS
    by_sid = {s["session_id"]: s for s in rep["sessions"]}
    bad_cols, bad_roles, bad_rows, bad_dt = [], [], [], []
    yaw_corr, gnss_gain = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        meta = json.loads(pf.schema_arrow.metadata[b"sih26168"])
        if pf.schema_arrow.names != expected:
            bad_cols.append(f.stem)
        try:
            columns.assert_model_inputs(meta["input_columns"])
            if any(not c.startswith("gt_") for c in meta["ground_truth_columns"]):
                raise columns.ColumnRoleError("unprefixed ground truth")
        except columns.ColumnRoleError:
            bad_roles.append(f.stem)
        s = by_sid[f.stem]
        if not (s["rows_out"] == pf.metadata.num_rows == s["rows_in"] - s["dropped_no_time"]
                - s["dropped_exact_duplicates"]):
            bad_rows.append(f.stem)
        d = pf.read(columns=["dt_s", "t_utc", "ph_gnss_epoch_utc", "ph_gnss_new_fix",
                             "ph_gnss_fix_age_s", "gyro_y_radps", "gt_yaw_rate_radps",
                             "gt_can_speed_mps", "ph_gnss_speed_mps", "gt_speed_mps"]).to_pandas()
        dt = d["dt_s"].dropna()
        if len(dt) and (dt.min() < 0 or dt.max() > 1.0):
            bad_dt.append(f.stem)
        if s["gnss_fix_delay_s"] is not None:
            age = d["ph_gnss_fix_age_s"].to_numpy()
            new = d["ph_gnss_new_fix"].to_numpy()
            # a new fix's age equals the delay; held rows age by exactly dt until the next fix
            if not (np.allclose(age[new], s["gnss_fix_delay_s"]) and np.nanmin(age) >= s["gnss_fix_delay_s"] - 1e-9):
                bad_dt.append(f"{f.stem}:epoch")
        mv = np.nan_to_num(d["gt_can_speed_mps"].to_numpy()) > 3.0
        if s["alignment_verified"] and mv.sum() > 200:
            yaw_corr.append(np.corrcoef(smooth(d["gyro_y_radps"].to_numpy())[mv],
                                        smooth(np.nan_to_num(d["gt_yaw_rate_radps"].to_numpy()))[mv])[0, 1])
        if s["gnss_fix_delay_source"].startswith("measured"):
            # epoch tagging must IMPROVE agreement with truth: every row's held fix vs truth
            # at the row's own time (what naive use would assume) vs truth at the fix epoch
            t0 = d["t_utc"].iloc[0]
            tt = (d["t_utc"] - t0).dt.total_seconds().to_numpy()
            te = (d["ph_gnss_epoch_utc"] - t0).dt.total_seconds().to_numpy()
            ph, gt = d["ph_gnss_speed_mps"].to_numpy(), d["gt_speed_mps"].to_numpy()
            ok = np.isfinite(ph) & np.isfinite(gt) & np.isfinite(te)
            if ok.sum() > 600:
                same = np.median(np.abs(ph - gt)[ok])
                at_epoch = np.median(np.abs(ph[ok] - np.interp(te[ok], tt[ok], gt[ok])))
                gnss_gain.append((same, at_epoch))
    record(CRITICAL, "synced columns are exactly META + INPUT + GT", not bad_cols, str(bad_cols[:5]))
    record(CRITICAL, "input/ground-truth roles in file metadata obey the CAN rule", not bad_roles,
           str(bad_roles[:5]))
    record(CRITICAL, "row accounting: out = in - no-time - exact duplicates (and = parquet rows)",
           not bad_rows, str(bad_rows[:5]))
    record(CRITICAL, "real dt inside segments in (0, 1] s; fix age = delay at a new fix, grows while held", not bad_dt,
           str(bad_dt[:5]))
    t = rep["sync_totals"]
    record(CRITICAL, "ground truth matched for >= 95 % of phone rows", t["gt_match_fraction"] >= 0.95,
           f"{t['gt_match_fraction']:.4f} ({t['gt_matched']:,}/{t['rows_out']:,})")
    ymed = float(np.median(yaw_corr)) if yaw_corr else float("nan")
    record(CRITICAL, "time join verified: gyro_y vs CAN yaw rate (recomputed, verified sessions)",
           len(yaw_corr) >= 20 and ymed >= 0.5, f"median r {ymed:.3f} over {len(yaw_corr)} sessions")
    g = np.array(gnss_gain)
    improved = int((g[:, 1] < g[:, 0]).sum()) if g.size else 0
    record(CRITICAL, "fix-epoch tagging improves phone-vs-truth speed agreement",
           g.size > 0 and improved >= 0.9 * len(g),
           f"{improved}/{len(g)} sessions; median |err| {np.median(g[:, 0]):.3f} -> {np.median(g[:, 1]):.3f} m/s"
           if g.size else "no measured sessions")


# ---- split & normalisation -------------------------------------------------------------------

@safe(CRITICAL, "split v2")
def check_split():
    m = json.loads(CFG.split_manifest.read_text(encoding="utf-8"))
    leaks = splits.find_leaks(m)
    record(CRITICAL, "split v2 passes the independent leak audit (incl. temporal adjacency)",
           not leaks, f"{len(leaks)} violations" + (f": {leaks[:3]}" if leaks else ""))
    synced = {f.stem for f in CFG.synced_root.glob("*.parquet")}
    listed = [x for ids in m["splits"].values() for x in ids]
    record(CRITICAL, "every synced session assigned exactly once", sorted(listed) == sorted(synced),
           f"{len(listed)} assigned, {len(synced)} synced")
    record(INFO, "split v2", None, f"test drivers {m['test_drivers']}; hours {m['split_duration_h']}; "
           f"fractions {m['split_fraction']}; drives "
           f"{ {n: sum(d['split'] == n for d in m['drives'].values()) for n in ('train', 'val', 'test')} }")


@safe(CRITICAL, "normalisation")
def check_norm():
    mean = json.loads((CFG.normalization_dir / "imu_mean.json").read_text(encoding="utf-8"))
    std = json.loads((CFG.normalization_dir / "imu_std.json").read_text(encoding="utf-8"))
    m = json.loads(CFG.split_manifest.read_text(encoding="utf-8"))
    train = sorted(m["splits"]["train"])
    record(CRITICAL, "normalisation fitted on exactly the train sessions",
           mean["sessions_used"] == train == std["sessions_used"]
           and mean["split_manifest_sha256"] == normalization.manifest_sha256(CFG.split_manifest),
           f"{len(train)} train sessions; manifest sha256 matches: "
           f"{mean['split_manifest_sha256'] == normalization.manifest_sha256(CFG.split_manifest)}")
    re_ = normalization.compute_imu_stats(CFG.split_manifest, CFG.synced_root)
    dm = max(abs(re_["mean"][c] - mean["values"][c]) for c in mean["channels"])
    ds = max(abs(re_["std"][c] - std["values"][c]) for c in std["channels"])
    record(CRITICAL, "normalisation reproduces from scratch", dm < 1e-12 and ds < 1e-12,
           f"max |d mean| {dm:.1e}, max |d std| {ds:.1e}")
    try:
        normalization.compute_imu_stats(CFG.split_manifest, CFG.synced_root,
                                        sessions=[m["splits"]["test"][0]])
        refused = False
    except normalization.SplitLeakError:
        refused = True
    record(CRITICAL, "normalisation refuses a test-split session", refused, "SplitLeakError raised")


def main() -> int:
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    check_tooling()
    check_config()
    check_mutations()
    check_wheel()
    check_synced(rep)
    check_split()
    check_norm()
    gl = rep["gnss_fix_timing"]
    record(INFO, "phone GNSS fix timing", None,
           f"new-fix delay pooled median {gl['pooled_median_delay_s']} s, IQR {gl['measured_delay_iqr_s']}; "
           f"sources {gl['source_counts']}; sessions with <= 2 s fix interval: "
           f"{gl['sessions_fix_interval_le_2s']}")
    record(INFO, "split v1 (Phase 2) under the drive rule", None,
           f"{len(rep['split_v1_leaks_under_drive_rule'])} violations -> superseded by v2")
    crit = [r for r in results if r[0] == CRITICAL]
    passed = sum(1 for r in crit if r[2] == "PASS")
    lines = ["=" * 100, "SIH26168 -- PHASE 3 PREPROCESSING, FRAMES, SPLITS & NORMALISATION VALIDATION",
             f"generated: {datetime.now().isoformat(timespec='seconds')}",
             "script:    scripts/phase3_validate.py", "=" * 100, ""]
    lines += [f"  {sev:<9} {st:<9} {name:<70} {detail}" for sev, name, st, detail in results]
    lines += ["", "-" * 100, f"CRITICAL checks: {passed}/{len(crit)} passed",
              f"RESULT: {'PASS' if passed == len(crit) else 'FAIL'}", "-" * 100]
    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0 if passed == len(crit) else 1


if __name__ == "__main__":
    sys.exit(main())
