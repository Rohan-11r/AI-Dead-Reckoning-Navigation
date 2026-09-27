#!/usr/bin/env python3
"""Phase 4 validation: baseline navigation core (sensors, alignment, INS, EKF).

Re-derives the Phase 4 claims from code and artefacts (AGENTS.md 2.2). Run after:

    .venv/Scripts/python.exe -m training.preprocessing.pipeline        (Phase 3, corrected)
    .venv/Scripts/python.exe scripts/analysis/gyro_axis_resolution.py
    .venv/Scripts/python.exe scripts/analysis/imu_noise.py
    .venv/Scripts/python.exe scripts/analysis/mounting_alignment.py
    .venv/Scripts/python.exe scripts/evaluate/phase4_baseline.py [--all-verified]
    .venv/Scripts/python.exe scripts/phase4_validate.py

Writes reports/phase4_validation.txt. Exit 0 iff every CRITICAL check passes.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from navcore.sensors.samples import IOVNBD_AXIS_MAP  # noqa: E402
from training.preprocessing.columns import assert_model_inputs  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
R4 = CFG.reports_dir / "phase4"
OUT = CFG.reports_dir / "phase4_validation.txt"
PY = sys.executable
CRITICAL, INFO = "CRITICAL", "INFO"
results: list[tuple[str, str, str, str]] = []


def record(sev, name, ok, detail):
    results.append((sev, name, "OBSERVED" if ok is None else ("PASS" if ok else "FAIL"), detail))


def safe(sev, name):
    def wrap(fn):
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:  # report, never hide
                record(sev, name, False, f"raised {type(exc).__name__}: {exc}")
                return None
        return inner
    return wrap


def load(name):
    return json.loads((R4 / name).read_text(encoding="utf-8"))


def run(cmd):
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONUTF8": "1"})


@safe(CRITICAL, "lint + tests")
def check_lint_tests():
    r = run([PY, "-m", "ruff", "check", "."])
    record(CRITICAL, "ruff check . is clean", r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1])
    r = run([PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"])
    record(CRITICAL, "full test suite passes (incl. INS physics, EKF FD Jacobians, MC NEES/NIS)",
           r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1])
    r = run([PY, "-m", "pytest", "tests/navigation/test_ekf.py", "-q", "-p", "no:cacheprovider",
             "-k", "finite_difference or documented_plus_sign or monte_carlo"])
    record(CRITICAL, "EKF: F and H match finite differences; spec +[f]x psi rejected; NEES/NIS consistent",
           r.returncode == 0 and "5 passed" in r.stdout, (r.stdout.strip().splitlines() or [""])[-1])


@safe(CRITICAL, "architecture")
def check_architecture():
    ins = (REPO_ROOT / "navigation-core" / "ins" / "mechanization.py").read_text(encoding="utf-8")
    tree = ast.parse(ins)
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    gnss_use = [m for m in imported if m and "gnss" in m.lower()] + re.findall(r"GnssSample|update_gnss", ins)
    record(CRITICAL, "INS module reads no GNSS (spec sec 7)", not gnss_use, str(gnss_use))
    bad = []
    for p in (REPO_ROOT / "navigation-core").rglob("*.py"):
        code = "\n".join(line for line in p.read_text(encoding="utf-8").splitlines()
                         if not line.lstrip().startswith("#"))
        if re.search(r"\bgt_[a-z]|vbox_|can_speed|wheel_speed|yaw_rate_radps|steering", code):
            bad.append(p.relative_to(REPO_ROOT).as_posix())
    record(CRITICAL, "navigation-core never references CAN/VBOX ground-truth channels", not bad, str(bad))
    src = (REPO_ROOT / "scripts" / "evaluate" / "phase4_baseline.py").read_text(encoding="utf-8")
    inputs = ast.literal_eval(re.search(r"^INPUTS = (\[.*?\])", src, re.S | re.M).group(1))
    try:
        assert_model_inputs(inputs)
        ok = True
    except Exception as exc:  # report, never hide
        ok, inputs = False, str(exc)
    record(CRITICAL, "baseline filter inputs pass the CAN-rule guard (phone channels only)", ok, str(inputs)[:120])


@safe(CRITICAL, "gyro axes")
def check_gyro():
    g = load("gyro_axis_resolution.json")
    null = g["hypotheses"]["null (horizontal = 0)"]["median_err_deg"]
    best = g["hypotheses"][g["best_permutation"]]["median_err_deg"]
    consistent = (not g["any_permutation_beats_null"]) == (IOVNBD_AXIS_MAP.gyro_valid == (False, False, True))
    record(CRITICAL, "gyro axis map in code matches the stop-to-stop evidence", consistent and g["n_stop_pairs"] >= 50,
           f"{g['n_stop_pairs']} stop pairs; null {null:.2f} deg vs best permutation {best:.2f} deg; "
           f"map gyro_valid={IOVNBD_AXIS_MAP.gyro_valid}")


@safe(CRITICAL, "noise")
def check_noise():
    fp = load("imu_noise.json")["filter_parameters"]
    vals = [fp[k] for k in ("accel_white_noise_density", "gyro_white_noise_density", "accel_bias_sigma",
                            "accel_bias_tau_s", "gyro_bias_sigma", "gyro_bias_tau_s")]
    record(CRITICAL, "EKF process noise measured (Allan / moving vibration), finite and positive",
           all(np.isfinite(v) and v > 0 for v in vals),
           f"accel N {fp['accel_white_noise_density']:.3f} m/s^2*sqrt(s); gyro N {fp['gyro_white_noise_density']:.4f} "
           f"rad/s*sqrt(s); accel B {fp['accel_bias_sigma']:.4f}; gyro B {fp['gyro_bias_sigma']:.2e}")


@safe(CRITICAL, "alignment")
def check_alignment():
    s = load("mounting_alignment.json")["summary"]
    d = s["yaw_phone_vs_reference_abs_deg"]
    record(CRITICAL, "phone-only mount yaw agrees with the VBOX-referenced yaw where both are observable",
           s["both_acceptable"] >= 3 and d["median"] is not None and d["median"] < 5.0,
           f"{s['both_acceptable']} sessions; |diff| median {d['median']:.2f} deg, p90 {d['p90']:.2f} deg")
    record(INFO, "mount observability", None,
           f"phone-only acceptable {s['phone_only_acceptable']}/{s['sessions_evaluated']}; reference acceptable "
           f"{s['reference_acceptable']}/{s['sessions_evaluated']}; tilt median {s['tilt_deg']['median']:.2f} deg")


@safe(CRITICAL, "phase 3 fix timing")
def check_fix_timing():
    rep = json.loads((CFG.reports_dir / "phase3" / "preprocessing_report.json").read_text(encoding="utf-8"))
    g = rep["gnss_fix_timing"]
    record(CRITICAL, "Phase 3 corrected: GNSS 'latency' re-measured per NEW fix (sample-and-hold aware)",
           "gnss_fix_timing" in rep and abs(g["pooled_median_delay_s"]) < 1.0,
           f"pooled new-fix delay {g['pooled_median_delay_s']} s; sources {g['source_counts']}; "
           f"1 Hz sessions {g['sessions_fix_interval_le_2s']}")


@safe(CRITICAL, "baseline")
def check_baseline():
    split = json.loads(CFG.split_manifest.read_text(encoding="utf-8"))
    test = set(split["splits"]["test"])
    for name in ("baseline_evaluation.json", "baseline_evaluation_all_verified.json"):
        p = R4 / name
        if not p.exists():
            record(CRITICAL, f"{name} present", False, "missing")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        used = set(r["sessions_used"])
        ran = [s for s in r["sessions"] if s.get("aided_horizontal_err_m")]
        record(CRITICAL, f"{name}: no held-out TEST session used; >= 1 session scored",
               not (used & test) and len(ran) >= 1, f"{len(ran)}/{len(used)} sessions scored; test overlap "
               f"{sorted(used & test)}")
        pl = r["pooled"]
        record(INFO, f"{name} pooled", None, json.dumps({k: (round(v, 1) if isinstance(v, float) else v)
                                                        for k, v in pl.items()}))


def main() -> int:
    check_lint_tests()
    check_architecture()
    check_gyro()
    check_noise()
    check_alignment()
    check_fix_timing()
    check_baseline()
    crit = [r for r in results if r[0] == CRITICAL]
    passed = sum(1 for r in crit if r[2] == "PASS")
    lines = ["=" * 100, "SIH26168 -- PHASE 4 BASELINE NAVIGATION CORE VALIDATION",
             f"generated: {datetime.now().isoformat(timespec='seconds')}",
             "script:    scripts/phase4_validate.py", "=" * 100, ""]
    lines += [f"  {sev:<9} {st:<9} {name:<78} {detail}" for sev, name, st, detail in results]
    lines += ["", "-" * 100, f"CRITICAL checks: {passed}/{len(crit)} passed",
              f"RESULT: {'PASS' if passed == len(crit) else 'FAIL'}", "-" * 100]
    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text)
    return 0 if passed == len(crit) else 1


if __name__ == "__main__":
    sys.exit(main())
