#!/usr/bin/env python3
"""Phase 0 environment + dataset validation for SIH26168.

Re-runnable, standard-library only (no third-party packages are installed yet, and
this script must work on a clean machine).  It re-derives every factual claim made in
PROJECT_STATUS.md Phase 0 rather than trusting it.

Per AGENTS.md: nothing here fabricates a value.  Every row of the output table is the
result of an actual query against this machine or an actual read of the dataset.

Usage:
    python scripts/phase0_validate.py
    python scripts/phase0_validate.py --no-network

Exit code 0 if all CRITICAL checks pass, 1 otherwise.  INFO checks never fail the run;
they record environment facts (including missing Android tooling) for the status file.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------------------
# Configuration.  Single source of truth for the dataset location during Phase 0.
# Phase 2 replaces this with a proper config module.
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(
    r"C:\Users\Shreeyash\OneDrive\Desktop\SIH26168"
    r"\data raw  IO-VNBD"
    r"\Synchronised V abd S datasets"
    r"\Synchronised V abd S datasets"
)

EXPECTED_CSV_COUNT = 288
EXPECTED_JPG_COUNT = 72
EXPECTED_COLUMNS = 24
EXPECTED_DT_MS = 100          # 10 Hz
DATASET_ENCODING = "cp1252"

# Coventry, UK operating area.  Deliberately loose -- this is a sanity screen for
# "did we read the right file", not a precision assertion.
UK_LAT_RANGE = (49.0, 61.0)
UK_LON_RANGE = (-9.0, 2.5)

STANDARD_GRAVITY = 9.80665

DOC_FILES = ("AGENTS.md", "PROJECT_STATUS.md", "TODO.md")

NETWORK_ENDPOINTS = (
    "https://pypi.org/simple/",
    "https://github.com",
)

# --------------------------------------------------------------------------------------
# Result collection
# --------------------------------------------------------------------------------------

CRITICAL = "CRITICAL"
INFO = "INFO"

results: list[tuple[str, str, str, str]] = []  # (severity, name, status, detail)


def record(severity: str, name: str, ok: bool | None, detail: str) -> None:
    """ok=True -> PASS, ok=False -> FAIL, ok=None -> a recorded observation."""
    status = "OBSERVED" if ok is None else ("PASS" if ok else "FAIL")
    results.append((severity, name, status, detail))


def safe(severity: str, name: str):
    """Decorator: a raising check becomes a FAIL, never a crashed run."""

    def wrap(fn):
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:  # noqa: BLE001 - report, do not hide
                record(severity, name, False, f"raised {type(exc).__name__}: {exc}")
                return None

        return inner

    return wrap


# --------------------------------------------------------------------------------------
# Environment checks
# --------------------------------------------------------------------------------------


def check_python() -> None:
    v = sys.version_info
    record(
        CRITICAL,
        "Python >= 3.10",
        v >= (3, 10),
        f"{platform.python_version()} at {sys.executable}",
    )


def check_platform() -> None:
    record(INFO, "OS", None, f"{platform.system()} {platform.release()} {platform.version()}")
    record(INFO, "Machine", None, f"{platform.machine()} / {platform.processor()}")
    record(INFO, "Logical CPUs", None, str(os.cpu_count()))


@safe(INFO, "Physical RAM")
def check_ram() -> None:
    import ctypes

    class MemStat(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    st = MemStat()
    st.dwLength = ctypes.sizeof(MemStat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    total_gb = st.ullTotalPhys / 1024**3
    avail_gb = st.ullAvailPhys / 1024**3
    record(INFO, "Physical RAM", None, f"{total_gb:.2f} GB total, {avail_gb:.2f} GB available")


@safe(CRITICAL, "Free disk space >= 20 GB")
def check_disk() -> None:
    usage = shutil.disk_usage(str(REPO_ROOT))
    free_gb = usage.free / 1024**3
    record(
        CRITICAL,
        "Free disk space >= 20 GB",
        free_gb >= 20,
        f"{free_gb:.1f} GB free of {usage.total / 1024**3:.0f} GB",
    )


def check_tool(name: str, exe: str, severity: str = INFO) -> None:
    path = shutil.which(exe)
    record(severity, f"tool: {name}", None, path if path else "NOT FOUND on PATH")


def check_android_toolchain() -> None:
    """Recorded as INFO, not FAIL: Phase 0 does not need Android, Phases 12-14 do."""
    for label, exe in (("java (JDK)", "java"), ("gradle", "gradle"), ("adb", "adb")):
        check_tool(label, exe)
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT", "JAVA_HOME"):
        record(INFO, f"env: {var}", None, os.environ.get(var) or "unset")
    default_sdk = Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk"
    record(
        INFO,
        "default Android SDK dir",
        None,
        str(default_sdk) if default_sdk.is_dir() else f"absent ({default_sdk})",
    )


def check_ml_packages() -> None:
    import importlib.util

    present, missing = [], []
    for mod in (
        "numpy", "scipy", "pandas", "sklearn", "torch", "tensorflow",
        "matplotlib", "onnx", "onnxruntime", "pyarrow",
    ):
        (present if importlib.util.find_spec(mod) else missing).append(mod)
    record(INFO, "ML packages present", None, ", ".join(present) or "none")
    record(INFO, "ML packages missing", None, ", ".join(missing) or "none")


@safe(INFO, "Network")
def check_network(enabled: bool) -> None:
    if not enabled:
        record(INFO, "Network", None, "skipped (--no-network)")
        return
    import urllib.request

    for url in NETWORK_ENDPOINTS:
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
                record(INFO, f"network: {url}", None, f"HTTP {resp.status}")
        except Exception as exc:  # noqa: BLE001
            record(INFO, f"network: {url}", None, f"unreachable ({type(exc).__name__})")


# --------------------------------------------------------------------------------------
# Repository checks
# --------------------------------------------------------------------------------------


def check_docs() -> None:
    for name in DOC_FILES:
        p = REPO_ROOT / name
        record(CRITICAL, f"doc: {name}", p.is_file(), f"{p.stat().st_size} bytes" if p.is_file() else "missing")

    agents = REPO_ROOT / "AGENTS.md"
    if agents.is_file():
        text = agents.read_text(encoding="utf-8", errors="replace")
        phrases = (
            "Sensors -> Calibration -> AI Signal Processing -> INS -> Fusion -> NHC -> Map Matching",
            "Never hallucinate latitude/longitude directly",
            "Never fake data",
        )
        found = [p for p in phrases if p in text]
        record(
            CRITICAL,
            "AGENTS.md states the core directive",
            len(found) == len(phrases),
            f"{len(found)}/{len(phrases)} directive clauses present",
        )


# --------------------------------------------------------------------------------------
# Dataset checks
# --------------------------------------------------------------------------------------


def normalise_header(name: str) -> str:
    """Collapse the dataset's irregular column spellings to a stable key."""
    return " ".join(name.replace("\ufeff", "").split()).upper()


@safe(CRITICAL, "IO-VNBD dataset")
def check_dataset() -> None:
    if not DATASET_ROOT.is_dir():
        record(
            CRITICAL,
            "IO-VNBD dataset root exists",
            False,
            f"not found: {DATASET_ROOT}  (edit DATASET_ROOT in this script)",
        )
        return
    record(CRITICAL, "IO-VNBD dataset root exists", True, str(DATASET_ROOT))

    csvs = sorted(DATASET_ROOT.rglob("*.csv"))
    jpgs = [p for p in DATASET_ROOT.rglob("*") if p.suffix.lower() == ".jpg"]
    total_bytes = sum(p.stat().st_size for p in csvs)

    record(
        CRITICAL,
        f"CSV count == {EXPECTED_CSV_COUNT}",
        len(csvs) == EXPECTED_CSV_COUNT,
        f"found {len(csvs)} CSV files, {total_bytes / 1024**2:.1f} MB",
    )
    record(
        INFO,
        f"JPG count == {EXPECTED_JPG_COUNT}",
        None,
        f"found {len(jpgs)}",
    )

    cat = len(list((DATASET_ROOT / "Categorised IOVNB Dataset").rglob("*.csv")))
    unc = len(list((DATASET_ROOT / "Uncategorised IOVNB Dataset").rglob("*.csv")))
    record(INFO, "Categorised / Uncategorised CSV split", None, f"{cat} / {unc}")

    s_files = [p for p in csvs if p.name.upper().startswith("S-")]
    v_files = [p for p in csvs if p.name.upper().startswith("V-")]
    record(
        INFO,
        "S- / V- prefix split",
        None,
        f"{len(s_files)} smartphone / {len(v_files)} vehicle "
        f"(unclassified: {len(csvs) - len(s_files) - len(v_files)})",
    )

    if not csvs:
        return

    sample = csvs[0]
    record(INFO, "sample file", None, str(sample.relative_to(DATASET_ROOT)))

    # --- encoding: cp1252 must work, strict UTF-8 must NOT --------------------------
    raw = sample.read_bytes()[:4096]
    try:
        raw.decode("utf-8")
        utf8_fails = False
    except UnicodeDecodeError:
        utf8_fails = True
    record(
        CRITICAL,
        "strict UTF-8 read of dataset fails (encoding is cp1252)",
        utf8_fails,
        "confirmed non-UTF-8 bytes present" if utf8_fails
        else "UTF-8 decoded cleanly -- PROJECT_STATUS.md 0.4 claim is WRONG, investigate",
    )
    try:
        raw.decode(DATASET_ENCODING)
        cp_ok = True
        cp_detail = f"{DATASET_ENCODING} decoded cleanly"
    except UnicodeDecodeError as exc:
        cp_ok = False
        cp_detail = f"{DATASET_ENCODING} failed: {exc}"
    record(CRITICAL, f"dataset decodes as {DATASET_ENCODING}", cp_ok, cp_detail)

    # --- schema and first rows ------------------------------------------------------
    with sample.open("r", encoding=DATASET_ENCODING, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [next(reader) for _ in range(4)]

    record(
        CRITICAL,
        f"column count == {EXPECTED_COLUMNS}",
        len(header) == EXPECTED_COLUMNS,
        f"found {len(header)}",
    )

    cols = {normalise_header(h): i for i, h in enumerate(header)}
    required = [
        "GPS LATITUDE (DEGREES)", "GPS LONGITUDE (DEGREES)", "GPS SPEED (KMH)",
        "TIME SINCE START (MS)", "GYROSCOPE YAW (RAD/S)",
    ]
    absent = [c for c in required if c not in cols]
    record(
        CRITICAL,
        "expected columns resolve after name normalisation",
        not absent,
        f"all {len(required)} present" if not absent else f"missing: {absent}",
    )

    # --- sampling interval ----------------------------------------------------------
    t_idx = cols.get("TIME SINCE START (MS)")
    if t_idx is not None:
        ts = [int(r[t_idx]) for r in rows]
        deltas = [b - a for a, b in zip(ts, ts[1:])]
        record(
            CRITICAL,
            f"sample interval == {EXPECTED_DT_MS} ms (10 Hz)",
            all(d == EXPECTED_DT_MS for d in deltas),
            f"observed deltas {deltas} ms from t={ts[0]} ms",
        )

    # --- geodetic plausibility (NOT a model output -- this is raw GNSS truth) -------
    lat_i, lon_i = cols.get("GPS LATITUDE (DEGREES)"), cols.get("GPS LONGITUDE (DEGREES)")
    if lat_i is not None and lon_i is not None:
        lat, lon = float(rows[0][lat_i]), float(rows[0][lon_i])
        in_uk = (UK_LAT_RANGE[0] <= lat <= UK_LAT_RANGE[1]
                 and UK_LON_RANGE[0] <= lon <= UK_LON_RANGE[1])
        record(
            CRITICAL,
            "first GNSS fix within the UK operating area",
            in_uk,
            f"lat={lat}, lon={lon}",
        )

    # --- gravity magnitude ----------------------------------------------------------
    g_idx = [cols.get(f"GRAVITY {a} (M/S²)") for a in "XYZ"]
    if all(i is not None for i in g_idx):
        gx, gy, gz = (float(rows[0][i]) for i in g_idx)
        g_mag = (gx * gx + gy * gy + gz * gz) ** 0.5
        record(
            CRITICAL,
            "gravity magnitude within 0.5 m/s^2 of 9.80665",
            abs(g_mag - STANDARD_GRAVITY) < 0.5,
            f"|g| = {g_mag:.5f} m/s^2 (deviation {g_mag - STANDARD_GRAVITY:+.5f})",
        )

    # --- non-numeric satellite field -----------------------------------------------
    sat_i = cols.get("GPS SATELLITES IN RANGE")
    if sat_i is not None:
        val = rows[0][sat_i].strip()
        record(
            INFO,
            "GPS SATELLITES IN RANGE is non-numeric (needs splitting)",
            None,
            f"observed {val!r}",
        )

    # --- datetime format ------------------------------------------------------------
    d_i = cols.get("DATE (YYYY-MO-DD HH-MI-SS_SSS)")
    if d_i is not None:
        val = rows[0][d_i].strip()
        fmt = "%Y-%m-%d %H:%M:%S:%f"
        try:
            parsed = datetime.strptime(val, fmt)
            record(
                CRITICAL,
                f"DATE parses with explicit format {fmt!r}",
                True,
                f"{val!r} -> {parsed.isoformat()}",
            )
        except ValueError as exc:
            record(CRITICAL, f"DATE parses with explicit format {fmt!r}", False, f"{val!r}: {exc}")


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def emit(lines: list[str]) -> None:
    sev_w = max(len(r[0]) for r in results)
    name_w = max(len(r[1]) for r in results)
    for sev, name, status, detail in results:
        lines.append(f"  {sev:<{sev_w}}  {status:<8}  {name:<{name_w}}  {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description="SIH26168 Phase 0 validation")
    ap.add_argument("--no-network", action="store_true", help="skip outbound HTTPS probes")
    args = ap.parse_args()

    check_python()
    check_platform()
    check_ram()
    check_disk()
    check_tool("git", "git")
    check_android_toolchain()
    check_ml_packages()
    check_network(enabled=not args.no_network)
    check_docs()
    check_dataset()

    crit = [r for r in results if r[0] == CRITICAL]
    failed = [r for r in crit if r[2] == "FAIL"]

    lines = [
        "=" * 100,
        "SIH26168 -- PHASE 0 ENVIRONMENT & DATASET VALIDATION",
        f"generated: {datetime.now().isoformat(timespec='seconds')}",
        f"script:    {Path(__file__).name}",
        f"repo:      {REPO_ROOT}",
        "=" * 100,
        "",
    ]
    emit(lines)
    lines += [
        "",
        "-" * 100,
        f"CRITICAL checks: {len(crit) - len(failed)}/{len(crit)} passed",
        f"RESULT: {'PASS' if not failed else 'FAIL'}",
    ]
    if failed:
        lines.append("Failed critical checks:")
        lines += [f"  - {r[1]}: {r[3]}" for r in failed]
    lines += [
        "-" * 100,
        "",
        "NOTE: missing java/gradle/adb/Android SDK are recorded as OBSERVED, not FAIL.",
        "      Phase 0 does not require them; Phases 12-14 do. See PROJECT_STATUS.md 0.8.",
    ]

    report = "\n".join(lines)
    print(report)

    out = REPO_ROOT / "reports" / "phase0_validation.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n[written] {out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
