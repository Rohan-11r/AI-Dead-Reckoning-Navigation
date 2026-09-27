#!/usr/bin/env python3
"""Phase 1 validation for SIH26168: structure, environment, packages, imports.

Re-runnable and self-contained. It re-derives every claim made in PROJECT_STATUS.md
Phase 1 rather than trusting it (AGENTS.md 2.2).

MUST be run with the project virtual environment's interpreter:

    .venv/Scripts/python.exe scripts/phase1_validate.py

Checks performed:
  1. directory structure matches the LAYOUT manifest in scripts/setup/create_structure.py
  2. the running interpreter really is the project venv, isolated from the global one
  3. every pin in requirements.txt is installed at EXACTLY that version
  4. requirements.txt and pyproject.toml pins agree (no silent drift)
  5. pyproject `packages` agrees with the LAYOUT manifest (no silent drift)
  6. every declared package imports, including the navcore -> navigation-core remap
  7. a real torch -> ONNX -> onnxruntime round trip, compared numerically
  8. foundational docs exist and contain the pipeline order

Exit code 0 if all CRITICAL checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import importlib.util
import os
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CRITICAL = "CRITICAL"
INFO = "INFO"

results: list[tuple[str, str, str, str]] = []


def record(severity: str, name: str, ok: bool | None, detail: str) -> None:
    status = "OBSERVED" if ok is None else ("PASS" if ok else "FAIL")
    results.append((severity, name, status, detail))


def safe(severity: str, name: str):
    def wrap(fn):
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:  # noqa: BLE001 - report, never hide
                record(severity, name, False, f"raised {type(exc).__name__}: {exc}")
                return None
        return inner
    return wrap


# --------------------------------------------------------------------------------------
# Shared helpers: parse the manifest and the pin files.
# --------------------------------------------------------------------------------------

def load_layout() -> list[tuple[str, str, str]]:
    """Import LAYOUT from the structure script without executing its main()."""
    spec = importlib.util.spec_from_file_location(
        "_create_structure", REPO_ROOT / "scripts" / "setup" / "create_structure.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/setup/create_structure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LAYOUT


PIN_RE = re.compile(r"^([A-Za-z0-9._-]+)==([A-Za-z0-9.+!-]+)$")


def parse_requirements(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = PIN_RE.match(line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def parse_pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def expected_packages(layout) -> set[str]:
    """Derive the importable package list from the layout manifest."""
    out: set[str] = set()
    for kind, rel, _doc in layout:
        if kind != "pkg":
            continue
        parts = rel.split("/")
        parts[0] = "navcore" if parts[0] == "navigation-core" else parts[0]
        if parts[0] == "tests":          # tests are not an installed package
            continue
        out.add(".".join(parts))
    return out


# --------------------------------------------------------------------------------------
# 1. Structure
# --------------------------------------------------------------------------------------

@safe(CRITICAL, "directory structure")
def check_structure(layout) -> None:
    missing_dirs, missing_markers = [], []
    for kind, rel, _doc in layout:
        target = REPO_ROOT / rel
        if not target.is_dir():
            missing_dirs.append(rel)
            continue
        marker = target / ("__init__.py" if kind == "pkg" else ".gitkeep")
        if not marker.is_file():
            missing_markers.append(f"{rel}/{marker.name}")

    record(
        CRITICAL,
        f"all {len(layout)} manifest directories exist",
        not missing_dirs,
        "all present" if not missing_dirs else f"missing {len(missing_dirs)}: {missing_dirs[:5]}",
    )
    record(
        CRITICAL,
        "every package has __init__.py / dir has .gitkeep",
        not missing_markers,
        "all present" if not missing_markers else f"missing {len(missing_markers)}: {missing_markers[:5]}",
    )
    n_pkg = sum(1 for k, _, _ in layout if k == "pkg")
    record(INFO, "manifest composition", None,
           f"{len(layout)} entries = {n_pkg} python packages + {len(layout) - n_pkg} plain dirs")


# --------------------------------------------------------------------------------------
# 2. Virtual environment
# --------------------------------------------------------------------------------------

def check_venv() -> None:
    expected = (REPO_ROOT / ".venv").resolve()
    running = Path(sys.prefix).resolve()

    record(
        CRITICAL,
        "venv directory exists",
        expected.is_dir(),
        str(expected) if expected.is_dir() else f"missing: {expected}",
    )
    record(
        CRITICAL,
        "interpreter is isolated from the global install",
        sys.prefix != sys.base_prefix,
        f"prefix={sys.prefix}  base={sys.base_prefix}",
    )
    record(
        CRITICAL,
        "running interpreter IS the project venv",
        running == expected,
        f"running from {running}"
        + ("" if running == expected else f"  -- expected {expected}. Re-run with .venv/Scripts/python.exe"),
    )
    record(INFO, "python version", None, f"{sys.version.split()[0]} ({sys.executable})")
    record(
        INFO,
        "PYTHONUTF8",
        None,
        os.environ.get("PYTHONUTF8", "unset")
        + ("" if os.environ.get("PYTHONUTF8") == "1"
           else "  -- set to 1 for the dynamo ONNX exporter (see scripts/setup/activate.*)"),
    )
    record(INFO, "stdout encoding", None, str(sys.stdout.encoding))


# --------------------------------------------------------------------------------------
# 3-5. Dependency pins and manifest agreement
# --------------------------------------------------------------------------------------

@safe(CRITICAL, "installed package versions")
def check_pins() -> None:
    pins = parse_requirements(REPO_ROOT / "requirements.txt")
    record(INFO, "requirements.txt pins parsed", None, f"{len(pins)} exact pins")

    wrong, absent = [], []
    for name, want in sorted(pins.items()):
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            absent.append(name)
            continue
        # torch reports 2.14.0+cpu for the pinned 2.14.0 CPU wheel; the local version
        # segment is a build variant, not a version mismatch.
        if got != want and got.split("+")[0] != want.split("+")[0]:
            wrong.append(f"{name}: want {want}, got {got}")

    record(
        CRITICAL,
        "all pinned requirements installed at the pinned version",
        not absent and not wrong,
        f"{len(pins) - len(absent) - len(wrong)}/{len(pins)} exact"
        + (f"; MISSING {absent}" if absent else "")
        + (f"; MISMATCH {wrong}" if wrong else ""),
    )


@safe(CRITICAL, "requirements.txt vs pyproject.toml")
def check_pin_agreement() -> None:
    req = parse_requirements(REPO_ROOT / "requirements.txt")
    proj = parse_pyproject()
    py: dict[str, str] = {}
    for dep in proj["project"]["dependencies"]:
        m = PIN_RE.match(dep.strip())
        if m:
            py[m.group(1).lower().replace("_", "-")] = m.group(2)

    only_req = sorted(set(req) - set(py))
    only_py = sorted(set(py) - set(req))
    differ = sorted(f"{k}: req={req[k]} proj={py[k]}" for k in set(req) & set(py) if req[k] != py[k])

    ok = not (only_req or only_py or differ)
    record(
        CRITICAL,
        "requirements.txt and pyproject.toml pins agree",
        ok,
        f"{len(req)} pins identical" if ok else
        f"only-in-requirements={only_req}; only-in-pyproject={only_py}; differing={differ}",
    )


@safe(CRITICAL, "pyproject packages vs layout manifest")
def check_package_list(layout) -> None:
    proj = parse_pyproject()
    declared = set(proj["tool"]["setuptools"]["packages"])
    derived = expected_packages(layout)

    missing = sorted(derived - declared)
    extra = sorted(declared - derived)
    ok = not missing and not extra
    record(
        CRITICAL,
        "pyproject `packages` matches the layout manifest",
        ok,
        f"{len(declared)} packages agree" if ok else
        f"in manifest but not pyproject={missing}; in pyproject but not manifest={extra}",
    )

    pkg_dir = proj["tool"]["setuptools"].get("package-dir", {})
    record(
        CRITICAL,
        "navcore -> navigation-core remap declared",
        pkg_dir.get("navcore") == "navigation-core",
        f"package-dir = {pkg_dir}",
    )


# --------------------------------------------------------------------------------------
# 6. Imports
# --------------------------------------------------------------------------------------

@safe(CRITICAL, "third-party imports")
def check_third_party_imports() -> None:
    mods = [
        ("numpy", "__version__"), ("scipy", "__version__"), ("pandas", "__version__"),
        ("sklearn", "__version__"), ("torch", "__version__"), ("onnx", "__version__"),
        ("onnxruntime", "__version__"), ("onnxscript", "__version__"),
        ("pyarrow", "__version__"), ("pyproj", "__version__"), ("shapely", "__version__"),
        ("networkx", "__version__"), ("matplotlib", "__version__"), ("yaml", "__version__"),
        ("allantools", None), ("tqdm", "__version__"), ("rich", "__version__"),
        ("psutil", "__version__"), ("pytest", "__version__"), ("joblib", "__version__"),
        ("seaborn", "__version__"), ("tensorboard", None),
    ]
    failures = []
    versions = []
    for name, attr in mods:
        try:
            m = importlib.import_module(name)
            v = getattr(m, attr, "?") if attr else "ok"
            versions.append(f"{name}={v}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}")
    record(
        CRITICAL,
        f"all {len(mods)} third-party modules import",
        not failures,
        f"{len(mods) - len(failures)}/{len(mods)} imported" + (f"; FAILED {failures}" if failures else ""),
    )
    record(INFO, "key versions", None,
           ", ".join(v for v in versions if v.split("=")[0] in
                     {"numpy", "scipy", "pandas", "torch", "onnx", "onnxruntime", "onnxscript"}))


@safe(CRITICAL, "project package imports")
def check_project_imports(layout) -> None:
    targets = sorted(expected_packages(layout))
    failures = []
    navcore_file = None
    for mod in targets:
        try:
            m = importlib.import_module(mod)
            if mod == "navcore.ins":
                navcore_file = getattr(m, "__file__", None)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mod}: {type(exc).__name__}")
    record(
        CRITICAL,
        f"all {len(targets)} project packages import",
        not failures,
        f"{len(targets) - len(failures)}/{len(targets)} imported" + (f"; FAILED {failures}" if failures else ""),
    )
    # The remap is the fragile part: setuptools' package auto-discovery cannot see
    # through package-dir, so this asserts the explicit list actually resolves on disk.
    ok = bool(navcore_file) and "navigation-core" in str(navcore_file).replace("\\", "/")
    record(
        CRITICAL,
        "navcore.* resolves to navigation-core/ on disk",
        ok,
        str(navcore_file) if navcore_file else "navcore.ins has no __file__",
    )


# --------------------------------------------------------------------------------------
# 7. Export toolchain round trip
# --------------------------------------------------------------------------------------

@safe(CRITICAL, "torch -> ONNX -> onnxruntime round trip")
def check_onnx_roundtrip() -> None:
    import tempfile

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from torch import nn

    torch.manual_seed(0)

    class Tiny(nn.Module):
        """Shape-representative of a Phase 7 head: IMU window -> (speed, log-variance)."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(6, 16, 5, padding=2)
            self.head = nn.Linear(16, 2)

        def forward(self, x):
            return self.head(torch.relu(self.conv(x)).mean(dim=-1))

    model = Tiny().eval()
    x = torch.randn(3, 6, 20)
    with torch.no_grad():
        ref = model(x).numpy()

    out = Path(tempfile.mkdtemp()) / "phase1_roundtrip.onnx"
    torch.onnx.export(
        model, (x,), str(out),
        input_names=["imu_window"], output_names=["speed_logvar"],
        opset_version=17, dynamo=False,          # legacy path: no PYTHONUTF8 dependency
    )
    onnx.checker.check_model(onnx.load(str(out)))
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
        None, {"imu_window": x.numpy()}
    )[0]

    max_abs = float(np.max(np.abs(ref - got)))
    record(
        CRITICAL,
        "ONNX export numerically matches PyTorch (tol 1e-5)",
        max_abs < 1e-5,
        f"max |torch - onnxruntime| = {max_abs:.3e} over {ref.size} outputs",
    )
    record(INFO, "onnxruntime providers", None, ", ".join(ort.get_available_providers()))
    record(
        INFO,
        "torch CUDA",
        None,
        f"available={torch.cuda.is_available()}  build={torch.__version__}"
        + ("  (CPU wheel; see requirements-gpu.txt for Phase 7)" if not torch.cuda.is_available() else ""),
    )


# --------------------------------------------------------------------------------------
# 8. Documentation
# --------------------------------------------------------------------------------------

def check_docs() -> None:
    required = {
        "docs/architecture.md": "Sensors -> Calibration -> AI",
        "docs/navigation_math.md": None,
        "docs/ml_pipeline.md": None,
        "docs/map_matching.md": None,
        "README.md": None,
        "AGENTS.md": "Never hallucinate latitude/longitude directly",
        "PROJECT_STATUS.md": None,
        "TODO.md": None,
        "requirements.txt": None,
        "pyproject.toml": None,
        ".gitignore": None,
        ".gitattributes": None,
    }
    missing, empty = [], []
    for rel in required:
        p = REPO_ROOT / rel
        if not p.is_file():
            missing.append(rel)
        elif p.stat().st_size == 0:
            empty.append(rel)
    record(
        CRITICAL,
        f"all {len(required)} foundational files present and non-empty",
        not missing and not empty,
        "all present" if not missing and not empty else f"missing={missing} empty={empty}",
    )

    # The architecture doc must actually carry the mandated pipeline order.
    #
    # Scoped to the FIRST fenced block (the pipeline diagram), not the whole document:
    # the prose introduction legitimately mentions NHC and the road network before the
    # diagram, so a whole-file scan would report a false ordering failure.
    #
    # Anchors use \b word boundaries because bare substring matching is wrong here --
    # "OUTPUT" matches inside "OUTPUTS ARE PHYSICAL QUANTITIES" in the AI stage box,
    # which places OUTPUT before FUSION and fails a correct diagram.
    arch = REPO_ROOT / "docs" / "architecture.md"
    if arch.is_file():
        text = arch.read_text(encoding="utf-8", errors="replace")
        blocks = re.findall(r"```.*?\n(.*?)```", text, flags=re.DOTALL)
        diagram = blocks[0] if blocks else ""
        record(
            CRITICAL,
            "architecture.md contains a pipeline diagram block",
            bool(diagram.strip()),
            f"{len(diagram.splitlines())} lines in the first fenced block"
            if diagram.strip() else "no fenced code block found",
        )

        stages = ["SENSORS", "CALIBRATION", "AI SIGNAL PROCESSING", "INS", "FUSION",
                  "NHC", "MAP MATCHING", "OUTPUT"]
        positions: dict[str, int] = {}
        for s in stages:
            m = re.search(rf"\b{re.escape(s)}\b", diagram)
            if m:
                positions[s] = m.start()

        missing = [s for s in stages if s not in positions]
        record(
            CRITICAL,
            "diagram contains all 8 pipeline stages",
            not missing,
            f"{len(positions)}/{len(stages)} stages present"
            + ("" if not missing else f"; missing {missing}"),
        )
        if not missing:
            order = [positions[s] for s in stages]
            record(
                CRITICAL,
                "pipeline stages appear in the mandated order",
                order == sorted(order),
                "Sensors -> Calibration -> AI -> INS -> Fusion -> NHC -> Map -> Output"
                if order == sorted(order)
                else "out of order: " + " ".join(
                    s for s in sorted(stages, key=lambda k: positions[k])
                ),
            )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="SIH26168 Phase 1 validation")
    ap.add_argument("--skip-onnx", action="store_true", help="skip the export round trip")
    args = ap.parse_args()

    layout = load_layout()

    check_structure(layout)
    check_venv()
    check_pins()
    check_pin_agreement()
    check_package_list(layout)
    check_third_party_imports()
    check_project_imports(layout)
    if args.skip_onnx:
        record(INFO, "ONNX round trip", None, "skipped (--skip-onnx)")
    else:
        check_onnx_roundtrip()
    check_docs()

    crit = [r for r in results if r[0] == CRITICAL]
    failed = [r for r in crit if r[2] == "FAIL"]

    sev_w = max(len(r[0]) for r in results)
    name_w = max(len(r[1]) for r in results)

    lines = [
        "=" * 104,
        "SIH26168 -- PHASE 1 VALIDATION (structure, environment, packages, imports, export)",
        f"generated: {datetime.now().isoformat(timespec='seconds')}",
        f"script:    {Path(__file__).name}",
        f"repo:      {REPO_ROOT}",
        f"python:    {sys.version.split()[0]}  ({sys.executable})",
        "=" * 104,
        "",
    ]
    for sev, name, status, detail in results:
        lines.append(f"  {sev:<{sev_w}}  {status:<8}  {name:<{name_w}}  {detail}")
    lines += [
        "",
        "-" * 104,
        f"CRITICAL checks: {len(crit) - len(failed)}/{len(crit)} passed",
        f"RESULT: {'PASS' if not failed else 'FAIL'}",
    ]
    if failed:
        lines.append("Failed critical checks:")
        lines += [f"  - {r[1]}: {r[3]}" for r in failed]
    lines += ["-" * 104]

    report = "\n".join(lines)
    print(report)

    out = REPO_ROOT / "reports" / "phase1_validation.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\n[written] {out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
