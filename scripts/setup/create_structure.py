#!/usr/bin/env python3
"""Create the SIH26168 monorepo directory structure.

Idempotent and re-runnable: existing directories and non-empty files are left alone.
Standard library only, so it works before the virtual environment exists.

The manifest below is the single source of truth for the repository layout.  Anything
that needs to know the layout (validation scripts, packaging config) should agree with
this file rather than restate it.

Usage:
    python scripts/setup/create_structure.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------------------
# Layout manifest.
#
#   pkg   -> importable Python package: gets an __init__.py carrying the docstring
#   dir   -> plain directory (data, artefacts, docs, non-Python source): gets .gitkeep
#
# The docstring text is not decoration -- it is the contract for what belongs in each
# directory, so that the pipeline order in AGENTS.md is legible from the tree alone.
# --------------------------------------------------------------------------------------

LAYOUT: list[tuple[str, str, str]] = [
    # ---- documentation ---------------------------------------------------------------
    ("dir", "docs", "Architecture, navigation mathematics, ML pipeline, map matching."),

    # ---- data (read-only raw; everything else regenerable) ---------------------------
    ("dir", "data", "Dataset tree. Raw data is READ-ONLY (AGENTS.md 5)."),
    ("dir", "data/raw", "Immutable source recordings. Never written to by any pipeline."),
    ("dir", "data/interim", "Partially processed intermediates. Fully regenerable."),
    ("dir", "data/processed", "Analysis-ready columnar data with a versioned schema."),
    ("dir", "data/splits", "Route/driver-level split manifests. Never random-row splits."),

    # ---- training (offline ML) -------------------------------------------------------
    ("pkg", "training", "Offline ML: data plumbing, models, training and export."),
    ("pkg", "training/configs", "Declarative experiment configs. No hyperparameters in code."),
    ("pkg", "training/datasets", "Dataset classes and windowing over the processed data."),
    ("pkg", "training/preprocessing", "Loaders, schema normalisation, unit conversion, integrity audit."),
    ("pkg", "training/features", "Feature extraction from IMU windows. Physical units only."),
    ("pkg", "training/models", "Model definitions. Outputs are physical quantities, NEVER coordinates."),
    ("pkg", "training/losses", "Loss functions, including uncertainty/NLL objectives."),
    ("pkg", "training/trainers", "Training loops, seeding, checkpointing, reproducibility."),
    ("pkg", "training/evaluation", "Model-level (not trajectory-level) evaluation."),
    ("pkg", "training/export", "ONNX/TFLite export plus numerical verification of the export."),
    ("pkg", "training/visualization", "Training diagnostics and plots."),
    ("pkg", "training/scripts", "Entry points for training and export runs."),

    # ---- model artefacts -------------------------------------------------------------
    ("dir", "models", "Model artefacts. Binary outputs, not source."),
    ("dir", "models/checkpoints", "Training checkpoints."),
    ("dir", "models/exported", "Verified ONNX/TFLite graphs cleared for on-device use."),
    ("dir", "models/metadata", "Model cards: inputs, outputs, units, training provenance."),
    ("dir", "models/normalization", "Feature normalisation statistics. Shared Python/Android."),

    # ---- navigation core (Python reference implementation) ---------------------------
    ("pkg", "navigation-core", "Python REFERENCE implementation of the navigation pipeline."),
    ("pkg", "navigation-core/common", "Constants, units, logging, typed containers."),
    ("pkg", "navigation-core/sensors", "Sensor models, sample containers, rate/timestamp handling."),
    ("pkg", "navigation-core/calibration", "Bias, scale factor, misalignment, Allan variance."),
    ("pkg", "navigation-core/alignment", "Initial and in-motion alignment; device-to-vehicle mounting."),
    ("pkg", "navigation-core/imu", "IMU preprocessing: coning/sculling, gravity separation."),
    ("pkg", "navigation-core/ins", "Strapdown INS mechanization. The physics core."),
    ("pkg", "navigation-core/gnss", "GNSS quality gating. Masked during simulated outages."),
    ("pkg", "navigation-core/fusion", "Error-state EKF. INS + AI + GNSS-when-valid + ZUPT."),
    ("pkg", "navigation-core/nhc", "Non-holonomic constraints and vehicle kinematics."),
    ("pkg", "navigation-core/map_matching", "HMM map matching. REFINEMENT ONLY, never a position source."),
    ("pkg", "navigation-core/outage", "GNSS outage detection and pipeline mode switching."),
    ("pkg", "navigation-core/recovery", "Re-acquisition and state recovery after an outage."),
    ("pkg", "navigation-core/state", "Navigation state and covariance representation."),
    ("pkg", "navigation-core/geometry", "Geodesy and rotations. THE ONLY path to lat/lon (AGENTS.md 2.1)."),
    ("pkg", "navigation-core/filtering", "Reusable estimator primitives and numerical guards."),
    ("pkg", "navigation-core/diagnostics", "Consistency checks: NIS/NEES, covariance PD, kinematic sanity."),

    # ---- edge engine (portable implementation for on-device) -------------------------
    ("dir", "edge-engine", "Portable on-device engine. Second implementation of the one spec."),
    ("dir", "edge-engine/src", "Portable core sources."),
    ("dir", "edge-engine/adapters", "Platform adapters (Android/JNI, desktop, test harness)."),
    ("dir", "edge-engine/configs", "Engine runtime configuration."),
    ("dir", "edge-engine/tests", "Engine tests, including golden-vector parity fixtures."),
    ("dir", "edge-engine/examples", "Minimal embedding examples."),

    # ---- android application ---------------------------------------------------------
    ("dir", "android-app", "Android application. Scaffolded in Phase 13 (blocked: no JDK/SDK)."),

    # ---- simulation ------------------------------------------------------------------
    ("pkg", "simulation", "Replay and scenario simulation over real recordings."),
    ("pkg", "simulation/replay", "Deterministic replay of recorded sensor streams."),
    ("pkg", "simulation/outage", "GNSS outage injection. Masks columns; does not set flags."),
    ("pkg", "simulation/scenarios", "Named scenarios: tunnel, urban canyon, car park."),
    ("pkg", "simulation/synthetic", "Analytically-known motion for unit tests. CLEARLY LABELLED synthetic."),
    ("pkg", "simulation/visualization", "Trajectory and scenario visualisation."),

    # ---- evaluation ------------------------------------------------------------------
    ("pkg", "evaluation", "Trajectory-level evaluation. The single source of published numbers."),
    ("pkg", "evaluation/metrics", "ATE, RPE, CDF percentiles, drift %, heading error."),
    ("pkg", "evaluation/baselines", "Honest baselines: pure INS, GNSS-only, classical DR."),
    ("dir", "evaluation/reports", "Generated evaluation reports. Regenerable, stamped with provenance."),
    ("dir", "evaluation/plots", "Generated figures. Regenerable."),

    # ---- hardware --------------------------------------------------------------------
    ("pkg", "hardware", "External IMU hardware integration."),
    ("pkg", "hardware/imu_protocol", "Wire protocol parsing and framing."),
    ("pkg", "hardware/bluetooth", "Bluetooth transport."),
    ("pkg", "hardware/usb", "USB transport."),
    ("pkg", "hardware/serial", "Serial transport."),
    ("dir", "hardware/examples", "Capture examples for external IMUs."),

    # ---- operational scripts ---------------------------------------------------------
    ("dir", "scripts", "Operational entry points. Run directly, not imported."),
    ("dir", "scripts/setup", "Environment and repository bootstrap."),
    ("dir", "scripts/download", "Dataset and map-data acquisition."),
    ("dir", "scripts/preprocess", "Raw -> interim -> processed pipeline runners."),
    ("dir", "scripts/train", "Training run launchers."),
    ("dir", "scripts/evaluate", "Evaluation and benchmark runners."),
    ("dir", "scripts/export", "Model export and export-verification runners."),
    ("dir", "scripts/validate", "Phase gates and invariant checks."),

    # ---- tests -----------------------------------------------------------------------
    ("pkg", "tests", "Test suite."),
    ("pkg", "tests/unit", "Pure unit tests for individual functions."),
    ("pkg", "tests/integration", "Multi-component pipeline tests."),
    ("pkg", "tests/navigation", "Navigation correctness: frames, rotations, INS, filter consistency."),
    ("pkg", "tests/ml", "ML plumbing tests. Includes the no-coordinate-target invariant."),
    ("pkg", "tests/android", "Python side of the Python/Android parity suite."),
    ("pkg", "tests/hardware", "Hardware interface tests."),
    ("pkg", "tests/regression", "Golden-vector and numerical-regression tests."),

    # ---- demo ------------------------------------------------------------------------
    ("dir", "demo", "Demonstration material for SIH evaluation."),

    # ---- reports (created in Phase 0, declared here for completeness) ----------------
    ("dir", "reports", "Phase validation outputs and provenance records."),
]

INIT_TEMPLATE = '''"""{doc}

Part of SIH26168 -- AI-ML based Intelligent Dead Reckoning.
Governed by AGENTS.md: physical units and frames explicit; no coordinate outputs
outside navigation-core/geometry; no fabricated data.
"""
'''

GITKEEP_TEMPLATE = (
    "# {path}\n"
    "# {doc}\n"
    "#\n"
    "# Placeholder so the directory is tracked while empty. Safe to delete once the\n"
    "# directory holds real, tracked content.\n"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create the SIH26168 repository structure")
    ap.add_argument("--dry-run", action="store_true", help="report actions without writing")
    args = ap.parse_args()

    created_dirs = 0
    created_files = 0
    skipped = 0

    for kind, rel, doc in LAYOUT:
        target = REPO_ROOT / rel

        if not target.exists():
            if not args.dry_run:
                target.mkdir(parents=True, exist_ok=True)
            created_dirs += 1
            print(f"  mkdir   {rel}")
        else:
            skipped += 1

        if kind == "pkg":
            marker = target / "__init__.py"
            content = INIT_TEMPLATE.format(doc=doc)
        else:
            marker = target / ".gitkeep"
            content = GITKEEP_TEMPLATE.format(path=rel, doc=doc)

        if marker.exists() and marker.stat().st_size > 0:
            skipped += 1
            continue
        if not args.dry_run:
            marker.write_text(content, encoding="utf-8")
        created_files += 1
        print(f"  write   {rel}/{marker.name}")

    total_pkg = sum(1 for k, _, _ in LAYOUT if k == "pkg")
    total_dir = sum(1 for k, _, _ in LAYOUT if k == "dir")

    print()
    print(f"manifest entries : {len(LAYOUT)}  ({total_pkg} python packages, {total_dir} plain dirs)")
    print(f"directories made : {created_dirs}")
    print(f"marker files made: {created_files}")
    print(f"already present  : {skipped}")
    if args.dry_run:
        print("\n(dry run -- nothing was written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
