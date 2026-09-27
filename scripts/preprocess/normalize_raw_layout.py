#!/usr/bin/env python3
"""Normalise the on-disk layout of the IO-VNBD raw dataset -- rename only, never edit.

The dataset arrived as

    data/raw/data raw  IO-VNBD/                       (double space in the name)
        Synchronised V abd S datasets/
            Synchronised V abd S datasets/            (same name nested twice)
                Categorised IOVNB Dataset/...
                Uncategorised IOVNB Dataset/...

and is normalised to

    data/raw/IO-VNBD/
        Categorised IOVNB Dataset/...
        Uncategorised IOVNB Dataset/...

Only the three wrapper directories change. Everything below `Categorised` /
`Uncategorised` keeps its original name, because those names carry metadata (driver,
route id) that the ingestion step parses, and renaming them would destroy provenance.

This is a same-volume directory rename: file bytes are never read or rewritten
(AGENTS.md 5, raw data is read-only). The script proves it by taking a (relative path,
size, mtime) inventory before the move and asserting it is identical afterwards. It is
idempotent: re-running it on an already-normalised tree verifies and exits 0.

    .venv/Scripts/python.exe scripts/preprocess/normalize_raw_layout.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
TARGET = CFG.raw_root
RAW = TARGET.parent
LEGACY_TOP = "data raw  IO-VNBD"
LEGACY_INNER = ("Synchronised V abd S datasets", "Synchronised V abd S datasets")
EXPECTED_CHILDREN = {"Categorised IOVNB Dataset", "Uncategorised IOVNB Dataset"}
REPORT = CFG.reports_dir / "phase2_raw_layout.json"


def inventory(root: Path) -> dict[str, tuple[int, int]]:
    """Relative POSIX path -> (size_bytes, mtime_ns) for every file under root."""
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def find_legacy_root() -> Path | None:
    """Locate the directory holding Categorised/Uncategorised in the as-delivered layout.

    Tolerates the top-level name differing in whitespace ("data raw IO-VNBD" vs the
    double-space original), since the user noted the exact spelling may vary.
    """
    for top in RAW.iterdir():
        if not top.is_dir() or top.name == TARGET.name:
            continue
        if " ".join(top.name.split()).lower() != " ".join(LEGACY_TOP.split()).lower():
            continue
        inner = top.joinpath(*LEGACY_INNER)
        if inner.is_dir():
            return inner
        if {c.name for c in top.iterdir()} >= EXPECTED_CHILDREN:
            return top
    return None


def main() -> int:
    legacy = find_legacy_root()

    if legacy is None:
        if TARGET.is_dir() and {c.name for c in TARGET.iterdir()} >= EXPECTED_CHILDREN:
            inv = inventory(TARGET)
            if not REPORT.exists():
                write_report(inv, moved_this_run=False)
            print(f"already normalised: {TARGET} ({len(inv)} files)")
            return 0
        print(f"ERROR: no IO-VNBD dataset found under {RAW}", file=sys.stderr)
        return 1

    if TARGET.exists():
        print(f"ERROR: both {legacy} and {TARGET} exist; refusing to merge", file=sys.stderr)
        return 1

    children = {c.name for c in legacy.iterdir()}
    if children != EXPECTED_CHILDREN:
        print(f"ERROR: unexpected contents {sorted(children)} in {legacy}", file=sys.stderr)
        return 1

    before = inventory(legacy)
    legacy_top = legacy
    while legacy_top.parent != RAW:
        legacy_top = legacy_top.parent

    TARGET.mkdir()
    for name in sorted(EXPECTED_CHILDREN):
        (legacy / name).rename(TARGET / name)  # same-volume rename: no byte copy

    after = inventory(TARGET)
    if before != after:
        missing = sorted(set(before) - set(after))[:10]
        changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])[:10]
        print(f"ERROR: inventory mismatch; missing={missing} changed={changed}", file=sys.stderr)
        return 1

    # The wrappers must now be empty directories; anything else means we missed data.
    leftovers = [p for p in legacy_top.rglob("*") if p.is_file()]
    if leftovers:
        print(f"ERROR: files left behind in {legacy_top}: {leftovers[:5]}", file=sys.stderr)
        return 1
    write_report(after, moved_this_run=True)
    try:
        shutil.rmtree(legacy_top)
    except PermissionError as exc:
        # Seen on OneDrive: an open shell/sync handle on the now-EMPTY wrapper. The data
        # move is already verified, so this is a warning, not a failure.
        print(f"WARNING: could not remove empty wrapper {legacy_top}: {exc}", file=sys.stderr)
    print(f"moved {len(after)} files: {legacy} -> {TARGET}; inventory identical")
    return 0


def write_report(inv: dict[str, tuple[int, int]], moved_this_run: bool) -> None:
    by_ext: dict[str, int] = {}
    for rel in inv:
        ext = rel.rsplit(".", 1)[-1].lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/preprocess/normalize_raw_layout.py",
        "legacy_layout": f"data/raw/{LEGACY_TOP}/" + "/".join(LEGACY_INNER),
        "normalised_root": TARGET.relative_to(REPO_ROOT).as_posix(),
        "operation": "directory rename (same volume); file bytes untouched",
        "moved_this_run": moved_this_run,
        "files": len(inv),
        "files_by_extension": by_ext,
        "total_bytes": sum(s for s, _ in inv.values()),
    }
    if moved_this_run:
        report["inventory_identical_before_after"] = True
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
