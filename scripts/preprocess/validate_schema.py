#!/usr/bin/env python3
"""Raw-level schema validator for IO-VNBD -- independent of the pandas loader.

For every CSV under data/raw/IO-VNBD it checks, using plain byte/line processing:
  * the file decodes as cp1252 and has CRLF line endings,
  * the header maps onto the canonical S (24-col) or V (29-col) schema,
  * EVERY data line has exactly the schema's field count (no ragged rows),
  * the body is pure ASCII (only the header may carry non-ASCII unit symbols),
  * the data-row count (reported so the processed Parquet can be cross-checked).

    .venv/Scripts/python.exe scripts/preprocess/validate_schema.py

Writes reports/phase2/schema_validation.json. Exit 0 iff every file passes.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.preprocessing import iovnbd as io  # noqa: E402
from training.preprocessing.config import load_dataset_config  # noqa: E402

CFG = load_dataset_config()
RAW_ROOT = CFG.raw_root
OUT = CFG.reports_dir / "phase2" / "schema_validation.json"


def validate_file(path: Path) -> dict:
    info = io.describe_raw_file(path, RAW_ROOT)
    res = {"rel": info.rel, "kind": info.kind, "session_key": info.session_key, "problems": []}
    data = path.read_bytes()
    try:
        data.decode(io.RAW_ENCODING)
    except UnicodeDecodeError as exc:
        res["problems"].append(f"not cp1252: {exc}")
    lines = data.split(b"\r\n")
    if lines and lines[-1] == b"":
        lines = lines[:-1]
    res["line_ending"] = "CRLF" if b"\r\n" in data else "LF"
    if res["line_ending"] != "CRLF" or b"\n" in data.replace(b"\r\n", b""):
        res["problems"].append("mixed or non-CRLF line endings")
    try:
        names, _ = io.map_header(lines[0].decode(io.RAW_ENCODING), info.kind)
        res["header_ok"] = True
    except io.SchemaError as exc:
        res["header_ok"] = False
        res["problems"].append(str(exc))
        names = io.SCHEMAS[info.kind]
    n_fields = len(names)
    body = lines[1:]
    ragged = [i + 2 for i, ln in enumerate(body) if ln.count(b",") != n_fields - 1]
    blank = sum(1 for ln in body if not ln.strip())
    nonascii = sum(1 for ln in body if any(b > 127 for b in ln))
    res.update(n_data_rows=len(body), n_ragged_rows=len(ragged), ragged_line_numbers=ragged[:10],
               n_blank_rows=blank, n_nonascii_body_rows=nonascii)
    if ragged:
        res["problems"].append(f"{len(ragged)} rows with wrong field count")
    if nonascii:
        res["problems"].append(f"{nonascii} body rows contain non-ASCII bytes")
    res["ok"] = not res["problems"]
    return res


def main() -> int:
    files = sorted(RAW_ROOT.rglob("*.csv"))
    results = [validate_file(f) for f in files]
    bad = [r for r in results if not r["ok"]]
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/preprocess/validate_schema.py",
        "n_files": len(results),
        "n_ok": len(results) - len(bad),
        "n_by_kind": {k: sum(1 for r in results if r["kind"] == k) for k in "SV"},
        "total_data_rows": sum(r["n_data_rows"] for r in results),
        "failures": bad,
        "files": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"{report['n_ok']}/{report['n_files']} files pass schema validation "
          f"({report['n_by_kind']}), {report['total_data_rows']:,} data rows")
    for r in bad:
        print(f"  FAIL {r['rel']}: {r['problems']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
