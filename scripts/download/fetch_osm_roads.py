#!/usr/bin/env python3
"""ONE-TIME, OFFLINE-PREPARATION download of the drivable OSM road network (Phase 8).

    .venv/Scripts/python.exe scripts/download/fetch_osm_roads.py

This is the ONLY network access in the map-matching stack, and it runs at data-preparation
time, never during navigation: the matcher (navcore.map_matching) reads the cached file and
has no network code at all (enforced by a static test).

Area: the Coventry core, covering the clock-verified VAL drives (S3a-c), the TRAIN drives
S1/S2/S4 and the held-out TEST drives M/Y1 (the TEST drives' extent is used only so their
one-time final evaluation can run offline later; nothing about them is inspected here).

Output: data/raw/osm/coventry_drivable.osm (gitignored) + a tracked provenance manifest
reports/phase8/osm_manifest.json (query, bbox, time, bytes, SHA-256, licence).
Map data (c) OpenStreetMap contributors, available under the Open Database License (ODbL).
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data" / "raw" / "osm" / "coventry_drivable.osm"
MANIFEST = REPO_ROOT / "reports" / "phase8" / "osm_manifest.json"
ENDPOINT = "https://overpass-api.de/api/interpreter"
BBOX = (52.34, -1.62, 52.58, -1.21)  # south, west, north, east (degrees)
HIGHWAYS = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|"
            "living_street|road|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
QUERY = (f'[out:xml][timeout:300];(way["highway"~"^({HIGHWAYS})$"]'
         f"({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]}););(._;>;);out body;")


def main() -> int:
    if OUT.exists():
        print(f"already present: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) -- not re-downloading")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(ENDPOINT, data=urllib.parse.urlencode({"data": QUERY}).encode(),
                                 headers={"User-Agent": "SIH26168-dead-reckoning/phase8 (offline cache)"})
    started = datetime.now(UTC)
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    if b"<osm" not in body[:500]:
        raise SystemExit(f"unexpected response: {body[:300]!r}")
    OUT.write_bytes(body)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "file": str(OUT.relative_to(REPO_ROOT)).replace("\\", "/"), "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(), "downloaded_utc": started.isoformat(timespec="seconds"),
        "endpoint": ENDPOINT, "query": QUERY, "bbox_s_w_n_e_deg": BBOX,
        "licence": "Map data (c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
        "note": "one-time offline cache; the matcher never accesses the network"}, indent=1) + "\n",
        encoding="utf-8")
    print(f"wrote {OUT} ({len(body) / 1e6:.1f} MB) and {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
