#!/usr/bin/env python3
"""Export the cached OSM road network as the phone's offline road bundle (Phase 11).

    .venv/Scripts/python.exe scripts/export/export_road_bundle.py [--osm data/raw/osm/coventry_drivable.osm]

Writes models/roads/<name>.roads.bin (gitignored: a derived, regenerable artefact of ODbL
data) and a small tracked manifest models/roads/<name>.roads.json (SHA-256, sizes,
origin, source), which the app's Gradle task bundles next to it. Map data (c)
OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from navcore.map_matching.bundle import read_bundle, write_bundle  # noqa: E402
from navcore.map_matching.road_network import RoadNetwork  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", default=str(REPO_ROOT / "data" / "raw" / "osm" / "coventry_drivable.osm"))
    ap.add_argument("--name", default="coventry")
    args = ap.parse_args()
    net = RoadNetwork.from_osm_xml(args.osm)
    out = REPO_ROOT / "models" / "roads" / f"{args.name}.roads.bin"
    write_bundle(net, out)
    back = read_bundle(out)  # lossless round trip, or fail here
    assert back.n_segments == net.n_segments and (back.node_xy == net.node_xy).all()
    data = out.read_bytes()
    manifest = {"file": out.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "nodes": len(net.node_xy), "directed_segments": net.n_segments, "ways": len(net.way_tags),
                "origin_lat_rad": net.ltp.lat0_rad, "origin_lon_rad": net.ltp.lon0_rad,
                "source": str(Path(args.osm).relative_to(REPO_ROOT)).replace("\\", "/"),
                "source_manifest": "reports/phase8/osm_manifest.json",
                "licence": "Map data (c) OpenStreetMap contributors, ODbL 1.0",
                "generated": datetime.now().isoformat(timespec="seconds"),
                "script": "scripts/export/export_road_bundle.py"}
    (out.with_suffix("").with_suffix(".roads.json")).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(data) / 1e6:.1f} MB, {net.n_segments} segments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
