#!/usr/bin/env python3
"""Export an OSM road network as the phone's offline road bundle (Phase 13).

    .venv/Scripts/python.exe scripts/export/export_road_bundle.py                      # Coventry (cached OSM)
    .venv/Scripts/python.exe scripts/export/export_road_bundle.py --city "Nagpur, India"
    .venv/Scripts/python.exe scripts/export/export_road_bundle.py --name nagpur --bbox 21.03 78.97 21.22 79.18
    .venv/Scripts/python.exe scripts/export/export_road_bundle.py --name x --osm path/to/file.osm

With --city or --bbox the drivable network is first downloaded once (Overpass) through
scripts/download/fetch_osm_roads.py -- the same query as Phase 8 -- then exported. That
download is data preparation; the app and the matcher never touch the network.

Writes models/roads/<name>.roads.bin (gitignored: a derived, regenerable artefact of ODbL
data) and a small manifest models/roads/<name>.roads.json (SHA-256, sizes, origin, bbox,
source), which the app's Gradle task bundles next to it. The app loads exactly one bundle:
keep only the one for the area you will drive in models/roads/. Map data (c) OpenStreetMap
contributors, ODbL.
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "download"))

import fetch_osm_roads as osm  # noqa: E402

from navcore.map_matching.bundle import read_bundle, write_bundle  # noqa: E402
from navcore.map_matching.road_network import RoadNetwork  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    osm.add_area_args(ap)
    ap.add_argument("--osm", help="export an existing OSM XML file instead of the area's cache")
    args = ap.parse_args()

    if args.osm:
        if args.bbox or args.city:
            raise SystemExit("--osm exports a given file; do not combine it with --bbox / --city")
        name = osm.area_name(args.name or "coventry")
        src, src_manifest = Path(args.osm).resolve(), osm.area_paths(name)[1]
    else:
        name, bbox, extra = osm.resolve_area(args)
        src, src_manifest = osm.area_paths(name)
        osm.fetch(bbox, src, src_manifest, extra)  # no-op when already cached for this box

    net = RoadNetwork.from_osm_xml(src)
    if net.n_segments == 0:
        raise SystemExit(f"{src}: no drivable road segments -- nothing to match against")
    out = REPO_ROOT / "models" / "roads" / f"{name}.roads.bin"
    write_bundle(net, out)
    back = read_bundle(out)  # lossless round trip, or fail here
    assert back.n_segments == net.n_segments and (back.node_xy == net.node_xy).all()
    data = out.read_bytes()
    prov = json.loads(src_manifest.read_text(encoding="utf-8")) if src_manifest.exists() else None
    manifest = {"file": out.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "nodes": len(net.node_xy), "directed_segments": net.n_segments, "ways": len(net.way_tags),
                "origin_lat_rad": net.ltp.lat0_rad, "origin_lon_rad": net.ltp.lon0_rad,
                "bbox_s_w_n_e_deg": prov["bbox_s_w_n_e_deg"] if prov else None,
                "source": str(src.relative_to(REPO_ROOT) if src.is_relative_to(REPO_ROOT) else src).replace("\\", "/"),
                "source_manifest": str(src_manifest.relative_to(REPO_ROOT)).replace("\\", "/") if prov else None,
                "licence": "Map data (c) OpenStreetMap contributors, ODbL 1.0",
                "generated": datetime.now().isoformat(timespec="seconds"),
                "script": "scripts/export/export_road_bundle.py"}
    (out.with_suffix("").with_suffix(".roads.json")).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(data) / 1e6:.1f} MB, {net.n_segments} segments)")

    others = sorted(p.name for p in out.parent.glob("*.roads.bin") if p != out)
    if others:
        print(f"WARNING: models/roads/ also holds {', '.join(others)}. The app refuses to choose between "
              f"bundles (no map matching); move the other .roads.bin files out before building.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
