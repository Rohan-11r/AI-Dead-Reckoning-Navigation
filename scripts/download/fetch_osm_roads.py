#!/usr/bin/env python3
"""ONE-TIME, OFFLINE-PREPARATION download of the drivable OSM road network (Phase 8).

    .venv/Scripts/python.exe scripts/download/fetch_osm_roads.py                      # Coventry (Phase 8)
    .venv/Scripts/python.exe scripts/download/fetch_osm_roads.py --name nagpur --bbox 21.03 78.97 21.22 79.18
    .venv/Scripts/python.exe scripts/download/fetch_osm_roads.py --city "Nagpur, India"

This is the ONLY network access in the map-matching stack, and it runs at data-preparation
time, never during navigation: the matcher (navcore.map_matching) reads the cached file and
has no network code at all (enforced by a static test).

Default area: the Coventry core, covering the clock-verified VAL drives (S3a-c), the TRAIN
drives S1/S2/S4 and the held-out TEST drives M/Y1 (the TEST drives' extent is used only so
their one-time final evaluation can run offline later; nothing about them is inspected here).
Output: data/raw/osm/coventry_drivable.osm (gitignored) + a tracked provenance manifest
reports/phase8/osm_manifest.json (query, bbox, time, bytes, SHA-256, licence).

Any other area (field testing): --bbox SOUTH WEST NORTH EAST in degrees, or --city, which
takes the bounding box of the first OpenStreetMap Nominatim match (printed and recorded; pass
--bbox when the city's administrative box is not the area you will drive). Output:
data/raw/osm/<name>_drivable.osm + reports/osm/<name>_osm_manifest.json. The Coventry files
are never touched by a named area.

Map data (c) OpenStreetMap contributors, available under the Open Database License (ODbL).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data" / "raw" / "osm" / "coventry_drivable.osm"
MANIFEST = REPO_ROOT / "reports" / "phase8" / "osm_manifest.json"
ENDPOINT = "https://overpass.kumi.systems/api/interpreter"
GEOCODER = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SIH26168-dead-reckoning/phase8 (offline cache)"
BBOX = (52.34, -1.62, 52.58, -1.21)  # south, west, north, east (degrees)
HIGHWAYS = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|"
            "living_street|road|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
# Larger boxes time out on the public Overpass server and give a bundle too big for a phone's
# heap (the 0.24 deg x 0.41 deg Coventry box is already 13 MB); --allow-large overrides.
MAX_SPAN_DEG = 0.6
MIN_SPAN_DEG = 0.005  # a geocoder "box" smaller than ~500 m is a point, not a city


def build_query(bbox_deg: tuple[float, float, float, float]) -> str:
    """Overpass QL for every drivable way in (south, west, north, east) degrees, with its nodes."""
    s, w, n, e = bbox_deg
    return (f'[out:xml][timeout:300];(way["highway"~"^({HIGHWAYS})$"]'
            f"({s},{w},{n},{e}););(._;>;);out body;")


QUERY = build_query(BBOX)


def validate_bbox(bbox_deg: tuple[float, float, float, float], allow_large: bool = False) -> tuple[float, float, float, float]:
    """(south, west, north, east) degrees, checked; raises ValueError on a malformed box."""
    s, w, n, e = (float(v) for v in bbox_deg)
    if not (-90.0 <= s < n <= 90.0):
        raise ValueError(f"need -90 <= south < north <= 90, got south={s}, north={n}")
    if not (-180.0 <= w < e <= 180.0):
        raise ValueError(f"need -180 <= west < east <= 180 (no antimeridian boxes), got west={w}, east={e}")
    if min(n - s, e - w) < MIN_SPAN_DEG:
        raise ValueError(f"box {n - s:.4f} deg x {e - w:.4f} deg is too small to hold a road network")
    if max(n - s, e - w) > MAX_SPAN_DEG and not allow_large:
        raise ValueError(f"box {n - s:.3f} deg x {e - w:.3f} deg exceeds {MAX_SPAN_DEG} deg on a side; "
                         "pass a tighter --bbox around the test routes (or --allow-large)")
    return s, w, n, e


def area_name(text: str) -> str:
    """A file-safe area name: 'Nagpur, India' -> 'nagpur'."""
    name = re.sub(r"[^a-z0-9]+", "_", text.split(",")[0].strip().lower()).strip("_")
    if not name:
        raise ValueError(f"cannot derive an area name from {text!r}; pass --name")
    return name


def bbox_from_nominatim(hit: dict) -> tuple[float, float, float, float]:
    """Nominatim's boundingbox is [south, north, west, east] as strings -> (s, w, n, e) floats."""
    s, n, w, e = (float(v) for v in hit["boundingbox"])
    return s, w, n, e


def geocode_city(city: str) -> tuple[tuple[float, float, float, float], dict]:
    """Bounding box of the first Nominatim match for `city`, plus the fields worth recording."""
    url = GEOCODER + "?" + urllib.parse.urlencode({"q": city, "format": "json", "limit": 1})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        hits = json.loads(r.read())
    if not hits:
        raise SystemExit(f"geocoder found nothing for {city!r}; pass --bbox SOUTH WEST NORTH EAST")
    hit = hits[0]
    record = {"geocoder": GEOCODER, "geocoder_query": city, "display_name": hit.get("display_name"),
              "osm_type": hit.get("osm_type"), "osm_id": hit.get("osm_id"), "boundingbox": hit["boundingbox"]}
    return bbox_from_nominatim(hit), record


def area_paths(name: str) -> tuple[Path, Path]:
    """(OSM cache, provenance manifest) for an area; 'coventry' keeps its Phase 8 paths."""
    if name == "coventry":
        return OUT, MANIFEST
    return (REPO_ROOT / "data" / "raw" / "osm" / f"{name}_drivable.osm",
            REPO_ROOT / "reports" / "osm" / f"{name}_osm_manifest.json")


def fetch(bbox_deg: tuple[float, float, float, float], out: Path, manifest: Path,
          extra: dict | None = None) -> Path:
    """Download the drivable network in bbox_deg to `out` once; record provenance in `manifest`.

    An existing cache is reused only if its manifest records the same box; a different box
    under the same name is refused rather than silently mixed up.
    """
    if out.exists():
        if manifest.exists():
            old = tuple(json.loads(manifest.read_text(encoding="utf-8"))["bbox_s_w_n_e_deg"])
            if old != tuple(bbox_deg):
                raise SystemExit(f"{out} was fetched for bbox {old}, not {tuple(bbox_deg)}: delete it "
                                 "(and its manifest) or use another --name")
        print(f"already present: {out} ({out.stat().st_size / 1e6:.1f} MB) -- not re-downloading")
        return out
    query = build_query(bbox_deg)
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(ENDPOINT, data=urllib.parse.urlencode({"data": query}).encode(),
                                 headers={"User-Agent": USER_AGENT})
    started = datetime.now(UTC)
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    if b"<osm" not in body[:500]:
        raise SystemExit(f"unexpected response: {body[:300]!r}")
    if b"<way" not in body:
        raise SystemExit(f"Overpass returned no drivable ways in bbox {tuple(bbox_deg)}: check the box")
    out.write_bytes(body)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "file": str(out.relative_to(REPO_ROOT)).replace("\\", "/"), "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(), "downloaded_utc": started.isoformat(timespec="seconds"),
        "endpoint": ENDPOINT, "query": query, "bbox_s_w_n_e_deg": tuple(bbox_deg),
        **(extra or {}),
        "licence": "Map data (c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)",
        "note": "one-time offline cache; the matcher never accesses the network"}, indent=1) + "\n",
        encoding="utf-8")
    print(f"wrote {out} ({len(body) / 1e6:.1f} MB) and {manifest}")
    return out


def add_area_args(ap: argparse.ArgumentParser) -> None:
    """--name / --bbox / --city / --allow-large, shared with scripts/export/export_road_bundle.py."""
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--bbox", nargs=4, type=float, metavar=("SOUTH", "WEST", "NORTH", "EAST"),
                   help="area to fetch, WGS84 degrees (e.g. 21.03 78.97 21.22 79.18)")
    g.add_argument("--city", help='place name for the Nominatim geocoder, e.g. "Nagpur, India"')
    ap.add_argument("--name", help="area name used in file names (default: from --city, else 'coventry')")
    ap.add_argument("--allow-large", action="store_true", help=f"permit a box wider than {MAX_SPAN_DEG} deg")


def resolve_area(args: argparse.Namespace) -> tuple[str, tuple[float, float, float, float], dict]:
    """(name, checked bbox, extra provenance) from parsed add_area_args() arguments."""
    extra: dict = {}
    if args.city:
        bbox, extra = geocode_city(args.city)
        print(f"geocoded {args.city!r} -> {extra['display_name']}\n  bbox S W N E = {bbox}")
        name = args.name or area_name(args.city)
    elif args.bbox:
        if not args.name:
            raise SystemExit("--bbox needs --name (used in the output file names)")
        bbox, name = tuple(args.bbox), area_name(args.name)
    else:
        bbox, name = BBOX, area_name(args.name or "coventry")
        if name != "coventry":
            raise SystemExit(f"--name {name} needs --bbox or --city")
    if name == "coventry" and tuple(bbox) != BBOX:
        raise SystemExit("the name 'coventry' is reserved for the Phase 8 box; pick another --name")
    try:
        bbox = validate_bbox(bbox, args.allow_large)
    except ValueError as x:
        raise SystemExit(f"bad area: {x}") from None
    return name, bbox, extra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_area_args(ap)
    name, bbox, extra = resolve_area(ap.parse_args())
    out, manifest = area_paths(name)
    fetch(bbox, out, manifest, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
