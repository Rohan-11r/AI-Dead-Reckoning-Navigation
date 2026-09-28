"""Area selection for the offline road bundle (scripts/download/fetch_osm_roads.py).

Pure argument handling only -- no network access. The Nominatim record below is a
hand-written SYNTHETIC fixture in the documented response shape, not a recorded response.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fetch_osm_roads", REPO / "scripts" / "download" / "fetch_osm_roads.py")
osm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(osm)


def _args(*argv: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    osm.add_area_args(ap)
    return ap.parse_args(list(argv))


def test_coventry_query_is_the_recorded_phase8_query():
    """Refactoring must not change the query behind reports/phase8/osm_manifest.json."""
    recorded = json.loads((REPO / "reports" / "phase8" / "osm_manifest.json").read_text(encoding="utf-8"))
    assert osm.QUERY == recorded["query"]
    assert list(osm.BBOX) == recorded["bbox_s_w_n_e_deg"]


def test_default_area_is_coventry_with_its_phase8_paths():
    name, bbox, extra = osm.resolve_area(_args())
    assert (name, bbox, extra) == ("coventry", osm.BBOX, {})
    assert osm.area_paths("coventry") == (osm.OUT, osm.MANIFEST)


def test_named_area_never_writes_the_coventry_files():
    out, manifest = osm.area_paths("nagpur")
    assert out.name == "nagpur_drivable.osm" and manifest.name == "nagpur_osm_manifest.json"
    assert manifest != osm.MANIFEST


def test_bbox_area():
    name, bbox, _ = osm.resolve_area(_args("--name", "Nagpur", "--bbox", "21.03", "78.97", "21.22", "79.18"))
    assert name == "nagpur" and bbox == (21.03, 78.97, 21.22, 79.18)
    assert "(21.03,78.97,21.22,79.18)" in osm.build_query(bbox)


@pytest.mark.parametrize("argv", [
    ("--bbox", "21.03", "78.97", "21.22", "79.18"),                  # no --name
    ("--name", "nagpur"),                                             # no area
    ("--name", "coventry", "--bbox", "21.03", "78.97", "21.22", "79.18"),  # reserved name
    ("--name", "x", "--bbox", "21.22", "78.97", "21.03", "79.18"),   # south > north
    ("--name", "x", "--bbox", "21.0", "79.18", "21.2", "78.97"),     # west > east
    ("--name", "x", "--bbox", "20.0", "78.0", "22.0", "80.0"),       # 2 deg: too large
    ("--name", "x", "--bbox", "21.0", "79.0", "21.001", "79.001"),   # a point
])
def test_bad_areas_are_refused(argv):
    with pytest.raises(SystemExit):
        osm.resolve_area(_args(*argv))


def test_allow_large():
    _, bbox, _ = osm.resolve_area(_args("--name", "x", "--bbox", "20", "78", "22", "80", "--allow-large"))
    assert bbox == (20.0, 78.0, 22.0, 80.0)


def test_nominatim_box_order_is_s_n_w_e():
    hit = {"boundingbox": ["21.0", "21.3", "78.9", "79.2"]}  # synthetic, documented order
    assert osm.bbox_from_nominatim(hit) == (21.0, 78.9, 21.3, 79.2)


def test_area_name():
    assert osm.area_name("Nagpur, Maharashtra, India") == "nagpur"
    assert osm.area_name("New Delhi") == "new_delhi"
    with pytest.raises(ValueError):
        osm.area_name(", India")
