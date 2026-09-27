"""Offline road network: OSM XML -> directed road segments in a local tangent plane.

OFFLINE ONLY. This module reads a file that was cached in advance
(scripts/download/fetch_osm_roads.py); it contains no network code, and a static test
enforces that. Nothing here needs a connection during a GNSS outage.

Representation (all geometry in METRES, never degrees -- docs/map_matching.md sec 7):
* every OSM way of a drivable highway class is cut into straight pieces between
  consecutive nodes, projected to the local ENU tangent plane (``LocalTangentPlane``,
  the single audited geodesy path, AGENTS.md 2.1);
* each piece becomes one DIRECTED segment per permitted travel direction: two-way roads
  give two segments (A->B and B->A), one-way roads one. ``heading`` is the ENU yaw of the
  travel direction, atan2(dN, dE), in radians;
* segments join at OSM nodes; route distances are shortest paths over this directed graph
  (bounded Dijkstra), so one-way restrictions are respected by construction;
* a uniform grid index returns candidate segments near a point.

One-way rules (OSM conventions): ``oneway=yes|true|1`` forward; ``oneway=-1|reverse``
backward; ``junction=roundabout|circular`` and ``highway=motorway`` imply forward unless
``oneway=no``. Ways with ``area=yes`` or ``access=no`` are skipped.
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from navcore.geometry.geodesy import LocalTangentPlane

DRIVABLE = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "service", "living_street", "road", "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link"})
_FORWARD = {"yes", "true", "1"}
_BACKWARD = {"-1", "reverse"}


@dataclass(frozen=True)
class Candidate:
    """A point on a directed segment near an observation (ENU metres)."""

    seg: int
    x: float  # projection, east [m]
    y: float  # projection, north [m]
    frac: float  # position along the segment, 0 = start node, 1 = end node
    dist_m: float  # |observation - projection|


def _direction(tags: dict[str, str]) -> tuple[bool, bool]:
    """(forward allowed, backward allowed) along the way's node order."""
    ow = tags.get("oneway", "").lower()
    if ow in _FORWARD:
        return True, False
    if ow in _BACKWARD:
        return False, True
    if ow == "no":
        return True, True
    if tags.get("junction") in ("roundabout", "circular") or tags.get("highway") == "motorway":
        return True, False
    return True, True


class RoadNetwork:
    """Directed road segments in a local tangent plane, with a grid index and routing."""

    def __init__(self, ltp: LocalTangentPlane, node_xy: np.ndarray, seg_from: np.ndarray,
                 seg_to: np.ndarray, seg_way: np.ndarray, way_tags: dict[int, dict[str, str]],
                 cell_m: float = 50.0) -> None:
        self.ltp = ltp
        self.node_xy = np.asarray(node_xy, dtype=np.float64)
        self.seg_from = np.asarray(seg_from, dtype=np.int64)
        self.seg_to = np.asarray(seg_to, dtype=np.int64)
        self.seg_way = np.asarray(seg_way, dtype=np.int64)
        self.way_tags = way_tags
        a, b = self.node_xy[self.seg_from], self.node_xy[self.seg_to]
        d = b - a
        self.seg_a, self.seg_d = a, d
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.seg_heading = np.arctan2(d[:, 1], d[:, 0])
        self.cell_m = float(cell_m)
        # outgoing segments per node, sorted by segment id (deterministic routing)
        out: dict[int, list[int]] = defaultdict(list)
        for s, n in enumerate(self.seg_from.tolist()):
            out[n].append(s)
        self.out_segs = dict(out)
        self._grid = self._build_grid()

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def from_osm_xml(cls, path: str | Path, origin_deg: tuple[float, float] | None = None,
                     cell_m: float = 50.0) -> RoadNetwork:
        """Parse an OSM XML file (``.osm``) holding drivable ways and their nodes."""
        nodes: dict[int, tuple[float, float]] = {}
        ways: list[tuple[int, list[int], dict[str, str]]] = []
        for _, el in ET.iterparse(str(path), events=("end",)):
            if el.tag == "node":
                nodes[int(el.get("id"))] = (float(el.get("lat")), float(el.get("lon")))
                el.clear()
            elif el.tag == "way":
                tags = {t.get("k"): t.get("v") for t in el.iter("tag")}
                refs = [int(n.get("ref")) for n in el.iter("nd")]
                if (tags.get("highway") in DRIVABLE and tags.get("area") != "yes"
                        and tags.get("access") != "no" and len(refs) >= 2):
                    ways.append((int(el.get("id")), refs, tags))
                el.clear()
        return cls.from_ways(nodes, ways, origin_deg, cell_m)

    @classmethod
    def from_ways(cls, nodes: dict[int, tuple[float, float]],
                  ways: list[tuple[int, list[int], dict[str, str]]],
                  origin_deg: tuple[float, float] | None = None, cell_m: float = 50.0) -> RoadNetwork:
        """Build from ``{node_id: (lat_deg, lon_deg)}`` and ``[(way_id, [node ids], tags)]``."""
        used = sorted({n for _, refs, _ in ways for n in refs if n in nodes})
        if not used:
            raise ValueError("RoadNetwork: no drivable way with known nodes")
        lat = np.array([nodes[n][0] for n in used])
        lon = np.array([nodes[n][1] for n in used])
        if origin_deg is None:
            origin_deg = (float(np.median(lat)), float(np.median(lon)))
        ltp = LocalTangentPlane(math.radians(origin_deg[0]), math.radians(origin_deg[1]), 0.0)
        xy = ltp.geodetic_to_enu(np.radians(lat), np.radians(lon), np.zeros_like(lat))[:, :2]
        idx = {n: i for i, n in enumerate(used)}
        sf, st, sw, way_tags = [], [], [], {}
        for wid, refs, tags in sorted(ways, key=lambda w: w[0]):
            fwd, bwd = _direction(tags)
            chain = [idx[n] for n in refs if n in idx]
            way_tags[wid] = {k: tags[k] for k in ("highway", "name", "oneway", "junction", "tunnel", "bridge")
                             if k in tags}
            for a, b in itertools.pairwise(chain):
                if a == b:
                    continue
                if fwd:
                    sf.append(a), st.append(b), sw.append(wid)
                if bwd:
                    sf.append(b), st.append(a), sw.append(wid)
        return cls(ltp, xy, np.array(sf), np.array(st), np.array(sw), way_tags, cell_m)

    # ---- offline cache ---------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, node_xy=self.node_xy, seg_from=self.seg_from, seg_to=self.seg_to,
                            seg_way=self.seg_way,
                            origin=np.array([self.ltp.lat0_rad, self.ltp.lon0_rad, self.ltp.h0_m]),
                            cell_m=np.array(self.cell_m),
                            way_tags=np.array(json.dumps({str(k): v for k, v in self.way_tags.items()})))

    @classmethod
    def load(cls, path: str | Path) -> RoadNetwork:
        z = np.load(path, allow_pickle=False)
        o = z["origin"]
        tags = {int(k): v for k, v in json.loads(str(z["way_tags"])).items()}
        return cls(LocalTangentPlane(float(o[0]), float(o[1]), float(o[2])), z["node_xy"], z["seg_from"],
                   z["seg_to"], z["seg_way"], tags, float(z["cell_m"]))

    # ---- spatial queries -------------------------------------------------------------------
    def _build_grid(self) -> dict[tuple[int, int], np.ndarray]:
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        b = self.seg_a + self.seg_d
        lo = np.floor(np.minimum(self.seg_a, b) / self.cell_m).astype(np.int64)
        hi = np.floor(np.maximum(self.seg_a, b) / self.cell_m).astype(np.int64)
        for s in range(len(self.seg_len)):
            for i in range(lo[s, 0], hi[s, 0] + 1):
                for j in range(lo[s, 1], hi[s, 1] + 1):
                    grid[(int(i), int(j))].append(s)
        return {k: np.array(v, dtype=np.int64) for k, v in grid.items()}

    def segments_near(self, x: float, y: float, radius_m: float) -> np.ndarray:
        c = self.cell_m
        i0, i1 = math.floor((x - radius_m) / c), math.floor((x + radius_m) / c)
        j0, j1 = math.floor((y - radius_m) / c), math.floor((y + radius_m) / c)
        hits = [self._grid[(i, j)] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1) if (i, j) in self._grid]
        return np.unique(np.concatenate(hits)) if hits else np.empty(0, dtype=np.int64)

    def project(self, segs: np.ndarray, x: float, y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Closest point on each segment: (projection xy [n,2], frac [n], distance [n])."""
        a, d = self.seg_a[segs], self.seg_d[segs]
        L2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
        frac = np.clip(np.einsum("ij,ij->i", np.array([x, y]) - a, d) / L2, 0.0, 1.0)
        p = a + frac[:, None] * d
        return p, frac, np.hypot(p[:, 0] - x, p[:, 1] - y)

    def candidates(self, x: float, y: float, radius_m: float) -> list[Candidate]:
        """Every directed segment within ``radius_m`` of (x, y), ordered by segment id."""
        segs = self.segments_near(x, y, radius_m)
        if segs.size == 0:
            return []
        p, frac, dist = self.project(segs, x, y)
        keep = dist <= radius_m
        return [Candidate(int(s), float(px), float(py), float(f), float(dd))
                for s, (px, py), f, dd in zip(segs[keep], p[keep], frac[keep], dist[keep], strict=True)]

    # ---- routing ---------------------------------------------------------------------------
    def distances_from_node(self, node: int, cutoff_m: float) -> dict[int, float]:
        """Shortest directed distances from ``node`` to every node within ``cutoff_m``."""
        dist = {node: 0.0}
        heap = [(0.0, node)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, math.inf):
                continue
            for s in self.out_segs.get(n, ()):
                nd = d + float(self.seg_len[s])
                m = int(self.seg_to[s])
                if nd <= cutoff_m and nd < dist.get(m, math.inf):
                    dist[m] = nd
                    heapq.heappush(heap, (nd, m))
        return dist

    def route_distance(self, a: Candidate, b: Candidate, cutoff_m: float,
                       cache: dict[int, dict[int, float]] | None = None) -> float:
        """Driving distance from candidate ``a`` to ``b`` along permitted directions; inf if
        no route exists within ``cutoff_m`` (an impossible transition, not a long one)."""
        if a.seg == b.seg and b.frac >= a.frac:
            return (b.frac - a.frac) * float(self.seg_len[a.seg])
        rest = (1.0 - a.frac) * float(self.seg_len[a.seg])
        into = b.frac * float(self.seg_len[b.seg])
        if rest + into > cutoff_m:
            return math.inf
        src = int(self.seg_to[a.seg])
        if cache is not None:
            if src not in cache:
                cache[src] = self.distances_from_node(src, cutoff_m)
            table = cache[src]
        else:
            table = self.distances_from_node(src, cutoff_m)
        mid = table.get(int(self.seg_from[b.seg]), math.inf)
        total = rest + mid + into
        return total if total <= cutoff_m else math.inf

    # ---- conversions (through the single geodesy path) --------------------------------------
    def to_xy(self, lat_rad: float, lon_rad: float) -> tuple[float, float, float]:
        e, n, u = self.ltp.geodetic_to_enu(lat_rad, lon_rad, 0.0)
        return float(e), float(n), float(u)

    def to_geodetic(self, x: float, y: float, u: float = 0.0) -> tuple[float, float]:
        lat, lon, _ = self.ltp.enu_to_geodetic(np.array([x, y, u]))
        return float(lat), float(lon)

    def way_of(self, seg: int) -> dict[str, str | int]:
        wid = int(self.seg_way[seg])
        return {"way_id": wid, **self.way_tags.get(wid, {})}

    @property
    def n_segments(self) -> int:
        return len(self.seg_len)


__all__ = ["DRIVABLE", "Candidate", "RoadNetwork"]
