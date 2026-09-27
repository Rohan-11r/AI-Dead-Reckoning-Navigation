"""Map matching (Phase 8): offline road network + HMM matcher.

All road networks and trajectories here are SYNTHETIC with known answers (AGENTS.md 2.2),
built in metres and converted to lat/lon through the single geodesy path. Every
"the HMM does X" test also shows that NEAREST-ROAD SNAPPING fails the same scenario, so the
test cannot pass on a scenario that does not discriminate.
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import numpy as np
import pytest

from navcore.filtering.ekf import N_STATES, ErrorStateEKF, NoiseParams
from navcore.geometry.geodesy import LocalTangentPlane
from navcore.geometry.quaternion import q_from_euler
from navcore.ins.mechanization import NavState
from navcore.map_matching import (
    DeadReckoningMapMatcher,
    HmmMapMatcher,
    MatcherConfig,
    RoadNetwork,
)
from navcore.map_matching.outage import course_and_sigma
from navcore.map_matching.road_network import Candidate

ORIGIN = (52.4, -1.5)
LTP = LocalTangentPlane(math.radians(ORIGIN[0]), math.radians(ORIGIN[1]), 0.0)
EAST, NORTH, WEST = 0.0, math.pi / 2, math.pi


def build(nodes_xy: dict[int, tuple[float, float]], ways: list[tuple[int, list[int], dict]]) -> RoadNetwork:
    ll = {}
    for n, (x, y) in nodes_xy.items():
        lat, lon, _ = LTP.enu_to_geodetic(np.array([x, y, 0.0]))
        ll[n] = (math.degrees(float(lat)), math.degrees(float(lon)))
    return RoadNetwork.from_ways(ll, ways, origin_deg=ORIGIN, cell_m=50.0)


def crossroads() -> RoadNetwork:
    """Way 1 runs west-east along y = 0, way 2 south-north along x = 0; they meet at node 0."""
    nodes = {1: (-600, 0), 0: (0, 0), 2: (600, 0), 3: (0, -600), 4: (0, 600)}
    return build(nodes, [(1, [1, 0, 2], {"highway": "primary"}), (2, [3, 0, 4], {"highway": "residential"})])


def nearest_way(net: RoadNetwork, x: float, y: float) -> int:
    c = min(net.candidates(x, y, 200.0), key=lambda k: (k.dist_m, k.seg))
    return int(net.seg_way[c.seg])


def run(net, obs, heading, speed=15.0, sigma=15.0, cfg=None):
    m = HmmMapMatcher(net, cfg)
    out = []
    for k, (x, y) in enumerate(obs):
        h = heading[k] if isinstance(heading, list | np.ndarray) else heading
        out.append(m.step(float(k), x, y, np.eye(2) * sigma ** 2, h, speed))
    return m, out


# ---- network --------------------------------------------------------------------------------

OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
 <node id="1" lat="52.4000" lon="-1.5000"/><node id="2" lat="52.4010" lon="-1.5000"/>
 <node id="3" lat="52.4020" lon="-1.5000"/><node id="4" lat="52.4020" lon="-1.4990"/>
 <node id="5" lat="52.4010" lon="-1.4990"/>
 <way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/></way>
 <way id="11"><nd ref="2"/><nd ref="3"/><tag k="highway" v="primary"/><tag k="oneway" v="yes"/></way>
 <way id="12"><nd ref="3"/><nd ref="4"/><tag k="highway" v="tertiary"/><tag k="oneway" v="-1"/></way>
 <way id="13"><nd ref="4"/><nd ref="5"/><nd ref="2"/><nd ref="4"/><tag k="highway" v="unclassified"/>
   <tag k="junction" v="roundabout"/></way>
 <way id="14"><nd ref="1"/><nd ref="5"/><tag k="highway" v="motorway"/><tag k="oneway" v="no"/></way>
 <way id="15"><nd ref="1"/><nd ref="4"/><tag k="highway" v="footway"/></way>
 <way id="16"><nd ref="1"/><nd ref="3"/><tag k="highway" v="service"/><tag k="area" v="yes"/></way>
 <way id="17"><nd ref="2"/><nd ref="99"/><nd ref="5"/><tag k="highway" v="service"/></way>
</osm>"""


def test_osm_xml_parsing_and_one_way_rules(tmp_path):
    p = tmp_path / "t.osm"
    p.write_text(OSM, encoding="utf-8")
    net = RoadNetwork.from_osm_xml(p)
    per_way = {w: [] for w in (10, 11, 12, 13, 14, 17)}
    for s in range(net.n_segments):
        per_way[int(net.seg_way[s])].append(s)
    assert len(per_way[10]) == 2  # two-way
    assert len(per_way[11]) == 1  # oneway=yes
    assert len(per_way[12]) == 1  # oneway=-1: only against the node order
    assert len(per_way[13]) == 3  # roundabout: implied one-way, 3 pieces
    assert len(per_way[14]) == 2  # motorway with explicit oneway=no
    assert len(per_way[17]) == 2  # unknown node 99 skipped -> one piece, two-way
    assert 15 not in net.way_tags and 16 not in net.way_tags  # footway, area excluded
    (s12,) = per_way[12]
    assert math.cos(net.seg_heading[s12]) == pytest.approx(-1.0, abs=1e-4)  # westward: 4 -> 3, against node order
    (s11,) = per_way[11]
    assert net.seg_heading[s11] == pytest.approx(NORTH, abs=0.01)


def test_candidates_equal_brute_force_projection():
    rng = np.random.default_rng(0)
    nodes = {i: tuple(rng.uniform(-400, 400, 2)) for i in range(60)}
    ways = [(100 + i, [2 * i, 2 * i + 1], {"highway": "residential"}) for i in range(30)]
    net = build(nodes, ways)
    for _ in range(50):
        x, y = rng.uniform(-400, 400, 2)
        got = {(c.seg, round(c.dist_m, 9)) for c in net.candidates(x, y, 80.0)}
        allseg = np.arange(net.n_segments)
        _, _, d = net.project(allseg, x, y)
        want = {(int(s), round(float(dd), 9)) for s, dd in zip(allseg, d, strict=True) if dd <= 80.0}
        assert got == want


def test_route_distance_respects_topology_and_one_way():
    nodes = {0: (0, 0), 1: (100, 0), 2: (200, 0), 5: (0, 1000), 6: (100, 1000)}
    net = build(nodes, [(1, [0, 1], {"highway": "residential"}),
                        (2, [1, 2], {"highway": "residential", "oneway": "yes"}),
                        (3, [5, 6], {"highway": "residential"})])  # disconnected road

    def cand(way, fwd, frac):
        s = next(s for s in range(net.n_segments) if net.seg_way[s] == way
                 and (math.cos(net.seg_heading[s]) > 0) == fwd)
        p = net.node_xy[net.seg_from[s]] + frac * net.seg_d[s]
        return Candidate(s, float(p[0]), float(p[1]), frac, 0.0)

    a, b = cand(1, True, 0.2), cand(1, True, 0.7)
    assert net.route_distance(a, b, 500) == pytest.approx(50.0)
    # no turning round mid-segment: going back means driving on to node 1, turning there,
    # and coming back (30 + 100 via node 0's dead end + 20)
    assert net.route_distance(b, a, 500) == pytest.approx(150.0)
    assert net.route_distance(b, a, 120) == math.inf  # beyond the cutoff
    c = cand(2, True, 0.5)
    assert net.route_distance(a, c, 500) == pytest.approx(80 + 50)
    back = cand(1, False, 0.5)  # westbound on way 1
    assert net.route_distance(c, back, 500) == math.inf  # one-way: cannot come back west
    assert net.route_distance(a, cand(3, True, 0.5), 5000) == math.inf  # disconnected


# ---- HMM behaviour --------------------------------------------------------------------------

def test_crossroads_hmm_stays_on_the_through_road_where_nearest_snapping_jumps():
    net = crossroads()
    xs = np.arange(-150.0, 151.0, 15.0)
    obs = [(x, 12.0) for x in xs]  # dead-reckoning drifted 12 m north of the road
    snapped = [nearest_way(net, x, y) for x, y in obs]
    assert 2 in snapped  # nearest-road snapping jumps onto the crossing road at the junction
    _, out = run(net, obs, EAST)
    assert all(r.matched and r.way["way_id"] == 1 for r in out)
    assert all(abs(r.y_m) < 0.5 for r in out)  # constrained back onto the road
    assert min(r.map_match_confidence for r in out) > 0.5


def test_hmm_follows_a_real_turn_at_the_junction():
    net = crossroads()
    rng = np.random.default_rng(1)
    path = [(x, 0.0, EAST) for x in np.arange(-150.0, 0.0, 15.0)] + [(0.0, y, NORTH) for y in np.arange(0.0, 151.0, 15.0)]
    obs = [(x + rng.normal(0, 6), y + rng.normal(0, 6)) for x, y, _ in path]
    heading = [h for *_, h in path]
    m, out = run(net, obs, heading)
    ways = [r.way["way_id"] for r in out]
    assert ways[:8] == [1] * 8 and ways[-8:] == [2] * 8
    path_c = m.decode()
    for (_, a), (_, b) in itertools.pairwise(path_c):
        assert math.isfinite(net.route_distance(a, b, 1000.0))  # no impossible transition


def test_parallel_roads_never_flip_across_without_a_connection():
    nodes = {1: (-3000, 0), 2: (3000, 0), 3: (-3000, 30), 4: (3000, 30)}
    net = build(nodes, [(1, [1, 2], {"highway": "primary"}), (2, [3, 4], {"highway": "primary"})])
    obs = [(x, 25.0 * k / 40) for k, x in enumerate(np.arange(0.0, 601.0, 15.0))]  # drift 0 -> 25 m
    assert any(nearest_way(net, x, y) == 2 for x, y in obs)
    _, out = run(net, obs, EAST)
    assert all(r.way["way_id"] == 1 for r in out)


def test_dual_carriageway_direction_of_travel_decides():
    nodes = {1: (-1000, 0), 2: (1000, 0), 3: (1000, 20), 4: (-1000, 20)}
    net = build(nodes, [(1, [1, 2], {"highway": "trunk", "oneway": "yes"}),    # eastbound, y = 0
                        (2, [3, 4], {"highway": "trunk", "oneway": "yes"})])   # westbound, y = 20
    obs = [(x, 7.0) for x in np.arange(300.0, -1.0, -15.0)]  # heading WEST, but nearer y = 0
    assert all(nearest_way(net, x, y) == 1 for x, y in obs)
    _, out = run(net, obs, WEST, sigma=10.0)
    assert all(r.way["way_id"] == 2 for r in out)


def test_off_network_suspends_then_reenters():
    net = crossroads()
    cfg = MatcherConfig(r_max_m=60.0)
    on = [(x, 3.0) for x in np.arange(-500.0, -380.0, 15.0)]
    off = [(-380.0 + 10 * k, 250.0 + 10 * k) for k in range(8)]  # a car park 250 m off any road
    back = [(x, 3.0) for x in np.arange(-300.0, -180.0, 15.0)]
    m = HmmMapMatcher(net, cfg)
    res = [m.step(float(k), x, y, np.eye(2) * 25.0, None, 0.0) for k, (x, y) in enumerate(on + off + back)]
    offr = res[len(on):len(on) + len(off)]
    assert [r.mode for r in offr[:2]] == ["no_candidates"] * 2
    assert all(r.mode == "suspended" and not r.matched and r.map_match_confidence == 0.0 for r in offr[2:])
    assert all(r.lat_rad is None for r in offr)  # no invented position off the network
    br = res[len(on) + len(off):]
    assert br[0].mode == "suspended" and br[-1].matched  # re-entry needs consecutive good epochs
    assert m.counts["suspensions"] == 1 and m.counts["reentries"] == 1


def test_search_radius_follows_the_filter_covariance():
    net = crossroads()
    m = HmmMapMatcher(net, MatcherConfig())
    far = (-300.0, 120.0)  # 120 m from way 1, 300 m from way 2
    small = m.step(0.0, *far, np.eye(2) * 5.0 ** 2)
    m.reset()
    large = m.step(0.0, *far, np.eye(2) * 60.0 ** 2)
    assert small.mode == "no_candidates" and small.search_radius_m < 60
    assert large.matched and large.search_radius_m > 120 and large.way["way_id"] == 1
    m.reset()
    huge = m.step(0.0, *far, np.eye(2) * 1e6)
    assert huge.search_radius_m == MatcherConfig().r_max_m  # clipped


def test_confidence_reflects_ambiguity():
    nodes = {1: (-1000, 0), 2: (1000, 0), 3: (-1000, 40), 4: (1000, 40)}
    net = build(nodes, [(1, [1, 2], {"highway": "primary", "oneway": "yes"}),
                        (2, [3, 4], {"highway": "primary", "oneway": "yes"})])
    m = HmmMapMatcher(net)
    mid = m.step(0.0, 0.0, 20.0, np.eye(2) * 30.0 ** 2, EAST, 15.0)  # equidistant, same direction
    assert 0.4 < mid.map_match_confidence < 0.6
    m.reset()
    clear = m.step(0.0, 0.0, 2.0, np.eye(2) * 30.0 ** 2, EAST, 15.0)
    assert clear.map_match_confidence > mid.map_match_confidence and 0.0 <= clear.map_match_confidence <= 1.0


def test_matched_point_lies_on_its_segment_and_near_the_fix():
    net = crossroads()
    _, out = run(net, [(x, 9.0) for x in np.arange(-300.0, -100.0, 15.0)], EAST)
    for r in out:
        _, _, d = net.project(np.array([r.segment]), r.x_m, r.y_m)
        assert d[0] < 1e-6 and r.correction_m <= r.search_radius_m
        x, y, _ = net.to_xy(r.lat_rad, r.lon_rad)
        assert math.hypot(x - r.x_m, y - r.y_m) < 1e-3  # lat/lon via the geodesy path round-trips


def test_offline_cache_gives_identical_matches(tmp_path):
    net = crossroads()
    net.save(tmp_path / "net.npz")
    net2 = RoadNetwork.load(tmp_path / "net.npz")
    obs = [(x, 12.0) for x in np.arange(-150.0, 151.0, 15.0)]
    a = [(r.segment, r.x_m, r.y_m, r.map_match_confidence) for r in run(net, obs, EAST)[1]]
    b = [(r.segment, r.x_m, r.y_m, r.map_match_confidence) for r in run(net2, obs, EAST)[1]]
    assert a == b


def test_map_matching_package_has_no_network_code():
    src = Path(__file__).resolve().parents[2] / "navigation-core" / "map_matching"
    bad = re.compile(r"^\s*(import|from)\s+(urllib|requests|http|socket|aiohttp|httpx)\b", re.M)
    for f in src.glob("*.py"):
        assert not bad.search(f.read_text(encoding="utf-8")), f


# ---- integration with the filter ------------------------------------------------------------

def test_dead_reckoning_matcher_reads_but_never_writes_the_filter():
    net = crossroads()
    lat, lon, _ = LTP.enu_to_geodetic(np.array([-200.0, 8.0, 0.0]))
    s = NavState(0.0, float(lat), float(lon), 0.0, np.array([15.0, 0.0, 0.0]), q_from_euler(0.0, 0, 0))
    ekf = ErrorStateEKF(s, np.eye(N_STATES) * 25.0, NoiseParams(0.05, 0.005, 0.02, 100.0, 1e-3, 100.0))
    P0, n0 = ekf.P.copy(), ekf.nominal
    dr = DeadReckoningMapMatcher(net)
    r = dr.update_from_ekf(ekf)
    assert r.matched and r.way["way_id"] == 1 and abs(r.y_m) < 1e-6 and 0 < r.map_match_confidence <= 1
    assert np.array_equal(ekf.P, P0) and ekf.nominal is n0  # severable: the filter is untouched
    assert dr.update_from_ekf(ekf) is None  # rate-limited to 1 Hz


def test_course_sigma_matches_monte_carlo():
    rng = np.random.default_rng(3)
    v, Pv = np.array([8.0, 6.0]), np.diag([0.3, 0.5])
    h, sh, sp = course_and_sigma(v, Pv)
    samp = rng.multivariate_normal(v, Pv, 200_000)
    mc = np.arctan2(samp[:, 1], samp[:, 0])
    assert h == pytest.approx(math.atan2(6, 8)) and sp == pytest.approx(10.0)
    assert sh == pytest.approx(float(np.std(mc)), rel=0.03)


def test_road_bundle_round_trip_is_lossless_and_guarded(tmp_path):
    """The phone's offline format (navcore.map_matching.bundle): same arrays, same matches."""
    from navcore.map_matching.bundle import read_bundle, write_bundle

    net = crossroads()
    write_bundle(net, tmp_path / "x.roads.bin")
    back = read_bundle(tmp_path / "x.roads.bin")
    assert np.array_equal(back.node_xy, net.node_xy) and np.array_equal(back.seg_from, net.seg_from)
    assert np.array_equal(back.seg_to, net.seg_to) and np.array_equal(back.seg_way, net.seg_way)
    assert back.way_tags == net.way_tags and back.cell_m == net.cell_m
    obs = [(x, 12.0) for x in np.arange(-150.0, 151.0, 15.0)]
    a = [(r.segment, r.x_m, r.y_m, r.map_match_confidence) for r in run(net, obs, EAST)[1]]
    b = [(r.segment, r.x_m, r.y_m, r.map_match_confidence) for r in run(back, obs, EAST)[1]]
    assert a == b
    raw = (tmp_path / "x.roads.bin").read_bytes()
    (tmp_path / "bad.bin").write_bytes(b"NOTROADS" + raw[8:])
    with pytest.raises(ValueError, match="not a road bundle"):
        read_bundle(tmp_path / "bad.bin")
    (tmp_path / "long.bin").write_bytes(raw + b"\0")
    with pytest.raises(ValueError, match="trailing"):
        read_bundle(tmp_path / "long.bin")
