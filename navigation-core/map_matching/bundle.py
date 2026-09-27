"""Road-network BUNDLE: the offline graph in a format the phone reads without numpy.

The Python cache (``RoadNetwork.save``, .npz) is not readable on Android; this is a plain
big-endian binary file (java.io.DataInputStream order) with the SAME arrays, so the Kotlin
port (android-app :core ``mapmatch.RoadNetwork``) rebuilds the identical network: same
segment ids, same geometry, same tangent-plane origin.

Layout (all big-endian):
    8 bytes   magic  b"SIHROAD1"
    int32     version (1)
    float64   lat0_rad, lon0_rad, h0_m          tangent-plane origin
    float64   cell_m                            grid cell size
    int32     n_nodes, then n_nodes x (float64 x_m, float64 y_m)
    int32     n_segs, then n_segs x int32 seg_from, n_segs x int32 seg_to, n_segs x int64 seg_way
    int32     n bytes, then UTF-8 JSON {way_id: {tag: value}}
The grid index is NOT stored: both implementations rebuild it with the same rule.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from navcore.geometry.geodesy import LocalTangentPlane
from navcore.map_matching.road_network import RoadNetwork

MAGIC = b"SIHROAD1"
VERSION = 1


def write_bundle(net: RoadNetwork, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = json.dumps({str(k): v for k, v in sorted(net.way_tags.items())}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    n, m = len(net.node_xy), len(net.seg_from)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">i", VERSION))
        f.write(struct.pack(">dddd", net.ltp.lat0_rad, net.ltp.lon0_rad, net.ltp.h0_m, net.cell_m))
        f.write(struct.pack(">i", n))
        f.write(np.ascontiguousarray(net.node_xy, dtype=">f8").tobytes())
        f.write(struct.pack(">i", m))
        f.write(np.ascontiguousarray(net.seg_from, dtype=">i4").tobytes())
        f.write(np.ascontiguousarray(net.seg_to, dtype=">i4").tobytes())
        f.write(np.ascontiguousarray(net.seg_way, dtype=">i8").tobytes())
        f.write(struct.pack(">i", len(tags)))
        f.write(tags)


def read_bundle(path: str | Path) -> RoadNetwork:
    b = Path(path).read_bytes()
    if b[:8] != MAGIC:
        raise ValueError(f"{path}: not a road bundle")
    o = 8
    (ver,) = struct.unpack_from(">i", b, o)
    o += 4
    if ver != VERSION:
        raise ValueError(f"{path}: bundle version {ver}, expected {VERSION}")
    lat0, lon0, h0, cell = struct.unpack_from(">dddd", b, o)
    o += 32
    (n,) = struct.unpack_from(">i", b, o)
    o += 4
    xy = np.frombuffer(b, ">f8", 2 * n, o).reshape(n, 2).astype(np.float64)
    o += 16 * n
    (m,) = struct.unpack_from(">i", b, o)
    o += 4
    sf = np.frombuffer(b, ">i4", m, o).astype(np.int64)
    o += 4 * m
    st = np.frombuffer(b, ">i4", m, o).astype(np.int64)
    o += 4 * m
    sw = np.frombuffer(b, ">i8", m, o).astype(np.int64)
    o += 8 * m
    (k,) = struct.unpack_from(">i", b, o)
    o += 4
    tags = {int(w): v for w, v in json.loads(b[o:o + k].decode("utf-8")).items()}
    if o + k != len(b):
        raise ValueError(f"{path}: {len(b) - o - k} trailing bytes")
    return RoadNetwork(LocalTangentPlane(lat0, lon0, h0), xy, sf, st, sw, tags, cell)


__all__ = ["MAGIC", "VERSION", "read_bundle", "write_bundle"]
