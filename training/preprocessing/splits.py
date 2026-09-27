"""Group-based, leak-free train/val/test splitting (docs/ml_pipeline.md section 3).

Unit of assignment = the DRIVE, not the session. IO-VNBD sessions are often consecutive
slices of one continuous drive, seconds apart (e.g. Vta11 ends 12 s before Vta12
starts). A drive is the connected component of sessions that share a driver AND either
share a route group (S3a/S3b/S3c) or are separated by less than ``DRIVE_GAP_S``. Adjacent
samples of one drive can therefore never land in different splits.

* test  = held-out DRIVER(S) entirely -- the driver subset whose share of hours is closest
          to TEST_FRACTION_TARGET within TEST_FRACTION_BOUNDS;
* val   = whole drives of the remaining drivers, chosen by a seeded shuffle up to
          VAL_FRACTION_TARGET, never emptying a driver's train set;
* train = the rest.

``find_leaks`` re-checks a manifest independently of how it was built.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, pairwise

import numpy as np
import pandas as pd

DRIVE_GAP_S = 600.0  # conservative: largest observed same-drive gap is 444 s (Vw7)
TEST_FRACTION_TARGET = 0.20
TEST_FRACTION_BOUNDS = (0.10, 0.35)
VAL_FRACTION_TARGET = 0.15
SEED = 26168
SPLIT_VERSION = 2


def assign_drives(sessions: pd.DataFrame, gap_s: float = DRIVE_GAP_S) -> pd.Series:
    """Drive id per session. ``sessions`` needs session_id, driver, route_group,
    start_utc, end_utc."""
    parent = {s: s for s in sessions["session_id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for (_, _), g in sessions.groupby(["driver", "route_group"]):
        ids = list(g["session_id"])
        for a, b in pairwise(ids):
            union(a, b)
    for _, g in sessions.sort_values("start_utc").groupby("driver"):
        g = g.sort_values("start_utc")
        latest_end, latest_id = None, None
        for sid, st, en in zip(g["session_id"], g["start_utc"], g["end_utc"], strict=True):
            if latest_end is not None and (st - latest_end).total_seconds() < gap_s:
                union(sid, latest_id)
            if latest_end is None or en > latest_end:
                latest_end, latest_id = en, sid
    roots = {s: find(s) for s in parent}
    # stable, readable drive ids: driver + first session (by start time) of the drive
    first = (sessions.assign(root=sessions["session_id"].map(roots))
             .sort_values("start_utc").groupby("root")["session_id"].first())
    drv = dict(zip(sessions["session_id"], sessions["driver"], strict=True))
    return sessions["session_id"].map(lambda s: f"{drv[s]}:{first[roots[s]]}")


def make_split(sessions: pd.DataFrame, seed: int = SEED) -> dict:
    s = sessions.copy()
    s["drive_id"] = assign_drives(s)
    dur = dict(zip(s["session_id"], s["duration_s"], strict=True))
    total = float(s["duration_s"].sum())
    by_driver = s.groupby("driver")["duration_s"].sum().to_dict()
    drivers = sorted(by_driver)

    cands = []
    for r in range(1, len(drivers)):
        for combo in combinations(drivers, r):
            frac = sum(by_driver[d] for d in combo) / total
            if TEST_FRACTION_BOUNDS[0] <= frac <= TEST_FRACTION_BOUNDS[1]:
                cands.append((abs(frac - TEST_FRACTION_TARGET), r, combo))
    if not cands:
        raise RuntimeError(f"no driver subset gives a test share within {TEST_FRACTION_BOUNDS}")
    test_drivers = min(cands)[2]

    rest = s[~s["driver"].isin(test_drivers)]
    drives = rest.groupby("drive_id")["duration_s"].sum()
    drive_driver = rest.groupby("drive_id")["driver"].first()
    per_driver = drive_driver.value_counts().to_dict()
    keys = sorted(drives.index)
    order = [keys[i] for i in np.random.default_rng(seed).permutation(len(keys))]
    rest_total = float(rest["duration_s"].sum())
    val, val_dur, taken = [], 0.0, defaultdict(int)
    for k in order:
        if val_dur >= VAL_FRACTION_TARGET * rest_total:
            break
        d = drive_driver[k]
        if taken[d] + 1 >= per_driver[d]:
            continue  # keep at least one drive of every driver in train
        if val_dur + drives[k] > 2 * VAL_FRACTION_TARGET * rest_total:
            continue  # one huge drive must not swallow the train set
        val.append(k)
        taken[d] += 1
        val_dur += float(drives[k])

    split_of = {}
    for sid, drv, did in zip(s["session_id"], s["driver"], s["drive_id"], strict=True):
        split_of[sid] = "test" if drv in test_drivers else ("val" if did in val else "train")
    names = ("train", "val", "test")
    splits = {n: sorted(k for k, v in split_of.items() if v == n) for n in names}
    return {
        "split_version": SPLIT_VERSION,
        "unit": "drive (connected sessions: same driver and same route group or gap < "
                f"{DRIVE_GAP_S:.0f} s)",
        "rule": "test = held-out drivers; val = whole drives of remaining drivers; train = rest",
        "seed": seed,
        "drive_gap_s": DRIVE_GAP_S,
        "test_fraction_target": TEST_FRACTION_TARGET,
        "test_fraction_bounds": list(TEST_FRACTION_BOUNDS),
        "val_fraction_target_of_remaining": VAL_FRACTION_TARGET,
        "test_drivers": list(test_drivers),
        "splits": splits,
        "split_duration_h": {n: round(sum(dur[x] for x in ids) / 3600, 3) for n, ids in splits.items()},
        "split_fraction": {n: round(sum(dur[x] for x in ids) / total, 4) for n, ids in splits.items()},
        "drives": {
            did: {"driver": g["driver"].iloc[0], "split": split_of[g["session_id"].iloc[0]],
                  "sessions": sorted(g["session_id"]), "duration_h": round(g["duration_s"].sum() / 3600, 3)}
            for did, g in s.groupby("drive_id")
        },
        "sessions": {
            r.session_id: {"split": split_of[r.session_id], "driver": r.driver,
                           "route_group": r.route_group, "drive_id": r.drive_id,
                           "start_utc": str(r.start_utc), "end_utc": str(r.end_utc),
                           "duration_s": round(float(r.duration_s), 3)}
            for r in s.sort_values("session_id").itertuples()
        },
    }


def find_leaks(manifest: dict, gap_s: float = DRIVE_GAP_S) -> list[str]:
    """Independent leak audit. Returns human-readable violations (empty = clean)."""
    out = []
    listed = [x for ids in manifest["splits"].values() for x in ids]
    if len(listed) != len(set(listed)):
        out.append("a session appears in more than one split")
    sess = manifest["sessions"]
    if set(listed) != set(sess):
        out.append("split lists and session table disagree")
    for key in ("drive_id", "route_group"):
        groups = defaultdict(set)
        for m in sess.values():
            groups[(m["driver"], m[key])].add(m["split"])
        out += [f"{key} {g} spans splits {sorted(v)}" for g, v in groups.items() if len(v) > 1]
    drv = defaultdict(set)
    for m in sess.values():
        drv[m["driver"]].add(m["split"])
    out += [f"test driver {d} also in {sorted(v - {'test'})}" for d, v in drv.items()
            if "test" in v and len(v) > 1]
    # temporal adjacency, recomputed from timestamps (does not trust drive_id)
    rows = sorted(((m["driver"], pd.Timestamp(m["start_utc"]), pd.Timestamp(m["end_utc"]),
                    m["split"], sid) for sid, m in sess.items()))
    for a, b in combinations(rows, 2):
        if a[0] != b[0] or a[3] == b[3]:
            continue
        gap = max((b[1] - a[2]).total_seconds(), (a[1] - b[2]).total_seconds())
        if gap < gap_s:
            out.append(f"{a[4]} ({a[3]}) and {b[4]} ({b[3]}) are {gap:.1f} s apart")
    return out
