#!/usr/bin/env python3
"""Golden vectors: the Python reference's outputs on fixed inputs, for the Kotlin port.

    .venv/Scripts/python.exe scripts/parity/generate_golden_vectors.py [--check]

AGENTS.md 4: one specification, two implementations, compared on the SAME inputs.
Writes tests/regression/golden/*.json; the Android :core tests replay them
(android-app/core/src/test/...), and tests/regression/test_golden_vectors.py fails if the
Python reference no longer reproduces a committed file. ``--check`` compares, writes nothing.

Vectors:
  gnss_state_machine.json   an event sequence (on_time / on_fix) covering every transition,
                            plus a seeded random stretch; the expected state after each event
  sample_validation.json    GnssSample inputs and whether the reference accepts them
  wgs84_radii.json          meridian / prime-vertical radii at fixed latitudes
  nav_*.json, mm_*.bin      Phase 11: geometry, INS, EKF ops, features, Model A, fused
                            navigator, streaming engine, map matcher (scripts/parity/nav_golden.py)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius  # noqa: E402
from navcore.sensors.samples import GnssSample, SensorSampleError  # noqa: E402
from navcore.state.gnss_state import GnssStateConfig, GnssStateMachine  # noqa: E402

OUT = REPO_ROOT / "tests" / "regression" / "golden"


def gnss_state_machine() -> dict:
    cfg = GnssStateConfig()
    sm = GnssStateMachine(cfg)
    events = []

    def fix(t, acc, ok):
        events.append({"op": "fix", "t": t, "accuracy_m": acc, "accepted": ok, "expect": sm.on_fix(t, acc, ok).value})

    def tick(t):
        events.append({"op": "time", "t": t, "expect": sm.on_time(t).value})

    # scripted: every transition at least once, including the boundaries
    tick(0.5)
    fix(1.0, 3.0, True)                     # LOST -> RECOVERING
    fix(2.0, 3.0, True)
    fix(3.0, 3.0, True)                     # -> GOOD (3 good)
    fix(4.0, 10.0, True)                    # accuracy == threshold: still good
    fix(5.0, 10.0001, True)                 # -> DEGRADED (accuracy)
    fix(6.0, 3.0, True)
    fix(7.0, 3.0, True)                     # -> GOOD (2 good)
    fix(8.0, 3.0, False)                    # -> DEGRADED (gate)
    fix(9.0, 3.0, True)
    fix(10.0, 3.0, True)                    # -> GOOD
    tick(22.0)                              # age 12.0: not > 12 -> GOOD
    tick(22.1)                              # age 12.1 -> DEGRADED
    tick(35.0)                              # age 25.0: not > 25
    tick(35.2)                              # -> LOST
    fix(36.0, 30.0, True)                   # LOST -> RECOVERING (inaccurate: streak 0)
    fix(37.0, 3.0, True)
    fix(38.0, 3.0, False)                   # RECOVERING -> DEGRADED (gate)
    tick(70.0)                              # -> LOST
    fix(71.0, 3.0, True)                    # -> RECOVERING (streak 1)
    tick(97.0)                              # RECOVERING, age 26 -> LOST
    # seeded random stretch
    rng = np.random.default_rng(26168)
    t = 100.0
    for _ in range(400):
        t = round(t + float(rng.choice([0.1, 1.0, 9.0, 13.0, 27.0])), 4)
        if rng.random() < 0.6:
            fix(t, round(float(rng.choice([2.0, 5.0, 9.9, 10.0, 12.0, 40.0])), 4), bool(rng.random() < 0.85))
        else:
            tick(t)
    return {"generator": "scripts/parity/generate_golden_vectors.py", "reference": "navcore.state.gnss_state",
            "config": {"degraded_accuracy_m": cfg.degraded_accuracy_m, "degraded_age_s": cfg.degraded_age_s,
                       "lost_age_s": cfg.lost_age_s, "good_streak": cfg.good_streak,
                       "recover_streak": cfg.recover_streak},
            "initial": "LOST", "events": events}


def sample_validation() -> dict:
    d = math.radians
    cases = [
        ("ok", dict(t_s=1.0, lat_rad=d(52.4), lon_rad=d(-1.5), h_m=100.0, horizontal_accuracy_m=4.0)),
        ("ok_speed_bearing", dict(t_s=1.0, lat_rad=d(52.4), lon_rad=d(-1.5), h_m=100.0, horizontal_accuracy_m=4.0,
                                  speed_mps=12.0, bearing_rad=d(270.0))),
        ("lat_in_degrees", dict(t_s=1.0, lat_rad=52.4, lon_rad=d(-1.5), h_m=0.0, horizontal_accuracy_m=4.0)),
        ("lon_in_degrees", dict(t_s=1.0, lat_rad=d(52.4), lon_rad=4.0, h_m=0.0, horizontal_accuracy_m=4.0)),
        ("zero_accuracy", dict(t_s=1.0, lat_rad=d(52.4), lon_rad=d(-1.5), h_m=0.0, horizontal_accuracy_m=0.0)),
        ("received_before_epoch", dict(t_s=2.0, t_received_s=1.0, lat_rad=d(52.4), lon_rad=d(-1.5), h_m=0.0,
                                       horizontal_accuracy_m=4.0)),
        ("speed_without_bearing", dict(t_s=1.0, lat_rad=d(52.4), lon_rad=d(-1.5), h_m=0.0, horizontal_accuracy_m=4.0,
                                       speed_mps=3.0)),
        ("speed_too_high", dict(t_s=1.0, lat_rad=d(52.4), lon_rad=d(-1.5), h_m=0.0, horizontal_accuracy_m=4.0,
                                speed_mps=100.5, bearing_rad=0.0)),
        ("height_implausible", dict(t_s=1.0, lat_rad=d(52.4), lon_rad=d(-1.5), h_m=20001.0, horizontal_accuracy_m=4.0)),
        ("negative_speed", dict(t_s=1.0, lat_rad=d(52.4), lon_rad=d(-1.5), h_m=0.0, horizontal_accuracy_m=4.0,
                                speed_mps=-0.1, bearing_rad=0.0)),
    ]
    out = []
    for name, kw in cases:
        try:
            GnssSample(**kw)
            ok = True
        except SensorSampleError:
            ok = False
        out.append({"name": name, "input": kw, "accepted": ok})
    return {"generator": "scripts/parity/generate_golden_vectors.py", "reference": "navcore.sensors.samples.GnssSample",
            "cases": out}


def wgs84_radii() -> dict:
    lats = [-89.9, -52.4, -30.0, 0.0, 1e-6, 30.0, 45.0, 52.4, 60.0, 89.9]
    return {"generator": "scripts/parity/generate_golden_vectors.py", "reference": "navcore.geometry.geodesy",
            "rows": [{"lat_rad": math.radians(d), "meridian_m": float(meridian_radius(math.radians(d))),
                      "prime_vertical_m": float(prime_vertical_radius(math.radians(d)))} for d in lats]}


VECTORS = {"gnss_state_machine.json": gnss_state_machine, "sample_validation.json": sample_validation,
           "wgs84_radii.json": wgs84_radii}


def render(obj: dict, compact: bool = False) -> str:
    if compact:  # the large Phase 11 vectors: one line, still exact (repr floats)
        return json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def build(bin_dir: Path) -> dict[str, str]:
    from scripts.parity.nav_golden import all_vectors  # heavier imports (onnxruntime, engine)

    out = {name: render(fn()) for name, fn in VECTORS.items()}
    out |= {name: render(obj, compact=True) for name, obj in all_vectors(REPO_ROOT, bin_dir).items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    bad = 0
    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, text in build(tmp).items():
                path = OUT / name
                # a Windows checkout may turn LF into CRLF: compare content, not line endings
                same = path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") == text
                print(f"{'OK  ' if same else 'DIFF'} {path.relative_to(REPO_ROOT)}")
                bad += not same
            for b in sorted(tmp.glob("*.bin")):
                same = (OUT / b.name).exists() and (OUT / b.name).read_bytes() == b.read_bytes()
                print(f"{'OK  ' if same else 'DIFF'} {(OUT / b.name).relative_to(REPO_ROOT)}")
                bad += not same
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, text in build(OUT).items():
            (OUT / name).write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {(OUT / name).relative_to(REPO_ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
