"""Golden vectors for the Android port must still be what the Python reference produces."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_committed_golden_vectors_match_the_python_reference():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "parity" / "generate_golden_vectors.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr


def test_state_machine_vector_covers_every_state_and_transition():
    ev = json.loads((REPO / "tests" / "regression" / "golden" / "gnss_state_machine.json").read_text(encoding="utf-8"))
    states = [e["expect"] for e in ev["events"]]
    pairs = {(a, b) for a, b in zip(["LOST", *states[:-1]], states, strict=True) if a != b}
    assert {"GOOD", "DEGRADED", "LOST", "RECOVERING"} <= set(states)
    assert {("LOST", "RECOVERING"), ("RECOVERING", "GOOD"), ("GOOD", "DEGRADED"), ("DEGRADED", "GOOD"),
            ("DEGRADED", "LOST"), ("RECOVERING", "DEGRADED"), ("RECOVERING", "LOST")} <= pairs
