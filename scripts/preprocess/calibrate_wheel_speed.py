#!/usr/bin/env python3
"""Thin wrapper: empirical wheel-speed unit/scale calibration.

The implementation lives in training/preprocessing/wheel_speed.py (it is also step 1 of
``python -m training.preprocessing.pipeline``).

    .venv/Scripts/python.exe scripts/preprocess/calibrate_wheel_speed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.preprocessing.wheel_speed import calibrate

if __name__ == "__main__":
    calibrate()
    sys.exit(0)
