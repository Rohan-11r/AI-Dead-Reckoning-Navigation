# SIH26168 — AI-ML based Intelligent Dead Reckoning System

Seamless navigation through GNSS outages using a **hybrid physics + AI** architecture.

> **Read [`AGENTS.md`](./AGENTS.md) first.** It is the core directive and it governs every
> file in this repository.

```
Sensors -> Calibration -> AI Signal Processing -> INS -> Fusion -> NHC -> Map Matching
```

Three rules that are never relaxed:

1. **No model ever outputs latitude or longitude.** Networks predict physical quantities
   (speed, displacement, bias, heading change) with units. Coordinates come only from
   integrating physics through `navigation-core/geometry`.
2. **No fabricated data or metrics.** Every number in every document is produced by a
   script in this repo that anyone can re-run.
3. **Python and Android must agree numerically.** A parity failure blocks release.

## Status

Phases 0 and 1 complete of 0-14; Phase 2 (dataset ingestion) is next. See [`PROJECT_STATUS.md`](./PROJECT_STATUS.md) for verified detail and
[`TODO.md`](./TODO.md) for the roadmap.

## Layout

| Path | Role |
| --- | --- |
| `navigation-core/` | Python **reference** implementation (imports as `navcore.*`) |
| `training/` | Offline ML: datasets, models, training, export |
| `edge-engine/` | Portable on-device engine — second implementation of the one spec |
| `android-app/` | Android application (Phase 13) |
| `simulation/` | Replay and GNSS-outage scenario injection over real recordings |
| `evaluation/` | Trajectory metrics — the single source of every published number |
| `hardware/` | External IMU integration (Bluetooth / USB / serial) |
| `data/` | `raw/` is **read-only**; `interim/`, `processed/` are regenerable |
| `models/` | Checkpoints, verified exported graphs, model cards |
| `scripts/` | Operational entry points and phase gates |
| `tests/` | Unit, integration, navigation, ML, parity, regression |
| `docs/` | Architecture, navigation maths, ML pipeline, map matching |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .

# verify
.venv/Scripts/python.exe scripts/phase1_validate.py
```

Optional extras: `requirements-dev.txt` (linting, typing), `requirements-geo.txt`
(Phase 10 map matching), `requirements-gpu.txt` (CUDA PyTorch for Phase 7 training).

Interpreter is **CPython 3.14.4**; 3.13.5 is a verified fallback.

## Dataset

IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset), 288 CSV recordings
at 10 Hz from Coventry, UK. Held **outside** the repository and read-only. Files are
cp1252-encoded, not UTF-8 — see `PROJECT_STATUS.md` §0.4 for the full list of parsing
hazards before writing any loader.
