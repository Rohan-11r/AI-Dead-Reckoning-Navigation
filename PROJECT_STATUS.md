# PROJECT_STATUS.md — SIH26168

**Project:** AI-ML based Intelligent Dead Reckoning system for seamless navigation
**Workspace:** `C:\Users\Shreeyash\OneDrive\Desktop\dead reckening`
**Governing document:** [`AGENTS.md`](./AGENTS.md) — the core directive overrides anything here.
**Last updated:** 2026-09-27 (Phase 2)

> **Status honesty rule.** A phase is `COMPLETED` only when its exit criteria are met by
> code in this repo that anyone can re-run. Nothing is marked done on intent. No metric
> appears in this file unless a script in this repo produced it.

---

## Status Legend

| Marker | Meaning |
| --- | --- |
| `NOT STARTED` | No work begun. |
| `IN PROGRESS` | Actively being built. |
| `BLOCKED` | Cannot proceed; blocker named explicitly. |
| `COMPLETED` | Exit criteria met and verified by a re-runnable check. |

---

## Phase Overview

| # | Phase | Status |
| --- | --- | --- |
| 0 | Environment Bootstrap & Reconnaissance | **COMPLETED** |
| 1 | Architecture & Repository Creation | **COMPLETED** |
| 2 | Dataset Ingestion, Schema Normalisation & Integrity Audit | **COMPLETED** |
| 3 | Geodesy & Reference-Frame Foundations | `NOT STARTED` |
| 4 | Sensor Characterisation & Calibration | `NOT STARTED` |
| 5 | Attitude Estimation (AHRS) | `NOT STARTED` |
| 6 | Strapdown INS Mechanization | `NOT STARTED` |
| 7 | AI Signal Processing (denoising/bias + context/speed) | `NOT STARTED` |
| 8 | Fusion Engine (Error-State EKF) | `NOT STARTED` |
| 9 | Non-Holonomic Constraints (NHC) | `NOT STARTED` |
| 10 | Map Matching | `NOT STARTED` |
| 11 | Evaluation Harness & Benchmarking | `NOT STARTED` |
| 12 | Model Export & Python/Android Numerical Parity | `NOT STARTED` |
| 13 | Android Application | `NOT STARTED` |
| 14 | End-to-End Validation, Documentation & Demo | `NOT STARTED` |

> **Roadmap revision (2026-09-27).** "Architecture & Repository Creation" was inserted as
> Phase 1, displacing the original numbering. To keep the roadmap at 0-14 as originally
> scoped, the two AI phases (originally 6 "Denoising & Bias" and 7 "Motion Context &
> Displacement") are merged into a single Phase 7 — they share one data pipeline, one
> training harness and one export path, and were already noted as parallelisable.
> Phases 8-14 keep their original numbers. Cross-references in `docs/` and `TODO.md` were
> renumbered mechanically in the same commit.

---

## Phase 0 — Environment Bootstrap & Reconnaissance

**Status:** `COMPLETED`
**Date:** 2026-09-27

### Objective

Establish exactly what this machine can and cannot do, confirm whether real IO-VNBD data
is available, and lay down the governing directive before any code is written.

### 0.1 Hardware (verified by `Get-CimInstance`, `nvidia-smi`)

| Item | Value |
| --- | --- |
| OS | Windows 11 Home Single Language, build 10.0.26200 |
| CPU | Intel Core i5-12500H — 12 physical cores, 16 logical threads, 2.50 GHz base |
| RAM | 15.67 GB total (3.13 GB free at probe time) |
| GPU 0 | Intel Iris Xe Graphics (integrated), driver 32.0.101.5542 |
| GPU 1 | **NVIDIA GeForce RTX 2050, 4096 MiB VRAM**, driver 555.97 |
| CUDA (driver) | 12.5 reported by `nvidia-smi` |
| CUDA toolkit | **Absent** — `nvcc` not on PATH |
| Disk (C:) | 476 GB total, 281 GB available |

**Implication for model design:** 4 GB of VRAM is the binding constraint. All AI stages
must be designed as small sequence models (1-D CNN / GRU / TCN scale, single-digit
millions of parameters) trained on windowed 10 Hz data with modest batch sizes. This is
consistent with the requirement that the same models run on a phone, so the constraint is
an alignment, not a compromise. No large-transformer designs.

### 0.2 Software toolchain (verified)

| Tool | Status |
| --- | --- |
| Python 3.14.4 | Present at `C:\Python314\python.exe` — **first on PATH** |
| Python 3.13.5 | Present at `C:\Program Files\Python313\python.exe` |
| pip | 26.0.1 |
| numpy / scipy / pandas / scikit-learn | **Not installed** |
| torch / tensorflow / onnx / onnxruntime | **Not installed** |
| matplotlib / filterpy / shapely / geopandas | **Not installed** |
| git | 2.54.0.windows.1 |
| Shell | PowerShell 5.1 + MINGW64 Git Bash 3.6.7 |
| **Java JDK** | **ABSENT** (`java` not on PATH) |
| **Gradle** | **ABSENT** |
| **Android SDK** | **ABSENT** — `ANDROID_HOME` and `ANDROID_SDK_ROOT` both unset; no `%LOCALAPPDATA%\Android\Sdk` |
| **adb** | **ABSENT** |

**Wheel availability check (live PyPI query):** latest `torch` is **2.14.0** and publishes
Windows wheels for `cp312`, `cp313`, and `cp314`. Python 3.14.4 is therefore viable for
the ML stack. Decision deferred to Phase 2 (see Open Decisions) because the wider
scientific ecosystem is less uniformly built for 3.14 than for 3.13.

### 0.3 Network (verified by HTTPS probe)

| Endpoint | HTTP status |
| --- | --- |
| `https://pypi.org/simple/` | 200 |
| `https://github.com` | 200 |
| `https://figshare.com` | 202 |
| `https://www.google.com` | 200 |

Outbound HTTPS works. Package installation and OSM / map-data download are feasible.

### 0.4 IO-VNBD dataset — **PRESENT, REAL, VERIFIED**

Located on this machine by filesystem scan. **Not** downloaded, **not** synthesised.

```
C:\Users\Shreeyash\OneDrive\Desktop\SIH26168\data raw  IO-VNBD\
  Synchronised V abd S datasets\Synchronised V abd S datasets\
    Categorised IOVNB Dataset\
      M (Driver B)\          S (Driver A)\      Vf (Driver E)\
      Vta (Driver E)\        Vtb (Driver E)\    Vw (Driver E)\
      Y (Driver D)\
    Uncategorised IOVNB Dataset\
      S-Dataset\             V-Dataset\
```

| Property | Verified value |
| --- | --- |
| Total size on disk | **816.1 MB of CSV** (825 MB including the 72 JPGs), materialised locally — not a OneDrive cloud placeholder |
| CSV files | **288** total — 144 under `Categorised`, 144 under `Uncategorised` |
| File-prefix split | 144 `S-*` (smartphone) / 144 `V-*` (vehicle) |
| Supporting JPGs | 72 (route/context imagery) |
| Drivers represented | A, B, D, E (from directory naming) |
| Sampling rate | **10 Hz** — consecutive `TIME SINCE START` values 4227 / 4327 / 4427 ms |
| Recording locale | Coventry, UK — first fix `52.402565, -1.503471` |
| Recording date (sample) | 2019-09-07 |
| Sample file length | `S-M.csv` = 105,975 lines including header (~2.9 h at 10 Hz) |
| **Text encoding** | **cp1252 / latin-1 — NOT UTF-8.** A strict UTF-8 read fails on byte `0xb2` at position 223 of the header (the `m/s²` superscript). |

**Confirmed 24-column schema** (verbatim header, whitespace irregularities included):

```
GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), GPS SPEED (Kmh),
GPS ACCURACY (m), GPS ORIENTATION (°),GPS SATELLITES IN RANGE,
TIME SINCE START (ms), DATE (YYYY-MO-DD HH-MI-SS_SSS),
ACCELEROMETER X (m/s²) , ACCELEROMETER Y (m/s²), ACCELEROMETER Z (m/s²),
GRAVITY X (m/s²), GRAVITY Y (m/s²), GRAVITY Z (m/s²),
GYROSCOPE Yaw (rad/s), GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s),
MAGNETIC FIELD X (μT), MAGNETIC FIELD Y (μT), MAGNETIC FIELD Z (μT),
ORIENTATION (Yaw) (°), ORIENTATION (Pitch) (°), ORIENTATION (Roll ) (°)
```

**Parsing hazards recorded now so Phase 2 handles them deliberately:**

1. Encoding is cp1252; a naive UTF-8 read raises.
2. Header has inconsistent spacing — a missing space after `(°),` before
   `GPS SATELLITES`, and a trailing space after `ACCELEROMETER X (m/s²) `.
   `ORIENTATION (Roll )` has a space before the closing paren. Column names must be
   normalised, not matched literally.
3. `GPS SATELLITES IN RANGE` is **not numeric** — observed value `18 / 19`. It must be
   split into used/visible integers.
4. `DATE` uses `YYYY-MM-DD HH:MM:SS:mmm` — a **colon** before milliseconds, not a
   decimal point. `strptime` needs an explicit format; auto-inference will misparse.
5. `GPS SPEED` is in **km/h**, while accelerations are SI. Conversion belongs at the
   loader boundary, per `AGENTS.md` §3.
6. Gyroscope axes are labelled **Yaw / Pitch / Roll**, not X / Y / Z. The mapping from
   these labels to a body frame must be established empirically in Phase 4 and never
   assumed.
7. `ACCELEROMETER` and `GRAVITY` are reported separately, so the accelerometer channel
   appears to be gravity-inclusive (`Z ≈ 9.97` at rest while `GRAVITY Z = 9.8066`).
   To be confirmed, not assumed, in Phase 2/3.
8. `Categorised` and `Uncategorised` contain files of identical names (e.g. `S-M.csv` in
   both). These are *likely* two organisations of the same recordings, which would make
   the unique data roughly half of 825 MB. **Unverified** — Phase 2 must checksum before
   either set is treated as independent data.

### 0.5 Deliverables created

| File | Purpose |
| --- | --- |
| `AGENTS.md` | Core directive, hard prohibitions, physical-consistency and parity requirements. |
| `PROJECT_STATUS.md` | This file — phase tracking, Phases 0–14. |
| `TODO.md` | High-level roadmap with per-phase task breakdown. |

### 0.6 Validation performed

`scripts/phase0_validate.py` re-runs the environment and dataset checks from scratch and
prints a PASS/FAIL table. Latest run output is recorded in
`reports/phase0_validation.txt`. The script asserts, among other things, that the dataset
directory exists, that the CSV count is 288, that the header parses under cp1252 and
fails under UTF-8, and that the sample interval is 100 ms.

### 0.7 Exit criteria

- [x] Hardware, OS, CPU/GPU/RAM enumerated from live system queries
- [x] Python, ML-package, and Android/Gradle/JDK availability established
- [x] Network reachability confirmed
- [x] IO-VNBD presence confirmed on disk, schema and encoding inspected
- [x] `AGENTS.md`, `PROJECT_STATUS.md`, `TODO.md` created
- [x] Re-runnable validation script written and executed — 16/16 CRITICAL checks PASS

### 0.8 Blockers carried forward

| Blocker | Affects | Severity |
| --- | --- | --- |
| **No JDK, no Android SDK, no Gradle, no adb** | Phases 12–14 | **High** — must be resolved before any Android work. Android Studio (or command-line tools + JDK 17) needs installing. |
| No physical Android test device confirmed | Phases 13–14 | Medium — an emulator cannot produce real IMU data, so a real handset is required for genuine field validation. |
| 4 GB VRAM | Phase 7 | Low — constrains model size, already factored into the design. |
| No CUDA toolkit (`nvcc`) | Phase 7 | Low — PyTorch ships its own CUDA runtime in the wheel; a toolkit is only needed for custom kernels, which are not planned. |
| ML/scientific stack not installed | Phase 2 onward | Low — network is available; this is the first task of Phase 2. |

### 0.9 Open decisions for Phase 2

1. **Python version:** 3.14.4 (default on PATH, torch 2.14 wheels exist) vs 3.13.5
   (broader ecosystem maturity). To be settled by attempting a pinned venv install and
   reporting what actually resolves — not by assumption.
2. **Workspace vs dataset location:** raw data lives under `Desktop\SIH26168`, outside
   this workspace, and inside a OneDrive-synced tree. Proposal: keep raw data read-only
   where it is, reference it through a single config path, and write all derived
   artefacts inside this workspace — ideally to a non-synced location to avoid OneDrive
   churn on multi-GB intermediates.
3. **Version control:** this workspace is not yet a git repository. Initialising one is
   a Phase 2 task, with the dataset and derived artefacts excluded via `.gitignore`.

---

## Phase 1 — Architecture & Repository Creation

**Status:** `COMPLETED`
**Date:** 2026-09-27
**Evidence:** `reports/phase1_validation.txt` — **17/17 CRITICAL checks PASS**
**Re-run with:** `.venv/Scripts/python.exe scripts/phase1_validate.py`

### Objective

Turn an empty directory into a working monorepo: the full structure, an isolated pinned
environment, the foundational design documents, and a re-runnable gate that proves all of
it rather than asserting it.

### 1.1 Directory structure

Created by `scripts/setup/create_structure.py`, which is **idempotent and manifest-driven**
— its `LAYOUT` list is the single source of truth for the repository shape, and re-running
it is safe.

| | Count |
| --- | --- |
| Manifest entries | **82** |
| Importable Python packages (given `__init__.py` with a role docstring) | **51** |
| Plain directories (given `.gitkeep` describing what belongs there) | **31** |

Top level: `docs/`, `data/{raw,interim,processed,splits}/`,
`training/{configs,datasets,preprocessing,features,models,losses,trainers,evaluation,export,visualization,scripts}/`,
`models/{checkpoints,exported,metadata,normalization}/`,
`navigation-core/{common,sensors,calibration,alignment,imu,ins,gnss,fusion,nhc,map_matching,outage,recovery,state,geometry,filtering,diagnostics}/`,
`edge-engine/{src,adapters,configs,tests,examples}/`, `android-app/`,
`simulation/{replay,outage,scenarios,synthetic,visualization}/`,
`evaluation/{metrics,reports,plots,baselines}/`,
`hardware/{imu_protocol,bluetooth,usb,serial,examples}/`,
`scripts/{setup,download,preprocess,train,evaluate,export,validate}/`,
`tests/{unit,integration,navigation,ml,android,hardware,regression}/`, `demo/`, `reports/`.

Each `__init__.py` docstring states what belongs in that module, so the mandated pipeline
order is legible from the tree alone rather than only from `AGENTS.md`.

### 1.2 The `navcore` import mapping — a real problem, found and fixed

`navigation-core` contains a hyphen and is therefore **not a legal Python identifier**, so
`import navigation-core.ins` is a syntax error. The requested directory name is kept and
mapped onto a legal import root in `pyproject.toml`:

```
package-dir = { navcore = "navigation-core" }

navigation-core/ins/mechanization.py   ->   import navcore.ins.mechanization
```

**The first attempt silently failed.** Using `[tool.setuptools.packages.find]` with
`include = ["navcore*"]` resolved **zero** navcore packages, because `find` scans the
filesystem for a directory literally named `navcore` and cannot see through the
`package-dir` remap. The editable install succeeded, reported no error, and every
`import navcore.*` then raised `ModuleNotFoundError`.

Fixed by listing all 43 packages explicitly. Verified:

```
navcore.ins -> C:\...\dead reckening\navigation-core\ins\__init__.py
43/43 project packages import
```

This is exactly the class of failure that a green install log hides, which is why the
validation script asserts the resolved `__file__` path and not merely that the import
succeeds.

### 1.3 Python version — decided on evidence, as promised in §0.9

**CPython 3.14.4**, with 3.13.5 as a verified fallback (`requires-python = ">=3.13,<3.15"`).

Live PyPI query on 2026-09-27 for `win_amd64` wheel availability:

| Package | cp313 | cp314 |
| --- | --- | --- |
| numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1 | native | native |
| torch 2.14.0 | native | native |
| onnxruntime 1.30.0 | native | native |
| onnx 1.23.0 | `cp312-abi3` | `cp312-abi3` |
| pyarrow, matplotlib, shapely, pyproj, numba | native | native |
| networkx, geopandas, osmnx, pytest, allantools, tqdm, rich | pure Python | pure Python |

Coverage is **equivalent** on both, so the tie broke on 3.14.4 already being first on
PATH — one fewer moving part. Note that an initial probe wrongly reported "no cp314 wheel"
for `onnx`; direct inspection of the filenames showed it ships `cp312-abi3`, which installs
on 3.12+. Recorded because the corrected finding is what the decision rests on.

### 1.4 Environment

| Item | Value |
| --- | --- |
| venv | `.venv/` (gitignored), created from `C:\Python314\python.exe` |
| Isolation | verified: `sys.prefix != sys.base_prefix`, and `sys.prefix` **is** the project `.venv` |
| pip / setuptools / wheel | 26.2.1 / 84.0.0 / 0.48.0 |
| Pinned dependencies installed | **22 / 22 at exactly the pinned version** |
| Project install | `pip install -e .` — editable, imports resolve to the working tree |

Requirement files, split by when they are actually needed:

| File | Purpose |
| --- | --- |
| `requirements.txt` | core stack, Phases 2–9 (22 exact pins) |
| `requirements-dev.txt` | ruff, mypy, pytest-cov, pre-commit, ipykernel |
| `requirements-geo.txt` | geopandas / pyogrio / osmnx / rtree — deferred to Phase 10 |
| `requirements-gpu.txt` | CUDA torch swap — deferred to Phase 7 |

**Anti-drift measure.** `requirements.txt` and `pyproject.toml` necessarily restate the
same pins, and `pyproject`'s package list restates the layout manifest. Both duplications
are **asserted equal by `scripts/phase1_validate.py`**, so they cannot drift silently —
the same principle `AGENTS.md` §4 applies to the Python/Android constants.

### 1.5 Two toolchain defects found by actually running the export path

Neither would have surfaced until Phase 12 had Phase 1 merely installed packages and
declared success.

1. **`onnxscript` is required but undeclared.** `torch.onnx.export` in torch 2.14 imports
   `onnxscript` (the dynamo-based exporter), but torch does not list it as a dependency.
   Without it, export fails with `ModuleNotFoundError` — at export time only. Now pinned
   at `0.7.2`.

2. **The dynamo exporter crashes under the cp1252 console codepage.** It prints a `U+2705`
   check mark in its own progress logging and dies:
   `UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'`.
   Fixed by `PYTHONUTF8=1`, which `scripts/setup/activate.ps1` and `activate.sh` now set,
   and which `.env.example` documents. (Unrelated to the IO-VNBD CSVs also being cp1252 —
   that is handled explicitly in the loader.)

**Export round-trip, measured on this machine:**

| Path | `max |torch − onnxruntime|` | Verdict |
| --- | --- | --- |
| `dynamo=True`, opset 18, dynamic batch | `1.490e-08` | works; needs `PYTHONUTF8=1` |
| `dynamo=False`, opset 17 (legacy TorchScript) | `1.863e-08` | works; deprecated since torch 2.9 |
| validation-script model, opset 17 | `2.980e-08` | gate uses this, tol `1e-5` |

The dynamic batch axis was confirmed genuinely dynamic (a batch of 9 through a graph
exported with batch 4 returned shape `(9, 2)`). The validation gate deliberately uses the
legacy path so the Phase 1 check does not itself depend on `PYTHONUTF8`; Phase 12 will
migrate to dynamo, which is the supported path going forward.

### 1.6 Documentation

| File | Contents |
| --- | --- |
| `docs/architecture.md` | Hybrid ML/physics diagram (Sensors → Calibration → AI → INS → Fusion → NHC → Map → Output), why that order, uncertainty flow, two-implementations-one-spec, repo map, runtime modes, toolchain decision |
| `docs/navigation_math.md` | **The specification.** Frames, WGS84 constants, geodetic↔ECEF↔ENU, quaternion conventions, gravity model, strapdown mechanization, 15-state error-state EKF with Jacobians, NHC, ZUPT, and how every convention is test-enforced |
| `docs/ml_pipeline.md` | Why AI at all, data flow, route/driver splitting, windowing, the three model stages, losses, reproducibility, export + parity, two-level evaluation, risks |
| `docs/map_matching.md` | HMM formulation, covariance-driven candidate search, emission/transition models, Viterbi, off-network detection, why fusion feedback is disabled by default |
| `README.md` | Entry point, layout table, setup commands |

The validation script checks that `architecture.md` contains all eight pipeline stages
**in the mandated order**, so the diagram cannot silently drift away from `AGENTS.md`.

Two quantitative findings were derived while writing `navigation_math.md`, both of which
change the implementation and are recorded there in full with reproducible arithmetic:

- **§5.2 — the dataset's `GRAVITY` channel is normalised to standard gravity `9.80665`,
  not local normal gravity.** At the dataset's location Somigliana + free-air gives
  `9.812381 m/s²` *(originally written as `9.812377` — an arithmetic slip corrected in
  Phase 2)*; the channel reports `9.806600`. The `0.005781 m/s²` difference is
  `≈ 10.4 m` of vertical error over a 60 s outage, so Phase 6 must use the gravity model
  and must not treat that channel as truth. *(Computed from one sample row; confirmed
  across all 72 unique sessions in Phase 2.)*
- **§6 — Coriolis and Earth-rate terms are not negligible.** At 30 m/s, Coriolis reaches
  `4.375e-3 m/s²` ≈ **7.9 m over 60 s**, and omitting Earth rate from attitude propagation
  drifts heading by `0.199°/minute` at this latitude. Both terms are retained.

### 1.7 Version control

`git init` run; `.gitignore` excludes the venv, all data trees, checkpoints, exported
binaries, generated reports/plots, Gradle/Android build output and secrets — while
deliberately **keeping** split manifests, model cards and phase validation reports, which
are provenance rather than artefacts. `.gitattributes` normalises line endings to LF, which
matters because golden-vector fixtures are shared between Python and Android and a CRLF
difference changes file hashes.

**Committed at the start of Phase 2** as `14bcbb5` ("Phase 1: repository scaffold, docs,
tooling, and validation reports"), on the user's instruction.

### 1.8 Exit criteria

- [x] Full monorepo structure created from a re-runnable manifest (82 entries)
- [x] Python version resolved on wheel evidence, not preference
- [x] `requirements.txt` + `pyproject.toml` with exact pins, asserted to agree
- [x] Isolated venv created; 22/22 dependencies installed at the pinned versions
- [x] Project installed editable; all 43 packages import, `navcore` remap verified on disk
- [x] Four foundational docs written, pipeline order test-enforced
- [x] `scripts/phase1_validate.py` written and run — **17/17 CRITICAL PASS**

### 1.9 Blockers and open items carried forward

| Item | Affects | Severity |
| --- | --- | --- |
| **No JDK / Android SDK / Gradle / adb** | Phases 12–14 | **High** — unchanged from Phase 0; long install lead time, worth starting now |
| No physical Android handset confirmed | Phases 13–14 | Medium — an emulator cannot produce real IMU data |
| **CUDA untested**: driver 555.97 (CUDA 12.5) vs the available `cu126` build | Phase 7 | Medium — expected to work via minor-version compatibility, **not verified**. Confirm before Phase 7; update to r560+ if `torch.cuda.is_available()` is False. |
| `.venv` sits inside a OneDrive-synced tree — **42,926 files, 1.27 GB logical / 1.4 GB on disk** | build hygiene | Low–Medium — the file *count* is what drives sync churn more than the size. It is gitignored; excluding `.venv` from OneDrive sync, or moving the workspace off OneDrive, is advisable. |
| Raw dataset still outside the repo, in OneDrive | Phase 2 | Low — intended (read-only), but derived multi-GB artefacts should target a non-synced path |
| `Categorised` vs `Uncategorised` duplication | Phase 2 | Medium — unresolved; could halve the effective dataset. Checksum first. |
| Citations in `architecture.md` §9 recorded from prior knowledge | reporting | Low — **must be verified against the actual publications** before appearing in any submitted report (`AGENTS.md` §2.2) |

---

## Phase 2 — Dataset Ingestion, Schema Normalisation & Integrity Audit

**Status:** `COMPLETED`
**Date:** 2026-09-27
**Evidence:** `reports/phase2_validation.txt` — **26/26 CRITICAL checks PASS**
**Report:** `reports/phase2/dataset_report.html` / `.json`, `reports/phase2/schema_validation.json`

**Re-run the whole phase with:**

```
.venv/Scripts/python.exe scripts/preprocess/normalize_raw_layout.py   # idempotent
.venv/Scripts/python.exe scripts/preprocess/validate_schema.py        # raw-level, pandas-free
.venv/Scripts/python.exe scripts/preprocess/ingest_iovnbd.py          # ~30 s, 8 workers
.venv/Scripts/python.exe scripts/phase2_validate.py                   # the gate
```

### 2.1 What was built

| File | Role |
| --- | --- |
| `scripts/preprocess/normalize_raw_layout.py` | Renames `data/raw/data raw  IO-VNBD/Synchronised…/Synchronised…/` to `data/raw/IO-VNBD/`. Same-volume rename; (path, size, mtime) inventory asserted identical before/after; idempotent |
| `training/preprocessing/iovnbd.py` | Loader: two schemas, header repair and mapping, satellite/Excel repair, explicit datetime, Europe/London→UTC, SI conversion, byte and content hashing |
| `training/preprocessing/iovnbd_audit.py` | Per-session timing, NaN, GNSS, gravity, kinematics, and phone↔vehicle cross-checks |
| `scripts/preprocess/validate_schema.py` | Independent raw check: decode, header map, per-line field count, ASCII body, row counts |
| `scripts/preprocess/ingest_iovnbd.py` | Dedupe → SI → audit → Parquet → split → report |
| `scripts/preprocess/dataset_report_html.py` | Renders the JSON report as a self-contained HTML page |
| `scripts/phase2_validate.py` | Gate: re-derives the claims from the artefacts |
| `tests/unit/test_iovnbd_loader.py` | 22 tests on inline, explicitly synthetic fixtures |

Outputs (gitignored, regenerable): `data/processed/iovnbd/v1/{phone,vehicle}/<session>.parquet`
(72 + 72 files, 110 MB, zstd; units, source path and source SHA-256 in each file's
metadata) and `manifest.json`. Tracked: `data/splits/iovnbd_split_v1.json`.

### 2.2 The dataset is not what Phase 0 assumed

Phase 0 took all 288 CSVs to be one 24-column smartphone schema. **They are two schemas:**

| Prefix | Columns | Content |
| --- | --- | --- |
| `S-*` | 24 | Smartphone: GNSS, accelerometer, gravity, gyro, magnetometer, orientation |
| `V-*` | 29 | Vehicle: VBOX GNSS (UTC time-of-day, 10 Hz) **plus CAN bus** — four wheel speeds, yaw rate, steering angle, indicated speed, long/lat acceleration, gear, rpm, pedals, brake |

The dataset therefore contains **wheel odometry and a CAN yaw-rate reference**, which
`docs/architecture.md` had explicitly ruled out. That changes what Phases 4, 7 and 11 can
do. Whether CAN channels may be model *inputs* is an open decision (§2.10).

### 2.3 Deduplication — settled by checksum

| | Result |
| --- | --- |
| CSV files | 288 → **72 unique sessions**, each an S + V pair |
| `V-*` pairs | **72/72 byte-identical** (SHA-256) |
| `S-*` pairs | **0/72 byte-identical, 72/72 content-identical** |
| Largest numeric difference inside any S pair | `7.1e-15` |

The S copies differ only in (a) float text — `1.539` vs `1.5390000000000001`, `-6` vs
`-6.0` — and (b) header labels. A byte checksum alone would have wrongly reported 144
distinct smartphone recordings. The loader therefore computes a **content digest** over
the parsed values (rounded to 1e-9, header labels excluded) as well as SHA-256. The
Categorised copy is canonical: it carries the driver/category directory metadata.

Unique data: **1,070,745 phone rows, 25.08 h, 1,341 km** (VBOX speed integrated with real
`dt`), 4 drivers, 7 categories, 66 route groups.

### 2.4 Parsing hazards — the 8 from Phase 0, plus the ones found here

| # | Hazard | Handling | Count |
| --- | --- | --- | --- |
| 1 | cp1252 encoding | Read as cp1252 (total over every byte present). **The header is mixed:** `°`/`μ` are UTF-8 byte pairs, `²` a single cp1252 byte, so no single codec decodes it correctly. Each token is repaired individually (`Â°` → `°`) | 288 files |
| 2 | Irregular headers | Keys normalised (NFKC, ASCII transliteration, whitespace, parens), then matched against an **explicit** list of accepted variants — never fuzzy | 3 variants |
| 2b | *new:* `DATE (…SS_SSS` without `)` | Uncategorised `S-Vfa01`/`S-Vfa02` only; accepted as a named variant | 2 files |
| 3 | `"18 / 19"` satellites | Split into `gps_sats_used` / `gps_sats_visible` (`Int16`); `used ≤ visible` holds on every row | 0 malformed |
| 3b | *new:* **Excel date damage** | The CSVs went through Excel, which turned `12 / 14` into `Dec-14` and `10 / 12` into `10-Dec` (UK d/m). Deterministic and lossless, so the integers are recovered exactly and rows flagged `gps_sats_excel_recovered` | **2,118 cells, 12 sessions** |
| 4 | `HH:MM:SS:mmm` datetime | Explicit `%Y-%m-%d %H:%M:%S:%f`; a decimal-point form is counted as a failure, not accepted | 0 failures |
| 4b | *new:* timezone | Phone `DATE` is UK civil time. Localised to Europe/London → UTC; DST-ambiguous times become NaT and are counted | 0 lost |
| 5 | km/h | → m/s at the loader (phone GPS, VBOX, CAN); CAN `g` × 9.80665; psi → Pa | — |
| 6 | Gyro Yaw/Pitch/Roll | See §2.6 — the labels are positional aliases, and "Yaw" on column 1 is **wrong** | — |
| 7 | Accel gravity-inclusive? | **Yes** — stationary \|a\| = 9.863 m/s² over 103,449 rows | settled |
| 8 | Categorised vs Uncategorised | §2.3 | settled |
| 9 | *new:* literal `nan` text | `S-Y1`: 94 bearing + 4 speed cells; coerced to NaN and counted | 98 cells |
| 10 | *new:* mislabelled V units | `Height (km)` is metres (MSL; minus phone ellipsoidal alt = −51.1 m ≈ −geoid); `Vertical velocity (km/hr)` behaves as m/s (dh/dt slope 0.986); pedal "0 or 1" spans 0–99 (%). Kept as `_raw` columns, not silently converted | — |
| 11 | *new:* wheel speed "rad/sec" | Ratio to VBOX speed in km/h is **1.0043** (1.0 to CAN indicated). Consistent with km/h, but rad/s would need a 0.277 m rolling radius — physically possible. **Unresolved**; kept `_raw` | Phase 4 |

`validate_schema.py` independently confirms: 288/288 files decode, map, and have **zero
ragged rows**, pure-ASCII bodies and CRLF endings; 4,283,560 data rows; Parquet row counts
equal raw line counts for all 144 canonical files.

### 2.5 Gravity — the 9.80665 vs local-gravity question, settled dataset-wide

| Quantity | m/s² |
| --- | --- |
| Standard gravity g₀ (defined) | 9.806650 |
| Local normal gravity (Somigliana + free-air), range over sessions | 9.812049 – 9.813116 |
| `GRAVITY` channel norm, pooled over 1.07 M rows | **9.806596** |
| `GRAVITY` norm, per-session means | 9.806592 – 9.806622 |
| Largest within-session std of the norm | 0.000043 |
| Sessions closer to g₀ than to local gravity | **72 / 72** |
| Stationary accelerometer norm (42 sessions) | 9.8627 (session range 9.837 – 10.100) |

**Conclusion for the filter:** the `GRAVITY` channel is Android's fused vector normalised to
g₀, not a measurement. Use its direction for levelling, never its magnitude; the INS uses
Somigliana (`navigation_math.md` §5.1), about 0.0058 m/s² ≈ 10.4 m over 60 s above g₀
here. The stationary accelerometer reads ~0.05 m/s² above local gravity; separating bias
from scale error is Phase 4.

**Correction to Phase 1:** `navigation_math.md` §5.2 had `sin²φ = 0.627722` (a hand-arithmetic
slip), giving 9.812377. The correct value is **0.627766 → 9.812381 m/s²**. The doc is fixed
and the value is now pinned by a unit test. No conclusion changes.

### 2.6 Gyroscope axis mapping — partly settled, with evidence

- **The labels are positional aliases.** The Uncategorised copies label the gyro columns
  `X/Y/Z` and orientation `Azimuth`; the Categorised copies relabel the *same bytes*
  `Yaw/Pitch/Roll` / `Yaw`.
- **Vehicle yaw is on column 2.** Against the CAN yaw rate (an independent sensor),
  time-aligned and 1 s smoothed, **21 of 22** strongly-correlated sessions (|r| ≥ 0.5) put
  it on `gyro_y` with sign **+**, median r 0.73, **gain 0.997** (so units and scale agree).
- **Accelerometer/gravity Z is vertical in 72/72 sessions.** So the gyro columns are *not*
  in the accelerometer's axis order, and the Categorised label "Yaw" on column 1 is wrong.
- **Not resolved:** the two horizontal gyro axes. The gravity-rotation test
  (`ω⊥ = −(g × ġ)/|g|²`) gave |r| < 0.3: the GRAVITY channel is too heavily smoothed and
  car pitch/roll rates too small. → Phase 4. Not guessed here.

### 2.7 The "Synchronised" S/V files are NOT row-synchronised

| Finding | Evidence |
| --- | --- |
| Phone `DATE` (as UTC) − VBOX UTC **on the same row index** is > 1 s in **49/72** sessions, from −168 s to **+314 s** | `clock_offset_s` per session |
| The gyro↔CAN-yaw best lag on row-paired data ≈ **−(that offset)**: S4 +313.75 → −313.9 s, Y1 +114.9 → −115.7 s, S2 +8.45 → −8.7 s | FFT xcorr, ±400 s |
| So the phone clock is right and the **row pairing is wrong** | — |
| The row offset **drifts within** M, S2, S3b, S4, Vta17, Vtb1, Y1 | 5 are concatenations of 2–4 phone recordings (`TIME SINCE START` resets); Vtb1 has **5,370 exact duplicate rows** walking the pairing off by up to 628 s |
| 9 sessions have unequal S/V row counts | Vfa01, Vfa02, Vta1b, Vtb10, Vw1, Vw15, Vw2, Vw4, Vw9 |

**Rule adopted:** join S and V on **UTC time** (nearest, ≤ 60 ms), **never on row index**.
The vehicle Parquet carries an absolute `t_utc` for this, and each file's metadata says so.

| Per-session verification (gyro vs CAN yaw) | Sessions |
| --- | --- |
| `time_join_verified` (r ≥ 0.5, residual clock correction −0.87…+1.52 s) | **25** |
| `time_join_contradicted` | **0** |
| `unverified_weak_signal` (r < 0.5) | 43 |
| `unverified_vehicle_not_moving` — **stationary-only recordings** `Vw1` (34 min), `Vw15` | 2 |
| `unverified_time_join_no_overlap` — `Vw7`, `Vw8`, whose phone clocks are ~165 s off | 2 |

Honest limits: even verified sessions agree only to about ±1.5 s, which includes CAN and
filter latencies. Before any V-file channel becomes a 10 Hz training target, Phase 7 must
refine alignment per session. **Phone GNSS lags VBOX by a median 4.1 s**, which Phase 8's
GNSS measurement model must account for.

### 2.8 Timing & missing values

| | Phone | Vehicle |
| --- | --- | --- |
| Intervals in (95, 105] ms | 1,062,455 of 1,070,673 (99.23 %) | 1,070,957 of 1,070,961 |
| Duplicate timestamps | 5,460 (5,456 in Vtb1) | 2 |
| Backwards steps (recording restarts) | 8 | 0 |
| Gaps > 1 s | 3 | 2 |
| Continuous segments (break on dt < 0 or > 1 s) | 83 | — |
| Exact duplicate rows | 5,374 | — |
| NaN | `gps_bearing_deg` 94, `gps_speed_mps` 4 (all S-Y1); every other column 0 | 0 |

Nothing was imputed or dropped: duplicates and NaNs remain in the Parquet, and counts are
in the report for downstream stages to handle explicitly.

### 2.9 Route/driver split — `data/splits/iovnbd_split_v1.json`

Per `docs/ml_pipeline.md` §3, assigned from real durations. Test = the held-out driver(s)
whose share is closest to 20 % (bounds 10–35 %); val = whole route groups of the remaining
drivers, seed 26168, never emptying a driver's train set.

| Split | Sessions | Hours | Fraction |
| --- | --- | --- | --- |
| train | 48 | 16.65 | 0.664 |
| val | 18 | 3.19 | 0.127 |
| test | 6 (**driver A**, held out entirely) | 5.25 | 0.209 |

Driver hours: A 5.25, B 1.71, D 1.86, E 16.27. Leak checks (gate-enforced): every session
in exactly one split, no route group (e.g. S3a/b/c, Vw14a/b/c) spans splits, and the test
driver appears nowhere else. *Caveat:* B and D have one session each, so they can only
ever be in train; val is therefore all driver E.

### 2.10 Exit criteria

- [x] Every one of the 288 files loads cleanly (0 excluded)
- [x] Integrity report with real counts, no estimates (`reports/phase2/`)
- [x] Duplication resolved by checksum (SHA-256 + content digest)
- [x] Split manifest committed and shown leak-free
- [x] Loader unit tests pass (22), including encoding and datetime edge cases
- [x] Raw data read-only: 144/144 source SHA-256s re-verified against Parquet metadata

### 2.11 Open items and decisions carried forward

| Item | Affects | Severity |
| --- | --- | --- |
| **Decision needed:** may CAN channels (wheel speed, yaw rate) be model *inputs*? A phone-only Android deployment has no CAN bus; using them as inputs would train a model that cannot run on-device. Default until decided: reference/evaluation truth only | Phases 4, 7, 11 | **High** |
| 45 of 72 sessions have unverified S/V alignment | Phase 7 targets from V | Medium |
| Horizontal gyro axes unresolved | Phases 4–6 | Medium |
| Wheel-speed unit (km/h vs rad/s label) | Phase 4 odometry | Low — calibrated against VBOX anyway |
| Phone GNSS ~4.1 s latency | Phase 8 | Medium |
| Stationary rows are audit-only; not exported as reusable segments | Phases 4, 8 | Low |
| No single dataset-path config module yet (path is a constant per script + `.env.example`) | hygiene | Low |
| WGS84/gravity constants duplicated in `iovnbd_audit.py` (test-asserted equal to the doc) until Phase 3 generates them | Phase 3 | Low |
| HTML report checked structurally only; the Chrome extension was not connected, so it was not visually reviewed | reporting | Low |
| Android built externally / in CI in later phases — Android Studio not installed locally (per user, 2026-09-27) | Phases 12–14 | Tracked |

---

## Phase 3 — Geodesy & Reference-Frame Foundations

**Status:** `NOT STARTED`

**Objective:** The single audited coordinate path that all position output must flow
through, per `AGENTS.md` §2.1.

**Planned scope**
- WGS84 constants from one source of truth, exported for both Python and Android.
- Geodetic ↔ ECEF ↔ local-tangent ENU conversions, plus NED where needed.
- Great-circle / geodesic distance for evaluation.
- Rotation library: quaternions, rotation matrices, Euler — with the convention declared
  once and asserted.
- Round-trip and analytically-known test cases (poles, equator, prime meridian,
  known baselines).

**Exit criteria**
- [ ] Round-trip conversions accurate to sub-millimetre over the dataset's operating region
- [ ] Rotation conventions documented and test-enforced
- [ ] Constants file generated, not hand-copied

---

## Phase 4 — Sensor Characterisation & Calibration

**Status:** `NOT STARTED`

**Objective:** Know each sensor's real error behaviour before trying to correct it.

**Planned scope**
- Stationary-segment detection from the real data (engine idle, traffic stops).
- Accelerometer and gyroscope bias, scale factor, and axis-misalignment estimation.
- Allan-variance analysis for noise density and random-walk terms — these become the EKF
  process-noise values rather than hand-tuned guesses.
- Magnetometer hard-iron / soft-iron ellipsoid fit; magnetic-disturbance rejection.
- Empirically establish the Gyroscope Yaw/Pitch/Roll to body-axis mapping (§0.4 hazard 6).
- Establish the accelerometer / gravity relationship (§0.4 hazard 7).
- Device-to-vehicle mounting-misalignment estimation.

**Exit criteria**
- [ ] Per-recording calibration parameters produced, with uncertainties
- [ ] Allan-variance plots and derived noise parameters committed
- [ ] Gyro axis mapping and gravity convention proven from data, not assumed
- [ ] Calibration raises rather than silently defaulting when inputs are missing

---

## Phase 5 — Attitude Estimation (AHRS)

**Status:** `NOT STARTED`

**Objective:** Reliable orientation, the largest single lever on dead-reckoning drift.

**Planned scope**
- Gyro integration baseline; quantify drift honestly.
- Complementary / Madgwick / Mahony filters and a quaternion EKF; compare on real data.
- Gravity-vector levelling for roll and pitch; magnetometer heading with disturbance gating.
- Static and in-motion initial alignment.
- Validation against the dataset's own `ORIENTATION` channels **and** against
  GPS-derived heading while moving, with the limitations of each reference stated.

**Exit criteria**
- [ ] Attitude error characterised over full routes with real numbers
- [ ] Heading drift rate reported per method
- [ ] Chosen filter justified by measured comparison, not preference

---

## Phase 6 — Strapdown INS Mechanization

**Status:** `NOT STARTED`

**Objective:** The physics core. Attitude → velocity → position by integration, nothing else.

**Planned scope**
- Strapdown equations in the local tangent frame with real non-uniform `dt`.
- Gravity model; Coriolis and transport-rate terms evaluated for relevance at vehicle speeds.
- Comparison of integration schemes (trapezoidal, RK, coning/sculling compensation).
- Pure-INS drift characterisation over 10 s / 30 s / 60 s / 300 s GNSS outages — the
  honest baseline every later phase must beat.

**Exit criteria**
- [ ] Open-loop INS drift curves from real IO-VNBD recordings
- [ ] Integration verified against analytically-known synthetic motion (clearly labelled fixtures)
- [ ] No GNSS field read anywhere in the INS code path

---

## Phase 7 — AI Signal Processing (Denoising / Bias + Motion Context / Speed)

**Status:** `NOT STARTED`

> Merged from the original Phases 6 and 7 — see the roadmap revision note above. The two
> stages share one windowed dataset, one training harness and one export path, so the
> split bought nothing but a phase boundary.

**Objective:** Learn what analytic calibration and physics cannot: the residual sensor-error
structure, and a **direct, non-integrating speed estimate**. That second output is the
mechanism that turns quadratic position drift into bounded drift, and is the single most
important model in the system. Outputs are physical quantities in SI units with
uncertainties — **never coordinates** (`AGENTS.md` §2.1).

**Planned scope — Stage A: denoising / residual bias regression**
- Windowed 10 Hz datasets built from the Phase 2 route/driver splits.
- Small sequence models (1-D CNN / TCN / GRU) sized for 4 GB VRAM and phone inference.
- Targets: accel/gyro bias and noise correction, supervised against GNSS-derived
  references where those references are valid.
- Ablation against Phase 4 analytic calibration — if AI does not beat physics, that is the
  reported result and the physics is kept.

**Planned scope — Stage B: motion context classification**
- Classes: stationary / accelerating / cruising / braking / turning, plus the dataset's
  own route categories.
- Feeds ZUPT triggering and NHC relaxation; operating point chosen on a cost-weighted
  curve because a false `stationary` permanently loses real distance.

**Planned scope — Stage C: speed / displacement / heading-change regression**
- Speed and displacement-magnitude regression from IMU windows.
- Heading-change regression as an independent check on the Phase 5 AHRS.
- Target-quality auditing: GNSS speed is noisy and GNSS bearing is meaningless at low
  speed, so windows failing a validity gate are excluded and the exclusion count reported.

**Planned scope — common to all stages**
- Per-prediction variance heads, so the Phase 8 filter can weight these properly.
- Heteroscedastic Gaussian NLL; variance calibration validated separately from RMSE.
- Evaluation on entirely held-out **drivers**, not just held-out routes.
- GPU swap (`requirements-gpu.txt`) installed and CUDA availability confirmed here —
  it is untested as of Phase 1.

**Exit criteria**
- [ ] Training reproducible from a committed script with a pinned seed; two fixed-seed runs agree
- [ ] Stage A improvement (or lack thereof) over Phase 4 reported on held-out routes
- [ ] Speed/displacement error characterised on held-out drivers
- [ ] Classifier confusion matrix and false-positive `stationary` rate committed
- [ ] Uncertainty calibration checked (predicted variance vs realised squared error)
- [ ] Parameter count, inference latency and model size recorded per model
- [ ] Zero coordinate targets anywhere in the training code, enforced by a static check

---

## Phase 8 — Fusion Engine (Error-State EKF)

**Status:** `NOT STARTED`

**Objective:** Fuse INS with AI estimates and GNSS-when-valid, with statistically
defensible uncertainty.

**Planned scope**
- Error-state (indirect) EKF: position, velocity, attitude, IMU bias states.
- Process noise from Phase 4 Allan variance; measurement noise from Phase 7 variance heads.
- Measurement models: GNSS position/velocity (gated by accuracy and satellite count),
  AI speed/displacement, ZUPT during detected stationarity.
- Innovation gating and outlier rejection; NIS/NEES consistency testing.
- Covariance symmetry and positive-definiteness assertions (`AGENTS.md` §3).
- GNSS-outage simulation driven by the evaluation harness masking columns, not by flags.

**Exit criteria**
- [ ] Filter consistency demonstrated (NEES within bounds), not merely low error
- [ ] Outage-duration vs error curves versus the Phase 6 pure-INS baseline
- [ ] Covariance assertions active and passing

---

## Phase 9 — Non-Holonomic Constraints (NHC)

**Status:** `NOT STARTED`

**Objective:** Exploit the fact that a road vehicle cannot slide sideways or fly.

**Planned scope**
- Body-frame lateral and vertical velocity pseudo-measurements constrained near zero.
- Bicycle-model yaw-rate / speed consistency coupling.
- Mounting-misalignment coupling with Phase 4 (a wrong body frame makes NHC actively harmful).
- Constraint relaxation on detected wheel slip, sharp manoeuvres, and rough terrain.
- Measured ablation: fusion with NHC vs without, on the same held-out routes.

**Exit criteria**
- [ ] NHC contribution quantified by ablation, with cases where it hurts reported too
- [ ] Constraint-relaxation logic validated against real turning and braking segments

---

## Phase 10 — Map Matching

**Status:** `NOT STARTED`

**Objective:** Snap the fused trajectory to the road network as a **refinement**. Per
`AGENTS.md` §1, map matching never originates a position.

**Planned scope**
- OSM road network for the Coventry, UK operating area; spatial index.
- HMM map matching (emission from filter covariance, transition from route topology).
- Off-network detection — car parks and private roads must not be force-snapped.
- Optional feedback of matched heading into fusion, only with an honest assessment of
  the correlation risk it introduces.
- Ablation with map matching disabled.

**Exit criteria**
- [ ] Match rate and error improvement measured on real routes
- [ ] Failure modes documented (parallel roads, junctions, off-network travel)
- [ ] Positions still traceable to integrated physics with map matching removed

---

## Phase 11 — Evaluation Harness & Benchmarking

**Status:** `NOT STARTED`

**Objective:** One harness that produces every number that appears in any report.

**Planned scope**
- Metrics: absolute trajectory error, relative pose error, CDF percentiles (50/68/95),
  final-position error, drift as a percentage of distance travelled, heading error.
- Outage-scenario battery: outage durations from seconds to several minutes, at varied
  start points across routes.
- GNSS masking enforced at the harness level.
- Per-stage ablation table: INS → +AI → +Fusion → +NHC → +Map Matching.
- Every result stamped with script, commit, split, seed, and timestamp (`AGENTS.md` §2.2).

**Exit criteria**
- [ ] Single command regenerates every published number
- [ ] Ablation table complete, including regressions where they exist
- [ ] Results reproduce across two runs with a fixed seed

---

## Phase 12 — Model Export & Python/Android Numerical Parity

**Status:** `NOT STARTED`
**Depends on:** resolving the JDK / Android SDK blocker (§0.8)

**Objective:** Enforce the parity contract in `AGENTS.md` §4.

**Planned scope**
- Export Phase 7–7 models to ONNX and/or TFLite; re-run the exported graph on the
  trainer's own inputs and compare numerically before acceptance.
- Generate the shared constants file for both languages from one source.
- Golden-vector fixtures covering geodesy, rotations, INS steps, filter updates, NHC,
  and the full pipeline.
- Cross-language parity test runner reporting maximum absolute and relative deviation.
- Explicit float/double policy per module, documented.

**Exit criteria**
- [ ] Exported models match the Python trainer within a stated tolerance
- [ ] Golden-vector parity within the §4 targets across all modules
- [ ] Parity suite runnable in CI; any failure blocks release

---

## Phase 13 — Android Application

**Status:** `NOT STARTED`
**Depends on:** JDK + Android SDK + Gradle installation, and a real test handset

**Objective:** Run the identical pipeline on-device, in real time, in a GNSS outage.

**Planned scope**
- Sensor acquisition with real hardware timestamps and honest sample-rate handling.
- Port of calibration, AHRS, INS, EKF, and NHC against the Phase 12 golden vectors.
- On-device inference (ONNX Runtime Mobile / TFLite / NNAPI), with the export verified.
- Background-service lifecycle, battery and thermal behaviour measured, not estimated.
- Offline map tiles and on-device map matching.
- On-device parity harness: replay a golden recording, compare against Python output.
- UI showing position, uncertainty, and current pipeline mode.

**Exit criteria**
- [ ] On-device replay of a golden recording matches Python within tolerance
- [ ] Real-time performance measured on real hardware (latency, CPU, battery, thermals)
- [ ] Works fully offline with GNSS disabled
- [ ] No coordinate ever produced outside the ported geodesy path

---

## Phase 14 — End-to-End Validation, Documentation & Demo

**Status:** `NOT STARTED`

**Objective:** Prove the system on data it has never seen, and describe it honestly.

**Planned scope**
- Field collection on new routes (tunnels, underpasses, urban canyon, multi-storey car
  parks) with GNSS truth logged for evaluation only.
- Comparison of on-device results against the Phase 11 Python benchmarks.
- Architecture documentation, algorithm specification, dataset card, model cards.
- Limitations document: where the system fails and why.
- Reproducibility guide — clean-machine setup to published results.
- SIH demo materials, with every claim traceable to a re-runnable measurement.

**Exit criteria**
- [ ] Field results reported for novel routes, including failures
- [ ] Documentation complete, no unsupported numbers anywhere
- [ ] A reviewer can reproduce the headline results from a clean checkout

---

## Change Log

| Date | Change |
| --- | --- |
| 2026-09-27 | Phase 0 opened and COMPLETED. Environment, toolchain, network, and dataset surveyed. `AGENTS.md`, `PROJECT_STATUS.md`, `TODO.md` created. IO-VNBD confirmed present and real (288 CSV files, 816 MB, 10 Hz, Coventry UK). JDK/Android SDK found absent and logged as a high-severity blocker for Phases 12–14. Validation script `scripts/phase0_validate.py` written and run: 16/16 CRITICAL checks PASS, output in `reports/phase0_validation.txt`. |
| 2026-09-27 | Phase 1 COMPLETED: monorepo structure created from a re-runnable manifest (82 dirs, 51 packages); git repo initialised with `.gitignore`/`.gitattributes`; Python 3.14.4 chosen on wheel evidence; venv built and 22 pinned dependencies installed at exact versions; `navigation-core` mapped to the `navcore` import namespace; `docs/architecture.md`, `navigation_math.md`, `ml_pipeline.md`, `map_matching.md` written; `scripts/phase1_validate.py` run with 17/17 CRITICAL checks PASS. Roadmap renumbered (see revision note); two AI phases merged to keep the range at 0-14. |
| 2026-09-27 | Phase 2 COMPLETED: raw layout normalised to `data/raw/IO-VNBD/`; two schemas found (S phone 24-col, V vehicle/CAN 29-col); 288 files → 72 unique sessions (V pairs byte-identical, S pairs content-identical, max diff 7.1e-15); 2,118 Excel-damaged satellite cells recovered; GRAVITY channel shown normalised to g0 in 72/72 sessions; gyro vertical axis = column 2 (gain 0.997); S/V row pairing shown misaligned by the phone clock error (up to 314 s), UTC-time join adopted (25 verified, 0 contradicted); driver-held-out split v1; `navigation_math.md` §5.2 arithmetic corrected (9.812377 → 9.812381); `scripts/phase2_validate.py` 26/26 CRITICAL PASS. |
