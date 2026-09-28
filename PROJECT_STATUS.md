# PROJECT_STATUS.md — SIH26168

**Project:** AI-ML based Intelligent Dead Reckoning system for seamless navigation
**Workspace:** `C:\Users\Shreeyash\OneDrive\Desktop\dead reckening`
**Governing document:** [`AGENTS.md`](./AGENTS.md) — the core directive overrides anything here.
**Last updated:** 2026-09-28 (Phase 5-9 crosswalk to TODO.md; road bundle for any area; Phase 12/13 labels in code)

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
| 3 | Preprocessing Pipeline & Coordinate Systems | **COMPLETED** |
| 4 | Baseline Navigation Core (sensors, alignment, INS, EKF) | **COMPLETED** |
| 5 | ML Training Pipeline (pipeline + smoke test; full training not yet run) | **COMPLETED** |
| 6 | ML Evaluation & Export (sweep, selection, evaluation, ONNX + parity gate) | **COMPLETED** |
| 7 | Sensor Fusion & NHC (AI-assisted EKF, NHC, GNSS state machine; real-data gain mixed) | **COMPLETED** |
| 8 | Offline Map Matching (OSM road graph, HMM matcher; real gain only while drift < ~30 m). *TODO.md 8 is "Fusion Engine (Error-State EKF)": see crosswalk* | **COMPLETED** |
| 9 | GNSS Outage Detection, Recovery & Replay (code + tests; recovery manager OFF pending real-drive evidence). *TODO.md 9 is "Non-Holonomic Constraints (NHC)": see crosswalk* | **COMPLETED** |
| 10 | *(TODO.md "Map Matching": delivered as Phase 8 above)* | — |
| 11 | Evaluation Harness & Benchmarking | `NOT STARTED` |
| 12 | Model Export & Python/Android Numerical Parity (Python side + golden vectors done; Kotlin parity tests + CI written, **never run**) | `IN PROGRESS` |
| 13 | Android Application (Part A base & sensors, Part B ML & dead reckoning: source complete, **never compiled**) | `BLOCKED` (build machine + handset) |
| 14 | End-to-End Validation, Documentation & Demo | `NOT STARTED` |

> **Roadmap revision (2026-09-27).** "Architecture & Repository Creation" was inserted as
> Phase 1, displacing the original numbering. To keep the roadmap at 0-14 as originally
> scoped, the two AI phases (originally 6 "Denoising & Bias" and 7 "Motion Context &
> Displacement") are merged into a single Phase 7 — they share one data pipeline, one
> training harness and one export path, and were already noted as parallelisable.
> Phases 8-14 keep their original numbers. Cross-references in `docs/` and `TODO.md` were
> renumbered mechanically in the same commit.
>
> **Renumbering (2026-09-28), Android work.** The Kotlin work committed as "Phase 10"
> (`4556c2e`) and "Phase 11" (`f832719`) is re-filed to match TODO.md: its parity parts
> under **Phase 12**, the app under **Phase 13** (Parts A and B). Phase 11 is the
> Evaluation Harness again. Commit messages keep the old labels; code comments, docstrings
> and the app's `versionName` (`0.13.0-phase13`) were updated to Phases 12/13. Remaining
> "Phase 10" mentions in `docs/map_matching.md`, `requirements*.txt` and `README.md` mean
> TODO.md's Phase 10 (Map Matching), delivered as Phase 8 below.

**Crosswalk, Phases 5-9: executed phase (this file) vs TODO.md's title.** Phases 5-9 were
redefined by the owner as the work went, so from 8 on the executed phase under a number is
not the phase TODO.md lists under it. Each TODO.md scope is either delivered elsewhere or
still open:

| # | TODO.md title | Executed here under that number | Where TODO.md's scope for that number stands |
| --- | --- | --- | --- |
| 5 | ML Training Pipeline (was AHRS) | same | done. Original AHRS scope not scheduled (attitude from the Phase 4 EKF + gravity levelling) |
| 6 | ML Evaluation & Export | same | done |
| 7 | Sensor Fusion & NHC | same | done. Also delivers much of TODO.md 8 and 9 (below) |
| 8 | Fusion Engine (Error-State EKF) | Offline Map Matching (= TODO.md **10**) | 15-state EKF built in Phase 4 (Monte Carlo NEES/NIS consistent); AI-speed update, NIS gating and stillness ZUPT added in Phase 7. TODO.md's Phase 8 checklist is ticked against this work; open: real-data consistency (2σ covers 75 %), satellite-count gating, displacement model |
| 9 | Non-Holonomic Constraints (NHC) | GNSS Outage Detection, Recovery & Replay (feeds TODO.md **11**'s outage battery) | NHC built in Phase 7; open per §7.7: correlated-error model, time-varying tilt, lateral NHC untested on real data |
| 10 | Map Matching | *(not reused)* | delivered as Phase 8 |

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
   **⚠ WRONG — corrected in Phase 3:** the header says `Kmh` but the values are **m/s**
   (Android `Location.getSpeed`). Logged / VBOX [m/s] = 0.997 over 40 sessions
   (`reports/phase3/phone_gps_speed_unit.json`). Phases 0–2 divided by 3.6.
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

Unique data: **1,070,745 phone rows, 29.56 h, 1,341 km** (VBOX speed integrated with real
`dt`), 4 drivers, 7 categories, 66 route groups.

> **Correction (Phase 3).** Phase 2 originally reported **25.08 h**. That figure was
> wrong: session duration was computed as `TIME SINCE START` last − first, which breaks
> in the 5 concatenated sessions where that clock resets (S3b came out *negative*,
> S4 0.10 h instead of 2.63 h). Durations are now the sum of real intervals
> `0 < dt ≤ 1 s` (`iovnbd_audit.recorded_duration_s`, regression-tested), and the
> Phase 2 report was regenerated. The v1 split had been balanced on the wrong numbers.

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
| 5 | km/h | → m/s at the loader (VBOX, CAN); CAN `g` × 9.80665; psi → Pa. **Phone GPS speed: the "Kmh" label is wrong, values are m/s** — found and fixed in Phase 3 (§3.4) | — |
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

> **SUPERSEDED in Phase 3 by `iovnbd_split_v2.json` (§3.6).** v1 grouped sessions only by
> route-group letter suffix. Most IO-VNBD sessions are consecutive slices of one drive, a
> few seconds apart, so v1 put adjacent samples of the same drive on both sides of a
> split — **49 violations** under the drive-adjacency rule (e.g. Vta1b in train ends 6.2 s
> before Vta2 in val starts). v1 is kept only as a record; nothing may train on it. The
> table below is from the original Phase 2 run, which also used the wrong durations
> (§2.3 correction).

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
| **DECIDED 2026-09-27 (user): CAN channels are ground truth only, never inputs** — enforced in Phase 3 §3.5. Original question: may CAN channels (wheel speed, yaw rate) be model *inputs*? A phone-only Android deployment has no CAN bus; using them as inputs would train a model that cannot run on-device. Default until decided: reference/evaluation truth only | Phases 4, 7, 11 | **High** |
| 45 of 72 sessions have unverified S/V alignment | Phase 7 targets from V | Medium |
| Horizontal gyro axes unresolved | Phases 4–6 | Medium |
| Wheel-speed unit (km/h vs rad/s label) | Phase 4 odometry | Low — calibrated against VBOX anyway |
| ~~Phone GNSS ~4.1 s latency~~ -- WRONG, a 9 s sample-and-hold artefact (Phase 4 §4.2) | -- | resolved |
| Stationary rows are audit-only; not exported as reusable segments | Phases 4, 8 | Low |
| No single dataset-path config module yet (path is a constant per script + `.env.example`) | hygiene | Low |
| WGS84/gravity constants duplicated in `iovnbd_audit.py` (test-asserted equal to the doc) until Phase 3 generates them | Phase 3 | Low |
| HTML report checked structurally only; the Chrome extension was not connected, so it was not visually reviewed | reporting | Low |
| Android built externally / in CI in later phases — Android Studio not installed locally (per user, 2026-09-27) | Phases 12–14 | Tracked |

---

## Phase 3 — Preprocessing Pipeline & Coordinate Systems

**Status:** `COMPLETED`
**Date:** 2026-09-27
**Evidence:** `reports/phase3_validation.txt` — **20/20 CRITICAL checks PASS**; 128 tests pass; `ruff check .` clean
**Scope note:** renamed from "Geodesy & Reference-Frame Foundations" at the user's
direction; the original geodesy/frames scope is included, plus the preprocessing pipeline.

**Re-run with:**

```
.venv/Scripts/python.exe -m training.preprocessing.pipeline   # ~17 s: calibrate, sync, split, normalise
.venv/Scripts/python.exe scripts/phase3_validate.py           # the gate (runs lint, tests, mutation check)
```

### 3.1 Housekeeping requested at the end of Phase 2

| Item | Result |
| --- | --- |
| Phase 2 commit | `e22997b` |
| CAN data rule | **Ground truth only, never a model/filter input** — enforced in code (§3.5), not by convention |
| `ruff` | Already pinned `ruff==0.16.9` in `requirements-dev.txt` but never installed; now installed at that version. 16 pre-existing findings fixed; `known-first-party` set so `navcore` imports sort correctly; `SIM300` ignored in tests (misfires on `CONST == pytest.approx(...)`) |
| `configs/dataset.yaml` | Created; loaded only via `training/preprocessing/config.py` (env override `SIH26168_DATASET_ROOT`; missing keys raise). All Phase 0/2 scripts switched to it — including `phase0_validate.py`, whose hardcoded `Desktop\SIH26168\…` path no longer existed. The gate greps for path construction in code |
| Single constants source | `navigation-core/common/constants.py`; the Phase 2 duplicate WGS84/gravity constants in the audit and loader were removed |

### 3.2 Wheel-speed unit — resolved as far as the data allows (`reports/phase3/wheel_speed_calibration.json`)

The label question (km/h vs rad/s) **cannot be proven from this data**: the two hypotheses
differ by a constant factor, and every kinematic relation (speed, left/right turn
differential, gear ratios) scales identically. What the ground truth needs — the factor to
m/s and its accuracy — is measured against VBOX Doppler speed:

| | Result |
| --- | --- |
| Driven axle | **Front-wheel drive**: (front − rear) correlates with longitudinal acceleration in 98.6 % of sessions, so the **rear (non-driven) axle** is the slip-free reference |
| Rear-axle mean ÷ VBOX speed in km/h | **k = 1.00019** (330,853 steady, straight, well-tracked samples, 62 sessions; session IQR 0.9991–1.0031) |
| Residual of `rear / (3.6 k)` vs VBOX | median 0.000, median \|·\| 0.058, p95 \|·\| 0.233 m/s |
| Speed dependence (honest limit) | ratio 1.0033 at 5–10 m/s → 0.9943 at 30–45 m/s; bias **−0.19 m/s above 30 m/s** (0.6 %) — not corrected, documented |
| Label decision | **km/h, by strong inference**: under rad/s the rear rolling radius would have to be 0.27772 m ≈ exactly 1/3.6 m |

`gt_wheel_speed_mps = mean(RL, RR) / (3.6 × 1.00019)`.

### 3.3 Coordinate frames — `navigation-core/geometry/`

| Module | Content |
| --- | --- |
| `quaternion.py` | `[w,x,y,z]`, Hamilton, `q_ab ≙ R_b^a` (spec §4.1). Product, rotate, conj/inverse, DCM ↔ q (Shepperd, stable at 180°), exp/log with the shared small-angle Taylor branch (`SMALL_ANGLE_THRESHOLD_RAD = 1e-6`), integration with real `dt`, slerp (short arc), ZYX Euler with explicit gimbal-lock handling, ENU-yaw ↔ compass heading |
| `rotation.py` | skew/unskew, elementary rotations, Euler ↔ DCM, proper-rotation check (rejects reflections), SVD orthonormalisation |
| `geodesy.py` | WGS84 geodetic ↔ ECEF (Heikkinen closed form for the inverse: exact, non-iterative, parity-friendly), `R_e^n`, **`LocalTangentPlane` with an immutable origin**, ENU ↔ NED, radii of curvature, Somigliana gravity |
| `frames.py` | `Frame` enum (BODY = DEVICE, VEHICLE [FLU], ENU, NED, ECEF, GEODETIC), `FramedVector`, `Rotation(dst ← src)` stored as a quaternion. Applying or composing across mismatched frames raises `FrameMismatchError`; untagged arrays are refused; GEODETIC is not a vector frame |

**Tests (`tests/navigation/`, 78):** the conventions §11 of the spec requires (Hamilton
`ij = k`, active right-handed rotation, composition order, ENU axis order), algebraic
identities on 1000s of seeded random quaternions, DCM/rotvec/Euler round trips including
180° and gimbal lock, small-angle branch continuity from 0 to 1e-4 rad, constant-rate
integration equal to the exact rotation to 1e-11 rad, geodesy vs **pyproj as an independent
oracle** (5,000 points, ≤ 1e-6 m), round-trip closure **< 1e-4 m globally** (20,000 points,
−1 km to 100 km, poles included), and frame-mismatch rejection.

**Mutation check (in the gate):** injecting a JPL product, a passive rotation, a transposed
`R_e^n`, a removed small-angle branch, or an ignored height makes the suite fail — **5/5
caught** — so the tests pin the conventions rather than merely running them.

One test bug was found and fixed while writing them: "1 m east" at height h uses
radius `(N + h)cos φ`, not `N cos φ`; the geodesy code was right.

### 3.4 Synchronisation — `training/preprocessing/sync.py`

One frame per session on the phone timeline, joined to ground truth by **UTC time**
(nearest ≤ 60 ms), never by row index:

| Step | Result |
| --- | --- |
| Rows | 1,070,745 in → **1,065,371** out; **5,374 exact duplicate rows dropped and counted**; 86 duplicate timestamps with different values kept (dt = 0) |
| Clock | Phase 2's measured phone-clock correction applied in the **25 verified** sessions (−0.87…+1.52 s); 0 elsewhere, flagged `alignment_verified = False` |
| Ground truth matched | **98.96 %** of phone rows |
| Time-join sanity (recomputed by the gate) | gyro_y vs CAN yaw rate, verified sessions: median r = **0.742** over 23 sessions |
| Segments | 83 (break where real `dt` > 1 s) |
| Stationary rows (`gt_stationary`) | 103,447 — the Phase 2 carry-over, now exported for Phases 4 and 8 |

> **⚠ SUPERSEDED — this paragraph is WRONG; corrected in Phase 4 (§4.2).** The phone
> logs a new GNSS fix only every ~9 s in 64 of the 67 moving sessions and repeats it on
> every row in between (sample-and-hold). The "4.1 s latency" below was measured by
> correlating that *held* column with truth, so it measured the average *age* of a held
> fix, not a delay. The true reporting delay of a new fix is ~0 s. The pipeline now tags
> every row with the epoch of the fix it holds plus `ph_gnss_fix_age_s`. The text is kept
> as a record of the error.

**Phone GNSS latency (the "4.1 s").** The fix is kept **exactly as received** — shifting it
earlier would feed future information to a real-time filter — and every row carries
`ph_gnss_epoch_utc = t_utc − latency`, the instant the fix actually describes (the
delayed-measurement form Phase 8 needs). Latency is measured per session (phone GNSS speed
vs VBOX): **41 sessions measured, pooled median 4.1 s, IQR 3.1–4.7 s, range 0.1–8.3 s**;
the other 31 use the pooled median, and the source is recorded per session. Checked by the
gate: comparing each fix with truth *at its epoch* lowers the median speed error from
**0.82 to 0.53 m/s**, better in 40 of 41 sessions.

**Two defects found by the gate's value-level checks — both would have survived a
"runs without error" standard:**

1. **Phone GPS speed is m/s, not km/h.** The header says `Kmh`; logged ÷ VBOX [m/s] =
   **0.997** (IQR 0.993–1.000, 40 sessions) — the km/h reading predicts 3.6. Phases 0–2
   divided by 3.6, so every phone speed was 3.6× too small. Latency estimation is
   scale-invariant and could not see it; the error-magnitude check could. Fixed in the
   loader, regression-tested, evidence in `reports/phase3/phone_gps_speed_unit.json`.
2. **Cross-correlation lag bias.** The Phase 2 `fft_xcorr_lag` maximised the raw xcorr
   sum, which is weighted by overlap length and pulls broad peaks (slow signals like speed)
   toward zero — a known 3.0 s synthetic lag read as 2.0 s. Now normalised by overlap and
   refined with exact Pearson correlation. **Phase 2 conclusions re-run: unchanged**
   (alignment statuses, clock corrections and gyro results identical — yaw-rate peaks are
   sharp). The Phase 3 GNSS latency was the affected quantity (biased 3.75 s → 4.1 s).

### 3.5 The CAN rule, enforced — `training/preprocessing/columns.py`

Synced columns are exactly `META + INPUT + GT`. All 15 vehicle-derived columns are
`gt_*`. `assert_model_inputs` rejects any `gt_`, `can_`, `wheel`, `steering`, `yaw_rate`,
`vbox_`, `brake`, `gear`, `engine`, `pedal` or `clutch` column and any undeclared input;
`assert_model_targets` rejects latitude/longitude targets (AGENTS.md §2.1). Normalisation
calls the input guard, each synced file records its input/ground-truth lists in its
metadata, and the gate re-checks them.

### 3.6 Leak-free split v2 — `data/splits/iovnbd_split_v2.json`

**Unit = the drive**: sessions connected by the same driver AND (same route group OR a gap
under 600 s). The largest same-drive gap observed is 444 s; the next gaps are 809 s and
over. 72 sessions → **14 drives** (e.g. `E:Vta1a` = 28 consecutive sessions, 2.44 h).

| Split | Drives | Sessions | Hours | Fraction |
| --- | --- | --- | --- | --- |
| train | 10 | 56 | 20.48 | 0.693 |
| val | 2 (`A:S3a`, `E:Vtb2`) | 14 | 4.18 | 0.141 |
| test | 2 (**drivers B and D held out entirely**) | 2 | 4.90 | 0.166 |

With the corrected durations (A 8.58 h, B 2.94, D 1.95, E 16.08), the held-out-driver rule
selects **B + D** (16.6 %, closest to 20 %) rather than v1's A. `find_leaks` re-derives
temporal adjacency from the timestamps (it does not trust `drive_id`): **v2: 0
violations; v1: 49.** *Caveat:* the test set is two drivers with one long drive each —
a real held-out-driver test, but a small one (2 drives).

### 3.7 Normalisation — `models/normalization/imu_{mean,std}.json` (tracked)

Per-channel mean and population std over the **56 train sessions only** (738,516 samples,
float64 two-pass) for 12 phone channels (accel, gyro, mag, gravity). Each file records the
sessions used, the split manifest path and its SHA-256, script and commit. The code raises
`SplitLeakError` if offered a val/test session, and the gate recomputes the files from
scratch (difference 0.0).

| | acc x/y/z m/s² | gyro x/y/z rad/s | grav z |
| --- | --- | --- | --- |
| mean | 0.023 / −0.058 / 9.841 | −0.0004 / −0.0021 / −0.0001 | 9.8065 |
| std | 1.887 / 1.732 / 0.831 | 0.122 / **0.257** / 0.161 | 0.0019 |

`gyro_y` having the largest spread is consistent with it being the yaw axis (§2.6).
`.gitignore` gained `!models/normalization/*.json`: these small files are part of the
parity contract, while checkpoints stay ignored.

### 3.8 Exit criteria

- [x] Reproducible preprocessing pipeline, one command, deterministic
- [x] Phone/vehicle synchronised on UTC time; GNSS latency measured and epoch-tagged
- [x] Frame system (Device/Body, Vehicle, ENU, NED, ECEF, WGS84) with quaternions internally
- [x] Strict math tests incl. pyproj oracle and mutation check; round trip < 1e-4 m
- [x] Group-based split with an independent leak audit; adjacent samples never split
- [x] Normalisation from train only, reproducible, leak-guarded
- [x] ruff clean; all 128 tests pass

### 3.9 Not done / carried forward

| Item | Affects |
| --- | --- |
| **Geodesic (Vincenty/Karney) distance for evaluation** (original Phase 3 scope) — not implemented; pyproj's `Geod` is the planned oracle | Phase 11 |
| Constants **generated** for Kotlin — Python single source exists; generator deferred | Phase 12 |
| Horizontal gyro axes still unresolved; `BODY` mapping of raw gyro columns is a Phase 4 output | Phases 4–6 |
| 47 sessions keep unverified S/V alignment (flagged per row) | Phase 7 labels |
| Wheel-speed GT has a −0.19 m/s bias above 30 m/s | Phase 11 |
| ~~GNSS latency varies 0.1–8.3 s~~ -- superseded: new-fix delay ~0 s; the real constraint is the 9 s fix interval (Phase 4 §4.2) | Phase 8 |
| mypy not run (not requested; not installed) | hygiene |

---

## Phase 4 — Baseline Navigation Core (sensors, alignment, INS, EKF)

**Status:** `COMPLETED`
**Date:** 2026-09-27
**Commit:** code in `13759cf`; final evaluation files and this write-up in the follow-up commit
**Evidence:** `reports/phase4_validation.txt` — **12/12 CRITICAL checks PASS**; 179 tests pass; `ruff check .` clean
**Scope note:** redefined by the user as "Baseline Navigation Core". It absorbs the INS
(original Phase 6), the classical EKF core (original Phase 8) and the calibration items of
the original Phase 4. Original Phases 5–14 keep their titles below and are re-scoped as
they are reached.

**Re-run with:**

```
.venv/Scripts/python.exe scripts/analysis/gyro_axis_resolution.py
.venv/Scripts/python.exe scripts/analysis/imu_noise.py
.venv/Scripts/python.exe scripts/analysis/mounting_alignment.py
.venv/Scripts/python.exe scripts/evaluate/phase4_baseline.py [--all-verified]
.venv/Scripts/python.exe scripts/phase4_validate.py
```

### 4.1 What was built (`navigation-core/`)

| Module | Content | Tests |
| --- | --- | --- |
| `sensors/samples.py` | `SensorSample`, `ImuSample`, `GnssSample` (frame-tagged, validated, raise on bad input; GNSS **epoch** vs **receipt** time), `TimestampGuard`, `ImuAxisMap` + evidence-based `IOVNBD_AXIS_MAP` | 7 |
| `alignment/mounting.py` | R_b^v: gravity levelling (pitch/roll) + mount yaw from GNSS-derived along-track/centripetal acceleration (closed-form 2-D Procrustes), with r²/scale **observability flags** | 8 |
| `ins/mechanization.py` | ENU strapdown: q ← exp(−ω_in dt) ⊗ q ⊗ exp(ω_ib dt); Heun velocity with **Coriolis + transport rate**; **Somigliana gravity**; curvilinear position + tangent-plane cross-check form. Reads no GNSS (gate-checked) | 17 |
| `filtering/ekf.py` | 15-state error-state EKF: `predict`, epoch-aware `update_gnss` (position + horizontal velocity, NIS-gated), `update_zupt`, `update_levelling`; Joseph form, symmetrisation, Cholesky PD guard that **raises** | 13 |
| `recovery/gnss_reset.py` | Gate lock-out recovery: reset to GNSS after 3 consecutive rejected fixes; attitude inflation capped | 2 |
| `calibration/allan.py` | Overlapping Allan deviation; white-noise density and bias instability | 3 |

### 4.2 Findings that change the plan

**1. The horizontal gyro axes are not angular rates.** Stop-to-stop test (142 pairs; the
accelerometer at a stop measures the exact tilt, and the gyro bias is observable there). For all 8 signed
permutations of columns 1/3, integrating them predicts the tilt change worse than
ignoring them: **best 5.95° vs 1.49°** median, never better in more than 13 % of pairs. Three other
references (GRAVITY rotation, ORIENTATION derivatives, road grade) were also negative.
→ IO-VNBD is a **reduced IMU**: vertical gyro (column 2: counter-clockwise +, gain 0.9–1.0 vs VBOX heading
rate) + 3-axis accelerometer; pitch/roll come from gravity levelling.

**2. The Phase 3 "4.1 s GNSS latency" was wrong.** The phone logs a new fix only every
**~9 s** in 64/67 moving sessions (1 Hz only in Vta1a, Vta1b, Vta2) and repeats it on
every row. Correlating the held column against truth measured the average *age* of a
held fix. Measured per new fix, the **reporting delay is ~0 s** (pooled median 0.0,
IQR −0.7…0.1 s). Sync now tags each row with its held fix's epoch plus
`ph_gnss_fix_age_s`, and clamps negative estimates (residual clock offset) at 0 so epochs stay
causal. With fixes at their true epochs, phone-vs-truth speed error drops from **0.82 to 0.15 m/s** (33/33
sessions). Phase 3 gate re-run: 20/20. §3.4 is annotated as superseded.

**3. Spec sign error.** `docs/navigation_math.md` §8.3 printed `+[f^n]×ψ`. Under the
spec's own truth-minus-estimate convention the finite-difference Jacobian test rejects it
(error ≈ 2g·Δt) and confirms `−[f^n]×ψ`. Doc corrected.

**4. The phone mount is not rigid.** Phone-only mount yaw is observable (r² ≥ 0.3) in
**8/68** sessions, and there it agrees with the VBOX-referenced fit to **1.5° median
(p90 8.7°)**. Even the 10 Hz VBOX reference is acceptable in only 28/68. The estimate differs
between halves of a drive by a median 12.7° (n = 2), and between consecutive sessions of
one drive by tens of degrees. Tilt is small: median 0.7°, max 6.8°. NHC (Phase 9) will need
per-segment re-alignment.

**5. Scoring requires a verified clock.** In unverified sessions a ~1 s residual phone
clock error puts the truth lookup ~15 m off (Vtb2: phone fixes "18.6 m" vs 2.8 m in
verified S3a). Evaluation is restricted to `alignment_verified` sessions.

### 4.3 Verification of the math

| Check | Result |
| --- | --- |
| 1 h at rest on the rotating Earth, 3 attitudes | position < 1 mm, velocity < 1e-6 m/s, attitude < 1e-9 rad |
| Constant 30 m/s along a parallel for 18 km (Coriolis + transport) | stays on the parallel to < 1 cm |
| Coriolis left uncompensated | deflects **right** (south) at the predicted rate, within 2 % |
| Earth rate omitted | heading drifts 0.199°/min as §6 predicts |
| g₀ fed instead of local γ | INS falls at ½·Δγ·t², within 1 % |
| Curvilinear vs tangent-plane position over 1.8 km | 1 cm horizontal, < 0.3 m vertical (curvature) |
| EKF F vs finite differences of the real propagation | agree to 5e-4 at dt = 0.01 s (all 15 × 15) |
| GNSS / levelling H vs finite differences | exact / quadratic remainder (ratio < 0.02 per 10× step) |
| Monte Carlo consistency (20 runs, model-consistent truth) | **mean NEES 16.5 (expected 15), mean NIS 5.00 (expected 5)** |

The Monte Carlo test first failed at NEES 43.9. The cause was the test, not the filter: a 29° initial yaw prior and a ZUPT
σ on exactly-stationary truth both broke the error-state premises. Two genuine filter
defects were found on real data and fixed: pinning unmeasured gyro-bias states at 1e-12
made P numerically indefinite (S3c), and there was no recovery from GNSS gate lock-out.

### 4.4 Process noise — measured, not tuned (`reports/phase4/imu_noise.json`)

| | At rest (Vw1, Allan) | Moving (vibration) | Used by the EKF |
| --- | --- | --- | --- |
| Accelerometer white noise | 0.021–0.069 m/s²·√s | **0.33** m/s²·√s | moving |
| Vertical gyro white noise | 0.018 rad/s·√s | **0.021** rad/s·√s | moving |
| Accel bias instability | **0.024** m/s² (τ ≈ 134 s) | — | at rest |
| Gyro bias instability | **1.8e-3** rad/s (τ ≈ 107 s) | — | at rest |

Caveat: only **one** qualifying at-rest window exists (Vw1, 34 min).

### 4.5 Real-data baseline (phone-only filter, VBOX used only to score)

Phone IMU + phone GNSS at fix epochs; phone-only stillness → ZUPT + levelling; lock-out
recovery; 8-hypothesis coarse heading alignment where the mount is unobservable; GNSS
withheld in 30 s / 60 s outages every 4 min; every continuous segment ≥ 5 min scored.
The held-out TEST drivers were never touched.

| Set | Sessions scored | Error with GNSS (median of session medians) | End of 30 s outage | End of 60 s outage |
| --- | --- | --- | --- | --- |
| Validation ∩ verified (`baseline_evaluation.json`) | 3/5 | **9.8 m** | **96 m** (14 outages) | **1,425 m** (12) |
| Train+val ∩ verified (`…_all_verified.json`) | 12/23 | **7.8 m** | **291 m** (75) | **995 m** (70) |
| — of which 1 Hz-fix sessions (Vta1a, Vta2) | 2 | 2.8 m | 235 m | 1,087 m |
| — of which 9 s-fix sessions | 10 | 9.1 m | 344 m | 943 m |

(Train sessions are legitimate here: nothing in this classical filter is fitted to them.)

> **Corrected twice since this was written:** an EKF tilt fix (§6.6) and a GNSS leak in
> this script's stillness detector (§7.3). Current figures: validation 9.9 m / 100 m /
> **1,989 m**; train+val 7.8 m / 290 m / **1,096 m** (with GNSS / 30 s / 60 s). The 60 s
> figures above were optimistic.
The filter's own 2σ covers its error in a median 88 % of rows (range 56–99 %). 30 s outage
errors split evenly between along-track (170 m) and cross-track (193 m).

**Verdict, stated plainly:** with GNSS the phone filter tracks to about 3–9 m. **Without GNSS
it is not usable as a dead-reckoning solution**: ~300 m after 30 s, ~1 km after 60 s. The
drift is roughly quadratic, the signature of acceleration errors: gravity leaking through
unseen pitch/roll changes (no horizontal gyro), the phone's 0.33 m/s²·√s vibration, and heading error from an
unobservable, non-rigid mount. This is the number that the learned speed (Phase 5+) and NHC must
beat, and the reason they exist.

### 4.6 Exit criteria

- [x] SensorSample abstraction with frames, validation, epoch/receipt time
- [x] Phone-to-vehicle alignment (gravity + GNSS motion), validated against a reference
- [x] Horizontal gyro axes resolved empirically (negatively: not usable)
- [x] INS with gravity, Coriolis, transport rate; analytic physics tests
- [x] EKF with position/velocity/attitude/accel bias/gyro bias; predict + GNSS update;
      F and H checked by finite differences; Monte Carlo consistency
- [x] Real-data baseline measured on verified, non-test drives
- [x] ruff clean; 179 tests pass; gate 12/12

### 4.7 Carried forward

| Item | Affects |
| --- | --- |
| Reduced IMU → unobservable pitch/roll during outages; the dominant drift source | Phase 5+ (learned speed), NHC |
| Non-rigid mount: per-segment re-alignment needed before NHC | NHC phase |
| 9 s phone fix interval in 64/67 sessions: sparse aiding | fusion |
| Only 12/23 verified non-test sessions have a scoreable (≥ 5 min) segment | evaluation |
| Accelerometer scale factor / misalignment, magnetometer calibration, Allan plots | calibration (deferred) |
| One at-rest window for bias instability | noise model |

---

## Phase 5 — ML Training Pipeline

**Status:** `COMPLETED` for its defined scope (pipeline + smoke test). **Full training has
NOT been run**: the numbers below prove the pipeline works, not model accuracy.
**Date:** 2026-09-27
**Evidence:** `reports/phase5/train_smoke.json`, `reports/phase5/train_limited.json`; 199 tests pass; `ruff check .` clean
**Scope note:** redefined by the user as "ML Training Pipeline", replacing the original
"Attitude Estimation (AHRS)". Attitude is currently handled by the Phase 4 EKF plus gravity
levelling (reduced IMU).

**Run:**

```
.venv/Scripts/python.exe scripts/train/train_all.py --smoke                   # 1 epoch, 20 batches
.venv/Scripts/python.exe scripts/train/train_all.py --epochs 2 --max-batches 150   # "limited"
.venv/Scripts/python.exe scripts/train/train_all.py                           # full, per configs/training.yaml
```

### 5.1 What was built

| Component | Content |
| --- | --- |
| `configs/training.yaml` + `training/configs/loader.py` | All hyperparameters; every key required, a missing key raises |
| `training/datasets/windows.py` | 5 s windows (50 × 10 Hz); train stride 1 s, eval stride 5 s (non-overlapping, ml_pipeline §4); windows never span a gap or segment; a target-quality gate counts every exclusion; TEST split refused; inputs pass the CAN-rule guard, targets the no-coordinate guard |
| Features | device-frame acc, gravity, gyro `[0, 0, ω_z]` (the only real rate axis, §4.2) **plus mount-invariant** \|f\|, f·ĝ, \|f×ĝ\|, ω·ĝ. Fixed physical scaling (not data-fitted, so no leakage). The Phase 3 per-axis statistics are deliberately not used: rotation augmentation makes per-device-axis statistics meaningless |
| Augmentation (train only) | random device rotation (yaw about gravity + ≤ 15° tilt, or full SO(3)) for the non-rigid mount; accel/gyro noise; per-window accel bias; timestamp jitter. Applied *before* invariant features are computed |
| `training/models/registry.py` | `tcn` (causal, dilated), `cnn1d`, `gru`, `lstm`; heteroscedastic head (mean, log σ²); softplus mean for speed |
| `training/losses/regression.py` | Huber(mean) + 0.5 · Gaussian NLL (ml_pipeline §6); calibration metrics (1σ / 2σ coverage) |
| `training/trainers/trainer.py` | seeded; device auto → CUDA if available else CPU; **CUDA OOM → halve batch down to 16 → CPU** (logged); gradient clipping; early stopping; best-epoch checkpoint with commit, seed, split-manifest hash, inputs, output unit |
| `scripts/train/train_all.py` | Models A and B; reports; model cards `models/metadata/<model>.json` (output name, unit, range); prints the constant-mean reference every model must beat |

### 5.2 The two models

| | Model A — forward speed | Model B — longitudinal acceleration |
| --- | --- | --- |
| Output | speed [m/s] ≥ 0, + log σ² | a_long [m/s²], + log σ² |
| Target | `gt_speed_mps` (VBOX) | d(`gt_speed_mps`)/dt, central difference over ±0.5 s on real timestamps, within one segment |
| Why | non-integrating speed: the measurement that bounds outage drift | a learned, mount-independent replacement for the raw accelerometer's along-track component, which drives the Phase 4 outage drift |
| Architecture | TCN, 4 levels, 48 ch — **51,410 params** | GRU, 48 units — **16,130 params** |
| Sessions | train 56 / val 14 (all) | train 18 / val 5 (**alignment-verified only**: the acceleration target is timing-sensitive) |
| Windows | train 72,206 / val 2,956 | train 31,130 / val 1,355 |
| Excluded (train) | 907 unmatched ground truth, 465 low VBOX sats, 25 gap/segment | 358, 22, 15, 7 target not finite |

### 5.3 Smoke test and a short sanity run (CPU; the installed torch is the CPU build)

| Run | Model | Val MAE | Constant-mean reference | 1σ / 2σ coverage | Time |
| --- | --- | --- | --- | --- | --- |
| smoke (1 epoch × 20 batches) | A | 7.57 m/s | 7.02 m/s | 1.00 / 1.00 (σ still ≈ 20 m/s) | 3.3 s |
| smoke | B | 0.484 m/s² | 0.479 m/s² | 0.75 / 0.93 | 2.0 s |
| limited (2 epochs × 150 batches) | A | **5.92 m/s** | 7.02 m/s | 0.71 / 0.92 | ~40 s |
| limited | B | 0.467 m/s² | 0.479 m/s² | 0.53 / 0.80 | ~20 s |

Reading this honestly: the pipeline runs end to end with no fallback triggered, and Model A
**learns** (16 % below the constant reference after 300 batches, with σ already near
calibrated). Model B barely beats its reference so far. Nothing here is a trained-model
result. Two identical runs of the limited configuration gave bit-identical metrics
(val MAE 5.920392…), so training is deterministic.

### 5.4 Tests (`tests/ml/test_phase5_pipeline.py`, 20)

Every architecture builds, stays under 1 M params, and returns the right shapes; TCN causality
(future samples cannot change past outputs); the NLL formula and its optimum at the true
variance; loss finite and differentiable; coverage metric on a calibrated Gaussian;
**static check that no config target, model output or training target is a coordinate**;
a missing config key raises; OOM policy (halve → CPU); invariant features unchanged under
random rotation; TEST split refused; windows never cross gaps or segments and inputs hold no
`gt_` column; augmentation changes inputs but never targets.

### 5.5 Not done / carried forward

| Item | Note |
| --- | --- |
| **Full training** of A and B, hyperparameter sweep (window 1–5 s, arch) | next step; CPU is workable at this model size |
| **CUDA build not installed**: GPU path and OOM fallback untested on the RTX 2050 | `pip install -r requirements-gpu.txt`, then confirm `torch.cuda.is_available()` (driver 555.97 vs cu126 untested) |
| Variance calibration by bins (ml_pipeline §6) | only 1σ/2σ coverage so far |
| Model B on 18 train sessions only (verified alignment) | per-session alignment refinement would widen it |
| Motion-context classifier (ZUPT trigger) | not in this phase's scope |
| ONNX export + parity check | export phase |

---

## Phase 6 — ML Evaluation & Export

**Status:** `COMPLETED`
**Date:** 2026-09-27
**Evidence:** `reports/phase6/sweep.json`, `selection.json`, `evaluation.json`, `onnx_validation.json`;
`models/exported/*.onnx` + `*.model_card.json`; `ruff check .` clean; 222 tests pass
**Scope note:** redefined by the user as "ML Evaluation & Export", replacing the original
"Strapdown INS Mechanization", which Phase 4 already delivered (`navcore.ins`).

**Run:**

```
.venv/Scripts/python.exe scripts/train/sweep.py               # 16 trainings, GPU, ~37 min
.venv/Scripts/python.exe scripts/train/sweep.py --reselect    # re-time latency, re-apply the rule
.venv/Scripts/python.exe scripts/evaluate/evaluate_models.py  # VAL, then TEST once
.venv/Scripts/python.exe scripts/export/export_models.py      # ONNX opset 17 + model cards
.venv/Scripts/python.exe scripts/validate/validate_models.py  # PyTorch vs ONNX gate; non-zero exit = blocker
```

### 6.1 GPU

`torch 2.14.0+cu126`; `torch.cuda.is_available()` → **True**, NVIDIA GeForce RTX 2050 (4 GB).
All 16 sweep runs trained on CUDA; the OOM fallback (halve batch → CPU) never triggered.

### 6.2 Sweep: 4 architectures × 2 windows × 2 models

12 epochs × 150 batches each, hidden width 48, same seed; scored on the **full** VAL split
(non-overlapping windows). Latency = batch-1, single-thread CPU (a phone proxy),
re-measured after training in 7 interleaved rounds (min of round medians).

| Candidate | Params | Model A val MAE (m/s) | Model B val MAE (m/s²) | CPU ms (A / B) |
| --- | --- | --- | --- | --- |
| TCN, 2 s | 51,410 | 5.18 | 0.4366 | 1.17 / 1.16 |
| 1D-CNN, 2 s | 26,690 | 4.12 | 0.4407 | 0.29 / 0.29 |
| GRU, 2 s | 16,130 | 4.09 | 0.4391 | 0.60 / 0.59 |
| LSTM, 2 s | 20,834 | 4.99 | 0.4403 | 0.27 / **0.27** |
| TCN, 5 s | 51,410 | 4.34 | 0.4376 | 1.31 / 1.25 |
| 1D-CNN, 5 s | 26,690 | **3.31** | 0.4424 | **0.32** / 0.32 |
| GRU, 5 s | 16,130 | 3.49 | **0.4337** | 1.22 / 1.24 |
| LSTM, 5 s | 20,834 | 5.11 | 0.4396 | 0.33 / 0.32 |
| *constant train mean* | — | *7.02* | *0.478* | — |

**Selection rule** (`configs/training.yaml` `sweep.selection`): lowest val MAE; candidates
within 3 % of it compete on CPU latency; latencies within 10 % of the fastest are a tie,
broken on parameter count, then val MAE.

- **Model A → `model_a_cnn1d_w50`**: the only candidate within 3 % (next best GRU-5 s is 5 % worse).
- **Model B → `model_b_lstm_w20`**: all 8 candidates lie within 3 % (0.434–0.442), i.e.
  architecture does not matter for this target; LSTM-2 s and 1D-CNN-2 s tie on latency and
  the LSTM has fewer parameters.

**Two corrections made during selection, both disclosed:**
1. The sweep's own latency numbers were taken while other programs loaded the CPU and moved
   by up to 3.5× between measurements (e.g. A-TCN-5 s 1.13 → 3.89 ms). A single re-measure
   flipped Model B's pick on noise. Latency is now measured interleaved over 7 rounds, where
   each model's rounds agree within ~15 %.
2. The 10 % latency tie rule was added **after** seeing that noise, not after seeing which
   model it picks. Without it Model B would be the LSTM anyway (fastest at 0.266 ms).

Most runs had their best epoch at 10–11 of 12: the models are **not trained to convergence**.
Longer training is the obvious next lever (`scripts/train/train_all.py` full mode).

### 6.3 Evaluation (`reports/phase6/evaluation.json`)

Selection used VAL only. TEST (held-out drivers B and D) was evaluated **once**, after selection.

| | Model A val | Model A **test** | Model B val | Model B **test** |
| --- | --- | --- | --- | --- |
| Windows | 2,956 | 3,484 | 3,405 | 8,718 |
| MAE | 3.31 m/s | **3.91 m/s** | 0.440 m/s² | **0.455 m/s²** |
| RMSE | 4.54 | 5.14 | 0.663 | 0.728 |
| Bias | −0.00 | **+2.81** | −0.01 | +0.01 |
| Abs. error p50 / p90 / p95 / p99 | 2.44 / 7.42 / 9.58 / 14.5 | 3.17 / 8.74 / 10.4 / 13.4 | 0.29 / 1.04 / 1.29 / 2.23 | 0.27 / 1.12 / 1.47 / 2.52 |
| Constant-mean reference | 7.02 | 5.47 | 0.478 | 0.474 |
| Coverage 1σ / 2σ / 3σ (ideal .68/.95/.997) | .68 / .93 / .98 | .57 / .90 / .98 | .72 / .95 / .99 | .68 / .88 / .94 |
| z std (ideal 1) / ENCE | 1.16 / 0.21 | 1.14 / 0.31 | 1.12 / 0.18 | **1.52** / 0.52 |

Read plainly:
- **Model A generalises to new drivers, but with a bias.** Test MAE is 29 % below the constant
  reference (val: 53 %). On test it over-reads speed by +2.8 m/s: +3.8 m/s at 2–10 m/s,
  +3.0 m/s at 10–20 m/s. On val it reads high at low speed (+2.9 m/s at 2–5 m/s) and low at
  high speed (−12.3 m/s above 30 m/s, 52 windows): the classic shrink-toward-the-mean of an
  under-trained regressor. Its variance head stays near-calibrated (z std 1.14).
- **Model B is weak.** It beats "predict the mean" by 8 % on val and **4 % on test**. Per
  target bin, its bias equals its MAE in every tail (e.g. −1.97 m/s² when the truth is above
  1.5 m/s²), i.e. it mostly predicts ≈ 0 and **under-reports every real acceleration**. On
  test it is also over-confident (z std 1.52, 2σ coverage 0.88). Phase 7 uses it only through
  its own variance (§7) and measures whether it helps.

### 6.4 ONNX export and the parity gate (`reports/phase6/onnx_validation.json`)

Opset 17, TorchScript exporter, input `window` [batch, 13, W], outputs `mean`, `logvar`,
dynamic batch axis; `onnx.checker` passes. Each model card records the SHA-256, the ordered
feature list, fixed scaling, output unit and range, and the training commit.

| Model | ONNX | Worst \|torch − onnx\| | Gate usage (≤ 1 passes) | SHA-256 = card |
| --- | --- | --- | --- | --- |
| `model_a_cnn1d_w50` | 107 kB | 4.6e-05 (stress set); 8.6e-06 on real windows | 0.47 | ✓ |
| `model_b_lstm_w20` | 85 kB | 2.9e-06 | 0.19 | ✓ |

**The gate was changed from a flat 1e-5, and why (both changes agreed with the project owner):**
- At first Model A **failed** the flat 1e-5: 1.1e-5 on random batches, 4.6e-5 on the 10×
  stress set. Those inputs drive it to 59–283 m/s, where one float32 step is already
  ≈ 1.5e-5. PyTorch float32 itself differs from exact float64 by 3.7e-5 there, and turning
  off every onnxruntime graph optimisation changes nothing. It is float32 rounding, not an
  export fault.
- **Real and random inputs:** every element must satisfy \|torch − onnx\| ≤ 1e-5 + 1e-6·\|torch\|.
- **10× stress set:** its hidden activations reach ~300, giving logvar up to 33, and a
  logvar gap of 1.24e-5 at small \|logvar\| is not covered by the relative term. There ONNX
  must be as accurate as float32 PyTorch against a float64 PyTorch reference, per output:
  max\|onnx − fp64\| ≤ 2·max\|torch32 − fp64\| + 1e-5. Model A: logvar 1.25e-5 vs torch32 8.3e-6.
- Both models pass with margin; the raw worst absolute gap is still reported.

### 6.5 Tests added

`tests/ml/test_onnx_export.py` (8): every architecture exports and matches onnxruntime,
with and without the positive head. `tests/navigation/test_nhc_state_fusion.py` includes the
runtime tamper check: a byte appended to the `.onnx` file is refused by its card's SHA-256.

### 6.6 Phase 4 baseline, re-run after an EKF fix found in Phase 7

Phase 7 found that the EKF fed **zero** for the unmeasured horizontal gyro axes, while a level
vehicle actually sees the Earth-rate and transport-rate components there. At rest, that tilts
the attitude at Ω cos φ (0.26 rad after 1 h). Fixed in `ErrorStateEKF.predict`: unmeasured
axes now take R_nbᵀ(ω_ie + ω_en). Regression test:
`test_reduced_imu_stays_level_on_the_rotating_earth`. Phase 4 re-run
(`reports/phase4/baseline_evaluation*.json`):

| Set | With GNSS | End of 30 s outage | End of 60 s outage |
| --- | --- | --- | --- |
| Train+val ∩ verified (was) | 7.8 m | 291 m | 995 m |
| Train+val ∩ verified (**now**) | 7.8 m | **290 m** | **1,009 m** |
| Validation ∩ verified (was) | 9.8 m | 96 m | 1,425 m |
| Validation ∩ verified (**now**) | 9.9 m | **100 m** | **1,226 m** |

The real-data effect is small. Over 30–60 s the Earth-rate tilt error is too small to matter;
my earlier estimate of a ~50 m improvement was wrong. The fix matters for long runs and for
the synthetic tests (INS-only dropout drift 369 → 293 m).

### 6.7 Exit criteria

- [x] CUDA torch installed; `cuda.is_available()` True
- [x] Sweep over TCN / 1D-CNN / GRU / LSTM × 2 s / 5 s; selection by a stated rule on VAL
- [x] MAE, RMSE, error percentiles, error by target range, 1/2/3σ coverage, ENCE, reliability table
- [x] TEST evaluated once, after selection
- [x] ONNX export with model cards; PyTorch-vs-ONNX gate passes; SHA-256 verified at load
- [x] Parameter count, CPU latency and size recorded per model

### 6.8 Carried forward

| Item | Note |
| --- | --- |
| Models under-trained (best epoch 10–11 of 12) | full-length training run |
| Model A +2.8 m/s bias on test drivers; shrinks toward the mean at high speed | more epochs; per-driver analysis |
| Model B ≈ constant; over-confident on test | reconsider the target (timing-sensitive d/dt on 18 sessions) |
| On-device (Android) ONNX parity | Phase 12, needs the Android toolchain |

---

## Phase 7 — Sensor Fusion & NHC

**Status:** `COMPLETED` for its defined scope (fusion code, NHC, state machine, integration
tests, real-data ablation). **The real-data benefit is mixed and is reported as measured (§7.5).**
**Date:** 2026-09-27
**Evidence:** `reports/phase7/fusion_evaluation.json`; `tests/navigation/test_nhc_state_fusion.py` (12),
`tests/integration/test_fused_navigation.py` (3); `ruff check .` clean; full suite passes
**Scope note:** redefined by the user as "Sensor Fusion & NHC". It delivers much of the
original Phase 8 (fusion with AI measurements) and Phase 9 (NHC) plans; those phases keep
the parts not done here (§7.7).

**Run:**

```
.venv/Scripts/python.exe scripts/evaluate/phase7_fusion_eval.py          # real-data ablation (VAL, verified)
.venv/Scripts/python.exe -m pytest tests/integration tests/navigation/test_nhc_state_fusion.py
```

### 7.1 What was built

| Component | Content |
| --- | --- |
| `navcore.fusion.features` | the single feature implementation (13 channels), used by training AND inference, so they cannot drift apart |
| `navcore.fusion.ai_models` | `OnnxSequenceModel`: loads an exported model, **refuses** it if its SHA-256 or feature order differs from the model card; `ImuWindow` (restarts on a data gap); `speed_innovation` (H = [0, v_h/\|v_h\|, …]); `correct_specific_force` (Model B, below) |
| `navcore.nhc.constraints` | vehicle-frame lateral / vertical velocity ≈ 0 as pseudo-measurements; exact Jacobian in velocity **and** attitude error; skipped below 2 m/s and above 3 m/s² lateral acceleration; lateral row only with an **accepted** mount |
| `navcore.state.gnss_state` | GOOD / DEGRADED / LOST / RECOVERING, from fix age (12 s / 25 s), accuracy (> 10 m) and gate outcome; per-state policy: GNSS noise ×1 / ×2 / off / ×3, AI speed off in GOOD only |
| `navcore.fusion.navigator` | `FusedNavigator.step()`: axis map → window → models → **Model B correction → predict()** → state machine + GNSS (+ lock-out reset) → **Model A speed update** (not GOOD) → NHC → stillness ZUPT. Every component is a switch; everything is counted |
| `ErrorStateEKF.update()` | public generic update (Joseph form, NIS gate, PD check) used by NHC and AI speed |

**How Model B is applied.** It predicts longitudinal acceleration, not a 3-axis correction.
Before `predict()`, the specific force's along-track component (along the current horizontal
velocity) is replaced by the inverse-variance blend of the IMU's value and Model B's, using
Model B's own variance. The cross-track and vertical components are untouched. Below 3 m/s
the direction is undefined and nothing is changed.

### 7.2 A bug found by the real data: AI speed counted as independent evidence

The first real-data run applied Model A's speed at **every 10 Hz sample**, even with no new
prediction, using its per-prediction variance. Consecutive 5 s windows share 49 of 50 samples,
so their errors are nearly identical, and the filter treated a biased speed as ~10× more
certain than it is. On S3a: 18,921 AI updates, none rejected; the filter grew over-confident
and the GNSS gate then **rejected 146 of 204 real fixes** (baseline: 32), holding the state
machine in DEGRADED 62 % of the time; aided error went from **7.2 m to 41.9 m**.

**Fix** (derived from the model design, not tuned): a speed update only on a **fresh**
prediction, at most once per second, with variance × (window length / interval) = ×5 for the
5 s model, so one window's worth of overlapping predictions counts once. After the fix, S3a
aided error is 7.3 m and GNSS rejections are 27. Unit test:
`test_ai_speed_is_applied_once_per_interval_with_inflated_variance`.

### 7.3 A leak found in the Phase 4 baseline (corrected)

Phase 7's baseline arm reproduced Phase 4 exactly on S3a and S3b but not on S3c (aided 101 m
vs 82 m). Cause: in `scripts/evaluate/phase4_baseline.py`, **a fix withheld during a simulated
outage still updated the stillness detector's speed hint**, so the filter used withheld GNSS
speed to recognise stops during outages. Confirmed: re-inserting the leak into Phase 7
reproduces Phase 4 on S3c (77 m / 484 m / 1,448 m vs 82 / 493 / 1,466). Fixed; Phase 4 re-run:

| Phase 4 baseline | With GNSS | End of 30 s outage | End of 60 s outage |
| --- | --- | --- | --- |
| Validation ∩ verified, as committed | 9.9 m | 100 m | 1,226 m |
| Validation ∩ verified, **leak fixed** | 9.9 m | 100 m | **1,989 m** |
| Train+val ∩ verified, as committed | 7.8 m | 290 m | 1,009 m |
| Train+val ∩ verified, **leak fixed** | 7.8 m | 290 m | **1,096 m** |

The earlier 60 s figures were optimistic; the phone-only filter is worse without GNSS than
reported. Phase 4 gate re-run: 12/12 PASS.

### 7.4 Synthetic integration test (`tests/integration/test_fused_navigation.py`)

Clearly synthetic: a 260 s drive with acceleration, braking and turns, IO-VNBD column layout,
IMU biases and noise, GNSS at 1 Hz with a **100 s dropout**. Model outputs come from
noisy-oracle stubs (truth + N(0, σ)). The trained models are meaningless on synthetic motion,
so real-model fusion is judged on real data (§7.5). Horizontal error at the end of the
dropout, over **8 seeds with identical noise in every arm**:

| Arm | Median | Range |
| --- | --- | --- |
| INS only | 475 m | 125–889 m |
| + NHC | 197 m | 47–373 m |
| + AI speed | 314 m | 88–730 m |
| + AI speed + NHC | 49 m | 29–218 m |
| **Full** (+ Model B) | **35 m** | 12–216 m |

Speed bounds along-track error and NHC bounds heading / cross-track error; each alone does
little. The test asserts the 8-seed median (< 25 % of INS-only, < 60 m), not one seed: per
seed the full/INS ratio ranges 0.02–0.48, and the old single-seed test passed on a lucky seed.
Other integration tests: the state machine walks DEGRADED → LOST → RECOVERING → GOOD across
the dropout, with LOST 12–26 s after the last fix; P stays positive-definite throughout.

*Correction:* an earlier single-seed run reported NHC alone as harmful (373 m vs 293 m). That
predated the §6.6 tilt fix and was one seed; over 8 seeds NHC alone helps.

### 7.5 Real data (`reports/phase7/fusion_evaluation.json`)

Validation split ∩ clock-verified: 3 scoreable drives (S3a, S3b, S3c; driver A, 9 s phone
fixes, mount **not** observable in any, so lateral NHC is off and only vertical NHC runs).
Same initialisation, multi-hypothesis heading, outage schedule and scoring as Phase 4. Phone
inputs only; VBOX scores. **Caveat:** Phase 6 selected the models on this split, so these lean
optimistic; TEST is reserved.

Pooled over all outages (14 × 30 s, 12 × 60 s):

| Arm | With GNSS p50 | 30 s p50 / p90 | 30 s along / cross | 60 s p50 / p90 | Filter 2σ covers error |
| --- | --- | --- | --- | --- | --- |
| baseline | 34.2 m | 274 / 1,123 m | 204 / 163 m | 2,814 / 5,110 m | 94 % |
| state machine only | 31.8 m | 297 / 1,208 m | 228 / 184 m | 2,860 / 4,691 m | 94 % |
| NHC | 12.9 m | 282 / 2,170 m (n 6) | 245 / 167 m | 1,579 / 11,997 m (n 5) | 68 % |
| **AI speed** | **11.9 m** | 300 / **520 m** | **117** / 146 m | **490 / 746 m** | 79 % |
| AI speed + NHC | 12.5 m | 294 / **380 m** | 117 / 204 m | **364** / 1,037 m | 75 % |
| full (+ Model B) | 37.9 m | 321 / 1,173 m | 76 / 238 m | 633 / 5,082 m | 63 % |

Per drive (with GNSS / 30 s / 60 s, median per drive):

| Drive | baseline | AI speed | AI speed + NHC | full |
| --- | --- | --- | --- | --- |
| S3a | 7.2 / 100 / 960 | 7.3 / 317 / 388 | 9.6 / 321 / 455 | 12.6 / 273 / 550 |
| S3b | 9.9 / 64 / — | 11.3 / 186 / — | 11.5 / 184 / — | 10.7 / 136 / — |
| S3c | 101 / 916 / 2,992 | 17.4 / 300 / 505 | 16.3 / 279 / 183 | 81.5 / 404 / 742 |

What this says, plainly:
- **Model A speed is the one component that clearly helps on real data.** 60 s outage error
  falls 5.7× (2,814 → 490 m), and the 30 s tail halves (p90 1,123 → 520 m). Along-track error
  at 30 s falls from 204 to 117 m. On the hardest drive (S3c) it also fixes the aided solution
  (101 → 17 m).
- **It does not help every outage.** On the two easier drives the median 30 s error roughly
  triples (S3a 100 → 317 m, S3b 64 → 186 m). The pooled 30 s median is unchanged (274 vs 300 m).
  The cause is **not found**. A stillness hypothesis was tested and rejected: with AI there are
  more ZUPTs (2,115 vs 1,889), not fewer. Model A's +1–3 m/s bias at low speed (§6.3) is the
  next suspect.
- **NHC alone hurts on real data** and **diverged on S3c** (`P not positive-definite after
  reset_to_fix`). Only vertical NHC runs here, and it needs the phone-to-vehicle tilt to well
  under 1°: 1° of tilt error leaks 0.17 m/s² (≈ 77 m in 30 s) into along-track acceleration.
  The tilt is one drive-wide estimate, and the phone's 60 s tilt differs from it by a median
  0.8–1.3° (up to 6.3° on S3c; this includes road gradient, so it is an upper bound). NHC is also
  applied at 10 Hz as if independent, and real NHC violations (sideslip, mount wobble) are
  time-correlated. The filter becomes over-confident (2σ coverage 94 % → 68 %), and the
  near-singular covariance is what failed at the reset. Combined with AI speed it is
  **neutral-to-helpful** (best 30 s tail, p90 380 m; best 60 s median, 364 m).
- **Model B makes things worse** (full vs AI speed + NHC): aided 12.5 → 37.9 m, 60 s 364 →
  633 m, 2σ coverage 63 %. This matches Phase 6: Model B barely beats a constant and is
  over-confident on new data, so the blend trusts it more than it deserves.
- **The GNSS state machine is neutral** on its own (vs baseline), as it should be.

**Shipped defaults** (`FusionConfig()`, agreed with the project owner after this ablation):
AI speed **ON**; NHC **ON only together with AI speed** (`nhc_requires_ai_speed`); Model B
**OFF**. **Model B was fully trained, selected, exported, parity-validated and evaluated, and
is disabled by default because empirical testing showed it degraded real-world performance**
(the `full` vs `ai_speed_nhc` rows above). It remains available behind `use_ai_accel=True`.
The report's `default` arm runs `FusionConfig()` as shipped. See `docs/architecture.md`,
"As built".

**Final re-run (with the defaults in place):** every arm reproduced its earlier numbers to
the last digit (the evaluation is deterministic), and the `default` arm equals
`ai_speed_nhc` exactly: aided p50 12.5 m, 30 s outage p50 294 m (p90 380 m), 60 s outage p50
364 m (pooled; n = 14 / 12).

### 7.6 Exit criteria

- [x] ONNX inference wrapper in `navigation-core/fusion/`, with integrity checks against the model card
- [x] Model B applied to the IMU before `predict()`; Model A speed as a measurement when GNSS is DEGRADED / LOST / RECOVERING
- [x] NHC in `navigation-core/nhc/` (lateral + vertical), exact Jacobian verified by finite differences
- [x] GNSS state machine GOOD / DEGRADED / LOST / RECOVERING in `navigation-core/state/`, driving the filter policy
- [x] Integration tests: raw IMU + AI outputs + simulated GNSS dropouts through the fused filter; all pass
- [x] Real-data ablation per component, including where components hurt

### 7.7 Carried forward

| Item | Where |
| --- | --- |
| NHC: correlated-error model (rate / σ) and a time-varying tilt estimate; covariance robustness at reset | Phase 9 (NHC) |
| Why AI speed worsens some 30 s outages; low-speed bias of Model A | Phase 8 / model retraining |
| Model B: retrain or drop; its variance is not trustworthy on new drivers | model retraining |
| Evaluation on only 3 drives / 14 + 12 outages — too few for fine conclusions; TEST drivers still reserved | Phase 11 |
| Lateral NHC is untested on real data (no validation drive has an observable mount) | Phase 9 |

---

## Phase 8 — Offline Map Matching

**Status:** `COMPLETED` for its defined scope. **Measured real-drive benefit: marginal
overall; a real gain only while dead reckoning is within ~30 m of the truth (§8.4).**
**Date:** 2026-09-27
**Evidence:** `reports/phase8/map_matching_evaluation.json`, `reports/phase8/osm_manifest.json`;
`tests/navigation/test_map_matching.py` (15); `ruff check .` clean
**TODO.md title for #8:** "Fusion Engine (Error-State EKF)", delivered in Phases 4 and 7 (crosswalk above).
**Scope note:** redefined by the user as "Offline Map Matching". It delivers the original
Phase 10 plan (`docs/map_matching.md`, which now opens with an "as built" section). The
original Phase 8 (fusion EKF) and Phase 9 (NHC) plans were delivered in Phases 4 and 7.

**Run:**

```
.venv/Scripts/python.exe scripts/download/fetch_osm_roads.py            # ONCE: offline road cache
.venv/Scripts/python.exe scripts/evaluate/phase8_map_matching_eval.py  # record, tune on TRAIN, report VAL
```

### 8.1 What was built (`navigation-core/map_matching/`)

| Component | Content |
| --- | --- |
| `road_network.RoadNetwork` | OSM XML → drivable ways → **directed** straight segments in the local tangent plane (metres, via `LocalTangentPlane`); one-way / `oneway=-1` / roundabout / motorway rules; 50 m grid index; bounded Dijkstra route distance (U-turns only at nodes); `.npz` offline cache |
| `hmm.HmmMapMatcher` | online HMM: emission = Mahalanobis with the **filter's 2×2 covariance** (+ 5 m map error) + heading agreement with the directed segment; transition = −\|route − travelled\|/β, **probability 0 when no route exists** (no teleports); forward-filter output + Viterbi `decode()`; `map_match_confidence` = posterior mass within 10 m of the chosen point; off-network suspension (χ² gate, 3 epochs) and re-entry; deterministic ordering |
| `outage.DeadReckoningMapMatcher` | takes the fused INS/AI/NHC trajectory (**reads the EKF, never writes it**), runs at 1 Hz, accumulates the path travelled between epochs, returns a road-constrained lat/lon + `map_match_confidence`, or `matched=False` with no position |
| Offline data | `scripts/download/fetch_osm_roads.py`: one Overpass query (Coventry core, drivable classes), 30.6 MB, 231,943 nodes → 450,775 directed segments, 46,301 ways; parsed in 2.2 s, cache reload 1.1 s; query 0.05 ms, 300 m route search 0.07 ms. Map data © OpenStreetMap contributors, ODbL. **No network code in the package** (static test) |

### 8.2 Tests (15)

OSM parsing and one-way rules; spatial index = brute force; route distance respects
topology, one-way streets and the cutoff; **crossroads**: dead reckoning 12 m off the road
makes nearest-road snapping jump onto the crossing road, the HMM stays on the through road
(and a mutation check shows it jumps too once transitions and heading are disabled); a real
**turn** at the junction is followed, with a Viterbi path that has no impossible
transition; **parallel roads** never flip; **dual carriageway**: direction of travel picks
the carriageway although the other is nearer; **off-network** suspension and re-entry with
no invented position; search radius follows the covariance; confidence ≈ 0.5 when two
roads are equally likely; matched point lies on its segment; cache round-trip identical;
the matcher leaves the EKF untouched; course σ matches Monte Carlo.

### 8.3 Real-drive protocol

The Phase 7 fused navigator with the shipped defaults (AI speed + NHC, Model B off) and
the Phase 7 outage schedule (30 s / 60 s every 4 min), recorded once per drive. β and a
covariance inflation were chosen on **TRAIN drives only** (S1, S2, S4; grid β ∈ {5, 10,
20, 40} m × inflation ∈ {1, 4, 9}). The criterion was the median output error in outages.
Every setting landed within 125.4–127.9 m vs 128.9 m fused; selected β = 5 m, ×9. Reported
on **VAL** (S3a, S3b, S3c). Output policy: the matched position if matched and confidence
≥ 0.5, else the fused one. "Correct road" = within 20 m of the VBOX track's own match
(evaluation only).

### 8.4 Results (VAL, `reports/phase8/map_matching_evaluation.json`)

| | Fused p50 / p90 | Map-matched p50 / p90 | Cross-track p50 | Along-track p50 | Correct road |
| --- | --- | --- | --- | --- | --- |
| All outage epochs (n 1,154) | 83.9 / 374 m | **80.7** / 366 m | 47.7 → **39.2** m | 47.6 → 47.3 m | 20 % |
| End of 30 s outage (n 14) | 287 / 374 m | 270 / 368 m | 205 → 203 m | 96 → 97 m | **0 %** |
| End of 60 s outage (n 12) | 348 / 1,032 m | 344 / 1,012 m | 160 → 152 m | 201 → 191 m | **0 %** |
| With GNSS (n 4,906) | 12.6 / 224 m | 12.3 / 220 m | 6.9 → **4.9** m | 7.9 → 9.9 m | 68 % |

**Where it helps — outage epochs by how far dead reckoning has already drifted:**

| Fused error | n | Fused → matched p50 | Cross-track | Correct road |
| --- | --- | --- | --- | --- |
| < 15 m | 158 | 8.3 → 8.6 m | 4.2 → 4.5 m | 82 % |
| **15–30 m** | 84 | **23.5 → 17.6 m** | **11.4 → 5.4 m** | 63 % |
| 30–60 m | 191 | 42.5 → 43.9 m | 29.7 → 22.0 m | 9 % |
| 60–120 m | 251 | 81.7 → 79.6 m | 62.0 → 50.5 m | 0.5 % |
| > 120 m | 470 | 255 → 239 m | 167 → 150 m | 0 % |

Confidence vs correctness: conf ≥ 0.95 → 69 % correct, 0.8–0.95 → 62 %, 0.5–0.8 → 35 %.

Read plainly:
- **Map matching cannot rescue a drifted trajectory in a city.** Once dead reckoning is
  more than ~30 m off, the candidate set holds many roads that fit its shape, and the matcher
  picks a wrong one almost every time (0 % correct at the end of 30 s and 60 s outages). The
  numbers improve slightly only because the wrong road is often a little nearer.
- **It does help while drift is small** (15–30 m: cross-track error halves, 11.4 → 5.4 m,
  63 % on the correct road). That is the first seconds of an outage, and short outages.
  With GNSS, it cuts cross-track error by 29 % but adds 2 m along-track (snapping moves the
  point along the road).
- **Confidence is informative but over-stated**: it ranks correctness correctly, but
  "≥ 0.95" is right only 69 % of the time. A consumer should gate on both confidence and
  the fused σ.
- It is a refinement, as designed. It never feeds back into the filter, so it cannot make
  the navigation estimate worse; only the displayed point.
- Caveat: 3 VAL drives, all driver A, urban Coventry.

### 8.5 Exit criteria

- [x] Map-matching engine in `navigation-core/map_matching/`
- [x] Offline OSM road network load and query; no cloud access at run time (static test)
- [x] Probabilistic (HMM) matcher using distance, road heading and route continuity; no nearest-road snapping
- [x] Integration: the dead-reckoned trajectory in, a constrained position + `map_match_confidence` out
- [x] Unit tests on synthetic intersections, turns, parallel roads, dual carriageways, off-network
- [x] Real-drive evaluation, including where it does not help

### 8.6 Carried forward

| Item | Note |
| --- | --- |
| Gate the map-matched output on the fused σ (e.g. use it only while σ < ~30 m) | the band table says where it helps |
| Confidence calibration (69 % at "≥ 0.95") | recalibrate on TRAIN |
| Feed matched heading back into the filter | deliberately not done (`docs/map_matching.md` §5) |
| On-device graph size / memory for the operating area | Android phase |

---

## Phase 9 — GNSS Outage Detection, Recovery & Replay

**Status:** `COMPLETED` for code and tests: outage simulator, replay engine, streaming
engine, recovery manager, output smoother. **The recovery manager is OFF by default**
(`FusionConfig.recovery = None`). Two real-drive benchmark runs showed earlier designs of
it failing badly (§9.4). The third run, of the corrected design, completed after the
Phase 9 commit (§9.4a): no divergence, but **no clear benefit either**, so it stays OFF.
The same run shows the output smoother **triples the typical error** of the output it
smooths: continuity costs accuracy on this data.
**Date:** 2026-09-28
**Evidence:** `tests/simulation/` (14), `tests/navigation/test_recovery_manager.py` (13),
`tests/navigation/test_recovery.py` (2, Phase 4, unchanged); **full suite 271 passed**, `ruff` clean;
`reports/phase9/outage_benchmark_first_design.json`, `..._second_design.json` (failed designs,
kept on purpose); `reports/phase9/outage_benchmark.json` (third run, when complete)
**TODO.md title for #9:** "Non-Holonomic Constraints (NHC)", delivered in Phase 7 (crosswalk above).
**Scope note:** redefined by the user as "GNSS Outage Detection, Recovery & Replay". The
original Phase 9 (NHC) was delivered in Phase 7.

**Run:**

```
.venv/Scripts/python.exe -m pytest tests/simulation tests/navigation/test_recovery_manager.py
.venv/Scripts/python.exe scripts/evaluate/phase9_outage_benchmark.py     # ~30-60 min on battery
```

### 9.1 What was built

| Component | Content |
| --- | --- |
| `simulation/outage/simulator.py` | `OutageSchedule` of `OutageEvent`s over RECORDED fixes: `full` (10/30/60/120/300 s …), `tunnel` (degraded approach, a confident 150 m multipath outlier at the exit, degraded exit ramp), `intermittent` (random on/off, degraded fixes), `degraded` (urban canyon); builders `periodic` (the Phase 4/7 protocol) and `benchmark` (the full battery with 150 s gaps). Seeded and deterministic; never mutates its input; every perturbation logged and labelled synthetic |
| `simulation/replay/replay.py` | `load_segments` (synced `.parquet` → phone arrays + fixes with epoch and receipt time; VBOX truth kept separate; CAN rule enforced on the input list); `ReplayEngine` streams sample by sample, delivering each fix once, at its receipt time, never early; optional real-time pacing with an injectable clock; `ReplayResult.digest` (SHA-256) proves bit-identical runs |
| `navcore/fusion/engine.py` | `NavigationEngine`: the streaming, causal engine: WAITING → ALIGNING (8 heading hypotheses, 120 s) → NAVIGATING; causal mount (tilt from the accelerometer before start-up, or the phone's gravity channel when a drive starts moving in its first second, found on real data), or `mount_mode="given"` for the Phase 4/7 whole-drive lookahead |
| `navcore/recovery/manager.py` | `RecoveryManager` (opt-in): after a LOSS (not the start-up LOST), holds fixes until two agree along the dead-reckoned path within a tolerance that includes the filter's own velocity and heading uncertainty; fails open after 30 s; fuses the first recovery fixes with graded noise (×9, ×4, ×2); inflates the POSITION covariance (≤ ×100) to admit a confirmed fix that fails the gate on position, and leaves P untouched otherwise. `OutputSmoother` (ON): the reported position = filter + an offset that absorbs every correction jump and glides back linearly at max(15 m/s, offset ÷ 10 s): never a teleport, never more than 10 s of lag |
| `navcore/fusion/navigator.py` | `output()` (smoothed position); optional `recovery`; unchanged filter behaviour when `recovery=None` |
| `simulation/synthetic/drive.py` | the synthetic drive generator, clearly labelled, shared by tests |
| `scripts/evaluate/phase9_outage_benchmark.py` | the battery replayed on every verified TRAIN+VAL drive: `phase7` vs `recovery` arms (+ `given_mount` on VAL); error at event end, jump size, error +10/+30 s, re-convergence time, wrong jumps |

**Replay engine verified against Phase 7:** replaying S3b through `NavigationEngine` with the
Phase 7 mount and schedule reproduces the Phase 7 default arm to every printed digit
(aided p50 **11.468584092819786 m**, 30 s outage **183.6279361708867 m**).

### 9.2 Tests (27 new)

Position continuity across LOST → RECOVERING → GOOD: the filter jumps by more than 100 m at
recovery; the output never moves by more than its glide bound, and it converges. This holds
with and without the manager. A confident 150 m multipath fix at recovery is flagged and
never fused (in the shipped configuration the NIS gate also rejects it — stated in the
test). A filter that honestly knows nothing about its velocity accepts it (it cannot tell).
Fixes are held until two agree. Fail-open timeout. The tolerance widens with the filter's
heading σ. Inflation admits a confirmed fix, never touches velocity, is capped, and is
refused (P unchanged) beyond the cap. Start-up LOST is not a recovery. Graded schedule.
Smoother absorbs a 300 m jump without moving, and a 1.5 km correction within 10 s.
Simulator: each dropout length removes exactly its window; tunnel approach, outlier and
exit; intermittent; degraded noise statistics; deterministic by seed; input never mutated;
battery layout; Phase 7 periodic protocol. Replay: causal delivery at receipt, outages never
reach the engine, bit-identical digests, lifecycle and heading choice, real-time pacing,
time-reversal refused, CAN/VBOX columns never exposed, start-up in the first second.

*Process note:* while adding these, `tests/navigation/test_recovery.py` (the Phase 4
lock-out tests) was accidentally overwritten; it was restored from git unchanged and the
new tests live in `test_recovery_manager.py`.

### 9.3 Real-drive benchmark battery

12 verified TRAIN + VAL drives, 144 events (full 10/30/60/120/300 s, tunnel 45 s,
intermittent 120 s, degraded 60 s), causal engine, shipped fusion defaults.
Train-drive absolute errors are optimistic (Model A was trained on them); arms are compared
on identical inputs.

### 9.4 What the benchmark found — two failed designs, kept on record

**Run 1** (`outage_benchmark_first_design.json`), recovery arm vs Phase 7 behaviour, output
error at event end (p50): degraded 60 s 16.5 → **1,309 m**; 10 s 47 → **244 m**; 60 s 453 →
**955 m**; tunnel 472 → **1,538 m**. Causes:
- the consistency check assumed a metre-accurate DR displacement over 9 s; with this phone's
  reduced IMU, heading is tens of degrees off after an outage, so **1,390 good fixes were
  rejected** and GNSS was held out while the filter drifted;
- velocity covariance was inflated with position;
- separately, a fixed 15 m/s output glide lagged kilometre corrections by minutes (S3a:
  output p50 **375 m** vs filter **22 m**).

**Run 2** (`outage_benchmark_second_design.json`), after: tolerance from the filter's own
velocity/heading σ, fail-open, position-only inflation, time-bounded glide.
- Inconsistency rejections fell from 1,390 to 25.
- The Phase 7 arm's output now tracks its filter (e.g. 30 s event end 256.6 m / 256.6 m).
- **But the recovery arm diverged, with filter jumps up to 1,186 km.** Cause: when inflating
  position could not admit a fix (its mismatch was in velocity), the covariance was
  inflated ×10,000 **and left inflated**; a second attempt compounded it to σ ≈ 2,000 km.
  It also acted during start-up, corrupting the 8-hypothesis heading choice.
- Lookahead measured on VAL (filter error at event end, whole-drive mount vs causal):
  10 s 42 vs 37 m, 30 s 287 vs 308 m, 60 s 221 vs 294 m, tunnel 250 vs 400 m (n 3–5 each):
  **the Phase 4/7 whole-drive mount was a mild advantage**, not decisive.

**Run 3** (corrected: inflation only for a position mismatch, capped ×100, refused and
undone otherwise; start-up excluded; linear glide smoother): completed after the commit.

### 9.4a Run 3 results (`reports/phase9/outage_benchmark.json`, 144 events, 0 divergences)

Recovery manager ON vs OFF (train + val, p50 per event kind):

| Event | Filter error at event end | Output error 30 s after | Time back within 20 m |
| --- | --- | --- | --- |
| full 10 s | 44.5 → 53.9 m | **16.4 → 9.2 m** | 8.4 → 8.1 s |
| full 30 s | 256.6 → 239.1 m | 31.3 → **111.9 m** | 30.5 → 41.0 s |
| full 60 s | 449.1 → 639.1 m | 283.5 → 296.1 m | 46.8 → 48.5 s |
| full 120 s | 827.2 → 817.3 m | 355.4 → 342.6 m | 51.1 → 54.6 s |
| full 300 s | 2,070 → 2,168 m | 626.5 → 546.4 m | 58.3 → 61.4 s |
| tunnel 45 s | 311.2 → 363.0 m | **195.3 → 121.7 m** | 49.3 → 41.7 s |
| intermittent 120 s | 926.2 → 984.8 m | 110.9 → **576.2 m** | 38.6 → 55.0 s |
| degraded 60 s | 11.0 → 13.9 m | 16.7 → 26.5 m | 0.1 → 0.1 s |

- **No divergence**: the largest filter jump is 8.2 km with the manager vs 8.5 km without (after 300 s
  outages: the correction itself). Inflations: 41 applied, 92 refused, 47 not a position
  mismatch; 22 inconsistent pairs; 12 fail-open timeouts.
- **Mixed, not better**: it helps after short dropouts and tunnels, hurts after 30 s and
  intermittent events, and mostly re-converges slower. Event-end errors differ too, because
  each recovery changes the state the next event starts from. **Decision: stays OFF.**
- **Output smoother cost**: in the manager-OFF arm, the median over drives of the per-drive
  median error is **63.0 m for the smoothed output vs 21.9 m for the filter**. With a fix only
  every 9 s, a recent correction is almost always still gliding in (10 s bound), so the
  output lags the filter's best estimate. The continuity guarantee is real, but so is the cost.
  The app should display a smoothed position only where continuity matters more than
  accuracy; the filter position is always available (`EngineOutput.filter_*`). **Open
  decision for the project owner.**
- Lookahead (VAL, filter error at event end, causal vs whole-drive mount): 10 s 37 vs 42 m,
  30 s 308 vs 287 m, 60 s 294 vs 221 m, tunnel 400 vs 250 m, 300 s 7,660 vs 6,299 m (n 2–5):
  the whole-drive mount of Phases 4/7 was a mild advantage.

### 9.5 What is and is not established

| Claim | Status |
| --- | --- |
| Outage simulator and replay are correct and deterministic | **verified** (tests; exact Phase 7 reproduction) |
| Output never teleports across LOST → RECOVERING → GOOD | **verified** by construction and tests (synthetic); run 2 real-drive: a *previous* smoother version |
| Final linear-glide smoother on real drives | **measured (run 3)**: continuous (max 90 m per 0.1 s glide after 300 s outages), but triples the typical error (63 vs 22 m) |
| Recovery manager improves real-drive recovery | **not established**: two designs failed; the third is neutral-to-mixed. OFF by default |
| Unit tests passing ⇒ correct on real data | **false in this phase**: all tests passed while runs 1 and 2 failed. Synthetic heading was near-perfect; real heading is not |

### 9.6 Exit criteria

- [x] Outage simulator in `simulation/outage/` (10/30/60/120/300 s, tunnel, intermittent, degraded)
- [x] Recovery logic in `navigation-core/recovery/` with consistency checks and gradual fusion; output continuity
- [x] Replay engine in `simulation/replay/`, deterministic, streaming, causal
- [x] Unit tests for continuity across LOST → RECOVERING → GOOD; full suite passes
- [x] Real-drive evaluation of the recovery manager (run 3): mixed, no clear benefit -> OFF
- [x] Decide what the app displays: **the filter position** (project owner, 2026-09-28: accuracy
      21.9 m beats continuity 63.0 m for an engineering demonstration, and the visible jump marks
      the moment GNSS is re-fused). `EngineConfig.display = "filter"` is the default; the
      smoothed position stays available (`display="smoothed"`, used by the Phase 9 benchmark)

---

## Phase 11 — Evaluation Harness & Benchmarking

**Status:** `NOT STARTED`
**Note:** this is TODO.md's Phase 11. The Kotlin work committed as "Phase 10" (`4556c2e`)
and "Phase 11" (`f832719`) is re-filed under Phases 12 and 13 below (renumbering of
2026-09-28). Nothing has been built for the harness yet; the Phase 9 outage benchmark
(`scripts/evaluate/phase9_outage_benchmark.py`) is the closest existing piece.

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

**Status:** `IN PROGRESS` — Python side done and checked here; Kotlin side **written, never
run** (no JDK / Android SDK on this machine, §0.8). Blocked on the first `:core:test` run,
which the CI workflow (§12.3) or the build machine (`TESTING_GUIDE.md`) will provide.
**Date:** 2026-09-28
**Scope note:** the parity work was committed under the labels "Phase 10" (`4556c2e`) and
"Phase 11" (`f832719`); it is filed here to match TODO.md.

**Objective:** Enforce the parity contract in `AGENTS.md` §4.

### 12.1 What exists

| Item | Where | State |
| --- | --- | --- |
| ONNX export of Model A / Model B + model cards (SHA-256, feature order) | Phase 6 §6.4, `models/exported/` | **done**; Python parity gate (atol 1e-5 + rtol 1e-6) passes |
| Golden vectors from the Python reference | `scripts/parity/generate_golden_vectors.py`, `nav_golden.py` → `tests/regression/golden/` | **done**; `--check` regenerates them byte-identically (on this Windows machine); `tests/regression/test_golden_vectors.py` passes |
| Kotlin replay of the vectors | `android-app/core/src/test/.../ParityTest.kt`, `nav/NavParityTest.kt` | **written, never run** |
| Model A through ONNX Runtime (JVM) against the Python outputs | `NavParityTest` (`nav_model_a.json`) | **written, never run** |
| CI on every push | `.github/workflows/android.yml` | **written, never run** (§12.3) |
| Shared constants file generated for both languages | — | **not done**: `Geometry.kt` copies the Python constants by hand; the golden vectors would catch a drift, but no generator exists |
| Float vs double policy per module | — | **partly**: stated in code comments (double math; float32 cast for model features); no standalone document |
| Cross-language runner reporting max abs / rel deviation | — | **not done**: the Kotlin tests assert tolerances, they do not print a deviation report |
| On-device (not JVM) replay of a golden recording | — | **not done** (Phase 13 exit criterion) |

#### Base-layer vectors (written with Phase 13 Part A, commit `4556c2e`)

**Golden-vector parity (AGENTS.md §4)**, replayed by the Kotlin `ParityTest`:
`gnss_state_machine.json` (421 events, every transition, boundary values), `sample_validation.json`
(10 accept/reject cases), `wgs84_radii.json` (10 latitudes, 1e-12 relative). Generated from
the Python reference; `tests/regression/test_golden_vectors.py` **passes** and fails if the
reference and the committed vectors ever diverge.

### 12.2 Navigation-core golden vectors (written with Phase 13 Part B, commit `f832719`)

| Vector | Content | Kotlin tolerance |
| --- | --- | --- |
| `nav_geometry.json` | quaternion ops (incl. the small-angle branch), ECEF ↔ geodetic, gravity, earth/transport rate, tangent plane | 1e-15 … 1e-13 (quaternions), 1e-8 m (ECEF) |
| `nav_ins.json` | 300 propagation steps, variable dt | 1e-6 m, 1e-9 m/s, 1e-11 |
| `nav_ekf_ops.json` | 40 predicts (full and reduced gyro), GNSS, ZUPT, levelling, NHC (vertical, lateral), a generic update, a gate rejection, a reset — full P after each | gate decisions **exact**; P 1e-7 relative |
| `nav_features.json` | 4 windows × 13 × 50 features | float32, 2 ulp |
| `nav_model_a.json` | Model A outputs on 5 windows | the Phase 6 gate: 1e-5 + 1e-6·\|y\| |
| `nav_navigator.json` | 180 s synthetic drive, 60 s dropout, full pipeline (deterministic speed stand-in) | GNSS state + all counters **exact**; 1 mm, P 1e-5 relative |
| `nav_engine.json` | 150 s drive, 40 s dropout, 8 hypotheses from a first-second start | mode, state, chosen heading, tilt source **exact**; 1 mm |
| `nav_map_matching.json` + `mm_*.roads.bin` | 4 HMM scenarios (drift at a junction, turn with a break, off-network suspend/re-enter, dual carriageway), Viterbi paths, the DR wrapper, course σ | modes, segments, paths, counts **exact**; 1e-9 m |

The same engine vector is also replayed through the **device path** (raw channel events →
10 Hz resampler → fix queue → engine) and must reproduce the reference, so the device-only
glue cannot silently change the navigation. Kotlin tests: 37 in `:core` (27 from Part A + 10 parity), none run yet.

### 12.3 CI

`.github/workflows/android.yml`, on every push and pull request, two independent jobs:

- **python-golden** (Ubuntu, Python 3.14, CPU torch): `tests/regression/test_golden_vectors.py`,
  i.e. the committed vectors must still be what the Python reference produces. The vectors
  were generated on Windows; byte-identical regeneration on Linux has **never been checked**,
  so a failure of this job on its first run may be a platform difference rather than a
  regression, and must be investigated, not waved through.
- **android** (Ubuntu, JDK 17, Android SDK): `./gradlew :core:test :app:assembleDebug`;
  uploads the `:core` test reports and the debug APK as artifacts.

The Gradle wrapper JAR is not committed (it could not be generated here). Until the build
machine commits `gradlew` + `gradle/wrapper/gradle-wrapper.jar`, the workflow creates the
wrapper itself with Gradle 8.11.1. **The workflow has not run yet**: its first run is the
first time any of this Kotlin is compiled; expect first-build fixes.

### 12.4 Exit criteria

- [x] Exported models match the Python trainer within a stated tolerance (Phase 6 gate, Python ONNX Runtime)
- [x] Golden-vector fixtures: geodesy, rotations, INS steps, filter updates, NHC, features, Model A, full pipeline, map matching
- [ ] Kotlin parity within the §4 targets across all modules — **`:core:test` has never run**
- [ ] Parity suite runs in CI; any failure blocks release — workflow written, **not yet run**
- [ ] Shared constants file generated from one source
- [ ] Float vs double policy documented per module
- [ ] Deviation report (max abs / rel) from the cross-language runner

---

## Phase 13 — Android Application

**Status:** `BLOCKED` (build machine + handset) — **source complete for Parts A and B, never
compiled**. `COMPLETED` only when `./gradlew :core:test :app:assembleDebug` passes (CI or the
build machine) and the exit criteria in §13D are met on a real phone.
**Build / test:** `TESTING_GUIDE.md`; CI: `.github/workflows/android.yml`.

### 13A — Base & Sensors (commit `4556c2e`, formerly labelled "Phase 10")

**Status:** `BLOCKED` on verification — **source complete, never compiled**. This machine has
no JDK, Gradle, Kotlin compiler or Android SDK (§0.8), so not one line of the Kotlin below has
been compiled or executed, and none of its unit tests has run. By this file's honesty rule the
part is not `COMPLETED` until `./gradlew :core:test :app:assembleDebug` passes on a build
machine. Everything that CAN be checked here was: the Python side of the parity contract
(golden vectors + their regression test) runs and passes.
**Date:** 2026-09-28
**Evidence:** `android-app/` (33 files, `README.md`); `scripts/parity/generate_golden_vectors.py`,
`tests/regression/golden/*.json`, `tests/regression/test_golden_vectors.py` (passing)
**Scope note:** committed as "Phase 10 — Android Application: Base & Sensors" (`4556c2e`);
re-filed as Part A of Phase 13 to match TODO.md. Its golden vectors are listed in Phase 12 §12.1.

**Build (on a machine with JDK 17 + Android SDK 35):**

```
cd android-app
gradle wrapper --gradle-version 8.11.1   # the wrapper JAR is a binary: not committed
./gradlew :core:test                     # pure JVM: needs no Android SDK
./gradlew :app:assembleDebug
```

#### 13A.1 Structure (Kotlin, Gradle Kotlin DSL, version catalog)

| Module | Kind | Contents |
| --- | --- | --- |
| `:core` | Kotlin/JVM | shared schema `ChannelSample` / `ImuSample` / `GnssSample` / `TimestampGuard` (ports of `navcore.sensors.samples`: same limits and rules); `AndroidMapping` (`SensorEvent` / `Location` → schema, one session clock = `elapsedRealtimeNanos`); `ImuAssembler` (pairs accel + gyro, stale gyro → invalid and zeroed, never guessed); `GnssStateMachine` (port of `navcore.state.gnss_state`); `RateMeter`; `SimulatedOutage`; `positionConfidence`; `SessionLogger`; `TrackProjector` (display only); `Display`; `NavigationEngine` contract + `GnssOnlyEngine` placeholder |
| `:sensors` | Android lib | `SensorSource`: accelerometer, gyroscope, magnetometer, gravity, registered on a **dedicated HandlerThread** → cold `Flow`; a bounded buffer whose drops and rejections are **counted** |
| `:gnss` | Android lib | `LocationManagerGnssSource` (GPS provider, **default**: pure GNSS, like IO-VNBD); `FusedGnssSource` (optional; may blend Wi-Fi/cell; provider logged on every line) |
| `:app` | Android app, Compose, SDK 35 (min 26) | `AcquisitionService`: foreground (type `location`); sensor, GNSS and outage-toggle collectors in **independent coroutines on `Dispatchers.Default`**, feeding ONE processing coroutine that alone owns engine, logger and meters (ordered, lock-free); the UI gets conflated `StateFlow` snapshots (≤ 10 Hz), decoupled from sensor rate. `NavigationScreen`, `DiagnosticsScreen`, `OnnxLatencyProbe`, `NavigationRepository` |

Versions (AGP 8.7.3, Kotlin 2.1.0, Compose BOM 2024.12.01, coroutines 1.9.0, ONNX Runtime
Android 1.20.0, Play services location 21.3.0) are the last known to the author and may need
bumping on the build machine.

#### 13A.2 Screens

- **Navigation**: track canvas (a **placeholder** for a map renderer: the reported positions
  in local metres, auto-scaled, north up; no tiles); HUD with speed (km/h), navigation state
  (`GOOD` / `DEGRADED` / `DEAD_RECKONING` / `RECOVERING`; the reference's `LOST` is displayed
  as `DEAD_RECKONING`), confidence %, horizontal σ, road name, engine; the **Simulate GNSS outage**
  toggle, which withholds fixes from the engine exactly like the Phase 9 simulator's `full`
  event, logs them flagged `withheld_by_sim`, and labels the state "(SIMULATED OUTAGE)".
- **Diagnostics**: sensor rates measured on the sensors' own timestamps, raw XYZ per channel,
  every counter (mapped / rejected / dropped / stale-gyro / withheld), EKF covariance diagonal
  (**n/a**: no EKF on the device yet, said so), AI inference latency, logger line counts.

**Confidence %** is a defined quantity, not a score: P(true position within 10 m) =
1 − exp(−R²/2σ²) for a circular Gaussian with per-axis σ; σ from Android's accuracy (a 68 %
radius) is accuracy / 1.5096.

**AI latency** = batch-1, single-thread ONNX Runtime inference of the bundled Model A on an
**all-zero** input: compute latency only. The model's SHA-256 is checked against its card
first, as in Python. The on-device feature pipeline is not ported, so the model does not drive
navigation, and the screen says so. The exported `.onnx` files and cards are copied into the
APK assets by a Gradle task: the same bytes the Phase 6 parity gate validated.

#### 13A.3 What the app does NOT do yet — stated on screen

| Item | On the device |
| --- | --- |
| Dead reckoning (INS + EKF + AI speed + NHC) | **no**. `GnssOnlyEngine` runs the ported state machine over GNSS; in `DEAD_RECKONING` it shows **no position** (and says why) rather than a stale or invented one |
| Model A driving navigation | no (latency probe only) |
| Map matching / road name | no ("—", with the reason) |
| Map tiles | no (track canvas) |
| Satellites used | not collected (needs a `GnssStatus` callback) |

#### 13A.4 Session logger (dataset expansion)

One directory per drive under the app's external files dir: `manifest.json` (schema v1,
session start in UTC ms **and** elapsed-realtime ns, device, SDK, app version, sensors
available, GNSS source); `sensors.csv` (`t_s,channel,x,y,z,accuracy`, raw DEVICE frame);
`gnss.csv` (fix epoch, receipt time, degrees, 68 % accuracy, speed, bearing, provider,
`withheld_by_sim`); `events.csv` (state transitions, outage toggles, rejected fixes).
Locale-independent full-precision numbers; buffered; synchronized; a write after close
throws instead of losing data; the service closes the log on cancellation.

#### 13A.5 Tests

Golden-vector parity for this part: Phase 12 §12.1.

**Kotlin (`:core`, 4 files, 27 tests; written, NOT yet run):**
- sensor-event mapping: channels, session clock, axes copied unchanged, rejections counted;
- location mapping: degrees → radians; a missing accuracy is rejected, never invented;
  speed without bearing; receipt vs epoch with jitter tolerance;
- IMU assembly: fresh vs stale gyro;
- schema rules; timestamp guard; rate meter; simulated outage windows; the confidence
  formula, including the 68 % identity; the logger (a German locale still writes `9.81`,
  CSV quoting, manifest, write-after-close fails); the placeholder engine giving no position
  when lost; HUD formatting.

#### 13A.6 Exit criteria

- [x] Android project structure: Kotlin, Gradle KTS, version catalog, SDK 35, Compose, coroutines
- [x] Sensor (accel / gyro / mag / gravity) and GNSS (LocationManager + Fused) wrappers mapped into the shared schema
- [x] Acquisition in a foreground service on background coroutines, decoupled from the UI
- [x] Navigation screen (track placeholder, HUD, outage toggle) and diagnostics screen
- [x] Session logger (CSV + JSON)
- [x] Kotlin unit tests written; golden vectors generated and checked on the Python side
- [ ] **Compiles** (`:app:assembleDebug`) — needs the build machine
- [ ] **Kotlin tests pass** (`:core:test`, incl. parity) — needs the build machine
- [ ] Runs on a real handset: rates, logging, outage toggle verified — needs a phone

### 13B — ML & Dead Reckoning Integration (commit `f832719`, formerly labelled "Phase 11")

**Status:** `BLOCKED` on verification — **source complete, never compiled** (no JDK, Gradle,
Kotlin compiler or Android SDK on this machine). The port is pinned by **golden vectors from
the Python reference**; those are generated and regression-checked here (passing), and the
Kotlin tests that replay them have not run yet. `COMPLETED` only after
`./gradlew :core:test :app:assembleDebug` passes on a build machine and a real drive is
recorded.
**Date:** 2026-09-28
**Evidence:** `android-app/core/src/main/kotlin/.../core/nav/`, `.../core/mapmatch/`;
`scripts/parity/nav_golden.py` → `tests/regression/golden/nav_*.json`, `mm_*.roads.bin`
(13 golden files, regenerated byte-identically by `generate_golden_vectors.py --check`);
`navigation-core/map_matching/bundle.py` + `scripts/export/export_road_bundle.py`;
full Python suite **275 passed**, `ruff` clean
**Scope note:** committed as "Phase 11 — Android ML & Dead Reckoning Integration" (`f832719`);
re-filed as Part B of Phase 13 to match TODO.md. Its golden vectors and tolerances are listed
in Phase 12 §12.2.

**Preceding owner decision (applied first, commit `cc73ed2`):** the app displays the
**filter position** (Phase 9: 21.9 m vs 63.0 m for the smoothed output; the visible jump
marks GNSS re-fusion). `EngineConfig.display = "filter"` in Python; the Kotlin engine
reports the filter position and has no smoother.

#### 13B.1 What was ported (Kotlin, `:core`, pure JVM)

| Kotlin | Python reference | Notes |
| --- | --- | --- |
| `nav/Linalg.kt` | numpy | row-major `Mat`, Cholesky, partial-pivot solve |
| `nav/Geometry.kt` | `navcore.common.constants`, `geometry.quaternion`, `geometry.geodesy` | same constants, same small-angle threshold, Heikkinen ECEF→geodetic, `LocalTangentPlane` |
| `nav/Ins.kt` | `ins.mechanization.propagate` | Heun velocity, trapezoidal position, earth/transport rate, Somigliana gravity |
| `nav/Ekf.kt` | `filtering.ekf`, `recovery.gnss_reset` | 15-state error-state EKF: F, Q, predict (incl. the Phase 7 unmeasured-axis fix and bias decoupling), Joseph update, NIS gate, Cholesky check, fix-epoch history, GNSS / ZUPT / levelling, reset_to_fix, lock-out reset |
| `nav/Aiding.kt` | `nhc.constraints`, `fusion.features`, `fusion.ai_models` | NHC (vertical + lateral), the 13 model features (double math, float32 cast), `ImuWindow`, speed innovation, logvar clamp |
| `nav/OnnxSpeedModel.kt` | `fusion.ai_models.OnnxSequenceModel` | ONNX Runtime; refuses the model unless SHA-256 **and** feature order match its card |
| `nav/FusedNavigator.kt` | `fusion.navigator` | the shipped defaults: AI speed (fresh, ≤ 1 Hz, variance × window/interval), NHC only with AI speed, state machine, stillness ZUPT; Model B and the recovery manager (both OFF by default) not ported |
| `nav/DrEngine.kt` | `fusion.engine.NavigationEngine` | WAITING → ALIGNING (8 heading hypotheses, NIS score) → NAVIGATING; causal tilt incl. the gravity-channel start-up |
| `mapmatch/RoadNetwork.kt`, `HmmMapMatcher.kt` | `map_matching.road_network`, `hmm`, `outage` | reads the road BUNDLE; same grid, Dijkstra, HMM, Viterbi, off-network suspension, `map_match_confidence`, dead-reckoning wrapper |

**Device-only additions (no Python counterpart, stated as such):**
- `TenHzResampler`: a phone delivers ~50–200 Hz; the model and the reference run at IO-VNBD's
  10 Hz. It averages each 0.1 s bin (boxcar), with bin ends from an integer index (no drift),
  and SKIPS and counts a bin missing a channel. Averaging vs decimation is a deviation from
  the (undocumented) IO-VNBD logging to check against on-device recordings.
- `AxisMap.ANDROID_LIVE`: a live phone's three gyro axes are all real rates (IO-VNBD's
  reduced gyro was a property of that dataset); `AxisMap.IOVNBD` is kept for parity tests.
- `DeadReckoningNavigation`: plugs the engine into the app. Channels → resampler → engine;
  fixes are released with the first 10 Hz sample at or after their RECEIPT time; the map
  matcher runs on the filter output once NAVIGATING; the map-matched position is reported
  **alongside** the filter position, never instead of it, never fed back.

#### 13B.2 Offline road bundle

`navcore.map_matching.bundle`: a documented big-endian binary (magic `SIHROAD1`) holding
the same arrays as the Python network, so both implementations build the identical graph
(same segment ids). `scripts/export/export_road_bundle.py` writes
`models/roads/coventry.roads.bin` (13.2 MB, 450,775 directed segments; **gitignored**, ODbL,
regenerable) and a tracked manifest with its SHA-256. The app bundles it when present, checks
the SHA-256, and loads it in the background. It takes seconds, and is handed to the
processing coroutine as a message. Round trip tested in Python (`test_road_bundle_round_trip…`).

#### 13B.3 App integration

- `EngineFactory` builds the engine from APK assets: measured IMU noise
  (`config/imu_noise.json`, Phase 4), Model A + card, road bundle + manifest. Downgrades are
  explicit and shown under "Engine resources":
  - **no noise file** → GNSS-only engine (no dead reckoning on invented Q);
  - **Model A missing or refused** → AI speed OFF, and therefore NHC OFF too (the Phase 7
    pairing rule);
  - **no road bundle** → no map matching.
- **Navigation screen:**
  - track canvas: the engine's track (blue), the current position (red, **keeps moving on
    dead reckoning when GNSS is lost**), map-matched positions (green);
  - HUD: speed, state (`DEAD_RECKONING` when lost), confidence P(within 10 m), σ, road name
    (or why none: no map / no confident match), map-match confidence, engine, mode;
  - "(provisional)" note while the heading is being aligned.
- **Diagnostics:** the EKF covariance diagonal is live; the model latency is shown both live
  (each real inference) and from the zero-input probe; skipped 10 Hz bins.
- Session logs record the engine, its notes, the display choice and the road bundle used.

#### 13B.4 What is not established

| Claim | Status |
| --- | --- |
| The Kotlin compiles | **unknown**: never compiled; expect first-build fixes |
| Kotlin = Python within the tolerances above | **unknown until `:core:test` runs**; the reference side is fixed and checked |
| The phone dead-reckons as well as Phase 7 measured | **unknown**: a live phone differs from IO-VNBD: full 3-axis gyro, 10 Hz resampling of a faster stream, a different mount. Needs recorded drives replayed through both stacks |
| Model A generalises to other phones | **unknown**: trained on one IO-VNBD phone model |
| Map matching helps on the device | Phase 8 measured a marginal gain (only while drift < ~30 m); nothing new here |

#### 13B.5 Exit criteria

- [x] Model A through ONNX Runtime from APK assets, integrity-checked (SHA-256 + feature order)
- [x] Rolling 5 s window at 10 Hz fed exactly as the model card defines (parity vectors)
- [x] INS mechanization and EKF predict/update ported; the phone computes its position in `DEAD_RECKONING`
- [x] Map matcher ported; reads the offline road bundle; constrains (reports alongside) the DR trajectory
- [x] Compose navigation screen shows the moving DR marker and the map-matched positions
- [x] Golden vectors for every ported layer, regenerated and checked on the Python side
- [ ] Compiles; `:core:test` (37, incl. parity) passes — build machine
- [ ] A real drive recorded on a phone, replayed through Python and Kotlin — handset

### 13C — Against the original Phase 13 plan (TODO.md)

| Planned | State |
| --- | --- |
| Android project scaffolding | written (13A) |
| Sensor acquisition with real hardware timestamps; honest rate handling | written (13A; 13B resampler) |
| Port calibration, AHRS, INS, EKF, NHC against golden vectors | INS, EKF, NHC ported with vectors (13B); calibration and AHRS are not ported as separate stages |
| On-device inference | ONNX Runtime, Model A (13B) |
| Background-service lifecycle, doze behaviour | foreground service written; doze **not tested** |
| Battery, CPU, thermal **measured** | **not done** (needs a handset) |
| Offline map tiles; on-device map matching | map matching written (13B); **no tiles** |
| On-device parity harness (replay a golden recording) | **not done** |
| UI: position, uncertainty, pipeline mode, GNSS status | written (σ, confidence, state); no uncertainty ellipse drawn |
| Full operation with GNSS disabled and no network | **not tested** |

### 13D — Exit criteria

- [ ] Compiles; `:core:test` passes (CI or build machine)
- [ ] On-device replay of a golden recording matches Python within tolerance
- [ ] Real-time performance measured on real hardware (latency, CPU, battery, thermals)
- [ ] Works fully offline with GNSS disabled
- [ ] No coordinate produced outside the ported geodesy path (true by construction in the source; to be confirmed once it builds)

---

## Phase 14 — End-to-End Validation, Documentation & Demo

**Status:** `NOT STARTED`

**Objective:** Prove the system on data it has never seen, and describe it honestly.

**Planned scope**
- Field collection on new routes (tunnels, underpasses, urban canyon, multi-storey car
  parks) with GNSS truth logged for evaluation only.
- Comparison of on-device results against the Phase 11 Python benchmarks.
- Build and field test on a physical handset by a teammate (`TESTING_GUIDE.md`).
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
| 2026-09-27 | Phase 3 COMPLETED (renamed "Preprocessing Pipeline & Coordinate Systems"): Phase 2 committed (`e22997b`); ruff installed at the pinned 0.16.9, repo lint-clean; `configs/dataset.yaml` replaces hardcoded paths; single constants source; wheel speed calibrated vs VBOX (rear axle, k = 1.00019, km/h by inference, FWD); `navcore.geometry` quaternion/rotation/geodesy/frames with 78 tests (pyproj oracle, mutation check 5/5); UTC-time sync of 72 sessions with measured GNSS latency (median 4.1 s) as epoch tags; CAN-as-ground-truth enforced in code; drive-level split v2 (0 leaks; v1 had 49); train-only normalisation. **Corrections to earlier phases:** phone GPS speed is m/s not km/h (Phases 0-2 divided by 3.6); total recorded time 29.56 h not 25.08 h; xcorr lag bias fixed (Phase 2 conclusions unchanged). `scripts/phase3_validate.py` 20/20 CRITICAL PASS. |
| 2026-09-27 | Phase 4 COMPLETED (redefined "Baseline Navigation Core"): sensors, alignment, INS, 15-state EKF, recovery, Allan; horizontal gyro columns shown unusable (reduced IMU); Phase 3 GNSS "4.1 s latency" corrected to a 9 s sample-and-hold artefact (new-fix delay ~0 s); spec sec 8.3 sign corrected; EKF consistent in Monte Carlo (NEES 16.5/15, NIS 5.00/5); real-data baseline on verified non-test drives: 7.8 m with GNSS, 291 m after 30 s and 995 m after 60 s without; `scripts/phase4_validate.py` 12/12 CRITICAL PASS. |
| 2026-09-27 | Phase 5 COMPLETED (redefined "ML Training Pipeline"): configs/training.yaml, windowed datasets with gap/segment/target gates and CAN/coordinate guards, mount-invariant features + rotation/noise/bias/jitter augmentation, model registry (TCN/CNN1D/GRU/LSTM, heteroscedastic heads), Huber+NLL loss, trainer with CUDA-OOM -> smaller batch -> CPU fallback, scripts/train/train_all.py. Smoke test passes on CPU; limited run: Model A val MAE 5.92 m/s vs 7.02 constant reference. Full training NOT yet run; CUDA torch not installed. 199 tests pass. |
| 2026-09-27 | Phase 6 COMPLETED (redefined "ML Evaluation & Export"): CUDA torch on RTX 2050; 16-run sweep; selected model_a_cnn1d_w50 (test MAE 3.91 m/s vs 5.47 constant, bias +2.81) and model_b_lstm_w20 (test 0.455 vs 0.474 m/s², weak); ONNX export + model cards; parity gate atol 1e-5 + rtol 1e-6 with an fp64-referenced stress set (agreed with the owner), both PASS; EKF reduced-IMU tilt fix. Committed e0eca3c. |
| 2026-09-27 | Phase 7 COMPLETED (redefined "Sensor Fusion & NHC"): ONNX wrapper with integrity checks, Model B pre-predict correction, Model A speed update by GNSS state, NHC, GNSS state machine, FusedNavigator; 15 fusion tests incl. 8-seed dropout test. Fixed: AI speed applied at 10 Hz as independent (aided 7 -> 42 m on S3a). **Phase 4 correction:** withheld GNSS speed leaked into stillness detection (val 60 s 1,226 -> 1,989 m). Real data: AI speed cuts 60 s outage error 2,814 -> 490 m but worsens some 30 s outages; NHC alone hurts and diverged once; Model B hurts. |
| 2026-09-27 | Phase 8 COMPLETED (redefined "Offline Map Matching"): OSM drivable network cached once (Overpass, 30.6 MB, 450,775 directed segments, ODbL) with provenance manifest; `navcore.map_matching` RoadNetwork (directed, one-way rules, grid index, bounded Dijkstra, npz cache), online HMM (covariance emission + heading, route-continuity transition, Viterbi, off-network suspension, map_match_confidence), DeadReckoningMapMatcher (read-only on the EKF); 15 tests. VAL: outage epochs 83.9 -> 80.7 m, cross-track 47.7 -> 39.2 m, correct road 20 % (0 % at outage ends); gain only while drift < ~30 m (15-30 m band: cross-track 11.4 -> 5.4 m). |
| 2026-09-28 | Phase 9 COMPLETED for code + tests (redefined "GNSS Outage Detection, Recovery & Replay"): outage simulator (full/tunnel/intermittent/degraded, battery), deterministic causal replay engine (reproduces Phase 7 S3b exactly), streaming NavigationEngine, RecoveryManager (opt-in, OFF) + OutputSmoother (ON); 27 new tests. Real-drive benchmark: runs 1 and 2 showed the recovery manager failing (1,390 good fixes rejected; then 1,186 km divergence from a kept failed inflation) -- both fixed and kept on record; run 3 in progress at commit time. |
| 2026-09-28 | Phase 10 source complete, BLOCKED on verification (redefined "Android Application — Base & Sensors"): android-app/ Gradle KTS monorepo (:core JVM, :sensors, :gnss, :app), shared schema + GNSS state machine ported, foreground acquisition service with independent coroutines, Compose navigation + diagnostics screens, outage toggle, CSV/JSON session logger, ONNX latency probe, 27 Kotlin tests; golden vectors from the Python reference (state machine, sample validation, WGS84 radii) with a passing Python regression test. Nothing compiled: no JDK/Android SDK on this machine. |
| 2026-09-28 | Display default set to the filter position (owner decision; cc73ed2). Phase 11 source complete, BLOCKED on verification (redefined "Android ML & Dead Reckoning Integration"): Kotlin ports of geometry, INS, 15-state EKF, NHC, features, ONNX Model A (SHA-256 + feature-order checked), fused navigator, streaming engine with heading hypotheses, 10 Hz resampler, map matcher reading a new binary road bundle (Coventry: 13.2 MB, gitignored, manifest tracked); app shows the DR marker moving when GNSS is lost plus map-matched positions; 13 golden files from the Python reference (regenerated byte-identically) replayed by 37 Kotlin tests. Nothing compiled: no JDK/Android SDK here. |
| 2026-09-28 | **Renumbering to match TODO.md:** the Android work committed as "Phase 10" (`4556c2e`) and "Phase 11" (`f832719`) is re-filed as Phase 12 (parity: golden vectors + Kotlin parity tests; `IN PROGRESS`) and Phase 13 Parts A/B (the app; `BLOCKED` on build machine + handset); Phase 11 is the Evaluation Harness again (`NOT STARTED`). Added `.github/workflows/android.yml` (golden-vector check; `./gradlew :core:test :app:assembleDebug`) and `TESTING_GUIDE.md`. Neither the workflow nor any Kotlin has run yet. No code or result changed. |
| 2026-09-28 | Housekeeping before field testing: Phase 5-9 crosswalk to TODO.md titles added (no phase renamed; executed 8/9 differ from TODO.md 8/9). Old "Phase 10"/"Phase 11" labels for the Android work replaced by Phase 12/13 in code comments, docstrings, `docs/architecture.md`, `.gitignore`; `versionName` and the session-manifest version `0.10.0-phase10` -> `0.13.0-phase13`. Road bundle for any area: `scripts/export/export_road_bundle.py` (and `scripts/download/fetch_osm_roads.py`) take `--city` or `--bbox S W N E`; per-area provenance `reports/osm/<name>_osm_manifest.json` (Coventry's Phase 8 files unchanged, its query asserted byte-identical by `tests/unit/test_osm_area.py`, 14 tests); run once for "Nagpur, India" (7.8 MB bundle, 279,967 directed segments; not committed). `EngineFactory` now refuses to choose between several bundles instead of loading the alphabetically first. `TESTING_GUIDE.md` step 5 added. The Kotlin change is uncompiled like the rest. |
| 2026-09-28 | `TODO.md` aligned with the project state: numbering typos from the mechanical renumbering fixed ("Phases 3 and 3", "Phase 7–7"); every checklist re-ticked against this file, each item saying where it was delivered; Immediate Next Actions now start with the first build and the Nagpur field test. No code or result changed. |
