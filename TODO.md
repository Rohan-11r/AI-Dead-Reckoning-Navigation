# TODO.md — High-Level Roadmap (SIH26168)

**AI-ML based Intelligent Dead Reckoning system for seamless navigation**

Read [`AGENTS.md`](./AGENTS.md) first — it governs everything below.
Detailed per-phase status lives in [`PROJECT_STATUS.md`](./PROJECT_STATUS.md).

**Numbering.** This file keeps the planned phase numbers and titles. PROJECT_STATUS.md files
the work as it was executed; Phases 0-7 and 11-14 match, but the work executed as "Phase 8"
is this file's Phase 10 (Map Matching), and the work executed as "Phase 9" (GNSS outage
detection, recovery & replay) is spread over Phases 8 and 11 below. PROJECT_STATUS.md has
the crosswalk. Each item below says where it was delivered.

---

## The Shape of the Work

```
Phase 0   Environment                          [COMPLETED]
Phase 1   Architecture & repo                  [COMPLETED]
Phase 2   Data                                 [COMPLETED]
Phase 3   Preprocessing & frames               [COMPLETED]
Phase 4   Baseline nav core (INS + EKF)        [COMPLETED]
              |
Phase 5   ML training pipeline  ------------\
Phase 6   ML evaluation & export ----------- +-- learned layer    [COMPLETED]
              |
Phase 7   Sensor fusion & NHC   ------------\                     [COMPLETED]
Phase 8   Fusion EKF            ------------ |                    [mostly done; follow-ups open]
Phase 9   NHC                   ------------ +-- estimation       [core done; follow-ups open]
Phase 10  Map matching          ------------/                     [COMPLETED as executed Phase 8]
              |
Phase 11  Evaluation harness    -- the truth-teller               [NOT STARTED]
              |
Phase 12  Export + parity       ------------\                     [IN PROGRESS]
Phase 13  Android               ------------ +-- deployment       [BLOCKED: build machine + handset]
Phase 14  Field validation + docs ----------/                     [NOT STARTED]
```

Phases 5 and 6 replaced the original AHRS and INS phases (the INS and EKF were delivered in
Phase 4); the original two AI phases were merged into Phase 7. Everything from Phase 12 on
needs a JDK and the Android SDK, which the authoring machine does not have: the Kotlin is
built in CI and on the teammate's machine (`TESTING_GUIDE.md`).

---

## Immediate Next Actions (top of the stack)

1. **First build (Phases 12-13):** on the teammate's machine or in CI, run
   `./gradlew :core:test` and `./gradlew :app:assembleDebug`; fix first-build errors;
   commit the Gradle wrapper. No Kotlin has been compiled yet.
2. **Field test in Nagpur (Phases 13-14):** generate the local road bundle
   (`scripts/export/export_road_bundle.py --bbox ...`), install the APK, record drives with
   the session logger, and bring the logs back for replay through Python.
3. **Models:** full-length training; fix Model A's +2.8 m/s bias on test drivers; retrain or
   drop Model B.
4. **NHC follow-ups (Phase 9):** correlated-error model (rate / σ), time-varying mount tilt,
   covariance robustness at a GNSS reset (the NHC-only arm diverged on S3c).
5. **Evaluation (Phase 11):** only 3 validation drives are scoreable — too few; build the
   harness and plan the one-time TEST run.
6. **Housekeeping:** the laptop throttles on battery (runs 3-10x slower); plug in for long runs.

---

## Phase 0 — Environment Bootstrap & Reconnaissance

- [x] Enumerate OS, CPU, RAM, GPU, disk from live system queries
- [x] Establish Python versions and which ML packages are present
- [x] Check JDK / Gradle / Android SDK / adb availability
- [x] Verify outbound network reachability
- [x] Locate and verify the IO-VNBD dataset on disk
- [x] Inspect the real CSV schema, encoding, sampling rate, and parsing hazards
- [x] Write `AGENTS.md` (core directive)
- [x] Write `PROJECT_STATUS.md` (phases 0–14)
- [x] Write `TODO.md` (this file)
- [x] Write and run a re-runnable environment validation script (`scripts/phase0_validate.py`, 16/16 PASS)

---

## Phase 1 — Architecture & Repository Creation

**Evidence:** `reports/phase1_validation.txt` — 17/17 CRITICAL checks PASS

- [x] Monorepo structure from a re-runnable manifest (82 entries: 51 packages, 31 dirs)
- [x] `scripts/setup/create_structure.py` — idempotent, manifest-driven
- [x] `git init` + `.gitignore` (data, venv, checkpoints, build) + `.gitattributes` (LF)
- [x] Python version resolved on live wheel evidence: **3.14.4**, 3.13.5 as fallback
- [x] `requirements.txt` (22 exact pins) + `-dev` / `-geo` / `-gpu` variants
- [x] `pyproject.toml`: packaging, `navcore` remap, pytest/ruff/mypy config
- [x] `.venv` created and isolated; 22/22 dependencies installed at pinned versions
- [x] `pip install -e .`; all 43 packages import; `navcore.*` verified to resolve on disk
- [x] `navigation-core` → `navcore` mapping (hyphen is not a legal identifier)
- [x] `docs/architecture.md` — hybrid ML/physics diagram, order rationale, runtime modes
- [x] `docs/navigation_math.md` — the specification: frames, WGS84, INS, EKF, NHC, ZUPT
- [x] `docs/ml_pipeline.md` — splits, windowing, models, losses, export, parity
- [x] `docs/map_matching.md` — HMM, off-network detection, feedback policy
- [x] `README.md`, `.env.example`, `scripts/setup/activate.{ps1,sh}`
- [x] `scripts/phase1_validate.py` — structure, venv, pins, imports, ONNX round trip
- [x] Anti-drift assertions: requirements ↔ pyproject pins, pyproject packages ↔ manifest
- [x] Verified torch → ONNX → onnxruntime round trip (max |Δ| = 2.98e-08, tol 1e-5)
- [x] First git commit (made at the start of Phase 2)

Defects found and fixed in this phase (each would otherwise have surfaced in Phase 12):

- [x] `onnxscript` required by `torch.onnx.export` but undeclared by torch — now pinned
- [x] dynamo ONNX exporter crashes on its own `✅` under cp1252 — `PYTHONUTF8=1` set
- [x] `packages.find` cannot see through `package-dir`; silently found zero navcore
      packages — replaced with an explicit list, now asserted against the manifest

---

## Phase 2 — Dataset Ingestion, Schema Normalisation & Integrity Audit

**Goal: know exactly what is wrong with the data before any model sees it.**
**Status: COMPLETED 2026-09-27** — `reports/phase2_validation.txt`, 26/26 CRITICAL PASS.

- [x] Pinned venv + `requirements.txt` *(done in Phase 1)*
- [x] Python version decided on evidence *(done in Phase 1)*
- [x] `git init` + `.gitignore` *(done in Phase 1; initial commit made at Phase 2 start)*
- [x] Raw layout normalised to `data/raw/IO-VNBD/` (rename only, inventory verified)
- [x] Single config module holding the one dataset path *(done in Phase 3: `configs/dataset.yaml`)*
- [x] Loader: cp1252 + per-token mojibake repair, normalised column names, explicit datetime
- [x] Split `GPS SATELLITES IN RANGE` into used/visible; 2,118 Excel-damaged cells repaired
- [x] Convert km/h to m/s at the loader boundary; SI everywhere inside *(Phase 3: phone GPS speed was actually m/s -- fixed)*
- [x] Checksum `Categorised` vs `Uncategorised` — settled: 72 unique sessions
- [x] Per-recording integrity audit: rows, duration, real `dt` histogram, monotonicity,
      gaps, duplicates, per-column NaN counts
- [x] GPS quality profile: accuracy, satellite counts, fix update interval
- [~] Kinematic plausibility screen: gravity vs g0 and local g, sensor ranges, GPS speed vs
      VBOX speed. **GPS speed vs differentiated GPS position not done.**
- [~] Stationary rows detected (CAN + wheel + VBOX speed all zero) for the audit;
      **not exported as reusable segments** (the filter detects stillness from phone data)
- [x] Parquet with an explicit versioned schema (`iovnbd-v1`, units in metadata)
- [x] **Route/driver-level** split manifest `data/splits/iovnbd_split_v1.json`
      *(superseded by the drive-level v2 in Phase 3)*
- [x] Leak check: no session, route group, or test driver spans two splits
- [x] Dataset card: `reports/phase2/dataset_report.html` (+ `.json`)
- [x] Loader unit tests (22)

**Items found by Phase 2, and where they went:**
- [x] Resolve the two horizontal gyro axes — Phase 4: columns 1/3 are not angular rates (reduced IMU)
- [x] Allan variance from stationary sessions — Phase 4 (`reports/phase4/imu_noise.json`)
- [x] Wheel-speed unit and odometry scale vs VBOX speed — Phase 3 (k = 1.00019)
- [x] Phone-GNSS "4.1 s latency" — Phase 4: a 9 s sample-and-hold artefact, not latency; no latency model needed
- [x] **Decided (user, 2026-09-27):** V-file CAN channels are ground truth only, never model
      or filter inputs — enforced in code (Phase 3)
- [ ] Per-session S/V time-alignment for the 47 unverified sessions — until then only
      clock-verified sessions are scored

---

## Phase 3 — Preprocessing Pipeline & Coordinate Systems

**Status: COMPLETED 2026-09-27** — `reports/phase3_validation.txt`, 20/20 CRITICAL PASS.

- [x] Commit Phase 2; install pinned ruff; repo lint-clean
- [x] `configs/dataset.yaml` + loader; no hardcoded dataset paths (gate-checked)
- [x] Wheel-speed unit/scale calibrated vs VBOX (rear axle, FWD, k = 1.00019)
- [x] Single Python constants source (`navcore.common.constants`)
- [x] Quaternion / rotation / geodesy / frames in `navigation-core/geometry/`
- [x] 78 math tests incl. pyproj oracle and a 5/5 mutation check
- [x] UTC-time sync pipeline; GNSS timing measured and epoch-tagged
- [x] CAN = ground truth only, enforced by `columns.assert_model_inputs`
- [x] Drive-level split v2 with independent leak audit (0 leaks)
- [x] Train-only normalisation `models/normalization/imu_{mean,std}.json`
- [ ] Geodesic (Vincenty/Karney) distance for evaluation -> Phase 11
- [ ] Generate Kotlin constants from the Python source -> Phase 12

---

## Phase 4 — Baseline Navigation Core (sensors, alignment, INS, EKF)

**Status: COMPLETED 2026-09-27** — `reports/phase4_validation.txt`. Redefined by the user:
absorbs the INS and the core of the classical EKF (planned for Phases 6 and 8), plus the
calibration items below.

- [x] `SensorSample` / `ImuSample` / `GnssSample` with frames, validation, epoch vs receipt time
- [x] Gyro axis question settled empirically: columns 1/3 are NOT angular rates -> reduced IMU
- [x] Accelerometer gravity inclusion confirmed (Phase 2)
- [x] Allan variance (stationary Vw1) + moving vibration level -> EKF process noise
- [x] Phone-to-vehicle alignment (gravity levelling + GNSS-motion yaw), with observability flags
- [x] Strapdown INS (Earth rate, transport rate, Coriolis, Somigliana gravity), 17 analytic tests
- [x] 15-state error-state EKF: predict, GNSS (epoch-aware), ZUPT, levelling; Joseph; PD guards
- [x] F and H validated by finite differences; Monte Carlo NEES/NIS consistency
- [x] GNSS lock-out recovery (navcore/recovery)
- [x] Real-data baseline on verified validation drives (+ all verified non-test drives)
- [x] Phase 3 GNSS-timing error found and corrected (9 s sample-and-hold, not 4.1 s latency)
- [ ] Accelerometer scale factor / axis misalignment calibration
- [ ] Magnetometer hard/soft-iron fit and disturbance rejection
- [ ] Allan-variance PLOTS (numbers exist; figures not produced)
- [ ] Per-segment re-alignment for a non-rigid phone mount -> Phase 9 (NHC depends on it)

---

## Phase 5 — ML Training Pipeline (redefined; was AHRS) — DONE

- [x] Windowed datasets, gates, CAN / coordinate guards, augmentation, model registry,
      Huber + NLL, trainer with CUDA-OOM fallback (see PROJECT_STATUS.md Phase 5)
- [ ] AHRS work of the original plan (Madgwick/Mahony/magnetometer) — not scheduled; attitude
      is the EKF + gravity levelling (reduced IMU)

---

## Phase 6 — ML Evaluation & Export (redefined; the INS was delivered in Phase 4) — DONE

- [x] CUDA torch; sweep TCN/1D-CNN/GRU/LSTM × 2 s/5 s; selection rule on VAL
- [x] Evaluation incl. calibration; TEST once; ONNX + model cards; parity gate PASS
- [ ] **Train to convergence** (best epochs were 10–11 of 12)
- [ ] Model A: +2.8 m/s bias on test drivers; shrinks toward the mean at high speed
- [ ] Model B: ≈ constant and over-confident on test — retrain with a better target or drop it

---

## Phase 7 — Sensor Fusion & NHC (redefined; merges the two planned AI phases) — DONE

- [x] ONNX wrapper, AI-assisted EKF, NHC, GNSS state machine, integration tests
- [x] Real-data ablation (PROJECT_STATUS.md §7.5)
- [x] Defaults agreed with the owner: AI speed ON; NHC only together with AI speed; Model B OFF
- [ ] Find why AI speed worsens some 30 s outages (S3a, S3b)

---

## Phase 8 — Fusion Engine (Error-State EKF)

**Goal: statistically defensible fusion with honest uncertainty.**
Most of this was delivered in Phase 4 (filter) and Phase 7 (AI measurements); the outage
simulation came with executed Phase 9.

- [x] Error-state EKF: position, velocity, attitude, IMU-bias states (15-state, Phase 4)
- [x] Process noise from Phase 4 Allan variance (`reports/phase4/imu_noise.json`)
- [x] Measurement noise from Model A's variance head (Phase 7)
- [~] GNSS updates gated on reported accuracy (state machine) and the NIS gate;
      **satellite count not used**
- [~] AI speed measurement model (Phase 7); **no displacement model**
- [x] ZUPT on detected stationarity (Phase 4; phone-only stillness since Phase 7)
- [x] Innovation gating and outlier rejection (NIS gate)
- [x] **NIS / NEES consistency tests** — Monte Carlo, synthetic (Phase 4)
- [ ] Real-data consistency: the shipped default's 2σ covers the error only 75 % of the time
      on VAL (Phase 7 §7.5) — the filter is overconfident on real data
- [x] Covariance symmetry and positive-definiteness assertions (Joseph form, Cholesky check)
- [x] GNSS-outage simulation over recorded fixes, enforced by the replay harness
      (`simulation/outage`, `simulation/replay`; executed Phase 9)
- [~] Outage-duration vs error: tables vs the Phase 4 baseline at 30 / 60 s (Phase 7) and a
      10-300 s battery (executed Phase 9); **no plotted curves**

---

## Phase 9 — Non-Holonomic Constraints (NHC)

**Goal: exploit vehicle kinematics — a car cannot slide sideways or fly.**
The constraint itself was delivered in Phase 7 (`navcore.nhc`).

- [x] Body-frame lateral-velocity pseudo-measurement (≈ 0) — **untested on real data**: no
      validation drive has an observable mount
- [x] Body-frame vertical-velocity pseudo-measurement (≈ 0)
- [ ] Bicycle-model yaw-rate / speed consistency coupling
- [x] Couple with Phase 4 mounting misalignment (lateral row only with an accepted mount)
- [~] Constraint relaxation: skipped below 2 m/s and above 3 m/s² lateral acceleration;
      **no wheel-slip or rough-terrain handling**
- [x] Ablation: fusion with vs without NHC — NHC alone hurts and diverged once (Phase 7 §7.5)
- [ ] Validate relaxation logic against real turning and braking segments
- [ ] Correlated-error model (rate / σ) and a time-varying mount-tilt estimate
- [ ] Covariance robustness at a GNSS reset

---

## Phase 10 — Map Matching (delivered as executed Phase 8)

**Goal: refinement only. Map matching never originates a position.**

- [x] Fetch the OSM road network for the Coventry, UK operating area (`scripts/download/fetch_osm_roads.py`)
- [x] Same for any field-test area: `--city` or `--bbox` (`scripts/export/export_road_bundle.py`)
- [x] Build a spatial index over road geometry (50 m grid)
- [x] HMM map matching: emission from filter covariance, transition from topology
- [x] Off-network detection — do not force-snap car parks and private roads
- [ ] Matched-heading feedback into fusion — **not done, by decision** (`docs/map_matching.md` §5:
      the correlation risk outweighs it)
- [x] Ablation with map matching disabled (VAL: marginal gain, only while drift < ~30 m)
- [x] Failure modes documented (`docs/map_matching.md`) and tested on synthetic junctions,
      parallel roads, dual carriageways, off-network travel
- [x] Positions remain valid and physics-derived with map matching removed (the matcher never
      writes to the EKF; tested)
- [ ] Gate the matched output on the fused σ; recalibrate `map_match_confidence` on TRAIN
- [ ] Measure the bundle's load time and memory on a phone

---

## Phase 11 — Evaluation Harness & Benchmarking

**Goal: one harness generates every number that ever gets published.**
The executed Phase 9 outage benchmark (`scripts/evaluate/phase9_outage_benchmark.py`) is the
closest existing piece.

- [ ] Metrics: ATE, RPE, error CDF (p50 / p68 / p95), final-position error
- [ ] Drift as a percentage of distance travelled
- [ ] Heading error over time
- [~] Outage battery: multiple durations and start points — exists for TRAIN+VAL drives
      (executed Phase 9); **not yet over the test routes**
- [~] GNSS masking enforced in the harness, not by convention — done in the replay engine;
      the harness itself does not exist yet
- [ ] **Per-stage ablation table:** INS → +AI → +Fusion → +NHC → +Map Matching
- [ ] Stamp every result with script, commit, split, seed, timestamp
- [ ] Single command regenerates all published numbers
- [ ] Verify reproducibility across two fixed-seed runs

---

## Phase 12 — Model Export & Python/Android Numerical Parity

**IN PROGRESS.** Python side done; Kotlin side written but **never run** (no JDK / Android
SDK on the authoring machine).

- [x] Export Phase 6 models to ONNX (with model cards: SHA-256 + feature order)
- [x] **Re-run each exported graph on the trainer's own inputs and compare numerically** (Phase 6 gate)
- [ ] Generate the shared constants file for both languages from one source
- [x] Golden-vector fixtures: geodesy, rotations, INS steps, filter updates, NHC, features,
      Model A, full pipeline, map matching (`tests/regression/golden/`)
- [~] Kotlin parity tests replaying the vectors — written, **never run**; they assert
      tolerances but do not print a max abs / rel deviation report
- [ ] Document the float vs double policy per module (only in code comments so far)
- [~] Parity suite wired into CI (`.github/workflows/android.yml`) — written, **not yet run**

---

## Phase 13 — Android Application

**BLOCKED on a build machine (JDK 17 + Android SDK) and a handset.** Source complete for
Part A (base & sensors) and Part B (ML & dead reckoning); **never compiled**.

- [x] Android project scaffolding (`:core`, `:sensors`, `:gnss`, `:app`)
- [x] Sensor acquisition using real hardware timestamps; honest rate handling (+ 10 Hz resampler)
- [~] INS, EKF and NHC ported against golden vectors; calibration and AHRS not ported as
      separate stages
- [x] On-device inference (ONNX Runtime, Model A, integrity-checked)
- [~] Foreground service written; **doze-mode behaviour not tested**
- [ ] **Measure** battery, CPU, and thermal behaviour — do not estimate
- [~] On-device map matching written; **no map tiles** (track canvas only)
- [ ] On-device parity harness: replay a golden recording, diff against Python
- [~] UI: position, σ and confidence, pipeline mode, GNSS status; **no uncertainty ellipse drawn**
- [ ] Verify full operation with GNSS disabled and no network
- [ ] Compiles; `./gradlew :core:test` and `./gradlew :app:assembleDebug` pass

---

## Phase 14 — End-to-End Validation, Documentation & Demo

- [ ] Teammate build and field test in Nagpur (`TESTING_GUIDE.md`), with a local road bundle
- [ ] Collect new field data: tunnels, underpasses, urban canyon, multi-storey car parks
- [ ] Log GNSS truth for **evaluation only**, never fed to outage-mode code
- [ ] Compare on-device results against the Phase 11 Python benchmarks
- [ ] Architecture document and algorithm specification
- [ ] Dataset card and model cards
- [ ] **Limitations document: where the system fails and why**
- [ ] Reproducibility guide: clean machine → published results
- [ ] SIH demo materials, every claim traceable to a re-runnable measurement

---

## Standing Rules (never tick these off — they always apply)

- No model ever outputs latitude or longitude.
- No number is written down that a script in this repo cannot regenerate.
- Units and frames are explicit in every symbol.
- Real `dt` always; never an assumed sample rate.
- Raw data is read-only.
- Failures, regressions, and limitations get reported as plainly as successes.
- Python and Android must agree within a stated tolerance; a parity failure blocks release.
