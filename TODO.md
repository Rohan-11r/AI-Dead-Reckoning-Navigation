# TODO.md — High-Level Roadmap (SIH26168)

**AI-ML based Intelligent Dead Reckoning system for seamless navigation**

Read [`AGENTS.md`](./AGENTS.md) first — it governs everything below.
Detailed per-phase status lives in [`PROJECT_STATUS.md`](./PROJECT_STATUS.md).

---

## The Shape of the Work

```
Phase 0   Environment                         [COMPLETED]
Phase 1   Architecture & repo                  [COMPLETED]
Phase 2   Data                                [COMPLETED]
Phase 3   Geodesy  ------------------------------- + -- foundations
Phase 4   Calibration                              |
              |
Phase 5   AHRS          ------------------\
Phase 6   INS           ------------------ +-- physics core
              |
Phase 7   AI: bias + speed/context ---------- learned layer
              |
Phase 8   Fusion EKF    ------------------\
Phase 9   NHC           ------------------ +-- estimation
Phase 10  Map Matching  ------------------/
              |
Phase 11  Evaluation    -- the truth-teller
              |
Phase 12  Export + Parity ----------------\
Phase 13  Android       ------------------ +-- deployment
Phase 14  Field validation + docs --------/
```

Phases 3 and 3 can proceed in parallel once Phase 2 lands. Phases 7 and 7 can be trained
in parallel. Everything from Phase 12 on is gated by installing a JDK and the Android SDK.

---

## Immediate Next Actions (top of the stack)

1. **Decision needed (user):** may V-file CAN channels (wheel speed, CAN yaw rate) be model
   *inputs*, or only reference/evaluation truth? See `PROJECT_STATUS.md` §2.10.
2. **Phase 3:** single constants source (WGS84, g0, normal-gravity coefficients) generated
   for Python and Kotlin — replaces the audit-only copy in `training/preprocessing/iovnbd_audit.py`.
3. **Phase 3:** single dataset-path config module (carried over from Phase 2).
4. **Phase 4:** horizontal gyro axes; Allan variance on stationary sessions `Vw1`, `Vw15`.
5. **Android:** built externally / in CI in later phases (Android Studio not installed locally).
   This blocks Phases 12–14 and has a long lead time.
5. **Out of band:** confirm a physical Android handset is available for Phases 13–14.
   An emulator cannot generate real IMU data.
6. **Before Phase 7:** install `requirements-gpu.txt` and confirm
   `torch.cuda.is_available()`. Driver 555.97 vs a `cu126` build is expected to work but
   is untested.
7. **Housekeeping:** consider moving the workspace out of the OneDrive-synced tree, or
   excluding `.venv/` from sync — 42,926 files, 1.27 GB logical / 1.4 GB on disk.

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

Defects found and fixed in this phase (each would otherwise have surfaced in Phase 12):

- [x] `onnxscript` required by `torch.onnx.export` but undeclared by torch — now pinned
- [x] dynamo ONNX exporter crashes on its own `✅` under cp1252 — `PYTHONUTF8=1` set
- [x] `packages.find` cannot see through `package-dir`; silently found zero navcore
      packages — replaced with an explicit list, now asserted against the manifest

Not done (deliberately):

- [ ] No git commit made — committing was not requested

---

## Phase 2 — Dataset Ingestion, Schema Normalisation & Integrity Audit

**Goal: know exactly what is wrong with the data before any model sees it.**
**Status: COMPLETED 2026-09-27** — `reports/phase2_validation.txt`, 26/26 CRITICAL PASS.

- [x] Pinned venv + `requirements.txt` *(done in Phase 1)*
- [x] Python version decided on evidence *(done in Phase 1)*
- [x] `git init` + `.gitignore` *(done in Phase 1; initial commit made at Phase 2 start)*
- [x] Raw layout normalised to `data/raw/IO-VNBD/` (rename only, inventory verified)
- [ ] Single config module holding the one dataset path — **not done**: path is a
      constant in each Phase 2 script + `.env.example`. Carry to Phase 3.
- [x] Loader: cp1252 + per-token mojibake repair, normalised column names, explicit datetime
- [x] Split `GPS SATELLITES IN RANGE` into used/visible; 2,118 Excel-damaged cells repaired
- [x] Convert km/h to m/s at the loader boundary; SI everywhere inside
- [x] Checksum `Categorised` vs `Uncategorised` — settled: 72 unique sessions
- [x] Per-recording integrity audit: rows, duration, real `dt` histogram, monotonicity,
      gaps, duplicates, per-column NaN counts
- [x] GPS quality profile: accuracy, satellite counts, fix update interval
- [~] Kinematic plausibility screen: gravity vs g0 and local g, sensor ranges, GPS speed vs
      VBOX speed. **GPS speed vs differentiated GPS position not done.**
- [~] Stationary rows detected (CAN + wheel + VBOX speed all zero) for the audit;
      **not yet exported as reusable segments** for Phase 4 / Phase 8
- [x] Parquet with an explicit versioned schema (`iovnbd-v1`, units in metadata)
- [x] **Route/driver-level** split manifest `data/splits/iovnbd_split_v1.json`
- [x] Leak check: no session, route group, or test driver spans two splits
- [x] Dataset card: `reports/phase2/dataset_report.html` (+ `.json`)
- [x] Loader unit tests (22)

**New items found by Phase 2 (carried forward):**
- [ ] Phase 4: resolve the two horizontal gyro axes (vertical = column 2 `gyro_y`, proven)
- [ ] Phase 4: Allan variance from stationary-only sessions `Vw1` (34 min), `Vw15`
- [ ] Phase 4: wheel-speed unit (km/h vs rad/s label) and odometry scale vs VBOX speed
- [ ] Phase 7: per-session S/V time-alignment refinement for the 47 unverified sessions
      before any V-file channel is used as a 10 Hz training target
- [ ] Phase 8: model phone-GNSS latency (median 4.1 s behind VBOX) in the measurement model
- [ ] **Decision needed:** whether V-file CAN channels (wheel speed, yaw rate) may be
      model *inputs*, or only reference/evaluation truth — a phone-only deployment has
      no CAN bus

---

## Phase 3 — Geodesy & Reference-Frame Foundations

**Goal: the one audited code path all position output must flow through.**

- [ ] WGS84 constants from a single source of truth
- [ ] Constants generator emitting both Python and Kotlin/C++ (for Phase 12)
- [ ] Geodetic ↔ ECEF conversions
- [ ] ECEF ↔ local tangent ENU (and NED where needed)
- [ ] Geodesic distance for evaluation metrics
- [ ] Quaternion / rotation-matrix / Euler library
- [ ] **Declare rotation conventions once** and assert them in tests
- [ ] Round-trip tests to sub-millimetre over the dataset's operating region
- [ ] Analytically-known test cases (equator, poles, prime meridian, known baselines)

---

## Phase 4 — Sensor Characterisation & Calibration

**Goal: know each sensor's real error behaviour before correcting it.**

- [ ] Stationary-segment extraction from real recordings
- [ ] Accelerometer bias, scale factor, axis misalignment
- [ ] Gyroscope bias and bias stability
- [ ] **Allan variance** analysis → noise density and random-walk parameters
- [ ] Feed those parameters into the Phase 8 process noise (no hand-tuning)
- [ ] Magnetometer hard-iron / soft-iron ellipsoid fit
- [ ] Magnetic-disturbance detection and rejection
- [ ] **Empirically determine the Gyroscope Yaw/Pitch/Roll → body-axis mapping**
- [ ] **Empirically confirm whether ACCELEROMETER includes gravity**
- [ ] Device-to-vehicle mounting misalignment estimation
- [ ] Calibration raises on missing inputs — no zero-filled fallbacks
- [ ] Allan-variance plots and a calibration report committed

---

## Phase 5 — Attitude Estimation (AHRS)

**Goal: orientation accuracy, the biggest single lever on drift.**

- [ ] Raw gyro-integration baseline; quantify its drift honestly
- [ ] Complementary filter
- [ ] Madgwick and Mahony filters
- [ ] Quaternion EKF with gyro-bias states
- [ ] Gravity-vector levelling for roll/pitch
- [ ] Magnetometer heading with disturbance gating
- [ ] Static and in-motion initial alignment
- [ ] Validate against the dataset `ORIENTATION` channels **and** GPS-derived heading,
      stating the limitations of each reference
- [ ] Report heading drift rate per method; choose by measurement, not preference

---

## Phase 6 — Strapdown INS Mechanization

**Goal: the physics core. Position by integration only.**

- [ ] Strapdown equations in the local tangent frame
- [ ] **Real non-uniform `dt`** throughout — no assumed 10 Hz
- [ ] Gravity model; assess Coriolis and transport-rate relevance at vehicle speeds
- [ ] Compare integration schemes (trapezoidal, RK, coning/sculling compensation)
- [ ] Verify against analytically-known synthetic motion (clearly labelled fixtures)
- [ ] **Pure-INS drift baseline** over 10 s / 30 s / 60 s / 300 s outages
- [ ] Assert no GNSS field is read anywhere in the INS path

---

## Phase 7 — AI Signal Processing (Denoising / Bias + Motion Context / Speed)

**Goal: learn what physics cannot — residual sensor error, and a direct, NON-INTEGRATING
speed estimate. No coordinates, ever.**

Merged from the original Phases 6 and 7: one dataset, one training harness, one export path.

Stage A — denoising / residual bias:
- [ ] Windowed dataset builder from the Phase 2 splits
- [ ] Baseline: classical filtering, for an honest comparison point
- [ ] Small sequence models (1-D CNN / TCN / GRU) sized for 4 GB VRAM and phone inference
- [ ] Targets: accel/gyro bias and noise corrections in SI units
- [ ] Ablation vs Phase 4 analytic calibration — report it even if AI loses

Stage B — motion context:
- [ ] Classifier: stationary / accelerating / cruising / braking / turning
- [ ] Exploit the dataset's own route categorisation as additional labels
- [ ] Operating point chosen on a cost-weighted curve; report false-`stationary` rate

Stage C — speed / displacement / heading change:
- [ ] **Speed regression** from IMU windows (the drift-arresting output)
- [ ] **Displacement-magnitude regression** over windows
- [ ] Heading-change regression as an independent check on Phase 5
- [ ] Audit target quality: gate on GNSS accuracy, satellites, minimum speed for bearing;
      report the exclusion count

Common:
- [ ] **Uncertainty / variance heads** on every output so Phase 8 can weight them
- [ ] Heteroscedastic NLL; validate variance calibration separately from RMSE
- [ ] **Evaluate on held-out drivers**, not just held-out routes
- [ ] Pinned seeds; two fixed-seed runs must agree
- [ ] Record parameter count, inference latency, model size
- [ ] Install `requirements-gpu.txt` and confirm CUDA actually works (untested as of Phase 1)
- [ ] Static check: no coordinate appears as a target anywhere in the training code

---

## Phase 8 — Fusion Engine (Error-State EKF)

**Goal: statistically defensible fusion with honest uncertainty.**

- [ ] Error-state EKF: position, velocity, attitude, IMU-bias states
- [ ] Process noise from Phase 4 Allan variance
- [ ] Measurement noise from the Phase 7–7 variance heads
- [ ] GNSS position/velocity updates gated on accuracy and satellite count
- [ ] AI speed / displacement measurement models
- [ ] ZUPT on detected stationarity
- [ ] Innovation gating and outlier rejection
- [ ] **NIS / NEES consistency tests** — a filter that is accurate but inconsistent is broken
- [ ] Covariance symmetry and positive-definiteness assertions
- [ ] GNSS-outage simulation enforced by harness-level column masking
- [ ] Outage-duration vs error curves against the Phase 6 baseline

---

## Phase 9 — Non-Holonomic Constraints (NHC)

**Goal: exploit vehicle kinematics — a car cannot slide sideways or fly.**

- [ ] Body-frame lateral-velocity pseudo-measurement (≈ 0)
- [ ] Body-frame vertical-velocity pseudo-measurement (≈ 0)
- [ ] Bicycle-model yaw-rate / speed consistency coupling
- [ ] Couple with Phase 4 mounting misalignment (a wrong body frame makes NHC harmful)
- [ ] Constraint relaxation on wheel slip, sharp manoeuvres, rough terrain
- [ ] Ablation: fusion with vs without NHC — report cases where it hurts
- [ ] Validate relaxation logic against real turning and braking segments

---

## Phase 10 — Map Matching

**Goal: refinement only. Map matching never originates a position.**

- [ ] Fetch OSM road network for the Coventry, UK operating area
- [ ] Build a spatial index over road geometry
- [ ] HMM map matching: emission from filter covariance, transition from topology
- [ ] Off-network detection — do not force-snap car parks and private roads
- [ ] Optional matched-heading feedback into fusion, with the correlation risk stated
- [ ] Ablation with map matching disabled
- [ ] Document failure modes: parallel roads, complex junctions, off-network travel
- [ ] Assert positions remain valid and physics-derived with map matching removed

---

## Phase 11 — Evaluation Harness & Benchmarking

**Goal: one harness generates every number that ever gets published.**

- [ ] Metrics: ATE, RPE, error CDF (p50 / p68 / p95), final-position error
- [ ] Drift as a percentage of distance travelled
- [ ] Heading error over time
- [ ] Outage battery: multiple durations, multiple start points, across all test routes
- [ ] GNSS masking enforced in the harness, not by convention
- [ ] **Per-stage ablation table:** INS → +AI → +Fusion → +NHC → +Map Matching
- [ ] Stamp every result with script, commit, split, seed, timestamp
- [ ] Single command regenerates all published numbers
- [ ] Verify reproducibility across two fixed-seed runs

---

## Phase 12 — Model Export & Python/Android Numerical Parity

**BLOCKED until a JDK and the Android SDK are installed.**

- [ ] Export Phase 7–7 models to ONNX and/or TFLite
- [ ] **Re-run each exported graph on the trainer's own inputs and compare numerically**
- [ ] Generate the shared constants file for both languages from one source
- [ ] Golden-vector fixtures: geodesy, rotations, INS steps, filter updates, NHC, full pipeline
- [ ] Cross-language parity runner reporting max absolute and relative deviation
- [ ] Document the float vs double policy per module
- [ ] Wire the parity suite into CI; any failure blocks release

---

## Phase 13 — Android Application

**BLOCKED until JDK + Android SDK + Gradle are installed and a handset is confirmed.**

- [ ] Android project scaffolding
- [ ] Sensor acquisition using real hardware timestamps; honest rate handling
- [ ] Port calibration, AHRS, INS, EKF, NHC — each validated against golden vectors
- [ ] On-device inference (ONNX Runtime Mobile / TFLite / NNAPI)
- [ ] Background-service lifecycle and doze-mode behaviour
- [ ] **Measure** battery, CPU, and thermal behaviour — do not estimate
- [ ] Offline map tiles and on-device map matching
- [ ] On-device parity harness: replay a golden recording, diff against Python
- [ ] UI: position, uncertainty ellipse, current pipeline mode, GNSS status
- [ ] Verify full operation with GNSS disabled and no network

---

## Phase 14 — End-to-End Validation, Documentation & Demo

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
