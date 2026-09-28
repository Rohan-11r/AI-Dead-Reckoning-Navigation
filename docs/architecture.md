# Architecture — SIH26168 Hybrid Physics + AI Dead Reckoning

**Governed by [`AGENTS.md`](../AGENTS.md).** Where this document and the core directive
appear to disagree, the directive wins and this document is the bug.

---

## 1. The Problem

GNSS fails predictably: tunnels, underpasses, multi-storey car parks, urban canyons,
dense foliage, deliberate jamming. A consumer smartphone IMU integrated open-loop drifts
quadratically in position — a bias of 0.01 m/s² unmodelled for 60 s produces
`½ · 0.01 · 60² = 18 m` of error, and real MEMS biases are larger than that.

So the useful question is not "can we integrate an IMU" (we can, badly) but **what
additional information can bound the drift**. This architecture supplies three kinds:

1. **Physics we know for free** — a road vehicle cannot slide sideways or leave the road
   surface (NHC), gravity has a known magnitude and direction, and stationarity is
   detectable (ZUPT).
2. **Structure learned from data** — MEMS error behaviour and the mapping from IMU
   signatures to *speed* is learnable. This is where AI earns its place.
3. **The road network** — position is not free in ℝ², it lies on a graph.

---

## 2. The Hybrid ML / Physics Pipeline

```
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  SENSORS                                                                     │
   │  accelerometer [m/s²] · gyroscope [rad/s] · magnetometer [µT]                 │
   │  gravity [m/s²] · GNSS (lat, lon, alt, speed, accuracy, sats) · time [ms]     │
   │  navigation-core/sensors                                                      │
   └────────────────────────────────────┬─────────────────────────────────────────┘
                                        │  raw samples, real non-uniform dt
   ┌────────────────────────────────────▼─────────────────────────────────────────┐
   │  CALIBRATION                                          [ P H Y S I C S ]      │
   │  bias · scale factor · axis misalignment · hard/soft-iron · Allan variance     │
   │  device→vehicle mounting alignment                                            │
   │  navigation-core/calibration · navigation-core/alignment                      │
   └────────────────────────────────────┬─────────────────────────────────────────┘
                                        │  corrected IMU + noise parameters (→ Q)
   ┌────────────────────────────────────▼─────────────────────────────────────────┐
   │  AI SIGNAL PROCESSING                                      [ A I ]           │
   │                                                                               │
   │   Stage A  denoise / residual bias regression                                 │
   │            → δb_a [m/s²], δb_g [rad/s], + variance                            │
   │   Stage B  motion context classification                                      │
   │            → {stationary, accelerating, cruising, braking, turning}            │
   │   Stage C  speed / displacement / heading-change regression                    │
   │            → v [m/s], Δs [m], Δψ [rad], + variance                            │
   │                                                                               │
   │   ⚠ OUTPUTS ARE PHYSICAL QUANTITIES WITH UNITS AND UNCERTAINTIES.             │
   │   ⚠ NO MODEL EVER OUTPUTS A COORDINATE.  (AGENTS.md §2.1)                     │
   │   training/models  ·  models/exported                                         │
   └────────────────────────────────────┬─────────────────────────────────────────┘
                                        │  corrections + pseudo-measurements + σ²
   ┌────────────────────────────────────▼─────────────────────────────────────────┐
   │  INS — STRAPDOWN MECHANIZATION                        [ P H Y S I C S ]      │
   │  attitude:  q̇ = ½ q ⊗ ω                                                      │
   │  velocity:  v̇ⁿ = Rᵇₙ fᵇ − (2Ωᵢₑ + Ωₑₙ) vⁿ + gⁿ                               │
   │  position:  ṗⁿ = vⁿ                                                           │
   │  navigation-core/imu · navigation-core/ins                                     │
   └────────────────────────────────────┬─────────────────────────────────────────┘
                                        │  nominal trajectory + Jacobians
   ┌────────────────────────────────────▼─────────────────────────────────────────┐
   │  FUSION — ERROR-STATE EKF (15 states)                 [ P H Y S I C S ]      │
   │  δx = [δp, δv, δψ, bₐ, b_g]                                                   │
   │  measurements: GNSS (gated on accuracy/sats) · AI speed · ZUPT                │
   │  Q from Allan variance · R from AI variance heads                             │
   │  innovation gating · NIS/NEES consistency · covariance PD guard                │
   │  navigation-core/fusion · navigation-core/state · navigation-core/diagnostics  │
   └────────────────────────────────────┬─────────────────────────────────────────┘
                                        │  fused state + covariance
   ┌────────────────────────────────────▼─────────────────────────────────────────┐
   │  NHC — NON-HOLONOMIC CONSTRAINTS                      [ P H Y S I C S ]      │
   │  vᵇ_y ≈ 0 (no sideslip) · vᵇ_z ≈ 0 (stays on the surface)                     │
   │  bicycle-model yaw-rate / speed coupling                                      │
   │  relaxed on slip, sharp manoeuvre, rough terrain                              │
   │  navigation-core/nhc                                                          │
   └────────────────────────────────────┬─────────────────────────────────────────┘
                                        │  kinematically admissible state
   ┌────────────────────────────────────▼─────────────────────────────────────────┐
   │  MAP MATCHING — HMM + Viterbi                    [ R E F I N E M E N T ]     │
   │  emission from filter covariance · transition from route topology             │
   │  off-network detection (car parks are not roads)                              │
   │  ⚠ NEVER ORIGINATES A POSITION. Remove it and the output is still valid.      │
   │  navigation-core/map_matching                                                 │
   └────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
   ┌────────────────────────────────────▼─────────────────────────────────────────┐
   │  OUTPUT                                                                       │
   │  (lat, lon, alt) + covariance + mode + provenance                             │
   │  ── produced ONLY by navigation-core/geometry from the integrated state ──     │
   └──────────────────────────────────────────────────────────────────────────────┘
```

### Why this order, and not another

- **Calibration before AI.** If the network has to learn the deterministic part of the
  error (constant bias, scale factor) it wastes capacity on something a least-squares fit
  solves exactly, and it will not transfer to a different handset. Physics first leaves
  the network only the genuinely hard residual.
- **AI before INS.** The learned corrections and pseudo-measurements have to exist before
  integration consumes them. AI feeds the physics; it does not post-process its output.
- **Fusion before NHC.** NHC is expressed in the *body* frame, so it needs an attitude
  estimate to apply. Applying it to a badly-aligned body frame actively injects error —
  which is why NHC is coupled to the Phase 4 mounting-alignment estimate.
- **Map matching last, and severable.** It is the only stage that uses information
  outside the vehicle. Keeping it last and optional is what lets us prove the position
  is physics-derived: delete the stage, and the system still produces a valid trajectory.

### Where the uncertainty flows

The architecture is only defensible if uncertainty is carried, not discarded:

```
Allan variance (Phase 4) ────────────────► Q   process noise
AI variance heads (Phase 7) ──────────► R   measurement noise, per-sample
GNSS accuracy + satellite count ─────────► R   gated, rejected when implausible
filter covariance P ─────────────────────► map-matching emission probability
                   └───────────────────────► reported output uncertainty
```

A point estimate with no covariance is not an answer to a navigation problem.

### As built (Phase 7): what the shipped fusion actually runs

The diagram above is the design space. What `navcore.fusion.navigator.FusionConfig()` runs
by default was chosen by a **real-data ablation**, not by preference
(`reports/phase7/fusion_evaluation.json`, 3 validation drives, pooled outages):

| Component | Default | Evidence |
| --- | --- | --- |
| Model A — forward speed (1D-CNN, 5 s) as an EKF measurement when GNSS is DEGRADED / LOST / RECOVERING | **ON** | 60 s outage error 2,814 → 490 m; 30 s p90 1,123 → 520 m |
| NHC (vertical; lateral only with an accepted mount) | **ON, only paired with AI speed** | alone: worse and diverged on 1 drive (mount-tilt error, correlated violations); with AI speed: best 30 s p90 (380 m) and 60 s median (364 m) |
| Model B — longitudinal acceleration (LSTM, 2 s) correcting the IMU before `predict()` | **OFF** | see below |
| GNSS state machine (GOOD / DEGRADED / LOST / RECOVERING) | ON | neutral alone; it is what switches AI speed on |

**Model B was fully trained, selected, exported, parity-validated and evaluated, and is
disabled by default because it made real-world navigation worse.** Adding it on top of AI
speed + NHC raised the aided error from 12.5 to 37.9 m and the 60 s outage error from 364 to
633 m, and dropped the filter's 2σ coverage to 63 %. That is consistent with its offline
evaluation: 4 % better than a constant on the held-out drivers, and over-confident there
(z std 1.52). The inverse-variance blend trusts it more than it deserves. It remains in the
code behind `use_ai_accel=True` for research, and is re-enabled only on new evidence
(a guard test, `test_default_fusion_config_is_the_measured_one`, pins these defaults).

---

## 3. Two Implementations, One Specification

`AGENTS.md` §4 forbids the Android app from being a reinterpretation of the Python code.

```
                   docs/navigation_math.md
              (equations, constants, operation order)
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
  navigation-core/                        edge-engine/src/
  Python REFERENCE                        portable on-device core
  readable, tested, slow                  deterministic, fast
         │                                       │
         │        ┌──────────────────┐           │
         └───────►│  GOLDEN VECTORS  │◄──────────┘
                  │ tests/regression │
                  └────────┬─────────┘
                           │  max |Δ| must stay within tolerance
                           ▼
                  parity failure = release blocker
                           │
                           ▼
                    android-app/  (Kotlin UI over edge-engine via JNI)
```

Shared constants (WGS84 `a`, `f`, `e²`, gravity coefficients, unit conversions) are
**generated** from one source into both languages, so they cannot drift by hand-copying.

Exported models get the same treatment: an ONNX graph is not trusted because export
returned success. It is re-run on the trainer's own inputs and compared numerically
before it is allowed into `models/exported/`.

---

## 4. Repository Map

| Path | Contains | Notes |
| --- | --- | --- |
| `navigation-core/` | Python reference pipeline | imports as `navcore.*` |
| `navigation-core/geometry/` | geodesy + rotations | **the only path to lat/lon** |
| `navigation-core/ins/` | strapdown mechanization | must never read a GNSS field |
| `navigation-core/diagnostics/` | NIS/NEES, covariance PD, kinematic sanity | the honesty enforcement layer |
| `training/` | datasets, models, losses, trainers, export | models output physical quantities only |
| `edge-engine/` | portable on-device engine | second implementation |
| `android-app/` | Android application (Kotlin, Gradle KTS: `:core` JVM, `:sensors`, `:gnss`, `:app`) | Phase 13 (Part A base & sensors, Part B ML & dead reckoning; parity tests under Phase 12): source complete, not yet compiled (no JDK/SDK on the authoring machine) |
| `simulation/outage/` | GNSS outage injection | masks columns, does not set a flag |
| `simulation/synthetic/` | analytically-known motion | test fixtures, explicitly labelled |
| `evaluation/` | ATE, RPE, drift %, ablations | sole source of published numbers |
| `data/raw/` | source recordings | **read-only** |
| `data/splits/` | route/driver split manifests | never random-row |
| `models/metadata/` | model cards | inputs, outputs, units, provenance |
| `tests/regression/` | golden vectors | the parity contract |
| `scripts/validate/` | phase gates | re-runnable invariant checks |

### The `navcore` import mapping

`navigation-core` is not a legal Python identifier, so `pyproject.toml` maps the
directory onto the `navcore` namespace:

```
navigation-core/ins/mechanization.py   →   import navcore.ins.mechanization
navigation-core/geometry/wgs84.py      →   import navcore.geometry.wgs84
```

This keeps the requested monorepo directory name while giving generic subdirectory names
(`common`, `state`, `filtering`) a namespace instead of letting them collide at top level.

---

## 5. Runtime Modes

The system is not one algorithm; it is a state machine over GNSS availability.
`navigation-core/outage` owns the transitions, `navigation-core/recovery` owns re-entry.

| Mode | Condition | Behaviour |
| --- | --- | --- |
| `GNSS_GOOD` | fix valid, accuracy and satellite count within gates | full fusion; IMU biases observable and actively estimated |
| `GNSS_DEGRADED` | fix present but accuracy poor or inconsistent | inflate R, tighten innovation gates, lean on AI + NHC |
| `DEAD_RECKONING` | no usable fix | INS + AI speed + NHC + ZUPT only; **no GNSS field is read** |
| `RECOVERING` | fix returns after an outage | reject the first fixes until consistent; correct without a position discontinuity |

Bias estimates learned during `GNSS_GOOD` are what make `DEAD_RECKONING` survivable —
the observable period funds the unobservable one. Mode is part of the output, so a
consumer always knows which regime produced a position.

---

## 6. Toolchain Decision (Phase 2, decided on evidence)

**CPython 3.14.4**, with 3.13.5 as a verified fallback.

Phase 0 deferred this rather than guessing. The deciding evidence, from a live PyPI
query on 2026-09-27 for `win_amd64` wheels:

| Package | cp313 | cp314 | Resolution |
| --- | --- | --- | --- |
| numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1 | native | native | either works |
| torch 2.14.0 | native | native | either works |
| onnxruntime 1.30.0 | native | native | either works |
| onnx 1.23.0 | via `cp312-abi3` | via `cp312-abi3` | identical on both |
| shapely, pyproj, pyarrow, matplotlib, numba | native | native | either works |
| geopandas, osmnx, networkx, pytest, allantools | pure Python | pure Python | either works |

Coverage is equivalent, so the tie broke on 3.14.4 already being the default interpreter
on PATH — no PATH juggling, one fewer moving part. Recorded as a *decision*, not a
preference: if any dependency later fails on 3.14, `requires-python = ">=3.13,<3.15"`
permits the fallback without a structural change.

**PyTorch is installed CPU-only.** The Windows PyPI wheel is 118 MB and CPU-only; the
CUDA build is ~2.5 GB. Phases 3–6 involve no training whatsoever, so the CUDA swap is
deferred to `requirements-gpu.txt` and Phase 7. Note the honest caveat recorded there:
the installed driver is 555.97 (CUDA 12.5) and the available build is `cu126`. That is
*expected* to work via CUDA minor-version compatibility but **has not been tested on this
machine**, and must be confirmed before Phase 7 rather than assumed.

### Hardware envelope

The RTX 2050 has **4 GB of VRAM**. This bounds the AI stages to small sequence models
(1-D CNN / TCN / GRU, single-digit millions of parameters) over windowed 10 Hz data.
That is the same bound on-device inference imposes, so the two constraints agree — the
GPU limit is not a compromise, it is the deployment requirement arriving early.

---

## 7. Data Flow

```
IO-VNBD (outside repo, READ-ONLY)
  288 CSV · 10 Hz · cp1252-encoded · Coventry UK
        │
        │  training/preprocessing  — cp1252 decode, column normalisation,
        │                            km/h→m/s, satellite-string split,
        │                            explicit datetime format, integrity audit
        ▼
  data/interim/        per-recording normalised frames + audit report
        │
        │  schema versioning, SI units throughout
        ▼
  data/processed/      Parquet, analysis-ready
        │
        │  split by ROUTE and DRIVER — never by random row
        ▼
  data/splits/         train / val / test manifests (tracked, reviewable)
        │
        ├──► training/datasets  → windowed tensors → training/models
        │                                                  │
        │                                            models/exported/
        │                                                  │
        └──► simulation/replay ──► navigation-core ◄────────┘
                                         │
                                  evaluation/  → metrics, ablations, reports
```

A random row split would put samples 100 ms apart on both sides of the boundary and
report a result that is nearly meaningless. Splitting by route *and* driver is what makes
the held-out numbers mean something (`AGENTS.md` §2.3).

---

## 8. Known Constraints and Honest Gaps

| Item | Status |
| --- | --- |
| JDK, Android SDK, Gradle, adb | **absent on this machine** — blocks Phases 12–14 |
| Physical Android handset | unconfirmed — an emulator cannot produce real IMU data |
| CUDA runtime verification | *Resolved in Phase 6:* `torch 2.14.0+cu126`, `cuda.is_available()` True on the RTX 2050 |
| `Categorised` vs `Uncategorised` duplication | unresolved; Phase 2 settles it by checksum |
| Gyro `Yaw/Pitch/Roll` → body-axis mapping | **unknown**; must be derived from data in Phase 4, never assumed |
| Accelerometer gravity-inclusive? | indicated by one sample row; to be confirmed in Phase 2 |
| Dataset `GRAVITY` magnitude vs local normal gravity | discrepancy noted — see `navigation_math.md` §5 |
| Wheel-odometry channel | *Decided (Phase 2, by the user):* IO-VNBD's CAN channels (wheel speed, steering, yaw rate) are **ground truth only**, never a model or filter input. The problem statement requires velocity without OBD-II; enforced in code (`training/preprocessing/columns.py`). |

Every row above is a thing we do not yet know. None of them will be filled in with a
plausible-sounding guess.

---

## 9. References

Listed for traceability. **Citations are to be verified against the actual publications
before they appear in any submitted report** — they are recorded here from prior
knowledge, which `AGENTS.md` §2.2 does not accept as a source.

- Onyekpe, Palade, Kanarachos et al. — *IO-VNBD: Inertial and odometry benchmark dataset
  for ground vehicle positioning* (the dataset used here).
- Onyekpe et al. — *WhONet: Wheel Odometry neural Network for vehicular localisation in
  GNSS-deprived environments* (prior art on the same dataset; a baseline to compare against).
- Newson & Krumm — *Hidden Markov Map Matching Through Noise and Sparseness* (the HMM
  map-matching formulation in `map_matching.md`).
- Groves — *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*
  (strapdown mechanization and error-state formulation conventions).
- Titterton & Weston — *Strapdown Inertial Navigation Technology*.
- El-Sheimy, Hou & Niu — Allan-variance analysis of MEMS inertial sensors.
