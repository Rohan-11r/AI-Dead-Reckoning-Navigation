# AGENTS.md — Core Directive (SIH26168)

**Project:** AI-ML based Intelligent Dead Reckoning system for seamless navigation
**Problem Statement ID:** SIH26168

---

## THE CORE DIRECTIVE

> **Hybrid Physics + AI architecture. Sensors -> Calibration -> AI Signal Processing -> INS -> Fusion -> NHC -> Map Matching. Never hallucinate latitude/longitude directly. Never fake data. Ensure strict Python vs Android numerical consistency.**

This sentence is the contract. Every commit, model, and line of code in this repository
must be defensible against it. If a change cannot be justified under the directive, it
does not land.

---

## 1. Mandated Pipeline Order

```
+-------------------------------------------------+
|                 RAW SENSORS                     |
|  accel, gyro, magnetometer, gravity, (GNSS)     |
+----------------------+--------------------------+
                       |
+----------------------v--------------------------+
|              CALIBRATION                        |
|  bias / scale-factor / axis-misalignment,       |
|  temperature & Allan-variance characterisation  |
+----------------------+--------------------------+
                       |
+----------------------v--------------------------+
|          AI SIGNAL PROCESSING                   |
|  learned denoising, bias regression, motion     |
|  context, speed / displacement estimation       |
|  --> OUTPUTS PHYSICAL QUANTITIES ONLY           |
+----------------------+--------------------------+
                       |
+----------------------v--------------------------+
|          INS (STRAPDOWN MECHANIZATION)          |
|  attitude -> velocity -> position, in a         |
|  well-defined local tangent (ENU) frame         |
+----------------------+--------------------------+
                       |
+----------------------v--------------------------+
|          FUSION (Error-State EKF)               |
|  INS + AI estimates + GNSS-when-valid + ZUPT    |
+----------------------+--------------------------+
                       |
+----------------------v--------------------------+
|     NHC (NON-HOLONOMIC CONSTRAINTS)             |
|  vehicle body-frame lateral/vertical velocity   |
|  ~ 0, wheelbase & yaw-rate kinematics           |
+----------------------+--------------------------+
                       |
+----------------------v--------------------------+
|          MAP MATCHING                           |
|  HMM snap of the fused trajectory to a road     |
|  network. Refinement ONLY -- never the source   |
|  of position.                                   |
+-------------------------------------------------+
```

**No stage may be skipped, reordered, or short-circuited.** In particular, no component
downstream of INS may write a position that was not produced by integrating physics.

---

## 2. Hard Prohibitions

### 2.1 Never hallucinate latitude/longitude directly

- **No model may have latitude or longitude as a regression target.** Not as a raw
  degree value, not as a normalised value, not as a delta-lat/delta-lon pair.
- Neural networks are permitted to output **only** physically-typed intermediate
  quantities with units, for example: `speed [m/s]`, `displacement magnitude [m]`,
  `heading change [rad]`, `accel bias [m/s^2]`, `gyro bias [rad/s]`,
  `motion-context class`, or a **variance / uncertainty** on any of these.
- Geodetic coordinates are produced by exactly one code path: the INS mechanization plus
  the fusion filter state, converted from the local tangent frame through a single
  audited geodesy module. Every lat/lon in any output must be traceable to an
  integration of accelerations/velocities from a known initial fix.
- A trajectory that "looks right" but was not integrated is a **failure**, not a result.

### 2.2 Never fake data

- No synthetic, mock, placeholder, or invented sensor samples, coordinates, or metrics
  outside of clearly-named unit-test fixtures whose synthetic nature is explicit in the
  filename and docstring (e.g. `tests/fixtures/synthetic_circle_imu.py`).
- No fabricated training runs, accuracy numbers, loss curves, or benchmark tables.
  A number that was not produced by a script in this repo that anyone can re-run does
  not get written down.
- Every reported metric must record: the exact script, the commit, the dataset split,
  the random seed, and the run timestamp.
- **Report failure honestly.** If drift is 8% of distance travelled, the document says
  8%. Degraded results are information; invented results are fraud.
- Missing / corrupt / NaN sensor rows are to be detected, counted, logged, and handled
  explicitly. They are never silently imputed or dropped without a recorded count.

### 2.3 Never leak ground truth

- GNSS ground truth is used for **initialisation**, **fusion when genuinely available
  and valid**, and **evaluation**. During a simulated GNSS outage, no component may
  read a GNSS field — enforced by the evaluation harness masking those columns, not by
  convention.
- Train / validation / test splits are by **route and driver**, never by random row, so
  that adjacent 10 Hz samples cannot appear on both sides of a split.

---

## 3. Physical Consistency Requirements

- **Units are explicit** in every variable name, docstring, and serialised schema
  (`speed_mps`, `yaw_rate_radps`, `alt_m`). SI throughout the core; conversion happens
  only at I/O boundaries.
- **Frames are explicit and named:** sensor/body frame (b), local tangent ENU/NED (n),
  ECEF (e), geodetic WGS84. Every transform is a named, tested function. No implicit
  frame changes.
- **Rotation conventions are declared once** (quaternion component order, rotation
  direction, Euler sequence) and asserted in tests. Mixed conventions are a defect class.
- **Time is monotonic and explicit.** Real, non-uniform sample intervals `dt` are used;
  a nominal rate is never assumed. Gaps, duplicates, and backwards timestamps are
  detected and reported.
- **Covariance must stay positive-definite.** The filter asserts symmetry and
  PD-ness; it does not quietly proceed on a broken `P`.
- **Kinematic sanity checks are mandatory:** no faster-than-physical accelerations, no
  negative speeds, no discontinuous jumps in position, gravity magnitude within
  tolerance of local `g`.

---

## 4. Python vs Android Numerical Consistency

The Android implementation is not a re-interpretation of the Python one. It is a
**port with a numerical contract.**

- **One specification, two implementations.** The algorithm spec (equations, constants,
  ordering of operations) is written down once and both implementations cite it.
- **Golden-vector parity tests.** Fixed input sequences are pushed through both stacks;
  outputs must agree to a documented tolerance. Targets:
  - deterministic scalar/vector math: `< 1e-6` relative
  - full-pipeline state after N steps: documented, bounded, regression-tested
- **Deterministic float behaviour.** No dependence on unspecified iteration order, no
  fast-math style reassociation, explicit `float` vs `double` choices recorded per
  module. Accumulators are `double` unless proven otherwise.
- **Shared constants file** (WGS84 `a`, `f`, `e^2`, `g`, conversion factors) generated
  from a single source of truth so Python and Kotlin/C++ cannot drift.
- **Exported models are verified, not assumed.** After ONNX / TFLite export, the
  exported graph is re-run on the same inputs as the trainer and compared numerically
  before it is allowed into the app.
- Any parity failure is a **release blocker**, not a known issue.

---

## 5. Engineering Standards

- **Honesty over optimism** in every status file, docstring, and report. "Not
  implemented" and "does not work yet" are acceptable and expected states.
- **Reproducibility:** seeds pinned, dependencies locked, data paths configured in one
  place, every result regenerable by a single documented command.
- **No silent fallbacks.** If calibration data is missing, raise. Do not substitute
  zeros and continue.
- **Tests before trust.** Every math module ships with unit tests that include
  analytically-known cases.
- **Raw data is read-only.** The dataset is never modified in place; all derived
  artefacts land in a separate, regenerable directory.

---

## 6. Review Checklist (apply to every change)

- [ ] Does any model output a coordinate? -> **reject**
- [ ] Is any number in a doc unbacked by a re-runnable script? -> **reject**
- [ ] Are units and frames explicit in every new symbol?
- [ ] Is real `dt` used rather than an assumed rate?
- [ ] Does the mandated pipeline order remain intact?
- [ ] Is GNSS genuinely unavailable to outage-mode code paths?
- [ ] Do Python and Android still agree within tolerance?
- [ ] Are failures and limitations stated plainly?
