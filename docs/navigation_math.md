# Navigation Mathematics — SIH26168

**This document is the specification.** Per [`AGENTS.md`](../AGENTS.md) §4 there is one
spec and two implementations (`navigation-core/` in Python, `edge-engine/` on-device).
Both cite this file. If they disagree with each other, this file decides; if either
disagrees with *this file*, that implementation is the bug.

Every convention below is **declared, not assumed**. §11 states how each one is enforced
by a test rather than by good intentions.

---

## 1. Notation and Frames

| Symbol | Frame | Definition |
| --- | --- | --- |
| `b` | body / sensor | Fixed to the IMU. Axis mapping to the vehicle is **unknown until Phase 4** and must be derived from data. |
| `v` | vehicle | x forward, y left, z up. Related to `b` by the mounting misalignment. |
| `n` | navigation (local tangent) | **ENU**: x East, y North, z Up. Origin at a fixed reference fix per run. |
| `e` | ECEF | Earth-centred, Earth-fixed, rotating with the Earth. |
| — | geodetic | WGS84 latitude `φ`, longitude `λ`, ellipsoidal height `h`. |

Conventions used throughout:

- `R_b^n` rotates a vector **from** `b` **into** `n`: `x^n = R_b^n x^b`.
- `[a]×` is the skew-symmetric matrix with `[a]× u = a × u`.
- Superscripts name the frame a quantity is *expressed in*; subscripts name what it
  describes. `ω_ib^b` is the rate of `b` w.r.t. inertial, expressed in `b`.
- **All angles in radians, all distances in metres, all times in seconds, internally.**
  Degrees and km/h exist only at I/O boundaries (`AGENTS.md` §3).

> **ENU, not NED.** Chosen because the vertical channel is the weakest and keeping "up"
> positive makes sign errors in the gravity term visible rather than plausible. The
> choice is arbitrary but it is made **once**, here, and asserted in tests.

---

## 2. WGS84 Constants

Single source of truth. Generated into both Python and the on-device language; never
hand-copied (`AGENTS.md` §4).

| Constant | Symbol | Value |
| --- | --- | --- |
| Semi-major axis | `a` | `6378137.0` m (exact by definition) |
| Flattening | `f` | `1 / 298.257223563` |
| Semi-minor axis | `b = a(1−f)` | `6356752.314245` m |
| First eccentricity squared | `e² = 2f − f²` | `6.69437999014e-3` |
| Second eccentricity squared | `e′² = e²/(1−e²)` | `6.73949674228e-3` |
| Earth rotation rate | `ω_ie` | `7.292115e-5` rad/s |
| Gravitational parameter | `GM` | `3.986004418e14` m³/s² |
| Normal gravity at equator | `γ_e` | `9.7803253359` m/s² |
| Somigliana coefficient | `k` | `1.93185265241e-3` |
| Standard gravity | `g₀` | `9.80665` m/s² (a *defined constant*, not local gravity — see §5) |

Radii of curvature at latitude `φ`:

```
Meridian:        M(φ) = a(1 − e²) / (1 − e² sin²φ)^(3/2)
Prime vertical:  N(φ) = a / (1 − e² sin²φ)^(1/2)
```

---

## 3. Coordinate Conversions

### 3.1 Geodetic → ECEF (closed form)

```
x^e = (N(φ) + h) cosφ cosλ
y^e = (N(φ) + h) cosφ sinλ
z^e = (N(φ)(1 − e²) + h) sinφ
```

### 3.2 ECEF → Geodetic

No closed form for `φ` in elementary functions. Two admissible implementations:

- **Bowring's method** — one trigonometric iteration, sub-millimetre for terrestrial
  heights. Cheap enough for the on-device path.
- **Ferrari / Vermeille closed-form** — exact, no iteration, more operations.

Whichever is used, it must be the **same** in both implementations, and the round-trip
`geodetic → ECEF → geodetic` must close to better than 1e-4 m over the operating region.

### 3.3 ECEF → Local Tangent ENU

Given reference `(φ₀, λ₀, h₀)` with ECEF position `p₀^e`, for a point `p^e`:

```
Δ = p^e − p₀^e

            ⎡ −sinλ₀            cosλ₀           0     ⎤
R_e^n  =    ⎢ −sinφ₀ cosλ₀     −sinφ₀ sinλ₀     cosφ₀ ⎥        (ENU row order)
            ⎣  cosφ₀ cosλ₀      cosφ₀ sinλ₀     sinφ₀ ⎦

p^n = R_e^n Δ  =  [East, North, Up]ᵀ
```

`R_e^n` is orthonormal, so the inverse is its transpose. **The reference origin is fixed
per run and recorded with the output.** A drifting origin silently corrupts every
position; tests assert the origin is immutable once set.

> **This is the only path to latitude and longitude.** `AGENTS.md` §2.1: output
> coordinates are produced by inverting §3.3 and §3.2 on the *integrated* navigation
> state. Nothing else in the codebase may construct a coordinate.

### 3.4 Distance for Evaluation

Trajectory error metrics use geodesic distance on the ellipsoid (Vincenty or Karney),
**not** the ENU Euclidean distance, so that evaluation does not inherit the tangent-plane
approximation it is meant to be measuring. `pyproj` serves as the independent cross-check
oracle here; it is never in the runtime position path.

---

## 4. Attitude

### 4.1 Quaternion Convention — declared once

- Storage order **`q = [w, x, y, z]`**, scalar first.
- `q` represents `R_b^n`, i.e. it rotates body vectors into the navigation frame.
- Hamilton product (not JPL). `q₁ ⊗ q₂` composes rotations, applying `q₂` first.
- Unit norm maintained; renormalise every step.

Storage order and the Hamilton/JPL choice are the two most common sources of silent
cross-implementation error. Both are asserted by tests (§11), and the on-device code
must state its convention in a comment citing this section.

### 4.2 Kinematics

```
q̇_b^n = ½ q_b^n ⊗ [0, ω_nb^b]
```

where `ω_nb^b = ω_ib^b − R_n^b (ω_ie^n + ω_en^n)` is the body rate relative to the
navigation frame — the gyroscope measures `ω_ib^b`, relative to *inertial* space, and the
Earth-rate and transport-rate terms must be removed. §6 quantifies what happens if they
are not.

### 4.3 Discrete Propagation

For a rotation increment `Δθ = ω_nb^b Δt` with `θ = ‖Δθ‖`:

```
                  ⎡ cos(θ/2)                  ⎤
Δq  =             ⎢ (Δθ/θ) sin(θ/2)           ⎥ ,        q_{k+1} = q_k ⊗ Δq
                  ⎣                           ⎦

θ → 0:   Δq ≈ [1, Δθ/2]  followed by renormalisation
```

The small-angle branch is required: `Δθ/θ` is `0/0` at rest, and a vehicle is stationary
for a large fraction of any real recording. The branch threshold is a **shared constant**,
because a different threshold in the two implementations is a parity failure.

Higher-order **coning and sculling** compensation is evaluated in Phase 6. At 10 Hz with
vehicle-scale rotation rates the effect is expected to be small, but "expected" is not a
measurement, so it gets measured.

### 4.4 Rotation Matrix from Quaternion

```
R_b^n = (w² − vᵀv) I + 2 v vᵀ + 2w [v]×        where q = [w, v], v = [x,y,z]ᵀ
```

---

## 5. Gravity

### 5.1 Normal Gravity (Somigliana)

```
γ(φ)   = γ_e (1 + k sin²φ) / √(1 − e² sin²φ)
γ(φ,h) ≈ γ(φ) − 3.086e-6 · h              [m/s²]   (free-air correction)
```

### 5.2 A real discrepancy at the dataset's location — worth being explicit about

At the first IO-VNBD fix (`φ = 52.402565°`, `h = 144.59 m`):

```
sin²φ            = 0.627766
γ(φ)             = 9.7803253359 × 1.001212752 / 0.997896535   = 9.812827 m/s²
free-air         = −3.086e-6 × 144.59                         = −0.000446 m/s²
γ(φ,h)           ≈ 9.812381 m/s²                       ← local normal gravity
```

> **Correction (Phase 2, 2026-09-27).** The Phase 1 version of this block had
> `sin²φ = 0.627722` — a hand-arithmetic slip — giving `9.812377`. The value above is
> recomputed and is pinned by `tests/unit/test_iovnbd_loader.py::test_normal_gravity_reproduces_doc_value`.
> The difference (4.2e-6 m/s²) changes none of the conclusions below.

But the dataset's own `GRAVITY` channel, on that same row
(`0.0089, −0.0009, 9.8066`), has magnitude:

```
‖g_dataset‖ = 9.806600 m/s²                            ← ≈ the DEFINED constant g₀
```

The Android `TYPE_GRAVITY` virtual sensor is reporting a vector normalised to standard
gravity `9.80665`, **not** local normal gravity. The difference is:

```
9.812381 − 9.806600 = 0.005781 m/s²
```

That is small, and it is *not* negligible in dead reckoning. As an unmodelled vertical
acceleration over a 60 s outage:

```
½ × 0.005781 × 60²  ≈  10.4 m
```

**Consequence for the implementation.** Phase 6 must use the §5.1 model for `g^n` and
must *not* treat the dataset `GRAVITY` channel as ground-truth gravity. The channel is
useful as an attitude-levelling reference (its *direction* is informative) but not as a
magnitude reference. This also means the accelerometer's gravity-inclusive/exclusive
question (Phase 0 §0.4 hazard 7) has to be settled from the data before any integration.

*Scope of this measurement:* the dataset figure above is from **one row of one file**.
**Phase 2 result (all 72 unique sessions, `reports/phase2/dataset_report.json`):**
confirmed. The pooled `‖GRAVITY‖` is `9.806596 m/s²`, the per-session means span
`9.806592–9.806622`, and every session is closer to `g₀` than to its own local normal
gravity (`9.81205–9.81312` over the dataset's latitudes and heights). The stationary
accelerometer norm is `9.863 m/s²`, so the accelerometer channel is gravity-inclusive
(hazard 7 settled).

### 5.3 Specific force at rest, for calibration

A stationary, level IMU measures `f^b = −R_n^b g^n`, i.e. it reads `+g` on the axis
opposing gravity. On the same sample row the accelerometer magnitude is:

```
‖f‖ = √(0.9684² + 0.0227² + 9.9665²) = 10.0135 m/s²
```

against `‖g‖ = 9.8066`. The residual `0.207 m/s²` is **not** attributable to bias: GPS
speed on that row is 5.84 km/h, so the vehicle was moving, and the residual mixes true
acceleration, engine vibration, accelerometer bias and scale error. Separating them is
exactly the job of Phase 4, using detected stationary segments. Recorded here to prevent
a later temptation to call it "the bias".

---

## 6. Earth Rotation and Transport Rate — magnitudes, not hand-waving

A common shortcut is to drop Earth-rate and Coriolis terms for "low-dynamics ground
vehicles". At the scale of a multi-minute GNSS outage this is wrong. At `φ = 52.4°`:

```
ω_ie^n = ω_ie [0, cosφ, sinφ]ᵀ                   (ENU: no East component)
       = 7.292115e-5 × [0, 0.610, 0.792]ᵀ  rad/s

ω_en^n ≈ [ −v_N / (M + h),  v_E / (N + h),  ... ]  ≈ v / R_earth
```

**Coriolis, at 30 m/s (108 km/h):**

```
‖2 ω_ie × v‖  ≤  2 × 7.292115e-5 × 30  =  4.375e-3 m/s²
position error over 60 s:  ½ × 4.375e-3 × 60²  ≈  7.9 m
```

**Transport rate, at 30 m/s:**

```
‖ω_en‖ ≈ 30 / 6.37e6 = 4.71e-6 rad/s
‖ω_en × v‖ ≈ 1.41e-4 m/s²
position error over 60 s:  ½ × 1.41e-4 × 60²  ≈  0.25 m
```

**Earth rate omitted from attitude propagation:**

```
heading drift = ω_ie sinφ · t = 5.777e-5 rad/s  =  0.199° per minute
```

A heading error of that order, over ~900 m of travel in 60 s, puts several metres of
cross-track error on the trajectory.

**Verdict:** Coriolis and Earth-rate terms are **retained**. Transport rate is retained
because it costs almost nothing once Coriolis is already computed. Every number above is
arithmetic anyone can redo from the constants in §2.

---

## 7. Strapdown Mechanization

The physics core. `navigation-core/ins`. **No GNSS field may be read in this module.**

```
Attitude:   q̇_b^n = ½ q_b^n ⊗ [0, ω_ib^b − R_n^b(ω_ie^n + ω_en^n)]

Velocity:   v̇^n   = R_b^n f^b − (2 ω_ie^n + ω_en^n) × v^n + g^n

Position:   ṗ^n   = v^n
```

with `f^b` the calibrated specific force and `g^n = [0, 0, −γ(φ,h)]ᵀ` in ENU.

Geodetic position is updated either by integrating `ṗ^n` in the tangent frame and
converting once (§3.3), or directly in curvilinear form:

```
φ̇ = v_N / (M(φ) + h)
λ̇ = v_E / ((N(φ) + h) cosφ)
ḣ = v_U
```

**Both forms are implemented and cross-checked against each other** — they should agree
to numerical precision over an outage-length window, and a disagreement is a bug in one
of them. Cheap, and it catches whole classes of frame errors.

### 7.1 Integration and timing

- `Δt` comes from the sample timestamps. **Never a nominal rate.** IO-VNBD is nominally
  10 Hz, and the Phase 2 audit exists precisely to find where it is not.
- Velocity and position use trapezoidal integration at minimum; Phase 6 compares schemes
  on real data rather than asserting one is better.
- Accumulators are `double`. The float/double policy per module is recorded and is part
  of the parity contract.

---

## 8. Error-State EKF

`navigation-core/fusion`. The filter estimates *errors* in the INS solution, which keeps
the linearisation valid because the errors stay small — an INS that must be linearised
about a bad state is already lost.

### 8.1 State

```
δx = [ δp^n(3), δv^n(3), ψ(3), δb_a(3), δb_g(3) ]ᵀ  ∈ ℝ¹⁵
```

### 8.2 Attitude-error convention — declared once

```
R̂_b^n = (I − [ψ]×) R_b^n
```

`ψ` is a small rotation of the *estimated* navigation frame relative to the true one.
Every Jacobian below follows from this choice. Swapping it flips signs throughout, which
is why it is declared here and enforced in §11.

### 8.3 Continuous error dynamics

```
δṗ^n  =  δv^n
δv̇^n  =  [f^n]× ψ  −  R_b^n δb_a  −  (2ω_ie^n + ω_en^n) × δv^n  +  w_a
ψ̇     =  −[ω_in^n]× ψ  −  R_b^n δb_g  +  w_g
δḃ_a  =  −(1/τ_a) δb_a  +  w_ba
δḃ_g  =  −(1/τ_g) δb_g  +  w_bg
```

with `f^n = R_b^n f^b`. Biases are first-order Gauss-Markov; `τ` and the driving noise
come from the Phase 4 **Allan variance**, not from tuning.

> **These signs are not taken on trust.** §11 requires `F` to be validated against a
> finite-difference Jacobian of the actual propagation function. A sign error in a
> document is a typo; a sign error that survives into the filter is a silent 50 %
> accuracy loss, so the test, not the document, is the guarantee.

### 8.4 Process noise

```
Q_k = ∫₀^Δt Φ(τ) G Q_c Gᵀ Φ(τ)ᵀ dτ        (Van Loan, or a first-order approximation)
```

`Q_c` is diagonal, populated from Allan-variance parameters: angle random walk, velocity
random walk, and bias instability per axis.

### 8.5 Measurement models

| Update | Available when | Innovation `z` | Jacobian `H` |
| --- | --- | --- | --- |
| GNSS position | fix valid, gates passed | `p_gnss − p̂` | `[I₃ 0 0 0 0]` |
| GNSS velocity | fix valid, speed reliable | `v_gnss − v̂` | `[0 I₃ 0 0 0]` |
| AI speed (Phase 7) | always | `v_AI − ‖v̂^n‖` | `[0  (v̂^n)ᵀ/‖v̂^n‖  0 0 0]` |
| ZUPT | stationarity detected | `0 − v̂^n` | `[0 I₃ 0 0 0]` |
| NHC lateral/vertical | moving, no slip | see §9 | see §9 |

The AI-speed Jacobian is singular at `‖v̂^n‖ = 0`; that branch is guarded and the update
is skipped in favour of ZUPT, which is the correct measurement at rest anyway.

**GNSS gating.** A fix is used only if reported accuracy, satellite count, and the
innovation itself all pass. Urban-canyon multipath produces fixes that are confidently
wrong, and an ungated filter will happily believe them.

### 8.6 Correction and numerical guards

```
K   = P Hᵀ (H P Hᵀ + R)⁻¹
δx̂  = K (z − H δx̂⁻)
P⁺  = (I − K H) P (I − K H)ᵀ + K R Kᵀ          ← Joseph form, mandatory
```

Joseph form is required, not optional: the naive `(I − KH)P` loses symmetry and drifts
non-positive-definite over the tens of thousands of updates in a single recording.

After every update:

- symmetrise: `P ← ½(P + Pᵀ)`
- assert positive-definiteness (Cholesky succeeds); **raise on failure, never continue**
- apply the attitude correction multiplicatively via `R_b^n ← (I + [ψ]×) R̂_b^n` followed
  by renormalisation, then **zero** `δp, δv, ψ` — they have been absorbed into the
  nominal state. Biases persist.

### 8.7 Consistency, not just accuracy

```
NIS_k  = νₖᵀ (H P Hᵀ + R)⁻¹ νₖ       ~ χ²(dim z)     — per-update
NEES_k = δxₖᵀ P⁻¹ δxₖ                ~ χ²(dim δx)    — needs truth, so evaluation-only
```

A filter that reports 2 m of uncertainty while making 20 m of error is broken even if
20 m happens to beat the baseline. `navigation-core/diagnostics` computes both; Phase 8's
exit criteria require consistency, not merely low error.

---

## 9. Non-Holonomic Constraints

`navigation-core/nhc`. Free information from the fact that this is a road vehicle.

Body-frame velocity, and its perturbation under the §8.2 convention:

```
v^b   = R_n^b v^n
δv^b  = R_n^b δv^n  −  R_n^b [v^n]× ψ
```

Two pseudo-measurements, with `S = [[0,1,0],[0,0,1]]` selecting the lateral and vertical
body axes:

```
z_NHC = −S v̂^b  (target: zero)

H_NHC = S [ 0₃   R_n^b   −R_n^b [v^n]×   0₃   0₃ ]
```

The `ψ` coupling is the important part. Lateral velocity depends on *attitude* as much as
on velocity, which is what lets NHC observe heading error — and equally what makes it
destructive if the body frame is wrong. **NHC is only enabled once the Phase 4
device-to-vehicle mounting alignment is available**, never with an assumed mounting.

Bicycle-model coupling provides a second relation:

```
ω_z ≈ v_x^b · tan(δ_steer) / L        ⇒        ‖v^b‖ ≈ ω_z · r_turn
```

usable as a consistency check between the gyroscope and the speed estimate even with `L`
and `δ_steer` unknown.

**Relaxation.** `R_NHC` is inflated or the update skipped on detected wheel slip, hard
cornering beyond a lateral-acceleration threshold, and rough terrain. The constraint
`v_y^b = 0` is an idealisation; asserting it during a skid injects error. Phase 9's
ablation must report the cases where NHC *hurts*, not only where it helps.

---

## 10. Zero-Velocity Update

Stationarity is detected from a combination of accelerometer variance, gyroscope
magnitude, and the AI motion-context classifier — not from GNSS speed, which is
unavailable in the mode where ZUPT matters most.

While stationary:

```
v^n = 0   (measurement, §8.5)
```

This is the single most effective drift arrest in urban driving, where a vehicle spends a
large fraction of its time at traffic lights. It also makes accelerometer bias directly
observable, which is why stationary segments feed Phase 4.

False positives are costly — clamping velocity to zero while creeping forward in traffic
loses real distance — so the detector's threshold is calibrated against labelled
stationary segments from the real data and its false-positive rate is reported.

---

## 11. How Every Convention Above Is Enforced

Declarations in a document drift. These are the tests that make them binding
(`tests/navigation/`, `tests/regression/`):

| Convention | Test |
| --- | --- |
| Quaternion order `[w,x,y,z]`, Hamilton, `R_b^n` | Rotate known vectors through known rotations; assert against hand-computed results. A JPL or `[x,y,z,w]` implementation fails. |
| ENU axis order | Assert a pure +East displacement appears in component 0 and moves longitude, not latitude. |
| Geodetic ↔ ECEF ↔ ENU | Round-trip closure < 1e-4 m; cross-check against `pyproj` as an independent oracle. |
| Error-state Jacobian `F` | **Compare analytic `F` against a finite-difference Jacobian** of the propagation function. Catches every sign error in §8.3. |
| Measurement Jacobians `H` | Same finite-difference treatment for GNSS, AI-speed, ZUPT and NHC. |
| Covariance stays PD | Cholesky assertion after every update; failure raises. |
| Gravity model | Reproduce the §5.2 arithmetic to 1e-6; assert local `γ ≠ g₀`, so nobody can quietly substitute the constant. |
| Earth-rate terms present | Run mechanization with and without; assert the difference matches §6 to within an order of magnitude. A silently-dropped term fails. |
| Small-angle branch threshold | Shared constant; assert both implementations use the same value. |
| INS reads no GNSS | Static check over `navigation-core/ins` plus a runtime test with GNSS columns masked. |
| No model outputs a coordinate | Static check over `training/models` for latitude/longitude targets (`AGENTS.md` §2.1). |
| Python ↔ on-device parity | Golden vectors: deterministic math `< 1e-6` relative; full-pipeline state bounded and regression-tracked. |

---

## 12. Open Mathematical Questions

Honest list of what is **not yet decided**, to be settled by measurement in the named
phase — not by picking the conventional answer now:

1. **Body-axis mapping.** *Partly settled in Phase 2.* The `Yaw/Pitch/Roll` labels are
   positional aliases of `X/Y/Z` (the Uncategorised copies of the same files say X/Y/Z).
   Against the vehicle's CAN yaw rate, the rotation about the vertical appears on the
   **second** gyro column (`gyro_y`, sign +, gain 0.997, 21 of 22 strongly-correlated
   sessions), while accelerometer/gravity **Z** is vertical in all 72 sessions. The gyro
   columns are therefore not in the accelerometer's axis order, and the label "Yaw" on
   column 1 is wrong. The two horizontal gyro axes are unresolved (the smoothed GRAVITY
   channel is too quiet to identify them). → Phase 4.
2. **Accelerometer gravity inclusion.** *Settled in Phase 2:* gravity-inclusive
   (stationary norm 9.863 m/s² over 103,449 rows). See §5.2.
3. **ECEF → geodetic method.** Bowring vs Ferrari, chosen on measured on-device cost
   against the accuracy requirement. → Phase 3.
4. **Integration scheme.** Trapezoidal vs higher-order, and whether coning/sculling
   compensation is measurable at 10 Hz. → Phase 6.
5. **Vertical channel.** Whether to estimate height at all, constrain it via NHC, or damp
   it with a barometer/road-surface assumption. Unaided vertical INS is divergent. → Phase 8.
6. **Attitude-error parameterisation.** 3-component `ψ` vs a full quaternion error state,
   over long outages. → Phase 8.
7. **Map-matching feedback.** Whether matched heading may re-enter fusion, given the
   correlation it introduces between the filter and its own refinement. → Phase 10.
