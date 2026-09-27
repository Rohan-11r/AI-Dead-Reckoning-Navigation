# ML Pipeline — SIH26168

**Governed by [`AGENTS.md`](../AGENTS.md).** The two rules that shape every design choice
in this document:

> **§2.1 — No model may have latitude or longitude as a regression target.**
> **§2.2 — No fabricated metrics. A number nobody can regenerate does not get written down.**

The AI in this system is not a trajectory predictor. It is a **sensor-error model and a
speed estimator**. Position comes from physics (`docs/navigation_math.md`).

---

## 1. Why AI at All

A fair question, given that the physics stack is complete without it. The honest answer
is that three specific things resist analytic modelling:

| Problem | Why physics alone struggles | What the model provides |
| --- | --- | --- |
| **Residual IMU error** | Bias is not constant. It drifts with temperature, vibration and time in a device- and duty-cycle-specific way. A constant-bias fit removes the easy part and leaves structured residual. | `δb_a`, `δb_g` per window, + variance |
| **Speed from an IMU** | Speed requires integrating acceleration, which integrates the bias too. There is no analytic escape from this. | `v [m/s]` directly from the IMU signature, which **does not integrate** and therefore does not drift |
| **Motion regime** | Constraint validity is regime-dependent — ZUPT needs stationarity, NHC breaks under slip. | motion-context class + confidence |

The second row is the load-bearing one. A *direct, non-integrating* speed estimate turns
unbounded quadratic position drift into bounded linear drift. That is the mechanism by
which this architecture works, and it is why speed regression gets the most attention.

**What AI does NOT do here:** predict coordinates, predict trajectories, replace the
filter, or decide where the vehicle is. If a proposed model would do any of those, it is
rejected at review (`AGENTS.md` §6).

---

## 2. Data Flow

```
IO-VNBD  288 CSV · 10 Hz · cp1252 · Coventry UK · READ-ONLY, outside the repo
    │
    │  training/preprocessing        (Phase 2)
    │  • cp1252 decode (UTF-8 raises)
    │  • normalise irregular column names
    │  • "18 / 19" -> (sats_used, sats_visible)
    │  • explicit datetime "%Y-%m-%d %H:%M:%S:%f"
    │  • km/h -> m/s at the boundary; SI inside
    │  • integrity audit: real dt histogram, gaps, duplicates, NaN counts
    │  • kinematic screen: ‖gravity‖, sensor ranges, GPS speed vs differentiated position
    ▼
data/interim/        per-recording normalised frames + audit report
    │
    │  schema versioning
    ▼
data/processed/      Parquet, SI units, explicit schema
    │
    │  split by ROUTE and DRIVER          <-- never by random row
    ▼
data/splits/         train / val / test manifests (tracked in git, reviewable)
    │
    ├─► training/datasets     windowing -> (B, C, W) tensors
    ├─► training/features     derived features, physical units only
    ├─► training/models       Stage A / B / C
    ├─► training/losses       incl. heteroscedastic NLL
    ├─► training/trainers     seeded, checkpointed, reproducible
    └─► training/export       ONNX + numerical verification of the export
                                     │
                              models/exported/   (only after parity passes)
                                     │
                                     ▼
                              navcore.fusion consumes the outputs as measurements
```

---

## 3. Splitting — the thing most easily got wrong

At 10 Hz, consecutive samples are 100 ms apart and almost identical. A random row split
therefore puts near-duplicate samples in train and test, and reports a number that
measures interpolation rather than generalisation. Published dead-reckoning results have
been inflated this way.

**The rule:** split by **route**, and hold out **drivers** entirely.

| Split | Contents | Answers |
| --- | --- | --- |
| `train` | subset of routes from a subset of drivers | — |
| `val` | different routes, same drivers | "does it generalise to new roads?" |
| `test` | different routes **and** held-out drivers | "does it generalise to a new person and vehicle?" |

The `test` split is the only one that supports a claim about deployment. Driver E has
several route groups (`Vf`, `Vta`, `Vtb`, `Vw`) and is a natural held-out driver candidate;
the final assignment is decided in Phase 2 from the actual per-recording durations so the
splits are balanced, and recorded in a tracked manifest.

**Leak checks, enforced as tests:**

- no recording ID appears in two splits
- no driver appears in both `train` and `test`
- no window crosses a recording boundary
- window stride is large enough that train and val windows do not overlap in time
- normalisation statistics are computed on `train` **only**, then frozen

That last one is a classic quiet leak: fitting a scaler on the full dataset leaks test
distribution into training. Statistics live in `models/normalization/` and are shipped to
Android as part of the parity contract — a mismatch there breaks inference silently rather
than loudly, which is why it is version-pinned with the model.

---

## 4. Windowing

| Parameter | Value | Reason |
| --- | --- | --- |
| Sample rate | 10 Hz (verified) | dataset native; real `dt` retained per sample, never assumed |
| Window length | 1–5 s (10–50 samples), swept | long enough to contain a manoeuvre, short enough to bound latency |
| Stride (train) | to be set in Phase 7 | balances sample count against redundancy |
| Stride (eval) | = window length | non-overlapping, so metrics are not inflated by correlated windows |
| Channels | accel (3), gyro (3), + optional mag (3), gravity (3) | magnetometer gated on disturbance detection |

Windows carry their real timestamps. A window spanning a data gap is either rejected or
flagged — never silently treated as contiguous. Phase 2's `dt` histogram is what tells us
how often that happens.

---

## 5. Models

Sized by two constraints that agree: **4 GB VRAM** on the RTX 2050 and **phone inference**.
Small 1-D convolutional / recurrent architectures, single-digit millions of parameters.
No large transformers — not for principle, but because neither constraint permits one and
the data volume (288 recordings) would not support one.

### Stage A — Denoise / residual bias regression (Phase 7)

```
in   : (B, 6, W)   calibrated accel + gyro window
out  : δb_a (3) [m/s²], δb_g (3) [rad/s], log σ² (6)
arch : 1-D CNN / TCN, dilated, causal
```

Runs **after** analytic calibration, so its job is only the residual that least-squares
cannot reach. Supervised against GNSS-derived references where those are valid.

**Mandatory ablation:** Stage A versus Phase 4 analytic calibration alone. If the network
does not beat physics, that is the reported result and the physics is kept.

### Stage B — Motion context classification (Phase 7)

```
in   : (B, 6, W)
out  : P(class) over {stationary, accelerating, cruising, braking, turning}
arch : small 1-D CNN + GRU
```

Feeds ZUPT triggering and NHC relaxation. Errors here are asymmetric: a false
`stationary` clamps velocity while the vehicle creeps and permanently loses distance,
which is worse than a missed detection. The operating point is therefore chosen on the
**cost-weighted** curve, not on accuracy, and the false-positive rate is reported
explicitly.

### Stage C — Speed / displacement / heading-change regression (Phase 7)

```
in   : (B, 6, W)
out  : v [m/s], Δs [m], Δψ [rad], each with log σ²
arch : 1-D CNN + GRU, separate heads
```

The most important model in the system, per §1. Targets derive from GNSS: speed from the
GNSS speed channel (converted from km/h), displacement from geodesic distance between
consecutive valid fixes, heading change from successive GNSS bearings while moving.

**Target quality is audited, not assumed.** GNSS speed is itself noisy and its bearing is
meaningless at low speed. Windows whose targets fail a validity gate (accuracy, satellite
count, minimum speed for bearing) are excluded, and the exclusion count is reported.
Training against a bad label is a quiet way to fabricate a result.

### The coordinate prohibition, enforced mechanically

Every model above outputs quantities with **units**. The invariant is not left to
reviewer diligence:

- `tests/ml/` contains a static check over `training/models/` and the target builders for
  any latitude/longitude symbol appearing as a regression target
- every model card in `models/metadata/` declares each output's name, unit and physical
  range; a coordinate-ranged output fails the schema
- the fusion layer consumes these as *measurements with covariance* — there is no code
  path by which a model output becomes a position except through the EKF and
  `navcore.geometry`

---

## 6. Losses

| Target | Loss | Why |
| --- | --- | --- |
| Speed, displacement, Δψ, biases | **Gaussian NLL** (heteroscedastic) | The filter needs `σ²`, not just a point estimate. MSE gives no uncertainty, and an unweighted measurement corrupts the EKF. |
| Motion context | class-weighted cross-entropy | classes are strongly imbalanced (a lot of `cruising`) |
| Optional | Huber on the mean term | robustness to residual label outliers |

```
L_NLL = ½ [ (y − μ)² / σ²  +  log σ² ]
```

Predicting `log σ²` keeps `σ² > 0` without a constraint. This loss can collapse by
inflating `σ²` to excuse a bad mean, so **calibration is validated separately**: predicted
variance is checked against realised squared error across bins. An overconfident or
uniformly-underconfident model is a broken measurement source even at good RMSE, and this
check is part of Phase 7's exit criteria.

---

## 7. Reproducibility

`AGENTS.md` §2.2 — a number that cannot be regenerated does not exist.

Every run records: git commit, the config file, seed, split manifest hash, resolved
package versions, hardware, start/end timestamps, and full metrics. Determinism measures:

```python
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # benchmark=True picks kernels nondeterministically
```

Hyperparameters live in `training/configs/`, never in code. Two runs of a fixed seed must
produce the same metrics; that equality is itself a test. Where full determinism is
unattainable (some CUDA reductions), the *residual nondeterminism is measured and
reported* rather than described as "negligible".

---

## 8. Export and the Parity Contract

Export success is not evidence the export is correct. Verified in Phase 2 on this machine:

| Path | `max |torch − onnxruntime|` | Status |
| --- | --- | --- |
| `dynamo=True` (default, opset 18) | `1.490e-08` | works; **requires `PYTHONUTF8=1`** |
| `dynamo=False` (legacy TorchScript) | `1.863e-08` | works; deprecated since torch 2.9 |

Two real findings from that smoke test, both of which would otherwise have surfaced in
Phase 12:

1. **`onnxscript` is required** by `torch.onnx.export` in torch 2.14 but is **not** a
   declared torch dependency. It is now pinned in `requirements.txt`.
2. **The dynamo exporter crashes under the cp1252 console codepage** — it prints a `✅`
   and dies with `UnicodeEncodeError` inside its own progress logging. `PYTHONUTF8=1`
   fixes it and is set by the activation helpers in `scripts/setup/`.

The export pipeline (`training/export/`) must therefore:

1. export with a pinned opset and declared dynamic axes
2. `onnx.checker.check_model` the graph
3. **re-run the exported graph on the trainer's own inputs** and compare against the
   PyTorch outputs
4. write a model card to `models/metadata/` with units, ranges and provenance
5. only then place the graph in `models/exported/`

A graph that fails step 3 never reaches the app. Tolerances: `< 1e-5` absolute on this
smoke-test scale; the production tolerance is set per model from its output range and
recorded in the card, because `1e-5` on a speed in m/s and `1e-5` on a bias in m/s² are
not the same claim.

---

## 9. Evaluation — two distinct levels

Keeping these apart matters, because good model metrics do not imply good navigation.

**Model level** (`training/evaluation/`) — per-window: RMSE/MAE on speed, displacement,
heading change; classifier confusion matrix; variance calibration. Fast, used to select
architectures.

**Trajectory level** (`evaluation/`) — the numbers that get published: ATE, RPE, error
CDF percentiles (p50/p68/p95), final-position error, **drift as a percentage of distance
travelled**, heading error. Computed over real recordings with GNSS masked by the harness.

The required ablation table, reported in full including regressions:

| Configuration | What it isolates |
| --- | --- |
| Pure INS | the honest baseline all else must beat |
| + Stage A (denoise/bias) | value of learned error modelling |
| + Stage C (AI speed) | value of the non-integrating speed estimate |
| + Fusion (EKF) | value of statistical fusion |
| + NHC | value of vehicle kinematics |
| + Map matching | value of road-network refinement |

Plus the comparison that keeps us honest about the AI: **Phase 4 analytic calibration
alone versus Stage A**. If physics wins, the table says physics won.

---

## 10. Known Risks

| Risk | Mitigation | Residual |
| --- | --- | --- |
| Label noise in GNSS-derived targets | validity gating; report exclusion counts | irreducible; bounds achievable accuracy |
| Overfitting to 4 drivers / one region | driver-held-out test split; report per-driver spread | **real** — Coventry, 2019, specific handsets. Generalisation beyond that is unevidenced and will be stated as such. |
| Speed model failing at low speed | ZUPT covers stationary; report error vs speed | low-speed creep remains hardest |
| Variance heads mis-calibrated | explicit calibration check in exit criteria | — |
| Single-vehicle dynamics | NHC parameters kept explicit, not baked into weights | — |
| Dataset duplication (`Categorised` vs `Uncategorised`) | Phase 2 checksum before counting either as independent | **unresolved** — could halve the effective data |

The generalisation row is the one to state plainly in any report: this is trained and
tested on IO-VNBD, which is four drivers in one English city in 2019. Results are
evidence about that domain. Anything beyond it is a hypothesis.
