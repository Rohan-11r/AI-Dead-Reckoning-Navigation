# Map Matching — SIH26168

**Governed by [`AGENTS.md`](../AGENTS.md).** The constraint that defines this entire stage:

> **§1 — Map matching is REFINEMENT ONLY. It never originates a position.**
> **§2.1 — Every lat/lon must be traceable to an integration of physics.**

Module: `navigation-core/map_matching` (imports as `navcore.map_matching`). Phase 10.

---

## 0. As built (Phase 8) — read this first

Implemented in `navcore.map_matching` (Phase 8 of the revised roadmap; this document's
"Phase 10" references predate the renumbering):

| Design item | As built |
| --- | --- |
| Road network source | OSM XML cached once by `scripts/download/fetch_osm_roads.py` (Overpass, drivable classes, Coventry core; provenance + SHA-256 in `reports/phase8/osm_manifest.json`). **Not `osmnx`**: a stdlib parser, so no new dependency and no network code in the package (static test) |
| Graph | directed segments per travel direction; one-way / roundabout / motorway rules; U-turns only at nodes |
| Spatial index | uniform 50 m grid (not an R-tree); checked equal to brute force |
| Emission | Mahalanobis with the filter's 2×2 covariance + (5 m)² map error, **plus a heading term** (directed segment vs course) — §3.2 had no heading term |
| Transition | `-|route − travelled| / β`, route by bounded Dijkstra; no route within the cutoff ⇒ probability 0 |
| Decoding | online forward filter for the live output + windowed Viterbi `decode()` |
| `map_match_confidence` | filtered posterior mass within 10 m of the chosen point |
| Off-network | χ² gate; suspend after 3 implausible epochs, re-enter after 3 plausible |
| Feedback into the filter | **none** (§5) — tested: the matcher never writes to the EKF |
| β, covariance inflation | chosen on TRAIN drives (S1, S2, S4) as §3.3 requires |

**Measured result (VAL, `reports/phase8/map_matching_evaluation.json`): a marginal gain.**
The road network cannot identify the road once dead reckoning has drifted by ~80 m or more
in a dense street grid; see PROJECT_STATUS.md Phase 8 for the numbers.

## 1. What This Stage Is, and What It Is Not

**It is:** a way to exploit the fact that a road vehicle's position is not free in ℝ² — it
lies on a graph. Snapping a drifted trajectory onto the road network removes the component
of error perpendicular to the road, which is often most of it.

**It is not:** a position source. This distinction is architectural, not rhetorical.

| Approach | Verdict |
| --- | --- |
| Take the fused trajectory, find the most likely road path, report the *corrected* position | **Correct.** Refines a physics-derived estimate. |
| Detect that the car is "probably on the A45" and report a point on the A45 | **Forbidden.** The position now comes from the map, not from integration. Violates §2.1. |

### The severability test

> **Delete the map-matching stage entirely. The system must still produce a valid,
> physics-derived trajectory with honest uncertainty.**

This is an enforced test, not a design note (`tests/navigation/`). If removing map
matching breaks the output, then the map had become load-bearing and the architecture has
drifted into hallucinating position from a road graph. The Phase 10 ablation table
therefore always includes a map-matching-disabled row.

A corollary: **map matching cannot rescue a broken filter.** A trajectory snapped
confidently onto the wrong road is worse than an unsnapped one with honest covariance,
because it converts a known uncertainty into an unknown error.

---

## 2. Road Network

| Item | Choice |
| --- | --- |
| Source | OpenStreetMap, via `osmnx` (Phase 10, `requirements-geo.txt`) |
| Area | Coventry, UK and surroundings — the IO-VNBD operating region |
| Graph | directed; one-way restrictions respected |
| Retained tags | geometry, `highway` class, `oneway`, `maxspeed`, `name`, `bridge`, `tunnel` |
| Spatial index | R-tree over segment bounding boxes |
| Storage | cached locally; **offline capable** (the app must work without network) |

`tunnel=yes` is not decoration: tunnels are exactly where GNSS fails, so tunnel segments
are a strong prior on *where an outage is expected*. Using them to explain a loss of fix is
legitimate. Using them to place the vehicle is not.

**Honest caveat:** OSM is community-mapped and neither complete nor current, especially
for private roads, car parks and recent construction. Its errors are not zero-mean and can
be locally systematic. Map-matching error attributable to map error will be reported as
such rather than folded into the system's error budget.

---

## 3. HMM Formulation

Following the standard Newson & Krumm construction (see `architecture.md` §9; the citation
is to be verified against the publication before it appears in any report).

- **Hidden states** at time `t`: candidate road segments near the fused position, with the
  perpendicular projection of the estimate onto each.
- **Observations**: the fused position from `navcore.fusion` — *after* NHC, *before* any
  map influence.
- **Goal**: the maximum-likelihood *sequence* of segments, via Viterbi.

The sequence is the point. Snapping each fix independently to its nearest road produces
physically impossible paths that teleport between parallel carriageways. The HMM enforces
that consecutive states be connected by a plausible route.

### 3.1 Candidate generation

For each fix, retrieve segments whose bounding box intersects a search radius derived from
the **filter covariance**, not a fixed constant:

```
r_search = k · √(λ_max(P_pos))        k ≈ 3, clipped to [r_min, r_max]
```

This is the right coupling: after a long outage the filter reports large uncertainty, the
search widens, and the true road stays in the candidate set. A fixed radius either
excludes the truth when drift is large or admits noise when the fix is good.

### 3.2 Emission probability

How well does candidate `i` explain the observation?

```
p(z_t | c_t^i)  ∝  exp( −½ d_iᵀ P_pos⁻¹ d_i )
```

where `d_i` is the 2-D offset from the fused position to its projection on segment `i`.

Using the **actual filter covariance** `P_pos` rather than a scalar σ is what makes this
stage statistically coherent: an elongated uncertainty ellipse (typical after an outage —
along-track error grows faster than cross-track) correctly makes an along-track offset
cheap and a cross-track offset expensive. A scalar σ would throw that structure away.

### 3.3 Transition probability

Is the move from segment `i` to segment `j` consistent with how far the vehicle actually
travelled?

```
Δ_route = shortest-path distance on the graph from proj_i to proj_j
Δ_gc    = geodesic distance between the two observations

p(c_{t+1}^j | c_t^i)  ∝  exp( −|Δ_route − Δ_gc| / β )
```

A detour far longer than the straight-line movement is improbable; so is a transition
requiring the vehicle to pass through a segment it cannot reach. `β` is fitted on the
training split, not hand-tuned on test.

Additional gates:

- transitions violating `oneway` are rejected outright
- implied speed above the segment's plausible maximum is penalised
- `Δ_route = ∞` (no route exists) is rejected

### 3.4 Decoding

Viterbi in log-space (products underflow badly over thousands of steps):

```
δ_t(j) = max_i [ δ_{t−1}(i) + log p(c_t^j | c_{t−1}^i) ] + log p(z_t | c_t^j)
```

For real-time on-device use, a **windowed / online Viterbi** over a lag of a few seconds
is used instead of full-sequence decoding, accepting a small optimality loss for bounded
latency and memory. The offline Python reference implements both, and the difference
between them is measured rather than assumed negligible.

---

## 4. Off-Network Detection — the failure mode that matters most

Not all driving happens on mapped roads. IO-VNBD explicitly includes **multi-storey car
parks**, which are both unmapped and exactly where GNSS is worst.

Forcing a snap there is the single most damaging thing this stage can do: it injects
confident, structured error at the moment the filter is least able to detect it.

**Detection:** map matching is suspended when

- the best candidate's emission probability stays below a floor for several consecutive epochs
- no candidate route explains the observed displacement
- motion context (Phase 7) indicates sustained low-speed manoeuvring inconsistent with road travel
- the trajectory's own shape (tight repeated turns, spiral ramps) is not road-like

**Behaviour when suspended:** fall back to the pure fusion + NHC output and **say so** in
the output mode. Position quality degrades — honestly, and visibly to the consumer —
rather than looking confident and being wrong.

Re-entry requires several consecutive confident matches, to avoid flapping at car-park
exits.

---

## 5. Feedback Into Fusion — deliberately conservative

Matched heading could be fed back into the EKF as a heading measurement, and it is
tempting because road bearing is accurate where the match is right.

**The problem:** the map-matching emission model already consumed the filter's own
estimate. Feeding its output back creates a correlated loop — the filter would treat its
own prior, laundered through the map, as independent evidence. That produces artificially
shrinking covariance and an overconfident, self-reinforcing solution. It is exactly the
kind of error that looks like an accuracy improvement.

**Position:** feedback is **disabled by default**. Phase 10 may evaluate it as an explicit
experiment, but enabling it requires:

1. a NEES consistency check showing the filter is not overconfident with feedback on
2. an ablation quantifying the benefit against that risk
3. inflated `R` on the fed-back measurement reflecting the correlation
4. the coupling documented in the limitations section

Absent all four, the map refines the *output* and does not touch the *state*.

---

## 6. Evaluation

Reported by the Phase 11 harness like any other stage:

| Metric | Meaning |
| --- | --- |
| Match rate | fraction of epochs with a confident match |
| Correct-segment rate | against manually verified segments on a sample of routes |
| Cross-track error reduction | the component map matching is actually meant to remove |
| Along-track error change | may **worsen**; snapping does not fix distance travelled |
| Off-network detection rate | true/false positives on known car-park segments |
| Route-topology violations | should be zero by construction; non-zero is a bug |

The along-track row is the honest one. Map matching corrects position *perpendicular* to
the road well and does comparatively little for error *along* it — that is Stage C's job
(`ml_pipeline.md` §5). Reporting only total error would obscure which mechanism did the work.

### Expected failure modes, to be documented rather than hidden

| Situation | Failure |
| --- | --- |
| Parallel carriageways, service road beside a dual carriageway | wrong-but-adjacent match |
| Complex junctions, roundabouts with multiple exits | brief mis-assignment |
| Elevated road above a surface road | 2-D matching cannot separate them; needs altitude, which is the weakest channel |
| Unmapped or newly built roads | no candidate exists; must be detected as off-network, not forced |
| Car parks, private land | off-network; suspension must trigger |
| Long outage with large drift | candidate set may exclude the truth; widened by §3.1 but not unboundedly |

---

## 7. Implementation Notes

- **Geometry in metres.** Projections and distances are computed in the local tangent
  frame or a projected CRS, never in degrees — a degree of longitude at 52.4°N is ~61 % of
  a degree of latitude, and mixing them produces a quiet, latitude-dependent bias.
- **Final coordinates still come from `navcore.geometry`.** The matched position is a
  point in the tangent frame; converting it to lat/lon uses the same single audited path
  as everything else (`AGENTS.md` §2.1). No exceptions for this module.
- **Determinism.** Tie-breaking among equal-probability candidates must be deterministic
  (stable ordering by segment ID), or Python and the on-device engine will diverge on
  ambiguous input and fail parity for a reason that looks like a numerical bug.
- **Shared parameters.** `β`, the search-radius constant `k`, and all thresholds are
  generated into both implementations from one source, like every other constant.
- **Offline first.** No network call at inference time. The graph is pre-fetched and cached.

---

## 8. Open Questions for Phase 10

1. Full-sequence versus windowed Viterbi: what lag is needed, and what is the measured
   optimality loss?
2. Is altitude usable at all for elevated/surface disambiguation, given how weak the
   vertical channel is (`navigation_math.md` §12.5)?
3. Should `maxspeed` inform the transition model, given OSM's patchy coverage of it?
4. On-device graph representation and memory budget for the operating area.
5. Do off-network thresholds transfer across the car-park recordings, or do they need
   per-site tuning? If the latter, that is a limitation to state, not to tune away on test data.
