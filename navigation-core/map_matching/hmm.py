"""Online HMM map matcher (Newson & Krumm construction, docs/map_matching.md sec 3).

REFINEMENT ONLY (AGENTS.md sec 1): the matcher consumes the fused, physics-derived position
and its covariance, and returns a point ON A ROAD near it, with a confidence. It never
feeds back into the filter, and when no road plausibly explains the observation it returns
NO position (``matched=False``) rather than inventing one.

Hidden state at epoch t: a candidate point on a DIRECTED road segment near the observation.

* Emission  log p(z | c) = -1/2 d^T S^-1 d  +  kappa (cos(dpsi) - 1)
    d      offset observation -> projection [m, EN]
    S      filter position covariance (2x2) + sigma_map^2 I (centreline / OSM error)
    dpsi   travel heading vs the directed segment's heading; used only above
           ``heading_min_speed_mps``; kappa = 1 / (sigma_heading^2 + sigma_heading_road^2)
* Transition  log p(c_t | c_t-1) = -|route - travelled| / beta
    route      shortest DRIVABLE distance between the two candidates (one-way respected)
    travelled  distance the dead-reckoned trajectory moved between the epochs
    no route within the cutoff -> probability 0: an impossible jump is impossible, it is
    not merely unlikely. This is what stops nearest-road snapping's teleports.
* Decoding
    online   forward filter (log-sum-exp) -> filtered posterior; output = its argmax
    offline  Viterbi over everything since the last break (``decode``)
* map_match_confidence = filtered posterior mass of the candidates lying within
  ``agree_radius_m`` of the chosen point (several directed segments meet at one node, so
  the mass of "this spot on the road" is split across them).
* Breaks: if no transition is possible, the chain restarts from emissions (counted).
* Off-network (car parks, unmapped roads): an epoch is IMPLAUSIBLE when no candidate is
  within the search radius or the best Mahalanobis distance exceeds ``chi2_gate``.
  ``suspend_after`` consecutive implausible epochs -> SUSPENDED (no output);
  ``reenter_after`` consecutive plausible ones -> resume with a fresh chain.

Determinism: candidates are ordered by segment id; ties break on the lowest index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from navcore.map_matching.road_network import Candidate, RoadNetwork


@dataclass(frozen=True)
class MatcherConfig:
    k_sigma: float = 3.0              # search radius = k * sqrt(lambda_max(S)), clipped
    r_min_m: float = 30.0
    r_max_m: float = 300.0
    max_candidates: int = 12          # nearest (Mahalanobis) kept per epoch
    sigma_map_m: float = 5.0          # OSM centreline vs vehicle path (lane offset, mapping error)
    beta_m: float = 10.0              # transition scale
    route_cutoff_factor: float = 2.0  # route search limit = factor * travelled + slack
    route_cutoff_slack_m: float = 100.0
    heading_min_speed_mps: float = 2.0
    sigma_heading_road_rad: float = math.radians(20.0)  # lane changes, curvature within a segment
    chi2_gate: float = 13.8           # chi^2(2 dof) 99.9 %: beyond this no road explains the fix
    suspend_after: int = 3
    reenter_after: int = 3
    agree_radius_m: float = 10.0
    viterbi_max_epochs: int = 3600    # bounded memory: smoothed decode looks back at most this far


@dataclass
class MapMatchResult:
    t_s: float
    matched: bool
    mode: str                      # matched | implausible | suspended | no_candidates
    x_m: float | None = None       # matched point (ENU, network tangent plane)
    y_m: float | None = None
    lat_rad: float | None = None
    lon_rad: float | None = None
    map_match_confidence: float = 0.0
    segment: int | None = None
    way: dict = field(default_factory=dict)
    correction_m: float | None = None   # |matched - observed|
    n_candidates: int = 0
    search_radius_m: float = 0.0


@dataclass
class _Epoch:
    t_s: float
    cands: list[Candidate]
    delta: np.ndarray              # Viterbi log-scores (normalised: max = 0)
    back: np.ndarray | None        # Viterbi back-pointers into the previous epoch


def _wrap(a: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):  # an all -inf column (unreachable candidate) -> -inf
        return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


class HmmMapMatcher:
    def __init__(self, network: RoadNetwork, cfg: MatcherConfig | None = None) -> None:
        self.net = network
        self.cfg = cfg or MatcherConfig()
        self.counts: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self._prev: list[Candidate] | None = None
        self._alpha: np.ndarray | None = None
        self._prev_xy: tuple[float, float] | None = None
        self._epochs: list[_Epoch] = []
        self._bad = 0
        self._good = 0
        self._travel = 0.0  # distance travelled since the chain last advanced
        self.suspended = False

    def _count(self, k: str) -> None:
        self.counts[k] = self.counts.get(k, 0) + 1

    # ---- model terms ---------------------------------------------------------------------
    def _emission(self, cands: list[Candidate], x: float, y: float, S: np.ndarray,
                  heading: float | None, sigma_heading: float) -> tuple[np.ndarray, np.ndarray]:
        d = np.array([[c.x - x, c.y - y] for c in cands])
        Si = np.linalg.inv(S)
        m2 = np.einsum("ij,jk,ik->i", d, Si, d)
        loge = -0.5 * m2
        if heading is not None:
            kappa = 1.0 / (sigma_heading ** 2 + self.cfg.sigma_heading_road_rad ** 2)
            hs = self.net.seg_heading[[c.seg for c in cands]]
            loge = loge + kappa * (np.cos(_wrap(hs - heading)) - 1.0)
        return loge, m2

    def _transition(self, prev: list[Candidate], cands: list[Candidate], travelled: float) -> np.ndarray:
        cutoff = self.cfg.route_cutoff_factor * travelled + self.cfg.route_cutoff_slack_m
        cache: dict[int, dict[int, float]] = {}
        T = np.full((len(prev), len(cands)), -np.inf)
        for i, a in enumerate(prev):
            for j, b in enumerate(cands):
                r = self.net.route_distance(a, b, cutoff, cache)
                if math.isfinite(r):
                    T[i, j] = -abs(r - travelled) / self.cfg.beta_m
        return T

    # ---- one epoch -----------------------------------------------------------------------
    def step(self, t_s: float, x: float, y: float, P_pos: np.ndarray, heading_rad: float | None = None,
             speed_mps: float = 0.0, sigma_heading_rad: float = math.radians(10.0),
             travelled_m: float | None = None, u: float = 0.0) -> MapMatchResult:
        """Match one observation.

        (x, y): fused position in the network's tangent plane [m]; P_pos: its 2x2 EN
        covariance [m^2]; heading_rad: ENU travel yaw atan2(v_N, v_E); travelled_m:
        path length the dead-reckoned trajectory covered since the previous epoch
        (default: straight-line distance between the observations).
        """
        cfg = self.cfg
        S = np.asarray(P_pos, dtype=np.float64)[:2, :2] + cfg.sigma_map_m ** 2 * np.eye(2)
        r = float(np.clip(cfg.k_sigma * math.sqrt(float(np.linalg.eigvalsh(S)[-1])), cfg.r_min_m, cfg.r_max_m))
        if travelled_m is None:
            travelled_m = 0.0 if self._prev_xy is None else math.hypot(x - self._prev_xy[0], y - self._prev_xy[1])
        self._prev_xy = (x, y)
        self._travel += travelled_m  # skipped (implausible) epochs keep accumulating
        cands = self.net.candidates(x, y, r)
        res = MapMatchResult(t_s=t_s, matched=False, mode="no_candidates", search_radius_m=r)
        heading = heading_rad if (heading_rad is not None and speed_mps >= cfg.heading_min_speed_mps) else None
        if cands:
            loge, m2 = self._emission(cands, x, y, S, heading, sigma_heading_rad)
            keep = np.argsort(m2, kind="stable")[: cfg.max_candidates]
            keep.sort()  # back to segment-id order: deterministic
            cands = [cands[k] for k in keep]
            loge, m2 = loge[keep], m2[keep]
            plausible = float(m2.min()) <= cfg.chi2_gate
        else:
            plausible = False
        res.n_candidates = len(cands)

        # ---- off-network bookkeeping
        if plausible:
            self._good += 1
            self._bad = 0
        else:
            self._bad += 1
            self._good = 0
        if not self.suspended and self._bad >= cfg.suspend_after:
            self.suspended = True
            self._count("suspensions")
            self._prev, self._alpha, self._travel = None, None, 0.0
        if self.suspended:
            if self._good >= cfg.reenter_after:
                self.suspended = False
                self._count("reentries")
            else:
                res.mode = "suspended"
                self._count("epochs_suspended")
                return res
        if not plausible:
            res.mode = "implausible" if cands else "no_candidates"
            self._count(f"epochs_{res.mode}")
            return res

        # ---- forward filter + Viterbi
        back = None
        if self._prev is None:
            alpha, delta = loge.copy(), loge.copy()
        else:
            T = self._transition(self._prev, cands, self._travel)
            if not np.isfinite(T).any():
                self._count("breaks")  # no road route connects the epochs: restart the chain
                alpha, delta = loge.copy(), loge.copy()
            else:
                alpha = _logsumexp(self._alpha[:, None] + T, axis=0) + loge
                prev_delta = self._epochs[-1].delta
                sc = prev_delta[:, None] + T
                back = np.argmax(sc, axis=0)
                delta = sc[back, np.arange(len(cands))] + loge
                if not np.isfinite(alpha).any():
                    self._count("breaks")
                    alpha, delta, back = loge.copy(), loge.copy(), None
        alpha = alpha - _logsumexp(alpha)
        delta = delta - np.max(delta)
        if back is None:
            self._epochs.append(_Epoch(t_s, cands, delta, None))
            self._epochs = self._epochs[-1:]  # a break starts a new decodable chain
        else:
            self._epochs.append(_Epoch(t_s, cands, delta, back))
            del self._epochs[: -cfg.viterbi_max_epochs]
        self._prev, self._alpha, self._travel = cands, alpha, 0.0

        post = np.exp(alpha)
        j = int(np.argmax(post))
        c = cands[j]
        near = [k for k, o in enumerate(cands) if math.hypot(o.x - c.x, o.y - c.y) <= cfg.agree_radius_m]
        lat, lon = self.net.to_geodetic(c.x, c.y, u)
        res.matched, res.mode = True, "matched"
        res.x_m, res.y_m, res.lat_rad, res.lon_rad = c.x, c.y, lat, lon
        res.map_match_confidence = float(min(1.0, post[near].sum()))
        res.segment, res.way = c.seg, self.net.way_of(c.seg)
        res.correction_m = math.hypot(c.x - x, c.y - y)
        self._count("epochs_matched")
        return res

    def decode(self) -> list[tuple[float, Candidate]]:
        """Viterbi (smoothed) path over the current chain, i.e. since the last break."""
        if not self._epochs:
            return []
        j = int(np.argmax(self._epochs[-1].delta))
        path = []
        for ep in reversed(self._epochs):
            path.append((ep.t_s, ep.cands[j]))
            if ep.back is None:
                break
            j = int(ep.back[j])
        return path[::-1]


__all__ = ["HmmMapMatcher", "MapMatchResult", "MatcherConfig"]
