"""Integration: dead-reckoned (INS/AI/NHC) trajectory -> road-constrained position.

``DeadReckoningMapMatcher`` takes the fused filter's output -- position, its 2x2 horizontal
covariance and the velocity -- and returns a ``MapMatchResult`` with a position ON a road and
a ``map_match_confidence`` in [0, 1]. It is meant to run continuously (so that when GNSS is
lost it already holds the road context) and matters most during an outage, when the fused
covariance grows and the road network is the only external constraint left.

REFINEMENT ONLY (AGENTS.md sec 1, docs/map_matching.md sec 5): nothing here writes to the
filter. Remove this module and the fused trajectory is unchanged (tested). Use the matched
position only when ``matched`` is True and the confidence meets the caller's threshold;
otherwise use the fused position, which is always valid. Offline: the road network is a
local file; there is no network access anywhere in navcore.map_matching.
"""

from __future__ import annotations

import math

import numpy as np

from navcore.filtering.ekf import P_, V_, ErrorStateEKF
from navcore.map_matching.hmm import HmmMapMatcher, MapMatchResult, MatcherConfig
from navcore.map_matching.road_network import RoadNetwork


def course_and_sigma(v_en: np.ndarray, P_v: np.ndarray) -> tuple[float, float, float]:
    """Travel direction atan2(v_N, v_E) [rad], its 1-sigma from the velocity covariance
    (first-order), and the horizontal speed [m/s]."""
    ve, vn = float(v_en[0]), float(v_en[1])
    s2 = ve * ve + vn * vn
    speed = math.sqrt(s2)
    if speed < 1e-3:
        return 0.0, math.pi, speed
    J = np.array([-vn, ve]) / s2
    var = float(J @ np.asarray(P_v)[:2, :2] @ J)
    return math.atan2(vn, ve), math.sqrt(max(var, 0.0)), speed


class DeadReckoningMapMatcher:
    """Runs the HMM matcher on the fused state at a fixed rate (default 1 Hz)."""

    def __init__(self, network: RoadNetwork, cfg: MatcherConfig | None = None, rate_hz: float = 1.0) -> None:
        self.net = network
        self.hmm = HmmMapMatcher(network, cfg)
        self.period_s = 1.0 / rate_hz
        self._last_t = -math.inf
        self._last_xy: tuple[float, float] | None = None
        self._path_m = 0.0
        self.last: MapMatchResult | None = None

    def update(self, t_s: float, lat_rad: float, lon_rad: float, P_pos_en: np.ndarray,
               v_en: np.ndarray, P_v_en: np.ndarray) -> MapMatchResult | None:
        """Feed one fused sample; returns a new result at most every ``period_s``, else None.

        The path length between matcher epochs is accumulated from every sample fed here, so
        curves are not cut short by the straight line between epochs.
        """
        x, y, u = self.net.to_xy(lat_rad, lon_rad)
        if self._last_xy is not None:
            self._path_m += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)
        if t_s - self._last_t < self.period_s - 1e-9:
            return None
        heading, sig_h, speed = course_and_sigma(np.asarray(v_en), np.asarray(P_v_en))
        travelled = self._path_m if math.isfinite(self._last_t) else None
        self._last_t, self._path_m = t_s, 0.0
        self.last = self.hmm.step(t_s, x, y, P_pos_en, heading, speed, sig_h, travelled, u)
        return self.last

    def update_from_ekf(self, ekf: ErrorStateEKF) -> MapMatchResult | None:
        """Read (never write) the filter: nominal position/velocity and covariance blocks."""
        s = ekf.nominal
        return self.update(s.t_s, s.lat_rad, s.lon_rad, ekf.P[P_, P_][:2, :2], s.v_enu_mps, ekf.P[V_, V_])


__all__ = ["DeadReckoningMapMatcher", "course_and_sigma"]
