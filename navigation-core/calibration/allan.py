"""Overlapping Allan deviation and the noise parameters read from it.

For a rate signal y sampled at interval ``dt`` (IEEE Std 952 / 1293 conventions):

* white noise (angle / velocity random walk) N:   sigma(tau) = N / sqrt(tau)  -> N = sigma(1 s)
  [units: (signal units) * sqrt(s), e.g. rad/s/sqrt(Hz) == rad/sqrt(s)]
* bias instability B: the flat floor, B = min(sigma) / sqrt(2 ln 2 / pi) = min / 0.664

Values are MEASURED here and fed to the filter's process noise (docs/navigation_math.md
sec 8.4) -- they are not tuned.
"""

from __future__ import annotations

import math

import numpy as np

BIAS_INSTABILITY_FACTOR = math.sqrt(2.0 * math.log(2.0) / math.pi)  # 0.664


def overlapping_adev(y: np.ndarray, dt_s: float, taus_s: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of rate samples ``y`` taken every ``dt_s``.

    Returns (tau_s, adev, n_terms). NaNs are not allowed -- pass contiguous data only.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.size < 10 or not np.all(np.isfinite(y)):
        raise ValueError("overlapping_adev needs a finite 1-D series of >= 10 samples")
    n = y.size
    if taus_s is None:
        m = np.unique(np.logspace(0, np.log10(n // 3), 40).astype(int))
    else:
        m = np.unique(np.maximum(1, np.round(np.asarray(taus_s) / dt_s)).astype(int))
        m = m[m <= n // 3]
    theta = np.concatenate([[0.0], np.cumsum(y) * dt_s])  # integrated signal
    out_tau, out_adev, out_n = [], [], []
    for mi in m:
        tau = mi * dt_s
        d = theta[2 * mi:] - 2.0 * theta[mi:-mi] + theta[: -2 * mi]
        if d.size < 1:
            continue
        out_tau.append(tau)
        out_adev.append(math.sqrt(np.mean(d * d) / (2.0 * tau * tau)))
        out_n.append(d.size)
    return np.array(out_tau), np.array(out_adev), np.array(out_n)


def noise_parameters(y: np.ndarray, dt_s: float) -> dict:
    """White-noise density N (read at tau = 1 s, log-interpolated) and bias instability B."""
    tau, adev, _ = overlapping_adev(y, dt_s)
    if tau.size < 2 or tau[0] > 1.0 or tau[-1] < 1.0:
        raise ValueError("series too short to read the Allan deviation at tau = 1 s")
    n_1s = float(np.exp(np.interp(0.0, np.log(tau), np.log(adev))))
    i = int(np.argmin(adev))
    return {
        "white_noise_density": n_1s,
        "bias_instability": float(adev[i] / BIAS_INSTABILITY_FACTOR),
        "bias_instability_tau_s": float(tau[i]),
        "tau_max_s": float(tau[-1]),
    }
