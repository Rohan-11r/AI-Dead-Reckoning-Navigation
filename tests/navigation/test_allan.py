"""Allan deviation against analytically known SYNTHETIC noise (fixed seeds)."""

from __future__ import annotations

import numpy as np
import pytest

from navcore.calibration import allan


def test_white_noise_slope_and_density():
    dt, N = 0.1, 2e-3  # density [unit*sqrt(s)] -> per-sample std N / sqrt(dt)
    y = np.random.default_rng(0).normal(0.0, N / np.sqrt(dt), 200_000)
    tau, adev, _ = allan.overlapping_adev(y, dt)
    # -1/2 slope wherever the estimate has many independent windows (tau <= T/100); at
    # tau ~ T/3 only ~3 windows remain and the Allan estimate is legitimately noisy
    ok = tau <= y.size * dt / 100
    np.testing.assert_allclose(adev[ok], N / np.sqrt(tau[ok]), rtol=0.05)
    assert allan.noise_parameters(y, dt)["white_noise_density"] == pytest.approx(N, rel=0.03)


def test_constant_bias_is_invisible_and_random_walk_rises():
    dt = 0.1
    rng = np.random.default_rng(1)
    white = rng.normal(0, 1e-3, 100_000)
    _, a1, _ = allan.overlapping_adev(white + 5.0, dt)
    _, a0, _ = allan.overlapping_adev(white, dt)
    np.testing.assert_allclose(a1, a0, rtol=1e-6)  # a constant offset has zero Allan variance
    rw = np.cumsum(rng.normal(0, 1e-5, 100_000))
    _, a2, _ = allan.overlapping_adev(white + rw, dt)
    assert a2[-1] > a2[np.argmin(a2)] * 2  # rate random walk turns the curve upward


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        allan.overlapping_adev(np.array([1.0, np.nan] * 20), 0.1)
    with pytest.raises(ValueError):
        allan.noise_parameters(np.zeros(20), 0.1)
