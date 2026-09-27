"""Heteroscedastic regression losses (docs/ml_pipeline.md sec 6).

    L = Huber(mu - y)  +  w * 1/2 [ (y - mu)^2 / sigma^2 + log sigma^2 ]

The Gaussian NLL term gives the filter a variance; the Huber term on the mean keeps the
point estimate robust to residual label outliers and stops the NLL from "explaining away"
a bad mean by inflating sigma^2 (the known collapse mode). log sigma^2 is clamped to a
configured range for numerical safety. Calibration is checked separately (``coverage``).
"""

from __future__ import annotations

import torch


def gaussian_nll(mean, logvar, y):
    return 0.5 * ((y - mean) ** 2 * torch.exp(-logvar) + logvar)


def huber_nll_loss(mean, logvar, y, delta: float, nll_weight: float, logvar_clamp=(-8.0, 6.0)):
    lv = logvar.clamp(*logvar_clamp)
    huber = torch.nn.functional.huber_loss(mean, y, delta=delta, reduction="mean")
    nll = gaussian_nll(mean, lv, y).mean()
    return huber + nll_weight * nll, {"huber": float(huber.detach()), "nll": float(nll.detach())}


@torch.no_grad()
def regression_metrics(mean, logvar, y, logvar_clamp=(-8.0, 6.0)) -> dict:
    err = mean - y
    sigma = torch.exp(0.5 * logvar.clamp(*logvar_clamp))
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(torch.sqrt((err ** 2).mean())),
        "nll": float(gaussian_nll(mean, logvar.clamp(*logvar_clamp), y).mean()),
        # calibration: a Gaussian puts 68.3 % within 1 sigma and 95.4 % within 2 sigma
        "coverage_1sigma": float((err.abs() <= sigma).float().mean()),
        "coverage_2sigma": float((err.abs() <= 2 * sigma).float().mean()),
        "mean_sigma": float(sigma.mean()),
    }
