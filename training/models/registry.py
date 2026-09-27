"""Model registry: small sequence models for phone IMU windows (Phase 5).

Sized for the two constraints of docs/ml_pipeline.md sec 5 -- 4 GB of VRAM and on-phone
inference: tens of thousands of parameters, not millions. Every model maps a window
(B, C, W) to a HETEROSCEDASTIC output (mean, log-variance) of ONE physical quantity:
the fusion filter consumes it as a measurement with covariance (sec 6). No model outputs
a coordinate (enforced in training.datasets / tests.ml).

    model = build("tcn", in_channels=13, positive=True, hidden=48, levels=4, kernel=3, dropout=0.1)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register(name: str):
    def deco(cls):
        if name in REGISTRY:
            raise KeyError(f"model {name!r} already registered")
        REGISTRY[name] = cls
        return cls
    return deco


def build(arch: str, in_channels: int, positive: bool, **kw) -> nn.Module:
    if arch not in REGISTRY:
        raise KeyError(f"unknown arch {arch!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[arch](in_channels=in_channels, positive=positive, **kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HeteroscedasticHead(nn.Module):
    """features (B, H) -> (mean, logvar). ``positive`` passes the mean through softplus
    (speed can never be negative)."""

    def __init__(self, hidden: int, positive: bool) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden, 2)
        self.positive = positive

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.lin(h)
        mean, logvar = out[:, 0], out[:, 1]
        if self.positive:
            mean = nn.functional.softplus(mean)
        return mean, logvar


class _CausalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv(c_in, c_out, kernel, dilation), nn.GELU(), nn.Dropout(dropout),
            _CausalConv(c_out, c_out, kernel, dilation), nn.GELU(), nn.Dropout(dropout))
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


@register("tcn")
class TCN(nn.Module):
    """Causal dilated temporal ConvNet; output read at the LAST time step (causal: no
    future samples, so it can run on-device in real time)."""

    def __init__(self, in_channels: int, positive: bool, hidden: int = 48, levels: int = 4,
                 kernel: int = 3, dropout: float = 0.1, **_) -> None:
        super().__init__()
        layers, c = [], in_channels
        for i in range(levels):
            layers.append(_TCNBlock(c, hidden, kernel, 2 ** i, dropout))
            c = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = HeteroscedasticHead(hidden, positive)
        self.receptive_field = 1 + 2 * (kernel - 1) * (2 ** levels - 1)

    def forward(self, x):
        return self.head(self.tcn(x)[:, :, -1])


@register("cnn1d")
class CNN1D(nn.Module):
    """Plain 1-D CNN with global average pooling."""

    def __init__(self, in_channels: int, positive: bool, hidden: int = 48, layers: int = 3,
                 kernel: int = 5, dropout: float = 0.1, **_) -> None:
        super().__init__()
        mods, c = [], in_channels
        for _ in range(layers):
            mods += [nn.Conv1d(c, hidden, kernel, padding=kernel // 2), nn.BatchNorm1d(hidden),
                     nn.GELU(), nn.Dropout(dropout)]
            c = hidden
        self.net = nn.Sequential(*mods)
        self.head = HeteroscedasticHead(hidden, positive)

    def forward(self, x):
        return self.head(self.net(x).mean(dim=2))


class _Recurrent(nn.Module):
    cell: type[nn.Module] = nn.GRU

    def __init__(self, in_channels: int, positive: bool, hidden: int = 48, layers: int = 1,
                 dropout: float = 0.1, **_) -> None:
        super().__init__()
        self.inp = nn.Conv1d(in_channels, hidden, 3, padding=1)  # light local smoothing
        self.rnn = self.cell(hidden, hidden, num_layers=layers, batch_first=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = HeteroscedasticHead(hidden, positive)

    def forward(self, x):
        h = nn.functional.gelu(self.inp(x)).transpose(1, 2)  # (B, W, H)
        out, _ = self.rnn(h)
        return self.head(self.drop(out[:, -1]))


@register("gru")
class GRU(_Recurrent):
    cell = nn.GRU


@register("lstm")
class LSTM(_Recurrent):
    cell = nn.LSTM
