"""Density head for dense bee capacity estimation."""

from __future__ import annotations

import torch
from torch import nn


class DensityHead(nn.Module):
    """P2 density head: DWConv, pointwise conv and Softplus."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
