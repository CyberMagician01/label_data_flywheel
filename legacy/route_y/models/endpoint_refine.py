"""Endpoint-local refinement for head/tail pose structure."""

from __future__ import annotations

import torch
from torch import nn


class EndpointRefine(nn.Module):
    """Two endpoint attention maps that refine local P2/P3 features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.attn(x)
        return x + self.refine(x * (1.0 + maps.mean(1, keepdim=True)))
