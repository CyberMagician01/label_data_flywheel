"""Re-parameterizable temporal difference convolution blocks for BeePoseTrack-Y."""

from __future__ import annotations

import torch
from torch import nn


class RepTDC(nn.Module):
    """Temporal difference block with 1/2/4 frame lags.

    The block accepts either a single-frame feature tensor ``[B,C,H,W]`` or a
    causal clip tensor ``[B,T,C,H,W]``. Single-frame inputs are treated as the
    deployment boundary case from the route document: temporal differences are
    zero and the spatial branch remains active.
    """

    def __init__(self, channels: int, reduction: int = 4, strides: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.strides = tuple(strides)
        self.center = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.diff = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            )
            for _ in self.strides
        )
        self.local3d = nn.Sequential(
            nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False),
            nn.BatchNorm3d(channels),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels + hidden * len(self.strides) + channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            current = x
            zeros = [torch.zeros_like(self.reduce(current)) for _ in self.strides]
            temporal = torch.zeros_like(current)
        elif x.ndim == 5:
            current = x[:, -1]
            reduced = self.reduce(current)
            zeros = torch.zeros_like(reduced)
            diffs = []
            for lag, branch in zip(self.strides, self.diff):
                hist = x[:, max(x.shape[1] - 1 - lag, 0)]
                diffs.append(branch(self.reduce(current - hist)))
            zeros = diffs if diffs else [zeros]
            temporal = self.local3d(x.transpose(1, 2))[:, :, -1]
        else:
            raise ValueError(f"RepTDC expects 4D or 5D tensor, got {tuple(x.shape)}")
        return self.project(torch.cat([self.center(current), *zeros, temporal], dim=1))


class RepTDCFuse(nn.Module):
    """Residual fusion wrapper used in YOLO feature maps."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.tdc = RepTDC(channels, reduction=reduction)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        motion = self.tdc(x)
        return x + self.gate(motion) * motion
