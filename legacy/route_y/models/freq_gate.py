"""Dynamic frequency-band fusion for BeePoseTrack-Y."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HaarBandDecompose(nn.Module):
    """Fixed Haar-like low/mid/high band decomposition."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, -1.0], [1.0, -1.0]],
                [[1.0, 1.0], [-1.0, -1.0]],
            ],
            dtype=torch.float32,
        )
        weight = kernels[:, None] / 2.0
        weight = weight.repeat(channels, 1, 1, 1)
        self.channels = channels
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = F.conv2d(x, self.weight, padding=1, groups=self.channels)
        y = y[:, :, : x.shape[-2], : x.shape[-1]]
        return y[:, 0::3], y[:, 1::3], y[:, 2::3]


class DynamicFrequencyGate(nn.Module):
    """Scale/domain-conditioned low/mid/high frequency gate."""

    def __init__(self, channels: int, stats_dim: int = 6, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.bands = HaarBandDecompose(channels)
        self.weight_mlp = nn.Sequential(
            nn.Linear(channels + stats_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, stats: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        if stats is None:
            stats = x.new_zeros((b, 6))
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        weights = torch.softmax(self.weight_mlp(torch.cat([pooled, stats], dim=1)), dim=1)
        low, mid, high = self.bands(x)
        fused = weights[:, 0, None, None, None] * low
        fused = fused + weights[:, 1, None, None, None] * mid
        fused = fused + weights[:, 2, None, None, None] * high
        return x + self.mix(fused)
