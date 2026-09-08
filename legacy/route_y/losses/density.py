"""Density-map target generation and losses."""

from __future__ import annotations

import torch
from torch import nn


def gaussian_density(boxes_xyxy: torch.Tensor, image_hw: tuple[int, int], map_hw: tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    mh, mw = map_hw
    device = boxes_xyxy.device
    yy, xx = torch.meshgrid(torch.arange(mh, device=device), torch.arange(mw, device=device), indexing="ij")
    density = torch.zeros((mh, mw), device=device)
    for box in boxes_xyxy:
        cx = ((box[0] + box[2]) * 0.5 / max(w, 1)) * mw
        cy = ((box[1] + box[3]) * 0.5 / max(h, 1)) * mh
        sigma = torch.clamp(torch.minimum(box[2] - box[0], box[3] - box[1]) / max(w, h) * max(mw, mh), min=1.0, max=8.0)
        density = density + torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return density


class DensityLoss(nn.Module):
    """Huber density loss plus count consistency."""

    def __init__(self, count_weight: float = 0.1) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.count_weight = count_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        field = self.huber(pred, target)
        count = torch.abs(pred.flatten(1).sum(1) - target.flatten(1).sum(1)).mean()
        return field + self.count_weight * count
