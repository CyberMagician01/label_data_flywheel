"""Temporal auxiliary losses for matched track observations."""

from __future__ import annotations

import torch
from torch import nn


class TemporalConsistencyLoss(nn.Module):
    """Smooth center, size and directed-axis changes along GT-matched tracks."""

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, prev_boxes: torch.Tensor, curr_boxes: torch.Tensor, prev_kpts: torch.Tensor, curr_kpts: torch.Tensor) -> torch.Tensor:
        if prev_boxes.numel() == 0 or curr_boxes.numel() == 0:
            return curr_boxes.new_tensor(0.0)
        prev_center = (prev_boxes[:, :2] + prev_boxes[:, 2:]) * 0.5
        curr_center = (curr_boxes[:, :2] + curr_boxes[:, 2:]) * 0.5
        center_loss = torch.linalg.norm(curr_center - prev_center, dim=-1).mean()
        prev_size = (prev_boxes[:, 2:] - prev_boxes[:, :2]).clamp_min(1.0)
        curr_size = (curr_boxes[:, 2:] - curr_boxes[:, :2]).clamp_min(1.0)
        size_loss = torch.abs(torch.log(curr_size / prev_size)).mean()
        prev_axis = torch.nn.functional.normalize(prev_kpts[:, 1] - prev_kpts[:, 0], dim=-1)
        curr_axis = torch.nn.functional.normalize(curr_kpts[:, 1] - curr_kpts[:, 0], dim=-1)
        axis_loss = (1.0 - (prev_axis * curr_axis).sum(-1).clamp(-1.0, 1.0)).mean()
        return self.weight * (center_loss + size_loss + axis_loss)
