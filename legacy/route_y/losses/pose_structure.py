"""Head-tail structural losses and metrics."""

from __future__ import annotations

import torch
from torch import nn


def _axis(points: torch.Tensor) -> torch.Tensor:
    return points[..., 1, :] - points[..., 0, :]


class PoseStructureLoss(nn.Module):
    """OKS, visibility, axis, inside-box and length-ratio penalties."""

    def __init__(self, oks: float = 1.0, visibility: float = 0.5, axis: float = 0.5, inside: float = 0.2, length: float = 0.2) -> None:
        super().__init__()
        self.oks_weight = oks
        self.visibility_weight = visibility
        self.axis_weight = axis
        self.inside_weight = inside
        self.length_weight = length
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        pred_xy: torch.Tensor,
        pred_vis: torch.Tensor,
        gt_xy: torch.Tensor,
        gt_vis: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        quality: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visible = (gt_vis > 0).float()
        if quality is None:
            quality = pred_xy.new_ones(pred_xy.shape[:-2])
        diag = torch.linalg.norm(boxes_xyxy[:, 2:] - boxes_xyxy[:, :2], dim=-1).clamp_min(1.0)
        err = torch.linalg.norm(pred_xy - gt_xy, dim=-1) / diag[:, None]
        oks = (err * visible).sum(-1) / visible.sum(-1).clamp_min(1.0)
        vis_loss = (self.bce(pred_vis, visible) * visible).sum(-1) / visible.sum(-1).clamp_min(1.0)

        pred_axis = torch.nn.functional.normalize(_axis(pred_xy), dim=-1)
        gt_axis = torch.nn.functional.normalize(_axis(gt_xy), dim=-1)
        axis_loss = 1.0 - (pred_axis * gt_axis).sum(-1).clamp(-1.0, 1.0)

        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        px, py = pred_xy.unbind(-1)
        margin_x = 0.05 * (x2 - x1).clamp_min(1.0)
        margin_y = 0.05 * (y2 - y1).clamp_min(1.0)
        outside = (x1[:, None] - margin_x[:, None] - px).relu()
        outside = outside + (px - x2[:, None] - margin_x[:, None]).relu()
        outside = outside + (y1[:, None] - margin_y[:, None] - py).relu()
        outside = outside + (py - y2[:, None] - margin_y[:, None]).relu()
        inside_loss = (outside / diag[:, None]).mean(-1)

        pred_len = torch.linalg.norm(_axis(pred_xy), dim=-1)
        gt_len = torch.linalg.norm(_axis(gt_xy), dim=-1).clamp_min(1.0)
        length_loss = torch.abs(pred_len / gt_len - 1.0)

        total = self.oks_weight * oks
        total = total + self.visibility_weight * vis_loss
        total = total + self.axis_weight * axis_loss
        total = total + self.inside_weight * inside_loss
        total = total + self.length_weight * length_loss
        return (total * quality).mean()
