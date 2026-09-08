"""Route-Best loss wrapper for YOLO26 pose training."""

from __future__ import annotations

from typing import Any
import os

import torch
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh

from losses.density import DensityLoss, gaussian_density
from losses.pose_structure import PoseStructureLoss
from models.route_context import get_route_context


class RouteBestPoseLoss(PoseLoss26):
    """YOLO26 pose loss plus Route-Best structural and auxiliary terms."""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.model = model
        route_cfg = getattr(model, "route_config", {}) or {}
        loss_cfg = route_cfg.get("loss", {})
        self.structure_loss = PoseStructureLoss(
            oks=float(loss_cfg.get("oks", 1.0)),
            visibility=float(loss_cfg.get("visibility", 0.5)),
            axis=float(loss_cfg.get("axis", 0.5)),
            inside=float(loss_cfg.get("inside", 0.2)),
            length=float(loss_cfg.get("length", 0.2)),
        ).to(self.device)
        self.density_loss = DensityLoss().to(self.device)
        self.structure_weight = float(loss_cfg.get("structure", 1.0))
        self.density_weight = float(loss_cfg.get("density", 0.1))
        self.prototype_weight = float(loss_cfg.get("prototype", 0.05))
        self.temporal_weight = float(loss_cfg.get("track", 0.1))
        self.warmup_epochs = float((route_cfg.get("training", {}) or {}).get("warmup_epochs", 10))
        self.stage = os.environ.get("BEEPOSETRACK_STAGE", "")

    def warmup_gain(self) -> float:
        epoch = float(getattr(self.model, "route_epoch", self.warmup_epochs) or 0)
        return min(max(epoch / max(self.warmup_epochs, 1.0), 0.0), 1.0)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        total, items = self._masked_pose_loss(preds, batch)
        gain = self.warmup_gain()
        aux_terms = self._auxiliary_terms(preds, batch)
        aux_total = total.new_zeros(())
        aux_values = []
        for name in ("structure", "density", "prototype", "temporal"):
            value = aux_terms.get(name, total.new_zeros(()))
            weight = self._stage_weight(name)
            weighted = value * weight * gain
            aux_total = aux_total + weighted
            aux_values.append(weighted.detach())
        batch_size = int(preds["feats"][0].shape[0])
        total = total + aux_total * batch_size
        items = torch.cat([items, torch.stack(aux_values)])
        return total, items

    def _masked_pose_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(6 if self.rle_loss else 5, device=self.device)
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        if self._pose_supervision_enabled() and fg_mask.sum():
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]
            pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)
            if self.rle_loss and preds.get("kpts_sigma", None) is not None:
                pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
                pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)
                pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)
            pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle
        return loss * batch_size, loss.detach()

    def _pose_supervision_enabled(self) -> bool:
        stage = self.stage.lower()
        return not ("det" in stage and "only" in stage)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        zero = pred_kpts[..., :0].sum()
        kpts_loss = zero
        kpts_obj_loss = zero
        rle_loss = zero
        if not masks.any():
            return kpts_loss, kpts_obj_loss, rle_loss

        target_bboxes = target_bboxes / stride_tensor
        gt_kpt_all = selected_keypoints[masks]
        pred_kpt_all = pred_kpts[masks]
        target_boxes_all = target_bboxes[masks]
        kpt_mask_all = gt_kpt_all[..., 2] != 0 if gt_kpt_all.shape[-1] == 3 else torch.full_like(gt_kpt_all[..., 0], True)
        pose_supervised = kpt_mask_all.any(dim=1)
        if not pose_supervised.any():
            return kpts_loss, kpts_obj_loss, rle_loss

        gt_kpt = gt_kpt_all[pose_supervised]
        pred_kpt = pred_kpt_all[pose_supervised]
        kpt_mask = kpt_mask_all[pose_supervised]
        area = xyxy2xywh(target_boxes_all[pose_supervised])[:, 2:].prod(1, keepdim=True)
        kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

        if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
            rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask).clamp(min=0)
        if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
            kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss, rle_loss

    def _stage_weight(self, name: str) -> float:
        stage = self.stage.lower()
        if "det" in stage and "only" in stage:
            return 0.0
        if stage.startswith("y1"):
            return 0.0
        if stage.startswith("y2"):
            return self.structure_weight if name == "structure" else 0.0
        return float(getattr(self, f"{name}_weight"))

    def _auxiliary_terms(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        structure = self._structure_term(preds, batch)
        if structure is not None:
            out["structure"] = structure
        density = self._density_term(preds, batch)
        if density is not None:
            out["density"] = density
        ctx = get_route_context()
        if ctx is not None:
            if "prototype" in ctx.aux_losses:
                out["prototype"] = ctx.aux_losses["prototype"].to(self.device)
            if ctx.clip is not None and ctx.clip.shape[1] > 1:
                out["temporal"] = self._clip_temporal_term(ctx.clip.to(self.device))
        return out

    def _structure_term(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        try:
            pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), _, _ = self.get_assigned_targets_and_loss(preds, batch)
        except Exception:
            return None
        if not bool(fg_mask.sum()):
            return None
        batch_size = pred_kpts.shape[0]
        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)
        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous().view(batch_size, -1, self.kpt_shape[0], 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)[..., :3]

        keypoints = batch["keypoints"].to(self.device).float().clone()
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]
        keypoints[..., 0] *= imgsz[1]
        keypoints[..., 1] *= imgsz[0]
        selected = self._select_target_keypoints(keypoints, batch["batch_idx"].view(-1, 1), target_gt_idx, fg_mask)
        selected[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        boxes = target_bboxes / stride_tensor
        pred_fg = pred_kpts[fg_mask]
        gt_fg = selected[fg_mask]
        boxes_fg = boxes[fg_mask]
        pose_supervised = gt_fg[..., 2].ne(0).any(dim=1)
        if not pose_supervised.any():
            return None
        pred_fg = pred_fg[pose_supervised]
        gt_fg = gt_fg[pose_supervised]
        boxes_fg = boxes_fg[pose_supervised]
        return self.structure_loss(pred_fg[..., :2], pred_fg[..., 2], gt_fg[..., :2], gt_fg[..., 2], boxes_fg)

    def _density_term(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        ctx = get_route_context()
        if ctx is None or not ctx.density_maps:
            return None
        pred = ctx.density_maps[-1]
        b, _, mh, mw = pred.shape
        image_hw = (int(preds["feats"][0].shape[2] * self.stride[0]), int(preds["feats"][0].shape[3] * self.stride[0]))
        targets = []
        batch_idx = batch["batch_idx"].to(self.device).long().view(-1)
        boxes = xywh2xyxy(batch["bboxes"].to(self.device).float())
        boxes[:, [0, 2]] *= image_hw[1]
        boxes[:, [1, 3]] *= image_hw[0]
        for bi in range(b):
            targets.append(gaussian_density(boxes[batch_idx == bi], image_hw, (mh, mw)))
        target = torch.stack(targets, dim=0).unsqueeze(1).to(dtype=pred.dtype)
        return self.density_loss(pred, target)

    @staticmethod
    def _clip_temporal_term(clip: torch.Tensor) -> torch.Tensor:
        diffs = clip[:, 1:] - clip[:, :-1]
        return diffs.abs().mean()


class RouteBestE2ELoss(E2ELoss):
    """End-to-end YOLO26 loss using RouteBestPoseLoss on both branches."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model, RouteBestPoseLoss)
