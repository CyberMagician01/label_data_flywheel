#!/usr/bin/env python3
"""Train YOLO26-Pose with RGB/IR separated detection heads.

This is an isolated experimental variant of the aligned single-frame baseline.
Backbone, neck, and pose heads stay shared. Only the box/class detection heads
are duplicated for RGB and IR. Training still supports the two-stage protocol:
det-only first, then pose-from-det with pose losses masked to complete poses.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.modules.head import Detect, Pose26
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from ultralytics.utils.ops import xyxy2xywh


def infer_domain_id(path: str | Path) -> int:
    """Return 0 for RGB/A videos and 1 for IR/B videos."""
    name = Path(path).name.upper()
    return 1 if name.startswith("B-") or "_IR" in name or "/IR/" in str(path).upper() else 0


def _select_domain_tensor(rgb: torch.Tensor, ir: torch.Tensor, domain_ids: torch.Tensor | None) -> torch.Tensor:
    if domain_ids is None:
        return rgb
    mask = domain_ids.to(device=rgb.device, dtype=torch.bool).view(-1, 1, 1)
    return torch.where(mask, ir, rgb)


def dual_domain_forward_head(
    self,
    x: list[torch.Tensor],
    box_head: torch.nn.Module = None,
    cls_head: torch.nn.Module = None,
    pose_head: torch.nn.Module = None,
    kpts_head: torch.nn.Module = None,
    kpts_sigma_head: torch.nn.Module = None,
) -> dict[str, torch.Tensor]:
    """Forward pass using RGB/IR-specific box/class heads and shared pose head."""
    if not hasattr(self, "cv2_rgb"):
        return self._bee_standard_forward_head(x, box_head, cls_head, pose_head, kpts_head, kpts_sigma_head)

    rgb_preds = Detect.forward_head(self, x, self.cv2_rgb, self.cv3_rgb)
    ir_preds = Detect.forward_head(self, x, self.cv2_ir, self.cv3_ir)
    domain_ids = getattr(self, "bee_domain_ids", None)
    preds = {
        "boxes": _select_domain_tensor(rgb_preds["boxes"], ir_preds["boxes"], domain_ids),
        "scores": _select_domain_tensor(rgb_preds["scores"], ir_preds["scores"], domain_ids),
        "feats": x,
    }

    if pose_head is not None:
        bs = x[0].shape[0]
        features = [self.cv4[i](x[i]) for i in range(self.nl)]
        preds["kpts"] = torch.cat([self.cv4_kpts[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        if self.training:
            preds["kpts_sigma"] = torch.cat(
                [self.cv4_sigma[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
            )
    return preds


def install_dual_domain_class_patch() -> None:
    """Patch Pose26 at class level so checkpoints remain reloadable."""
    if not hasattr(Pose26, "_bee_standard_forward_head"):
        Pose26._bee_standard_forward_head = Pose26.forward_head
    Pose26.dual_domain_forward_head = dual_domain_forward_head
    Pose26.forward_head = dual_domain_forward_head


def install_dual_domain_detection_heads(model: torch.nn.Module) -> None:
    """Attach RGB/IR detection heads to the final YOLO26-Pose head."""
    head = model.model[-1]
    if "forward_head" in head.__dict__:
        delattr(head, "forward_head")
    if "_bee_original_forward_head" in head.__dict__:
        delattr(head, "_bee_original_forward_head")
    if not hasattr(head, "cv2_rgb"):
        head.cv2_rgb = copy.deepcopy(head.cv2)
        head.cv3_rgb = copy.deepcopy(head.cv3)
        head.cv2_ir = copy.deepcopy(head.cv2)
        head.cv3_ir = copy.deepcopy(head.cv3)
        head.bee_domain_ids = None


class DualDomainDecoupledPoseLoss(PoseLoss26):
    """YOLO26 Pose loss with det-only stage and det/pose mask separation."""

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
        zero = pred_kpts[..., :0].sum()
        if os.environ.get("BEEPOSETRACK_DET_ONLY", "0") == "1":
            return zero, zero, zero

        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

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


def init_dual_domain_criterion(model):
    return E2ELoss(model, DualDomainDecoupledPoseLoss) if getattr(model, "end2end", False) else DualDomainDecoupledPoseLoss(model)


class DualDomainPoseTrainer(PoseTrainer):
    """Single-frame YOLO26-Pose trainer with RGB/IR detection-head routing."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        install_dual_domain_detection_heads(model)
        model.__class__.init_criterion = init_dual_domain_criterion
        model.criterion = None
        return model

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        files = batch.get("im_file", [])
        domain_ids = torch.tensor([infer_domain_id(p) for p in files], device=self.device, dtype=torch.long)
        self.model.model[-1].bee_domain_ids = domain_ids
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/beeposetrack_y"))
    parser.add_argument("--name", default="y_aligned_single_frame_pose_dual_det_head")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--det-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_dual_domain_class_patch()
    old_det_only = os.environ.get("BEEPOSETRACK_DET_ONLY")
    os.environ["BEEPOSETRACK_DET_ONLY"] = "1" if args.det_only else "0"
    print(
        f"BeePoseTrack training mode: {'det-only' if args.det_only else 'det+pose-decoupled'}; "
        "detection heads: RGB/IR separated; pose head: shared",
        flush=True,
    )
    model = YOLO(args.model)
    try:
        model.train(
            trainer=DualDomainPoseTrainer,
            task="pose",
            data=str(args.data),
            project=str(args.project),
            name=args.name,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            optimizer="MuSGD",
            lr0=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_bias_lr=0.0,
            nbs=args.nbs,
            amp=True,
            seed=args.seed,
            deterministic=True,
            max_det=768,
            close_mosaic=0,
            mosaic=0.0,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            erasing=0.0,
            degrees=10.0,
            translate=0.10,
            scale=0.30,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            fliplr=0.5,
            flipud=0.0,
            auto_augment=None,
            patience=0,
            save=True,
            save_period=10,
            val=True,
            exist_ok=True,
            resume=args.resume,
        )
    finally:
        if old_det_only is None:
            os.environ.pop("BEEPOSETRACK_DET_ONLY", None)
        else:
            os.environ["BEEPOSETRACK_DET_ONLY"] = old_det_only


if __name__ == "__main__":
    main()
