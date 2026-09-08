#!/usr/bin/env python3
"""Train single-frame YOLO26-Pose with decoupled det/pose supervision masks.

This keeps the aligned baseline architecture and parameters unchanged. Instances
with bbox but no complete head/tail remain detection positives, while pose,
keypoint-objectness, and RLE losses are computed only for instances with at
least one visible keypoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.utils.loss import E2ELoss, PoseLoss26
from ultralytics.utils.ops import xyxy2xywh


class DecoupledPoseLoss(PoseLoss26):
    """YOLO26 Pose loss that ignores det-only targets for all pose losses."""

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


def init_decoupled_criterion(model):
    return E2ELoss(model, DecoupledPoseLoss) if getattr(model, "end2end", False) else DecoupledPoseLoss(model)


class DecoupledPoseTrainer(PoseTrainer):
    """Plain single-frame pose trainer with only the decoupled loss swapped in."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.__class__.init_criterion = init_decoupled_criterion
        model.criterion = None
        return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--project", type=Path, default=Path("runs/beeposetrack_y"))
    parser.add_argument("--name", default="y_aligned_single_frame_pose_decoupled")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--nbs", type=int, default=16, help="Nominal/effective batch size for gradient accumulation.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--det-only",
        action="store_true",
        help="Train only bbox/class/DFL losses; force pose, keypoint-objectness, and RLE losses to zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    old_det_only = os.environ.get("BEEPOSETRACK_DET_ONLY")
    if args.det_only:
        os.environ["BEEPOSETRACK_DET_ONLY"] = "1"
    else:
        os.environ["BEEPOSETRACK_DET_ONLY"] = "0"
    print(f"BeePoseTrack training mode: {'det-only' if args.det_only else 'det+pose-decoupled'}", flush=True)
    model = YOLO(args.model)
    try:
        model.train(
            trainer=DecoupledPoseTrainer,
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
