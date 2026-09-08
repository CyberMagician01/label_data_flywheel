"""Ultralytics PoseTrainer with manifest-driven five-frame input tensors."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.data.build import build_dataloader
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import torch_distributed_zero_first

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import FiveFrameManifestIndex, read_rgb_or_ir
from data.dataset import frame_statistics
from losses.route_best_loss import RouteBestE2ELoss
from models.route_context import begin_route_context
from models.yolo26_bee_pose import BeeDensityTap, BeeRouteBlock, FiveFrameInputAdapter


def init_route_best_criterion(model):
    """Pickle-safe criterion factory installed on the model class during training."""

    return RouteBestE2ELoss(model)


class RouteBestPoseTrainer(PoseTrainer):
    """Single-frame Route-Best trainer with domain context and auxiliary loss."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        route_config = Path(os.environ.get("BEEPOSETRACK_ROUTE_CONFIG", ""))
        self.route_config = yaml.safe_load(route_config.read_text(encoding="utf-8")) if route_config.exists() else {}
        self.clip_len = int(os.environ.get("BEEPOSETRACK_CLIP_LEN", "5"))
        self.temporal_stride = int(os.environ.get("BEEPOSETRACK_TEMPORAL_STRIDE", "1"))
        manifest_value = os.environ.get("BEEPOSETRACK_MANIFEST", "")
        manifest = Path(manifest_value) if manifest_value else None
        self.frame_indices = {}
        self.frame_by_path = {}
        if manifest is not None and manifest.is_file():
            for split in ("train", "test"):
                index = FiveFrameManifestIndex(manifest, split, self.clip_len, self.temporal_stride)
                self.frame_indices[split] = index
                for frame in index.frames:
                    self.frame_by_path[str(frame.image_path.resolve())] = (index, frame)
        self.loss_names = ("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "rle_loss", "structure_loss", "density_loss", "prototype_loss", "temporal_loss")

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model.route_config = getattr(self, "route_config", {})
        model.route_epoch = 0
        model.__class__.init_criterion = init_route_best_criterion
        model.criterion = None
        return model

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)
        self._apply_route_lr_multipliers(model, optimizer, lr)
        return optimizer

    def _apply_route_lr_multipliers(self, model, optimizer, base_lr: float) -> None:
        param_scales = self._route_param_scales(model)
        if not param_scales:
            return
        new_groups = []
        summary = {"route_new": 0, "early_backbone": 0, "base": 0}
        for group in optimizer.param_groups:
            buckets = {"route_new": [], "early_backbone": [], "base": []}
            for param in group["params"]:
                scale, bucket = param_scales.get(id(param), (1.0, "base"))
                buckets[bucket].append(param)
            for bucket, params in buckets.items():
                if not params:
                    continue
                scale = 2.0 if bucket == "route_new" else 0.5 if bucket == "early_backbone" else 1.0
                cloned = {k: v for k, v in group.items() if k != "params"}
                cloned["params"] = params
                cloned["lr"] = base_lr * scale
                cloned["route_lr_group"] = bucket
                new_groups.append(cloned)
                summary[bucket] += sum(p.numel() for p in params)
        optimizer.param_groups[:] = new_groups
        print(
            "RouteBest LR groups "
            f"route_new={summary['route_new']}@{base_lr * 2.0:g}, "
            f"early_backbone={summary['early_backbone']}@{base_lr * 0.5:g}, "
            f"base={summary['base']}@{base_lr:g}; grad_clip=10.0",
            flush=True,
        )

    @staticmethod
    def _route_param_scales(model) -> dict[int, tuple[float, str]]:
        scales: dict[int, tuple[float, str]] = {}
        new_module_types = (BeeRouteBlock, BeeDensityTap, FiveFrameInputAdapter)
        for module_name, module in model.named_modules():
            if isinstance(module, new_module_types):
                for param in module.parameters(recurse=True):
                    scales[id(param)] = (2.0, "route_new")
                continue
            if RouteBestPoseTrainer._is_early_backbone_module(module_name):
                for param in module.parameters(recurse=False):
                    scales.setdefault(id(param), (0.5, "early_backbone"))
        return scales

    @staticmethod
    def _is_early_backbone_module(module_name: str) -> bool:
        parts = module_name.split(".")
        if len(parts) < 2 or parts[0] != "model":
            return False
        try:
            layer_index = int(parts[1])
        except ValueError:
            return False
        return layer_index in {0, 1, 2}

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        self._apply_balanced_domain_order(dataset)
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers,
            shuffle=False,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def _apply_balanced_domain_order(self, dataset) -> None:
        if not self.frame_by_path or not hasattr(dataset, "im_files") or not hasattr(dataset, "labels"):
            return
        buckets = {"RGB": [], "IR": []}
        for idx, path in enumerate(dataset.im_files):
            item = self.frame_by_path.get(str(Path(path).resolve()))
            if item is None:
                continue
            _, frame = item
            domain = frame.domain.upper()
            if domain in buckets:
                buckets[domain].append(idx)
        if not buckets["RGB"] or not buckets["IR"]:
            return
        ordered = []
        quota = max(len(buckets["RGB"]), len(buckets["IR"]))
        for i in range(quota):
            ordered.append(buckets["RGB"][i % len(buckets["RGB"])])
            ordered.append(buckets["IR"][i % len(buckets["IR"])])
        dataset.im_files = [dataset.im_files[i] for i in ordered]
        dataset.labels = [dataset.labels[i] for i in ordered]

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        if getattr(self, "model", None) is not None:
            self.model.route_epoch = int(getattr(self, "epoch", 0) or 0)
        self._begin_single_frame_context(batch)
        return batch

    def _begin_single_frame_context(self, batch: dict) -> None:
        files = batch.get("im_file") or batch.get("im_files") or []
        domain_ids = []
        stats = []
        for file_name in files:
            item = self.frame_by_path.get(str(Path(file_name).resolve()))
            if item is None:
                domain_ids.append(0)
                stats.append(torch.zeros(6, device=self.device))
                continue
            _, frame = item
            image = read_rgb_or_ir(frame.image_path, frame.domain)
            domain_ids.append(1 if frame.domain.upper() == "IR" else 0)
            stats.append(torch.from_numpy(frame_statistics(image, frame.domain, len(frame.instances), 0.0)).to(self.device))
        if not stats:
            stats = [torch.zeros(6, device=self.device) for _ in range(batch["img"].shape[0])]
            domain_ids = [0 for _ in range(batch["img"].shape[0])]
        begin_route_context(
            clip=batch["img"][:, None],
            domain_ids=torch.tensor(domain_ids, device=self.device, dtype=torch.long),
            stats=torch.stack(stats, dim=0).float(),
        )


class FiveFramePoseTrainer(RouteBestPoseTrainer):
    """Build [B, 15, H, W] causal clip tensors before model forward."""

    def preprocess_batch(self, batch: dict) -> dict:
        batch = PoseTrainer.preprocess_batch(self, batch)
        if getattr(self, "model", None) is not None:
            self.model.route_epoch = int(getattr(self, "epoch", 0) or 0)
        if not self.frame_by_path:
            clip = batch["img"].repeat(1, self.clip_len, 1, 1)
            begin_route_context(clip=clip.view(clip.shape[0], self.clip_len, 3, clip.shape[-2], clip.shape[-1]))
            batch["img"] = clip
            return batch
        h, w = batch["img"].shape[-2:]
        clips = []
        domain_ids = []
        stats = []
        files = batch.get("im_file") or batch.get("im_files") or []
        for i, file_name in enumerate(files):
            item = self.frame_by_path.get(str(Path(file_name).resolve()))
            if item is None:
                current = batch["img"][i]
                clips.append(current.repeat(self.clip_len, 1, 1))
                domain_ids.append(0)
                stats.append(torch.zeros(6, device=self.device))
                continue
            frame_index, frame = item
            tensors = []
            clip_images = []
            for ref in frame_index.causal_clip(frame):
                img = read_rgb_or_ir(ref.image_path, ref.domain)
                clip_images.append(img)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                ten = torch.from_numpy(img).to(self.device, non_blocking=self.device.type == "cuda").permute(2, 0, 1).float() / 255.0
                tensors.append(ten)
            clips.append(torch.cat(tensors, dim=0))
            motion = 0.0
            if len(clip_images) > 1:
                a = cv2.cvtColor(clip_images[-1], cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
                b = cv2.cvtColor(clip_images[-2], cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
                if a.shape != b.shape:
                    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
                motion = float(abs(a - b).mean())
            domain_ids.append(1 if frame.domain.upper() == "IR" else 0)
            stats.append(torch.from_numpy(frame_statistics(clip_images[-1], frame.domain, len(frame.instances), motion)).to(self.device))
        if clips:
            batch["img"] = torch.stack(clips, dim=0)
            begin_route_context(
                clip=batch["img"].view(batch["img"].shape[0], self.clip_len, 3, batch["img"].shape[-2], batch["img"].shape[-1]),
                domain_ids=torch.tensor(domain_ids, device=self.device, dtype=torch.long),
                stats=torch.stack(stats, dim=0).float(),
            )
        else:
            batch["img"] = batch["img"].repeat(1, self.clip_len, 1, 1)
            begin_route_context(clip=batch["img"].view(batch["img"].shape[0], self.clip_len, 3, batch["img"].shape[-2], batch["img"].shape[-1]))
        return batch
