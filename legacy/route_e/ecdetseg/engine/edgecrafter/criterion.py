"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RF-DETR (https://github.com/roboflow/rf-detr)
Copyright (c) 2025 Roboflow. All Rights Reserved.
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .segmentation_head import (get_uncertain_point_coords_with_randomness,
                                point_sample)
from .utils import bbox2distance


@register()
class ECCriterion(nn.Module):
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        mask_point_sample_ratio=None,
        loss_warmup_epochs=0,
        one2many_topk=0,
        one2many_loss_weight=0.1,
        use_adaptive_density_target=False,
        density_scale_bandwidth=0.15,
        density_neighbor_bandwidth=0.25,
        density_min_sigma=1.0,
        density_max_sigma=12.0,
        density_foreground_weight=8.0,
        ignore_query_iou_threshold=0.3,
        domain_foreground_quality_threshold=0.7,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.mal_alpha = mal_alpha
        self.use_uni_set = use_uni_set
        self.mask_point_sample_ratio = matcher.mask_point_sample_ratio
        self.loss_warmup_epochs = loss_warmup_epochs
        self.current_epoch = 0
        self.one2many_topk = one2many_topk
        self.base_one2many_topk = int(one2many_topk)
        self.one2many_loss_weight = one2many_loss_weight
        self.use_adaptive_density_target = bool(use_adaptive_density_target)
        self.density_scale_bandwidth = float(density_scale_bandwidth)
        self.density_neighbor_bandwidth = float(density_neighbor_bandwidth)
        self.density_min_sigma = float(density_min_sigma)
        self.density_max_sigma = float(density_max_sigma)
        self.density_foreground_weight = float(density_foreground_weight)
        self.ignore_query_iou_threshold = float(ignore_query_iou_threshold)
        self.domain_foreground_quality_threshold = float(
            domain_foreground_quality_threshold
        )
        self.current_stage = None
        self.stage_progress = 0.0
        self.stage_loss_multipliers = {}

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_stage_context(self, stage, progress, pose_match_multiplier,
                          recall_debt=None):
        self.current_stage = str(stage)
        self.stage_progress = float(progress)
        if hasattr(self.matcher, 'set_pose_cost_multiplier'):
            self.matcher.set_pose_cost_multiplier(pose_match_multiplier)
        if stage == 'E-S0':
            self.one2many_topk = 1
        elif stage == 'E-S4':
            debt = max((float(value) for value in (recall_debt or {}).values()), default=0.0)
            self.one2many_topk = max(
                self.base_one2many_topk,
                min(self.base_one2many_topk + math.ceil(debt), self.base_one2many_topk * 2),
            )
        elif stage == 'E-S5':
            self.one2many_topk = self.base_one2many_topk
        else:
            self.one2many_topk = self.base_one2many_topk
        stage_index = int(str(stage).split('S')[-1])
        self.stage_loss_multipliers = {
            'pose': (
                self.stage_progress if stage == 'E-S0' else 1.0
            ),
            'structure_domain': (
                self.stage_progress if stage == 'E-S2'
                else float(stage_index > 2)
            ),
            'temporal_density_track': (
                self.stage_progress if stage == 'E-S3'
                else float(stage_index > 3)
            ),
            'consistency': float(stage in ('E-S3', 'E-S4')),
        }

    def _stage_loss_scale(self, name):
        if not self.stage_loss_multipliers:
            return 1.0
        if 'consistency' in name:
            return self.stage_loss_multipliers['consistency']
        if any(token in name for token in (
            'density', 'track_center', 'track_axis',
        )):
            return self.stage_loss_multipliers['temporal_density_track']
        if any(token in name for token in (
            'domain', 'prototype', 'pattern', 'hard_negative',
        )):
            return self.stage_loss_multipliers['structure_domain']
        if any(token in name for token in (
            'keypoint', 'pose_', 'direction', 'visibility', 'endpoint',
        )):
            return self.stage_loss_multipliers['pose']
        return 1.0

    @staticmethod
    def _valid_query_weight(outputs, reference):
        valid = outputs.get('pred_query_valid')
        if valid is None:
            return torch.ones(reference.shape[:2], dtype=reference.dtype, device=reference.device)
        if valid.shape != reference.shape[:2]:
            raise ValueError('pred_query_valid must have shape [batch, num_queries].')
        return valid.to(device=reference.device, dtype=reference.dtype)

    def _background_query_weight(self, outputs, targets, indices, reference):
        weight = self._valid_query_weight(outputs, reference)
        if 'pred_boxes' not in outputs:
            return weight
        for batch_index, target in enumerate(targets):
            ignored = torch.zeros(
                reference.shape[1], dtype=torch.bool, device=reference.device,
            )
            ignore_boxes = target.get('ignore_boxes')
            if ignore_boxes is not None and len(ignore_boxes):
                ignore_boxes = ignore_boxes.to(
                    device=reference.device, dtype=outputs['pred_boxes'].dtype,
                )
                overlap, _ = box_iou(
                    box_cxcywh_to_xyxy(outputs['pred_boxes'][batch_index]),
                    box_cxcywh_to_xyxy(ignore_boxes),
                )
                ignored |= overlap.amax(dim=1) >= self.ignore_query_iou_threshold
            ignore_mask = target.get('ignore_mask')
            if ignore_mask is not None:
                mask = torch.as_tensor(
                    ignore_mask, device=reference.device, dtype=reference.dtype,
                ).reshape(1, 1, *ignore_mask.shape[-2:])
                grid = outputs['pred_boxes'][batch_index, :, :2].mul(2).sub(1).view(
                    1, -1, 1, 2,
                )
                ignored |= F.grid_sample(
                    mask, grid, mode='bilinear', padding_mode='zeros', align_corners=False,
                ).view(-1) >= 0.5
            source_indices = indices[batch_index][0].to(reference.device)
            ignored[source_indices] = False
            weight[batch_index, ignored] = 0.0
        return weight

    @staticmethod
    def _matched_instance_weight(targets, indices, device, mode='base'):
        values = []
        structure_weights = []
        domain_ids = [
            int(target.get('domain_id', torch.zeros(1, device=device)).flatten()[0])
            for target in targets
        ]
        domain_image_counts = {
            domain_id: max(domain_ids.count(domain_id), 1)
            for domain_id in set(domain_ids)
        }
        for target, (_, target_indices) in zip(targets, indices):
            count = len(target['boxes'])
            pseudo = target.get('is_pseudo', torch.zeros(count, dtype=torch.bool, device=device))
            score = target.get('pseudo_score', torch.ones(count, device=device))
            weight = torch.where(pseudo, score, torch.ones_like(score)).to(torch.float32)
            if mode == 'pose':
                weight = weight * target.get('pose_quality', torch.ones(count, device=device))
            elif mode == 'track':
                weight = weight * target.get('trajectory_stability', torch.ones(count, device=device))
            values.append(weight[target_indices])
            matched_count = max(len(target_indices), 1)
            domain_id = int(target.get('domain_id', torch.zeros(1, device=device)).flatten()[0])
            structure_weights.append(torch.full(
                (len(target_indices),),
                1.0 / (matched_count * domain_image_counts[domain_id]),
                dtype=torch.float32,
                device=device,
            ))
        if not values:
            return torch.zeros(0, device=device)
        value = torch.cat(values).to(device)
        structure = torch.cat(structure_weights)
        structure = structure / structure.mean().clamp_min(1e-6)
        return value * structure

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(
            target_classes, num_classes=self.num_classes + 1
        )[..., :-1].to(src_logits.dtype)
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        query_weight = self._background_query_weight(
            outputs, targets, indices, src_logits,
        )
        query_weight[idx] = self._matched_instance_weight(
            targets, indices, src_logits.device
        ).to(query_weight.dtype)
        loss = (loss * query_weight.unsqueeze(-1)).sum() / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype) * self._matched_instance_weight(
            targets, indices, src_logits.device
        ).to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = (loss * self._background_query_weight(
            outputs, targets, indices, src_logits,
        ).unsqueeze(-1)).sum() / num_boxes
        return {'loss_vfl': loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype) * self._matched_instance_weight(
            targets, indices, src_logits.device
        ).to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        target_score = target_score.pow(self.gamma)
        if self.mal_alpha != None:
            weight = self.mal_alpha * pred_score.pow(self.gamma) * (1 - target) + target
        else:
            weight = pred_score.pow(self.gamma) * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = (loss * self._background_query_weight(
            outputs, targets, indices, src_logits,
        ).unsqueeze(-1)).sum() / num_boxes
        return {'loss_mal': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        instance_weight = self._matched_instance_weight(
            targets, indices, src_boxes.device
        ).to(src_boxes.dtype)
        losses['loss_bbox'] = (loss_bbox * instance_weight[:, None]).sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = (loss_giou * instance_weight).sum() / num_boxes

        return losses

    def loss_pose(self, outputs, targets, indices, num_boxes):
        """监督匹配查询的头尾坐标、方向和关键点可见性。"""
        if 'pred_keypoints' not in outputs:
            return {}

        # Detection-only images legitimately carry boxes without pose labels.
        # Materialize masked zero keypoints so every DDP rank emits the same
        # graph-connected pose-loss keys while contributing exactly zero pose
        # gradient for those instances.  This does not fabricate supervision:
        # pose_mask remains zero and the original target dictionaries are not
        # mutated.
        num_keypoints = outputs['pred_keypoints'].shape[-2]
        pose_targets = []
        for target in targets:
            if 'keypoints' in target:
                pose_targets.append(target)
                continue
            pose_target = dict(target)
            pose_target['keypoints'] = target['boxes'].new_zeros(
                (len(target['boxes']), num_keypoints, 3)
            )
            pose_target['pose_mask'] = target['boxes'].new_zeros(
                len(target['boxes'])
            )
            pose_targets.append(pose_target)
        targets = pose_targets

        idx = self._get_src_permutation_idx(indices)
        pred_keypoints = outputs['pred_keypoints'][idx]
        pred_visibility = outputs['pred_visibility'][idx]
        target_keypoints = torch.cat([
            target['keypoints'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0)
        pose_mask = torch.cat([
            ((target['pose_state'] > 0) if 'pose_state' in target else
             target.get('pose_mask', torch.ones(len(target['boxes']), device=target['boxes'].device)).bool())[target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0).to(pred_keypoints.dtype)

        target_xy = target_keypoints[..., :2]
        target_boxes = torch.cat([
            target['boxes'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0)
        coordinate_visible = (target_keypoints[..., 2] > 0).to(pred_keypoints.dtype)
        target_visible = (target_keypoints[..., 2] == 2).to(pred_keypoints.dtype)
        pose_quality = self._matched_instance_weight(
            targets, indices, pred_keypoints.device, mode='pose'
        ).to(pred_keypoints.dtype)
        coordinate_weight = coordinate_visible * pose_mask[:, None] * pose_quality[:, None]
        coordinate_denominator = (2.0 * coordinate_weight.sum()).clamp_min(1.0)
        loss_keypoint = (
            (pred_keypoints - target_xy).abs() * coordinate_weight[..., None]
        ).sum() / coordinate_denominator

        bbox_diagonal = target_boxes[..., 2:].norm(dim=-1).clamp_min(1e-4)
        squared_distance = (pred_keypoints - target_xy).pow(2).sum(dim=-1)
        oks_error = 1.0 - torch.exp(
            -squared_distance / (2.0 * (0.10 * bbox_diagonal[:, None]).pow(2))
        )
        loss_pose_oks = (oks_error * coordinate_weight).sum() / coordinate_weight.sum().clamp_min(1.0)

        pred_boxes = outputs['pred_boxes'][idx]
        box_min = pred_boxes[..., :2] - 0.5 * pred_boxes[..., 2:]
        box_max = pred_boxes[..., :2] + 0.5 * pred_boxes[..., 2:]
        outside = F.relu(box_min[:, None] - pred_keypoints) + F.relu(pred_keypoints - box_max[:, None])
        loss_pose_box = (outside * coordinate_weight[..., None]).sum() / coordinate_denominator

        if pred_keypoints.shape[-2] >= 2:
            pred_axis = F.normalize(pred_keypoints[:, 1] - pred_keypoints[:, 0], dim=-1, eps=1e-6)
            target_axis = F.normalize(target_xy[:, 1] - target_xy[:, 0], dim=-1, eps=1e-6)
            direction_weight = (
                coordinate_visible[:, 0] * coordinate_visible[:, 1]
                * pose_mask * pose_quality
            )
            loss_direction = ((1.0 - (pred_axis * target_axis).sum(-1)) * direction_weight).sum()
            loss_direction = loss_direction / direction_weight.sum().clamp_min(1.0)
            pred_length = (pred_keypoints[:, 1] - pred_keypoints[:, 0]).norm(dim=-1)
            target_length = (target_xy[:, 1] - target_xy[:, 0]).norm(dim=-1)
            pred_length_normalized = pred_length / bbox_diagonal
            target_length_normalized = target_length / bbox_diagonal
            if all(
                'pose_length_median' in target and 'pose_length_mad' in target
                for target in targets
            ):
                length_median = torch.cat([
                    target['pose_length_median'][target_indices]
                    for target, (_, target_indices) in zip(targets, indices)
                ]).to(pred_length)
                length_mad = torch.cat([
                    target['pose_length_mad'][target_indices]
                    for target, (_, target_indices) in zip(targets, indices)
                ]).to(pred_length).clamp_min(1e-3)
                length_error = (
                    (pred_length_normalized - length_median) / length_mad
                    - (target_length_normalized - length_median) / length_mad
                ).abs()
            else:
                length_error = (pred_length_normalized - target_length_normalized).abs()
            loss_pose_length = (length_error * direction_weight).sum() / direction_weight.sum().clamp_min(1.0)
        else:
            loss_direction = pred_keypoints.sum() * 0.0
            loss_pose_length = pred_keypoints.sum() * 0.0

        visibility_weight = (
            pose_mask[:, None] * pose_quality[:, None]
        ).expand_as(pred_visibility)
        loss_visibility = F.binary_cross_entropy_with_logits(
            pred_visibility,
            target_visible,
            reduction='none',
        )
        loss_visibility = (loss_visibility * visibility_weight).sum() / visibility_weight.sum().clamp_min(1.0)

        loss_endpoint_uncertainty = pred_keypoints.sum() * 0.0
        if 'pred_endpoint_uncertainty' in outputs:
            endpoint_uncertainty = outputs['pred_endpoint_uncertainty'][idx].clamp_min(1e-4)
            endpoint_error = (pred_keypoints - target_xy).abs().mean(dim=-1)
            endpoint_nll = endpoint_error / endpoint_uncertainty + endpoint_uncertainty.log()
            loss_endpoint_uncertainty = (
                endpoint_nll * coordinate_weight
            ).sum() / coordinate_weight.sum().clamp_min(1.0)

        return {
            'loss_keypoint': loss_keypoint,
            'loss_pose_oks': loss_pose_oks,
            'loss_pose_box': loss_pose_box,
            'loss_pose_length': loss_pose_length,
            'loss_direction': loss_direction,
            'loss_visibility': loss_visibility,
            'loss_endpoint_uncertainty': loss_endpoint_uncertainty,
        }

    def _build_adaptive_density_target(self, prediction, targets):
        batch_size, _, height, width = prediction.shape
        target_density = prediction.new_zeros((batch_size, 1, height, width))
        target_counts = prediction.new_zeros(batch_size)
        for batch_idx, target in enumerate(targets):
            boxes = target['boxes']
            pseudo = target.get(
                'is_pseudo',
                torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device),
            )
            pseudo_score = target.get(
                'pseudo_score', torch.ones(len(boxes), device=boxes.device),
            )
            instance_weight = torch.where(
                pseudo, pseudo_score, torch.ones_like(pseudo_score),
            ).to(device=prediction.device, dtype=prediction.dtype)
            target_counts[batch_idx] = instance_weight.sum()
            if len(boxes) == 0:
                continue
            boxes = boxes.to(device=prediction.device, dtype=prediction.dtype)
            centers = boxes[:, :2].clamp(0, 1) * boxes.new_tensor([width, height])
            box_scale = (
                boxes[:, 2] * width * boxes[:, 3] * height
            ).clamp_min(1e-6).sqrt()
            scale_sigma = self.density_scale_bandwidth * box_scale
            if len(boxes) > 1:
                distances = torch.cdist(centers, centers)
                distances.fill_diagonal_(torch.inf)
                neighbor_sigma = self.density_neighbor_bandwidth * distances.amin(dim=1)
                sigma = torch.sqrt(scale_sigma.clamp_min(1e-6) * neighbor_sigma.clamp_min(1e-6))
            else:
                sigma = scale_sigma
            sigma = sigma.clamp(self.density_min_sigma, self.density_max_sigma)

            for center, current_sigma, weight in zip(centers, sigma, instance_weight):
                radius = max(1, int(torch.ceil(3.0 * current_sigma).item()))
                center_x, center_y = center
                left = max(0, int(torch.floor(center_x).item()) - radius)
                right = min(width, int(torch.floor(center_x).item()) + radius + 1)
                top = max(0, int(torch.floor(center_y).item()) - radius)
                bottom = min(height, int(torch.floor(center_y).item()) + radius + 1)
                grid_y = torch.arange(top, bottom, device=prediction.device, dtype=prediction.dtype)
                grid_x = torch.arange(left, right, device=prediction.device, dtype=prediction.dtype)
                gaussian = torch.exp(-(
                    (grid_x[None] + 0.5 - center_x).square()
                    + (grid_y[:, None] + 0.5 - center_y).square()
                ) / (2.0 * current_sigma.square()))
                gaussian = gaussian / gaussian.sum().clamp_min(1e-8)
                target_density[batch_idx, 0, top:bottom, left:right] += weight * gaussian
        return target_density, target_counts

    def loss_density(self, outputs, targets, indices, num_boxes):
        if 'pred_density' not in outputs:
            return {}

        prediction = outputs['pred_density']
        batch_size, _, height, width = prediction.shape
        if self.use_adaptive_density_target:
            target_density, target_counts = self._build_adaptive_density_target(
                prediction, targets,
            )
            pred_counts = prediction.flatten(1).sum(1)
            target_peak = target_density.flatten(1).amax(dim=1).clamp_min(1e-8)
            foreground = target_density / target_peak[:, None, None, None]
            pixel_weight = 1.0 + self.density_foreground_weight * foreground
            map_error = F.smooth_l1_loss(
                prediction, target_density, reduction='none', beta=0.05,
            )
            loss_density_map = (
                map_error * pixel_weight
            ).flatten(1).sum(1) / pixel_weight.flatten(1).sum(1).clamp_min(1.0)
            loss_density_count = F.smooth_l1_loss(
                torch.log1p(pred_counts), torch.log1p(target_counts), reduction='mean',
            )
            return {
                'loss_density_map': loss_density_map.mean(),
                'loss_density_count': loss_density_count,
            }
        target_density = prediction.new_zeros((batch_size, 1, height, width))
        target_counts = prediction.new_zeros(batch_size)

        for batch_idx, target in enumerate(targets):
            boxes = target['boxes']
            pseudo = target.get('is_pseudo', torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device))
            pseudo_score = target.get('pseudo_score', torch.ones(len(boxes), device=boxes.device))
            instance_weight = torch.where(pseudo, pseudo_score, torch.ones_like(pseudo_score)).to(prediction.dtype)
            target_counts[batch_idx] = instance_weight.sum()
            if len(boxes) == 0:
                continue
            centers = boxes[:, :2].clamp(0, 1)
            x_index = (centers[:, 0] * (width - 1)).round().long()
            y_index = (centers[:, 1] * (height - 1)).round().long()
            flat_index = y_index * width + x_index
            target_density[batch_idx, 0].view(-1).scatter_add_(
                0, flat_index, instance_weight
            )

        gaussian_axis = torch.arange(-4, 5, device=prediction.device, dtype=prediction.dtype)
        gaussian = torch.exp(-(gaussian_axis[:, None] ** 2 + gaussian_axis[None, :] ** 2) / 8.0)
        gaussian = (gaussian / gaussian.sum()).view(1, 1, 9, 9)
        target_density = F.conv2d(target_density, gaussian, padding=4)
        target_sum = target_density.flatten(1).sum(1).clamp_min(1e-6)
        target_density = target_density * (target_counts / target_sum)[:, None, None, None]

        pred_counts = prediction.flatten(1).sum(1)
        pred_distribution = prediction / pred_counts.clamp_min(1e-6)[:, None, None, None]
        target_distribution = target_density / target_counts.clamp_min(1.0)[:, None, None, None]
        loss_density_map = (pred_distribution - target_distribution).abs().flatten(1).sum(1).mean()
        loss_density_count = ((pred_counts - target_counts).abs() / (target_counts + 1.0)).mean()
        return {
            'loss_density_map': loss_density_map,
            'loss_density_count': loss_density_count,
        }

    def loss_domain(self, outputs, targets, indices, num_boxes):
        losses = {}
        domain_device = next(iter(outputs.values())).device
        domain_targets = torch.cat([
            target.get('domain_id', torch.zeros(1, dtype=torch.long, device=domain_device))
            for target in targets
        ]).long().to(domain_device)
        if 'pred_domain_logits' in outputs:
            losses['loss_domain'] = F.cross_entropy(
                outputs['pred_domain_logits'], domain_targets,
            )
        if 'pred_domain_features' in outputs and 'domain_prototypes' in outputs:
            # Prototype similarities are small reductions and should remain
            # FP32 under AMP. This also keeps learned FP32 prototypes
            # compatible with FP16/BF16 activation features.
            features = F.normalize(outputs['pred_domain_features'].float(), dim=-1)
            prototypes = F.normalize(outputs['domain_prototypes'].float(), dim=-1)
            losses['loss_domain_route_prototype'] = F.cross_entropy(
                features @ prototypes.t() / 0.1, domain_targets
            )
        if (
            'pred_query_features' in outputs
            and 'domain_prototypes' in outputs
            and 'shared_bee_prototype' in outputs
        ):
            idx = self._get_src_permutation_idx(indices)
            if idx[0].numel() > 0:
                query_features = F.normalize(
                    outputs['pred_query_features'][idx].float(), dim=-1,
                )
                prototypes = F.normalize(outputs['domain_prototypes'].float(), dim=-1)
                query_domains = domain_targets[idx[0]]
                quality = (
                    outputs['pred_quality'][idx].sigmoid().detach()
                    if 'pred_quality' in outputs else torch.ones_like(query_domains, dtype=query_features.dtype)
                )
                prototype_loss = F.cross_entropy(
                    query_features @ prototypes.t() / 0.1,
                    query_domains,
                    reduction='none',
                )
                losses['loss_domain_prototype'] = (
                    prototype_loss * quality
                ).sum() / quality.sum().clamp_min(1.0)
                shared = F.normalize(
                    outputs['shared_bee_prototype'].float(), dim=-1,
                )
                losses['loss_shared_bee_prototype'] = (
                    (1.0 - query_features @ shared) * quality
                ).sum() / quality.sum().clamp_min(1.0)
        if (
            'pred_domain_p3_features' in outputs
            and 'shared_bee_prototype' in outputs
        ):
            feature_map = outputs['pred_domain_p3_features']
            rois, roi_weights, roi_images, roi_domains = [], [], [], []
            height, width = feature_map.shape[-2:]
            for batch_index, (target, (_, target_indices)) in enumerate(
                zip(targets, indices)
            ):
                if not len(target_indices):
                    continue
                target_indices = target_indices.to(target['boxes'].device)
                count = len(target['boxes'])
                pseudo = target.get(
                    'is_pseudo',
                    torch.zeros(count, dtype=torch.bool, device=target['boxes'].device),
                ).bool()[target_indices]
                quality = torch.stack([
                    target.get(
                        field, torch.ones(count, device=target['boxes'].device),
                    )[target_indices].float()
                    for field in (
                        'inter_group_quality', 'intra_group_quality',
                        'hierarchy_quality',
                    )
                ]).amin(dim=0)
                keep = (~pseudo) & (
                    quality >= self.domain_foreground_quality_threshold
                )
                if not bool(keep.any()):
                    continue
                boxes = box_cxcywh_to_xyxy(
                    target['boxes'][target_indices][keep].to(feature_map)
                )
                boxes = boxes * boxes.new_tensor([width, height, width, height])
                batch_column = boxes.new_full((len(boxes), 1), batch_index)
                rois.append(torch.cat((batch_column, boxes), dim=1))
                roi_weights.append(quality[keep].to(feature_map))
                roi_images.append(torch.full(
                    (int(keep.sum()),), batch_index,
                    dtype=torch.long, device=feature_map.device,
                ))
                roi_domains.append(torch.full(
                    (int(keep.sum()),), int(domain_targets[batch_index]),
                    dtype=torch.long, device=feature_map.device,
                ))
            if rois:
                pooled = torchvision.ops.roi_align(
                    feature_map, torch.cat(rois), output_size=(3, 3),
                    spatial_scale=1.0, sampling_ratio=-1, aligned=True,
                ).mean(dim=(-2, -1))
                pooled = F.normalize(pooled.float(), dim=-1)
                shared = F.normalize(
                    outputs['shared_bee_prototype'].float(), dim=-1,
                )
                item_loss = 1.0 - pooled @ shared
                weights = torch.cat(roi_weights)
                image_ids = torch.cat(roi_images)
                domains = torch.cat(roi_domains)
                image_losses = []
                image_domains = []
                for image_id in image_ids.unique(sorted=True):
                    mask = image_ids == image_id
                    image_losses.append(
                        (item_loss[mask] * weights[mask]).sum()
                        / weights[mask].sum().clamp_min(1e-6)
                    )
                    image_domains.append(domains[mask][0])
                image_losses = torch.stack(image_losses)
                image_domains = torch.stack(image_domains)
                domain_losses = [
                    image_losses[image_domains == domain].mean()
                    for domain in image_domains.unique(sorted=True)
                ]
                losses['loss_shared_bee_prototype'] = torch.stack(
                    domain_losses
                ).mean()
            else:
                losses['loss_shared_bee_prototype'] = (
                    feature_map.sum() + outputs['shared_bee_prototype'].sum()
                ) * 0.0
        return losses

    def loss_track(self, outputs, targets, indices, num_boxes):
        """Supervise displacement, scale change and oriented axis; never classify IDs."""
        if 'pred_track_geometry' not in outputs or not all('track_geometry' in target for target in targets):
            return {}
        idx = self._get_src_permutation_idx(indices)
        prediction = outputs['pred_track_geometry'][idx]
        target_geometry = torch.cat([
            target['track_geometry'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0)
        mask = torch.cat([
            target.get(
                'track_geometry_mask',
                torch.ones(len(target['boxes']), device=target['boxes'].device),
            )[target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ], dim=0).to(prediction.dtype)
        mask = mask * self._matched_instance_weight(
            targets, indices, prediction.device, mode='track'
        ).to(prediction.dtype)
        denominator = mask.sum().clamp_min(1.0)
        center_scale = (
            F.smooth_l1_loss(prediction[:, :4], target_geometry[:, :4], reduction='none').mean(-1)
            * mask
        ).sum() / denominator
        pred_axis = F.normalize(prediction[:, 4:6], dim=-1, eps=1e-6)
        target_axis = F.normalize(target_geometry[:, 4:6], dim=-1, eps=1e-6)
        axis = ((1.0 - (pred_axis * target_axis).sum(-1)) * mask).sum() / denominator
        return {
            'loss_track_center_scale': center_scale,
            'loss_track_axis': axis,
        }

    def loss_ir_hard_negative(self, outputs, targets, indices, num_boxes):
        if (
            'pred_ir_hard_negative_prior' not in outputs
            or 'query_initial_references' not in outputs
        ):
            return {}
        prior = outputs['pred_ir_hard_negative_prior']
        centers = outputs['query_initial_references'][..., :2].clamp(0, 1)
        grid = centers.mul(2).sub(1).unsqueeze(2)
        weights = F.grid_sample(
            prior, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        ).squeeze(1).squeeze(-1)
        matched = torch.zeros_like(weights, dtype=torch.bool)
        for batch_index, (source_indices, _) in enumerate(indices):
            matched[batch_index, source_indices] = True
        valid = outputs.get('pred_query_valid', torch.ones_like(matched)).bool()
        weights = weights * (~matched & valid).to(weights.dtype)
        confidence = outputs['pred_logits'].amax(dim=-1)
        loss = F.binary_cross_entropy_with_logits(
            confidence, torch.zeros_like(confidence), reduction='none'
        )
        return {
            'loss_ir_hard_negative': (loss * weights).sum() / weights.sum().clamp_min(1.0)
        }

    def loss_quality(self, outputs, targets, indices, num_boxes):
        """Predict localization/pose quality without changing set matching."""
        if 'pred_quality' not in outputs:
            return {}

        quality_logits = outputs['pred_quality']
        target_quality = torch.zeros_like(quality_logits)
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() > 0:
            pred_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([
                target['boxes'][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ], dim=0)
            iou = torch.diag(box_iou(
                box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes)
            )[0]).detach().clamp(0, 1)
            target_labels = torch.cat([
                target['labels'][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ], dim=0)
            matched_class_logits = outputs['pred_logits'][idx]
            classification_correctness = (
                (matched_class_logits.argmax(dim=-1) == target_labels).to(iou.dtype)
                if matched_class_logits.shape[-1] > 1 else torch.ones_like(iou)
            ).detach()

            if 'pred_keypoints' in outputs and all('keypoints' in target for target in targets):
                pred_keypoints = outputs['pred_keypoints'][idx]
                target_keypoints = torch.cat([
                    target['keypoints'][target_indices]
                    for target, (_, target_indices) in zip(targets, indices)
                ], dim=0)
                pose_mask = torch.cat([
                    ((target['pose_state'] > 0) if 'pose_state' in target else
                     target.get('pose_mask', torch.ones(len(target['boxes']), device=target['boxes'].device)).bool())[target_indices]
                    for target, (_, target_indices) in zip(targets, indices)
                ], dim=0).to(pred_keypoints.dtype)
                visible = (target_keypoints[..., 2] > 0).to(pred_keypoints.dtype)
                target_diagonal = target_boxes[..., 2:].norm(dim=-1).clamp_min(1e-4)
                squared_distance = (
                    pred_keypoints - target_keypoints[..., :2]
                ).square().sum(dim=-1)
                endpoint_oks = torch.exp(
                    -squared_distance
                    / (2.0 * (0.10 * target_diagonal[:, None]).square())
                )
                visible_count = visible.sum(dim=-1)
                endpoint_oks = (
                    endpoint_oks * visible
                ).sum(dim=-1) / visible_count.clamp_min(1.0)
                endpoint_oks = torch.where(
                    visible_count > 0, endpoint_oks, torch.ones_like(endpoint_oks),
                ).detach()
                iou = iou * torch.where(
                    pose_mask > 0, endpoint_oks, torch.ones_like(endpoint_oks),
                )

            # A focal-loss target is a probability and must stay in [0, 1].
            # Domain/instance balancing is a loss weight, not part of the
            # target; putting it here can make BCE targets exceed one and
            # drive the quality logit toward +infinity with a negative loss.
            matched_quality = (
                classification_correctness * iou
            ).to(target_quality.dtype).clamp(0.0, 1.0)
            if not torch.isfinite(matched_quality).all():
                raise FloatingPointError('Non-finite matched quality target.')
            target_quality[idx] = matched_quality

        loss = torchvision.ops.sigmoid_focal_loss(
            quality_logits,
            target_quality,
            alpha=0.25,
            gamma=2.0,
            reduction='none',
        )
        query_weight = self._background_query_weight(
            outputs, targets, indices, quality_logits,
        )
        if idx[0].numel() > 0:
            matched_weight = self._matched_instance_weight(
                targets, indices, quality_logits.device,
            ).to(query_weight.dtype)
            if not torch.isfinite(matched_weight).all() or bool((matched_weight < 0).any()):
                raise FloatingPointError('Invalid matched quality loss weight.')
            query_weight[idx] = matched_weight
        loss = (loss * query_weight).sum() / num_boxes
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite quality loss.')
        class_confidence = outputs['pred_logits'].sigmoid().amax(dim=-1).detach()
        consistency = F.smooth_l1_loss(
            quality_logits.sigmoid(), class_confidence, reduction='none'
        )
        consistency = (
            consistency * query_weight
        ).sum() / query_weight.sum().clamp_min(1.0)
        return {
            'loss_quality': loss,
            'loss_quality_consistency': consistency,
        }

    def loss_consistency(self, outputs, targets, indices, num_boxes):
        """EMA-teacher soft set consistency on matched pseudo instances."""
        if 'pred_logits' not in outputs:
            return {}
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = outputs['pred_logits'].sum() * 0.0
            return {
                'loss_consistency_class': zero,
                'loss_consistency_box': zero,
                'loss_consistency_pose': zero,
            }
        pseudo = torch.cat([
            target.get(
                'is_pseudo',
                torch.zeros(
                    len(target['boxes']), dtype=torch.bool,
                    device=target['boxes'].device,
                ),
            )[target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ]).bool().to(outputs['pred_logits'].device)
        if not bool(pseudo.any()):
            zero = outputs['pred_logits'].sum() * 0.0
            return {
                'loss_consistency_class': zero,
                'loss_consistency_box': zero,
                'loss_consistency_pose': zero,
            }
        score = torch.cat([
            target.get(
                'pseudo_score',
                torch.ones(len(target['boxes']), device=target['boxes'].device),
            )[target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ]).to(outputs['pred_logits']).clamp(0, 1)
        weight = score * pseudo.to(score.dtype)
        denominator = weight.sum().clamp_min(1e-6)

        student_logits = outputs['pred_logits'][idx]
        teacher_probability_parts = []
        for target, (_, target_indices) in zip(targets, indices):
            teacher_probability = target.get('teacher_class_probability')
            if teacher_probability is None:
                teacher_probability = F.one_hot(
                    target['labels'], num_classes=self.num_classes,
                ).to(student_logits.dtype)
            teacher_probability_parts.append(teacher_probability[target_indices])
        teacher_probability = torch.cat(teacher_probability_parts).to(student_logits)
        class_loss = F.binary_cross_entropy_with_logits(
            student_logits, teacher_probability, reduction='none',
        ).mean(dim=-1)
        class_loss = (class_loss * weight).sum() / denominator

        student_boxes = outputs['pred_boxes'][idx]
        teacher_boxes = torch.cat([
            target['boxes'][target_indices]
            for target, (_, target_indices) in zip(targets, indices)
        ]).to(student_boxes)
        box_loss = F.smooth_l1_loss(
            student_boxes, teacher_boxes, reduction='none',
        ).mean(dim=-1)
        box_loss = (box_loss * weight).sum() / denominator

        pose_loss = student_boxes.sum() * 0.0
        if 'pred_keypoints' in outputs and all('keypoints' in target for target in targets):
            student_keypoints = outputs['pred_keypoints'][idx]
            teacher_keypoints = torch.cat([
                target['keypoints'][target_indices]
                for target, (_, target_indices) in zip(targets, indices)
            ]).to(student_keypoints)
            coordinate_mask = (teacher_keypoints[..., 2] > 0).to(student_keypoints.dtype)
            pose_error = (
                (student_keypoints - teacher_keypoints[..., :2]).abs()
                * coordinate_mask[..., None]
            ).sum(dim=(-1, -2)) / (2.0 * coordinate_mask.sum(-1)).clamp_min(1.0)
            pose_loss = (pose_error * weight).sum() / denominator
        return {
            'loss_consistency_class': class_loss,
            'loss_consistency_box': box_loss,
            'loss_consistency_pose': pose_loss,
        }

    def loss_pattern(self, outputs, targets, indices, num_boxes):
        """Keep PaQ prototypes used without forcing each individual query uniform."""
        usage = outputs.get('pattern_usage')
        if usage is None:
            return {}
        uniform = torch.full_like(usage, 1.0 / usage.shape[-1])
        loss = F.kl_div(
            usage.clamp_min(1e-8).log(), uniform, reduction='batchmean'
        )
        return {'loss_pattern_balance': loss}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute BCE-with-logits and Dice losses for segmentation masks on matched pairs.
        Expects outputs to contain 'pred_masks' of shape [B, Q, H, W] and targets with key 'masks'.
        """
        assert 'pred_masks' in outputs, "pred_masks missing in model outputs"
        pred_masks = outputs['pred_masks']  # [B, Q, H, W]
        # gather matched prediction masks
        idx = self._get_src_permutation_idx(indices)
        src_masks = pred_masks[idx]  # [N, H, W]
        # handle no matches
        if src_masks.numel() == 0:
            return {
                'loss_mask_ce': src_masks.sum(),
                'loss_mask_dice': src_masks.sum(),
            }
        # gather matched target masks
        target_masks = torch.cat([t['masks'][j] for t, (_, j) in zip(targets, indices)], dim=0)  # [N, Ht, Wt]
        
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks.unsqueeze(1)
        target_masks = target_masks.unsqueeze(1).float()

        num_points = max(src_masks.shape[-2], src_masks.shape[-2] * src_masks.shape[-1] // self.mask_point_sample_ratio)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                num_points,
                3,
                0.75,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_ce": sigmoid_ce_loss_jit(point_logits, point_labels, num_boxes),
            "loss_mask_dice": dice_loss_jit(point_logits, point_labels, num_boxes),
        }

        del src_masks
        del target_masks
        return losses
    
    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
            and Decoupled Distillation Focal (DDF) Loss. """

        losses = {}
        if 'pred_corners' in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max+1))
            ref_points = outputs['ref_points'][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and 'is_dn' in outputs:
                        self.fgl_targets_dn= bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])
                if self.fgl_targets is None and 'is_dn' not in outputs:
                        self.fgl_targets = bbox2distance(ref_points, box_cxcywh_to_xyxy(target_boxes),
                                                        self.reg_max, outputs['reg_scale'], outputs['up'])

            target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets

            ious = torch.diag(box_iou(\
                        box_cxcywh_to_xyxy(outputs['pred_boxes'][idx]), box_cxcywh_to_xyxy(target_boxes))[0])
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
                pred_corners, target_corners, weight_right, weight_left, weight_targets, avg_factor=num_boxes)

            if 'teacher_corners' in outputs:
                pred_corners = outputs['pred_corners'].reshape(-1, (self.reg_max+1))
                target_corners = outputs['teacher_corners'].reshape(-1, (self.reg_max+1))
                if torch.equal(pred_corners, target_corners):
                    # Keep a graph-connected exact zero without allowing the
                    # FP16 reduction itself to overflow before multiplication.
                    losses['loss_ddf'] = pred_corners.sum(dtype=torch.float32) * 0.0
                else:
                    # DDF is a probability-space KL term.  Finite FP16 logits can
                    # still overflow during max subtraction (for example
                    # 65504 - (-65504)), producing -inf and then 0 * inf = NaN.
                    # Compute only this probability/KL path in FP32; targets,
                    # weights and gradients are otherwise unchanged.
                    pred_corners_kl = pred_corners.float()
                    target_corners_kl = target_corners.detach().float()
                    weight_targets_local = outputs['teacher_logits'].float().sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).float()
                    weight_targets_local = weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

                    loss_match_local = weight_targets_local * (T ** 2) * (nn.KLDivLoss(reduction='none')
                    (F.log_softmax(pred_corners_kl / T, dim=1), F.softmax(target_corners_kl / T, dim=1))).sum(-1)
                    if 'is_dn' not in outputs:
                        batch_scale = 8 / outputs['pred_boxes'].shape[0]  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (mask.sum() * batch_scale) ** 0.5, ((~mask).sum() * batch_scale) ** 0.5
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses['loss_ddf'] = (loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                        for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None
        
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'mal': self.loss_labels_mal,
            'local': self.loss_local,
            'masks': self.loss_masks,
            'pose': self.loss_pose,
            'density': self.loss_density,
            'domain': self.loss_domain,
            'quality': self.loss_quality,
            'consistency': self.loss_consistency,
            'pattern': self.loss_pattern,
            'track': self.loss_track,
            'ir_hard_negative': self.loss_ir_hard_negative,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_o2m, cached_indices_enc = [], [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                needs_o2m = i < len(outputs['aux_outputs']) and self.one2many_topk > 1
                match_result = self.matcher(
                    aux_outputs,
                    targets,
                    return_topk=self.one2many_topk if needs_o2m else False,
                )
                indices_aux = match_result['indices']
                cached_indices.append(indices_aux)
                if needs_o2m:
                    cached_indices_o2m.append(match_result['indices_o2m'])
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
        else:
            assert 'aux_outputs' in outputs, ''

        # Reduce both normalization counts in one collective. Packing two sums
        # does not change either value or the resulting loss normalization.
        num_boxes = sum(len(t["labels"]) for t in targets)
        normalizers = torch.as_tensor(
            [num_boxes_go, num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(normalizers)
        normalizers = torch.clamp(normalizers / get_world_size(), min=1)
        num_boxes_go, num_boxes = normalizers.tolist()

        # Compute all the requested losses, main loss
        losses = {}
        for loss in self.losses:
            use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
            indices_in = indices_go if use_uni_set else indices
            num_boxes_in = num_boxes_go if use_uni_set else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.one2many_topk > 1:
                    indices_o2m = cached_indices_o2m[i]
                    for loss in self.losses:
                        if loss in ('local', 'density', 'domain'):
                            continue
                        meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_o2m)
                        l_dict = self.get_loss(
                            loss,
                            aux_outputs,
                            targets,
                            indices_o2m,
                            num_boxes * self.one2many_topk,
                            **meta,
                        )
                        l_dict = {
                            key: value * self.weight_dict[key] * self.one2many_loss_weight
                            for key, value in l_dict.items() if key in self.weight_dict
                        }
                        losses.update({key + f'_o2m_{i}': value for key, value in l_dict.items()})

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                use_uni_set = self.use_uni_set and (loss in ['boxes', 'local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    use_uni_set = self.use_uni_set and (loss == 'boxes')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            if 'dn_positive_count' in outputs['dn_meta']:
                dn_num_boxes = torch.as_tensor(
                    [outputs['dn_meta']['dn_positive_count']],
                    dtype=torch.float,
                    device=next(iter(outputs.values())).device,
                )
                if is_dist_available_and_initialized():
                    torch.distributed.all_reduce(dn_num_boxes)
                dn_num_boxes = torch.clamp(
                    dn_num_boxes / get_world_size(), min=1,
                ).item()
            else:
                dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
                dn_num_boxes = 1 if dn_num_boxes == 0 else dn_num_boxes
            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses:      # only work for local loss
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if self.loss_warmup_epochs > 0:
            warmup_scale = min((self.current_epoch + 1) / self.loss_warmup_epochs, 1.0)
            warmup_names = (
                'keypoint', 'direction', 'visibility', 'density', 'domain',
                'quality', 'pattern', 'prototype', 'track', 'hard_negative',
                'consistency',
            )
            losses = {
                key: value * warmup_scale if any(name in key for name in warmup_names) else value
                for key, value in losses.items()
            }

        losses = {
            key: value * self._stage_loss_scale(key)
            for key, value in losses.items()
        }

        # Invalid optimization signals must propagate to the engine's finite
        # loss gate instead of being silently replaced by zeros.
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', 'mal'):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx = dn_meta["dn_positive_idx"]
        if 'dn_positive_target_idx' in dn_meta:
            return [
                (positive_idx, target_idx)
                for positive_idx, target_idx in zip(
                    dn_positive_idx, dn_meta['dn_positive_target_idx'],
                )
            ]
        dn_num_group = dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))

        return dn_match_indices


    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)


    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum', avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
             + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5  + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))
    
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule
