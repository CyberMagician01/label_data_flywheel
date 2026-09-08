"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..core import register
from .box_ops import (batch_dice_loss, batch_sigmoid_ce_loss,
                      box_cxcywh_to_xyxy, generalized_box_iou)
from .segmentation_head import point_sample
from .utils import weighting_function


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = [
        "use_focal_loss",
    ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0,
                 mask_point_sample_ratio=None, aux_positive_radius=0.0, **kwargs):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]
        self.cost_keypoint = weight_dict.get("cost_keypoint", 0.0)
        self.cost_oks = weight_dict.get("cost_oks", 0.0)
        self.cost_direction = weight_dict.get("cost_direction", 0.0)
        self.cost_ddf = weight_dict.get("cost_ddf", 0.0)
        self.pose_cost_multiplier = 1.0
        self.aux_positive_radius = float(aux_positive_radius)

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        
        self.mask_point_sample_ratio = mask_point_sample_ratio
        if self.mask_point_sample_ratio:
            self.cost_mask_ce = weight_dict["cost_mask_ce"]
            self.cost_mask_dice = weight_dict["cost_mask_dice"]

        assert (
            self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0
        ), "all costs cant be 0"

    def set_pose_cost_multiplier(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError('pose cost multiplier must be in [0, 1]')
        self.pose_cost_multiplier = value

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (
                (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        cost_ddf = None
        if (
            self.cost_ddf > 0
            and 'pred_corners' in outputs
            and 'ref_points' in outputs
            and 'up' in outputs
            and 'reg_scale' in outputs
        ):
            corner_logits = outputs['pred_corners'].flatten(0, 1)
            num_bins = corner_logits.shape[-1] // 4
            corner_logits = corner_logits.view(-1, 4, num_bins)
            reference = outputs['ref_points'].flatten(0, 1)
            target_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            scale = outputs['reg_scale'].abs().to(out_bbox).reshape(-1)[0]
            width_scale = reference[:, 2:3] / scale.clamp_min(1e-8)
            height_scale = reference[:, 3:4] / scale.clamp_min(1e-8)
            target_distance = torch.stack([
                (reference[:, None, 0] - target_xyxy[None, :, 0]) / width_scale - 0.5 * scale,
                (reference[:, None, 1] - target_xyxy[None, :, 1]) / height_scale - 0.5 * scale,
                (target_xyxy[None, :, 2] - reference[:, None, 0]) / width_scale - 0.5 * scale,
                (target_xyxy[None, :, 3] - reference[:, None, 1]) / height_scale - 0.5 * scale,
            ], dim=-1)
            support = weighting_function(
                num_bins - 1, outputs['up'].to(out_bbox), outputs['reg_scale'].to(out_bbox)
            ).to(out_bbox)
            right_index = torch.searchsorted(support.contiguous(), target_distance.contiguous())
            right_index = right_index.clamp(1, num_bins - 1)
            left_index = right_index - 1
            left_value = support[left_index]
            right_value = support[right_index]
            right_weight = (
                (target_distance - left_value) / (right_value - left_value).clamp_min(1e-8)
            ).clamp(0, 1)
            log_probability = corner_logits.log_softmax(dim=-1)
            pair_log_probability = log_probability[:, None].expand(
                -1, target_distance.shape[1], -1, -1,
            )
            left_nll = -torch.gather(
                pair_log_probability, -1, left_index.unsqueeze(-1),
            ).squeeze(-1)
            right_nll = -torch.gather(
                pair_log_probability, -1, right_index.unsqueeze(-1),
            ).squeeze(-1)
            cost_ddf = (
                (1.0 - right_weight) * left_nll + right_weight * right_nll
            ).mean(dim=-1)

        pose_present = (
            (self.cost_keypoint > 0 or self.cost_oks > 0 or self.cost_direction > 0)
            and 'pred_keypoints' in outputs
            and all('keypoints' in target for target in targets)
        )
        if pose_present:
            out_keypoints = outputs['pred_keypoints'].flatten(0, 1)
            tgt_keypoints = torch.cat([target['keypoints'] for target in targets])
            tgt_xy = tgt_keypoints[..., :2]
            tgt_visible = (tgt_keypoints[..., 2] > 0).to(out_keypoints.dtype)
            tgt_pose_mask = torch.cat([
                (target['pose_state'] > 0) if 'pose_state' in target else
                target.get('pose_mask', torch.ones(len(target['boxes']), device=tgt_xy.device)).bool()
                for target in targets
            ]).to(out_keypoints.dtype)
            tgt_pose_quality = torch.cat([
                target.get(
                    'pose_quality', torch.ones(len(target['boxes']), device=tgt_xy.device)
                )
                for target in targets
            ]).to(out_keypoints.dtype)
            tgt_pose_weight = tgt_pose_mask * tgt_pose_quality

            coordinate_error = (out_keypoints[:, None] - tgt_xy[None]).abs()
            coordinate_weight = tgt_visible[None, ..., None]
            coordinate_denominator = coordinate_weight.sum(dim=(-1, -2)).clamp_min(1.0)
            cost_keypoint = (coordinate_error * coordinate_weight).sum(dim=(-1, -2)) / coordinate_denominator
            cost_keypoint = cost_keypoint * tgt_pose_weight[None]

            if self.cost_oks > 0:
                target_diagonal = tgt_bbox[:, 2:].norm(dim=-1).clamp_min(1e-4)
                squared_distance = (out_keypoints[:, None] - tgt_xy[None]).pow(2).sum(dim=-1)
                oks_error = 1.0 - torch.exp(
                    -squared_distance / (2.0 * (0.10 * target_diagonal[None, :, None]).pow(2))
                )
                oks_denominator = tgt_visible.sum(dim=-1).clamp_min(1.0)
                cost_oks = (oks_error * tgt_visible[None]).sum(dim=-1) / oks_denominator[None]
                cost_oks = cost_oks * tgt_pose_weight[None]

            if self.cost_direction > 0 and out_keypoints.shape[-2] >= 2:
                pred_axis = F.normalize(out_keypoints[:, 1] - out_keypoints[:, 0], dim=-1, eps=1e-6)
                target_axis = F.normalize(tgt_xy[:, 1] - tgt_xy[:, 0], dim=-1, eps=1e-6)
                direction_mask = tgt_pose_weight * tgt_visible[:, 0] * tgt_visible[:, 1]
                cost_direction = (1.0 - pred_axis @ target_axis.transpose(0, 1)) * direction_mask[None]
        
        masks_present = "masks" in targets[0] and 'pred_masks' in outputs
        if masks_present:
            tgt_masks = torch.cat([v["masks"] for v in targets])
            out_masks = outputs["pred_masks"].flatten(0, 1)
            # Resize predicted masks to target mask size if needed
            # if out_masks.shape[-2:] != tgt_masks.shape[-2:]:
            #     # out_masks = F.interpolate(out_masks.unsqueeze(1), size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            #     tgt_masks = F.interpolate(tgt_masks.unsqueeze(1).float(), size=out_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            # # Flatten masks
            # pred_masks_logits = out_masks.flatten(1)  # [P, HW]
            # tgt_masks_flat = tgt_masks.flatten(1).float()  # [T, HW]

            num_points = out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio

            tgt_masks = tgt_masks.to(out_masks.dtype)

            point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
            pred_masks_logits = point_sample(out_masks.unsqueeze(1), point_coords.repeat(out_masks.shape[0], 1, 1), align_corners=False).squeeze(1)
            tgt_masks_flat = point_sample(tgt_masks.unsqueeze(1), point_coords.repeat(tgt_masks.shape[0], 1, 1), align_corners=False, mode="nearest").squeeze(1)

            # Binary cross-entropy with logits cost (mean over pixels), computed pairwise efficiently
            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)

            # Dice loss cost (1 - dice coefficient)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)
            
        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if cost_ddf is not None:
            C = C + self.cost_ddf * cost_ddf
        if pose_present:
            C = C + self.pose_cost_multiplier * self.cost_keypoint * cost_keypoint
            if self.cost_oks > 0:
                C = C + self.pose_cost_multiplier * self.cost_oks * cost_oks
            if self.cost_direction > 0:
                C = C + self.pose_cost_multiplier * self.cost_direction * cost_direction
        if masks_present:
            C = C + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice
        target_match_quality = torch.cat([
            torch.where(
                target.get(
                    'is_pseudo',
                    torch.zeros(
                        len(target['boxes']), dtype=torch.bool,
                        device=target['boxes'].device,
                    ),
                ).bool(),
                target.get(
                    'pseudo_score',
                    torch.ones(len(target['boxes']), device=target['boxes'].device),
                ).clamp(0, 1),
                torch.ones(len(target['boxes']), device=target['boxes'].device),
            )
            for target in targets
        ]).to(device=C.device, dtype=C.dtype)
        C = C * target_match_quality[None]
        C = C.view(bs, num_queries, -1)
        query_valid = outputs.get('pred_query_valid')
        if query_valid is not None:
            query_valid = query_valid.to(device=C.device, dtype=torch.bool)
            if query_valid.shape != (bs, num_queries):
                raise ValueError('pred_query_valid must have shape [batch, num_queries].')
            C = C.masked_fill(~query_valid[:, :, None], 1e6)
        C = C.cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = []
        for batch_idx, batch_cost in enumerate(C.split(sizes, -1)):
            local_cost = batch_cost[batch_idx]
            if sizes[batch_idx] == 0:
                indices_pre.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
                continue
            if query_valid is None:
                valid_rows = np.arange(num_queries)
            else:
                valid_rows = torch.nonzero(query_valid[batch_idx].cpu(), as_tuple=False).squeeze(1).numpy()
            local_sources, target_indices = linear_sum_assignment(local_cost[valid_rows])
            indices_pre.append((valid_rows[local_sources], target_indices))
        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            candidate_masks = None
            if self.aux_positive_radius > 0:
                candidate_masks = []
                for batch_index, target in enumerate(targets):
                    query_center = outputs['pred_boxes'][batch_index, :, :2].cpu()
                    target_boxes = target['boxes'].cpu()
                    if len(target_boxes) == 0:
                        candidate_masks.append(torch.zeros(num_queries, 0, dtype=torch.bool))
                        continue
                    target_radius = target_boxes[:, 2:].norm(dim=-1).clamp_min(1e-4)
                    distance = torch.cdist(query_center, target_boxes[:, :2])
                    candidate_masks.append(
                        distance <= self.aux_positive_radius * target_radius[None]
                    )
            return {
                "indices": indices,
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre,
                    candidate_masks=candidate_masks,
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None,
                          candidate_masks=None):
        split_costs = C.split(sizes, -1)
        results = []
        for batch_index, target_count in enumerate(sizes):
            if target_count == 0:
                empty = torch.zeros(0, dtype=torch.int64)
                results.append((empty, empty.clone()))
                continue
            local_cost = split_costs[batch_index][batch_index].clone().numpy()
            if candidate_masks is not None:
                allowed = candidate_masks[batch_index].numpy()
                local_cost[~allowed] = 1e6
            source_parts, target_parts = [], []
            initial_sources, initial_targets = initial_indices[batch_index]
            if len(initial_sources):
                source_parts.append(torch.as_tensor(initial_sources, dtype=torch.int64))
                target_parts.append(torch.as_tensor(initial_targets, dtype=torch.int64))
                local_cost[initial_sources, :] = 1e6
            for _ in range(max(k - 1, 0)):
                available_rows = np.flatnonzero(np.min(local_cost, axis=1) < 1e5)
                if len(available_rows) == 0:
                    break
                local_sources, target_indices = linear_sum_assignment(local_cost[available_rows])
                source_indices = available_rows[local_sources]
                source_parts.append(torch.as_tensor(source_indices, dtype=torch.int64))
                target_parts.append(torch.as_tensor(target_indices, dtype=torch.int64))
                # A query may supervise only one GT across all auxiliary
                # assignments, while every GT can receive up to k queries.
                local_cost[source_indices, :] = 1e6
            results.append((torch.cat(source_parts), torch.cat(target_parts)))
        return results
