"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import math

import torch

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .utils import inverse_sigmoid


def _monotonic_denoising_count(
    max_gt_num,
    stage_cap,
    effective_query_capacity,
    matching_entropy,
    base_fraction,
    entropy_gain,
):
    """Return an even DN slot count monotone in GT, capacity and entropy."""
    if max_gt_num <= 0 or stage_cap <= 1 or effective_query_capacity <= 1:
        return 0
    cap = min(int(stage_cap), int(effective_query_capacity))
    demand_inputs = {
        'matching_entropy': float(matching_entropy),
        'base_fraction': float(base_fraction),
        'entropy_gain': float(entropy_gain),
    }
    nonfinite = {
        name: value for name, value in demand_inputs.items()
        if not math.isfinite(value)
    }
    if nonfinite:
        raise FloatingPointError(
            f'Non-finite denoising demand inputs before allocation: {nonfinite}'
        )
    entropy = min(max(demand_inputs['matching_entropy'], 0.0), 1.0)
    calibrated_demand = math.ceil(
        cap * (demand_inputs['base_fraction'] + demand_inputs['entropy_gain'] * entropy)
    )
    requested = min(cap, max(2 * int(max_gt_num), calibrated_demand, 2))
    return requested - requested % 2


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    num_keypoints=0,
    keypoint_noise_scale=0.0,
    head_tail_swap_ratio=0.0,
    no_object_noise_ratio=0.0,
    effective_query_capacity=None,
    matching_entropy=0.0,
    denoising_base_fraction=0.25,
    denoising_entropy_gain=0.75,
):
    """Build exact-capacity contrastive DN queries.

    The slot count is an explicit monotone function of the batch GT count,
    effective normal-query capacity and current matching entropy. Positive
    target indices are stored directly, so a partially filled reconstruction
    cycle remains exactly supervised and exactly normalized.
    """
    if num_denoising <= 0:
        return None, None, None, None
    if not targets:
        raise ValueError('Denoising requires a non-empty target batch.')
    for name, value in (
        ('label_noise_ratio', label_noise_ratio),
        ('head_tail_swap_ratio', head_tail_swap_ratio),
        ('no_object_noise_ratio', no_object_noise_ratio),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f'{name} must be in [0, 1].')
    if not 0.0 <= float(denoising_base_fraction) <= 1.0:
        raise ValueError('denoising_base_fraction must be in [0, 1].')
    if float(denoising_entropy_gain) < 0.0:
        raise ValueError('denoising_entropy_gain must be non-negative.')

    num_gts = [len(t['labels']) for t in targets]
    device = targets[0]['labels'].device
    max_gt_num = max(num_gts)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_max_gt = torch.as_tensor(max_gt_num, dtype=torch.long, device=device)
        torch.distributed.all_reduce(global_max_gt, op=torch.distributed.ReduceOp.MAX)
        max_gt_num = int(global_max_gt.item())
    if torch.is_tensor(matching_entropy):
        matching_entropy = matching_entropy.detach().float().mean().item()
    if effective_query_capacity is None:
        effective_query_capacity = num_queries
    if torch.is_tensor(effective_query_capacity):
        effective_query_capacity = effective_query_capacity.detach().max().item()
    num_denoising = _monotonic_denoising_count(
        max_gt_num,
        num_denoising,
        effective_query_capacity,
        matching_entropy,
        denoising_base_fraction,
        denoising_entropy_gain,
    )
    if num_denoising == 0:
        return None, None, None, None

    bs = len(num_gts)
    positive_count = num_denoising // 2
    input_query_class = torch.full(
        [bs, num_denoising], num_classes, dtype=torch.int32, device=device,
    )
    input_query_bbox = torch.zeros([bs, num_denoising, 4], device=device)
    input_query_keypoints = (
        torch.zeros([bs, num_denoising, num_keypoints, 2], device=device)
        if num_keypoints > 0 else None
    )
    input_query_pose_valid = (
        torch.zeros([bs, num_denoising, num_keypoints], dtype=torch.bool, device=device)
        if num_keypoints > 0 else None
    )
    pad_gt_mask = torch.zeros([bs, num_denoising], dtype=torch.bool, device=device)
    dn_positive_idx = []
    dn_positive_target_idx = []

    for batch_idx, num_gt in enumerate(num_gts):
        if num_gt == 0:
            empty = torch.zeros(0, dtype=torch.int64, device=device)
            dn_positive_idx.append(empty)
            dn_positive_target_idx.append(empty)
            continue
        # A seeded cyclic rotation provides uniform target coverage without
        # binding reconstruction targets to persistent query identities.
        offset = torch.randint(num_gt, (1,), device=device)
        target_idx = (torch.arange(positive_count, device=device) + offset) % num_gt
        all_target_idx = torch.cat([target_idx, target_idx], dim=0)
        input_query_class[batch_idx] = targets[batch_idx]['labels'][all_target_idx]
        input_query_bbox[batch_idx] = targets[batch_idx]['boxes'][all_target_idx]
        pad_gt_mask[batch_idx] = True
        dn_positive_idx.append(torch.arange(positive_count, device=device))
        dn_positive_target_idx.append(target_idx.to(torch.int64))
        if input_query_keypoints is not None and 'keypoints' in targets[batch_idx]:
            keypoints = targets[batch_idx]['keypoints'][all_target_idx, :num_keypoints]
            input_query_keypoints[batch_idx] = keypoints[..., :2]
            input_query_pose_valid[batch_idx] = keypoints[..., 2] > 0

    negative_gt_mask = torch.zeros(
        [bs, num_denoising, 1], dtype=torch.bool, device=device,
    )
    negative_gt_mask[:, positive_count:] = True

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(
            input_query_class, 0, num_classes, dtype=input_query_class.dtype,
        )
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    no_object_mask = torch.zeros_like(pad_gt_mask)
    if no_object_noise_ratio > 0:
        no_object_mask = (
            torch.rand_like(input_query_class, dtype=torch.float) < no_object_noise_ratio
        ) & pad_gt_mask
        input_query_class = torch.where(
            no_object_mask,
            torch.full_like(input_query_class, num_classes),
            input_query_class,
        )

    known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
    if box_noise_scale > 0:
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (~negative_gt_mask)
        known_bbox += rand_sign * rand_part * diff
    known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
    input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
    input_query_bbox[input_query_bbox < 0] *= -1
    input_query_bbox_unact = inverse_sigmoid(input_query_bbox)
    input_query_logits = class_embed(input_query_class)

    dn_keypoint_refs = None
    swap_count = torch.zeros((), device=device)
    noise_count = torch.zeros((), device=device)
    if input_query_keypoints is not None:
        dn_keypoint_refs = input_query_keypoints.clone()
        if keypoint_noise_scale > 0:
            diagonal = input_query_bbox[..., 2:].norm(dim=-1, keepdim=True)
            noise = (
                torch.rand_like(dn_keypoint_refs).mul(2).sub(1)
                * diagonal[..., None]
                * keypoint_noise_scale
            )
            valid_noise = input_query_pose_valid[..., None] & pad_gt_mask[..., None, None]
            dn_keypoint_refs = torch.where(valid_noise, dn_keypoint_refs + noise, dn_keypoint_refs)
            noise_count = valid_noise.sum()
        if num_keypoints == 2 and head_tail_swap_ratio > 0:
            swap_mask = (
                torch.rand([bs, num_denoising], device=device) < head_tail_swap_ratio
            ) & pad_gt_mask
            dn_keypoint_refs = torch.where(
                swap_mask[..., None, None], dn_keypoint_refs.flip(2), dn_keypoint_refs,
            )
            swap_count = swap_mask.sum()
        dn_keypoint_refs = dn_keypoint_refs.clamp(0, 1)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True
    attn_mask[:num_denoising, num_denoising:] = True
    # Every positive/negative reconstruction pair is isolated from every
    # other pair, while the two members of one pair can exchange evidence.
    attn_mask[:num_denoising, :num_denoising] = True
    pair_idx = torch.arange(positive_count, device=device)
    pair_members = torch.stack([pair_idx, pair_idx + positive_count], dim=1)
    for members in pair_members:
        attn_mask[members[:, None], members[None, :]] = False

    dn_meta = {
        'dn_positive_idx': tuple(dn_positive_idx),
        'dn_positive_target_idx': tuple(dn_positive_target_idx),
        'dn_positive_count': int(sum(index.numel() for index in dn_positive_idx)),
        'dn_num_group': 1,
        'dn_num_split': [num_denoising, num_queries],
        'dn_matching_entropy': float(matching_entropy),
        'dn_effective_query_capacity': int(effective_query_capacity),
        'dn_no_object_noise_count': no_object_mask.sum(),
    }
    if dn_keypoint_refs is not None:
        dn_meta.update({
            'dn_keypoint_refs': dn_keypoint_refs,
            'dn_keypoint_noise_count': noise_count,
            'dn_head_tail_swap_count': swap_count,
        })

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
