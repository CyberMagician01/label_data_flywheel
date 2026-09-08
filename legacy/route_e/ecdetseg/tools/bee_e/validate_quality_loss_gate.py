#!/usr/bin/env python3
"""Direct launch gate for the quality-target probability and loss weighting."""

import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.edgecrafter.criterion import ECCriterion
from engine.edgecrafter.matcher import HungarianMatcher


def build_criterion():
    matcher = HungarianMatcher({
        'cost_class': 1,
        'cost_bbox': 1,
        'cost_giou': 1,
    })
    return ECCriterion(matcher, {}, [])


def check_endpoint_quality():
    criterion = build_criterion()
    target = {
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
        'keypoints': torch.tensor([[[0.4, 0.5, 2.0], [0.6, 0.5, 2.0]]]),
        'pose_state': torch.tensor([1]),
        'domain_id': torch.tensor([0]),
    }
    base = {
        'pred_logits': torch.tensor([[[10.0]]]),
        'pred_boxes': target['boxes'][None].clone(),
        'pred_quality': torch.tensor([[10.0]]),
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    perfect = dict(
        base,
        pred_keypoints=target['keypoints'][None, ..., :2].clone(),
    )
    displaced = dict(
        base,
        pred_keypoints=torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]]),
    )
    perfect_loss = criterion.loss_quality(
        perfect, [target], indices, 1,
    )['loss_quality']
    displaced_loss = criterion.loss_quality(
        displaced, [target], indices, 1,
    )['loss_quality']
    if not torch.isfinite(perfect_loss) or not torch.isfinite(displaced_loss):
        raise AssertionError('Endpoint quality gate produced a non-finite loss.')
    if perfect_loss < 0 or displaced_loss < 0:
        raise AssertionError('Endpoint quality gate produced a negative loss.')
    if not perfect_loss < displaced_loss:
        raise AssertionError('Endpoint OKS did not improve the quality target.')
    return float(perfect_loss), float(displaced_loss)


def check_balancing_weight():
    criterion = build_criterion()
    target = {
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
        'domain_id': torch.tensor([0]),
    }
    outputs = {
        'pred_logits': torch.tensor([[[10.0]]]),
        'pred_boxes': target['boxes'][None].clone(),
        'pred_quality': torch.tensor([[0.0]]),
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    original = ECCriterion.__dict__['_matched_instance_weight']

    def set_weight(value):
        ECCriterion._matched_instance_weight = staticmethod(
            lambda targets, indices, device, mode='base': torch.tensor(
                [value], dtype=torch.float32, device=device,
            )
        )

    try:
        set_weight(1.0)
        base_loss = criterion.loss_quality(
            outputs, [target], indices, 1,
        )['loss_quality']
        set_weight(4.0)
        weighted_loss = criterion.loss_quality(
            outputs, [target], indices, 1,
        )['loss_quality']
    finally:
        ECCriterion._matched_instance_weight = original

    if not torch.isfinite(base_loss) or not torch.isfinite(weighted_loss):
        raise AssertionError('Balancing gate produced a non-finite loss.')
    if base_loss < 0 or weighted_loss < 0:
        raise AssertionError('Balancing gate produced a negative loss.')
    torch.testing.assert_close(weighted_loss, base_loss * 4.0)
    return float(base_loss), float(weighted_loss)


def main():
    perfect_loss, displaced_loss = check_endpoint_quality()
    base_loss, weighted_loss = check_balancing_weight()
    print(json.dumps({
        'status': 'QUALITY_LOSS_GATE_OK',
        'perfect_endpoint_loss': perfect_loss,
        'displaced_endpoint_loss': displaced_loss,
        'weight_1_loss': base_loss,
        'weight_4_loss': weighted_loss,
        'weight_ratio': weighted_loss / base_loss,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
