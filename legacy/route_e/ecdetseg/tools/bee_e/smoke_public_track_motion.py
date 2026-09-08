#!/usr/bin/env python3
"""Small read-only gate for BEE24 motion supervision and axis masking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.core import YAMLConfig
from engine.edgecrafter.criterion import ECCriterion


class TrackLossHarness:
    loss_track = ECCriterion.loss_track
    _get_src_permutation_idx = ECCriterion._get_src_permutation_idx

    @staticmethod
    def _matched_instance_weight(targets, indices, device, mode):
        assert mode == "track"
        count = sum(len(target_indices) for _, target_indices in indices)
        return torch.ones(count, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = YAMLConfig(str(args.config))
    loader = config.train_dataloader
    iterator = iter(loader)
    try:
        _, targets = next(iterator)
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()

    geometry_count = sum(int(target["track_geometry_mask"].sum()) for target in targets)
    axis_count = sum(int(target["track_axis_mask"].sum()) for target in targets)
    if geometry_count <= 0:
        raise RuntimeError("First BEE24 batch has no center/scale track supervision")
    if axis_count != 0:
        raise RuntimeError("BEE24 batch unexpectedly enables unlabeled body-axis supervision")

    prediction = torch.tensor(
        [[[0.4, -0.2, 0.3, -0.1, 0.6, 0.8]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = [{
        "boxes": torch.zeros((1, 4)),
        "track_geometry": torch.zeros((1, 6)),
        "track_geometry_mask": torch.tensor([True]),
        "track_axis_mask": torch.tensor([False]),
    }]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    losses = TrackLossHarness().loss_track(
        {"pred_track_geometry": prediction}, targets, indices, 1
    )
    sum(losses.values()).backward()
    if not bool(prediction.grad[0, 0, :4].abs().sum() > 0):
        raise RuntimeError("Center/scale track gradient gate failed")
    if not bool(prediction.grad[0, 0, 4:].abs().sum() == 0):
        raise RuntimeError("Masked BEE24 axis produced a gradient")
    if float(losses["loss_track_axis"]) != 0.0:
        raise RuntimeError("Masked BEE24 axis produced a loss")
    print(
        "PUBLIC_TRACK_MOTION_SMOKE_OK "
        f"batch_geometry={geometry_count} batch_axis={axis_count} "
        f"center_loss={float(losses['loss_track_center_scale']):.6f}"
    )


if __name__ == "__main__":
    main()
