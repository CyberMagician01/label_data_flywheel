#!/usr/bin/env python3
"""Compare training-critical checkpoint state after sync/async smoke runs."""

import argparse
import json
from pathlib import Path

import torch


REQUIRED_STATE = ('model', 'ema', 'optimizer', 'lr_scheduler', 'scaler', 'last_epoch')


def compare_values(left, right, path, differences):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            differences.append(f'{path}: tensor metadata differs')
        elif not torch.equal(left, right):
            maximum = float((left.float() - right.float()).abs().max())
            differences.append(f'{path}: tensor differs, max_abs={maximum}')
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            differences.append(f'{path}: mapping keys differ')
            return
        for key in left:
            compare_values(left[key], right[key], f'{path}.{key}', differences)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            differences.append(f'{path}: sequence length differs')
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(
                left_item, right_item, f'{path}[{index}]', differences
            )
        return
    if left != right:
        differences.append(f'{path}: {left!r} != {right!r}')


def main(args):
    left = torch.load(args.left, map_location='cpu', weights_only=True)
    right = torch.load(args.right, map_location='cpu', weights_only=True)
    differences = []
    for name in REQUIRED_STATE:
        if name not in left or name not in right:
            differences.append(f'{name}: required state is missing')
            continue
        compare_values(left[name], right[name], name, differences)
    report = {
        'left': str(Path(args.left).resolve()),
        'right': str(Path(args.right).resolve()),
        'required_state': list(REQUIRED_STATE),
        'status': 'accepted' if not differences else 'rejected',
        'differences': differences[:100],
        'difference_count': len(differences),
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if differences:
        raise SystemExit('sync/async training state mismatch')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', required=True)
    parser.add_argument('--right', required=True)
    parser.add_argument('--report', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
