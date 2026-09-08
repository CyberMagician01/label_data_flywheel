#!/usr/bin/env python3
"""Prove full-validation metric equivalence before increasing eval batch size."""

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.misc import dist_utils
from engine.solver.pareto_checkpoint import atomic_write_json
from tools.bee_e.eval_checkpoint_queue import build_solver, evaluate_one


def numeric_values(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(numeric_values(item, f'{prefix}.{key}' if prefix else str(key)))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(numeric_values(item, f'{prefix}[{index}]'))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def evaluate_batch(config, checkpoint, output_dir, device, seed, batch_size):
    worker_args = SimpleNamespace(
        config=config,
        output_dir=Path(output_dir),
        device=device,
        seed=seed,
        evaluation_profile='detection',
        val_batch_size=batch_size,
    )
    solver = build_solver(worker_args)
    try:
        return evaluate_one(solver, checkpoint, 'detection')
    finally:
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(args):
    dist_utils.setup_distributed(0, 'builtin', seed=args.seed)
    first = evaluate_batch(
        args.config, args.checkpoint, args.output_dir, args.device, args.seed,
        args.first_batch,
    )
    second = evaluate_batch(
        args.config, args.checkpoint, args.output_dir, args.device, args.seed,
        args.second_batch,
    )
    first_values = numeric_values(first)
    second_values = numeric_values(second)
    if first_values.keys() != second_values.keys():
        missing_first = sorted(second_values.keys() - first_values.keys())
        missing_second = sorted(first_values.keys() - second_values.keys())
        raise SystemExit(
            f'metric key mismatch: missing_batch1={missing_first}, '
            f'missing_batch2={missing_second}'
        )
    deltas = {
        key: abs(first_values[key] - second_values[key])
        for key in first_values
        if math.isfinite(first_values[key]) and math.isfinite(second_values[key])
    }
    maximum = max(deltas.values(), default=0.0)
    report = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'first_batch': args.first_batch,
        'second_batch': args.second_batch,
        'tolerance': args.tolerance,
        'max_absolute_delta': maximum,
        'status': 'accepted' if maximum <= args.tolerance else 'rejected',
        'deltas': deltas,
        'first_stats': first,
        'second_stats': second,
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if maximum > args.tolerance:
        raise SystemExit(
            f'validation batch equivalence rejected: {maximum} > {args.tolerance}'
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--first-batch', type=int, default=1)
    parser.add_argument('--second-batch', type=int, default=2)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    return parser.parse_args()


if __name__ == '__main__':
    try:
        main(parse_args())
    finally:
        dist_utils.cleanup()
