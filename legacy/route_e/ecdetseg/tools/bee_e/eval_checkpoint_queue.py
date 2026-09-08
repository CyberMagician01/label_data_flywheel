#!/usr/bin/env python3
"""Evaluate atomically published E1 checkpoints without blocking training."""

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS
from engine.solver.ec_engine import evaluate
from engine.solver.pareto_checkpoint import ParetoCheckpointManager, atomic_write_json


CHECKPOINT_PATTERN = re.compile(r'^checkpoint(?P<epoch>\d{4})\.ready$')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def epoch_paths(output_dir, epoch):
    output_dir = Path(output_dir)
    metrics_dir = output_dir / 'checkpoint_metrics'
    return {
        'checkpoint': output_dir / f'checkpoint{epoch:04}.pth',
        'ready': output_dir / f'checkpoint{epoch:04}.ready',
        'metrics': metrics_dir / f'epoch{epoch:04}.json',
        'done': metrics_dir / f'epoch{epoch:04}.done',
        'failed': metrics_dir / f'epoch{epoch:04}.failed',
    }


def completed_with_matching_sha(paths):
    if not all(paths[name].is_file() for name in ('checkpoint', 'metrics', 'done')):
        return False
    try:
        record = json.loads(paths['metrics'].read_text(encoding='utf-8'))
        done = json.loads(paths['done'].read_text(encoding='utf-8'))
        actual = sha256(paths['checkpoint'])
        return record.get('sha256') == actual and done.get('sha256') == actual
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def invalidate_stale_outputs(paths):
    for name in ('metrics', 'done', 'failed'):
        path = paths[name]
        if path.exists():
            path.unlink()


def discover_ready_epochs(output_dir):
    epochs = []
    for marker in Path(output_dir).glob('checkpoint*.ready'):
        match = CHECKPOINT_PATTERN.match(marker.name)
        if match:
            epochs.append(int(match.group('epoch')))
    return sorted(epochs)


def build_solver(args):
    cfg = YAMLConfig(
        args.config,
        device=args.device,
        seed=args.seed,
        output_dir=str(args.output_dir),
        use_amp=True,
    )
    cfg.yaml_cfg['evaluation_profile'] = args.evaluation_profile
    cfg.evaluation_profile = args.evaluation_profile
    if 'ViTAdapter' in cfg.yaml_cfg:
        # Every async evaluation loads a complete E1 checkpoint immediately;
        # reloading/downloading the backbone here is redundant and fragile.
        cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
    if args.val_batch_size is not None:
        cfg.yaml_cfg['val_dataloader']['total_batch_size'] = args.val_batch_size
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    return solver


def evaluate_one(solver, checkpoint, evaluation_profile):
    solver.load_resume_state(str(checkpoint))
    module = solver.ema.module if solver.ema else solver.model
    return evaluate(
        module,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device,
        evaluation_profile=evaluation_profile,
    )[0]


def run(args):
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dist_utils.setup_distributed(0, 'builtin', seed=args.seed)
    solver = build_solver(args)
    manager = ParetoCheckpointManager(
        args.output_dir,
        metric_directions=solver.cfg.pareto_metrics,
    )
    failed_this_run = set()

    while True:
        completed = 0
        progressed = False
        for epoch in discover_ready_epochs(args.output_dir):
            paths = epoch_paths(args.output_dir, epoch)
            if not paths['checkpoint'].is_file():
                continue
            if completed_with_matching_sha(paths):
                completed += 1
                continue
            if epoch in failed_this_run:
                continue
            invalidate_stale_outputs(paths)
            try:
                stats = evaluate_one(
                    solver, paths['checkpoint'], args.evaluation_profile
                )
                manager.record(epoch, paths['checkpoint'], stats)
                checkpoint_sha = sha256(paths['checkpoint'])
                atomic_write_json(paths['done'], {
                    'epoch': epoch,
                    'checkpoint': str(paths['checkpoint']),
                    'sha256': checkpoint_sha,
                    'status': 'done',
                })
                completed += 1
                progressed = True
                print(
                    f'Async evaluation complete: epoch={epoch} '
                    f'sha256={checkpoint_sha}',
                    flush=True,
                )
            except Exception:
                failed_this_run.add(epoch)
                atomic_write_text(paths['failed'], traceback.format_exc())
                print(
                    f'Async evaluation failed: epoch={epoch}; '
                    f'details={paths["failed"]}',
                    flush=True,
                )

        if args.once:
            return 0 if completed else 1
        ready_count = len(discover_ready_epochs(args.output_dir))
        if ready_count >= args.expected_checkpoints and failed_this_run:
            print(
                f'Async evaluation queue failed for epochs: '
                f'{sorted(failed_this_run)}',
                flush=True,
            )
            return 2
        if completed >= args.expected_checkpoints:
            print(
                f'Async evaluation queue complete: {completed}/'
                f'{args.expected_checkpoints}',
                flush=True,
            )
            return 0
        if not progressed:
            time.sleep(args.poll_seconds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--evaluation-profile', default='detection')
    parser.add_argument('--val-batch-size', type=int)
    parser.add_argument('--expected-checkpoints', type=int, default=74)
    parser.add_argument('--poll-seconds', type=float, default=5.0)
    parser.add_argument('--once', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        raise SystemExit(run(parse_args()))
    finally:
        dist_utils.cleanup()
