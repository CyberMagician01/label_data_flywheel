#!/usr/bin/env python3
"""Freeze E-S5 calibration state and evaluate dev_holdout exactly once."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS
from engine.solver.ec_engine import evaluate
from engine.solver.pareto_checkpoint import atomic_write_json
from tools.bee_e.calibrate_inference_contract import build_contract
from tools.bee_e.split_contract import validate_schema_v2_split_contract
from tools.bee_e.validate_formal_configs import _load_config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _verified_path(path, expected_sha, label):
    if not path or len(str(expected_sha or '')) != 64:
        raise ValueError(f'{label} requires an existing file and complete SHA256.')
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')
    actual = sha256(path)
    if actual != str(expected_sha).lower():
        raise ValueError(
            f'{label} SHA256 mismatch: expected={expected_sha}, actual={actual}'
        )
    return path, actual


def validate_completed_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'E-S5 selected checkpoint not found: {path}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    required = {
        'model', 'optimizer', 'lr_scheduler', 'scaler', 'ema',
        'last_epoch', 'stage_controller',
    }
    missing = required - set(state)
    if missing:
        raise ValueError(f'E-S5 selected checkpoint misses {sorted(missing)}')
    stage = state['stage_controller']
    if not stage.get('completed') or int(stage.get('current_stage_index', -1)) != 5:
        raise ValueError('Checkpoint is not a completed continuous E-S5 lineage.')
    if not stage.get('best', {}).get('E-S5'):
        raise ValueError('Completed checkpoint does not record an E-S5 selected best state.')
    provenance = stage.get('provenance', {})
    if set(provenance) != {
        'initialization', 'design_evidence', 'split_contract', 'input_distribution',
    }:
        raise ValueError('Completed checkpoint misses the four-part route provenance.')
    sha_fields = {
        'initialization': ('sha256', 'selection_evidence_sha256'),
        'design_evidence': ('sha256',),
        'split_contract': (
            'source_manifest_sha256', 'derived_manifest_sha256',
            'train_annotation_sha256', 'calibration_annotation_sha256',
            'dev_holdout_annotation_sha256',
        ),
        'input_distribution': ('sha256',),
    }
    for group, fields in sha_fields.items():
        evidence = provenance[group]
        if not isinstance(evidence, dict) or any(
            len(str(evidence.get(field) or '')) != 64 for field in fields
        ):
            raise ValueError(f'Completed checkpoint has incomplete {group} provenance.')
    model_state = state['ema'].get('module') if isinstance(state['ema'], dict) else None
    if not isinstance(model_state, dict):
        raise ValueError('Completed checkpoint EMA does not contain module weights.')
    return state, sha256(path)


def domain_statistics_fingerprint(model_state):
    """Hash the two frozen DomainSpecificBatchNorm running-stat branches."""
    branch_tokens = ('.norms.', '.branches.')
    selected = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model_state.items()
        if any(token in name for token in branch_tokens)
        and name.endswith(('running_mean', 'running_var'))
    }
    domains = {
        int(part)
        for name in selected
        for part in name.split('.')
        if part in {'0', '1'}
        and any(f'{token}{part}.' in name for token in branch_tokens)
    }
    if domains != {0, 1}:
        raise ValueError('Frozen checkpoint must contain RGB/IR domain BN statistics.')
    digest = hashlib.sha256()
    for name in sorted(selected):
        tensor = selected[name]
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return {
        'sha256': digest.hexdigest(),
        'tensor_count': len(selected),
        'domains': ['rgb', 'ir'],
    }


def validate_split_contract(config):
    split = config.get('bee_e_split_contract', {})
    train_dataset = config.get('train_dataloader', {}).get('dataset', {})
    calibration_dataset = config.get('val_dataloader', {}).get('dataset', {})
    verified = validate_schema_v2_split_contract(
        split, train_dataset, calibration_dataset,
    )
    return {
        role: (
            Path(verified[f'{role}_annotation']),
            verified[f'{role}_annotation_sha256'],
        )
        for role in ('train', 'calibration', 'dev_holdout')
    } | {
        'manifest': (
            Path(verified['derived_manifest']), verified['derived_manifest_sha256'],
        ),
    }


def freeze_calibration_contract(observations_path, output_path, checkpoint,
                                checkpoint_sha, domain_stats,
                                calibration_annotation_sha, preprocessing):
    observations_path = Path(observations_path).expanduser().resolve()
    observations = json.loads(observations_path.read_text(encoding='utf-8'))
    contract = build_contract(
        observations, observations_path,
        expected_calibration_annotation_sha256=calibration_annotation_sha,
    )
    contract['frozen_model'] = {
        'checkpoint': str(Path(checkpoint).resolve()),
        'checkpoint_sha256': checkpoint_sha,
        'weights': 'ema.module',
        'domain_statistics': domain_stats,
    }
    contract['preprocessing'] = preprocessing
    atomic_write_json(output_path, contract)
    return contract, sha256(output_path)


def _json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def preprocessing_contract(config):
    def operation(dataset, name):
        matches = [
            item for item in dataset.get('transforms', {}).get('ops', [])
            if item.get('type') == name
        ]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {name} preprocessing operation.')
        return matches[0]

    train = config['train_dataloader']['dataset']
    calibration = config['val_dataloader']['dataset']
    if train.get('temporal_offsets') != [-4, -1, 0] or calibration.get(
        'temporal_offsets'
    ) != [-4, -1, 0]:
        raise ValueError('Frozen preprocessing requires temporal offsets [-4,-1,0].')
    train_prepare = operation(train, 'PrepareTemporalFrames')
    calibration_prepare = operation(calibration, 'PrepareTemporalFrames')
    train_ir = operation(train, 'IRPercentileNormalize')
    calibration_ir = operation(calibration, 'IRPercentileNormalize')
    stabilizer = dict(train_prepare.get('stabilizer', {}))
    ir_names = ('lower', 'upper', 'foreground_residual_quantile')
    ir_normalization = {name: float(train_ir[name]) for name in ir_names}
    if stabilizer != dict(calibration_prepare.get('stabilizer', {})):
        raise ValueError('Train/calibration stabilizer contracts disagree.')
    if ir_normalization != {
        name: float(calibration_ir[name]) for name in ir_names
    }:
        raise ValueError('Train/calibration IR normalization contracts disagree.')
    return {
        'temporal_frames': 3,
        'temporal_offsets': [-4, -1, 0],
        'input_size': list(config['eval_spatial_size']),
        'ir_normalization': ir_normalization,
        'stabilizer': stabilizer,
    }


def run(args):
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    checkpoint_state, checkpoint_sha = validate_completed_checkpoint(args.checkpoint)
    split_files = validate_split_contract(config)
    dev_path, dev_sha = split_files['dev_holdout']
    calibration_annotation_path, calibration_annotation_sha = (
        split_files['calibration']
    )
    manifest_path, manifest_sha = split_files['manifest']
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = output_dir / 'DEV_HOLDOUT_COMPLETE.json'
    running = output_dir / 'DEV_HOLDOUT_RUNNING.json'
    if complete.exists():
        raise RuntimeError(
            'dev_holdout has already been evaluated; the fixed protocol forbids rerunning it.'
        )
    if running.exists() and not args.resume_incomplete:
        raise RuntimeError(
            'An incomplete dev_holdout attempt exists; inspect it and pass '
            '--resume-incomplete only for infrastructure recovery.'
        )

    domain_stats = domain_statistics_fingerprint(
        checkpoint_state['ema']['module']
    )
    calibration_path = output_dir / 'inference_calibration_contract.json'
    preprocessing = preprocessing_contract(config)
    calibration, calibration_sha = freeze_calibration_contract(
        args.calibration_observations, calibration_path, args.checkpoint,
        checkpoint_sha, domain_stats, calibration_annotation_sha, preprocessing,
    )
    atomic_write_json(running, {
        'status': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint_sha256': checkpoint_sha,
        'calibration_contract_sha256': calibration_sha,
        'calibration_annotation': str(calibration_annotation_path),
        'calibration_annotation_sha256': calibration_annotation_sha,
        'dev_holdout_annotation_sha256': dev_sha,
    })

    cfg = YAMLConfig(
        str(config_path), device=args.device, seed=args.seed,
        output_dir=str(output_dir), use_amp=True,
    )
    cfg.yaml_cfg['enable_continuous_stage_training'] = False
    cfg.yaml_cfg['resume'] = None
    cfg.yaml_cfg['tuning'] = None
    cfg.yaml_cfg['evaluation_profile'] = 'full'
    cfg.evaluation_profile = 'full'
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = str(dev_path)
    cfg.yaml_cfg['val_dataloader']['dataset']['expected_ann_sha256'] = dev_sha
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = args.val_batch_size
    if 'ViTAdapter' in cfg.yaml_cfg:
        cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True

    dist_utils.setup_distributed(0, 'builtin', seed=args.seed)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    solver.load_resume_state(str(Path(args.checkpoint).resolve()))
    module = solver.ema.module if solver.ema else solver.model
    stats = evaluate(
        module, solver.criterion, solver.postprocessor,
        solver.val_dataloader, solver.evaluator, solver.device,
        evaluation_profile='full',
        inference_calibration_contract=calibration,
    )[0]
    result = {
        'status': 'complete',
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'split': 'dev_holdout',
        'evaluation_count': 1,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': checkpoint_sha,
        'calibration_contract': str(calibration_path),
        'calibration_contract_sha256': calibration_sha,
        'split_manifest': str(manifest_path),
        'split_manifest_sha256': manifest_sha,
        'dev_holdout_annotation': str(dev_path),
        'dev_holdout_annotation_sha256': dev_sha,
        'metrics': _json_value(stats),
    }
    atomic_write_json(complete, result)
    running.unlink(missing_ok=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--calibration-observations', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--val-batch-size', type=int, default=1)
    parser.add_argument('--resume-incomplete', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    try:
        run(parse_args())
    finally:
        dist_utils.cleanup()
