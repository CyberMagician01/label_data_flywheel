"""Create an honest pre-training registration for the full E route."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.bee_e.design_evidence import FRONTIER_METRICS, _transform_operation
from tools.bee_e.validate_formal_configs import _load_config


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_values(config):
    adapter = config['ViTAdapter']
    projection = adapter['proj_dim']
    prepare = _transform_operation(config, 'PrepareTemporalFrames')
    ir_normalize = _transform_operation(config, 'IRPercentileNormalize')
    return {
        'p2_channels': int(projection[0] if isinstance(projection, list) else projection),
        'endpoint_sampling_points': int(config['ECTransformer']['endpoint_sampling_points']),
        'domain_adapter_rank': int(adapter['domain_adapter_rank']),
        'stabilizer': dict(prepare.get('stabilizer', {})),
        'ir_normalization': {
            name: float(ir_normalize[name])
            for name in ('lower', 'upper', 'foreground_residual_quantile')
        },
        'foreground_prototype_quality_threshold': float(
            config['ECCriterion']['domain_foreground_quality_threshold']
        ),
    }


def build(config, plan_sha256, alignment_sha256):
    selected = selected_values(config)
    alternatives = {
        'p2_channels': [128, selected['p2_channels']],
        'endpoint_sampling_points': [4, selected['endpoint_sampling_points']],
        'domain_adapter_rank': [8, selected['domain_adapter_rank'], 32],
        'stabilizer': [
            {**selected['stabilizer'], 'stable_translation': 0.25},
            selected['stabilizer'],
        ],
        'ir_normalization': [
            {**selected['ir_normalization'], 'lower': 0.02, 'upper': 0.98},
            selected['ir_normalization'],
        ],
        'foreground_prototype_quality_threshold': [
            0.6, selected['foreground_prototype_quality_threshold'], 0.8,
        ],
    }
    optimizer = config['optimizer']
    train_loader = config['train_dataloader']
    optimization = {
        'selected': {
            'optimizer': 'AdamW',
            'base_lr': float(optimizer['lr']),
            'weight_decay': float(optimizer['weight_decay']),
            'total_batch_size': int(train_loader['total_batch_size']),
            'grad_accum_steps': int(config['grad_accum_steps']),
            'effective_batch_size': int(
                train_loader['total_batch_size'] * config['grad_accum_steps']
            ),
            'use_amp': bool(config['use_amp']),
            'clip_max_norm': float(config['clip_max_norm']),
            'query_clip_max_norm': float(config['query_clip_max_norm']),
            'ema_decay': float(config.get('ema_decay', 0.9999)),
            'loss_weights': config['ECCriterion']['weight_dict'],
        },
        'measurement_protocol': {
            'status': 'pending_training_measurement',
            'gradient_trace_fields': [
                'parameter_group', 'gradient_norm', 'train_calibration_gap',
            ],
            'mixed_precision_fields': [
                'optimizer_step_equivalent', 'max_loss_error', 'max_gradient_error',
            ],
            'optimizer_step_gate': 1,
            'seed': int(config['seed']),
        },
    }
    return {
        'evidence_type': 'preregistered_design_experiment',
        'status': 'registered_before_training',
        'split': 'calibration',
        'calibration_annotation_sha256': config['val_dataloader']['dataset'][
            'expected_ann_sha256'
        ],
        'implementation_plan_sha256': plan_sha256,
        'alignment_contract_sha256': alignment_sha256,
        'selected': selected,
        'experiments': {
            axis: {
                'candidate_values': alternatives[axis],
                'metrics': list(FRONTIER_METRICS[axis]),
                'measurement_status': 'pending_training_measurement',
                'seed': int(config['seed']),
            }
            for axis in selected
        },
        'optimization': optimization,
        'stage_budget': [
            {
                'stage': stage['name'],
                'min_cycles': int(stage['min_cycles']),
                'max_cycles': int(stage['max_cycles']),
                'metrics': [
                    'coverage_speed', 'metric_autocorrelation', 'bootstrap_noise_bound',
                ],
                'measurement_status': 'pending_training_measurement',
            }
            for stage in config['bee_e_stage_specs']
        ],
        'truthfulness': {
            'no_pretraining_metrics_fabricated': True,
            'measured_results_must_be_written_during_training': True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--implementation-plan-sha256', required=True)
    parser.add_argument('--alignment-contract-sha256', required=True)
    args = parser.parse_args()
    config = _load_config(Path(args.config).resolve())
    evidence = build(
        config, args.implementation_plan_sha256.lower(),
        args.alignment_contract_sha256.lower(),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    temporary.replace(output)
    print(json.dumps({'path': str(output), 'sha256': sha256(output)}))


if __name__ == '__main__':
    main()
