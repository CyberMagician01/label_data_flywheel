"""Fail-fast validation for the BeePoseTrack-E 1280/256 formal chain."""

import argparse
import copy
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge_dict(target, source):
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _merge_dict(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _load_config(path, config=None):
    """Load inherited YAML without importing the training runtime."""
    path = Path(path)
    config = {} if config is None else config
    document = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    for include in document.get('__include__', []):
        include_path = Path(include).expanduser()
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        _load_config(include_path.resolve(), config)
    return _merge_dict(config, document)


STAGES = {
    'e1_formal_native_p2_q768_1280.yml': {
        'epochs': 74, 'keypoints': 0, 'density': False,
        'temporal': False, 'domain': False, 'pseudo': False,
        'evaluation_profile': 'detection', 'inline_eval': False,
    },
    'e2_formal_joint_1280.yml': {
        'epochs': 92, 'keypoints': 2, 'density': False,
        'temporal': False, 'domain': False, 'pseudo': False,
        'evaluation_profile': 'joint', 'inline_eval': True,
    },
    'e3_formal_query_1280.yml': {
        'epochs': 40, 'keypoints': 2, 'density': True,
        'temporal': False, 'domain': False, 'pseudo': False,
        'evaluation_profile': 'joint', 'inline_eval': True,
    },
    'e4_formal_temporal_domain_1280.yml': {
        'epochs': 50, 'keypoints': 2, 'density': True,
        'temporal': True, 'domain': True, 'pseudo': False,
        'evaluation_profile': 'full', 'inline_eval': True,
    },
    'e5_formal_semisupervised_1280.yml': {
        'epochs': 60, 'keypoints': 2, 'density': True,
        'temporal': True, 'domain': True, 'pseudo': True,
        'evaluation_profile': 'full', 'inline_eval': True,
    },
    'e6_formal_final_1280.yml': {
        'epochs': 30, 'keypoints': 2, 'density': True,
        'temporal': True, 'domain': True, 'pseudo': False,
        'evaluation_profile': 'full', 'inline_eval': True,
    },
}


def validate_continuous_route(config_path):
    config_path = Path(config_path)
    config = _load_config(config_path)
    errors = []
    stages = config.get('bee_e_stage_specs', [])
    train = config.get('train_dataloader', {}).get('dataset', {})
    val = config.get('val_dataloader', {}).get('dataset', {})
    detector = config.get('ECDet', {})
    adapter = config.get('ViTAdapter', {})
    decoder = config.get('ECTransformer', {})
    criterion = config.get('ECCriterion', {})
    encoder = config.get('HybridEncoder', {})
    postprocessor = config.get('PostProcessor', {})
    recheck = config.get('bee_e_density_recheck', {})
    initialization = config.get('bee_e_initialization', {})
    split_contract = config.get('bee_e_split_contract', {})
    design_evidence = config.get('bee_e_design_evidence', {})
    distribution_evidence = config.get('bee_e_input_distribution_evidence', {})
    _require(errors, config.get('enable_continuous_stage_training') is True,
             'continuous route: stage controller must be enabled')
    _require(errors, [stage.get('name') for stage in stages] == [
        'E-S0', 'E-S1', 'E-S2', 'E-S3', 'E-S4', 'E-S5'
    ], 'continuous route: E-S0..E-S5 order is invalid')
    _require(errors, sum(int(stage.get('max_cycles', 0)) for stage in stages) ==
             int(config.get('bee_e_total_coverage_budget', -1)) == config.get('epochs'),
             'continuous route: coverage budget must equal stage maxima and epochs')
    external_eval = bool(config.get('external_eval', {}).get('enabled', False))
    _require(errors, bool(config.get('inline_eval')) != external_eval and
             config.get('evaluation_profile') == 'full',
             'continuous route: every cycle requires exactly one SHA-bound full calibration mode')
    _require(errors, detector.get('temporal_frames') == 3 and
             detector.get('enable_stabilized_temporal') is True and
             detector.get('use_temporal_difference') is False and
             detector.get('use_frequency_gate') is False,
             'continuous route: stabilized [long, short, current] input is required')
    _require(errors, train.get('temporal_offsets') == [-4, -1, 0] and
             val.get('temporal_offsets') == [-4, -1, 0],
             'continuous route: train/val temporal offsets must be [-4,-1,0]')
    _require(errors, decoder.get('num_queries') == 768 and
             decoder.get('num_layers') == 4 and
             decoder.get('endpoint_decoder_type') == 'decomposed' and
             decoder.get('use_full_endpoint_route') is True,
             'continuous route: q768/four-layer/decomposed endpoint contract failed')
    _require(errors, 0.0 < float(decoder.get('no_object_noise_ratio', 0.0)) < 1.0 and
             float(decoder.get('denoising_base_fraction', 0.0)) > 0.0 and
             float(decoder.get('denoising_entropy_gain', 0.0)) > 0.0,
             'continuous route: monotone GT/capacity/entropy denoising contract failed')
    _require(errors, encoder.get('feat_strides') == [4, 8, 16, 32] and
             encoder.get('activation_checkpoint_high_resolution') is True,
             'continuous route: P2-P5/checkpointed P2-P3 encoder is required')
    _require(errors, int(adapter.get('domain_adapter_rank', 0)) > 0 and
             adapter.get('enable_dual_domain_norm') is True and
             adapter.get('enable_domain_layernorm') is True,
             'continuous route: P2/P3 low-rank RGB/IR adapters and domain normalization are required')
    _require(errors, detector.get('enable_domain_adapter') is False and
             detector.get('enable_shared_foreground_prototype') is True,
             'continuous route: legacy pyramid experts must be disabled and the shared P3 foreground prototype enabled')
    _require(errors,
             float(criterion.get('domain_foreground_quality_threshold', -1.0)) == 0.7,
             'continuous route: the evidence-selected foreground prototype quality threshold must be 0.7')
    domain_loss_weights = {
        key for key, value in criterion.get('weight_dict', {}).items()
        if (key == 'loss_domain' or key.startswith('loss_domain_') or
            key == 'loss_shared_bee_prototype') and float(value) > 0.0
    }
    _require(errors, domain_loss_weights == {'loss_shared_bee_prototype'} and
             float(criterion.get('weight_dict', {}).get(
                 'loss_shared_bee_prototype', 0.0
             )) > 0.0,
             'continuous route: only the exact shared foreground prototype domain loss may be active')
    _require(errors, train.get('expected_ann_sha256') ==
             split_contract.get('train_annotation_sha256'),
             'continuous route: train annotation SHA is not bound to the split contract')
    _require(errors, val.get('expected_ann_sha256') ==
             split_contract.get('calibration_annotation_sha256'),
             'continuous route: calibration annotation SHA is not bound to the split contract')
    _require(errors, postprocessor.get('enable_calibrated_filter') is True,
             'continuous route: calibrated domain score filtering is required')
    _require(errors, postprocessor.get('enable_set_dedup') is False and
             recheck.get('deduplicate_after_merge') is True,
             'continuous route: structural set dedup must execute exactly once after recheck')
    _require(errors, config.get('pareto_metrics') is None,
             'continuous route: legacy disjoint-stage Pareto selector must be disabled')
    _require(errors, initialization.get('source') == 'official_ecdet_coco' and
             initialization.get('selected_scale') == 'M' and
             {'selection_evidence', 'selection_evidence_sha256',
               'checkpoint', 'sha256'} <= set(initialization),
             'continuous route: selected official ECDet-M COCO initialization contract is required')
    _require(errors, set(initialization.get('reinitialize_output_heads', [])) ==
             {'category', 'quality', 'density', 'endpoints'},
             'continuous route: all task output heads must be reinitialized')
    _require(errors, config.get('require_equal_domain_updates') is True,
             'continuous route: every optimizer update must enforce equal RGB/IR views')
    _require(errors, config.get('enable_ddp_no_sync') is True,
             'continuous route: accumulated DDP micro-batches must synchronize once per update')
    _require(errors, split_contract.get('protocol') ==
             'video_section_near_duplicate_track_scope_fixed_5_1_1' and
             {'source_manifest', 'source_manifest_sha256',
              'derived_manifest', 'derived_manifest_sha256',
              'train_annotation', 'train_annotation_sha256',
              'calibration_annotation', 'calibration_annotation_sha256',
              'dev_holdout_annotation', 'dev_holdout_annotation_sha256'} <= set(split_contract),
             'continuous route: fixed train/calibration/dev_holdout 5:1:1 contract is required')
    _require(errors, {'path', 'sha256'} <= set(design_evidence),
             'continuous route: SHA-bound design evidence contract is required')
    _require(errors, {'path', 'sha256'} <= set(distribution_evidence),
             'continuous route: SHA-bound input distribution evidence is required')
    if errors:
        raise ValueError('\n'.join(errors))
    return {
        'config': config_path.name,
        'stages': [stage['name'] for stage in stages],
        'coverage_budget': config['bee_e_total_coverage_budget'],
        'temporal_offsets': train['temporal_offsets'],
        'queries': decoder['num_queries'],
        'status': 'ok',
    }


def _require(errors, condition, message):
    if not condition:
        errors.append(message)


def _learning_rate_contract(config, final_stage):
    optimizer = config['optimizer']
    expected = (
        {'base': 1e-5, 'trunk': 5e-7, 'new': 2e-5}
        if final_stage else
        {'base': 1e-4, 'trunk': 5e-6, 'new': 2e-4}
    )
    groups = optimizer.get('params', [])
    trunk_groups = [
        group for group in groups
        if str(group.get('params', '')).startswith('^backbone\\.backbone\\.')
    ]
    new_groups = [
        group for group in groups
        if 'lr' in group
        and 'native_p2_adapter' in str(group.get('params', ''))
        and 'endpoint' in str(group.get('params', ''))
        and 'domain' in str(group.get('params', ''))
    ]
    return expected, optimizer, trunk_groups, new_groups


def validate_stage(config_path):
    config_path = Path(config_path)
    config = _load_config(config_path)
    expected_stage = STAGES[config_path.name]
    errors = []
    prefix = config_path.name

    adapter = config.get('ViTAdapter', {})
    encoder = config.get('HybridEncoder', {})
    detector = config.get('ECDet', {})
    decoder = config.get('ECTransformer', {})
    criterion = config.get('ECCriterion', {})
    train_loader = config.get('train_dataloader', {})
    train_dataset = train_loader.get('dataset', {})
    val_dataset = config.get('val_dataloader', {}).get('dataset', {})

    _require(errors, config.get('eval_spatial_size') == [1280, 1280],
             f'{prefix}: eval_spatial_size must be [1280, 1280]')
    _require(errors, '960' not in str(config.get('output_dir', '')),
             f'{prefix}: legacy 960 output directory is forbidden')
    _require(errors, adapter.get('name') == 'ecvittplus',
             f'{prefix}: formal backbone must be ECDet-M/ecvittplus')
    _require(errors, adapter.get('use_native_p2') is True,
             f'{prefix}: native stem P2 must be enabled')
    _require(errors, adapter.get('embed_dim') == 256 and adapter.get('proj_dim') == 256,
             f'{prefix}: ViT/P2 projection width must be 256')
    _require(errors, encoder.get('in_channels') == [256, 256, 256, 256],
             f'{prefix}: Hybrid Encoder must consume 256-channel P2-P5')
    _require(errors, encoder.get('feat_strides') == [4, 8, 16, 32],
             f'{prefix}: Hybrid Encoder strides must be P2-P5')
    _require(errors, encoder.get('hidden_dim') == 256,
             f'{prefix}: Hybrid Encoder hidden_dim must be 256')
    _require(errors, decoder.get('hidden_dim') == 256,
             f'{prefix}: decoder hidden_dim must be 256')
    _require(errors, decoder.get('num_queries') == 768,
             f'{prefix}: deployment query slots must be 768')
    _require(errors, decoder.get('num_layers') == 4,
             f'{prefix}: decoder must have four layers')
    _require(errors, decoder.get('endpoint_sampling_points') == 8,
             f'{prefix}: endpoint sampling points must be 8')
    _require(errors, config.get('warmup_iter') == 2000,
             f'{prefix}: warm-up must be 2000 iterations')
    _require(errors, config.get('lrsheduler') == 'flatcosine',
             f'{prefix}: learning-rate schedule must be cosine')
    _require(errors, config.get('checkpoint_freq') == 1,
             f'{prefix}: every epoch must have a checkpoint')
    _require(errors, config.get('clip_max_norm') == 1.0,
             f'{prefix}: global gradient clip must be 1.0')
    _require(errors, config.get('query_clip_max_norm') == 0.1,
             f'{prefix}: query gradient clip must be 0.1')
    effective_batch = (
        int(train_loader.get('total_batch_size', 0))
        * int(config.get('grad_accum_steps', 1))
    )
    _require(errors, effective_batch >= 16,
             f'{prefix}: effective batch must be at least 16')
    _require(errors, train_dataset.get('strict_schema') is True,
             f'{prefix}: strict schema v2 must be enabled for training')
    _require(errors, train_dataset.get('schema_version') == 2,
             f'{prefix}: training schema_version must be 2')
    _require(errors, val_dataset.get('strict_schema') is True,
             f'{prefix}: strict schema v2 must be enabled for validation')
    _require(errors, val_dataset.get('schema_version') == 2,
             f'{prefix}: validation schema_version must be 2')

    _require(errors, config.get('epochs') == expected_stage['epochs'],
             f"{prefix}: epochs must be {expected_stage['epochs']}")
    _require(errors, decoder.get('num_keypoints') == expected_stage['keypoints'],
             f"{prefix}: num_keypoints must be {expected_stage['keypoints']}")
    _require(errors, detector.get('enable_density') is expected_stage['density'],
             f"{prefix}: density stage flag is incorrect")
    _require(errors, detector.get('enable_temporal') is expected_stage['temporal'],
             f"{prefix}: temporal stage flag is incorrect")
    _require(errors, detector.get('enable_domain_adapter') is expected_stage['domain'],
             f"{prefix}: domain stage flag is incorrect")
    _require(errors, config.get('enable_online_pseudo') is expected_stage['pseudo'],
             f"{prefix}: semi-supervised stage flag is incorrect")
    _require(errors, criterion.get('loss_warmup_epochs') == 10,
             f'{prefix}: new loss warm-up must be 10 epochs')
    _require(
        errors,
        config.get('evaluation_profile') == expected_stage['evaluation_profile'],
        f"{prefix}: evaluation_profile must be "
        f"{expected_stage['evaluation_profile']}",
    )
    _require(
        errors,
        config.get('inline_eval') is expected_stage['inline_eval'],
        f"{prefix}: inline_eval must be {expected_stage['inline_eval']}",
    )
    if config_path.name == 'e1_formal_native_p2_q768_1280.yml':
        _require(
            errors,
            config.get('pareto_metrics') == {
                'bbox_ap': 'max',
                'rgb_ap': 'max',
                'ir_ap': 'max',
                'query_utilization': 'max',
            },
            f'{prefix}: E1 Pareto metrics must be detection/domain/query only',
        )

    if expected_stage['keypoints']:
        _require(errors, decoder.get('endpoint_decoder_type') == 'decomposed',
                 f'{prefix}: decomposed endpoint decoder is required')
    if expected_stage['density']:
        _require(errors, detector.get('enable_density_capacity') is True,
                 f'{prefix}: prediction-only density capacity mask is required')
        _require(errors, decoder.get('use_density_query') is True,
                 f'{prefix}: density query initialization is required')
        _require(errors, decoder.get('use_pattern_query') is True,
                 f'{prefix}: pattern query initialization is required')
    if expected_stage['temporal']:
        _require(errors, train_dataset.get('type') == 'TemporalCocoDetection',
                 f'{prefix}: temporal training dataset is required')
        _require(errors, val_dataset.get('type') == 'TemporalCocoDetection',
                 f'{prefix}: temporal validation dataset is required')
        _require(errors, train_dataset.get('temporal_frames') == 5,
                 f'{prefix}: training clip length must be five')
        _require(errors, val_dataset.get('temporal_frames') == 5,
                 f'{prefix}: validation clip length must be five')

    if config_path.name == 'e2_formal_joint_1280.yml':
        _require(errors, config.get('freeze_backbone_epochs') == 10,
                 f'{prefix}: E2 must freeze only the pretrained trunk for 10 epochs')
        _require(errors, config.get('backbone_unfreeze_layers_per_epoch') == 2,
                 f'{prefix}: E2 gradual unfreeze rate must be explicit')

    expected_lr, optimizer, trunk_groups, new_groups = _learning_rate_contract(
        config, config_path.name == 'e6_formal_final_1280.yml'
    )
    _require(errors, optimizer.get('type') == 'AdamW',
             f'{prefix}: optimizer must be AdamW')
    _require(errors, optimizer.get('lr') == expected_lr['base'],
             f"{prefix}: base LR must be {expected_lr['base']}")
    _require(errors, optimizer.get('weight_decay') == 1.25e-4,
             f'{prefix}: weight decay must be 1.25e-4')
    _require(errors, len(trunk_groups) == 2 and all(
        group.get('lr') == expected_lr['trunk'] for group in trunk_groups
    ), f"{prefix}: pretrained trunk LR must be {expected_lr['trunk']}")
    _require(errors, len(new_groups) == 2 and all(
        group.get('lr') == expected_lr['new'] for group in new_groups
    ), f"{prefix}: P2/new-module LR must be {expected_lr['new']}")

    if errors:
        raise ValueError('\n'.join(errors))
    return {
        'config': config_path.name,
        'input': 1280,
        'hidden_dim': 256,
        'p2_channels': 256,
        'queries': 768,
        'effective_batch': effective_batch,
        'base_lr': expected_lr['base'],
        'trunk_lr': expected_lr['trunk'],
        'new_module_lr': expected_lr['new'],
    }


def validate_all(config_dir):
    config_dir = Path(config_dir)
    return [validate_stage(config_dir / name) for name in STAGES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-dir', type=Path,
        default=REPO_ROOT / 'configs' / 'bee_e',
    )
    args = parser.parse_args()
    records = validate_all(args.config_dir)
    continuous = validate_continuous_route(
        args.config_dir / 'e_route_continuous_1280.yml'
    )
    print(json.dumps({
        'status': 'ok', 'stages': records, 'continuous_route': continuous,
    }, indent=2))


if __name__ == '__main__':
    main()
