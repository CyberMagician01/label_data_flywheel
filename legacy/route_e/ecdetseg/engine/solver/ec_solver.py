"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import datetime
import hashlib
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from .ec_engine import evaluate, train_one_epoch
from .pareto_checkpoint import ParetoCheckpointManager
from .semi_supervised import OnlinePseudoLabeler
from .stage_controller import BeeEStageController


class ECSolver(BaseSolver):

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def _external_eval_config(self):
        return dict(self.cfg.yaml_cfg.get('external_eval') or {})

    def _external_eval_enabled(self):
        return bool(self._external_eval_config().get('enabled', False))

    def _wait_for_external_evaluation(self, epoch, checkpoint_path):
        """Wait for a SHA-bound evaluator result published by another GPU."""
        if dist_utils.is_dist_available_and_initialized():
            raise RuntimeError('External evaluation currently requires single-process training.')
        config = self._external_eval_config()
        timeout = float(config.get('timeout_seconds', 7200))
        poll_seconds = max(float(config.get('poll_seconds', 5)), 0.1)
        metrics_dir = Path(self.output_dir) / 'checkpoint_metrics'
        metrics_path = metrics_dir / f'epoch{epoch:04}.json'
        done_path = metrics_dir / f'epoch{epoch:04}.done'
        failed_path = metrics_dir / f'epoch{epoch:04}.failed'
        checkpoint_sha = self._sha256(checkpoint_path)
        started = time.monotonic()
        next_status = started
        while True:
            if failed_path.is_file():
                raise RuntimeError(
                    f'External evaluation failed for epoch {epoch}: {failed_path}'
                )
            if metrics_path.is_file() and done_path.is_file():
                try:
                    record = json.loads(metrics_path.read_text(encoding='utf-8'))
                    done = json.loads(done_path.read_text(encoding='utf-8'))
                    eval_stats = record.get('eval_stats')
                    if (
                        record.get('sha256') == checkpoint_sha
                        and done.get('sha256') == checkpoint_sha
                        and isinstance(eval_stats, dict)
                    ):
                        print(
                            'External evaluation accepted: '
                            f'epoch={epoch} sha256={checkpoint_sha}',
                            flush=True,
                        )
                        return eval_stats
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            now = time.monotonic()
            if now - started >= timeout:
                raise TimeoutError(
                    f'External evaluation timed out for epoch {epoch} '
                    f'after {timeout:.0f}s.'
                )
            if now >= next_status:
                print(
                    'Waiting for external evaluation: '
                    f'epoch={epoch} sha256={checkpoint_sha}',
                    flush=True,
                )
                next_status = now + 60.0
            time.sleep(poll_seconds)

    def _validate_continuous_initialization(self):
        """Fresh E-S0 must start from the selected official COCO ECDet."""
        args = self.cfg
        if not args.enable_continuous_stage_training:
            return None
        if args.resume:
            return {'mode': 'resume', 'checkpoint': str(args.resume)}
        contract = args.bee_e_initialization or {}
        required = {
            'source', 'selected_scale', 'selection_evidence',
            'selection_evidence_sha256', 'checkpoint', 'sha256',
            'reinitialize_output_heads',
        }
        missing = sorted(required - set(contract))
        if missing:
            raise ValueError(
                'Fresh continuous BeePoseTrack-E initialization contract misses: '
                + ', '.join(missing)
            )
        source = str(contract['source'])
        public_preadaptation = source == 'public_bee_preadaptation'
        if source not in {'official_ecdet_coco', 'public_bee_preadaptation'}:
            raise ValueError(
                'Fresh E-S0 must use source=official_ecdet_coco or a '
                'SHA-verified public_bee_preadaptation checkpoint; '
                f"received {source!r}."
            )
        expected_reinitialized = (
            set() if public_preadaptation
            else {'category', 'quality', 'density', 'endpoints'}
        )
        if set(contract['reinitialize_output_heads']) != expected_reinitialized:
            raise ValueError(
                'Official COCO initialization must reinitialize the exact '
                'category/quality/density/endpoints heads, while a public '
                'bee-preadapted checkpoint must retain all learned heads.'
            )
        scale = str(contract['selected_scale']).upper()
        if scale not in {'S', 'M', 'L'}:
            raise ValueError('selected_scale must be one of S/M/L.')
        evidence_path = contract['selection_evidence']
        evidence_sha = str(contract['selection_evidence_sha256'] or '').lower()
        if not evidence_path or len(evidence_sha) != 64:
            raise ValueError(
                'Fresh E-S0 requires SHA-verified S/M/L frontier selection evidence.'
            )
        evidence_path = Path(evidence_path).expanduser().resolve()
        if not evidence_path.is_file():
            raise FileNotFoundError(f'S/M/L selection evidence not found: {evidence_path}')
        actual_evidence_sha = self._sha256(evidence_path)
        if actual_evidence_sha != evidence_sha:
            raise ValueError(
                f'S/M/L selection evidence SHA256 mismatch: expected={evidence_sha}, '
                f'actual={actual_evidence_sha}'
            )
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        candidates = evidence.get('candidates', [])
        candidate_scales = {str(item.get('scale', '')).upper() for item in candidates}
        if candidate_scales != {'S', 'M', 'L'}:
            raise ValueError('Scale selection evidence must contain exact S/M/L candidates.')
        if str(evidence.get('selected_scale', '')).upper() != scale:
            raise ValueError('Scale selection evidence disagrees with selected_scale.')
        frontier = {str(value).upper() for value in evidence.get('pareto_frontier', [])}
        if scale not in frontier or not frontier <= {'S', 'M', 'L'}:
            raise ValueError('selected_scale must belong to the recorded S/M/L Pareto frontier.')
        if evidence.get('dataset') != 'COCO2017' or evidence.get('extra_supervision') != '--':
            raise ValueError('Scale selection evidence must identify the official COCO-only model zoo.')
        required_measurements = {
            'official_coco_ap', 'parameters_million',
            'official_t4_trt_fp16_latency_ms', 'official_checkpoint_url',
        }
        if any(required_measurements - set(item) for item in candidates):
            raise ValueError(
                'Every official S/M/L candidate requires COCO AP, parameter count, '
                'T4 TensorRT FP16 latency, and the official checkpoint URL.'
            )
        directions = {
            'official_coco_ap': 'max', 'parameters_million': 'min',
            'official_t4_trt_fp16_latency_ms': 'min',
        }

        def dominates(left, right):
            no_worse = all(
                float(left[name]) >= float(right[name])
                if direction == 'max' else float(left[name]) <= float(right[name])
                for name, direction in directions.items()
            )
            strictly_better = any(
                float(left[name]) > float(right[name])
                if direction == 'max' else float(left[name]) < float(right[name])
                for name, direction in directions.items()
            )
            return no_worse and strictly_better

        measured_frontier = {
            str(candidate['scale']).upper()
            for candidate in candidates
            if not any(
                dominates(other, candidate)
                for other in candidates if other is not candidate
            )
        }
        if frontier != measured_frontier:
            raise ValueError(
                'Recorded S/M/L Pareto frontier disagrees with the measured '
                f'official three-objective frontier: recorded={sorted(frontier)}, '
                f'measured={sorted(measured_frontier)}.'
            )
        checkpoint = contract['checkpoint'] or args.tuning
        expected_sha = str(contract['sha256'] or '').lower()
        if not checkpoint or len(expected_sha) != 64:
            raise ValueError(
                'Fresh E-S0 requires a local official ECDet COCO checkpoint '
                'and its complete SHA256; a backbone-only weight is not sufficient.'
            )
        if str(checkpoint).startswith(('http://', 'https://')):
            raise ValueError(
                'The COCO checkpoint must be downloaded first so its SHA256 '
                'can be verified before model construction.'
            )
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'COCO initialization checkpoint not found: {checkpoint}')
        actual_sha = self._sha256(checkpoint)
        if actual_sha != expected_sha:
            raise ValueError(
                f'COCO initialization SHA256 mismatch: expected={expected_sha}, '
                f'actual={actual_sha}'
            )
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        has_model = isinstance(state, dict) and (
            isinstance(state.get('model'), dict)
            or (
                isinstance(state.get('ema'), dict)
                and isinstance(state['ema'].get('module'), dict)
            )
        )
        if not has_model:
            raise ValueError(
                'COCO initialization checkpoint must contain model or ema.module state.'
            )
        public_evidence_path = None
        actual_public_evidence_sha = None
        if public_preadaptation:
            complete_state = {
                'model', 'optimizer', 'lr_scheduler', 'scaler', 'ema', 'last_epoch',
            }
            missing_state = sorted(complete_state - set(state))
            if missing_state:
                raise ValueError(
                    'Public preadaptation checkpoint misses complete training state: '
                    + ', '.join(missing_state)
                )
            if not isinstance(state.get('ema'), dict) or not isinstance(
                state['ema'].get('module'), dict
            ):
                raise ValueError('Public preadaptation checkpoint EMA state is incomplete.')

            public_evidence_path = contract.get('public_preadaptation_evidence')
            expected_public_evidence_sha = str(
                contract.get('public_preadaptation_evidence_sha256') or ''
            ).lower()
            if not public_evidence_path or len(expected_public_evidence_sha) != 64:
                raise ValueError(
                    'public_bee_preadaptation requires a SHA-verified '
                    'public_preadaptation_evidence file.'
                )
            public_evidence_path = Path(public_evidence_path).expanduser().resolve()
            if not public_evidence_path.is_file():
                raise FileNotFoundError(
                    f'Public preadaptation evidence not found: {public_evidence_path}'
                )
            actual_public_evidence_sha = self._sha256(public_evidence_path)
            if actual_public_evidence_sha != expected_public_evidence_sha:
                raise ValueError(
                    'Public preadaptation evidence SHA256 mismatch: '
                    f'expected={expected_public_evidence_sha}, '
                    f'actual={actual_public_evidence_sha}'
                )
            public_evidence = json.loads(
                public_evidence_path.read_text(encoding='utf-8')
            )
            if int(public_evidence.get('schema_version', -1)) != 1:
                raise ValueError('Unsupported public preadaptation evidence schema.')
            if public_evidence.get('selected_checkpoint_sha256') != actual_sha:
                raise ValueError(
                    'Public preadaptation evidence does not bind the selected '
                    'checkpoint SHA256.'
                )
            if public_evidence.get('base_official_checkpoint_sha256') != (
                'c4cdf8bcd3b27c7903e422acd03caf733b5e1bfd664550bce43e50e2a3bbdc6e'
            ):
                raise ValueError(
                    'Public preadaptation evidence must trace back to the '
                    'selected official ECDet-M COCO checkpoint.'
                )
            if set(public_evidence.get('public_datasets', [])) != {
                'BEE24', 'BeePose', 'MendeleyBeePose',
            }:
                raise ValueError(
                    'Public preadaptation evidence must cover exact BEE24, '
                    'BeePose, and MendeleyBeePose sources.'
                )
            if public_evidence.get('handoff_policy') != (
                'model_and_ema_tuning_with_fresh_E-S0_optimizer_scheduler'
            ):
                raise ValueError('Public preadaptation handoff policy is invalid.')

            public_manifest_path = Path(
                public_evidence.get('public_data_manifest', '')
            ).expanduser().resolve()
            expected_manifest_sha = str(
                public_evidence.get('public_data_manifest_sha256') or ''
            ).lower()
            if not public_manifest_path.is_file() or len(expected_manifest_sha) != 64:
                raise ValueError('Public data manifest evidence is incomplete.')
            actual_manifest_sha = self._sha256(public_manifest_path)
            if actual_manifest_sha != expected_manifest_sha:
                raise ValueError('Public data manifest SHA256 mismatch.')
            public_manifest = json.loads(
                public_manifest_path.read_text(encoding='utf-8')
            )
            if set(public_manifest.get('public_datasets', [])) != {
                'BEE24', 'BeePose', 'MendeleyBeePose',
            } or any(public_manifest.get('leakage_checks', {}).values()):
                raise ValueError('Public data manifest content or leakage gate is invalid.')

            p0a = public_evidence.get('p0a') or {}
            p0a_path = Path(p0a.get('checkpoint', '')).expanduser().resolve()
            expected_p0a_sha = str(p0a.get('checkpoint_sha256') or '').lower()
            if not p0a_path.is_file() or len(expected_p0a_sha) != 64:
                raise ValueError('E-P0A lineage evidence is incomplete.')
            if self._sha256(p0a_path) != expected_p0a_sha:
                raise ValueError('E-P0A lineage checkpoint SHA256 mismatch.')
            p0a_state = torch.load(p0a_path, map_location='cpu', weights_only=True)
            if int(p0a_state.get('last_epoch', -1)) != int(p0a.get('last_epoch', -2)):
                raise ValueError('E-P0A lineage epoch disagrees with its checkpoint.')

            p0b = public_evidence.get('p0b') or {}
            selected_path = Path(
                p0b.get('selected_checkpoint', '')
            ).expanduser().resolve()
            selected_epoch = int(p0b.get('selected_epoch', -1))
            if selected_path != checkpoint:
                raise ValueError(
                    'Public preadaptation evidence selected checkpoint path disagrees '
                    'with the E-S0 initialization checkpoint.'
                )
            if not 0 <= selected_epoch < 12:
                raise ValueError('Selected E-P0B epoch is outside the complete 12-cycle stage.')
            if int(state['last_epoch']) != selected_epoch or int(
                p0b.get('last_epoch', -1)
            ) != selected_epoch:
                raise ValueError('E-P0B selected epoch disagrees with its checkpoint.')
        args.tuning = str(checkpoint)
        verified = {
            'mode': 'fresh', 'source': source,
            'selected_scale': scale, 'checkpoint': str(checkpoint),
            'sha256': actual_sha, 'selection_evidence': str(evidence_path),
            'selection_evidence_sha256': actual_evidence_sha,
        }
        if public_preadaptation:
            verified.update({
                'public_preadaptation_evidence': str(public_evidence_path),
                'public_preadaptation_evidence_sha256': actual_public_evidence_sha,
            })
        print(f'Verified BeePoseTrack-E initialization: {verified}')
        return verified

    def _validate_continuous_split_contract(self):
        """Bind training/calibration/dev-holdout files to one leakage-free 5:1:1 split."""
        args = self.cfg
        if not args.enable_continuous_stage_training:
            return None
        yaml_cfg = getattr(args, 'yaml_cfg', {})
        from tools.bee_e.split_contract import validate_schema_v2_split_contract
        return validate_schema_v2_split_contract(
            args.bee_e_split_contract or {},
            yaml_cfg.get('train_dataloader', {}).get('dataset', {}),
            yaml_cfg.get('val_dataloader', {}).get('dataset', {}),
        )

    def _validate_continuous_design_evidence(self):
        args = self.cfg
        if not args.enable_continuous_stage_training or args.resume:
            return {'mode': 'resume'} if args.resume else None
        contract = args.bee_e_design_evidence or {}
        evidence_path = contract.get('path')
        expected_sha = str(contract.get('sha256') or '').lower()
        if not evidence_path or len(expected_sha) != 64:
            raise ValueError(
                'Fresh E-S0 requires a SHA-verified route design evidence artifact.'
            )
        evidence_path = Path(evidence_path).expanduser().resolve()
        if not evidence_path.is_file():
            raise FileNotFoundError(f'route design evidence not found: {evidence_path}')
        actual_sha = self._sha256(evidence_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f'route design evidence SHA256 mismatch: expected={expected_sha}, '
                f'actual={actual_sha}'
            )
        from tools.bee_e.design_evidence import validate_route_design_evidence
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        verified = validate_route_design_evidence(evidence, args.yaml_cfg)
        return {
            'path': str(evidence_path), 'sha256': actual_sha, **verified,
        }

    def _validate_continuous_input_distribution_evidence(self):
        """Require the pre-training distribution report and every hard gate."""
        args = self.cfg
        if not args.enable_continuous_stage_training or args.resume:
            return {'mode': 'resume'} if args.resume else None
        contract = args.bee_e_input_distribution_evidence or {}
        report_path = contract.get('path')
        expected_sha = str(contract.get('sha256') or '').lower()
        if not report_path or len(expected_sha) != 64:
            raise ValueError(
                'Fresh E-S0 requires a SHA-verified input distribution report.'
            )
        report_path = Path(report_path).expanduser().resolve()
        if not report_path.is_file():
            raise FileNotFoundError(f'input distribution report not found: {report_path}')
        actual_sha = self._sha256(report_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f'input distribution report SHA256 mismatch: expected={expected_sha}, '
                f'actual={actual_sha}'
            )
        report = json.loads(report_path.read_text(encoding='utf-8'))
        expected_gates = {
            'dual_domain_coverage', 'video_coverage', 'group_isolation',
            'effective_sample_size', 'query_capacity', 'supervision_masks',
        }
        gates = report.get('gates', {})
        if set(gates) != expected_gates or not all(gates.values()):
            raise ValueError('Input distribution report has incomplete or failed gates.')
        if report.get('ready_for_training') is not True:
            raise ValueError('Input distribution report is not ready_for_training.')
        yaml_cfg = args.yaml_cfg
        expected_capacity = int(yaml_cfg['ECTransformer']['num_queries'])
        if int(report.get('query_capacity', -1)) != expected_capacity:
            raise ValueError('Input distribution report query capacity is stale.')
        if set(report.get('splits', {})) != {'train', 'calibration', 'dev_holdout'}:
            raise ValueError('Input distribution report must cover all three fixed splits.')
        expected_artifact_shas = {
            'train': str(yaml_cfg['train_dataloader']['dataset'][
                'expected_ann_sha256'
            ]).lower(),
            'calibration': str(yaml_cfg['val_dataloader']['dataset'][
                'expected_ann_sha256'
            ]).lower(),
            'dev_holdout': str(args.bee_e_split_contract[
                'dev_holdout_annotation_sha256'
            ]).lower(),
        }
        artifacts = report.get('source_artifacts', {})
        actual_artifact_shas = {
            role: str(artifacts.get(role, {}).get('sha256') or '').lower()
            for role in expected_artifact_shas
        }
        if actual_artifact_shas != expected_artifact_shas:
            raise ValueError(
                'Input distribution report is not bound to the fixed train, '
                'calibration and dev_holdout annotations.'
            )
        return {
            'path': str(report_path), 'sha256': actual_sha,
            'query_capacity': expected_capacity,
            'source_artifacts': artifacts,
        }

    def _build_stage_controller(self):
        args = self.cfg
        if not args.enable_continuous_stage_training:
            return None
        required = {
            'bee_e_stage_specs': args.bee_e_stage_specs,
            'bee_e_metric_directions': args.bee_e_metric_directions,
            'bee_e_official_primary_metric': args.bee_e_official_primary_metric,
            'bee_e_stage_constraints': args.bee_e_stage_constraints,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                'Continuous BeePoseTrack-E training requires: ' + ', '.join(missing)
            )
        external_eval = self._external_eval_enabled()
        if bool(args.inline_eval) == external_eval:
            raise ValueError(
                'Continuous stages require exactly one evaluation mode: '
                'inline_eval=true or external_eval.enabled=true.'
            )
        if args.evaluation_profile != 'full':
            raise ValueError(
                'Continuous stages require evaluation_profile=full.'
            )
        if int(args.checkpoint_freq) != 1:
            raise ValueError('Continuous stages require checkpoint_freq=1.')
        initialization = self._validate_continuous_initialization()
        design_evidence = self._validate_continuous_design_evidence()
        split_contract = self._validate_continuous_split_contract()
        input_distribution = self._validate_continuous_input_distribution_evidence()
        print(f'Verified BeePoseTrack-E route design evidence: {design_evidence}')
        print(f'Verified BeePoseTrack-E fixed split: {split_contract}')
        print(f'Verified BeePoseTrack-E input distribution: {input_distribution}')
        controller = BeeEStageController(
            args.bee_e_stage_specs,
            args.bee_e_metric_directions,
            args.bee_e_official_primary_metric,
            args.bee_e_stage_constraints,
            total_coverage_budget=args.bee_e_total_coverage_budget,
            bootstrap_samples=args.bee_e_bootstrap_samples,
            minimum_plateau_window=args.bee_e_minimum_plateau_window,
            seed=args.seed,
        )
        if int(args.epochs) != controller.total_coverage_budget:
            raise ValueError(
                'epochs must exactly equal bee_e_total_coverage_budget so the '
                'continuous learning-rate envelope and coverage budget share one clock.'
            )
        controller.provenance = {
            'initialization': initialization,
            'design_evidence': design_evidence,
            'split_contract': split_contract,
            'input_distribution': input_distribution,
        }
        return controller

    @staticmethod
    def _load_checkpoint_file(path):
        if str(path).startswith('http'):
            return torch.hub.load_state_dict_from_url(
                str(path), map_location='cpu', weights_only=True,
            )
        return torch.load(path, map_location='cpu', weights_only=True)

    def _restore_stage_best_training_state(self, path):
        """Restore train state while preserving the advanced stage clock."""
        state = self._load_checkpoint_file(path)
        required = ['model', 'optimizer', 'lr_scheduler']
        if self.scaler is not None:
            required.append('scaler')
        if self.ema is not None:
            required.append('ema')
        missing = [name for name in required if name not in state]
        if missing:
            raise RuntimeError(
                f'Stage-selected checkpoint {path} misses state: {missing}'
            )
        dist_utils.de_parallel(self.model).load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        if self.lr_warmup_scheduler is not None and 'lr_warmup_scheduler' in state:
            self.lr_warmup_scheduler.load_state_dict(state['lr_warmup_scheduler'])
        if self.scaler is not None:
            self.scaler.load_state_dict(state['scaler'])
        if self.ema is not None:
            dist_utils.de_parallel(self.ema).load_state_dict(state['ema'])
        print(f'Restored stage-best full training state from {path}')

    def _restore_rebuilt_scheduler_on_resume(self, path):
        state = self._load_checkpoint_file(path)
        if 'lr_scheduler' not in state:
            raise RuntimeError('Resume checkpoint does not contain lr_scheduler state.')
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        if self.lr_warmup_scheduler is not None and 'lr_warmup_scheduler' in state:
            self.lr_warmup_scheduler.load_state_dict(state['lr_warmup_scheduler'])

    def _stage_debts_from_metrics(self, metrics):
        definitions = self.cfg.bee_e_debt_metrics or {}
        result = {}
        for debt_kind in ('coverage', 'recall', 'generalization', 'matching'):
            values = {}
            for metric_name, rule in definitions.get(debt_kind, {}).items():
                if metric_name not in metrics:
                    continue
                metric = metrics[metric_name]
                if isinstance(metric, (list, tuple)):
                    metric = metric[0]
                if torch.is_tensor(metric):
                    metric = metric.detach().cpu().reshape(-1)[0].item()
                target = float(rule['target'])
                direction = rule.get(
                    'direction',
                    self.cfg.bee_e_metric_directions.get(metric_name, 'max'),
                )
                value = float(metric)
                values[metric_name] = max(
                    0.0, target - value if direction == 'max' else value - target,
                )
            result[debt_kind] = values
        return result

    @staticmethod
    @contextmanager
    def _preserve_training_rng():
        """Make validation observational: it must not advance training RNG."""
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _evaluate_without_training_side_effects(self, module):
        with self._preserve_training_rng():
            return evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                evaluation_profile=self.cfg.evaluation_profile,
            )

    @staticmethod
    def _set_backbone_schedule(model, epoch, freeze_epochs, layers_per_epoch):
        module = dist_utils.de_parallel(model)
        adapter = getattr(module, 'backbone', None)
        if adapter is None:
            return False, 0, 0

        # ViTAdapter contains both the pretrained ViT trunk and newly added
        # task modules (native P2, shallow-history P3 and projectors).  The E2
        # freeze contract applies only to the pretrained trunk.  Freezing the
        # whole adapter would silently freeze the very P2 branch E2 must learn.
        trunk = getattr(adapter, 'backbone', adapter)
        named_parameters = list(trunk.named_parameters())

        blocks = getattr(trunk, 'blocks', None)
        if isinstance(blocks, torch.nn.ModuleList):
            groups = {'stem': []}
            for name, parameter in named_parameters:
                if name.startswith('blocks.'):
                    block_index = name.split('.')[1]
                    group = f'blocks.{block_index}'
                else:
                    group = 'stem'
                groups.setdefault(group, []).append(parameter)
            group_names = ['stem'] + [
                f'blocks.{index}' for index in range(len(blocks))
                if f'blocks.{index}' in groups
            ]
        else:
            groups = {}
            for name, parameter in named_parameters:
                group = name.split('.')[0]
                groups.setdefault(group, []).append(parameter)
            group_names = list(groups)

        # A zero-length freeze means ordinary end-to-end fine-tuning.  It must
        # not accidentally activate the gradual-unfreeze path used by E2.
        if freeze_epochs <= 0:
            enabled = set(group_names)
        elif epoch < freeze_epochs:
            enabled = set()
        elif layers_per_epoch <= 0:
            enabled = set(group_names)
        else:
            count = min(
                len(group_names), (epoch - freeze_epochs + 1) * layers_per_epoch
            )
            enabled = set(group_names[-count:])
        for group_name, parameters in groups.items():
            trainable = group_name in enabled
            for parameter in parameters:
                parameter.requires_grad_(trainable)

        # Task-specific adapter modules remain trainable throughout the
        # pretrained-trunk freeze and gradual-unfreeze schedule.
        if trunk is not adapter:
            trunk_parameter_ids = {id(parameter) for _, parameter in named_parameters}
            for parameter in adapter.parameters():
                if id(parameter) not in trunk_parameter_ids:
                    parameter.requires_grad_(True)

        trainable_count = sum(parameter.numel() for _, parameter in named_parameters if parameter.requires_grad)
        total_count = sum(parameter.numel() for _, parameter in named_parameters)
        return trainable_count == 0, trainable_count, total_count

    def fit(self, ):
        self.stage_controller = self._build_stage_controller()
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)
        
        stop_aug_epoch = self.train_dataloader.dataset._transforms.stop_epoch  # epoch to stop augmentation
        if args.lrsheduler is not None:
            no_aug_epochs = args.epochs - stop_aug_epoch
            flat_epochs = self.train_dataloader.dataset._transforms.mosaic_epoch if args.flat_epoch is None else args.flat_epoch
            # The ECDet contract defines warm-up in data iterations.  Gradient
            # accumulation changes how often optimizer.step() runs, but must
            # not stretch a 2000-iteration warm-up to 2000 optimizer updates.
            iter_per_epoch = (
                args.lr_schedule_iterations_per_epoch
                or len(self.train_dataloader)
            )
            warmup_iter = args.warmup_iter
            
            print(
                'FlatCosineLRScheduler with '
                f'flat_epochs: {flat_epochs}, no_aug_epochs: {no_aug_epochs}, '
                f'warmup_data_iterations: {warmup_iter}, '
                f'data_iterations_per_epoch: {iter_per_epoch}'
            )
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                args.lr_gamma,
                iter_per_epoch,
                total_epochs=args.epochs,
                warmup_iter=warmup_iter,
                flat_epochs=flat_epochs,
                no_aug_epochs=no_aug_epochs,
                base_lrs=self.optimizer_base_lrs,
            )
            self.self_lr_scheduler = True
            if args.resume:
                self._restore_rebuilt_scheduler_on_resume(args.resume)
        else:
            self.self_lr_scheduler = False

        best_stat = {'epoch': -1, }
        # Inline evaluation is disabled for E1 async mode.  Skipping the
        # resume evaluation is important: evaluation must never perturb the
        # training queue or consume GPU1 time.
        if (
            self.last_epoch > 0 and args.inline_eval
            and self.stage_controller is None
        ):
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = self._evaluate_without_training_side_effects(
                module
            )
            for k in test_stats:
                if k not in ('coco_eval_bbox', 'coco_eval_mask'):
                    continue
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        effective_batch_size = (
            self.train_dataloader.batch_size
            * max(1, int(args.grad_accum_steps))
            * dist_utils.get_world_size()
        )
        if effective_batch_size < args.min_effective_batch_size:
            raise ValueError(
                f'effective batch size {effective_batch_size} is below '
                f'{args.min_effective_batch_size}; increase grad_accum_steps.'
            )
        print(f'Effective batch size: {effective_batch_size}')
        pareto_manager = None
        if (
            args.inline_eval
            and self.output_dir
            and args.pareto_metrics
            and dist_utils.is_main_process()
        ):
            pareto_manager = ParetoCheckpointManager(
                self.output_dir, metric_directions=args.pareto_metrics
            )
        pseudo_labeler = None
        if args.enable_online_pseudo:
            if self.ema is None:
                raise ValueError('Online pseudo labels require use_ema=True.')
            pseudo_labeler = OnlinePseudoLabeler(
                score_threshold=args.pseudo_score_threshold,
                quality_threshold=args.pseudo_quality_threshold,
                set_iou_threshold=args.pseudo_set_iou_threshold,
                endpoint_threshold=args.pseudo_endpoint_threshold,
                trajectory_iou_threshold=args.pseudo_trajectory_iou_threshold,
                max_pseudo_instances=args.max_pseudo_instances,
                ignore_score_threshold=args.pseudo_ignore_score_threshold,
            )
        for epoch in range(start_epoch, args.epochs):

            stage_context = None
            if self.stage_controller is not None:
                if self.stage_controller.completed:
                    print('Continuous BeePoseTrack-E stages are already complete.')
                    break
                if self.stage_controller.blocked_reason:
                    raise RuntimeError(self.stage_controller.blocked_reason)
                stage_context = self.stage_controller.apply(
                    self.model, self.criterion, self.train_dataloader,
                )
                print(f'BeePoseTrack-E stage context: {stage_context}')

            self.train_dataloader.set_epoch(epoch)
            if hasattr(self.criterion, 'set_epoch'):
                self.criterion.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
                
            if self.stage_controller is None and epoch == stop_aug_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                if args.stop_aug_resume_checkpoint:
                    print(
                        'Stop-augmentation transition explicitly loads: '
                        f'{args.stop_aug_resume_checkpoint}'
                    )
                    self.load_resume_state(args.stop_aug_resume_checkpoint)
                    self.last_epoch = epoch - 1
                else:
                    print('Stop-augmentation transition continues current weights.')

            if self.stage_controller is None:
                backbone_frozen, trainable_backbone, total_backbone = self._set_backbone_schedule(
                    self.model,
                    epoch,
                    args.freeze_backbone_epochs,
                    args.backbone_unfreeze_layers_per_epoch,
                )
            else:
                backbone_frozen = False
                bare_model = dist_utils.de_parallel(self.model)
                backbone = getattr(bare_model, 'backbone', None)
                total_backbone = sum(
                    parameter.numel() for parameter in backbone.parameters()
                ) if backbone is not None else 0
                trainable_backbone = sum(
                    parameter.numel() for parameter in backbone.parameters()
                    if parameter.requires_grad
                ) if backbone is not None else 0
            print(
                f'Backbone schedule epoch={epoch}: '
                f'{trainable_backbone}/{total_backbone} trainable parameters'
            )

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                grad_accum_steps=args.grad_accum_steps,
                query_max_norm=args.query_clip_max_norm,
                freeze_backbone=backbone_frozen,
                pseudo_labeler=(
                    pseudo_labeler
                    if stage_context is None or stage_context['soft_pseudo_enabled']
                    else None
                ),
                log_gradient_conflicts_every=args.log_gradient_conflicts_every,
                enable_ddp_no_sync=args.enable_ddp_no_sync,
                enable_explicit_accumulated_gradient_sync=(
                    args.enable_explicit_accumulated_gradient_sync
                ),
                require_equal_domain_updates=args.require_equal_domain_updates,
                scheduler_reference_batch_size=args.scheduler_reference_batch_size,
                lr_schedule_iterations_per_epoch=args.lr_schedule_iterations_per_epoch,
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            epoch_checkpoint = None
            if self.output_dir:
                checkpoint_paths = [(self.output_dir / 'last.pth', False)]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    epoch_checkpoint = self.output_dir / f'checkpoint{epoch:04}.pth'
                    checkpoint_paths.append((epoch_checkpoint, True))
                checkpoint_state = self.state_dict()
                for checkpoint_path, write_ready in checkpoint_paths:
                    dist_utils.atomic_save_on_master(
                        checkpoint_state,
                        checkpoint_path,
                        write_ready=write_ready,
                    )

            test_stats = {}
            coco_evaluator = None
            should_inline_eval = (
                args.inline_eval
                and (
                    (epoch + 1) % max(1, int(args.inline_eval_freq)) == 0
                    or (epoch + 1) == args.epochs
                )
            )
            if should_inline_eval:
                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = self._evaluate_without_training_side_effects(
                    module
                )
            elif self._external_eval_enabled():
                test_stats = self._wait_for_external_evaluation(
                    epoch, epoch_checkpoint,
                )

            for k in test_stats:
                values = test_stats[k] if isinstance(test_stats[k], (list, tuple)) else [test_stats[k]]
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(values):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k not in ('coco_eval_bbox', 'coco_eval_mask'):
                    continue

                if k in best_stat:
                    best_stat['epoch'] = epoch if values[0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], values[0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = values[0]

                best_stat_print[k] = best_stat[k]
                print(f'best_stat: {best_stat_print}')  # global best

            if pareto_manager is not None and epoch_checkpoint is not None:
                selected = pareto_manager.record(epoch, epoch_checkpoint, test_stats)
                if selected is None:
                    print('Pareto selection pending: required metrics are incomplete.')
                else:
                    print(
                        'Pareto selected checkpoint: '
                        f"epoch={selected['epoch']} score={selected['balanced_score']:.6f}"
                    )

            stage_decision = None
            if self.stage_controller is not None:
                debts = self._stage_debts_from_metrics(test_stats)
                self.stage_controller.update_debts(**debts)
                stage_decision = self.stage_controller.record_cycle(
                    test_stats, epoch_checkpoint,
                )
                if self.output_dir:
                    checkpoint_state = self.state_dict()
                    dist_utils.atomic_save_on_master(
                        checkpoint_state, self.output_dir / 'last.pth',
                    )
                    dist_utils.atomic_save_on_master(
                        checkpoint_state, epoch_checkpoint, write_ready=True,
                    )
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                print(f'BeePoseTrack-E stage decision: {stage_decision}')
                if stage_decision.get('blocked'):
                    raise RuntimeError(stage_decision['reason'])
                if stage_decision.get('transition'):
                    self._restore_stage_best_training_state(
                        stage_decision['selected']['checkpoint'],
                    )
                elif stage_decision.get('completed'):
                    self._restore_stage_best_training_state(
                        stage_decision['selected']['checkpoint'],
                    )
                if stage_decision.get('transition') or stage_decision.get('completed'):
                    transition_name = (
                        f"{stage_decision.get('from_stage', 'E-S5')}_to_"
                        f"{stage_decision.get('to_stage', 'complete')}.pth"
                    )
                    selected_state = self.state_dict()
                    dist_utils.atomic_save_on_master(
                        selected_state, self.output_dir / 'last.pth',
                    )
                    dist_utils.atomic_save_on_master(
                        selected_state,
                        self.output_dir / f'stage_transition_{transition_name}',
                        write_ready=True,
                    )
                    if dist_utils.is_dist_available_and_initialized():
                        torch.distributed.barrier()

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if self.iou_type in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval[self.iou_type].eval,
                                    self.output_dir / "eval" / name)
            accelerator = getattr(torch, dist_utils.current_device(), None)
            if accelerator is not None and hasattr(accelerator, 'empty_cache'):
                accelerator.empty_cache()
            if stage_decision is not None and stage_decision.get('completed'):
                print('BeePoseTrack-E E-S0..E-S5 continuous training completed.')
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = self._evaluate_without_training_side_effects(module)

        if self.output_dir and coco_evaluator is not None:
            dist_utils.save_on_master(coco_evaluator.coco_eval[self.iou_type].eval, self.output_dir / "eval.pth")

        return
