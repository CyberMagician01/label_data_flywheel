"""Continuous E-S0..E-S5 controller with coverage and calibration gates."""

import math
import random
from copy import deepcopy

import torch

from ..edgecrafter.bee_e_layers import (
    DomainConditionedLayerNorm,
    DomainLowRankAdapter2d,
    DomainSpecificBatchNorm2d,
    StabilizedTemporalResidual,
)
from ..misc import dist_utils


STAGES = ('E-S0', 'E-S1', 'E-S2', 'E-S3', 'E-S4', 'E-S5')

_DISABLED_PARAMETER_TOKENS = {
    'E-S0': ('native_p2', 'density', 'temporal', 'history', 'domain', 'prototype', 'track', 'pattern'),
    'E-S1': ('temporal', 'history', 'domain', 'prototype', 'track'),
    'E-S2': ('temporal', 'history', 'density', 'track'),
    'E-S3': (),
    'E-S4': (),
    'E-S5': (),
}


def _metric_value(metrics, name):
    value = metrics.get(name)
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)[0].item()
    return None if value is None else float(value)


class BeeEStageController:
    """Stateful stage controller saved inside every training checkpoint."""

    def __init__(self, stage_specs, metric_directions, official_primary_metric,
                 constraints, total_coverage_budget=None, bootstrap_samples=1000,
                 minimum_plateau_window=4, seed=2026):
        if [item['name'] for item in stage_specs] != list(STAGES):
            raise ValueError('stage_specs must define E-S0 through E-S5 in order')
        self.stage_specs = deepcopy(stage_specs)
        for spec in self.stage_specs:
            if not 0 < int(spec['min_cycles']) <= int(spec['max_cycles']):
                raise ValueError(f"invalid coverage bounds for {spec['name']}")
        self.metric_directions = dict(metric_directions)
        self.official_primary_metric = str(official_primary_metric)
        if self.official_primary_metric not in self.metric_directions:
            raise ValueError('official primary metric requires a direction')
        self.constraints = deepcopy(constraints)
        maximum_budget = sum(int(spec['max_cycles']) for spec in self.stage_specs)
        self.total_coverage_budget = int(total_coverage_budget or maximum_budget)
        if self.total_coverage_budget < sum(int(spec['min_cycles']) for spec in self.stage_specs):
            raise ValueError('total coverage budget is below stage minimums')
        if self.total_coverage_budget != maximum_budget:
            raise ValueError(
                'total_coverage_budget must equal the sum of stage max_cycles; '
                'early-stage saved cycles are reallocated to E-S4/E-S5.'
            )
        self.bootstrap_samples = int(bootstrap_samples)
        self.minimum_plateau_window = int(minimum_plateau_window)
        if self.bootstrap_samples <= 0:
            raise ValueError('bootstrap_samples must be positive')
        if self.minimum_plateau_window < 3:
            raise ValueError('minimum_plateau_window must be at least 3')
        self.seed = int(seed)
        self.current_stage_index = 0
        self.cycle_in_stage = 0
        self.global_cycle = 0
        self.history = {stage: [] for stage in STAGES}
        self.best = {stage: None for stage in STAGES}
        self.completed = False
        self.blocked_reason = None
        self.coverage_debt = {}
        self.recall_debt = {}
        self.generalization_gap = {}
        self.matching_debt = {}
        self.provenance = {}

    @property
    def stage(self):
        return STAGES[self.current_stage_index]

    @property
    def spec(self):
        return self.stage_specs[self.current_stage_index]

    @property
    def progress(self):
        return min(self.cycle_in_stage / max(int(self.spec['max_cycles']), 1), 1.0)

    def state_dict(self):
        return {
            'current_stage_index': self.current_stage_index,
            'cycle_in_stage': self.cycle_in_stage,
            'global_cycle': self.global_cycle,
            'history': deepcopy(self.history),
            'best': deepcopy(self.best),
            'completed': self.completed,
            'blocked_reason': self.blocked_reason,
            'coverage_debt': deepcopy(self.coverage_debt),
            'recall_debt': deepcopy(self.recall_debt),
            'generalization_gap': deepcopy(self.generalization_gap),
            'matching_debt': deepcopy(self.matching_debt),
            'stage_specs': deepcopy(self.stage_specs),
            'provenance': deepcopy(self.provenance),
        }

    def load_state_dict(self, state):
        for key in (
            'current_stage_index', 'cycle_in_stage', 'global_cycle', 'history',
            'best', 'completed', 'blocked_reason', 'coverage_debt',
            'recall_debt', 'generalization_gap', 'stage_specs',
            'matching_debt',
            'provenance',
        ):
            if key in state:
                setattr(self, key, deepcopy(state[key]))

    def update_debts(self, coverage=None, recall=None, generalization=None,
                     matching=None):
        if coverage is not None:
            self.coverage_debt = {str(key): float(value) for key, value in coverage.items()}
        if recall is not None:
            self.recall_debt = {str(key): float(value) for key, value in recall.items()}
        if generalization is not None:
            self.generalization_gap = {
                str(key): float(value) for key, value in generalization.items()
            }
        if matching is not None:
            self.matching_debt = {
                str(key): float(value) for key, value in matching.items()
            }

    def _stage_constraints(self):
        merged = deepcopy(self.constraints.get('all', {}))
        merged.update(self.constraints.get(self.stage, {}))
        return merged

    def _constraint_pass(self, metrics):
        stage_constraints = self._stage_constraints()
        for name, rule in stage_constraints.items():
            value = _metric_value(metrics, name)
            if value is None:
                return False
            if isinstance(rule, dict):
                if 'min' in rule and value < float(rule['min']):
                    return False
                if 'max' in rule and value > float(rule['max']):
                    return False
            else:
                direction = self.metric_directions.get(name, 'max')
                if direction == 'max' and value < float(rule):
                    return False
                if direction == 'min' and value > float(rule):
                    return False
        return True

    def _record_score(self, record):
        value = _metric_value(record['metrics'], self.official_primary_metric)
        if value is None:
            return -float('inf')
        direction = self.metric_directions[self.official_primary_metric]
        primary = value if direction == 'max' else -value
        margins = []
        stage_constraints = self._stage_constraints()
        for name, rule in stage_constraints.items():
            metric = _metric_value(record['metrics'], name)
            if metric is None:
                continue
            if isinstance(rule, dict) and 'min' in rule:
                margins.append(metric - float(rule['min']))
            elif isinstance(rule, dict) and 'max' in rule:
                margins.append(float(rule['max']) - metric)
        return primary, min(margins, default=0.0), -record['cycle']

    def _update_best(self):
        feasible = [
            record for record in self.history[self.stage]
            if self._constraint_pass(record['metrics'])
            and _metric_value(record['metrics'], self.official_primary_metric) is not None
        ]
        if feasible:
            self.best[self.stage] = max(feasible, key=self._record_score)

    def _bootstrap_noise(self, values, window):
        generator = random.Random(self.seed + self.global_cycle * 7919 + window)
        slopes = []
        x = list(range(window))
        x_mean = sum(x) / window
        denominator = sum((item - x_mean) ** 2 for item in x)
        for _ in range(self.bootstrap_samples):
            sample = [values[generator.randrange(window)] for _ in range(window)]
            mean = sum(sample) / window
            slopes.append(abs(sum(
                (index - x_mean) * (value - mean)
                for index, value in enumerate(sample)
            ) / max(denominator, 1e-12)))
        slopes.sort()
        return slopes[min(len(slopes) - 1, math.floor(0.95 * len(slopes)))]

    def _plateau(self):
        values = [
            _metric_value(record['metrics'], self.official_primary_metric)
            for record in self.history[self.stage]
        ]
        values = [value for value in values if value is not None]
        for window in range(self.minimum_plateau_window, len(values) + 1):
            tail = values[-window:]
            mean = sum(tail) / window
            centered = [value - mean for value in tail]
            variance = sum(value ** 2 for value in centered)
            autocorrelation = (
                sum(centered[index] * centered[index - 1] for index in range(1, window))
                / max(variance, 1e-12)
            )
            critical = 1.96 / math.sqrt(window)
            x_mean = (window - 1) / 2
            denominator = sum((index - x_mean) ** 2 for index in range(window))
            slope = abs(sum(
                (index - x_mean) * (value - mean)
                for index, value in enumerate(tail)
            ) / max(denominator, 1e-12))
            if abs(autocorrelation) <= critical and slope <= self._bootstrap_noise(tail, window):
                return True, window
        return False, None

    def _allocate_saved_budget(self, saved_cycles):
        if saved_cycles <= 0 or self.current_stage_index >= 5:
            return
        if self.current_stage_index == 4:
            self.stage_specs[5]['max_cycles'] += saved_cycles
            return
        recall = sum(max(value, 0.0) for value in self.recall_debt.values())
        generalization = sum(max(value, 0.0) for value in self.generalization_gap.values())
        denominator = recall + generalization
        s4_extra = round(saved_cycles * (recall / denominator)) if denominator else saved_cycles // 2
        s5_extra = saved_cycles - s4_extra
        self.stage_specs[4]['max_cycles'] += s4_extra
        self.stage_specs[5]['max_cycles'] += s5_extra

    def record_cycle(self, metrics, checkpoint_path):
        if self.completed or self.blocked_reason:
            raise RuntimeError('cannot record a completed or blocked stage controller')
        self.cycle_in_stage += 1
        self.global_cycle += 1
        record = {
            'stage': self.stage, 'cycle': self.cycle_in_stage,
            'global_cycle': self.global_cycle, 'checkpoint': str(checkpoint_path),
            'metrics': deepcopy(metrics),
        }
        self.history[self.stage].append(record)
        self._update_best()
        minimum = int(self.spec['min_cycles'])
        maximum = int(self.spec['max_cycles'])
        plateau, plateau_window = self._plateau()
        feasible = self.best[self.stage] is not None
        minimum_progress = float(self.spec.get(
            'minimum_progress_for_transition', minimum / maximum,
        ))
        if not 0.0 < minimum_progress <= 1.0:
            raise ValueError('minimum_progress_for_transition must be in (0, 1]')
        should_transition = (
            self.cycle_in_stage >= minimum
            and self.progress >= minimum_progress
            and plateau
            and feasible
        )
        forced = self.cycle_in_stage >= maximum
        if forced and not feasible:
            self.blocked_reason = (
                f'{self.stage} reached max_cycles={maximum} without a checkpoint '
                'satisfying all calibration constraints'
            )
            return {'transition': False, 'blocked': True, 'reason': self.blocked_reason}
        if not should_transition and not forced:
            return {
                'transition': False, 'blocked': False,
                'plateau': plateau, 'plateau_window': plateau_window,
            }
        selected = deepcopy(self.best[self.stage])
        if self.stage == 'E-S5':
            self.completed = True
            return {
                'transition': False, 'completed': True,
                'selected': selected, 'plateau_window': plateau_window,
            }
        saved = max(0, maximum - self.cycle_in_stage)
        self.stage_specs[self.current_stage_index]['max_cycles'] = self.cycle_in_stage
        self._allocate_saved_budget(saved)
        previous_stage = self.stage
        self.current_stage_index += 1
        self.cycle_in_stage = 0
        return {
            'transition': True, 'blocked': False,
            'from_stage': previous_stage, 'to_stage': self.stage,
            'selected': selected, 'plateau_window': plateau_window,
        }

    def apply(self, model, criterion, train_dataloader):
        """Apply branch gates, query budget, matching ramp and data policy."""
        bare_model = dist_utils.de_parallel(model)
        disabled = _DISABLED_PARAMETER_TOKENS[self.stage]
        for name, parameter in bare_model.named_parameters():
            # Some deploy-time caches (for example integer anchor indices) are
            # registered as non-floating Parameters. They are valid model
            # state but can never carry gradients, so stage gating must leave
            # them non-trainable instead of calling requires_grad_(True).
            if parameter.is_floating_point() or parameter.is_complex():
                parameter.requires_grad_(
                    not any(token in name for token in disabled)
                )
        backbone_adapter = getattr(bare_model, 'backbone', None)
        backbone_trunk = getattr(backbone_adapter, 'backbone', backbone_adapter)
        if backbone_trunk is not None and self.stage == 'E-S0':
            for parameter in backbone_trunk.parameters():
                parameter.requires_grad_(False)
            blocks = getattr(backbone_trunk, 'blocks', None)
            trainable_blocks = int(self.spec.get('trainable_backbone_blocks', 0))
            if trainable_blocks > 0 and isinstance(blocks, torch.nn.ModuleList):
                for block in blocks[-trainable_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad_(True)
        domain_trainable = self.current_stage_index >= 2
        for module in bare_model.modules():
            if isinstance(module, (
                DomainConditionedLayerNorm, DomainLowRankAdapter2d,
            )):
                for parameter in module.parameters():
                    parameter.requires_grad_(domain_trainable)
            if isinstance(module, DomainSpecificBatchNorm2d):
                for parameter in module.parameters():
                    parameter.requires_grad_(domain_trainable)
                module.freeze_running_stats = (
                    not domain_trainable or self.stage == 'E-S5'
                )
        if self.stage == 'E-S0':
            query_fraction = float(self.spec.get('initial_query_fraction', 0.25))
        elif self.stage == 'E-S1':
            initial = float(self.stage_specs[0].get('initial_query_fraction', 0.25))
            query_fraction = initial + (1.0 - initial) * self.progress
        elif self.stage == 'E-S2':
            query_fraction = 1.0
        else:
            query_fraction = 1.0
        decoder = getattr(bare_model, 'decoder', None)
        if decoder is not None and hasattr(decoder, 'num_queries'):
            minimum = int(getattr(decoder, 'min_active_queries', 1))
            decoder.stage_query_limit = max(
                minimum, min(decoder.num_queries, math.ceil(decoder.num_queries * query_fraction)),
            )
            if hasattr(decoder, 'set_stage_denoising'):
                matching_debt = max(self.matching_debt.values(), default=0.0)
                decoder.set_stage_denoising(
                    self.stage, self.progress, matching_debt=matching_debt,
                )
        temporal_multiplier = (
            self.progress if self.stage == 'E-S3' else float(self.current_stage_index > 3)
        )
        for module in bare_model.modules():
            if isinstance(module, StabilizedTemporalResidual):
                module.set_stage_multiplier(temporal_multiplier)
        pose_multiplier = (
            self.progress if self.stage == 'E-S2'
            else float(self.current_stage_index > 2)
        )
        if hasattr(criterion, 'set_stage_context'):
            criterion.set_stage_context(
                self.stage, self.progress, pose_multiplier,
                recall_debt=self.recall_debt,
            )
        transforms = getattr(train_dataloader.dataset, '_transforms', None)
        if transforms is not None and hasattr(transforms, 'set_stage'):
            transforms.set_stage(
                self.stage, self.progress,
                feedback={
                    'initial_query_fraction': self.stage_specs[0].get(
                        'initial_query_fraction', 0.25,
                    ),
                    'recall_debt': self.recall_debt,
                },
            )
        sampler = getattr(train_dataloader, 'sampler', None)
        if self.stage != 'E-S5' and hasattr(sampler, 'update_feedback'):
            sampler.update_feedback(
                self.coverage_debt, self.recall_debt, self.generalization_gap,
            )
        return {
            'stage': self.stage, 'progress': self.progress,
            'query_fraction': query_fraction,
            'temporal_multiplier': temporal_multiplier,
            'pose_match_multiplier': pose_multiplier,
            'soft_pseudo_enabled': self.stage in ('E-S3', 'E-S4'),
        }
