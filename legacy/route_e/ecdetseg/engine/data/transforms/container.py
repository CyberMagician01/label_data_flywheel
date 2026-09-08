"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T

from ...core import GLOBAL_CONFIG, register
from ._transforms import EmptyTransform

torchvision.disable_beta_transforms_warning()
import random


BEE_E_STAGE_ORDER = ('E-S0', 'E-S1', 'E-S2', 'E-S3', 'E-S4', 'E-S5')

# Probabilities multiply, rather than replace, each transform's calibrated p.
# Zero is an explicit prohibition from the route specification.
BEE_E_STAGE_VIEW_MULTIPLIERS = {
    'E-S0': {
        'BeeDirectionRotation': 1.0,
        'RandomHorizontalFlipWithKeypoints': 1.0,
        'IRPercentileNormalize': 1.0,
        'IRSensorAugment': 0.35,
        'BeeRGBSensorAugment': 0.35,
        'PrepareTemporalFrames': 1.0,
    },
    'E-S1': {
        'BeeTwoImageStitch': 1.0,
        'BeeTargetCenteredCrop': 1.0,
        'BeeDensityConstrainedCrop': 1.0,
        'RandomHorizontalFlipWithKeypoints': 1.0,
        'IRPercentileNormalize': 1.0,
        'IRSensorAugment': 0.60,
        'BeeRGBSensorAugment': 0.60,
        'PrepareTemporalFrames': 1.0,
    },
    'E-S2': {
        'BeeTargetCenteredCrop': 1.0,
        'BeeDensityConstrainedCrop': 1.0,
        'BeeDirectionGapRotation': 1.0,
        'RandomBeeKeypointOcclusion': 1.0,
        'RandomHorizontalFlipWithKeypoints': 1.0,
        'IRPercentileNormalize': 1.0,
        'IRSensorAugment': 1.0,
        'BeeRGBSensorAugment': 1.0,
        'PrepareTemporalFrames': 1.0,
    },
    'E-S3': {
        'BeeTargetCenteredCrop': 0.25,
        'BeeDensityConstrainedCrop': 0.25,
        'BeeTrajectoryTailAugment': 1.0,
        'RandomHorizontalFlipWithKeypoints': 0.5,
        'IRPercentileNormalize': 1.0,
        'IRSensorAugment': 0.25,
        'BeeRGBSensorAugment': 0.25,
        'PrepareTemporalFrames': 1.0,
    },
    'E-S4': {
        'BeeTargetCenteredCrop': 1.0,
        'BeeDensityConstrainedCrop': 1.0,
        'BeeTrajectoryTailAugment': 1.0,
        'RandomBeeKeypointOcclusion': 0.5,
        'RandomHorizontalFlipWithKeypoints': 0.5,
        'IRPercentileNormalize': 1.0,
        'IRSensorAugment': 1.0,
        'BeeRGBSensorAugment': 1.0,
        'PrepareTemporalFrames': 1.0,
    },
    'E-S5': {
        'RandomHorizontalFlipWithKeypoints': 0.25,
        'IRPercentileNormalize': 1.0,
        'IRSensorAugment': 0.20,
        'BeeRGBSensorAugment': 0.20,
        'PrepareTemporalFrames': 1.0,
    },
}

_BEE_E_ALWAYS_ON = {
    'EmptyTransform', 'LetterBox', 'SanitizeBoundingBoxes', 'ConvertPILImage',
    'Normalize', 'ConvertBoxes', 'ConvertKeypoints', 'PadToSize', 'Resize',
}

_BEE_E_DOMAIN_RESTRICTIONS = {
    'BeeTargetCenteredCrop': {0},
    'BeeDensityConstrainedCrop': {1},
}


@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None, remove_ops=None, mosaic_epoch=-1,
                 mosaic_prob=-1, stop_epoch=None, bee_e_stage='E-S0',
                 stage_progress=0.0, bee_e_resolutions=None) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transform = getattr(GLOBAL_CONFIG[name]['_pymodule'], GLOBAL_CONFIG[name]['_name'])(**op)
                    transforms.append(transform)
                    op['type'] = name

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]

        super().__init__(transforms=transforms)

        self.mosaic_prob = mosaic_prob
        if policy is None:
            policy = 'default'
        self.global_samples = 0
        self.policy = policy
        
        self.strong_augmentation = remove_ops
        self.mosaic_epoch = mosaic_epoch
        self.stop_epoch = stop_epoch
        self.cur_epoch = 0
        self.bee_e_stage = None
        self.stage_progress = 0.0
        self.stage_feedback = {}
        self.bee_e_resolutions = dict(bee_e_resolutions or {})
        self.current_bee_e_resolution = None
        self.set_stage(bee_e_stage, stage_progress)
        
    def set_epoch(self, epoch: int):
        self.cur_epoch = epoch

    def set_stage(self, stage, progress=0.0, feedback=None):
        if stage not in BEE_E_STAGE_ORDER:
            raise ValueError(f'Unknown BeePoseTrack-E stage: {stage!r}')
        if not 0.0 <= float(progress) <= 1.0:
            raise ValueError('stage progress must be in [0, 1]')
        self.bee_e_stage = stage
        self.stage_progress = float(progress)
        if feedback is not None:
            self.stage_feedback = dict(feedback)
        for transform in self.transforms:
            setter = getattr(transform, 'set_stage_context', None)
            if setter is not None:
                setter(
                    stage=stage, progress=self.stage_progress,
                    feedback=self.stage_feedback,
                )

    def _select_bee_e_resolution(self):
        if not self.bee_e_resolutions:
            return None
        required = {'low', 'mid', 'full'}
        if not required <= self.bee_e_resolutions.keys():
            raise ValueError('bee_e_resolutions requires low, mid and full values')
        if self.bee_e_stage == 'E-S0':
            key = 'low'
        elif self.bee_e_stage == 'E-S1':
            key = ('low' if self.stage_progress < 1 / 3 else
                   'mid' if self.stage_progress < 2 / 3 else 'full')
        elif self.bee_e_stage == 'E-S2':
            key = 'full' if random.random() < self.stage_progress else 'mid'
        else:
            key = 'full'
        value = self.bee_e_resolutions[key]
        return (int(value), int(value)) if isinstance(value, int) else tuple(value)

    def _apply_bee_e_resolution(self):
        size = self._select_bee_e_resolution()
        if size is None:
            return
        self.current_bee_e_resolution = size
        for transform in self.transforms:
            if type(transform).__name__ in {'LetterBox', 'PrepareTemporalFrames', 'Resize'}:
                transform.size = size
     
    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy)(*inputs)

    def get_forward(self, name):
        forwards = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
            'bee_e_staged': self.bee_e_staged_forward,
        }
        return forwards[name]

    @staticmethod
    def _sample_domain(sample):
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            return 0
        target = sample[1]
        value = target.get('domain_id', torch.tensor([0]))
        return int(torch.as_tensor(value).reshape(-1)[0].item())

    def _annotate_bee_e_stage(self, sample):
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            return sample
        image, target, *extra = sample
        target = dict(target)
        stage_index = BEE_E_STAGE_ORDER.index(self.bee_e_stage)
        initial_fraction = float(self.stage_feedback.get('initial_query_fraction', 0.25))
        if self.bee_e_stage == 'E-S0':
            query_fraction = initial_fraction
        elif self.bee_e_stage == 'E-S1':
            query_fraction = initial_fraction + (1.0 - initial_fraction) * self.stage_progress
        elif self.bee_e_stage == 'E-S2':
            query_fraction = max(initial_fraction, self.stage_progress)
        else:
            query_fraction = 1.0
        target['bee_e_stage_index'] = torch.tensor(stage_index, dtype=torch.int64)
        target['bee_e_stage_progress'] = torch.tensor(self.stage_progress, dtype=torch.float32)
        target['query_capacity_fraction'] = torch.tensor(query_fraction, dtype=torch.float32)
        target['pose_match_multiplier'] = torch.tensor(
            self.stage_progress if self.bee_e_stage == 'E-S2'
            else float(stage_index > 2), dtype=torch.float32,
        )
        target['temporal_stage_multiplier'] = torch.tensor(
            self.stage_progress if self.bee_e_stage == 'E-S3'
            else float(stage_index > 3), dtype=torch.float32,
        )
        target['soft_pseudo_enabled'] = torch.tensor(
            self.bee_e_stage in ('E-S3', 'E-S4'), dtype=torch.bool,
        )
        if self.current_bee_e_resolution is not None:
            target['bee_e_input_size'] = torch.tensor(
                self.current_bee_e_resolution, dtype=torch.int64,
            )
        output = (image, target, *extra)
        return output if extra else output[:2]

    def bee_e_staged_forward(self, *inputs: Any):
        self._apply_bee_e_resolution()
        sample = inputs if len(inputs) > 1 else inputs[0]
        sample = self._annotate_bee_e_stage(sample)
        domain = self._sample_domain(sample)
        enabled = BEE_E_STAGE_VIEW_MULTIPLIERS[self.bee_e_stage]
        for transform in self.transforms:
            name = type(transform).__name__
            if name in _BEE_E_ALWAYS_ON:
                sample = transform(sample)
                continue
            allowed_domains = _BEE_E_DOMAIN_RESTRICTIONS.get(name)
            if allowed_domains is not None and domain not in allowed_domains:
                continue
            multiplier = float(enabled.get(name, 0.0))
            if multiplier <= 0.0 or random.random() > multiplier:
                continue
            sample = transform(sample)
        return sample

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]  # image, target, dataset


        if self.mosaic_prob > 0 and self.mosaic_epoch > self.cur_epoch:
            with_mosaic = random.random() <= self.mosaic_prob       
        else:
            with_mosaic = False
            
        for transform in self.transforms:
            # Removing strong augmentation after stop_epoch 
            if type(transform).__name__ in self.strong_augmentation and self.cur_epoch >= self.stop_epoch:
                pass
             # Using Mosaic for [policy_epoch[0], policy_epoch[1]] with probability
            elif (type(transform).__name__ == 'Mosaic' and not with_mosaic):      
                pass
            # Mosaic and Zoomout/IoUCrop can not be co-existed in the same sample
            elif (type(transform).__name__ == 'RandomZoomOut' or type(transform).__name__ == 'RandomIoUCrop') and with_mosaic:      
                pass
            else:
                sample = transform(sample)

        return sample


    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]

        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
