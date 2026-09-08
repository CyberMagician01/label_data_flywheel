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
import os
import math
import random
from collections import defaultdict, deque
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as VT
from PIL import Image, ImageDraw
from torch.utils.data import default_collate
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as VF

from ..core import register

torchvision.disable_beta_transforms_warning()


__all__ = [
    'DataLoader',
    'DomainBalancedSampler',
    'BaseCollateFunction',
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


class DomainBalancedSampler(data.Sampler):
    """Base coverage followed by debt-weighted, globally paired correction."""

    is_domain_balanced = True

    def __init__(self, dataset, hard_negative_ratio=0.25,
                 unlabelled_ratio=0.25, correction_ratio=0.25, seed=2026,
                 rank_interleave_domains=False,
                 synchronize_domains_across_ranks=False,
                 local_batch_size=None):
        self.dataset = dataset
        self.hard_negative_ratio = hard_negative_ratio
        self.unlabelled_ratio = unlabelled_ratio
        self.correction_ratio = float(correction_ratio)
        if self.correction_ratio < 0:
            raise ValueError('correction_ratio must be non-negative')
        self.seed = seed
        self.rank_interleave_domains = bool(rank_interleave_domains)
        self.synchronize_domains_across_ranks = bool(
            synchronize_domains_across_ranks
        )
        if self.rank_interleave_domains and self.synchronize_domains_across_ranks:
            raise ValueError(
                'Rank-domain interleaving and synchronized domains are mutually exclusive.'
            )
        self.local_batch_size = (
            int(local_batch_size) if local_batch_size is not None else None
        )
        self.epoch = 0
        self.coverage_debt = defaultdict(float)
        self.recall_debt = defaultdict(float)
        self.generalization_gap = defaultdict(float)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_feedback(self, coverage_debt=None, recall_debt=None,
                        generalization_gap=None):
        """Update per-video debts measured on the previous coverage cycle."""
        for destination, source in (
            (self.coverage_debt, coverage_debt),
            (self.recall_debt, recall_debt),
            (self.generalization_gap, generalization_gap),
        ):
            if source is not None:
                destination.update({str(key): float(value) for key, value in source.items()})

    def _weighted_pick(self, indices, generator):
        base_weights = getattr(self.dataset, 'sampling_weights', {})
        index_video = getattr(self.dataset, 'index_video', {})
        weights = []
        for index in indices:
            video = str(index_video.get(index, 'unknown'))
            debt = (
                max(self.coverage_debt[video], 0.0)
                + max(self.recall_debt[video], 0.0)
                + max(self.generalization_gap[video], 0.0)
            )
            weights.append(max(float(base_weights.get(index, 1.0)), 1e-6) * (1.0 + debt))
        weights = torch.tensor(weights, dtype=torch.float64)
        selected = torch.multinomial(weights, 1, generator=generator).item()
        return indices[selected]

    @staticmethod
    def _shuffle(values, generator):
        if not values:
            return []
        order = torch.randperm(len(values), generator=generator).tolist()
        return [values[index] for index in order]

    def _base_coverage_order(self, domain, generator):
        domain_indices = list(getattr(self.dataset, 'domain_indices', {}).get(domain, []))
        if not domain_indices:
            raise ValueError(f'Domain {domain} has no samples; equal RGB/IR updates are impossible.')
        video_indices = getattr(self.dataset, 'video_indices', {}).get(domain, {})
        if not video_indices:
            return self._shuffle(domain_indices, generator)
        queues = {
            str(video): self._shuffle(
                [index for index in indices if index in set(domain_indices)], generator,
            )
            for video, indices in video_indices.items()
        }
        queues = {video: indices for video, indices in queues.items() if indices}
        videos = self._shuffle(sorted(queues), generator)
        order = []
        while any(queues.values()):
            for video in videos:
                if queues[video]:
                    order.append(queues[video].pop())
        if set(order) != set(domain_indices):
            missing = sorted(set(domain_indices) - set(order))
            order.extend(self._shuffle(missing, generator))
        return order

    def _correction_pick(self, domain, step, generator):
        domain_indices = getattr(self.dataset, 'domain_indices', {})
        hard_indices = getattr(self.dataset, 'hard_negative_indices', {})
        labelled_indices = getattr(self.dataset, 'labelled_indices', {})
        unlabelled_indices = getattr(self.dataset, 'unlabelled_indices', {})
        video_indices = getattr(self.dataset, 'video_indices', {})
        use_unlabelled = (
            bool(unlabelled_indices.get(domain))
            and float(torch.rand((), generator=generator)) < self.unlabelled_ratio
        )
        pool = (
            unlabelled_indices.get(domain) if use_unlabelled
            else labelled_indices.get(domain)
        ) or domain_indices.get(domain)
        videos = sorted(video_indices.get(domain, {}))
        if videos:
            video = videos[(step + self.epoch) % len(videos)]
            video_pool = set(video_indices[domain][video])
            constrained = [index for index in pool if index in video_pool]
            if constrained:
                pool = constrained
        hard_pool = [index for index in hard_indices.get(domain, []) if index in pool]
        use_hard = (
            bool(hard_pool)
            and float(torch.rand((), generator=generator)) < self.hard_negative_ratio
        )
        return self._weighted_pick(hard_pool if use_hard else pool, generator)

    def _global_sequence(self, world_size):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1009)
        domain_orders = {
            domain: self._base_coverage_order(domain, generator) for domain in (0, 1)
        }
        base_pairs = max(len(domain_orders[0]), len(domain_orders[1]))
        pairs = []
        for step in range(base_pairs):
            pairs.append((
                domain_orders[0][step % len(domain_orders[0])],
                domain_orders[1][step % len(domain_orders[1])],
            ))
        correction_pairs = int(math.ceil(base_pairs * self.correction_ratio))
        for step in range(correction_pairs):
            pairs.append((
                self._correction_pick(0, step, generator),
                self._correction_pick(1, step, generator),
            ))
        while (2 * len(pairs)) % world_size:
            step = len(pairs)
            pairs.append((
                self._correction_pick(0, step, generator),
                self._correction_pick(1, step, generator),
            ))
        return [index for pair in pairs for index in pair]

    def __iter__(self):
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else 1
        sequence = self._global_sequence(world_size)
        if self.synchronize_domains_across_ranks:
            if world_size != 2:
                raise ValueError('Synchronized rank domains require world_size=2.')
            if not self.local_batch_size or self.local_batch_size < 1:
                raise ValueError('Synchronized rank domains require a local batch size.')
            pairs = list(zip(sequence[0::2], sequence[1::2]))
            usable = (len(pairs) // self.local_batch_size) * self.local_batch_size
            micro_batches = usable // self.local_batch_size
            domains = {
                0: [pair[0] for pair in pairs[:usable]],
                1: [pair[1] for pair in pairs[:usable]],
            }
            local_sequence = []
            synchronized_micro_batches = micro_batches - (micro_batches % 2)
            global_batch_size = world_size * self.local_batch_size
            for micro_batch in range(synchronized_micro_batches):
                domain = micro_batch % 2
                domain_batch = micro_batch // 2
                start = domain_batch * global_batch_size
                rank_start = start + rank * self.local_batch_size
                local_sequence.extend(
                    domains[domain][rank_start:rank_start + self.local_batch_size]
                )
            if micro_batches % 2:
                start = (micro_batches // 2) * global_batch_size
                local_sequence.extend(
                    domains[rank][start:start + self.local_batch_size]
                )
            if len(local_sequence) != usable:
                raise RuntimeError('Synchronized rank-domain sampler length mismatch.')
            return iter(local_sequence)
        if self.rank_interleave_domains:
            if world_size != 2:
                raise ValueError('Rank-domain interleaving requires world_size=2.')
            if not self.local_batch_size or self.local_batch_size < 1:
                raise ValueError('Rank-domain interleaving requires a local batch size.')
            pairs = list(zip(sequence[0::2], sequence[1::2]))
            local_sequence = []
            for start in range(0, len(pairs), self.local_batch_size):
                micro_batch = start // self.local_batch_size
                domain = (micro_batch + rank) % 2
                local_sequence.extend(
                    pair[domain] for pair in pairs[start:start + self.local_batch_size]
                )
            return iter(local_sequence)
        return iter(sequence[rank::world_size])

    def __len__(self):
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        return len(self._global_sequence(world_size)) // world_size


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __init__(self, *args, domain_balanced=False, hard_negative_ratio=0.25,
                 unlabelled_ratio=0.25, correction_ratio=0.25,
                 sampler_seed=2026, rank_interleave_domains=False,
                 synchronize_domains_across_ranks=False, **kwargs):
        dataset = kwargs.get('dataset', args[0] if args else None)
        if domain_balanced:
            batch_size = kwargs.get('batch_size')
            world_size = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 1
            )
            if batch_size is not None and (int(batch_size) * world_size) % 2:
                raise ValueError(
                    'Domain-balanced training requires an even global micro-batch.'
                )
            kwargs['shuffle'] = False
            kwargs['sampler'] = DomainBalancedSampler(
                dataset,
                hard_negative_ratio=hard_negative_ratio,
                unlabelled_ratio=unlabelled_ratio,
                correction_ratio=correction_ratio,
                seed=sampler_seed,
                rank_interleave_domains=rank_interleave_domains,
                synchronize_domains_across_ranks=synchronize_domains_across_ranks,
                local_batch_size=batch_size,
            )
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


@register() 
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        mixup_prob=0.0,
        mixup_epoch=0,
    ) -> None:
        super().__init__()
        self.mixup_prob, self.mixup_epoch = mixup_prob, mixup_epoch

    def apply_mixup(self, images, targets):
        """
        Applies Mixup augmentation to the batch if conditions are met.

        Args:
            images (torch.Tensor): Batch of images.
            targets (list[dict]): List of target dictionaries corresponding to images.

        Returns:
            tuple: Updated images and targets
        """
        beta = round(random.uniform(0.45, 0.55), 6)
        # Apply Mixup if within specified epoch range and probability threshold
        if random.random() < self.mixup_prob and self.epoch < self.mixup_epoch:
            # Generate mixup ratio
            beta = round(random.uniform(0.45, 0.55), 6)

            # Mix images
            images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))

            # Prepare targets for Mixup
            shifted_targets = targets[-1:] + targets[:-1]
            updated_targets = deepcopy(targets)

            for i in range(len(targets)):
                # Combine boxes, labels, and areas from original and shifted targets
                updated_targets[i]['boxes'] = torch.cat([targets[i]['boxes'], shifted_targets[i]['boxes']], dim=0)
                updated_targets[i]['labels'] = torch.cat([targets[i]['labels'], shifted_targets[i]['labels']], dim=0)
                updated_targets[i]['area'] = torch.cat([targets[i]['area'], shifted_targets[i]['area']], dim=0)
                if 'masks' in targets[i]:
                    updated_targets[i]['masks'] = torch.cat([targets[i]['masks'], shifted_targets[i]['masks']], dim=0)

                # Add mixup ratio to targets
                updated_targets[i]['mixup'] = torch.tensor(
                    [beta] * len(targets[i]['labels']) + [1.0 - beta] * len(shifted_targets[i]['labels']), 
                    dtype=torch.float32
                    )
            targets = updated_targets
            
        return images, targets

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]
        images, targets = self.apply_mixup(images, targets)

        return images, targets
