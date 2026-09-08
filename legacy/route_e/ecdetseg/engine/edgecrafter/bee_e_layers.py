"""BeePoseTrack-E layers shared by the full ECDet route.

The modules in this file keep the RGB/IR and temporal contracts explicit.  In
particular, domain selection is never inferred from image appearance: callers
must provide ``domain_id`` (0=RGB, 1=IR).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW features."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, value):
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        value = (value - mean) * torch.rsqrt(variance + self.eps)
        return value * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class _ExplicitDomainMixin:
    def set_domain_id(self, domain_id):
        if domain_id is None:
            self._domain_id = None
            return
        domain_id = torch.as_tensor(domain_id, dtype=torch.long)
        if domain_id.ndim == 0:
            domain_id = domain_id[None]
        self._domain_id = domain_id.reshape(-1)

    def _checked_domain_id(self, batch_size, device):
        domain_id = getattr(self, '_domain_id', None)
        if domain_id is None:
            raise RuntimeError('Explicit RGB/IR routing requires domain_id for every sample.')
        domain_id = domain_id.to(device=device, dtype=torch.long)
        if domain_id.numel() == 1 and batch_size != 1:
            domain_id = domain_id.expand(batch_size)
        if domain_id.shape != (batch_size,):
            raise ValueError(
                f'domain_id must have shape [{batch_size}], got {tuple(domain_id.shape)}.'
            )
        if not bool(((domain_id == 0) | (domain_id == 1)).all()):
            raise ValueError('domain_id values must be 0 (RGB) or 1 (IR).')
        return domain_id


class DomainSpecificBatchNorm2d(_ExplicitDomainMixin, nn.Module):
    """Two independent BN statistics/affines selected by explicit domain_id."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, freeze_running_stats=False):
        super().__init__()
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(
                num_features, eps=eps, momentum=momentum, affine=affine,
                track_running_stats=track_running_stats,
            )
            for _ in range(2)
        ])
        self.freeze_running_stats = bool(freeze_running_stats)
        self._domain_id = None

    @classmethod
    def from_batch_norm(cls, source, freeze_running_stats=False):
        module = cls(
            source.num_features,
            eps=source.eps,
            momentum=source.momentum,
            affine=source.affine,
            track_running_stats=source.track_running_stats,
            freeze_running_stats=freeze_running_stats,
        )
        for norm in module.norms:
            norm.load_state_dict(copy.deepcopy(source.state_dict()))
        return module

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_running_stats:
            for norm in self.norms:
                norm.eval()
        return self

    def forward(self, value):
        domain_id = self._checked_domain_id(value.shape[0], value.device)
        if not self.training or self.freeze_running_stats:
            rgb = self.norms[0](value)
            ir = self.norms[1](value)
            mask = domain_id[:, None, None, None].bool()
            return torch.where(mask, ir, rgb)

        output = torch.empty_like(value)
        for current_domain, norm in enumerate(self.norms):
            mask = domain_id == current_domain
            if bool(mask.any()):
                output[mask] = norm(value[mask])
        return output


class DomainConditionedLayerNorm(_ExplicitDomainMixin, nn.Module):
    """Shared LayerNorm plus zero-initialized RGB/IR affine increments."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.domain_weight_delta = nn.Parameter(torch.zeros(2, *shape))
        self.domain_bias_delta = nn.Parameter(torch.zeros(2, *shape))
        self._domain_id = None

    @classmethod
    def from_layer_norm(cls, source):
        module = cls(source.normalized_shape, source.eps, source.elementwise_affine)
        module.norm.load_state_dict(copy.deepcopy(source.state_dict()))
        return module

    def forward(self, value, domain_id=None):
        normalized = self.norm(value)
        if domain_id is None:
            domain_id = self._checked_domain_id(value.shape[0], value.device)
        else:
            domain_id = torch.as_tensor(
                domain_id, device=value.device, dtype=torch.long,
            ).reshape(-1)
            if domain_id.shape != (value.shape[0],):
                raise ValueError(
                    f'domain_id must have shape [{value.shape[0]}], '
                    f'got {tuple(domain_id.shape)}.'
                )
            if not bool(((domain_id == 0) | (domain_id == 1)).all()):
                raise ValueError('domain_id values must be 0 (RGB) or 1 (IR).')
        delta_weight = self.domain_weight_delta[domain_id]
        delta_bias = self.domain_bias_delta[domain_id]
        while delta_weight.ndim < value.ndim:
            delta_weight = delta_weight.unsqueeze(1)
            delta_bias = delta_bias.unsqueeze(1)
        return normalized * (1.0 + delta_weight) + delta_bias


class DomainLowRankAdapter2d(_ExplicitDomainMixin, nn.Module):
    """Per-domain low-rank residual adapter with an exactly-zero initial path."""

    def __init__(self, channels, rank):
        super().__init__()
        if rank <= 0 or rank > channels:
            raise ValueError('Domain adapter rank must be in [1, channels].')
        self.down = nn.ModuleList([nn.Conv2d(channels, rank, 1, bias=False) for _ in range(2)])
        self.up = nn.ModuleList([nn.Conv2d(rank, channels, 1, bias=False) for _ in range(2)])
        self.activation = nn.GELU()
        self._domain_id = None
        for layer in self.up:
            nn.init.zeros_(layer.weight)

    def forward(self, value):
        domain_id = self._checked_domain_id(value.shape[0], value.device)
        branches = [self.up[index](self.activation(self.down[index](value))) for index in range(2)]
        mask = domain_id[:, None, None, None].bool()
        return value + torch.where(mask, branches[1], branches[0])


class StabilizedTemporalResidual(nn.Module):
    """P2/P3 three-frame residual block from the route definition."""

    def __init__(self, channels, bottleneck_channels, gate_bias=-4.0):
        super().__init__()
        expanded = channels * 4
        self.compress = nn.Sequential(
            nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False),
            LayerNorm2d(expanded),
            nn.GELU(),
            nn.Conv2d(expanded, bottleneck_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels, channels, 1, bias=False),
        )
        self.gate = nn.Conv2d(expanded, channels, 1)
        self.register_buffer('stage_multiplier', torch.tensor(0.0))
        nn.init.zeros_(self.compress[-1].weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def set_stage_multiplier(self, value):
        self.stage_multiplier.fill_(float(value))

    @staticmethod
    def _warp(value, theta):
        grid = F.affine_grid(theta.to(dtype=value.dtype), value.shape, align_corners=False)
        return F.grid_sample(
            value, grid, mode='bilinear', padding_mode='border', align_corners=False,
        )

    def forward(self, current, short_support, long_support, short_theta, long_theta,
                short_valid, long_valid):
        short_aligned = self._warp(short_support, short_theta)
        long_aligned = self._warp(long_support, long_theta)
        short_valid = short_valid[:, None, None, None].to(current.dtype)
        long_valid = long_valid[:, None, None, None].to(current.dtype)
        short_delta = (current - short_aligned) * short_valid
        long_delta = (current - long_aligned) * long_valid
        residual_input = torch.cat(
            [short_delta, long_delta, short_delta.abs(), long_delta.abs()], dim=1,
        )
        residual = self.compress(residual_input)
        gate = torch.sigmoid(self.gate(residual_input))
        return current + self.stage_multiplier.to(current.dtype) * gate * residual


def replace_layer_norms(module):
    """Recursively replace LayerNorm while preserving pretrained parameters."""
    for name, child in list(module.named_children()):
        if isinstance(child, DomainConditionedLayerNorm):
            # Model components can request domain conversion independently and
            # again during top-level assembly. Keep the transformation
            # idempotent so the wrapped vanilla LayerNorm is never wrapped a
            # second time and domain context has exactly one owner.
            continue
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, DomainConditionedLayerNorm.from_layer_norm(child))
        else:
            replace_layer_norms(child)


def set_explicit_domain(module, domain_id):
    """Set domain context on all explicit-domain layers below ``module``."""
    for child in module.modules():
        if isinstance(child, _ExplicitDomainMixin):
            child.set_domain_id(domain_id)
