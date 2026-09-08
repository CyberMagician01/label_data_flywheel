"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from .bee_e_layers import (
    DomainSpecificBatchNorm2d,
    StabilizedTemporalResidual,
    replace_layer_norms,
    set_explicit_domain,
)

__all__ = ['ECDet', 'ECSeg']


class _DomainExpert(nn.Module):
    """Zero-initialized DWConv bottleneck residual expert."""

    def __init__(self, channels, bottleneck_ratio=0.25):
        super().__init__()
        hidden = max(16, int(channels * bottleneck_ratio))
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        nn.init.zeros_(self.block[-1].weight)

    def forward(self, feature):
        return self.block(feature)


class _ECBase(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward_features(self, x, **backbone_kwargs):
        x = self.backbone(x, **backbone_kwargs)
        x = self.encoder(x)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


@register()
class ECDet(_ECBase):

    def __init__(self, backbone, encoder, decoder, enable_density=False,
                 density_in_channels=256, density_hidden_channels=128,
                 enable_density_capacity=False, min_density_queries=128,
                 density_budget_ratio=1.25, density_budget_padding=32,
                 training_density_budget_ratio=2.0,
                 enable_temporal=False, temporal_frames=5,
                 enable_stabilized_temporal=False,
                 temporal_feature_channels=None,
                 temporal_bottleneck_channels=64,
                 temporal_gate_bias=-4.0,
                 temporal_stage_multiplier=0.0,
                 use_temporal_difference=False, use_frequency_gate=False,
                 temporal_feature_levels=2,
                 enable_explicit_domain=False,
                 enable_domain_layernorm=False,
                 enable_domain_adapter=False, domain_channels=256, domain_levels=3,
                 domain_expert_bottleneck_ratio=0.25,
                 domain_prototype_scale=4.0,
                 enable_shared_foreground_prototype=False,
                 use_route_density_head=False,
                 density_group_norm_groups=32):
        super().__init__(backbone, encoder, decoder)
        self.enable_density = enable_density
        self.enable_density_capacity = enable_density_capacity
        self.min_density_queries = min_density_queries
        self.density_budget_ratio = density_budget_ratio
        self.density_budget_padding = density_budget_padding
        # Kept only for checkpoint/config compatibility. Query validity must
        # follow the same prediction-only rule during training and inference.
        self.training_density_budget_ratio = training_density_budget_ratio
        self.enable_temporal = enable_temporal
        self.enable_stabilized_temporal = bool(enable_stabilized_temporal)
        self.use_temporal_difference = use_temporal_difference
        self.use_frequency_gate = use_frequency_gate
        self.temporal_feature_levels = temporal_feature_levels
        self.enable_explicit_domain = bool(enable_explicit_domain)
        self.enable_domain_layernorm = bool(enable_domain_layernorm)
        self.enable_domain_adapter = enable_domain_adapter
        self.enable_shared_foreground_prototype = bool(
            enable_shared_foreground_prototype
        )
        self.domain_prototype_scale = domain_prototype_scale
        if self.enable_temporal:
            if self.enable_stabilized_temporal and temporal_frames != 3:
                raise ValueError('The stabilized temporal route requires exactly three frames.')
            self.temporal_logits = nn.Parameter(torch.zeros(temporal_frames))
            # The first two high-resolution feature levels carry the temporal
            # signal. Zero initialization keeps the model identical to simple
            # temporal averaging at the start of E4 training.
            if self.use_temporal_difference:
                self.temporal_difference_scales = nn.Parameter(
                    torch.zeros(temporal_feature_levels, 3)
                )
            if self.use_frequency_gate:
                self.frequency_scales = nn.Parameter(
                    torch.zeros(temporal_feature_levels, 3)
                )
            if self.enable_stabilized_temporal:
                if temporal_feature_levels != 2:
                    raise ValueError('The stabilized temporal route is defined on P2 and P3 only.')
                temporal_feature_channels = temporal_feature_channels or [
                    density_in_channels, density_in_channels,
                ]
                if len(temporal_feature_channels) != 2:
                    raise ValueError('temporal_feature_channels must contain P2 and P3 channels.')
                self.stabilized_temporal_blocks = nn.ModuleList([
                    StabilizedTemporalResidual(
                        channels, temporal_bottleneck_channels, temporal_gate_bias,
                    )
                    for channels in temporal_feature_channels
                ])
                for block in self.stabilized_temporal_blocks:
                    block.set_stage_multiplier(temporal_stage_multiplier)

        if self.enable_domain_layernorm:
            replace_layer_norms(self.encoder)
            replace_layer_norms(self.decoder)
        if self.enable_domain_adapter:
            self.domain_classifier = nn.Linear(domain_channels, 2)
            self.domain_statistics_router = nn.Sequential(
                nn.Linear(4, max(16, domain_channels // 4)),
                nn.SiLU(),
                nn.Linear(max(16, domain_channels // 4), 2),
            )
            self.domain_prototypes = nn.Parameter(torch.empty(2, domain_channels))
            nn.init.normal_(self.domain_prototypes, std=0.02)
            self.shared_domain_adapters = nn.ModuleList([
                _DomainExpert(domain_channels, domain_expert_bottleneck_ratio)
                for _ in range(domain_levels)
            ])
        if self.enable_domain_adapter or self.enable_shared_foreground_prototype:
            self.shared_bee_prototype = nn.Parameter(torch.empty(domain_channels))
            nn.init.normal_(self.shared_bee_prototype, std=0.02)
            self.domain_adapters = nn.ModuleList([
                nn.ModuleList([
                    _DomainExpert(domain_channels, domain_expert_bottleneck_ratio)
                    for _ in range(2)
                ])
                for _ in range(domain_levels)
            ])
        if self.enable_density:
            if use_route_density_head:
                groups = min(int(density_group_norm_groups), density_in_channels)
                while density_in_channels % groups:
                    groups -= 1
                self.density_head = nn.Sequential(
                    nn.Conv2d(
                        density_in_channels, density_in_channels, 3, padding=1,
                        groups=density_in_channels, bias=False,
                    ),
                    nn.GroupNorm(groups, density_in_channels),
                    nn.GELU(),
                    nn.Conv2d(density_in_channels, 1, 1),
                    nn.Softplus(),
                )
                nn.init.normal_(self.density_head[0].weight, std=0.01)
                nn.init.ones_(self.density_head[1].weight)
                nn.init.zeros_(self.density_head[1].bias)
                nn.init.normal_(self.density_head[3].weight, std=0.01)
                nn.init.constant_(self.density_head[3].bias, -6.0)
            else:
                self.density_head = nn.Sequential(
                    nn.Conv2d(density_in_channels, density_hidden_channels, 3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(density_hidden_channels, 1, 1),
                )
                nn.init.normal_(self.density_head[0].weight, std=0.01)
                nn.init.zeros_(self.density_head[0].bias)
                nn.init.normal_(self.density_head[-1].weight, std=0.01)
                nn.init.constant_(self.density_head[-1].bias, -6.0)
        self.use_route_density_head = bool(use_route_density_head)

    @staticmethod
    def _resolve_domain_id(x, targets, domain_id):
        if domain_id is None and targets is not None:
            if not all('domain_id' in target for target in targets):
                raise RuntimeError('Explicit RGB/IR routing requires domain_id in every target.')
            domain_id = torch.stack([
                torch.as_tensor(target['domain_id']).reshape(-1)[0]
                for target in targets
            ])
        if domain_id is None:
            raise RuntimeError('Explicit RGB/IR routing requires domain_id for every sample.')
        domain_id = torch.as_tensor(domain_id, device=x.device, dtype=torch.long).reshape(-1)
        if domain_id.shape != (x.shape[0],):
            raise ValueError(f'domain_id must have shape [{x.shape[0]}].')
        if not bool(((domain_id == 0) | (domain_id == 1)).all()):
            raise ValueError('domain_id values must be 0 (RGB) or 1 (IR).')
        return domain_id

    @staticmethod
    def _resolve_scene_id(x, targets, scene_id):
        if scene_id is None and targets is not None:
            if not all('scene_id' in target for target in targets):
                raise RuntimeError('Scene-routed detection requires scene_id in every target.')
            scene_id = torch.stack([
                torch.as_tensor(target['scene_id']).reshape(-1)[0]
                for target in targets
            ])
        if scene_id is None:
            raise RuntimeError('Scene-routed detection requires scene_id for every sample.')
        scene_id = torch.as_tensor(scene_id, device=x.device, dtype=torch.long).reshape(-1)
        if scene_id.shape != (x.shape[0],):
            raise ValueError(f'scene_id must have shape [{x.shape[0]}].')
        if not bool(((scene_id == 0) | (scene_id == 1)).all()):
            raise ValueError('scene_id values must be 0 (indoor/B) or 1 (outdoor/A).')
        return scene_id

    @staticmethod
    def _resolve_stabilization_theta(x, targets, stabilization_theta, temporal_valid_mask):
        if stabilization_theta is None and targets is not None and all(
            'stabilization_theta' in target for target in targets
        ):
            stabilization_theta = torch.stack([
                torch.as_tensor(target['stabilization_theta']) for target in targets
            ])
        if stabilization_theta is None:
            if bool(temporal_valid_mask[:, :-1].any()):
                raise RuntimeError(
                    'Valid temporal support frames require explicit stabilization_theta.'
                )
            identity = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                device=x.device, dtype=x.dtype,
            )
            return identity[None, None].expand(x.shape[0], 2, -1, -1).clone()
        stabilization_theta = torch.as_tensor(
            stabilization_theta, device=x.device, dtype=x.dtype,
        )
        if stabilization_theta.shape != (x.shape[0], 2, 2, 3):
            raise ValueError('stabilization_theta must have shape [batch, 2, 2, 3].')
        return stabilization_theta

    def _predict_valid_query_mask(self, density_prediction):
        predicted_count = density_prediction.flatten(1).sum(dim=1)
        valid_count = torch.ceil(
            predicted_count * self.density_budget_ratio + self.density_budget_padding
        ).long().clamp(min=self.min_density_queries, max=self.decoder.num_queries)
        slot_index = torch.arange(
            self.decoder.num_queries, device=density_prediction.device
        )
        return slot_index.unsqueeze(0) < valid_count.unsqueeze(1)

    @staticmethod
    def _ir_hard_negative_prior(frames, targets, temporal_valid_mask=None):
        """Training-only causal-history fixed-texture anomaly map with GT protection."""
        if frames.ndim != 5 or frames.shape[1] < 2 or targets is None:
            return None
        gray = frames.mean(dim=2)
        history = gray[:, :-1]
        if temporal_valid_mask is None:
            history_valid = torch.ones(
                history.shape[:2], dtype=torch.bool, device=history.device
            )
        else:
            history_valid = temporal_valid_mask[:, :-1].to(
                device=history.device, dtype=torch.bool
            )
        weights = history_valid.to(history.dtype)[:, :, None, None]
        valid_count = weights.sum(dim=1).clamp_min(1.0)
        background = (history * weights).sum(dim=1) / valid_count
        variance = (
            (history - background[:, None]).square() * weights
        ).sum(dim=1) / valid_count
        temporal_stability = torch.exp(-variance.sqrt())
        local_background = F.avg_pool2d(background[:, None], 5, stride=1, padding=2)[:, 0]
        fixed_texture = (background - local_background).abs() * temporal_stability
        fixed_texture = fixed_texture * history_valid.any(dim=1)[:, None, None]
        prior = fixed_texture[:, None]
        prior = prior / prior.flatten(1).amax(dim=1).clamp_min(1e-6)[:, None, None, None]
        protected = torch.zeros_like(prior, dtype=torch.bool)
        for batch_index, target in enumerate(targets):
            is_ir = int(target.get('domain_id', torch.tensor([0], device=prior.device)).item()) == 1
            explicit_negative = bool(target.get(
                'hard_negative', torch.tensor([False], device=prior.device)
            ).item())
            if not is_ir or not explicit_negative:
                prior[batch_index] = 0
                continue
            for box in target.get('boxes', []):
                center_x, center_y, width, height = box
                left = int(((center_x - 0.6 * width).clamp(0, 1) * prior.shape[-1]).item())
                right = int(((center_x + 0.6 * width).clamp(0, 1) * prior.shape[-1]).item()) + 1
                top = int(((center_y - 0.6 * height).clamp(0, 1) * prior.shape[-2]).item())
                bottom = int(((center_y + 0.6 * height).clamp(0, 1) * prior.shape[-2]).item()) + 1
                protected[batch_index, :, top:bottom, left:right] = True
        return prior.masked_fill(protected, 0.0)

    def forward(self, x, targets=None, temporal_valid_mask=None, domain_id=None,
                stabilization_theta=None, scene_id=None):
        scene_routing_enabled = bool(getattr(
            self.decoder, 'enable_scene_detection_heads', False
        ))
        if scene_routing_enabled:
            scene_id = self._resolve_scene_id(x, targets, scene_id)
        if self.enable_explicit_domain:
            domain_id = self._resolve_domain_id(x, targets, domain_id)
            set_explicit_domain(self.encoder, domain_id)
            set_explicit_domain(self.decoder, domain_id)
        if x.ndim == 5:
            batch_size, frame_count, channels, height, width = x.shape
            if temporal_valid_mask is None and targets is not None and all(
                'temporal_valid_mask' in target for target in targets
            ):
                temporal_valid_mask = torch.stack([
                    target['temporal_valid_mask'] for target in targets
                ], dim=0)
            if temporal_valid_mask is None:
                temporal_valid_mask = torch.ones(
                    batch_size, frame_count, dtype=torch.bool, device=x.device
                )
            else:
                temporal_valid_mask = temporal_valid_mask.to(
                    device=x.device, dtype=torch.bool
                ).clone()
            if temporal_valid_mask.shape != (batch_size, frame_count):
                raise ValueError('temporal_valid_mask must have shape [batch, frames]')
            temporal_valid_mask[:, -1] = True
            if self.enable_stabilized_temporal:
                if frame_count != 3:
                    raise ValueError('The stabilized temporal route requires [long, short, current].')
                stabilization_theta = self._resolve_stabilization_theta(
                    x, targets, stabilization_theta, temporal_valid_mask,
                )
        hard_negative_prior = (
            self._ir_hard_negative_prior(x, targets, temporal_valid_mask)
            if self.training else None
        )

        current_image = x[:, -1] if x.ndim == 5 else x
        channel_means = current_image.mean(dim=(-2, -1))
        color_spread = channel_means.std(dim=-1)
        contrast = current_image.std(dim=(-2, -1)).mean(dim=-1)
        smooth = F.avg_pool2d(current_image, 3, stride=1, padding=1)
        noise_energy = (current_image - smooth).abs().mean(dim=(1, 2, 3))
        motion_energy = torch.zeros_like(noise_energy)
        if x.ndim == 5 and frame_count > 1:
            history_mask = temporal_valid_mask[:, :-1]
            history_indices = torch.arange(frame_count - 1, device=x.device)[None]
            latest_history = history_indices.masked_fill(~history_mask, -1).amax(dim=1)
            has_history = latest_history >= 0
            reference = x[
                torch.arange(batch_size, device=x.device),
                latest_history.clamp_min(0),
            ]
            motion_energy = (
                (current_image - reference).abs().mean(dim=(1, 2, 3))
                * has_history.to(current_image.dtype)
            )
        domain_router_features = torch.stack(
            [color_spread, contrast, noise_energy, motion_energy], dim=-1
        )
        if x.ndim == 5:
            try:
                backbone_kwargs = {'return_shallow': True}
                if self.enable_explicit_domain:
                    backbone_kwargs['domain_id'] = domain_id
                current_features, current_shallow_features = self.backbone(
                    x[:, -1], **backbone_kwargs
                )
            except TypeError as error:
                raise RuntimeError(
                    'Temporal E path requires backbone.forward(..., return_shallow=True).'
                ) from error
            if frame_count > 1:
                if not hasattr(self.backbone, 'forward_shallow_history'):
                    raise RuntimeError(
                        'Temporal E path requires backbone.forward_shallow_history().'
                    )
                history_kwargs = {}
                if self.enable_explicit_domain:
                    history_kwargs['domain_id'] = domain_id[:, None].expand(
                        -1, frame_count - 1
                    ).reshape(-1)
                history_features = self.backbone.forward_shallow_history(
                    x[:, :-1].reshape(batch_size * (frame_count - 1), channels, height, width),
                    **history_kwargs,
                )
            else:
                history_features = []
            temporal_logits = self.temporal_logits[:frame_count][None].expand(batch_size, -1)
            temporal_weights = temporal_logits.masked_fill(
                ~temporal_valid_mask, -torch.finfo(temporal_logits.dtype).max
            ).softmax(dim=1)
            fused_features = []
            for level_idx, current in enumerate(current_features):
                if (
                    self.enable_stabilized_temporal
                    and level_idx < self.temporal_feature_levels
                    and frame_count == 3
                ):
                    history = history_features[level_idx].reshape(
                        batch_size, 2, *history_features[level_idx].shape[1:]
                    )
                    fused = self.stabilized_temporal_blocks[level_idx](
                        current,
                        history[:, 1],
                        history[:, 0],
                        stabilization_theta[:, 1],
                        stabilization_theta[:, 0],
                        temporal_valid_mask[:, 1],
                        temporal_valid_mask[:, 0],
                    )
                    sequence = torch.cat(
                        [history, current_shallow_features[level_idx][:, None]], dim=1,
                    )
                    current_shallow = current_shallow_features[level_idx]
                elif level_idx < self.temporal_feature_levels and frame_count > 1:
                    history = history_features[level_idx].reshape(
                        batch_size, frame_count - 1, *history_features[level_idx].shape[1:]
                    )
                    current_shallow = current_shallow_features[level_idx]
                    sequence = torch.cat([history, current_shallow[:, None]], dim=1)
                    shallow_fused = (
                        sequence * temporal_weights.view(batch_size, frame_count, 1, 1, 1)
                    ).sum(dim=1)
                    fused = current + (shallow_fused - current_shallow)
                else:
                    sequence = current[:, None]
                    current_shallow = current
                    fused = current

                if self.use_temporal_difference and level_idx < self.temporal_feature_levels:
                    # Causal temporal-difference convolution at 1/2/4-frame
                    # spans. Repeating the earliest available frame handles
                    # clips shorter than five frames without a separate path.
                    difference = 0.0
                    for scale_idx, gap in enumerate((1, 2, 4)):
                        history_idx = max(sequence.shape[1] - 1 - gap, 0)
                        gap_valid = temporal_valid_mask[:, history_idx, None, None, None]
                        difference = difference + (
                            self.temporal_difference_scales[level_idx, scale_idx]
                            * (current_shallow - sequence[:, history_idx]) * gap_valid
                        )
                    fused = fused + difference

                if self.use_frequency_gate and level_idx < self.temporal_feature_levels:
                    # Fixed low/mid/high spatial bands with learnable per-level
                    # gates. All operators are standard ONNX convolution/pool
                    # primitives and the initial residual is exactly zero.
                    low = F.avg_pool2d(current, kernel_size=5, stride=1, padding=2)
                    smooth3 = F.avg_pool2d(current, kernel_size=3, stride=1, padding=1)
                    mid = smooth3 - low
                    high = current - smooth3
                    bands = (low, mid, high)
                    for band_idx, band in enumerate(bands):
                        fused = fused + self.frequency_scales[level_idx, band_idx] * band

                fused_features.append(fused)
            features = self.encoder(fused_features)
        else:
            backbone_kwargs = {'domain_id': domain_id} if self.enable_explicit_domain else {}
            features = self.forward_features(x, **backbone_kwargs)

        domain_logits = None
        domain_features = None
        domain_adapter_responses = []
        if self.enable_domain_adapter:
            domain_features = features[-1].mean(dim=(-2, -1))
            normalized_features = F.normalize(domain_features, dim=-1)
            normalized_prototypes = F.normalize(self.domain_prototypes, dim=-1)
            prototype_logits = normalized_features @ normalized_prototypes.t()
            domain_logits = (
                self.domain_classifier(domain_features)
                + self.domain_statistics_router(domain_router_features)
                + self.domain_prototype_scale * prototype_logits
            )
            domain_weights = domain_logits.softmax(dim=-1)
            adapted_features = []
            for level_idx, feature in enumerate(features):
                if level_idx >= len(self.domain_adapters):
                    adapted_features.append(feature)
                    continue
                residual = self.shared_domain_adapters[level_idx](feature) + sum(
                    domain_weights[:, domain_idx, None, None, None]
                    * self.domain_adapters[level_idx][domain_idx](feature)
                    for domain_idx in range(2)
                )
                domain_adapter_responses.append(residual.abs().flatten(1).mean(dim=1))
                adapted_features.append(feature + residual)
            features = adapted_features

        density_prediction = None
        if self.enable_density:
            density_prediction = self.density_head(features[0])
            if not self.use_route_density_head:
                density_prediction = F.softplus(density_prediction)
        valid_query_mask = None
        if (
            self.enable_density_capacity
            and density_prediction is not None
            and not getattr(self.decoder, 'enable_monotonic_query_capacity', False)
        ):
            valid_query_mask = self._predict_valid_query_mask(density_prediction)
        explicit_domain_context = None
        if self.enable_explicit_domain:
            explicit_domain_context = F.one_hot(domain_id, num_classes=2).to(features[0].dtype)
        outputs = self.decoder(
            features,
            targets,
            density_prior=density_prediction,
            valid_query_mask=valid_query_mask,
            domain_context=(
                domain_logits.softmax(dim=-1)
                if domain_logits is not None else explicit_domain_context
            ),
            scene_id=scene_id,
        )
        if domain_logits is not None:
            outputs['pred_domain_logits'] = domain_logits
            outputs['pred_domain_features'] = domain_features
            # Return graph-connected tensors rather than raw leaf parameters.
            # DDP's unused-parameter traversal starts from forward outputs; a
            # raw leaf consumed later by the criterion can otherwise be marked
            # unused before backward and then marked ready a second time.
            outputs['domain_prototypes'] = self.domain_prototypes * 1.0
            outputs['shared_bee_prototype'] = self.shared_bee_prototype * 1.0
            outputs['domain_expert_weights'] = domain_weights
            outputs['domain_rgb_weight'] = domain_weights[:, 0]
            outputs['domain_ir_weight'] = domain_weights[:, 1]
            outputs['domain_router_features'] = domain_router_features
            outputs['domain_router_entropy'] = -(
                domain_weights.clamp_min(1e-8) * domain_weights.clamp_min(1e-8).log()
            ).sum(-1)
            outputs['domain_prototype_distance'] = 1.0 - F.cosine_similarity(
                self.domain_prototypes[0:1], self.domain_prototypes[1:2]
            )
            outputs['shared_bee_prototype_distance'] = (
                1.0 - F.cosine_similarity(
                    self.domain_prototypes,
                    self.shared_bee_prototype[None].expand_as(self.domain_prototypes),
                )
            ).mean()
            if domain_adapter_responses:
                outputs['domain_adapter_response'] = torch.stack(
                    domain_adapter_responses, dim=1
                ).mean(dim=1)
            bn_mean_distances = []
            bn_logvar_distances = []
            for module in self.modules():
                if not isinstance(module, DomainSpecificBatchNorm2d):
                    continue
                rgb, ir = module.norms
                if rgb.running_mean is None or ir.running_mean is None:
                    continue
                bn_mean_distances.append(
                    (rgb.running_mean - ir.running_mean).square().mean().sqrt()
                )
                bn_logvar_distances.append((
                    rgb.running_var.clamp_min(1e-8).log()
                    - ir.running_var.clamp_min(1e-8).log()
                ).square().mean().sqrt())
            if bn_mean_distances:
                outputs['domain_bn_mean_distance'] = torch.stack(bn_mean_distances).mean()
                outputs['domain_bn_logvar_distance'] = torch.stack(bn_logvar_distances).mean()
        if self.enable_shared_foreground_prototype:
            if len(features) < 2:
                raise RuntimeError('Shared foreground prototype requires Encoder P3.')
            outputs['pred_domain_p3_features'] = features[1]
            outputs['shared_bee_prototype'] = self.shared_bee_prototype * 1.0
        if self.enable_density:
            outputs['pred_density'] = density_prediction
        if self.enable_explicit_domain:
            outputs['route_domain_id'] = domain_id
        if scene_routing_enabled:
            outputs['route_scene_id'] = scene_id
        if hard_negative_prior is not None:
            outputs['pred_ir_hard_negative_prior'] = hard_negative_prior
            outputs['ir_hard_negative_pixels'] = (
                hard_negative_prior > 0.5
            ).float().flatten(1).sum(1)
        if x.ndim == 5:
            outputs['temporal_valid_ratio'] = temporal_valid_mask.float().mean(dim=1)
        return outputs


@register()
class ECSeg(_ECBase):

    def forward(self, x, targets=None):
        x = self.forward_features(x)
        spatial_feat = x[0]
        return self.decoder(x, targets, spatial_feat)
