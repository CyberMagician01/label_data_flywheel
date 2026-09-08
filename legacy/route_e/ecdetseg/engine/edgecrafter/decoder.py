"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import functools
import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ..core import register
from .bee_e_layers import replace_layer_norms
from .denoising import get_contrastive_denoising_training_group
from .segmentation_head import SegmentationHead
from .utils import (bias_init_with_prob, deformable_attention_core_func_v2,
                    distance2bbox, get_activation, inverse_sigmoid,
                    weighting_function)

__all__ = ['ECTransformer']
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DecomposedEndpointDecoder(nn.Module):
    """Decode head/tail queries around each fixed instance query.

    Instance queries remain global. Endpoint queries are derived per decoder
    layer, sample only a fixed local neighbourhood on P2-P5, and carry their
    refined point references into the next layer.
    """

    def __init__(self, hidden_dim, num_keypoints, num_layers, num_levels,
                 num_sampling_points=4, act='silu', use_full_route=False,
                 body_axis_radius=0.25, endpoint_delta_limit=2.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_keypoints = num_keypoints
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.num_sampling_points = num_sampling_points
        self.use_full_route = bool(use_full_route)
        self.body_axis_radius = float(body_axis_radius)
        self.endpoint_delta_limit = float(endpoint_delta_limit)

        self.endpoint_semantic_embed = nn.Parameter(
            torch.empty(num_keypoints, hidden_dim)
        )
        self.endpoint_query_layers = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, hidden_dim, 2, act=act)
            for _ in range(num_layers)
        ])
        self.initial_reference_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(num_layers)
        ])
        self.sampling_offset_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_levels * num_sampling_points * 2)
            for _ in range(num_layers)
        ])
        # Sampling weights are instance-level and shared by head/tail.  Only
        # offsets and endpoint refinement are endpoint-specific.
        self.instance_sampling_attention_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_levels * num_sampling_points)
            for _ in range(num_layers)
        ])
        self.sample_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.endpoint_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.reference_refine_heads = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 2, 2, act=act)
            for _ in range(num_layers)
        ])
        self.visibility_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_layers)
        ])
        if self.use_full_route:
            if num_keypoints != 2:
                raise ValueError('The full BeePoseTrack-E endpoint route requires head and tail.')
            if hidden_dim % 8:
                raise ValueError('Box sine embedding requires hidden_dim divisible by 8.')
            self.full_query_generators = nn.ModuleList([
                MLP(hidden_dim * 3, hidden_dim, hidden_dim, 2, act=act)
                for _ in range(num_layers)
            ])
            self.body_axis_head = nn.Linear(hidden_dim, 2)
            self.shared_sampling_offset_head = nn.Linear(
                hidden_dim, num_levels * num_sampling_points * 2,
            )
            self.shared_instance_sampling_attention_head = nn.Linear(
                hidden_dim, num_levels * num_sampling_points,
            )
            self.shared_sample_projection = nn.Linear(hidden_dim, hidden_dim)
            self.endpoint_output_mlps = nn.ModuleList([
                nn.ModuleList([
                    MLP(hidden_dim, hidden_dim, hidden_dim, 2, act=act)
                    for _ in range(num_keypoints)
                ])
                for _ in range(num_layers)
            ])
            self.full_reference_refine_heads = nn.ModuleList([
                nn.ModuleList([
                    MLP(hidden_dim, hidden_dim, 2, 2, act=act)
                    for _ in range(num_keypoints)
                ])
                for _ in range(num_layers)
            ])
            self.full_visibility_heads = nn.ModuleList([
                nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_keypoints)])
                for _ in range(num_layers)
            ])
            self.full_uncertainty_heads = nn.ModuleList([
                nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_keypoints)])
                for _ in range(num_layers)
            ])
        canonical = torch.tensor([
            [0.0, 0.0], [-0.5, 0.0], [0.5, 0.0],
            [0.0, -0.5], [0.0, 0.5],
            [-0.5, -0.5], [0.5, -0.5],
            [-0.5, 0.5], [0.5, 0.5],
        ])
        if num_sampling_points > len(canonical):
            angles = torch.arange(
                num_sampling_points - len(canonical), dtype=torch.float32
            ) * (2.0 * math.pi / max(num_sampling_points - len(canonical), 1))
            extra = 0.35 * torch.stack((angles.cos(), angles.sin()), dim=-1)
            canonical = torch.cat((canonical, extra), dim=0)
        self.register_buffer(
            'base_sampling_offsets', canonical[:num_sampling_points], persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.endpoint_semantic_embed, std=0.02)
        for offset_head, attention_head, refine_head, visibility_head in zip(
            self.sampling_offset_heads,
            self.instance_sampling_attention_heads,
            self.reference_refine_heads,
            self.visibility_heads,
        ):
            init.zeros_(offset_head.weight)
            init.zeros_(offset_head.bias)
            init.zeros_(attention_head.weight)
            init.zeros_(attention_head.bias)
            init.zeros_(refine_head.layers[-1].weight)
            init.zeros_(refine_head.layers[-1].bias)
            init.zeros_(visibility_head.bias)
        if self.use_full_route:
            init.zeros_(self.shared_sampling_offset_head.weight)
            init.zeros_(self.shared_sampling_offset_head.bias)
            init.zeros_(self.shared_instance_sampling_attention_head.weight)
            init.zeros_(self.shared_instance_sampling_attention_head.bias)
            for layer_refine, layer_visibility, layer_uncertainty in zip(
                self.full_reference_refine_heads,
                self.full_visibility_heads,
                self.full_uncertainty_heads,
            ):
                for refine, visibility, uncertainty in zip(
                    layer_refine, layer_visibility, layer_uncertainty,
                ):
                    init.zeros_(refine.layers[-1].weight)
                    init.zeros_(refine.layers[-1].bias)
                    init.zeros_(visibility.bias)
                    init.zeros_(uncertainty.bias)

    @staticmethod
    def _box_sine_embedding(boxes, hidden_dim, temperature=10000.0):
        frequency_count = hidden_dim // 8
        frequencies = torch.arange(
            frequency_count, device=boxes.device, dtype=boxes.dtype,
        )
        frequencies = temperature ** (frequencies / max(frequency_count, 1))
        angles = boxes.unsqueeze(-1) * (2.0 * math.pi) / frequencies
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-3)

    def _sample_local_features_full(self, endpoint_query, instance_query,
                                    reference_points, box_centers, box_sizes,
                                    projected_features):
        batch_size, num_queries, num_keypoints, _ = endpoint_query.shape
        offsets = self.shared_sampling_offset_head(endpoint_query).view(
            batch_size, num_queries, num_keypoints, self.num_levels,
            self.num_sampling_points, 2,
        )
        learned_offsets = 0.25 * torch.tanh(offsets)
        base = self.base_sampling_offsets.view(
            1, 1, 1, 1, self.num_sampling_points, 2,
        ).expand(
            batch_size, num_queries, num_keypoints, self.num_levels,
            self.num_sampling_points, 2,
        ).clone()
        reference_offset = (
            (reference_points - box_centers[:, :, None, :])
            / box_sizes[:, :, None, :]
        ).clamp(-0.75, 0.75)
        base[..., 0, :] = reference_offset[:, :, :, None, :]
        sampling_points = (
            box_centers[:, :, None, None, None, :]
            + (base + learned_offsets) * box_sizes[:, :, None, None, None, :]
        ).clamp(0.0, 1.0)

        sampled_levels = []
        for level_idx, feature in enumerate(projected_features):
            grid = sampling_points[:, :, :, level_idx].mul(2).sub(1)
            grid = grid.reshape(
                batch_size, num_queries * num_keypoints * self.num_sampling_points, 1, 2,
            )
            sampled = F.grid_sample(
                feature, grid, mode='bilinear', padding_mode='zeros', align_corners=False,
            )
            sampled_levels.append(sampled.squeeze(-1).transpose(1, 2).reshape(
                batch_size, num_queries, num_keypoints,
                self.num_sampling_points, self.hidden_dim,
            ))
        sampled_features = torch.cat(sampled_levels, dim=3)
        attention = self.shared_instance_sampling_attention_head(instance_query).view(
            batch_size, num_queries, 1,
            self.num_levels * self.num_sampling_points, 1,
        ).softmax(dim=3)
        return (sampled_features * attention).sum(dim=3)

    def _forward_full_route(self, instance_queries, boxes, projected_features,
                            initial_references=None):
        if len(projected_features) != self.num_levels:
            raise ValueError('Endpoint decoder requires one projected feature per level.')
        keypoints, visibility, uncertainty, endpoint_features = [], [], [], []
        previous_reference = None
        semantic = self.endpoint_semantic_embed.view(
            1, 1, self.num_keypoints, self.hidden_dim,
        )
        for layer_idx, instance_query in enumerate(instance_queries):
            box = boxes[layer_idx]
            box_centers = box[..., :2]
            box_sizes = box[..., 2:].clamp_min(1e-4)
            box_sine = self._box_sine_embedding(box, self.hidden_dim)
            endpoint_query = self.full_query_generators[layer_idx](torch.cat([
                instance_query[:, :, None, :].expand(-1, -1, self.num_keypoints, -1),
                semantic.expand(instance_query.shape[0], instance_query.shape[1], -1, -1),
                box_sine[:, :, None, :].expand(-1, -1, self.num_keypoints, -1),
            ], dim=-1))

            if previous_reference is None:
                body_axis = F.normalize(self.body_axis_head(instance_query), dim=-1, eps=1e-6)
                long_side = box_sizes.amax(dim=-1, keepdim=True)
                signs = box.new_tensor([1.0, -1.0]).view(1, 1, 2, 1)
                reference_points = (
                    box_centers[:, :, None, :]
                    + signs * self.body_axis_radius * long_side[:, :, None, :] * body_axis[:, :, None, :]
                )
                if initial_references is not None:
                    use_override = torch.isfinite(initial_references).all(-1, keepdim=True)
                    reference_points = torch.where(
                        use_override, initial_references, reference_points,
                    )
            else:
                reference_points = previous_reference.detach()
            reference_points = reference_points.clamp(0.0, 1.0)

            local_feature = self._sample_local_features_full(
                endpoint_query, instance_query, reference_points,
                box_centers, box_sizes, projected_features,
            )
            endpoint_query = self.endpoint_norms[layer_idx](
                endpoint_query + self.shared_sample_projection(local_feature)
            )
            endpoint_query = torch.stack([
                self.endpoint_output_mlps[layer_idx][endpoint_idx](
                    endpoint_query[:, :, endpoint_idx]
                )
                for endpoint_idx in range(self.num_keypoints)
            ], dim=2)
            refinement = torch.stack([
                self.full_reference_refine_heads[layer_idx][endpoint_idx](
                    endpoint_query[:, :, endpoint_idx]
                )
                for endpoint_idx in range(self.num_keypoints)
            ], dim=2)
            reference_points = torch.sigmoid(
                inverse_sigmoid(reference_points)
                + self.endpoint_delta_limit * torch.tanh(refinement)
            )
            previous_reference = reference_points
            keypoints.append(reference_points)
            visibility.append(torch.cat([
                self.full_visibility_heads[layer_idx][endpoint_idx](
                    endpoint_query[:, :, endpoint_idx]
                )
                for endpoint_idx in range(self.num_keypoints)
            ], dim=-1))
            uncertainty.append(F.softplus(torch.cat([
                self.full_uncertainty_heads[layer_idx][endpoint_idx](
                    endpoint_query[:, :, endpoint_idx]
                )
                for endpoint_idx in range(self.num_keypoints)
            ], dim=-1)))
            endpoint_features.append(endpoint_query)
        return (
            torch.stack(keypoints), torch.stack(visibility),
            torch.stack(uncertainty), torch.stack(endpoint_features),
        )

    def _sample_local_features(self, endpoint_query, instance_query,
                               reference_points, box_centers, box_sizes,
                               projected_features, layer_idx):
        batch_size, num_queries, num_keypoints, _ = endpoint_query.shape
        offsets = self.sampling_offset_heads[layer_idx](endpoint_query).view(
            batch_size,
            num_queries,
            num_keypoints,
            self.num_levels,
            self.num_sampling_points,
            2,
        )
        learned_offsets = 0.25 * torch.tanh(offsets)
        base = self.base_sampling_offsets.view(
            1, 1, 1, 1, self.num_sampling_points, 2
        ).expand(
            batch_size, num_queries, num_keypoints,
            self.num_levels, self.num_sampling_points, 2,
        ).clone()
        # Slot zero always samples the endpoint reference propagated from the
        # previous decoder layer. Remaining slots are the box centre, long/
        # short-axis ends and diagonal context locations.
        reference_offset = (
            (reference_points - box_centers[:, :, None, :])
            / box_sizes[:, :, None, :]
        ).clamp(-0.75, 0.75)
        base[..., 0, :] = reference_offset[:, :, :, None, :]
        sampling_points = (
            box_centers[:, :, None, None, None, :]
            + (base + learned_offsets)
            * box_sizes[:, :, None, None, None, :]
        )
        sampling_points = sampling_points.clamp(0.0, 1.0)

        sampled_levels = []
        for level_idx, feature in enumerate(projected_features):
            grid = sampling_points[:, :, :, level_idx].mul(2).sub(1)
            grid = grid.reshape(batch_size, num_queries * num_keypoints * self.num_sampling_points, 1, 2)
            sampled = F.grid_sample(
                feature,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(
                batch_size,
                num_queries,
                num_keypoints,
                self.num_sampling_points,
                self.hidden_dim,
            )
            sampled_levels.append(sampled)

        sampled_features = torch.cat(sampled_levels, dim=3)
        attention = self.instance_sampling_attention_heads[layer_idx](instance_query)
        attention = attention.view(
            batch_size, num_queries, 1,
            self.num_levels * self.num_sampling_points, 1,
        ).softmax(dim=3)
        return (sampled_features * attention).sum(dim=3)

    def forward(self, instance_queries, boxes, projected_features,
                initial_references=None):
        if self.use_full_route:
            return self._forward_full_route(
                instance_queries, boxes, projected_features, initial_references,
            )
        if len(projected_features) != self.num_levels:
            raise ValueError('Endpoint decoder requires one projected feature per level.')

        keypoints, visibility = [], []
        previous_reference = None
        semantic = self.endpoint_semantic_embed.view(1, 1, self.num_keypoints, self.hidden_dim)
        for layer_idx, instance_query in enumerate(instance_queries):
            endpoint_query = instance_query.unsqueeze(2) + semantic
            endpoint_query = self.endpoint_query_layers[layer_idx](endpoint_query)
            box_centers = boxes[layer_idx][..., :2]
            box_sizes = boxes[layer_idx][..., 2:].clamp_min(1e-4)

            if previous_reference is None:
                initial_offset = torch.tanh(
                    self.initial_reference_heads[layer_idx](endpoint_query)
                )
                default_reference = (
                    box_centers[:, :, None, :]
                    + 0.5 * initial_offset * box_sizes[:, :, None, :]
                )
                if initial_references is not None:
                    use_override = torch.isfinite(initial_references).all(-1, keepdim=True)
                    reference_points = torch.where(
                        use_override, initial_references, default_reference
                    )
                else:
                    reference_points = default_reference
            else:
                reference_points = previous_reference.detach()
            reference_points = reference_points.clamp(0.0, 1.0)

            local_feature = self._sample_local_features(
                endpoint_query,
                instance_query,
                reference_points,
                box_centers,
                box_sizes,
                projected_features,
                layer_idx,
            )
            endpoint_query = self.endpoint_norms[layer_idx](
                endpoint_query + self.sample_projections[layer_idx](local_feature)
            )
            refinement = torch.tanh(
                self.reference_refine_heads[layer_idx](endpoint_query)
            )
            reference_points = (
                reference_points + 0.25 * refinement * box_sizes[:, :, None, :]
            ).clamp(0.0, 1.0)
            previous_reference = reference_points
            keypoints.append(reference_points)
            visibility.append(self.visibility_heads[layer_idx](endpoint_query).squeeze(-1))

        return torch.stack(keypoints), torch.stack(visibility)
    

class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)
    

class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max+1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score

class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method)

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int]):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None,
                 ):
        super(TransformerDecoderLayer, self).__init__()

        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # self.ffn = SwiGLUFFN(d_model, dim_feedforward, d_model)
        
        self.dropout4 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                target,
                reference_points,
                value,
                spatial_shapes,
                attn_mask=None,
                query_pos_embed=None,
                query_key_padding_mask=None):

        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(
            q,
            k,
            value=target,
            attn_mask=attn_mask,
            key_padding_mask=query_key_padding_mask,
        )
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes)

        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        # target2 = self.ffn(target)
        
        target = target + self.dropout4(target2)
        target = self.norm2(target.clamp(min=-65504, max=65504))

        return target


class SceneRoutedHead(nn.Module):
    """Run every scene branch and select one branch per sample."""

    def __init__(self, head, num_scenes=2):
        super().__init__()
        if int(num_scenes) != 2:
            raise ValueError('The BeePoseTrack-E scene route requires indoor/outdoor heads.')
        self.heads = nn.ModuleList([copy.deepcopy(head) for _ in range(int(num_scenes))])

    def forward(self, x, scene_id):
        if scene_id is None:
            raise RuntimeError('Scene-routed detection heads require scene_id.')
        scene_id = torch.as_tensor(scene_id, device=x.device, dtype=torch.long).reshape(-1)
        if scene_id.shape != (x.shape[0],):
            raise ValueError(f'scene_id must have shape [{x.shape[0]}].')
        if not bool(((scene_id == 0) | (scene_id == 1)).all()):
            raise ValueError('scene_id values must be 0 (indoor/B) or 1 (outdoor/A).')
        branch_outputs = torch.stack([head(x) for head in self.heads], dim=1)
        selector = F.one_hot(scene_id, num_classes=len(self.heads)).to(branch_outputs.dtype)
        selector = selector.reshape(selector.shape[0], selector.shape[1], *(
            [1] * (branch_outputs.ndim - 2)
        ))
        return (branch_outputs * selector).sum(dim=1)


def _apply_detection_head(head, x, scene_id=None):
    if isinstance(head, SceneRoutedHead):
        return head(x, scene_id)
    return head(x)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, segmentation_head, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act='relu'):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] \
                    + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
        self.segmentation_head = segmentation_head
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                spatial_features,
                target,
                ref_points_unact,
                memory,
                spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None,
                query_valid_mask=None,
                scene_id=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        dec_out_hs = []
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            query_key_padding_mask = (
                ~query_valid_mask if query_valid_mask is not None else None
            )
            output = layer(
                output,
                ref_points_input,
                value,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
                query_key_padding_mask,
            )
            if query_valid_mask is not None:
                output = output.masked_fill(~query_valid_mask.unsqueeze(-1), 0.0)

            if i == 0 :
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(
                    _apply_detection_head(pre_bbox_head, output, scene_id)
                    + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = _apply_detection_head(score_head[0], output, scene_id)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = _apply_detection_head(
                bbox_head[i], output + output_detach, scene_id
            ) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = _apply_detection_head(score_head[i], output, scene_id)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                dec_out_hs.append(output)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        if spatial_features is not None:
            dec_out_segs = self.segmentation_head(
                spatial_features=spatial_features,   # fused_feats[0], backbone feature
                query_features=dec_out_hs,           # list[Tensor], [N,B,Nq,C]
            )

            return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
                torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), torch.stack(dec_out_hs), torch.stack(dec_out_segs), \
                pre_bboxes, pre_scores, dec_out_segs[-1]
        else:
            return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
                torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), torch.stack(dec_out_hs), None, \
                pre_bboxes, pre_scores, None

@register()
class ECTransformer(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="silu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True,
                 cross_attn_method='default',
                 query_select_method='default',
                 reg_max=32,
                 reg_scale=4.,
                 layer_scale=1,
                 share_bbox_head=False,
                 share_score_head=False,
                 mask_downsample_ratio=None,
                 num_keypoints=0,
                 use_local_endpoint_sampling=False,
                 endpoint_decoder_type='direct',
                 endpoint_sampling_points=4,
                 use_full_endpoint_route=False,
                 body_axis_radius=0.25,
                 endpoint_delta_limit=2.0,
                 use_density_query=False,
                 use_density_peaks=False,
                 density_peak_ratio=0.5,
                 density_peak_kernel=3,
                 density_peak_dedup_radius=0.02,
                 enable_monotonic_query_capacity=False,
                 min_active_queries=128,
                 density_capacity_ratio=1.25,
                 density_capacity_padding=32.0,
                 uncertainty_capacity_scale=64.0,
                 stage_query_limit=None,
                 query_capacity_by_scene=None,
                 use_pattern_query=False,
                 num_query_patterns=8,
                 domain_context_dim=2,
                 use_quality_head=False,
                 use_shared_query_content=False,
                 use_track_geometry_head=False,
                 keypoint_noise_scale=0.0,
                 head_tail_swap_ratio=0.0,
                 no_object_noise_ratio=0.0,
                 denoising_base_fraction=0.25,
                 denoising_entropy_gain=0.75,
                 initial_denoising_fraction=0.25,
                 denoising_debt_gain=0.5,
                 max_denoising_multiplier=1.5,
                 enable_domain_layernorm=False,
                 enable_scene_detection_heads=False,
                 num_scene_heads=2,
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale*hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.num_keypoints = num_keypoints
        self.use_local_endpoint_sampling = use_local_endpoint_sampling
        self.endpoint_decoder_type = endpoint_decoder_type
        self.use_full_endpoint_route = bool(use_full_endpoint_route)
        self.use_density_query = use_density_query
        self.use_density_peaks = use_density_peaks
        self.density_peak_ratio = density_peak_ratio
        self.density_peak_kernel = density_peak_kernel
        self.density_peak_dedup_radius = density_peak_dedup_radius
        self.enable_monotonic_query_capacity = bool(enable_monotonic_query_capacity)
        self.min_active_queries = int(min_active_queries)
        self.density_capacity_ratio = float(density_capacity_ratio)
        self.density_capacity_padding = float(density_capacity_padding)
        self.uncertainty_capacity_scale = float(uncertainty_capacity_scale)
        self.stage_query_limit = (
            self.num_queries if stage_query_limit is None else int(stage_query_limit)
        )
        if (
            self.enable_monotonic_query_capacity
            and not self.min_active_queries <= self.stage_query_limit <= self.num_queries
        ):
            raise ValueError('Query capacity must satisfy min <= stage limit <= N_cap.')
        self.query_capacity_by_scene = dict(query_capacity_by_scene or {})
        for scene_name, capacity in self.query_capacity_by_scene.items():
            if scene_name not in ('indoor', 'outdoor', '0', '1', 0, 1):
                raise ValueError(f'Unsupported scene query-capacity key: {scene_name}')
            minimum = int(capacity.get('min_active_queries', self.min_active_queries))
            limit = int(capacity.get('stage_query_limit', self.stage_query_limit))
            if not 1 <= minimum <= limit <= self.num_queries:
                raise ValueError(
                    f'Invalid query capacity for scene {scene_name}: {minimum}/{limit}'
                )
        self.use_pattern_query = use_pattern_query
        self.use_quality_head = use_quality_head
        self.use_shared_query_content = bool(use_shared_query_content)
        self.use_track_geometry_head = use_track_geometry_head
        self.keypoint_noise_scale = keypoint_noise_scale
        self.head_tail_swap_ratio = head_tail_swap_ratio
        self.no_object_noise_ratio = float(no_object_noise_ratio)
        self.denoising_base_fraction = float(denoising_base_fraction)
        self.denoising_entropy_gain = float(denoising_entropy_gain)
        self.initial_denoising_fraction = float(initial_denoising_fraction)
        self.denoising_debt_gain = float(denoising_debt_gain)
        self.max_denoising_multiplier = float(max_denoising_multiplier)
        if not 0.0 < self.initial_denoising_fraction <= 1.0:
            raise ValueError('initial_denoising_fraction must be in (0, 1].')
        if self.denoising_debt_gain < 0.0:
            raise ValueError('denoising_debt_gain must be non-negative.')
        if self.max_denoising_multiplier < 1.0:
            raise ValueError('max_denoising_multiplier must be at least 1.')
        if not 0.0 <= self.no_object_noise_ratio <= 1.0:
            raise ValueError('no_object_noise_ratio must be in [0, 1].')
        if not 0.0 <= self.denoising_base_fraction <= 1.0:
            raise ValueError('denoising_base_fraction must be in [0, 1].')
        if self.denoising_entropy_gain < 0.0:
            raise ValueError('denoising_entropy_gain must be non-negative.')
        self.enable_domain_layernorm = bool(enable_domain_layernorm)
        self.enable_scene_detection_heads = bool(enable_scene_detection_heads)
        self.num_scene_heads = int(num_scene_heads)
        if self.enable_scene_detection_heads and self.num_scene_heads != 2:
            raise ValueError('Scene detection routing requires exactly two heads.')

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        assert endpoint_decoder_type in ('direct', 'decomposed'), ''
        if endpoint_decoder_type == 'decomposed' and scaled_dim != hidden_dim:
            raise ValueError('Decomposed endpoint queries require layer_scale=1.')
        if self.use_full_endpoint_route and endpoint_decoder_type != 'decomposed':
            raise ValueError('The full endpoint route requires endpoint_decoder_type=decomposed.')
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method)
        decoder_layer_wide = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, layer_scale=layer_scale)
        
        # SegmetationHead
        segmentation_head = SegmentationHead(hidden_dim, num_layers, downsample_ratio=mask_downsample_ratio, image_size=eval_spatial_size) if mask_downsample_ratio else None
        
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, decoder_layer_wide, segmentation_head, num_layers, nhead,
                                          reg_max, self.reg_scale, self.up, eval_idx, layer_scale, act=activation)
      # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.base_num_denoising = int(num_denoising)
        self.base_label_noise_ratio = float(label_noise_ratio)
        self.base_box_noise_scale = float(box_noise_scale)
        self.base_keypoint_noise_scale = float(keypoint_noise_scale)
        self.base_head_tail_swap_ratio = float(head_tail_swap_ratio)
        self.base_no_object_noise_ratio = float(no_object_noise_ratio)
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        if self.use_shared_query_content:
            self.shared_query_content = nn.Parameter(torch.empty(1, hidden_dim))

        if query_select_method == 'agnostic':
            enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_score_head = self._scene_head(enc_score_head)
        self.enc_bbox_head = self._scene_head(
            MLP(hidden_dim, hidden_dim, 4, 3, act=activation)
        )

        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=activation)
        if self.use_density_query:
            self.density_query_scale = nn.Parameter(torch.tensor(1.0))
        if self.use_pattern_query:
            self.query_patterns = nn.Parameter(torch.empty(num_query_patterns, hidden_dim))
            self.pattern_router = nn.Linear(hidden_dim * 2, num_query_patterns)
            self.global_pattern_router = nn.Linear(hidden_dim, num_query_patterns)
            self.domain_pattern_router = nn.Linear(domain_context_dim, num_query_patterns)
            self.content_query_generator = MLP(hidden_dim * 2, hidden_dim, hidden_dim, 2, act=activation)

        # decoder head
        self.pre_bbox_head = self._scene_head(
            MLP(hidden_dim, hidden_dim, 4, 3, act=activation)
        )
        self.integral = Integral(self.reg_max)

        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        dec_score_head = nn.Linear(hidden_dim, num_classes)
        raw_dec_score_head = nn.ModuleList(
            [dec_score_head if share_score_head else copy.deepcopy(dec_score_head) for _ in range(self.eval_idx + 1)]
          + [copy.deepcopy(dec_score_head) for _ in range(num_layers - self.eval_idx - 1)])
        self.dec_score_head = nn.ModuleList([
            self._scene_head(head) for head in raw_dec_score_head
        ])

        # Share the same bbox head for all layers
        dec_bbox_head = MLP(hidden_dim, hidden_dim, 4 * (self.reg_max+1), 3, act=activation)
        raw_dec_bbox_head = nn.ModuleList(
            [dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head) for _ in range(self.eval_idx + 1)]
          + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max+1), 3, act=activation) for _ in range(num_layers - self.eval_idx - 1)])
        self.dec_bbox_head = nn.ModuleList([
            self._scene_head(head) for head in raw_dec_bbox_head
        ])

        if self.num_keypoints > 0:
            pose_dims = [hidden_dim] * (self.eval_idx + 1) + [scaled_dim] * (num_layers - self.eval_idx - 1)
            if self.endpoint_decoder_type == 'decomposed':
                self.endpoint_decoder = DecomposedEndpointDecoder(
                    hidden_dim,
                    self.num_keypoints,
                    num_layers,
                    num_levels,
                    num_sampling_points=endpoint_sampling_points,
                    act=activation,
                    use_full_route=self.use_full_endpoint_route,
                    body_axis_radius=body_axis_radius,
                    endpoint_delta_limit=endpoint_delta_limit,
                )
            else:
                self.dec_keypoint_head = nn.ModuleList([
                    MLP(dim, dim, self.num_keypoints * 2, 3, act=activation) for dim in pose_dims
                ])
                self.dec_visibility_head = nn.ModuleList([
                    nn.Linear(dim, self.num_keypoints) for dim in pose_dims
                ])
                if self.use_local_endpoint_sampling:
                    self.dec_keypoint_refine_head = nn.ModuleList([
                        MLP(dim + self.num_keypoints * hidden_dim, dim, self.num_keypoints * 2, 3, act=activation)
                        for dim in pose_dims
                    ])
                    self.dec_local_visibility_head = nn.ModuleList([
                        nn.Linear(dim + self.num_keypoints * hidden_dim, self.num_keypoints)
                        for dim in pose_dims
                    ])

        if self.use_quality_head:
            quality_dims = [hidden_dim] * (self.eval_idx + 1) + [scaled_dim] * (num_layers - self.eval_idx - 1)
            if self.use_full_endpoint_route:
                self.dec_quality_head = nn.ModuleList([
                    MLP(dim * 3, dim, 1, 2, act=activation) for dim in quality_dims
                ])
            else:
                self.dec_quality_head = nn.ModuleList([
                    nn.Linear(dim, 1) for dim in quality_dims
                ])
        if self.use_track_geometry_head:
            self.track_geometry_head = MLP(hidden_dim, hidden_dim, 6, 3, act=activation)

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()


        self._reset_parameters(feat_channels)
        if self.enable_domain_layernorm:
            replace_layer_norms(self)

    def set_stage_denoising(self, stage, progress, matching_debt=0.0):
        """Apply the calibrated E-S0..E-S5 denoising curriculum.

        The underlying denoising builder already maps the requested capacity
        monotonically through ``N_gt``. This method supplies the stage cap and
        noise envelope without changing reconstruction targets or attention
        isolation.
        """
        if stage not in ('E-S0', 'E-S1', 'E-S2', 'E-S3', 'E-S4', 'E-S5'):
            raise ValueError(f'Unsupported BeePoseTrack-E stage: {stage}')
        progress = min(max(float(progress), 0.0), 1.0)
        matching_debt = max(float(matching_debt), 0.0)
        if stage == 'E-S0':
            capacity_scale = self.initial_denoising_fraction
            box_scale = self.initial_denoising_fraction
            endpoint_scale = 0.0
        elif stage == 'E-S1':
            capacity_scale = self.initial_denoising_fraction + (
                1.0 - self.initial_denoising_fraction
            ) * progress
            box_scale = capacity_scale
            endpoint_scale = 0.0
        elif stage == 'E-S2':
            capacity_scale = 1.0
            box_scale = 1.0
            endpoint_scale = progress
        elif stage == 'E-S4':
            capacity_scale = min(
                self.max_denoising_multiplier,
                1.0 + self.denoising_debt_gain * matching_debt,
            )
            box_scale = capacity_scale
            endpoint_scale = capacity_scale
        else:
            capacity_scale = 1.0
            box_scale = 1.0
            endpoint_scale = 1.0

        active_query_cap = max(1, int(self.stage_query_limit))
        requested = int(round(self.base_num_denoising * capacity_scale))
        self.num_denoising = min(active_query_cap, max(0, requested))
        self.label_noise_ratio = min(
            1.0, self.base_label_noise_ratio * box_scale,
        )
        self.box_noise_scale = self.base_box_noise_scale * box_scale
        self.keypoint_noise_scale = (
            self.base_keypoint_noise_scale * endpoint_scale
        )
        self.head_tail_swap_ratio = min(
            1.0, self.base_head_tail_swap_ratio * endpoint_scale,
        )
        self.no_object_noise_ratio = min(
            1.0, self.base_no_object_noise_ratio * box_scale,
        )
        return {
            'num_denoising': self.num_denoising,
            'label_noise_ratio': self.label_noise_ratio,
            'box_noise_scale': self.box_noise_scale,
            'keypoint_noise_scale': self.keypoint_noise_scale,
            'head_tail_swap_ratio': self.head_tail_swap_ratio,
            'no_object_noise_ratio': self.no_object_noise_ratio,
        }

    def apply_query_capacity_contract(self, contract):
        """Freeze the E-S5 calibration-selected monotone capacity mapping."""
        capacity = contract.get('query_capacity', contract)
        if capacity.get('source_data') != 'calibration':
            raise ValueError('Query capacity must be fitted on calibration.')
        if capacity.get('effective_stage') != 'E-S5':
            raise ValueError('Query capacity must be frozen at E-S5.')
        parameters = capacity.get('parameters', {})
        required = {
            'density_capacity_ratio', 'density_capacity_padding',
            'uncertainty_capacity_scale', 'min_active_queries',
            'stage_query_limit',
        }
        missing = required - set(parameters)
        if missing:
            raise ValueError(f'Query capacity contract misses {sorted(missing)}')
        minimum = int(parameters['min_active_queries'])
        limit = int(parameters['stage_query_limit'])
        if not 1 <= minimum <= limit <= self.num_queries:
            raise ValueError('Query capacity must satisfy 1 <= min <= limit <= N_cap.')
        self.density_capacity_ratio = float(parameters['density_capacity_ratio'])
        self.density_capacity_padding = float(parameters['density_capacity_padding'])
        self.uncertainty_capacity_scale = float(parameters['uncertainty_capacity_scale'])
        self.min_active_queries = minimum
        self.stage_query_limit = limit
        return self

    def _scene_head(self, head):
        if not self.enable_scene_detection_heads:
            return head
        return SceneRoutedHead(head, self.num_scene_heads)

    @staticmethod
    def _head_branches(head):
        if isinstance(head, SceneRoutedHead):
            return head.heads
        return (head,)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        for head in self._head_branches(self.enc_score_head):
            init.constant_(head.bias, bias)
        for head in self._head_branches(self.enc_bbox_head):
            init.constant_(head.layers[-1].weight, 0)
            init.constant_(head.layers[-1].bias, 0)

        for head in self._head_branches(self.pre_bbox_head):
            init.constant_(head.layers[-1].weight, 0)
            init.constant_(head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            for head in self._head_branches(cls_):
                init.constant_(head.bias, bias)
            for head in self._head_branches(reg_):
                if hasattr(head, 'layers'):
                    init.constant_(head.layers[-1].weight, 0)
                    init.constant_(head.layers[-1].bias, 0)

        if self.num_keypoints > 0 and self.endpoint_decoder_type == 'direct':
            for keypoint_head, visibility_head in zip(self.dec_keypoint_head, self.dec_visibility_head):
                init.constant_(keypoint_head.layers[-1].weight, 0)
                init.constant_(keypoint_head.layers[-1].bias, 0)
                init.constant_(visibility_head.bias, 0)
            if self.use_local_endpoint_sampling:
                for refine_head, visibility_head in zip(
                    self.dec_keypoint_refine_head, self.dec_local_visibility_head
                ):
                    init.constant_(refine_head.layers[-1].weight, 0)
                    init.constant_(refine_head.layers[-1].bias, 0)
                    init.constant_(visibility_head.bias, 0)

        if self.use_quality_head:
            for quality_head in self.dec_quality_head:
                if hasattr(quality_head, 'bias'):
                    init.constant_(quality_head.bias, 0)
                else:
                    init.zeros_(quality_head.layers[-1].weight)
                    init.zeros_(quality_head.layers[-1].bias)
        if self.use_track_geometry_head:
            init.zeros_(self.track_geometry_head.layers[-1].weight)
            init.zeros_(self.track_geometry_head.layers[-1].bias)

        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        if self.use_shared_query_content:
            init.normal_(self.shared_query_content, std=0.02)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)
        if self.use_pattern_query:
            init.normal_(self.query_patterns, std=0.02)
            init.xavier_uniform_(self.pattern_router.weight)
            init.zeros_(self.pattern_router.bias)
            init.xavier_uniform_(self.global_pattern_router.weight)
            init.zeros_(self.global_pattern_router.bias)
            init.xavier_uniform_(self.domain_pattern_router.weight)
            init.zeros_(self.domain_pattern_router.bias)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                        ('norm', nn.BatchNorm2d(self.hidden_dim))])
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes, proj_feats

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _extract_density_peak_candidates(self, density_prior, p2_feature):
        """Build real query references from local maxima on the P2 density map."""
        if density_prior is None:
            return None
        batch_size, _, height, width = density_prior.shape
        peak_count = min(
            self.num_queries,
            height * width,
            max(1, int(round(self.num_queries * self.density_peak_ratio))),
        )
        kernel = max(1, int(self.density_peak_kernel))
        if kernel % 2 == 0:
            kernel += 1
        pooled = F.max_pool2d(density_prior, kernel, stride=1, padding=kernel // 2)
        peak_scores = density_prior.flatten(1)
        is_peak = (density_prior >= pooled).flatten(1)
        ranked_scores = peak_scores.masked_fill(~is_peak, -torch.inf)
        values, indices = ranked_scores.topk(peak_count, dim=1)

        y = indices // width
        x = indices % width
        xy = torch.stack([
            (x.to(density_prior.dtype) + 0.5) / width,
            (y.to(density_prior.dtype) + 0.5) / height,
        ], dim=-1)
        wh = xy.new_full(xy.shape, 0.05)
        anchors_unact = inverse_sigmoid(torch.cat([xy, wh], dim=-1))

        grid = xy.mul(2).sub(1).unsqueeze(2)
        content = F.grid_sample(
            p2_feature, grid, mode='bilinear', padding_mode='zeros', align_corners=False
        ).squeeze(-1).transpose(1, 2)
        return content, anchors_unact, values, xy

    @staticmethod
    def _query_diagnostics(initial_boxes, pattern_weights=None, valid_query_mask=None):
        centers = initial_boxes[..., :2].clamp(0, 1)
        cells = (centers * 8).long().clamp(0, 7)
        cell_ids = cells[..., 1] * 8 + cells[..., 0]
        coverage = initial_boxes.new_tensor([
            ids.unique().numel() / 64.0 for ids in cell_ids
        ])
        sample = centers[:, :min(128, centers.shape[1])]
        if sample.shape[1] > 1:
            distance = torch.cdist(sample, sample)
            eye = torch.eye(sample.shape[1], dtype=torch.bool, device=sample.device)[None]
            collapse = ((distance < 0.01) & ~eye).float().sum((1, 2))
            collapse = collapse / (sample.shape[1] * (sample.shape[1] - 1))
        else:
            collapse = initial_boxes.new_zeros(initial_boxes.shape[0])
        diagnostics = {
            'query_spatial_coverage': coverage,
            'query_collapse': collapse,
            'query_initial_references': initial_boxes,
        }
        if valid_query_mask is not None:
            diagnostics['query_utilization'] = valid_query_mask.float().mean(1)
        if pattern_weights is not None:
            usage = pattern_weights.mean(1)
            diagnostics['pred_pattern_weights'] = pattern_weights
            diagnostics['pattern_usage'] = usage
            diagnostics['pattern_entropy'] = -(
                usage.clamp_min(1e-8) * usage.clamp_min(1e-8).log()
            ).sum(-1)
        return diagnostics

    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None,
                           density_prior_flat=None,
                           density_prior=None,
                           p2_feature=None,
                           domain_context=None,
                           scene_id=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory

        enc_outputs_logits :torch.Tensor = _apply_detection_head(
            self.enc_score_head, memory, scene_id
        )
        if self.use_density_query and density_prior_flat is not None:
            normalized_density = (
                density_prior_flat - density_prior_flat.mean(dim=1, keepdim=True)
            ) / density_prior_flat.std(dim=1, keepdim=True).clamp_min(1e-6)
            enc_outputs_logits = enc_outputs_logits + self.density_query_scale * normalized_density

        # Anchors outside the valid grid are stored as +inf sentinels.  They
        # must never enter top-k selection, otherwise a single invalid
        # reference can spread NaNs to every query through self-attention.
        enc_outputs_logits = enc_outputs_logits.masked_fill(~valid_mask, -torch.inf)
        foreground_probability = enc_outputs_logits.sigmoid().amax(dim=-1)
        candidate_count = min(self.num_queries, foreground_probability.shape[1])
        candidate_probability = foreground_probability.topk(candidate_count, dim=1).values
        candidate_entropy = -(
            candidate_probability.clamp(1e-6, 1 - 1e-6)
            * candidate_probability.clamp(1e-6, 1 - 1e-6).log()
            + (1 - candidate_probability).clamp(1e-6, 1 - 1e-6)
            * (1 - candidate_probability).clamp(1e-6, 1 - 1e-6).log()
        ) / math.log(2.0)
        query_uncertainty = candidate_entropy.mean(dim=1)

        density_candidates = None
        if self.use_density_peaks and density_prior is not None and p2_feature is not None:
            density_candidates = self._extract_density_peak_candidates(density_prior, p2_feature)

        if density_candidates is None:
            enc_topk_memory, enc_topk_logits, enc_topk_anchors, _ = \
                self._select_topk(memory, enc_outputs_logits, anchors, self.num_queries)
        else:
            peak_memory, peak_anchors, peak_scores, peak_xy = density_candidates
            refill_count = self.num_queries - peak_memory.shape[1]
            if refill_count > 0:
                encoder_xy = anchors.sigmoid()[..., :2]
                near_peak = torch.cdist(encoder_xy, peak_xy).amin(dim=-1) < self.density_peak_dedup_radius
                dedup_logits = enc_outputs_logits.masked_fill(near_peak.unsqueeze(-1), -torch.inf)
                refill_memory, refill_logits, refill_anchors, _ = self._select_topk(
                    memory, dedup_logits, anchors, refill_count
                )
                enc_topk_memory = torch.cat([peak_memory, refill_memory], dim=1)
                enc_topk_anchors = torch.cat([peak_anchors, refill_anchors], dim=1)
                peak_logits = _apply_detection_head(
                    self.enc_score_head, peak_memory, scene_id
                )
                if self.training:
                    enc_topk_logits = torch.cat([peak_logits, refill_logits], dim=1)
            else:
                enc_topk_memory = peak_memory
                enc_topk_anchors = peak_anchors
                enc_topk_logits = (
                    _apply_detection_head(self.enc_score_head, peak_memory, scene_id)
                    if self.training else None
                )

        enc_topk_bbox_unact :torch.Tensor = _apply_detection_head(
            self.enc_bbox_head, enc_topk_memory, scene_id
        ) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.use_shared_query_content:
            content = self.shared_query_content.view(1, 1, -1).expand(
                memory.shape[0], self.num_queries, -1,
            )
        elif self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        pattern_weights = None
        if self.use_pattern_query:
            global_context = memory.mean(dim=1, keepdim=True).expand_as(enc_topk_memory)
            router_input = torch.cat([enc_topk_memory, global_context], dim=-1)
            router_logits = self.pattern_router(router_input)
            router_logits = router_logits + self.global_pattern_router(global_context)
            if domain_context is not None:
                router_logits = router_logits + self.domain_pattern_router(domain_context)[:, None]
            pattern_weights = router_logits.softmax(dim=-1)
            generated_content = self.content_query_generator(router_input)
            content = content + generated_content + pattern_weights @ self.query_patterns

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        diagnostics = self._query_diagnostics(
            enc_topk_bbox_unact.sigmoid(), pattern_weights=pattern_weights
        )
        diagnostics['query_candidate_uncertainty'] = query_uncertainty
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list, diagnostics

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_anchors_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1]))

        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])) if self.training else None

        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_anchors, topk_ind
    
    @staticmethod
    def _split(x, dim, s_idx):
        return torch.split(x, s_idx, dim=dim) if x is not None else (None, None)

    def _monotonic_query_budget(self, predicted_count, candidate_uncertainty,
                                scene_id=None):
        requested = torch.ceil(
            self.density_capacity_ratio * predicted_count
            + self.density_capacity_padding
            + self.uncertainty_capacity_scale * candidate_uncertainty
        ).long()
        minimum = torch.full_like(requested, self.min_active_queries)
        limit = torch.full_like(requested, self.stage_query_limit)
        if self.query_capacity_by_scene:
            if scene_id is None:
                raise RuntimeError('Scene-specific query capacity requires scene_id.')
            scene_id = torch.as_tensor(
                scene_id, device=requested.device, dtype=torch.long,
            ).reshape_as(requested)
            for numeric_id, scene_name in ((0, 'indoor'), (1, 'outdoor')):
                capacity = self.query_capacity_by_scene.get(
                    scene_name,
                    self.query_capacity_by_scene.get(
                        str(numeric_id), self.query_capacity_by_scene.get(numeric_id)
                    ),
                )
                if capacity is None:
                    continue
                scene_minimum = int(capacity.get(
                    'min_active_queries', self.min_active_queries,
                ))
                scene_limit = int(capacity.get(
                    'stage_query_limit', self.stage_query_limit,
                ))
                selected = scene_id == numeric_id
                minimum = torch.where(selected, scene_minimum, minimum)
                limit = torch.where(selected, scene_limit, limit)
        return torch.minimum(torch.maximum(requested, minimum), limit)

    def _predict_pose(self, query_features, boxes, projected_features=None,
                      initial_references=None):
        if self.num_keypoints <= 0:
            return None, None, None, None
        if self.endpoint_decoder_type == 'decomposed':
            result = self.endpoint_decoder(
                query_features, boxes, projected_features,
                initial_references=initial_references,
            )
            if self.use_full_endpoint_route:
                return result
            keypoints, visibility = result
            return keypoints, visibility, None, None

        keypoints, visibility = [], []
        for layer_idx, features in enumerate(query_features):
            offsets = self.dec_keypoint_head[layer_idx](features)
            offsets = offsets.view(*offsets.shape[:-1], self.num_keypoints, 2)
            centers = boxes[layer_idx][..., :2].unsqueeze(-2)
            sizes = boxes[layer_idx][..., 2:].unsqueeze(-2)
            points = centers + 0.5 * torch.tanh(offsets) * sizes
            if self.use_local_endpoint_sampling and projected_features is not None:
                pose_feature = projected_features[0]
                batch_size, num_queries = points.shape[:2]
                sampling_grid = points.mul(2).sub(1).reshape(batch_size, num_queries * self.num_keypoints, 1, 2)
                sampled = F.grid_sample(
                    pose_feature,
                    sampling_grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                )
                sampled = sampled.squeeze(-1).transpose(1, 2).reshape(batch_size, num_queries, -1)
                local_features = torch.cat([features, sampled], dim=-1)
                refinement = self.dec_keypoint_refine_head[layer_idx](local_features)
                refinement = refinement.view(batch_size, num_queries, self.num_keypoints, 2)
                points = points + 0.25 * torch.tanh(refinement) * sizes
                visibility.append(self.dec_local_visibility_head[layer_idx](local_features))
            else:
                visibility.append(self.dec_visibility_head[layer_idx](features))
            keypoints.append(points)
        return torch.stack(keypoints), torch.stack(visibility), None, None

    def forward(self, feats, targets=None, spatial_feat=None, density_prior=None,
                valid_query_mask=None, query_budget=None, domain_context=None,
                scene_id=None):
        if self.enable_scene_detection_heads:
            if scene_id is None and targets is not None:
                if not all('scene_id' in target for target in targets):
                    raise RuntimeError('Scene-routed detection requires scene_id in every target.')
                scene_id = torch.stack([
                    torch.as_tensor(target['scene_id']).reshape(-1)[0]
                    for target in targets
                ])
            if scene_id is None:
                raise RuntimeError('Scene-routed detection requires scene_id for every sample.')
            scene_id = torch.as_tensor(
                scene_id, device=feats[0].device, dtype=torch.long
            ).reshape(-1)
            if scene_id.shape != (feats[0].shape[0],):
                raise ValueError(f'scene_id must have shape [{feats[0].shape[0]}].')
            if not bool(((scene_id == 0) | (scene_id == 1)).all()):
                raise ValueError('scene_id values must be 0 (indoor/B) or 1 (outdoor/A).')
        else:
            scene_id = None
        # input projection and embedding
        memory, spatial_shapes, projected_features = self._get_encoder_input(feats)
        density_prior_flat = None
        if self.use_density_query and density_prior is not None:
            density_levels = [
                F.interpolate(density_prior, size=shape, mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
                for shape in spatial_shapes
            ]
            density_prior_flat = torch.cat(density_levels, dim=1)

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list, query_diagnostics = \
            self._get_decoder_input(
                memory, spatial_shapes, None, None,
                density_prior_flat, density_prior, projected_features[0], domain_context,
                scene_id
            )

        fixed_query_valid = valid_query_mask
        if fixed_query_valid is None and self.enable_monotonic_query_capacity:
            if density_prior is None:
                raise RuntimeError('Monotonic query capacity requires the P2 density prediction.')
            predicted_count = density_prior.flatten(1).sum(dim=1)
            query_budget = self._monotonic_query_budget(
                predicted_count,
                query_diagnostics['query_candidate_uncertainty'],
                scene_id,
            )
            query_diagnostics['predicted_instance_count'] = predicted_count
            query_diagnostics['active_query_count'] = query_budget
        if fixed_query_valid is None and query_budget is not None:
            fixed_query_valid = (
                torch.arange(self.num_queries, device=memory.device)[None]
                < query_budget[:, None]
            )
        if fixed_query_valid is None:
            fixed_query_valid = torch.ones(
                memory.shape[0], self.num_queries,
                dtype=torch.bool, device=memory.device,
            )
        fixed_query_valid = fixed_query_valid.to(device=memory.device, dtype=torch.bool)
        expected_shape = (memory.shape[0], self.num_queries)
        if fixed_query_valid.shape != expected_shape:
            raise ValueError(
                f'valid_query_mask must have shape {expected_shape}, got {tuple(fixed_query_valid.shape)}.'
            )

        # Build DN only after the normal-query entropy and effective capacity
        # are known. This makes the route's N_gt/capacity/entropy contract an
        # actual batch-time dependency instead of a static configuration hint.
        if self.training and self.num_denoising > 0:
            effective_query_capacity = fixed_query_valid.sum(dim=1).max()
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    num_keypoints=self.num_keypoints,
                    keypoint_noise_scale=self.keypoint_noise_scale,
                    head_tail_swap_ratio=self.head_tail_swap_ratio,
                    no_object_noise_ratio=self.no_object_noise_ratio,
                    effective_query_capacity=effective_query_capacity,
                    matching_entropy=query_diagnostics['query_candidate_uncertainty'],
                    denoising_base_fraction=self.denoising_base_fraction,
                    denoising_entropy_gain=self.denoising_entropy_gain,
                )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
        if denoising_bbox_unact is not None:
            init_ref_points_unact = torch.cat(
                [denoising_bbox_unact, init_ref_points_unact], dim=1,
            )
            init_ref_contents = torch.cat(
                [denoising_logits, init_ref_contents], dim=1,
            )
            query_diagnostics['denoising_query_count'] = init_ref_points_unact.new_full(
                (memory.shape[0],), denoising_bbox_unact.shape[1],
            )
        if denoising_bbox_unact is not None:
            denoising_valid = torch.ones(
                memory.shape[0],
                denoising_bbox_unact.shape[1],
                dtype=torch.bool,
                device=memory.device,
            )
            decoder_query_valid = torch.cat([denoising_valid, fixed_query_valid], dim=1)
        else:
            decoder_query_valid = fixed_query_valid

        # decoder
        out_bboxes, out_logits, out_corners, out_refs, out_hs, out_masks, pre_bboxes, pre_logits, pre_segs = self.decoder(
                spatial_feat,
                init_ref_contents,
                init_ref_points_unact,
                memory,
                spatial_shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                self.pre_bbox_head,
                self.integral,
                self.up,
                self.reg_scale,
                attn_mask=attn_mask,
                dn_meta=dn_meta,
                query_valid_mask=decoder_query_valid,
                scene_id=scene_id)

        endpoint_initial_references = None
        if dn_meta is not None and 'dn_keypoint_refs' in dn_meta:
            normal_refs = dn_meta['dn_keypoint_refs'].new_full(
                (memory.shape[0], self.num_queries, self.num_keypoints, 2), float('nan')
            )
            endpoint_initial_references = torch.cat([
                dn_meta['dn_keypoint_refs'], normal_refs
            ], dim=1)
        out_keypoints, out_visibility, out_uncertainty, out_endpoint_features = self._predict_pose(
            out_hs, out_bboxes, projected_features,
            initial_references=endpoint_initial_references,
        )
        out_quality = None
        if self.use_quality_head:
            if self.use_full_endpoint_route:
                out_quality = torch.stack([
                    quality_head(torch.cat([
                        instance_features,
                        endpoint_features[:, :, 0],
                        endpoint_features[:, :, 1],
                    ], dim=-1)).squeeze(-1)
                    for quality_head, instance_features, endpoint_features in zip(
                        self.dec_quality_head, out_hs, out_endpoint_features,
                    )
                ])
            else:
                out_quality = torch.stack([
                    quality_head(features).squeeze(-1)
                    for quality_head, features in zip(self.dec_quality_head, out_hs)
                ])
        out_track_geometry = (
            self.track_geometry_head(out_hs[-1])
            if self.use_track_geometry_head else None
        )

        s_idx = dn_meta['dn_num_split'] if dn_meta is not None else None

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = self._split(pre_logits, 1, s_idx)
            dn_pre_bboxes, pre_bboxes = self._split(pre_bboxes, 1, s_idx)
            dn_pre_segs, pred_segs = self._split(pre_segs, 1, s_idx)

            dn_out_logits, out_logits = self._split(out_logits, 2, s_idx)
            dn_out_bboxes, out_bboxes = self._split(out_bboxes, 2, s_idx)
            dn_out_masks, out_masks = self._split(out_masks, 2, s_idx)
            dn_out_corners, out_corners =self._split(out_corners, 2, s_idx)
            dn_out_refs, out_refs = self._split(out_refs, 2, s_idx)
            dn_out_keypoints, out_keypoints = self._split(out_keypoints, 2, s_idx)
            dn_out_visibility, out_visibility = self._split(out_visibility, 2, s_idx)
            dn_out_uncertainty, out_uncertainty = self._split(out_uncertainty, 2, s_idx)
            dn_out_quality, out_quality = self._split(out_quality, 2, s_idx)
            dn_out_track_geometry, out_track_geometry = self._split(
                out_track_geometry, 1, s_idx
            )
            dn_query_features, final_query_features = self._split(out_hs[-1], 1, s_idx)
        else:
            final_query_features = out_hs[-1]
            # Without denoising, the decoder already returned the normal
            # pre-mask tensor directly. Keep the same name used by the
            # auxiliary-output assembly below.
            pred_segs = pre_segs

        query_valid = fixed_query_valid
        if query_valid is not None:
            query_valid = query_valid.to(device=out_logits.device, dtype=torch.bool)
            if query_valid.shape != out_logits.shape[1:3]:
                raise ValueError('valid_query_mask must have shape [batch, num_queries].')
            out_logits = out_logits.masked_fill(~query_valid[None, :, :, None], -20.0)

        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                    'pred_masks': out_masks[-1] if out_masks is not None else None, 
                    'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_masks': out_masks[-1] if out_masks is not None else None}

        if out_keypoints is not None:
            out['pred_keypoints'] = out_keypoints[-1]
            out['pred_visibility'] = out_visibility[-1]
        if out_uncertainty is not None:
            out['pred_endpoint_uncertainty'] = out_uncertainty[-1]
        if out_quality is not None:
            out['pred_quality'] = out_quality[-1]
        if self.use_track_geometry_head:
            out['pred_track_geometry'] = out_track_geometry
        if self.training:
            out['pred_query_features'] = final_query_features
        if query_valid is not None:
            out['pred_query_valid'] = query_valid
            query_diagnostics['query_utilization'] = query_valid.float().mean(1)
        out.update(query_diagnostics)

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss2(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], 
                                                     out_refs[:-1], out_masks[:-1] if out_masks is not None else None,
                                                     out_corners[-1], out_logits[-1],
                                                     out_keypoints[:-1] if out_keypoints is not None else None,
                                                     out_visibility[:-1] if out_visibility is not None else None,
                                                     out_uncertainty[:-1] if out_uncertainty is not None else None,
                                                     out_quality[:-1] if out_quality is not None else None,
                                                     query_valid)
            for auxiliary in out['aux_outputs']:
                auxiliary['up'] = self.up
                auxiliary['reg_scale'] = self.reg_scale
            out['enc_aux_outputs'] = self._set_aux_loss(
                enc_topk_logits_list, enc_topk_bboxes_list, query_valid
            )
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes, 'pred_masks': pred_segs}
            if query_valid is not None:
                out['pre_outputs']['pred_query_valid'] = query_valid
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs, dn_out_masks,
                                                        dn_out_corners[-1], dn_out_logits[-1],
                                                        dn_out_keypoints, dn_out_visibility,
                                                        dn_out_uncertainty, dn_out_quality)
                for auxiliary in out['dn_outputs']:
                    auxiliary['up'] = self.up
                    auxiliary['reg_scale'] = self.reg_scale
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes, 'pred_masks': dn_pre_segs}
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_query_valid=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        results = [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class, outputs_coord)]
        if outputs_query_valid is not None:
            for result in results:
                result['pred_query_valid'] = outputs_query_valid
        return results


    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        outputs_masks=None,
        teacher_corners=None,
        teacher_logits=None,
        outputs_keypoints=None,
        outputs_visibility=None,
        outputs_uncertainty=None,
        outputs_quality=None,
        outputs_query_valid=None,
    ):
        if outputs_masks is None:
            res = zip(
                outputs_class,
                outputs_coord,
                outputs_corners,
                outputs_ref
            )
        else:
            res = zip(
                outputs_class,
                outputs_coord,
                outputs_corners,
                outputs_ref,
                outputs_masks
            )

        results = []
        for layer_idx, items in enumerate(res):
            result = {
                'pred_logits': items[0],
                'pred_boxes': items[1],
                'pred_corners': items[2],
                'ref_points': items[3],
                'teacher_corners': teacher_corners,
                'teacher_logits': teacher_logits
            }

            if outputs_masks is not None:
                result['pred_masks'] = items[4]
            if outputs_keypoints is not None:
                result['pred_keypoints'] = outputs_keypoints[layer_idx]
                result['pred_visibility'] = outputs_visibility[layer_idx]
            if outputs_uncertainty is not None:
                result['pred_endpoint_uncertainty'] = outputs_uncertainty[layer_idx]
            if outputs_quality is not None:
                result['pred_quality'] = outputs_quality[layer_idx]
            if outputs_query_valid is not None:
                result['pred_query_valid'] = outputs_query_valid

            results.append(result)

        return results

