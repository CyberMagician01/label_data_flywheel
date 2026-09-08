"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

import torch
from calflops import calculate_flops


def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:
    base_size = cfg.yaml_cfg["eval_spatial_size"]
    input_shape = (1, 3, *base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()

    profile_kwargs = {}
    has_domain_route = getattr(model_for_info, 'enable_explicit_domain', False)
    has_scene_route = getattr(
        getattr(model_for_info, 'decoder', None),
        'enable_scene_detection_heads',
        False,
    )
    if has_domain_route or has_scene_route:
        # Route models deliberately reject inputs without an explicit RGB/IR
        # domain. Profiling is inference-only, so use a valid RGB route rather
        # than weakening the model's runtime contract.
        parameter = next(model_for_info.parameters())
        profile_input = torch.empty(
            input_shape, dtype=parameter.dtype, device=parameter.device,
        )
        if has_domain_route:
            profile_kwargs['domain_id'] = torch.zeros(
                input_shape[0], dtype=torch.long, device=parameter.device,
            )
        if has_scene_route:
            profile_kwargs['scene_id'] = torch.ones(
                input_shape[0], dtype=torch.long, device=parameter.device,
            )
        flops, macs, _ = calculate_flops(
            model=model_for_info,
            args=[profile_input],
            kwargs=profile_kwargs,
            output_as_string=True,
            output_precision=4,
            print_detailed=False,
        )
    else:
        flops, macs, _ = calculate_flops(
            model=model_for_info,
            input_shape=input_shape,
            output_as_string=True,
            output_precision=4,
            print_detailed=False,
        )
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
