"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import inspect
import math
import sys
import time
from contextlib import nullcontext
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA


def _move_target_value(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_target_value(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_target_value(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_target_value(item, device) for item in value)
    return value


def _supported_model_kwargs(model, kwargs):
    """只把模型显式声明支持的路线输入传入，兼容检测-only基线。"""
    unwrapped = model.module if hasattr(model, 'module') else model
    parameters = inspect.signature(unwrapped.forward).parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    supported = {parameter.name for parameter in parameters}
    return {name: value for name, value in kwargs.items() if name in supported}


def _iter_tensor_paths(value, prefix='outputs'):
    """Yield tensor leaves without synchronizing healthy training."""
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensor_paths(item, f'{prefix}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensor_paths(item, f'{prefix}[{index}]')


def _raise_if_nonfinite_loss(loss, loss_dict, outputs, targets, epoch, step):
    """Stop before backward and diagnose the failed forward only on demand."""
    if bool(torch.isfinite(loss).item()):
        return

    nonfinite_losses = [
        name for name, value in loss_dict.items()
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item())
    ]
    nonfinite_outputs = []
    for path, value in _iter_tensor_paths(outputs):
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            count = int((~torch.isfinite(value)).sum().item())
            nonfinite_outputs.append(
                f'{path}:dtype={value.dtype},shape={tuple(value.shape)},count={count}'
            )
    image_ids = [
        target.get('image_id').detach().cpu().reshape(-1).tolist()
        if torch.is_tensor(target.get('image_id')) else target.get('image_id')
        for target in targets
    ]
    raise FloatingPointError(
        'Non-finite loss detected before backward: '
        f'epoch={epoch}, step={step}, image_ids={image_ids}, '
        f'losses={nonfinite_losses}, outputs={nonfinite_outputs}'
    )


@torch.no_grad()
def _explicit_accumulated_gradient_sync(model):
    """Average accumulated gradients after every rank has seen both domains."""
    if not dist_utils.is_dist_available_and_initialized():
        used = sum(parameter.grad is not None for parameter in model.parameters())
        total = sum(parameter.requires_grad for parameter in model.parameters())
        return used, total

    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        return 0, 0

    device = named_parameters[0][1].device
    local_mask = torch.tensor(
        [parameter.grad is not None for _, parameter in named_parameters],
        dtype=torch.int32,
        device=device,
    )
    mask_min = local_mask.clone()
    mask_max = local_mask.clone()
    torch.distributed.all_reduce(mask_min, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(mask_max, op=torch.distributed.ReduceOp.MAX)
    if not torch.equal(mask_min, mask_max):
        mismatched = torch.nonzero(mask_min != mask_max, as_tuple=False).flatten().tolist()
        names = [named_parameters[index][0] for index in mismatched[:16]]
        raise RuntimeError(
            'Explicit accumulated gradient sync requires identical used-parameter '
            f'masks on every rank; mismatched={names}, total={len(mismatched)}.'
        )

    used_parameters = [
        parameter for (_, parameter), is_used in zip(named_parameters, local_mask.tolist())
        if is_used
    ]
    buckets = {}
    for parameter in used_parameters:
        key = (parameter.grad.device, parameter.grad.dtype)
        buckets.setdefault(key, []).append(parameter)

    world_size = torch.distributed.get_world_size()
    for parameters in buckets.values():
        flat_gradient = torch.cat([
            parameter.grad.detach().reshape(-1) for parameter in parameters
        ])
        torch.distributed.all_reduce(flat_gradient)
        flat_gradient.div_(world_size)
        offset = 0
        for parameter in parameters:
            count = parameter.numel()
            parameter.grad.copy_(flat_gradient[offset:offset + count].view_as(parameter))
            offset += count
    return len(used_parameters), len(named_parameters)


def _apply_backbone_train_modes(model):
    """Keep frozen trunk groups in eval mode during gradual unfreezing."""
    bare_model = dist_utils.de_parallel(model)
    adapter = getattr(bare_model, 'backbone', None)
    if adapter is None:
        return
    trunk = getattr(adapter, 'backbone', adapter)
    parameters = list(trunk.parameters())
    if not parameters or all(parameter.requires_grad for parameter in parameters):
        return
    trunk.eval()
    blocks = getattr(trunk, 'blocks', None)
    if isinstance(blocks, torch.nn.ModuleList):
        for block in blocks:
            if any(parameter.requires_grad for parameter in block.parameters()):
                block.train()
        for name, child in trunk.named_children():
            if name != 'blocks' and any(
                parameter.requires_grad for parameter in child.parameters()
            ):
                child.train()
    else:
        for child in trunk.children():
            if any(parameter.requires_grad for parameter in child.parameters()):
                child.train()


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.3e}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler = kwargs.get('lr_warmup_scheduler', None)
    grad_accum_steps = max(1, int(kwargs.get('grad_accum_steps', 1)))
    query_max_norm = float(kwargs.get('query_max_norm', 0.0))
    pseudo_labeler = kwargs.get('pseudo_labeler')
    conflict_interval = int(kwargs.get('log_gradient_conflicts_every', 0))
    enable_ddp_no_sync = bool(kwargs.get('enable_ddp_no_sync', False))
    enable_explicit_gradient_sync = bool(
        kwargs.get('enable_explicit_accumulated_gradient_sync', False)
    )
    if enable_explicit_gradient_sync and not enable_ddp_no_sync:
        raise ValueError(
            'Explicit accumulated gradient sync requires enable_ddp_no_sync=True.'
        )
    ddp_active = (
        dist_utils.is_dist_available_and_initialized()
        and dist_utils.get_world_size() > 1
    )
    if ddp_active and conflict_interval > 0:
        # torch.autograd.grad fires the DDP parameter hooks before the real
        # backward pass, so the same parameter is marked ready twice.  This
        # diagnostic is logging-only; disabling it under DDP preserves the
        # exact training loss and gradient while avoiding hook corruption.
        if dist_utils.is_main_process():
            print('Gradient-conflict diagnostics disabled under DDP.')
        conflict_interval = 0
    require_equal_domain_updates = bool(
        kwargs.get('require_equal_domain_updates', False)
    )
    _apply_backbone_train_modes(model)
    query_parameters = [
        parameter for name, parameter in model.named_parameters()
        if any(token in name for token in ('query', 'endpoint', 'pattern', 'density_peak'))
    ]
    shared_monitor_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(token in name for token in ('backbone', 'encoder'))
    ][:8]

    def clip_gradients():
        query_norm = torch.zeros((), device=device)
        if query_max_norm > 0:
            active_query_parameters = [
                parameter for parameter in query_parameters
                if parameter.grad is not None
            ]
            if active_query_parameters:
                query_norm = torch.nn.utils.clip_grad_norm_(
                    active_query_parameters, query_max_norm
                )
        full_norm = torch.zeros((), device=device)
        if max_norm > 0:
            full_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return query_norm, full_norm

    def domain_monitor_losses(outputs, targets):
        logits = outputs.get('pred_domain_logits')
        if logits is None:
            return {}
        domain_targets = torch.cat([
            target.get(
                'domain_id', torch.zeros(1, dtype=torch.long, device=logits.device)
            ).reshape(-1)[:1]
            for target in targets
        ]).long()
        monitored = {}
        for domain_id, domain_name in ((0, 'rgb_domain'), (1, 'ir_domain')):
            mask = domain_targets == domain_id
            if mask.any():
                monitored[domain_name] = F.cross_entropy(logits[mask], domain_targets[mask])
        return monitored

    def gradient_conflicts(loss_dict, global_step, monitor_losses=None):
        if conflict_interval <= 0 or global_step % conflict_interval:
            return {}
        groups = {
            'det': ('mal', 'focal', 'bbox', 'giou', 'fgl', 'ddf'),
            'pose': ('keypoint', 'pose', 'direction', 'visibility'),
            'density': ('density',),
            'domain': ('domain', 'prototype'),
            'track': ('track',),
        }
        if not shared_monitor_parameters:
            return {}
        gradients = {}
        for group_name, tokens in groups.items():
            group_losses = [
                value for name, value in loss_dict.items()
                if any(token in name for token in tokens)
            ]
            if not group_losses:
                continue
            values = torch.autograd.grad(
                sum(group_losses), shared_monitor_parameters,
                retain_graph=True, allow_unused=True,
            )
            gradients[group_name] = values
        for group_name, monitor_loss in (monitor_losses or {}).items():
            gradients[group_name] = torch.autograd.grad(
                monitor_loss, shared_monitor_parameters,
                retain_graph=True, allow_unused=True,
            )
        metrics = {}
        names = list(gradients)
        for name, values in gradients.items():
            norm = torch.zeros((), device=device)
            for value in values:
                if value is not None:
                    norm += value.square().sum()
            metrics[f'grad_norm_{name}'] = float(norm.sqrt().detach())
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1:]:
                dot = torch.zeros((), device=device)
                left_norm = torch.zeros((), device=device)
                right_norm = torch.zeros((), device=device)
                for left, right in zip(gradients[left_name], gradients[right_name]):
                    if left is None or right is None:
                        continue
                    dot += (left * right).sum()
                    left_norm += left.square().sum()
                    right_norm += right.square().sum()
                cosine = dot / (left_norm.sqrt() * right_norm.sqrt()).clamp_min(1e-12)
                metrics[f'grad_cos_{left_name}_{right_name}'] = float(cosine.detach())
        return metrics

    data_iterations_per_epoch = len(data_loader)
    scheduler_reference_batch_size = kwargs.get('scheduler_reference_batch_size')
    if scheduler_reference_batch_size:
        scheduler_reference_batch_size = int(scheduler_reference_batch_size)
        sampler_size = len(data_loader.sampler)
        physical_batch_size = int(data_loader.batch_size)
        world_size = dist_utils.get_world_size()
        scheduler_epoch_start = int(lr_scheduler.current_iter)

        def canonical_iteration_after(loader_steps):
            local_samples = min(
                int(loader_steps) * physical_batch_size,
                sampler_size,
            )
            global_samples = local_samples * world_size
            return scheduler_epoch_start + (
                global_samples // scheduler_reference_batch_size
            )
    else:
        scheduler_epoch_start = epoch * data_iterations_per_epoch

        def canonical_iteration_after(loader_steps):
            return scheduler_epoch_start + int(loader_steps)
    optimizer.zero_grad()
    accumulated_domain_views = torch.zeros(2, dtype=torch.long, device=device)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = [
            {key: _move_target_value(value, device) for key, value in target.items()}
            for target in targets
        ]
        pseudo_stats = None
        if pseudo_labeler is not None:
            if ema is None:
                raise RuntimeError('Online pseudo labeling requires an EMA teacher.')
            targets, pseudo_stats = pseudo_labeler.generate(ema.module, samples, targets)
            samples = pseudo_labeler.strong_view(samples, targets)
        global_step = canonical_iteration_after(i)
        update_step = i // grad_accum_steps
        group_start = update_step * grad_accum_steps
        group_size = min(grad_accum_steps, len(data_loader) - group_start)
        should_step = (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader)
        optimizer_step_performed = False
        amp_step_skipped = 0.0
        amp_fp32_fallback = 0.0
        metas = dict(
            epoch=epoch,
            step=i,
            global_step=global_step,
            epoch_step=(
                int(kwargs.get('lr_schedule_iterations_per_epoch'))
                if kwargs.get('lr_schedule_iterations_per_epoch')
                else len(data_loader)
            ),
        )

        update_domain_views = None
        if require_equal_domain_updates:
            missing_domain = [
                index for index, target in enumerate(targets)
                if 'domain_id' not in target
            ]
            if missing_domain:
                raise RuntimeError(
                    'Equal-domain update contract requires domain_id on every image; '
                    f'missing local target indices={missing_domain}.'
                )
            domain_ids = torch.stack([
                target['domain_id'].reshape(-1)[0] for target in targets
            ]).long()
            if bool(((domain_ids < 0) | (domain_ids > 1)).any()):
                raise RuntimeError('BeePoseTrack-E domain_id must be exactly RGB=0 or IR=1.')
            accumulated_domain_views += torch.bincount(domain_ids, minlength=2)[:2]
            if should_step:
                update_domain_views = accumulated_domain_views.clone()
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.all_reduce(update_domain_views)
                if (
                    int(update_domain_views[0]) == 0
                    or int(update_domain_views[1]) == 0
                    or int(update_domain_views[0]) != int(update_domain_views[1])
                ):
                    raise RuntimeError(
                        'Each optimizer update must contain equal non-zero RGB/IR views; '
                        f'global counts={update_domain_views.tolist()}, epoch={epoch}, '
                        f'update_step={update_step}.'
                    )
                accumulated_domain_views.zero_()

        if self_lr_scheduler and i == group_start:
            # Set the LR before accumulating the gradients for this optimizer
            # update.  The schedule clock remains the number of consumed data
            # iterations, so grad_accum_steps does not alter warm-up duration.
            group_end_iteration = canonical_iteration_after(
                group_start + group_size
            )
            optimizer = lr_scheduler.step(group_end_iteration, optimizer)

        sync_context = (
            model.no_sync()
            if enable_ddp_no_sync
            and (enable_explicit_gradient_sync or not should_step)
            and hasattr(model, 'no_sync')
            else nullcontext()
        )
        # DDP requires both forward and backward inside no_sync. The final
        # micro-batch synchronizes the accumulated gradient exactly once.
        with sync_context:
            if scaler is not None:
                amp_forward_error = None
                try:
                    with torch.autocast(device_type=device.type, cache_enabled=True):
                        outputs = model(samples, targets=targets)
                except FloatingPointError as error:
                    amp_forward_error = error
                    outputs = None

                if (
                    amp_forward_error is not None
                    or not torch.isfinite(outputs['pred_boxes']).all()
                ):
                    amp_fp32_fallback = 1.0
                    image_ids = [
                        target.get('image_id').detach().cpu().reshape(-1).tolist()
                        if torch.is_tensor(target.get('image_id')) else target.get('image_id')
                        for target in targets
                    ]
                    reason = (
                        str(amp_forward_error)
                        if amp_forward_error is not None
                        else 'non-finite pred_boxes'
                    )
                    print(
                        f"AMP forward failed finite-value gate at epoch={epoch}, "
                        f"step={i}, image_ids={image_ids}, reason={reason}; "
                        "retrying this batch in FP32."
                    )
                    # Do not keep the failed AMP graph alive while constructing
                    # the FP32 graph; at 1280px both graphs together exceed 24GB.
                    if outputs is not None:
                        del outputs
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    try:
                        with torch.autocast(device_type=device.type, enabled=False):
                            outputs = model(samples, targets=targets)
                    except FloatingPointError as error:
                        raise FloatingPointError(
                            f"FP32 retry failed at epoch={epoch}, step={i}, "
                            f"image_ids={image_ids}: {error}"
                        ) from error
                    if not torch.isfinite(outputs['pred_boxes']).all():
                        raise FloatingPointError(
                            f"FP32 retry still produced non-finite boxes at "
                            f"epoch={epoch}, step={i}, image_ids={image_ids}."
                        )

                with torch.autocast(device_type=device.type, enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)

                loss = sum(loss_dict.values())
                _raise_if_nonfinite_loss(loss, loss_dict, outputs, targets, epoch, i)
                conflict_stats = gradient_conflicts(
                    loss_dict, global_step, domain_monitor_losses(outputs, targets)
                )
                scaler.scale(loss / group_size).backward()
            else:
                outputs = model(samples, targets=targets)
                loss_dict = criterion(outputs, targets, **metas)

                loss : torch.Tensor = sum(loss_dict.values())
                _raise_if_nonfinite_loss(loss, loss_dict, outputs, targets, epoch, i)
                conflict_stats = gradient_conflicts(
                    loss_dict, global_step, domain_monitor_losses(outputs, targets)
                )
                (loss / group_size).backward()

        if should_step:
            if scaler is not None:
                scale_before = float(scaler.get_scale())
                if max_norm > 0 or query_max_norm > 0 or enable_explicit_gradient_sync:
                    scaler.unscale_(optimizer)
                nonfinite_gradients = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ]
                if nonfinite_gradients:
                    raise FloatingPointError(
                        'Non-finite gradients detected before optimizer.step: '
                        f'epoch={epoch}, update_step={update_step}, '
                        f'parameters={nonfinite_gradients[:16]}'
                    )
                if enable_explicit_gradient_sync:
                    explicit_used, explicit_total = _explicit_accumulated_gradient_sync(model)
                    if update_step == 0 and dist_utils.is_main_process():
                        print(
                            'Explicit accumulated gradient sync passed: '
                            f'used={explicit_used}/{explicit_total}, epoch={epoch}.'
                        )
                if max_norm > 0 or query_max_norm > 0:
                    query_grad_norm, full_grad_norm = clip_gradients()
                else:
                    query_grad_norm = torch.zeros((), device=device)
                    full_grad_norm = torch.zeros((), device=device)

                scaler.step(optimizer)
                scaler.update()
                # GradScaler skips optimizer.step when it observes non-finite
                # gradients. EMA and LR scheduling follow real updates only.
                optimizer_step_performed = float(scaler.get_scale()) >= scale_before
                amp_step_skipped = float(not optimizer_step_performed)
            else:
                if enable_explicit_gradient_sync:
                    explicit_used, explicit_total = _explicit_accumulated_gradient_sync(model)
                    if update_step == 0 and dist_utils.is_main_process():
                        print(
                            'Explicit accumulated gradient sync passed: '
                            f'used={explicit_used}/{explicit_total}, epoch={epoch}.'
                        )
                query_grad_norm, full_grad_norm = clip_gradients()
                optimizer.step()
                optimizer_step_performed = True
            optimizer.zero_grad()

        if optimizer_step_performed:
            if ema is not None:
                ema.update(model)

            if not self_lr_scheduler and lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(amp_fp32_fallback=amp_fp32_fallback)
        if pseudo_stats is not None:
            metric_logger.update(
                pseudo_instances=float(pseudo_stats['pseudo_instances']),
                ema_teacher_updates=float(ema.updates),
            )
        if conflict_stats:
            metric_logger.update(**conflict_stats)
        if should_step:
            metric_logger.update(
                query_grad_norm=float(query_grad_norm),
                full_grad_norm=float(full_grad_norm),
                amp_step_skipped=amp_step_skipped,
            )
            if update_domain_views is not None:
                metric_logger.update(
                    rgb_views_per_update=float(update_domain_views[0]),
                    ir_views_per_update=float(update_domain_views[1]),
                )

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor,
             data_loader, coco_evaluator: CocoEvaluator, device,
             evaluation_profile='full', inference_calibration_contract=None):
    profiles = {'detection', 'pose', 'joint', 'density', 'temporal', 'full'}
    if evaluation_profile not in profiles:
        raise ValueError(f'unsupported evaluation profile: {evaluation_profile}')
    enable_pose = evaluation_profile in {'pose', 'joint', 'density', 'temporal', 'full'}
    enable_density = evaluation_profile in {'density', 'joint', 'temporal', 'full'}
    enable_tracking = evaluation_profile in {'temporal', 'full'}
    enable_latency = evaluation_profile == 'full'
    density_rechecker = None
    tracker_config = None
    if inference_calibration_contract is not None:
        from ..edgecrafter.inference import DensityResidualRechecker
        from ..tracking import TrackerConfig
        if not hasattr(postprocessor, 'apply_calibration_contract'):
            raise TypeError('Calibrated route evaluation requires BeePoseTrack-E PostProcessor.')
        postprocessor.apply_calibration_contract(inference_calibration_contract)
        bare_model = dist_utils.de_parallel(model)
        decoder = getattr(bare_model, 'decoder', None)
        if decoder is None or not hasattr(decoder, 'apply_query_capacity_contract'):
            raise TypeError('Calibrated route evaluation requires ECTransformer capacity mapping.')
        decoder.apply_query_capacity_contract(inference_calibration_contract)
        density_rechecker = DensityResidualRechecker.from_calibration_contract(
            inference_calibration_contract,
            input_size=int(getattr(postprocessor, 'eval_spatial_size', 1280) or 1280),
        )
        tracker_config = TrackerConfig.from_calibration_contract(
            inference_calibration_contract
        )
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    from ..edgecrafter.pose_metrics import BeeDetectionMetrics, BeeQueryMetrics
    pose_metrics = {}
    if enable_pose:
        from ..edgecrafter.pose_metrics import BeePoseMetrics
        pose_metrics = {
        'all': BeePoseMetrics(),
        'rgb': BeePoseMetrics(),
        'ir': BeePoseMetrics(),
        }
    detection_metrics = {
        name: BeeDetectionMetrics() for name in ('all', 'rgb', 'ir')
    }
    video_detection_metrics = {}
    query_metrics = {
        name: BeeQueryMetrics() for name in ('all', 'rgb', 'ir')
    }
    tracking_metrics = {}
    if enable_tracking:
        from ..tracking import PoseMotionTracker, TrackerConfig, TrackingMetricsAccumulator
        tracking_metrics = {
            name: TrackingMetricsAccumulator() for name in ('all', 'rgb', 'ir')
        }
    tracking_frames = []
    density_metrics = {
        name: {'absolute_error': 0.0, 'squared_error': 0.0, 'frames': 0, 'gt_count': 0.0}
        for name in ('all', 'rgb', 'ir')
    }
    inference_timings = []
    query_diagnostics = {
        name: {'sum': 0.0, 'count': 0}
        for name in (
            'query_utilization', 'query_spatial_coverage', 'query_collapse',
            'pattern_entropy', 'domain_rgb_weight', 'domain_ir_weight',
            'domain_router_entropy', 'domain_prototype_distance',
            'shared_bee_prototype_distance', 'domain_adapter_response',
            'domain_bn_mean_distance', 'domain_bn_logvar_distance',
        )
    }

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [
            {key: _move_target_value(value, device) for key, value in target.items()}
            for target in targets
        ]

        if enable_latency and device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        temporal_mask = None
        if samples.ndim == 5 and all('temporal_valid_mask' in target for target in targets):
            temporal_mask = torch.stack([
                target['temporal_valid_mask'] for target in targets
            ], dim=0)
        route_kwargs = {}
        if all('domain_id' in target for target in targets):
            route_kwargs['domain_id'] = torch.stack([
                target['domain_id'].reshape(-1)[0] for target in targets
            ])
        if all('scene_id' in target for target in targets):
            route_kwargs['scene_id'] = torch.stack([
                target['scene_id'].reshape(-1)[0] for target in targets
            ])
        if temporal_mask is not None:
            route_kwargs['temporal_valid_mask'] = temporal_mask
        if all('stabilization_theta' in target for target in targets):
            route_kwargs['stabilization_theta'] = torch.stack([
                target['stabilization_theta'] for target in targets
            ])
        outputs = model(samples, **_supported_model_kwargs(model, route_kwargs))
        for name, accumulator in query_diagnostics.items():
            value = outputs.get(name)
            if value is not None:
                accumulator['sum'] += float(value.float().sum().item())
                accumulator['count'] += value.numel()

        has_letterbox = all('letterbox_scale' in target for target in targets)
        target_sizes = torch.stack([
            target['input_size'] if has_letterbox else target['orig_size']
            for target in targets
        ], dim=0)

        results = postprocessor(outputs, target_sizes)
        if density_rechecker is not None:
            calibrated_results = []
            for batch_index, (result, target) in enumerate(zip(results, targets)):
                current_image = (
                    samples[batch_index, -1]
                    if samples.ndim == 5 else samples[batch_index]
                )
                domain_id = int(target.get(
                    'domain_id', torch.tensor([0], device=device),
                ).reshape(-1)[0].item())

                def infer_local(crop, local_domain_id):
                    if samples.ndim == 5:
                        local_samples = crop[None, None].repeat(
                            1, samples.shape[1], 1, 1, 1,
                        )
                        local_temporal_mask = torch.zeros(
                            1, samples.shape[1], dtype=torch.bool, device=crop.device,
                        )
                        local_temporal_mask[:, -1] = True
                    else:
                        local_samples = crop[None]
                        local_temporal_mask = None
                    local_kwargs = {
                        'domain_id': torch.tensor(
                            [local_domain_id], dtype=torch.long, device=crop.device,
                        ),
                    }
                    if local_temporal_mask is not None:
                        local_kwargs['temporal_valid_mask'] = local_temporal_mask
                    local_outputs = model(
                        local_samples,
                        **_supported_model_kwargs(model, local_kwargs),
                    )
                    local_size = torch.tensor(
                        [[density_rechecker.input_size, density_rechecker.input_size]],
                        dtype=target_sizes.dtype, device=target_sizes.device,
                    )
                    return postprocessor(local_outputs, local_size)[0]

                calibrated_results.append(density_rechecker(
                    current_image, result, infer_local, domain_id,
                ))
            results = calibrated_results
        if has_letterbox:
            for result, target in zip(results, targets):
                scale = target['letterbox_scale']
                pad = target['letterbox_pad']
                boxes = result['boxes']
                boxes[:, 0::2] = (boxes[:, 0::2] - pad[0]) / scale
                boxes[:, 1::2] = (boxes[:, 1::2] - pad[1]) / scale
                original_w, original_h = target['orig_size']
                boxes[:, 0::2].clamp_(0, original_w)
                boxes[:, 1::2].clamp_(0, original_h)
                if 'keypoints' in result:
                    keypoints = result['keypoints']
                    keypoints[..., 0] = (keypoints[..., 0] - pad[0]) / scale
                    keypoints[..., 1] = (keypoints[..., 1] - pad[1]) / scale
                    keypoints[..., 0].clamp_(0, original_w)
                    keypoints[..., 1].clamp_(0, original_h)
        if enable_latency and device.type == 'cuda':
            torch.cuda.synchronize(device)
        if enable_latency:
            inference_timings.append(
                (time.perf_counter() - started) * 1000.0 / max(len(results), 1)
            )

        for batch_index, (result, target) in enumerate(zip(results, targets)):
            domain_name = 'ir' if int(target.get('domain_id', torch.tensor([0], device=device)).item()) == 1 else 'rgb'
            if enable_pose:
                pose_metrics['all'].update(result, target)
                pose_metrics[domain_name].update(result, target)
            detection_metrics['all'].update(result, target)
            detection_metrics[domain_name].update(result, target)
            sequence_id = int(target.get('sequence_id', target['image_id']).item())
            video_key = f'{domain_name}:{sequence_id}'
            video_detection_metrics.setdefault(
                video_key, BeeDetectionMetrics(),
            ).update(result, target)
            valid_query_mask = outputs.get('pred_query_valid')
            if valid_query_mask is None:
                valid_query_mask = torch.ones(
                    outputs['pred_logits'].shape[1], dtype=torch.bool, device=device
                )
            else:
                valid_query_mask = valid_query_mask[batch_index]
            query_metrics['all'].update(result, target, valid_query_mask)
            query_metrics[domain_name].update(result, target, valid_query_mask)

            track_ids = target.get('track_id')
            track_mask = target.get('track_mask')
            track_supervised = bool(target.get(
                'track_supervised', torch.tensor([False], device=device)
            ).item())
            if enable_tracking and track_supervised and track_ids is not None and track_mask is not None:
                sequence_id = int(target.get('sequence_id', target['image_id']).item())
                frame_id = int(target.get('frame_id', target['image_id']).item())
                sensor_id = int(target.get('sensor_id', torch.tensor([0], device=device)).item())
                detections = {
                    'boxes': result['boxes'].detach().cpu(),
                    'scores': result['scores'].detach().cpu(),
                    'quality': result.get('quality_score', result['scores']).detach().cpu(),
                    'query_indices': result.get(
                        'query_indices', torch.arange(len(result['boxes']), device=device)
                    ).detach().cpu(),
                }
                if 'keypoints' in result:
                    detections['keypoints'] = result['keypoints'].detach().cpu()
                valid_tracks = track_mask.bool()
                tracking_target = {
                    'boxes': target.get('orig_boxes', target['boxes'])[valid_tracks].detach().cpu(),
                    'track_ids': track_ids[valid_tracks].detach().cpu(),
                    'frame_id': frame_id,
                }
                if 'orig_keypoints' in target:
                    tracking_target['keypoints'] = (
                        target['orig_keypoints'][valid_tracks][..., :2].detach().cpu()
                    )
                tracking_frames.append({
                    'domain': domain_name,
                    'sensor_id': sensor_id,
                    'sequence_id': sequence_id,
                    'frame_id': frame_id,
                    'image_id': int(target['image_id'].item()),
                    'detections': detections,
                    'target': tracking_target,
                })
            if enable_density and 'density_count' in result:
                predicted_count = float(result['density_count'].item())
                gt_count = float(len(target['boxes']))
                error = predicted_count - gt_count
                for density_domain in ('all', domain_name):
                    density_metrics[density_domain]['absolute_error'] += abs(error)
                    density_metrics[density_domain]['squared_error'] += error * error
                    density_metrics[density_domain]['frames'] += 1
                    density_metrics[density_domain]['gt_count'] += gt_count

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    def merge_metric_group(group):
        gathered = dist_utils.all_gather({
            name: accumulator.state_dict() for name, accumulator in group.items()
        })
        merged = {}
        for name, accumulator in group.items():
            combined = accumulator.empty_copy()
            for rank_state in gathered:
                combined.merge_state(rank_state[name])
            merged[name] = combined
        return merged

    if enable_pose:
        pose_metrics = merge_metric_group(pose_metrics)
    detection_metrics = merge_metric_group(detection_metrics)
    query_metrics = merge_metric_group(query_metrics)
    gathered_video_metrics = dist_utils.all_gather({
        name: accumulator.state_dict()
        for name, accumulator in video_detection_metrics.items()
    })
    video_names = sorted({
        name for rank_state in gathered_video_metrics for name in rank_state
    })
    merged_video_metrics = {}
    for name in video_names:
        combined = BeeDetectionMetrics()
        for rank_state in gathered_video_metrics:
            if name in rank_state:
                combined.merge_state(rank_state[name])
        merged_video_metrics[name] = combined
    video_detection_metrics = merged_video_metrics

    if enable_density:
        gathered_density = dist_utils.all_gather(density_metrics)
        density_metrics = {
            name: {'absolute_error': 0.0, 'squared_error': 0.0, 'frames': 0, 'gt_count': 0.0}
            for name in ('all', 'rgb', 'ir')
        }
        for rank_values in gathered_density:
            for domain_name, values in rank_values.items():
                for metric_name, value in values.items():
                    density_metrics[domain_name][metric_name] += value

    gathered_diagnostics = dist_utils.all_gather(query_diagnostics)
    query_diagnostics = {
        name: {'sum': 0.0, 'count': 0} for name in query_diagnostics
    }
    for rank_values in gathered_diagnostics:
        for name, values in rank_values.items():
            query_diagnostics[name]['sum'] += values['sum']
            query_diagnostics[name]['count'] += values['count']
    inference_timings = [
        value for rank_values in dist_utils.all_gather(inference_timings)
        for value in rank_values
    ]

    gathered_frames = (
        [frame for rank_frames in dist_utils.all_gather(tracking_frames)
         for frame in rank_frames]
        if enable_tracking else []
    )
    unique_frames = {}
    for frame in gathered_frames:
        key = (
            frame['domain'], frame['sensor_id'], frame['sequence_id'],
            frame['frame_id'], frame['image_id'],
        )
        unique_frames.setdefault(key, frame)
    fixed_trackers = {}
    tracker_last_frames = {}
    for frame in sorted(unique_frames.values(), key=lambda item: (
        item['domain'], item['sensor_id'], item['sequence_id'],
        item['frame_id'], item['image_id'],
    )):
        tracker_key = (frame['domain'], frame['sensor_id'], frame['sequence_id'])
        previous_frame = tracker_last_frames.get(tracker_key)
        if previous_frame is not None and frame['frame_id'] <= previous_frame:
            raise ValueError(
                'Fixed tracking evaluation requires strictly increasing frame_id '
                f"within each sequence; got {frame['frame_id']} after {previous_frame} "
                f'for {tracker_key}.'
            )
        tracker_last_frames[tracker_key] = frame['frame_id']
        tracker = fixed_trackers.setdefault(
            tracker_key, PoseMotionTracker(tracker_config or TrackerConfig())
        )
        tracked = tracker.update(frame['detections'], frame['frame_id'])
        frame['target']['sequence_id'] = (frame['sensor_id'], frame['sequence_id'])
        tracking_metrics['all'].update(tracked, frame['target'])
        tracking_metrics[frame['domain']].update(tracked, frame['target'])

    # gather the standard metric logger and COCO evaluator state.
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator.labels is not None:
        
        import numpy as np
        from tabulate import tabulate
        
        res_per_type = {}
        headers = ['class']

        for iou_type in coco_evaluator.iou_types:
            if iou_type not in coco_evaluator.coco_eval:
                continue
            
            precisions = coco_evaluator.coco_eval[iou_type].eval['precision']
            ap = np.mean(precisions[..., 0, -1], axis=(0, 1)) * 100
            ap_50 = np.mean(precisions[0, :, :, 0, -1], axis=0) * 100
            
            prefix = 'bbox' if iou_type == 'bbox' else 'segm'
            headers.extend([f'{prefix}-AP', f'{prefix}-AP50'])
            res_per_type[iou_type] = (ap, ap_50)

        # Construct rows by merging metrics for each class
        table_data = []
        for k, name in enumerate(coco_evaluator.labels):
            row = [name]
            for iou_type in coco_evaluator.iou_types:
                if iou_type in res_per_type:
                    ap, ap_50 = res_per_type[iou_type]
                    row.extend([f'{ap[k]:.2f}', f'{ap_50[k]:.2f}'])
            table_data.append(row)

        print(f"\n### Class-wise Evaluation Metrics ###")
        print(tabulate(table_data, headers=headers, tablefmt='pretty'))
        
    
    if coco_evaluator is not None:
        if 'segm' in iou_types:
            stats['coco_eval_mask'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        elif 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

    if enable_pose and pose_metrics['all'].gt_pose_instances > 0:
        print('\n### Bee joint pose metrics (detections matched at IoU>=0.50) ###')
        for domain_name, accumulator in pose_metrics.items():
            domain_metrics = accumulator.summarize()
            print(f'{domain_name}: {domain_metrics}')
            for metric_name, value in domain_metrics.items():
                stats[f'pose_{domain_name}_{metric_name}'] = value

    if detection_metrics['all'].gt_instances > 0:
        print('\n### Bee detection metrics by domain ###')
        for domain_name, accumulator in detection_metrics.items():
            domain_metrics = accumulator.summarize()
            print(f'{domain_name}: {domain_metrics}')
            for metric_name, value in domain_metrics.items():
                stats[f'detection_{domain_name}_{metric_name}'] = value
            video_recalls = {}
            for video_name, accumulator in video_detection_metrics.items():
                values = accumulator.summarize()
                recall = values.get('recall')
                if recall is None:
                    continue
                video_recalls[video_name] = recall
                stats[f'detection_video_{video_name}_recall'] = recall
        if video_recalls:
            weakest_video = min(video_recalls, key=video_recalls.get)
            stats['detection_weakest_video_recall'] = video_recalls[weakest_video]

    print('\n### Bee query capacity and matching diagnostics ###')
    for domain_name, accumulator in query_metrics.items():
        domain_metrics = accumulator.summarize()
        print(f'{domain_name}: {domain_metrics}')
        for metric_name, value in domain_metrics.items():
            stats[f'query_{domain_name}_{metric_name}'] = value
    stats['query_utilization'] = stats.get('query_all_query_utilization', 0.0)

    if enable_tracking and tracking_metrics['all'].gt_detections > 0:
        print('\n### Fixed-parameter pose-motion tracking metrics ###')
        for domain_name, accumulator in tracking_metrics.items():
            if accumulator.gt_detections == 0:
                continue
            domain_metrics = accumulator.summarize()
            print(f'{domain_name}: {domain_metrics}')
            for metric_name, value in domain_metrics.items():
                if isinstance(value, (int, float)):
                    stats[f'tracking_{domain_name}_{metric_name}'] = value

    if enable_density and density_metrics['all']['frames'] > 0:
        print('\n### Bee density/count metrics ###')
        for domain_name, values in density_metrics.items():
            frame_count = max(values['frames'], 1)
            mae = values['absolute_error'] / frame_count
            rmse = (values['squared_error'] / frame_count) ** 0.5
            mean_gt = values['gt_count'] / frame_count
            relative_mae = mae / max(mean_gt, 1e-6)
            domain_metrics = {
                'frames': values['frames'],
                'mae': mae,
                'rmse': rmse,
                'mean_gt_count': mean_gt,
                'relative_mae': relative_mae,
            }
            print(f'{domain_name}: {domain_metrics}')
            for metric_name, value in domain_metrics.items():
                stats[f'density_{domain_name}_{metric_name}'] = value

    if inference_timings:
        ordered = sorted(inference_timings)
        p99_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.99) - 1)
        stats['latency_mean_ms'] = sum(ordered) / len(ordered)
        stats['latency_p99_ms'] = ordered[p99_index]
    for name, accumulator in query_diagnostics.items():
        if accumulator['count']:
            stats[name] = accumulator['sum'] / accumulator['count']

    return stats, coco_evaluator
