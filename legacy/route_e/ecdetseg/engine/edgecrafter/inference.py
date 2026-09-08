"""BeePoseTrack-E calibrated inference without ordinary NMS."""

from collections import deque

import torch
import torch.nn.functional as F

from .postprocessor import structural_set_deduplicate

__all__ = ['DensityResidualRechecker']


def _domain_config(config, domain_id):
    name = 'ir' if int(domain_id) == 1 else 'rgb'
    if name not in config:
        raise KeyError(f'missing calibrated density recheck contract for {name}')
    required = {
        'trigger_count', 'residual_threshold', 'expansion_rate',
        'gaussian_sigma_ratio',
    }
    missing = required - set(config[name])
    if missing:
        raise KeyError(f'{name} density recheck contract missing {sorted(missing)}')
    return config[name]


def _residual_density(density, boxes, image_size, sigma_ratio):
    residual = density.float().clone().clamp_min_(0)
    map_h, map_w = residual.shape
    image_h, image_w = image_size
    yy = torch.arange(map_h, device=residual.device, dtype=residual.dtype)[:, None]
    xx = torch.arange(map_w, device=residual.device, dtype=residual.dtype)[None, :]
    for box in boxes:
        center_x = (box[0] + box[2]) * 0.5 / max(float(image_w), 1.0) * map_w
        center_y = (box[1] + box[3]) * 0.5 / max(float(image_h), 1.0) * map_h
        box_w = (box[2] - box[0]).clamp_min(1.0) / max(float(image_w), 1.0) * map_w
        box_h = (box[3] - box[1]).clamp_min(1.0) / max(float(image_h), 1.0) * map_h
        sigma = (torch.sqrt(box_w * box_h) * float(sigma_ratio)).clamp_min(0.5)
        gaussian = torch.exp(-((xx - center_x).square() + (yy - center_y).square()) / (2 * sigma.square()))
        gaussian = gaussian / gaussian.sum().clamp_min(1e-8)
        residual.sub_(gaussian)
    return residual.clamp_min_(0)


def _largest_integrated_component(residual, threshold):
    residual_cpu = residual.detach().float().cpu()
    mask = residual_cpu >= float(threshold)
    if not bool(mask.any()):
        return None
    height, width = mask.shape
    visited = torch.zeros_like(mask)
    best = None
    for row in range(height):
        for column in range(width):
            if not bool(mask[row, column]) or bool(visited[row, column]):
                continue
            queue = deque([(row, column)])
            visited[row, column] = True
            pixels = []
            while queue:
                current_row, current_column = queue.popleft()
                pixels.append((current_row, current_column))
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if (
                        0 <= next_row < height and 0 <= next_column < width
                        and bool(mask[next_row, next_column])
                        and not bool(visited[next_row, next_column])
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
            rows = [pixel[0] for pixel in pixels]
            columns = [pixel[1] for pixel in pixels]
            integral = sum(float(residual_cpu[r, c]) for r, c in pixels)
            candidate = (integral, min(columns), min(rows), max(columns) + 1, max(rows) + 1)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def _square_crop(component, density_shape, image_size, expansion_rate):
    _, left, top, right, bottom = component
    map_h, map_w = density_shape
    image_h, image_w = image_size
    left = left / map_w * image_w
    right = right / map_w * image_w
    top = top / map_h * image_h
    bottom = bottom / map_h * image_h
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    side = max(right - left, bottom - top) * (1.0 + 2.0 * float(expansion_rate))
    side = min(max(side, 1.0), float(min(image_w, image_h)))
    left = min(max(center_x - side * 0.5, 0.0), image_w - side)
    top = min(max(center_y - side * 0.5, 0.0), image_h - side)
    return tuple(int(round(value)) for value in (left, top, left + side, top + side))


def _map_local_result(result, crop_box, model_input_size):
    left, top, right, bottom = crop_box
    scale_x = (right - left) / float(model_input_size)
    scale_y = (bottom - top) / float(model_input_size)
    mapped = dict(result)
    if 'boxes' in mapped:
        boxes = mapped['boxes'].clone()
        boxes[:, 0::2] = boxes[:, 0::2] * scale_x + left
        boxes[:, 1::2] = boxes[:, 1::2] * scale_y + top
        mapped['boxes'] = boxes
    if 'keypoints' in mapped:
        keypoints = mapped['keypoints'].clone()
        keypoints[..., 0] = keypoints[..., 0] * scale_x + left
        keypoints[..., 1] = keypoints[..., 1] * scale_y + top
        mapped['keypoints'] = keypoints
    return mapped


def _append_instances(base, extra):
    base_count = len(base.get('scores', []))
    extra_count = len(extra.get('scores', []))
    merged = dict(base)
    for name in set(base) | set(extra):
        left = base.get(name)
        right = extra.get(name)
        if (
            torch.is_tensor(left) and torch.is_tensor(right)
            and left.ndim and right.ndim
            and left.shape[0] == base_count and right.shape[0] == extra_count
            and left.shape[1:] == right.shape[1:]
        ):
            merged[name] = torch.cat((left, right), dim=0)
    return merged


class DensityResidualRechecker:
    """Single-region residual-density recheck using the same inference callback."""

    def __init__(self, calibration_by_domain, input_size=1280, max_passes=1,
                 dedup_thresholds_by_domain=None, endpoint_oks_sigma=0.10):
        if int(max_passes) < 0:
            raise ValueError('max_passes must be non-negative')
        self.calibration_by_domain = calibration_by_domain
        self.input_size = int(input_size)
        self.max_passes = int(max_passes)
        self.dedup_thresholds_by_domain = dedup_thresholds_by_domain or {}
        self.endpoint_oks_sigma = float(endpoint_oks_sigma)

    @classmethod
    def from_calibration_contract(cls, contract, input_size=1280,
                                  endpoint_oks_sigma=0.10):
        if contract.get('effective_stage') != 'E-S5':
            raise ValueError('Density recheck contract must be frozen at E-S5.')
        domains = contract.get('domains', {})
        if set(domains) != {'rgb', 'ir'}:
            raise ValueError('Density recheck contract requires exact rgb/ir domains.')
        density = {}
        dedup = {}
        for domain_name in ('rgb', 'ir'):
            domain = domains[domain_name]
            if domain.get('source_data') != 'calibration':
                raise ValueError(
                    f'{domain_name} density parameters were not fitted on calibration.'
                )
            density[domain_name] = dict(domain.get('density_recheck', {}))
            dedup[domain_name] = dict(domain.get('set_deduplication', {}))
            _domain_config(density, 1 if domain_name == 'ir' else 0)
            if 'max_passes' not in density[domain_name]:
                raise ValueError(f'{domain_name} density recheck misses max_passes.')
        return cls(
            density,
            input_size=input_size,
            max_passes=max(int(values['max_passes']) for values in density.values()),
            dedup_thresholds_by_domain=dedup,
            endpoint_oks_sigma=endpoint_oks_sigma,
        )

    def __call__(self, image, result, infer, domain_id):
        if image.ndim != 3:
            raise ValueError('recheck image must have shape [C, H, W]')
        if 'density' not in result:
            return structural_set_deduplicate(
                result, self.dedup_thresholds_by_domain.get(
                    'ir' if int(domain_id) == 1 else 'rgb', {}
                ), self.endpoint_oks_sigma,
            )
        config = _domain_config(self.calibration_by_domain, domain_id)
        merged = dict(result)
        image_h, image_w = image.shape[-2:]
        executed = []
        domain_max_passes = min(
            self.max_passes, int(config.get('max_passes', self.max_passes))
        )
        for pass_index in range(domain_max_passes):
            residual_count = max(
                0.0,
                float(result['density'].sum()) - len(merged.get('scores', [])),
            )
            if residual_count < float(config['trigger_count']):
                break
            residual = _residual_density(
                result['density'], merged['boxes'], (image_h, image_w),
                config['gaussian_sigma_ratio'],
            )
            component = _largest_integrated_component(
                residual, config['residual_threshold']
            )
            if component is None:
                break
            crop_box = _square_crop(
                component, residual.shape, (image_h, image_w),
                config['expansion_rate'],
            )
            left, top, right, bottom = crop_box
            crop = image[:, top:bottom, left:right]
            crop = F.interpolate(
                crop[None], size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False,
            )[0]
            local = _map_local_result(infer(crop, int(domain_id)), crop_box, self.input_size)
            local_count = len(local.get('scores', []))
            if local_count == 0:
                break
            local['recheck_pass'] = torch.full(
                (local_count,), pass_index + 1, dtype=torch.long,
                device=local['scores'].device,
            )
            if 'recheck_pass' not in merged:
                merged['recheck_pass'] = torch.zeros(
                    len(merged['scores']), dtype=torch.long,
                    device=merged['scores'].device,
                )
            merged = _append_instances(merged, local)
            executed.append(crop_box)
        domain_name = 'ir' if int(domain_id) == 1 else 'rgb'
        merged = structural_set_deduplicate(
            merged, self.dedup_thresholds_by_domain.get(domain_name, {}),
            self.endpoint_oks_sigma,
        )
        merged['density_recheck_regions'] = executed
        merged['density_recheck_passes'] = len(executed)
        return merged
