"""BeePoseTrack-E stage-specific geometric and trajectory augmentation."""

import copy
import math
import random

import PIL.Image
import torch
import torch.nn.functional as nnF
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import convert_to_tv_tensor
from ._transforms import (_INSTANCE_FIELDS, _btca_transform_crop,
                          _btca_transform_rotate)


def _tensor(value):
    return value.as_subclass(torch.Tensor) if hasattr(value, 'as_subclass') else value


def _clone_target(target):
    return {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in target.items()
    }


def _domain_id(target):
    value = target.get('domain_id', torch.tensor([0]))
    return int(torch.as_tensor(value).reshape(-1)[0].item())


def _image_id(target):
    value = target.get('image_id', torch.tensor([-1]))
    return int(torch.as_tensor(value).reshape(-1)[0].item())


def _place_image(image, canvas, region, fill):
    region_left, region_top, region_width, region_height = region
    source_height, source_width = F.get_size(image)
    scale = min(region_width / source_width, region_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    left = region_left + (region_width - resized_width) // 2
    top = region_top + (region_height - resized_height) // 2
    resized = F.resize(image, [resized_height, resized_width], antialias=True)
    if isinstance(canvas, PIL.Image.Image):
        canvas.paste(resized, (left, top))
    else:
        canvas[..., top:top + resized_height, left:left + resized_width] = resized
    return scale, left, top, resized_height, resized_width


def _transform_target(target, scale, left, top, canvas_size, track_id_offset=0):
    canvas_height, canvas_width = canvas_size
    result = _clone_target(target)
    count = len(result.get('boxes', ()))
    boxes = result.get('boxes')
    if boxes is not None:
        boxes = _tensor(boxes).clone() * scale
        boxes[:, 0::2] += left
        boxes[:, 1::2] += top
        result['boxes'] = convert_to_tv_tensor(
            boxes, key='boxes', box_format='XYXY', spatial_size=canvas_size,
        )
    if 'ignore_boxes' in result:
        ignore_boxes = _tensor(result['ignore_boxes']).clone() * scale
        ignore_boxes[:, 0::2] += left
        ignore_boxes[:, 1::2] += top
        result['ignore_boxes'] = convert_to_tv_tensor(
            ignore_boxes, key='boxes', box_format='XYXY', spatial_size=canvas_size,
        )
    if 'keypoints' in result:
        keypoints = result['keypoints'].clone().to(torch.float32)
        keypoints[..., :2] *= scale
        keypoints[..., 0] += left
        keypoints[..., 1] += top
        result['keypoints'] = keypoints
    if 'masks' in result:
        masks = result['masks'].to(torch.uint8)
        source_height, source_width = masks.shape[-2:]
        resized_height = max(1, round(source_height * scale))
        resized_width = max(1, round(source_width * scale))
        masks = F.resize(masks, [resized_height, resized_width], antialias=False)
        placed = torch.zeros(
            count, canvas_height, canvas_width, dtype=masks.dtype, device=masks.device,
        )
        placed[:, top:top + resized_height, left:left + resized_width] = masks
        result['masks'] = placed.bool()
    if 'area' in result:
        result['area'] = result['area'] * (scale ** 2)
    if track_id_offset and 'track_id' in result:
        track_ids = result['track_id'].clone()
        valid = track_ids >= 0
        track_ids[valid] += int(track_id_offset)
        result['track_id'] = track_ids
    return result


def _merge_instance_targets(primary, secondary, canvas_size):
    result = {
        key: value for key, value in primary.items()
        if key not in _INSTANCE_FIELDS and key != 'ignore_boxes'
    }
    primary_count = len(primary.get('boxes', ()))
    secondary_count = len(secondary.get('boxes', ()))
    for key in _INSTANCE_FIELDS:
        first = primary.get(key)
        second = secondary.get(key)
        if first is None and second is None:
            continue
        if first is None or second is None:
            raise ValueError(
                f'Two-image stitch requires aligned instance field {key!r} in both samples.'
            )
        if first.shape[0] != primary_count or second.shape[0] != secondary_count:
            raise ValueError(f'Instance field {key!r} is not aligned with boxes.')
        result[key] = torch.cat([_tensor(first), _tensor(second)], dim=0)
    if 'boxes' in result:
        result['boxes'] = convert_to_tv_tensor(
            result['boxes'], key='boxes', box_format='XYXY', spatial_size=canvas_size,
        )
    ignore_parts = [
        _tensor(target['ignore_boxes']) for target in (primary, secondary)
        if 'ignore_boxes' in target and len(target['ignore_boxes'])
    ]
    if ignore_parts:
        result['ignore_boxes'] = convert_to_tv_tensor(
            torch.cat(ignore_parts, dim=0), key='boxes', box_format='XYXY',
            spatial_size=canvas_size,
        )
    result['stitched_image_ids'] = torch.tensor(
        [_image_id(primary), _image_id(secondary)], dtype=torch.int64,
    )
    result['is_two_image_stitch'] = torch.tensor(True)
    return result


def _rotate_target(image, target, angle, fill):
    """Rotate all spatial supervision and temporal geometric metadata."""
    height, width = F.get_size(image)
    rotated_image = F.rotate(image, angle, fill=fill)
    result = _clone_target(target)
    if 'boxes' in result:
        result['boxes'] = F.rotate(result['boxes'], angle)
    if 'ignore_boxes' in result:
        result['ignore_boxes'] = F.rotate(result['ignore_boxes'], angle)
    if 'masks' in result:
        result['masks'] = F.rotate(result['masks'], angle, fill=0)
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    if 'keypoints' in result:
        keypoints = result['keypoints'].clone()
        x = keypoints[..., 0] - width / 2
        y = keypoints[..., 1] - height / 2
        keypoints[..., 0] = cosine * x + sine * y + width / 2
        keypoints[..., 1] = -sine * x + cosine * y + height / 2
        inside = (
            (keypoints[..., 0] >= 0) & (keypoints[..., 0] < width)
            & (keypoints[..., 1] >= 0) & (keypoints[..., 1] < height)
        )
        keypoints[..., 2] = torch.where(
            inside, keypoints[..., 2], torch.zeros_like(keypoints[..., 2]),
        )
        keypoints[..., 0].clamp_(0, width)
        keypoints[..., 1].clamp_(0, height)
        result['keypoints'] = keypoints
    if 'track_geometry' in result:
        geometry = result['track_geometry'].clone()
        for x_index, y_index in ((0, 1), (4, 5)):
            x = geometry[..., x_index].clone()
            y = geometry[..., y_index].clone()
            geometry[..., x_index] = cosine * x + sine * y
            geometry[..., y_index] = -sine * x + cosine * y
        result['track_geometry'] = geometry
    if 'btca_tubes' in result:
        result['btca_tubes'] = _btca_transform_rotate(
            result['btca_tubes'], angle, height, width,
        )
    previous = float(torch.as_tensor(result.get('temporal_rotation', 0.0)).item())
    result['temporal_rotation'] = torch.tensor(previous + angle, dtype=torch.float32)
    return rotated_image, result


def _crop_target(image, target, top, left, crop_height, crop_width,
                 keep, clipped, new_area):
    result = _clone_target(target)
    count = len(keep)
    for key in _INSTANCE_FIELDS:
        value = result.get(key)
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count:
            result[key] = value[keep]
    result['boxes'] = convert_to_tv_tensor(
        clipped[keep], key='boxes', box_format='XYXY',
        spatial_size=(crop_height, crop_width),
    )
    if 'area' in result:
        result['area'] = new_area[keep]
    if 'keypoints' in result:
        keypoints = result['keypoints'].clone()
        keypoints[..., 0] = (keypoints[..., 0] - left).clamp(0, crop_width)
        keypoints[..., 1] = (keypoints[..., 1] - top).clamp(0, crop_height)
        result['keypoints'] = keypoints
    if 'masks' in result:
        result['masks'] = F.crop(
            result['masks'], top, left, crop_height, crop_width,
        )
    if 'ignore_boxes' in result:
        boxes = _tensor(result['ignore_boxes']).clone()
        boxes[:, 0::2] = (boxes[:, 0::2] - left).clamp(0, crop_width)
        boxes[:, 1::2] = (boxes[:, 1::2] - top).clamp(0, crop_height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        result['ignore_boxes'] = convert_to_tv_tensor(
            boxes[valid], key='boxes', box_format='XYXY',
            spatial_size=(crop_height, crop_width),
        )
    if 'btca_tubes' in result:
        result['btca_tubes'] = _btca_transform_crop(
            result['btca_tubes'], top, left, crop_height, crop_width,
        )
    result['temporal_crop'] = torch.tensor(
        [top, left, crop_height, crop_width], dtype=torch.int64,
    )
    return F.crop(image, top, left, crop_height, crop_width), result


@register()
class BeeTwoImageStitch:
    """E-S1 two-image stitch with domain and query-capacity contracts.

    Samples are cached separately per domain.  Each source is isotropically
    letterboxed into one half of the original canvas, so bee geometry is not
    anisotropically distorted.  Every per-instance field is kept aligned.
    """

    def __init__(self, p=0.15, orientations=('horizontal', 'vertical'),
                 max_cached_images=64, effective_query_capacity=768,
                 capacity_safety_ratio=0.80, max_resample_attempts=12,
                 fill=114):
        self.p = float(p)
        self.orientations = tuple(orientations)
        if not self.orientations or any(
            orientation not in ('horizontal', 'vertical')
            for orientation in self.orientations
        ):
            raise ValueError('orientations must contain horizontal and/or vertical')
        self.max_cached_images = int(max_cached_images)
        self.effective_query_capacity = int(effective_query_capacity)
        self.capacity_safety_ratio = float(capacity_safety_ratio)
        self.max_resample_attempts = int(max_resample_attempts)
        self.fill = fill
        self.cache = {0: [], 1: []}

    @staticmethod
    def _clone_sample(image, target):
        image = image.copy() if hasattr(image, 'copy') else image.clone()
        return image, _clone_target(target)

    def _remember(self, image, target):
        domain = _domain_id(target)
        cache = self.cache.setdefault(domain, [])
        cache.append(self._clone_sample(image, target))
        if len(cache) > self.max_cached_images:
            cache.pop(0)

    def _capacity(self, target):
        value = target.get('effective_query_count', self.effective_query_capacity)
        return int(torch.as_tensor(value).reshape(-1)[0].item())

    def _canvas(self, image, height, width):
        if isinstance(image, PIL.Image.Image):
            fill = self.fill
            if image.mode == 'RGB' and isinstance(fill, int):
                fill = (fill, fill, fill)
            return PIL.Image.new(image.mode, (width, height), color=fill)
        return torch.full(
            (*image.shape[:-2], height, width), self.fill,
            dtype=image.dtype, device=image.device,
        )

    def _stitch(self, image, target, peer_image, peer_target, orientation):
        height, width = F.get_size(image)
        peer_height, peer_width = F.get_size(peer_image)
        if (height, width) != (peer_height, peer_width):
            raise ValueError('Cached stitch inputs must share the pre-stitch canvas size.')
        if orientation == 'horizontal':
            split = width // 2
            regions = ((0, 0, split, height), (split, 0, width - split, height))
        else:
            split = height // 2
            regions = ((0, 0, width, split), (0, split, width, height - split))
        canvas = self._canvas(image, height, width)
        primary_placement = _place_image(image, canvas, regions[0], self.fill)
        secondary_placement = _place_image(peer_image, canvas, regions[1], self.fill)
        primary = _transform_target(
            target, primary_placement[0], primary_placement[1], primary_placement[2],
            (height, width),
        )
        current_ids = primary.get('track_id', torch.empty(0, dtype=torch.int64))
        positive = current_ids[current_ids >= 0]
        track_offset = int(positive.max().item()) + 1 if len(positive) else 1
        secondary = _transform_target(
            peer_target, secondary_placement[0], secondary_placement[1],
            secondary_placement[2], (height, width), track_offset,
        )
        merged = _merge_instance_targets(primary, secondary, (height, width))
        merged['stitch_orientation'] = orientation
        merged['stitch_capacity'] = torch.tensor(self._capacity(target), dtype=torch.int64)
        merged['stitch_gt_count'] = torch.tensor(len(merged.get('boxes', ())), dtype=torch.int64)
        # E-S1 uses a copied current-frame history; real temporal input starts at E-S3.
        merged['force_current_frame_history'] = torch.tensor(True)
        return canvas, merged

    def __call__(self, sample):
        image, target, *extra = sample
        domain = _domain_id(target)
        peers = list(self.cache.setdefault(domain, []))
        self._remember(image, target)
        if torch.rand(()) >= self.p or not peers:
            return sample
        capacity_limit = math.floor(
            self._capacity(target) * self.capacity_safety_ratio
        )
        current_count = len(target.get('boxes', ()))
        attempts = min(self.max_resample_attempts, max(1, len(peers)))
        for _ in range(attempts):
            peer_image, peer_target = random.choice(peers)
            if F.get_size(peer_image) != F.get_size(image):
                continue
            if current_count + len(peer_target.get('boxes', ())) > capacity_limit:
                continue
            orientation = random.choice(self.orientations)
            stitched = self._stitch(
                image, target, peer_image, peer_target, orientation,
            )
            outputs = (*stitched, *extra)
            return outputs if extra else outputs[:2]
        return sample


@register()
class BeeDirectionGapRotation:
    """Rotate labelled body axes toward under-covered circular bins."""

    def __init__(self, p=0.3, bins=12, max_degrees=30.0,
                 min_endpoint_retention=1.0, fill=114):
        self.p = float(p)
        self.bins = int(bins)
        self.max_degrees = float(max_degrees)
        self.min_endpoint_retention = float(min_endpoint_retention)
        self.fill = fill
        self.coverage = torch.zeros(self.bins, dtype=torch.float64)

    def update_coverage(self, counts):
        counts = torch.as_tensor(counts, dtype=torch.float64).reshape(-1)
        if len(counts) != self.bins or (counts < 0).any():
            raise ValueError('Direction coverage must be a non-negative vector of length bins.')
        self.coverage.copy_(counts)

    def _angles(self, target):
        keypoints = target.get('keypoints')
        if keypoints is None or keypoints.shape[1] < 2:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        visible = (keypoints[:, 0, 2] > 0) & (keypoints[:, 1, 2] > 0)
        if 'pose_mask' in target:
            visible &= target['pose_mask'].bool()
        indices = visible.nonzero(as_tuple=False).flatten()
        if not len(indices):
            return torch.empty(0), indices
        delta = keypoints[indices, 1, :2] - keypoints[indices, 0, :2]
        angles = torch.atan2(delta[:, 1], delta[:, 0]).remainder(2 * math.pi)
        return angles, indices

    def __call__(self, sample):
        image, target, *extra = sample
        angles, indices = self._angles(target)
        if not len(angles):
            return sample
        observed_bins = torch.floor(angles / (2 * math.pi) * self.bins).long()
        self.coverage.scatter_add_(
            0, observed_bins.cpu(), torch.ones_like(observed_bins, dtype=torch.float64).cpu(),
        )
        if torch.rand(()) >= self.p:
            return sample
        minimum = self.coverage.min()
        gap_bins = (self.coverage == minimum).nonzero(as_tuple=False).flatten()
        target_bin = int(gap_bins[int(torch.randint(len(gap_bins), ()))].item())
        target_angle = (
            (target_bin + float(torch.rand(()))) * 2 * math.pi / self.bins
        )
        source_angle = float(angles[int(torch.randint(len(angles), ()))].item())
        delta = (target_angle - source_angle + math.pi) % (2 * math.pi) - math.pi
        angle = max(-self.max_degrees, min(self.max_degrees, math.degrees(delta)))
        before_visible = (target['keypoints'][..., 2] > 0).sum().clamp_min(1)
        rotated_image, rotated_target = _rotate_target(image, target, angle, self.fill)
        after_visible = (rotated_target['keypoints'][..., 2] > 0).sum()
        if float(after_visible / before_visible) < self.min_endpoint_retention:
            return sample
        rotated_target['direction_gap_bin'] = torch.tensor(target_bin, dtype=torch.int64)
        rotated_target['direction_rotation_angle'] = torch.tensor(angle, dtype=torch.float32)
        outputs = (rotated_image, rotated_target, *extra)
        return outputs if extra else outputs[:2]


@register()
class BeeDensityConstrainedCrop:
    """IR/RGB density crop constrained by visibility and query pressure."""

    def __init__(self, p=0.3, domains=(1,), scale=(0.55, 0.95),
                 occupancy=(0.10, 0.75), effective_query_capacity=768,
                 capacity_safety_ratio=0.80, min_visible_ratio=0.80,
                 min_short_side=2.0, require_complete_endpoints=True,
                 attempts=24):
        self.p = float(p)
        self.domains = tuple(int(value) for value in domains)
        self.scale = tuple(float(value) for value in scale)
        self.occupancy = tuple(float(value) for value in occupancy)
        self.effective_query_capacity = int(effective_query_capacity)
        self.capacity_safety_ratio = float(capacity_safety_ratio)
        self.min_visible_ratio = float(min_visible_ratio)
        self.min_short_side = float(min_short_side)
        self.require_complete_endpoints = bool(require_complete_endpoints)
        self.attempts = int(attempts)

    def _capacity(self, target):
        value = target.get('effective_query_count', self.effective_query_capacity)
        return int(torch.as_tensor(value).reshape(-1)[0].item())

    def __call__(self, sample):
        image, target, *extra = sample
        boxes = target.get('boxes')
        if (
            _domain_id(target) not in self.domains or boxes is None or not len(boxes)
            or torch.rand(()) >= self.p
        ):
            return sample
        height, width = F.get_size(image)
        raw = _tensor(boxes)
        capacity = self._capacity(target)
        minimum_count = max(1, math.ceil(capacity * self.occupancy[0]))
        maximum_count = min(
            math.floor(capacity * self.occupancy[1]),
            math.floor(capacity * self.capacity_safety_ratio),
        )
        best = None
        target_count = (minimum_count + maximum_count) / 2
        centers = (raw[:, :2] + raw[:, 2:]) / 2
        for _ in range(self.attempts):
            scale = self.scale[0] + float(torch.rand(())) * (self.scale[1] - self.scale[0])
            crop_height = max(1, round(height * scale))
            crop_width = max(1, round(width * scale))
            center = centers[int(torch.randint(len(centers), ()))].clone()
            # Small jitter prevents repeatedly selecting the same dense window.
            center[0] += (float(torch.rand(())) - 0.5) * crop_width * 0.2
            center[1] += (float(torch.rand(())) - 0.5) * crop_height * 0.2
            left = min(max(round(float(center[0]) - crop_width / 2), 0), width - crop_width)
            top = min(max(round(float(center[1]) - crop_height / 2), 0), height - crop_height)
            clipped = raw.clone()
            clipped[:, 0::2] = (clipped[:, 0::2] - left).clamp(0, crop_width)
            clipped[:, 1::2] = (clipped[:, 1::2] - top).clamp(0, crop_height)
            old_area = (
                (raw[:, 2] - raw[:, 0]) * (raw[:, 3] - raw[:, 1])
            ).clamp_min(1e-6)
            new_width = clipped[:, 2] - clipped[:, 0]
            new_height = clipped[:, 3] - clipped[:, 1]
            new_area = (new_width * new_height).clamp_min(0)
            keep = (
                (new_area / old_area >= self.min_visible_ratio)
                & (torch.minimum(new_width, new_height) >= self.min_short_side)
            )
            count = int(keep.sum())
            if not minimum_count <= count <= maximum_count:
                continue
            if self.require_complete_endpoints and 'keypoints' in target:
                points = target['keypoints'][keep]
                labelled = points[..., 2] > 0
                inside = (
                    (points[..., 0] >= left) & (points[..., 0] < left + crop_width)
                    & (points[..., 1] >= top) & (points[..., 1] < top + crop_height)
                )
                if bool((labelled & ~inside).any()):
                    continue
            score = abs(count - target_count)
            if best is None or score < best[0]:
                best = (score, top, left, crop_height, crop_width, keep, clipped, new_area)
        if best is None:
            return sample
        _, top, left, crop_height, crop_width, keep, clipped, new_area = best
        cropped_image, cropped_target = _crop_target(
            image, target, top, left, crop_height, crop_width,
            keep, clipped, new_area,
        )
        cropped_target['density_crop_gt_count'] = torch.tensor(int(keep.sum()))
        cropped_target['density_crop_capacity'] = torch.tensor(capacity)
        cropped_target['density_crop_occupancy'] = torch.tensor(
            float(keep.sum()) / max(capacity, 1), dtype=torch.float32,
        )
        outputs = (cropped_image, cropped_target, *extra)
        return outputs if extra else outputs[:2]


def _pixel_to_grid_theta(pixel_mapping, height, width):
    norm_to_pixel = torch.tensor([
        [width / 2.0, 0.0, width / 2.0 - 0.5],
        [0.0, height / 2.0, height / 2.0 - 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=pixel_mapping.dtype, device=pixel_mapping.device)
    pixel_to_norm = torch.linalg.inv(norm_to_pixel)
    return (pixel_to_norm @ pixel_mapping @ norm_to_pixel)[:2]


def _target_to_source_mapping(source_box, target_box, angle, height, width,
                              device, dtype):
    source_box = torch.as_tensor(source_box, device=device, dtype=dtype)
    target_box = torch.as_tensor(target_box, device=device, dtype=dtype)
    source_center = (source_box[:2] + source_box[2:]) * 0.5
    target_center = (target_box[:2] + target_box[2:]) * 0.5
    source_size = (source_box[2:] - source_box[:2]).clamp_min(1e-6)
    target_size = (target_box[2:] - target_box[:2]).clamp_min(1e-6)
    source_center = source_center * torch.tensor([width, height], device=device, dtype=dtype)
    target_center = target_center * torch.tensor([width, height], device=device, dtype=dtype)
    source_size = source_size * torch.tensor([width, height], device=device, dtype=dtype)
    target_size = target_size * torch.tensor([width, height], device=device, dtype=dtype)
    radians = torch.as_tensor(-angle, device=device, dtype=dtype)
    cosine, sine = radians.cos(), radians.sin()
    rotation = torch.stack([
        torch.stack([cosine, -sine]), torch.stack([sine, cosine]),
    ])
    linear = torch.diag(source_size / target_size) @ rotation
    translation = source_center - linear @ target_center
    mapping = torch.eye(3, device=device, dtype=dtype)
    mapping[:2, :2] = linear
    mapping[:2, 2] = translation
    return mapping, torch.linalg.inv(mapping)


def _transform_points_forward(points, forward_mapping, height, width):
    points = points.clone()
    visible = points[..., 2:3]
    xy = points[..., :2] * torch.tensor(
        [width, height], device=points.device, dtype=points.dtype,
    )
    homogeneous = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    transformed = homogeneous @ forward_mapping.T
    points[..., :2] = transformed[..., :2] / torch.tensor(
        [width, height], device=points.device, dtype=points.dtype,
    )
    inside = (
        (points[..., 0] >= 0) & (points[..., 0] <= 1)
        & (points[..., 1] >= 0) & (points[..., 1] <= 1)
    )
    points[..., 2:3] = visible * inside.unsqueeze(-1)
    points[..., :2].clamp_(0, 1)
    return points


@register()
class BeeTrajectoryTailAugment:
    """B-TCA on reliable short three-frame track tubes.

    The source foreground is derived from its residual to a robust temporal
    median.  Motion, scale, axis and placement are sampled only from the
    domain-video tail distribution attached by ``TemporalCocoDetection``.
    """

    def __init__(self, p=0.15, residual_threshold=0.04,
                 residual_temperature=0.02, minimum_track_quality=0.7,
                 capacity_safety_ratio=0.80, effective_query_capacity=768,
                 minimum_mask_mass=4.0, rgb_gain=(0.95, 1.05)):
        self.p = float(p)
        self.residual_threshold = float(residual_threshold)
        self.residual_temperature = float(residual_temperature)
        self.minimum_track_quality = float(minimum_track_quality)
        self.capacity_safety_ratio = float(capacity_safety_ratio)
        self.effective_query_capacity = int(effective_query_capacity)
        self.minimum_mask_mass = float(minimum_mask_mass)
        self.rgb_gain = tuple(float(value) for value in rgb_gain)

    def _capacity(self, target):
        value = target.get('effective_query_count', self.effective_query_capacity)
        return int(torch.as_tensor(value).reshape(-1)[0].item())

    @staticmethod
    def _choose(tensor):
        tensor = torch.as_tensor(tensor)
        if not len(tensor):
            raise ValueError('B-TCA tail distributions may not be empty.')
        return tensor[int(torch.randint(len(tensor), ()))].clone()

    def _source_mask(self, clip, boxes, valid):
        frame_count, _, height, width = clip.shape
        background = clip.median(dim=0).values
        residual = (clip - background).abs().mean(dim=1, keepdim=True)
        masks = torch.zeros(
            frame_count, 1, height, width, device=clip.device, dtype=clip.dtype,
        )
        for frame in range(frame_count):
            if not bool(valid[frame]):
                continue
            box = boxes[frame]
            x1 = max(0, min(width, math.floor(float(box[0]) * width)))
            y1 = max(0, min(height, math.floor(float(box[1]) * height)))
            x2 = max(0, min(width, math.ceil(float(box[2]) * width)))
            y2 = max(0, min(height, math.ceil(float(box[3]) * height)))
            if x2 <= x1 or y2 <= y1:
                continue
            local = torch.sigmoid(
                (residual[frame, :, y1:y2, x1:x2] - self.residual_threshold)
                / max(self.residual_temperature, 1e-6)
            )
            # Retain local background texture around residual foreground.
            local = nnF.max_pool2d(local[None], 5, stride=1, padding=2)[0]
            masks[frame, :, y1:y2, x1:x2] = local
        return masks

    @staticmethod
    def _motion_blur(foreground, velocity):
        if float(velocity.norm()) < 1e-6:
            return foreground
        horizontal = abs(float(velocity[0])) >= abs(float(velocity[1]))
        if horizontal:
            kernel = foreground.new_ones(1, 1, 1, 3) / 3
            padding = (0, 1)
        else:
            kernel = foreground.new_ones(1, 1, 3, 1) / 3
            padding = (1, 0)
        channels = foreground.shape[1]
        return nnF.conv2d(
            foreground, kernel.expand(channels, 1, -1, -1),
            padding=padding, groups=channels,
        )

    def _append_instance(self, target, source_index, target_box, forward_mapping,
                         target_axis, target_velocity, domain, height, width):
        result = _clone_target(target)
        count = len(result['boxes'])
        for key in _INSTANCE_FIELDS:
            if key == 'boxes':
                continue
            value = result.get(key)
            if value is None or not torch.is_tensor(value) or value.ndim == 0:
                continue
            if value.shape[0] != count:
                raise ValueError(f'B-TCA instance field {key!r} is misaligned.')
            result[key] = torch.cat([value, value[source_index:source_index + 1].clone()])
        boxes = result['boxes']
        box_format = getattr(getattr(boxes, 'format', None), 'value', 'XYXY').upper()
        normalized = bool(float(_tensor(boxes).max()) <= 1.5)
        xyxy = target_box.clone()
        if box_format == 'CXCYWH':
            appended = torch.cat([(xyxy[:2] + xyxy[2:]) * 0.5, xyxy[2:] - xyxy[:2]])
        else:
            appended = xyxy
        if not normalized:
            appended = appended * torch.tensor(
                [width, height, width, height], device=appended.device,
                dtype=appended.dtype,
            )
        merged_boxes = torch.cat([_tensor(boxes), appended[None]], dim=0)
        result['boxes'] = convert_to_tv_tensor(
            merged_boxes, key='boxes', box_format=box_format,
            spatial_size=getattr(boxes, 'canvas_size', (height, width)),
        )
        if 'keypoints' in result:
            keypoints = result['keypoints']
            source_points = keypoints[source_index:source_index + 1].clone()
            point_normalized = bool(float(source_points[..., :2].max()) <= 1.5)
            if not point_normalized:
                source_points[..., 0] /= width
                source_points[..., 1] /= height
            transformed = _transform_points_forward(
                source_points, forward_mapping, height, width,
            )
            if not point_normalized:
                transformed[..., 0] *= width
                transformed[..., 1] *= height
            result['keypoints'][-1:] = transformed
        if 'track_id' in result:
            valid_ids = result['track_id'][:-1]
            valid_ids = valid_ids[valid_ids >= 0]
            result['track_id'][-1] = int(valid_ids.max().item()) + 1 if len(valid_ids) else 0
        if 'track_geometry' in result:
            geometry = result['track_geometry'][-1]
            diagonal = torch.linalg.vector_norm(target_box[2:] - target_box[:2]).clamp_min(1e-6)
            geometry.zero_()
            geometry[:2] = target_velocity / diagonal
            geometry[4:6] = target_axis / target_axis.norm().clamp_min(1e-6)
            result['track_geometry'][-1] = geometry
        for key in ('track_mask', 'track_geometry_mask', 'track_axis_mask'):
            if key in result:
                result[key][-1] = True
        if 'area' in result:
            pixel_area = (
                (target_box[2] - target_box[0]) * width
                * (target_box[3] - target_box[1]) * height
            )
            result['area'][-1] = pixel_area
        if 'masks' in result:
            # Pixel mask is installed by the caller after spatial warping.
            result['masks'][-1].zero_()
        if 'btca_augmented' not in result:
            result['btca_augmented'] = torch.cat([
                torch.zeros(count, dtype=torch.bool, device=boxes.device),
                torch.ones(1, dtype=torch.bool, device=boxes.device),
            ])
        else:
            result['btca_augmented'][-1] = True
        result['btca_domain'] = torch.tensor(domain, dtype=torch.int64)
        return result

    def __call__(self, sample):
        clip, target, *extra = sample
        tubes = target.get('btca_tubes', ())
        if torch.rand(()) >= self.p or not tubes:
            return sample
        if not torch.is_tensor(clip) or clip.ndim != 4 or clip.shape[0] != 3:
            raise ValueError('B-TCA requires a prepared three-frame tensor clip.')
        if len(target.get('boxes', ())) + 1 > math.floor(
            self._capacity(target) * self.capacity_safety_ratio
        ):
            return sample
        candidates = [
            tube for tube in tubes
            if float(tube.get('quality', 0.0)) >= self.minimum_track_quality
            and int(tube.get('observation_length', 10 ** 9))
            <= int(tube.get('short_track_threshold', -1))
            and bool(torch.as_tensor(tube['valid_mask'])[-1])
            and int(torch.as_tensor(tube['valid_mask']).sum()) >= 2
            and all(
                name in tube.get('tail_distribution', {})
                and len(tube['tail_distribution'][name])
                for name in ('centers', 'velocities', 'sizes', 'axes')
            )
        ]
        if not candidates:
            return sample
        tube = random.choice(candidates)
        domain = _domain_id(target)
        boxes = torch.as_tensor(tube['boxes'], device=clip.device, dtype=clip.dtype)
        valid = torch.as_tensor(tube['valid_mask'], device=clip.device, dtype=torch.bool)
        source_mask = self._source_mask(clip, boxes, valid)
        if float(source_mask.sum()) < self.minimum_mask_mass:
            return sample
        distribution = tube.get('tail_distribution', {})
        required = ('centers', 'velocities', 'sizes', 'axes')
        if any(name not in distribution or not len(distribution[name]) for name in required):
            raise ValueError('Reliable B-TCA tubes require empirical center/velocity/size/axis tails.')
        target_center = self._choose(distribution['centers']).to(clip)
        velocity = self._choose(distribution['velocities']).to(clip)
        target_size = self._choose(distribution['sizes']).to(clip).clamp_min(1e-4)
        target_axis = self._choose(distribution['axes']).to(clip)
        target_axis = target_axis / target_axis.norm().clamp_min(1e-6)
        target_angle = torch.atan2(target_axis[1], target_axis[0])
        source_points = torch.as_tensor(tube.get('keypoints', []), device=clip.device, dtype=clip.dtype)
        if source_points.numel() and source_points.shape[-2] >= 2:
            source_axis = source_points[-1, 1, :2] - source_points[-1, 0, :2]
            source_angle = torch.atan2(source_axis[1], source_axis[0])
        else:
            source_angle = clip.new_tensor(0.0)
        angle = target_angle - source_angle
        frame_count, _, height, width = clip.shape
        output = clip.clone()
        transformed_masks = []
        forward_mappings = []
        target_boxes = []
        for frame in range(frame_count):
            steps_from_current = frame - (frame_count - 1)
            center = target_center + velocity * steps_from_current
            half_size = target_size * 0.5
            target_box = torch.cat([center - half_size, center + half_size]).clamp(0, 1)
            if bool((target_box[2:] <= target_box[:2]).any()):
                return sample
            mapping, forward = _target_to_source_mapping(
                boxes[frame], target_box, angle, height, width,
                clip.device, clip.dtype,
            )
            theta = _pixel_to_grid_theta(mapping, height, width)
            grid = nnF.affine_grid(
                theta[None], (1, clip.shape[1], height, width), align_corners=False,
            )
            foreground = clip[frame:frame + 1] * source_mask[frame:frame + 1]
            transformed_foreground = nnF.grid_sample(
                foreground, grid, mode='bilinear', padding_mode='zeros',
                align_corners=False,
            )
            transformed_mask = nnF.grid_sample(
                source_mask[frame:frame + 1], grid, mode='bilinear',
                padding_mode='zeros', align_corners=False,
            ).clamp(0, 1)
            if domain == 0:
                gain = self.rgb_gain[0] + float(torch.rand(())) * (
                    self.rgb_gain[1] - self.rgb_gain[0]
                )
                transformed_foreground = self._motion_blur(
                    transformed_foreground * gain, velocity,
                )
            output[frame:frame + 1] = (
                output[frame:frame + 1] * (1 - transformed_mask)
                + transformed_foreground
            )
            transformed_masks.append(transformed_mask[0, 0])
            forward_mappings.append(forward)
            target_boxes.append(target_box)
        track_ids = target.get('track_id')
        if track_ids is None:
            return sample
        matches = (track_ids == int(tube['track_id'])).nonzero(as_tuple=False).flatten()
        if not len(matches):
            return sample
        result = self._append_instance(
            target, int(matches[0]), target_boxes[-1], forward_mappings[-1],
            target_axis, velocity, domain, height, width,
        )
        if 'masks' in result:
            result['masks'][-1] = transformed_masks[-1] >= 0.5
        result['btca_applied'] = torch.tensor(True)
        result['btca_source_track_id'] = torch.tensor(int(tube['track_id']))
        outputs = (output, result, *extra)
        return outputs if extra else outputs[:2]
