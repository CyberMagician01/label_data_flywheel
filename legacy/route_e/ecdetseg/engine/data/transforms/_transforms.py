"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import PIL.ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import (BoundingBoxes, Image, Mask, Video, _boxes_keys,
                     convert_to_tv_tensor)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt


@register()
class ConvertKeypoints:
    """将 COCO 关键点坐标归一化到当前网络输入画布。"""

    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, sample):
        image, target, *extra = sample
        if self.normalize and 'keypoints' in target:
            height, width = F.get_size(image)
            target = dict(target)
            keypoints = target['keypoints'].clone()
            keypoints[..., 0] = keypoints[..., 0] / width
            keypoints[..., 1] = keypoints[..., 1] / height
            target['keypoints'] = keypoints
        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]


class StaticCameraStabilizer:
    """Estimate support-to-current similarity motion for a static camera.

    The branch order follows the route contract: phase-correlation identity
    check, ORB+RANSAC similarity, then bounded phase-translation fallback.
    Returned ``theta`` maps current output coordinates to support input
    coordinates and can therefore be passed directly to ``affine_grid``.
    """

    BRANCH_INVALID = 0
    BRANCH_IDENTITY = 1
    BRANCH_ORB_RANSAC = 2
    BRANCH_PHASE_TRANSLATION = 3

    def __init__(self, stable_translation=0.35, min_phase_response=0.05,
                 orb_features=1000, min_orb_matches=12,
                 ransac_threshold=2.0, max_translation=96.0,
                 max_rotation_degrees=8.0, min_scale=0.95, max_scale=1.05,
                 max_scale_change=None, max_rotation_deg=None):
        # The route contract records symmetric scale change and rotation with
        # concise names. Convert them to the explicit similarity bounds used
        # internally instead of silently dropping either safety limit.
        if max_rotation_deg is not None:
            max_rotation_degrees = float(max_rotation_deg)
        if max_scale_change is not None:
            max_scale_change = float(max_scale_change)
            if not 0.0 <= max_scale_change < 1.0:
                raise ValueError('max_scale_change must be in [0, 1).')
            min_scale = 1.0 - max_scale_change
            max_scale = 1.0 + max_scale_change
        self.stable_translation = float(stable_translation)
        self.min_phase_response = float(min_phase_response)
        self.orb_features = int(orb_features)
        self.min_orb_matches = int(min_orb_matches)
        self.ransac_threshold = float(ransac_threshold)
        self.max_translation = float(max_translation)
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    @staticmethod
    def _pixel_forward_to_grid_theta(forward, height, width):
        forward_h = np.eye(3, dtype=np.float64)
        forward_h[:2] = np.asarray(forward, dtype=np.float64)
        output_norm_to_pixel = np.array([
            [width / 2.0, 0.0, width / 2.0 - 0.5],
            [0.0, height / 2.0, height / 2.0 - 0.5],
            [0.0, 0.0, 1.0],
        ])
        input_pixel_to_norm = np.linalg.inv(output_norm_to_pixel)
        current_to_support = np.linalg.inv(forward_h)
        theta = input_pixel_to_norm @ current_to_support @ output_norm_to_pixel
        return torch.tensor(theta[:2], dtype=torch.float32)

    def _valid_similarity(self, affine):
        linear = affine[:, :2]
        scale = math.sqrt(max(float(np.linalg.det(linear)), 0.0))
        rotation = abs(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))
        translation = float(np.linalg.norm(affine[:, 2]))
        return (
            self.min_scale <= scale <= self.max_scale
            and rotation <= self.max_rotation_degrees
            and translation <= self.max_translation
        )

    def estimate(self, support_gray, current_gray):
        import cv2

        support = np.asarray(support_gray, dtype=np.float32)
        current = np.asarray(current_gray, dtype=np.float32)
        if support.shape != current.shape or support.ndim != 2:
            raise ValueError('Stabilization inputs must be equal-size grayscale images.')
        height, width = support.shape
        (shift_x, shift_y), phase_response = cv2.phaseCorrelate(support, current)
        if (
            phase_response >= self.min_phase_response
            and math.hypot(shift_x, shift_y) <= self.stable_translation
        ):
            forward = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            return self._pixel_forward_to_grid_theta(forward, height, width), {
                'branch': self.BRANCH_IDENTITY,
                'confidence': float(phase_response),
                'forward_affine': torch.tensor(forward, dtype=torch.float32),
            }

        support_u8 = np.clip(support, 0, 255).astype(np.uint8)
        current_u8 = np.clip(current, 0, 255).astype(np.uint8)
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        support_keypoints, support_descriptors = orb.detectAndCompute(support_u8, None)
        current_keypoints, current_descriptors = orb.detectAndCompute(current_u8, None)
        affine = None
        inlier_ratio = 0.0
        if support_descriptors is not None and current_descriptors is not None:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
                support_descriptors, current_descriptors,
            )
            matches = sorted(matches, key=lambda match: match.distance)
            if len(matches) >= self.min_orb_matches:
                source = np.float32([
                    support_keypoints[match.queryIdx].pt for match in matches
                ]).reshape(-1, 1, 2)
                destination = np.float32([
                    current_keypoints[match.trainIdx].pt for match in matches
                ]).reshape(-1, 1, 2)
                affine, inliers = cv2.estimateAffinePartial2D(
                    source, destination, method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_threshold,
                    maxIters=2000, confidence=0.995, refineIters=10,
                )
                if inliers is not None:
                    inlier_ratio = float(inliers.mean())
        if affine is not None and self._valid_similarity(affine):
            return self._pixel_forward_to_grid_theta(affine, height, width), {
                'branch': self.BRANCH_ORB_RANSAC,
                'confidence': inlier_ratio,
                'forward_affine': torch.tensor(affine, dtype=torch.float32),
            }

        shift_x = float(np.clip(shift_x, -self.max_translation, self.max_translation))
        shift_y = float(np.clip(shift_y, -self.max_translation, self.max_translation))
        forward = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
        return self._pixel_forward_to_grid_theta(forward, height, width), {
            'branch': self.BRANCH_PHASE_TRANSLATION,
            'confidence': max(float(phase_response), 0.0),
            'forward_affine': torch.tensor(forward, dtype=torch.float32),
        }


@register()
class PrepareTemporalFrames:
    """读取相邻帧，执行与中心帧一致的 IR、LetterBox 和归一化处理。"""

    def __init__(self, size, fill=114, lower=0.01, upper=0.99,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 stabilize=False, stabilizer=None):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.fill = fill
        self.lower = lower
        self.upper = upper
        self.mean = list(mean)
        self.std = list(std)
        self.stabilize = bool(stabilize)
        self.stabilizer = StaticCameraStabilizer(**(stabilizer or {}))
        self.bee_e_stage = None

    def set_stage_context(self, stage, progress=0.0, feedback=None):
        del progress, feedback
        self.bee_e_stage = str(stage)

    @staticmethod
    def _apply_rgb_augment(frame, parameters):
        brightness, contrast, saturation, shadow_start, shadow_end, shadow_factor, direction, kernel = parameters.tolist()
        frame = F.adjust_brightness(frame, brightness)
        frame = F.adjust_contrast(frame, contrast)
        frame = F.adjust_saturation(frame, saturation)
        tensor = F.pil_to_tensor(frame).float()
        height = tensor.shape[-2]
        top = min(max(int(shadow_start * height), 0), height)
        bottom = min(max(int(shadow_end * height), top), height)
        tensor[:, top:bottom] *= shadow_factor
        kernel = int(kernel)
        if kernel > 1:
            if direction == 0:
                weight = tensor.new_ones(1, 1, 1, kernel) / kernel
                padding = (0, kernel // 2)
            elif direction == 1:
                weight = tensor.new_ones(1, 1, kernel, 1) / kernel
                padding = (kernel // 2, 0)
            else:
                weight = torch.eye(kernel, device=tensor.device, dtype=tensor.dtype)[None, None] / kernel
                padding = (kernel // 2, kernel // 2)
            tensor = torch.nn.functional.conv2d(
                tensor[None], weight.expand(tensor.shape[0], 1, -1, -1),
                padding=padding, groups=tensor.shape[0],
            )[0]
        return F.to_pil_image(tensor.clamp(0, 255).to(torch.uint8))

    def _load_geometric(self, path, crop=None, rotation=None, horizontal_flip=False):
        with PIL.Image.open(path) as handle:
            frame = handle.convert('RGB')
        if crop is not None:
            top, left, height, width = [int(value) for value in crop.tolist()]
            frame = F.crop(frame, top, left, height, width)
        if rotation is not None:
            frame = F.rotate(frame, float(rotation.item()), fill=self.fill)
        if horizontal_flip:
            frame = F.horizontal_flip(frame)
        return frame

    def _letterbox(self, frame, fill=None):
        fill = self.fill if fill is None else fill
        source_h, source_w = F.get_size(frame)
        output_h, output_w = self.size
        scale = min(output_w / source_w, output_h / source_h)
        resized_w = max(1, round(source_w * scale))
        resized_h = max(1, round(source_h * scale))
        left = (output_w - resized_w) // 2
        top = (output_h - resized_h) // 2
        right = output_w - resized_w - left
        bottom = output_h - resized_h - top
        frame = F.resize(frame, [resized_h, resized_w], antialias=True)
        return F.pad(frame, [left, top, right, bottom], fill=fill)

    def _stabilization_view(self, frame):
        gray = F.rgb_to_grayscale(frame, num_output_channels=1)
        gray = self._letterbox(gray, fill=0)
        return F.pil_to_tensor(gray)[0].cpu().numpy()

    def _prepare_frame(self, frame, is_ir, sensor_augment=None, rgb_augment=None,
                       ir_quantile_bounds=None):
        if is_ir:
            tensor = F.pil_to_tensor(frame).float()
            gray = tensor.mean(dim=0, keepdim=True)
            if ir_quantile_bounds is None:
                low = torch.quantile(gray, self.lower)
                high = torch.quantile(gray, self.upper)
            else:
                bounds = torch.as_tensor(ir_quantile_bounds, dtype=gray.dtype)
                if bounds.numel() != 2:
                    raise ValueError('ir_quantile_bounds must contain [q_low, q_high].')
                low, high = bounds.reshape(-1)
            gray = ((gray.clamp(low, high) - low) / (high - low).clamp_min(1.0) * 255.0).to(torch.uint8)
            if sensor_augment is not None:
                gain, offset, noise_std, blur_flag, hot_pixel_p = sensor_augment.tolist()
                gray = gray.float() * gain + offset
                if noise_std > 0:
                    gray = gray + torch.randn_like(gray) * noise_std
                if blur_flag > 0:
                    gray = F.gaussian_blur(gray, kernel_size=[3, 3], sigma=[0.4, 1.0])
                if hot_pixel_p > 0:
                    hot = torch.rand_like(gray) < hot_pixel_p
                    gray = torch.where(hot, torch.full_like(gray, 255.0), gray)
                gray = gray.clamp(0, 255).to(torch.uint8)
            frame = F.to_pil_image(gray.repeat(3, 1, 1))
        elif rgb_augment is not None:
            frame = self._apply_rgb_augment(frame, rgb_augment)
        frame = self._letterbox(frame)
        frame = F.pil_to_tensor(frame).float() / 255.0
        return F.normalize(frame, mean=self.mean, std=self.std)

    def _prepare(self, path, is_ir, sensor_augment=None, crop=None,
                 rotation=None, horizontal_flip=False, rgb_augment=None):
        frame = self._load_geometric(path, crop, rotation, horizontal_flip)
        return self._prepare_frame(frame, is_ir, sensor_augment, rgb_augment)

    def __call__(self, sample):
        image, target, *extra = sample
        paths = target.get('temporal_paths')
        if not paths:
            return sample
        is_ir = int(target.get('domain_id', torch.tensor([0])).item()) == 1
        sensor_augment = target.get('ir_sensor_augment')
        crop = target.get('temporal_crop')
        rotation = target.get('temporal_rotation')
        horizontal_flip = bool(target.get(
            'temporal_horizontal_flip', torch.tensor(False)
        ).item())
        rgb_augment = target.get('rgb_sensor_augment')
        ir_quantile_bounds = target.get('ir_quantile_bounds')
        geometric_frames = [
            self._load_geometric(path, crop, rotation, horizontal_flip) for path in paths
        ]
        frames = [
            self._prepare_frame(
                frame, is_ir, sensor_augment, rgb_augment,
                ir_quantile_bounds=ir_quantile_bounds,
            )
            for frame in geometric_frames
        ]
        frames[-1] = image.as_subclass(torch.Tensor) if hasattr(image, 'as_subclass') else image
        target = dict(target)
        stage_index = (
            int(self.bee_e_stage.split('S')[-1])
            if self.bee_e_stage is not None else None
        )
        force_current = bool(torch.as_tensor(target.get(
            'force_current_frame_history', False,
        )).item())
        if (stage_index is not None and stage_index < 3) or force_current:
            frames = [frames[-1].clone(), frames[-1].clone(), frames[-1]]
            target['temporal_valid_mask'] = torch.tensor([False, False, True])
        if self.stabilize:
            if len(paths) != 3:
                raise ValueError('The stabilized temporal route requires exactly three frames.')
            valid_mask = torch.as_tensor(
                target.get('temporal_valid_mask', torch.ones(3)), dtype=torch.bool,
            )
            current_gray = self._stabilization_view(geometric_frames[-1])
            identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            thetas, branches, confidences, affines = [], [], [], []
            for support_index in range(2):
                if not bool(valid_mask[support_index]):
                    theta = identity.clone()
                    diagnostic = {
                        'branch': StaticCameraStabilizer.BRANCH_INVALID,
                        'confidence': 0.0,
                        'forward_affine': identity.clone(),
                    }
                else:
                    theta, diagnostic = self.stabilizer.estimate(
                        self._stabilization_view(geometric_frames[support_index]),
                        current_gray,
                    )
                thetas.append(theta)
                branches.append(diagnostic['branch'])
                confidences.append(diagnostic['confidence'])
                affines.append(diagnostic['forward_affine'])
            target['stabilization_theta'] = torch.stack(thetas)
            target['stabilization_branch'] = torch.tensor(branches, dtype=torch.int64)
            target['stabilization_confidence'] = torch.tensor(confidences, dtype=torch.float32)
            target['stabilization_forward_affine'] = torch.stack(affines)
        target.pop('temporal_paths', None)
        outputs = (torch.stack(frames, dim=0), target, *extra)
        return outputs if extra else outputs[:2]


@register()
class IRPercentileNormalize:
    """Use one background/foreground-derived percentile contract per IR clip."""

    def __init__(self, lower=0.01, upper=0.99,
                 foreground_residual_quantile=0.75):
        self.lower = lower
        self.upper = upper
        self.foreground_residual_quantile = float(foreground_residual_quantile)

    @staticmethod
    def _geometric_frame(path, target, output_size):
        with PIL.Image.open(path) as handle:
            frame = handle.convert('RGB')
        crop = target.get('temporal_crop')
        if crop is not None:
            top, left, height, width = [int(value) for value in crop.tolist()]
            frame = F.crop(frame, top, left, height, width)
        rotation = target.get('temporal_rotation')
        if rotation is not None:
            frame = F.rotate(frame, float(rotation.item()), fill=0)
        if bool(torch.as_tensor(target.get(
            'temporal_horizontal_flip', False,
        )).item()):
            frame = F.horizontal_flip(frame)
        if F.get_size(frame) != list(output_size):
            frame = F.resize(frame, list(output_size), antialias=True)
        return F.pil_to_tensor(frame).float().mean(dim=0)

    def _joint_clip_bounds(self, image, target):
        current = F.pil_to_tensor(image).float().mean(dim=0)
        frames = []
        for path in target.get('temporal_paths') or []:
            try:
                frames.append(self._geometric_frame(path, target, current.shape))
            except (FileNotFoundError, OSError):
                continue
        if not frames:
            frames = [current]
        elif len(frames) >= 1:
            # The transformed in-memory current frame is authoritative because
            # all preceding spatial augmentations have already been applied.
            frames[-1] = current
        clip = torch.stack(frames)
        background = clip.median(dim=0).values
        residual = (clip - background).abs()
        residual_cut = torch.quantile(
            residual, self.foreground_residual_quantile,
        )
        foreground = clip[residual >= residual_cut]
        joint = torch.cat((background.flatten(), foreground.flatten()))
        low = torch.quantile(joint, self.lower)
        high = torch.quantile(joint, self.upper)
        return torch.stack((low, high))

    def __call__(self, sample):
        image, target, *extra = sample
        if int(target.get('domain_id', torch.tensor([0])).item()) == 1:
            tensor = F.pil_to_tensor(image).float()
            gray = tensor.mean(dim=0, keepdim=True)
            bounds = self._joint_clip_bounds(image, target)
            low, high = bounds
            gray = ((gray.clamp(low, high) - low) / (high - low).clamp_min(1.0) * 255.0).to(torch.uint8)
            image = F.to_pil_image(gray.repeat(3, 1, 1))
            target = dict(target)
            target['ir_quantile_bounds'] = bounds
        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]


@register()
class IRSensorAugment:
    """IR-only gain/offset, sensor noise, blur and sparse hot-pixel augmentation."""

    def __init__(self, p=0.5, gain=(0.9, 1.1), offset=(-8.0, 8.0),
                 noise_std=(0.0, 4.0), blur_p=0.2, hot_pixel_p=0.001):
        self.p = p
        self.gain = gain
        self.offset = offset
        self.noise_std = noise_std
        self.blur_p = blur_p
        self.hot_pixel_p = hot_pixel_p

    @staticmethod
    def _sample(bounds):
        return bounds[0] + float(torch.rand(())) * (bounds[1] - bounds[0])

    def __call__(self, sample):
        image, target, *extra = sample
        is_ir = int(target.get('domain_id', torch.tensor([0])).item()) == 1
        if not is_ir or torch.rand(()) >= self.p:
            return sample
        tensor = F.pil_to_tensor(image).float()
        gray = tensor.mean(dim=0, keepdim=True)
        gain = self._sample(self.gain)
        offset = self._sample(self.offset)
        gray = gray * gain + offset
        noise_std = self._sample(self.noise_std)
        if noise_std > 0:
            gray = gray + torch.randn_like(gray) * noise_std
        blur_flag = float(torch.rand(()) < self.blur_p)
        if blur_flag:
            gray = F.gaussian_blur(gray, kernel_size=[3, 3], sigma=[0.4, 1.0])
        if self.hot_pixel_p > 0:
            hot = torch.rand_like(gray) < self.hot_pixel_p
            gray = torch.where(hot, torch.full_like(gray, 255.0), gray)
        image = F.to_pil_image(gray.clamp(0, 255).to(torch.uint8).repeat(3, 1, 1))
        target = dict(target)
        target['ir_sensor_augment'] = torch.tensor(
            [gain, offset, noise_std, blur_flag, self.hot_pixel_p], dtype=torch.float32
        )
        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]


@register()
class BeeRGBSensorAugment:
    """RGB-only color/shadow and direction-aware motion-blur augmentation."""

    def __init__(self, p=0.5, brightness=(0.85, 1.15), contrast=(0.85, 1.15),
                 saturation=(0.85, 1.15), shadow_factor=(0.65, 0.95),
                 motion_kernels=(1, 3, 5)):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.shadow_factor = shadow_factor
        self.motion_kernels = tuple(int(value) for value in motion_kernels)

    @staticmethod
    def _sample(bounds):
        return bounds[0] + float(torch.rand(())) * (bounds[1] - bounds[0])

    def __call__(self, sample):
        image, target, *extra = sample
        if int(target.get('domain_id', torch.tensor([0])).item()) == 1 or torch.rand(()) >= self.p:
            return sample
        start = float(torch.rand(())) * 0.6
        end = min(start + 0.2 + float(torch.rand(())) * 0.3, 1.0)
        parameters = torch.tensor([
            self._sample(self.brightness), self._sample(self.contrast),
            self._sample(self.saturation), start, end,
            self._sample(self.shadow_factor), int(torch.randint(3, ())),
            self.motion_kernels[int(torch.randint(len(self.motion_kernels), ()))],
        ], dtype=torch.float32)
        image = PrepareTemporalFrames._apply_rgb_augment(image, parameters)
        target = dict(target)
        target['rgb_sensor_augment'] = parameters
        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]


_INSTANCE_FIELDS = {
    'boxes', 'labels', 'area', 'iscrowd', 'masks', 'keypoints', 'pose_state',
    'pose_mask', 'track_geometry', 'track_geometry_mask', 'track_axis_mask', 'track_id', 'track_mask',
    'annotator_id', 'inter_group_quality', 'intra_group_quality',
    'hierarchy_quality', 'is_pseudo', 'pseudo_score', 'pose_quality',
    'track_quality', 'trajectory_stability', 'supervision_mask',
    'btca_augmented',
}


def _clone_btca_tubes(tubes):
    cloned = []
    for tube in tubes:
        item = {}
        for key, value in tube.items():
            if key == 'tail_distribution':
                item[key] = {
                    name: tensor.clone() for name, tensor in value.items()
                }
            else:
                item[key] = value.clone() if torch.is_tensor(value) else value
        cloned.append(item)
    return cloned


def _btca_transform_crop(tubes, top, left, height, width):
    result = _clone_btca_tubes(tubes)
    for tube in result:
        boxes = tube['boxes']
        old_area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ).clamp_min(1e-6)
        boxes[:, 0::2] = (boxes[:, 0::2] - left).clamp(0, width)
        boxes[:, 1::2] = (boxes[:, 1::2] - top).clamp(0, height)
        new_area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ).clamp_min(0)
        tube['valid_mask'] &= new_area / old_area >= 0.8
        points = tube['keypoints']
        points[..., 0] -= left
        points[..., 1] -= top
        inside = (
            (points[..., 0] >= 0) & (points[..., 0] < width)
            & (points[..., 1] >= 0) & (points[..., 1] < height)
        )
        points[..., 2] = torch.where(
            inside, points[..., 2], torch.zeros_like(points[..., 2]),
        )
        points[..., 0].clamp_(0, width)
        points[..., 1].clamp_(0, height)
        distribution = tube['tail_distribution']
        distribution['centers'][:, 0] -= left
        distribution['centers'][:, 1] -= top
        valid_centers = (
            (distribution['centers'][:, 0] >= 0)
            & (distribution['centers'][:, 0] < width)
            & (distribution['centers'][:, 1] >= 0)
            & (distribution['centers'][:, 1] < height)
        )
        distribution['centers'] = distribution['centers'][valid_centers]
    return result


def _rotate_xy(points, angle, height, width, *, translate_center=True):
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    output = points.clone()
    x = output[..., 0] - (width / 2 if translate_center else 0)
    y = output[..., 1] - (height / 2 if translate_center else 0)
    output[..., 0] = cosine * x + sine * y + (width / 2 if translate_center else 0)
    output[..., 1] = -sine * x + cosine * y + (height / 2 if translate_center else 0)
    return output


def _btca_transform_rotate(tubes, angle, height, width):
    result = _clone_btca_tubes(tubes)
    for tube in result:
        boxes = tube['boxes']
        corners = torch.stack([
            boxes[:, [0, 1]], boxes[:, [2, 1]],
            boxes[:, [2, 3]], boxes[:, [0, 3]],
        ], dim=1)
        corners = _rotate_xy(corners, angle, height, width)
        tube['boxes'] = torch.cat([
            corners.amin(dim=1), corners.amax(dim=1),
        ], dim=1)
        tube['keypoints'][..., :2] = _rotate_xy(
            tube['keypoints'][..., :2], angle, height, width,
        )
        inside = (
            (tube['keypoints'][..., 0] >= 0) & (tube['keypoints'][..., 0] < width)
            & (tube['keypoints'][..., 1] >= 0) & (tube['keypoints'][..., 1] < height)
        )
        tube['keypoints'][..., 2] = torch.where(
            inside, tube['keypoints'][..., 2],
            torch.zeros_like(tube['keypoints'][..., 2]),
        )
        distribution = tube['tail_distribution']
        distribution['centers'] = _rotate_xy(
            distribution['centers'], angle, height, width,
        )
        distribution['velocities'] = _rotate_xy(
            distribution['velocities'], angle, height, width,
            translate_center=False,
        )
        distribution['axes'] = _rotate_xy(
            distribution['axes'], angle, height, width,
            translate_center=False,
        )
    return result


def _btca_transform_flip(tubes, width):
    result = _clone_btca_tubes(tubes)
    for tube in result:
        boxes = tube['boxes']
        old_left = boxes[:, 0].clone()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - old_left
        tube['keypoints'][..., 0] = width - tube['keypoints'][..., 0]
        distribution = tube['tail_distribution']
        distribution['centers'][:, 0] = width - distribution['centers'][:, 0]
        distribution['velocities'][:, 0] *= -1
        distribution['axes'][:, 0] *= -1
    return result


def _btca_transform_letterbox(tubes, scale, left, top, output_size):
    output_height, output_width = output_size
    result = _clone_btca_tubes(tubes)
    normalizer = torch.tensor(
        [output_width, output_height], dtype=torch.float32,
    )
    for tube in result:
        tube['boxes'][:, 0::2] = tube['boxes'][:, 0::2] * scale + left
        tube['boxes'][:, 1::2] = tube['boxes'][:, 1::2] * scale + top
        tube['keypoints'][..., 0] = tube['keypoints'][..., 0] * scale + left
        tube['keypoints'][..., 1] = tube['keypoints'][..., 1] * scale + top
        distribution = tube['tail_distribution']
        distribution['centers'] = distribution['centers'] * scale
        distribution['centers'] += torch.tensor([left, top], dtype=torch.float32)
        distribution['velocities'] *= scale
        distribution['sizes'] *= scale
        tube['boxes'] /= torch.tensor(
            [output_width, output_height, output_width, output_height],
            dtype=torch.float32,
        )
        tube['keypoints'][..., :2] /= normalizer
        distribution['centers'] /= normalizer
        distribution['velocities'] /= normalizer
        distribution['sizes'] /= normalizer
        tube['coordinate_space'] = 'normalized'
    return result


def _clip_independent_boxes(boxes, height, width, *, min_size=1.0):
    """Clip an independent box set without aligning it to instance fields."""
    if boxes is None:
        return None
    raw = boxes.as_subclass(torch.Tensor) if hasattr(boxes, 'as_subclass') else boxes
    clipped = raw.clone()
    clipped[:, 0::2] = clipped[:, 0::2].clamp(0, width)
    clipped[:, 1::2] = clipped[:, 1::2].clamp(0, height)
    keep = (
        torch.isfinite(clipped).all(dim=1)
        & ((clipped[:, 2] - clipped[:, 0]) >= min_size)
        & ((clipped[:, 3] - clipped[:, 1]) >= min_size)
    )
    return convert_to_tv_tensor(
        clipped[keep], key='boxes', box_format='XYXY', spatial_size=(height, width)
    )


@register(name='SanitizeBoundingBoxes')
class SanitizeBoundingBoxes:
    """Remove invalid boxes and keep every per-instance target field aligned."""

    def __init__(self, min_size=1.0, min_area=1.0):
        self.min_size = float(min_size)
        self.min_area = float(min_area)

    def __call__(self, sample):
        image, target, *extra = sample
        boxes = target.get('boxes')
        height, width = F.get_size(image)
        target = dict(target)
        if 'ignore_boxes' in target:
            target['ignore_boxes'] = _clip_independent_boxes(
                target['ignore_boxes'], height, width,
                min_size=max(self.min_size, 1.0),
            )
        if boxes is None or len(boxes) == 0:
            outputs = (image, target, *extra)
            return outputs if extra else outputs[:2]

        raw = boxes.as_subclass(torch.Tensor) if hasattr(boxes, 'as_subclass') else boxes
        clipped = raw.clone()
        clipped[:, 0::2] = clipped[:, 0::2].clamp(0, width)
        clipped[:, 1::2] = clipped[:, 1::2].clamp(0, height)
        box_width = clipped[:, 2] - clipped[:, 0]
        box_height = clipped[:, 3] - clipped[:, 1]
        area = box_width * box_height
        keep = (
            torch.isfinite(clipped).all(dim=1)
            & (box_width >= self.min_size)
            & (box_height >= self.min_size)
            & (area >= self.min_area)
        )

        instance_count = len(keep)
        for key in _INSTANCE_FIELDS:
            value = target.get(key)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == instance_count:
                target[key] = value[keep]
        target['boxes'] = convert_to_tv_tensor(
            clipped[keep], key='boxes', box_format='XYXY', spatial_size=(height, width)
        )
        if 'area' in target:
            target['area'] = area[keep]

        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]


@register()
class BeeTargetCenteredCrop:
    """Target-centred crop with explicit truncation and instance-count guards."""

    def __init__(self, p=0.3, scale=(0.65, 0.95), min_instances=1,
                 min_short_side=2.0, max_truncation=0.20):
        self.p = p
        self.scale = scale
        self.min_instances = min_instances
        self.min_short_side = min_short_side
        self.max_truncation = max_truncation

    def __call__(self, sample):
        image, target, *extra = sample
        boxes = target.get('boxes')
        if boxes is None or len(boxes) == 0 or torch.rand(()) >= self.p:
            return sample
        height, width = F.get_size(image)
        scale = self.scale[0] + float(torch.rand(())) * (self.scale[1] - self.scale[0])
        crop_height = max(1, round(height * scale))
        crop_width = max(1, round(width * scale))
        selected = boxes[int(torch.randint(len(boxes), ()))]
        center_x = float((selected[0] + selected[2]) / 2)
        center_y = float((selected[1] + selected[3]) / 2)
        left = min(max(round(center_x - crop_width / 2), 0), width - crop_width)
        top = min(max(round(center_y - crop_height / 2), 0), height - crop_height)
        raw = boxes.as_subclass(torch.Tensor) if hasattr(boxes, 'as_subclass') else boxes
        clipped = raw.clone()
        clipped[:, 0::2] = (clipped[:, 0::2] - left).clamp(0, crop_width)
        clipped[:, 1::2] = (clipped[:, 1::2] - top).clamp(0, crop_height)
        old_area = ((raw[:, 2] - raw[:, 0]) * (raw[:, 3] - raw[:, 1])).clamp_min(1e-6)
        new_width = clipped[:, 2] - clipped[:, 0]
        new_height = clipped[:, 3] - clipped[:, 1]
        new_area = (new_width * new_height).clamp_min(0)
        keep = (
            (new_area / old_area >= 1.0 - self.max_truncation)
            & (torch.minimum(new_width, new_height) >= self.min_short_side)
        )
        if int(keep.sum()) < self.min_instances:
            return sample
        target = dict(target)
        for key in _INSTANCE_FIELDS:
            value = target.get(key)
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == len(keep):
                target[key] = value[keep]
        target['boxes'] = convert_to_tv_tensor(
            clipped[keep], key='boxes', box_format='XYXY',
            spatial_size=(crop_height, crop_width),
        )
        if 'ignore_boxes' in target:
            ignore_boxes = target['ignore_boxes']
            ignore_raw = (
                ignore_boxes.as_subclass(torch.Tensor)
                if hasattr(ignore_boxes, 'as_subclass') else ignore_boxes
            ).clone()
            ignore_raw[:, 0::2] -= left
            ignore_raw[:, 1::2] -= top
            target['ignore_boxes'] = _clip_independent_boxes(
                ignore_raw, crop_height, crop_width
            )
        if 'area' in target:
            target['area'] = new_area[keep]
        if 'keypoints' in target:
            keypoints = target['keypoints'].clone()
            keypoints[..., 0] = (keypoints[..., 0] - left).clamp(0, crop_width)
            keypoints[..., 1] = (keypoints[..., 1] - top).clamp(0, crop_height)
            target['keypoints'] = keypoints
        if 'masks' in target:
            target['masks'] = F.crop(target['masks'], top, left, crop_height, crop_width)
        if 'btca_tubes' in target:
            target['btca_tubes'] = _btca_transform_crop(
                target['btca_tubes'], top, left, crop_height, crop_width,
            )
        target['temporal_crop'] = torch.tensor([top, left, crop_height, crop_width])
        image = F.crop(image, top, left, crop_height, crop_width)
        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]


@register()
class BeeDirectionRotation:
    """Small synchronized rotation for image, boxes, endpoints and history frames."""

    def __init__(self, p=0.3, degrees=12.0, fill=114):
        self.p = p
        self.degrees = float(degrees)
        self.fill = fill

    def __call__(self, sample):
        image, target, *extra = sample
        if torch.rand(()) >= self.p:
            return sample
        angle = (float(torch.rand(())) * 2.0 - 1.0) * self.degrees
        height, width = F.get_size(image)
        image = F.rotate(image, angle, fill=self.fill)
        target = dict(target)
        if 'boxes' in target:
            target['boxes'] = F.rotate(target['boxes'], angle)
        if 'ignore_boxes' in target:
            target['ignore_boxes'] = F.rotate(target['ignore_boxes'], angle)
        if 'masks' in target:
            target['masks'] = F.rotate(target['masks'], angle, fill=0)
        if 'keypoints' in target:
            radians = torch.tensor(math.radians(angle), dtype=target['keypoints'].dtype)
            cosine, sine = radians.cos(), radians.sin()
            keypoints = target['keypoints'].clone()
            dx = keypoints[..., 0] - width / 2
            dy = keypoints[..., 1] - height / 2
            keypoints[..., 0] = cosine * dx + sine * dy + width / 2
            keypoints[..., 1] = -sine * dx + cosine * dy + height / 2
            inside = (
                (keypoints[..., 0] >= 0) & (keypoints[..., 0] < width)
                & (keypoints[..., 1] >= 0) & (keypoints[..., 1] < height)
            )
            keypoints[..., 2] = torch.where(
                inside, keypoints[..., 2], torch.zeros_like(keypoints[..., 2])
            )
            keypoints[..., 0].clamp_(0, width)
            keypoints[..., 1].clamp_(0, height)
            target['keypoints'] = keypoints
        if 'track_geometry' in target:
            geometry = target['track_geometry'].clone()
            radians = torch.tensor(
                math.radians(angle), dtype=geometry.dtype, device=geometry.device
            )
            cosine, sine = radians.cos(), radians.sin()
            for x_index, y_index in ((0, 1), (4, 5)):
                x_component = geometry[..., x_index].clone()
                y_component = geometry[..., y_index].clone()
                geometry[..., x_index] = cosine * x_component + sine * y_component
                geometry[..., y_index] = -sine * x_component + cosine * y_component
            target['track_geometry'] = geometry
        if 'btca_tubes' in target:
            target['btca_tubes'] = _btca_transform_rotate(
                target['btca_tubes'], angle, height, width,
            )
        target['temporal_rotation'] = torch.tensor(angle, dtype=torch.float32)
        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]


@register()
class RandomHorizontalFlipWithKeypoints:
    """同步翻转图像、框和头尾坐标，不交换 head/tail 语义。"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, target, *extra = sample
        if torch.rand(()) < self.p:
            _, width = F.get_size(image)
            image = F.horizontal_flip(image)
            target = dict(target)
            target['boxes'] = F.horizontal_flip(target['boxes'])
            if 'ignore_boxes' in target:
                target['ignore_boxes'] = F.horizontal_flip(target['ignore_boxes'])
            if 'keypoints' in target:
                keypoints = target['keypoints'].clone()
                keypoints[..., 0] = width - keypoints[..., 0]
                target['keypoints'] = keypoints
            if 'track_geometry' in target:
                geometry = target['track_geometry'].clone()
                geometry[..., 0] = -geometry[..., 0]
                geometry[..., 4] = -geometry[..., 4]
                target['track_geometry'] = geometry
            if 'btca_tubes' in target:
                target['btca_tubes'] = _btca_transform_flip(
                    target['btca_tubes'], width,
                )
            target['temporal_horizontal_flip'] = torch.tensor(True)
        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]


@register()
class RandomBeeKeypointOcclusion:
    """在蜜蜂局部生成遮挡，并将被遮挡但仍有坐标的关键点标为 COCO v=1。"""

    def __init__(self, p=0.5, max_regions=3, min_scale=0.25, max_scale=0.60, fill=114):
        self.p = p
        self.max_regions = max_regions
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fill = fill

    def __call__(self, sample):
        image, target, *extra = sample
        if torch.rand(()) >= self.p or 'keypoints' not in target or len(target['boxes']) == 0:
            return sample

        image = image.copy()
        target = dict(target)
        keypoints = target['keypoints'].clone()
        boxes = target['boxes'].as_subclass(torch.Tensor)
        image_width, image_height = image.size
        region_count = int(torch.randint(1, self.max_regions + 1, ()).item())
        draw = PIL.ImageDraw.Draw(image)

        for _ in range(region_count):
            object_idx = int(torch.randint(0, len(boxes), ()).item())
            point_idx = int(torch.randint(0, keypoints.shape[1], ()).item())
            center_x, center_y = keypoints[object_idx, point_idx, :2]
            box_width = (boxes[object_idx, 2] - boxes[object_idx, 0]).clamp_min(2)
            box_height = (boxes[object_idx, 3] - boxes[object_idx, 1]).clamp_min(2)
            scale = self.min_scale + float(torch.rand(())) * (self.max_scale - self.min_scale)
            half_width = float(box_width * scale * 0.5)
            half_height = float(box_height * scale * 0.5)
            left = max(0, int(float(center_x) - half_width))
            top = max(0, int(float(center_y) - half_height))
            right = min(image_width, int(float(center_x) + half_width))
            bottom = min(image_height, int(float(center_y) + half_height))
            draw.rectangle([left, top, right, bottom], fill=self.fill)

            inside = (
                (keypoints[..., 0] >= left) & (keypoints[..., 0] <= right)
                & (keypoints[..., 1] >= top) & (keypoints[..., 1] <= bottom)
                & (keypoints[..., 2] > 0)
            )
            keypoints[..., 2][inside] = 1

        target['keypoints'] = keypoints
        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]


@register()
class LetterBox:
    """等比例缩放并居中填充，同时变换框、关键点和恢复元数据。"""

    def __init__(self, size, fill=114):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.fill = fill

    def __call__(self, sample):
        image, target, *extra = sample
        source_h, source_w = F.get_size(image)
        output_h, output_w = self.size
        scale = min(output_w / source_w, output_h / source_h)
        resized_w = max(1, round(source_w * scale))
        resized_h = max(1, round(source_h * scale))
        left = (output_w - resized_w) // 2
        top = (output_h - resized_h) // 2
        right = output_w - resized_w - left
        bottom = output_h - resized_h - top

        image = F.resize(image, [resized_h, resized_w], antialias=True)
        image = F.pad(image, [left, top, right, bottom], fill=self.fill)

        target = dict(target)
        boxes = target['boxes'].as_subclass(torch.Tensor).clone()
        target['orig_boxes'] = boxes.clone()
        boxes[:, 0::2] = boxes[:, 0::2] * scale + left
        boxes[:, 1::2] = boxes[:, 1::2] * scale + top
        target['boxes'] = convert_to_tv_tensor(
            boxes,
            key='boxes',
            box_format='XYXY',
            spatial_size=(output_h, output_w),
        )

        if 'ignore_boxes' in target:
            ignore_boxes = target['ignore_boxes'].as_subclass(torch.Tensor).clone()
            ignore_boxes[:, 0::2] = ignore_boxes[:, 0::2] * scale + left
            ignore_boxes[:, 1::2] = ignore_boxes[:, 1::2] * scale + top
            target['ignore_boxes'] = convert_to_tv_tensor(
                ignore_boxes, key='boxes', box_format='XYXY',
                spatial_size=(output_h, output_w),
            )
        if 'masks' in target:
            masks = F.resize(
                target['masks'].to(torch.uint8), [resized_h, resized_w],
                antialias=False,
            )
            target['masks'] = F.pad(
                masks, [left, top, right, bottom], fill=0,
            ).bool()
        if 'area' in target:
            target['area'] = target['area'] * (scale ** 2)

        if 'keypoints' in target:
            keypoints = target['keypoints'].clone()
            target['orig_keypoints'] = keypoints.clone()
            keypoints[..., 0] = keypoints[..., 0] * scale + left
            keypoints[..., 1] = keypoints[..., 1] * scale + top
            target['keypoints'] = keypoints

        if 'btca_tubes' in target:
            target['btca_tubes'] = _btca_transform_letterbox(
                target['btca_tubes'], scale, left, top, (output_h, output_w),
            )

        target['input_size'] = torch.tensor([output_w, output_h], dtype=torch.int64)
        target['letterbox_scale'] = torch.tensor([scale], dtype=torch.float32)
        target['letterbox_pad'] = torch.tensor([left, top], dtype=torch.float32)
        outputs = (image, target, *extra)
        return outputs if extra else outputs[:2]
