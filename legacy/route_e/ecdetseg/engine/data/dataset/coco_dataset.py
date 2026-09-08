"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# import faster_coco_eval
import copy
import hashlib
import math
import pycocotools.mask as coco_mask
import re
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pathlib import Path

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset

torchvision.disable_beta_transforms_warning()
# faster_coco_eval.init_as_pycocotools()
Image.MAX_IMAGE_PIXELS = None

__all__ = ['CocoDetection', 'TemporalCocoDetection']


@register()
class CocoDetection(torchvision.datasets.CocoDetection, DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, transforms, return_masks=False,
                 remap_mscoco_category=False, strict_domain_schema=False,
                 strict_schema=False, schema_version=2,
                 expected_ann_sha256=None, strict_scene_schema=False,
                 default_scene=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        if expected_ann_sha256 is not None:
            digest = hashlib.sha256()
            with open(ann_file, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != str(expected_ann_sha256).lower():
                raise ValueError(
                    f'Annotation SHA256 mismatch for {ann_file}: '
                    f'expected={expected_ann_sha256}, actual={actual}'
                )
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.strict_domain_schema = strict_domain_schema
        self.strict_scene_schema = bool(strict_scene_schema)
        self.default_scene = default_scene
        self.strict_schema = strict_schema
        self.schema_version = int(schema_version)
        if self.strict_schema:
            self.validate_schema_dict(self.coco.dataset, self.schema_version)
        sensor_names = []
        sequence_names = []
        annotator_names = []
        self.domain_indices = {0: [], 1: []}
        self.hard_negative_indices = {0: [], 1: []}
        self.labelled_indices = {0: [], 1: []}
        self.unlabelled_indices = {0: [], 1: []}
        self.video_indices = {0: {}, 1: {}}
        self.index_video = {}
        self.sampling_weights = {}
        # 飞轮的无效/存疑帧不能进入基础覆盖或被当成无目标负样本。
        self.ids = [image_id for image_id in self.ids
                    if self.coco.imgs[image_id].get('status', 'valid') not in ('suspect', 'invalid')
                    and float(self.coco.imgs[image_id].get('sampling_weight', 1.0)) > 0]
        for index, image_id in enumerate(self.ids):
            image_info = self.coco.loadImgs(image_id)[0]
            domain_id = self._domain_id(image_info)
            self.domain_indices[domain_id].append(index)
            is_unlabelled = bool(image_info.get(
                'is_unlabeled', image_info.get('unlabeled', False)
            ))
            destination = self.unlabelled_indices if is_unlabelled else self.labelled_indices
            destination[domain_id].append(index)
            is_hard_negative = bool(image_info.get('hard_negative', False)) or (
                domain_id == 1 and len(self.coco.imgToAnns.get(image_id, [])) == 0
            )
            if is_hard_negative:
                self.hard_negative_indices[domain_id].append(index)
            sensor_names.append(str(image_info.get('sensor_id', image_info.get('camera_id', 'unknown'))))
            sequence_names.append(str(image_info.get('sequence_id', image_info.get('video_id', 'unknown'))))
            sequence_name = str(
                image_info.get('sequence_id', image_info.get('video_id', 'unknown'))
            )
            self.video_indices[domain_id].setdefault(sequence_name, []).append(index)
            self.index_video[index] = sequence_name
            self.sampling_weights[index] = float(image_info.get('sampling_weight', 1.0))
            annotator_names.extend(
                str(annotation.get('annotator_id', annotation.get('source', 'unknown')))
                for annotation in self.coco.imgToAnns.get(image_id, [])
            )
        self.sensor_to_id = {name: index for index, name in enumerate(sorted(set(sensor_names)))}
        self.sequence_to_id = {name: index for index, name in enumerate(sorted(set(sequence_names)))}
        self.annotator_to_id = {
            name: index for index, name in enumerate(sorted(set(annotator_names or ['unknown'])))
        }

    @staticmethod
    def validate_schema_dict(dataset, expected_version=2):
        """Validate the versioned BeePoseTrack-E instance contract."""
        actual_version = dataset.get(
            'schema_version', dataset.get('info', {}).get('schema_version')
        )
        if int(actual_version or -1) != int(expected_version):
            raise ValueError(
                f'BeePoseTrack-E schema_version must be {expected_version}, '
                f'got {actual_version!r}.'
            )

        images = {image['id']: image for image in dataset.get('images', [])}
        for image in images.values():
            missing = []
            if not any(key in image for key in ('domain', 'modality')):
                missing.append('domain/modality')
            if not any(key in image for key in ('sensor_id', 'camera_id')):
                missing.append('sensor_id/camera_id')
            if not any(key in image for key in ('sequence_id', 'video_id')):
                missing.append('sequence_id/video_id')
            if not any(key in image for key in ('frame_id', 'frame_index')):
                missing.append('frame_id/frame_index')
            if 'track_supervised' not in image:
                missing.append('track_supervised')
            if missing:
                raise ValueError(
                    f"image {image.get('id')} is missing schema fields: {', '.join(missing)}"
                )

        quality_fields = (
            'inter_group_quality', 'intra_group_quality',
            'hierarchy_quality', 'pose_quality', 'track_quality',
        )
        for annotation in dataset.get('annotations', []):
            annotation_id = annotation.get('id', '<unknown>')
            if annotation.get('image_id') not in images:
                raise ValueError(f'annotation {annotation_id} references an unknown image')
            required = (
                'bbox', 'category_id', 'pose_state', 'pose_mask',
                'track_id', 'track_mask', 'supervision_mask',
            )
            missing = [key for key in required if key not in annotation]
            if not any(key in annotation for key in ('annotator_id', 'source')):
                missing.append('annotator_id/source')
            missing.extend(key for key in quality_fields if key not in annotation)
            if missing:
                raise ValueError(
                    f"annotation {annotation_id} is missing schema fields: {', '.join(missing)}"
                )

            pose_state = int(annotation['pose_state'])
            pose_mask = bool(annotation['pose_mask'])
            if pose_state not in (0, 1, 2) or pose_mask != (pose_state > 0):
                raise ValueError(
                    f'annotation {annotation_id} has inconsistent pose_state/pose_mask'
                )
            keypoints = annotation.get('keypoints', [])
            if pose_state == 0 and keypoints:
                raise ValueError(
                    f'annotation {annotation_id} must not use zero-filled keypoints for a missing pose'
                )
            if pose_state > 0 and len(keypoints) != 6:
                raise ValueError(
                    f'annotation {annotation_id} must contain exactly two COCO keypoints'
                )
            visibility = keypoints[2::3]
            if pose_state == 1 and any(value > 0 for value in visibility):
                raise ValueError(f'annotation {annotation_id} marks an invisible pose as visible')
            if pose_state == 2 and not any(value > 0 for value in visibility):
                raise ValueError(f'annotation {annotation_id} has no visible labelled endpoint')

            track_mask = bool(annotation['track_mask'])
            track_geometry_mask = bool(annotation.get('track_geometry_mask', False))
            if track_mask and int(annotation['track_id']) < 0:
                raise ValueError(
                    f'annotation {annotation_id} enables track supervision without a valid track_id'
                )
            if track_geometry_mask:
                if not track_mask or len(annotation.get('track_geometry', [])) != 6:
                    raise ValueError(
                        f'annotation {annotation_id} enables track geometry without identity supervision'
                    )
            supervision = annotation['supervision_mask']
            if len(supervision) != 4:
                raise ValueError(f'annotation {annotation_id} supervision_mask must have four entries')
            if not bool(supervision[0]) or bool(supervision[1]) != pose_mask or bool(supervision[2]) != track_mask:
                raise ValueError(f'annotation {annotation_id} has inconsistent supervision_mask')
            for field in quality_fields:
                value = float(annotation[field])
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f'annotation {annotation_id} field {field} must be in [0, 1]'
                    )

    def _domain_id(self, image_info):
        value = image_info.get('domain', image_info.get('modality'))
        if value is None:
            if self.strict_domain_schema:
                raise ValueError('Every BeePoseTrack-E image must declare domain/modality as RGB or IR.')
            return 0
        normalized = str(value).strip().upper()
        if normalized in ('IR', 'INFRARED', 'THERMAL', 'TIR'):
            return 1
        if normalized in ('RGB', 'VISIBLE', 'VIS'):
            return 0
        raise ValueError(f'Unsupported domain/modality value: {value!r}.')

    def _scene_id(self, image_info):
        value = image_info.get('environment', image_info.get('scene'))
        if value is None:
            value = self.default_scene
        if value is None:
            if self.strict_scene_schema:
                raise ValueError(
                    'Every scene-routed image must declare environment/scene as A/outdoor or B/indoor.'
                )
            return 1
        normalized = str(value).strip().lower()
        if normalized in ('b', 'inside', 'indoor', 'nest_inside', '巢内'):
            return 0
        if normalized in ('a', 'outside', 'outdoor', 'nest_outside', '巢外'):
            return 1
        raise ValueError(f'Unsupported environment/scene value: {value!r}.')

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            self._transforms.set_epoch(self.epoch)
            img, target = self._transforms(img, target)
        return img, target

    def load_item(self, idx):
        image, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        ignored_annotations = [
            annotation for annotation in target
            if bool(annotation.get('ignore_region', annotation.get('ignore', False)))
        ]
        target = {
            'image_id': image_id,
            'annotations': [
                annotation for annotation in target
                if not bool(annotation.get('ignore_region', annotation.get('ignore', False)))
            ],
        }

        if self.remap_mscoco_category:
            image, target = self.prepare(
                image, target,
                category2label=mscoco_category2label,
                annotator2id=self.annotator_to_id,
            )
        else:
            image, target = self.prepare(
                image, target, annotator2id=self.annotator_to_id
            )

        target['idx'] = torch.tensor([idx])
        domain_id = self._domain_id(image_info)
        sensor_name = str(image_info.get('sensor_id', image_info.get('camera_id', 'unknown')))
        sequence_name = str(image_info.get('sequence_id', image_info.get('video_id', 'unknown')))
        is_hard_negative = bool(image_info.get('hard_negative', False)) or (
            domain_id == 1 and len(target['boxes']) == 0
        )
        target['domain_id'] = torch.tensor([domain_id], dtype=torch.int64)
        target['scene_id'] = torch.tensor([
            self._scene_id(image_info)
        ], dtype=torch.int64)
        target['sensor_id'] = torch.tensor([self.sensor_to_id[sensor_name]], dtype=torch.int64)
        target['sequence_id'] = torch.tensor([self.sequence_to_id[sequence_name]], dtype=torch.int64)
        target['frame_id'] = torch.tensor([
            int(image_info.get('frame_id', image_info.get('frame_index', image_id)))
        ], dtype=torch.int64)
        target['hard_negative'] = torch.tensor([is_hard_negative], dtype=torch.bool)
        target['is_unlabeled'] = torch.tensor([
            bool(image_info.get('is_unlabeled', image_info.get('unlabeled', False)))
        ], dtype=torch.bool)
        target['track_supervised'] = torch.tensor([
            bool(image_info.get('track_supervised', False))
        ], dtype=torch.bool)
        target['schema_version'] = torch.tensor([self.schema_version], dtype=torch.int64)

        ignore_xywh = list(image_info.get('ignore_boxes', [])) + [
            annotation['bbox'] for annotation in ignored_annotations
        ]
        if ignore_xywh:
            ignore_boxes = torch.as_tensor(ignore_xywh, dtype=torch.float32).reshape(-1, 4)
            ignore_boxes[:, 2:] += ignore_boxes[:, :2]
            ignore_boxes[:, 0::2].clamp_(0, image.size[0])
            ignore_boxes[:, 1::2].clamp_(0, image.size[1])
            target['ignore_boxes'] = convert_to_tv_tensor(
                ignore_boxes, key='boxes', spatial_size=image.size[::-1], box_format='XYXY',
            )

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        if hasattr(self, '_preset') and self._preset is not None:
            s += f' preset:\n   {repr(self._preset)}'
        return s

    @property
    def categories(self, ):
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


@register()
class TemporalCocoDetection(CocoDetection):
    """Return a configurable causal frame tuple ending at the supervised frame."""

    def __init__(self, *args, temporal_frames=5, enable_btca_metadata=False,
                 btca_short_quantile=0.35, btca_min_observations=2,
                 temporal_offsets=None, **kwargs):
        super().__init__(*args, **kwargs)
        if temporal_offsets is None:
            temporal_offsets = list(range(-(temporal_frames - 1), 1))
        temporal_offsets = [int(value) for value in temporal_offsets]
        if len(temporal_offsets) != int(temporal_frames):
            raise ValueError('temporal_offsets length must equal temporal_frames')
        if temporal_offsets[-1] != 0 or temporal_offsets != sorted(set(temporal_offsets)):
            raise ValueError('temporal_offsets must be unique, increasing, causal and end in 0')
        if any(value > 0 for value in temporal_offsets):
            raise ValueError('temporal_offsets cannot contain future frames')
        self.temporal_frames = temporal_frames
        self.temporal_offsets = tuple(temporal_offsets)
        self.enable_btca_metadata = bool(enable_btca_metadata)
        self.btca_short_quantile = float(btca_short_quantile)
        self.btca_min_observations = int(btca_min_observations)
        groups = {}
        explicit_neighbors = {}
        for image_id in self.ids:
            image_info = self.coco.loadImgs(image_id)[0]
            file_name = str(image_info['file_name']).replace('\\', '/')
            if image_info.get('temporal_paths'):
                paths = [
                    str(path).replace('\\', '/') for path in image_info['temporal_paths']
                ]
                if len(paths) != temporal_frames:
                    raise ValueError('temporal_paths length must equal temporal_frames')
                explicit_neighbors[image_id] = paths
            path = Path(file_name)
            sequence = image_info.get('sequence_id', image_info.get('video_id'))
            frame_value = image_info.get('frame_id', image_info.get('frame_index'))
            if sequence is not None and frame_value is not None:
                group_key = f"{self._domain_id(image_info)}:{sequence}"
                frame_id = int(frame_value)
            else:
                match = re.search(r'^(.*)_frame_(\d+)$', path.stem)
                prefix = match.group(1) if match else path.stem
                frame_id = int(match.group(2)) if match else int(image_id)
                group_key = f'{path.parent.as_posix()}/{prefix}'
            groups.setdefault(group_key, []).append((frame_id, image_id, file_name))

        self.temporal_neighbors = {}
        self.temporal_valid_masks = {}
        for sequence in groups.values():
            sequence.sort(key=lambda item: item[0])
            frame_ids = [item[0] for item in sequence]
            if len(frame_ids) != len(set(frame_ids)):
                raise ValueError('A temporal sequence contains duplicate frame_id values')
            for position, (_, image_id, _) in enumerate(sequence):
                neighbors = []
                valid_mask = []
                for offset in self.temporal_offsets:
                    requested_position = position + offset
                    neighbor_position = min(max(requested_position, 0), len(sequence) - 1)
                    neighbors.append(sequence[neighbor_position][2])
                    valid_mask.append(0 <= requested_position < len(sequence))
                self.temporal_neighbors[image_id] = neighbors
                self.temporal_valid_masks[image_id] = valid_mask
        self.temporal_neighbors.update(explicit_neighbors)
        for image_id, paths in explicit_neighbors.items():
            image_info = self.coco.loadImgs(image_id)[0]
            explicit_mask = image_info.get('temporal_valid_mask', [True] * len(paths))
            if len(explicit_mask) != temporal_frames:
                raise ValueError('temporal_valid_mask length must equal temporal_frames')
            self.temporal_valid_masks[image_id] = [bool(value) for value in explicit_mask]
        self.btca_tubes = {}
        if self.enable_btca_metadata:
            if self.temporal_frames != 3:
                raise ValueError('B-TCA metadata is defined for the three-frame E route.')
            self._build_btca_tubes(groups)

    @staticmethod
    def _btca_observation(annotation, image_info):
        width = float(image_info['width'])
        height = float(image_info['height'])
        x, y, box_width, box_height = [float(value) for value in annotation['bbox']]
        if box_width <= 0 or box_height <= 0:
            return None
        box = torch.tensor([x, y, x + box_width, y + box_height], dtype=torch.float32)
        center = torch.tensor(
            [x + box_width / 2, y + box_height / 2], dtype=torch.float32,
        )
        size = torch.tensor([box_width, box_height], dtype=torch.float32)
        raw_keypoints = annotation.get('keypoints', [])
        if len(raw_keypoints) >= 6:
            keypoints = torch.as_tensor(raw_keypoints, dtype=torch.float32).reshape(-1, 3)[:2]
        else:
            keypoints = torch.zeros(2, 3, dtype=torch.float32)
        if bool((keypoints[:, 2] > 0).all()):
            axis = keypoints[1, :2] - keypoints[0, :2]
            axis = axis / axis.norm().clamp_min(1e-6)
        else:
            axis = None
        quality = min(
            float(annotation.get('inter_group_quality', annotation.get('group_quality', 1.0))),
            float(annotation.get('intra_group_quality', annotation.get('instance_quality', 1.0))),
            float(annotation.get('hierarchy_quality', annotation.get('annotation_quality', 1.0))),
            float(annotation.get('track_quality', 1.0)),
        )
        return {
            'box': box, 'center': center, 'size': size, 'keypoints': keypoints,
            'axis': axis, 'quality': quality,
        }

    def _build_btca_tubes(self, groups):
        """Build domain-video-local short tubes and empirical tail distributions."""
        for sequence in groups.values():
            sequence = sorted(sequence, key=lambda item: item[0])
            observations = {}
            image_positions = {}
            for position, (_, image_id, _) in enumerate(sequence):
                image_positions[image_id] = position
                image_info = self.coco.loadImgs(image_id)[0]
                for annotation in self.coco.imgToAnns.get(image_id, []):
                    if bool(annotation.get('ignore_region', annotation.get('ignore', False))):
                        continue
                    track_id = int(annotation.get('track_id', -1))
                    if track_id < 0:
                        continue
                    observation = self._btca_observation(annotation, image_info)
                    if observation is None:
                        continue
                    observation.update({'position': position, 'image_id': image_id})
                    observations.setdefault(track_id, []).append(observation)
            lengths = [
                len(track) for track in observations.values()
                if len(track) >= self.btca_min_observations
            ]
            if not lengths:
                continue
            short_threshold = max(
                self.btca_min_observations,
                int(math.ceil(float(torch.quantile(
                    torch.tensor(lengths, dtype=torch.float32), self.btca_short_quantile,
                )))),
            )
            centers, velocities, sizes, axes = [], [], [], []
            for track in observations.values():
                track.sort(key=lambda item: item['position'])
                centers.extend(item['center'] for item in track)
                sizes.extend(item['size'] for item in track)
                axes.extend(item['axis'] for item in track if item['axis'] is not None)
                for previous, current in zip(track, track[1:]):
                    if current['position'] == previous['position'] + 1:
                        velocities.append(current['center'] - previous['center'])
            if not centers or not velocities or not sizes or not axes:
                continue
            distribution = {
                'centers': torch.stack(centers),
                'velocities': torch.stack(velocities),
                'sizes': torch.stack(sizes),
                'axes': torch.stack(axes),
            }
            by_position = {
                track_id: {item['position']: item for item in track}
                for track_id, track in observations.items()
            }
            for image_id, current_position in image_positions.items():
                tubes = []
                for track_id, position_map in by_position.items():
                    track_length = len(position_map)
                    if (
                        track_length < self.btca_min_observations
                        or track_length > short_threshold
                        or current_position not in position_map
                    ):
                        continue
                    current = position_map[current_position]
                    boxes, keypoints, valid_mask, qualities = [], [], [], []
                    for offset in self.temporal_offsets:
                        position = current_position + offset
                        observation = position_map.get(position)
                        valid_mask.append(observation is not None)
                        observation = current if observation is None else observation
                        boxes.append(observation['box'])
                        keypoints.append(observation['keypoints'])
                        qualities.append(observation['quality'])
                    if sum(valid_mask) < self.btca_min_observations:
                        continue
                    tubes.append({
                        'track_id': track_id,
                        'boxes': torch.stack(boxes),
                        'keypoints': torch.stack(keypoints),
                        'valid_mask': torch.tensor(valid_mask, dtype=torch.bool),
                        'quality': min(qualities),
                        'observation_length': track_length,
                        'short_track_threshold': short_threshold,
                        'tail_distribution': distribution,
                        'coordinate_space': 'pixel',
                    })
                if tubes:
                    self.btca_tubes[image_id] = tubes

    def load_item(self, idx):
        image, target = super().load_item(idx)
        image_id = self.ids[idx]
        target['temporal_paths'] = [
            str(Path(self.img_folder) / relative_path)
            for relative_path in self.temporal_neighbors[image_id]
        ]
        target['temporal_valid_mask'] = torch.tensor(
            self.temporal_valid_masks[image_id], dtype=torch.bool
        )
        if image_id in self.btca_tubes:
            target['btca_tubes'] = copy.deepcopy(self.btca_tubes[image_id])
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        keypoint_lengths = [len(obj.get('keypoints', [])) for obj in anno]
        keypoint_length = max(keypoint_lengths, default=0)
        if keypoint_length:
            if any(length not in (0, keypoint_length) for length in keypoint_lengths):
                raise ValueError('All keypoint annotations in one image must use the same layout.')
            keypoints = [
                obj.get('keypoints') or [0.0] * keypoint_length
                for obj in anno
            ]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        pose_states = []
        for obj in anno:
            if 'pose_state' in obj:
                pose_state = int(obj['pose_state'])
            elif 'pose_mask' in obj:
                if not bool(obj['pose_mask']):
                    pose_state = 0
                else:
                    visibility = obj.get('keypoints', [])[2::3]
                    pose_state = 2 if any(value > 0 for value in visibility) else 1
            else:
                # COCO's num_keypoints alone cannot distinguish an unlabelled
                # zero-filled pose from an intentionally fully-invisible pose.
                # Fully-invisible annotations must therefore set pose_state=1
                # (or pose_mask=true) explicitly.
                pose_state = 2 if int(obj.get('num_keypoints', 0)) > 0 else 0
            if pose_state not in (0, 1, 2):
                raise ValueError('pose_state must be 0 (missing), 1 (fully invisible), or 2 (coordinate-labelled).')
            if pose_state > 0 and keypoint_length == 0:
                raise ValueError('Annotated poses must include a keypoint vector, even when all points are invisible.')
            pose_states.append(pose_state)
        pose_state = torch.tensor(pose_states, dtype=torch.int64)
        pose_mask = (pose_state > 0).to(torch.float32)
        track_geometry = torch.as_tensor([
            obj.get('track_geometry', [0.0] * 6) for obj in anno
        ], dtype=torch.float32).reshape(-1, 6)
        track_geometry_mask = torch.tensor([
            bool(obj.get('track_geometry_mask', obj.get('track_mask', 'track_geometry' in obj)))
            for obj in anno
        ], dtype=torch.bool)
        track_id = torch.tensor([int(obj.get('track_id', -1)) for obj in anno], dtype=torch.int64)
        track_mask = torch.tensor([
            bool(obj.get('track_mask', obj.get('track_geometry_mask', 'track_geometry' in obj)))
            for obj in anno
        ], dtype=torch.bool) & (track_id >= 0)
        annotator2id = kwargs.get('annotator2id', {'unknown': 0})
        annotator_id = torch.tensor([
            annotator2id.get(str(obj.get('annotator_id', obj.get('source', 'unknown'))), -1)
            for obj in anno
        ], dtype=torch.int64)
        inter_group_quality = torch.tensor([
            float(obj.get('inter_group_quality', obj.get('group_quality', 1.0))) for obj in anno
        ], dtype=torch.float32)
        intra_group_quality = torch.tensor([
            float(obj.get('intra_group_quality', obj.get('instance_quality', 1.0))) for obj in anno
        ], dtype=torch.float32)
        hierarchy_quality = torch.tensor([
            float(obj.get('hierarchy_quality', obj.get('annotation_quality', 1.0))) for obj in anno
        ], dtype=torch.float32)
        is_pseudo = torch.tensor([bool(obj.get('is_pseudo', False)) for obj in anno], dtype=torch.bool)
        pseudo_score = torch.tensor([float(obj.get('pseudo_score', 1.0)) for obj in anno], dtype=torch.float32)
        pose_quality = torch.tensor([float(obj.get('pose_quality', 1.0)) for obj in anno], dtype=torch.float32)
        track_quality = torch.tensor([float(obj.get('track_quality', 1.0)) for obj in anno], dtype=torch.float32)
        trajectory_stability = torch.tensor([
            float(obj.get('trajectory_stability', 1.0)) for obj in anno
        ], dtype=torch.float32)
        supervision_mask = torch.tensor([
            obj.get(
                'supervision_mask',
                [True, pose_states[index] > 0, bool(track_mask[index]), True],
            )
            for index, obj in enumerate(anno)
        ], dtype=torch.bool).reshape(-1, 4)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        pose_state = pose_state[keep]
        pose_mask = pose_mask[keep]
        track_geometry = track_geometry[keep]
        track_geometry_mask = track_geometry_mask[keep]
        track_id = track_id[keep]
        track_mask = track_mask[keep]
        annotator_id = annotator_id[keep]
        inter_group_quality = inter_group_quality[keep]
        intra_group_quality = intra_group_quality[keep]
        hierarchy_quality = hierarchy_quality[keep]
        is_pseudo = is_pseudo[keep]
        pseudo_score = pseudo_score[keep]
        pose_quality = pose_quality[keep]
        track_quality = track_quality[keep]
        trajectory_stability = trajectory_stability[keep]
        supervision_mask = supervision_mask[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks.bool()
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints
        target["pose_state"] = pose_state
        target["pose_mask"] = pose_mask
        target['track_geometry'] = track_geometry
        target['track_geometry_mask'] = track_geometry_mask
        target['track_id'] = track_id
        target['track_mask'] = track_mask
        target['annotator_id'] = annotator_id
        target['inter_group_quality'] = inter_group_quality
        target['intra_group_quality'] = intra_group_quality
        target['hierarchy_quality'] = hierarchy_quality
        target['is_pseudo'] = is_pseudo
        target['pseudo_score'] = pseudo_score
        target['pose_quality'] = pose_quality
        target['track_quality'] = track_quality
        target['trajectory_stability'] = trajectory_stability
        target['supervision_mask'] = supervision_mask

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        # target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
mscoco_label2name_remap80 = {i: k for i, k in enumerate(mscoco_category2name.values())}
