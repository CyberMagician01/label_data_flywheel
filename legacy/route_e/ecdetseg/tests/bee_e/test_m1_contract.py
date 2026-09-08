import itertools
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image


def _install_optional_dependency_stubs():
    tensorboard = types.ModuleType('torch.utils.tensorboard')
    tensorboard.SummaryWriter = object
    sys.modules.setdefault('torch.utils.tensorboard', tensorboard)

    pycocotools = types.ModuleType('pycocotools')
    pycocotools.__path__ = []
    for name, class_name in [('mask', None), ('coco', 'COCO'), ('cocoeval', 'COCOeval')]:
        module = types.ModuleType(f'pycocotools.{name}')
        if class_name:
            setattr(module, class_name, type(class_name, (), {}))
        setattr(pycocotools, name, module)
        sys.modules.setdefault(f'pycocotools.{name}', module)
    sys.modules.setdefault('pycocotools', pycocotools)

    calflops = types.ModuleType('calflops')
    calflops.calculate_flops = lambda *args, **kwargs: None
    sys.modules.setdefault('calflops', calflops)

    def exact_linear_sum_assignment(cost):
        cost = np.asarray(cost)
        row_count, column_count = cost.shape
        if row_count == 0 or column_count == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if column_count > row_count:
            columns, rows = exact_linear_sum_assignment(cost.T)
            return rows, columns
        best = None
        for rows in itertools.permutations(range(row_count), column_count):
            score = sum(cost[row, column] for column, row in enumerate(rows))
            if best is None or score < best[0]:
                best = score, rows
        return np.asarray(best[1], dtype=np.int64), np.arange(column_count, dtype=np.int64)

    scipy = types.ModuleType('scipy')
    scipy.__path__ = []
    scipy_optimize = types.ModuleType('scipy.optimize')
    scipy_optimize.linear_sum_assignment = exact_linear_sum_assignment
    scipy.optimize = scipy_optimize
    sys.modules['scipy'] = scipy
    sys.modules['scipy.optimize'] = scipy_optimize


_install_optional_dependency_stubs()

from engine.data.dataset.coco_dataset import CocoDetection, ConvertCocoPolysToMask
from engine.data._misc import convert_to_tv_tensor
from engine.data.transforms._transforms import SanitizeBoundingBoxes
from engine.edgecrafter.criterion import ECCriterion
from engine.edgecrafter.decoder import DecomposedEndpointDecoder
from engine.edgecrafter.ecvit import ConvPyramidPatchEmbed
from engine.edgecrafter.ecvit import ViTAdapter
from engine.edgecrafter.matcher import HungarianMatcher
from engine.edgecrafter.modeling import ECDet
from engine.core import YAMLConfig
from tools.bee_e.migrate_schema_v2 import migrate


def test_native_p2_is_direct_stride4_stem_output_at_1280():
    patch_embed = ConvPyramidPatchEmbed(embed_dim=32).eval()
    image = torch.randn(1, 3, 1280, 1280)
    with torch.no_grad():
        patch_tokens, p2, p3 = patch_embed.forward_with_stem(image)
    assert p2.shape == (1, 8, 320, 320)
    assert p3.shape == (1, 16, 160, 160)
    assert patch_tokens.shape == (1, 32, 80, 80)


def test_decomposed_endpoint_queries_sample_all_levels_and_refine_references():
    decoder = DecomposedEndpointDecoder(
        hidden_dim=32,
        num_keypoints=2,
        num_layers=3,
        num_levels=4,
        num_sampling_points=3,
    ).eval()
    decoder.reference_refine_heads[1].layers[-1].bias.data.fill_(0.5)
    instance_queries = torch.randn(3, 2, 5, 32)
    boxes = torch.tensor([0.5, 0.5, 0.4, 0.3]).view(1, 1, 1, 4).repeat(3, 2, 5, 1)
    features = [
        torch.randn(2, 32, 32, 32),
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 32, 4, 4),
    ]
    with torch.no_grad():
        keypoints, visibility = decoder(instance_queries, boxes, features)
    assert keypoints.shape == (3, 2, 5, 2, 2)
    assert visibility.shape == (3, 2, 5, 2)
    assert torch.all((keypoints >= 0) & (keypoints <= 1))
    assert not torch.allclose(keypoints[0], keypoints[1])


def test_pose_annotation_states_are_explicit_and_distinct():
    converter = ConvertCocoPolysToMask(return_masks=False)
    image = Image.new('RGB', (32, 32))
    annotations = [
        {'bbox': [1, 1, 8, 8], 'category_id': 1, 'area': 64,
         'keypoints': [2, 2, 2, 7, 7, 2], 'num_keypoints': 2},
        {'bbox': [2, 2, 8, 8], 'category_id': 1, 'area': 64,
         'keypoints': [3, 3, 2, 0, 0, 0], 'num_keypoints': 1},
        {'bbox': [3, 3, 8, 8], 'category_id': 1, 'area': 64,
         'keypoints': [0, 0, 0, 0, 0, 0], 'num_keypoints': 0, 'pose_state': 1},
        {'bbox': [4, 4, 8, 8], 'category_id': 1, 'area': 64,
         'keypoints': [0, 0, 0, 0, 0, 0], 'num_keypoints': 0},
    ]
    _, target = converter(image, {'image_id': 1, 'annotations': annotations})
    assert target['pose_state'].tolist() == [2, 2, 1, 0]
    assert target['pose_mask'].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_detection_only_box_is_kept_beside_pose_annotated_instances():
    converter = ConvertCocoPolysToMask(return_masks=False)
    image = Image.new('RGB', (32, 32))
    annotations = [
        {
            'bbox': [1, 1, 8, 8], 'category_id': 1, 'area': 64,
            'keypoints': [2, 2, 2, 7, 7, 2], 'num_keypoints': 2,
            'pose_mask': True,
        },
        {
            'bbox': [12, 12, 8, 8], 'category_id': 1, 'area': 64,
            'num_keypoints': 0, 'pose_mask': False,
        },
    ]
    _, target = converter(image, {'image_id': 1, 'annotations': annotations})

    assert target['boxes'].shape[0] == 2
    assert target['keypoints'].shape == (2, 2, 3)
    assert target['keypoints'][1].eq(0).all()
    assert target['pose_state'].tolist() == [2, 0]
    assert target['pose_mask'].tolist() == [1.0, 0.0]


def test_track_identity_is_kept_when_geometry_is_not_available():
    converter = ConvertCocoPolysToMask(return_masks=False)
    image = Image.new('RGB', (32, 32))
    annotations = [{
        'bbox': [1, 1, 8, 8], 'category_id': 1, 'area': 64,
        'track_id': 7, 'track_mask': True, 'track_geometry_mask': False,
    }]
    _, target = converter(image, {'image_id': 1, 'annotations': annotations})
    assert target['track_id'].tolist() == [7]
    assert target['track_mask'].tolist() == [True]
    assert target['track_geometry_mask'].tolist() == [False]


def test_schema_migration_preserves_cross_frame_group_ids_as_tracks(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({
        'images': [
            {
                'id': 1, 'file_name': 'frame_000000.jpg', 'width': 100, 'height': 80,
                'domain': 'RGB', 'scene': 'A', 'video': '1', 'frame': 0,
                'source': 'annotator_03',
                'source_json': '标注员_03/A-5-1_区段_01/frame_000000.json',
            },
            {
                'id': 2, 'file_name': 'frame_000005.jpg', 'width': 100, 'height': 80,
                'domain': 'RGB', 'scene': 'A', 'video': '1', 'frame': 5,
                'source': 'annotator_03',
                'source_json': '标注员_03/A-5-1_区段_01/frame_000005.json',
            },
        ],
        'annotations': [
            {
                'id': 1, 'image_id': 1, 'bbox': [10, 10, 10, 20],
                'category_id': 1, 'track_id': '42', 'pose_mask': True,
                'num_keypoints': 2, 'keypoints': [11, 15, 2, 19, 15, 2],
                'source': 'annotator_03', 'pairing_method': 'group_id',
            },
            {
                'id': 2, 'image_id': 2, 'bbox': [12, 10, 10, 20],
                'category_id': 1, 'track_id': '42', 'pose_mask': True,
                'num_keypoints': 2, 'keypoints': [13, 15, 2, 19, 15, 2],
                'source': 'annotator_03', 'pairing_method': 'group_id',
            },
        ],
        'categories': [{'id': 1, 'name': 'bee'}],
    }, ensure_ascii=False), encoding='utf-8')

    dataset, summary = migrate(source)
    first, second = dataset['annotations']
    assert first['source_group_id'] == second['source_group_id'] == '42'
    assert first['track_id'] == second['track_id'] == 0
    assert first['track_mask'] and second['track_mask']
    assert not first['track_geometry_mask']
    assert second['track_geometry_mask']
    assert second['track_geometry'][0] == pytest.approx(2 / (10 ** 2 + 20 ** 2) ** 0.5)
    assert second['track_geometry'][1:4] == pytest.approx([0.0, 0.0, 0.0])
    assert second['track_geometry'][4:] == pytest.approx([1.0, 0.0])
    assert all(image['track_supervised'] for image in dataset['images'])
    assert summary['track_supervised_annotations'] == 2
    assert summary['track_geometry_annotations'] == 1
    CocoDetection.validate_schema_dict(dataset)


def test_sanitize_boxes_keeps_pose_and_quality_fields_aligned():
    image = Image.new('RGB', (32, 32))
    target = {
        'boxes': convert_to_tv_tensor(
            torch.tensor([[2.0, 2.0, 12.0, 12.0], [20.0, 20.0, 20.0, 24.0]]),
            key='boxes', box_format='XYXY', spatial_size=(32, 32),
        ),
        'labels': torch.tensor([0, 0]),
        'area': torch.tensor([100.0, 0.0]),
        'keypoints': torch.tensor([
            [[4.0, 4.0, 2.0], [10.0, 10.0, 2.0]],
            [[20.0, 20.0, 2.0], [20.0, 24.0, 2.0]],
        ]),
        'pose_state': torch.tensor([2, 2]),
        'pose_mask': torch.tensor([1.0, 1.0]),
        'track_quality': torch.tensor([0.9, 0.1]),
        'supervision_mask': torch.tensor([[True, True, False, True], [True, True, False, True]]),
    }
    _, sanitized = SanitizeBoundingBoxes(min_size=1)((image, target))

    assert sanitized['boxes'].shape[0] == 1
    assert sanitized['labels'].shape[0] == 1
    assert sanitized['keypoints'].shape[0] == 1
    assert sanitized['pose_state'].tolist() == [2]
    assert sanitized['track_quality'].tolist() == pytest.approx([0.9])
    assert sanitized['supervision_mask'].shape == (1, 4)


def _matcher():
    return HungarianMatcher(
        weight_dict={
            'cost_class': 2,
            'cost_bbox': 5,
            'cost_giou': 2,
            'cost_keypoint': 4,
            'cost_oks': 2,
            'cost_direction': 1,
        },
        use_focal_loss=True,
    )


def test_unlabelled_pose_does_not_change_hungarian_matching():
    outputs = {
        'pred_logits': torch.tensor([[[8.0], [2.0], [-2.0]]]),
        'pred_boxes': torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]]]),
        'pred_keypoints': torch.zeros(1, 3, 2, 2),
        'pred_query_valid': torch.tensor([[True, True, False]]),
    }
    targets = [{
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        'keypoints': torch.zeros(1, 2, 3),
        'pose_state': torch.tensor([0]),
        'pose_mask': torch.tensor([0.0]),
    }]
    first = _matcher()(outputs, targets)['indices'][0][0]
    outputs['pred_keypoints'] = torch.randn(1, 3, 2, 2) * 1000
    second = _matcher()(outputs, targets)['indices'][0][0]
    assert first.tolist() == second.tolist() == [0]


def test_invalid_fixed_slot_never_participates_in_matching_or_classification_loss():
    outputs = {
        'pred_logits': torch.tensor([[[1.0], [-1.0], [20.0]]]),
        'pred_boxes': torch.tensor([[[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]]]),
        'pred_keypoints': torch.zeros(1, 3, 2, 2),
        'pred_query_valid': torch.tensor([[True, True, False]]),
    }
    targets = [{
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        'keypoints': torch.zeros(1, 2, 3),
        'pose_state': torch.tensor([0]),
    }]
    matched_query = _matcher()(outputs, targets)['indices'][0][0]
    assert matched_query.item() != 2

    criterion = ECCriterion(
        matcher=types.SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={},
        losses=[],
        num_classes=1,
    )
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    first = criterion.loss_labels_focal(outputs, targets, indices, num_boxes=1)['loss_focal']
    outputs['pred_logits'][0, 2, 0] = -20.0
    second = criterion.loss_labels_focal(outputs, targets, indices, num_boxes=1)['loss_focal']
    assert torch.allclose(first, second)


def test_fully_invisible_pose_keeps_visibility_loss_but_missing_pose_is_ignored():
    criterion = ECCriterion(
        matcher=types.SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={},
        losses=[],
        num_classes=1,
    )
    outputs = {
        'pred_boxes': torch.tensor([[[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]]]),
        'pred_keypoints': torch.zeros(1, 2, 2, 2),
        'pred_visibility': torch.tensor([[[4.0, 4.0], [-10.0, -10.0]]]),
    }
    targets = [{
        'boxes': outputs['pred_boxes'][0].clone(),
        'keypoints': torch.zeros(2, 2, 3),
        'pose_state': torch.tensor([1, 0]),
    }]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    first = criterion.loss_pose(outputs, targets, indices, num_boxes=2)
    outputs['pred_visibility'][0, 1] = 10.0
    second = criterion.loss_pose(outputs, targets, indices, num_boxes=2)
    assert first['loss_visibility'] > 0
    assert torch.allclose(first['loss_visibility'], second['loss_visibility'])
    assert first['loss_keypoint'] == 0
    assert first['loss_direction'] == 0


def test_density_valid_slots_are_fixed_shape_and_train_eval_identical():
    model = ECDet.__new__(ECDet)
    nn.Module.__init__(model)
    model.decoder = types.SimpleNamespace(num_queries=8)
    model.density_budget_ratio = 1.25
    model.density_budget_padding = 1
    model.min_density_queries = 2
    density = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    model.train()
    train_mask = model._predict_valid_query_mask(density)
    model.eval()
    eval_mask = model._predict_valid_query_mask(density)
    assert train_mask.shape == (1, 8)
    assert torch.equal(train_mask, eval_mask)
    assert train_mask.sum().item() == 4


def test_formal_contract_has_normative_m1_shape_and_switches():
    config_path = Path(__file__).resolve().parents[2] / 'configs' / 'bee_e' / 'e_formal_contract_1280.yml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert config['eval_spatial_size'] == [1280, 1280]
    assert config['ViTAdapter']['use_native_p2'] is True
    assert config['ViTAdapter']['proj_dim'] == 256
    assert config['ECTransformer']['hidden_dim'] == 256
    assert config['ECTransformer']['num_queries'] == 768
    assert config['ECTransformer']['endpoint_decoder_type'] == 'decomposed'
    assert config['ECTransformer']['endpoint_sampling_points'] == 8


def test_formal_contract_keys_bind_to_real_modules(monkeypatch):
    monkeypatch.setattr(ViTAdapter, '_load_weights', lambda self, path: None)
    config_path = Path(__file__).resolve().parents[2] / 'configs' / 'bee_e' / 'e2_formal_joint_1280.yml'
    model = YAMLConfig(str(config_path)).model
    assert model.backbone.feature_sources[0] == 'conv_stem_stride4'
    assert model.decoder.endpoint_decoder_type == 'decomposed'
    assert model.decoder.num_queries == 768
    assert model.decoder.hidden_dim == 256


def test_formal_contract_end_to_end_fixed_slot_smoke(monkeypatch):
    monkeypatch.setattr(ViTAdapter, '_load_weights', lambda self, path: None)
    config_path = Path(__file__).resolve().parents[2] / 'configs' / 'bee_e' / 'e2_formal_joint_1280.yml'
    config = YAMLConfig(str(config_path))
    config.yaml_cfg['eval_spatial_size'] = [128, 128]
    model = config.model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 128, 128))
    assert outputs['pred_logits'].shape[:2] == (1, 768)
    assert outputs['pred_boxes'].shape == (1, 768, 4)
    assert outputs['pred_keypoints'].shape == (1, 768, 2, 2)
    assert outputs['pred_visibility'].shape == (1, 768, 2)
    assert outputs['pred_query_valid'].shape == (1, 768)
    assert outputs['pred_query_valid'].all()


def test_formal_contract_training_outputs_and_losses_are_finite(monkeypatch):
    monkeypatch.setattr(ViTAdapter, '_load_weights', lambda self, path: None)
    config_path = Path(__file__).resolve().parents[2] / 'configs' / 'bee_e' / 'e2_formal_joint_1280.yml'
    config = YAMLConfig(str(config_path))
    config.yaml_cfg['eval_spatial_size'] = [128, 128]
    model = config.model.train()
    criterion = config.criterion
    targets = [{
        'labels': torch.tensor([0]),
        'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        'keypoints': torch.tensor([[[0.46, 0.5, 2.0], [0.54, 0.5, 2.0]]]),
        'pose_state': torch.tensor([2]),
        'pose_mask': torch.tensor([1.0]),
    }]
    outputs = model(torch.randn(1, 3, 128, 128), targets)
    losses = criterion(outputs, targets)
    assert len(outputs['aux_outputs']) == 3
    assert outputs['pred_query_valid'].shape == (1, 768)
    assert losses
    assert all(torch.isfinite(loss).item() for loss in losses.values())
