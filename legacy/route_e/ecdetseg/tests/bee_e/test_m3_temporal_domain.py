import sys
import types
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as VF
import yaml
from PIL import Image


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

from engine.data.dataloader import DomainBalancedSampler
from engine.data.dataset.coco_dataset import CocoDetection, ConvertCocoPolysToMask
from engine.data.transforms._transforms import (
    BeeDirectionRotation,
    BeeRGBSensorAugment,
    BeeTargetCenteredCrop,
    IRSensorAugment,
    PrepareTemporalFrames,
    RandomHorizontalFlipWithKeypoints,
)
from engine.data._misc import convert_to_tv_tensor
from engine.edgecrafter.criterion import ECCriterion
from engine.edgecrafter.modeling import ECDet


class _TemporalBackbone(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.channels = channels
        self.full_calls = []
        self.shallow_calls = []

    def forward(self, image, return_shallow=False):
        self.full_calls.append(tuple(image.shape))
        batch, _, height, width = image.shape
        full = [
            torch.ones(batch, self.channels, height // stride, width // stride)
            for stride in (4, 8, 16, 32)
        ]
        if return_shallow:
            return full, [feature.clone() for feature in full[:2]]
        return full

    def forward_shallow_history(self, image):
        self.shallow_calls.append(tuple(image.shape))
        batch, _, height, width = image.shape
        return [
            torch.ones(batch, self.channels, height // stride, width // stride)
            for stride in (4, 8)
        ]


class _Decoder(nn.Module):
    num_queries = 6

    def forward(self, features, targets=None, **kwargs):
        batch = features[0].shape[0]
        return {
            'pred_logits': torch.zeros(batch, self.num_queries, 1),
            'pred_boxes': torch.zeros(batch, self.num_queries, 4),
        }


def test_history_frames_use_shallow_p2_p3_but_current_uses_full_backbone():
    backbone = _TemporalBackbone()
    model = ECDet(
        backbone=backbone,
        encoder=nn.Identity(),
        decoder=_Decoder(),
        enable_temporal=True,
        temporal_frames=5,
        use_temporal_difference=True,
        use_frequency_gate=True,
        temporal_feature_levels=2,
    ).eval()
    model(torch.randn(2, 5, 3, 64, 64))
    assert backbone.full_calls == [(2, 3, 64, 64)]
    assert backbone.shallow_calls == [(8, 3, 64, 64)]


def test_temporal_p3_residual_preserves_current_vit_semantics():
    class SemanticBackbone(_TemporalBackbone):
        def forward(self, image, return_shallow=False):
            batch, _, height, width = image.shape
            full = [
                torch.full((batch, self.channels, height // stride, width // stride), value)
                for stride, value in zip((4, 8, 16, 32), (2.0, 10.0, 20.0, 30.0))
            ]
            shallow = [
                torch.full_like(full[0], 2.0),
                torch.full_like(full[1], 2.0),
            ]
            return (full, shallow) if return_shallow else full

        def forward_shallow_history(self, image):
            batch, _, height, width = image.shape
            return [
                torch.full((batch, self.channels, height // stride, width // stride), 2.0)
                for stride in (4, 8)
            ]

    class CaptureDecoder(_Decoder):
        def forward(self, features, targets=None, **kwargs):
            self.features = features
            return super().forward(features, targets, **kwargs)

    decoder = CaptureDecoder()
    model = ECDet(
        backbone=SemanticBackbone(), encoder=nn.Identity(), decoder=decoder,
        enable_temporal=True, temporal_frames=5,
        temporal_feature_levels=2,
    ).eval()
    model(torch.randn(1, 5, 3, 64, 64))
    assert torch.allclose(decoder.features[1], torch.full_like(decoder.features[1], 10.0))


def test_single_frame_replication_has_explicit_mask_and_zero_temporal_difference():
    model = ECDet(
        backbone=_TemporalBackbone(), encoder=nn.Identity(), decoder=_Decoder(),
        enable_temporal=True, temporal_frames=5,
        use_temporal_difference=True, use_frequency_gate=True,
    ).eval()
    image = torch.randn(1, 3, 64, 64)
    clip = image[:, None].repeat(1, 5, 1, 1, 1)
    single_mask = torch.tensor([[False, False, False, False, True]])
    all_mask = torch.ones(1, 5, dtype=torch.bool)
    single = model(clip, temporal_valid_mask=single_mask)
    repeated = model(clip, temporal_valid_mask=all_mask)
    assert torch.equal(single['pred_logits'], repeated['pred_logits'])
    assert abs(single['temporal_valid_ratio'].item() - 0.2) < 1e-6


def test_domain_motion_router_ignores_invalid_history_padding():
    model = ECDet(
        backbone=_TemporalBackbone(), encoder=nn.Identity(), decoder=_Decoder(),
        enable_temporal=True, temporal_frames=5,
        enable_domain_adapter=True, domain_channels=8, domain_levels=2,
    ).eval()
    clip = torch.zeros(1, 5, 3, 64, 64)
    clip[:, -1] = 1.0
    only_current = torch.tensor([[False, False, False, False, True]])
    outputs = model(clip, temporal_valid_mask=only_current)
    assert outputs['domain_router_features'][0, -1] == 0


def test_ir_hard_negative_prior_ignores_invalid_history_padding():
    model = ECDet(
        backbone=_TemporalBackbone(), encoder=nn.Identity(), decoder=_Decoder(),
        enable_temporal=True, temporal_frames=5,
    ).train()
    clip = torch.zeros(1, 5, 3, 32, 32)
    clip[:, :-1, :, 8:24, 8:24] = 100.0
    target = [{
        'domain_id': torch.tensor([1]),
        'hard_negative': torch.tensor([True]),
        'boxes': torch.zeros(0, 4),
    }]
    only_current = torch.tensor([[False, False, False, False, True]])
    outputs = model(clip, targets=target, temporal_valid_mask=only_current)
    assert outputs['pred_ir_hard_negative_prior'].count_nonzero() == 0


def test_history_shallow_path_keeps_training_gradients():
    class Backbone(_TemporalBackbone):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, image, return_shallow=False):
            result = super().forward(image, return_shallow=return_shallow)
            if return_shallow:
                full, shallow = result
                return (
                    [self.scale * value for value in full],
                    [self.scale * value for value in shallow],
                )
            return [self.scale * value for value in result]

        def forward_shallow_history(self, image):
            return [self.scale * value for value in super().forward_shallow_history(image)]

    class Decoder(_Decoder):
        def forward(self, features, targets=None, **kwargs):
            result = super().forward(features, targets, **kwargs)
            result['pred_logits'] = result['pred_logits'] + features[0].mean()
            return result

    backbone = Backbone()
    model = ECDet(
        backbone=backbone, encoder=nn.Identity(), decoder=Decoder(),
        enable_temporal=True, temporal_frames=5,
    ).train()
    model(torch.randn(1, 5, 3, 32, 32))['pred_logits'].sum().backward()
    assert backbone.scale.grad is not None
    assert backbone.scale.grad.abs() > 0


def test_domain_experts_are_dwconv_bottlenecks_with_prototypes():
    model = ECDet(
        backbone=_TemporalBackbone(),
        encoder=nn.Identity(),
        decoder=_Decoder(),
        enable_domain_adapter=True,
        domain_channels=8,
        domain_levels=4,
    ).eval()
    outputs = model(torch.randn(2, 3, 64, 64))
    depthwise = model.domain_adapters[0][0].block[0]
    assert depthwise.kernel_size == (3, 3)
    assert depthwise.groups == depthwise.in_channels == 8
    assert model.domain_prototypes.shape == (2, 8)
    assert outputs['pred_domain_logits'].shape == (2, 2)
    assert outputs['pred_domain_features'].shape == (2, 8)


def test_domain_prototype_and_track_geometry_losses_are_finite():
    criterion = ECCriterion(
        matcher=types.SimpleNamespace(mask_point_sample_ratio=None),
        weight_dict={}, losses=[], num_classes=1,
    )
    outputs = {
        'pred_domain_logits': torch.tensor([[1.0, -1.0]]),
        'pred_domain_features': torch.randn(1, 8),
        'domain_prototypes': torch.randn(2, 8),
        'pred_track_geometry': torch.tensor([[[0.1, 0.0, 0.0, 0.0, 1.0, 0.0]]]),
    }
    targets = [{
        'domain_id': torch.tensor([0]),
        'boxes': torch.zeros(1, 4),
        'track_geometry': torch.tensor([[0.1, 0.0, 0.0, 0.0, 1.0, 0.0]]),
        'track_geometry_mask': torch.tensor([True]),
    }]
    domain = criterion.loss_domain(outputs, targets, [], 1)
    track = criterion.loss_track(
        outputs, targets, [(torch.tensor([0]), torch.tensor([0]))], 1
    )
    assert all(torch.isfinite(value) for value in {**domain, **track}.values())
    assert track['loss_track_center_scale'] == 0
    assert track['loss_track_axis'] == 0


def test_track_geometry_schema_is_preserved_per_instance():
    converter = ConvertCocoPolysToMask(return_masks=False)
    _, target = converter(Image.new('RGB', (32, 32)), {
        'image_id': 1,
        'annotations': [{
            'bbox': [1, 1, 8, 8], 'category_id': 1, 'area': 64,
            'track_geometry': [0.1, 0.2, 0.0, 0.0, 1.0, 0.0],
        }],
    })
    assert target['track_geometry'].shape == (1, 6)
    assert target['track_geometry_mask'].tolist() == [True]


def test_domain_sampler_balances_rgb_ir_and_includes_ir_hard_negatives():
    class Dataset:
        domain_indices = {0: [0, 1], 1: [2, 3]}
        hard_negative_indices = {0: [], 1: [3]}

        def __len__(self):
            return 20

    dataset = Dataset()
    sampler = DomainBalancedSampler(dataset, hard_negative_ratio=1.0, seed=1)
    indices = list(iter(sampler))
    assert len(indices) == 6  # four base-cover samples plus one correction pair
    assert set(indices[:4]) == {0, 1, 2, 3}
    assert all(index in (0, 1) for index in indices[0::2])
    assert indices[-1] == 3


def test_domain_sampler_pairs_rgb_ir_across_distributed_ranks_per_microbatch():
    class Dataset:
        domain_indices = {0: [0, 1], 1: [2, 3]}
        hard_negative_indices = {0: [], 1: []}
        labelled_indices = domain_indices
        unlabelled_indices = {0: [], 1: []}

        def __len__(self):
            return 20

    rank_sequences = []
    for rank in (0, 1):
        with mock.patch('torch.distributed.is_available', return_value=True), \
             mock.patch('torch.distributed.is_initialized', return_value=True), \
             mock.patch('torch.distributed.get_world_size', return_value=2), \
             mock.patch('torch.distributed.get_rank', return_value=rank):
            rank_sequences.append(list(DomainBalancedSampler(Dataset(), seed=1)))
    assert all(len(indices) == 3 for indices in rank_sequences)
    for rank_zero, rank_one in zip(*rank_sequences):
        assert rank_zero in (0, 1)
        assert rank_one in (2, 3)


def test_strict_schema_rejects_zero_filled_missing_pose():
    dataset = {
        'schema_version': 2,
        'images': [{
            'id': 1, 'file_name': 'a.png', 'domain': 'RGB', 'sensor_id': 'cam',
            'sequence_id': 'seq', 'frame_id': 0, 'track_supervised': False,
        }],
        'annotations': [{
            'id': 1, 'image_id': 1, 'bbox': [0, 0, 4, 4], 'category_id': 1,
            'pose_state': 0, 'pose_mask': False, 'keypoints': [0, 0, 0, 0, 0, 0],
            'track_id': -1, 'track_mask': False,
            'supervision_mask': [True, False, False, True],
            'annotator_id': 'human', 'inter_group_quality': 1.0,
            'intra_group_quality': 1.0, 'hierarchy_quality': 1.0,
            'pose_quality': 0.0, 'track_quality': 0.0,
        }],
    }
    try:
        CocoDetection.validate_schema_dict(dataset)
    except ValueError as error:
        assert 'zero-filled keypoints' in str(error)
    else:
        raise AssertionError('strict schema accepted a zero-filled missing pose')


def test_ir_sensor_augment_is_domain_gated_and_records_shared_parameters():
    augment = IRSensorAugment(p=1.0, blur_p=0.0, hot_pixel_p=0.0)
    image = Image.new('RGB', (16, 16), color=(100, 100, 100))
    rgb_image, rgb_target = augment((image, {'domain_id': torch.tensor([0])}))
    ir_image, ir_target = augment((image, {'domain_id': torch.tensor([1])}))
    assert rgb_image is image
    assert 'ir_sensor_augment' not in rgb_target
    assert ir_image.size == image.size
    assert ir_target['ir_sensor_augment'].shape == (5,)


def test_rgb_crop_rotation_flip_record_parameters_for_all_history_frames(tmp_path):
    path = tmp_path / 'frame.png'
    Image.new('RGB', (32, 32), color=(100, 120, 140)).save(path)
    target = {
        'domain_id': torch.tensor([0]),
        'boxes': convert_to_tv_tensor(
            torch.tensor([[8.0, 8.0, 24.0, 24.0]]), key='boxes',
            box_format='XYXY', spatial_size=(32, 32),
        ),
        'labels': torch.tensor([0]),
        'keypoints': torch.tensor([[[10.0, 16.0, 2.0], [22.0, 16.0, 2.0]]]),
        'temporal_paths': [str(path)] * 5,
    }
    image = Image.open(path).convert('RGB')
    image, target = BeeRGBSensorAugment(p=1.0)((image, target))
    image, target = BeeTargetCenteredCrop(p=1.0, scale=(0.8, 0.8))((image, target))
    image, target = BeeDirectionRotation(p=1.0, degrees=5)((image, target))
    image, target = RandomHorizontalFlipWithKeypoints(p=1.0)((image, target))
    assert 'rgb_sensor_augment' in target
    assert 'temporal_crop' in target
    assert 'temporal_rotation' in target
    assert target['temporal_horizontal_flip']
    current = VF.pil_to_tensor(VF.resize(image, [32, 32])).float() / 255.0
    clip, _ = PrepareTemporalFrames(size=(32, 32))((current, target))
    assert clip.shape == (5, 3, 32, 32)


def test_rotation_and_flip_transform_track_geometry_vectors():
    image = Image.new('RGB', (32, 32), color=(100, 120, 140))
    target = {
        'boxes': convert_to_tv_tensor(
            torch.tensor([[8.0, 8.0, 24.0, 24.0]]), key='boxes',
            box_format='XYXY', spatial_size=(32, 32),
        ),
        'track_geometry': torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
    }
    with mock.patch('torch.rand', side_effect=[torch.tensor(0.0), torch.tensor(1.0)]):
        _, rotated = BeeDirectionRotation(p=1.0, degrees=90)((image, target))
    assert torch.allclose(
        rotated['track_geometry'][0, [0, 1, 4, 5]],
        torch.tensor([0.0, -1.0, 0.0, -1.0]), atol=1e-6,
    )
    flip_target = dict(target)
    flip_target['track_geometry'] = target['track_geometry'].clone()
    _, flipped = RandomHorizontalFlipWithKeypoints(p=1.0)((image, flip_target))
    assert torch.allclose(
        flipped['track_geometry'][0, [0, 1, 4, 5]],
        torch.tensor([-1.0, 0.0, -1.0, 0.0]), atol=1e-6,
    )


def test_formal_m3_contract_is_explicit():
    root = Path(__file__).resolve().parents[2] / 'configs' / 'bee_e'
    base = yaml.safe_load((root / 'e_formal_contract_1280.yml').read_text(encoding='utf-8'))
    query_stage = yaml.safe_load((root / 'e3_formal_query_1280.yml').read_text(encoding='utf-8'))
    stage = yaml.safe_load((root / 'e4_formal_temporal_domain_1280.yml').read_text(encoding='utf-8'))
    assert stage['ECDet']['enable_temporal'] is True
    assert base['ECDet']['temporal_feature_levels'] == 2
    assert stage['ECDet']['enable_domain_adapter'] is True
    assert base['train_dataloader']['domain_balanced'] is True
    assert base['train_dataloader']['dataset']['strict_schema'] is True
    assert query_stage['train_dataloader']['hard_negative_ratio'] > 0
