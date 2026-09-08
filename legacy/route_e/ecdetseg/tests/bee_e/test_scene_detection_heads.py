import torch
import torch.nn as nn

from engine.data.dataset.coco_dataset import CocoDetection
from engine.edgecrafter.decoder import ECTransformer, SceneRoutedHead
from engine.edgecrafter.postprocessor import PostProcessor
from engine.solver._solver import BaseSolver


def test_scene_head_routes_per_sample_and_keeps_both_branches_in_graph():
    routed = SceneRoutedHead(nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        routed.heads[0].weight.fill_(1.0)
        routed.heads[1].weight.fill_(3.0)

    inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
    output = routed(inputs, torch.tensor([0, 1]))
    assert torch.equal(output, torch.tensor([[3.0], [15.0]]))

    output.sum().backward()
    assert torch.equal(routed.heads[0].weight.grad, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(routed.heads[1].weight.grad, torch.tensor([[2.0, 3.0]]))


def test_scene_labels_follow_a_outdoor_b_indoor_contract():
    dataset = object.__new__(CocoDetection)
    dataset.strict_scene_schema = True
    dataset.default_scene = None
    assert dataset._scene_id({'scene': 'A'}) == 1
    assert dataset._scene_id({'scene': 'B'}) == 0
    assert dataset._scene_id({'environment': 'outdoor'}) == 1
    assert dataset._scene_id({'environment': 'indoor'}) == 0


def test_shared_bbox_state_is_copied_to_both_scene_heads():
    source = {'decoder.enc_bbox_head.layers.0.weight': torch.ones(2, 2)}
    current = {
        'decoder.enc_bbox_head.heads.0.layers.0.weight': torch.zeros(2, 2),
        'decoder.enc_bbox_head.heads.1.layers.0.weight': torch.zeros(2, 2),
    }
    expanded = BaseSolver._expand_scene_detection_heads(current, source)
    assert torch.equal(expanded[next(iter(current))], torch.ones(2, 2))
    assert torch.equal(expanded[list(current)[1]], torch.ones(2, 2))


def test_scene_specific_logical_query_capacity_keeps_indoor_fallback():
    decoder = object.__new__(ECTransformer)
    nn.Module.__init__(decoder)
    decoder.density_capacity_ratio = 1.0
    decoder.density_capacity_padding = 0.0
    decoder.uncertainty_capacity_scale = 0.0
    decoder.min_active_queries = 512
    decoder.stage_query_limit = 768
    decoder.query_capacity_by_scene = {
        'outdoor': {'min_active_queries': 64, 'stage_query_limit': 256},
    }
    budget = decoder._monotonic_query_budget(
        torch.tensor([20.0, 20.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0, 1]),
    )
    assert torch.equal(budget, torch.tensor([512, 64]))


def test_outdoor_postprocess_can_exclude_visibility_and_apply_scene_nms():
    postprocessor = PostProcessor(
        num_classes=1,
        num_top_queries=4,
        enable_calibrated_filter=True,
        score_exponents_by_domain={'rgb': [1.0, 1.0, 1.0]},
        score_thresholds_by_domain={'rgb': 0.0},
        score_exponents_by_scene={'outdoor': [1.0, 1.0, 0.0]},
        num_top_queries_by_scene={'outdoor': 2},
        box_nms_iou_by_scene={'outdoor': 0.70},
    )
    logits = torch.tensor([[[4.0], [3.0], [2.0], [1.0]]]).repeat(2, 1, 1)
    boxes = torch.tensor([[
        [0.50, 0.50, 0.20, 0.20],
        [0.50, 0.50, 0.20, 0.20],
        [0.20, 0.20, 0.10, 0.10],
        [0.80, 0.80, 0.10, 0.10],
    ]]).repeat(2, 1, 1)
    outputs = {
        'pred_logits': logits,
        'pred_boxes': boxes,
        'pred_quality': torch.zeros(2, 4),
        'pred_visibility': torch.full((2, 4, 2), -20.0),
        'pred_query_valid': torch.ones(2, 4, dtype=torch.bool),
        'route_domain_id': torch.zeros(2, dtype=torch.long),
        'route_scene_id': torch.tensor([0, 1]),
    }
    indoor, outdoor = postprocessor(outputs, torch.tensor([[100, 100], [100, 100]]))
    assert len(indoor['scores']) == 4
    assert float(indoor['scores'].max()) < 1e-6
    assert len(outdoor['scores']) == 1
    assert float(outdoor['scores'].max()) > 0.40
    assert int(outdoor['scene_box_nms_removed']) == 1
