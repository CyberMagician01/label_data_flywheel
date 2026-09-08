import time

import pytest
import torch

from tools.bee_e.export_onnx import BeeEExportWrapper, single_frame_as_clip
from tools.bee_e.validate_onnx import EndToEndProfiler, compare_outputs


class _DeployModel(torch.nn.Module):
    def forward(self, images, temporal_valid_mask=None):
        batch = images.shape[0]
        device = images.device
        return {
            'pred_logits': torch.zeros(batch, 4, 1, device=device),
            'pred_boxes': torch.full((batch, 4, 4), 0.5, device=device),
            'pred_keypoints': torch.full((batch, 4, 2, 2), 0.5, device=device),
            'pred_visibility': torch.zeros(batch, 4, 2, device=device),
            'pred_quality': torch.zeros(batch, 4, device=device),
            'pred_query_valid': torch.tensor(
                [[True, True, False, False]], device=device
            ).expand(batch, -1),
            'aux_outputs': [{'training_only': torch.ones(1, device=device)}],
        }


def test_export_wrapper_keeps_fixed_slots_and_removes_training_outputs():
    wrapper = BeeEExportWrapper(_DeployModel(), num_queries=4)
    outputs = wrapper(
        torch.zeros(1, 3, 3, 16, 16), torch.ones(1, 3, dtype=torch.bool)
    )
    assert len(outputs) == len(BeeEExportWrapper.output_names)
    assert outputs[1].shape == (1, 4, 4)
    assert outputs[2].shape == (1, 4, 2, 2)
    assert outputs[5].tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert outputs[-1][0, 2] == 0


def test_single_frame_is_repeated_into_same_three_frame_graph():
    image = torch.rand(2, 3, 8, 8)
    clip = single_frame_as_clip(image, 3)
    assert clip.shape == (2, 3, 3, 8, 8)
    assert torch.equal(clip[:, 0], clip[:, -1])
    with pytest.raises(ValueError, match='three frames'):
        single_frame_as_clip(image, 5)


def test_numerical_comparator_checks_boxes_endpoints_and_ranking():
    reference = BeeEExportWrapper(_DeployModel(), num_queries=4)(
        torch.zeros(1, 3, 3, 16, 16), torch.ones(1, 3, dtype=torch.bool)
    )
    candidate = [tensor.detach().numpy() for tensor in reference]
    report = compare_outputs(reference, candidate, topk=4)
    assert report['boxes_max_abs_error'] == 0
    assert report['keypoints_max_abs_error'] == 0
    assert report['topk_query_ranking_overlap'] == 1


def test_exported_query_score_uses_frozen_domain_ranking_exponents():
    contract = {
        'effective_stage': 'E-S5',
        'domains': {
            'rgb': {'ranking': {'score_exponents': [1.0, 1.0, 0.0]}},
            'ir': {'ranking': {'score_exponents': [2.0, 1.0, 0.0]}},
        },
    }
    wrapper = BeeEExportWrapper(
        _DeployModel(), num_queries=4, calibration_contract=contract,
    )
    scores = wrapper(
        torch.zeros(2, 3, 3, 16, 16),
        torch.ones(2, 3, dtype=torch.bool),
        domain_id=torch.tensor([0, 1]),
    )[-1]
    assert torch.all(scores[1, :2] < scores[0, :2])


def test_end_to_end_profiler_reports_required_percentiles():
    profiler = EndToEndProfiler()
    for _ in range(3):
        for stage in profiler.stages:
            profiler.measure(stage, lambda: time.sleep(0.00001))
    report = profiler.summarize()
    assert set(('mean_ms', 'p50_ms', 'p95_ms', 'p99_ms')) <= set(report['model'])
    assert 'passes_10_second_limit' in report['end_to_end']
