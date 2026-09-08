import torch

from engine.tracking import (
    BeeBehaviorAnalyzer,
    PoseMotionTracker,
    TrackerConfig,
    TrackingMetricsAccumulator,
    validate_result_schema,
)
from engine.edgecrafter.pose_metrics import BeeQueryMetrics
from engine.edgecrafter.postprocessor import PostProcessor


def _detections(x_offset=0.0, reversed_pose=False, score=0.9):
    keypoints = torch.tensor([[[12.0 + x_offset, 20.0], [28.0 + x_offset, 20.0]]])
    if reversed_pose:
        keypoints = keypoints.flip(1)
    return {
        'boxes': torch.tensor([[10.0 + x_offset, 10.0, 30.0 + x_offset, 30.0]]),
        'scores': torch.tensor([score]),
        'quality': torch.tensor([0.8]),
        'keypoints': keypoints,
        'scene_region': torch.tensor([1]),
        'query_indices': torch.tensor([17]),
    }


def test_bytetrack_ocsort_keeps_id_and_dp_corrects_head_tail():
    tracker = PoseMotionTracker(TrackerConfig(min_hits=1, max_missed=2))
    first = tracker.update(_detections(), frame_index=0)
    second = tracker.update(_detections(2.0, reversed_pose=True), frame_index=1)
    assert first[0]['track_id'] == second[0]['track_id']
    assert second[0]['query_index'] == 17
    assert second[0]['keypoints'][0, 0] < second[0]['keypoints'][1, 0]
    assert abs(float(tracker.tracks[0].state[2])) > 0


def test_low_score_second_stage_updates_existing_track_but_creates_no_new_track():
    tracker = PoseMotionTracker(TrackerConfig(min_hits=1))
    tracker.update(_detections(), frame_index=0)
    result = tracker.update(_detections(1.0, score=0.3), frame_index=1)
    assert len(result) == 1
    assert result[0]['track_id'] == 1


def test_tracking_metrics_report_perfect_sequence_and_bins():
    tracker = PoseMotionTracker(TrackerConfig(min_hits=1))
    metrics = TrackingMetricsAccumulator(iou_threshold=0.5)
    for frame in range(3):
        predictions = tracker.update(_detections(float(frame)), frame)
        metrics.update(predictions, {
            'boxes': _detections(float(frame))['boxes'],
            'track_ids': torch.tensor([4]),
            'keypoints': _detections(float(frame))['keypoints'],
        })
    summary = metrics.summarize()
    assert summary['mota'] == 1.0
    assert summary['idf1'] == 1.0
    assert summary['hota'] == 1.0
    assert summary['cast'] == 1.0
    assert summary['head_tail_swap_rate'] == 0.0
    assert 'density_low' in summary['bins']


def test_tracking_metrics_scope_reused_track_ids_by_sequence():
    metrics = TrackingMetricsAccumulator(iou_threshold=0.5)
    for sequence_id, prediction_id in ((10, 1), (20, 7)):
        metrics.update(
            [{
                'track_id': prediction_id,
                'box': torch.tensor([0.0, 0.0, 10.0, 10.0]),
                'keypoints': None,
            }],
            {
                'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                'track_ids': torch.tensor([5]),
                'sequence_id': sequence_id,
            },
        )
    assert metrics.summarize()['id_switches'] == 0


def test_idf1_uses_global_one_to_one_trajectory_assignment():
    metrics = TrackingMetricsAccumulator(iou_threshold=0.5)
    for frame, gt_id in enumerate((10, 20)):
        metrics.update(
            [{
                'track_id': 1,
                'box': torch.tensor([0.0, 0.0, 10.0, 10.0]),
                'keypoints': None,
            }],
            {
                'boxes': torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                'track_ids': torch.tensor([gt_id]),
                'sequence_id': 1,
                'frame_id': frame,
            },
        )
    assert metrics.summarize()['idf1'] == 0.5


def test_behavior_schema_units_and_outputs():
    tracker = PoseMotionTracker(TrackerConfig(min_hits=1))
    records = []
    for frame in range(3):
        records.extend(tracker.update(_detections(float(frame) * 2), frame))
    assert validate_result_schema(records)
    analyzer = BeeBehaviorAnalyzer(
        fps=10, pixels_per_mm=2, image_size=(100, 100),
        heatmap_size=(4, 4), entrance_region=1,
    )
    analyzer.add(records)
    result = analyzer.summarize()
    assert result['units']['speed'] == 'mm/s'
    assert len(result['individual']) == 1
    assert result['individual'][0]['mean_speed_mm_s'] > 0
    assert len(result['group']['density_heatmap']) == 4


def test_query_capacity_metrics_report_450_plus_shortfall():
    metrics = BeeQueryMetrics(score_threshold=0.05)
    metrics.update({
        'scores': torch.tensor([0.9]),
        'boxes': torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        'query_valid': torch.tensor([True]),
    }, {
        'orig_boxes': torch.zeros(500, 4),
        'boxes': torch.zeros(500, 4),
    }, torch.cat((torch.ones(400, dtype=torch.bool), torch.zeros(368, dtype=torch.bool))))
    result = metrics.summarize()
    assert result['450_plus_capacity_truncation_rate'] == 0.2
    assert result['query_utilization'] == 400 / 768


def test_postprocessor_ranking_uses_quality_validity_and_traces_query_index():
    processor = PostProcessor(num_classes=1, num_top_queries=2)
    results = processor({
        'pred_logits': torch.tensor([[[8.0], [7.0], [6.0]]]),
        'pred_boxes': torch.full((1, 3, 4), 0.5),
        'pred_quality': torch.tensor([[-8.0, 8.0, 8.0]]),
        'pred_query_valid': torch.tensor([[True, True, False]]),
    }, torch.tensor([[100, 100]]))
    assert results[0]['query_indices'].tolist() == [1, 0]
    assert results[0]['query_valid'].all()
