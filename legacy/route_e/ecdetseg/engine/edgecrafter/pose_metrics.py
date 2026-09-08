import math

import torch
import torchvision


class BeeDetectionMetrics:
    """Single-class COCO-style AP accumulator for RGB/IR slices."""

    def __init__(self, score_threshold=0.001):
        self.score_threshold = score_threshold
        self.thresholds = [0.50 + 0.05 * index for index in range(10)]
        self.records = {threshold: [] for threshold in self.thresholds}
        self.gt_instances = 0

    def update(self, prediction, target):
        scores = prediction['scores']
        boxes = prediction['boxes']
        keep = scores >= self.score_threshold
        scores = scores[keep]
        boxes = boxes[keep]
        gt_boxes = target.get('orig_boxes', target['boxes'])
        self.gt_instances += len(gt_boxes)
        order = scores.argsort(descending=True)
        scores = scores[order]
        boxes = boxes[order]
        ious = torchvision.ops.box_iou(boxes, gt_boxes) if len(boxes) and len(gt_boxes) else None
        for threshold in self.thresholds:
            used = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
            for prediction_index, score in enumerate(scores):
                matched = False
                if ious is not None and len(gt_boxes):
                    available = ious[prediction_index].clone()
                    available[used] = -1
                    best_iou, gt_index = available.max(dim=0)
                    if float(best_iou) >= threshold:
                        used[gt_index] = True
                        matched = True
                self.records[threshold].append((float(score), float(matched)))

    def empty_copy(self):
        return type(self)(score_threshold=self.score_threshold)

    def state_dict(self):
        return {
            'gt_instances': self.gt_instances,
            'records': self.records,
        }

    def merge_state(self, state):
        self.gt_instances += int(state['gt_instances'])
        for threshold, records in state['records'].items():
            self.records[float(threshold)].extend(records)

    def _average_precision(self, threshold):
        if self.gt_instances == 0:
            return 0.0, 0.0
        records = sorted(self.records[threshold], key=lambda item: item[0], reverse=True)
        if not records:
            return 0.0, 0.0
        true_positive = torch.tensor([item[1] for item in records]).cumsum(0)
        false_positive = torch.tensor([1.0 - item[1] for item in records]).cumsum(0)
        recall = true_positive / self.gt_instances
        precision = true_positive / (true_positive + false_positive).clamp_min(1e-8)
        sampled = []
        for recall_level in torch.linspace(0, 1, 101):
            valid = recall >= recall_level
            sampled.append(float(precision[valid].max()) if valid.any() else 0.0)
        return sum(sampled) / len(sampled), float(recall[-1])

    def summarize(self):
        values = [self._average_precision(threshold) for threshold in self.thresholds]
        return {
            'map': sum(item[0] for item in values) / len(values),
            'ap50': values[0][0],
            'recall': sum(item[1] for item in values) / len(values),
            'gt_instances': self.gt_instances,
        }


class BeePoseMetrics:
    """对 IoU>=0.5 的检测匹配统计联合模型姿态指标。"""

    def __init__(self, score_threshold=0.05, iou_threshold=0.5):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.matched_instances = 0
        self.gt_pose_instances = 0
        self.keypoint_count = 0
        self.normalized_error_sum = 0.0
        self.pck05_count = 0
        self.pck10_count = 0
        self.pck20_count = 0
        self.direction_correct = 0
        self.angle_count = 0
        self.angle_sum = 0.0
        self.angle_over_45 = 0

    def update(self, prediction, target):
        if 'keypoints' not in prediction or 'orig_keypoints' not in target:
            return

        pose_mask = target.get('pose_mask', torch.ones(len(target['orig_boxes']), device=target['orig_boxes'].device)) > 0
        self.gt_pose_instances += int(pose_mask.sum().item())

        keep = prediction['scores'] >= self.score_threshold
        pred_boxes = prediction['boxes'][keep]
        pred_keypoints = prediction['keypoints'][keep]
        pred_scores = prediction['scores'][keep]
        gt_boxes = target.get('orig_boxes', target['boxes'])
        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            return

        ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)
        used_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
        matched = []
        for pred_idx in pred_scores.argsort(descending=True):
            available = ious[pred_idx].clone()
            available[used_gt] = -1
            best_iou, gt_idx = available.max(dim=0)
            if best_iou < self.iou_threshold:
                continue
            used_gt[gt_idx] = True
            if pose_mask[gt_idx]:
                matched.append((int(pred_idx.item()), int(gt_idx.item())))

        if not matched:
            return

        for pred_idx, gt_idx in matched:
            pred = pred_keypoints[pred_idx]
            target_keypoints = target['orig_keypoints'][gt_idx]
            visible = target_keypoints[:, 2] > 0
            if not visible.any():
                continue

            target_xy = target_keypoints[:, :2]
            box = gt_boxes[gt_idx]
            diagonal = torch.linalg.vector_norm(box[2:] - box[:2]).clamp_min(1.0)
            normalized_errors = torch.linalg.vector_norm(pred[visible] - target_xy[visible], dim=-1) / diagonal
            self.keypoint_count += int(normalized_errors.numel())
            self.normalized_error_sum += float(normalized_errors.sum().item())
            self.pck05_count += int((normalized_errors <= 0.05).sum().item())
            self.pck10_count += int((normalized_errors <= 0.10).sum().item())
            self.pck20_count += int((normalized_errors <= 0.20).sum().item())
            self.matched_instances += 1

            if len(pred) >= 2 and visible[:2].all():
                direct = torch.linalg.vector_norm(pred[:2] - target_xy[:2], dim=-1).sum()
                swapped = torch.linalg.vector_norm(pred[:2].flip(0) - target_xy[:2], dim=-1).sum()
                self.direction_correct += int((direct <= swapped).item())

                pred_axis = pred[1] - pred[0]
                target_axis = target_xy[1] - target_xy[0]
                cosine = torch.dot(pred_axis, target_axis) / (
                    torch.linalg.vector_norm(pred_axis).clamp_min(1e-6)
                    * torch.linalg.vector_norm(target_axis).clamp_min(1e-6)
                )
                angle = math.degrees(math.acos(float(cosine.clamp(-1, 1).item())))
                self.angle_count += 1
                self.angle_sum += angle
                self.angle_over_45 += int(angle > 45.0)

    def empty_copy(self):
        return type(self)(
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
        )

    def state_dict(self):
        names = (
            'matched_instances', 'gt_pose_instances', 'keypoint_count',
            'normalized_error_sum', 'pck05_count', 'pck10_count',
            'pck20_count', 'direction_correct', 'angle_count',
            'angle_sum', 'angle_over_45',
        )
        return {name: getattr(self, name) for name in names}

    def merge_state(self, state):
        for name, value in state.items():
            setattr(self, name, getattr(self, name) + value)

    def summarize(self):
        keypoint_denominator = max(self.keypoint_count, 1)
        angle_denominator = max(self.angle_count, 1)
        gt_denominator = max(self.gt_pose_instances, 1)
        return {
            'matched_instances': self.matched_instances,
            'gt_pose_instances': self.gt_pose_instances,
            'joint_pose_recall_iou50': self.matched_instances / gt_denominator,
            'nme_bbox_diagonal': self.normalized_error_sum / keypoint_denominator,
            'pck@0.05': self.pck05_count / keypoint_denominator,
            'pck@0.10': self.pck10_count / keypoint_denominator,
            'pck@0.20': self.pck20_count / keypoint_denominator,
            'orientation_accuracy': self.direction_correct / angle_denominator,
            'head_tail_swap_rate': 1.0 - self.direction_correct / angle_denominator,
            'pose_decidable_rate': self.angle_count / gt_denominator,
            'mean_angle_error_deg': self.angle_sum / angle_denominator,
            'angle_over_45deg': self.angle_over_45 / angle_denominator,
        }


class BeeQueryMetrics:
    """Capacity, unmatched-GT and duplicate-query diagnostics by density bin."""

    def __init__(self, score_threshold=0.05, match_iou=0.5, duplicate_iou=0.7):
        self.score_threshold = score_threshold
        self.match_iou = match_iou
        self.duplicate_iou = duplicate_iou
        self.frames = 0
        self.slots = 0
        self.active = 0
        self.gt = 0
        self.unmatched_gt = 0
        self.duplicates = 0
        self.capacity_shortfall = 0
        self.bins = {
            name: {'frames': 0, 'gt': 0, 'unmatched_gt': 0, 'capacity_shortfall': 0}
            for name in ('0_49', '50_199', '200_449', '450_plus')
        }

    @staticmethod
    def _bin(count):
        if count < 50:
            return '0_49'
        if count < 200:
            return '50_199'
        if count < 450:
            return '200_449'
        return '450_plus'

    def update(self, prediction, target, valid_query_mask):
        gt_boxes = target.get('orig_boxes', target['boxes'])
        keep = prediction['scores'] >= self.score_threshold
        if 'query_valid' in prediction:
            keep &= prediction['query_valid'].bool()
        boxes = prediction['boxes'][keep]
        active = int(valid_query_mask.sum())
        gt_count = len(gt_boxes)
        unmatched = gt_count
        if len(boxes) and gt_count:
            iou = torchvision.ops.box_iou(gt_boxes, boxes)
            unmatched = int((iou.max(dim=1).values < self.match_iou).sum())
        duplicates = 0
        if len(boxes) > 1:
            pair_iou = torchvision.ops.box_iou(boxes, boxes)
            pair_iou = torch.triu(pair_iou, diagonal=1)
            duplicates = int((pair_iou >= self.duplicate_iou).sum())
        shortfall = max(gt_count - active, 0)
        self.frames += 1
        self.slots += valid_query_mask.numel()
        self.active += active
        self.gt += gt_count
        self.unmatched_gt += unmatched
        self.duplicates += duplicates
        self.capacity_shortfall += shortfall
        values = self.bins[self._bin(gt_count)]
        values['frames'] += 1
        values['gt'] += gt_count
        values['unmatched_gt'] += unmatched
        values['capacity_shortfall'] += shortfall

    def empty_copy(self):
        return type(self)(
            score_threshold=self.score_threshold,
            match_iou=self.match_iou,
            duplicate_iou=self.duplicate_iou,
        )

    def state_dict(self):
        names = (
            'frames', 'slots', 'active', 'gt', 'unmatched_gt',
            'duplicates', 'capacity_shortfall',
        )
        state = {name: getattr(self, name) for name in names}
        state['bins'] = {
            name: dict(values) for name, values in self.bins.items()
        }
        return state

    def merge_state(self, state):
        for name, value in state.items():
            if name == 'bins':
                for bin_name, bin_values in value.items():
                    for metric_name, metric_value in bin_values.items():
                        self.bins[bin_name][metric_name] += metric_value
            else:
                setattr(self, name, getattr(self, name) + value)

    def summarize(self):
        result = {
            'query_utilization': self.active / max(self.slots, 1),
            'unmatched_gt_rate': self.unmatched_gt / max(self.gt, 1),
            'duplicate_queries_per_frame': self.duplicates / max(self.frames, 1),
            'capacity_truncation_rate': self.capacity_shortfall / max(self.gt, 1),
        }
        for name, values in self.bins.items():
            result[f'{name}_unmatched_gt_rate'] = values['unmatched_gt'] / max(values['gt'], 1)
            result[f'{name}_capacity_truncation_rate'] = values['capacity_shortfall'] / max(values['gt'], 1)
        return result
