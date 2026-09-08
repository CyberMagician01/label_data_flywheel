"""Standard fixed-tracker MOTA/IDF1/HOTA and frame-context CAST metrics."""

from collections import Counter, defaultdict

import torch
import torchvision

from .pose_motion_tracker import _linear_sum_assignment


class TrackingMetricsAccumulator:
    def __init__(self, iou_threshold=0.5, cast_context_frames=(1, 5, 10, 20)):
        self.iou_threshold = iou_threshold
        self.cast_context_frames = tuple(int(value) for value in cast_context_frames)
        self.gt_detections = 0
        self.pred_detections = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.id_switches = 0
        self.fragments = 0
        self.iou_sum = 0.0
        self.head_tail_correct = 0
        self.head_tail_count = 0
        self.previous_prediction_for_gt = {}
        self.previous_matched_for_gt = {}
        self.pair_counts = Counter()
        self.gt_track_counts = Counter()
        self.bin_counts = defaultdict(lambda: Counter())
        self.frame_records = []
        self.matched_events = []
        self.sequence_next_frame = Counter()

    @staticmethod
    def _prediction_tensors(predictions, device):
        if not predictions:
            return (
                torch.zeros(0, 4, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                None,
            )
        boxes = torch.stack([prediction['box'].to(device) for prediction in predictions])
        ids = torch.tensor([prediction['track_id'] for prediction in predictions], device=device)
        if all(prediction.get('keypoints') is not None for prediction in predictions):
            keypoints = torch.stack([prediction['keypoints'].to(device) for prediction in predictions])
        else:
            keypoints = None
        return boxes, ids, keypoints

    @staticmethod
    def _density_bin(count):
        if count < 50:
            return 'density_low'
        if count < 200:
            return 'density_medium'
        return 'density_high'

    @staticmethod
    def _overlap_bin(gt_boxes):
        if len(gt_boxes) < 2:
            return 'overlap_low'
        overlaps = torchvision.ops.box_iou(gt_boxes, gt_boxes)
        overlaps.fill_diagonal_(0)
        value = float(overlaps.max(dim=1).values.mean())
        if value < 0.10:
            return 'overlap_low'
        if value < 0.30:
            return 'overlap_medium'
        return 'overlap_high'

    def update(self, predictions, target):
        gt_boxes = target['boxes']
        gt_ids = target['track_ids'].long()
        sequence_id = target.get('sequence_id', 0)
        if isinstance(sequence_id, torch.Tensor):
            sequence_id = int(sequence_id.item())
        frame_id = target.get('frame_id')
        if isinstance(frame_id, torch.Tensor):
            frame_id = int(frame_id.item())
        if frame_id is None:
            frame_id = self.sequence_next_frame[sequence_id]
        self.sequence_next_frame[sequence_id] = max(
            self.sequence_next_frame[sequence_id], int(frame_id) + 1
        )
        gt_keypoints = target.get('keypoints')
        pred_boxes, pred_ids, pred_keypoints = self._prediction_tensors(predictions, gt_boxes.device)
        self.frame_records.append({
            'sequence_id': sequence_id,
            'frame_id': int(frame_id),
            'gt_boxes': gt_boxes.detach().float().cpu().clone(),
            'gt_ids': gt_ids.detach().long().cpu().clone(),
            'pred_boxes': pred_boxes.detach().float().cpu().clone(),
            'pred_ids': pred_ids.detach().long().cpu().clone(),
        })
        self.gt_detections += len(gt_boxes)
        self.pred_detections += len(pred_boxes)
        for gt_id in gt_ids.tolist():
            self.gt_track_counts[(sequence_id, int(gt_id))] += 1

        matches = []
        if len(gt_boxes) and len(pred_boxes):
            iou = torchvision.ops.box_iou(gt_boxes, pred_boxes)
            rows, columns = _linear_sum_assignment(1.0 - iou)
            for row, column in zip(rows.tolist(), columns.tolist()):
                if float(iou[row, column]) >= self.iou_threshold:
                    matches.append((row, column, float(iou[row, column])))
        matched_gt = {row for row, _, _ in matches}
        matched_pred = {column for _, column, _ in matches}
        true_positive = len(matches)
        false_negative = len(gt_boxes) - true_positive
        false_positive = len(pred_boxes) - true_positive
        frame_switches = 0
        for gt_index, pred_index, match_iou in matches:
            gt_id = (sequence_id, int(gt_ids[gt_index]))
            pred_id = (sequence_id, int(pred_ids[pred_index]))
            previous_id = self.previous_prediction_for_gt.get(gt_id)
            if previous_id is not None and previous_id != pred_id:
                self.id_switches += 1
                frame_switches += 1
            if self.previous_matched_for_gt.get(gt_id) is False:
                self.fragments += 1
            self.previous_prediction_for_gt[gt_id] = pred_id
            self.previous_matched_for_gt[gt_id] = True
            self.pair_counts[(gt_id, pred_id)] += 1
            self.matched_events.append({
                'sequence_id': sequence_id,
                'frame_id': int(frame_id),
                'gt_id': gt_id,
                'pred_id': pred_id,
            })
            self.iou_sum += match_iou
            if gt_keypoints is not None and pred_keypoints is not None:
                direct = (pred_keypoints[pred_index] - gt_keypoints[gt_index]).norm(dim=-1).sum()
                swapped = (pred_keypoints[pred_index].flip(0) - gt_keypoints[gt_index]).norm(dim=-1).sum()
                self.head_tail_correct += int(direct <= swapped)
                self.head_tail_count += 1
        for gt_index, gt_id in enumerate(gt_ids.tolist()):
            if gt_index not in matched_gt:
                self.previous_matched_for_gt[(sequence_id, int(gt_id))] = False

        self.true_positives += true_positive
        self.false_negatives += false_negative
        self.false_positives += false_positive
        for bin_name in (self._density_bin(len(gt_boxes)), self._overlap_bin(gt_boxes)):
            self.bin_counts[bin_name].update({
                'gt': len(gt_boxes), 'tp': true_positive, 'fp': false_positive,
                'fn': false_negative, 'id_switches': frame_switches,
            })

    def _identity_counts(self):
        """Global one-to-one trajectory assignment used by IDF1."""
        gt_ids = list(self.gt_track_counts)
        pred_ids = list(dict.fromkeys(
            pred_id for _, pred_id in self.pair_counts
        ))
        if not gt_ids or not pred_ids:
            return 0, self.pred_detections, self.gt_detections
        matrix = torch.zeros(len(gt_ids), len(pred_ids), dtype=torch.float64)
        gt_index = {value: index for index, value in enumerate(gt_ids)}
        pred_index = {value: index for index, value in enumerate(pred_ids)}
        for (gt_id, pred_id), count in self.pair_counts.items():
            matrix[gt_index[gt_id], pred_index[pred_id]] = count
        rows, columns = _linear_sum_assignment(-matrix)
        idtp = int(matrix[rows, columns].sum().item()) if len(rows) else 0
        return (
            idtp,
            max(self.pred_detections - idtp, 0),
            max(self.gt_detections - idtp, 0),
        )

    def _hota(self):
        """HOTA averaged over IoU alpha=0.05..0.95, following TrackEval."""
        gt_keys = {}
        pred_keys = {}
        prepared = []
        for frame in self.frame_records:
            sequence = frame['sequence_id']
            gt_indices = []
            for value in frame['gt_ids'].tolist():
                gt_indices.append(gt_keys.setdefault((sequence, int(value)), len(gt_keys)))
            pred_indices = []
            for value in frame['pred_ids'].tolist():
                pred_indices.append(pred_keys.setdefault((sequence, int(value)), len(pred_keys)))
            similarity = (
                torchvision.ops.box_iou(frame['gt_boxes'], frame['pred_boxes'])
                if len(frame['gt_boxes']) and len(frame['pred_boxes'])
                else torch.zeros(len(frame['gt_boxes']), len(frame['pred_boxes']))
            )
            prepared.append((
                torch.tensor(gt_indices, dtype=torch.long),
                torch.tensor(pred_indices, dtype=torch.long),
                similarity,
            ))

        gt_count = torch.zeros(len(gt_keys), dtype=torch.float64)
        pred_count = torch.zeros(len(pred_keys), dtype=torch.float64)
        potential = torch.zeros(len(gt_keys), len(pred_keys), dtype=torch.float64)
        for gt_indices, pred_indices, similarity in prepared:
            if len(gt_indices):
                gt_count.index_add_(0, gt_indices, torch.ones(len(gt_indices), dtype=torch.float64))
            if len(pred_indices):
                pred_count.index_add_(0, pred_indices, torch.ones(len(pred_indices), dtype=torch.float64))
            if not len(gt_indices) or not len(pred_indices):
                continue
            similarity = similarity.to(torch.float64)
            denominator = (
                similarity.sum(dim=1, keepdim=True)
                + similarity.sum(dim=0, keepdim=True)
                - similarity
            )
            normalized = torch.where(
                denominator > 0, similarity / denominator.clamp_min(1e-12),
                torch.zeros_like(similarity),
            )
            potential[gt_indices[:, None], pred_indices[None, :]] += normalized
        alignment_denominator = (
            gt_count[:, None] + pred_count[None, :] - potential
        )
        global_alignment = torch.where(
            alignment_denominator > 0,
            potential / alignment_denominator.clamp_min(1e-12),
            torch.zeros_like(potential),
        )

        hota_values = []
        deta_values = []
        assa_values = []
        loca_values = []
        for alpha in [index / 20.0 for index in range(1, 20)]:
            pair_matches = torch.zeros_like(potential)
            true_positive = 0
            false_positive = 0
            false_negative = 0
            localization_sum = 0.0
            for gt_indices, pred_indices, similarity in prepared:
                if not len(gt_indices) or not len(pred_indices):
                    false_negative += len(gt_indices)
                    false_positive += len(pred_indices)
                    continue
                score = global_alignment[gt_indices[:, None], pred_indices[None, :]]
                score = score * similarity.to(score.dtype)
                rows, columns = _linear_sum_assignment(-score)
                keep = similarity[rows, columns] >= alpha
                rows = rows[keep]
                columns = columns[keep]
                matched = len(rows)
                true_positive += matched
                false_negative += len(gt_indices) - matched
                false_positive += len(pred_indices) - matched
                if matched:
                    localization_sum += float(similarity[rows, columns].sum().item())
                    pair_matches[gt_indices[rows], pred_indices[columns]] += 1
            detection_accuracy = true_positive / max(
                true_positive + false_positive + false_negative, 1
            )
            association_numerator = 0.0
            if true_positive:
                nonzero = torch.nonzero(pair_matches > 0, as_tuple=False)
                for gt_index, pred_index in nonzero.tolist():
                    count = float(pair_matches[gt_index, pred_index])
                    denominator = (
                        float(gt_count[gt_index]) + float(pred_count[pred_index]) - count
                    )
                    association_numerator += count * count / max(denominator, 1e-12)
            association_accuracy = association_numerator / max(true_positive, 1)
            localization_accuracy = localization_sum / max(true_positive, 1)
            deta_values.append(detection_accuracy)
            assa_values.append(association_accuracy)
            loca_values.append(localization_accuracy)
            hota_values.append((detection_accuracy * association_accuracy) ** 0.5)
        average = lambda values: sum(values) / max(len(values), 1)
        return {
            'hota': average(hota_values),
            'hota_deta': average(deta_values),
            'hota_assa': average(assa_values),
            'hota_loca': average(loca_values),
        }

    def _cast(self, detection_accuracy):
        """WACV-2026 CAST using its explicit T-previous-frames option."""
        events_by_sequence = defaultdict(list)
        for event in self.matched_events:
            events_by_sequence[event['sequence_id']].append(event)
        curve = {}
        switch_curve = {}
        transfer_curve = {}
        association_curve = {}
        for context in self.cast_context_frames:
            switch_scores = []
            transfer_scores = []
            association_scores = []
            for events in events_by_sequence.values():
                ordered = sorted(events, key=lambda item: item['frame_id'])
                for current in ordered:
                    history = [
                        event for event in ordered
                        if current['frame_id'] - context <= event['frame_id'] < current['frame_id']
                    ]
                    agreeing = [
                        event for event in history
                        if event['gt_id'] == current['gt_id']
                        and event['pred_id'] == current['pred_id']
                    ]
                    same_gt = [event for event in history if event['gt_id'] == current['gt_id']]
                    same_pred = [event for event in history if event['pred_id'] == current['pred_id']]
                    if same_gt:
                        switch_scores.append(len(agreeing) / len(same_gt))
                    if same_pred:
                        transfer_scores.append(len(agreeing) / len(same_pred))
                    union_count = len(same_gt) + len(same_pred) - len(agreeing)
                    if union_count:
                        association_scores.append(len(agreeing) / union_count)
            switch_score = sum(switch_scores) / max(len(switch_scores), 1)
            transfer_score = sum(transfer_scores) / max(len(transfer_scores), 1)
            association_score = (
                sum(association_scores) / len(association_scores)
                if association_scores else 1.0
            )
            switch_curve[context] = switch_score
            transfer_curve[context] = transfer_score
            association_curve[context] = association_score
            curve[context] = (detection_accuracy * association_score) ** 0.5
        selected_context = 5 if 5 in curve else self.cast_context_frames[0]
        return {
            'cast': curve[selected_context],
            'cast_context_frames': selected_context,
            'cast_curve': curve,
            'cast_switch_curve': switch_curve,
            'cast_transfer_curve': transfer_curve,
            'cast_association_curve': association_curve,
        }

    def summarize(self):
        gt = max(self.gt_detections, 1)
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        mota = 1.0 - (fn + fp + self.id_switches) / gt
        idtp, idfp, idfn = self._identity_counts()
        idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)
        detection_accuracy = tp / max(tp + fp + fn, 1)
        hota_metrics = self._hota()
        cast_metrics = self._cast(detection_accuracy)
        bins = {}
        for name, counts in self.bin_counts.items():
            bin_gt = max(counts['gt'], 1)
            bins[name] = {
                'mota': 1.0 - (counts['fn'] + counts['fp'] + counts['id_switches']) / bin_gt,
                'recall': counts['tp'] / bin_gt,
                'gt_instances': counts['gt'],
            }
        return {
            'mota': mota,
            'idf1': idf1,
            **hota_metrics,
            **cast_metrics,
            'id_switches': self.id_switches,
            'fragments': self.fragments,
            'fragment_rate': self.fragments / gt,
            'head_tail_swap_rate': 1.0 - self.head_tail_correct / max(self.head_tail_count, 1),
            'fixed_iou_threshold': self.iou_threshold,
            'bins': bins,
        }
