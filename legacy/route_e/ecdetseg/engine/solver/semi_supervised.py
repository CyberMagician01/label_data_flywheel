import torch
import torch.nn.functional as F
import torchvision


class OnlinePseudoLabeler:
    """EMA teacher pseudo sets with flip-set and reciprocal trajectory checks."""

    def __init__(self, score_threshold=0.35, quality_threshold=0.30,
                 set_iou_threshold=0.50, endpoint_threshold=0.15,
                 trajectory_iou_threshold=0.30, max_pseudo_instances=256,
                 ignore_score_threshold=0.55):
        self.score_threshold = score_threshold
        self.quality_threshold = quality_threshold
        self.set_iou_threshold = set_iou_threshold
        self.endpoint_threshold = endpoint_threshold
        self.trajectory_iou_threshold = trajectory_iou_threshold
        self.max_pseudo_instances = max_pseudo_instances
        self.ignore_score_threshold = float(ignore_score_threshold)

    @staticmethod
    def _box_iou(boxes_a, boxes_b):
        if not len(boxes_a) or not len(boxes_b):
            return boxes_a.new_zeros((len(boxes_a), len(boxes_b)))
        return torchvision.ops.box_iou(
            torchvision.ops.box_convert(boxes_a, 'cxcywh', 'xyxy'),
            torchvision.ops.box_convert(boxes_b, 'cxcywh', 'xyxy'),
        )

    def _extract(self, outputs, batch_index, unflip=False):
        logits = outputs['pred_logits'][batch_index]
        class_scores, labels = logits.sigmoid().max(dim=-1)
        quality = outputs.get('pred_quality')
        quality = quality[batch_index].sigmoid() if quality is not None else torch.ones_like(class_scores)
        score = class_scores * quality
        valid = outputs.get('pred_query_valid')
        valid = valid[batch_index].bool() if valid is not None else torch.ones_like(score, dtype=torch.bool)
        keep = valid & (class_scores >= self.score_threshold) & (quality >= self.quality_threshold)
        indices = torch.nonzero(keep).flatten()
        if len(indices) > self.max_pseudo_instances:
            indices = indices[score[indices].topk(self.max_pseudo_instances).indices]
        boxes = outputs['pred_boxes'][batch_index, indices].detach().clone()
        keypoints = outputs.get('pred_keypoints')
        keypoints = keypoints[batch_index, indices].detach().clone() if keypoints is not None else None
        visibility = outputs.get('pred_visibility')
        visibility = visibility[batch_index, indices].sigmoid().detach() if visibility is not None else None
        masks = outputs.get('pred_masks')
        masks = masks[batch_index, indices].detach().clone() if masks is not None else None
        if unflip:
            boxes[:, 0] = 1.0 - boxes[:, 0]
            if keypoints is not None:
                keypoints[..., 0] = 1.0 - keypoints[..., 0]
            if masks is not None:
                masks = masks.flip(-1)
        return {
            'boxes': boxes,
            'labels': labels[indices].detach(),
            'scores': score[indices].detach(),
            'class_probabilities': logits.sigmoid()[indices].detach(),
            'keypoints': keypoints,
            'visibility': visibility,
            'masks': masks,
        }

    @staticmethod
    def _index_candidates(candidates, keep):
        count = len(candidates['boxes'])
        return {
            key: (value[keep] if torch.is_tensor(value) and value.ndim > 0
                  and value.shape[0] == count else value)
            for key, value in candidates.items()
        }

    def _set_consistency(self, original, flipped):
        if not len(original['boxes']) or not len(flipped['boxes']):
            return original['boxes'].new_zeros(len(original['boxes']))
        iou = self._box_iou(original['boxes'], flipped['boxes'])
        best_iou, matched = iou.max(dim=1)
        consistent = best_iou >= self.set_iou_threshold
        if original['keypoints'] is not None and flipped['keypoints'] is not None:
            diagonal = original['boxes'][:, 2:].norm(dim=-1).clamp_min(1e-4)
            endpoint_error = (
                original['keypoints'] - flipped['keypoints'][matched]
            ).norm(dim=-1).mean(dim=-1) / diagonal
            consistent &= endpoint_error <= self.endpoint_threshold
        return best_iou * consistent.to(best_iou.dtype)

    def _trajectory_consistency(self, current, frame_candidates):
        score = current['boxes'].new_zeros(len(current['boxes']))
        if not len(current['boxes']):
            return score
        # A single-frame prediction is not a trajectory. Padding or a missing
        # history must never silently turn into perfect temporal consistency.
        if not frame_candidates:
            return score

        def reciprocal_step(source_boxes, target_boxes):
            if not len(source_boxes) or not len(target_boxes):
                return (
                    torch.zeros(0, dtype=torch.long, device=source_boxes.device),
                    torch.zeros(0, dtype=torch.bool, device=source_boxes.device),
                    torch.zeros(0, device=source_boxes.device),
                )
            iou = self._box_iou(source_boxes, target_boxes)
            values, target_index = iou.max(dim=1)
            _, source_index = iou.max(dim=0)
            reciprocal = source_index[target_index] == torch.arange(
                len(source_boxes), device=source_boxes.device
            )
            return target_index, reciprocal & (values >= self.trajectory_iou_threshold), values

        original_indices = torch.arange(len(current['boxes']), device=current['boxes'].device)
        active_indices = original_indices
        active_boxes = current['boxes']
        path_score = current['boxes'].new_ones(len(active_boxes))
        for previous in reversed(frame_candidates):
            target_index, valid, values = reciprocal_step(active_boxes, previous['boxes'])
            active_indices = active_indices[valid]
            path_score = torch.minimum(path_score[valid], values[valid])
            active_boxes = previous['boxes'][target_index[valid]]
            if not len(active_boxes):
                return score

        # Traverse the same path forward. A pseudo instance is accepted only
        # when the oldest observation returns to the same current-frame set item.
        for next_frame in list(frame_candidates[1:]) + [current]:
            target_index, valid, values = reciprocal_step(active_boxes, next_frame['boxes'])
            active_indices = active_indices[valid]
            path_score = torch.minimum(path_score[valid], values[valid])
            active_boxes = next_frame['boxes'][target_index[valid]]
            final_target_index = target_index[valid]
            if not len(active_boxes):
                return score
        returned = final_target_index == active_indices
        score[active_indices[returned]] = path_score[returned]
        return score

    @staticmethod
    def _empty_instances(target, device, num_keypoints=2, mask_size=None):
        result = {key: value for key, value in target.items() if value.ndim <= 1 and value.numel() == 1}
        result.update({
            'labels': torch.zeros(0, dtype=torch.long, device=device),
            'boxes': torch.zeros(0, 4, device=device),
            'keypoints': torch.zeros(0, num_keypoints, 3, device=device),
            'pose_state': torch.zeros(0, dtype=torch.long, device=device),
            'pose_mask': torch.zeros(0, device=device),
            'track_geometry': torch.zeros(0, 6, device=device),
            'track_geometry_mask': torch.zeros(0, dtype=torch.bool, device=device),
            'track_mask': torch.zeros(0, dtype=torch.bool, device=device),
            'track_id': torch.full((0,), -1, dtype=torch.long, device=device),
            'is_pseudo': torch.zeros(0, dtype=torch.bool, device=device),
            'pseudo_score': torch.zeros(0, device=device),
            'pose_quality': torch.zeros(0, device=device),
            'track_quality': torch.zeros(0, device=device),
            'trajectory_stability': torch.zeros(0, device=device),
            'supervision_mask': torch.zeros(0, 4, device=device),
            'ignore_boxes': torch.zeros(0, 4, device=device),
        })
        if mask_size is not None:
            result['masks'] = torch.zeros(
                0, int(mask_size[0]), int(mask_size[1]),
                dtype=torch.bool, device=device,
            )
        return result

    def _build_target(self, base_target, candidates, set_score, trajectory_score,
                      mask_size=None):
        device = candidates['boxes'].device
        combined = candidates['scores'] * set_score * trajectory_score
        keep = combined > 0
        independent_support = (set_score > 0).to(torch.int64) + (
            trajectory_score > 0
        ).to(torch.int64)
        ignore_keep = (~keep) & (
            (independent_support > 0)
            | (candidates['scores'] >= self.ignore_score_threshold)
        )
        if 'masks' in base_target and candidates.get('masks') is None:
            # Never manufacture a foreground mask. A candidate without the
            # supervision required by a mask-labelled sample is ignore-only.
            ignore_keep |= keep
            keep = torch.zeros_like(keep)
        trex_boxes = base_target.get('trex2_boxes')
        trex_scores = None
        if trex_boxes is not None and len(trex_boxes) and len(candidates['boxes']):
            trex_scores = base_target.get(
                'trex2_scores', torch.ones(len(trex_boxes), device=device) * 0.5
            )
            all_overlap = self._box_iou(candidates['boxes'], trex_boxes)
            support_iou, _ = all_overlap.max(dim=1)
            ignore_keep |= (~keep) & (support_iou >= self.set_iou_threshold)
        boxes = candidates['boxes'][keep]
        labels = candidates['labels'][keep]
        score = combined[keep]
        trajectory = trajectory_score[keep]
        keypoint_xy = candidates['keypoints'][keep] if candidates['keypoints'] is not None else None
        visibility = candidates['visibility'][keep] if candidates['visibility'] is not None else None

        # T-Rex2 is proposal-only: it may support a teacher set but never creates
        # a target by itself and never overwrites human annotations.
        if trex_boxes is not None and len(trex_boxes) and len(boxes):
            overlap = self._box_iou(boxes, trex_boxes)
            support_iou, support_index = overlap.max(dim=1)
            supported = support_iou >= self.set_iou_threshold
            support = trex_scores[support_index].clamp(0, 1)
            score = torch.where(
                supported, torch.maximum(score, 0.5 * score + 0.5 * support), score
            )

        result = self._empty_instances(
            base_target, device,
            num_keypoints=(
                candidates['keypoints'].shape[1]
                if candidates['keypoints'] is not None else 2
            ),
            mask_size=(mask_size if candidates.get('masks') is not None else None),
        )
        existing_ignore = base_target.get(
            'ignore_boxes', candidates['boxes'].new_zeros((0, 4))
        ).to(device)
        result['ignore_boxes'] = torch.cat([
            existing_ignore, candidates['boxes'][ignore_keep].detach(),
        ], dim=0)
        instance_count = len(boxes)
        result['boxes'] = boxes
        result['labels'] = labels
        result['is_pseudo'] = torch.ones(instance_count, dtype=torch.bool, device=device)
        result['pseudo_score'] = score
        result['teacher_class_probability'] = candidates['class_probabilities'][keep]
        result['trajectory_stability'] = trajectory
        if candidates.get('masks') is not None:
            pseudo_masks = candidates['masks'][keep, None].sigmoid()
            if mask_size is not None and tuple(pseudo_masks.shape[-2:]) != tuple(mask_size):
                pseudo_masks = F.interpolate(
                    pseudo_masks, size=tuple(mask_size), mode='bilinear', align_corners=False
                )
            result['masks'] = pseudo_masks[:, 0] >= 0.5
        if keypoint_xy is not None:
            keypoint_count = keypoint_xy.shape[1]
            keypoints = torch.zeros(instance_count, keypoint_count, 3, device=device)
            teacher_count = len(keypoint_xy)
            keypoints[:teacher_count, :, :2] = keypoint_xy
            pose_visible = visibility >= 0.5
            keypoints[:teacher_count, :, 2] = pose_visible.to(keypoints.dtype) * 2
            pose_quality = visibility.mean(dim=-1)
            if instance_count > teacher_count:
                pose_quality = torch.cat([
                    pose_quality, torch.zeros(instance_count - teacher_count, device=device)
                ])
            pose_state = torch.zeros(instance_count, dtype=torch.long, device=device)
            pose_state[:teacher_count] = torch.where(
                pose_visible.any(dim=-1),
                torch.full((teacher_count,), 2, dtype=torch.long, device=device),
                torch.ones(teacher_count, dtype=torch.long, device=device),
            )
            result['keypoints'] = keypoints
            result['pose_state'] = pose_state
            result['pose_mask'] = (pose_state > 0).to(keypoints.dtype)
            result['pose_quality'] = pose_quality
        else:
            result['keypoints'] = torch.zeros(
                instance_count, 2, 3, device=device
            )
            result['pose_state'] = torch.zeros(
                instance_count, dtype=torch.long, device=device
            )
            result['pose_mask'] = torch.zeros(instance_count, device=device)
            result['pose_quality'] = torch.zeros(instance_count, device=device)
        result['track_geometry'] = torch.zeros(instance_count, 6, device=device)
        # Reciprocal consistency is a confidence signal, not an identity
        # annotation. Track loss is disabled until a real pseudo track id and
        # geometry are assigned by a dedicated association step.
        result['track_geometry_mask'] = torch.zeros(
            instance_count, dtype=torch.bool, device=device
        )
        result['track_mask'] = torch.zeros(
            instance_count, dtype=torch.bool, device=device
        )
        result['track_quality'] = torch.zeros(instance_count, device=device)
        result['track_id'] = torch.full((instance_count,), -1, dtype=torch.long, device=device)
        result['supervision_mask'] = torch.stack([
            torch.ones(instance_count, device=device),
            result['pose_mask'],
            result['track_mask'].to(torch.float32),
            torch.ones(instance_count, device=device),
        ], dim=-1)
        return result

    @staticmethod
    def _merge_with_human_target(base_target, pseudo_target):
        """Append only missing-instance pseudo targets while preserving human labels."""
        human_count = len(base_target.get('boxes', ()))
        pseudo_count = len(pseudo_target['boxes'])
        result = dict(base_target)
        result['ignore_boxes'] = pseudo_target['ignore_boxes']
        if pseudo_count == 0:
            return result

        instance_fields = {
            'boxes', 'labels', 'area', 'iscrowd', 'masks', 'keypoints',
            'pose_state', 'pose_mask', 'track_geometry', 'track_geometry_mask',
            'track_id', 'track_mask', 'annotator_id', 'inter_group_quality',
            'intra_group_quality', 'hierarchy_quality', 'is_pseudo',
            'pseudo_score', 'pose_quality', 'track_quality',
            'trajectory_stability', 'supervision_mask',
            'teacher_class_probability',
        }

        def filler(template, count, key, human_side):
            shape = (count, *template.shape[1:])
            value = torch.zeros(shape, dtype=template.dtype, device=template.device)
            if human_side and key in {
                'pseudo_score', 'pose_quality', 'track_quality',
                'trajectory_stability', 'supervision_mask',
            }:
                value.fill_(1)
            if key == 'track_id':
                value.fill_(-1)
            return value

        for key in instance_fields:
            human_value = base_target.get(key)
            pseudo_value = pseudo_target.get(key)
            human_aligned = (
                torch.is_tensor(human_value) and human_value.ndim > 0
                and human_value.shape[0] == human_count
            )
            pseudo_aligned = (
                torch.is_tensor(pseudo_value) and pseudo_value.ndim > 0
                and pseudo_value.shape[0] == pseudo_count
            )
            if not human_aligned and not pseudo_aligned:
                continue
            template = human_value if human_aligned else pseudo_value
            if not human_aligned:
                human_value = filler(template, human_count, key, True)
            if not pseudo_aligned:
                pseudo_value = filler(template, pseudo_count, key, False)
            result[key] = torch.cat([
                human_value.to(pseudo_value.device), pseudo_value
            ], dim=0)
        return result

    @torch.no_grad()
    def generate(self, teacher, samples, targets):
        teacher.eval()
        temporal_mask = None
        if samples.ndim == 5 and all('temporal_valid_mask' in target for target in targets):
            temporal_mask = torch.stack([
                target['temporal_valid_mask'] for target in targets
            ], dim=0)
        if temporal_mask is None:
            original_outputs = teacher(samples)
        else:
            original_outputs = teacher(samples, temporal_valid_mask=temporal_mask)
        flipped_samples = samples.flip(-1)
        if temporal_mask is None:
            flipped_outputs = teacher(flipped_samples)
        else:
            flipped_outputs = teacher(flipped_samples, temporal_valid_mask=temporal_mask)
        frame_outputs = None
        frame_count = 1
        if samples.ndim == 5 and samples.shape[1] > 1:
            batch, frame_count, channels, height, width = samples.shape
            frame_outputs = teacher(samples.reshape(batch * frame_count, channels, height, width))

        pseudo_targets = []
        accepted = 0
        for batch_index, target in enumerate(targets):
            is_unlabeled = bool(target.get(
                'is_unlabeled', torch.tensor([False], device=samples.device)
            ).item())
            original = self._extract(original_outputs, batch_index)
            if not is_unlabeled and len(target.get('boxes', ())):
                overlap = self._box_iou(original['boxes'], target['boxes'])
                missing_candidate = overlap.max(dim=1).values < self.set_iou_threshold
                original = self._index_candidates(original, missing_candidate)
            flipped = self._extract(flipped_outputs, batch_index, unflip=True)
            set_score = self._set_consistency(original, flipped)
            frame_candidates = []
            if frame_outputs is not None:
                if temporal_mask is None:
                    valid_history = range(frame_count - 1)
                else:
                    valid_history = torch.nonzero(
                        temporal_mask[batch_index, :frame_count - 1].bool()
                    ).flatten().tolist()
                frame_candidates = [
                    self._extract(frame_outputs, batch_index * frame_count + frame_index)
                    for frame_index in valid_history
                ]
            trajectory_score = self._trajectory_consistency(original, frame_candidates)
            pseudo_target = self._build_target(
                target, original, set_score, trajectory_score,
                mask_size=samples.shape[-2:],
            )
            accepted += len(pseudo_target['boxes'])
            pseudo_targets.append(
                pseudo_target if is_unlabeled
                else self._merge_with_human_target(target, pseudo_target)
            )
        return pseudo_targets, {
            'pseudo_instances': accepted,
            'ema_teacher_updates': getattr(teacher, 'updates', 0),
        }

    @staticmethod
    def strong_view(samples, targets):
        strong = samples.clone()
        for batch_index, target in enumerate(targets):
            if not bool(target.get('is_pseudo', torch.zeros(0, device=strong.device)).any()):
                continue
            gain = 0.8 + 0.4 * torch.rand((), device=strong.device)
            noise = torch.randn_like(strong[batch_index]) * 0.03
            strong[batch_index] = strong[batch_index] * gain + noise
        return strong
