"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register

__all__ = ['PostProcessor', 'structural_set_deduplicate']


def mod(a, b):
    out = a - a // b * b
    return out


def _domain_value(values, domain_id, default):
    if values is None:
        return default
    domain_name = 'ir' if int(domain_id) == 1 else 'rgb'
    value = values.get(domain_name, values.get(str(int(domain_id)), default))
    return value


def _scene_value(values, scene_id, default):
    if values is None or scene_id is None:
        return default
    scene_name = 'outdoor' if int(scene_id) == 1 else 'indoor'
    return values.get(scene_name, values.get(str(int(scene_id)), default))


def _filter_result(result, keep):
    count = len(result.get('scores', []))
    return {
        name: value[keep]
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count
        else value
        for name, value in result.items()
    }


def _endpoint_structure(keypoints, boxes, sigma):
    axes = keypoints[:, 1] - keypoints[:, 0]
    lengths = axes.norm(dim=-1).clamp_min(1e-6)
    diagonals = (boxes[:, 2:] - boxes[:, :2]).norm(dim=-1).clamp_min(1.0)
    squared = (keypoints[:, None] - keypoints[None]).square().sum(dim=-1)
    scale = ((diagonals[:, None] + diagonals[None, :]) * 0.5 * sigma).clamp_min(1e-6)
    oks = torch.exp(-squared / (2.0 * scale[..., None].square())).mean(dim=-1)
    length_difference = (
        (lengths[:, None] - lengths[None, :]).abs()
        / torch.maximum(lengths[:, None], lengths[None, :])
    )
    cosine = (
        axes @ axes.t()
        / (lengths[:, None] * lengths[None, :])
    ).clamp(-1.0, 1.0)
    oriented_axis_difference = torch.rad2deg(torch.acos(cosine))
    return oks, length_difference, oriented_axis_difference


def structural_set_deduplicate(result, thresholds=None, endpoint_oks_sigma=0.10):
    """执行一次集合结构去重；四个条件必须同时成立才删除低分查询。"""
    thresholds = thresholds or {}
    count = len(result.get('scores', []))
    if count < 2 or 'keypoints' not in result:
        return result
    keypoints = result['keypoints']
    boxes = result['boxes']
    if keypoints.shape[:2] != (count, 2):
        return result
    iou = torchvision.ops.box_iou(boxes, boxes)
    oks, length_difference, axis_difference = _endpoint_structure(
        keypoints, boxes, float(endpoint_oks_sigma)
    )
    duplicate = (
        (iou >= float(thresholds.get('box_iou_min', 0.75)))
        & (oks >= float(thresholds.get('endpoint_oks_min', 0.80)))
        & (length_difference <= float(thresholds.get('relative_length_max', 0.20)))
        & (axis_difference <= float(thresholds.get('oriented_axis_deg_max', 20.0)))
    )
    duplicate.fill_diagonal_(False)
    order = result['scores'].argsort(descending=True)
    keep = torch.ones(count, dtype=torch.bool, device=boxes.device)
    for position, query_index in enumerate(order.tolist()):
        if not bool(keep[query_index]):
            continue
        lower_ranked = order[position + 1:]
        if len(lower_ranked):
            keep[lower_ranked[duplicate[query_index, lower_ranked]]] = False
    filtered = {}
    for name, value in result.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
            filtered[name] = value[keep]
        else:
            filtered[name] = value
    filtered['set_dedup_removed'] = torch.tensor(
        count - int(keep.sum()), device=boxes.device
    )
    return filtered


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        enable_calibrated_filter=False,
        score_exponents_by_domain=None,
        score_thresholds_by_domain=None,
        score_exponents_by_scene=None,
        score_thresholds_by_scene=None,
        num_top_queries_by_scene=None,
        box_nms_iou_by_scene=None,
        enable_set_dedup=False,
        dedup_thresholds_by_domain=None,
        endpoint_oks_sigma=0.10,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.enable_calibrated_filter = bool(enable_calibrated_filter)
        self.score_exponents_by_domain = score_exponents_by_domain or {
            'rgb': [1.0, 1.0, 0.0], 'ir': [1.0, 1.0, 0.0],
        }
        self.score_thresholds_by_domain = score_thresholds_by_domain or {
            'rgb': 0.0, 'ir': 0.0,
        }
        self.score_exponents_by_scene = score_exponents_by_scene or {}
        self.score_thresholds_by_scene = score_thresholds_by_scene or {}
        self.num_top_queries_by_scene = num_top_queries_by_scene or {}
        self.box_nms_iou_by_scene = box_nms_iou_by_scene or {}
        self.enable_set_dedup = bool(enable_set_dedup)
        self.dedup_thresholds_by_domain = dedup_thresholds_by_domain or {}
        self.endpoint_oks_sigma = float(endpoint_oks_sigma)
        self.deploy_mode = False

    def apply_calibration_contract(self, contract):
        """Atomically load the frozen E-S5 RGB/IR ranking and dedup contract."""
        if contract.get('effective_stage') != 'E-S5':
            raise ValueError('Inference calibration contract must be frozen at E-S5.')
        domains = contract.get('domains', {})
        if set(domains) != {'rgb', 'ir'}:
            raise ValueError('Inference calibration contract requires exact rgb/ir domains.')
        score_exponents = {}
        score_thresholds = {}
        dedup_thresholds = {}
        for domain_name in ('rgb', 'ir'):
            domain = domains[domain_name]
            if domain.get('source_data') != 'calibration':
                raise ValueError(
                    f'{domain_name} inference parameters were not fitted on calibration.'
                )
            ranking = domain.get('ranking', {})
            dedup = domain.get('set_deduplication', {})
            if len(ranking.get('score_exponents', [])) != 3:
                raise ValueError(f'{domain_name} ranking must contain three score exponents.')
            required_dedup = {
                'box_iou_min', 'endpoint_oks_min', 'relative_length_max',
                'oriented_axis_deg_max',
            }
            if required_dedup - set(dedup):
                raise ValueError(f'{domain_name} structural dedup contract is incomplete.')
            score_exponents[domain_name] = [
                float(value) for value in ranking['score_exponents']
            ]
            score_thresholds[domain_name] = float(ranking['score_threshold'])
            dedup_thresholds[domain_name] = {
                name: float(dedup[name]) for name in required_dedup
            }
        self.score_exponents_by_domain = score_exponents
        self.score_thresholds_by_domain = score_thresholds
        self.dedup_thresholds_by_domain = dedup_thresholds
        self.enable_calibrated_filter = True
        return self

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        mask_pred = outputs.get('pred_masks', None)
        keypoint_pred = outputs.get('pred_keypoints', None)
        visibility_pred = outputs.get('pred_visibility', None)
        density_pred = outputs.get('pred_density', None)
        domain_pred = outputs.get('pred_domain_logits', None)
        quality_pred = outputs.get('pred_quality', None)
        query_valid_pred = outputs.get('pred_query_valid', None)
        route_scene_id = outputs.get('route_scene_id')
        if (
            self.score_exponents_by_scene or self.score_thresholds_by_scene
            or self.num_top_queries_by_scene or self.box_nms_iou_by_scene
        ) and route_scene_id is None:
            raise RuntimeError('Scene-specific postprocessing requires route_scene_id.')
        route_domain_id = outputs.get('route_domain_id')
        if route_domain_id is None and domain_pred is not None:
            route_domain_id = domain_pred.argmax(dim=-1)
        if route_domain_id is None:
            route_domain_id = logits.new_zeros(logits.shape[0], dtype=torch.long)

        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            if self.enable_calibrated_filter:
                quality_for_rank = (
                    quality_pred.sigmoid() if quality_pred is not None
                    else scores.new_ones(scores.shape[:2])
                )
                visibility_for_rank = (
                    visibility_pred.sigmoid().mean(dim=-1) if visibility_pred is not None
                    else scores.new_ones(scores.shape[:2])
                )
                calibrated = []
                for batch_index, domain_id in enumerate(route_domain_id.tolist()):
                    exponents = _domain_value(
                        self.score_exponents_by_domain, domain_id, [1.0, 1.0, 0.0]
                    )
                    if route_scene_id is not None:
                        exponents = _scene_value(
                            self.score_exponents_by_scene,
                            int(route_scene_id[batch_index]),
                            exponents,
                        )
                    alpha, beta, gamma = exponents
                    calibrated.append(
                        scores[batch_index].clamp_min(1e-8).pow(float(alpha))
                        * quality_for_rank[batch_index, :, None].clamp_min(1e-8).pow(float(beta))
                        * visibility_for_rank[batch_index, :, None].clamp_min(1e-8).pow(float(gamma))
                    )
                scores = torch.stack(calibrated, dim=0)
            elif quality_pred is not None:
                scores = scores * quality_pred.sigmoid().unsqueeze(-1)
            if query_valid_pred is not None:
                scores = scores * query_valid_pred.to(scores.dtype).unsqueeze(-1)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            masks = mask_pred.gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mask_pred.shape[-2], 
                                                                                           mask_pred.shape[-1])) if mask_pred is not None else None

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            index = torch.arange(scores.shape[1], device=scores.device)[None].expand(scores.shape[0], -1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if keypoint_pred is not None:
            keypoints = keypoint_pred.gather(
                dim=1,
                index=index[..., None, None].expand(-1, -1, keypoint_pred.shape[-2], 2),
            )
            keypoints = keypoints * orig_target_sizes[:, None, None, :]
            keypoint_scores = visibility_pred.sigmoid().gather(
                dim=1,
                index=index[..., None].expand(-1, -1, visibility_pred.shape[-1]),
            )
            pose_scores = keypoint_scores.mean(dim=-1)

        if quality_pred is not None:
            quality_scores = quality_pred.sigmoid().gather(dim=1, index=index)
        if query_valid_pred is not None:
            query_valid = query_valid_pred.gather(dim=1, index=index)

        if self.deploy_mode:
            if keypoint_pred is not None:
                deploy_outputs = [labels, boxes, scores, keypoints, keypoint_scores, pose_scores]
                if quality_pred is not None:
                    deploy_outputs.append(quality_scores)
                if density_pred is not None:
                    deploy_outputs.append(density_pred)
                if domain_pred is not None:
                    deploy_outputs.append(domain_pred.softmax(dim=-1))
                return tuple(deploy_outputs)
            if mask_pred is not None:
                return labels, boxes, scores, masks
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        if mask_pred is not None:
            for (i, (s, l, b, m, q)) in enumerate(zip(scores, labels, boxes, masks, index)):
                res = {'scores': s, 'labels': l, 'boxes': b, 'query_indices': q}
                w, h = orig_target_sizes[i].tolist()
                m = F.interpolate(m.unsqueeze(1), size=(int(h), int(w)), mode='bilinear', align_corners=False)
                res['masks'] = m > 0.0
                results.append(res)
        else:
            results = [
                {'scores': s, 'labels': l, 'boxes': b, 'query_indices': q}
                for s, l, b, q in zip(scores, labels, boxes, index)
            ]

        if keypoint_pred is not None:
            for result, keypoints_i, keypoint_scores_i, pose_scores_i in zip(
                results, keypoints, keypoint_scores, pose_scores
            ):
                result['keypoints'] = keypoints_i
                result['kpt_scores'] = keypoint_scores_i
                result['pose_score'] = pose_scores_i

        if quality_pred is not None:
            for result, quality_i in zip(results, quality_scores):
                result['quality_score'] = quality_i
                if 'pose_score' in result:
                    result['pose_score'] = result['pose_score'] * quality_i
        if query_valid_pred is not None:
            for result, query_valid_i in zip(results, query_valid):
                result['query_valid'] = query_valid_i

        if density_pred is not None:
            for result, density_i in zip(results, density_pred):
                result['density'] = density_i[0]
                result['density_count'] = density_i.sum()

        if domain_pred is not None:
            for result, domain_i in zip(results, domain_pred.softmax(dim=-1)):
                result['domain_scores'] = domain_i

        calibrated_results = []
        scene_ids = (
            route_scene_id.tolist() if route_scene_id is not None
            else [None] * len(results)
        )
        for result, domain_id, scene_id in zip(
            results, route_domain_id.tolist(), scene_ids,
        ):
            result['domain_id'] = torch.tensor(domain_id, device=boxes.device)
            if scene_id is not None:
                result['scene_id'] = torch.tensor(scene_id, device=boxes.device)
            scene_top_queries = _scene_value(
                self.num_top_queries_by_scene, scene_id, None,
            )
            if scene_top_queries is not None and len(result['scores']) > int(scene_top_queries):
                result = _filter_result(result, slice(0, int(scene_top_queries)))
            if self.enable_calibrated_filter:
                threshold = _domain_value(
                    self.score_thresholds_by_domain, domain_id, 0.0,
                )
                threshold = float(_scene_value(
                    self.score_thresholds_by_scene, scene_id, threshold,
                ))
                keep = result['scores'] >= threshold
                if 'query_valid' in result:
                    keep &= result['query_valid'].bool()
                result = _filter_result(result, keep)
            nms_iou = _scene_value(self.box_nms_iou_by_scene, scene_id, None)
            if nms_iou is not None and len(result['scores']) > 1:
                count = len(result['scores'])
                keep = torchvision.ops.nms(
                    result['boxes'], result['scores'], float(nms_iou),
                )
                result = _filter_result(result, keep)
                result['scene_box_nms_removed'] = torch.tensor(
                    count - len(keep), device=boxes.device,
                )
            if self.enable_set_dedup:
                result = structural_set_deduplicate(
                    result,
                    _domain_value(self.dedup_thresholds_by_domain, domain_id, {}),
                    self.endpoint_oks_sigma,
                )
            calibrated_results.append(result)
        results = calibrated_results

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
