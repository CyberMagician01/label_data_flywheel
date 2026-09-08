"""Export one fixed-shape BeePoseTrack-E graph for RGB/IR and 1/3-frame use."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn


class BeeEExportWrapper(nn.Module):
    """Expose only deployable tensors; DN, one-to-many and EMA stay outside."""

    output_names = (
        'logits', 'boxes', 'keypoints', 'visibility', 'quality',
        'query_valid', 'density', 'domain_scores', 'query_scores',
    )

    def __init__(self, model, num_queries=768, num_keypoints=2,
                 calibration_contract=None):
        super().__init__()
        self.model = model
        self.num_queries = num_queries
        self.num_keypoints = num_keypoints
        self.register_buffer(
            'score_exponents',
            self._score_exponents(calibration_contract), persistent=True,
        )

    @staticmethod
    def _score_exponents(contract):
        if contract is None:
            return torch.ones(2, 3, dtype=torch.float32)
        if contract.get('effective_stage') != 'E-S5':
            raise ValueError('ONNX export requires an E-S5 calibration contract.')
        domains = contract.get('domains', {})
        if set(domains) != {'rgb', 'ir'}:
            raise ValueError('ONNX calibration contract requires exact rgb/ir domains.')
        values = []
        for name in ('rgb', 'ir'):
            ranking = domains[name].get('ranking', {})
            exponents = ranking.get('score_exponents', [])
            if len(exponents) != 3:
                raise ValueError(f'{name} ranking requires three score exponents.')
            values.append([float(value) for value in exponents])
        return torch.tensor(values, dtype=torch.float32)

    def forward(self, images, temporal_valid_mask, domain_id=None,
                stabilization_theta=None):
        if images.ndim != 5 or images.shape[1] != 3:
            raise ValueError('BeePoseTrack-E ONNX input must be [B,3,C,H,W].')
        if temporal_valid_mask.shape != (images.shape[0], 3):
            raise ValueError('temporal_valid_mask must have shape [B,3].')
        if domain_id is None:
            domain_id = torch.zeros(
                images.shape[0], dtype=torch.long, device=images.device
            )
        if stabilization_theta is None:
            stabilization_theta = identity_stabilization(
                images.shape[0], dtype=images.dtype, device=images.device
            )
        route_inputs = {
            'temporal_valid_mask': temporal_valid_mask,
            'domain_id': domain_id,
            'stabilization_theta': stabilization_theta,
        }
        parameters = inspect.signature(self.model.forward).parameters.values()
        if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
            supported = {parameter.name for parameter in parameters}
            route_inputs = {
                name: value for name, value in route_inputs.items() if name in supported
            }
        outputs = self.model(images, **route_inputs)
        logits = outputs['pred_logits'][:, :self.num_queries]
        boxes = outputs['pred_boxes'][:, :self.num_queries]
        batch, queries = logits.shape[:2]
        keypoints = outputs.get('pred_keypoints')
        if keypoints is None:
            keypoints = boxes.new_zeros((batch, queries, self.num_keypoints, 2))
        else:
            keypoints = keypoints[:, :self.num_queries]
        visibility = outputs.get('pred_visibility')
        if visibility is None:
            visibility = logits.new_zeros((batch, queries, self.num_keypoints))
        else:
            visibility = visibility[:, :self.num_queries]
        quality = outputs.get('pred_quality')
        if quality is None:
            quality = logits.new_zeros((batch, queries))
        else:
            quality = quality[:, :self.num_queries]
        valid = outputs.get('pred_query_valid')
        if valid is None:
            valid = logits.new_ones((batch, queries))
        else:
            valid = valid[:, :self.num_queries].to(logits.dtype)
        density = outputs.get('pred_density')
        if density is None:
            density = logits.new_zeros((batch, 1, images.shape[-2] // 4, images.shape[-1] // 4))
        domain_logits = outputs.get('pred_domain_logits')
        if domain_logits is None:
            domain_scores = torch.nn.functional.one_hot(
                domain_id.long(), num_classes=2
            ).to(logits.dtype)
        else:
            domain_scores = domain_logits.softmax(dim=-1)
        pose_visibility = visibility.sigmoid().mean(dim=-1)
        exponents = self.score_exponents[domain_id.long()].to(logits.dtype)
        query_scores = (
            logits.sigmoid().amax(dim=-1).clamp_min(1e-8).pow(exponents[:, 0:1])
            * quality.sigmoid().clamp_min(1e-8).pow(exponents[:, 1:2])
            * pose_visibility.clamp_min(1e-8).pow(exponents[:, 2:3])
            * valid
        )
        return (
            logits, boxes, keypoints, visibility, quality, valid,
            density, domain_scores, query_scores,
        )


def single_frame_as_clip(image, clip_len=3):
    if image.ndim != 4:
        raise ValueError('single image input must have shape [B, C, H, W]')
    if int(clip_len) != 3:
        raise ValueError('BeePoseTrack-E single-image mode must repeat into three frames')
    return image[:, None].expand(-1, clip_len, -1, -1, -1).contiguous()


def identity_stabilization(batch, dtype=torch.float32, device=None):
    identity = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device,
    )
    return identity[None, None].expand(batch, 2, -1, -1).contiguous()


def export_fixed_graph(model, output, input_size=1280, clip_len=3,
                       num_queries=768, opset=18, calibration_contract=None):
    if clip_len != 3:
        raise ValueError('BeePoseTrack-E deploy contract requires exactly three frames')
    model = model.deploy() if hasattr(model, 'deploy') else model
    wrapper = BeeEExportWrapper(
        model.eval(), num_queries=num_queries,
        calibration_contract=calibration_contract,
    ).eval()
    sample = torch.zeros(1, clip_len, 3, input_size, input_size)
    temporal_valid_mask = torch.ones(1, clip_len, dtype=torch.bool)
    domain_id = torch.zeros(1, dtype=torch.long)
    stabilization_theta = identity_stabilization(1)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (sample, temporal_valid_mask, domain_id, stabilization_theta),
        str(output),
        input_names=[
            'images', 'temporal_valid_mask', 'domain_id', 'stabilization_theta'
        ],
        output_names=list(wrapper.output_names),
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True,
    )
    return output


def convert_graph_to_fp16(fp32_path, fp16_path):
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(fp32_path))
    model = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=False
    )
    fp16_path = Path(fp16_path)
    fp16_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(fp16_path))
    return fp16_path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_model(config_path, checkpoint_path, calibration_contract):
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    cfg.yaml_cfg['ViTAdapter']['skip_load_backbone'] = True
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('ema', {}).get('module') if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state, strict=True)
    frozen = calibration_contract.get('frozen_model', {})
    if frozen.get('checkpoint_sha256') != _sha256(checkpoint_path):
        raise ValueError('Calibration contract is not bound to this export checkpoint.')
    decoder = getattr(cfg.model, 'decoder', None)
    if decoder is None or not hasattr(decoder, 'apply_query_capacity_contract'):
        raise TypeError('Export model does not expose the calibrated query-capacity route.')
    decoder.apply_query_capacity_contract(calibration_contract)
    return cfg.model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration-contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fp16-output', type=Path)
    parser.add_argument('--input-size', type=int, default=1280)
    parser.add_argument('--clip-len', type=int, default=3)
    parser.add_argument('--num-queries', type=int, default=768)
    parser.add_argument('--opset', type=int, default=18)
    args = parser.parse_args()
    calibration_contract = json.loads(
        args.calibration_contract.read_text(encoding='utf-8')
    )
    model = _load_model(args.config, args.checkpoint, calibration_contract)
    output = export_fixed_graph(
        model, args.output, input_size=args.input_size, clip_len=args.clip_len,
        num_queries=args.num_queries, opset=args.opset,
        calibration_contract=calibration_contract,
    )
    if args.fp16_output:
        convert_graph_to_fp16(output, args.fp16_output)


if __name__ == '__main__':
    main()
