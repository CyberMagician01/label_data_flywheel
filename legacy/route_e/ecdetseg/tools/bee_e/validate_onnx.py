"""Numerical, graph-shape and end-to-end latency validation for BeePoseTrack-E."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .export_onnx import (
    BeeEExportWrapper, _load_model, identity_stabilization, single_frame_as_clip,
)


REQUIRED_DEPLOY_OPS = {'Conv', 'Gather', 'GridSample', 'MatMul', 'Softmax'}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_fixed_graph(path, input_size=1280, clip_len=3, num_queries=768):
    import onnx

    if int(clip_len) != 3:
        raise ValueError('BeePoseTrack-E deploy graph must contain exactly three frames.')
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph_input = model.graph.input[0]
    shape = [dimension.dim_value for dimension in graph_input.type.tensor_type.shape.dim]
    expected = [1, clip_len, 3, input_size, input_size]
    if shape != expected:
        raise ValueError(f'ONNX input shape {shape} != fixed contract {expected}')
    mask_shape = [
        dimension.dim_value
        for dimension in model.graph.input[1].type.tensor_type.shape.dim
    ]
    if mask_shape != [1, clip_len]:
        raise ValueError(f'ONNX temporal mask shape {mask_shape} != [1, {clip_len}]')
    domain_shape = [
        dimension.dim_value
        for dimension in model.graph.input[2].type.tensor_type.shape.dim
    ]
    theta_shape = [
        dimension.dim_value
        for dimension in model.graph.input[3].type.tensor_type.shape.dim
    ]
    if domain_shape != [1]:
        raise ValueError(f'ONNX domain input shape {domain_shape} != [1]')
    if theta_shape != [1, 2, 2, 3]:
        raise ValueError(f'ONNX stabilization shape {theta_shape} != [1, 2, 2, 3]')
    output_shapes = {
        output.name: [dimension.dim_value for dimension in output.type.tensor_type.shape.dim]
        for output in model.graph.output
    }
    if output_shapes.get('boxes', [None, None])[1] != num_queries:
        raise ValueError('ONNX boxes output does not retain all fixed query slots')
    operators = {node.op_type for node in model.graph.node}
    custom = sorted({
        f'{node.domain}:{node.op_type}' for node in model.graph.node
        if node.domain not in ('', 'ai.onnx')
    })
    if custom:
        raise ValueError(f'custom/non-standard ONNX operators: {custom}')
    if operators & {'If', 'Loop', 'Scan'}:
        raise ValueError('dynamic control flow is forbidden in the deploy graph')
    missing = sorted(REQUIRED_DEPLOY_OPS - operators)
    if missing:
        raise ValueError(f'deploy graph is missing required ECDet operators: {missing}')
    return {
        'input_shape': shape, 'temporal_mask_shape': mask_shape,
        'domain_shape': domain_shape, 'stabilization_shape': theta_shape,
        'output_shapes': output_shapes, 'operators': sorted(operators),
    }


def compare_outputs(reference, candidate, fp16=False, topk=100):
    tolerances = {
        'logits': 3e-3 if fp16 else 1e-4,
        'boxes': 3e-3 if fp16 else 1e-4,
        'keypoints': 5e-3 if fp16 else 2e-4,
        'visibility': 3e-3 if fp16 else 1e-4,
        'quality': 3e-3 if fp16 else 1e-4,
        'query_valid': 0.0,
        'density': 3e-3 if fp16 else 2e-4,
        'domain_scores': 3e-3 if fp16 else 1e-4,
        'query_scores': 3e-3 if fp16 else 1e-4,
    }
    report = {}
    for name, expected, actual in zip(BeeEExportWrapper.output_names, reference, candidate):
        expected = torch.as_tensor(expected).detach().float().cpu()
        actual = torch.as_tensor(actual).detach().float().cpu()
        error = float((expected - actual).abs().max()) if expected.numel() else 0.0
        report[f'{name}_max_abs_error'] = error
        if error > tolerances[name]:
            raise ValueError(f'{name} error {error:.6g} exceeds {tolerances[name]:.6g}')
    reference_rank = torch.as_tensor(reference[-1]).detach().cpu().flatten().topk(min(topk, reference[-1].shape[-1])).indices
    candidate_rank = torch.as_tensor(candidate[-1]).detach().cpu().flatten().topk(min(topk, candidate[-1].shape[-1])).indices
    overlap = len(set(reference_rank.tolist()) & set(candidate_rank.tolist())) / max(len(reference_rank), 1)
    report['topk_query_ranking_overlap'] = overlap
    if overlap < (0.98 if fp16 else 1.0):
        raise ValueError(f'query ranking overlap {overlap:.3f} is below tolerance')
    return report


def _read_image(path, input_size, is_ir=False, return_meta=False,
                ir_quantile_bounds=None, ir_normalization=None):
    """Match validation preprocessing: IR percentile, letterbox, normalize."""
    with Image.open(path) as handle:
        image = handle.convert('RGB')
    source_w, source_h = image.size
    if is_ir:
        ir_normalization = ir_normalization or {
            'lower': 0.01, 'upper': 0.99,
            'foreground_residual_quantile': 0.75,
        }
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.uint8).copy()
        ).permute(2, 0, 1).float()
        gray = tensor.mean(dim=0, keepdim=True)
        if ir_quantile_bounds is None:
            low = torch.quantile(gray, float(ir_normalization['lower']))
            high = torch.quantile(gray, float(ir_normalization['upper']))
        else:
            low, high = torch.as_tensor(ir_quantile_bounds).reshape(-1)
        gray = (
            (gray.clamp(low, high) - low)
            / (high - low).clamp_min(1.0) * 255.0
        ).to(torch.uint8)
        image = Image.fromarray(
            gray.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(), mode='RGB'
        )

    scale = min(input_size / source_w, input_size / source_h)
    resized_w = max(1, round(source_w * scale))
    resized_h = max(1, round(source_h * scale))
    left = (input_size - resized_w) // 2
    top = (input_size - resized_h) // 2
    canvas = Image.new('RGB', (input_size, input_size), color=(114, 114, 114))
    resized = image.resize((resized_w, resized_h), resample=Image.Resampling.BILINEAR)
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array.copy())
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    tensor = ((tensor - mean) / std)[None]
    if not return_meta:
        return tensor
    return tensor, {
        'scale': scale, 'left': left, 'top': top,
        'source_width': source_w, 'source_height': source_h,
    }


def validate_four_modes(torch_model, session, rgb_image, ir_image, clip_len=3,
                        rgb_clip=None, ir_clip=None, fp16=False):
    modes = {
        'rgb_single_repeated': single_frame_as_clip(rgb_image, clip_len),
        'rgb_three_frame': rgb_clip if rgb_clip is not None else single_frame_as_clip(rgb_image, clip_len),
        'ir_single_repeated': single_frame_as_clip(ir_image, clip_len),
        'ir_three_frame': ir_clip if ir_clip is not None else single_frame_as_clip(ir_image, clip_len),
    }
    report = {}
    with torch.no_grad():
        for mode, samples in modes.items():
            temporal_mask = torch.ones(samples.shape[:2], dtype=torch.bool)
            if mode.endswith('single_repeated'):
                temporal_mask[:, :-1] = False
            domain_id = torch.tensor([
                1 if mode.startswith('ir_') else 0
            ], dtype=torch.long)
            stabilization_theta = identity_stabilization(
                samples.shape[0], dtype=samples.dtype, device=samples.device
            )
            reference = torch_model(
                samples, temporal_mask, domain_id, stabilization_theta
            )
            candidate = session.run(None, {
                'images': samples.cpu().numpy(),
                'temporal_valid_mask': temporal_mask.numpy(),
                'domain_id': domain_id.numpy(),
                'stabilization_theta': stabilization_theta.cpu().numpy(),
            })
            report[mode] = compare_outputs(reference, candidate, fp16=fp16)
    return report


def load_stratified_calibration_manifest(path, expected_annotation_sha256=None):
    """Load the RGB/IR video-density-scale-occlusion calibration evidence."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('split') != 'calibration':
        raise ValueError('ONNX numerical validation may only use split=calibration.')
    annotation = manifest.get('calibration_annotation', {})
    annotation_path = Path(annotation.get('path', '')).expanduser().resolve()
    annotation_sha = str(annotation.get('sha256') or '').lower()
    if not annotation_path.is_file() or len(annotation_sha) != 64:
        raise ValueError(
            'ONNX calibration manifest must bind the calibration annotation and SHA256.'
        )
    actual_annotation_sha = _sha256(annotation_path)
    if actual_annotation_sha != annotation_sha:
        raise ValueError('ONNX calibration annotation SHA256 mismatch.')
    if (
        expected_annotation_sha256 is not None
        and actual_annotation_sha != str(expected_annotation_sha256).lower()
    ):
        raise ValueError('ONNX evidence is not bound to the frozen calibration split.')
    records = manifest.get('frames', [])
    if not records:
        raise ValueError('calibration manifest contains no frames')
    required_strata = {'video_id', 'density_bin', 'scale_bin', 'occlusion_bin'}
    domains = set()
    identities = set()
    normalized = []
    for index, record in enumerate(records):
        domain = str(record.get('domain', '')).lower()
        if domain not in {'rgb', 'ir'}:
            raise ValueError(f'calibration frame {index} has invalid domain={domain!r}')
        image = Path(record['image']).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f'calibration image not found: {image}')
        clip = record.get('clip')
        if clip is not None:
            if len(clip) != 3:
                raise ValueError(f'calibration frame {index} clip must contain exactly 3 paths')
            clip = [Path(item).expanduser().resolve() for item in clip]
            missing = [str(item) for item in clip if not item.is_file()]
            if missing:
                raise FileNotFoundError(f'calibration clip paths not found: {missing}')
        strata = record.get('strata', {})
        missing_strata = required_strata - set(strata)
        if missing_strata:
            raise ValueError(
                f'calibration frame {index} misses strata={sorted(missing_strata)}'
            )
        identity = (domain, str(image), tuple(strata[name] for name in sorted(required_strata)))
        if identity in identities:
            raise ValueError(f'duplicate calibration evidence row: {identity}')
        identities.add(identity)
        domains.add(domain)
        normalized.append({
            'domain': domain,
            'image': image,
            'clip': clip,
            'strata': {name: strata[name] for name in sorted(required_strata)},
            'image_sha256': _sha256(image),
            'clip_sha256': [_sha256(item) for item in clip] if clip else None,
        })
    if domains != {'rgb', 'ir'}:
        raise ValueError('calibration manifest must contain both RGB and IR frames')
    return normalized


def validate_stratified_calibration(torch_model, session, records, input_size=1280,
                                    clip_len=3, fp16=False, preprocessing=None):
    preprocessing = preprocessing or {}
    if preprocessing.get('temporal_frames') != 3:
        raise ValueError('ONNX validation requires the frozen three-frame preprocessing contract.')
    if list(preprocessing.get('input_size', [])) != [input_size, input_size]:
        raise ValueError('ONNX validation input size disagrees with frozen preprocessing.')
    report = []
    for index, record in enumerate(records):
        is_ir = record['domain'] == 'ir'
        ir_normalization = preprocessing.get('ir_normalization', {})
        image = _read_image(
            record['image'], input_size, is_ir=is_ir,
            ir_normalization=ir_normalization,
        )
        clip = (
            _read_clip(
                record['clip'], input_size, is_ir=is_ir,
                ir_normalization=ir_normalization,
            )
            if record['clip'] else single_frame_as_clip(image, clip_len)
        )
        samples_by_mode = {
            'single_repeated': single_frame_as_clip(image, clip_len),
            'three_frame': clip,
        }
        modes = {}
        with torch.no_grad():
            for mode, samples in samples_by_mode.items():
                temporal_mask = torch.ones(samples.shape[:2], dtype=torch.bool)
                if mode == 'single_repeated':
                    temporal_mask[:, :-1] = False
                domain_id = torch.tensor([int(is_ir)], dtype=torch.long)
                theta = (
                    identity_stabilization(
                        samples.shape[0], dtype=samples.dtype, device=samples.device,
                    )
                    if mode == 'single_repeated' or record['clip'] is None
                    else _stabilization_theta(
                        record['clip'], input_size,
                        preprocessing.get('stabilizer', {}),
                    ).to(device=samples.device, dtype=samples.dtype)
                )
                reference = torch_model(samples, temporal_mask, domain_id, theta)
                candidate = session.run(None, {
                    'images': samples.cpu().numpy(),
                    'temporal_valid_mask': temporal_mask.numpy(),
                    'domain_id': domain_id.numpy(),
                    'stabilization_theta': theta.cpu().numpy(),
                })
                modes[mode] = compare_outputs(reference, candidate, fp16=fp16)
        report.append({
            'index': index, 'domain': record['domain'],
            'image': str(record['image']), 'image_sha256': record['image_sha256'],
            'clip_sha256': record['clip_sha256'], 'strata': record['strata'],
            'modes': modes,
        })
    return report


def _joint_ir_clip_bounds(paths, ir_normalization):
    frames = []
    reference_size = None
    for path in paths:
        with Image.open(path) as handle:
            image = handle.convert('RGB')
        reference_size = reference_size or image.size
        if image.size != reference_size:
            image = image.resize(reference_size, resample=Image.Resampling.BILINEAR)
        frames.append(torch.from_numpy(
            np.asarray(image, dtype=np.uint8).copy()
        ).float().mean(dim=-1))
    clip = torch.stack(frames)
    background = clip.median(dim=0).values
    residual = (clip - background).abs()
    foreground = clip[residual >= torch.quantile(
        residual, float(ir_normalization['foreground_residual_quantile']),
    )]
    joint = torch.cat((background.flatten(), foreground.flatten()))
    return torch.stack((
        torch.quantile(joint, float(ir_normalization['lower'])),
        torch.quantile(joint, float(ir_normalization['upper'])),
    ))


def _read_clip(paths, input_size, is_ir=False, ir_normalization=None):
    ir_normalization = ir_normalization or {
        'lower': 0.01, 'upper': 0.99,
        'foreground_residual_quantile': 0.75,
    }
    bounds = _joint_ir_clip_bounds(paths, ir_normalization) if is_ir else None
    return torch.stack([
        _read_image(
            path, input_size, is_ir=is_ir, ir_quantile_bounds=bounds,
            ir_normalization=ir_normalization,
        )[0] for path in paths
    ], dim=0)[None]


def _stabilization_theta(paths, input_size, stabilizer_config):
    from engine.data.transforms._transforms import PrepareTemporalFrames

    if len(paths) != 3:
        raise ValueError('stabilization requires exactly [long, short, current] paths')
    preparer = PrepareTemporalFrames(
        size=(input_size, input_size), stabilize=True,
        stabilizer=stabilizer_config,
    )
    frames = []
    for path in paths:
        with Image.open(path) as handle:
            frames.append(handle.convert('RGB'))
    current = preparer._stabilization_view(frames[-1])
    theta = [
        preparer.stabilizer.estimate(
            preparer._stabilization_view(frame), current,
        )[0]
        for frame in frames[:2]
    ]
    return torch.stack(theta)[None]


def profile_onnx_pipeline(session, image_sources, input_size, clip_len,
                          iterations=100, warmup_iterations=10, tracker=None,
                          preprocessing=None):
    from engine.tracking import PoseMotionTracker, TrackerConfig

    tracker = tracker or PoseMotionTracker(TrackerConfig(min_hits=1))
    preprocessing = preprocessing or {}
    if preprocessing.get('temporal_frames') != 3:
        raise ValueError('Latency acceptance requires the frozen three-frame contract.')
    if list(preprocessing.get('input_size', [])) != [input_size, input_size]:
        raise ValueError('Latency input size disagrees with frozen preprocessing.')
    ir_normalization = preprocessing.get('ir_normalization', {})
    warm_image = _read_image(
        image_sources[0][0], input_size, is_ir=image_sources[0][1],
        ir_normalization=ir_normalization,
    )
    warm_clip = single_frame_as_clip(warm_image, clip_len)
    warm_inputs = {
        'images': warm_clip.numpy(),
        'temporal_valid_mask': np.asarray(
            [[False] * (clip_len - 1) + [True]], dtype=np.bool_
        ),
        'domain_id': np.asarray([int(image_sources[0][1])], dtype=np.int64),
        'stabilization_theta': identity_stabilization(1).numpy(),
    }
    for _ in range(warmup_iterations):
        session.run(None, warm_inputs)
    profiler = EndToEndProfiler()
    for iteration in range(iterations):
        image_path, is_ir = image_sources[iteration % len(image_sources)]
        image, metadata = profiler.measure(
            'preprocess', _read_image, image_path, input_size, is_ir, True,
            None, ir_normalization,
        )
        clip = single_frame_as_clip(image, clip_len)
        outputs = profiler.measure(
            'model', session.run, None, {
            'images': clip.numpy(),
                'temporal_valid_mask': np.asarray(
                    [[False] * (clip_len - 1) + [True]], dtype=np.bool_
                ),
                'domain_id': np.asarray([int(is_ir)], dtype=np.int64),
                'stabilization_theta': identity_stabilization(1).numpy(),
            }
        )

        def postprocess():
            names = BeeEExportWrapper.output_names
            result = dict(zip(names, outputs))
            scores = torch.from_numpy(result['query_scores'][0])
            keep = scores.topk(min(300, len(scores))).indices
            boxes = torch.from_numpy(result['boxes'][0])[keep]
            boxes_xyxy = torch.stack((
                boxes[:, 0] - boxes[:, 2] / 2,
                boxes[:, 1] - boxes[:, 3] / 2,
                boxes[:, 0] + boxes[:, 2] / 2,
                boxes[:, 1] + boxes[:, 3] / 2,
            ), dim=-1) * input_size
            boxes_xyxy[:, 0::2] = (
                boxes_xyxy[:, 0::2] - metadata['left']
            ) / metadata['scale']
            boxes_xyxy[:, 1::2] = (
                boxes_xyxy[:, 1::2] - metadata['top']
            ) / metadata['scale']
            boxes_xyxy[:, 0::2].clamp_(0, metadata['source_width'])
            boxes_xyxy[:, 1::2].clamp_(0, metadata['source_height'])
            keypoints = torch.from_numpy(result['keypoints'][0])[keep] * input_size
            keypoints[..., 0] = (
                keypoints[..., 0] - metadata['left']
            ) / metadata['scale']
            keypoints[..., 1] = (
                keypoints[..., 1] - metadata['top']
            ) / metadata['scale']
            keypoints[..., 0].clamp_(0, metadata['source_width'])
            keypoints[..., 1].clamp_(0, metadata['source_height'])
            return {
                'boxes': boxes_xyxy,
                'scores': scores[keep],
                'keypoints': keypoints,
                'quality': torch.from_numpy(result['quality'][0])[keep].sigmoid(),
                'query_indices': keep,
            }

        detections = profiler.measure('postprocess', postprocess)
        tracks = profiler.measure('tracking', tracker.update, detections, iteration)
        profiler.measure(
            'output', json.dumps,
            [{key: value.tolist() if torch.is_tensor(value) else value
              for key, value in track.items()} for track in tracks]
        )
        profiler.sample_process_gpu_memory()
    return profiler.summarize()


class EndToEndProfiler:
    stages = ('preprocess', 'model', 'postprocess', 'tracking', 'output')

    def __init__(self):
        self.values = defaultdict(list)
        self.peak_memory_mb = None

    def measure(self, stage, function, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = function(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.values[stage].append((time.perf_counter() - started) * 1000.0)
        return result

    def sample_process_gpu_memory(self):
        """Observe ONNX Runtime CUDA memory without mistaking torch stats for ORT."""
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-compute-apps=pid,used_gpu_memory',
                    '--format=csv,noheader,nounits',
                ],
                check=True, capture_output=True, text=True, timeout=5,
            )
            used = []
            for line in result.stdout.splitlines():
                fields = [field.strip() for field in line.split(',')]
                if len(fields) == 2 and int(fields[0]) == os.getpid():
                    used.append(float(fields[1]))
            if used:
                value = sum(used)
                self.peak_memory_mb = max(self.peak_memory_mb or 0.0, value)
        except (FileNotFoundError, ValueError, subprocess.SubprocessError):
            return

    @staticmethod
    def _percentile(values, percentile):
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile) - 1))
        return ordered[index]

    def summarize(self):
        report = {}
        all_totals = []
        frame_count = max((len(values) for values in self.values.values()), default=0)
        for stage in self.stages:
            values = self.values.get(stage, [])
            if values:
                report[stage] = {
                    'mean_ms': sum(values) / len(values),
                    'p50_ms': self._percentile(values, 0.50),
                    'p95_ms': self._percentile(values, 0.95),
                    'p99_ms': self._percentile(values, 0.99),
                }
        for index in range(frame_count):
            all_totals.append(sum(
                self.values[stage][index] for stage in self.stages
                if index < len(self.values.get(stage, []))
            ))
        if all_totals:
            report['end_to_end'] = {
                'mean_ms': sum(all_totals) / len(all_totals),
                'p50_ms': self._percentile(all_totals, 0.50),
                'p95_ms': self._percentile(all_totals, 0.95),
                'p99_ms': self._percentile(all_totals, 0.99),
                'passes_10_second_limit': self._percentile(all_totals, 0.99) <= 10000.0,
            }
        report['peak_memory_mb'] = self.peak_memory_mb
        report['peak_memory_source'] = (
            'nvidia-smi process used_gpu_memory sampled after each full iteration'
            if self.peak_memory_mb is not None else 'unavailable'
        )
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration-contract', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--onnx-fp16', type=Path)
    parser.add_argument('--calibration-manifest', type=Path, required=True)
    parser.add_argument('--input-size', type=int, default=1280)
    parser.add_argument('--clip-len', type=int, default=3)
    parser.add_argument('--num-queries', type=int, default=768)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmup-iterations', type=int, default=10)
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import onnxruntime as ort

    if args.clip_len != 3:
        raise ValueError('BeePoseTrack-E deployment acceptance requires three frames.')
    calibration_contract = json.loads(
        args.calibration_contract.read_text(encoding='utf-8')
    )
    torch_model = BeeEExportWrapper(
        _load_model(
            args.config, args.checkpoint, calibration_contract,
        ).eval(),
        num_queries=args.num_queries,
        calibration_contract=calibration_contract,
    ).eval()
    calibration_records = load_stratified_calibration_manifest(
        args.calibration_manifest,
        expected_annotation_sha256=calibration_contract.get('source', {}).get(
            'calibration_annotation_sha256'
        ),
    )
    session = ort.InferenceSession(
        str(args.onnx), providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    if not args.allow_cpu and 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError(
            'Windows CUDA deployment acceptance requires CUDAExecutionProvider; '
            'use --allow-cpu only for non-acceptance diagnostics.'
        )
    report = {
        'artifacts': {
            'config': str(args.config.resolve()),
            'config_sha256': _sha256(args.config),
            'checkpoint': str(args.checkpoint.resolve()),
            'checkpoint_sha256': _sha256(args.checkpoint),
            'calibration_contract': str(args.calibration_contract.resolve()),
            'calibration_contract_sha256': _sha256(args.calibration_contract),
            'onnx_fp32': str(args.onnx.resolve()),
            'onnx_fp32_sha256': _sha256(args.onnx),
            'onnx_fp32_bytes': args.onnx.stat().st_size,
            'calibration_manifest': str(args.calibration_manifest.resolve()),
            'calibration_manifest_sha256': _sha256(args.calibration_manifest),
        },
        'environment': {
            'platform': platform.platform(),
            'python': sys.version,
            'torch': torch.__version__,
            'onnxruntime': ort.__version__,
            'providers': session.get_providers(),
            'acceptance_provider': 'CUDAExecutionProvider',
        },
        'graph_fp32': inspect_fixed_graph(
        args.onnx, args.input_size, args.clip_len, args.num_queries
        ),
    }
    report['numerical_fp32'] = validate_stratified_calibration(
        torch_model, session, calibration_records,
        args.input_size, args.clip_len,
        preprocessing=calibration_contract.get('preprocessing'),
    )
    if args.onnx_fp16:
        report['artifacts'].update({
            'onnx_fp16': str(args.onnx_fp16.resolve()),
            'onnx_fp16_sha256': _sha256(args.onnx_fp16),
            'onnx_fp16_bytes': args.onnx_fp16.stat().st_size,
        })
        report['graph_fp16'] = inspect_fixed_graph(
            args.onnx_fp16, args.input_size, args.clip_len, args.num_queries
        )
        fp16_session = ort.InferenceSession(
            str(args.onnx_fp16), providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        report['numerical_fp16'] = validate_stratified_calibration(
            torch_model, fp16_session, calibration_records,
            args.input_size, args.clip_len, fp16=True,
            preprocessing=calibration_contract.get('preprocessing'),
        )
    report['latency'] = profile_onnx_pipeline(
        session,
        [(record['image'], record['domain'] == 'ir') for record in calibration_records],
        args.input_size,
        args.clip_len, iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        preprocessing=calibration_contract.get('preprocessing'),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
