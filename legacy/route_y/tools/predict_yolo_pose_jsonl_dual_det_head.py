#!/usr/bin/env python3
"""Export dual-domain-head YOLO pose predictions in BeePoseTrack JSONL format."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect, Pose26

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.yolo26_bee_pose import register_ultralytics_modules
from data.dataset import FiveFrameManifestIndex, frame_statistics, read_rgb_or_ir
from models.route_context import begin_route_context


FRAME_RE = re.compile(r"(?P<video>[AB]-5-\d+)(?:_section(?P<section>\d+))?.*?frame_?0*(?P<frame>\d+)", re.IGNORECASE)


def infer_domain_id(path: str | Path, domain: str | None = None) -> int:
    if domain:
        return 1 if domain.upper() == "IR" else 0
    name = Path(path).name.upper()
    return 1 if name.startswith("B-") or "_IR" in name or "/IR/" in str(path).upper() else 0


def _select_domain_tensor(rgb: torch.Tensor, ir: torch.Tensor, domain_ids: torch.Tensor | None) -> torch.Tensor:
    if domain_ids is None:
        return rgb
    mask = domain_ids.to(device=rgb.device, dtype=torch.bool).view(-1, 1, 1)
    return torch.where(mask, ir, rgb)


def dual_domain_forward_head(
    self,
    x: list[torch.Tensor],
    box_head: torch.nn.Module = None,
    cls_head: torch.nn.Module = None,
    pose_head: torch.nn.Module = None,
    kpts_head: torch.nn.Module = None,
    kpts_sigma_head: torch.nn.Module = None,
) -> dict[str, torch.Tensor]:
    if not hasattr(self, "cv2_rgb"):
        return self._bee_standard_forward_head(x, box_head, cls_head, pose_head, kpts_head, kpts_sigma_head)

    rgb_preds = Detect.forward_head(self, x, self.cv2_rgb, self.cv3_rgb)
    ir_preds = Detect.forward_head(self, x, self.cv2_ir, self.cv3_ir)
    domain_ids = getattr(self, "bee_domain_ids", None)
    preds = {
        "boxes": _select_domain_tensor(rgb_preds["boxes"], ir_preds["boxes"], domain_ids),
        "scores": _select_domain_tensor(rgb_preds["scores"], ir_preds["scores"], domain_ids),
        "feats": x,
    }
    if pose_head is not None:
        bs = x[0].shape[0]
        features = [self.cv4[i](x[i]) for i in range(self.nl)]
        preds["kpts"] = torch.cat([self.cv4_kpts[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
    return preds


def install_dual_domain_class_patch() -> None:
    if not hasattr(Pose26, "_bee_standard_forward_head"):
        Pose26._bee_standard_forward_head = Pose26.forward_head
    Pose26.dual_domain_forward_head = dual_domain_forward_head
    Pose26.forward_head = dual_domain_forward_head


def install_dual_domain_detection_heads(model: torch.nn.Module) -> None:
    head = model.model[-1]
    if "forward_head" in head.__dict__:
        delattr(head, "forward_head")
    if "_bee_original_forward_head" in head.__dict__:
        delattr(head, "_bee_original_forward_head")
    if not hasattr(head, "cv2_rgb"):
        head.cv2_rgb = copy.deepcopy(head.cv2)
        head.cv3_rgb = copy.deepcopy(head.cv3)
        head.cv2_ir = copy.deepcopy(head.cv2)
        head.cv3_ir = copy.deepcopy(head.cv3)
        head.bee_domain_ids = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True, help="Image directory, image file, or txt list.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="1")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=768)
    parser.add_argument("--model-name", default="BeePoseTrack-Y-Aligned-DualDetHead")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--five-frame", action="store_true")
    return parser.parse_args()


def parse_ids(path: Path) -> tuple[str, str, int, str]:
    match = FRAME_RE.search(path.stem)
    if match:
        video_id = match.group("video").upper()
        section = match.group("section")
        section_id = f"{video_id}_section_{int(section):02d}" if section else video_id
        return video_id, section_id, int(match.group("frame")), "RGB" if video_id.startswith("A") else "IR"
    fallback = path.parent.name
    return fallback, fallback, -1, "unknown"


def point_or_null(xy, conf):
    if xy is None or conf is None or float(conf) <= 0:
        return None
    return [round(float(xy[0]), 3), round(float(xy[1]), 3)]


def frame_to_tensor(frame, index: FiveFrameManifestIndex, imgsz: int) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = []
    images = []
    for ref in index.causal_clip(frame):
        img = read_rgb_or_ir(ref.image_path, ref.domain)
        images.append(img)
        img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        tensors.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
    clip = torch.cat(tensors, dim=0).unsqueeze(0)
    motion = 0.0
    if len(images) > 1:
        a = cv2.cvtColor(images[-1], cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
        b = cv2.cvtColor(images[-2], cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        motion = float(abs(a - b).mean())
    stats = torch.from_numpy(frame_statistics(images[-1], frame.domain, len(frame.instances), motion)).float().unsqueeze(0)
    return clip, stats


def set_inference_domain(model: YOLO, domain_id: int) -> None:
    head = model.model.model[-1]
    head.bee_domain_ids = torch.tensor([domain_id], dtype=torch.long)


def write_record(handle, result, image_path: Path, video_id: str, section_id: str, frame_id: int, domain: str, model_name: str, checkpoint: str, scale_xy: tuple[float, float] = (1.0, 1.0)) -> None:
    boxes = result.boxes
    keypoints = result.keypoints
    detections = []
    n = 0 if boxes is None else len(boxes)
    xyxy = boxes.xyxy.cpu().numpy() if n else []
    scores = boxes.conf.cpu().numpy() if n else []
    kxy = keypoints.xy.cpu().numpy() if keypoints is not None else None
    kconf = keypoints.conf.cpu().numpy() if keypoints is not None and keypoints.conf is not None else None
    sx, sy = scale_xy
    for i in range(n):
        head_score = float(kconf[i, 0]) if kconf is not None else 0.0
        tail_score = float(kconf[i, 1]) if kconf is not None else 0.0
        pose_score = (head_score + tail_score) / 2.0 if kconf is not None else 0.0
        box = xyxy[i].copy()
        box[[0, 2]] *= sx
        box[[1, 3]] *= sy
        pts = None if kxy is None else kxy[i].copy()
        if pts is not None:
            pts[:, 0] *= sx
            pts[:, 1] *= sy
        detections.append(
            {
                "bbox_xyxy": [round(float(v), 3) for v in box.tolist()],
                "score": round(float(scores[i]), 6),
                "head": point_or_null(pts[0] if pts is not None else None, head_score),
                "tail": point_or_null(pts[1] if pts is not None else None, tail_score),
                "head_score": round(head_score, 6),
                "tail_score": round(tail_score, 6),
                "pose_score": round(pose_score, 6),
            }
        )
    record = {
        "video_id": video_id,
        "section_id": section_id,
        "frame_id": frame_id,
        "domain": domain,
        "image_path": str(image_path),
        "model_name": model_name,
        "checkpoint": checkpoint,
        "detections": detections,
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_sources(source: str) -> list[Path]:
    path = Path(source)
    if path.is_file() and path.suffix.lower() == ".txt":
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def main() -> None:
    args = parse_args()
    register_ultralytics_modules()
    install_dual_domain_class_patch()
    model = YOLO(args.model)
    install_dual_domain_detection_heads(model.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        if args.five_frame:
            if args.manifest is None:
                raise ValueError("--five-frame requires --manifest")
            index = FiveFrameManifestIndex(args.manifest, args.eval_split, 5, 1)
            for frame in index.frames:
                clip, stats = frame_to_tensor(frame, index, args.imgsz)
                domain_id = infer_domain_id(frame.image_path, frame.domain)
                set_inference_domain(model, domain_id)
                domain_ids = torch.tensor([domain_id], dtype=torch.long)
                begin_route_context(clip=clip.view(1, 5, 3, args.imgsz, args.imgsz), domain_ids=domain_ids, stats=stats, source="five_frame_infer")
                result = model.predict(
                    source=clip,
                    imgsz=args.imgsz,
                    device=args.device,
                    conf=args.conf,
                    iou=args.iou,
                    max_det=args.max_det,
                    stream=False,
                    verbose=False,
                )[0]
                scale_xy = (float(frame.width or args.imgsz) / float(args.imgsz), float(frame.height or args.imgsz) / float(args.imgsz))
                write_record(handle, result, frame.image_path, frame.video_id, frame.section_id, frame.frame_id, frame.domain, args.model_name, str(Path(args.model).resolve()), scale_xy=scale_xy)
            return
        for image_path in iter_sources(args.source):
            video_id, section_id, frame_id, domain = parse_ids(image_path)
            set_inference_domain(model, infer_domain_id(image_path, domain))
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                stream=False,
                verbose=False,
            )[0]
            write_record(handle, result, image_path, video_id, section_id, frame_id, domain, args.model_name, str(Path(args.model).resolve()))


if __name__ == "__main__":
    main()
