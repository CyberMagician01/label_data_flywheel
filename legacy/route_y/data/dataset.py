"""Manifest dataset helpers for BeePoseTrack-Y Route-Best."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameRef:
    image_path: Path
    video_id: str
    section_id: str
    frame_id: int
    domain: str
    split: str
    width: int
    height: int
    instances: list[dict[str, Any]]


def load_manifest(manifest: Path, split: str | None = None) -> list[FrameRef]:
    rows: list[FrameRef] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split is not None and row.get("split") != split:
            continue
        rows.append(
            FrameRef(
                image_path=Path(row["image_path"]),
                video_id=str(row["video_id"]),
                section_id=str(row.get("section_id", row["video_id"])),
                frame_id=int(row["frame_id"]),
                domain=str(row.get("domain", "unknown")),
                split=str(row.get("split", "")),
                width=int(row.get("width", 0)),
                height=int(row.get("height", 0)),
                instances=list(row.get("instances", [])),
            )
        )
    return rows


class FiveFrameManifestIndex:
    """Causal five-frame index based on the frozen shared manifest."""

    def __init__(self, manifest: Path, split: str, clip_len: int = 5, temporal_stride: int = 1) -> None:
        self.frames = load_manifest(manifest, split)
        self.clip_len = clip_len
        self.temporal_stride = temporal_stride
        self.by_video: dict[str, list[FrameRef]] = {}
        for frame in self.frames:
            self.by_video.setdefault(frame.section_id, []).append(frame)
        for seq in self.by_video.values():
            seq.sort(key=lambda x: x.frame_id)
        self.position = {(f.section_id, f.frame_id): i for seq in self.by_video.values() for i, f in enumerate(seq)}

    def causal_clip(self, frame: FrameRef) -> list[FrameRef]:
        seq = self.by_video[frame.section_id]
        idx = self.position[(frame.section_id, frame.frame_id)]
        refs = []
        for lag in range((self.clip_len - 1) * self.temporal_stride, -1, -self.temporal_stride):
            refs.append(seq[max(idx - lag, 0)])
        return refs


def read_rgb_or_ir(path: Path, domain: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if str(domain).upper() == "IR":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray = gray.astype(np.float32)
        lo, hi = np.percentile(gray, [1.0, 99.0])
        if hi <= lo:
            out = np.zeros_like(gray, dtype=np.uint8)
        else:
            out = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
            out = (out * 255.0 + 0.5).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def frame_statistics(image: np.ndarray, domain: str, instance_count: int, motion_energy: float = 0.0) -> np.ndarray:
    arr = image.astype(np.float32) / 255.0
    contrast = float(arr.std())
    brightness = float(arr.mean())
    channel_delta = float(np.abs(arr[..., 0] - arr[..., 1]).mean() + np.abs(arr[..., 1] - arr[..., 2]).mean())
    domain_id = 1.0 if str(domain).upper() == "IR" else 0.0
    return np.asarray([domain_id, brightness, contrast, channel_delta, float(instance_count), float(motion_energy)], dtype=np.float32)
