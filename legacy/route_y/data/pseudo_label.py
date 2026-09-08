"""Pseudo-label containers for the semi-supervised Y4 stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PseudoLabelSource:
    detections_jsonl: Path
    teacher_checkpoint: Path
    quality_weight: float
    source_name: str = "ema_trex2_track_verified"


def pseudo_label_enabled(source: PseudoLabelSource | None) -> bool:
    return source is not None and source.detections_jsonl.exists() and source.teacher_checkpoint.exists()
