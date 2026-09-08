"""IR-specific augmentation primitives."""

from __future__ import annotations

import cv2
import numpy as np


def ir_percentile_stretch(image: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    lo, hi = np.percentile(gray.astype(np.float32), [lo_q, hi_q])
    if hi <= lo:
        out = np.zeros_like(gray, dtype=np.uint8)
    else:
        out = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        out = (out * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def ir_sensor_noise(image: np.ndarray, sigma: float = 5.0, fixed_pattern: float = 0.03) -> np.ndarray:
    arr = image.astype(np.float32)
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    col = np.random.normal(0.0, 255.0 * fixed_pattern, (1, arr.shape[1], 1)).astype(np.float32)
    arr = arr + noise + col
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)
