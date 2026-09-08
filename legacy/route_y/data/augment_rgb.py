"""RGB-specific augmentation primitives."""

from __future__ import annotations

import cv2
import numpy as np


def rgb_photometric(image: np.ndarray, brightness: float = 0.08, contrast: float = 0.12, temperature: float = 0.04) -> np.ndarray:
    arr = image.astype(np.float32) / 255.0
    arr = (arr - 0.5) * (1.0 + contrast) + 0.5 + brightness
    arr[..., 0] = np.clip(arr[..., 0] + temperature, 0.0, 1.0)
    arr[..., 2] = np.clip(arr[..., 2] - temperature, 0.0, 1.0)
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def directional_motion_blur(image: np.ndarray, kernel: int = 5, angle_deg: float = 0.0) -> np.ndarray:
    kernel = max(int(kernel) | 1, 3)
    k = np.zeros((kernel, kernel), dtype=np.float32)
    k[kernel // 2, :] = 1.0 / kernel
    m = cv2.getRotationMatrix2D((kernel / 2 - 0.5, kernel / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, m, (kernel, kernel))
    k /= max(float(k.sum()), 1e-6)
    return cv2.filter2D(image, -1, k)
