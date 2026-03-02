from __future__ import annotations

import cv2
import numpy as np


def extract_chromakey_ratio(image_bgr: np.ndarray, ratio_threshold: float = 1.25) -> np.ndarray:
    """Extract ROI by green-dominance ratio and invert background."""

    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)

    ratio = g / (np.maximum(r, b) + 1.0)
    green_dominant = (g > r) & (g > b)
    green_mask = green_dominant & (ratio > float(ratio_threshold))

    bg = (green_mask.astype(np.uint8) * 255)
    bg = cv2.dilate(bg, np.ones((3, 3), dtype=np.uint8), iterations=1)
    roi = cv2.bitwise_not(bg)
    return roi
