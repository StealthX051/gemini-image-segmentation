from __future__ import annotations

import cv2
import numpy as np


def extract_bw_threshold(image_bgr: np.ndarray, threshold: int = 127, use_otsu: bool = False) -> np.ndarray:
    """Extract ROI from black-white surrogate image."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if use_otsu:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask
