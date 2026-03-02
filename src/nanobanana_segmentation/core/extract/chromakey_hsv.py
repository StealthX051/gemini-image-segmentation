from __future__ import annotations

import cv2
import numpy as np


def extract_chromakey_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Extract ROI mask by identifying pure-green background in HSV space and inverting."""

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 180, 120], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    green = cv2.inRange(hsv, lower_green, upper_green)
    roi = cv2.bitwise_not(green)

    kernel = np.ones((3, 3), dtype=np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    return roi
