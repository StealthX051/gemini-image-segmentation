from __future__ import annotations

import cv2
import numpy as np


def fill_holes(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, flood_inv)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if mask.size == 0 or min_area <= 1:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == label] = 255
    return out


def keep_largest_components(mask: np.ndarray, max_components: int) -> np.ndarray:
    if max_components <= 0:
        return np.zeros_like(mask, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask

    areas = []
    for label in range(1, num_labels):
        areas.append((int(stats[label, cv2.CC_STAT_AREA]), label))
    areas.sort(reverse=True)
    keep = {label for _, label in areas[:max_components]}

    out = np.zeros_like(mask, dtype=np.uint8)
    for label in keep:
        out[labels == label] = 255
    return out


def apply_profile_morphology(mask: np.ndarray, task_profile: str) -> np.ndarray:
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    kernel5 = np.ones((5, 5), dtype=np.uint8)
    out = mask.copy()
    profile = (task_profile or "blob").strip().lower()

    if profile == "thin":
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel3, iterations=1)
    elif profile == "low_contrast":
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel5, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel3, iterations=1)
    else:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel3, iterations=1)

    return out


def standard_postprocess(
    mask: np.ndarray,
    *,
    image_area: int,
    task_profile: str,
    min_component_area_frac: float = 0.0001,
    max_components: int = 1024,
) -> np.ndarray:
    out = (mask > 0).astype(np.uint8) * 255
    out = fill_holes(out)
    min_area = max(1, int(image_area * float(min_component_area_frac)))
    out = remove_small_components(out, min_area=min_area)
    out = keep_largest_components(out, max_components=max_components)
    out = apply_profile_morphology(out, task_profile=task_profile)
    return out
