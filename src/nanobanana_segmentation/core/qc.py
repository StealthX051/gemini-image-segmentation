from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .types import ConstraintConfig, QCMetrics


def _component_stats(mask: np.ndarray) -> Tuple[int, np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0, np.zeros((0, 5), dtype=np.int32)
    return n - 1, stats[1:]


def compute_qc_metrics(
    *,
    mask: np.ndarray,
    surrogate: np.ndarray,
    input_shape: Tuple[int, int],
    attempt_mode: str,
) -> QCMetrics:
    h, w = input_shape
    resolution_match = surrogate.shape[:2] == (h, w)

    fg = (mask > 0)
    nonempty = bool(np.any(fg))
    area_frac = float(np.mean(fg)) if fg.size else 0.0

    component_count, component_stats = _component_stats(mask)
    largest_component_frac = 0.0
    speckle_score = 1.0
    if component_count > 0 and area_frac > 0:
        component_areas = component_stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        largest_component_frac = float(component_areas.max() / max(1.0, component_areas.sum()))
        tiny = component_areas[component_areas <= 0.001 * (h * w)]
        speckle_score = float(tiny.sum() / max(1.0, component_areas.sum()))

    border_touch = bool(
        np.any(fg[0, :]) or np.any(fg[-1, :]) or np.any(fg[:, 0]) or np.any(fg[:, -1])
    )

    green_coverage = None
    green_uniformity = None
    if attempt_mode == "chromakey":
        b = surrogate[:, :, 0].astype(np.int16)
        g = surrogate[:, :, 1].astype(np.int16)
        r = surrogate[:, :, 2].astype(np.int16)
        green_bg = (g > 220) & (r < 40) & (b < 40)
        green_coverage = float(np.mean(green_bg))
        green_pixels = surrogate[green_bg]
        if green_pixels.size:
            green_uniformity = float(np.std(green_pixels.reshape(-1, 3), axis=0).mean())
        else:
            green_uniformity = float("inf")

    return QCMetrics(
        resolution_match=resolution_match,
        mask_nonempty=nonempty,
        mask_area_frac=area_frac,
        component_count=component_count,
        largest_component_frac=largest_component_frac,
        speckle_score=speckle_score,
        border_touch=border_touch,
        green_coverage=green_coverage,
        green_uniformity_proxy=green_uniformity,
    )


def evaluate_qc(
    metrics: QCMetrics,
    constraints: ConstraintConfig,
    *,
    attempt_mode: str,
) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    if not metrics.resolution_match:
        failures.append("resolution_mismatch")
    if not metrics.mask_nonempty:
        failures.append("empty_mask")
    if metrics.mask_area_frac < float(constraints.min_area_frac):
        failures.append("area_too_small")
    if metrics.mask_area_frac > float(constraints.max_area_frac):
        failures.append("area_too_large")
    if constraints.single_component and metrics.component_count > 1:
        failures.append("multiple_components")
    if metrics.component_count < int(constraints.min_components):
        failures.append("too_few_components")
    if metrics.component_count > int(constraints.max_components):
        failures.append("too_many_components")
    if not constraints.allow_border_touch and metrics.border_touch:
        failures.append("border_touch")
    if metrics.speckle_score > 0.20:
        failures.append("severe_speckle")

    if attempt_mode == "chromakey":
        if metrics.green_coverage is None or metrics.green_coverage < 0.10:
            failures.append("low_green_coverage")
        if metrics.green_uniformity_proxy is None or metrics.green_uniformity_proxy > 25.0:
            failures.append("low_green_uniformity")

    return len(failures) == 0, failures


def score_attempt(metrics: QCMetrics, *, qc_pass: bool, failures: List[str]) -> float:
    score = 0.0
    score += 1000.0 if qc_pass else 0.0
    score -= 100.0 * float(len(failures))
    score -= 0.8 * float(metrics.component_count)
    score -= 120.0 * float(metrics.speckle_score)
    score -= 20.0 if metrics.border_touch else 0.0

    if metrics.green_uniformity_proxy is not None:
        score -= 0.25 * float(metrics.green_uniformity_proxy)
    if metrics.green_coverage is not None:
        score += 10.0 * float(metrics.green_coverage)

    score += 2.5 * float(metrics.largest_component_frac)
    return score


def metrics_to_dict(metrics: QCMetrics) -> Dict[str, object]:
    return asdict(metrics)
