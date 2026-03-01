from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import color

from .config import ITAConfig


def to_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = np.max(arr, axis=-1)
    if arr.dtype == np.bool_:
        return arr

    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=bool)

    max_val = float(np.max(finite))
    if max_val <= 1.0:
        # Support masks stored as 0/1 or probabilities in [0, 1].
        return arr > 0.5
    return arr > 127.0


def build_field_mask(image_rgb: np.ndarray, floor: int = 5) -> np.ndarray:
    gray = np.mean(image_rgb.astype(np.float32), axis=-1)
    field = gray > float(floor)
    if not np.any(field):
        return np.ones(gray.shape, dtype=bool)
    field = ndimage.binary_fill_holes(field)
    field = ndimage.binary_opening(field, structure=np.ones((3, 3), dtype=bool))
    return field.astype(bool)


def _radius_from_frac(shape: Tuple[int, int], frac: float) -> int:
    h, w = shape
    base = min(h, w)
    if float(frac) <= 0.0:
        return 0
    return max(1, int(round(frac * float(base))))


def _bbox_from_mask(mask: np.ndarray, pad_px: int) -> Tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if yy.size == 0 or xx.size == 0:
        return (0, mask.shape[0], 0, mask.shape[1])
    y0 = max(0, int(yy.min()) - int(pad_px))
    y1 = min(mask.shape[0], int(yy.max()) + int(pad_px) + 1)
    x0 = max(0, int(xx.min()) - int(pad_px))
    x1 = min(mask.shape[1], int(xx.max()) + int(pad_px) + 1)
    return y0, y1, x0, x1


def _resize_pair(
    image_rgb: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    max_dim: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    h, w = image_rgb.shape[:2]
    max_hw = max(h, w)
    if max_hw <= max_dim:
        return image_rgb, lesion_mask.astype(bool), 1.0
    scale = float(max_dim) / float(max_hw)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(
        lesion_mask.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized_rgb, resized_mask > 0, scale


def _rgb_pixels_to_lab(rgb_pixels: np.ndarray) -> np.ndarray:
    if rgb_pixels.size == 0:
        return np.empty((0, 3), dtype=float)
    rgb01 = np.clip(rgb_pixels.astype(np.float32) / 255.0, 0.0, 1.0)
    lab = color.rgb2lab(rgb01.reshape(-1, 1, 3)).reshape(-1, 3)
    return np.asarray(lab, dtype=float)


def _build_ring_mask(
    lesion_mask: np.ndarray,
    *,
    ita_cfg: ITAConfig,
    field_mask: np.ndarray,
    outer_radius_cap: int,
    inner_radius_cap: int,
) -> np.ndarray:
    outer_r = min(
        int(outer_radius_cap),
        _radius_from_frac(lesion_mask.shape, ita_cfg.ring_outer_frac_min_dim),
    )
    inner_r = min(
        int(inner_radius_cap),
        _radius_from_frac(lesion_mask.shape, ita_cfg.ring_inner_frac_min_dim),
    )
    outer_r = max(1, int(outer_r))
    inner_r = max(0, int(min(inner_r, outer_r)))

    outer = ndimage.binary_dilation(
        lesion_mask,
        structure=np.ones((outer_r * 2 + 1, outer_r * 2 + 1), dtype=bool),
    )
    if inner_r <= 0:
        inner = lesion_mask
    else:
        inner = ndimage.binary_dilation(
            lesion_mask,
            structure=np.ones((inner_r * 2 + 1, inner_r * 2 + 1), dtype=bool),
        )
    ring = np.logical_and(outer, np.logical_not(inner))
    return np.logical_and(ring, field_mask)


def _filter_lab_with_lstar_window(
    lab: np.ndarray,
    *,
    enabled: bool,
    low_pct: float,
    high_pct: float,
) -> np.ndarray:
    if not enabled or lab.size == 0:
        return lab
    l_vals = lab[:, 0]
    lo = float(np.percentile(l_vals, max(0.0, min(100.0, low_pct))))
    hi = float(np.percentile(l_vals, max(0.0, min(100.0, high_pct))))
    mask = np.logical_and(l_vals >= lo, l_vals <= hi)
    filtered = lab[mask]
    return filtered if filtered.size else lab


def _aggregate_channel(values: np.ndarray, *, mode: str, trim_std: float) -> float:
    if values.size == 0:
        return math.nan
    if mode == "mean":
        return float(np.mean(values))
    if mode == "trimmed_mean_sd":
        mu = float(np.mean(values))
        sigma = float(np.std(values))
        if sigma <= 0.0 or not np.isfinite(sigma):
            return mu
        z = abs(float(trim_std))
        keep = np.abs(values - mu) <= (z * sigma)
        trimmed = values[keep]
        if trimmed.size == 0:
            return mu
        return float(np.mean(trimmed))
    return float(np.median(values))


def _aggregate_lab(lab: np.ndarray, *, mode: str, trim_std: float) -> Tuple[float, float]:
    if lab.size == 0:
        return math.nan, math.nan
    l_vals = lab[:, 0]
    b_vals = lab[:, 2]
    if mode == "trimmed_mean_sd":
        l_mu = float(np.mean(l_vals))
        l_sigma = float(np.std(l_vals))
        b_mu = float(np.mean(b_vals))
        b_sigma = float(np.std(b_vals))
        l_keep = np.ones(l_vals.shape, dtype=bool) if l_sigma <= 0.0 else np.abs(l_vals - l_mu) <= abs(trim_std) * l_sigma
        b_keep = np.ones(b_vals.shape, dtype=bool) if b_sigma <= 0.0 else np.abs(b_vals - b_mu) <= abs(trim_std) * b_sigma
        keep = np.logical_and(l_keep, b_keep)
        trimmed = lab[keep]
        if trimmed.size == 0:
            trimmed = lab
        return float(np.mean(trimmed[:, 0])), float(np.mean(trimmed[:, 2]))
    l_agg = _aggregate_channel(l_vals, mode=mode, trim_std=trim_std)
    b_agg = _aggregate_channel(b_vals, mode=mode, trim_std=trim_std)
    return l_agg, b_agg


def _ita_from_aggregated_lab(l_value: float, b_value: float, *, eps: float) -> float:
    if not np.isfinite(l_value) or not np.isfinite(b_value):
        return math.nan
    return float(np.arctan((l_value - 50.0) / (b_value + float(eps))) * (180.0 / np.pi))


def _ita_from_pixelwise_lab(lab: np.ndarray, *, eps: float) -> float:
    if lab.size == 0:
        return math.nan
    ita_values = np.arctan((lab[:, 0] - 50.0) / (lab[:, 2] + float(eps))) * (180.0 / np.pi)
    if ita_values.size == 0:
        return math.nan
    return float(np.median(ita_values))


def _default_ita_payload(image_rgb: np.ndarray) -> Dict[str, object]:
    empty = np.zeros(image_rgb.shape[:2], dtype=bool)
    return {
        "ita_deg": math.nan,
        "ring_pixel_count": 0,
        "ring_area_frac": 0.0,
        "ring_valid": False,
        "ita_candidate_count": 0,
        "ita_region_strategy": "none",
        "ita_estimator": "none",
        "ita_aggregation_stat": "none",
        "Lstar_skin_median": math.nan,
        "bstar_skin_median": math.nan,
        "ring_mask": empty,
        "field_mask": np.ones(image_rgb.shape[:2], dtype=bool),
        "lesion_mask": empty,
        "lab_ring_median": np.array([math.nan, math.nan, math.nan], dtype=float),
        "lab_lesion_median": np.array([math.nan, math.nan, math.nan], dtype=float),
        "Lstar_skin_mean": math.nan,
        "Lstar_skin_std": math.nan,
        "working_rgb": image_rgb,
        "working_scale": 1.0,
        "lab": None,
    }


def compute_ita_features(
    image_rgb: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    ita_cfg: ITAConfig,
    roi_max_dim: int = 768,
    ring_outer_radius_cap: int = 64,
    ring_inner_radius_cap: int = 24,
    ring_roi_pad_px: int = 24,
) -> Dict[str, object]:
    lesion_full = to_bool_mask(lesion_mask)
    if not np.any(lesion_full):
        return _default_ita_payload(image_rgb)

    region_strategy = str(getattr(ita_cfg, "region_strategy", "perilesional_ring")).strip().lower()
    if region_strategy not in {"perilesional_ring", "global_nonlesion"}:
        region_strategy = "perilesional_ring"

    if region_strategy == "global_nonlesion":
        working_rgb = image_rgb
        working_lesion = lesion_full
        scale = 1.0
        field_mask = (
            build_field_mask(working_rgb, floor=ita_cfg.field_intensity_floor)
            if ita_cfg.use_field_mask
            else np.ones(working_rgb.shape[:2], dtype=bool)
        )
        ring_mask = np.logical_and(np.logical_not(working_lesion), field_mask)
    else:
        y0, y1, x0, x1 = _bbox_from_mask(lesion_full, pad_px=ring_roi_pad_px)
        roi_rgb = image_rgb[y0:y1, x0:x1]
        roi_lesion = lesion_full[y0:y1, x0:x1]
        working_rgb, working_lesion, scale = _resize_pair(
            roi_rgb,
            roi_lesion,
            max_dim=max(64, int(roi_max_dim)),
        )
        field_mask = (
            build_field_mask(working_rgb, floor=ita_cfg.field_intensity_floor)
            if ita_cfg.use_field_mask
            else np.ones(working_rgb.shape[:2], dtype=bool)
        )
        ring_mask = _build_ring_mask(
            working_lesion,
            ita_cfg=ita_cfg,
            field_mask=field_mask,
            outer_radius_cap=max(1, int(ring_outer_radius_cap)),
            inner_radius_cap=max(0, int(ring_inner_radius_cap)),
        )

    ring_count = int(ring_mask.sum())
    ring_area_frac = float(ring_count / float(ring_mask.size)) if ring_mask.size else 0.0
    ring_valid = bool(
        ring_count >= int(ita_cfg.ring_min_pixels)
        and ring_area_frac >= float(ita_cfg.ring_min_area_frac)
    )

    ring_rgb = working_rgb[ring_mask]
    lesion_rgb = working_rgb[working_lesion]
    ring_lab = _rgb_pixels_to_lab(ring_rgb)
    lesion_lab = _rgb_pixels_to_lab(lesion_rgb)

    candidate_lab = _filter_lab_with_lstar_window(
        ring_lab,
        enabled=bool(getattr(ita_cfg, "apply_lstar_window", False)),
        low_pct=float(getattr(ita_cfg, "lstar_window_low_pct", 5.0)),
        high_pct=float(getattr(ita_cfg, "lstar_window_high_pct", 95.0)),
    )
    candidate_count = int(candidate_lab.shape[0])

    if ring_lab.size == 0:
        ring_lab_median = np.array([math.nan, math.nan, math.nan], dtype=float)
        l_med = math.nan
        b_med = math.nan
        l_mean = math.nan
        l_std = math.nan
    else:
        ring_lab_median = np.median(ring_lab, axis=0)
        l_vals = ring_lab[:, 0]
        l_med = float(np.median(l_vals))
        b_med = float(np.median(ring_lab[:, 2]))
        l_mean = float(np.mean(l_vals))
        l_std = float(np.std(l_vals))

    if lesion_lab.size == 0:
        lesion_lab_median = np.array([math.nan, math.nan, math.nan], dtype=float)
    else:
        lesion_lab_median = np.median(lesion_lab, axis=0)

    estimator = str(getattr(ita_cfg, "estimator", "aggregated_lab")).strip().lower()
    if estimator not in {"aggregated_lab", "pixelwise_median"}:
        estimator = "aggregated_lab"
    aggregation_stat = str(getattr(ita_cfg, "aggregation_stat", "median")).strip().lower()
    if aggregation_stat not in {"median", "mean", "trimmed_mean_sd"}:
        aggregation_stat = "median"
    trim_std = float(getattr(ita_cfg, "trim_std", 1.0))

    if estimator == "pixelwise_median":
        ita_val = _ita_from_pixelwise_lab(candidate_lab, eps=float(ita_cfg.eps))
    else:
        l_agg, b_agg = _aggregate_lab(candidate_lab, mode=aggregation_stat, trim_std=trim_std)
        ita_val = _ita_from_aggregated_lab(l_agg, b_agg, eps=float(ita_cfg.eps))

    return {
        "ita_deg": ita_val,
        "ring_pixel_count": ring_count,
        "ring_area_frac": ring_area_frac,
        "ring_valid": bool(ring_valid and candidate_count >= int(ita_cfg.ring_min_pixels)),
        "ita_candidate_count": candidate_count,
        "ita_region_strategy": region_strategy,
        "ita_estimator": estimator,
        "ita_aggregation_stat": aggregation_stat,
        "Lstar_skin_median": l_med,
        "bstar_skin_median": b_med,
        "ring_mask": ring_mask,
        "field_mask": field_mask,
        "lesion_mask": working_lesion,
        "lab_ring_median": ring_lab_median,
        "lab_lesion_median": lesion_lab_median,
        "Lstar_skin_mean": l_mean,
        "Lstar_skin_std": l_std,
        "working_rgb": working_rgb,
        "working_scale": float(scale),
        # Keep for backward compatibility: enhanced pipeline now avoids full-frame Lab allocations.
        "lab": None,
    }


def compute_ita_legacy_like(
    image_rgb: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    eps: float = 1e-6,
    use_field_mask: bool = False,
    field_intensity_floor: int = 5,
    min_pixels: int = 200,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> Tuple[float, int]:
    lesion = to_bool_mask(lesion_mask)
    if not np.any(lesion):
        return math.nan, 0

    region_mask = np.logical_not(lesion)
    if use_field_mask:
        field_mask = build_field_mask(image_rgb, floor=field_intensity_floor)
        region_mask = np.logical_and(region_mask, field_mask)
    if not np.any(region_mask):
        return math.nan, 0

    region_lab = _rgb_pixels_to_lab(image_rgb[region_mask])
    if region_lab.size == 0:
        return math.nan, 0

    l_vals = region_lab[:, 0]
    lo = float(np.percentile(l_vals, max(0.0, min(100.0, low_pct))))
    hi = float(np.percentile(l_vals, max(0.0, min(100.0, high_pct))))
    keep = np.logical_and(l_vals >= lo, l_vals <= hi)
    candidate = region_lab[keep]
    candidate_count = int(candidate.shape[0])
    if candidate_count < int(min_pixels):
        return math.nan, candidate_count

    ita_values = np.arctan((candidate[:, 0] - 50.0) / (candidate[:, 2] + float(eps))) * (180.0 / np.pi)
    if ita_values.size == 0:
        return math.nan, candidate_count
    return float(np.median(ita_values)), candidate_count


def ita_bin6(ita_value: float, ita_cfg: ITAConfig) -> str:
    if np.isnan(ita_value):
        return "Unknown"
    if ita_value > ita_cfg.very_light_threshold:
        return "Very Light"
    if ita_value > ita_cfg.light_threshold:
        return "Light"
    if ita_value > ita_cfg.intermediate_threshold:
        return "Intermediate"
    if ita_value > ita_cfg.tan_threshold:
        return "Tan"
    if ita_value > ita_cfg.brown_threshold:
        return "Brown"
    return "Dark"


def ita_binary(ita_value: float, *, cutoff: float) -> str:
    if np.isnan(ita_value):
        return "Unknown"
    if ita_value <= cutoff:
        return "Lower ITA"
    return "Higher ITA"
