from __future__ import annotations

import math
from typing import Dict

import cv2
import numpy as np
from skimage import color

from .config import CovariateConfig
from .ita import to_bool_mask


def _delta_e(
    lesion_lab: np.ndarray,
    skin_lab: np.ndarray,
    method: str,
) -> float:
    if np.isnan(lesion_lab).any() or np.isnan(skin_lab).any():
        return math.nan
    if method == "ciede2000":
        lhs = lesion_lab.reshape(1, 1, 3)
        rhs = skin_lab.reshape(1, 1, 3)
        return float(color.deltaE_ciede2000(lhs, rhs)[0, 0])
    return float(np.linalg.norm(lesion_lab - skin_lab))


def _build_directional_kernels(num_orientations: int) -> list[np.ndarray]:
    # Lightweight set of line kernels for hair-like ridge enhancement.
    if num_orientations <= 4:
        return [
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
        ]
    return [
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)),
    ]


def _hair_fraction(
    gray: np.ndarray,
    valid_mask: np.ndarray,
    *,
    quantile: float,
    mode: str,
) -> float:
    if mode == "off":
        return math.nan
    if not np.any(valid_mask):
        return math.nan
    kernels = _build_directional_kernels(4 if mode == "lite" else 8)
    responses = [cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel) for kernel in kernels]
    score = np.maximum.reduce(responses)
    valid_vals = score[valid_mask]
    if valid_vals.size == 0:
        return math.nan
    thresh = float(np.quantile(valid_vals, quantile))
    hair_mask = np.logical_and(score >= thresh, valid_mask)
    return float(hair_mask.sum() / max(1, valid_mask.sum()))


def _specular_fraction(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    l_cutoff: float,
    c_cutoff: float,
) -> float:
    if not np.any(valid_mask):
        return math.nan
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(float) * (100.0 / 255.0)
    val = hsv[:, :, 2].astype(float) * (100.0 / 255.0)
    # Approximate high-L*, low-chroma highlights via HSV.
    mask = np.logical_and.reduce((val >= l_cutoff, sat <= c_cutoff, valid_mask))
    return float(mask.sum() / max(1, valid_mask.sum()))


def _resize_texture_inputs(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_dim:
        return rgb, valid_mask
    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_valid = (
        cv2.resize(
            valid_mask.astype(np.uint8),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    return resized_rgb, resized_valid


def compute_covariates(
    *,
    image_rgb: np.ndarray,
    lesion_mask: np.ndarray,
    ring_mask: np.ndarray,
    field_mask: np.ndarray,
    lab_lesion_median: np.ndarray,
    lab_ring_median: np.ndarray,
    lstar_skin_mean: float,
    lstar_skin_std: float,
    cfg: CovariateConfig,
    deltae_method_override: str | None = None,
    hair_mode: str = "lite",
    include_specular: bool = True,
    texture_max_dim: int = 512,
) -> Dict[str, float | str]:
    lesion_bool = to_bool_mask(lesion_mask)
    total_pixels = float(lesion_bool.size) if lesion_bool.size else 1.0
    lesion_area_frac = float(lesion_bool.sum() / total_pixels)

    deltae_method = (deltae_method_override or cfg.deltae_method or "deltae76").strip().lower()
    deltae = _delta_e(
        np.asarray(lab_lesion_median, dtype=float),
        np.asarray(lab_ring_median, dtype=float),
        deltae_method,
    )

    if cfg.valid_pixels_base == "ring":
        valid_mask = ring_mask
    elif cfg.valid_pixels_base == "lesion_ring":
        valid_mask = np.logical_or(ring_mask, lesion_bool)
    else:
        valid_mask = field_mask

    texture_rgb, texture_valid = _resize_texture_inputs(
        image_rgb,
        valid_mask=valid_mask,
        max_dim=max(64, int(texture_max_dim)),
    )
    gray = cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2GRAY)

    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    sharpness = float(np.var(lap))
    hair_frac = _hair_fraction(
        gray,
        texture_valid,
        quantile=cfg.hair_threshold_quantile,
        mode=hair_mode,
    )
    specular_frac = (
        _specular_fraction(
            texture_rgb,
            texture_valid,
            l_cutoff=cfg.specular_lstar_cutoff,
            c_cutoff=cfg.specular_chroma_cutoff,
        )
        if include_specular
        else math.nan
    )

    return {
        "lesion_area_frac": lesion_area_frac,
        "deltaE_lesion_skin": deltae,
        "deltaE_method": deltae_method,
        "Lstar_skin_mean": float(lstar_skin_mean) if np.isfinite(lstar_skin_mean) else math.nan,
        "Lstar_skin_std": float(lstar_skin_std) if np.isfinite(lstar_skin_std) else math.nan,
        "sharpness_laplacian_var": sharpness,
        "hair_frac": hair_frac,
        "specular_frac": specular_frac,
    }
