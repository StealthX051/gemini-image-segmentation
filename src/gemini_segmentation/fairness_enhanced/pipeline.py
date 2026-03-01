from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .config import EnhancedFairnessConfig, config_to_dict
from .covariates import compute_covariates
from .dataset_index import build_or_load_source_index
from .dedup import apply_exact_dedup, apply_near_dedup, phash64_hex, sha256_file
from .effects import compute_endpoint_effects, threshold_sensitivity_table
from .ita import (
    compute_ita_features,
    compute_ita_legacy_like,
    ita_bin6,
    ita_binary,
    to_bool_mask,
)
from .labels import build_label_text
from .reporting import (
    ensure_output_dirs,
    write_analysis_frame,
    write_covariate_qc,
    write_dedup_outputs,
    write_endpoint_effects,
    write_ita_bins,
    write_json,
    write_threshold_sensitivity,
    write_trend_outputs,
)
from .trends import build_iou_trend_frames, build_success_trend_frames


FEATURES_MANIFEST_NAME = "features_manifest.json"
FEATURES_PART_PREFIX = "features_part_"
METRICS_CACHE_NAME = "metrics.parquet"
ITA_CACHE_NAME = "ita_features.parquet"
COVARIATES_CACHE_NAME = "covariates.parquet"
FINGERPRINTS_CACHE_NAME = "fingerprints.parquet"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        token = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return token
    except Exception:
        return None


def _ita_method_payload(cfg: EnhancedFairnessConfig, *, ita_cutoff: float) -> Dict[str, object]:
    region_strategy = str(cfg.ita.region_strategy).strip().lower()
    if region_strategy == "global_nonlesion":
        label = "image-derived non-lesional skin tone proxy (ITA)"
    elif region_strategy == "perilesional_ring":
        label = "image-derived perilesional skin tone proxy (ITA)"
    else:
        label = "image-derived skin tone proxy (ITA)"
    return {
        "label": label,
        "binary_strata": "lower-ITA (darker-appearing) vs higher-ITA (lighter-appearing)",
        "cutoff_degrees": float(ita_cutoff),
        "region_strategy": region_strategy,
        "estimator": str(cfg.ita.estimator),
        "aggregation_stat": str(cfg.ita.aggregation_stat),
        "trim_std": float(cfg.ita.trim_std),
        "apply_lstar_window": bool(cfg.ita.apply_lstar_window),
        "lstar_window_low_pct": float(cfg.ita.lstar_window_low_pct),
        "lstar_window_high_pct": float(cfg.ita.lstar_window_high_pct),
        "formula": "ITA = arctan((L* - 50) / (b* + eps)) * (180/pi)",
        "eps": float(cfg.ita.eps),
        "ring_outer_frac_min_dim": float(cfg.ita.ring_outer_frac_min_dim),
        "ring_inner_frac_min_dim": float(cfg.ita.ring_inner_frac_min_dim),
        "ring_min_pixels": int(cfg.ita.ring_min_pixels),
        "ring_min_area_frac": float(cfg.ita.ring_min_area_frac),
        "use_field_mask": bool(cfg.ita.use_field_mask),
        "field_intensity_floor": int(cfg.ita.field_intensity_floor),
        "legacy_like_sensitivity_enabled": bool(cfg.ita.include_legacy_like_sensitivity),
        "legacy_like_definition": (
            "Global non-lesion region with optional field-mask; ITA is median over pixelwise ITA "
            "after L* percentile windowing."
        ),
    }


def _write_ita_method_note(out_dir: Path, cfg: EnhancedFairnessConfig, *, ita_cutoff: float) -> None:
    payload = _ita_method_payload(cfg, ita_cutoff=ita_cutoff)
    write_json(out_dir / "ita_method_note.json", payload)
    lines = [
        "# ITA Method Note",
        "",
        "This run uses an " + str(payload["label"]) + ".",
        "",
        f"- Binary strata: lower-ITA vs higher-ITA using cutoff {float(ita_cutoff):.1f} degrees.",
        f"- Region strategy: {payload['region_strategy']}",
        f"- ITA estimator: {payload['estimator']}",
        f"- Aggregation statistic: {payload['aggregation_stat']}",
        f"- L* windowing: enabled={payload['apply_lstar_window']} ({payload['lstar_window_low_pct']}-{payload['lstar_window_high_pct']} percentile)",
        f"- Formula: {payload['formula']}",
        f"- Epsilon: {payload['eps']}",
        f"- Field mask: enabled={payload['use_field_mask']}, floor={payload['field_intensity_floor']}",
        f"- Minimum sample thresholds: min_pixels={payload['ring_min_pixels']}, min_area_frac={payload['ring_min_area_frac']}",
        "",
        "Legacy-like ITA sensitivity (global non-lesion, pixelwise-median ITA) is "
        + ("enabled." if bool(payload["legacy_like_sensitivity_enabled"]) else "disabled."),
    ]
    (out_dir / "ita_method_note.md").write_text("\n".join(lines), encoding="utf-8")


def _safe_open_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _safe_open_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img)


def _source_lookup_maps(source_index: pd.DataFrame) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    by_sha: Dict[str, Dict[str, object]] = {}
    by_id: Dict[str, Dict[str, object]] = {}

    if source_index.empty:
        return by_sha, by_id

    for sha, chunk in source_index.groupby("sha256", sort=False):
        first = chunk.iloc[0].to_dict()
        target_source = str(first.get("dataset_source_primary", "unknown"))
        preferred = chunk[chunk["dataset_source"] == target_source]
        row = preferred.iloc[0].to_dict() if not preferred.empty else first
        by_sha[str(sha)] = {
            "dataset_source_primary": str(row.get("dataset_source_primary", "unknown")),
            "dataset_source_memberships_json": str(
                row.get("dataset_source_memberships_json", "[]")
            ),
            "split": str(row.get("split", "unknown")),
            "mask_source": str(row.get("mask_source", "unknown")),
        }

    for image_id, chunk in source_index.groupby("image_id", sort=False):
        row = chunk.iloc[0].to_dict()
        by_id[str(image_id)] = {
            "dataset_source_primary": str(row.get("dataset_source_primary", "unknown")),
            "dataset_source_memberships_json": str(row.get("dataset_source_memberships_json", "[]")),
            "split": str(row.get("split", "unknown")),
            "mask_source": str(row.get("mask_source", "unknown")),
        }

    return by_sha, by_id


def _covariate_columns() -> List[str]:
    return [
        "lesion_area_frac",
        "deltaE_lesion_skin",
        "deltaE_method",
        "Lstar_skin_mean",
        "Lstar_skin_std",
        "sharpness_laplacian_var",
        "hair_frac",
        "specular_frac",
    ]


def _trend_covariate_columns(df: pd.DataFrame) -> List[str]:
    # Trend models require numeric covariates; exclude method labels and non-numeric columns.
    out: List[str] = []
    for col in _covariate_columns():
        if col == "deltaE_method" or col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if int(series.notna().sum()) <= 0:
            continue
        out.append(col)
    return out


def _cache_column_sets() -> Dict[str, List[str]]:
    return {
        "metrics": [
            "image_id",
            "image_name",
            "model_name",
            "run_id",
            "iou",
            "dice",
            "success_t050",
            "success_from_metrics",
        ],
        "ita_features": [
            "image_id",
            "image_name",
            "ita_deg",
            "ita_deg_legacy_like",
            "ita_delta_vs_legacy",
            "ita_bin6",
            "ita_binary",
            "ita_binary_legacy28",
            "ita_candidate_count",
            "ita_legacy_candidate_count",
            "ita_method_region",
            "ita_method_estimator",
            "ita_method_aggregation",
            "ring_pixel_count",
            "ring_area_frac",
            "ring_valid",
            "Lstar_skin_mean",
            "Lstar_skin_std",
        ],
        "covariates": [
            "image_id",
            "image_name",
            "lesion_area_frac",
            "deltaE_lesion_skin",
            "deltaE_method",
            "sharpness_laplacian_var",
            "hair_frac",
            "specular_frac",
        ],
        "fingerprints": [
            "image_id",
            "image_name",
            "image_path",
            "sha256",
            "phash64_hex",
        ],
    }


def _profile_name(cfg: EnhancedFairnessConfig) -> str:
    profile = str(cfg.features.profile or "balanced").strip().lower()
    if profile not in {"balanced", "full", "minimal"}:
        return "balanced"
    return profile


def _feature_plan(
    cfg: EnhancedFairnessConfig,
    *,
    stage: str,
    requested_columns: set[str] | None = None,
) -> Dict[str, object]:
    requested = requested_columns or set()
    profile = _profile_name(cfg)

    compute_ita = True
    compute_covariates = bool(cfg.covariates.enabled and profile in {"balanced", "full"})
    include_specular = bool(profile == "full" or cfg.features.include_specular_in_core)
    hair_mode = str(cfg.features.hair_mode or "lite").strip().lower()
    deltae_method = cfg.covariates.deltae_method if profile == "full" else "deltae76"
    compute_phash = bool(cfg.features.compute_phash_in_core or cfg.dedup.mode == "near")
    if stage == "all" and (cfg.sensitivity.include_near_dedup or cfg.dedup.mode == "near"):
        compute_phash = True

    if profile == "minimal":
        compute_covariates = False
        include_specular = False
        hair_mode = "off"

    if stage == "sensitivity":
        need_near = bool(cfg.sensitivity.include_near_dedup or cfg.dedup.mode == "near")
        compute_phash = need_near

    if stage == "augment":
        compute_ita = bool(
            requested.intersection(
                {
                    "ita_deg",
                    "ita_bin6",
                    "ita_binary",
                    "ring_pixel_count",
                    "ring_area_frac",
                    "ring_valid",
                    "ita_candidate_count",
                    "ita_method_region",
                    "ita_method_estimator",
                    "ita_method_aggregation",
                    "ita_deg_legacy_like",
                    "ita_binary_legacy28",
                    "ita_legacy_candidate_count",
                    "ita_delta_vs_legacy",
                    "deltaE_lesion_skin",
                    "Lstar_skin_mean",
                    "Lstar_skin_std",
                    "sharpness_laplacian_var",
                    "hair_frac",
                    "specular_frac",
                }
            )
        )
        compute_covariates = bool(
            requested.intersection(
                {
                    "lesion_area_frac",
                    "deltaE_lesion_skin",
                    "deltaE_method",
                    "Lstar_skin_mean",
                    "Lstar_skin_std",
                    "sharpness_laplacian_var",
                    "hair_frac",
                    "specular_frac",
                }
            )
        )
        compute_phash = "phash64_hex" in requested
        include_specular = "specular_frac" in requested or include_specular
        if "hair_frac" not in requested:
            hair_mode = "off"
        if "deltaE_lesion_skin" in requested and profile != "full":
            deltae_method = "deltae76"

    return {
        "profile": profile,
        "compute_sha": True,
        "compute_phash": compute_phash,
        "compute_ita": compute_ita,
        "compute_covariates": compute_covariates,
        "include_specular": include_specular,
        "hair_mode": hair_mode,
        "deltae_method": deltae_method,
    }


def _requested_feature_columns(
    cfg: EnhancedFairnessConfig,
    *,
    stage: str,
    user_requested: set[str] | None = None,
) -> set[str]:
    requested = set(user_requested or set())
    profile = _profile_name(cfg)
    if stage in {"all", "core"}:
        requested.update(
            {
                "iou",
                "dice",
                "success_t050",
                "ita_deg",
                "ita_bin6",
                "ita_binary",
                "ita_candidate_count",
                "ita_method_region",
                "ita_method_estimator",
                "ita_method_aggregation",
                "ring_pixel_count",
                "ring_area_frac",
                "ring_valid",
                "lesion_area_frac",
                "sha256",
            }
        )
        if cfg.ita.include_legacy_like_sensitivity:
            requested.update(
                {
                    "ita_deg_legacy_like",
                    "ita_binary_legacy28",
                    "ita_legacy_candidate_count",
                    "ita_delta_vs_legacy",
                }
            )
        if profile in {"balanced", "full"} and cfg.covariates.enabled:
            requested.update(
                {
                    "deltaE_lesion_skin",
                    "deltaE_method",
                    "Lstar_skin_mean",
                    "Lstar_skin_std",
                    "sharpness_laplacian_var",
                    "hair_frac",
                }
            )
            if profile == "full" or cfg.features.include_specular_in_core:
                requested.add("specular_frac")
        if (
            cfg.features.compute_phash_in_core
            or cfg.dedup.mode == "near"
            or (stage == "all" and cfg.sensitivity.include_near_dedup)
        ):
            requested.add("phash64_hex")
    if stage in {"sensitivity"} and (cfg.sensitivity.include_near_dedup or cfg.dedup.mode == "near"):
        requested.add("phash64_hex")
    return requested


def _available_memory_bytes() -> int | None:
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
            return None
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _resolve_workers(
    cfg: EnhancedFairnessConfig,
    *,
    requested_workers: int,
) -> Tuple[int, Dict[str, object]]:
    requested = max(1, int(requested_workers))
    profile = _profile_name(cfg)
    summary: Dict[str, object] = {
        "requested_workers": requested,
        "workers_auto": bool(cfg.runtime.workers_auto),
        "profile": profile,
    }
    if not cfg.runtime.workers_auto:
        summary["effective_workers"] = requested
        return requested, summary

    available = _available_memory_bytes()
    if available is None:
        summary["effective_workers"] = requested
        summary["workers_auto_reason"] = "mem_unavailable"
        return requested, summary

    if profile == "full":
        estimate_mb = int(cfg.runtime.per_worker_estimate_mb_full)
    elif profile == "minimal":
        estimate_mb = int(cfg.runtime.per_worker_estimate_mb_minimal)
    else:
        estimate_mb = int(cfg.runtime.per_worker_estimate_mb_balanced)

    target_frac = max(0.10, min(0.95, float(cfg.runtime.memory_target_frac)))
    target_bytes = int(float(available) * target_frac)
    per_worker_bytes = max(128 * 1024 * 1024, int(estimate_mb) * 1024 * 1024)
    cap = max(1, int(target_bytes // per_worker_bytes))
    effective = max(1, min(requested, cap))

    summary.update(
        {
            "available_memory_bytes": int(available),
            "memory_target_frac": target_frac,
            "per_worker_estimate_mb": int(estimate_mb),
            "workers_memory_cap": int(cap),
            "effective_workers": int(effective),
        }
    )
    return int(effective), summary


def _duplicate_examples(
    *,
    base_df: pd.DataFrame,
    dedup_map_exact: pd.DataFrame,
    out_dir: Path,
    limit: int,
) -> None:
    if dedup_map_exact.empty:
        return
    by_name = {
        str(row.image_name): Path(str(row.image_path))
        for _, row in base_df.iterrows()
        if str(row.image_name)
    }
    pairs = dedup_map_exact[dedup_map_exact["n_members"] > 1].head(max(0, int(limit)))
    manifest_rows: List[Dict[str, str]] = []
    for _, row in pairs.iterrows():
        members = str(row.get("all_members", "")).split("|")
        if len(members) < 2:
            continue
        left, right = members[0], members[1]
        left_src = by_name.get(left)
        right_src = by_name.get(right)
        if not left_src or not right_src:
            continue
        left_dst = out_dir / f"{left_src.stem}__dupA{left_src.suffix.lower()}"
        right_dst = out_dir / f"{right_src.stem}__dupB{right_src.suffix.lower()}"
        try:
            shutil.copy2(left_src, left_dst)
            shutil.copy2(right_src, right_dst)
            manifest_rows.append(
                {
                    "sha256": str(row.get("sha256", "")),
                    "left": left,
                    "right": right,
                    "left_copy": left_dst.name,
                    "right_copy": right_dst.name,
                }
            )
        except Exception as exc:
            logging.warning("Failed to write duplicate example pair (%s, %s): %s", left, right, exc)

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(out_dir / "duplicate_examples_manifest.csv", index=False)


def _near_canonical_rows(
    near_df: pd.DataFrame | None,
    dedup_map_near: pd.DataFrame | None,
) -> pd.DataFrame:
    if near_df is None or dedup_map_near is None or dedup_map_near.empty:
        return pd.DataFrame()
    canonical_map = dedup_map_near[["cluster_id", "canonical_image_name"]].rename(
        columns={"canonical_image_name": "near_canonical_image_name"}
    )
    merged = near_df.merge(
        canonical_map,
        left_on="near_dedup_group_id",
        right_on="cluster_id",
        how="left",
    )
    if "near_canonical_image_name" not in merged.columns:
        return pd.DataFrame()
    return merged[merged["image_name"] == merged["near_canonical_image_name"]].copy()


def _default_covariates(cfg: EnhancedFairnessConfig) -> Dict[str, float | str]:
    return {
        "lesion_area_frac": math.nan,
        "deltaE_lesion_skin": math.nan,
        "deltaE_method": cfg.covariates.deltae_method,
        "Lstar_skin_mean": math.nan,
        "Lstar_skin_std": math.nan,
        "sharpness_laplacian_var": math.nan,
        "hair_frac": math.nan,
        "specular_frac": math.nan,
    }


def _build_analysis_row(
    *,
    idx: int,
    img_path: Path,
    gt_path: Path,
    per_image_metrics: Dict[str, Tuple[float, float, bool]],
    by_sha: Dict[str, Dict[str, object]],
    by_id: Dict[str, Dict[str, object]],
    cfg: EnhancedFairnessConfig,
    model_name: str,
    prompt_variant: str,
    run_id: str,
    feature_plan: Dict[str, object],
) -> Tuple[int, Dict[str, object] | None, Dict[str, float]]:
    img_path = Path(img_path)
    gt_path = Path(gt_path)
    image_name = img_path.name
    image_id = img_path.stem
    t0 = time.perf_counter()
    timings: Dict[str, float] = {
        "read_seconds": 0.0,
        "sha_seconds": 0.0,
        "phash_seconds": 0.0,
        "ita_seconds": 0.0,
        "covariates_seconds": 0.0,
        "total_seconds": 0.0,
    }

    try:
        read_t0 = time.perf_counter()
        image_rgb = _safe_open_rgb(img_path)
        gt_mask = _safe_open_mask(gt_path)
        timings["read_seconds"] = float(time.perf_counter() - read_t0)
    except Exception as exc:
        logging.warning("Skipping %s due to read failure: %s", image_name, exc)
        timings["total_seconds"] = float(time.perf_counter() - t0)
        return idx, None, timings

    sha_t0 = time.perf_counter()
    sha = sha256_file(img_path) if bool(feature_plan.get("compute_sha", True)) else ""
    timings["sha_seconds"] = float(time.perf_counter() - sha_t0)

    if bool(feature_plan.get("compute_phash", False)):
        phash_t0 = time.perf_counter()
        phash = phash64_hex(np.mean(image_rgb, axis=-1))
        timings["phash_seconds"] = float(time.perf_counter() - phash_t0)
    else:
        phash = math.nan

    source = by_sha.get(sha)
    if source is None and cfg.allow_image_id_fallback:
        source = by_id.get(image_id)
    source = source or {
        "dataset_source_primary": "unknown",
        "dataset_source_memberships_json": "[]",
        "split": "unknown",
        "mask_source": "unknown",
    }

    lesion_bool = to_bool_mask(gt_mask)
    lesion_area_frac = float(lesion_bool.sum() / float(lesion_bool.size)) if lesion_bool.size else math.nan

    ita_features: Dict[str, object] | None = None
    if bool(feature_plan.get("compute_ita", True)) or bool(feature_plan.get("compute_covariates", False)):
        ita_t0 = time.perf_counter()
        ita_features = compute_ita_features(
            image_rgb,
            gt_mask,
            ita_cfg=cfg.ita,
            roi_max_dim=int(cfg.features.roi_max_dim),
            ring_outer_radius_cap=int(cfg.features.ring_outer_radius_cap),
            ring_inner_radius_cap=int(cfg.features.ring_inner_radius_cap),
            ring_roi_pad_px=int(cfg.features.ring_roi_pad_px),
        )
        timings["ita_seconds"] = float(time.perf_counter() - ita_t0)

    if ita_features is None:
        ita_features = {
            "ita_deg": math.nan,
            "ring_pixel_count": 0,
            "ring_area_frac": 0.0,
            "ring_valid": False,
            "ita_candidate_count": 0,
            "ita_region_strategy": "unknown",
            "ita_estimator": "unknown",
            "ita_aggregation_stat": "unknown",
            "ring_mask": np.zeros(gt_mask.shape[:2], dtype=bool),
            "field_mask": np.ones(gt_mask.shape[:2], dtype=bool),
            "lesion_mask": lesion_bool,
            "lab_ring_median": np.array([math.nan, math.nan, math.nan], dtype=float),
            "lab_lesion_median": np.array([math.nan, math.nan, math.nan], dtype=float),
            "Lstar_skin_mean": math.nan,
            "Lstar_skin_std": math.nan,
            "working_rgb": image_rgb,
        }
    ita_val = float(ita_features["ita_deg"]) if ita_features["ita_deg"] is not None else math.nan
    ita_legacy = math.nan
    ita_legacy_count = 0
    if bool(cfg.ita.include_legacy_like_sensitivity):
        ita_legacy, ita_legacy_count = compute_ita_legacy_like(
            image_rgb=image_rgb,
            lesion_mask=gt_mask,
            eps=float(cfg.ita.eps),
            use_field_mask=bool(cfg.ita.use_field_mask),
            field_intensity_floor=int(cfg.ita.field_intensity_floor),
            min_pixels=int(cfg.ita.ring_min_pixels),
            low_pct=float(cfg.ita.lstar_window_low_pct),
            high_pct=float(cfg.ita.lstar_window_high_pct),
        )
    ita_delta_vs_legacy = float(ita_val - ita_legacy) if np.isfinite(ita_val) and np.isfinite(ita_legacy) else math.nan

    iou, dice, success_existing = per_image_metrics.get(image_name, (math.nan, math.nan, False))
    success_t050 = bool(float(iou) >= 0.50) if not math.isnan(float(iou)) else False

    covariates = _default_covariates(cfg)
    covariates["lesion_area_frac"] = lesion_area_frac
    if bool(feature_plan.get("compute_covariates", False)):
        cov_t0 = time.perf_counter()
        covariates = compute_covariates(
            image_rgb=np.asarray(ita_features.get("working_rgb"), dtype=np.uint8),
            lesion_mask=np.asarray(ita_features.get("lesion_mask"), dtype=bool),
            ring_mask=np.asarray(ita_features["ring_mask"], dtype=bool),
            field_mask=np.asarray(ita_features["field_mask"], dtype=bool),
            lab_lesion_median=np.asarray(ita_features["lab_lesion_median"], dtype=float),
            lab_ring_median=np.asarray(ita_features["lab_ring_median"], dtype=float),
            lstar_skin_mean=float(ita_features.get("Lstar_skin_mean", math.nan)),
            lstar_skin_std=float(ita_features.get("Lstar_skin_std", math.nan)),
            cfg=cfg.covariates,
            deltae_method_override=str(feature_plan.get("deltae_method", cfg.covariates.deltae_method)),
            hair_mode=str(feature_plan.get("hair_mode", "lite")),
            include_specular=bool(feature_plan.get("include_specular", False)),
            texture_max_dim=int(cfg.features.hair_max_dim),
        )
        timings["covariates_seconds"] = float(time.perf_counter() - cov_t0)
        # Force lesion area fraction to remain the full-resolution GT quantity.
        covariates["lesion_area_frac"] = lesion_area_frac
    else:
        covariates["deltaE_method"] = str(feature_plan.get("deltae_method", cfg.covariates.deltae_method))
        covariates["Lstar_skin_mean"] = float(ita_features.get("Lstar_skin_mean", math.nan))
        covariates["Lstar_skin_std"] = float(ita_features.get("Lstar_skin_std", math.nan))

    row = {
        "extraction_index": int(idx),
        "image_id": image_id,
        "image_name": image_name,
        "image_path": str(img_path.resolve()),
        "mask_path": str(gt_path.resolve()),
        "dataset_source_primary": str(source["dataset_source_primary"]),
        "dataset_source_memberships_json": str(source["dataset_source_memberships_json"]),
        "split": str(source["split"]),
        "mask_source": str(source["mask_source"]),
        "sha256": sha,
        "phash64_hex": phash,
        "iou": float(iou),
        "dice": float(dice),
        "success_t050": bool(success_t050),
        "success_from_metrics": bool(success_existing),
        "ita_deg": ita_val,
        "ita_deg_legacy_like": float(ita_legacy) if np.isfinite(ita_legacy) else math.nan,
        "ita_delta_vs_legacy": ita_delta_vs_legacy,
        "ita_bin6": ita_bin6(ita_val, cfg.ita),
        "ita_binary": "Unknown",  # set after cutoff resolution
        "ita_binary_legacy28": ita_binary(float(ita_legacy), cutoff=28.0),
        "ita_candidate_count": int(ita_features.get("ita_candidate_count", 0) or 0),
        "ita_legacy_candidate_count": int(ita_legacy_count),
        "ita_method_region": str(ita_features.get("ita_region_strategy", "unknown")),
        "ita_method_estimator": str(ita_features.get("ita_estimator", "unknown")),
        "ita_method_aggregation": str(ita_features.get("ita_aggregation_stat", "unknown")),
        "ring_pixel_count": int(ita_features["ring_pixel_count"]),
        "ring_area_frac": float(ita_features["ring_area_frac"]),
        "ring_valid": bool(ita_features["ring_valid"]),
        "image_height": int(image_rgb.shape[0]),
        "image_width": int(image_rgb.shape[1]),
        "model_name": str(model_name),
        "prompt_variant": str(prompt_variant),
        "run_id": str(run_id),
        "audit_mode": "enhanced",
        **covariates,
    }
    timings["total_seconds"] = float(time.perf_counter() - t0)
    return idx, row, timings


def _current_rss_bytes() -> int | None:
    try:
        if os.name == "nt":
            import ctypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_uint32),
                    ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return int(counters.WorkingSetSize)
            return None

        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss * 1024)
    except Exception:
        return None


def _process_sample() -> Dict[str, float | int | None]:
    return {
        "cpu_seconds": float(time.process_time()),
        "rss_bytes": _current_rss_bytes(),
    }


def _new_runtime_profile(stage: str) -> Dict[str, object]:
    return {
        "status": "running",
        "stage": stage,
        "started_at": _now_iso(),
        "finished_at": None,
        "pid": int(os.getpid()),
        "stages": {},
    }


def _profile_stage_start(
    profile: Dict[str, object],
    *,
    stage_name: str,
    total_items: int | None = None,
) -> None:
    stages = profile.setdefault("stages", {})
    sample = _process_sample()
    stages[stage_name] = {
        "started_at": _now_iso(),
        "finished_at": None,
        "wall_seconds": None,
        "cpu_seconds": None,
        "rss_peak_bytes": sample.get("rss_bytes"),
        "items_total": int(total_items) if total_items is not None else None,
        "items_completed": 0,
        "throughput_items_per_sec": None,
        "queue_depth": 0,
        "_wall_start": float(time.perf_counter()),
        "_cpu_start": float(sample.get("cpu_seconds") or 0.0),
    }
    logging.info("[enhanced] Stage start: %s", stage_name)


def _profile_stage_sample(
    profile: Dict[str, object],
    *,
    stage_name: str,
    completed_items: int | None = None,
    queue_depth: int | None = None,
) -> None:
    stage = (profile.get("stages") or {}).get(stage_name)
    if not isinstance(stage, dict):
        return
    sample = _process_sample()
    rss = sample.get("rss_bytes")
    if rss is not None:
        existing = stage.get("rss_peak_bytes")
        if existing is None:
            stage["rss_peak_bytes"] = int(rss)
        else:
            stage["rss_peak_bytes"] = int(max(int(existing), int(rss)))
    if completed_items is not None:
        stage["items_completed"] = int(completed_items)
        wall_start = float(stage.get("_wall_start") or 0.0)
        if wall_start > 0.0:
            elapsed = max(1e-9, float(time.perf_counter()) - wall_start)
            stage["throughput_items_per_sec"] = float(completed_items) / elapsed
    if queue_depth is not None:
        stage["queue_depth"] = int(queue_depth)


def _profile_stage_end(
    profile: Dict[str, object],
    *,
    stage_name: str,
    completed_items: int | None = None,
) -> None:
    stage = (profile.get("stages") or {}).get(stage_name)
    if not isinstance(stage, dict):
        return

    _profile_stage_sample(profile, stage_name=stage_name, completed_items=completed_items)

    wall_end = float(time.perf_counter())
    cpu_end = float(time.process_time())
    wall_start = float(stage.get("_wall_start") or wall_end)
    cpu_start = float(stage.get("_cpu_start") or cpu_end)
    wall = max(0.0, wall_end - wall_start)
    cpu = max(0.0, cpu_end - cpu_start)

    stage["finished_at"] = _now_iso()
    stage["wall_seconds"] = wall
    stage["cpu_seconds"] = cpu
    if completed_items is not None:
        stage["items_completed"] = int(completed_items)

    items_completed = stage.get("items_completed")
    if isinstance(items_completed, int) and wall > 0:
        stage["throughput_items_per_sec"] = float(items_completed / wall)

    stage.pop("_wall_start", None)
    stage.pop("_cpu_start", None)

    logging.info("[enhanced] Stage end: %s (wall=%.2fs, cpu=%.2fs)", stage_name, wall, cpu)


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _load_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _default_features_manifest() -> Dict[str, object]:
    return {
        "version": 1,
        "updated_at": _now_iso(),
        "next_part": 1,
        "shards": [],
        "processed_image_names": [],
        "processed_count": 0,
        "restarts": 0,
    }


def _features_manifest_path(cache_dir: Path) -> Path:
    return cache_dir / FEATURES_MANIFEST_NAME


def _clear_feature_cache(cache_dir: Path) -> None:
    for shard in sorted(cache_dir.glob(f"{FEATURES_PART_PREFIX}*.parquet")):
        try:
            shard.unlink()
        except FileNotFoundError:
            continue
    manifest = _features_manifest_path(cache_dir)
    if manifest.exists():
        try:
            manifest.unlink()
        except FileNotFoundError:
            pass


def _load_features_manifest(cache_dir: Path) -> Dict[str, object]:
    manifest_path = _features_manifest_path(cache_dir)
    if not manifest_path.exists():
        return _default_features_manifest()
    payload = _load_json(manifest_path)
    if not payload:
        return _default_features_manifest()
    base = _default_features_manifest()
    base.update(payload)
    if not isinstance(base.get("shards"), list):
        base["shards"] = []
    if not isinstance(base.get("processed_image_names"), list):
        base["processed_image_names"] = []
    base["next_part"] = int(base.get("next_part", 1) or 1)
    base["processed_count"] = int(base.get("processed_count", 0) or 0)
    base["restarts"] = int(base.get("restarts", 0) or 0)
    return base


def _save_features_manifest(cache_dir: Path, manifest: Dict[str, object]) -> None:
    manifest["updated_at"] = _now_iso()
    _atomic_write_json(_features_manifest_path(cache_dir), manifest)


def _append_feature_shard(
    *,
    cache_dir: Path,
    manifest: Dict[str, object],
    rows: Sequence[Dict[str, object]],
    processed_names: set[str],
) -> int:
    if not rows:
        return 0

    part_index = int(manifest.get("next_part", 1) or 1)
    part_name = f"{FEATURES_PART_PREFIX}{part_index:05d}.parquet"
    part_path = cache_dir / part_name

    shard_df = pd.DataFrame(list(rows))
    if "extraction_index" in shard_df.columns:
        shard_df = shard_df.sort_values("extraction_index", kind="stable")
    shard_df.to_parquet(part_path, index=False)

    image_names = shard_df.get("image_name", pd.Series(dtype=str)).astype(str).tolist()
    processed_names.update(image_names)

    shards = list(manifest.get("shards", []))
    shards.append(
        {
            "file": part_name,
            "rows": int(len(shard_df)),
            "min_index": int(shard_df["extraction_index"].min()) if "extraction_index" in shard_df.columns else None,
            "max_index": int(shard_df["extraction_index"].max()) if "extraction_index" in shard_df.columns else None,
        }
    )
    manifest["shards"] = shards
    manifest["next_part"] = int(part_index + 1)
    manifest["processed_image_names"] = sorted(processed_names)
    manifest["processed_count"] = int(len(processed_names))

    _save_features_manifest(cache_dir, manifest)
    return int(len(shard_df))


def _load_feature_rows(cache_dir: Path, manifest: Dict[str, object]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for shard in manifest.get("shards", []):
        if not isinstance(shard, dict):
            continue
        fname = str(shard.get("file", "")).strip()
        if not fname:
            continue
        shard_path = cache_dir / fname
        if not shard_path.exists():
            logging.warning("Missing feature shard listed in manifest: %s", shard_path)
            continue
        frames.append(pd.read_parquet(shard_path))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "extraction_index" in df.columns:
        df = df.sort_values("extraction_index", kind="stable")
    if "image_name" in df.columns:
        df = df.drop_duplicates(subset=["image_name"], keep="last")
    return df.reset_index(drop=True)


def _ensure_analysis_schema_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required_defaults: Dict[str, object] = {
        "image_id": "",
        "image_name": "",
        "image_path": "",
        "mask_path": "",
        "dataset_source_primary": "unknown",
        "dataset_source_memberships_json": "[]",
        "split": "unknown",
        "mask_source": "unknown",
        "sha256": "",
        "phash64_hex": math.nan,
        "iou": math.nan,
        "dice": math.nan,
        "success_t050": False,
        "success_from_metrics": False,
        "ita_deg": math.nan,
        "ita_deg_legacy_like": math.nan,
        "ita_delta_vs_legacy": math.nan,
        "ita_bin6": "Unknown",
        "ita_binary": "Unknown",
        "ita_binary_legacy28": "Unknown",
        "ita_candidate_count": 0,
        "ita_legacy_candidate_count": 0,
        "ita_method_region": "unknown",
        "ita_method_estimator": "unknown",
        "ita_method_aggregation": "unknown",
        "lesion_area_frac": math.nan,
        "deltaE_lesion_skin": math.nan,
        "deltaE_method": "deltae76",
        "Lstar_skin_mean": math.nan,
        "Lstar_skin_std": math.nan,
        "sharpness_laplacian_var": math.nan,
        "hair_frac": math.nan,
        "specular_frac": math.nan,
        "model_name": "unknown",
        "prompt_variant": "none",
        "run_id": "unknown",
        "audit_mode": "enhanced",
        "image_height": 0,
        "image_width": 0,
        "ring_pixel_count": 0,
        "ring_area_frac": 0.0,
        "ring_valid": False,
    }
    for column, default in required_defaults.items():
        if column not in out.columns:
            out[column] = default
    return out


def _write_consolidated_caches(cache_dir: Path, base_df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for cache_name, columns in _cache_column_sets().items():
        present = [c for c in columns if c in base_df.columns]
        if not present:
            continue
        payload = base_df[present].copy()
        if cache_name == "metrics":
            payload.to_parquet(cache_dir / METRICS_CACHE_NAME, index=False)
        elif cache_name == "ita_features":
            payload.to_parquet(cache_dir / ITA_CACHE_NAME, index=False)
        elif cache_name == "covariates":
            payload.to_parquet(cache_dir / COVARIATES_CACHE_NAME, index=False)
        elif cache_name == "fingerprints":
            payload.to_parquet(cache_dir / FINGERPRINTS_CACHE_NAME, index=False)


def _append_to_consolidated_caches(cache_dir: Path, new_rows: pd.DataFrame) -> None:
    if new_rows.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_df = _ensure_analysis_schema_columns(new_rows)
    key_col = "image_name" if "image_name" in base_df.columns else "image_id"

    for cache_name, columns in _cache_column_sets().items():
        present = [c for c in columns if c in base_df.columns]
        if not present:
            continue
        payload = base_df[present].copy()
        if cache_name == "metrics":
            path = cache_dir / METRICS_CACHE_NAME
        elif cache_name == "ita_features":
            path = cache_dir / ITA_CACHE_NAME
        elif cache_name == "covariates":
            path = cache_dir / COVARIATES_CACHE_NAME
        else:
            path = cache_dir / FINGERPRINTS_CACHE_NAME

        if path.exists():
            try:
                existing = pd.read_parquet(path)
                payload = pd.concat([existing, payload], ignore_index=True)
            except Exception:
                pass
        if key_col in payload.columns:
            payload = payload.drop_duplicates(subset=[key_col], keep="last")
        payload.to_parquet(path, index=False)


def _effective_stage_config(cfg: EnhancedFairnessConfig) -> Tuple[EnhancedFairnessConfig, Dict[str, object]]:
    out = deepcopy(cfg)
    stage = str(out.runtime.stage or "all").strip().lower()
    if stage not in {"all", "core", "sensitivity", "augment"}:
        raise ValueError(f"Unsupported enhanced runtime stage: {stage}")
    out.runtime.stage = stage

    overrides: Dict[str, object] = {}
    if stage == "core":
        out.dedup.mode = "exact"
        out.dedup.include_near_map = False
        out.sensitivity.include_near_dedup = False
        out.sensitivity.include_dependence = False
        out.sensitivity.include_mask_source = False
        out.bootstrap.method = "percentile"
        out.bootstrap.n_resamples = 1000
        out.trends.bootstrap_resamples = 50
        overrides = {
            "dedup.mode": "exact",
            "dedup.include_near_map": False,
            "sensitivity.include_near_dedup": False,
            "sensitivity.include_dependence": False,
            "sensitivity.include_mask_source": False,
            "bootstrap.method": "percentile",
            "bootstrap.n_resamples": 1000,
            "trends.bootstrap_resamples": 50,
        }
    if stage == "augment":
        out.runtime.resume = True
        overrides["runtime.resume"] = True

    return out, overrides


def _extract_features_with_checkpoints(
    *,
    pair_list: List[Tuple[Path, Path]],
    per_image_metrics: Dict[str, Tuple[float, float, bool]],
    by_sha: Dict[str, Dict[str, object]],
    by_id: Dict[str, Dict[str, object]],
    cfg: EnhancedFairnessConfig,
    model_name: str,
    prompt_variant: str,
    run_id: str,
    workers: int,
    cache_dir: Path,
    runtime_profile: Dict[str, object],
    feature_plan: Dict[str, object],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    checkpoint_every = max(1, int(cfg.runtime.checkpoint_every))
    max_inflight = int(cfg.runtime.max_inflight_tasks)
    if max_inflight <= 0:
        max_inflight = max(1, int(workers))
    heartbeat_every = max(1, min(200, checkpoint_every))

    if not cfg.runtime.resume:
        _clear_feature_cache(cache_dir)

    manifest_path = _features_manifest_path(cache_dir)
    manifest_exists = manifest_path.exists()
    manifest = _load_features_manifest(cache_dir)
    if cfg.runtime.resume and manifest_exists:
        manifest["restarts"] = int(manifest.get("restarts", 0) or 0) + 1
    else:
        manifest["restarts"] = int(manifest.get("restarts", 0) or 0)
    _save_features_manifest(cache_dir, manifest)

    processed_names = set(str(v) for v in manifest.get("processed_image_names", []))

    indexed_pairs: List[Tuple[int, Path, Path]] = []
    skipped_from_resume = 0
    for idx, (img_path, gt_path) in enumerate(pair_list):
        image_name = Path(img_path).name
        if image_name in processed_names:
            skipped_from_resume += 1
            continue
        indexed_pairs.append((idx, Path(img_path), Path(gt_path)))

    stage_name = "feature_extraction"
    _profile_stage_start(
        runtime_profile,
        stage_name=stage_name,
        total_items=len(indexed_pairs),
    )

    completed_this_run = 0
    checkpoints_written = 0
    rows_buffer: List[Dict[str, object]] = []
    feature_seconds: Dict[str, float] = defaultdict(float)

    def flush_rows() -> None:
        nonlocal checkpoints_written
        if not rows_buffer:
            return
        row_df = pd.DataFrame(list(rows_buffer))
        if "extraction_index" in row_df.columns:
            row_df = row_df.sort_values("extraction_index", kind="stable")
        _append_feature_shard(
            cache_dir=cache_dir,
            manifest=manifest,
            rows=list(rows_buffer),
            processed_names=processed_names,
        )
        _append_to_consolidated_caches(cache_dir, row_df)
        rows_buffer.clear()
        checkpoints_written += 1

    logging.info(
        "[enhanced] Extraction pending=%s skipped_from_resume=%s workers=%s max_inflight=%s checkpoint_every=%s",
        len(indexed_pairs),
        skipped_from_resume,
        workers,
        max_inflight,
        checkpoint_every,
    )

    def handle_result(row: Dict[str, object] | None, timings: Dict[str, float], in_flight_count: int) -> None:
        nonlocal completed_this_run
        completed_this_run += 1
        for key, value in timings.items():
            feature_seconds[str(key)] += float(value)
        if row is not None:
            rows_buffer.append(row)
        if len(rows_buffer) >= checkpoint_every:
            flush_rows()
        if completed_this_run == len(indexed_pairs) or (completed_this_run % heartbeat_every == 0):
            elapsed = max(1e-9, float(feature_seconds.get("total_seconds", 0.0)))
            ips = float(completed_this_run) / max(
                1e-9,
                float(
                    time.perf_counter()
                    - float(
                        ((runtime_profile.get("stages") or {}).get(stage_name) or {}).get(
                            "_wall_start",
                            time.perf_counter(),
                        )
                    )
                ),
            )
            _profile_stage_sample(
                runtime_profile,
                stage_name=stage_name,
                completed_items=completed_this_run,
                queue_depth=int(in_flight_count),
            )
            stage_payload = (runtime_profile.get("stages") or {}).get(stage_name) or {}
            rss_peak = stage_payload.get("rss_peak_bytes")
            rss_peak_gb = (
                float(rss_peak) / (1024.0**3)
                if isinstance(rss_peak, (int, float)) and float(rss_peak) > 0.0
                else math.nan
            )
            logging.info(
                "[enhanced] Extraction heartbeat: %s/%s pending completed (%.3f img/s, queue=%s, cpu_total=%.1fs, rss_peak_gb=%.2f)",
                completed_this_run,
                len(indexed_pairs),
                ips,
                int(in_flight_count),
                elapsed,
                rss_peak_gb,
            )

    if indexed_pairs:
        if workers <= 1:
            for idx, img_path, gt_path in indexed_pairs:
                _, row, perf = _build_analysis_row(
                    idx=idx,
                    img_path=img_path,
                    gt_path=gt_path,
                    per_image_metrics=per_image_metrics,
                    by_sha=by_sha,
                    by_id=by_id,
                    cfg=cfg,
                    model_name=model_name,
                    prompt_variant=prompt_variant,
                    run_id=run_id,
                    feature_plan=feature_plan,
                )
                handle_result(row, perf, 0)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                iterator = iter(indexed_pairs)
                in_flight: Dict[object, int] = {}

                while True:
                    while len(in_flight) < max_inflight:
                        try:
                            idx, img_path, gt_path = next(iterator)
                        except StopIteration:
                            break
                        future = executor.submit(
                            _build_analysis_row,
                            idx=idx,
                            img_path=img_path,
                            gt_path=gt_path,
                            per_image_metrics=per_image_metrics,
                            by_sha=by_sha,
                            by_id=by_id,
                            cfg=cfg,
                            model_name=model_name,
                            prompt_variant=prompt_variant,
                            run_id=run_id,
                            feature_plan=feature_plan,
                        )
                        in_flight[future] = idx

                    if not in_flight:
                        break

                    queue_depth = int(len(in_flight))
                    done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        idx = in_flight.pop(future)
                        try:
                            _, row, perf = future.result()
                        except Exception as exc:
                            logging.warning(
                                "Failed to process pair index %s in enhanced fairness worker: %s",
                                idx,
                                exc,
                            )
                            row = None
                            perf = {}
                        handle_result(row, perf, len(in_flight))

    flush_rows()
    _profile_stage_end(
        runtime_profile,
        stage_name=stage_name,
        completed_items=completed_this_run,
    )
    stage_payload = (runtime_profile.get("stages") or {}).get(stage_name)
    if isinstance(stage_payload, dict):
        stage_payload["feature_seconds"] = {k: float(v) for k, v in feature_seconds.items()}
        stage_payload["resume_cache_hits"] = int(skipped_from_resume)
        stage_payload["resume_cache_misses"] = int(len(indexed_pairs))

    base_df = _load_feature_rows(cache_dir, manifest)
    if base_df.empty:
        raise ValueError("Enhanced fairness extraction produced no rows from checkpoint shards.")
    if "extraction_index" in base_df.columns:
        base_df = base_df.sort_values("extraction_index", kind="stable")
        base_df = base_df.drop(columns=["extraction_index"]) 

    extraction_info = {
        "manifest_path": str(manifest_path),
        "checkpoint_every": checkpoint_every,
        "max_inflight_tasks": max_inflight,
        "resume_enabled": bool(cfg.runtime.resume),
        "manifest_restarts": int(manifest.get("restarts", 0) or 0),
        "processed_count_total": int(manifest.get("processed_count", 0) or 0),
        "processed_this_run": int(completed_this_run),
        "skipped_from_resume": int(skipped_from_resume),
        "checkpoints_written": int(checkpoints_written),
        "shard_count": int(len(manifest.get("shards", []))),
        "feature_seconds": {k: float(v) for k, v in feature_seconds.items()},
    }
    return base_df.reset_index(drop=True), extraction_info


def _load_existing_core_state(
    *,
    out_dir: Path,
    cfg: EnhancedFairnessConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, pd.DataFrame, pd.DataFrame]:
    analysis_path = out_dir / "analysis_frame.parquet"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Sensitivity stage requires existing {analysis_path}. Run --enhanced-stage core or all first."
        )

    exact_df = _ensure_analysis_schema_columns(pd.read_parquet(analysis_path))
    if exact_df.empty:
        raise ValueError("Existing analysis_frame.parquet is empty; cannot run sensitivity stage.")

    base_df = exact_df.copy()

    if "selected_for_primary" in exact_df.columns:
        primary_df = exact_df[exact_df["selected_for_primary"]].copy()
    elif "is_canonical" in exact_df.columns:
        primary_df = exact_df[exact_df["is_canonical"]].copy()
    else:
        exact_df2, _, _ = apply_exact_dedup(base_df)
        primary_df = exact_df2[exact_df2["is_canonical"]].copy()

    if primary_df.empty:
        primary_df = exact_df.copy()

    metadata_path = out_dir / "run_metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    ita_cutoff = float(metadata.get("ita_cutoff", cfg.ita.binary_cutoff))

    if "ita_binary" not in base_df.columns:
        base_df["ita_binary"] = base_df["ita_deg"].apply(lambda v: ita_binary(float(v), cutoff=ita_cutoff))
    if "ita_binary" not in primary_df.columns:
        primary_df["ita_binary"] = primary_df["ita_deg"].apply(lambda v: ita_binary(float(v), cutoff=ita_cutoff))

    dedup_map_exact_path = out_dir / "dedup_map_exact.csv"
    dedup_report_path = out_dir / "dedup_report.csv"
    if dedup_map_exact_path.exists() and dedup_report_path.exists():
        dedup_map_exact = pd.read_csv(dedup_map_exact_path)
        dedup_report = pd.read_csv(dedup_report_path)
    else:
        _, dedup_map_exact, dedup_report = apply_exact_dedup(base_df)

    return base_df, exact_df, primary_df, ita_cutoff, dedup_map_exact, dedup_report


def _augment_analysis_frame(
    *,
    out_dir: Path,
    cfg: EnhancedFairnessConfig,
    run_config: Dict[str, object],
    workers: int,
    requested_columns: set[str],
    runtime_profile: Dict[str, object],
) -> pd.DataFrame:
    analysis_path = out_dir / "analysis_frame.parquet"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Augment stage requires existing {analysis_path}. Run --enhanced-stage core or all first."
        )

    exact_df = _ensure_analysis_schema_columns(pd.read_parquet(analysis_path))
    if exact_df.empty:
        return exact_df

    requested = set(str(c).strip() for c in requested_columns if str(c).strip())
    if not requested:
        return exact_df

    missing_cols = [
        col
        for col in sorted(requested)
        if col not in exact_df.columns
        or bool(exact_df[col].isna().all() if hasattr(exact_df[col], "isna") else False)
    ]
    if not missing_cols:
        return exact_df

    missing_row_mask = pd.Series(False, index=exact_df.index)
    for col in missing_cols:
        if col not in exact_df.columns:
            missing_row_mask = pd.Series(True, index=exact_df.index)
            break
        missing_row_mask = missing_row_mask | exact_df[col].isna()
    target_df = exact_df[missing_row_mask].copy()
    if target_df.empty:
        return exact_df

    stage_name = "augment_features"
    _profile_stage_start(runtime_profile, stage_name=stage_name, total_items=len(target_df))

    profile_plan = _feature_plan(cfg, stage="augment", requested_columns=set(missing_cols))
    model_name = str(run_config.get("model_name", "unknown"))
    prompt_variant = str(run_config.get("prompt_family", "none"))
    run_id = str(run_config.get("run_id", out_dir.parent.name))

    records = target_df[
        ["image_name", "image_path", "mask_path", "iou", "dice", "success_t050"]
    ].to_dict("records")
    per_image_metrics: Dict[str, Tuple[float, float, bool]] = {
        str(r["image_name"]): (
            float(r.get("iou", math.nan)),
            float(r.get("dice", math.nan)),
            bool(r.get("success_t050", False)),
        )
        for r in records
    }

    updates: List[Dict[str, object]] = []
    completed = 0
    max_inflight = int(cfg.runtime.max_inflight_tasks) if int(cfg.runtime.max_inflight_tasks) > 0 else int(workers)
    max_inflight = max(1, max_inflight)

    def _consume_aug_result(result_row: Dict[str, object] | None, perf: Dict[str, float], in_flight_count: int) -> None:
        nonlocal completed
        completed += 1
        if result_row is not None:
            update = {"image_name": result_row["image_name"]}
            for col in missing_cols:
                update[col] = result_row.get(col, math.nan)
            updates.append(update)
        _profile_stage_sample(
            runtime_profile,
            stage_name=stage_name,
            completed_items=completed,
            queue_depth=in_flight_count,
        )

    if workers <= 1:
        for idx, row in enumerate(records):
            _, out_row, perf = _build_analysis_row(
                idx=idx,
                img_path=Path(str(row["image_path"])),
                gt_path=Path(str(row["mask_path"])),
                per_image_metrics=per_image_metrics,
                by_sha={},
                by_id={},
                cfg=cfg,
                model_name=model_name,
                prompt_variant=prompt_variant,
                run_id=run_id,
                feature_plan=profile_plan,
            )
            _consume_aug_result(out_row, perf, 0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            iterator = iter(enumerate(records))
            in_flight: Dict[object, int] = {}
            while True:
                while len(in_flight) < max_inflight:
                    try:
                        idx, row = next(iterator)
                    except StopIteration:
                        break
                    future = executor.submit(
                        _build_analysis_row,
                        idx=idx,
                        img_path=Path(str(row["image_path"])),
                        gt_path=Path(str(row["mask_path"])),
                        per_image_metrics=per_image_metrics,
                        by_sha={},
                        by_id={},
                        cfg=cfg,
                        model_name=model_name,
                        prompt_variant=prompt_variant,
                        run_id=run_id,
                        feature_plan=profile_plan,
                    )
                    in_flight[future] = idx

                if not in_flight:
                    break
                done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    _ = in_flight.pop(future)
                    try:
                        _, out_row, perf = future.result()
                    except Exception:
                        out_row, perf = None, {}
                    _consume_aug_result(out_row, perf, len(in_flight))

    if updates:
        update_df = pd.DataFrame(updates).drop_duplicates(subset=["image_name"], keep="last")
        merged = exact_df.merge(update_df, on="image_name", how="left", suffixes=("", "__aug"))
        for col in missing_cols:
            aug_col = f"{col}__aug"
            if aug_col not in merged.columns:
                continue
            if col in merged.columns:
                merged[col] = merged[col].where(~merged[col].isna(), merged[aug_col])
            else:
                merged[col] = merged[aug_col]
            merged = merged.drop(columns=[aug_col])
        exact_df = _ensure_analysis_schema_columns(merged)
        write_analysis_frame(exact_df, analysis_path)
        _write_consolidated_caches(out_dir / "cache", exact_df)

    _profile_stage_end(runtime_profile, stage_name=stage_name, completed_items=completed)
    return exact_df


def _run_sensitivity_suite(
    *,
    base_df: pd.DataFrame,
    exact_df: pd.DataFrame,
    near_canonical_df: pd.DataFrame,
    cfg: EnhancedFairnessConfig,
    out_dir: Path,
) -> None:
    sensitivity_rows: List[Dict[str, object]] = []
    for mode_name, frame in (
        ("none", base_df),
        ("exact", exact_df[exact_df["is_canonical"]].copy() if "is_canonical" in exact_df.columns else pd.DataFrame()),
        ("near", near_canonical_df.copy()),
    ):
        if frame.empty:
            continue
        frame = frame[frame["ita_binary"].isin(["Lower ITA", "Higher ITA"])].copy()
        if frame.empty:
            continue
        eff_table, _, _ = compute_endpoint_effects(
            frame,
            group_col="ita_binary",
            lower_label="Lower ITA",
            higher_label="Higher ITA",
            bootstrap_method=cfg.bootstrap.method,
            bootstrap_fallback_method=cfg.bootstrap.fallback_method,
            n_resamples=max(1000, cfg.bootstrap.n_resamples // 2),
            seed=cfg.bootstrap.seed + 900,
        )
        row = {"dedup_mode": mode_name}
        for metric in (
            "median_iou_diff_lower_minus_higher",
            "mean_iou_diff_lower_minus_higher",
            "success_risk_difference_lower_minus_higher",
        ):
            part = eff_table[eff_table["metric"] == metric]
            if part.empty:
                continue
            row[f"{metric}__estimate"] = float(part.iloc[0]["estimate"])
        sensitivity_rows.append(row)

    pd.DataFrame(sensitivity_rows).to_csv(out_dir / "dedup_sensitivity.csv", index=False)

    if cfg.sensitivity.include_dependence and "dedup_group_id" in exact_df.columns:
        cluster_sizes = exact_df.groupby("dedup_group_id").size()
        dep_payload = {
            "n_unique_dedup_groups": int(cluster_sizes.shape[0]),
            "max_cluster_size": int(cluster_sizes.max()) if not cluster_sizes.empty else 0,
            "mean_cluster_size": float(cluster_sizes.mean()) if not cluster_sizes.empty else 0.0,
        }
        write_json(out_dir / "dependence_sensitivity.json", dep_payload)

    if cfg.sensitivity.include_mask_source and "mask_source" in exact_df.columns:
        mask_rows: List[Dict[str, object]] = []
        for mask_source, part in exact_df.groupby("mask_source", dropna=False):
            part = part[part["ita_binary"].isin(["Lower ITA", "Higher ITA"])].copy()
            if part.empty:
                continue
            eff_table, _, _ = compute_endpoint_effects(
                part,
                group_col="ita_binary",
                lower_label="Lower ITA",
                higher_label="Higher ITA",
                bootstrap_method=cfg.bootstrap.method,
                bootstrap_fallback_method=cfg.bootstrap.fallback_method,
                n_resamples=max(1000, cfg.bootstrap.n_resamples // 2),
                seed=cfg.bootstrap.seed + 1200,
            )
            median_row = eff_table[eff_table["metric"] == "median_iou_diff_lower_minus_higher"]
            rd_row = eff_table[eff_table["metric"] == "success_risk_difference_lower_minus_higher"]
            mask_rows.append(
                {
                    "mask_source": str(mask_source),
                    "n": int(len(part)),
                    "median_iou_diff": float(median_row.iloc[0]["estimate"]) if not median_row.empty else math.nan,
                    "rd": float(rd_row.iloc[0]["estimate"]) if not rd_row.empty else math.nan,
                }
            )
        pd.DataFrame(mask_rows).to_csv(out_dir / "mask_source_sensitivity.csv", index=False)


def _stage_wall_times(runtime_profile: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    stages = runtime_profile.get("stages") or {}
    if not isinstance(stages, dict):
        return out
    for key, payload in stages.items():
        if not isinstance(payload, dict):
            continue
        value = payload.get("wall_seconds")
        if value is None:
            continue
        out[str(key)] = float(value)
    return out


def run_enhanced_fairness_audit(
    *,
    image_mask_pairs: Iterable[Tuple[Path, Path]],
    per_image_metrics: Dict[str, Tuple[float, float, bool]],
    run_dir: Path,
    run_config: Dict[str, object],
    cfg: EnhancedFairnessConfig,
    success_threshold: float,
    workers: int = 1,
    augment_columns: Sequence[str] | None = None,
) -> Dict[str, object]:
    pair_list = [(Path(a), Path(b)) for a, b in image_mask_pairs]
    out_dir = Path(run_dir) / "fairness_enhanced"
    paths = ensure_output_dirs(out_dir)

    effective_cfg, stage_overrides = _effective_stage_config(cfg)
    stage = effective_cfg.runtime.stage
    requested_workers = max(1, int(workers))
    workers, worker_summary = _resolve_workers(
        effective_cfg,
        requested_workers=requested_workers,
    )

    runtime_profile = _new_runtime_profile(stage)
    runtime_profile["worker_summary"] = worker_summary
    extraction_info: Dict[str, object] = {
        "manifest_path": str(_features_manifest_path(paths["cache"])),
        "checkpoint_every": int(effective_cfg.runtime.checkpoint_every),
        "max_inflight_tasks": int(effective_cfg.runtime.max_inflight_tasks),
        "resume_enabled": bool(effective_cfg.runtime.resume),
        "manifest_restarts": 0,
        "processed_count_total": 0,
        "processed_this_run": 0,
        "skipped_from_resume": 0,
        "checkpoints_written": 0,
        "shard_count": 0,
        "requested_workers": int(requested_workers),
        "effective_workers": int(workers),
    }

    model_name = str(run_config.get("model_name", "unknown"))
    prompt_variant = str(run_config.get("prompt_family", "none"))
    run_id = str(run_config.get("run_id", Path(run_dir).name))

    base_df = pd.DataFrame()
    exact_df = pd.DataFrame()
    primary_df = pd.DataFrame()
    dedup_map_exact = pd.DataFrame()
    dedup_report = pd.DataFrame()
    dedup_map_near = None
    near_df = None
    near_canonical_df = pd.DataFrame()
    ita_cutoff = float(effective_cfg.ita.binary_cutoff)
    warnings_out: List[str] = []
    augment_requested = set(str(c).strip() for c in (augment_columns or []) if str(c).strip())
    trend_covariates_used: List[str] = []

    try:
        if stage in {"all", "core"}:
            feature_plan = _feature_plan(effective_cfg, stage=stage)
            logging.info(
                "[enhanced] Feature profile=%s plan=%s workers(requested=%s,effective=%s)",
                _profile_name(effective_cfg),
                json.dumps(feature_plan, sort_keys=True),
                requested_workers,
                workers,
            )
            _profile_stage_start(runtime_profile, stage_name="source_index")
            source_index_path = paths["cache"] / "source_index.parquet"
            source_index = build_or_load_source_index(cache_path=source_index_path, cfg=effective_cfg)
            by_sha, by_id = _source_lookup_maps(source_index)
            _profile_stage_end(runtime_profile, stage_name="source_index")

            base_df, extraction_info = _extract_features_with_checkpoints(
                pair_list=pair_list,
                per_image_metrics=per_image_metrics,
                by_sha=by_sha,
                by_id=by_id,
                cfg=effective_cfg,
                model_name=model_name,
                prompt_variant=prompt_variant,
                run_id=run_id,
                workers=workers,
                cache_dir=paths["cache"],
                runtime_profile=runtime_profile,
                feature_plan=feature_plan,
            )
            base_df = _ensure_analysis_schema_columns(base_df)

            finite_ita = base_df["ita_deg"].astype(float)
            finite_ita = finite_ita[np.isfinite(finite_ita)]
            if effective_cfg.ita.binary_strategy == "median" and len(finite_ita):
                ita_cutoff = float(np.median(finite_ita))
            else:
                ita_cutoff = float(effective_cfg.ita.binary_cutoff)
            base_df["ita_binary"] = base_df["ita_deg"].apply(lambda v: ita_binary(float(v), cutoff=ita_cutoff))
            _write_consolidated_caches(paths["cache"], base_df)

            _profile_stage_start(runtime_profile, stage_name="fingerprint_write")
            fp_cols = [c for c in ["image_id", "image_name", "image_path", "sha256", "phash64_hex"] if c in base_df.columns]
            fingerprints = base_df[fp_cols].copy()
            fingerprints.to_parquet(out_dir / "fingerprints.parquet", index=False)
            _profile_stage_end(runtime_profile, stage_name="fingerprint_write", completed_items=len(fingerprints))

            _profile_stage_start(runtime_profile, stage_name="dedup_exact")
            exact_df, dedup_map_exact, dedup_report = apply_exact_dedup(base_df)
            _profile_stage_end(runtime_profile, stage_name="dedup_exact", completed_items=len(exact_df))

            _profile_stage_start(runtime_profile, stage_name="dedup_near")
            if (effective_cfg.sensitivity.include_near_dedup or effective_cfg.dedup.mode == "near") and "phash64_hex" in exact_df.columns:
                if exact_df["phash64_hex"].isna().all():
                    logging.info("[enhanced] Skipping near-dedup in %s: phash64_hex unavailable", stage)
                else:
                    near_df, dedup_map_near = apply_near_dedup(
                        exact_df[exact_df["is_canonical"]].copy(),
                        threshold=effective_cfg.dedup.near_hamming_threshold,
                    )
            elif effective_cfg.sensitivity.include_near_dedup or effective_cfg.dedup.mode == "near":
                logging.info("[enhanced] Skipping near-dedup in %s: phash64_hex column missing", stage)
            near_canonical_df = _near_canonical_rows(near_df, dedup_map_near)
            _profile_stage_end(
                runtime_profile,
                stage_name="dedup_near",
                completed_items=int(len(near_df)) if isinstance(near_df, pd.DataFrame) else 0,
            )

            if effective_cfg.dedup.mode == "none":
                primary_df = base_df.copy()
                primary_df["dedup_group_id"] = [f"none_{i:06d}" for i in range(len(primary_df))]
            elif effective_cfg.dedup.mode == "near" and not near_canonical_df.empty:
                primary_df = near_canonical_df.copy()
                primary_df["dedup_group_id"] = primary_df["near_dedup_group_id"]
            elif effective_cfg.dedup.mode == "near":
                logging.warning(
                    "Near-dedup selected but canonical near set is empty; falling back to exact-dedup canonical rows."
                )
                primary_df = exact_df[exact_df["is_canonical"]].copy()
                primary_df["dedup_group_id"] = primary_df["dedup_group_id"].astype(str)
            else:
                primary_df = exact_df[exact_df["is_canonical"]].copy()
                primary_df["dedup_group_id"] = primary_df["dedup_group_id"].astype(str)

            exact_df["selected_for_primary"] = exact_df["image_name"].isin(primary_df["image_name"].tolist())
            write_analysis_frame(exact_df, out_dir / "analysis_frame.parquet")

            label_text = build_label_text(
                grouping_strategy="binary",
                cutoff=ita_cutoff,
                region_strategy=str(effective_cfg.ita.region_strategy),
            )
            analysis_valid = primary_df[primary_df["ita_binary"].isin(["Lower ITA", "Higher ITA"])].copy()

            _profile_stage_start(runtime_profile, stage_name="effects_bootstrap")
            effects_table, effects_payload, warnings_out = compute_endpoint_effects(
                analysis_valid,
                group_col="ita_binary",
                lower_label="Lower ITA",
                higher_label="Higher ITA",
                bootstrap_method=effective_cfg.bootstrap.method,
                bootstrap_fallback_method=effective_cfg.bootstrap.fallback_method,
                n_resamples=effective_cfg.bootstrap.n_resamples,
                seed=effective_cfg.bootstrap.seed,
            )
            effects_payload["label_text"] = asdict(label_text)
            effects_payload["ita_cutoff"] = ita_cutoff
            effects_payload["audit_mode"] = "enhanced"
            effects_payload["runtime_stage"] = stage
            write_endpoint_effects(effects_table, effects_payload, out_dir)

            threshold_table = threshold_sensitivity_table(
                analysis_valid,
                thresholds=effective_cfg.sensitivity.success_thresholds,
                group_col="ita_binary",
                lower_label="Lower ITA",
                higher_label="Higher ITA",
                iou_col="iou",
                bootstrap_method=effective_cfg.bootstrap.method,
                bootstrap_fallback_method=effective_cfg.bootstrap.fallback_method,
                n_resamples=max(1000, effective_cfg.bootstrap.n_resamples // 2),
                seed=effective_cfg.bootstrap.seed + 500,
            )
            write_threshold_sensitivity(threshold_table, out_dir)
            _profile_stage_end(runtime_profile, stage_name="effects_bootstrap")

            _profile_stage_start(runtime_profile, stage_name="trend_models")
            trend_df = analysis_valid.copy()
            cov_cols = _trend_covariate_columns(trend_df)
            trend_covariates_used = list(cov_cols)
            logging.info("[enhanced] Trend covariates used: %s", ",".join(cov_cols) if cov_cols else "<none>")
            for cov_col in cov_cols:
                trend_df[cov_col] = pd.to_numeric(trend_df[cov_col], errors="coerce")
            success_trend_df = build_success_trend_frames(
                df=trend_df,
                ita_col="ita_deg",
                outcome_col="success_t050",
                covariate_cols=cov_cols,
                knots=effective_cfg.trends.knots,
                degree=effective_cfg.trends.degree,
                n_bootstrap=effective_cfg.trends.bootstrap_resamples,
                seed=effective_cfg.bootstrap.seed,
            )
            iou_trend_df, iou_summary = build_iou_trend_frames(
                df=trend_df,
                ita_col="ita_deg",
                iou_col="iou",
                covariate_cols=cov_cols,
                knots=effective_cfg.trends.knots,
                degree=effective_cfg.trends.degree,
                quantile=effective_cfg.trends.quantile,
                alpha=effective_cfg.trends.quantile_alpha,
                n_bootstrap=effective_cfg.trends.bootstrap_resamples,
                seed=effective_cfg.bootstrap.seed + 100,
            )
            write_trend_outputs(
                success_df=success_trend_df,
                iou_df=iou_trend_df,
                iou_summary=iou_summary,
                out_dir=out_dir,
            )
            _profile_stage_end(runtime_profile, stage_name="trend_models")

            write_ita_bins(primary_df, out_dir)
            write_covariate_qc(primary_df, paths["covariates_qc"])
            write_dedup_outputs(
                dedup_map_exact=dedup_map_exact,
                dedup_report=dedup_report,
                dedup_map_near=dedup_map_near if effective_cfg.dedup.include_near_map else None,
                out_dir=out_dir,
            )
            _duplicate_examples(
                base_df=base_df,
                dedup_map_exact=dedup_map_exact,
                out_dir=paths["duplicate_examples"],
                limit=effective_cfg.duplicate_examples_limit,
            )

            if stage == "all":
                _profile_stage_start(runtime_profile, stage_name="sensitivity_suite")
                _run_sensitivity_suite(
                    base_df=base_df,
                    exact_df=exact_df,
                    near_canonical_df=near_canonical_df,
                    cfg=effective_cfg,
                    out_dir=out_dir,
                )
                _profile_stage_end(runtime_profile, stage_name="sensitivity_suite")

        elif stage == "sensitivity":
            _profile_stage_start(runtime_profile, stage_name="sensitivity_load")
            base_df, exact_df, primary_df, ita_cutoff, dedup_map_exact, dedup_report = _load_existing_core_state(
                out_dir=out_dir,
                cfg=effective_cfg,
            )
            _profile_stage_end(runtime_profile, stage_name="sensitivity_load", completed_items=len(exact_df))

            need_near = bool(effective_cfg.sensitivity.include_near_dedup or effective_cfg.dedup.mode == "near")
            if need_near and ("phash64_hex" not in exact_df.columns or exact_df["phash64_hex"].isna().all()):
                logging.info("[enhanced] Auto-augmenting missing phash64_hex for sensitivity near-dedup")
                _profile_stage_start(runtime_profile, stage_name="auto_augment")
                _augment_analysis_frame(
                    out_dir=out_dir,
                    cfg=effective_cfg,
                    run_config=run_config,
                    workers=workers,
                    requested_columns={"phash64_hex"},
                    runtime_profile=runtime_profile,
                )
                _profile_stage_end(runtime_profile, stage_name="auto_augment")
                base_df, exact_df, primary_df, ita_cutoff, dedup_map_exact, dedup_report = _load_existing_core_state(
                    out_dir=out_dir,
                    cfg=effective_cfg,
                )

            _profile_stage_start(runtime_profile, stage_name="dedup_near")
            if need_near and "phash64_hex" in exact_df.columns and not exact_df["phash64_hex"].isna().all():
                near_df, dedup_map_near = apply_near_dedup(
                    exact_df[exact_df["is_canonical"]].copy() if "is_canonical" in exact_df.columns else exact_df.copy(),
                    threshold=effective_cfg.dedup.near_hamming_threshold,
                )
            near_canonical_df = _near_canonical_rows(near_df, dedup_map_near)
            _profile_stage_end(
                runtime_profile,
                stage_name="dedup_near",
                completed_items=int(len(near_df)) if isinstance(near_df, pd.DataFrame) else 0,
            )

            write_dedup_outputs(
                dedup_map_exact=dedup_map_exact,
                dedup_report=dedup_report,
                dedup_map_near=dedup_map_near if effective_cfg.dedup.include_near_map else None,
                out_dir=out_dir,
            )

            _profile_stage_start(runtime_profile, stage_name="sensitivity_suite")
            _run_sensitivity_suite(
                base_df=base_df,
                exact_df=exact_df,
                near_canonical_df=near_canonical_df,
                cfg=effective_cfg,
                out_dir=out_dir,
            )
            _profile_stage_end(runtime_profile, stage_name="sensitivity_suite")
        else:
            requested = _requested_feature_columns(
                effective_cfg,
                stage="augment",
                user_requested=augment_requested,
            )
            if not requested:
                requested = {"phash64_hex"}
            exact_df = _augment_analysis_frame(
                out_dir=out_dir,
                cfg=effective_cfg,
                run_config=run_config,
                workers=workers,
                requested_columns=requested,
                runtime_profile=runtime_profile,
            )
            base_df = exact_df.copy()
            if "selected_for_primary" in exact_df.columns:
                primary_df = exact_df[exact_df["selected_for_primary"]].copy()
            elif "is_canonical" in exact_df.columns:
                primary_df = exact_df[exact_df["is_canonical"]].copy()
            else:
                primary_df = exact_df.copy()
            dedup_map_exact_path = out_dir / "dedup_map_exact.csv"
            dedup_report_path = out_dir / "dedup_report.csv"
            if dedup_map_exact_path.exists() and dedup_report_path.exists():
                dedup_map_exact = pd.read_csv(dedup_map_exact_path)
                dedup_report = pd.read_csv(dedup_report_path)

        existing_metadata = _load_json(out_dir / "run_metadata.json")
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}

        metadata = dict(existing_metadata)
        effective_inflight = (
            int(extraction_info.get("max_inflight_tasks", 0) or 0)
            if stage in {"all", "core"}
            else max(1, int(effective_cfg.runtime.max_inflight_tasks) if int(effective_cfg.runtime.max_inflight_tasks) > 0 else int(workers))
        )
        checkpoint_payload = (
            extraction_info
            if stage in {"all", "core"}
            else existing_metadata.get("checkpoint", extraction_info)
        )
        metadata.update(
            {
                "git_commit": _git_commit(),
                "config": config_to_dict(effective_cfg),
                "runtime_stage": stage,
                "runtime_stage_overrides": stage_overrides,
                "runtime": {
                    "stage": stage,
                    "resume_enabled": bool(effective_cfg.runtime.resume),
                    "checkpoint_every": int(effective_cfg.runtime.checkpoint_every),
                    "max_inflight_tasks_configured": int(effective_cfg.runtime.max_inflight_tasks),
                    "max_inflight_tasks_effective": int(effective_inflight),
                    "requested_workers": int(requested_workers),
                    "workers": int(workers),
                    "workers_auto": bool(effective_cfg.runtime.workers_auto),
                    "worker_summary": dict(worker_summary),
                    "stage_wall_seconds": _stage_wall_times(runtime_profile),
                },
                "checkpoint": checkpoint_payload,
                "dataset_counts": {
                    "input_pairs": int(len(pair_list)),
                    "processed_rows": int(len(base_df)),
                    "primary_rows": int(len(primary_df)),
                },
                "ita_cutoff": float(ita_cutoff),
                "ita_method": _ita_method_payload(effective_cfg, ita_cutoff=float(ita_cutoff)),
                "bootstrap": {
                    "n_resamples": effective_cfg.bootstrap.n_resamples,
                    "method": effective_cfg.bootstrap.method,
                    "fallback_method": effective_cfg.bootstrap.fallback_method,
                    "seed": effective_cfg.bootstrap.seed,
                },
                "success_threshold_arg": float(success_threshold),
                "augment_columns": sorted(augment_requested),
                "trend_covariates_used": list(trend_covariates_used),
                "warnings": list(warnings_out),
                "updated_at": _now_iso(),
            }
        )
        _write_ita_method_note(out_dir, effective_cfg, ita_cutoff=float(ita_cutoff))
        write_json(out_dir / "run_metadata.json", metadata)

        runtime_profile["status"] = "ok"
        runtime_profile["finished_at"] = _now_iso()
        write_json(out_dir / "runtime_profile.json", runtime_profile)

        return {
            "out_dir": str(out_dir),
            "analysis_rows": int(len(primary_df)),
            "ita_cutoff": float(ita_cutoff),
            "stage": stage,
        }

    except Exception as exc:
        runtime_profile["status"] = "failed"
        runtime_profile["error"] = str(exc)
        runtime_profile["finished_at"] = _now_iso()
        try:
            write_json(out_dir / "runtime_profile.json", runtime_profile)
        except Exception:
            pass
        raise
