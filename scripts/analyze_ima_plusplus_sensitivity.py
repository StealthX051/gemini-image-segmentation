#!/usr/bin/env python3
"""Run IMA++ sensitivity analyses for a completed segmentation run.

Outputs are written under:
<run_dir>/ima_plusplus_sensitivity/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from gemini_segmentation.metrics import compute_metrics_for_masks


STAPLE_PATTERN = re.compile(r"_ST_ST_ST_ST\.png$", re.IGNORECASE)
MV_PATTERN = re.compile(r"_MV_MV_MV_MV\.png$", re.IGNORECASE)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image_obj:
        return np.asarray(image_obj)


def _resolve_path(dataset_root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return dataset_root / path


def _coerce_json_field(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return default
        try:
            return json.loads(token)
        except json.JSONDecodeError:
            return default
    return default


def _load_index_rows(index_path: Path) -> List[Dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(f"IMA++ index not found: {index_path}")

    if index_path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    if index_path.suffix.lower() == ".csv":
        df = pd.read_csv(index_path)
        rows = []
        for _, row in df.iterrows():
            payload = row.to_dict()
            payload["all_mask_paths"] = _coerce_json_field(payload.get("all_mask_paths"), default=[])
            payload["all_mask_metadata"] = _coerce_json_field(payload.get("all_mask_metadata"), default=[])
            rows.append(payload)
        return rows

    raise ValueError(f"Unsupported index format: {index_path}")


def _classify_mask(mask_path: str) -> str:
    name = Path(mask_path).name
    if STAPLE_PATTERN.search(name):
        return "consensus_staple"
    if MV_PATTERN.search(name):
        return "consensus_mv"
    return "annotator"


def _find_first(metadata: Dict[str, Any], candidates: Sequence[str], default: str = "unknown") -> str:
    lowered = {str(k).lower(): v for k, v in metadata.items()}
    for candidate in candidates:
        if candidate.lower() in lowered:
            value = lowered[candidate.lower()]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            token = str(value).strip()
            if token:
                return token
    return default


def _stats_from_series(series: pd.Series) -> Dict[str, float]:
    if series.empty:
        return {
            "mean": math.nan,
            "median": math.nan,
            "iqr": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    q75 = float(series.quantile(0.75))
    q25 = float(series.quantile(0.25))
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "iqr": q75 - q25,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    pd.DataFrame(rows_list).to_csv(path, index=False)


def analyze_sensitivity(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}

    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else Path(run_config.get("dataset_root", "")).expanduser().resolve()
    )
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}. Pass --dataset-root or ensure run_config has a valid path."
        )

    success_threshold = (
        float(args.success_threshold)
        if args.success_threshold is not None
        else float(run_config.get("success_threshold", 0.5))
    )

    index_path = (
        Path(args.index_path).expanduser().resolve()
        if args.index_path
        else dataset_root / "metadata" / "ima_plusplus_index.jsonl"
    )
    rows = _load_index_rows(index_path)

    pred_masks_dir = run_dir / "masks"
    if not pred_masks_dir.exists():
        raise FileNotFoundError(f"Prediction mask directory not found: {pred_masks_dir}")

    out_dir = run_dir / "ima_plusplus_sensitivity"

    metrics_mv: List[Dict[str, Any]] = []
    metrics_annotators: List[Dict[str, Any]] = []

    for row in rows:
        image_path = _resolve_path(dataset_root, row.get("image_path"))
        if image_path is None:
            continue
        image_name = image_path.name
        pred_path = pred_masks_dir / image_name
        if not pred_path.exists():
            logging.warning("Skipping %s; missing predicted mask %s", image_name, pred_path)
            continue

        pred_mask = _load_mask(pred_path)
        isic_id = str(row.get("ISIC_id", "")).strip()

        mv_path = _resolve_path(dataset_root, row.get("mv_mask_path"))
        if mv_path is not None and mv_path.exists():
            gt_mv = _load_mask(mv_path)
            iou_mv, dice_mv, success_mv = compute_metrics_for_masks(
                gt_mv,
                [pred_mask],
                success_threshold=success_threshold,
            )
            metrics_mv.append(
                {
                    "image_name": image_name,
                    "ISIC_id": isic_id,
                    "gt_policy": row.get("gt_policy"),
                    "mv_mask_path": str(mv_path),
                    "iou": iou_mv,
                    "dice": dice_mv,
                    "success": success_mv,
                }
            )

        all_mask_paths = _coerce_json_field(row.get("all_mask_paths"), default=[])
        all_mask_metadata = _coerce_json_field(row.get("all_mask_metadata"), default=[])

        for idx, mask_meta in enumerate(all_mask_metadata):
            if not isinstance(mask_meta, dict):
                continue

            mask_path_token = mask_meta.get("mask_path")
            if not mask_path_token and idx < len(all_mask_paths):
                mask_path_token = all_mask_paths[idx]
            if not mask_path_token:
                continue

            mask_kind = str(mask_meta.get("mask_kind") or _classify_mask(str(mask_path_token)))
            if mask_kind != "annotator":
                continue

            mask_path = _resolve_path(dataset_root, str(mask_path_token))
            if mask_path is None or not mask_path.exists():
                logging.warning("Skipping annotator mask for %s; path missing: %s", image_name, mask_path)
                continue

            gt_ann = _load_mask(mask_path)
            iou_ann, dice_ann, success_ann = compute_metrics_for_masks(
                gt_ann,
                [pred_mask],
                success_threshold=success_threshold,
            )

            metrics_annotators.append(
                {
                    "image_name": image_name,
                    "ISIC_id": isic_id,
                    "mask_path": str(mask_path),
                    "annotator": _find_first(mask_meta, ("annotator", "annotator_id", "rater", "rater_id")),
                    "tool": _find_first(mask_meta, ("tool", "annotation_tool", "tool_name")),
                    "skill_level": _find_first(mask_meta, ("skill_level", "skill", "expertise")),
                    "iou": iou_ann,
                    "dice": dice_ann,
                    "success": success_ann,
                }
            )

    _write_csv(out_dir / "metrics_mv.csv", metrics_mv)
    _write_csv(out_dir / "metrics_annotators.csv", metrics_annotators)

    annot_df = pd.DataFrame(metrics_annotators)
    mv_df = pd.DataFrame(metrics_mv)

    per_image_summary_rows: List[Dict[str, Any]] = []
    if not annot_df.empty:
        for (image_name, isic_id), group in annot_df.groupby(["image_name", "ISIC_id"], sort=True):
            iou_stats = _stats_from_series(group["iou"])
            dice_stats = _stats_from_series(group["dice"])
            per_image_summary_rows.append(
                {
                    "image_name": image_name,
                    "ISIC_id": isic_id,
                    "n_annotator_masks": int(len(group)),
                    "iou_mean": iou_stats["mean"],
                    "iou_median": iou_stats["median"],
                    "iou_iqr": iou_stats["iqr"],
                    "iou_min": iou_stats["min"],
                    "iou_max": iou_stats["max"],
                    "dice_mean": dice_stats["mean"],
                    "dice_median": dice_stats["median"],
                    "dice_iqr": dice_stats["iqr"],
                    "dice_min": dice_stats["min"],
                    "dice_max": dice_stats["max"],
                }
            )
    _write_csv(out_dir / "per_image_annotator_summary.csv", per_image_summary_rows)

    overall_rows: List[Dict[str, Any]] = []
    if not mv_df.empty:
        overall_rows.append(
            {
                "comparison_set": "mv_consensus",
                "count": int(len(mv_df)),
                "mean_iou": float(mv_df["iou"].mean()),
                "median_iou": float(mv_df["iou"].median()),
                "mean_dice": float(mv_df["dice"].mean()),
                "median_dice": float(mv_df["dice"].median()),
                "success_rate": float(mv_df["success"].mean()),
            }
        )
    if not annot_df.empty:
        overall_rows.append(
            {
                "comparison_set": "annotators",
                "count": int(len(annot_df)),
                "mean_iou": float(annot_df["iou"].mean()),
                "median_iou": float(annot_df["iou"].median()),
                "mean_dice": float(annot_df["dice"].mean()),
                "median_dice": float(annot_df["dice"].median()),
                "success_rate": float(annot_df["success"].mean()),
            }
        )
    _write_csv(out_dir / "summary_overall.csv", overall_rows)

    by_tool_rows: List[Dict[str, Any]] = []
    if not annot_df.empty:
        for tool, group in annot_df.groupby("tool", dropna=False, sort=True):
            by_tool_rows.append(
                {
                    "tool": str(tool),
                    "count": int(len(group)),
                    "mean_iou": float(group["iou"].mean()),
                    "median_iou": float(group["iou"].median()),
                    "mean_dice": float(group["dice"].mean()),
                    "median_dice": float(group["dice"].median()),
                    "success_rate": float(group["success"].mean()),
                }
            )
    _write_csv(out_dir / "summary_by_tool.csv", by_tool_rows)

    by_skill_rows: List[Dict[str, Any]] = []
    if not annot_df.empty:
        for skill, group in annot_df.groupby("skill_level", dropna=False, sort=True):
            by_skill_rows.append(
                {
                    "skill_level": str(skill),
                    "count": int(len(group)),
                    "mean_iou": float(group["iou"].mean()),
                    "median_iou": float(group["iou"].median()),
                    "mean_dice": float(group["dice"].mean()),
                    "median_dice": float(group["dice"].median()),
                    "success_rate": float(group["success"].mean()),
                }
            )
    _write_csv(out_dir / "summary_by_skill_level.csv", by_skill_rows)

    logging.info("IMA++ sensitivity analysis complete: %s", out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run IMA++ sensitivity analyses for a completed run.")
    parser.add_argument("--run-dir", required=True, help="Segmentation run directory under results/")
    parser.add_argument("--dataset-root", help="Dataset root containing metadata/ima_plusplus_index.jsonl")
    parser.add_argument("--index-path", help="Optional path to IMA++ index (.jsonl or .csv)")
    parser.add_argument("--success-threshold", type=float, help="Override IoU success threshold")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    try:
        analyze_sensitivity(args)
    except Exception as exc:  # pragma: no cover - entrypoint safety
        logging.exception("IMA++ sensitivity analysis failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
