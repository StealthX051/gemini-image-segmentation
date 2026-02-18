from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .types import BootstrapCI, PerImageMetrics, RunSummary


def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    if np.sum(union) == 0:
        return 1.0
    return float(np.sum(intersection) / np.sum(union))


def calculate_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    intersection = np.logical_and(y_true, y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    if denominator == 0:
        return 1.0
    return float(2.0 * np.sum(intersection) / denominator)


def _as_single_channel(mask: np.ndarray, *, label: str) -> np.ndarray:
    """Normalize masks to HxW for metric computation."""

    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        logging.info("Converting %s mask from shape %s to single-channel", label, arr.shape)
        return np.max(arr, axis=-1)
    raise ValueError(f"Unsupported {label} mask shape: {arr.shape}")


def calculate_bootstrap_ci(
    data: List[float],
    n_resamples: int = 5000,
    method: str = "bca",
) -> BootstrapCI:
    if not data or len(data) < 2:
        return BootstrapCI(0.0, 0.0)

    try:
        from scipy.stats import bootstrap

        res = bootstrap(
            (np.array(data),),
            np.mean,
            method=method,
            n_resamples=n_resamples,
            confidence_level=0.95,
            vectorized=False,
        )
        lower, upper = res.confidence_interval.low, res.confidence_interval.high
    except Exception:
        bootstrapped_means = []
        n_samples = len(data)
        for _ in range(n_resamples):
            resample = np.random.choice(data, size=n_samples, replace=True)
            bootstrapped_means.append(np.mean(resample))
        lower = float(np.percentile(bootstrapped_means, 2.5))
        upper = float(np.percentile(bootstrapped_means, 97.5))
    return BootstrapCI(lower=float(lower), upper=float(upper))


def aggregate_run(
    metrics: Iterable[PerImageMetrics],
    *,
    n_resamples: int = 5000,
    method: str = "bca",
) -> RunSummary:
    metrics_list = list(metrics)
    if not metrics_list:
        return RunSummary()

    ious = [m.iou for m in metrics_list]
    dices = [m.dice for m in metrics_list]
    mean_iou = float(np.mean(ious))
    median_iou = float(np.median(ious))
    mean_dice = float(np.mean(dices))
    median_dice = float(np.median(dices))
    ci_iou = calculate_bootstrap_ci(ious, n_resamples=n_resamples, method=method)
    ci_dice = calculate_bootstrap_ci(dices, n_resamples=n_resamples, method=method)
    success_rate = float(np.mean([m.success for m in metrics_list]))

    return RunSummary(
        metrics=metrics_list,
        mean_iou=mean_iou,
        median_iou=median_iou,
        ci_iou=ci_iou,
        mean_dice=mean_dice,
        median_dice=median_dice,
        ci_dice=ci_dice,
        success_rate=success_rate,
    )


def aggregate_from_map(
    metrics_map: Dict[str, PerImageMetrics],
    *,
    n_resamples: int = 5000,
    method: str = "bca",
) -> RunSummary:
    return aggregate_run(metrics_map.values(), n_resamples=n_resamples, method=method) if metrics_map else RunSummary()


def write_metrics(metrics: List[PerImageMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [m.__dict__ for m in metrics]
    pd.DataFrame(rows).to_csv(path, index=False)


def load_metrics(path: Path) -> Dict[str, PerImageMetrics]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    records: Dict[str, PerImageMetrics] = {}
    for _, row in df.iterrows():
        records[str(row["image_name"])] = PerImageMetrics(
            image_name=str(row["image_name"]),
            iou=float(row["iou"]),
            dice=float(row["dice"]),
            success=bool(row["success"]),
        )
    return records


def write_summary(summary: RunSummary, path: Path) -> None:
    payload = {
        "mean_iou": summary.mean_iou,
        "median_iou": summary.median_iou,
        "ci_iou_lower": summary.ci_iou.lower,
        "ci_iou_upper": summary.ci_iou.upper,
        "mean_dice": summary.mean_dice,
        "median_dice": summary.median_dice,
        "ci_dice_lower": summary.ci_dice.lower,
        "ci_dice_upper": summary.ci_dice.upper,
        "success_rate": summary.success_rate,
    }
    pd.DataFrame([payload]).to_csv(path, index=False)


def upsert_metrics(
    metrics_map: Dict[str, PerImageMetrics],
    metric: PerImageMetrics,
    *,
    metrics_path: Path,
    summary_path: Path,
    n_resamples: int = 5000,
    method: str = "bca",
) -> Dict[str, PerImageMetrics]:
    metrics_map[metric.image_name] = metric
    write_metrics(list(metrics_map.values()), metrics_path)
    write_summary(
        aggregate_from_map(metrics_map, n_resamples=n_resamples, method=method), summary_path
    )
    return metrics_map


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=np.uint8)
    combined = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined = np.maximum(combined, mask)
    return combined


def compute_metrics_for_masks(
    ground_truth: np.ndarray, predictions: List[np.ndarray], success_threshold: float = 0.5
) -> Tuple[float, float, bool]:
    if ground_truth.size == 0:
        logging.warning("Ground truth mask empty; marking as failure")
        return 0.0, 0.0, False

    ground_truth_2d = _as_single_channel(ground_truth, label="ground truth")
    pred_masks_2d = [_as_single_channel(mask, label="prediction") for mask in predictions]
    combined_pred = combine_masks(pred_masks_2d) if pred_masks_2d else np.zeros_like(ground_truth_2d)
    if combined_pred.shape != ground_truth_2d.shape:
        logging.warning(
            "Prediction/ground-truth shape mismatch (%s vs %s); marking as failure",
            combined_pred.shape,
            ground_truth_2d.shape,
        )
        return 0.0, 0.0, False

    y_true = ground_truth_2d > 127
    y_pred = combined_pred > 127
    iou = calculate_iou(y_true, y_pred)
    dice = calculate_dice(y_true, y_pred)
    success = iou >= success_threshold
    return iou, dice, success
