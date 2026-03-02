from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from gemini_segmentation.metrics import calculate_dice, calculate_iou


def _as_single_channel(mask: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(mask)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        logging.info("Converting %s mask from shape %s to single-channel", label, arr.shape)
        return np.max(arr, axis=-1)
    raise ValueError(f"Unsupported {label} mask shape: {arr.shape}")


def evaluate_segmentation(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    gt_2d = _as_single_channel(gt_mask, label="ground truth")
    pred_2d = _as_single_channel(pred_mask, label="prediction")
    if gt_2d.shape != pred_2d.shape:
        logging.warning("Skipping metric computation for shape mismatch gt=%s pred=%s", gt_2d.shape, pred_2d.shape)
        return {
            "iou": 0.0,
            "dice": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    gt = (gt_2d > 127)
    pred = (pred_2d > 127)

    iou = float(calculate_iou(gt, pred))
    dice = float(calculate_dice(gt, pred))

    tp = float(np.logical_and(gt, pred).sum())
    fp = float(np.logical_and(~gt, pred).sum())
    fn = float(np.logical_and(gt, ~pred).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "iou": iou,
        "dice": dice,
        "precision": float(precision),
        "recall": float(recall),
    }
