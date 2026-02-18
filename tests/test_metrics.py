import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gemini_segmentation.metrics import compute_metrics_for_masks


def test_compute_metrics_handles_rgb_ground_truth() -> None:
    gt = np.zeros((6, 7, 3), dtype=np.uint8)
    gt[1:5, 2:6, :] = 255

    pred = np.zeros((6, 7), dtype=np.uint8)
    pred[1:5, 2:6] = 255

    iou, dice, success = compute_metrics_for_masks(gt, [pred], success_threshold=0.5)
    assert iou == 1.0
    assert dice == 1.0
    assert success is True


def test_compute_metrics_handles_rgb_prediction() -> None:
    gt = np.zeros((6, 7), dtype=np.uint8)
    gt[1:5, 2:6] = 255

    pred = np.zeros((6, 7, 3), dtype=np.uint8)
    pred[1:5, 2:6, :] = 255

    iou, dice, success = compute_metrics_for_masks(gt, [pred], success_threshold=0.5)
    assert iou == 1.0
    assert dice == 1.0
    assert success is True


def test_compute_metrics_shape_mismatch_returns_failure() -> None:
    gt = np.zeros((6, 7, 3), dtype=np.uint8)
    pred = np.zeros((5, 7), dtype=np.uint8)

    iou, dice, success = compute_metrics_for_masks(gt, [pred], success_threshold=0.5)
    assert iou == 0.0
    assert dice == 0.0
    assert success is False
