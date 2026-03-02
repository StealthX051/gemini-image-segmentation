from __future__ import annotations

import numpy as np

from nanobanana_segmentation.study.eval import evaluate_segmentation


def test_evaluate_segmentation_rgb_gt_and_2d_pred() -> None:
    gt = np.zeros((16, 16, 3), dtype=np.uint8)
    gt[4:12, 5:11, :] = 255
    pred = np.zeros((16, 16), dtype=np.uint8)
    pred[4:12, 5:11] = 255

    metrics = evaluate_segmentation(gt, pred)
    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_evaluate_segmentation_shape_mismatch_returns_zeroes() -> None:
    gt = np.zeros((16, 16, 3), dtype=np.uint8)
    pred = np.zeros((12, 16), dtype=np.uint8)

    metrics = evaluate_segmentation(gt, pred)
    assert metrics == {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
