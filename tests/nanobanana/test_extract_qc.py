from __future__ import annotations

import numpy as np

from nanobanana_segmentation.core.extract.bw_threshold import extract_bw_threshold
from nanobanana_segmentation.core.extract.chromakey_hsv import extract_chromakey_hsv
from nanobanana_segmentation.core.extract.chromakey_ratio import extract_chromakey_ratio
from nanobanana_segmentation.core.extract.postprocess import standard_postprocess
from nanobanana_segmentation.core.qc import compute_qc_metrics, evaluate_qc
from nanobanana_segmentation.core.types import ConstraintConfig


def _synthetic_chromakey() -> np.ndarray:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 1] = 255  # green background in BGR
    img[16:48, 16:48] = [255, 255, 255]
    return img


def test_chromakey_extractors_detect_roi() -> None:
    image = _synthetic_chromakey()
    hsv_mask = extract_chromakey_hsv(image)
    ratio_mask = extract_chromakey_ratio(image)

    assert int((hsv_mask > 0).sum()) > 0
    assert int((ratio_mask > 0).sum()) > 0


def test_bw_threshold_and_postprocess() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:44, 20:44] = [255, 255, 255]

    mask = extract_bw_threshold(image, threshold=127, use_otsu=False)
    post = standard_postprocess(mask, image_area=64 * 64, task_profile="blob")
    assert int((post > 0).sum()) >= int((mask > 0).sum())


def test_qc_metrics_and_rules() -> None:
    image = _synthetic_chromakey()
    mask = extract_chromakey_hsv(image)
    metrics = compute_qc_metrics(
        mask=mask,
        surrogate=image,
        input_shape=(64, 64),
        attempt_mode="chromakey",
    )
    qc_pass, failures = evaluate_qc(metrics, ConstraintConfig(), attempt_mode="chromakey")
    assert qc_pass is True
    assert failures == []
