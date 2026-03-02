from __future__ import annotations

from pathlib import Path

import pandas as pd

from nanobanana_segmentation.study.reports import build_reports


def test_build_reports(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "image_name": "a.png",
                "replicate_idx": 0,
                "tool_mode": "closed",
                "iou": 0.5,
                "dice": 0.6,
                "precision": 0.7,
                "recall": 0.8,
                "qc_pass": True,
                "retrieval_duplicate": False,
                "retrieval_mask_source": False,
                "audit_unavailable": True,
            },
            {
                "image_name": "a.png",
                "replicate_idx": 0,
                "tool_mode": "text_image",
                "iou": 0.6,
                "dice": 0.7,
                "precision": 0.8,
                "recall": 0.9,
                "qc_pass": True,
                "retrieval_duplicate": False,
                "retrieval_mask_source": False,
                "audit_unavailable": True,
            },
        ]
    )

    outputs = build_reports(df, tmp_path / "reports")
    assert Path(outputs["summary"]).exists()
    assert Path(outputs["paired_delta"]).exists()
    assert Path(outputs["delta_plot"]).exists()
    assert Path(outputs["summary_primary"]).exists()
    assert Path(outputs["summary_sensitivity"]).exists()
    assert Path(outputs["analysis_partitions"]).exists()
