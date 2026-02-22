from pathlib import Path

import pandas as pd

from gemini_segmentation.paper.figures import generate_fairness_artifacts
from gemini_segmentation.types import BootstrapCI, FairnessResult, GroupSummary


def _summary(group: str, mean_iou: float, mean_dice: float, success_rate: float) -> GroupSummary:
    return GroupSummary(
        group_name=group,
        count=2,
        mean_iou=mean_iou,
        median_iou=mean_iou,
        ci_iou=BootstrapCI(lower=mean_iou - 0.05, upper=mean_iou + 0.05),
        mean_dice=mean_dice,
        median_dice=mean_dice,
        ci_dice=BootstrapCI(lower=mean_dice - 0.05, upper=mean_dice + 0.05),
        success_rate=success_rate,
    )


def test_generate_fairness_artifacts_from_objects(tmp_path: Path) -> None:
    results = [
        FairnessResult(
            image_name="img1.png",
            ita=35.0,
            chardon_label="Light",
            tone_group="Light",
            iou=0.8,
            dice=0.9,
            success=True,
            candidate_count=500,
        ),
        FairnessResult(
            image_name="img2.png",
            ita=15.0,
            chardon_label="Dark",
            tone_group="Dark",
            iou=0.6,
            dice=0.7,
            success=True,
            candidate_count=500,
        ),
        FairnessResult(
            image_name="img3.png",
            ita=12.0,
            chardon_label="Dark",
            tone_group="Dark",
            iou=0.3,
            dice=0.4,
            success=False,
            candidate_count=500,
        ),
        FairnessResult(
            image_name="img5.png",
            ita=20.0,
            chardon_label="Dark",
            tone_group="Dark",
            iou=0.2,
            dice=0.25,
            success=False,
            candidate_count=500,
        ),
        FairnessResult(
            image_name="img4.png",
            ita=40.0,
            chardon_label="Light",
            tone_group="Light",
            iou=0.55,
            dice=0.65,
            success=True,
            candidate_count=500,
        ),
    ]

    summaries = [
        _summary("Light", mean_iou=0.675, mean_dice=0.775, success_rate=1.0),
        _summary("Dark", mean_iou=0.3667, mean_dice=0.4833, success_rate=0.3333),
    ]

    stats_payload = {
        "kruskal_iou_p": 0.04,
        "kruskal_dice_p": 0.05,
        "cliffs_delta_iou_light_dark": 0.2,
        "chi2_success": 3.1,
        "chi2_success_p": 0.08,
    }

    figure_artifacts, table_artifacts = generate_fairness_artifacts(
        results=results,
        summaries=summaries,
        stats_payload=stats_payload,
        output_dir=tmp_path,
        seed=42,
    )

    assert figure_artifacts.pdf.exists()
    assert figure_artifacts.png.exists()
    assert figure_artifacts.svg.exists()
    assert table_artifacts.csv.exists()
    assert table_artifacts.html.exists()
    assert table_artifacts.docx.exists()

    table = pd.read_csv(table_artifacts.csv)
    assert set(table["metric"]) == {"IoU mean", "IoU median", "Dice mean", "Dice median"}
    assert "Light" in table.columns and "Dark" in table.columns

    def _point_estimate(cell: str) -> float:
        return float(str(cell).split(" ")[0])

    dark_mean = _point_estimate(table.loc[table["metric"] == "IoU mean", "Dark"].iloc[0])
    dark_median = _point_estimate(table.loc[table["metric"] == "IoU median", "Dark"].iloc[0])
    assert dark_mean != dark_median
