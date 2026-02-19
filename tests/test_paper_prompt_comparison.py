import json
from pathlib import Path

import pandas as pd

from gemini_segmentation.paper.prompt_comparison import generate_prompt_comparison_report


def _write_run(
    run_dir: Path,
    *,
    prompt_family: str,
    mean_iou: float,
    mean_dice: float,
    success_rate: float,
    provider: str = "gemini",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset_name": "polyp",
        "provider": provider,
        "model_name": "x",
        "prompt_family": prompt_family,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    summary = pd.DataFrame(
        [
            {
                "mean_iou": mean_iou,
                "median_iou": mean_iou,
                "ci_iou_lower": mean_iou - 0.05,
                "ci_iou_upper": mean_iou + 0.05,
                "mean_dice": mean_dice,
                "median_dice": mean_dice,
                "ci_dice_lower": mean_dice - 0.05,
                "ci_dice_upper": mean_dice + 0.05,
                "success_rate": success_rate,
            }
        ]
    )
    summary.to_csv(run_dir / "summary.csv", index=False)


def test_generate_prompt_comparison_report_outputs_grouped_files(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "reports"
    dataset = "polyp"
    gemini_run_id = "polyp_full_3x3_w10_20260218-100000"
    moondream_run_id = "moondream3_polyp_full_20260218-110000"

    _write_run(
        results_dir / dataset / "gemini-2.5-flash" / "label_v1-aaaa1111" / gemini_run_id,
        prompt_family="label_v1",
        mean_iou=0.70,
        mean_dice=0.80,
        success_rate=0.90,
    )
    _write_run(
        results_dir / dataset / "gemini-2.5-flash" / "desc_v1-bbbb2222" / gemini_run_id,
        prompt_family="desc_v1",
        mean_iou=0.75,
        mean_dice=0.84,
        success_rate=0.92,
    )
    _write_run(
        results_dir / dataset / "moondream-3" / "label_v1-cccc3333" / moondream_run_id,
        prompt_family="label_v1",
        mean_iou=0.62,
        mean_dice=0.73,
        success_rate=0.81,
    )

    artifacts = generate_prompt_comparison_report(
        results_dir=results_dir,
        output_dir=output_dir,
        dataset=dataset,
        gemini_run_id=gemini_run_id,
        moondream_run_id=moondream_run_id,
        replicate_run_id=None,
    )

    assert artifacts.markdown.exists()
    assert artifacts.html.exists()
    assert artifacts.pdf.exists()
    assert artifacts.csv.exists()

    md_text = artifacts.markdown.read_text(encoding="utf-8")
    assert "## Gemini 2.5 Flash" in md_text
    assert "## Moondream 3" in md_text
    assert "95% CI" in md_text
    assert "Median IoU" in md_text
    assert "Label-Only" in md_text

    csv_df = pd.read_csv(artifacts.csv)
    assert set(csv_df["model"]) == {"gemini-2.5-flash", "moondream-3"}
    assert "model_display" in csv_df.columns
    assert "prompt_family_display" in csv_df.columns
    assert "median_iou" in csv_df.columns
    assert "median_dice" in csv_df.columns
    assert "success_rate" in csv_df.columns


def test_generate_prompt_comparison_report_includes_replicate_run_when_requested(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "reports"
    dataset = "polyp"
    gemini_run_id = "polyp_full_3x3_w10_20260218-100000"
    moondream_run_id = "moondream3_polyp_full_20260218-110000"
    replicate_run_id = "replicate_sa2va_polyp_full_20260219-120000"
    replicate_model_dir = (
        "bytedance_sa2va-26b-image_addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f"
    )

    _write_run(
        results_dir / dataset / "gemini-2.5-flash" / "label_v1-aaaa1111" / gemini_run_id,
        prompt_family="label_v1",
        mean_iou=0.70,
        mean_dice=0.80,
        success_rate=0.90,
        provider="gemini",
    )
    _write_run(
        results_dir / dataset / "moondream-3" / "label_v1-bbbb2222" / moondream_run_id,
        prompt_family="label_v1",
        mean_iou=0.62,
        mean_dice=0.73,
        success_rate=0.81,
        provider="moondream",
    )
    _write_run(
        results_dir / dataset / replicate_model_dir / "label_v1-cccc3333" / replicate_run_id,
        prompt_family="label_v1",
        mean_iou=0.66,
        mean_dice=0.77,
        success_rate=0.84,
        provider="replicate",
    )

    artifacts = generate_prompt_comparison_report(
        results_dir=results_dir,
        output_dir=output_dir,
        dataset=dataset,
        gemini_run_id=gemini_run_id,
        moondream_run_id=moondream_run_id,
        replicate_run_id=replicate_run_id,
    )

    md_text = artifacts.markdown.read_text(encoding="utf-8")
    assert "Replicate run ID" in md_text
    assert "SA2VA 26B (Replicate)" in md_text

    csv_df = pd.read_csv(artifacts.csv)
    assert "replicate" in set(csv_df["source"])
    assert replicate_model_dir in set(csv_df["model"])
