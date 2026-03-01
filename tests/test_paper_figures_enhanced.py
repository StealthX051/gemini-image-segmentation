import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from gemini_segmentation.paper.figures_enhanced import generate_fairness_enhanced_artifacts


def _write_qc_png(path: Path, value: int) -> None:
    arr = np.full((48, 64, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _build_fixture(root: Path, *, include_optional: bool, include_covadj: bool = False) -> Path:
    fairness_dir = root / "fairness_enhanced"
    fairness_dir.mkdir(parents=True, exist_ok=True)

    analysis = pd.DataFrame(
        {
            "image_name": [f"img{i}.png" for i in range(1, 7)],
            "iou": [0.71, 0.65, 0.49, 0.55, 0.78, 0.31],
            "dice": [0.81, 0.75, 0.59, 0.66, 0.85, 0.41],
            "success_t050": [True, True, False, True, True, False],
            "ita_deg": [42.0, 35.5, 18.2, 10.0, 55.1, 21.3],
            "ita_binary": ["Higher ITA", "Higher ITA", "Lower ITA", "Lower ITA", "Higher ITA", "Lower ITA"],
            "dataset_source_primary": ["isic2017", "isic2018", "isic2017", "ima_plusplus", "isic2016", "isic2017"],
            "split": ["test", "test", "val", "train", "test", "val"],
            "mask_source": ["challenge_gt", "challenge_gt", "challenge_gt", "consensus_staple", "challenge_gt", "consensus_mv"],
            "selected_for_primary": [True, True, True, True, True, False],
        }
    )
    analysis.to_parquet(fairness_dir / "analysis_frame.parquet", index=False)

    effects = pd.DataFrame(
        {
            "metric": [
                "cliffs_delta_iou_lower_vs_higher",
                "median_iou_diff_lower_minus_higher",
                "mean_iou_diff_lower_minus_higher",
                "success_risk_difference_lower_minus_higher",
                "success_relative_risk_lower_over_higher",
                "success_odds_ratio_lower_over_higher",
            ],
            "estimate": [-0.12, -0.03, -0.02, -0.09, 0.86, 0.72],
            "ci_lower": [np.nan, -0.08, -0.06, -0.21, 0.65, 0.45],
            "ci_upper": [np.nan, 0.01, 0.02, 0.01, 1.03, 1.10],
            "ci_method": ["na", "percentile", "percentile", "percentile", "percentile", "percentile"],
            "p_value": [0.19, 0.19, 0.19, 0.07, 0.07, 0.07],
        }
    )
    effects.to_csv(fairness_dir / "endpoint_effects_table.csv", index=False)

    effects_payload = {
        "summary": {
            "lower_n": 2,
            "higher_n": 3,
            "lower_mean_iou": 0.52,
            "higher_mean_iou": 0.71,
            "lower_success_rate": 0.5,
            "higher_success_rate": 1.0,
        },
        "label_text": {
            "methods_snippet": "Fairness analyses were stratified by the image-derived perilesional skin tone proxy (ITA).",
            "figure_caption_snippet": "Groups reflect the image-derived perilesional skin tone proxy (ITA); lower-ITA (darker-appearing) vs higher-ITA (lighter-appearing) strata.",
        },
    }
    (fairness_dir / "endpoint_effects.json").write_text(json.dumps(effects_payload), encoding="utf-8")

    trend_success = pd.DataFrame(
        {
            "ita_deg": [-20, -5, 10, 25, 40, -20, -5, 10, 25, 40],
            "pred": [0.48, 0.55, 0.62, 0.71, 0.76, 0.50, 0.57, 0.65, 0.73, 0.79],
            "ci_lower": [0.40, 0.45, 0.53, 0.60, 0.66, 0.42, 0.48, 0.56, 0.63, 0.69],
            "ci_upper": [0.57, 0.64, 0.71, 0.80, 0.84, 0.59, 0.66, 0.74, 0.82, 0.87],
            "model": ["ita_only"] * 5 + ["ita_plus_covariates"] * 5,
        }
    )
    trend_success.to_csv(fairness_dir / "ita_trend_success.csv", index=False)

    iou_summary = {
        "base": {"status": "ok", "model": "quantile_regression_irls"},
        "covariate_adjusted": {"status": "ok", "covariates": ["lesion_area_frac", "deltaE_lesion_skin"]},
    }
    (fairness_dir / "ita_trend_iou_summary.json").write_text(json.dumps(iou_summary), encoding="utf-8")

    threshold = pd.DataFrame(
        {
            "threshold": [0.30, 0.40, 0.50, 0.60],
            "rd": [-0.08, -0.10, -0.09, -0.04],
            "rd_ci_lower": [-0.18, -0.21, -0.20, -0.14],
            "rd_ci_upper": [0.01, 0.00, 0.01, 0.06],
            "rr": [0.90, 0.86, 0.88, 0.95],
            "rr_ci_lower": [0.70, 0.63, 0.66, 0.76],
            "rr_ci_upper": [1.08, 1.01, 1.05, 1.11],
            "or": [0.72, 0.65, 0.69, 0.84],
            "or_ci_lower": [0.45, 0.38, 0.41, 0.54],
            "or_ci_upper": [1.09, 1.03, 1.07, 1.17],
        }
    )
    threshold.to_csv(fairness_dir / "threshold_sensitivity_table.csv", index=False)

    run_metadata = {
        "runtime_stage": "core",
        "dataset_counts": {"input_pairs": 6, "processed_rows": 6, "primary_rows": 5},
        "ita_cutoff": 28.0,
        "bootstrap": {"method": "percentile", "n_resamples": 1000},
        "trend_covariates_used": [
            "lesion_area_frac",
            "deltaE_lesion_skin",
            "Lstar_skin_mean",
            "Lstar_skin_std",
            "sharpness_laplacian_var",
            "hair_frac",
        ],
        "runtime": {
            "requested_workers": 8,
            "workers": 4,
            "max_inflight_tasks_effective": 4,
            "stage_wall_seconds": {"feature_extraction": 123.4, "trend_models": 1.2},
        },
        "checkpoint": {
            "checkpoint_every": 50,
            "processed_this_run": 6,
            "checkpoints_written": 1,
            "manifest_restarts": 0,
            "skipped_from_resume": 0,
            "shard_count": 1,
        },
        "config": {
            "runtime": {"stage": "core"},
            "trends": {"knots": 5, "degree": 3, "bootstrap_resamples": 50, "quantile": 0.5, "quantile_alpha": 0.001},
        },
    }
    (fairness_dir / "run_metadata.json").write_text(json.dumps(run_metadata), encoding="utf-8")

    if include_covadj:
        covadj_effects = pd.DataFrame(
            {
                "metric": [
                    "adjusted_risk_low",
                    "adjusted_risk_high",
                    "adjusted_rd_low_minus_high",
                    "adjusted_rr_low_over_high",
                    "adjusted_or_low_over_high",
                ],
                "estimate": [0.62, 0.74, -0.12, 0.84, 0.73],
                "ci_lower": [0.54, 0.68, -0.21, 0.72, 0.55],
                "ci_upper": [0.69, 0.79, -0.03, 0.95, 0.97],
                "ci_method": ["percentile_bootstrap"] * 5,
                "n": [5, 5, 5, 5, 5],
                "n_units": [5, 5, 5, 5, 5],
                "n_boot": [80, 80, 80, 80, 80],
                "resample_unit": ["dedup_group_id"] * 5,
            }
        )
        covadj_effects.to_csv(fairness_dir / "covadj_success_t050_effects.csv", index=False)

        covadj_payload = {
            "status": "ok",
            "summary": {
                "r_low_adj": 0.62,
                "r_high_adj": 0.74,
                "rd_adj": -0.12,
                "rr_adj": 0.84,
                "or_adj": 0.73,
                "rd_ci_lower": -0.21,
                "rd_ci_upper": -0.03,
                "n": 5,
                "n_units": 5,
                "n_boot": 80,
                "resample_unit": "dedup_group_id",
            },
        }
        (fairness_dir / "covadj_success_t050_effects.json").write_text(json.dumps(covadj_payload), encoding="utf-8")
        (fairness_dir / "covadj_model_spec.json").write_text(
            json.dumps({"type": "logistic_regression_predictive_margins", "covariates_used": ["lesion_area_frac", "deltaE_lesion_skin"]}),
            encoding="utf-8",
        )
        covadj_components = pd.DataFrame(
            {
                "component": [
                    "Lower ITA indicator (vs Higher ITA)",
                    "lesion_area_frac",
                    "deltaE_lesion_skin",
                ],
                "component_type": ["exposure", "covariate", "covariate"],
                "scale": [
                    "binary indicator",
                    "per 1 SD increase (standardized)",
                    "per 1 SD increase (standardized)",
                ],
                "coef_estimate": [-0.24, -0.09, 0.14],
                "coef_ci_lower": [-0.42, -0.21, 0.03],
                "coef_ci_upper": [-0.06, 0.02, 0.26],
                "coef_significant_95ci": ["yes", "no", "yes"],
                "or_estimate": [0.79, 0.91, 1.15],
                "or_ci_lower": [0.66, 0.81, 1.03],
                "or_ci_upper": [0.94, 1.02, 1.30],
                "or_significant_95ci": ["yes", "no", "yes"],
                "direction": ["lower success odds", "lower success odds", "higher success odds"],
                "bootstrap_p_two_sided": [0.008, 0.140, 0.020],
                "n_boot_feature": [80, 80, 80],
            }
        )
        covadj_components.to_csv(fairness_dir / "covadj_component_effects.csv", index=False)

    if include_optional:
        pd.DataFrame(
            {
                "dedup_mode": ["none", "exact", "near"],
                "median_iou_diff_lower_minus_higher__estimate": [-0.04, -0.03, -0.02],
                "mean_iou_diff_lower_minus_higher__estimate": [-0.03, -0.02, -0.01],
                "success_risk_difference_lower_minus_higher__estimate": [-0.11, -0.09, -0.06],
            }
        ).to_csv(fairness_dir / "dedup_sensitivity.csv", index=False)

        pd.DataFrame(
            {
                "mask_source": ["challenge_gt", "consensus_staple"],
                "n": [4, 1],
                "median_iou_diff": [-0.02, -0.07],
                "rd": [-0.08, -0.13],
            }
        ).to_csv(fairness_dir / "mask_source_sensitivity.csv", index=False)

        (fairness_dir / "dependence_sensitivity.json").write_text(
            json.dumps({"n_unique_dedup_groups": 5, "max_cluster_size": 2, "mean_cluster_size": 1.2}),
            encoding="utf-8",
        )

        pd.DataFrame(
            {
                "dataset_source_primary": ["isic2017", "ima_plusplus"],
                "total_images": [4, 1],
                "canonical_images": [4, 1],
                "collapsed_duplicates": [0, 0],
            }
        ).to_csv(fairness_dir / "dedup_report.csv", index=False)

        pd.DataFrame(
            {
                "ita_bin6": ["Light", "Intermediate", "Tan"],
                "n": [2, 2, 1],
                "mean_iou": [0.72, 0.58, 0.55],
                "median_iou": [0.71, 0.57, 0.55],
            }
        ).to_csv(fairness_dir / "ita_bins_table.csv", index=False)

        runtime_profile = {
            "status": "ok",
            "stages": {
                "source_index": {"throughput_items_per_sec": 120.5, "rss_peak_bytes": 800000000},
                "feature_extraction": {"throughput_items_per_sec": 6.4, "rss_peak_bytes": 1500000000},
            },
        }
        (fairness_dir / "runtime_profile.json").write_text(json.dumps(runtime_profile), encoding="utf-8")

        qc_dir = fairness_dir / "covariates_qc_plots"
        qc_dir.mkdir(parents=True, exist_ok=True)
        _write_qc_png(qc_dir / "lesion_area_frac_hist.png", 120)
        _write_qc_png(qc_dir / "deltaE_lesion_skin_hist.png", 150)

        _write_qc_png(fairness_dir / "ita_trend_iou_median.png", 100)

    return fairness_dir


def test_generate_fairness_enhanced_artifacts_core_only(tmp_path: Path) -> None:
    fairness_dir = _build_fixture(tmp_path, include_optional=False)

    bundle = generate_fairness_enhanced_artifacts(
        fairness_enhanced_dir=fairness_dir,
        output_dir=tmp_path / "artifacts",
        seed=7,
        include_supplement=True,
    )

    for key in ["E1", "E2", "E3", "E4", "E5"]:
        assert key in bundle.figures
        assert bundle.figures[key].png.exists()
        assert bundle.figures[key].svg.exists()
        assert bundle.figures[key].pdf.exists()

    for key in ["E1", "E2", "E3", "E4"]:
        assert key in bundle.tables
        assert bundle.tables[key].csv.exists()
        assert bundle.tables[key].html.exists()
        assert bundle.tables[key].markdown.exists()

    assert "ES1" in bundle.tables
    assert "ES4" in bundle.tables
    assert "ES1" not in bundle.figures

    assert bundle.report_markdown.exists()
    assert bundle.report_html.exists()
    assert bundle.report_docx.exists()
    assert bundle.report_pdf.exists()

    md = bundle.report_markdown.read_text(encoding="utf-8")
    assert "image-derived perilesional skin tone proxy" in md
    assert "Key Terms and Interpretation Guide" in md
    assert "Lesion-skin color contrast (Delta E)" in md
    assert "Image sharpness (variance of Laplacian)" in md


def test_generate_fairness_enhanced_artifacts_with_supplement(tmp_path: Path) -> None:
    fairness_dir = _build_fixture(tmp_path, include_optional=True)

    bundle = generate_fairness_enhanced_artifacts(
        fairness_enhanced_dir=fairness_dir,
        output_dir=tmp_path / "artifacts",
        seed=7,
        include_supplement=True,
    )

    for key in ["ES1", "ES2", "ES3"]:
        assert key in bundle.figures
        assert bundle.figures[key].png.exists()

    for key in ["ES1", "ES2", "ES3", "ES4"]:
        assert key in bundle.tables
        assert bundle.tables[key].csv.exists()


def test_generate_fairness_enhanced_artifacts_includes_covadj_when_available(tmp_path: Path) -> None:
    fairness_dir = _build_fixture(tmp_path, include_optional=False, include_covadj=True)

    bundle = generate_fairness_enhanced_artifacts(
        fairness_enhanced_dir=fairness_dir,
        output_dir=tmp_path / "artifacts",
        seed=3,
        include_supplement=False,
    )

    assert "E5" in bundle.tables
    assert "E6" in bundle.tables
    assert bundle.tables["E5"].csv.exists()
    assert bundle.tables["E6"].csv.exists()
    assert bundle.tables["E5"].html.exists()
    assert bundle.tables["E5"].markdown.exists()
    assert bundle.tables["E6"].html.exists()
    assert bundle.tables["E6"].markdown.exists()

    e5 = pd.read_csv(bundle.tables["E5"].csv)
    assert "n units" not in e5.columns
    assert "n boot" not in e5.columns
    assert "significant (95% ci)" in e5.columns

    e6 = pd.read_csv(bundle.tables["E6"].csv)
    assert "component" in e6.columns
    assert "odds ratio" in e6.columns
    assert "significant (95% ci)" in e6.columns

    md = bundle.report_markdown.read_text(encoding="utf-8")
    assert "Covariate-adjusted success RD (Lower - Higher)" in md
    assert "Table E5. Covariate-Adjusted Success Effects" in md
    assert "Table E6. Adjusted Model Component Contributions" in md


def test_generate_fairness_enhanced_artifacts_no_supplement_flag(tmp_path: Path) -> None:
    fairness_dir = _build_fixture(tmp_path, include_optional=True)

    bundle = generate_fairness_enhanced_artifacts(
        fairness_enhanced_dir=fairness_dir,
        output_dir=tmp_path / "artifacts",
        include_supplement=False,
    )

    assert all(not key.startswith("ES") for key in bundle.figures)
    assert all(not key.startswith("ES") for key in bundle.tables)


def test_generate_fairness_enhanced_artifacts_missing_required_file(tmp_path: Path) -> None:
    fairness_dir = _build_fixture(tmp_path, include_optional=False)
    (fairness_dir / "endpoint_effects.json").unlink()

    with pytest.raises(FileNotFoundError, match="endpoint_effects.json"):
        generate_fairness_enhanced_artifacts(
            fairness_enhanced_dir=fairness_dir,
            output_dir=tmp_path / "artifacts",
        )


def test_generate_fairness_enhanced_artifacts_stable_key_sets(tmp_path: Path) -> None:
    fairness_dir = _build_fixture(tmp_path, include_optional=True)

    first = generate_fairness_enhanced_artifacts(
        fairness_enhanced_dir=fairness_dir,
        output_dir=tmp_path / "a1",
        seed=11,
    )
    second = generate_fairness_enhanced_artifacts(
        fairness_enhanced_dir=fairness_dir,
        output_dir=tmp_path / "a2",
        seed=11,
    )

    assert set(first.figures.keys()) == set(second.figures.keys())
    assert set(first.tables.keys()) == set(second.tables.keys())
