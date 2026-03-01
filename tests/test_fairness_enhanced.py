import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
from PIL import Image

from gemini_segmentation.fairness_enhanced.config import (
    BootstrapConfig,
    CovariateConfig,
    DedupConfig,
    EnhancedFairnessConfig,
    FeaturesConfig,
    ITAConfig,
    RuntimeConfig,
    SensitivityConfig,
    SourceRootsConfig,
    TrendConfig,
)
from gemini_segmentation.fairness_enhanced.covadj import (
    compute_covariate_adjusted_success_effects,
)
from gemini_segmentation.fairness_enhanced.dedup import apply_exact_dedup, hamming_hex64, phash64_hex
from gemini_segmentation.fairness_enhanced.effects import compute_endpoint_effects
from gemini_segmentation.fairness_enhanced.ita import (
    compute_ita_features,
    compute_ita_legacy_like,
    ita_bin6,
    ita_binary,
    to_bool_mask,
)
from gemini_segmentation.fairness_enhanced.labels import build_label_text
import gemini_segmentation.fairness_enhanced.pipeline as pipeline_mod
from gemini_segmentation.fairness_enhanced.pipeline import run_enhanced_fairness_audit
from gemini_segmentation.fairness_enhanced.trends import build_iou_trend_frames, build_success_trend_frames


def _write_image(path: Path, value: int) -> None:
    arr = np.full((64, 64, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path: Path) -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 255
    Image.fromarray(mask, mode="L").save(path)


def test_label_text_enforces_proxy_language() -> None:
    payload = build_label_text(
        grouping_strategy="binary",
        cutoff=28.0,
        region_strategy="global_nonlesion",
    )
    assert "image-derived non-lesional skin tone proxy (ITA)" in payload.methods_snippet
    assert "lower-ITA" in payload.figure_caption_snippet


def test_default_ita_region_strategy_is_global_nonlesion() -> None:
    assert ITAConfig().region_strategy == "global_nonlesion"


def test_label_text_supports_perilesional_proxy_language() -> None:
    payload = build_label_text(
        grouping_strategy="binary",
        cutoff=28.0,
        region_strategy="perilesional_ring",
    )
    assert "image-derived perilesional skin tone proxy" in payload.methods_snippet
    assert "lower-ITA" in payload.figure_caption_snippet


def test_ita_bins_and_binary_mapping() -> None:
    cfg = ITAConfig()
    assert ita_bin6(60.0, cfg) == "Very Light"
    assert ita_bin6(-40.0, cfg) == "Dark"
    assert ita_binary(10.0, cutoff=28.0) == "Lower ITA"
    assert ita_binary(35.0, cutoff=28.0) == "Higher ITA"


def test_ita_uses_roi_and_no_full_frame_lab() -> None:
    image = np.full((240, 320, 3), 140, dtype=np.uint8)
    lesion = np.zeros((240, 320), dtype=np.uint8)
    lesion[90:150, 120:200] = 255
    cfg = ITAConfig(region_strategy="perilesional_ring")
    payload = compute_ita_features(
        image,
        lesion,
        ita_cfg=cfg,
        roi_max_dim=96,
        ring_outer_radius_cap=12,
        ring_inner_radius_cap=4,
        ring_roi_pad_px=20,
    )
    assert payload["lab"] is None
    assert max(payload["working_rgb"].shape[:2]) <= 96
    assert payload["ring_pixel_count"] > 0


def test_ita_ring_is_deterministic_with_radius_caps() -> None:
    image = np.full((300, 420, 3), 120, dtype=np.uint8)
    lesion = np.zeros((300, 420), dtype=np.uint8)
    lesion[100:200, 150:260] = 255
    cfg = ITAConfig(region_strategy="perilesional_ring")
    a = compute_ita_features(
        image,
        lesion,
        ita_cfg=cfg,
        roi_max_dim=128,
        ring_outer_radius_cap=10,
        ring_inner_radius_cap=3,
        ring_roi_pad_px=12,
    )
    b = compute_ita_features(
        image,
        lesion,
        ita_cfg=cfg,
        roi_max_dim=128,
        ring_outer_radius_cap=10,
        ring_inner_radius_cap=3,
        ring_roi_pad_px=12,
    )
    assert a["ring_pixel_count"] == b["ring_pixel_count"]
    assert np.array_equal(a["ring_mask"], b["ring_mask"])
    assert a["ring_pixel_count"] < int(a["ring_mask"].size)


def test_ita_inner_frac_zero_does_not_collapse_ring() -> None:
    image = np.full((96, 96, 3), 130, dtype=np.uint8)
    lesion = np.zeros((96, 96), dtype=np.uint8)
    lesion[28:68, 28:68] = 255
    cfg = ITAConfig(
        region_strategy="perilesional_ring",
        ring_outer_frac_min_dim=0.02,
        ring_inner_frac_min_dim=0.0,
    )
    payload = compute_ita_features(
        image,
        lesion,
        ita_cfg=cfg,
        roi_max_dim=96,
        ring_outer_radius_cap=8,
        ring_inner_radius_cap=4,
        ring_roi_pad_px=8,
    )
    assert int(payload["ring_pixel_count"]) > 0


def test_to_bool_mask_supports_unit_scale_binary_masks() -> None:
    mask_u8 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    mask_f32 = mask_u8.astype(np.float32)
    out_u8 = to_bool_mask(mask_u8)
    out_f32 = to_bool_mask(mask_f32)
    assert int(out_u8.sum()) == 2
    assert int(out_f32.sum()) == 2


def test_ita_global_nonlesion_pixelwise_matches_legacy_like_helper() -> None:
    image = np.full((96, 96, 3), 140, dtype=np.uint8)
    lesion = np.zeros((96, 96), dtype=np.uint8)
    lesion[28:68, 28:68] = 255
    cfg = ITAConfig(
        region_strategy="global_nonlesion",
        estimator="pixelwise_median",
        aggregation_stat="median",
        apply_lstar_window=True,
        lstar_window_low_pct=5.0,
        lstar_window_high_pct=95.0,
        use_field_mask=False,
    )
    payload = compute_ita_features(
        image,
        lesion,
        ita_cfg=cfg,
        roi_max_dim=96,
        ring_outer_radius_cap=12,
        ring_inner_radius_cap=4,
        ring_roi_pad_px=8,
    )
    legacy_like, candidate_count = compute_ita_legacy_like(
        image_rgb=image,
        lesion_mask=lesion,
        eps=cfg.eps,
        use_field_mask=cfg.use_field_mask,
        field_intensity_floor=cfg.field_intensity_floor,
        min_pixels=cfg.ring_min_pixels,
        low_pct=cfg.lstar_window_low_pct,
        high_pct=cfg.lstar_window_high_pct,
    )
    assert abs(float(payload["ita_deg"]) - float(legacy_like)) < 1e-9
    assert int(payload["ita_candidate_count"]) == int(candidate_count)


def test_exact_dedup_canonical_selection_prefers_mask_source_then_split() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_id": "A",
                "image_name": "a.png",
                "sha256": "x",
                "mask_source": "challenge_gt",
                "split": "train",
                "image_height": 100,
                "image_width": 100,
                "dataset_source_primary": "isic2017",
            },
            {
                "image_id": "B",
                "image_name": "b.png",
                "sha256": "x",
                "mask_source": "consensus_staple",
                "split": "val",
                "image_height": 90,
                "image_width": 90,
                "dataset_source_primary": "ima_plusplus",
            },
        ]
    )
    exact_df, dedup_map, _ = apply_exact_dedup(frame)
    canonical = dedup_map.iloc[0]["canonical_image_name"]
    assert canonical == "b.png"
    assert int(exact_df["is_canonical"].sum()) == 1


def test_phash_hex_and_hamming_distance() -> None:
    img_a = np.full((32, 32), 40, dtype=np.uint8)
    img_b = np.full((32, 32), 40, dtype=np.uint8)
    hash_a = phash64_hex(img_a)
    hash_b = phash64_hex(img_b)
    assert len(hash_a) == 16
    assert hamming_hex64(hash_a, hash_b) == 0


def test_endpoint_effects_table_contains_required_metrics() -> None:
    df = pd.DataFrame(
        {
            "ita_binary": ["Lower ITA"] * 6 + ["Higher ITA"] * 6,
            "iou": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            "success_t050": [False, False, False, False, True, True, True, True, True, True, True, True],
        }
    )
    table, payload, _ = compute_endpoint_effects(
        df,
        n_resamples=200,
        bootstrap_method="percentile",
        bootstrap_fallback_method="percentile",
        seed=7,
    )
    metrics = set(table["metric"].tolist())
    assert "median_iou_diff_lower_minus_higher" in metrics
    assert "success_risk_difference_lower_minus_higher" in metrics
    assert "tests" in payload


def test_covadj_success_effects_outputs_required_metrics() -> None:
    rng = np.random.default_rng(7)
    n = 240
    ita = np.where(np.arange(n) < (n // 2), "Lower ITA", "Higher ITA")
    lesion_area = rng.uniform(0.01, 0.20, size=n)
    deltae = rng.uniform(6.0, 30.0, size=n)
    sharp = rng.uniform(10.0, 150.0, size=n)
    hair = rng.uniform(0.01, 0.15, size=n)

    logit = (
        -0.35
        + 0.8 * (ita == "Higher ITA").astype(float)
        - 1.8 * lesion_area
        + 0.025 * deltae
        + 0.002 * sharp
        - 1.2 * hair
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    success = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "ita_binary": ita,
            "success_t050": success,
            "lesion_area_frac": lesion_area,
            "deltaE_lesion_skin": deltae,
            "sharpness_laplacian_var": sharp,
            "hair_frac": hair,
            "dedup_group_id": [f"g{i:03d}" for i in range(n)],
            "dataset_source_primary": np.where(np.arange(n) % 2 == 0, "isic2017", "interop4074"),
        }
    )

    table, payload, boot = compute_covariate_adjusted_success_effects(
        df,
        covariate_cols=[
            "lesion_area_frac",
            "deltaE_lesion_skin",
            "sharpness_laplacian_var",
            "hair_frac",
        ],
        include_dataset_source=True,
        n_resamples=80,
        seed=13,
    )

    metrics = set(table["metric"].tolist())
    assert "adjusted_rd_low_minus_high" in metrics
    assert "adjusted_rr_low_over_high" in metrics
    assert "adjusted_or_low_over_high" in metrics
    assert payload["status"] == "ok"
    assert "model_spec" in payload
    assert "component_effects" in payload
    comp = pd.DataFrame(payload["component_effects"])
    assert not comp.empty
    assert "component" in comp.columns
    assert "or_significant_95ci" in comp.columns
    assert not boot.empty


def test_covadj_single_class_is_handled() -> None:
    df = pd.DataFrame(
        {
            "ita_binary": ["Lower ITA", "Higher ITA", "Lower ITA", "Higher ITA"],
            "success_t050": [0, 0, 0, 0],
            "lesion_area_frac": [0.05, 0.06, 0.07, 0.08],
            "dedup_group_id": ["a", "b", "c", "d"],
        }
    )
    table, payload, boot = compute_covariate_adjusted_success_effects(
        df,
        covariate_cols=["lesion_area_frac"],
        n_resamples=20,
        seed=3,
    )
    row = table[table["metric"] == "adjusted_rd_low_minus_high"].iloc[0]
    assert float(row["estimate"]) == 0.0
    assert payload["model_spec"]["fit"]["fit_status"] in {"constant_class", "ok"}
    assert not boot.empty


def test_trend_builders_return_non_empty_frames() -> None:
    df = pd.DataFrame(
        {
            "ita_deg": np.linspace(-20, 60, 40),
            "iou": np.linspace(0.2, 0.9, 40),
            "success_t050": [int(x >= 0.5) for x in np.linspace(0.2, 0.9, 40)],
            "lesion_area_frac": np.linspace(0.01, 0.2, 40),
            "deltaE_lesion_skin": np.linspace(5, 30, 40),
        }
    )
    s_df = build_success_trend_frames(
        df=df,
        ita_col="ita_deg",
        outcome_col="success_t050",
        covariate_cols=["lesion_area_frac", "deltaE_lesion_skin"],
        knots=4,
        degree=3,
        n_bootstrap=20,
        seed=1,
    )
    i_df, i_payload = build_iou_trend_frames(
        df=df,
        ita_col="ita_deg",
        iou_col="iou",
        covariate_cols=["lesion_area_frac", "deltaE_lesion_skin"],
        knots=4,
        degree=3,
        quantile=0.5,
        alpha=0.001,
        n_bootstrap=20,
        seed=2,
    )
    assert not s_df.empty
    assert not i_df.empty
    assert "base" in i_payload


def test_trend_covariate_selection_excludes_non_numeric_method_labels() -> None:
    df = pd.DataFrame(
        {
            "lesion_area_frac": [0.1, 0.2],
            "deltaE_lesion_skin": [5.0, 6.0],
            "deltaE_method": ["ciede2000", "ciede2000"],
            "sharpness_laplacian_var": [10.0, 11.0],
        }
    )
    cov_cols = pipeline_mod._trend_covariate_columns(df)
    assert "deltaE_method" not in cov_cols
    assert "lesion_area_frac" in cov_cols
    assert "deltaE_lesion_skin" in cov_cols
    assert "sharpness_laplacian_var" in cov_cols


def test_success_trend_single_class_returns_constant_predictions() -> None:
    df = pd.DataFrame(
        {
            "ita_deg": np.linspace(-20, 60, 12),
            "success_t050": np.zeros(12, dtype=int),
            "lesion_area_frac": np.linspace(0.02, 0.08, 12),
        }
    )
    trend_df = build_success_trend_frames(
        df=df,
        ita_col="ita_deg",
        outcome_col="success_t050",
        covariate_cols=["lesion_area_frac"],
        knots=4,
        degree=3,
        n_bootstrap=8,
        seed=11,
    )
    assert not trend_df.empty
    assert trend_df["pred"].notna().all()
    assert np.allclose(trend_df["pred"].to_numpy(dtype=float), 0.0)


def test_build_analysis_row_uses_full_gt_for_lesion_area_fraction(tmp_path: Path) -> None:
    img_path = tmp_path / "images" / "ISIC_AREA.png"
    mask_path = tmp_path / "masks" / "ISIC_AREA.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((400, 400, 3), 100, dtype=np.uint8)
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[120:160, 120:160] = 255
    Image.fromarray(img, mode="RGB").save(img_path)
    Image.fromarray(mask, mode="L").save(mask_path)

    cfg = _base_cfg(tmp_path, stage="core", checkpoint_every=1, max_inflight_tasks=1)
    cfg.features.ring_roi_pad_px = 0
    cfg.features.roi_max_dim = 400

    feature_plan = pipeline_mod._feature_plan(cfg, stage="core")
    _, row, _ = pipeline_mod._build_analysis_row(
        idx=0,
        img_path=img_path,
        gt_path=mask_path,
        per_image_metrics={img_path.name: (0.8, 0.9, True)},
        by_sha={},
        by_id={},
        cfg=cfg,
        model_name="m",
        prompt_variant="p",
        run_id="r",
        feature_plan=feature_plan,
    )

    assert row is not None
    expected = float((40 * 40) / float(400 * 400))
    assert abs(float(row["lesion_area_frac"]) - expected) < 1e-9


def test_build_analysis_row_populates_legacy_like_ita_when_enabled(tmp_path: Path) -> None:
    img_path = tmp_path / "images" / "ISIC_LEGACY_ITA.png"
    mask_path = tmp_path / "masks" / "ISIC_LEGACY_ITA.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((128, 128, 3), 120, dtype=np.uint8)
    img[:, :64, :] = 95
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:88, 40:88] = 255
    Image.fromarray(img, mode="RGB").save(img_path)
    Image.fromarray(mask, mode="L").save(mask_path)

    cfg = _base_cfg(tmp_path, stage="core", checkpoint_every=1, max_inflight_tasks=1)
    cfg.ita.include_legacy_like_sensitivity = True

    feature_plan = pipeline_mod._feature_plan(cfg, stage="core")
    _, row, _ = pipeline_mod._build_analysis_row(
        idx=0,
        img_path=img_path,
        gt_path=mask_path,
        per_image_metrics={img_path.name: (0.8, 0.9, True)},
        by_sha={},
        by_id={},
        cfg=cfg,
        model_name="m",
        prompt_variant="p",
        run_id="r",
        feature_plan=feature_plan,
    )

    assert row is not None
    assert np.isfinite(float(row["ita_deg"]))
    assert np.isfinite(float(row["ita_deg_legacy_like"]))
    assert int(row["ita_legacy_candidate_count"]) > 0
    assert row["ita_binary_legacy28"] in {"Lower ITA", "Higher ITA", "Unknown"}
    assert row["ita_method_region"] in {"perilesional_ring", "global_nonlesion", "unknown"}


def test_pipeline_writes_required_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    img_a = img_dir / "ISIC_A.jpg"
    img_b = img_dir / "ISIC_B.jpg"
    mask_a = mask_dir / "ISIC_A.jpg"
    mask_b = mask_dir / "ISIC_B.jpg"

    _write_image(img_a, 180)
    _write_image(img_b, 35)
    _write_mask(mask_a)
    _write_mask(mask_b)

    pairs = [(img_a, mask_a), (img_b, mask_b)]
    per_metrics = {
        "ISIC_A.jpg": (0.82, 0.90, True),
        "ISIC_B.jpg": (0.41, 0.58, False),
    }

    cfg = EnhancedFairnessConfig(
        sources=SourceRootsConfig(
            interop_root=tmp_path / "missing_interop",
            isic2016_root=tmp_path / "missing16",
            isic2017_root=tmp_path / "missing17",
            isic2018_root=tmp_path / "missing18",
            ima_plusplus_root=tmp_path / "missing_ima",
        ),
        dedup=DedupConfig(mode="exact", near_hamming_threshold=6, include_near_map=True),
        ita=ITAConfig(binary_cutoff=28.0, binary_strategy="fixed"),
        covariates=CovariateConfig(enabled=True),
        bootstrap=BootstrapConfig(n_resamples=200, method="percentile", fallback_method="percentile", seed=3),
        trends=TrendConfig(enabled=True, knots=4, degree=3, bootstrap_resamples=20, quantile=0.5, quantile_alpha=0.001),
        sensitivity=SensitivityConfig(enabled=True, success_thresholds=[0.3, 0.5], include_near_dedup=True),
        runtime=RuntimeConfig(stage="all", resume=True, checkpoint_every=1, max_inflight_tasks=2),
        allow_image_id_fallback=False,
        duplicate_examples_limit=2,
        refresh_source_index=True,
    )

    run_payload = run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "run-test"},
        cfg=cfg,
        success_threshold=0.5,
    )

    out_dir = Path(run_payload["out_dir"])
    assert (out_dir / "analysis_frame.parquet").exists()
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "fingerprints.parquet").exists()
    assert (out_dir / "endpoint_effects_table.csv").exists()
    assert (out_dir / "endpoint_effects.json").exists()
    assert (out_dir / "ita_bins_table.csv").exists()
    assert (out_dir / "ita_bins_plot.png").exists()
    assert (out_dir / "ita_trend_success.csv").exists()
    assert (out_dir / "ita_trend_success.png").exists()
    assert (out_dir / "ita_trend_iou_median.png").exists()
    assert (out_dir / "covadj_success_t050_effects.csv").exists()
    assert (out_dir / "covadj_success_t050_effects.json").exists()
    assert (out_dir / "covadj_model_spec.json").exists()
    assert (out_dir / "covadj_component_effects.csv").exists()
    assert (out_dir / "covadj_component_effects.json").exists()
    assert (out_dir / "covadj_success_t050_bootstrap_samples.parquet").exists()
    assert (out_dir / "threshold_sensitivity_table.csv").exists()
    assert (out_dir / "threshold_sensitivity.png").exists()
    assert (out_dir / "runtime_profile.json").exists()
    assert (out_dir / "ita_method_note.json").exists()
    assert (out_dir / "ita_method_note.md").exists()
    assert (out_dir / "dedup_sensitivity.csv").exists()
    assert (out_dir / "cache" / "metrics.parquet").exists()
    assert (out_dir / "cache" / "ita_features.parquet").exists()
    assert (out_dir / "cache" / "covariates.parquet").exists()
    assert (out_dir / "cache" / "fingerprints.parquet").exists()

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["ita_cutoff"] == 28.0
    assert metadata["runtime_stage"] == "all"
    assert metadata["checkpoint"]["checkpoint_every"] == 1


def _prepare_fixture_case(tmp_path: Path, n_images: int = 4):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    pairs = []
    per_metrics = {}
    for idx in range(n_images):
        name = f"ISIC_{idx:03d}.jpg"
        img_path = img_dir / name
        mask_path = mask_dir / name
        _write_image(img_path, 40 + idx * 20)
        _write_mask(mask_path)
        pairs.append((img_path, mask_path))
        iou = 0.30 + (idx * 0.1)
        per_metrics[name] = (iou, min(1.0, iou + 0.1), bool(iou >= 0.5))
    return run_dir, pairs, per_metrics


def _base_cfg(tmp_path: Path, *, stage: str, checkpoint_every: int = 2, max_inflight_tasks: int = 0) -> EnhancedFairnessConfig:
    return EnhancedFairnessConfig(
        sources=SourceRootsConfig(
            interop_root=tmp_path / "missing_interop",
            isic2016_root=tmp_path / "missing16",
            isic2017_root=tmp_path / "missing17",
            isic2018_root=tmp_path / "missing18",
            ima_plusplus_root=tmp_path / "missing_ima",
        ),
        dedup=DedupConfig(mode="exact", near_hamming_threshold=6, include_near_map=True),
        ita=ITAConfig(binary_cutoff=28.0, binary_strategy="fixed"),
        covariates=CovariateConfig(enabled=True),
        features=FeaturesConfig(
            profile="balanced",
            compute_phash_in_core=False,
            hair_mode="lite",
            include_specular_in_core=False,
            roi_max_dim=128,
            hair_max_dim=96,
            ring_outer_radius_cap=12,
            ring_inner_radius_cap=4,
            ring_roi_pad_px=12,
        ),
        bootstrap=BootstrapConfig(n_resamples=200, method="percentile", fallback_method="percentile", seed=5),
        trends=TrendConfig(enabled=True, knots=4, degree=3, bootstrap_resamples=10, quantile=0.5, quantile_alpha=0.001),
        sensitivity=SensitivityConfig(enabled=True, success_thresholds=[0.3, 0.5], include_near_dedup=True),
        runtime=RuntimeConfig(stage=stage, resume=True, checkpoint_every=checkpoint_every, max_inflight_tasks=max_inflight_tasks),
        allow_image_id_fallback=False,
        duplicate_examples_limit=1,
        refresh_source_index=True,
    )


def test_pipeline_resume_skips_processed_rows(tmp_path: Path) -> None:
    run_dir, pairs, per_metrics = _prepare_fixture_case(tmp_path, n_images=3)
    cfg = _base_cfg(tmp_path, stage="core", checkpoint_every=1, max_inflight_tasks=2)

    run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "run-r1"},
        cfg=cfg,
        success_threshold=0.5,
        workers=2,
    )
    run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "run-r2"},
        cfg=cfg,
        success_threshold=0.5,
        workers=2,
    )

    out_dir = run_dir / "fairness_enhanced"
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    checkpoint = metadata["checkpoint"]
    assert checkpoint["skipped_from_resume"] == 3
    assert checkpoint["processed_this_run"] == 0
    assert checkpoint["shard_count"] >= 1


def test_pipeline_core_then_sensitivity_stage(tmp_path: Path) -> None:
    run_dir, pairs, per_metrics = _prepare_fixture_case(tmp_path, n_images=4)

    cfg_core = _base_cfg(tmp_path, stage="core", checkpoint_every=2, max_inflight_tasks=2)
    run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "core"},
        cfg=cfg_core,
        success_threshold=0.5,
        workers=2,
    )
    out_dir = run_dir / "fairness_enhanced"
    assert (out_dir / "analysis_frame.parquet").exists()
    assert not (out_dir / "dedup_sensitivity.csv").exists()

    cfg_sens = _base_cfg(tmp_path, stage="sensitivity", checkpoint_every=2, max_inflight_tasks=2)
    run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "sens"},
        cfg=cfg_sens,
        success_threshold=0.5,
        workers=2,
    )
    assert (out_dir / "dedup_sensitivity.csv").exists()
    frame = pd.read_parquet(out_dir / "analysis_frame.parquet")
    assert "phash64_hex" in frame.columns
    assert not frame["phash64_hex"].isna().all()
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["runtime_stage"] == "sensitivity"


def test_pipeline_augment_stage_fills_requested_columns(tmp_path: Path) -> None:
    run_dir, pairs, per_metrics = _prepare_fixture_case(tmp_path, n_images=4)
    cfg_core = _base_cfg(tmp_path, stage="core", checkpoint_every=2, max_inflight_tasks=2)
    run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "core"},
        cfg=cfg_core,
        success_threshold=0.5,
        workers=2,
    )
    out_dir = run_dir / "fairness_enhanced"
    pre = pd.read_parquet(out_dir / "analysis_frame.parquet")
    assert pre["phash64_hex"].isna().all()

    cfg_aug = _base_cfg(tmp_path, stage="augment", checkpoint_every=2, max_inflight_tasks=2)
    run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "aug"},
        cfg=cfg_aug,
        success_threshold=0.5,
        workers=2,
        augment_columns=["phash64_hex", "specular_frac"],
    )
    post = pd.read_parquet(out_dir / "analysis_frame.parquet")
    assert not post["phash64_hex"].isna().all()
    assert not post["specular_frac"].isna().all()
    assert pre["iou"].tolist() == post["iou"].tolist()


def test_pipeline_minimal_profile_keeps_covariate_columns_nan(tmp_path: Path) -> None:
    run_dir, pairs, per_metrics = _prepare_fixture_case(tmp_path, n_images=3)
    cfg = _base_cfg(tmp_path, stage="core", checkpoint_every=1, max_inflight_tasks=2)
    cfg.features.profile = "minimal"

    run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "minimal"},
        cfg=cfg,
        success_threshold=0.5,
        workers=2,
    )

    frame = pd.read_parquet(run_dir / "fairness_enhanced" / "analysis_frame.parquet")
    assert frame["phash64_hex"].isna().all()
    assert frame["deltaE_lesion_skin"].isna().all()
    assert frame["sharpness_laplacian_var"].isna().all()
    assert frame["hair_frac"].isna().all()
    assert frame["specular_frac"].isna().all()


def test_bounded_inflight_scheduler_respects_limit(tmp_path: Path, monkeypatch) -> None:
    run_dir, pairs, per_metrics = _prepare_fixture_case(tmp_path, n_images=6)
    cfg = _base_cfg(tmp_path, stage="core", checkpoint_every=2, max_inflight_tasks=2)

    lock = threading.Lock()
    active = {"value": 0, "max": 0}
    original = pipeline_mod._build_analysis_row

    def wrapped_build_analysis_row(**kwargs):
        with lock:
            active["value"] += 1
            active["max"] = max(active["max"], active["value"])
        try:
            time.sleep(0.02)
            return original(**kwargs)
        finally:
            with lock:
                active["value"] -= 1

    monkeypatch.setattr(pipeline_mod, "_build_analysis_row", wrapped_build_analysis_row)

    run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "bounded"},
        cfg=cfg,
        success_threshold=0.5,
        workers=4,
    )

    assert active["max"] <= 2


def test_workers_auto_caps_effective_workers(tmp_path: Path, monkeypatch) -> None:
    run_dir, pairs, per_metrics = _prepare_fixture_case(tmp_path, n_images=2)
    cfg = _base_cfg(tmp_path, stage="core", checkpoint_every=1, max_inflight_tasks=0)
    cfg.runtime.workers_auto = True
    cfg.runtime.memory_target_frac = 0.5
    cfg.runtime.per_worker_estimate_mb_balanced = 512

    monkeypatch.setattr(
        pipeline_mod,
        "_available_memory_bytes",
        lambda: int(1024 * 1024 * 1024),  # 1 GiB
    )

    run_enhanced_fairness_audit(
        image_mask_pairs=pairs,
        per_image_metrics=per_metrics,
        run_dir=run_dir,
        run_config={"model_name": "model-x", "prompt_family": "label_v1", "run_id": "auto-cap"},
        cfg=cfg,
        success_threshold=0.5,
        workers=10,
    )

    metadata = json.loads((run_dir / "fairness_enhanced" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["runtime"]["requested_workers"] == 10
    assert metadata["runtime"]["workers"] == 1
