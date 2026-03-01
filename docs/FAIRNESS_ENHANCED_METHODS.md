# Enhanced Fairness Audit Methods (Implementation-Aligned)

This document is the implementation-aligned methods specification for the enhanced dermoscopy fairness audit (`fairness --audit-mode enhanced`).

Version anchor:
- Code path: `src/gemini_segmentation/fairness_enhanced/`
- Effective defaults: `configs/fairness_enhanced.yaml`
- Current method family: `posthoc_v13` (see `docs/METHODS_CHANGELOG.md`)

This file is intended to be both:
- engineering-reference exactness (what the code does), and
- manuscript-ready methods source text (what to report).

## 1) Execution Scope and Stages

Enhanced fairness supports staged execution:
- `all`: full pipeline (`source_index`, feature extraction, dedup, effects, trends, sensitivities).
- `core`: faster first pass with constrained defaults.
- `sensitivity`: runs sensitivity analyses on existing core artifacts.
- `augment`: computes only missing/explicitly requested columns and merges into `analysis_frame.parquet`.

Stage-specific overrides in `core`:
- `dedup.mode = exact`
- `dedup.include_near_map = false`
- `sensitivity.include_near_dedup = false`
- `sensitivity.include_dependence = false`
- `sensitivity.include_mask_source = false`
- `bootstrap.method = percentile`
- `bootstrap.n_resamples = 1000`
- `trends.bootstrap_resamples = 50`

These overrides are applied at runtime only for the `core` stage and are written to `run_metadata.json` under `runtime_stage_overrides`.

## 2) Canonical Analysis Unit and Table

Primary analytic artifact:
- `<run_dir>/fairness_enhanced/analysis_frame.parquet`

Canonical row unit:
- one row per image per model/run after extraction, with canonical-selection flags added after dedup.

Key column families:
- identifiers/provenance: `image_id`, `image_name`, `dataset_source_primary`, `dataset_source_memberships_json`, `split`, `mask_source`, `sha256`, `phash64_hex`, `dedup_group_id`.
- outcomes: `iou`, `dice`, `success_t050`.
- ITA and grouping: `ita_deg`, `ita_bin6`, `ita_binary`.
- ITA method/QC: `ita_method_region`, `ita_method_estimator`, `ita_method_aggregation`, `ita_candidate_count`, `ring_pixel_count`, `ring_area_frac`, `ring_valid`.
- covariates: `lesion_area_frac`, `deltaE_lesion_skin`, `deltaE_method`, `Lstar_skin_mean`, `Lstar_skin_std`, `sharpness_laplacian_var`, `hair_frac`, `specular_frac`.
- reproducibility: `model_name`, `prompt_variant`, `run_id`, `audit_mode`.

## 3) Source Index and Provenance Resolution

Source index cache:
- `<run_dir>/fairness_enhanced/cache/source_index.parquet`

Source roots:
- interop 4074, ISIC 2016/2017/2018, IMA++ (from config).

Matching and metadata policy:
- source membership and primary source assignment are SHA-based (`sha256`), not image-ID-based.
- optional image-ID fallback is off by default (`allow_image_id_fallback: false`).
- ISIC 2017 split labels are loaded from metadata CSVs.
- IMA++ splits are loaded from `train/val/test_ima_plusplus.txt`.
- IMA++ mask provenance is loaded from `metadata/ima_plusplus_index.csv` (`gt_policy`).

## 4) Deduplication

Exact dedup (`dedup.mode=exact`):
- group by `sha256`.
- canonical member tie-break order:
1. mask provenance rank: `consensus_staple > consensus_mv > challenge_gt > single_annotator_only > assisted > unknown`
2. split rank: `test > val > train > unknown`
3. higher resolution (`image_height * image_width`)
4. lexical `image_id`

Near dedup (`dedup.mode=near` or sensitivity near-dedup):
- pHash (`phash64_hex`) computed from grayscale image resized to 32x32.
- 2D orthonormal DCT (NumPy implementation), 8x8 low-frequency block, median-threshold bit encoding to 64-bit hash.
- cluster edges defined by Hamming distance <= configured threshold.

Outputs:
- `dedup_map_exact.csv`, `dedup_report.csv`, optional `dedup_map_near.csv`.

## 5) Mask Handling and Metric Inputs

Ground-truth mask binarization (`to_bool_mask`):
- if mask max <= 1.0: threshold at `>0.5`.
- else threshold at `>127`.
- supports boolean, unit-scale, and 8-bit masks.

Outcome endpoint fields:
- `iou` and `dice` are consumed from upstream segmentation metrics.
- `success_t050` is recomputed in enhanced as `iou >= 0.50` (fixed primary usability endpoint).

## 6) ITA Computation

### 6.1 Default enhanced ITA definition

Current default:
- region strategy: `global_nonlesion`
- estimator: `aggregated_lab`
- aggregation statistic: `median`

Meaning:
- non-lesional pixels are selected as logical complement of GT lesion mask.
- optional field mask is applied (`use_field_mask=true` by default).
- representative L* and b* are aggregated over selected pixels.
- ITA is computed from aggregated values (not per-pixel ITA median).

### 6.2 Field mask (when enabled)

Field mask is generated from RGB image:
- grayscale proxy = channel mean.
- threshold `gray > field_intensity_floor`.
- fill holes and apply binary opening.

### 6.3 Optional perilesional-ring strategy

If `region_strategy=perilesional_ring`:
- lesion-centered ROI bbox with configurable padding.
- ROI downsampled to `roi_max_dim`.
- ring defined by outer minus inner binary dilations with radius caps.
- field mask applied after ring construction.

### 6.4 ITA formula

`ITA = arctan((L* - 50) / (b* + eps)) * (180/pi)`

### 6.5 Aggregation modes

`aggregated_lab` supports:
- `median`
- `mean`
- `trimmed_mean_sd` (joint L* and b* trimming by ±`trim_std` SD)

Optional luminance window:
- if enabled, L* percentile filtering is applied to candidate region (`lstar_window_low_pct`, `lstar_window_high_pct`) before ITA aggregation.

Alternative estimator:
- `pixelwise_median`: compute per-pixel ITA then take median.

### 6.6 ITA grouping

Binary grouping:
- default fixed cutoff (`binary_strategy=fixed`, default 28.0°).
- optional `binary_strategy=median` uses sample median ITA for cutoff.

6-bin mapping:
- Very Light > 55
- Light: 41 < ITA <= 55
- Intermediate: 28 < ITA <= 41
- Tan: 10 < ITA <= 28
- Brown: -30 < ITA <= 10
- Dark: ITA <= -30

### 6.7 Legacy-like sensitivity ITA (optional)

If `ita.include_legacy_like_sensitivity=true`, additional columns are written:
- `ita_deg_legacy_like`, `ita_binary_legacy28`, `ita_legacy_candidate_count`, `ita_delta_vs_legacy`.

Legacy-like comparator definition:
- global non-lesion region,
- optional field mask,
- L* percentile windowing,
- ITA = median of per-pixel ITA values.

## 7) Covariates

Covariates are profile-gated and computed in enhanced when enabled.

`lesion_area_frac`:
- computed from full-resolution GT lesion mask and image size.

`deltaE_lesion_skin`:
- between lesion Lab median and skin-region Lab median.
- method:
  - default in `balanced` profile: `deltae76` (Euclidean Lab distance).
  - configurable/full profile: `ciede2000`.

`Lstar_skin_mean`, `Lstar_skin_std`:
- derived from selected skin region.

`sharpness_laplacian_var`:
- variance of Laplacian on grayscale image (after texture-space resize).

`hair_frac`:
- morphological black-hat response with directional kernels.
- `lite` mode uses 4 kernels; `full` mode uses 8 kernels.
- threshold at configured quantile of valid-mask response.

`specular_frac`:
- HSV proxy thresholding (`value >= specular_lstar_cutoff` and `saturation <= specular_chroma_cutoff`) over valid mask.

Valid-mask basis for hair/specular:
- `field`, `ring`, or `lesion_ring` (`covariates.valid_pixels_base`).

## 8) Statistical Outputs

Primary endpoints:
- quality: `iou` (continuous)
- usability: `success_t050` (binary)

Effect-size outputs (`endpoint_effects_table.csv`):
- `cliffs_delta_iou_lower_vs_higher`
- `median_iou_diff_lower_minus_higher`
- `mean_iou_diff_lower_minus_higher`
- `success_risk_difference_lower_minus_higher`
- `success_relative_risk_lower_over_higher`
- `success_odds_ratio_lower_over_higher`

Bootstrap CI behavior:
- method: configured (`bca` or `percentile`), with fallback method.
- BCa implemented with `scipy.stats.bootstrap(..., method="BCa")`.
- if BCa fails or degenerates, fallback percentile CI is used and warning recorded.

Secondary tests:
- Mann-Whitney U for IoU groups.
- chi-square for success tables; Fisher exact is additionally reported when expected counts are small.

Threshold sensitivity:
- recompute RD/RR/OR over configured IoU thresholds and write table/plot artifacts.

## 9) Trend Models

Success trend (`ita_trend_success.csv`, `.png`):
- spline basis via `SplineTransformer`.
- logistic regression (`liblinear`).
- ITA-only and ITA+covariate curves (if numeric covariates available).
- bootstrap confidence bands from resampled refits.
- single-class guard: if resample has only one class, constant-probability curve is emitted.

IoU trend (`ita_trend_iou_median.png`, `ita_trend_iou_summary.json`):
- spline basis + IRLS-style quantile regression (q=0.5 by default).
- avoids SciPy HiGHS dependency for stability.
- ITA-only and ITA+covariate curves with bootstrap bands.

## 10) Feature Profiles and Gating

Profiles:
- `minimal`: ITA + core identifiers/metrics; covariates disabled.
- `balanced` (default): ITA + key covariates (`deltaE`, `L*`, sharpness, hair), specular optional.
- `full`: balanced plus full configured `deltaE` method and optional extras.

pHash in core:
- computed only if requested by profile/config or required by near-dedup logic.

## 11) Runtime, Concurrency, and Resume

Extraction execution:
- thread pool with bounded in-flight tasks.
- if `runtime.max_inflight_tasks <= 0`, effective in-flight defaults to `workers`.

Worker auto-cap:
- optional memory-aware cap (`runtime.workers_auto`).
- effective worker cap is computed from available memory, target fraction, and profile-specific per-worker estimate.

Checkpointing:
- row shards written to `cache/features_part_*.parquet`.
- manifest in `cache/features_manifest.json` tracks processed image names and restart counts.
- resume skips processed image names when enabled.

Runtime telemetry:
- `runtime_profile.json` stores per-stage wall/cpu durations, throughput, and RSS peaks (when available).

## 12) Required Reproducibility Artifacts

Core reproducibility artifacts:
- `run_metadata.json`
- `runtime_profile.json`
- `analysis_frame.parquet`
- `ita_method_note.json`
- `ita_method_note.md`

`run_metadata.json` includes:
- effective config snapshot,
- stage overrides,
- worker summary,
- dataset counts,
- ITA cutoff and ITA method payload,
- bootstrap settings,
- trend covariates used,
- warnings list.

## 13) Manuscript-Ready Methods Text (Template)

Use/adapt this text directly in manuscript methods:

"We performed fairness auditing on dermoscopy segmentation outputs using an image-derived skin-tone proxy based on the Individual Typology Angle (ITA). For each image, lesion masks were binarized from ground-truth labels, and non-lesional pixels were selected as the logical complement of the lesion mask. A field-of-view mask was optionally applied by thresholding image intensity, filling holes, and applying morphological opening. Skin-tone ITA was computed in CIE Lab color space from aggregated regional color statistics (default: median L* and median b*), using ITA = arctan((L* - 50)/(b* + eps)) * 180/pi. Binary strata were defined as lower-ITA versus higher-ITA using a prespecified cutoff (default 28 degrees), and six-category ITA bins were also recorded."

"Primary fairness endpoints were (1) continuous IoU and (2) success at IoU >= 0.50. We report effect sizes and uncertainty for group differences, including Cliff's delta, differences in median and mean IoU, risk difference, relative risk, and odds ratio, with bootstrap confidence intervals (BCa with percentile fallback). Secondary nonparametric tests (Mann-Whitney U and chi-square/Fisher exact) were reported as supportive analyses."

"We additionally fit continuous ITA trend models for success and IoU. Success trends used spline-expanded logistic regression with bootstrap confidence bands. IoU trends used spline-expanded median (quantile) regression with bootstrap confidence bands. Sensitivity analyses included IoU-threshold sweeps, deduplication sensitivity (none/exact/near), dependence summaries by dedup groups, and mask-source stratified summaries when available. All enhanced outputs were generated reproducibly with run metadata, runtime profiling, and checkpoint-resumable feature extraction."

## 14) Non-Causal Framing Requirement

All manuscript and artifact text should preserve proxy framing:
- use "image-derived non-lesional skin tone proxy (ITA)" (or perilesional if configured),
- avoid identity-language claims,
- interpret covariate-adjusted trends as descriptive attenuation checks, not causal effects.
