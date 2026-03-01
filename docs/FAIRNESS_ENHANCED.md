# Fairness Enhanced (Audit v2)

Operational guide for the enhanced dermoscopy fairness pipeline behind:
- `python -m gemini_segmentation.cli fairness ... --audit-mode enhanced`

For exact algorithmic details and manuscript-ready wording, see:
- `docs/FAIRNESS_ENHANCED_METHODS.md`

## Mode Contract
- Legacy remains default:
  - `--audit-mode legacy`
  - outputs in `<run_dir>/fairness/`
- Enhanced is opt-in:
  - `--audit-mode enhanced`
  - outputs in `<run_dir>/fairness_enhanced/`
- Legacy output schemas and legacy paper tooling are unchanged.

## CLI Surface (Enhanced)
- `--enhanced-config <yaml_or_json>`
- `--enhanced-stage all|core|sensitivity|augment`
- `--enhanced-feature-profile balanced|full|minimal`
- `--enhanced-augment-columns <col1,col2,...>`
- `--enhanced-resume|--no-enhanced-resume`
- `--enhanced-checkpoint-every <N>`
- `--enhanced-workers-auto|--no-enhanced-workers-auto`

## Config Defaults
Default config file:
- `configs/fairness_enhanced.yaml`

Current default ITA method configuration:
- `ita.region_strategy: global_nonlesion`
- `ita.estimator: aggregated_lab`
- `ita.aggregation_stat: median`
- `ita.binary_cutoff: 28.0`

## Stage Behavior

`all`:
- source index build/load
- feature extraction
- fingerprint write
- exact/near dedup as configured
- endpoint effects + trend outputs
- covariate-adjusted success effects via predictive margins
- threshold sensitivity + optional sensitivity suite
- report artifacts

Model-family clarification:
- continuous trend artifacts (`ita_trend_success.*`, `ita_trend_iou_*`) use continuous `ita_deg`.
- covariate-adjusted disparity artifacts (`covadj_success_t050_effects.*`, `covadj_component_effects.*`) use binary `ita_binary` (Lower vs Higher ITA at the configured cutoff).

`core`:
- executes a fast completion-first pass with runtime overrides:
  - exact dedup only
  - near/dependence/mask-source sensitivity disabled
  - bootstrap percentile with reduced resampling
  - reduced trend bootstrap resamples

`sensitivity`:
- reuses existing core artifacts from `fairness_enhanced/`
- auto-augments missing `phash64_hex` when near dedup is requested
- runs near-dedup/dependence/mask-source sensitivity outputs

`augment`:
- computes only requested missing columns
- merges updates into existing `analysis_frame.parquet`
- updates consolidated caches

## Feature Profiles

`balanced` (default):
- computes ITA + key covariates
- uses `deltaE76` override in core trend/covariate flow unless `full`

`full`:
- computes full covariate set with configured high-cost options

`minimal`:
- preserves schema, skips nonessential covariates (written as NaN)

## Output Artifacts
Enhanced mode writes:
- `analysis_frame.parquet`
- `fingerprints.parquet`
- `run_metadata.json`
- `runtime_profile.json`
- `ita_method_note.json`
- `ita_method_note.md`
- `endpoint_effects_table.csv`
- `endpoint_effects.json`
- `covadj_success_t050_effects.csv`
- `covadj_success_t050_effects.json`
- `covadj_model_spec.json`
- `covadj_success_t050_bootstrap_samples.parquet`
- `covadj_component_effects.csv`
- `covadj_component_effects.json`
- `ita_bins_table.csv`
- `ita_bins_plot.png`
- `ita_trend_success.csv`
- `ita_trend_success.png`
- `ita_trend_iou_median.png`
- `ita_trend_iou_summary.json`
- `threshold_sensitivity_table.csv`
- `threshold_sensitivity.png`
- `dedup_map_exact.csv`
- `dedup_report.csv`
- optional `dedup_map_near.csv`
- `covariates_qc_plots/*`
- `duplicate_examples/*`

Sensitivity-stage outputs (when enabled/executed):
- `dedup_sensitivity.csv`
- `dependence_sensitivity.json`
- `mask_source_sensitivity.csv`

Cache and resume artifacts:
- `cache/source_index.parquet`
- `cache/features_part_*.parquet`
- `cache/features_manifest.json`
- `cache/metrics.parquet`
- `cache/ita_features.parquet`
- `cache/covariates.parquet`
- `cache/fingerprints.parquet`

## Runtime and Resume Semantics
- Extraction is checkpointed by image name.
- Resume skips previously processed image names from `features_manifest.json`.
- In-flight worker queue is bounded (`max_inflight_tasks`; auto to `workers` when unset).
- Worker auto-cap can reduce effective worker count from requested workers based on available memory and profile estimates.
- Runtime telemetry is written to `runtime_profile.json` per stage.

## Proxy Language Requirement
Enhanced text output is region-aware and proxy-safe:
- global mode: "image-derived non-lesional skin tone proxy (ITA)"
- ring mode: "image-derived perilesional skin tone proxy (ITA)"
- strata: "lower-ITA (darker-appearing) vs higher-ITA (lighter-appearing)"

Avoid identity-language overreach in generated captions/tables.

## Related Docs
- Algorithm and manuscript methods: `docs/FAIRNESS_ENHANCED_METHODS.md`
- Manuscript policy constraints: `docs/MANUSCRIPT_ALIGNMENT.md`
- Method/version history: `docs/METHODS_CHANGELOG.md`
