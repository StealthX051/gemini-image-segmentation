# Methods Changelog

This changelog tracks method-level changes that affect manuscript interpretation, reproducibility, or experimental comparability.

## Current Effective Version
- `posthoc_v19` (provider expansion + prompt ablation families + caching/retry hardening + provider-parameter parity + Replicate batch/instruction-shaping hardening + Replicate input-serialization/runtime validation hardening + standardized comparison reporting + Replicate-inclusive comparison report run selection + completed operational validation record + fairness Figure 2 panel-level export parity + IMA++ STAPLE-first canonical GT preparation with retained multi-mask sensitivity workflow + IMA++ acquisition/prep hardening for live Zenodo resolution, threaded API downloads, and split-manifest compatibility + enhanced dermoscopy fairness audit v2 with legacy-safe mode routing, SHA/pHash dedup, canonical analysis frame, trend/sensitivity artifacts, and proxy-language templating + enhanced fairness lesion-area/binarization correctness hardening for unit-scale GT masks and full-resolution lesion fraction retention + enhanced ITA methodology controls/provenance notes with optional legacy-like ITA sensitivity output for discrepancy analysis + enhanced-default ITA region reset to global non-lesional with aggregated L*/b* ITA and region-aware proxy labeling + covariate-adjusted predictive-margin success effects with bootstrap CIs and model-spec artifacts + adjusted-model component contribution outputs with CI-based significance and publication-readable enhanced report tables + NanoBanana SDK-compatible tool wiring with explicit image-search fallback telemetry + NanoBanana retrieval-policy toggles with explicit primary-vs-sensitivity partition reporting + NanoBanana image-generation config compatibility fix removing invalid `response_mime_type` on GenerateContent requests + NanoBanana image-part decoding hardening for SDK-native inline bytes/base64 and non-PNG surrogate image bytes).

## Entries

### MTH-BASELINE-001
- Date: manuscript-era baseline (pre post hoc additions; exact date not pinned in repo docs).
- Status: historical baseline.
- Summary: baseline manuscript workflow centered on medical-image segmentation evaluation with Gemini prompts and JSON-schema mask outputs.
- Impact: reference point for comparing all post hoc method additions.
- Primary anchors: manuscript draft methods text and notebook-era workflow.

### MTH-POSTHOC-001
- Date: post hoc addition window (after manuscript draft; exact date tracked in commit history).
- Status: active.
- Summary: provider expansion to include Moondream and Replicate/Sa2VA adapters, plus Gemini model-ID flexibility including `gemini-robotics-er-1.5-preview`.
- Impact: enables cross-provider comparisons with provider-specific prompt shaping and run-config traceability.
- Code anchors: `src/gemini_segmentation/models.py`, `src/gemini_segmentation/cli.py`, `tests/test_replicate_segmenter.py`, `tests/test_cli.py`.

### MTH-POSTHOC-002
- Date: post hoc addition window (after manuscript draft; exact date tracked in commit history).
- Status: active.
- Summary: prompt ablation families formalized as `label_v1`, `desc_v1`, and `desc_neg_v1` with fixed semantics and provider-aware prompt materialization.
- Impact: supports controlled ablation over language specificity while preserving output schema expectations for Gemini and compatible behavior for other providers.
- Code anchors: `src/gemini_segmentation/prompts.py`, `configs/prompts.yaml`, `tests/test_prompts.py`, `tests/test_cli.py`.

### MTH-POSTHOC-003
- Date: 2026-02-18.
- Status: active.
- Summary: added dual-layer caching controls for high-volume benchmark runs: (1) local deterministic request cache across providers; (2) Gemini explicit context cache for supported Gemini models.
- Impact: reduces repeated API spend during reruns/resume workflows while preserving ablation isolation via cache keys that include provider/model/prompt/image identity.
- Provider notes: Gemini docs list caching support for `gemini-2.5-flash` and `gemini-2.5-flash-lite`; robotics docs list `gemini-robotics-er-1.5-preview` caching as unsupported, so the implementation auto-falls back to no explicit Gemini cache for robotics ER.
- Code anchors: `src/gemini_segmentation/cache.py`, `src/gemini_segmentation/cli.py`, `src/gemini_segmentation/models.py`, `src/gemini_segmentation/types.py`, `src/gemini_segmentation/config.py`.

### MTH-POSTHOC-004
- Date: 2026-02-18.
- Status: active.
- Summary: added bounded retry handling (`max_retries`, default 5) for timeout/parse-failure outcomes and normalized multi-channel masks to single-channel before IoU/Dice computation.
- Impact: improves run robustness for malformed/partial model outputs and prevents metric crashes on RGB/RGBA mask files.
- Code anchors: `src/gemini_segmentation/cli.py`, `src/gemini_segmentation/metrics.py`, `src/gemini_segmentation/types.py`, `src/gemini_segmentation/config.py`, `tests/test_cli.py`, `tests/test_metrics.py`.

### MTH-POSTHOC-005
- Date: 2026-02-18.
- Status: active.
- Summary: standardized reproducible benchmark orchestration with a config-driven batch runner for prompt-ablation matrices and robotics ER benchmarking, including strict preflight checks, deterministic command assembly, and run-level status artifacts.
- Impact: improves unattended execution reproducibility and auditability across dataset/model matrices without changing prompt semantics, provider contracts, or metric definitions.
- Code anchors: `src/gemini_segmentation/batch.py`, `configs/benchmarks/ablation_robotics_canonical.yaml`, `scripts/launch_batch.sh`, `tests/test_batch.py`, `docs/BATCH_ORCHESTRATION.md`.

### MTH-POSTHOC-006
- Date: 2026-02-19.
- Status: active.
- Summary: hardened cross-provider parity by constraining non-Gemini segment calls to provider-native arguments (Moondream adapter invocation fallbacks; batch command assembly omits Gemini-only sampling controls for Moondream/Replicate), and added standardized post-run model-vs-prompt comparison reporting artifacts.
- Impact: reduces provider-parameter mismatch risk in Moondream runs, improves reproducible auditability for cross-model prompt-ablation comparisons, and standardizes manuscript-facing summary exports (mean/median IoU-Dice, 95% CIs, success rate) without changing segmentation metric definitions.
- Code anchors: `src/gemini_segmentation/models.py`, `src/gemini_segmentation/batch.py`, `src/gemini_segmentation/paper/prompt_comparison.py`, `tests/test_moondream_segmenter.py`, `tests/test_batch.py`, `tests/test_paper_prompt_comparison.py`.

### MTH-POSTHOC-007
- Date: 2026-02-19.
- Status: active.
- Summary: completed Replicate/Sa2VA production-parity hardening by adding batch-schema parity fields and strict preflight validation, provider-specific command forwarding, Replicate output-url extraction robustness across API output variants, and prompt-family-specific Replicate instruction shaping (`label_v1`, `desc_v1`, `desc_neg_v1`).
- Impact: closes batch-orchestration parity gaps for Replicate runs, improves resilience to Replicate output-shape variation, preserves manuscript prompt-ablation semantics for Replicate defaults, and improves resume reproducibility through stronger run-config handling.
- Code anchors: `src/gemini_segmentation/models.py`, `src/gemini_segmentation/batch.py`, `src/gemini_segmentation/prompts.py`, `src/gemini_segmentation/cli.py`, `tests/test_replicate_segmenter.py`, `tests/test_batch.py`, `tests/test_cli.py`, `tests/test_prompts.py`.

### MTH-POSTHOC-008
- Date: 2026-02-19.
- Status: active.
- Summary: hardened Replicate input serialization by switching SA2VA image input from raw `bytes` to provider-supported upload payloads with a data-URI fallback path, then documented runtime gating caveats discovered during smoke validation (account-credit throttling and strict version-permission checks).
- Impact: prevents `TypeError: Object of type bytes is not JSON serializable` request failures, preserves provider contract while improving cross-client compatibility, and makes Replicate smoke/full-run prerequisites explicit for reproducible execution planning.
- Code anchors: `src/gemini_segmentation/models.py`, `tests/test_replicate_segmenter.py`, `README.md`, `docs/ARCHITECTURE.md`, `docs/AGENT_HANDOFF.md`, `docs/BATCH_ORCHESTRATION.md`.

### MTH-POSTHOC-009
- Date: 2026-02-19.
- Status: active.
- Summary: completed Replicate/Sa2VA cross-provider reporting parity by adding explicit Replicate run selection support to the prompt-comparison reporting pipeline (`--replicate-run-id`) and recording successful end-to-end Replicate validation state (smoke + full polyp 3-family batch run) in operational docs.
- Impact: allows manuscript-style consolidated prompt-comparison reports to include Replicate rows deterministically by run ID, and improves reproducibility handoff by pinning a validated Replicate run and model version in project docs without changing segmentation metric definitions.
- Code anchors: `src/gemini_segmentation/paper/prompt_comparison.py`, `tests/test_paper_prompt_comparison.py`, `README.md`, `docs/AGENT_HANDOFF.md`, `llms.txt`.

### MTH-POSTHOC-010
- Date: 2026-02-26.
- Status: active.
- Summary: aligned fairness Figure 2 rendering with legacy derm notebook styling and added panel-level artifact exports so a single run writes the combined figure and each panel as standalone files (`PNG/PDF/SVG`).
- Impact: improves manuscript assembly flexibility and reproducible figure reuse without changing fairness statistics, thresholds, or metric definitions.
- Code anchors: `src/gemini_segmentation/paper/figures.py`, `tests/test_paper_figures.py`, `README.md`, `docs/AGENT_HANDOFF.md`, `docs/NOTEBOOKS.md`.

### MTH-POSTHOC-011
- Date: 2026-02-26.
- Status: active.
- Summary: added IMA++ dataset preparation and sensitivity-analysis tooling with deterministic canonical GT policy (`STAPLE -> majority vote -> single annotator only`) while retaining all masks + per-mask metadata for optional post-run sensitivity analyses.
- Impact: preserves existing CLI ingestion contracts (`images/` + `masks/`) while enabling manuscript-defensible consensus-first evaluation on IMA++ and transparent robustness checks against MV/annotator variation.
- Code anchors: `scripts/prepare_ima_plusplus.py`, `scripts/analyze_ima_plusplus_sensitivity.py`, `src/gemini_segmentation/prompts.py`, `configs/prompts.yaml`, `tests/test_prepare_ima_plusplus.py`, `tests/test_ima_plusplus_sensitivity.py`, `tests/test_prompts.py`, `tests/test_cli.py`, `README.md`, `docs/MANUSCRIPT_ALIGNMENT.md`, `docs/ARCHITECTURE.md`, `docs/AGENT_HANDOFF.md`.

### MTH-POSTHOC-012
- Date: 2026-02-26.
- Status: active.
- Summary: hardened IMA++ preparation operations by (1) pinning live Zenodo record file defaults, (2) adding threaded ISIC API download mode with retry/backoff + resume-safe skip-existing behavior, (3) ingesting optional `seg_metadata_multiannotator_subset.csv`, and (4) fixing split-manifest generation to support split CSVs with `image` column values (including case/extension normalization).
- Impact: improves large-scale acquisition reliability and reproducibility of prepared dataset artifacts without changing canonical GT policy, segmentation metrics, or sensitivity-analysis definitions.
- Code anchors: `scripts/prepare_ima_plusplus.py`, `tests/test_prepare_ima_plusplus.py`, `README.md`, `docs/ARCHITECTURE.md`, `docs/AGENT_HANDOFF.md`.

### MTH-POSTHOC-013
- Date: 2026-02-26.
- Status: active.
- Summary: added fairness audit v2 as an opt-in CLI mode (`fairness --audit-mode enhanced`) while preserving legacy fairness as default; the enhanced mode adds SHA/pHash deduplication, source-index sidecar integration across interop ISIC and IMA++, canonical `analysis_frame.parquet`, endpoint effect-size reporting, continuous ITA trend modeling, and sensitivity artifacts.
- Impact: strengthens methodological defensibility for dermoscopy disparity analyses without breaking existing legacy fairness outputs or figure-generation workflows.
- Code anchors: `src/gemini_segmentation/cli.py`, `src/gemini_segmentation/fairness_enhanced/`, `configs/fairness_enhanced.yaml`, `tests/test_cli.py`, `tests/test_fairness_enhanced.py`, `README.md`, `docs/ARCHITECTURE.md`, `docs/MANUSCRIPT_ALIGNMENT.md`.

### MTH-POSTHOC-014
- Date: 2026-02-26.
- Status: active.
- Summary: hardened enhanced fairness trend/dedup internals for Windows stability by removing crash-prone native SciPy backends in two paths: pHash DCT now uses a NumPy orthonormal DCT implementation, and IoU quantile trend fitting now uses deterministic IRLS-style quantile fitting (pinball-loss approximation via asymmetric weighted least squares) instead of `QuantileRegressor` with HiGHS; success-trend logistic fitting was set to `solver=liblinear` to avoid SciPy L-BFGS deprecation noise during testing; enhanced fairness preprocessing now honors `--workers` with thread-safe per-image parallelism and deterministic row ordering.
- Impact: preserves enhanced fairness artifact contracts and trend outputs while preventing access-violation failures observed during test runs on Windows Conda environments, reducing non-actionable warning noise, and enabling faster enhanced runs with controlled worker concurrency.
- Code anchors: `src/gemini_segmentation/fairness_enhanced/dedup.py`, `src/gemini_segmentation/fairness_enhanced/trends.py`, `src/gemini_segmentation/fairness_enhanced/reporting.py`, `src/gemini_segmentation/fairness_enhanced/pipeline.py`, `src/gemini_segmentation/cli.py`, `tests/test_cli.py`.

### MTH-POSTHOC-015
- Date: 2026-02-27.
- Status: active.
- Summary: added runtime troubleshooting hardening for enhanced fairness with staged execution controls (`all|core|sensitivity`), resumable extraction checkpoints (`features_part_*.parquet` + `features_manifest.json`), bounded in-flight worker scheduling, stage-level runtime profiling (`runtime_profile.json`), and CLI/config runtime overrides for stage/resume/checkpoint cadence.
- Impact: improves completion reliability and restartability on large 4k-image runs, reduces memory-pressure risk from unbounded future submission, and enables fast core-first workflows before expensive sensitivity passes while preserving legacy mode behavior.
- Code anchors: `src/gemini_segmentation/cli.py`, `src/gemini_segmentation/fairness_enhanced/config.py`, `src/gemini_segmentation/fairness_enhanced/pipeline.py`, `configs/fairness_enhanced.yaml`, `tests/test_cli.py`, `tests/test_fairness_enhanced.py`, `README.md`, `docs/FAIRNESS_ENHANCED.md`.

### MTH-POSTHOC-016
- Date: 2026-02-28.
- Status: active.
- Summary: refactored enhanced fairness extraction for efficiency while preserving fairness-method rigor: ROI/downsampled ITA computation with sampled-pixel Lab conversion (no full-frame Lab arrays), feature-profile gating (`balanced|full|minimal`), sensitivity-oriented pHash computation with `augment` backfill stage, consolidated feature caches (`metrics/ita_features/covariates/fingerprints`), and memory-aware worker auto-capping.
- Impact: substantially reduces memory pressure and paging risk on large dermoscopy runs, preserves legacy fairness behavior by default, keeps enhanced artifact/schema compatibility, and enables targeted feature augmentation without rerunning full core extraction.
- Code anchors: `src/gemini_segmentation/fairness_enhanced/ita.py`, `src/gemini_segmentation/fairness_enhanced/covariates.py`, `src/gemini_segmentation/fairness_enhanced/config.py`, `src/gemini_segmentation/fairness_enhanced/pipeline.py`, `src/gemini_segmentation/cli.py`, `configs/fairness_enhanced.yaml`, `tests/test_cli.py`, `tests/test_fairness_enhanced.py`, `docs/FAIRNESS_ENHANCED.md`.

### MTH-POSTHOC-017
- Date: 2026-02-28.
- Status: active.
- Summary: added an enhanced fairness manuscript artifact generator (`paper/figures_enhanced.py`) that consumes `fairness_enhanced/` outputs and renders E-numbered main/supplement figures and tables plus a unified narrative report in `md|html|pdf|docx`.
- Impact: enables publication-oriented reporting for enhanced fairness analyses (cohort accountability, effect-size forests, trend interpretation, threshold/sensitivity summaries) without changing legacy fairness figure/table workflows or fairness metric computations.
- Code anchors: `src/gemini_segmentation/paper/figures_enhanced.py`, `src/gemini_segmentation/paper/__init__.py`, `tests/test_paper_figures_enhanced.py`, `README.md`, `docs/ARCHITECTURE.md`, `docs/MANUSCRIPT_ALIGNMENT.md`.

### MTH-POSTHOC-018
- Date: 2026-02-28.
- Status: active.
- Summary: corrected enhanced fairness lesion-area derivation robustness by (1) making GT mask binarization tolerant to unit-scale masks (`0/1` and `[0,1]`), and (2) forcing `lesion_area_frac` in `analysis_frame.parquet` to remain the full-resolution GT-mask fraction (preventing ROI/downsample covariate internals from overwriting it).
- Impact: prevents spurious near-zero lesion-area distributions when datasets store binary masks as unit-scale values, and preserves manuscript-meaningful lesion-size covariate semantics tied to full-image GT area.
- Code anchors: `src/gemini_segmentation/fairness_enhanced/ita.py`, `src/gemini_segmentation/fairness_enhanced/pipeline.py`, `tests/test_fairness_enhanced.py`.

### MTH-POSTHOC-019
- Date: 2026-02-28.
- Status: active.
- Summary: added explicit enhanced ITA-method controls and provenance outputs: configurable ITA region strategy (`perilesional_ring` vs `global_nonlesion`), estimator (`aggregated_lab` vs `pixelwise_median`), robust aggregation (`median|mean|trimmed_mean_sd`), optional L* percentile windowing, and run-level ITA method note artifacts (`ita_method_note.json/.md`); optional legacy-like ITA sensitivity columns were added to `analysis_frame.parquet` for side-by-side comparability checks.
- Impact: improves manuscript defensibility and reproducibility of proxy-stratification definitions, and enables transparent diagnosis of legacy-vs-enhanced ITA distribution shifts without altering legacy fairness behavior.
- Code anchors: `src/gemini_segmentation/fairness_enhanced/config.py`, `src/gemini_segmentation/fairness_enhanced/ita.py`, `src/gemini_segmentation/fairness_enhanced/pipeline.py`, `configs/fairness_enhanced.yaml`, `docs/FAIRNESS_ENHANCED.md`, `tests/test_fairness_enhanced.py`.

### MTH-POSTHOC-020
- Date: 2026-02-28.
- Status: active.
- Summary: switched the enhanced ITA default region strategy back to `global_nonlesion` while keeping the improved default ITA estimator (`aggregated_lab` with robust aggregation), and made proxy-language text region-aware so generated captions/method notes match the configured region strategy.
- Impact: restores legacy-aligned non-lesional sampling as the enhanced default for manuscript comparability, while retaining methodologically explicit aggregation controls and run-level ITA provenance artifacts.
- Code anchors: `src/gemini_segmentation/fairness_enhanced/config.py`, `configs/fairness_enhanced.yaml`, `src/gemini_segmentation/fairness_enhanced/labels.py`, `src/gemini_segmentation/fairness_enhanced/pipeline.py`, `tests/test_fairness_enhanced.py`, `docs/FAIRNESS_ENHANCED.md`, `docs/MANUSCRIPT_ALIGNMENT.md`.

### MTH-POSTHOC-021
- Date: 2026-02-28.
- Status: active.
- Summary: consolidated enhanced fairness documentation into an implementation-aligned methods specification (`docs/FAIRNESS_ENHANCED_METHODS.md`) and synchronized operational/manuscript docs to the current enhanced defaults and stage behavior.
- Impact: improves manuscript reproducibility and reporting consistency without changing fairness computation logic or output schemas.
- Code anchors: `docs/FAIRNESS_ENHANCED_METHODS.md`, `docs/FAIRNESS_ENHANCED.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/MANUSCRIPT_ALIGNMENT.md`.

### MTH-POSTHOC-022
- Date: 2026-02-28.
- Status: active.
- Summary: added covariate-adjusted binary-success disparity estimation to enhanced fairness using logistic-regression predictive margins, standardized adjusted risks (`r_low_adj`, `r_high_adj`), and derived adjusted RD/RR/OR with bootstrap confidence intervals over dedup-aware resampling units.
- Impact: adds a direct manuscript-ready answer to whether lower-ITA vs higher-ITA success disparity persists after adjustment for technical image covariates, while preserving non-causal proxy framing and legacy fairness mode behavior.
- Code anchors: `src/gemini_segmentation/fairness_enhanced/covadj.py`, `src/gemini_segmentation/fairness_enhanced/pipeline.py`, `tests/test_fairness_enhanced.py`, `docs/FAIRNESS_ENHANCED_METHODS.md`, `docs/FAIRNESS_ENHANCED.md`, `README.md`, `docs/MANUSCRIPT_ALIGNMENT.md`.

### MTH-POSTHOC-023
- Date: 2026-02-28.
- Status: active.
- Summary: extended enhanced covariate-adjusted outputs with per-term adjusted-model contributions (log-odds and odds-ratio views) plus bootstrap CI-based term significance flags, and updated enhanced manuscript artifact tables to include component contributions with publication-readable columns.
- Impact: enables transparent reporting of which adjusted-model contributors retain directional/significance evidence, while keeping report generation resilient when covariate-adjusted artifacts are absent.
- Code anchors: `src/gemini_segmentation/fairness_enhanced/covadj.py`, `src/gemini_segmentation/fairness_enhanced/pipeline.py`, `src/gemini_segmentation/paper/figures_enhanced.py`, `tests/test_fairness_enhanced.py`, `tests/test_paper_figures_enhanced.py`, `docs/FAIRNESS_ENHANCED_METHODS.md`, `docs/FAIRNESS_ENHANCED.md`, `README.md`.

### MTH-POSTHOC-024
- Date: 2026-03-01.
- Status: active.
- Summary: added an isolated NanoBanana 2 study lane (`src/nanobanana_segmentation/`) with a segmentation microservice (`/v1/segment`, `/health`, `/metrics`) and a staged tool-ablation runner (`stage0/1/2`) that logs raw request/response payloads, grounding/thought metadata, deterministic extraction/QC attempts, and run-record artifacts under dedicated roots (`results_nanobanana/`, `artifacts_nanobanana/`).
- Impact: enables independent NanoBanana capability studies without changing existing manuscript-aligned `gemini_segmentation` CLI/provider contracts; introduces retrieval leakage auditing (duplicate and mask-source heuristics) plus study-level summary/report outputs for tool-mode comparisons.
- Code anchors: `src/nanobanana_segmentation/core/engine.py`, `src/nanobanana_segmentation/core/qc.py`, `src/nanobanana_segmentation/core/extract/`, `src/nanobanana_segmentation/core/grounding/parse_grounding.py`, `src/nanobanana_segmentation/service/main.py`, `src/nanobanana_segmentation/study/runner.py`, `src/nanobanana_segmentation/study/leakage.py`, `configs/nanobanana/service.yaml`, `configs/nanobanana/study.yaml`, `docs/NANOBANANA_STUDY.md`, `tests/nanobanana/`.

### MTH-POSTHOC-025
- Date: 2026-03-01.
- Status: active.
- Summary: fixed NanoBanana `generateContent` tool wiring to use SDK-validated list payloads (`tools=[...]`) instead of dict payloads, moved `include_thoughts` to `ThinkingConfig`, and aligned tool-mode mapping to Google docs by using `google_search.search_types` (`web_search`/`image_search`) when available, with deterministic fallback telemetry when search-type selectors are missing in the installed SDK.
- Impact: prevents runtime request validation failure (`GenerateContentConfig.tools` list-type error), preserves smoke/study execution continuity, and makes tool-mode degradations auditable so ablation interpretation can exclude unsupported `image` tool claims.
- Code anchors: `src/nanobanana_segmentation/core/engine.py`, `tests/nanobanana/test_engine.py`, `docs/NANOBANANA_STUDY.md`, `docs/AGENT_HANDOFF.md`.

### MTH-POSTHOC-026
- Date: 2026-03-01.
- Status: active.
- Summary: added configurable NanoBanana retrieval-policy toggles (`query_policy`, `snapshot_policy`, `scope_policy`) through study config/API request wiring and propagated them into `run_record.tool_config`; implemented explicit primary vs sensitivity analysis partitioning with dedicated CSV/plot outputs and partition-count reporting.
- Impact: removes hardcoded retrieval-policy assumptions, makes ablation governance settings auditable per run, and aligns reporting with contamination-handling requirements by separating primary and sensitivity analysis sets.
- Code anchors: `src/nanobanana_segmentation/study/config.py`, `src/nanobanana_segmentation/study/runner.py`, `src/nanobanana_segmentation/study/reports.py`, `src/nanobanana_segmentation/core/types.py`, `src/nanobanana_segmentation/core/engine.py`, `src/nanobanana_segmentation/service/api_models.py`, `src/nanobanana_segmentation/service/main.py`, `configs/nanobanana/study.yaml`, `docs/NANOBANANA_STUDY.md`.

### MTH-POSTHOC-027
- Date: 2026-03-01.
- Status: active.
- Summary: fixed NanoBanana `generateContent` image-generation request compatibility by removing `generation_config.response_mime_type=image/png` and relying on `response_modalities=["IMAGE"]`.
- Impact: resolves runtime `400 INVALID_ARGUMENT` failures in current `google-genai` SDK where `response_mime_type` is restricted to text/structured MIME types for this endpoint.
- Code anchors: `src/nanobanana_segmentation/core/engine.py`.

### MTH-POSTHOC-028
- Date: 2026-03-01.
- Status: active.
- Summary: hardened NanoBanana surrogate image extraction to follow Python SDK response patterns (`response.parts` / `part.inline_data` / `part.as_image()`), accept both inline bytes and base64 string payloads, and decode non-PNG image bytes (e.g., JPEG) before deterministic mask extraction.
- Impact: resolves runtime surrogate decode failures (`Unable to decode PNG bytes`) when the model returns image bytes in SDK-native formats that are not strict PNG base64 strings.
- Code anchors: `src/nanobanana_segmentation/core/engine.py`, `tests/nanobanana/test_engine.py`.

### MTH-POSTHOC-029
- Date: 2026-03-01.
- Status: active.
- Summary: added configurable parallel execution for the NanoBanana study runner (`execution.workers`) with deterministic result ordering plus stage-level progress/stall diagnostics (`progress_poll_seconds`, `progress_log_interval_seconds`, `stall_warning_seconds`, `fail_fast`) and run-summary telemetry (`n_tasks`, `n_failures`, `failures`, `stall_events`).
- Impact: reduces wall-clock time for stage runs, improves visibility into long-running/frozen-looking tasks, and preserves auditable outputs under the existing study/result artifact contracts.
- Code anchors: `src/nanobanana_segmentation/study/config.py`, `src/nanobanana_segmentation/study/runner.py`, `configs/nanobanana/study.yaml`, `configs/nanobanana/study_pneumothorax_smoke_desc_neg.yaml`, `tests/nanobanana/test_study_config.py`, `tests/nanobanana/test_runner_parallel.py`, `docs/NANOBANANA_STUDY.md`, `docs/AGENT_HANDOFF.md`.

### MTH-POSTHOC-030
- Date: 2026-03-02.
- Status: active.
- Summary: hardened NanoBanana engine handling for surrogate/output resolution mismatches by resizing selected masks to input image dimensions with nearest-neighbor before overlay/final artifact writes, while preserving explicit QC failure signaling (`resolution_mismatch`) and warning telemetry (`resized_mask_to_input_shape`).
- Impact: prevents stage-run crashes on non-conforming model output sizes (for example `(960,1120)` surrogate masks for `(529,619)` inputs), allowing smoke/benchmark completion with auditable mismatch flags instead of hard failures.
- Code anchors: `src/nanobanana_segmentation/core/engine.py`, `tests/nanobanana/test_engine.py`.

### MTH-POSTHOC-031
- Date: 2026-03-02.
- Status: active.
- Summary: hardened NanoBanana study evaluation against multi-channel ground-truth masks by normalizing GT/pred arrays to single-channel before metric computation and returning zeroed metrics (with warning) on irreconcilable shape mismatch instead of crashing the run.
- Impact: prevents stage-run failures on datasets with RGB/JPEG mask files (for example polyp `masks/*.jpg`), enabling stable smoke diagnostics while preserving explicit mismatch handling.
- Code anchors: `src/nanobanana_segmentation/study/eval.py`, `tests/nanobanana/test_eval.py`.

## Update Protocol For New Method Changes
- Add a new `MTH-*` entry in this file describing what changed and why it matters.
- Update `docs/MANUSCRIPT_ALIGNMENT.md` if semantics, contracts, or provider behavior changed.
- Update method-facing docs in `README.md` when user-visible behavior changed.
- Add or adjust tests (`tests/test_cli.py`, `tests/test_prompts.py`, provider-specific tests) to lock behavior.
- Include the new changelog ID in PR/commit notes for traceability.
