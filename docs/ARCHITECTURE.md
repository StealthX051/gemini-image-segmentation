# Architecture

## Runtime Flow
0. Optional dataset adaptation scripts materialize external datasets into CLI-compatible layout (`scripts/prepare_ima_plusplus.py` for IMA++) while preserving source/raw assets.
1. `segment` command resolves dataset paths and manifest (`src/gemini_segmentation/data.py`).
2. Prompt payload is built from prompt family/preset/provider (`src/gemini_segmentation/prompts.py`, `src/gemini_segmentation/config.py`).
3. Optional local request cache lookup is performed before provider calls (`src/gemini_segmentation/cache.py`, wired in `src/gemini_segmentation/cli.py`).
4. Provider adapter performs inference (`src/gemini_segmentation/models.py`) with retry policy applied in CLI orchestration (`max_retries`).
   - Gemini Robotics-ER 1.6 ablations can optionally enable Gemini code execution (`--gemini-agentic-vision`) while leaving prompt-family text unchanged.
   - Gemini 2.5 models use provider-gated JSON structured output (`response_mime_type="application/json"` plus schema); Robotics-ER 1.6 keeps the same prompt wording but remains on text-response parsing in this repo unless explicitly allowlisted later.
5. Responses are parsed into normalized masks and persisted (`src/gemini_segmentation/io.py`).
6. IoU/Dice/summary metrics are updated incrementally (`src/gemini_segmentation/metrics.py`) with single-channel normalization for RGB/RGBA mask inputs.
7. Optional fairness analysis consumes saved masks/metrics:
   - legacy mode uses `src/gemini_segmentation/fairness.py`,
   - enhanced mode uses `src/gemini_segmentation/fairness_enhanced/` and writes expanded artifacts under `fairness_enhanced/`.
   - enhanced defaults currently use global non-lesional ITA regions with aggregated Lab ITA estimation (configurable), plus exact dedup as the primary dedup mode.
   Fairness preprocessing concurrency remains configurable via `fairness --workers`, with optional memory-aware worker auto-capping in enhanced mode.
8. Optional batch orchestration composes multiple `segment`/`fairness` CLI calls from config matrices (`src/gemini_segmentation/batch.py`) and mirrors child command output to terminal + per-job logs.

## Manuscript Alignment
- See `docs/MANUSCRIPT_ALIGNMENT.md` for method-level constraints that tie implementation to manuscript and post hoc extensions.
- See `docs/GEMINI_CACHING.md` for model-specific caching support and operational defaults.
- Prompt ablation is represented by `PromptFamily`: `label_v1`, `desc_v1`, `desc_neg_v1`.
- Provider expansion includes Gemini model switching, Moondream adapter support, and Replicate/Sa2VA adapter support.
- Cost controls now include local request caching (all providers) plus Gemini explicit context caching when supported by the selected model.
- Run reproducibility relies on `run_config.json` fields such as `provider`, `prompt_family`, `prompt_hash`, provider-specific targets/instructions, model identifier, optional `output_model_name`, and `gemini_agentic_vision`.

## Key Design Contracts
- Segmenter contract: `segment(image_obj) -> (masks, latency_s, parse_success, timed_out, raw_items)`.
- Mask contract: `SegmentationMask` stores full-image binary mask plus pixel-space bounding box.
- Output contract: run artifacts live under `results/<dataset>/<model>/<prompt_key>/<run_id>/`.
- Output-label contract: `--output-model-name` can decouple artifact/report path labels from the actual API model identifier so batch ablations can compare two conditions that call the same backend model.
- Replicate pathing note: Replicate model-version identifiers are stored exactly in run config (`replicate_model_version`) and mapped to filesystem-safe `<model>` directory tokens for cross-platform artifact writes.
- Retry contract: per-image retries are configured by `max_retries` and apply to timeout/parse-failure outcomes.
- Resume behavior depends on `predictions.jsonl` and per-image artifact regeneration.
- Batch contract: orchestration outputs live under `results/batches/<run_id>/` with `resolved_config.json`, `job_status.jsonl`, `summary.json`, and per-job logs.
- Provider-parameter contract: Gemini segment calls may include sampling controls (`thinking_budget`, `temperature`), provider-gated structured-output controls (`response_mime_type`, `response_json_schema`), and the Robotics-only `gemini_agentic_vision` code-execution toggle; Moondream/Replicate segment calls omit Gemini-only controls.
- Cache-isolation contract: local Gemini request-cache keys include the agentic-vision toggle so Robotics 1.6 off/on conditions never share cached predictions.
- Replicate input contract: adapter sends image payloads as provider-supported file uploads and falls back to data-URI form if the client rejects file-like serialization; raw `bytes` are never passed directly in JSON request bodies.
- Replicate batch contract: config-level Replicate fields (`replicate_model_version`, `replicate_targets`, `replicate_instructions`, `replicate_cache_dir`) are resolved into provider-specific CLI flags, and fairness run discovery uses the Replicate output model label (`replicate_model_version`) to match CLI artifact paths.

## Module Ownership
- `cli.py`: argument parsing, run orchestration, checkpointing loop.
- `batch.py`: config-driven multi-job orchestration, strict preflight, deterministic command assembly, status/summary artifacts.
- `models.py`: provider clients and provider-specific output adaptation.
- `io.py`: JSON parsing, base64 encoding/decoding, overlay rendering, JSONL persistence.
- `metrics.py`: IoU/Dice, bootstrap CI, rolling summaries.
- `fairness.py`: ITA extraction, tone grouping, statistical testing outputs.
- `fairness_enhanced/`: enhanced fairness v2 pipeline (source indexing, feature-profile-gated extraction, configurable ITA region strategy with aggregated Lab ITA estimation, covariate extraction, SHA/pHash dedup, canonical analysis frame, effect sizes, covariate-adjusted predictive-margin success estimates, trends, sensitivity artifacts, staged execution including `augment`, runtime profiling, and resumable extraction checkpoints).
- `paper/`: manuscript-ready tables and figures (legacy fairness Figure 2/Table 4 via `paper/figures.py`, enhanced fairness manuscript artifacts via `paper/figures_enhanced.py`, plus comparison/best-case utilities).

## Extension Points
- New provider: implement adapter in `models.py`, keep return contract stable, wire in `cli.py`.
- New prompt family: extend `PromptFamily` and dictionaries in `prompts.py`, add YAML presets in `configs/prompts.yaml`.
- New dataset layout: extend dataset discovery/manifest behavior in `data.py` while preserving `images/` + `masks/` assumptions where possible.
- IMA++ integration pattern: keep core CLI ingestion unchanged and adapt source metadata/masks/images into `images/` + `masks/` with a prep script, while retaining richer mask metadata in sidecar files for optional analyses.
- IMA++ prep download backends: API mode (default) performs threaded ISIC API v2 image downloads with retry/backoff + skip-existing resume semantics; template mode executes user-specified CLI command templates.
- New analysis metric: add metric computation in `metrics.py`, propagate to CSV/summary and tests.
- Enhanced fairness extensions: prefer adding new disparity/tone-proxy logic under `fairness_enhanced/` while keeping `fairness.py` stable for legacy reproducibility.
- New benchmark study matrix: add/update YAML under `configs/benchmarks/` and validate orchestration behavior with `tests/test_batch.py`.

## Operational Notes
- Large datasets are intentionally externalized; repository code should not assume local copies exist.
- IMA++ canonical GT policy is handled at prep time (`STAPLE -> MV -> single-annotator-only`) with all masks retained for optional sensitivity analyses.
- IMA++ split-manifest generation accepts either split `ISIC_id` or `image` columns and normalizes filename case/extensions (e.g., `.JPG` to local `.jpg`) before mapping to prepared images.
- IMA++ prep copies optional `seg_metadata_multiannotator_subset.csv` into dataset metadata when present.
- Notebook workflows are legacy but still relevant for provenance and parity checks.
- Paper tools expect stable CSV schemas and config-driven registries in `configs/`.
- Replicate preflight checks token availability but cannot verify billing/credits or model-version permissions ahead of runtime; those can still surface as provider `429`/`422` responses during execution.

## NanoBanana Isolated Lane
- Independent package root: `src/nanobanana_segmentation/`.
- Existing `gemini_segmentation` CLI/runtime contracts are unchanged.
- Shared helper reuse is limited to stable utilities (dataset discovery, metric formulas, hashing/cache primitives).
- Core NanoBanana flow:
  1. Build semantic retry-ladder prompts (`core/prompts.py`).
  2. Call NanoBanana model adapter with tool mode and thinking controls (`core/engine.py`).
  3. Parse grounding + thought metadata (`core/grounding/parse_grounding.py`).
  4. Run deterministic mask extraction (`core/extract/*`).
  5. Apply QC scoring and attempt selection (`core/qc.py`).
  6. Persist raw request/response + artifacts + run record (`core/logging/*`).
- Microservice entrypoint: `nanobanana_segmentation.service.main` (`/v1/segment`, `/health`, `/metrics`).
- Study entrypoint: `python -m nanobanana_segmentation.study.runner --config ...`.
- NanoBanana default output roots:
  - `results_nanobanana/`
  - `artifacts_nanobanana/`
