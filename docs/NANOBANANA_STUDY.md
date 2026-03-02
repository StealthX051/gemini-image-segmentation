# NanoBanana 2 Study Specification

This document is the implementation-facing source of truth for the isolated NanoBanana study lane.
It preserves the core requirements from the original coding brief and maps them to this repository.

## 1) Deliverables

### Deliverable A: Segmentation microservice
- FastAPI app at `src/nanobanana_segmentation/service/main.py`
- Endpoints:
  - `POST /v1/segment`
  - `GET /health`
  - `GET /metrics`
- Input: medical image + target + segmentation controls
- Output: deterministic binary mask + attempt/QC metadata + optional debug references

### Deliverable B: Tool ablation study runner
- CLI: `python -m nanobanana_segmentation.study.runner --config configs/nanobanana/study.yaml --stage <stage0|stage1|stage2|all>`
- Runs staged tool-mode matrices and writes quantitative + audit outputs

## 2) Study framing

### Primary research question
Does enabling NanoBanana tool access (`text`, `image`, `text_image`) improve OOD medical-image segmentation reliability and performance vs `closed`?

### Mechanistic hypotheses
- `text`: may improve conceptual/procedural interpretation.
- `image`: may improve exemplar-driven localization.
- tools may introduce leakage/instability risks that must be audited.

### Defensible claims requirement
Any performance claim must be paired-image comparable across modes with leakage flags and grounding metadata retained.

## 3) Model and API constraints

### Model IDs (configurable)
- default: `gemini-3.1-flash-image-preview`
- fallback: `gemini-2.5-flash-image`
- optional high-tier: `gemini-3-pro-image-preview`

### API surfaces
- baseline: `generateContent`
- optional: `interactions` behind `api_surface` flag

### Thinking controls
- `minimal` default
- `high` for targeted Stage 2 contrasts

### Thought signatures/summaries handling
For every model attempt:
- raw request/response JSON persisted
- thought signature presence + raw tokens logged if present
- thought summaries logged if present
- grounding metadata parsed and persisted

## 4) Attempt semantics
- Semantic attempt: prompt-ladder attempt 1/2/3
- Transport retry: same request replay on timeout/429/5xx
- Replicate: full pipeline rerun by design

## 5) Prompt system (chromakey-first)

Runtime prompt builder: `src/nanobanana_segmentation/core/prompts.py`

- Prompt templates are config-wired from `configs/nanobanana/prompts.yaml`
- Context-bound fields:
  - `{target}`
  - `{profile_addendum}`
  - `{constraint_text}`
  - `{tool_addendum}`
  - `{model_addendum}`

### Semantic retry ladder
1. `chromakey_v1`: strict green/white two-color mask-surrogate
2. `chromakey_v2_strict`: correction clause with stricter compliance wording
3. `bw_v1_fallback`: black/white binary fallback

### NanoBanana-specific prompt addendum
Model-style directive enforces image-only output (no markdown/json text) for image-generation mode.

## 6) Deterministic extraction and QC

### Extraction methods
- Chromakey HSV inversion: `core/extract/chromakey_hsv.py`
- Chromakey green-ratio method: `core/extract/chromakey_ratio.py`
- BW threshold fallback: `core/extract/bw_threshold.py`

### Standard postprocess
- fill holes
- remove small components
- component filtering
- task-profile morphology (`blob|thin|low_contrast`)

### QC metrics
- resolution match
- non-empty
- area fraction
- component count
- largest-component fraction
- speckle score
- border touch
- chromakey-only: green coverage + green uniformity proxy

### QC policy
- fixed pass/fail rules with structured failure taxonomy
- deterministic attempt scoring and best-attempt selection even on all-fail cases

## 7) Tool modes and grounding logs
- `closed`: no retrieval
- `text`: text/web grounding
- `image`: image grounding
- `text_image`: both

### SDK compatibility caveat (current environment)
- Google docs configure web/image grounding through `tools=[{google_search:{search_types:[...]}}]` using `web_search` and `image_search` search types.
- This implementation follows that schema when the installed SDK exposes `GoogleSearch.search_types` and `SearchType.{web_search,image_search}`.
- The currently installed `google-genai` SDK in this repo environment does not expose those search-type selectors.
- Runtime behavior is therefore:
  - requested `image` mode falls back to effective `text` mode
  - requested `text_image` mode falls back to effective `text` mode
- Every attempt logs this explicitly in raw request payload warnings:
  - `tool_image_search_unavailable_in_sdk`
  - `tool_mode_fallback:<requested_mode>-><effective_mode>`
- Before any publication-facing `image` vs `text` ablation claim, verify SDK/API supports distinct image-search tool wiring in your execution environment.

Grounding parser:
- `src/nanobanana_segmentation/core/grounding/parse_grounding.py`
- captures queries, chunks, supports, entry-point metadata when present

## 8) Leakage/contamination controls

Module: `src/nanobanana_segmentation/study/leakage.py`

### Duplicate detection
- exact: SHA-256
- near-duplicate: pHash hamming threshold (default `<= 8`)

### Mask-source detection
- regex on URL/title/snippet (`mask|seg|label|groundtruth|annotation`)
- binary-like retrieved image heuristic

### Analysis set policy
- Primary set excludes `retrieval_duplicate=true` or `retrieval_mask_source=true`
- Sensitivity set includes flagged runs with explicit labels
- If retrieval metadata is absent: `audit_unavailable=true` and tracked explicitly

### Retrieval policy toggles (study config)
Configured in `configs/nanobanana/study.yaml` under `retrieval`:
- `query_policy`: `model_generated|fixed_queries`
- `snapshot_policy`: `live_with_caching|frozen`
- `scope_policy`: `open_web|curated_domains`
- `primary_exclude_duplicates`: bool
- `primary_exclude_mask_source`: bool
- `include_audit_unavailable_in_primary`: bool

## 9) Microservice API contract

### `POST /v1/segment`
Supports multipart and JSON (`image_base64`).

Request fields include:
- `target`
- `mode` (`auto|chromakey|bw`)
- `task_profile` (`blob|thin|low_contrast`)
- `tool_mode` (`closed|text|image|text_image`)
- `query_policy` (`model_generated|fixed_queries`)
- `snapshot_policy` (`live_with_caching|frozen`)
- `scope_policy` (`open_web|curated_domains`)
- `thinking_level` (`minimal|high`)
- `include_thoughts`
- retry limits (`max_retries_semantic`, `max_retries_transport`)
- optional constraints/budget
- output mode (`png|rle|coco`)

Response includes:
- `mask_png_base64`
- optional `mask_rle`/`mask_coco`
- `meta` with attempt list + selected attempt
- optional `debug` artifact references

### `GET /metrics`
Prometheus counters/histograms:
- requests total
- semantic attempts
- transport retries
- QC failures by reason
- tool-mode usage
- request latency histogram

## 10) Study runner staged design

Runner: `src/nanobanana_segmentation/study/runner.py`

### Parallel execution controls
Configured in `execution`:
- `workers`: concurrent study tasks (`>1` enables thread-pooled task execution)
- `progress_poll_seconds`: scheduler poll interval
- `progress_log_interval_seconds`: periodic progress logging cadence
- `stall_warning_seconds`: emits long-running task diagnostics when exceeded
- `fail_fast`: abort stage on first task error (`true`) or continue and report failures (`false`)

The runner writes `n_tasks`, `n_failures`, `failures`, and `stall_events` in `run_summary.json`.

### Stage 0
- smoke subset (10–30 images)
- modes: closed/text/image/text_image
- `k=1`

### Stage 1
- full benchmark
- same 4 modes
- `k=1`
- thinking `minimal`

### Stage 2
- targeted contrast only (default closed vs text_image)
- thinking `minimal vs high`
- stratified subset (default size 50, configurable)
- `k=3`

### Freeze-risk checkpoints
Primary observed long-wait points:
- external model calls during `generate_content` / transport retries
- semantic retry ladder when multiple attempts fail QC
- retrieval-audit IO on runs with large grounding payloads

Mitigations now in code:
- bounded semantic + transport retries remain enforced
- long-running task warnings are emitted at `stall_warning_seconds`
- periodic progress logs prevent silent stalls during long stages

## 11) Output and artifact layout

### Study results root
- `results_nanobanana/<dataset>/<model>/<run_id>/`
  - `results.csv`
  - `run_summary.json`
  - `reports/*.csv`
  - `reports/*.png`
  - Primary/sensitivity partition outputs:
    - `reports/summary_by_mode_primary.csv`
    - `reports/summary_by_mode_sensitivity.csv`
    - `reports/paired_delta_vs_closed_primary.csv`
    - `reports/paired_delta_vs_closed_sensitivity.csv`
    - `reports/qc_failure_summary_primary.csv`
    - `reports/qc_failure_summary_sensitivity.csv`
    - `reports/analysis_partition_counts.csv`
    - `reports/retrieval_audit_summary.csv`

### Artifact root
- `artifacts_nanobanana/<image_id>/<tool_mode>/replicate_<k>/<run_id>/`
  - `input/`
  - `surrogate/`
  - `intermediate/`
  - `final/`
  - `overlay/`
  - `raw/`
  - `run_record.json`

## 12) RunRecord contract
Run record captures:
- IDs (run/image/split/mode/replicate)
- model and tool config
- prompt metadata
- per-attempt logs (prompts, raw IO, retries, QC, grounding, thought fields)
- tool policy toggles (`tool_mode`, `query_policy`, `snapshot_policy`, `scope_policy`)
- outputs (hashes + artifact refs)
- evaluation slot
- leakage flags

## 13) Default operational settings
- tool modes: `closed`, `text`, `image`, `text_image`
- thinking: `minimal`
- semantic attempts: `3`
- transport retries: `3`
- workers: `6` (configurable)
- replicates: `k=1` initial
- include thoughts: `false`
- retrieval policy: live with caching + audit logging

## 13.1) Local pneumothorax smoke default
- Local SIIM-ACR pneumothorax root in this repo is configured as:
  - `data/SLIM_ACR`
- Positive-only manifest used by the pneumothorax smoke config:
  - `data/SLIM_ACR/master_imagelist_pneumothorax_cxr_positive.txt`
- The NanoBanana study loader supports both directory conventions:
  - `images/` + `masks/` (CLI-compatible layout)
  - `image/` + `mask/` (legacy SIIM-ACR layout)
- The smoke config already pins this path:
  - `configs/nanobanana/study_pneumothorax_smoke_desc_neg.yaml`

## 13.2) Local polyp smoke defaults
- Standard smoke (all four tool modes, chromakey-first auto extraction):
  - `configs/nanobanana/study_polyp_smoke.yaml`
- Diagnostic smoke (closed-only, BW-only extraction mode):
  - `configs/nanobanana/study_polyp_smoke_bw_closed.yaml`

## 14) Tests
NanoBanana tests live under `tests/nanobanana/` and cover:
- extraction + QC
- grounding/thought parser behavior
- engine retry behavior
- API smoke behavior
- report generation smoke

## 15) PDF report export
After a study run completes, generate a styled multi-page PDF report from run artifacts:

```bash
python scripts/export_nanobanana_pdf_report.py \
  --run-dir results_nanobanana/<dataset>/<model>/<run_id>
```

Default output:
- `<run_dir>/reports/nanobanana_run_report.pdf`

## 16) Non-interference contract
This package is intentionally isolated.
No NanoBanana changes should break or alter:
- `gemini_segmentation.cli`
- `gemini_segmentation.batch`
- manuscript-aligned prompt-family semantics (`label_v1`, `desc_v1`, `desc_neg_v1`)

## 17) Current empirical status (preliminary)
- As of March 2, 2026 local smoke runs, both NanoBanana extraction lanes executed end-to-end:
  - chromakey-first (`mode=auto`, chromakey attempts with BW fallback),
  - BW-only (`mode=bw`).
- In these smoke runs, observed segmentation performance did not clearly exceed the local `gemini-robotics-er-1.5-preview` baseline.
- Treat this as a provisional engineering observation, not a final cross-model claim:
  - sample sizes were small smoke subsets,
  - prompt families and tool-mode capabilities are still under active ablation,
  - publication-facing conclusions require controlled paired benchmarking with leakage-audited primary analysis outputs.
