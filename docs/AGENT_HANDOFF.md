# Agent Handoff (Current State)

Last updated: 2026-04-22.

## Scope Of This File
- Read `docs/ENGINEERING_QUICKSTART.md` first if you are starting cold.
- Read `docs/SETUP.md` first if the task is about installation, environment repair, or fresh-clone reproducibility.
- This file is an operational snapshot and may include point-in-time validation notes (run IDs, observed throttling, local smoke outcomes).
- Documentation ownership map: `docs/DOCUMENTATION_MAP.md`.
- Canonical behavior/contracts live in: `docs/ARCHITECTURE.md`, `docs/MANUSCRIPT_ALIGNMENT.md`, `docs/METHODS_CHANGELOG.md`, and `docs/BATCH_ORCHESTRATION.md`.
- If any statement here conflicts with those canonical docs or current CLI help/code, prefer the canonical docs and code.

## Current Priorities
- Prompt-ablation runs (`label_v1`, `desc_v1`, `desc_neg_v1`) across Gemini models.
- Robotics ER benchmarking via `gemini-robotics-er-1.5-preview`.
- Cost control through local request cache plus Gemini explicit cache where supported.
- Current fairness workflow preference: run fairness analyses on dermoscopy-focused studies unless explicitly requested for other datasets.
- Fairness Figure 2/Table 4 rendering parity with legacy derm notebook styling and manuscript-facing annotations.

## Runtime Facts
- CLI entrypoint: `python -m gemini_segmentation.cli segment ...`
- Batch entrypoint: `python -m gemini_segmentation.batch --config ...`
- Batch runner mirrors active job stdout/stderr to terminal and writes the same stream to `results/batches/<run_id>/logs/*.log`.
- Prompt families are selected by repeating `--prompt-family` (no `--prompt-families` flag).
- Default retry policy: `--max-retries 5` (five retries after the first attempt) for timeout/parse-failure/exception retries.
- Local request cache is enabled by default; failed parses/timeouts are not persisted.
- Explicit Gemini context cache is enabled by default for supported models and auto-skipped for robotics ER.
- Moondream segment calls use provider-native target arguments and do not use Gemini-only `temperature`/`thinking_budget` controls.
- Replicate batch jobs now support explicit parity fields (`replicate_model_version`, `replicate_targets`, `replicate_instructions`, `replicate_cache_dir`) with strict preflight validation.
- Replicate default instructions are prompt-family aware (`label_v1`, `desc_v1`, `desc_neg_v1`) and remain overrideable per target via repeated `--replicate-instruction`.
- Replicate adapter sends image payloads as file uploads (with a data-URI fallback path for client-serialization compatibility).
- `python -m gemini_segmentation.paper.figures --fairness-dir <.../fairness>` expects fairness CSVs in that exact directory and writes `figure2.png|pdf|svg` plus standalone panel exports (`figure2_panel_a` through `figure2_panel_d` in PNG/PDF/SVG).
- Current Figure 2 panel behavior: left IoU panel uses full `0.0–1.0` range; right IoU panel is thresholded (`IoU >= 0.5`) and truncated at `0.5`.
- IMA++ prep utility: `python scripts/prepare_ima_plusplus.py` builds a separate CLI-compatible dataset root (`data/IMAplusplus_cli` by default) with canonical GT policy `STAPLE -> MV -> single annotator only`.
- IMA++ prep defaults now use threaded ISIC API download mode (`--download-images-mode api`) with retries/backoff and skip-existing behavior for resumable pulls.
- IMA++ prep also copies optional `seg_metadata_multiannotator_subset.csv` when present/downloaded.
- IMA++ split manifests now handle split CSVs using either `ISIC_id` or `image` columns (including `.JPG`/case normalization).
- IMA++ sensitivity utility: `python scripts/analyze_ima_plusplus_sensitivity.py --run-dir <...>` writes MV/annotator sensitivity summaries under `<run_dir>/ima_plusplus_sensitivity/`.
- New dataset key for prompts/presets: `ima_plusplus` (dermoscopy semantics aligned with `derm_lesion`).

## Critical Gotchas
- `.env` is not auto-loaded into shell process env vars by the CLI. Export env vars in the active shell before running.
- In Codex automation, prefer `conda run -n gemini_seg ...` over `conda activate gemini_seg`.
- The Windows convenience benchmark script `.\scripts\run_polyp_full_3x3_w10.ps1` auto-loads `.env` but writes `configs/benchmarks/polyp_full_w10.local.yaml`, which dirties the worktree.
- Ignore rules do not affect already tracked files. If NanoBanana run artifacts were committed previously, untrack them once with `git rm --cached -r results_nanobanana artifacts_nanobanana`.
- Replicate fairness discovery in batch uses the Replicate output model label (`replicate_model_version`) rather than the matrix display name.
- Replicate preflight validates token presence but cannot validate account credits/billing state; runtime can still fail with `429` create-prediction throttling on unfunded accounts.
- Replicate model-version IDs must be exact and accessible; invalid/inaccessible versions return `422 Invalid version or not permitted`.
- Using `--prompt-preset configs/prompts.yaml --preset-name polyp` can override `--model-name` because preset `polyp` sets a model in YAML.
  - For strict model comparisons, either:
    - avoid preset model-bearing entries, or
    - use family-only selection with explicit `--model-name`.
- Some datasets have RGB ground-truth masks; metrics now normalize masks to single-channel before IoU/Dice.
- Do not pass placeholder paths like `<your_run>` to `paper.figures`; pass the actual run fairness folder containing `fairness_results.csv`.
- `isic auth login` is not available in newer `isic-cli` versions; use the prep script's API mode (default) or template mode with current `isic image download` syntax when needed.
- Zenodo DOI `10.5281/zenodo.14201692` currently resolves to record `14201693`; prep script defaults point to live record URLs and are overrideable.

## Execution Patterns
- Segment smoke/parity pattern: one model per run, repeat `--prompt-family`, keep local cache enabled, and tune `--workers`/`--rate-limit` for provider limits.
- Batch benchmark pattern: `python -m gemini_segmentation.batch --config ... [--overrides ...] [--auto-fairness]`.
- Reporting pattern: `python -m gemini_segmentation.paper.prompt_comparison --dataset <name> [--*-run-id ...]`.
- Windows full-polyp convenience benchmark: `.\scripts\run_polyp_full_3x3_w10.ps1` (reuse `-RunId` to resume).
- Canonical runnable command examples live in `README.md`; keep this handoff focused on behavior deltas and gotchas.

## Validation Snapshots
- Historical validation snapshots (pinned run IDs, exact smoke command variants, and point-in-time metric highlights) are kept in `docs/history/VALIDATION_SNAPSHOTS.md`.

## Doc Routing
- For fast onboarding, use `AGENTS.md`, `docs/DOCUMENTATION_MAP.md`, and `docs/ENGINEERING_QUICKSTART.md`.
- Read deeper docs only for the scope you are touching.

## NanoBanana Study Lane (2026-03-01)
- New isolated package path: `src/nanobanana_segmentation/`.
- Only read or use this section when the task is explicitly about NanoBanana.
- Service entrypoint:
  - `uvicorn nanobanana_segmentation.service.main:app --host 0.0.0.0 --port 8000`
- Study runner entrypoint:
  - `python -m nanobanana_segmentation.study.runner --config configs/nanobanana/study.yaml --stage stage1`
- Default NanoBanana roots:
  - `results_nanobanana/`
  - `artifacts_nanobanana/`
- Configs:
  - `configs/nanobanana/service.yaml`
  - `configs/nanobanana/study.yaml`
  - `configs/nanobanana/prompts.yaml`
- Operational notes:
  - package is intentionally isolated; do not route NanoBanana logic through `gemini_segmentation.cli`.
  - model responses are persisted raw per attempt in artifact run directories.
  - grounding/thought metadata capture is best-effort and field-tolerant to API payload shape changes.
  - some image-generation responses can return masks at non-input resolution; engine now resizes selected masks to input shape for overlay/final artifact writes and logs `resized_mask_to_input_shape` plus QC `resolution_mismatch`.
  - study evaluation now normalizes multi-channel GT/pred masks to single-channel before IoU/Dice/Precision/Recall computation to support RGB/JPEG mask datasets (for example polyp).
  - Current `google-genai` environment does not expose Google Search type selectors (`GoogleSearch.search_types` with `web_search`/`image_search`); requested `image` and `text_image` modes therefore fall back to effective `text`, with explicit attempt warnings (`tool_image_search_unavailable_in_sdk`, `tool_mode_fallback:*`).
  - Retrieval-policy toggles are configurable in study configs via `retrieval.{query_policy,snapshot_policy,scope_policy,...}` and reporting now emits explicit primary/sensitivity partition artifacts.
  - Study runner execution parallelism and stall diagnostics are configurable via `execution.{workers,progress_poll_seconds,progress_log_interval_seconds,stall_warning_seconds,fail_fast}`; run summaries now include `n_tasks`, `n_failures`, `failures`, and `stall_events`.
  - Preliminary local smoke observation (March 2, 2026): both chromakey-first and BW-only NanoBanana lanes run end-to-end, but observed performance has not clearly exceeded the local `gemini-robotics-er-1.5-preview` baseline; treat as non-final until controlled paired benchmark runs are completed.
