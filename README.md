# Gemini Image Segmentation

This repository contains two parallel workflows for evaluating Google Gemini models on medical-image segmentation tasks:

1. **Legacy Jupyter notebooks** that encode the original experiments per dataset (polyp, optic disc, dermatology, BUSI breast ultrasound, chest X-ray pneumothorax, LiTS liver lesions, histopathology, and laparoscopy). Each notebook’s first cell is a linear Python script with helper functions, dataset discovery, Gemini calls, and metrics.
2. **A modular CLI** (`python -m gemini_segmentation.cli`) that lifts the notebook logic into reusable modules with resumable runs, centralized outputs, and fairness analysis.

Read this document top-to-bottom when onboarding: it explains the environment, directory layout, how the notebooks work, how the CLI mirrors them, and where to extend the system (prompts, models, fairness, outputs).

## Agentic development files
- `AGENTS.md`: repository-level instructions for coding agents.
- `.codex/config.toml`: repo-local Codex defaults (model/sandbox/doc discovery).
- `CONTRIBUTING.md`: contributor workflow and validation commands.
- `docs/ARCHITECTURE.md`: module boundaries, contracts, and extension points.
- `docs/MANUSCRIPT_ALIGNMENT.md`: manuscript + post hoc method alignment constraints.
- `docs/AGENT_HANDOFF.md`: current operational handoff for new agent sessions.
- `docs/GEMINI_CACHING.md`: Gemini/API caching support notes and benchmark run guidance.
- `docs/BATCH_ORCHESTRATION.md`: unattended matrix-runner workflow and config schema.
- `docs/METHODS_CHANGELOG.md`: ordered method-version and change-ID history.
- `docs/NOTEBOOKS.md`: legacy notebook map and migration guidance.
- `llms.txt`: compact LLM index of canonical files and commands.

## Environment
- **Python**: Use the conda environment in `environment.yml` (Python 3.11, scientific stack, stats, plotting, and `google-genai`).
- **Secrets**: Provide a `.env` file with the following keys before running notebooks or the CLI:
  - `GOOGLE_API_KEY`
  - `MOONDREAM_API_KEY` (or pass `--moondream-api-key`)
  - `REPLICATE_API_TOKEN`
- **Shell export note**: the CLI does not auto-load `.env`; export env vars in the active shell before running commands.
  - PowerShell: `Get-Content .env | ForEach-Object { if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }; $n,$v = $_ -split '=',2; Set-Item -Path "Env:$n" -Value $v.Trim().Trim('"').Trim("'") }`
- **GPU/CPU**: Workloads are CPU-bound by default; the code auto-resizes images to ≤1024 px as in the paper.

## Repository layout
- `01_*` … `16_*` notebooks: per-dataset pairs (`environment_and_data_prep` + `genai_segmentation_evaluation`).
- `ita_fitzpatrick_analysis.ipynb`: fairness/skin-tone analysis built on notebook outputs.
- `configs/`: prompt presets (`prompts.yaml`) used by the CLI.
- `configs/benchmarks/`: benchmark-matrix configs for unattended orchestration.
- `src/gemini_segmentation/`: modular Python package powering the CLI (data discovery, IO, Gemini client, metrics, fairness, prompts, and types).
- `results/`: created by the CLI to hold centralized outputs (safe to delete/regenerate).

## Legacy notebook workflow
The notebook pairs share the same structure and remain runnable without moving input files.

1. **Bootstrap**: First cell imports helpers, loads `.env`, checks dataset roots (e.g., `HK_SEG_DIR` for Hyper-Kvasir defaults to `segmented-images/` with `images/` and `masks/`), and optionally installs missing pip packages.
2. **Dataset discovery**: Confirms `images/` and `masks/` exist; builds `master_imagelist_<dataset>.txt` and optional pilot lists (`pilot50_*.txt`) in place.
3. **Segmentation primitives**: Defines `SegmentationMask` plus `parse_json`/`parse_segmentation_masks` to convert Gemini responses (bounding boxes + base64 masks) into full-resolution masks; overlay utilities draw masks/labels for QA.
4. **Gemini call**: `gemini_seg` resizes large images to ≤1024 px, encodes JPEG bytes, and calls `gemini-2.5-*` with configurable prompt, temperature, thinking budget, and safety settings. Calls were typically serial, with optional ad-hoc threads inside the notebook.
5. **Batch evaluation**: The evaluation notebook loops over `master_imagelist_*`, respects sampling and rate-limit sleeps, and can write per-image prediction JSON under the dataset root in `predictions_<model>/` without relocating inputs.
6. **Metrics**: Computes IoU/Dice per image, aggregates means/medians with bootstrap CIs, and records parse/time-out flags. Fairness analysis relies on these masks plus ground-truth masks.
7. **Fairness (notebook)**: `ita_fitzpatrick_analysis.ipynb` converts peri-lesional skin to L*a*b*, filters luminance (data-driven window with ≥2% area and ≥200 pixels), maps median ITA to Chardon/Fitzpatrick-like labels, and reports IoU/Dice summaries and statistical tests (Kruskal–Wallis, Dunn + Holm–Bonferroni, Cliff’s Delta, χ² success rates).

## Modular CLI
The CLI mirrors the notebook behavior while consolidating outputs under `results/<dataset>/<model>/<run_id>/` and preserving legacy inputs.

### Installation
```bash
conda env create -f environment.yml
conda activate gemini-segmentation
python -m pip install -e .
```
Ensure `.env` contains `GOOGLE_API_KEY`.
For CLI runs in PowerShell, export `.env` into process env vars before invoking the CLI.

For quick test runs without Conda, install the minimal test dependencies with:

```bash
pip install -r requirements-dev.txt
```

If tests fail with `site-packages/docx.py` and `ModuleNotFoundError: No module named 'exceptions'`, remove the legacy `docx` package and reinstall `python-docx`:

```bash
python -m pip uninstall -y docx
python -m pip install --upgrade python-docx
python -c "import docx; print(docx.__file__)"
```

### Commands
- `segment`: Run Gemini or Moondream on a dataset without changing source files.
  - Required: `segment <dataset_name> <dataset_root>` (must contain `images/` and `masks/`, plus any existing manifest files).
  - Key options: `--manifest` to target curated lists (e.g., `pilot50_*`) without rewriting `master_imagelist_*`; `--prompt`/`--prompt-file` or `--prompt-preset configs/prompts.yaml --preset-name <name>`; `--model-name`, `--temperature`, `--thinking-budget`, `--timeout`, `--max-retries`, `--workers`, `--rate-limit`, `--sample-size`, `--success-threshold`, `--bootstrap-method` (`bca` or `percentile`) and `--bootstrap-resamples` (default 5000) for summary stats; `--legacy-predictions` (emit notebook-style JSON near the inputs for back-compat); `--dry-run` (list pending images without calling the API).
  - Retry behavior: `--max-retries` defaults to `5` and retries timeout/parse-failure call outcomes.
  - Local request cache (all providers): `--local-cache/--no-local-cache` and `--local-cache-dir` to reuse prior request outputs across runs and reduce repeat API calls.
  - Gemini explicit context cache: `--gemini-explicit-cache/--no-gemini-explicit-cache` and `--gemini-cache-ttl` (seconds). Explicit cache is attempted automatically for Gemini models except robotics ER, where Gemini docs list caching as unsupported.
  - Provider selection: `--provider gemini` (default), `--provider moondream`, or `--provider replicate`. For Moondream, pass `--model-name moondream-3` (auto-applied if you keep the default) and optionally `--moondream-target` multiple times to request one API call per object label (otherwise the prompt text is used as the target). Use `--moondream-endpoint` for a local Moondream Station deployment or rely on `MOONDREAM_API_KEY`/`--moondream-api-key` for cloud calls.
  - Model selection: pass the Gemini model ID via `--model-name`. The default is `gemini-2.5-flash`, and you can explicitly target `gemini-2.5-flash-lite` or `gemini-robotics-er-1.5-preview` the same way.
  - Preset caveat: some presets in `configs/prompts.yaml` include a `model` field and can override CLI `--model-name`; avoid those presets when running strict cross-model comparisons.
  - Replicate example: `python -m gemini_segmentation.cli segment polyp /data/hk_seg --provider replicate --replicate-model-version your-org/seg-model:123abc --replicate-target polyp --replicate-instruction "Segment the visible polyp" --timeout 120 --workers 2`. The `--replicate-instruction` flags align 1:1 with `--replicate-target` entries to send label-specific instructions alongside each call.
- `fairness`: Compute ITA/Fitzpatrick statistics from a completed run: `fairness <dataset_name> <dataset_root> <results/.../run_id> [--manifest] [--sample-size] [--success-threshold] [--bootstrap-method] [--bootstrap-resamples]`. Defaults fall back to the stored `run_config.json` so fairness matches the originating segmentation subset and bootstrap settings.

### Benchmark matrix automation
Use the batch orchestrator to run full prompt-ablation matrices unattended across models and datasets:

- Entrypoint: `python -m gemini_segmentation.batch --config configs/benchmarks/ablation_robotics_canonical.yaml [flags]`
- Thin launcher (auto-loads `.env` into the shell process): `./scripts/launch_batch.sh --config configs/benchmarks/ablation_robotics_canonical.yaml`
- Optional local overrides: `--overrides configs/benchmarks/ablation_robotics_canonical.local.yaml`
- Stable run IDs: pass `--run-id`, otherwise default is `<study_id>_<YYYYMMDD-HHMMSS>`
- Filters: repeat `--only-dataset` and/or `--only-model` to run a subset.
- Optional fairness phase: add `--auto-fairness`
- Planning mode: `--dry-run` validates config/preflight and writes planned jobs without API calls.
- Windows convenience runner (polyp full dataset, 3x3, workers=10, live monitor): `.\scripts\run_polyp_full_3x3_w10.ps1`

Quick examples:

```bash
# Full matrix
python -m gemini_segmentation.batch \
  --config configs/benchmarks/ablation_robotics_canonical.yaml

# Only robotics ER on two datasets with fixed run-id
python -m gemini_segmentation.batch \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --only-model gemini-robotics-er-1.5-preview \
  --only-dataset polyp \
  --only-dataset derm_lesion \
  --run-id ablation_robotics_subset_20260218-1530

# Same matrix with fairness post-step enabled
python -m gemini_segmentation.batch \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --auto-fairness
```

PowerShell convenience run (full polyp 3x3 with live monitor):

```powershell
.\scripts\run_polyp_full_3x3_w10.ps1
```

Resume an interrupted run safely with the same run ID:

```powershell
.\scripts\run_polyp_full_3x3_w10.ps1 -RunId <existing_run_id>
```

Batch outputs are written under:

```
results/batches/<run_id>/
  resolved_config.json
  job_status.jsonl
  summary.json
  logs/*.log
```

During non-dry-run execution, the batch runner streams each active job's stdout/stderr to the terminal while also persisting full logs under `logs/*.log`.

The orchestrator executes jobs sequentially by default, continues after failures, and returns non-zero if any segment/fairness job fails.

Generated runtime artifacts (`results/`, `outputs/`, `artifacts/`) are ignored by default in `.gitignore` to keep repository status clean during long benchmark runs.

### Outputs (per run)
```
results/<dataset>/<model>/<run_id>/
  run_config.json           # exact parameters, prompts, model name, rate limit, workers
  predictions.jsonl         # one JSON record per image; rewritten atomically for resume safety
  masks/*.png               # binary masks per image (union of predicted objects)
  overlays/*.png            # preview overlays with bounding boxes and labels
  raw_responses/*.json      # raw Gemini payloads (box_2d, label, mask) preserved per image
  metrics.csv               # per-image IoU/Dice/success; updated after each image
  summary.csv               # rolling aggregates with bootstrap CIs
  fairness/                 # ITA distributions, Chardon labels, Kruskal–Wallis, Dunn+Holm, Cliff’s Delta, χ² success tables
```
- **Resume behavior**: If `predictions.jsonl` exists, the CLI rehydrates prior metrics/masks/overlays before processing remaining images. Writes are atomic per image to avoid corruption on interruption.
- **Legacy parity**: `--legacy-predictions` writes the notebook-style JSON (including bounding boxes and base64 masks) under `predictions_<model>/` near the dataset, matching the original consumers. Raw payloads are also saved under `raw_responses/` for byte-for-byte reproduction.
- **Replicate cost/latency**: Replicate provider calls incur the model’s metered costs and add mask-download latency. Use `--replicate-cache-dir <path>` to cache downloaded masks and skip re-fetching them when resuming or rerunning a job.

### Data flow
1. **Inputs** stay in place (no path changes): the CLI reads the same `images/`, `masks/`, and manifest text files the notebooks expect.
2. **Processing**: Each worker owns a Gemini client; a global rate limiter throttles calls. Images are resized to ≤1024 px before inference; model masks are resized to bounding-box extents, then to full resolution for metrics and overlays.
3. **Checkpointing**: After each image, the CLI rewrites `predictions.jsonl`, updates `metrics.csv`/`summary.csv`, and regenerates legacy prediction JSONs (when requested). Missing artifacts on resume are regenerated from stored masks.

### Prompt/model configuration
- Add or edit presets in `configs/prompts.yaml` to swap prompts/models without code changes.
- Built-in presets mirror the notebook prompts so CLI runs match legacy behavior:
  - `polyp` → colorectal polyp
  - `optic_disc_cup` → optic disc + optic cup
  - `derm_lesion` → skin lesion
  - `busi_mass` → BUSI breast mass
  - `pneumothorax_cxr` → chest X-ray pneumothorax
  - `lits_liver` / `lits_liver_mass` → LiTS liver vs. liver-mass targets
  - `laparoscopy_uterus_tools` → uterus + surgical tools in laparoscopy frames
  - `histopathology` → tissue regions of diagnostic interest (tumor, stroma, necrosis, etc.)
  - Structured prompt families (`label_v1`, `desc_v1`, `desc_neg_v1`) can be selected explicitly via presets (e.g., `polyp_desc_neg_v1`) or by repeating `--prompt-family`; `desc_neg_v1` appends negation-only guardrails.
- Override inline with `--prompt` or `--prompt-file`; the chosen text and model parameters are captured in `run_config.json` for reproducibility.

#### Prompt families and provider-aware shaping
- **Ablation families:** All tasks support three prompt families held constant across providers. `label_v1` uses only the class name; `desc_v1` adds modality context + short definition + stable attributes; `desc_neg_v1` equals `desc_v1` plus an exclusions block (the negation block is appended byte-for-byte so the only delta is the exclusions text). Enumerate families by repeating `--prompt-family`.
- **Provider-specific construction:**
  - **Gemini** receives the full JSON-schema prompt (keys `box_2d`, `mask`, `label`) built via the selected family.
  - **Moondream** ignores the JSON schema; it receives the target label(s) as the `object` string(s). Use `--moondream-target` overrides for multi-target tasks; otherwise the preset label(s) are sent. The adapter does not pass Gemini-style `temperature`/`thinking_budget` controls to Moondream segment calls.
  - **Replicate/Sa2VA** expects natural-language instructions rather than schemas. Each target gets a concise instruction like `Segment the optic disc.`; overrides can be provided with `--replicate-instruction` to align custom wording per label.
- **Caching/resume:** Cache keys include provider, prompt family, and a hash of the provider-specific payload to avoid collisions between JSON-schema prompts (Gemini) and object/instruction strings (Moondream/Replicate).

### Fairness analysis
- Consumes the masks/metrics from a completed `segment` run; does not rerun Gemini.
- Mirrors the notebook ITA pipeline: peri-lesional masking, luminance filtering (5–95th percentiles), ≥2% area and ≥200 valid-pixel thresholds, median ITA → Chardon labels → Light/Dark split.
- Reports per-group IoU/Dice means/medians with BCa CIs, Kruskal–Wallis, pairwise Dunn with Holm–Bonferroni correction, Cliff’s Delta effect sizes (with bootstrap CIs), and χ² comparisons of success rates.

### Paper artifacts
- **Tables/Figure placeholders:** `python -m gemini_segmentation.paper.make_all --results <path/to/long_form_results.csv>` (Parquet is also supported). The YAML registry in `configs/paper.yaml` documents required columns (task/model/prompt_strategy/iou/dice/success), display labels, and specifications for each table/figure. Artifacts land in `artifacts/` by default with `tables/*.csv|html|docx` and `figures/*.png|pdf`; override with `--artifacts` for CI.
- **Figure 1 best cases:** `python -m gemini_segmentation.paper.best_cases --config configs/figure1_best_cases.yaml` selects the highest-IoU image per configured dataset/target (persisting selection to `artifacts/figures/figure1_best_cases/selection.yaml`) and renders the montage to PDF/PNG in the same directory.
- **Fairness Figure 2 + Table 4:** `python -m gemini_segmentation.paper.figures --fairness-dir <results/.../fairness>` consumes the ITA fairness CSVs from a completed run and emits the four-panel plot plus Table 4 to `artifacts/fairness/` (paths are configurable via `--output-dir`).
- **Prompt-family comparison report (Markdown/HTML/PDF):** `python -m gemini_segmentation.paper.prompt_comparison --dataset polyp` reads completed run summaries and emits grouped model sections plus a consolidated PDF mega table with publication-style model/prompt labels. Rows include mean IoU/Dice (95% CI), median IoU/Dice, and success rate in `.md`, `.html`, `.pdf`, and `.csv` under `results/reports/`. Override run selection with `--gemini-run-id` / `--moondream-run-id` when needed.

## Extending the project
- **New datasets**: Add discovery helpers or manifest builders in `src/gemini_segmentation/data.py` if layout differs; keep `images/`/`masks/` naming stable to reuse the CLI.
- **New prompts/models**: Add YAML presets or extend `src/gemini_segmentation/models.py` to register additional providers while honoring the `segment_image` contract.
- **Custom metrics**: Extend `src/gemini_segmentation/metrics.py` to add new per-image metrics; aggregate outputs automatically join `metrics.csv`/`summary.csv`.
- **Fairness variations**: Modify `src/gemini_segmentation/fairness.py` to add new groupings or filters; outputs will land under the run’s `fairness/` directory.

## Key files to read
- **Notebook starters**: the first cell of any `NN_*_environment_and_data_prep.ipynb` for dataset discovery and segmentation helpers.
- **CLI entrypoint**: `src/gemini_segmentation/cli.py` (commands, arguments, workflow wiring).
- **Batch entrypoint**: `src/gemini_segmentation/batch.py` (matrix orchestration, strict preflight, status logs).
- **Gemini client + parsing**: `src/gemini_segmentation/models.py` and `src/gemini_segmentation/io.py` (request construction, response parsing, mask decoding, legacy exports).
- **Metrics and resume logic**: `src/gemini_segmentation/metrics.py` and `src/gemini_segmentation/io.py` (IoU/Dice, bootstrap, checkpointing, JSONL handling).
- **Fairness**: `src/gemini_segmentation/fairness.py` (ITA computation, statistical tests, output schemas).
- **Prompts**: `configs/prompts.yaml` (presets) or CLI flags for overrides.

## Notes on legacy vs. CLI
- **Inputs are unchanged**: keep all datasets where the notebooks expect them; the CLI reads the same locations.
- **Outputs are unified**: prefer the `results/` tree for new runs; enable `--legacy-predictions` only if older notebook consumers need the original JSON drops.
- **Tmux/parallelism**: you can still orchestrate multiple CLI runs with tmux; within a run, use `--workers` for thread-level parallelism guarded by the global rate limiter.
