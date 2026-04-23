# Gemini Image Segmentation

This repository contains two primary segmentation workflows plus one isolated study lane:

1. **Legacy Jupyter notebooks** that encode the original experiments per dataset (polyp, optic disc, dermatology, BUSI breast ultrasound, chest X-ray pneumothorax, LiTS liver lesions, histopathology, and laparoscopy). Each notebook’s first cell is a linear Python script with helper functions, dataset discovery, Gemini calls, and metrics.
2. **A modular CLI** (`python -m gemini_segmentation.cli`) that lifts the notebook logic into reusable modules with resumable runs, centralized outputs, and fairness analysis.
3. **An isolated NanoBanana package** (`src/nanobanana_segmentation/`) for retrieval/tool-ablation studies and a separate segmentation microservice path that does not modify `gemini_segmentation` runtime contracts.

## Agentic development files
- `AGENTS.md`: repository-level instructions for coding agents.
- `docs/ENGINEERING_QUICKSTART.md`: fast-path engineering setup and task-based doc routing.
- `docs/SETUP.md`: canonical fresh-clone setup and reproducible environment guide.
- `.codex/config.toml`: repo-local Codex defaults (model/sandbox/doc discovery).
- `CONTRIBUTING.md`: contributor workflow and validation commands.
- `docs/DOCUMENTATION_MAP.md`: canonical documentation ownership map and update policy.
- `docs/ARCHITECTURE.md`: module boundaries, contracts, and extension points.
- `docs/MANUSCRIPT_ALIGNMENT.md`: manuscript + post hoc method alignment constraints.
- `docs/AGENT_HANDOFF.md`: dated operational snapshot for current caveats and recent validation notes.
- `docs/history/VALIDATION_SNAPSHOTS.md`: archived point-in-time run IDs, metrics, and smoke command variants.
- `docs/GEMINI_CACHING.md`: dated cache-support snapshot and repo caching notes.
- `docs/BATCH_ORCHESTRATION.md`: unattended matrix-runner workflow and config schema.
- `docs/METHODS_CHANGELOG.md`: ordered method-version and change-ID history.
- `docs/NOTEBOOKS.md`: legacy notebook map and migration guidance.
- `llms.txt`: compact secondary index for non-Codex tooling; `AGENTS.md` and `docs/ENGINEERING_QUICKSTART.md` are the primary agent entrypoints.

## Environment
- **Python**: `environment.yml` bootstraps Python 3.11, `pip`, and the Cairo runtime; the editable package install then pulls the Python dependency set from `pyproject.toml`.
- **Codex/automation default**: for non-interactive runs, prefer `conda run -n gemini_seg ...` instead of relying on `conda activate`.
- **Windows + WSL note**: this repository lives on Windows. In bash/WSL sessions, use `/mnt/...` paths and bash commands. Use PowerShell only for `.ps1` scripts or clearly Windows-native tasks.
- **Fresh clone setup**: use `docs/SETUP.md` for the canonical Conda and `venv` setup flows.
- **Secrets**: Provide a `.env` file with the following keys before running notebooks or the CLI:
  - `GOOGLE_API_KEY`
  - `MOONDREAM_API_KEY` (or pass `--moondream-api-key`)
  - `REPLICATE_API_TOKEN`
- **Shell export note**: the CLI does not auto-load `.env`; export env vars in the active shell before running commands.
  - Bash/WSL: `set -a; source .env; set +a`
  - PowerShell: `Get-Content .env | ForEach-Object { if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }; $n,$v = $_ -split '=',2; Set-Item -Path "Env:$n" -Value $v.Trim().Trim('"').Trim("'") }`
- **Wrapper note**:
  - `python -m gemini_segmentation.cli ...` does not auto-load `.env`
  - `./scripts/launch_batch.sh ...` does auto-load `.env`
  - `.\scripts\run_polyp_full_3x3_w10.ps1` does auto-load `.env`
- **GPU/CPU**: Workloads are CPU-bound by default; the code auto-resizes images to ≤1024 px as in the paper.

## Repository layout
- `notebooks/`: legacy experiment notebooks (`01_*` … `16_*` dataset pairs plus exploratory notebooks).
- `ita_fitzpatrick_analysis.ipynb`: legacy fairness/skin-tone analysis notebook (currently at repository root).
- `configs/`: prompt presets (`prompts.yaml`) used by the CLI.
- `configs/benchmarks/`: benchmark-matrix configs for unattended orchestration.
- `src/gemini_segmentation/`: modular Python package powering the CLI (data discovery, IO, Gemini client, metrics, fairness, prompts, and types).
- `results/`: created by the CLI to hold centralized outputs (safe to delete/regenerate).

## Legacy notebook workflow
Notebook pairs share one pattern and remain runnable in-place:
1. **Bootstrap**: load helpers/env vars, validate dataset roots (`images/`, `masks/`), optionally install missing packages.
2. **Discovery/manifests**: build `master_imagelist_<dataset>.txt` and optional pilot lists (`pilot50_*.txt`) in place.
3. **Segmentation primitives**: parse Gemini (`box_2d`, base64 mask) into full-resolution masks and render QA overlays.
4. **Inference**: call `gemini-2.5-*` on ≤1024px resized images with configurable prompt/temperature/thinking/safety; typically serial with optional ad-hoc threading.
5. **Evaluation**: iterate manifests with optional sampling/rate-limit sleeps; optionally emit `predictions_<model>/` JSON under dataset roots.
6. **Metrics/fairness inputs**: compute IoU/Dice + bootstrap summaries + parse/timeout flags from predicted and GT masks.
7. **Notebook fairness**: `ita_fitzpatrick_analysis.ipynb` computes peri-lesional ITA groupings and reports IoU/Dice + nonparametric statistics (Kruskal–Wallis, Dunn+Holm-Bonferroni, Cliff’s Delta, χ² success rates).

## Modular CLI
The CLI mirrors the notebook behavior while consolidating outputs under `results/<dataset>/<model>/<prompt_key>/<run_id>/` and preserving legacy inputs.

### Installation
```bash
conda env create -f environment.yml
```
The Conda path above is the full fresh-clone bootstrap. For interactive local work you can still activate the env manually, but for agent automation and reproducible command snippets, prefer `conda run -n gemini_seg ...`.

If you change dependency metadata later and want to refresh an existing env:

```bash
conda run -n gemini_seg python scripts/bootstrap_env.py
```

Alternative `venv` path:

```bash
python3 -m venv .venv
source .venv/bin/activate
python scripts/bootstrap_env.py
```

On Debian/Ubuntu-like systems, install `python3-venv` first if `python3 -m venv` is unavailable.

Ensure `.env` contains API keys; CLI commands require env vars in the active shell process (the CLI does not auto-load `.env`).

Verify the interpreter when needed:

```bash
conda run -n gemini_seg python -c "import sys; print(sys.executable)"
```

Fast verification:

```bash
conda run -n gemini_seg gemini-seg --help
conda run -n gemini_seg nanobanana-study --help
conda run -n gemini_seg python -m gemini_segmentation.cli --help
conda run -n gemini_seg pytest -q tests/test_config.py tests/test_prompts.py tests/test_batch.py
```

If tests fail with `site-packages/docx.py` and `ModuleNotFoundError: No module named 'exceptions'`, remove the legacy `docx` package and reinstall `python-docx`:

```bash
conda run -n gemini_seg python -m pip uninstall -y docx
conda run -n gemini_seg python -m pip install --upgrade python-docx
conda run -n gemini_seg python -c "import docx; print(docx.__file__)"
```

### Commands
- `segment`: Run Gemini, Moondream, or Replicate on a dataset without changing source files.
  - Required: `segment <dataset_name> <dataset_root>` (must contain `images/` and `masks/`, plus any existing manifest files).
  - Prompt selection: `--prompt`, `--prompt-file`, or `--prompt-preset configs/prompts.yaml --preset-name <name> [--preset-branch legacy]`.
  - Execution controls: `--model-name`, `--provider`, `--temperature`, `--thinking-budget`, `--timeout`, `--max-retries` (default `5`), `--workers`, `--rate-limit`, `--sample-size`, `--results-dir`, `--run-id`, `--dry-run`.
  - Metrics controls: `--success-threshold`, `--bootstrap-method` (`bca|percentile`), `--bootstrap-resamples` (default `5000`).
  - Cache controls: `--local-cache/--no-local-cache`, `--local-cache-dir`; Gemini explicit cache: `--gemini-explicit-cache/--no-gemini-explicit-cache`, `--gemini-cache-ttl`.
  - Back-compat/output controls: `--manifest`, `--legacy-predictions`, `--replicate-cache-dir`.
  - Provider-specific target/instruction controls: `--moondream-target`, `--moondream-endpoint`, `--moondream-api-key`, `--replicate-model-version`, `--replicate-target`, `--replicate-instruction`.
  - Explicit Gemini cache is auto-skipped for robotics ER (`gemini-robotics-er-1.5-preview`).
  - Provider selection: `--provider gemini` (default), `--provider moondream`, or `--provider replicate`. For Moondream, pass `--model-name moondream-3` (auto-applied if you keep the default) and optionally `--moondream-target` multiple times to request one API call per object label (otherwise the prompt text is used as the target). Use `--moondream-endpoint` for a local Moondream Station deployment or rely on `MOONDREAM_API_KEY`/`--moondream-api-key` for cloud calls.
  - Model selection: pass the Gemini model ID via `--model-name`. The default is `gemini-2.5-flash`, and you can explicitly target `gemini-2.5-flash-lite` or `gemini-robotics-er-1.5-preview` the same way.
  - Preset caveat: some presets in `configs/prompts.yaml` include a `model` field and can override CLI `--model-name`; avoid those presets when running strict cross-model comparisons.
  - Replicate example: `python -m gemini_segmentation.cli segment polyp /data/hk_seg --provider replicate --replicate-model-version bytedance/sa2va-26b-image:addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f --replicate-target polyp --replicate-instruction "Segment the visible polyp" --timeout 120 --workers 2`. The `--replicate-instruction` flags align 1:1 with `--replicate-target` entries to send label-specific instructions alongside each call.
  - Replicate operational caveat (2026-02-19): accounts without payment method/credits can be throttled aggressively (for example, create-prediction `429` with ~`6/min` and burst `1` as observed in validation). For smoke tests on throttled accounts, start with `--workers 1 --rate-limit 12` (increase to `15` if needed).
  - Replicate version caveat: `--replicate-model-version` must be a valid, accessible version ID. Invalid or inaccessible IDs return `422 Invalid version or not permitted`.
  - Replicate validation status (2026-02-19): end-to-end SA2VA implementation is validated, including a completed full polyp 3-family batch run (`replicate_sa2va_polyp_full_20260219-162118`) with 1000 predictions per family. See `docs/history/VALIDATION_SNAPSHOTS.md` for exact run paths and per-family metrics.
- `fairness`: Compute dermoscopy fairness artifacts from a completed run: `fairness <dataset_name> <dataset_root> <results/.../run_id> [--audit-mode legacy|enhanced] [--enhanced-config configs/fairness_enhanced.yaml] [--enhanced-stage all|core|sensitivity|augment] [--enhanced-feature-profile balanced|full|minimal] [--enhanced-augment-columns <csv>] [--enhanced-resume|--no-enhanced-resume] [--enhanced-checkpoint-every N] [--enhanced-workers-auto|--no-enhanced-workers-auto] [--manifest] [--sample-size] [--workers] [--success-threshold] [--bootstrap-method] [--bootstrap-resamples]`.
  - `--audit-mode legacy` (default): preserves the historical ITA/Fitzpatrick CSV schema under `<run_dir>/fairness/`.
  - `--audit-mode enhanced`: runs Fairness Audit v2 and writes expanded artifacts under `<run_dir>/fairness_enhanced/`.
- `python scripts/prepare_ima_plusplus.py`: Build an IMA++ CLI-compatible dataset root (`images/`, `masks/`, `masks_all/`, `master_imagelist_ima_plusplus.txt`) with canonical GT policy `STAPLE -> majority vote -> single annotator only` while retaining all mask metadata for sensitivity analyses.
- `python scripts/analyze_ima_plusplus_sensitivity.py --run-dir <results/.../run_id> [--dataset-root <path>]`: Generate IMA++ sensitivity artifacts against MV consensus and annotator masks under `<run_dir>/ima_plusplus_sensitivity/`.

### Benchmark matrix automation
Use the batch orchestrator for unattended prompt-ablation matrices:

- Entrypoint: `python -m gemini_segmentation.batch --config configs/benchmarks/ablation_robotics_canonical.yaml [flags]`
- Thin launcher (auto-loads `.env` into the shell process): `./scripts/launch_batch.sh --config configs/benchmarks/ablation_robotics_canonical.yaml`
- Optional local overrides: `--overrides <your_local_override.yaml>` (seed from `configs/benchmarks/ablation_robotics_canonical.local.example.yaml`)
- Stable run IDs: pass `--run-id`, otherwise default is `<study_id>_<YYYYMMDD-HHMMSS>`
- Filters: repeat `--only-dataset` and/or `--only-model` to run a subset.
- Optional fairness phase: `--auto-fairness`
- Planning mode: `--dry-run` validates config/preflight and writes planned jobs without API calls.
- Windows convenience runner (polyp full dataset, 3x3, workers=10, live monitor): `.\scripts\run_polyp_full_3x3_w10.ps1`
  - PowerShell only
  - auto-loads `.env`
  - assumes the correct Conda Python is available on `PATH`, or should be wrapped with `conda run -n gemini_seg powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_polyp_full_3x3_w10.ps1`
  - writes `configs/benchmarks/polyp_full_w10.local.yaml`, which dirties the worktree

Quick examples:

```bash
# Full matrix
python -m gemini_segmentation.batch \
  --config configs/benchmarks/ablation_robotics_canonical.yaml

# Filtered subset with fixed run-id
python -m gemini_segmentation.batch \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --only-model gemini-robotics-er-1.5-preview \
  --only-dataset polyp \
  --only-dataset derm_lesion \
  --run-id ablation_robotics_subset_20260218-1530

# Replicate/Sa2VA batch using local override config
python -m gemini_segmentation.batch \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --overrides configs/benchmarks/replicate_sa2va_polyp.local.yaml \
  --run-id replicate_sa2va_polyp_full_20260219-162118
```

PowerShell convenience run (full polyp 3x3 with live monitor):

```powershell
.\scripts\run_polyp_full_3x3_w10.ps1
```

Resume an interrupted run safely with the same run ID:

```powershell
.\scripts\run_polyp_full_3x3_w10.ps1 -RunId <existing_run_id>
```

For additional unattended patterns (nohup/tmux, stop-on-failure, dry-run planning), see `docs/BATCH_ORCHESTRATION.md`.

Batch outputs:

```
results/batches/<run_id>/
  resolved_config.json
  job_status.jsonl
  summary.json
  logs/*.log
```

- Non-dry-run behavior: mirror child stdout/stderr to terminal and persist per-job logs under `logs/*.log`.
- Failure behavior: default continue-on-failure; process exits non-zero if any segment/fairness job fails.
- Generated runtime folders (`results/`, `outputs/`, `artifacts/`, `results_nanobanana/`, `artifacts_nanobanana/`) are ignored in `.gitignore`.

### Outputs (per run)
```
results/<dataset>/<model>/<prompt_key>/<run_id>/
  run_config.json           # exact parameters, prompts, model name, rate limit, workers
  predictions.jsonl         # one JSON record per image; rewritten atomically for resume safety
  masks/*.png               # binary masks per image (union of predicted objects)
  overlays/*.png            # preview overlays with bounding boxes and labels
  raw_responses/*.json      # raw Gemini payloads (box_2d, label, mask) preserved per image
  metrics.csv               # per-image IoU/Dice/success; updated after each image
  summary.csv               # rolling aggregates with bootstrap CIs
  fairness/                 # Legacy fairness artifacts (unchanged schema)
  fairness_enhanced/        # Enhanced fairness v2 artifacts (analysis_frame, dedup maps, trends, sensitivities)
```
- **Resume behavior**: If `predictions.jsonl` exists, the CLI rehydrates prior metrics/masks/overlays before processing remaining images. Writes are atomic per image to avoid corruption on interruption.
- **Legacy parity**: `--legacy-predictions` writes the notebook-style JSON (including bounding boxes and base64 masks) under `predictions_<model>/` near the dataset, matching the original consumers. Raw payloads are also saved under `raw_responses/` for byte-for-byte reproduction.
- **Replicate cost/latency**: Replicate provider calls incur the model’s metered costs and add mask-download latency. Use `--replicate-cache-dir <path>` to cache downloaded masks and skip re-fetching them when resuming or rerunning a job.
- **Replicate model path token**: Replicate model-version strings are normalized into a filesystem-safe model directory token under `results/<dataset>/<model>/<prompt_key>/...` while the exact model version remains recorded in `run_config.json` (`replicate_model_version`).
- **Replicate billing gate**: low/zero-credit Replicate accounts can block or heavily throttle smoke/full runs. Treat account funding/credits as an execution prerequisite for Replicate validation.

### Data flow
1. **Inputs unchanged**: read existing `images/`, `masks/`, and manifests in place.
2. **Processing**: worker-local clients + process-wide rate limiter; resize to ≤1024 px before inference; reproject mask outputs to full resolution.
3. **Checkpointing**: per-image atomic writes to `predictions.jsonl`, incremental `metrics.csv`/`summary.csv`, and artifact regeneration on resume.

### Prompt/model configuration
- Add or edit presets in `configs/prompts.yaml` to swap prompts/models without code changes.
- Built-in presets mirror the notebook prompts so CLI runs match legacy behavior:
  - `polyp` → colorectal polyp
  - `optic_disc_cup` → optic disc + optic cup
  - `derm_lesion` → skin lesion
  - `ima_plusplus` → skin lesion (IMA++ dermoscopy with STAPLE-first canonical GT policy handled in prep script)
  - `busi_mass` → BUSI breast mass
  - `pneumothorax_cxr` → chest X-ray pneumothorax
  - `lits_liver` / `lits_liver_mass` → LiTS liver vs. liver-mass targets
  - `laparoscopy_uterus_tools` → uterus + surgical tools in laparoscopy frames
  - `histopathology` → tissue regions of diagnostic interest (tumor, stroma, necrosis, etc.)
  - Structured prompt families (`label_v1`, `desc_v1`, `desc_neg_v1`) can be selected explicitly via presets (e.g., `polyp_desc_neg_v1`) or by repeating `--prompt-family`; `desc_neg_v1` appends negation-only guardrails.
- Override inline with `--prompt` or `--prompt-file`; the chosen text and model parameters are captured in `run_config.json` for reproducibility.

### IMA++ Preparation And Sensitivity
- Prep command (recommended):
  - `python scripts/prepare_ima_plusplus.py --download-zenodo --download-split-csvs --download-images --download-images-mode api --isic-api-workers 12`
- Download/order notes:
  - Zenodo first (`segs.zip`, `seg_metadata.csv`, `img_metadata.csv`), then ISIC image pulls by `ISIC_id`.
  - DOI `10.5281/zenodo.14201692` currently resolves to record `14201693`; script defaults are pinned to live record URLs and remain overrideable via `--*-url`.
  - API mode uses ISIC v2 with retries/backoff/skip-existing; older `isic auth login` docs are obsolete for current CLI flows.
- Outputs/policy:
  - CLI-ready `images/`, `masks/`; sidecar `masks_all/` + metadata/split manifests for sensitivity analyses.
  - Canonical GT policy: `STAPLE -> majority vote -> single annotator only (exactly one annotator mask)`.
- Sensitivity command:
  - `python scripts/analyze_ima_plusplus_sensitivity.py --run-dir <results/.../run_id> --dataset-root data/IMAplusplus_cli`
  - Writes `metrics_mv.csv`, `metrics_annotators.csv`, `per_image_annotator_summary.csv`, `summary_overall.csv`, `summary_by_tool.csv`, `summary_by_skill_level.csv` under `<run_dir>/ima_plusplus_sensitivity/`.

#### Prompt families and provider-aware shaping
- **Ablation families:** All tasks support three prompt families held constant across providers. `label_v1` uses only the class name; `desc_v1` adds modality context + short definition + stable attributes; `desc_neg_v1` equals `desc_v1` plus an exclusions block (the negation block is appended byte-for-byte so the only delta is the exclusions text). Enumerate families by repeating `--prompt-family`.
- **Provider-specific construction:**
  - **Gemini** receives the full JSON-schema prompt (keys `box_2d`, `mask`, `label`) built via the selected family.
  - **Moondream** ignores the JSON schema; it receives the target label(s) as the `object` string(s). Use `--moondream-target` overrides for multi-target tasks; otherwise the preset label(s) are sent. The adapter does not pass Gemini-style `temperature`/`thinking_budget` controls to Moondream segment calls.
  - **Replicate/Sa2VA** expects natural-language instructions rather than schemas. Defaults are prompt-family aware (`label_v1` label-only, `desc_v1` descriptor-enriched, `desc_neg_v1` descriptor plus exclusions). Overrides can be provided with `--replicate-instruction` to align custom wording per label.
- **Caching/resume:** Cache keys include provider, prompt family, and a hash of the provider-specific payload to avoid collisions between JSON-schema prompts (Gemini) and object/instruction strings (Moondream/Replicate).

### Fairness analysis
- Consumes the masks/metrics from a completed `segment` run; does not rerun Gemini.
- Legacy mode mirrors the historical notebook ITA/Fitzpatrick workflow.
- Legacy mode remains default for paper-parity workflows (`--audit-mode legacy`).
- Enhanced mode (`--audit-mode enhanced`) adds:
  - canonical `analysis_frame.parquet`,
  - SHA/pHash-based deduplication maps/reports,
  - endpoint effect-size tables (IoU + success RD/RR/OR),
  - covariate-adjusted success effects using predictive margins (`covadj_success_t050_effects.*` + `covadj_model_spec.json`),
  - adjusted model component contributions with CI-based significance flags (`covadj_component_effects.csv`),
  - continuous ITA trend plots/tables,
  - threshold and dedup sensitivity outputs.
  - default ITA method: global non-lesional region sampling with aggregated `L*`/`b*` ITA (documented per run in `fairness_enhanced/ita_method_note.json|md`).
  - interpretation note: trend models use continuous ITA (`ita_deg`); predictive-margin adjusted effects/components use binary cutoff (`ita_binary`).
- Runtime controls: stage selection (`all|core|sensitivity|augment`), feature profile (`balanced|full|minimal`), targeted augmentation columns, resumable checkpoints, checkpoint cadence, and optional memory-aware worker auto-capping.
- Runtime telemetry: `runtime_profile.json` includes per-stage wall/CPU/RAM/throughput; `core` intentionally defers sensitivity-only outputs to `sensitivity`.
- Enhanced defaults are configured in `configs/fairness_enhanced.yaml`; override with `--enhanced-config`.
- Deep references:
  - `docs/FAIRNESS_ENHANCED_METHODS.md` (implementation-aligned methods)
  - `docs/FAIRNESS_ENHANCED.md` (operational quick reference)

### Paper artifacts
- **Tables/Figure placeholders:** `python -m gemini_segmentation.paper.make_all --results <csv_or_parquet>`; registry in `configs/paper.yaml`; outputs to `artifacts/tables/*.csv|html|docx` and `artifacts/figures/*.png|pdf` (override `--artifacts`).
- **Figure 1 best cases:** `python -m gemini_segmentation.paper.best_cases --config configs/figure1_best_cases.yaml`; writes montage PDF/PNG and `selection.yaml` under `artifacts/figures/figure1_best_cases/`.
- **Fairness Figure 2 + Table 4:** `python -m gemini_segmentation.paper.figures --fairness-dir <results/.../fairness>`; writes combined + panel figure exports (`png|pdf|svg`) and Table 4 under `artifacts/fairness/` (override `--output-dir`).
- **Enhanced fairness artifacts/report:** `python -m gemini_segmentation.paper.figures_enhanced --fairness-enhanced-dir <results/.../fairness_enhanced>`; writes figures (`png|svg|pdf`), tables (`csv|html|md`), and combined report (`md|html|pdf|docx`) under `artifacts/fairness_enhanced/`.
- **Prompt-family comparison report:** `python -m gemini_segmentation.paper.prompt_comparison --dataset <name>`; writes `.md|.html|.pdf|.csv` under `results/reports/`; supports `--gemini-run-id`, `--moondream-run-id`, `--replicate-run-id`.

## Extending the project
- **New datasets**: Add discovery helpers or manifest builders in `src/gemini_segmentation/data.py` if layout differs; keep `images/`/`masks/` naming stable to reuse the CLI.
- **New prompts/models**: Add YAML presets or extend `src/gemini_segmentation/models.py` to register additional providers while honoring the `segment_image` contract.
- **Custom metrics**: Extend `src/gemini_segmentation/metrics.py` to add new per-image metrics; aggregate outputs automatically join `metrics.csv`/`summary.csv`.
- **Fairness variations**:
  - Legacy workflow lives in `src/gemini_segmentation/fairness.py` (`fairness/` outputs).
  - Enhanced workflow lives in `src/gemini_segmentation/fairness_enhanced/` (`fairness_enhanced/` outputs).

## Key files to read
- **Notebook starters**: the first cell of any `notebooks/NN_*_environment_and_data_prep*.ipynb` for dataset discovery and segmentation helpers.
- **CLI entrypoint**: `src/gemini_segmentation/cli.py` (commands, arguments, workflow wiring).
- **Batch entrypoint**: `src/gemini_segmentation/batch.py` (matrix orchestration, strict preflight, status logs).
- **Gemini client + parsing**: `src/gemini_segmentation/models.py` and `src/gemini_segmentation/io.py` (request construction, response parsing, mask decoding, legacy exports).
- **Metrics and resume logic**: `src/gemini_segmentation/metrics.py` and `src/gemini_segmentation/io.py` (IoU/Dice, bootstrap, checkpointing, JSONL handling).
- **Fairness**:
  - `src/gemini_segmentation/fairness.py` (legacy ITA computation/statistics/output schema),
  - `src/gemini_segmentation/fairness_enhanced/` (enhanced fairness v2 pipeline).
- **Prompts**: `configs/prompts.yaml` (presets) or CLI flags for overrides.

## Notes on legacy vs. CLI
- **Inputs unchanged**: CLI reads the same dataset roots/notebook manifests.
- **Outputs unified**: prefer `results/`; use `--legacy-predictions` only for notebook-era consumers.
- **Parallelism**: orchestrate multi-run concurrency externally (for example `tmux`); use per-run `--workers` with rate limiting.

## NanoBanana package (isolated study lane)
This repo also includes an isolated NanoBanana package under `src/nanobanana_segmentation/`.

- Microservice:
  - `uvicorn nanobanana_segmentation.service.main:app --host 0.0.0.0 --port 8000`
  - Endpoints: `POST /v1/segment`, `GET /health`, `GET /metrics`.
- Study runner:
  - `python -m nanobanana_segmentation.study.runner --config configs/nanobanana/study.yaml --stage stage1`
  - Stages: `stage0`, `stage1`, `stage2`, `all`.
- Default artifact roots:
  - `results_nanobanana/`
  - `artifacts_nanobanana/`
- Git hygiene for generated run data:
  - If these folders were previously committed, ignore rules will not retroactively untrack them.
  - Run once to remove them from the index while keeping local files:
    - `git rm --cached -r results_nanobanana artifacts_nanobanana`
- Full operational details:
  - `docs/NANOBANANA_STUDY.md`
