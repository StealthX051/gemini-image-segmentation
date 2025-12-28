# Gemini Image Segmentation

This repository contains two parallel workflows for evaluating Google Gemini models on medical-image segmentation tasks:

1. **Legacy Jupyter notebooks** that encode the original experiments per dataset (polyp, optic disc, dermatology, BUSI breast ultrasound, chest X-ray pneumothorax, LiTS liver lesions, histopathology, and laparoscopy). Each notebook’s first cell is a linear Python script with helper functions, dataset discovery, Gemini calls, and metrics.
2. **A modular CLI** (`python -m gemini_segmentation.cli`) that lifts the notebook logic into reusable modules with resumable runs, centralized outputs, and fairness analysis.

Read this document top-to-bottom when onboarding: it explains the environment, directory layout, how the notebooks work, how the CLI mirrors them, and where to extend the system (prompts, models, fairness, outputs).

## Environment
- **Python**: Use the conda environment in `environment.yml` (Python 3.11, scientific stack, stats, plotting, and `google-genai`).
- **Secrets**: Provide a `.env` file with `GOOGLE_API_KEY` before running notebooks or the CLI. For Moondream runs, also set `MOONDREAM_API_KEY` (or pass `--moondream-api-key`).
- **GPU/CPU**: Workloads are CPU-bound by default; the code auto-resizes images to ≤1024 px as in the paper.

## Repository layout
- `01_*` … `16_*` notebooks: per-dataset pairs (`environment_and_data_prep` + `genai_segmentation_evaluation`).
- `ita_fitzpatrick_analysis.ipynb`: fairness/skin-tone analysis built on notebook outputs.
- `configs/`: prompt presets (`prompts.yaml`) used by the CLI.
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
```
Ensure `.env` contains `GOOGLE_API_KEY`.

### Commands
- `segment`: Run Gemini or Moondream on a dataset without changing source files.
  - Required: `segment <dataset_name> <dataset_root>` (must contain `images/` and `masks/`, plus any existing manifest files).
  - Key options: `--manifest` to target curated lists (e.g., `pilot50_*`) without rewriting `master_imagelist_*`; `--prompt`/`--prompt-file` or `--prompt-preset configs/prompts.yaml --preset-name <name>`; `--model-name`, `--temperature`, `--thinking-budget`, `--timeout`, `--workers`, `--rate-limit`, `--sample-size`, `--success-threshold`, `--bootstrap-method` (`bca` or `percentile`) and `--bootstrap-resamples` (default 5000) for summary stats; `--legacy-predictions` (emit notebook-style JSON near the inputs for back-compat); `--dry-run` (list pending images without calling the API).
  - Provider selection: `--provider gemini` (default) or `--provider moondream`. For Moondream, pass `--model-name moondream-3` (auto-applied if you keep the default) and optionally `--moondream-target` multiple times to request one API call per object label (otherwise the prompt text is used as the target). Use `--moondream-endpoint` for a local Moondream Station deployment or rely on `MOONDREAM_API_KEY`/`--moondream-api-key` for cloud calls.
  - Model selection: pass the Gemini model ID via `--model-name`. The default is `gemini-2.5-flash`, and you can explicitly target `gemini-2.5-flash-lite` or `gemini-robotics-er-1.5-preview` the same way.
- `fairness`: Compute ITA/Fitzpatrick statistics from a completed run: `fairness <dataset_name> <dataset_root> <results/.../run_id> [--manifest] [--sample-size] [--success-threshold] [--bootstrap-method] [--bootstrap-resamples]`. Defaults fall back to the stored `run_config.json` so fairness matches the originating segmentation subset and bootstrap settings.

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
  - Hybrid variants (`*_hybrid`) provide parallel prompts with hybrid-expression wording; select them via `--preset-name polyp_hybrid` or by pairing `--preset-name polyp` with `--preset-branch hybrid`.
- Override inline with `--prompt` or `--prompt-file`; the chosen text and model parameters are captured in `run_config.json` for reproducibility.

### Fairness analysis
- Consumes the masks/metrics from a completed `segment` run; does not rerun Gemini.
- Mirrors the notebook ITA pipeline: peri-lesional masking, luminance filtering (5–95th percentiles), ≥2% area and ≥200 valid-pixel thresholds, median ITA → Chardon labels → Light/Dark split.
- Reports per-group IoU/Dice means/medians with BCa CIs, Kruskal–Wallis, pairwise Dunn with Holm–Bonferroni correction, Cliff’s Delta effect sizes (with bootstrap CIs), and χ² comparisons of success rates.

## Extending the project
- **New datasets**: Add discovery helpers or manifest builders in `src/gemini_segmentation/data.py` if layout differs; keep `images/`/`masks/` naming stable to reuse the CLI.
- **New prompts/models**: Add YAML presets or extend `src/gemini_segmentation/models.py` to register additional providers while honoring the `segment_image` contract.
- **Custom metrics**: Extend `src/gemini_segmentation/metrics.py` to add new per-image metrics; aggregate outputs automatically join `metrics.csv`/`summary.csv`.
- **Fairness variations**: Modify `src/gemini_segmentation/fairness.py` to add new groupings or filters; outputs will land under the run’s `fairness/` directory.

## Key files to read
- **Notebook starters**: the first cell of any `NN_*_environment_and_data_prep.ipynb` for dataset discovery and segmentation helpers.
- **CLI entrypoint**: `src/gemini_segmentation/cli.py` (commands, arguments, workflow wiring).
- **Gemini client + parsing**: `src/gemini_segmentation/models.py` and `src/gemini_segmentation/io.py` (request construction, response parsing, mask decoding, legacy exports).
- **Metrics and resume logic**: `src/gemini_segmentation/metrics.py` and `src/gemini_segmentation/io.py` (IoU/Dice, bootstrap, checkpointing, JSONL handling).
- **Fairness**: `src/gemini_segmentation/fairness.py` (ITA computation, statistical tests, output schemas).
- **Prompts**: `configs/prompts.yaml` (presets) or CLI flags for overrides.

## Notes on legacy vs. CLI
- **Inputs are unchanged**: keep all datasets where the notebooks expect them; the CLI reads the same locations.
- **Outputs are unified**: prefer the `results/` tree for new runs; enable `--legacy-predictions` only if older notebook consumers need the original JSON drops.
- **Tmux/parallelism**: you can still orchestrate multiple CLI runs with tmux; within a run, use `--workers` for thread-level parallelism guarded by the global rate limiter.

