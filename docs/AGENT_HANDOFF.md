# Agent Handoff (Current State)

Last updated: 2026-02-22.

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
- `python -m gemini_segmentation.paper.figures --fairness-dir <.../fairness>` expects fairness CSVs in that exact directory and now writes `figure2.png`, `figure2.pdf`, and `figure2.svg`.
- Current Figure 2 panel behavior: left IoU panel uses full `0.0–1.0` range; right IoU panel is thresholded (`IoU >= 0.5`) and truncated at `0.5`.

## Critical Gotchas
- `.env` is not auto-loaded into shell process env vars by the CLI. Export env vars in the active shell before running.
- Replicate fairness discovery in batch uses the Replicate output model label (`replicate_model_version`) rather than the matrix display name.
- Replicate preflight validates token presence but cannot validate account credits/billing state; runtime can still fail with `429` create-prediction throttling on unfunded accounts.
- Replicate model-version IDs must be exact and accessible; invalid/inaccessible versions return `422 Invalid version or not permitted`.
- Using `--prompt-preset configs/prompts.yaml --preset-name polyp` can override `--model-name` because preset `polyp` sets a model in YAML.
  - For strict model comparisons, either:
    - avoid preset model-bearing entries, or
    - use family-only selection with explicit `--model-name`.
- Some datasets have RGB ground-truth masks; metrics now normalize masks to single-channel before IoU/Dice.
- Do not pass placeholder paths like `<your_run>` to `paper.figures`; pass the actual run fairness folder containing `fairness_results.csv`.

## Current Replicate Validation State (2026-02-19)
- Focused parity/unit tests pass for Replicate-targeted suites.
- Direct 10-image Replicate smoke run succeeded after account funding was enabled.
- Full polyp 3-family Replicate batch run succeeded with run ID:
  - `replicate_sa2va_polyp_full_20260219-162118`
- Current validated SA2VA 26B version string for smoke/full runs:
  - `bytedance/sa2va-26b-image:addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f`
- Completed full-run artifact roots:
  - `results/polyp/bytedance_sa2va-26b-image_addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f/label_v1-33499fdf/replicate_sa2va_polyp_full_20260219-162118`
  - `results/polyp/bytedance_sa2va-26b-image_addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f/desc_v1-a60ffa93/replicate_sa2va_polyp_full_20260219-162118`
  - `results/polyp/bytedance_sa2va-26b-image_addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f/desc_neg_v1-b99674e5/replicate_sa2va_polyp_full_20260219-162118`
- Full-run summary highlights (`summary.csv`):
  - `label_v1`: mean IoU `0.7147`, mean Dice `0.7909`, success rate `0.775` (1000 predictions).
  - `desc_v1`: mean IoU `0.7119`, mean Dice `0.7882`, success rate `0.772` (1000 predictions).
  - `desc_neg_v1`: mean IoU `0.7231`, mean Dice `0.7944`, success rate `0.796` (1000 predictions).

## Recommended Replicate Smoke Command (Funded Accounts / Standard Parity)
Use parity settings (`workers=10`, `rate_limit=0.5`, `max_retries=5`):

```powershell
$env:POLYP_DATASET_ROOT = "D:\Projects\gemini_image_segmentation\segmented-images"
python -m gemini_segmentation.cli segment polyp "$env:POLYP_DATASET_ROOT" `
  --provider replicate `
  --replicate-model-version bytedance/sa2va-26b-image:addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f `
  --prompt-family label_v1 `
  --sample-size 10 `
  --workers 10 `
  --rate-limit 0.5 `
  --max-retries 5 `
  --local-cache `
  --local-cache-dir results/.request_cache `
  --replicate-cache-dir results/.replicate_mask_cache `
  --run-id replicate_sa2va_smoke_YYYYMMDD-HHMMSS
```

## Fallback Smoke Command (Throttled Accounts)
If provider-side throttling persists, use one worker with a conservative global interval:

```powershell
$env:POLYP_DATASET_ROOT = "D:\Projects\gemini_image_segmentation\segmented-images"
python -m gemini_segmentation.cli segment polyp "$env:POLYP_DATASET_ROOT" `
  --provider replicate `
  --replicate-model-version bytedance/sa2va-26b-image:addd35cc4f8e0761836ff1e4af324bd7b1f4fa67ee3d384b69202cb288a7dd4f `
  --prompt-family label_v1 `
  --sample-size 10 `
  --workers 1 `
  --rate-limit 12 `
  --max-retries 2 `
  --local-cache `
  --local-cache-dir results/.request_cache `
  --replicate-cache-dir results/.replicate_mask_cache `
  --run-id replicate_sa2va_smoke_20260219_slow
```

## Recommended Smoke Command Pattern
Use one model per run and repeat `--prompt-family`:

```bash
python -m gemini_segmentation.cli segment polyp segmented-images \
  --provider gemini \
  --model-name gemini-robotics-er-1.5-preview \
  --prompt-family label_v1 \
  --prompt-family desc_v1 \
  --prompt-family desc_neg_v1 \
  --sample-size 100 \
  --workers <cpus-minus-2> \
  --rate-limit 0.5 \
  --max-retries 5 \
  --local-cache \
  --local-cache-dir results/.request_cache \
  --no-gemini-explicit-cache
```

## Recommended Batch Pattern
Use config-driven orchestration for unattended benchmark runs:

```bash
python -m gemini_segmentation.batch \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --run-id ablation_robotics_<timestamp>
```

- Add `--overrides configs/benchmarks/ablation_robotics_canonical.local.yaml` for machine-specific dataset roots/manifests.
- Add `--only-model` and `--only-dataset` to run subsets.
- Add `--auto-fairness` to run fairness immediately after successful segmentation jobs.
- For non-interrupting planning/verification, use `--dry-run`.

## Comparison Reporting
- Use `python -m gemini_segmentation.paper.prompt_comparison --dataset polyp` to generate grouped model/prompt comparison artifacts under `results/reports/` (`.md`, `.html`, `.pdf`, `.csv`).
- Override run selection with `--gemini-run-id`, `--moondream-run-id`, and `--replicate-run-id` as needed.
- Report tables include mean IoU/Dice with 95% CIs, median IoU/Dice, and success rate, plus a consolidated PDF mega table for cross-model prompt-family comparison.

## PowerShell Convenience Runner
For a full polyp 3x3 run (three models × three prompt families) with `workers=10` and live monitoring:

```powershell
.\scripts\run_polyp_full_3x3_w10.ps1
```

- Reuse `-RunId <existing_run_id>` to resume interrupted runs.
- Use `-NoLiveMonitor` to suppress heartbeat/log-tail monitor output.

## Key Files For New Agents
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/BATCH_ORCHESTRATION.md`
- `docs/MANUSCRIPT_ALIGNMENT.md`
- `docs/GEMINI_CACHING.md`
- `docs/METHODS_CHANGELOG.md`
- `src/gemini_segmentation/cli.py`
- `src/gemini_segmentation/batch.py`
- `src/gemini_segmentation/models.py`
- `src/gemini_segmentation/cache.py`
- `src/gemini_segmentation/metrics.py`
- `tests/test_cli.py`
- `tests/test_batch.py`
- `tests/test_metrics.py`
