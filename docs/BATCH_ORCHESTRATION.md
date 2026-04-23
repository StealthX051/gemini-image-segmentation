# Batch Orchestration

This document describes the config-driven benchmark runner used for unattended prompt-ablation and robotics ER studies.

## Entrypoints
- Python module: `python -m gemini_segmentation.batch --config <path>`
- Thin shell launcher: `./scripts/launch_batch.sh --config <path>`
- Windows convenience runner: `.\scripts\run_polyp_full_3x3_w10.ps1`

The shell launcher exists only to load `.env` into the active process and delegate to the Python module.

## CLI Flags
- `--config <path>`: required benchmark config YAML.
- `--overrides <path>`: optional local override YAML, deep-merged into the base config.
- `--run-id <id>`: optional fixed run identifier. Default: `<study_id>_<YYYYMMDD-HHMMSS>`.
- `--only-dataset <name>`: repeat to include selected dataset names only.
- `--only-model <model>`: repeat to include selected model names only.
- `--auto-fairness`: run fairness immediately after successful segment jobs.
- `--dry-run`: validate and emit planned commands/status without executing API calls.
- `--stop-on-failure`: stop on first failed segment/fairness command.

## Config Schema (`schema_version: 1`)
Top-level keys:
- `schema_version` (required): currently `1`.
- `study_id` (required): used in default run-id generation.
- `results_dir` (optional): default `results`.
- `defaults` (optional): baseline job settings.
- `models` (required): model matrix entries.
- `datasets` (required): dataset matrix entries.

### `defaults` fields
- `provider`: `gemini`, `moondream`, or `replicate` (default `gemini`).
- `prompt_families`: list of prompt families (default `label_v1`, `desc_v1`, `desc_neg_v1`).
- `timeout`, `max_retries`, `workers`, `sample_size`, `rate_limit`
- `local_cache`, `local_cache_dir`
- `gemini_explicit_cache`, `gemini_cache_ttl`, `gemini_agentic_vision`
- `thinking_budget`, `temperature` (Gemini-only segment command options; not emitted for Moondream/Replicate jobs)
- Replicate fields: `replicate_model_version`, `replicate_targets`, `replicate_instructions`, `replicate_cache_dir`
- `legacy_predictions`
- `success_threshold`, `bootstrap_method`, `bootstrap_resamples`
- `manifest`

### `models` entry fields
- `name` (required)
- `api_model_name` (optional): actual backend model identifier when `name` is being used as the output/report label
- Optional overrides for any `defaults` field

### `datasets` entry fields
- `name` (required): CLI dataset name.
- `root` (required): dataset root containing `images/` and `masks/`.
- Optional overrides for any `defaults` field.

## Environment Placeholders
Config strings can include `${ENV_VAR}` placeholders. They are expanded before validation.

Example:

```yaml
datasets:
  - name: polyp
    root: ${POLYP_DATASET_ROOT}
```

If a placeholder variable is missing, the run exits before any job starts.

## Local Override Pattern
- Commit base matrix config, for example `configs/benchmarks/ablation_robotics_canonical.yaml`.
- Create machine-specific values in an untracked local override file (for example `<local_override.yaml>`) by copying from `configs/benchmarks/ablation_robotics_canonical.local.example.yaml`.
- Merge behavior:
  - mappings are merged recursively,
  - scalar values are replaced,
  - lists are replaced by the override list.

## Preflight Validation
Before execution, the runner validates:
- provider API keys are available (`GOOGLE_API_KEY`, `MOONDREAM_API_KEY`, `REPLICATE_API_TOKEN` as needed),
- every dataset root exists and has `images/` + `masks/`,
- explicit manifests exist when provided,
- prompt families are non-empty and valid,
- Replicate jobs define `replicate_model_version`,
- Replicate `replicate_instructions` are only used when `replicate_targets` are present,
- Replicate `replicate_targets` and `replicate_instructions` have matching cardinality when both are provided,
- models without explicit Gemini cache support must disable it,
- Gemini agentic vision is only enabled for `gemini-robotics-er-1.6-preview`.

In `--dry-run` mode, provider API key checks are skipped, but filesystem/config checks still run.
Preflight does not verify provider account credits/billing state or model-version permission scope; Replicate can still fail at runtime with `429` throttling or `422 Invalid version or not permitted`.

## Execution Lifecycle
1. Load base config.
2. Optionally merge override config.
3. Expand `${ENV_VAR}` placeholders.
4. Build deterministic dataset × model job matrix.
5. Run strict preflight.
6. Write `results/batches/<run_id>/resolved_config.json`.
7. Execute segment commands sequentially (provider-specific assembly: Gemini receives `--thinking-budget`/`--temperature`, optional `--gemini-agentic-vision`, and optional `--output-model-name` when the batch label differs from the API model; Moondream/Replicate do not receive Gemini-only flags; Replicate additionally receives `--replicate-model-version`, repeated `--replicate-target`, repeated `--replicate-instruction`, and optional `--replicate-cache-dir`).
8. Optionally execute fairness commands for discovered prompt-family run directories.
9. Append per-command status records to `job_status.jsonl`.
10. Write final `summary.json`.

Default failure mode is continue-on-failure, with non-zero process exit if any job fails.
During non-dry-run execution, child command output is mirrored to both the terminal and per-job log files for single-terminal monitoring. For Replicate jobs, fairness run discovery resolves filesystem-safe model directory tokens generated from `replicate_model_version` (and also checks the legacy raw path form for backward compatibility).
For Gemini Robotics-ER 1.6 agentic ablations, `name` remains the output/report label (for example `gemini-robotics-er-1.6-preview-agentic`) while `api_model_name` carries the actual API model ID (`gemini-robotics-er-1.6-preview`).

## Replicate Runtime Gating Notes
- For Replicate SA2VA, treat funded account credits/payment method as a run prerequisite.
- If account-level create-prediction throttling is active, lower concurrency and increase interval in config overrides:
  - `workers: 1`
  - `rate_limit: 12` (increase to `15` if throttling persists)
- Replicate model version IDs must be exact and accessible to the token used for the run.

## Output Files
All orchestration metadata for a batch run is stored under:

```text
results/batches/<run_id>/
  resolved_config.json
  job_status.jsonl
  summary.json
  logs/*.log
```

`job_status.jsonl` records one JSON object per executed (or planned, in dry-run) command.

## Unattended Usage Examples
Run in tmux:

```bash
tmux new -s seg_batch
./scripts/launch_batch.sh \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --overrides configs/benchmarks/ablation_robotics_canonical.local.example.yaml \
  --run-id ablation_robotics_20260218-1900 \
  --auto-fairness
```

Run with nohup:

```bash
nohup ./scripts/launch_batch.sh \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --overrides configs/benchmarks/ablation_robotics_canonical.local.example.yaml \
  --run-id ablation_robotics_20260218-1900 \
  > results/batches/ablation_robotics_20260218-1900/nohup.log 2>&1 &
```

Dry-run planning:

```bash
python -m gemini_segmentation.batch \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --only-model gemini-robotics-er-1.6-preview-agentic \
  --dry-run
```

PowerShell full polyp canonical matrix run (workers=10 + live monitor):

```powershell
.\scripts\run_polyp_full_3x3_w10.ps1
```

Disable the live monitor and only wait for process completion:

```powershell
.\scripts\run_polyp_full_3x3_w10.ps1 -NoLiveMonitor
```

Resume by reusing the same run ID:

```powershell
.\scripts\run_polyp_full_3x3_w10.ps1 -RunId polyp_full_3x3_w10_YYYYMMDD-HHMMSS
```

The live monitor prints:
- status events from `job_status.jsonl`,
- heartbeat updates every 30 seconds,
- tail lines from the currently active job log.
