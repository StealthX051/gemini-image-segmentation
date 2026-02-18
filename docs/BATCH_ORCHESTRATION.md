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
- `gemini_explicit_cache`, `gemini_cache_ttl`
- `thinking_budget`, `temperature`
- `legacy_predictions`
- `success_threshold`, `bootstrap_method`, `bootstrap_resamples`
- `manifest`

### `models` entry fields
- `name` (required)
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
- Keep machine-specific values in an untracked local override file, for example `configs/benchmarks/ablation_robotics_canonical.local.yaml`.
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
- robotics ER jobs do not use explicit Gemini context cache.

In `--dry-run` mode, provider API key checks are skipped, but filesystem/config checks still run.

## Execution Lifecycle
1. Load base config.
2. Optionally merge override config.
3. Expand `${ENV_VAR}` placeholders.
4. Build deterministic dataset × model job matrix.
5. Run strict preflight.
6. Write `results/batches/<run_id>/resolved_config.json`.
7. Execute segment commands sequentially.
8. Optionally execute fairness commands for discovered prompt-family run directories.
9. Append per-command status records to `job_status.jsonl`.
10. Write final `summary.json`.

Default failure mode is continue-on-failure, with non-zero process exit if any job fails.

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
  --overrides configs/benchmarks/ablation_robotics_canonical.local.yaml \
  --run-id ablation_robotics_20260218-1900 \
  --auto-fairness
```

Run with nohup:

```bash
nohup ./scripts/launch_batch.sh \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --overrides configs/benchmarks/ablation_robotics_canonical.local.yaml \
  --run-id ablation_robotics_20260218-1900 \
  > results/batches/ablation_robotics_20260218-1900/nohup.log 2>&1 &
```

Dry-run planning:

```bash
python -m gemini_segmentation.batch \
  --config configs/benchmarks/ablation_robotics_canonical.yaml \
  --only-model gemini-robotics-er-1.5-preview \
  --dry-run
```

PowerShell full polyp 3x3 run (workers=10 + live monitor):

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
