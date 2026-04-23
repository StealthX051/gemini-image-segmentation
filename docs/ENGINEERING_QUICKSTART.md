# Engineering Quickstart

This is the default fast-path document for Codex sessions and returning contributors.
Read this with `docs/DOCUMENTATION_MAP.md` before making changes.

For fresh-clone environment setup, use `docs/SETUP.md`.

## Repo Shape
- Core production lane: `src/gemini_segmentation/`
- Isolated study lane: `src/nanobanana_segmentation/`
- Legacy provenance lane: `notebooks/` and `ita_fitzpatrick_analysis.ipynb`

If a task does not explicitly involve NanoBanana or notebooks, assume you should stay in the core `gemini_segmentation` lane.

## Execution Baseline
- The repository lives on a Windows machine, but Codex may be running from bash/WSL over `/mnt/d/...`.
- Default shell for agent work: bash/WSL for file operations and repo inspection.
- On the original Windows-hosted repo, default Python/test execution is `conda run -n gemini_seg ...`.
- Do not rely on `conda activate gemini_seg` in non-interactive automation.
- On Linux or macOS fresh clones, use the documented `venv` bootstrap path in `docs/SETUP.md`.
- If you are on the original Windows-hosted repo and `conda` is unavailable in the current shell, switch to a Conda-initialized shell. Do not silently fall back to system `python`.
- Use PowerShell only for `.ps1` scripts or Windows-native tasks.
- Keep paths consistent within one command:
  - bash/WSL: `/mnt/d/Projects/...`
  - PowerShell: `D:\Projects\...` or repo-relative Windows paths
- Scratch environments and caches such as `.venv/`, `venv/`, `.tmp-bootstrap-venv/`, `.pytest_cache/`, and `__pycache__/` are local-only and gitignored.
- Respect `.gitattributes`: repo text files stay LF by default, while PowerShell scripts stay CRLF. Do not do broad line-ending rewrites.

## Verify The Interpreter
Run this before expensive commands if there is any doubt:

```bash
conda run -n gemini_seg python -c "import sys; print(sys.executable)"
```

## Secrets And Environment Loading
- The CLI does not auto-load `.env`.
- `scripts/launch_batch.sh` does auto-load `.env`.
- `scripts/run_polyp_full_3x3_w10.ps1` does auto-load `.env`.

Load `.env` manually when calling the CLI directly:

```bash
set -a
source .env
set +a
```

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $n, $v = $_ -split '=', 2
  Set-Item -Path "Env:$n" -Value $v.Trim().Trim('"').Trim("'")
}
```

## Common Commands

- Full Conda bootstrap: `conda env create -f environment.yml`
- Refresh an existing env: `conda run -n gemini_seg python scripts/bootstrap_env.py`
- Requirements-file wrapper: `python -m pip install -r requirements-dev.txt`
- Segment: `conda run -n gemini_seg python -m gemini_segmentation.cli segment <dataset_name> <dataset_root> [flags]`
- Fairness: `conda run -n gemini_seg python -m gemini_segmentation.cli fairness <dataset_name> <dataset_root> <run_dir> [flags]`
- Batch: `conda run -n gemini_seg python -m gemini_segmentation.batch --config <configs/benchmarks/*.yaml> [flags]`
- Tests: `conda run -n gemini_seg pytest -q`

PowerShell-only helper:

```powershell
conda run -n gemini_seg powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_polyp_full_3x3_w10.ps1
```

Notes:
- `scripts/run_polyp_full_3x3_w10.ps1` writes `configs/benchmarks/polyp_full_w10.local.yaml`, so it will dirty the worktree.
- When a clean worktree matters, prefer the Python batch entrypoint.

## Minimal Patch Expectations
- Prefer the smallest patch that fully solves the task.
- Do not refactor opportunistically.
- Deleting dead code, stale docs, or redundant files is encouraged when that is the smallest safe fix and does not remove required behavior.
- Preserve behavior unless the task explicitly changes behavior.
- Keep documentation concise and canonical: development docs should stay short; methods docs can stay long, but only read or edit them when semantics change.
- Subagents are allowed and often helpful for parallel exploration, but they are optional.
- End each substantive pass with the documentation updates needed to keep the repo accurate for the next agent session.

## Read By Task
- Always:
  - `docs/DOCUMENTATION_MAP.md`
  - `AGENTS.md`
- Read for setup or environment reproducibility work:
  - `docs/SETUP.md`
- Read for runtime changes:
  - `docs/ARCHITECTURE.md`
- Read for batch changes:
  - `docs/BATCH_ORCHESTRATION.md`
- Read for prompts, providers, fairness wording, or manuscript-facing outputs:
  - `docs/MANUSCRIPT_ALIGNMENT.md`
  - `docs/METHODS_CHANGELOG.md`
- Read for notebook work:
  - `docs/NOTEBOOKS.md`
- Read for the isolated NanoBanana lane:
  - `docs/NANOBANANA_STUDY.md`
  - `src/nanobanana_segmentation/AGENTS.md`
- Operational snapshots only:
  - `docs/AGENT_HANDOFF.md`
  - `docs/GEMINI_CACHING.md`
- History only:
  - `docs/history/VALIDATION_SNAPSHOTS.md`

## Safety Rails
- Do not modify source files under dataset roots.
- Write new runtime artifacts under `results/` unless explicitly asked otherwise.
- Preserve resume-safe write patterns.
- If you change CLI arguments or behavior, update `tests/test_cli.py`.
- If you change method semantics, update `docs/METHODS_CHANGELOG.md` in the same change.
