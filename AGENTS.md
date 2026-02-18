# AGENTS.md

Repository-level instructions for coding agents working in this project.

## Scope
- Applies to the entire repository.
- If a subdirectory later adds its own `AGENTS.md`, the deeper file is more specific for that subtree.

## First Steps
- Read `README.md` for the project workflow and CLI usage.
- Read `docs/AGENT_HANDOFF.md` for current operational caveats and recent changes.
- Read `docs/ARCHITECTURE.md` before changing runtime logic.
- Read `docs/MANUSCRIPT_ALIGNMENT.md` before changing providers, prompts, or manuscript-facing outputs.
- Read `docs/METHODS_CHANGELOG.md` to understand baseline vs post hoc method versions.
- Read `docs/NOTEBOOKS.md` before touching notebook-stage artifacts.

## Environment Setup
- `conda env create -f environment.yml`
- `conda activate gemini_seg`
- `python -m pip install -e .`
- `python -m pip install -r requirements-dev.txt`

## Core Commands
- Run segmentation: `python -m gemini_segmentation.cli segment <dataset_name> <dataset_root> [flags]`
- Run fairness: `python -m gemini_segmentation.cli fairness <dataset_name> <dataset_root> <run_dir> [flags]`
- Build paper artifacts: `python -m gemini_segmentation.paper.make_all --results <csv_or_parquet>`
- Build Figure 1 montage: `python -m gemini_segmentation.paper.best_cases --config configs/figure1_best_cases.yaml`
- Build fairness figure/table: `python -m gemini_segmentation.paper.figures --fairness-dir <results/.../fairness>`

## Repository Map
- `src/gemini_segmentation/cli.py`: CLI entrypoint and run orchestration.
- `src/gemini_segmentation/models.py`: provider adapters (`gemini`, `moondream`, `replicate`).
- `src/gemini_segmentation/io.py`: parsing, JSONL IO, overlays, mask encoding.
- `src/gemini_segmentation/data.py`: dataset discovery and manifest handling.
- `src/gemini_segmentation/metrics.py`: IoU/Dice, bootstrap CI, summary writing.
- `src/gemini_segmentation/fairness.py`: ITA grouping and statistical analysis.
- `src/gemini_segmentation/prompts.py`: prompt families and provider-specific shaping.
- `src/gemini_segmentation/paper/`: table/figure generation utilities.
- `configs/`: prompt presets and paper artifact registries.
- `tests/`: primary regression safety net.

## Non-Negotiable Invariants
- Keep dataset inputs immutable; do not alter source files under dataset roots.
- Write new run artifacts under `results/` unless explicitly asked otherwise.
- Preserve resume safety: avoid non-atomic prediction/metrics write patterns.
- Keep provider interfaces compatible with `segment()` returning `(masks, latency, parse_success, timed_out, raw_items)`.
- Preserve manuscript/post hoc prompt-ablation semantics (`label_v1`, `desc_v1`, `desc_neg_v1`) unless explicitly requested to revise methods.
- Preserve provider-aware prompt shaping: Gemini uses JSON-schema prompts, Moondream uses target labels, Replicate/Sa2VA uses instruction strings.
- If you change CLI args or behavior, update tests in `tests/test_cli.py`.

## Change-Specific Guidance
- Prompt-family changes: update `src/gemini_segmentation/prompts.py` and `configs/prompts.yaml` together, then run `pytest -q tests/test_prompts.py tests/test_cli.py`.
- Prompt-ablation changes: keep `desc_neg_v1` as `desc_v1` plus exclusions-only delta, and update `docs/MANUSCRIPT_ALIGNMENT.md` when wording/semantics change.
- Provider changes: keep output schema Gemini-compatible (`box_2d`, `mask`, `label`), then run `pytest -q tests/test_replicate_segmenter.py tests/test_cli.py`.
- Provider-expansion changes: update provider notes in `docs/MANUSCRIPT_ALIGNMENT.md` and `docs/ARCHITECTURE.md` in the same change.
- Metrics changes: run `pytest -q tests/test_metrics.py` and any affected CLI tests.
- Any method-level change: add an entry to `docs/METHODS_CHANGELOG.md` in the same change.
- Paper artifact changes: keep CSV/HTML/DOCX and PNG/PDF outputs stable unless intentionally revised, then run `pytest -q tests/test_paper.py tests/test_paper_figures.py tests/test_paper_best_cases.py`.
- Fairness changes: preserve file names under `fairness/` and run relevant fairness plus CLI tests.

## Notebook Policy
- Notebooks are legacy workflow records and often contain large saved outputs.
- Prefer implementing logic in `src/` and tests, then document notebook impact.
- Do not mass-rewrite notebook JSON or clear outputs unless explicitly requested.

## Data and Secrets
- Never commit secrets from `.env`.
- CLI runs require env vars to exist in the active shell process (a `.env` file alone is insufficient unless exported by the shell/user).
- Treat `data/`, `segmented-images/`, `outputs/`, and generated artifacts as local runtime assets.
- Keep `.gitignore` behavior intact unless there is a clear request to version new artifacts.

## Definition Of Done
- Code and docs updated in the correct module(s).
- Focused tests executed for touched behavior.
- `README.md` or docs updated when user-facing workflow changes.
- Final summary includes changed files and any unrun checks.
