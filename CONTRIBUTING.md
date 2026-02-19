# Contributing

## Development Environment
- Preferred path: `conda env create -f environment.yml` then `conda activate gemini_seg`.
- Install package in editable mode: `python -m pip install -e .`.
- Install test/dev dependencies: `python -m pip install -r requirements-dev.txt`.

## Common Dependency Pitfall
- If imports fail with `site-packages/docx.py` and `ModuleNotFoundError: No module named 'exceptions'`, the legacy `docx` package is installed instead of `python-docx`.
- Fix command 1: `python -m pip uninstall -y docx`
- Fix command 2: `python -m pip install --upgrade python-docx`
- Verify command: `python -c "import docx; print(docx.__file__)"`
- Expected verify output path includes `site-packages/docx/__init__.py` (package directory), not `site-packages/docx.py`.

## Code Layout
- Runtime package: `src/gemini_segmentation/`.
- Prompt and paper configuration: `configs/`.
- Benchmark matrix configs: `configs/benchmarks/`.
- Batch launcher script: `scripts/launch_batch.sh`.
- Regression tests: `tests/`.
- Legacy experiment notebooks: `notebooks/*.ipynb` (plus `ita_fitzpatrick_analysis.ipynb` currently at repo root).

## Local Validation
- Full test run: `pytest -q`.
- Prompt/CLI-focused checks: `pytest -q tests/test_prompts.py tests/test_cli.py`.
- Batch-runner checks: `pytest -q tests/test_batch.py`.
- Metrics-focused checks: `pytest -q tests/test_metrics.py`.
- Provider adapter checks: `pytest -q tests/test_replicate_segmenter.py`.
- Replicate parity checks: `pytest -q tests/test_replicate_segmenter.py tests/test_cli.py tests/test_batch.py tests/test_prompts.py`.
- Paper/fairness artifact checks: `pytest -q tests/test_paper.py tests/test_paper_figures.py tests/test_paper_best_cases.py`.

## Contribution Rules
- Keep changes scoped to the task; avoid broad notebook JSON rewrites.
- When changing CLI arguments, update parser/help behavior and `tests/test_cli.py`.
- When changing prompt families or defaults, update both `src/gemini_segmentation/prompts.py` and `configs/prompts.yaml`.
- For manuscript-facing method changes, update `docs/MANUSCRIPT_ALIGNMENT.md` in the same PR.
- For any method-level change, add an entry to `docs/METHODS_CHANGELOG.md` in the same PR.
- When changing output schemas or file names, update dependent paper/fairness code and tests.
- For matrix orchestration changes, update `docs/BATCH_ORCHESTRATION.md`, `README.md`, and `llms.txt` in the same PR.
- Prefer adding tests for behavior changes rather than only updating documentation.

## Data and Secrets
- Do not commit API keys or `.env` contents.
- Do not commit large generated artifacts from datasets unless explicitly requested.
- Preserve dataset roots and source input files; write outputs to `results/` or configured artifact directories.
- Generated runtime directories (`results/`, `outputs/`, `artifacts/`) are ignored by default via `.gitignore`.
- If generated artifacts were tracked historically, untrack them with `git rm -r --cached <path>` instead of deleting local runtime files.

## Replicate Troubleshooting
- `429 Request was throttled` on create-prediction:
  - lower concurrency (`--workers 1`) and increase spacing (`--rate-limit 12` or higher),
  - confirm account payment method/credits are active.
- `422 Invalid version or not permitted`:
  - verify the exact `--replicate-model-version` string and model access permissions for the token.

## Pull Request Checklist
- Relevant tests pass locally or skipped tests are clearly explained.
- User-facing workflow changes are reflected in `README.md`.
- New assumptions, flags, or side effects are documented in touched modules or docs.
