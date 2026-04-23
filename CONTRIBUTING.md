# Contributing

## Development Environment
- Create the Conda env: `conda env create -f environment.yml`.
- For Codex or other non-interactive automation, prefer `conda run -n gemini_seg ...` instead of relying on `conda activate`.
- `environment.yml` is the full fresh-clone Conda bootstrap path.
- Refresh an existing env with the bootstrap script: `conda run -n gemini_seg python scripts/bootstrap_env.py`.
- Alternative manual install: `conda run -n gemini_seg python -m pip install -e .[dev,notebooks]`.
- Linux/macOS `venv` path: `python3 -m venv .venv && source .venv/bin/activate && python scripts/bootstrap_env.py`.
- Verify interpreter when needed: `conda run -n gemini_seg python -c "import sys; print(sys.executable)"`.
- CLI note: direct `python -m gemini_segmentation.cli ...` does not auto-load `.env`.
- Wrapper note: `scripts/launch_batch.sh` and `scripts/run_polyp_full_3x3_w10.ps1` do auto-load `.env`.
- Fresh clone setup guide: `docs/SETUP.md`.
- Repo hygiene note: `.venv/`, `venv/`, `.tmp-bootstrap-venv/`, `.pytest_cache/`, and `__pycache__/` are local-only and gitignored.
- Line-ending note: keep Python/YAML/Markdown/shell files as LF text and PowerShell scripts as CRLF text; let `.gitattributes` and `.editorconfig` do the work.

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
- Documentation ownership map: `docs/DOCUMENTATION_MAP.md`.
- Batch launcher script: `scripts/launch_batch.sh`.
- Regression tests: `tests/`.
- Legacy experiment notebooks: `notebooks/*.ipynb` (plus `ita_fitzpatrick_analysis.ipynb` currently at repo root).

## Local Validation
- Full test run: `conda run -n gemini_seg pytest -q`.
- Prompt/CLI-focused checks: `conda run -n gemini_seg pytest -q tests/test_prompts.py tests/test_cli.py`.
- Batch-runner checks: `conda run -n gemini_seg pytest -q tests/test_batch.py`.
- Metrics-focused checks: `conda run -n gemini_seg pytest -q tests/test_metrics.py`.
- Provider adapter checks: `conda run -n gemini_seg pytest -q tests/test_replicate_segmenter.py`.
- Replicate parity checks: `conda run -n gemini_seg pytest -q tests/test_replicate_segmenter.py tests/test_cli.py tests/test_batch.py tests/test_prompts.py`.
- Paper/fairness artifact checks: `conda run -n gemini_seg pytest -q tests/test_paper.py tests/test_paper_figures.py tests/test_paper_best_cases.py`.

## Contribution Rules
- Keep changes scoped to the task; avoid broad notebook JSON rewrites.
- If deletion is the cleanest non-breaking fix, remove dead code, stale files, or redundant documentation instead of leaving them in place.
- When changing CLI arguments, update parser/help behavior and `tests/test_cli.py`.
- When changing prompt families or defaults, update both `src/gemini_segmentation/prompts.py` and `configs/prompts.yaml`.
- For manuscript-facing method changes, update `docs/MANUSCRIPT_ALIGNMENT.md` in the same PR.
- For any method-level change, add an entry to `docs/METHODS_CHANGELOG.md` in the same PR.
- When changing output schemas or file names, update dependent paper/fairness code and tests.
- For matrix orchestration changes, update `docs/BATCH_ORCHESTRATION.md`, `README.md`, and `docs/ENGINEERING_QUICKSTART.md` when execution guidance changed.
- For substantive changes, leave the relevant docs more accurate than you found them; do not defer obvious documentation drift.
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
