# AGENTS.md

Repository instructions for Codex and other coding agents.

## Scope
- Applies to the entire repository.
- A deeper `AGENTS.md` overrides this file for its subtree.

## Read Order
- Always read `docs/DOCUMENTATION_MAP.md` and `docs/ENGINEERING_QUICKSTART.md` first.
- Then read only the docs relevant to the task:
  - environment or bootstrap work: `docs/SETUP.md`
  - runtime logic: `docs/ARCHITECTURE.md`
  - batch runner: `docs/BATCH_ORCHESTRATION.md`
  - prompts, providers, or manuscript-facing outputs: `docs/MANUSCRIPT_ALIGNMENT.md` and `docs/METHODS_CHANGELOG.md`
  - notebooks: `docs/NOTEBOOKS.md`
  - NanoBanana lane: `docs/NANOBANANA_STUDY.md` and `src/nanobanana_segmentation/AGENTS.md`
- `docs/AGENT_HANDOFF.md` and `docs/GEMINI_CACHING.md` are operational snapshots. Read them when the task needs current caveats, not by default.
- `docs/history/VALIDATION_SNAPSHOTS.md` is history only.

## Codex-Specific Rules
- Always use the OpenAI developer documentation MCP server if you need to work with the OpenAI API, ChatGPT Apps SDK, Codex, or related docs without me having to explicitly ask.
- Prefer minimal, behavior-preserving patches over proactive refactors.
- Deletion is allowed when it is the cleanest non-breaking fix. Remove dead code, stale compatibility paths, and redundant or outdated documentation instead of preserving them by default.
- Keep code and documentation academically rigorous, professional, and complete, but do not expand scope without a clear need.
- Keep development docs concise and actionable. Put manuscript/method detail in the methods docs, not in the fast-path engineering docs.
- Use subagents when helpful for parallel exploration or verification, but do not spawn them reflexively.
- After every substantive change, update the relevant documentation in the same pass unless the change is purely internal and leaves no user-facing or maintainer-facing delta.

## Execution Rules
- This repository lives on a Windows machine. Many Codex sessions run from bash/WSL over `/mnt/...`.
- On the original Windows-hosted repo, prefer `conda run -n gemini_seg ...` for non-interactive Python commands. Do not assume `conda activate gemini_seg` works in automation.
- On Linux or macOS fresh clones, the supported `venv` bootstrap path is documented in `docs/SETUP.md`.
- If you are on the original Windows-hosted repo and `conda` is unavailable in the current shell, switch to a shell where Anaconda is initialized. Do not silently substitute system `python`.
- Use bash/WSL for normal shell work. Use PowerShell only for `.ps1` scripts or clearly Windows-native tasks.
- Keep path styles consistent inside one command: bash/WSL uses `/mnt/...`; PowerShell uses `D:\...` or repo-relative Windows paths.
- Repo-local scratch environments and caches such as `.venv/`, `venv/`, `.tmp-bootstrap-venv/`, `.pytest_cache/`, and `__pycache__/` are local-only and gitignored.
- Respect `.gitattributes`: keep Python, shell, Markdown, YAML, TOML, JSON, and notebook files as LF text, and keep PowerShell scripts as CRLF text. Do not mass-normalize line endings outside the files you touch.
- The CLI does not auto-load `.env`.
- `scripts/launch_batch.sh` and `scripts/run_polyp_full_3x3_w10.ps1` do auto-load `.env`.
- `scripts/run_polyp_full_3x3_w10.ps1` writes `configs/benchmarks/polyp_full_w10.local.yaml`; prefer the Python batch entrypoint when a clean worktree matters.

## Setup And Common Commands
- Full Conda bootstrap: `conda env create -f environment.yml`
- Refresh an existing env: `conda run -n gemini_seg python scripts/bootstrap_env.py`
- Linux/macOS `venv` bootstrap: `python3 -m venv .venv && source .venv/bin/activate && python scripts/bootstrap_env.py`
- Requirements-file wrapper: `python -m pip install -r requirements-dev.txt`
- Verify interpreter: `conda run -n gemini_seg python -c "import sys; print(sys.executable)"`
- Segment: `conda run -n gemini_seg python -m gemini_segmentation.cli segment <dataset_name> <dataset_root> [flags]`
- Fairness: `conda run -n gemini_seg python -m gemini_segmentation.cli fairness <dataset_name> <dataset_root> <run_dir> [flags]`
- Batch: `conda run -n gemini_seg python -m gemini_segmentation.batch --config <configs/benchmarks/*.yaml> [flags]`
- PowerShell benchmark: `conda run -n gemini_seg powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_polyp_full_3x3_w10.ps1`

## Non-Negotiable Invariants
- Keep dataset inputs immutable; do not alter source files under dataset roots.
- Write new run artifacts under `results/` unless explicitly asked otherwise.
- Preserve resume safety; avoid non-atomic prediction or metrics writes.
- Keep provider interfaces compatible with `segment()` returning `(masks, latency, parse_success, timed_out, raw_items)`.
- Preserve prompt-family semantics for `label_v1`, `desc_v1`, and `desc_neg_v1` unless the task explicitly changes methods.
- Preserve provider-aware prompt shaping: Gemini uses JSON-schema prompts, Moondream uses target labels, Replicate/Sa2VA uses instruction strings.

## Change-Specific Guidance
- Prompt-family changes: update `src/gemini_segmentation/prompts.py`, `configs/prompts.yaml`, `docs/MANUSCRIPT_ALIGNMENT.md`, and the affected prompt/CLI tests together.
- Provider changes: keep the Gemini-compatible output schema (`box_2d`, `mask`, `label`) and run the relevant provider plus CLI tests.
- CLI argument or behavior changes: update `tests/test_cli.py`.
- Metrics changes: run `tests/test_metrics.py` and any affected CLI tests.
- Batch changes: update `docs/BATCH_ORCHESTRATION.md` and run `tests/test_batch.py` plus affected CLI tests.
- Paper artifact changes: keep output filenames and schemas stable unless intentionally revised, then update docs and paper tests.
- Any method-level change: add an entry to `docs/METHODS_CHANGELOG.md` in the same change.

## Notebook And Study Lanes
- Notebooks are legacy provenance assets. Prefer implementing logic in `src/` and tests.
- Do not mass-rewrite notebook JSON or clear outputs unless explicitly requested.
- NanoBanana is intentionally isolated. Do not route NanoBanana behavior through `gemini_segmentation` unless explicitly requested.

## Data And Secrets
- Never commit secrets from `.env`.
- Treat `data/`, `segmented-images/`, `outputs/`, and generated artifacts as local runtime assets.
- Keep `.gitignore` behavior intact unless there is a clear request to version new artifacts.

## Definition Of Done
- Docs and code updated in the correct place with minimal scope.
- Obsolete code or documentation removed when it is safe and clearly improves maintainability.
- Focused validation run for changed behavior, or the reason it was not run is stated clearly.
- User-facing workflow changes are reflected in `README.md` and the relevant canonical docs.
