# Agent Notes for `gemini-image-segmentation`

## Scope
This file applies to the entire repository.

## Development guidelines
- Follow the repository README for background, layout, and workflows before making changes.
- Prefer small, focused edits that maintain notebook/CLI parity and avoid broad refactors unless required for correctness.
- Keep public code paths typed and documented; add concise docstrings or comments for non-obvious logic.
- Preserve existing logging patterns; use the standard `logging` module instead of `print` for operational messages.
- Align with the conda environment in `environment.yml`; avoid introducing new dependencies without strong justification.

## Testing
- After code changes, run a quick smoke test: `python -m compileall src`.
- If a change touches CLI behavior or parsing logic, prefer adding or running targeted unit checks when available.

## Outputs and paths
- The CLI centralizes artifacts under `results/<dataset>/<model>/<run_id>/`; avoid changing this layout unless explicitly requested.
- When adjusting mask or bounding-box handling, ensure outputs stay aligned with both notebook expectations and CLI consumers.

## Pull requests
- Summarize behavior changes clearly and note any user-visible effects on prompts, manifests, outputs, or statistics.
