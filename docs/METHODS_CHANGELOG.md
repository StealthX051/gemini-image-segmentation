# Methods Changelog

This changelog tracks method-level changes that affect manuscript interpretation, reproducibility, or experimental comparability.

## Current Effective Version
- `posthoc_v1` (provider expansion + prompt ablation families).

## Entries

### MTH-BASELINE-001
- Date: manuscript-era baseline (pre post hoc additions; exact date not pinned in repo docs).
- Status: historical baseline.
- Summary: baseline manuscript workflow centered on medical-image segmentation evaluation with Gemini prompts and JSON-schema mask outputs.
- Impact: reference point for comparing all post hoc method additions.
- Primary anchors: manuscript draft methods text and notebook-era workflow.

### MTH-POSTHOC-001
- Date: post hoc addition window (after manuscript draft; exact date tracked in commit history).
- Status: active.
- Summary: provider expansion to include Moondream and Replicate/Sa2VA adapters, plus Gemini model-ID flexibility including `gemini-robotics-er-1.5-preview`.
- Impact: enables cross-provider comparisons with provider-specific prompt shaping and run-config traceability.
- Code anchors: `src/gemini_segmentation/models.py`, `src/gemini_segmentation/cli.py`, `tests/test_replicate_segmenter.py`, `tests/test_cli.py`.

### MTH-POSTHOC-002
- Date: post hoc addition window (after manuscript draft; exact date tracked in commit history).
- Status: active.
- Summary: prompt ablation families formalized as `label_v1`, `desc_v1`, and `desc_neg_v1` with fixed semantics and provider-aware prompt materialization.
- Impact: supports controlled ablation over language specificity while preserving output schema expectations for Gemini and compatible behavior for other providers.
- Code anchors: `src/gemini_segmentation/prompts.py`, `configs/prompts.yaml`, `tests/test_prompts.py`, `tests/test_cli.py`.

## Update Protocol For New Method Changes
- Add a new `MTH-*` entry in this file describing what changed and why it matters.
- Update `docs/MANUSCRIPT_ALIGNMENT.md` if semantics, contracts, or provider behavior changed.
- Update method-facing docs in `README.md` when user-visible behavior changed.
- Add or adjust tests (`tests/test_cli.py`, `tests/test_prompts.py`, provider-specific tests) to lock behavior.
- Include the new changelog ID in PR/commit notes for traceability.
