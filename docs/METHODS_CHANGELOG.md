# Methods Changelog

This changelog tracks method-level changes that affect manuscript interpretation, reproducibility, or experimental comparability.

## Current Effective Version
- `posthoc_v9` (provider expansion + prompt ablation families + caching/retry hardening + provider-parameter parity + Replicate batch/instruction-shaping hardening + Replicate input-serialization/runtime validation hardening + standardized comparison reporting + Replicate-inclusive comparison report run selection + completed operational validation record + fairness Figure 2 panel-level export parity + IMA++ STAPLE-first canonical GT preparation with retained multi-mask sensitivity workflow + IMA++ acquisition/prep hardening for live Zenodo resolution, threaded API downloads, and split-manifest compatibility).

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

### MTH-POSTHOC-003
- Date: 2026-02-18.
- Status: active.
- Summary: added dual-layer caching controls for high-volume benchmark runs: (1) local deterministic request cache across providers; (2) Gemini explicit context cache for supported Gemini models.
- Impact: reduces repeated API spend during reruns/resume workflows while preserving ablation isolation via cache keys that include provider/model/prompt/image identity.
- Provider notes: Gemini docs list caching support for `gemini-2.5-flash` and `gemini-2.5-flash-lite`; robotics docs list `gemini-robotics-er-1.5-preview` caching as unsupported, so the implementation auto-falls back to no explicit Gemini cache for robotics ER.
- Code anchors: `src/gemini_segmentation/cache.py`, `src/gemini_segmentation/cli.py`, `src/gemini_segmentation/models.py`, `src/gemini_segmentation/types.py`, `src/gemini_segmentation/config.py`.

### MTH-POSTHOC-004
- Date: 2026-02-18.
- Status: active.
- Summary: added bounded retry handling (`max_retries`, default 5) for timeout/parse-failure outcomes and normalized multi-channel masks to single-channel before IoU/Dice computation.
- Impact: improves run robustness for malformed/partial model outputs and prevents metric crashes on RGB/RGBA mask files.
- Code anchors: `src/gemini_segmentation/cli.py`, `src/gemini_segmentation/metrics.py`, `src/gemini_segmentation/types.py`, `src/gemini_segmentation/config.py`, `tests/test_cli.py`, `tests/test_metrics.py`.

### MTH-POSTHOC-005
- Date: 2026-02-18.
- Status: active.
- Summary: standardized reproducible benchmark orchestration with a config-driven batch runner for prompt-ablation matrices and robotics ER benchmarking, including strict preflight checks, deterministic command assembly, and run-level status artifacts.
- Impact: improves unattended execution reproducibility and auditability across dataset/model matrices without changing prompt semantics, provider contracts, or metric definitions.
- Code anchors: `src/gemini_segmentation/batch.py`, `configs/benchmarks/ablation_robotics_canonical.yaml`, `scripts/launch_batch.sh`, `tests/test_batch.py`, `docs/BATCH_ORCHESTRATION.md`.

### MTH-POSTHOC-006
- Date: 2026-02-19.
- Status: active.
- Summary: hardened cross-provider parity by constraining non-Gemini segment calls to provider-native arguments (Moondream adapter invocation fallbacks; batch command assembly omits Gemini-only sampling controls for Moondream/Replicate), and added standardized post-run model-vs-prompt comparison reporting artifacts.
- Impact: reduces provider-parameter mismatch risk in Moondream runs, improves reproducible auditability for cross-model prompt-ablation comparisons, and standardizes manuscript-facing summary exports (mean/median IoU-Dice, 95% CIs, success rate) without changing segmentation metric definitions.
- Code anchors: `src/gemini_segmentation/models.py`, `src/gemini_segmentation/batch.py`, `src/gemini_segmentation/paper/prompt_comparison.py`, `tests/test_moondream_segmenter.py`, `tests/test_batch.py`, `tests/test_paper_prompt_comparison.py`.

### MTH-POSTHOC-007
- Date: 2026-02-19.
- Status: active.
- Summary: completed Replicate/Sa2VA production-parity hardening by adding batch-schema parity fields and strict preflight validation, provider-specific command forwarding, Replicate output-url extraction robustness across API output variants, and prompt-family-specific Replicate instruction shaping (`label_v1`, `desc_v1`, `desc_neg_v1`).
- Impact: closes batch-orchestration parity gaps for Replicate runs, improves resilience to Replicate output-shape variation, preserves manuscript prompt-ablation semantics for Replicate defaults, and improves resume reproducibility through stronger run-config handling.
- Code anchors: `src/gemini_segmentation/models.py`, `src/gemini_segmentation/batch.py`, `src/gemini_segmentation/prompts.py`, `src/gemini_segmentation/cli.py`, `tests/test_replicate_segmenter.py`, `tests/test_batch.py`, `tests/test_cli.py`, `tests/test_prompts.py`.

### MTH-POSTHOC-008
- Date: 2026-02-19.
- Status: active.
- Summary: hardened Replicate input serialization by switching SA2VA image input from raw `bytes` to provider-supported upload payloads with a data-URI fallback path, then documented runtime gating caveats discovered during smoke validation (account-credit throttling and strict version-permission checks).
- Impact: prevents `TypeError: Object of type bytes is not JSON serializable` request failures, preserves provider contract while improving cross-client compatibility, and makes Replicate smoke/full-run prerequisites explicit for reproducible execution planning.
- Code anchors: `src/gemini_segmentation/models.py`, `tests/test_replicate_segmenter.py`, `README.md`, `docs/ARCHITECTURE.md`, `docs/AGENT_HANDOFF.md`, `docs/BATCH_ORCHESTRATION.md`.

### MTH-POSTHOC-009
- Date: 2026-02-19.
- Status: active.
- Summary: completed Replicate/Sa2VA cross-provider reporting parity by adding explicit Replicate run selection support to the prompt-comparison reporting pipeline (`--replicate-run-id`) and recording successful end-to-end Replicate validation state (smoke + full polyp 3-family batch run) in operational docs.
- Impact: allows manuscript-style consolidated prompt-comparison reports to include Replicate rows deterministically by run ID, and improves reproducibility handoff by pinning a validated Replicate run and model version in project docs without changing segmentation metric definitions.
- Code anchors: `src/gemini_segmentation/paper/prompt_comparison.py`, `tests/test_paper_prompt_comparison.py`, `README.md`, `docs/AGENT_HANDOFF.md`, `llms.txt`.

### MTH-POSTHOC-010
- Date: 2026-02-26.
- Status: active.
- Summary: aligned fairness Figure 2 rendering with legacy derm notebook styling and added panel-level artifact exports so a single run writes the combined figure and each panel as standalone files (`PNG/PDF/SVG`).
- Impact: improves manuscript assembly flexibility and reproducible figure reuse without changing fairness statistics, thresholds, or metric definitions.
- Code anchors: `src/gemini_segmentation/paper/figures.py`, `tests/test_paper_figures.py`, `README.md`, `docs/AGENT_HANDOFF.md`, `docs/NOTEBOOKS.md`.

### MTH-POSTHOC-011
- Date: 2026-02-26.
- Status: active.
- Summary: added IMA++ dataset preparation and sensitivity-analysis tooling with deterministic canonical GT policy (`STAPLE -> majority vote -> single annotator only`) while retaining all masks + per-mask metadata for optional post-run sensitivity analyses.
- Impact: preserves existing CLI ingestion contracts (`images/` + `masks/`) while enabling manuscript-defensible consensus-first evaluation on IMA++ and transparent robustness checks against MV/annotator variation.
- Code anchors: `scripts/prepare_ima_plusplus.py`, `scripts/analyze_ima_plusplus_sensitivity.py`, `src/gemini_segmentation/prompts.py`, `configs/prompts.yaml`, `tests/test_prepare_ima_plusplus.py`, `tests/test_ima_plusplus_sensitivity.py`, `tests/test_prompts.py`, `tests/test_cli.py`, `README.md`, `docs/MANUSCRIPT_ALIGNMENT.md`, `docs/ARCHITECTURE.md`, `docs/AGENT_HANDOFF.md`.

### MTH-POSTHOC-012
- Date: 2026-02-26.
- Status: active.
- Summary: hardened IMA++ preparation operations by (1) pinning live Zenodo record file defaults, (2) adding threaded ISIC API download mode with retry/backoff + resume-safe skip-existing behavior, (3) ingesting optional `seg_metadata_multiannotator_subset.csv`, and (4) fixing split-manifest generation to support split CSVs with `image` column values (including case/extension normalization).
- Impact: improves large-scale acquisition reliability and reproducibility of prepared dataset artifacts without changing canonical GT policy, segmentation metrics, or sensitivity-analysis definitions.
- Code anchors: `scripts/prepare_ima_plusplus.py`, `tests/test_prepare_ima_plusplus.py`, `README.md`, `docs/ARCHITECTURE.md`, `docs/AGENT_HANDOFF.md`.

## Update Protocol For New Method Changes
- Add a new `MTH-*` entry in this file describing what changed and why it matters.
- Update `docs/MANUSCRIPT_ALIGNMENT.md` if semantics, contracts, or provider behavior changed.
- Update method-facing docs in `README.md` when user-visible behavior changed.
- Add or adjust tests (`tests/test_cli.py`, `tests/test_prompts.py`, provider-specific tests) to lock behavior.
- Include the new changelog ID in PR/commit notes for traceability.
