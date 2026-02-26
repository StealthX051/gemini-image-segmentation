# Manuscript Alignment

This document keeps implementation changes aligned with:
- The original manuscript draft (baseline workflow and task framing).
- Post hoc additions introduced after the manuscript draft, including provider expansion and prompt ablation.
- See `docs/METHODS_CHANGELOG.md` for ordered method-version history and change IDs.

## Current Scope
- Core task families remain medical-image segmentation workflows across endoscopy, dermoscopy, fundus imaging, laparoscopy, ultrasound, CT, and chest radiography.
- The production entrypoint is the CLI (`src/gemini_segmentation/cli.py`), with notebooks preserved as legacy provenance.

## Post Hoc Additions Reflected In Code
- Provider expansion includes Gemini model selection via `--model-name` (including `gemini-robotics-er-1.5-preview` support as a model identifier).
- Provider expansion includes Moondream integration (`--provider moondream`) via `MoondreamSegmenter`.
- Provider expansion includes Replicate/Sa2VA integration (`--provider replicate`) via `Sa2VAReplicateSegmenter`.
- Prompt ablation family `label_v1` is class-name-only target instruction.
- Prompt ablation family `desc_v1` is class-name plus short definition and stable visual/anatomic descriptors.
- Prompt ablation family `desc_neg_v1` is `desc_v1` plus explicit exclusions/negation block.

## Prompt-Ablation Method Contract
- Family semantics must remain stable unless methods are intentionally revised.
- For Gemini calls, prompt text remains JSON-schema-oriented with keys `box_2d`, `mask`, and `label`.
- Absence behavior for single-target tasks remains `[]`.
- Absence behavior for multi-target tasks allows omitted entries for absent targets (with `[]` when no targets are present).
- `desc_neg_v1` should remain an exclusions-focused extension of `desc_v1` rather than a separate redesign.

## Provider-Aware Prompt Shaping Contract
- Gemini: receives full JSON-schema prompt text.
- Moondream: receives object target label(s) (schema text is not sent as the segmentation instruction), using provider-native segment arguments only (no Gemini-style `temperature` / `thinking_budget` controls).
- Replicate/Sa2VA: receives natural-language instruction(s), optionally per target.
- Replicate/Sa2VA defaults are family-aware: `label_v1` uses label-only instructions; `desc_v1` adds descriptor context; `desc_neg_v1` appends exclusions-only deltas to `desc_v1`.
- Replicate/Sa2VA adapter calls must use provider-supported image input formats (file upload or equivalent URI form), not raw JSON `bytes` payloads.
- Any change to provider shaping should be treated as a methods change and documented here.

## Caching Contract
- Local request caching may be used to avoid duplicate inference calls across reruns, but cache keys must include model/provider/prompt/image identity so ablation conditions remain isolated.
- Local request cache should persist parse-success responses only; malformed/timeout responses should be retried rather than frozen into cache.
- Gemini explicit context caching may be enabled for supported Gemini models to reduce prompt-token costs.
- When Gemini caching is enabled, runs should monitor `usage_metadata.cached_content_token_count` to confirm cache-token reuse in practice.
- For `gemini-robotics-er-1.5-preview`, Gemini documentation currently lists context caching as unsupported; implementations must gracefully fall back without explicit cache.

## Reliability Contract
- Segmentation runs should use bounded retries for timeout/parse-failure outcomes to reduce one-off malformed-output artifacts.
- Default CLI behavior uses `max_retries=5` unless intentionally overridden for sensitivity analyses.
- Mask metrics should normalize multi-channel mask arrays (e.g., RGB PNG labels) to single-channel before IoU/Dice computation.
- Replicate runs should be configured with explicit per-process throttling when account limits are tight (for example `workers=1`, `rate_limit>=12s` in smoke validation) so provider throttling does not confound method validation.

## Reproducibility And Reporting
- Keep `run_config.json` comprehensive for post hoc analyses and manuscript traceability.
- Required traceability fields include provider, model identifier, prompt family, prompt hash, prompt text or provider-specific target/instruction payload, retry policy (`max_retries`), and bootstrap settings.
- Matrix orchestration runs should record resolved benchmark configuration and execution status under `results/batches/<run_id>/` (`resolved_config.json`, `job_status.jsonl`, `summary.json`) so multi-model ablation batches are auditable.
- Recommended run-id policy for matrix studies is `<study_id>_<YYYYMMDD-HHMMSS>`; reuse the same run-id only when explicitly resuming the same study settings.
- Matrix configs should define the full three-family ablation (`label_v1`, `desc_v1`, `desc_neg_v1`) per job unless a sensitivity analysis explicitly narrows scope.
- Comparative reporting can be generated post-run via `python -m gemini_segmentation.paper.prompt_comparison` (grouped per model and per prompt family; includes mean/median IoU-Dice, 95% CIs, and success rate) without altering segmentation outputs.
- Fairness paper artifact generation (`python -m gemini_segmentation.paper.figures --fairness-dir <.../fairness>`) may emit both combined Figure 2 and standalone panel exports (`figure2_panel_a` to `figure2_panel_d` in PNG/PDF/SVG); this is a presentation/export change only and does not alter fairness computations.
- When method semantics change, update `src/gemini_segmentation/prompts.py`.
- When method semantics change, update `configs/prompts.yaml`.
- When method semantics change, update relevant tests in `tests/test_prompts.py` and `tests/test_cli.py`.
- When method semantics change, update this document and manuscript-facing methods text in `README.md`.
