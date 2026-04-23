# Manuscript Alignment

This document keeps implementation changes aligned with:
- The original manuscript draft (baseline workflow and task framing).
- Post hoc additions introduced after the manuscript draft, including provider expansion and prompt ablation.
- See `docs/METHODS_CHANGELOG.md` for ordered method-version history and change IDs.

## Current Scope
- Core task families remain medical-image segmentation workflows across endoscopy, dermoscopy, fundus imaging, laparoscopy, ultrasound, CT, and chest radiography.
- The production entrypoint is the CLI (`src/gemini_segmentation/cli.py`), with notebooks preserved as legacy provenance.
- The isolated NanoBanana study lane (`src/nanobanana_segmentation/`) is intentionally separate from manuscript-baseline runtime contracts and does not alter existing CLI semantics unless explicitly promoted.

## Post Hoc Additions Reflected In Code
- Provider expansion includes Gemini model selection via `--model-name`, with `gemini-robotics-er-1.6-preview` now replacing the active Robotics ER 1.5 path.
- Provider expansion also includes a repo-scoped Robotics-only Gemini agentic-vision toggle via `--gemini-agentic-vision`, implemented as Gemini code execution on top of Robotics-ER 1.6 while leaving prompt-family wording unchanged.
- Provider expansion includes Moondream integration (`--provider moondream`) via `MoondreamSegmenter`.
- Provider expansion includes Replicate/Sa2VA integration (`--provider replicate`) via `Sa2VAReplicateSegmenter`.
- Prompt ablation family `label_v1` is class-name-only target instruction.
- Prompt ablation family `desc_v1` is class-name plus short definition and stable visual/anatomic descriptors.
- Prompt ablation family `desc_neg_v1` is `desc_v1` plus explicit exclusions/negation block.

## Prompt-Ablation Method Contract
- Family semantics must remain stable unless methods are intentionally revised.
- For Gemini calls, prompt text remains JSON-schema-oriented with keys `box_2d`, `mask`, and `label`, but runtime wording should stay close to Google’s recommended segmentation prompt skeleton rather than manuscript-style rationale text.
- The intended Gemini output contract requires `mask` to be described as a `base64 encoded png` and returned as a PNG data URI beginning with `data:image/png;base64,`; SVG paths, polygon coordinate lists, and other vector encodings are outside the intended contract.
- Absence behavior for single-target tasks remains `[]`.
- Absence behavior for multi-target tasks allows omitted entries for absent targets (with `[]` when no targets are present).
- `desc_neg_v1` should remain an exclusions-focused extension of `desc_v1` rather than a separate redesign.

## Provider-Aware Prompt Shaping Contract
- Gemini: receives a short Google-style segmentation prompt skeleton plus the selected family’s task-specific wording. For officially documented Gemini 2.5 models, the adapter also requests `application/json` structured output with a segmentation schema; `gemini-robotics-er-1.6-preview` reuses the same prompt wording but does not enable schema mode by default in this repo. Tool-enabled Robotics-ER 1.6 runs keep Gemini code execution enabled and structured output disabled. Source guidance: <https://ai.google.dev/gemini-api/docs/image-understanding>, <https://ai.google.dev/gemini-api/docs/structured-output>, <https://developers.googleblog.com/conversational-image-segmentation-gemini-2-5/>.
- Moondream: receives object target label(s) only (schema text is not sent as the segmentation instruction), using provider-native segment arguments only (no Gemini-style `temperature` / `thinking_budget` controls). The adapter expects Moondream’s native SVG path plus bounding-box output and rasterizes it into the repo’s normalized mask format. Source guidance: <https://docs.moondream.ai/skills/>, <https://docs.moondream.ai/skills/segment/>.
- Replicate/Sa2VA: receives natural-language instruction(s), optionally per target.
- Replicate/Sa2VA defaults are family-aware: `label_v1` uses short `Please segment the <target>.` instructions; `desc_v1` adds a short task descriptor/context sentence; `desc_neg_v1` appends exclusions-only deltas to `desc_v1`. Source guidance: <https://huggingface.co/ByteDance/Sa2VA-8B>, <https://github.com/bytedance/Sa2VA>.
- Replicate/Sa2VA adapter calls must use provider-supported image input formats (file upload or equivalent URI form), not raw JSON `bytes` payloads.
- Any change to provider shaping should be treated as a methods change and documented here.

## Caching Contract
- Local request caching may be used to avoid duplicate inference calls across reruns, but cache keys must include model/provider/prompt/image identity so ablation conditions remain isolated.
- Robotics-ER 1.6 off/on agentic-vision conditions must remain cache-isolated from each other.
- Local request cache should persist parse-success responses only; malformed/timeout responses should be retried rather than frozen into cache.
- Gemini explicit context caching may be enabled for supported Gemini models to reduce prompt-token costs.
- When Gemini caching is enabled, runs should monitor `usage_metadata.cached_content_token_count` to confirm cache-token reuse in practice.
- `gemini-robotics-er-1.6-preview` now remains eligible for explicit Gemini cache. Historical `gemini-robotics-er-1.5-preview` runs remain the unsupported-cache reference point.

## Robotics-ER 1.6 Agentic-Vision Contract
- In this repo, agentic vision is intentionally scoped to `gemini-robotics-er-1.6-preview`.
- The ablation is defined as tool enablement only: same prompt family, same retry policy, same dataset, same thinking-budget setting unless intentionally overridden.
- The tool-enabled condition uses Gemini code execution while preserving the existing JSON-schema segmentation output contract (`box_2d`, `mask`, `label`).
- Distinct output/report labels may be used via `output_model_name` so the plain and tool-enabled Robotics-ER 1.6 conditions remain separately auditable even though they call the same API model.

## Reliability Contract
- Segmentation runs should use bounded retries for timeout/parse-failure outcomes to reduce one-off malformed-output artifacts.
- Default CLI behavior uses `max_retries=5` unless intentionally overridden for sensitivity analyses.
- Mask metrics should normalize multi-channel mask arrays (e.g., RGB PNG labels) to single-channel before IoU/Dice computation.
- IMA++ dermoscopy integration uses a deterministic canonical-GT policy at dataset-prep time:
  - prefer STAPLE consensus masks,
  - fall back to majority-vote consensus masks,
  - fall back to single annotator masks only when exactly one annotator mask exists for the image.
- For IMA++, all masks and per-mask metadata (annotator/tool/skill labels when provided) should be retained for optional sensitivity analyses; canonical-GT evaluation should not discard this provenance.
- Replicate runs should be configured with explicit per-process throttling when account limits are tight (for example `workers=1`, `rate_limit>=12s` in smoke validation) so provider throttling does not confound method validation.
- Fairness CLI must preserve legacy reproducibility by default (`fairness --audit-mode legacy`) and only execute fairness-v2 logic when explicitly requested (`--audit-mode enhanced`).
- Fairness-v2 text output must use proxy language (for example “image-derived non-lesional skin tone proxy (ITA)” or “image-derived perilesional skin tone proxy (ITA)” plus “lower-ITA vs higher-ITA strata”) and avoid identity-language overreach in auto-generated captions/tables.

## Reproducibility And Reporting
- Keep `run_config.json` comprehensive for post hoc analyses and manuscript traceability.
- Required traceability fields include provider, model identifier, optional output model label, prompt family, prompt hash, prompt text or provider-specific target/instruction payload, retry policy (`max_retries`), bootstrap settings, and whether Gemini agentic vision was enabled.
- Matrix orchestration runs should record resolved benchmark configuration and execution status under `results/batches/<run_id>/` (`resolved_config.json`, `job_status.jsonl`, `summary.json`) so multi-model ablation batches are auditable.
- Recommended run-id policy for matrix studies is `<study_id>_<YYYYMMDD-HHMMSS>`; reuse the same run-id only when explicitly resuming the same study settings.
- Matrix configs should define the full three-family ablation (`label_v1`, `desc_v1`, `desc_neg_v1`) per job unless a sensitivity analysis explicitly narrows scope.
- Comparative reporting can be generated post-run via `python -m gemini_segmentation.paper.prompt_comparison` (grouped per model and per prompt family; includes mean/median IoU-Dice, 95% CIs, and success rate) without altering segmentation outputs.
- Prompt-comparison reporting supports Gemini-only runs; Moondream and Replicate rows are optional when those providers were not part of the selected comparison.
- Prompt-comparison reporting should keep the active Gemini defaults on the Robotics-ER 1.6 family while still auto-including legacy Gemini model rows, including historical `gemini-robotics-er-1.5-preview` outputs, whenever those directories exist for the selected Gemini run ID.
- IMA++ optional sensitivity analysis may be generated post-run via `python scripts/analyze_ima_plusplus_sensitivity.py` to report:
  - model-vs-MV consensus metrics,
  - model-vs-annotator metrics and per-image dispersion summaries (mean/median/IQR/min/max),
  - stratified summaries by annotator tool/skill metadata when available.
- Enhanced fairness mode (`fairness --audit-mode enhanced`) writes expanded artifacts to `<run_dir>/fairness_enhanced/` including canonical analysis frame, dedup maps/reports, endpoint effect tables, trend curves, and sensitivity analyses; staged execution (`--enhanced-stage all|core|sensitivity|augment`), feature-profile gating, and resumable checkpointing are operational/runtime controls and do not alter legacy fairness semantics. Legacy fairness outputs remain under `<run_dir>/fairness/`.
- Enhanced fairness now also writes covariate-adjusted success-effect artifacts (`covadj_success_t050_effects.csv|json`, `covadj_model_spec.json`) based on predictive margins from logistic models, plus optional per-term component contribution artifacts (`covadj_component_effects.csv|json`) with CI-based term significance flags; these are descriptive adjusted disparity summaries and should be interpreted with proxy-language/non-causal framing.
- Manuscript text should explicitly distinguish adjusted model families: continuous ITA trend models (`ita_deg`) versus binary-cutoff predictive-margin models (`ita_binary`, lower vs higher ITA), since they target different estimands.
- Manuscript-ready enhanced fairness method text should be sourced from `docs/FAIRNESS_ENHANCED_METHODS.md` to keep claims aligned with implementation details (region strategy, ITA estimator, covariates, trend models, and sensitivity definitions).
- Fairness paper artifact generation (`python -m gemini_segmentation.paper.figures --fairness-dir <.../fairness>`) may emit both combined Figure 2 and standalone panel exports (`figure2_panel_a` to `figure2_panel_d` in PNG/PDF/SVG); this is a presentation/export change only and does not alter fairness computations.
- Enhanced fairness manuscript artifact generation (`python -m gemini_segmentation.paper.figures_enhanced --fairness-enhanced-dir <.../fairness_enhanced>`) emits E-numbered figure/table assets plus a narrative report (`md|html|pdf|docx`) while preserving proxy-language/non-causal framing and leaving legacy Figure 2/Table 4 tooling unchanged.
- When method semantics change, update in the same change:
  - `src/gemini_segmentation/prompts.py`
  - `configs/prompts.yaml`
  - relevant tests (`tests/test_prompts.py`, `tests/test_cli.py`)
  - this document and manuscript-facing methods text in `README.md`
