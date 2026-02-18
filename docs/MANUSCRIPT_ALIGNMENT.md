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
- Moondream: receives object target label(s) (schema text is not sent as the segmentation instruction).
- Replicate/Sa2VA: receives natural-language instruction(s), optionally per target.
- Any change to provider shaping should be treated as a methods change and documented here.

## Reproducibility And Reporting
- Keep `run_config.json` comprehensive for post hoc analyses and manuscript traceability.
- Required traceability fields include provider, model identifier, prompt family, prompt hash, prompt text or provider-specific target/instruction payload, and bootstrap settings.
- When method semantics change, update `src/gemini_segmentation/prompts.py`.
- When method semantics change, update `configs/prompts.yaml`.
- When method semantics change, update relevant tests in `tests/test_prompts.py` and `tests/test_cli.py`.
- When method semantics change, update this document and manuscript-facing methods text in `README.md`.
