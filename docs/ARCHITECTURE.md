# Architecture

## Runtime Flow
1. `segment` command resolves dataset paths and manifest (`src/gemini_segmentation/data.py`).
2. Prompt payload is built from prompt family/preset/provider (`src/gemini_segmentation/prompts.py`, `src/gemini_segmentation/config.py`).
3. Provider adapter performs inference (`src/gemini_segmentation/models.py`).
4. Responses are parsed into normalized masks and persisted (`src/gemini_segmentation/io.py`).
5. IoU/Dice/summary metrics are updated incrementally (`src/gemini_segmentation/metrics.py`).
6. Optional fairness analysis consumes saved masks/metrics (`src/gemini_segmentation/fairness.py`).

## Manuscript Alignment
- See `docs/MANUSCRIPT_ALIGNMENT.md` for method-level constraints that tie implementation to manuscript and post hoc extensions.
- Prompt ablation is represented by `PromptFamily`: `label_v1`, `desc_v1`, `desc_neg_v1`.
- Provider expansion includes Gemini model switching, Moondream adapter support, and Replicate/Sa2VA adapter support.
- Run reproducibility relies on `run_config.json` fields such as `provider`, `prompt_family`, `prompt_hash`, provider-specific targets/instructions, and model identifier.

## Key Design Contracts
- Segmenter contract: `segment(image_obj) -> (masks, latency_s, parse_success, timed_out, raw_items)`.
- Mask contract: `SegmentationMask` stores full-image binary mask plus pixel-space bounding box.
- Output contract: run artifacts live under `results/<dataset>/<model>/<prompt_key>/<run_id>/`.
- Resume behavior depends on `predictions.jsonl` and per-image artifact regeneration.

## Module Ownership
- `cli.py`: argument parsing, run orchestration, checkpointing loop.
- `models.py`: provider clients and provider-specific output adaptation.
- `io.py`: JSON parsing, base64 encoding/decoding, overlay rendering, JSONL persistence.
- `metrics.py`: IoU/Dice, bootstrap CI, rolling summaries.
- `fairness.py`: ITA extraction, tone grouping, statistical testing outputs.
- `paper/`: manuscript-ready tables and figures.

## Extension Points
- New provider: implement adapter in `models.py`, keep return contract stable, wire in `cli.py`.
- New prompt family: extend `PromptFamily` and dictionaries in `prompts.py`, add YAML presets in `configs/prompts.yaml`.
- New dataset layout: extend dataset discovery/manifest behavior in `data.py` while preserving `images/` + `masks/` assumptions where possible.
- New analysis metric: add metric computation in `metrics.py`, propagate to CSV/summary and tests.

## Operational Notes
- Large datasets are intentionally externalized; repository code should not assume local copies exist.
- Notebook workflows are legacy but still relevant for provenance and parity checks.
- Paper tools expect stable CSV schemas and config-driven registries in `configs/`.
