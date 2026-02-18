# Gemini Caching Notes

Verified on **2026-02-18** against official Gemini docs.

## Model Support Snapshot
- `gemini-2.5-flash`: caching supported.
- `gemini-2.5-flash-lite`: caching supported.
- `gemini-robotics-er-1.5-preview`: robotics docs list caching as **not supported**.

## API-Side Caching Modes
- Implicit caching: enabled automatically on supported models when repeated prefix context is detected.
- Explicit caching (`cachedContents`): create a cache once and reference it in subsequent generate calls.
- Gemini docs list minimum input token sizes for caching:
  - `gemini-2.5-flash`: 1024 tokens
  - `gemini-2.5-flash-lite`: 2048 tokens

## Repo Implementation
- Local cache (all providers): `src/gemini_segmentation/cache.py` + `src/gemini_segmentation/cli.py`
  - Keyed by provider, model, prompt signature, image SHA-256, and decode settings.
  - Stores parse-success responses only.
- Gemini explicit cache (supported Gemini models): `src/gemini_segmentation/models.py`
  - Enabled by default.
  - Auto-skips explicit cache for robotics ER models.
  - Logs Gemini cached token usage when present.

## Recommended CLI Flags For Ablation Runs
- Keep local cache on: `--local-cache` (default).
- Reuse one shared cache directory across runs:
  - `--local-cache-dir results/.request_cache`
- Keep Gemini explicit cache on for 2.5 Flash / 2.5 Flash-Lite:
  - `--gemini-explicit-cache --gemini-cache-ttl 3600`
- Robotics ER:
  - Explicit cache is auto-disabled by adapter logic; local cache still works.

## Operational Gotchas
- `.env` keys must be exported into the active shell process before running CLI commands (CLI does not auto-load `.env`).
- Presets in `configs/prompts.yaml` can include `model` and override `--model-name`; avoid such presets during strict cross-model comparisons.

## References
- Context caching guide: https://ai.google.dev/gemini-api/docs/caching
- Models capability page: https://ai.google.dev/gemini-api/docs/models
- Robotics model docs: https://ai.google.dev/gemini-api/docs/robotics
- Pricing (context caching rows): https://ai.google.dev/gemini-api/docs/pricing
