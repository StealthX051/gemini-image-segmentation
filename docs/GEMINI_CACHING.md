# Gemini Caching Snapshot

This is a dated operational snapshot, not a permanent source of truth for external provider capabilities.
For repo behavior, defer to code plus `docs/ARCHITECTURE.md`.
When current provider support matters, re-check the upstream provider docs.

Verified on **2026-04-22** against official Gemini docs.

## Model Support Snapshot
- `gemini-2.5-flash`: caching supported.
- `gemini-2.5-flash-lite`: caching supported.
- `gemini-robotics-er-1.6-preview`: robotics docs list caching as supported.
- Historical note: `gemini-robotics-er-1.5-preview` was documented as **not supported** for explicit cache.

## API-Side Caching Modes
- Implicit caching: enabled automatically on supported models when repeated prefix context is detected.
- Explicit caching (`cachedContents`): create a cache once and reference it in subsequent generate calls.
- Gemini docs list minimum input token sizes for caching:
  - `gemini-2.5-flash`: 1024 tokens
  - `gemini-2.5-flash-lite`: 2048 tokens

## Repo Implementation
- Local cache (all providers): `src/gemini_segmentation/cache.py` + `src/gemini_segmentation/cli.py`
  - Keyed by provider, model, prompt signature, image SHA-256, decode settings, and Gemini agentic-vision state.
  - Stores parse-success responses only.
- Gemini explicit cache (supported Gemini models): `src/gemini_segmentation/models.py`
  - Enabled by default.
  - Auto-skips explicit cache only for models documented as unsupported.
  - Logs Gemini cached token usage when present.

## Recommended CLI Flags For Ablation Runs
- Keep local cache on: `--local-cache` (default).
- Reuse one shared cache directory across runs:
  - `--local-cache-dir results/.request_cache`
- Keep Gemini explicit cache on for 2.5 Flash / 2.5 Flash-Lite:
  - `--gemini-explicit-cache --gemini-cache-ttl 3600`
- Robotics ER 1.6:
  - Keep explicit cache on unless you are deliberately reproducing a historical unsupported model path.
  - Enable `--gemini-agentic-vision` only for the tool-enabled Robotics 1.6 ablation condition.
  - Use `--output-model-name gemini-robotics-er-1.6-preview-agentic` when you need separate artifact paths for that condition.

## Operational Gotchas
- `.env` keys must be exported into the active shell process before running CLI commands (CLI does not auto-load `.env`).
- Presets in `configs/prompts.yaml` can include `model` and override `--model-name`; avoid such presets during strict cross-model comparisons.

## References
- Context caching guide: https://ai.google.dev/gemini-api/docs/caching
- Models capability page: https://ai.google.dev/gemini-api/docs/models
- Robotics model docs: https://ai.google.dev/gemini-api/docs/robotics
- Code execution with images: https://ai.google.dev/gemini-api/docs/code-execution#images
- Pricing (context caching rows): https://ai.google.dev/gemini-api/docs/pricing
