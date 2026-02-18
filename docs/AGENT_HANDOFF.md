# Agent Handoff (Current State)

Last updated: 2026-02-18.

## Current Priorities
- Prompt-ablation runs (`label_v1`, `desc_v1`, `desc_neg_v1`) across Gemini models.
- Robotics ER benchmarking via `gemini-robotics-er-1.5-preview`.
- Cost control through local request cache plus Gemini explicit cache where supported.

## Runtime Facts
- CLI entrypoint: `python -m gemini_segmentation.cli segment ...`
- Prompt families are selected by repeating `--prompt-family` (no `--prompt-families` flag).
- Default retry policy: `--max-retries 5` (five retries after the first attempt) for timeout/parse-failure/exception retries.
- Local request cache is enabled by default; failed parses/timeouts are not persisted.
- Explicit Gemini context cache is enabled by default for supported models and auto-skipped for robotics ER.

## Critical Gotchas
- `.env` is not auto-loaded into shell process env vars by the CLI. Export env vars in the active shell before running.
- Using `--prompt-preset configs/prompts.yaml --preset-name polyp` can override `--model-name` because preset `polyp` sets a model in YAML.
  - For strict model comparisons, either:
    - avoid preset model-bearing entries, or
    - use family-only selection with explicit `--model-name`.
- Some datasets have RGB ground-truth masks; metrics now normalize masks to single-channel before IoU/Dice.

## Recommended Smoke Command Pattern
Use one model per run and repeat `--prompt-family`:

```bash
python -m gemini_segmentation.cli segment polyp segmented-images \
  --provider gemini \
  --model-name gemini-robotics-er-1.5-preview \
  --prompt-family label_v1 \
  --prompt-family desc_v1 \
  --prompt-family desc_neg_v1 \
  --sample-size 100 \
  --workers <cpus-minus-2> \
  --rate-limit 0.5 \
  --max-retries 5 \
  --local-cache \
  --local-cache-dir results/.request_cache \
  --no-gemini-explicit-cache
```

## Key Files For New Agents
- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/MANUSCRIPT_ALIGNMENT.md`
- `docs/GEMINI_CACHING.md`
- `docs/METHODS_CHANGELOG.md`
- `src/gemini_segmentation/cli.py`
- `src/gemini_segmentation/models.py`
- `src/gemini_segmentation/cache.py`
- `src/gemini_segmentation/metrics.py`
- `tests/test_cli.py`
- `tests/test_metrics.py`
