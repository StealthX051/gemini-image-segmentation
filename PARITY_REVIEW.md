# CLI vs. Legacy Notebook Parity Review

This review tracked differences between the CLI implementation and the original notebook workflows that could block identical reproduction. The previously flagged gaps have been closed.

## Resolution summary
- **Histopathology prompts restored**: `configs/prompts.yaml` now includes a histopathology preset aligned with the notebook wording.
- **Pilot/curated manifests selectable**: the CLI accepts `--manifest` overrides (with optional fallbacks) to load curated lists such as `pilot50_*` without mutating the master manifest.
- **Legacy prediction fidelity**: raw model responses (original `box_2d`, `label`, `mask`) are persisted per image and reused when emitting `--legacy-predictions`, matching the notebook JSON payloads instead of synthesized bounding boxes.
- **Statistical method parity**: run summaries and fairness outputs default to BCa bootstrapping with 5,000 resamples, mirroring the notebook configuration while allowing overrides.

No additional parity blockers are known at this time.
