# Documentation Map

This file defines documentation ownership so details stay comprehensive without drifting into conflicting duplicates.

## Canonical Sources By Topic

- Runtime architecture and contracts:
  - `docs/ARCHITECTURE.md`
  - Authority: module boundaries, runtime flow, path contracts, provider contract invariants.
- Manuscript-facing method semantics and non-causal framing:
  - `docs/MANUSCRIPT_ALIGNMENT.md`
  - Authority: prompt-family semantics, provider-aware shaping, fairness-language constraints.
- Method/version history:
  - `docs/METHODS_CHANGELOG.md`
  - Authority: ordered method change IDs, dates, and impact anchors.
- Day-to-day operational caveats and recent validation notes:
  - `docs/AGENT_HANDOFF.md`
  - Authority: current-state run notes, practical gotchas, recently validated commands.
  - Non-authority: long-term contract definitions (defer to canonical docs above).
- Historical run snapshots and pinned validation records:
  - `docs/VALIDATION_SNAPSHOTS.md`
  - Authority: point-in-time run IDs/metrics/command variants kept for reproducibility history.
  - Non-authority: current recommended defaults (defer to README + handoff + code).
- Batch orchestration behavior:
  - `docs/BATCH_ORCHESTRATION.md`
  - Authority: matrix config schema, preflight rules, batch lifecycle and outputs.
- Enhanced fairness algorithms:
  - `docs/FAIRNESS_ENHANCED_METHODS.md`
  - Authority: implementation-aligned fairness-v2 methods and manuscript-ready wording.
- Enhanced fairness operations:
  - `docs/FAIRNESS_ENHANCED.md`
  - Authority: CLI/runtime knobs, stage behavior, output file inventory.
- Gemini/API caching specifics:
  - `docs/GEMINI_CACHING.md`
  - Authority: model support snapshot and repo caching behavior.
- Legacy notebook scope and policy:
  - `docs/NOTEBOOKS.md`
  - Authority: notebook families, provenance role, editing boundaries.
- NanoBanana isolated lane:
  - `docs/NANOBANANA_STUDY.md`
  - Authority: service/study contracts specific to `src/nanobanana_segmentation/`.

## README Role

- `README.md` is the onboarding entrypoint and command reference.
- It should summarize and link to canonical docs rather than carry unique method contracts that can drift independently.

## Redundancy Policy

- Keep detail where it is most maintainable; avoid deleting substantive method detail.
- Prefer one canonical definition plus links in secondary docs.
- When a detail must appear in multiple places (for usability), copy from canonical text and update all copies in the same change.

## Conflict Resolution Order

When statements differ, use this precedence:

1. CLI/code behavior in `src/` and `--help` surfaces.
2. Canonical topic docs listed above.
3. Operational snapshot docs (`docs/AGENT_HANDOFF.md`).
4. Convenience summaries/examples in `README.md`.

## Update Checklist

For any user-visible behavior change:

1. Update code/tests first.
2. Update the canonical topic doc for that behavior.
3. Update `README.md` and `docs/AGENT_HANDOFF.md` only where needed for discoverability or operational guidance.
4. If method semantics changed, add a new entry to `docs/METHODS_CHANGELOG.md`.
