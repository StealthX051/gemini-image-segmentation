# Documentation Map

This file defines which docs are fast-path engineering guidance, which docs are canonical by topic, and which docs are historical or operational snapshots.

## Read First
- `AGENTS.md`
  - Repository-specific Codex rules, minimal-patch guidance, and execution constraints.
- `docs/ENGINEERING_QUICKSTART.md`
  - Fast-path setup, Windows/WSL/Conda execution policy, and task-based doc routing.
- `docs/SETUP.md`
  - Canonical fresh-clone setup and environment bootstrap guide.
- `README.md`
  - Human onboarding entrypoint and command overview.

## Canonical Sources By Topic
- Runtime architecture and contracts:
  - `docs/ARCHITECTURE.md`
  - Authority: module boundaries, runtime flow, path contracts, provider contract invariants.
- Batch orchestration behavior:
  - `docs/BATCH_ORCHESTRATION.md`
  - Authority: matrix config schema, preflight rules, batch lifecycle, outputs.
- Manuscript-facing method semantics and non-causal framing:
  - `docs/MANUSCRIPT_ALIGNMENT.md`
  - Authority: prompt-family semantics, provider-aware shaping, fairness-language constraints.
- Method/version history:
  - `docs/METHODS_CHANGELOG.md`
  - Authority: ordered method change IDs, dates, and impact anchors.
- Enhanced fairness algorithms:
  - `docs/FAIRNESS_ENHANCED_METHODS.md`
  - Authority: implementation-aligned fairness-v2 methods and manuscript-ready wording.
- Enhanced fairness operations:
  - `docs/FAIRNESS_ENHANCED.md`
  - Authority: CLI/runtime knobs, stage behavior, output inventory.
- Legacy notebook scope and policy:
  - `docs/NOTEBOOKS.md`
  - Authority: notebook families, provenance role, editing boundaries.
- Setup and environment bootstrap:
  - `docs/SETUP.md`
  - Authority: fresh-clone setup flow, Conda/venv bootstrap steps, and verification commands.

## Operational Snapshots
- `docs/AGENT_HANDOFF.md`
  - Current caveats, practical gotchas, and recent validation notes.
  - Non-authority for durable contracts; defer to canonical docs and code when they differ.
- `docs/GEMINI_CACHING.md`
  - Dated operational snapshot for cache support and repo caching behavior.
  - Re-verify external provider details when current accuracy matters.

## Isolated Study Lane
- `docs/NANOBANANA_STUDY.md`
  - Authority for the isolated `src/nanobanana_segmentation/` study/service lane.
  - Read only when working in NanoBanana-related code or configs.

## History Only
- `docs/history/VALIDATION_SNAPSHOTS.md`
  - Point-in-time validation records, run IDs, and smoke-command history.
  - Non-authority for current defaults or current recommended workflows.

## README Role
- `README.md` should summarize and route.
- It should not become the only place that defines method contracts, operational caveats, or codebase invariants.

## Redundancy Policy
- Keep one canonical definition per topic whenever possible.
- Duplicate details only when they materially improve usability, then update all copies in the same change.
- Prefer short development docs plus links over repeating long method descriptions in multiple places.
- Delete stale or duplicate documentation when it no longer adds unique value. Archive only when the historical record is genuinely useful.

## Conflict Resolution Order
When statements differ, use this precedence:

1. Actual code and current CLI `--help` behavior
2. Canonical topic docs listed above
3. Operational snapshot docs
4. README summaries and examples

## Update Checklist
For any user-visible behavior change:

1. Update code and tests first.
2. Update the canonical topic doc for that behavior.
3. Update `README.md`, `AGENTS.md`, or `docs/ENGINEERING_QUICKSTART.md` only where needed for discoverability or execution guidance.
4. Update `docs/AGENT_HANDOFF.md` only if there is a meaningful current-state caveat.
5. If method semantics changed, add a new entry to `docs/METHODS_CHANGELOG.md`.
6. Remove or trim stale documentation in the same pass instead of leaving known drift behind.
