# AGENTS.md

Additional instructions for the isolated NanoBanana study lane.

## Scope
- Applies within `src/nanobanana_segmentation/`.
- This subtree is intentionally separate from `src/gemini_segmentation/`.

## Read First
- `docs/NANOBANANA_STUDY.md`
- Root `AGENTS.md`
- `docs/DOCUMENTATION_MAP.md`

## Working Rules
- Do not route NanoBanana features through `gemini_segmentation` unless the task explicitly asks for cross-lane integration.
- Preserve the service endpoints `/v1/segment`, `/health`, and `/metrics` unless explicitly changing the service contract.
- Preserve staged study semantics and audit outputs unless the task explicitly changes study methods.
- Treat retrieval, grounding, leakage, and attempt/QC metadata as first-class reproducibility artifacts.
- Deleting dead NanoBanana-only code or stale study docs is allowed when it is the cleanest non-breaking fix.
- Update the relevant NanoBanana docs in the same pass after substantive changes.

## Output Rules
- Keep runtime artifacts under `results_nanobanana/` and `artifacts_nanobanana/` unless explicitly asked otherwise.
- When changing NanoBanana runtime behavior, update the affected tests under `tests/nanobanana/`.
