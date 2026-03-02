from __future__ import annotations

from pathlib import Path

from nanobanana_segmentation.study.config import load_study_config


def test_load_study_config_with_retrieval_toggles(tmp_path: Path) -> None:
    cfg_path = tmp_path / "study.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "study_id: s1",
                "dataset_name: ds",
                "dataset_root: .",
                "target: lesion",
                "execution:",
                "  workers: 3",
                "  client_timeout_s: 90",
                "  progress_poll_seconds: 1",
                "  progress_log_interval_seconds: 5",
                "  stall_warning_seconds: 30",
                "  fail_fast: false",
                "retrieval:",
                "  query_policy: fixed_queries",
                "  snapshot_policy: frozen",
                "  scope_policy: curated_domains",
                "  primary_exclude_duplicates: true",
                "  primary_exclude_mask_source: false",
                "  include_audit_unavailable_in_primary: false",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_study_config(cfg_path)
    assert cfg.retrieval.query_policy == "fixed_queries"
    assert cfg.retrieval.snapshot_policy == "frozen"
    assert cfg.retrieval.scope_policy == "curated_domains"
    assert cfg.retrieval.primary_exclude_duplicates is True
    assert cfg.retrieval.primary_exclude_mask_source is False
    assert cfg.retrieval.include_audit_unavailable_in_primary is False
    assert cfg.execution.workers == 3
    assert cfg.execution.client_timeout_s == 90
    assert cfg.execution.progress_poll_seconds == 1
    assert cfg.execution.progress_log_interval_seconds == 5
    assert cfg.execution.stall_warning_seconds == 30
    assert cfg.execution.fail_fast is False
