import argparse
import json
from pathlib import Path

import yaml

from gemini_segmentation import batch


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _make_dataset_root(path: Path) -> None:
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "masks").mkdir(parents=True, exist_ok=True)


def _base_config(dataset_root: str, *, results_dir: str) -> dict:
    return {
        "schema_version": 1,
        "study_id": "ablation_robotics",
        "results_dir": results_dir,
        "defaults": {
            "provider": "gemini",
            "prompt_families": ["label_v1", "desc_v1", "desc_neg_v1"],
            "max_retries": 5,
            "workers": 2,
            "timeout": 60.0,
            "local_cache": True,
            "local_cache_dir": "results/.request_cache",
            "gemini_explicit_cache": True,
            "gemini_cache_ttl": 3600,
            "temperature": 0.5,
            "thinking_budget": 0,
            "bootstrap_method": "bca",
            "bootstrap_resamples": 5000,
        },
        "datasets": [{"name": "polyp", "root": dataset_root}],
        "models": [{"name": "gemini-2.5-flash"}],
    }


def test_load_batch_config_expands_env_placeholders(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "segmented-images"
    _make_dataset_root(data_root)
    monkeypatch.setenv("POLYP_DATASET_ROOT", str(data_root))

    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        _base_config("${POLYP_DATASET_ROOT}", results_dir=str(tmp_path / "results")),
    )

    loaded = batch.load_batch_config(config_path)
    assert loaded["datasets"][0]["root"] == str(data_root)


def test_load_batch_config_merges_overrides(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    override_path = tmp_path / "override.yaml"

    _write_yaml(
        config_path,
        {
            "schema_version": 1,
            "study_id": "study",
            "defaults": {"workers": 4, "max_retries": 5},
            "datasets": [{"name": "polyp", "root": "/tmp/a"}],
            "models": [{"name": "gemini-2.5-flash"}],
        },
    )
    _write_yaml(override_path, {"defaults": {"workers": 1}})

    merged = batch.load_batch_config(config_path, override_path)
    assert merged["defaults"]["workers"] == 1
    assert merged["defaults"]["max_retries"] == 5


def test_build_jobs_allows_unresolved_env_for_filtered_datasets(tmp_path, monkeypatch) -> None:
    root_a = tmp_path / "a"
    _make_dataset_root(root_a)
    monkeypatch.setenv("DATASET_A_ROOT", str(root_a))

    config_path = tmp_path / "config.yaml"
    _write_yaml(
        config_path,
        {
            "schema_version": 1,
            "study_id": "study",
            "defaults": {"provider": "gemini", "prompt_families": ["label_v1"]},
            "datasets": [
                {"name": "dataset_a", "root": "${DATASET_A_ROOT}"},
                {"name": "dataset_b", "root": "${DATASET_B_ROOT}"},
            ],
            "models": [{"name": "gemini-2.5-flash"}],
        },
    )

    config = batch.load_batch_config(config_path)
    jobs = batch.build_jobs(config, only_datasets=["dataset_a"])
    assert len(jobs) == 1
    assert jobs[0].dataset_root == root_a


def test_build_jobs_applies_filters_and_robotics_cache_policy(tmp_path) -> None:
    config = {
        "schema_version": 1,
        "study_id": "study",
        "results_dir": str(tmp_path / "results"),
        "defaults": {
            "provider": "gemini",
            "prompt_families": ["label_v1", "desc_v1", "desc_neg_v1"],
            "gemini_explicit_cache": True,
        },
        "datasets": [
            {"name": "polyp", "root": str(tmp_path / "polyp")},
            {"name": "derm_lesion", "root": str(tmp_path / "derm")},
        ],
        "models": [
            {"name": "gemini-2.5-flash"},
            {"name": "gemini-robotics-er-1.5-preview"},
        ],
    }

    jobs = batch.build_jobs(
        config,
        only_datasets=["polyp"],
        only_models=["gemini-robotics-er-1.5-preview"],
    )
    assert len(jobs) == 1
    assert jobs[0].dataset_name == "polyp"
    assert jobs[0].model_name == "gemini-robotics-er-1.5-preview"
    assert jobs[0].gemini_explicit_cache is False

    cmd = batch.build_segment_command(jobs[0], run_id="run-id", results_dir=tmp_path / "results")
    assert "--no-gemini-explicit-cache" in cmd
    assert "--gemini-explicit-cache" not in cmd
    assert cmd.count("--prompt-family") == 3


def test_build_segment_command_includes_required_flags(tmp_path) -> None:
    job = batch.BatchJob(
        dataset_name="polyp",
        dataset_root=tmp_path / "dataset",
        model_name="gemini-2.5-flash",
        provider="gemini",
        prompt_families=("label_v1", "desc_v1", "desc_neg_v1"),
        manifest=None,
        timeout=60.0,
        max_retries=5,
        workers=4,
        sample_size=100,
        rate_limit=0.5,
        local_cache=True,
        local_cache_dir=Path("results/.request_cache"),
        gemini_explicit_cache=True,
        gemini_cache_ttl=3600,
        thinking_budget=0,
        temperature=0.5,
        legacy_predictions=False,
        success_threshold=0.5,
        bootstrap_method="bca",
        bootstrap_resamples=5000,
    )
    cmd = batch.build_segment_command(job, run_id="batch-run", results_dir=tmp_path / "results")

    assert "--max-retries" in cmd and "5" in cmd
    assert "--local-cache" in cmd
    assert "--local-cache-dir" in cmd
    assert "--gemini-explicit-cache" in cmd
    assert "--run-id" in cmd and "batch-run" in cmd
    assert cmd.count("--prompt-family") == 3


def test_run_batch_continues_on_failure_and_returns_nonzero(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset_root(dataset_root)
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")

    config = _base_config(str(dataset_root), results_dir=str(tmp_path / "results"))
    config["models"] = [
        {"name": "gemini-2.5-flash"},
        {"name": "gemini-2.5-flash-lite"},
    ]
    config_path = tmp_path / "batch.yaml"
    _write_yaml(config_path, config)

    calls = {"count": 0}

    def fake_run_command(cmd, log_path):
        calls["count"] += 1
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("log", encoding="utf-8")
        exit_code = 1 if calls["count"] == 1 else 0
        return exit_code, 0.1, "2026-02-18T10:00:00", "2026-02-18T10:00:01"

    monkeypatch.setattr(batch, "_run_command", fake_run_command)

    args = argparse.Namespace(
        config=str(config_path),
        overrides=None,
        run_id="unit-batch-run",
        only_dataset=None,
        only_model=None,
        auto_fairness=False,
        dry_run=False,
        stop_on_failure=False,
    )
    exit_code = batch.run_batch(args)
    assert exit_code == 1
    assert calls["count"] == 2

    summary_path = tmp_path / "results" / "batches" / "unit-batch-run" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["segment_jobs_executed"] == 2
    assert summary["segment_jobs_failed"] == 1


def test_run_batch_dry_run_emits_plan_without_execution(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset_root(dataset_root)

    config = _base_config(str(dataset_root), results_dir=str(tmp_path / "results"))
    config_path = tmp_path / "batch.yaml"
    _write_yaml(config_path, config)

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("_run_command should not be called during dry-run")

    monkeypatch.setattr(batch, "_run_command", should_not_run)

    args = argparse.Namespace(
        config=str(config_path),
        overrides=None,
        run_id="unit-batch-dry",
        only_dataset=None,
        only_model=None,
        auto_fairness=False,
        dry_run=True,
        stop_on_failure=False,
    )
    exit_code = batch.run_batch(args)
    assert exit_code == 0

    summary_path = tmp_path / "results" / "batches" / "unit-batch-dry" / "summary.json"
    status_path = tmp_path / "results" / "batches" / "unit-batch-dry" / "job_status.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status_lines = [line for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert summary["dry_run"] is True
    assert summary["planned_segment_jobs"] == 1
    assert len(status_lines) == 1
