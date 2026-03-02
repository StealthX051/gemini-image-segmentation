from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class StudyModelConfig:
    model_id: str = "gemini-3.1-flash-image-preview"
    api_surface: str = "generate_content"
    model_style: str = "nanobanana_v1"
    include_thoughts: bool = False
    cost_per_attempt_usd: float = 0.01
    use_otsu_bw: bool = False


@dataclass(frozen=True)
class StudyMatrixConfig:
    tool_modes: List[str] = field(default_factory=lambda: ["closed", "text", "image", "text_image"])
    thinking_levels: List[str] = field(default_factory=lambda: ["minimal"])
    replicates: int = 1


@dataclass(frozen=True)
class StageConfig:
    stage0_sample_size: int = 20
    stage1_sample_size: Optional[int] = None
    stage2_sample_size: int = 50
    stage2_enabled: bool = False
    stage2_tool_modes: List[str] = field(default_factory=lambda: ["closed", "text_image"])
    stage2_thinking_levels: List[str] = field(default_factory=lambda: ["minimal", "high"])
    stage2_replicates: int = 3


@dataclass(frozen=True)
class ExecutionConfig:
    workers: int = 6
    client_timeout_s: float = 120.0
    progress_poll_seconds: float = 2.0
    progress_log_interval_seconds: float = 20.0
    stall_warning_seconds: float = 180.0
    fail_fast: bool = True
    max_retries_semantic: int = 3
    max_retries_transport: int = 3
    task_profile: str = "blob"
    mode: str = "auto"
    prompts_path: str = "configs/nanobanana/prompts.yaml"
    constraints: Dict[str, Any] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalConfig:
    query_policy: str = "model_generated"
    snapshot_policy: str = "live_with_caching"
    scope_policy: str = "open_web"
    primary_exclude_duplicates: bool = True
    primary_exclude_mask_source: bool = True
    include_audit_unavailable_in_primary: bool = True


@dataclass(frozen=True)
class StudyPathsConfig:
    results_root: str = "results_nanobanana"
    artifacts_root: str = "artifacts_nanobanana"


@dataclass(frozen=True)
class LeakageConfig:
    near_hamming_threshold: int = 8


@dataclass(frozen=True)
class StudyConfig:
    study_id: str
    dataset_name: str
    dataset_root: Path
    manifest: Optional[str] = None
    target: str = "target"
    model: StudyModelConfig = field(default_factory=StudyModelConfig)
    matrix: StudyMatrixConfig = field(default_factory=StudyMatrixConfig)
    stages: StageConfig = field(default_factory=StageConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    paths: StudyPathsConfig = field(default_factory=StudyPathsConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)


def _coerce_path(value: str | Path) -> Path:
    expanded = os.path.expandvars(str(value))
    return Path(expanded).expanduser().resolve()


def load_study_config(path: Path) -> StudyConfig:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    model = StudyModelConfig(**(raw.get("model") or {}))
    matrix = StudyMatrixConfig(**(raw.get("matrix") or {}))
    stages = StageConfig(**(raw.get("stages") or {}))
    execution = ExecutionConfig(**(raw.get("execution") or {}))
    retrieval = RetrievalConfig(**(raw.get("retrieval") or {}))
    paths = StudyPathsConfig(**(raw.get("paths") or {}))
    leakage = LeakageConfig(**(raw.get("leakage") or {}))

    dataset_root = raw.get("dataset_root")
    if not dataset_root:
        raise ValueError("dataset_root is required in study config")

    return StudyConfig(
        study_id=str(raw.get("study_id") or "nanobanana_study"),
        dataset_name=str(raw.get("dataset_name") or "dataset"),
        dataset_root=_coerce_path(dataset_root),
        manifest=raw.get("manifest"),
        target=str(raw.get("target") or "target"),
        model=model,
        matrix=matrix,
        stages=stages,
        execution=execution,
        retrieval=retrieval,
        paths=paths,
        leakage=leakage,
    )
