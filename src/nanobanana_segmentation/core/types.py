from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ConstraintConfig:
    min_area_frac: float = 0.0
    max_area_frac: float = 1.0
    single_component: bool = False
    allow_border_touch: bool = True
    min_components: int = 1
    max_components: int = 1024


@dataclass(frozen=True)
class BudgetConfig:
    max_cost_usd: Optional[float] = None
    max_attempts_total: Optional[int] = None


@dataclass(frozen=True)
class EngineRequest:
    image_bytes: bytes
    image_name: str
    image_id: str
    target: str
    mode: str = "auto"
    task_profile: str = "blob"
    tool_mode: str = "closed"
    query_policy: str = "model_generated"
    snapshot_policy: str = "live_with_caching"
    scope_policy: str = "open_web"
    thinking_level: str = "minimal"
    include_thoughts: bool = False
    max_retries_semantic: int = 3
    max_retries_transport: int = 3
    constraints: ConstraintConfig = ConstraintConfig()
    budget: BudgetConfig = BudgetConfig()
    output_format: str = "png"
    return_debug: bool = False
    run_id: str = ""
    replicate_idx: int = 0
    split: str = "unknown"


@dataclass
class QCMetrics:
    resolution_match: bool
    mask_nonempty: bool
    mask_area_frac: float
    component_count: int
    largest_component_frac: float
    speckle_score: float
    border_touch: bool
    green_coverage: Optional[float] = None
    green_uniformity_proxy: Optional[float] = None


@dataclass
class AttemptResult:
    attempt_index: int
    prompt_id: str
    prompt_text: str
    attempt_mode: str
    extraction_method: str
    qc_metrics: QCMetrics
    qc_pass: bool
    failure_reasons: List[str]
    score: float
    transport_retries: int
    transport_retry_events: List[Dict[str, Any]]
    thought_signature_present: bool
    thought_signatures: List[str]
    thought_summaries: List[str]
    grounding: Dict[str, Any]
    raw_request_path: Optional[str]
    raw_response_path: Optional[str]
    surrogate_image_path: Optional[str]
    intermediate_mask_paths: List[str]
    overlay_path: Optional[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class EngineResult:
    run_id: str
    image_id: str
    image_name: str
    selected_attempt_index: int
    qc_pass: bool
    warnings: List[str]
    attempts: List[AttemptResult]
    mask: np.ndarray
    surrogate: Optional[np.ndarray]
    mask_hash: str
    surrogate_hash: Optional[str]
    run_record_path: Optional[str] = None


@dataclass(frozen=True)
class ModelCallResult:
    surrogate_png: Optional[bytes]
    raw_request: Dict[str, Any]
    raw_response: Dict[str, Any]
    thought_signature_present: bool
    thought_signatures: List[str]
    thought_summaries: List[str]
    grounding: Dict[str, Any]


@dataclass
class StudyRow:
    run_id: str
    image_id: str
    image_name: str
    split: str
    tool_mode: str
    query_policy: str
    snapshot_policy: str
    scope_policy: str
    thinking_level: str
    replicate_idx: int
    iou: float
    dice: float
    precision: float
    recall: float
    qc_pass: bool
    selected_attempt_index: int
    retrieval_duplicate: bool
    retrieval_mask_source: bool
    audit_unavailable: bool
    analysis_primary: bool
    analysis_sensitivity: bool
    run_record_path: str
    mask_path: str


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    mask_path: Path
    split: str = "unknown"
