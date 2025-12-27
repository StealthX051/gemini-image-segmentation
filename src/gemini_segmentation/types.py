from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class SegmentationMask:
    """A single segmentation mask aligned to the full image space."""

    y0: int
    x0: int
    y1: int
    x1: int
    mask: np.ndarray  # uint8 mask in image coordinates (0-255)
    label: str = ""


@dataclass
class PredictionRecord:
    """Structured metadata for a single image prediction."""

    image_name: str
    latency_s: float
    parse_success: bool
    timed_out: bool
    num_masks: int
    prediction_path: Optional[Path]
    overlay_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    legacy_json_path: Optional[Path] = None
    raw_response_path: Optional[Path] = None


@dataclass
class PerImageMetrics:
    image_name: str
    iou: float
    dice: float
    success: bool


@dataclass
class RunConfig:
    dataset_name: str
    dataset_root: Path
    model_name: str
    prompt: str
    thinking_budget: int = 0
    temperature: float = 0.5
    safety_settings: Optional[dict] = None
    timeout_s: float = 60.0
    workers: int = 1
    sample_size: Optional[int] = None
    manifest_path: Optional[Path] = None
    rate_limit_s: Optional[float] = None
    legacy_predictions: bool = False
    run_id: Optional[str] = None
    bootstrap_method: str = "bca"
    bootstrap_resamples: int = 5000


@dataclass
class FairnessResult:
    image_name: str
    ita: float
    chardon_label: str
    tone_group: str
    iou: float
    dice: float
    success: bool
    candidate_count: int = 0


@dataclass
class BootstrapCI:
    lower: float
    upper: float


@dataclass
class GroupSummary:
    group_name: str
    count: int
    mean_iou: float
    median_iou: float
    ci_iou: BootstrapCI
    mean_dice: float
    median_dice: float
    ci_dice: BootstrapCI
    success_rate: float


@dataclass
class RunSummary:
    metrics: List[PerImageMetrics] = field(default_factory=list)
    mean_iou: float = 0.0
    median_iou: float = 0.0
    ci_iou: BootstrapCI = field(default_factory=lambda: BootstrapCI(0.0, 0.0))
    mean_dice: float = 0.0
    median_dice: float = 0.0
    ci_dice: BootstrapCI = field(default_factory=lambda: BootstrapCI(0.0, 0.0))
    success_rate: float = 0.0
