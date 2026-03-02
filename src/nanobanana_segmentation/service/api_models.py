from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ConstraintPayload(BaseModel):
    min_area_frac: float = 0.0
    max_area_frac: float = 1.0
    single_component: bool = False
    allow_border_touch: bool = True
    min_components: int = 1
    max_components: int = 1024


class BudgetPayload(BaseModel):
    max_cost_usd: Optional[float] = None
    max_attempts_total: Optional[int] = None


class SegmentRequestJson(BaseModel):
    image_base64: str
    image_name: str = "upload.png"
    target: str
    mode: Literal["auto", "chromakey", "bw"] = "auto"
    task_profile: Literal["blob", "thin", "low_contrast"] = "blob"
    tool_mode: Literal["closed", "text", "image", "text_image"] = "closed"
    query_policy: Literal["model_generated", "fixed_queries"] = "model_generated"
    snapshot_policy: Literal["live_with_caching", "frozen"] = "live_with_caching"
    scope_policy: Literal["open_web", "curated_domains"] = "open_web"
    thinking_level: Literal["minimal", "high"] = "minimal"
    include_thoughts: bool = False
    max_retries_semantic: int = 3
    max_retries_transport: int = 3
    output: Literal["png", "rle", "coco"] = "png"
    return_debug: bool = False
    constraints: ConstraintPayload = Field(default_factory=ConstraintPayload)
    budget: BudgetPayload = Field(default_factory=BudgetPayload)


class SegmentResponse(BaseModel):
    mask_png_base64: str
    mask_rle: Optional[Dict[str, Any]] = None
    mask_coco: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
