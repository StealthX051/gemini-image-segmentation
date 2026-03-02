from __future__ import annotations

import base64
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import yaml
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from nanobanana_segmentation.core.engine import (
    NanoBananaClient,
    SegmentationEngine,
    encode_coco,
    encode_rle,
)
from nanobanana_segmentation.core.logging.artifact_store import ArtifactStore
from nanobanana_segmentation.core.types import BudgetConfig, ConstraintConfig, EngineRequest
from nanobanana_segmentation.service.api_models import HealthResponse, SegmentRequestJson
from nanobanana_segmentation.service.metrics import (
    LATENCY_HISTOGRAM,
    QC_FAILURE_COUNT,
    REQUEST_COUNT,
    SEMANTIC_ATTEMPT_COUNT,
    TRANSPORT_RETRY_COUNT,
    render_prometheus,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_SERVICE_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "nanobanana" / "service.yaml"


def _load_service_config(path: Optional[Path]) -> Dict[str, Any]:
    cfg_path = path or DEFAULT_SERVICE_CONFIG
    if not cfg_path.exists():
        return {
            "model_id": "gemini-3.1-flash-image-preview",
            "api_surface": "generate_content",
            "artifact_root": "artifacts_nanobanana",
            "results_root": "results_nanobanana",
            "cost_per_attempt_usd": 0.01,
            "use_otsu_bw": False,
            "prompts_path": "configs/nanobanana/prompts.yaml",
            "model_style": "nanobanana_v1",
        }
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_engine(config_path: Optional[Path] = None) -> SegmentationEngine:
    cfg = _load_service_config(config_path)
    model_id = str(cfg.get("model_id", "gemini-3.1-flash-image-preview"))
    api_surface = str(cfg.get("api_surface", "generate_content"))
    artifact_root = Path(str(cfg.get("artifact_root", "artifacts_nanobanana"))).expanduser().resolve()
    cost_per_attempt = float(cfg.get("cost_per_attempt_usd", 0.01))
    use_otsu_bw = bool(cfg.get("use_otsu_bw", False))
    prompts_path_raw = cfg.get("prompts_path", "configs/nanobanana/prompts.yaml")
    prompts_path = Path(str(prompts_path_raw)).expanduser().resolve() if prompts_path_raw else None
    model_style = str(cfg.get("model_style", "nanobanana_v1"))

    client = NanoBananaClient(model_id=model_id, api_surface=api_surface)
    store = ArtifactStore(artifact_root)
    return SegmentationEngine(
        client=client,
        artifact_store=store,
        cost_per_attempt_usd=cost_per_attempt,
        use_otsu_bw=use_otsu_bw,
        prompts_path=prompts_path,
        model_style=model_style,
    )


app = FastAPI(title="NanoBanana Segmentation Service", version="0.1.0")
ENGINE: SegmentationEngine | None = None


def get_engine() -> SegmentationEngine:
    global ENGINE
    if ENGINE is None:
        ENGINE = build_engine()
    return ENGINE


def _mask_to_base64(mask) -> str:
    ok, encoded = cv2.imencode(".png", mask.astype("uint8"))
    if not ok:
        raise RuntimeError("Failed to encode mask PNG")
    return base64.b64encode(bytes(encoded)).decode("utf-8")


def _attempts_to_meta(result) -> Dict[str, Any]:
    engine = get_engine()
    return {
        "selected_attempt_index": result.selected_attempt_index,
        "qc_pass": result.qc_pass,
        "warnings": result.warnings,
        "attempts": [
            {
                "attempt_index": a.attempt_index,
                "prompt_id": a.prompt_id,
                "attempt_mode": a.attempt_mode,
                "extraction_method": a.extraction_method,
                "qc_pass": a.qc_pass,
                "failure_reasons": a.failure_reasons,
                "qc_metrics": {
                    "resolution_match": a.qc_metrics.resolution_match,
                    "mask_nonempty": a.qc_metrics.mask_nonempty,
                    "mask_area_frac": a.qc_metrics.mask_area_frac,
                    "component_count": a.qc_metrics.component_count,
                    "largest_component_frac": a.qc_metrics.largest_component_frac,
                    "speckle_score": a.qc_metrics.speckle_score,
                    "border_touch": a.qc_metrics.border_touch,
                    "green_coverage": a.qc_metrics.green_coverage,
                    "green_uniformity_proxy": a.qc_metrics.green_uniformity_proxy,
                },
                "score": a.score,
                "transport_retries": a.transport_retries,
                "grounding": a.grounding,
                "thought_signature_present": a.thought_signature_present,
                "thought_summaries": a.thought_summaries,
            }
            for a in result.attempts
        ],
        "run_id": result.run_id,
        "run_record_path": result.run_record_path,
        "model_id": engine.client.model_id,
    }


async def _parse_request_payload(request: Request, image_file: UploadFile | None) -> Dict[str, Any]:
    ctype = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in ctype:
        form = await request.form()
        if image_file is None:
            img = form.get("image")
            if isinstance(img, UploadFile):
                image_file = img
        if image_file is None:
            raise HTTPException(status_code=400, detail="Multipart request missing image field")

        image_bytes = await image_file.read()
        payload = {
            "image_bytes": image_bytes,
            "image_name": image_file.filename or "upload.png",
            "target": str(form.get("target") or ""),
            "mode": str(form.get("mode") or "auto"),
            "task_profile": str(form.get("task_profile") or "blob"),
            "tool_mode": str(form.get("tool_mode") or "closed"),
            "query_policy": str(form.get("query_policy") or "model_generated"),
            "snapshot_policy": str(form.get("snapshot_policy") or "live_with_caching"),
            "scope_policy": str(form.get("scope_policy") or "open_web"),
            "thinking_level": str(form.get("thinking_level") or "minimal"),
            "include_thoughts": str(form.get("include_thoughts") or "false").lower() in {"1", "true", "yes"},
            "max_retries_semantic": int(form.get("max_retries_semantic") or 3),
            "max_retries_transport": int(form.get("max_retries_transport") or 3),
            "output": str(form.get("output") or "png"),
            "return_debug": str(form.get("return_debug") or "false").lower() in {"1", "true", "yes"},
            "constraints": {},
            "budget": {},
        }

        constraints_raw = form.get("constraints")
        if isinstance(constraints_raw, str) and constraints_raw.strip():
            payload["constraints"] = json.loads(constraints_raw)

        budget_raw = form.get("budget")
        if isinstance(budget_raw, str) and budget_raw.strip():
            payload["budget"] = json.loads(budget_raw)

        if not payload["target"]:
            raise HTTPException(status_code=400, detail="target is required")
        return payload

    body = await request.json()
    if hasattr(SegmentRequestJson, "model_validate"):
        parsed = SegmentRequestJson.model_validate(body)
    else:  # pragma: no cover - pydantic v1 fallback
        parsed = SegmentRequestJson.parse_obj(body)
    try:
        image_bytes = base64.b64decode(parsed.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64 payload: {exc}") from exc

    return {
        "image_bytes": image_bytes,
        "image_name": parsed.image_name,
        "target": parsed.target,
        "mode": parsed.mode,
        "task_profile": parsed.task_profile,
        "tool_mode": parsed.tool_mode,
        "query_policy": parsed.query_policy,
        "snapshot_policy": parsed.snapshot_policy,
        "scope_policy": parsed.scope_policy,
        "thinking_level": parsed.thinking_level,
        "include_thoughts": parsed.include_thoughts,
        "max_retries_semantic": parsed.max_retries_semantic,
        "max_retries_transport": parsed.max_retries_transport,
        "output": parsed.output,
        "return_debug": parsed.return_debug,
        "constraints": parsed.constraints.model_dump(),
        "budget": parsed.budget.model_dump(),
    }


@app.post("/v1/segment")
async def segment(request: Request, image: UploadFile | None = File(default=None)):
    started = datetime.utcnow()
    payload = await _parse_request_payload(request, image)
    engine = get_engine()

    tool_mode = str(payload.get("tool_mode", "closed"))
    REQUEST_COUNT.labels(tool_mode=tool_mode).inc()

    constraints = ConstraintConfig(**(payload.get("constraints") or {}))
    budget = BudgetConfig(**(payload.get("budget") or {}))

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    image_name = str(payload.get("image_name") or "upload.png")
    image_id = Path(image_name).stem or uuid.uuid4().hex[:12]

    engine_request = EngineRequest(
        image_bytes=payload["image_bytes"],
        image_name=image_name,
        image_id=image_id,
        target=payload["target"],
        mode=payload.get("mode", "auto"),
        task_profile=payload.get("task_profile", "blob"),
        tool_mode=tool_mode,
        query_policy=str(payload.get("query_policy", "model_generated")),
        snapshot_policy=str(payload.get("snapshot_policy", "live_with_caching")),
        scope_policy=str(payload.get("scope_policy", "open_web")),
        thinking_level=payload.get("thinking_level", "minimal"),
        include_thoughts=bool(payload.get("include_thoughts", False)),
        max_retries_semantic=int(payload.get("max_retries_semantic", 3)),
        max_retries_transport=int(payload.get("max_retries_transport", 3)),
        constraints=constraints,
        budget=budget,
        output_format=payload.get("output", "png"),
        return_debug=bool(payload.get("return_debug", False)),
        run_id=run_id,
    )

    try:
        result = engine.segment_once(engine_request)
    except Exception as exc:
        LOGGER.exception("Segmentation request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        elapsed = (datetime.utcnow() - started).total_seconds()
        LATENCY_HISTOGRAM.observe(elapsed)

    total_transport_retries = sum(a.transport_retries for a in result.attempts)
    if total_transport_retries:
        TRANSPORT_RETRY_COUNT.inc(total_transport_retries)
    SEMANTIC_ATTEMPT_COUNT.inc(len(result.attempts))

    for attempt in result.attempts:
        if not attempt.qc_pass:
            for reason in attempt.failure_reasons:
                QC_FAILURE_COUNT.labels(reason=reason).inc()

    response_payload: Dict[str, Any] = {
        "mask_png_base64": _mask_to_base64(result.mask),
        "meta": _attempts_to_meta(result),
    }

    output_mode = str(payload.get("output", "png"))
    if output_mode in {"rle", "coco"}:
        response_payload["mask_rle"] = encode_rle(result.mask)
    if output_mode == "coco":
        response_payload["mask_coco"] = encode_coco(result.mask)

    if bool(payload.get("return_debug", False)):
        response_payload["debug"] = {
            "run_record_path": result.run_record_path,
            "attempt_paths": [
                {
                    "attempt_index": a.attempt_index,
                    "raw_request_path": a.raw_request_path,
                    "raw_response_path": a.raw_response_path,
                    "surrogate_image_path": a.surrogate_image_path,
                    "intermediate_mask_paths": a.intermediate_mask_paths,
                    "overlay_path": a.overlay_path,
                    "prompt_text": a.prompt_text,
                    "grounding": a.grounding,
                    "thought_summary": a.thought_summaries,
                }
                for a in result.attempts
            ],
        }

    return JSONResponse(response_payload)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_prometheus()
    return Response(content=payload, media_type=content_type)
