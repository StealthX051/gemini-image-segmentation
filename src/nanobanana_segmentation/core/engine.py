from __future__ import annotations

import io
import json
import logging
import random
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .extract.bw_threshold import extract_bw_threshold
from .extract.chromakey_hsv import extract_chromakey_hsv
from .extract.chromakey_ratio import extract_chromakey_ratio
from .extract.postprocess import standard_postprocess
from .grounding.parse_grounding import parse_grounding_fields, parse_thought_fields
from .logging.artifact_store import ArtifactStore
from .logging.run_record import RunRecord, persist_run_record
from .prompts import PromptAttempt, build_retry_ladder_prompts, load_prompt_templates
from .qc import compute_qc_metrics, evaluate_qc, score_attempt
from .types import AttemptResult, EngineRequest, EngineResult, ModelCallResult


def _serialize_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {str(k): _serialize_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_obj(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _serialize_obj(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _serialize_obj({k: v for k, v in value.__dict__.items() if not k.startswith("_")})
    return str(value)


def _image_to_png_bytes(image: Image.Image) -> bytes:
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        return buf.getvalue()


def _png_bytes_to_bgr(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Unable to decode image bytes")
    return decoded


def _mask_to_png(mask: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8))
    if not ok:
        raise ValueError("Failed to encode mask PNG")
    return bytes(encoded)


def _resize_mask_to_shape(mask: np.ndarray, target_shape: Tuple[int, int]) -> Tuple[np.ndarray, bool]:
    target_h, target_w = target_shape
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeeze, got shape={arr.shape}")
    out = (arr > 0).astype(np.uint8) * 255
    if out.shape == (target_h, target_w):
        return out, False
    resized = cv2.resize(out, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.uint8), True


def _overlay_png(input_rgb: np.ndarray, mask: np.ndarray) -> bytes:
    overlay = input_rgb.copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    normalized_mask, _ = _resize_mask_to_shape(mask, overlay.shape[:2])
    fg = (normalized_mask > 0)[:, :, None]
    overlay = np.where(fg, ((0.65 * overlay) + (0.35 * red)).astype(np.uint8), overlay)
    with io.BytesIO() as buf:
        Image.fromarray(overlay).save(buf, format="PNG")
        return buf.getvalue()


def _precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    tp = float(np.logical_and(y_true, y_pred).sum())
    fp = float(np.logical_and(~y_true, y_pred).sum())
    fn = float(np.logical_and(y_true, ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    flat = (mask > 0).astype(np.uint8).ravel(order="F")
    counts: List[int] = []
    prev = 0
    run = 0
    for value in flat:
        if int(value) == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = int(value)
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _bbox_from_mask(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return [x0, y0, x1, y1]


def encode_coco(mask: np.ndarray) -> Dict[str, Any]:
    bbox = _bbox_from_mask(mask)
    area = int((mask > 0).sum())
    return {
        "bbox": bbox or [0, 0, 0, 0],
        "area": area,
        "segmentation": encode_rle(mask),
    }


class NanoBananaClient:
    """Adapter for NanoBanana image-generation segmentation surrogates."""

    def __init__(
        self,
        *,
        model_id: str,
        api_surface: str = "generate_content",
        timeout_s: float = 60.0,
        client: Any | None = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.api_surface = api_surface
        self.timeout_s = float(timeout_s)
        self._client = client or self._build_client(api_key=api_key)

    @staticmethod
    def _build_client(api_key: Optional[str] = None) -> Any:
        try:
            from google import genai
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise ImportError(
                "google-genai is required for NanoBanana calls. Install dependencies and export GOOGLE_API_KEY."
            ) from exc

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        return genai.Client(**kwargs)

    @staticmethod
    def _normalize_tool_mode(tool_mode: str) -> str:
        mode = (tool_mode or "closed").strip().lower()
        if mode in {"closed", "text", "image", "text_image"}:
            return mode
        return "closed"

    @classmethod
    def _resolve_tools_payload(
        cls,
        *,
        tool_mode: str,
        tool_fields: Sequence[str],
        google_search_fields: Sequence[str] = (),
        search_type_fields: Sequence[str] = (),
    ) -> Tuple[List[Dict[str, Any]], str, List[str]]:
        requested_mode = cls._normalize_tool_mode(tool_mode)
        fields = set(tool_fields)
        google_fields = set(google_search_fields)
        search_fields = set(search_type_fields)
        warnings: List[str] = []
        tools_payload: List[Dict[str, Any]] = []

        if requested_mode == "closed":
            return tools_payload, "closed", warnings

        text_requested = requested_mode in {"text", "text_image"}
        image_requested = requested_mode in {"image", "text_image"}

        if "google_search" in fields:
            supports_scoped_search = (
                "search_types" in google_fields
                and "web_search" in search_fields
                and "image_search" in search_fields
            )
            if supports_scoped_search:
                scoped_types: List[Dict[str, Dict[str, Any]]] = []
                if text_requested:
                    scoped_types.append({"web_search": {}})
                if image_requested:
                    scoped_types.append({"image_search": {}})
                if scoped_types:
                    tools_payload.append({"google_search": {"search_types": scoped_types}})
            else:
                # Older SDKs expose google_search without search-type selectors.
                tools_payload.append({"google_search": {}})
                if image_requested:
                    warnings.append("tool_image_search_unavailable_in_sdk")
        elif "google_search_retrieval" in fields:
            tools_payload.append({"google_search_retrieval": {}})
            if image_requested:
                warnings.append("tool_image_search_unavailable_in_sdk")
        else:
            if text_requested:
                warnings.append("tool_text_search_unavailable_in_sdk")
            if image_requested:
                warnings.append("tool_image_search_unavailable_in_sdk")

        effective_mode = "closed"
        has_text = any("google_search" in item or "google_search_retrieval" in item for item in tools_payload)
        has_image = False
        for item in tools_payload:
            google_search_cfg = item.get("google_search")
            if not isinstance(google_search_cfg, dict):
                continue
            scoped = google_search_cfg.get("search_types")
            if not isinstance(scoped, list):
                continue
            if any(isinstance(entry, dict) and "image_search" in entry for entry in scoped):
                has_image = True
                break
        if has_text and has_image:
            effective_mode = "text_image"
        elif has_image:
            effective_mode = "image"
        elif has_text:
            effective_mode = "text"

        if effective_mode != requested_mode:
            warnings.append(f"tool_mode_fallback:{requested_mode}->{effective_mode}")

        return tools_payload, effective_mode, warnings

    @staticmethod
    def _thinking_budget(thinking_level: str) -> int:
        return 1024 if (thinking_level or "minimal").strip().lower() == "high" else 0

    @staticmethod
    def _extract_surrogate_png(raw_response: Any) -> Optional[bytes]:
        import base64

        def _as_list(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return [value]

        def _field(node: Any, *names: str) -> Any:
            if isinstance(node, dict):
                for name in names:
                    if name in node:
                        return node.get(name)
                return None
            for name in names:
                if hasattr(node, name):
                    return getattr(node, name)
            return None

        def _decode_inline_data(data: Any) -> Optional[bytes]:
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
            if isinstance(data, str):
                try:
                    # Python SDK can expose base64 strings in some shapes.
                    return base64.b64decode(data, validate=True)
                except Exception:
                    return None
            return None

        def _extract_from_parts(parts: Any) -> Optional[bytes]:
            for part in _as_list(parts):
                inline_data = _field(part, "inline_data", "inlineData")
                if inline_data is not None:
                    decoded = _decode_inline_data(_field(inline_data, "data"))
                    if decoded:
                        return decoded

                file_data = _field(part, "file_data", "fileData")
                if file_data is not None:
                    uri = _field(file_data, "file_uri", "fileUri")
                    if isinstance(uri, str) and uri.startswith("data:image/") and ";base64," in uri:
                        try:
                            return base64.b64decode(uri.split(",", 1)[1], validate=True)
                        except Exception:
                            pass

                as_image = getattr(part, "as_image", None)
                if callable(as_image):
                    try:
                        image = as_image()
                        if image is not None:
                            return _image_to_png_bytes(image)
                    except Exception:
                        continue
            return None

        # Preferred path per Python SDK docs: response.parts / part.inline_data / part.as_image()
        direct = _extract_from_parts(_field(raw_response, "parts"))
        if direct:
            return direct

        # Candidate/content path used by many responses.
        for candidate in _as_list(_field(raw_response, "candidates")):
            content = _field(candidate, "content")
            decoded = _extract_from_parts(_field(content, "parts"))
            if decoded:
                return decoded

        # Fallback for unexpected payload shapes.
        payload = _serialize_obj(raw_response)

        def walk(node: Any):
            if isinstance(node, dict):
                yield node
                for value in node.values():
                    yield from walk(value)
            elif isinstance(node, list):
                for item in node:
                    yield from walk(item)

        for node in walk(payload):
            inline_data = node.get("inline_data") or node.get("inlineData")
            if isinstance(inline_data, dict):
                decoded = _decode_inline_data(inline_data.get("data"))
                if decoded:
                    return decoded
            file_data = node.get("file_data") or node.get("fileData")
            if isinstance(file_data, dict):
                uri = file_data.get("file_uri") or file_data.get("fileUri")
                if isinstance(uri, str) and uri.startswith("data:image/") and ";base64," in uri:
                    try:
                        return base64.b64decode(uri.split(",", 1)[1], validate=True)
                    except Exception:
                        continue
        return None

    def _generate_content_call(
        self,
        *,
        prompt: str,
        image_png: bytes,
        tool_mode: str,
        thinking_level: str,
        include_thoughts: bool,
    ) -> Tuple[Dict[str, Any], Any]:
        from google.genai import types as genai_types  # type: ignore

        tool_fields = tuple(getattr(genai_types.Tool, "model_fields", {}).keys())
        google_search_fields = tuple(getattr(getattr(genai_types, "GoogleSearch", object), "model_fields", {}).keys())
        search_type_fields = tuple(getattr(getattr(genai_types, "SearchType", object), "model_fields", {}).keys())
        tools_payload, effective_tool_mode, tool_warnings = self._resolve_tools_payload(
            tool_mode=tool_mode,
            tool_fields=tool_fields,
            google_search_fields=google_search_fields,
            search_type_fields=search_type_fields,
        )

        thinking_kwargs: Dict[str, Any] = {"thinking_budget": self._thinking_budget(thinking_level)}
        if include_thoughts:
            thinking_kwargs["include_thoughts"] = True
        config_kwargs: Dict[str, Any] = {
            "thinking_config": genai_types.ThinkingConfig(**thinking_kwargs),
            "response_modalities": ["IMAGE"],
        }
        if tools_payload:
            config_kwargs["tools"] = tools_payload

        config = genai_types.GenerateContentConfig(**config_kwargs)
        contents = [
            genai_types.Part.from_bytes(data=image_png, mime_type="image/png"),
            genai_types.Part(text=prompt),
        ]

        request_payload = {
            "api_surface": "generate_content",
            "model": self.model_id,
            "tool_mode_requested": self._normalize_tool_mode(tool_mode),
            "tool_mode_effective": effective_tool_mode,
            "thinking_level": thinking_level,
            "include_thoughts": include_thoughts,
            "contents": ["<image/png bytes>", prompt],
            "tools_payload": _serialize_obj(tools_payload),
            "warnings": tool_warnings,
            "config": _serialize_obj(config_kwargs),
        }
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
        )
        return request_payload, response

    def _interactions_call(
        self,
        *,
        prompt: str,
        image_png: bytes,
        tool_mode: str,
        thinking_level: str,
        include_thoughts: bool,
    ) -> Tuple[Dict[str, Any], Any]:
        # Feature-flagged adapter. Uses the same client while preserving request metadata.
        try:
            from google.genai import types as genai_types  # type: ignore

            tool_fields = tuple(getattr(genai_types.Tool, "model_fields", {}).keys())
            google_search_fields = tuple(getattr(getattr(genai_types, "GoogleSearch", object), "model_fields", {}).keys())
            search_type_fields = tuple(getattr(getattr(genai_types, "SearchType", object), "model_fields", {}).keys())
        except Exception:
            tool_fields = ("google_search",)
            google_search_fields = ()
            search_type_fields = ()
        tools_payload, effective_tool_mode, tool_warnings = self._resolve_tools_payload(
            tool_mode=tool_mode,
            tool_fields=tool_fields,
            google_search_fields=google_search_fields,
            search_type_fields=search_type_fields,
        )
        request_payload = {
            "api_surface": "interactions",
            "model": self.model_id,
            "tool_mode_requested": self._normalize_tool_mode(tool_mode),
            "tool_mode_effective": effective_tool_mode,
            "thinking_level": thinking_level,
            "include_thoughts": include_thoughts,
            "contents": ["<image/png bytes>", prompt],
            "tools_payload": _serialize_obj(tools_payload),
            "warnings": tool_warnings,
        }
        if hasattr(self._client, "interactions"):
            response = self._client.interactions.create(
                model=self.model_id,
                input={"image": image_png, "prompt": prompt},
                tools=tools_payload or None,
                thinking_level=thinking_level,
                include_thoughts=include_thoughts,
            )
            return request_payload, response
        return self._generate_content_call(
            prompt=prompt,
            image_png=image_png,
            tool_mode=tool_mode,
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
        )

    def generate_mask_surrogate(
        self,
        *,
        prompt: str,
        image_png: bytes,
        tool_mode: str,
        thinking_level: str,
        include_thoughts: bool,
    ) -> ModelCallResult:
        if self.api_surface == "interactions":
            raw_request, response = self._interactions_call(
                prompt=prompt,
                image_png=image_png,
                tool_mode=tool_mode,
                thinking_level=thinking_level,
                include_thoughts=include_thoughts,
            )
        else:
            raw_request, response = self._generate_content_call(
                prompt=prompt,
                image_png=image_png,
                tool_mode=tool_mode,
                thinking_level=thinking_level,
                include_thoughts=include_thoughts,
            )

        raw_response = _serialize_obj(response)
        thought = parse_thought_fields(raw_response)
        grounding = parse_grounding_fields(raw_response)
        surrogate_png = self._extract_surrogate_png(response)

        return ModelCallResult(
            surrogate_png=surrogate_png,
            raw_request=raw_request,
            raw_response=raw_response,
            thought_signature_present=bool(thought["thought_signature_present"]),
            thought_signatures=list(thought["thought_signatures"]),
            thought_summaries=list(thought["thought_summaries"]),
            grounding=grounding,
        )


class SegmentationEngine:
    def __init__(
        self,
        *,
        client: NanoBananaClient,
        artifact_store: ArtifactStore,
        cost_per_attempt_usd: float = 0.01,
        use_otsu_bw: bool = False,
        random_seed: int = 7,
        prompts_path: Optional[Path] = None,
        model_style: str = "nanobanana_v1",
    ) -> None:
        self.client = client
        self.artifact_store = artifact_store
        self.cost_per_attempt_usd = float(cost_per_attempt_usd)
        self.use_otsu_bw = bool(use_otsu_bw)
        self.rng = random.Random(random_seed)
        self.model_style = str(model_style or "nanobanana_v1")
        self.prompt_templates = load_prompt_templates(prompts_path)

    @staticmethod
    def _retryable_exception(exc: Exception) -> bool:
        text = str(exc).lower()
        code = getattr(exc, "status_code", None)
        if code in {429, 500, 502, 503, 504}:
            return True
        return any(token in text for token in ("timeout", "timed out", "429", "503", "500", "502", "504"))

    def _call_with_transport_retries(
        self,
        *,
        prompt: str,
        image_png: bytes,
        tool_mode: str,
        thinking_level: str,
        include_thoughts: bool,
        max_retries_transport: int,
    ) -> Tuple[ModelCallResult, int, List[Dict[str, Any]]]:
        retries = max(0, int(max_retries_transport))
        attempt = 0
        events: List[Dict[str, Any]] = []

        while True:
            attempt += 1
            try:
                result = self.client.generate_mask_surrogate(
                    prompt=prompt,
                    image_png=image_png,
                    tool_mode=tool_mode,
                    thinking_level=thinking_level,
                    include_thoughts=include_thoughts,
                )
                return result, attempt - 1, events
            except Exception as exc:
                if attempt > retries + 1 or not self._retryable_exception(exc):
                    raise
                delay = min(8.0, (2 ** (attempt - 1)) * 0.5)
                delay = delay + self.rng.uniform(0.0, 0.25)
                events.append(
                    {
                        "attempt": attempt,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "error": str(exc),
                        "sleep_s": round(delay, 3),
                    }
                )
                time.sleep(delay)

    @staticmethod
    def _choose_best_candidate(candidates: Sequence[Tuple[str, np.ndarray, np.ndarray, float]]) -> Tuple[str, np.ndarray, np.ndarray, float]:
        if not candidates:
            raise ValueError("No extraction candidates were generated")
        return sorted(candidates, key=lambda item: item[3], reverse=True)[0]

    def _extract_candidate_masks(
        self,
        *,
        surrogate_bgr: np.ndarray,
        task_profile: str,
        input_shape: Tuple[int, int],
        attempt_mode: str,
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        h, w = input_shape
        area = h * w
        candidates: List[Tuple[str, np.ndarray, np.ndarray]] = []

        if attempt_mode == "bw":
            raw = extract_bw_threshold(surrogate_bgr, threshold=127, use_otsu=self.use_otsu_bw)
            post = standard_postprocess(
                raw,
                image_area=area,
                task_profile=task_profile,
                max_components=1024,
            )
            candidates.append(("bw_threshold", raw, post))
            return candidates

        raw_hsv = extract_chromakey_hsv(surrogate_bgr)
        post_hsv = standard_postprocess(
            raw_hsv,
            image_area=area,
            task_profile=task_profile,
            max_components=1024,
        )
        candidates.append(("chromakey_hsv", raw_hsv, post_hsv))

        raw_ratio = extract_chromakey_ratio(surrogate_bgr)
        post_ratio = standard_postprocess(
            raw_ratio,
            image_area=area,
            task_profile=task_profile,
            max_components=1024,
        )
        candidates.append(("chromakey_ratio", raw_ratio, post_ratio))
        return candidates

    def segment_once(self, request: EngineRequest) -> EngineResult:
        run_id = request.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_dir = self.artifact_store.run_dir(
            image_id=request.image_id,
            mode=request.tool_mode,
            replicate_idx=int(request.replicate_idx),
            run_id=run_id,
        )

        image = Image.open(io.BytesIO(request.image_bytes)).convert("RGB")
        input_rgb = np.array(image)
        h, w = input_rgb.shape[:2]
        input_png = _image_to_png_bytes(image)

        input_path, input_hash = self.artifact_store.write_bytes(
            run_dir=run_dir,
            stem="input",
            payload=input_png,
            suffix=".png",
            subdir="input",
        )

        warnings: List[str] = []

        attempts_spec = build_retry_ladder_prompts(
            target=request.target,
            task_profile=request.task_profile,
            constraints=asdict(request.constraints),
            max_attempts=request.max_retries_semantic,
            tool_mode=request.tool_mode,
            model_style=self.model_style,
            templates=self.prompt_templates,
        )
        requested_mode = (request.mode or "auto").strip().lower()
        if requested_mode in {"chromakey", "bw"}:
            filtered_attempts = [spec for spec in attempts_spec if spec.attempt_mode == requested_mode]
            if filtered_attempts:
                attempts_spec = filtered_attempts
            else:
                warnings.append(f"mode_filter_empty:{requested_mode}")
        elif requested_mode not in {"", "auto"}:
            warnings.append(f"unknown_mode:{requested_mode}")

        max_attempts_total = request.budget.max_attempts_total
        attempts: List[AttemptResult] = []
        selected_masks: Dict[int, np.ndarray] = {}

        spent_usd = 0.0
        if request.budget.max_cost_usd is not None and request.budget.max_cost_usd <= 0:
            raise ValueError("budget.max_cost_usd must be positive when provided")

        for idx, spec in enumerate(attempts_spec, start=1):
            if max_attempts_total is not None and idx > int(max_attempts_total):
                warnings.append("budget_max_attempts_total_reached")
                break
            if request.budget.max_cost_usd is not None and spent_usd + self.cost_per_attempt_usd > request.budget.max_cost_usd:
                warnings.append("budget_max_cost_usd_reached")
                break

            model_result, transport_retries, retry_events = self._call_with_transport_retries(
                prompt=spec.prompt_text,
                image_png=input_png,
                tool_mode=request.tool_mode,
                thinking_level=request.thinking_level,
                include_thoughts=request.include_thoughts,
                max_retries_transport=request.max_retries_transport,
            )
            raw_call_warnings = model_result.raw_request.get("warnings")
            if isinstance(raw_call_warnings, list):
                call_warnings = [str(item) for item in raw_call_warnings]
            elif raw_call_warnings:
                call_warnings = [str(raw_call_warnings)]
            else:
                call_warnings = []
            for warn in call_warnings:
                if warn not in warnings:
                    warnings.append(warn)
            spent_usd += self.cost_per_attempt_usd

            req_path = self.artifact_store.write_json(
                run_dir=run_dir,
                name=f"attempt_{idx:02d}_request.json",
                payload=model_result.raw_request,
                subdir="raw",
            )
            resp_path = self.artifact_store.write_json(
                run_dir=run_dir,
                name=f"attempt_{idx:02d}_response.json",
                payload=model_result.raw_response,
                subdir="raw",
            )

            surrogate_png = model_result.surrogate_png
            if not surrogate_png:
                zero = np.zeros((h, w), dtype=np.uint8)
                metrics = compute_qc_metrics(
                    mask=zero,
                    surrogate=np.zeros((h, w, 3), dtype=np.uint8),
                    input_shape=(h, w),
                    attempt_mode=spec.attempt_mode,
                )
                qc_pass, failures = evaluate_qc(metrics, request.constraints, attempt_mode=spec.attempt_mode)
                score = score_attempt(metrics, qc_pass=False, failures=failures + ["no_surrogate_output"])
                attempts.append(
                    AttemptResult(
                        attempt_index=idx,
                        prompt_id=spec.prompt_id,
                        prompt_text=spec.prompt_text,
                        attempt_mode=spec.attempt_mode,
                        extraction_method="none",
                        qc_metrics=metrics,
                        qc_pass=False,
                        failure_reasons=failures + ["no_surrogate_output"],
                        score=score,
                        transport_retries=transport_retries,
                        transport_retry_events=retry_events,
                        thought_signature_present=model_result.thought_signature_present,
                        thought_signatures=model_result.thought_signatures,
                        thought_summaries=model_result.thought_summaries,
                        grounding=model_result.grounding,
                        raw_request_path=str(req_path),
                        raw_response_path=str(resp_path),
                        surrogate_image_path=None,
                        intermediate_mask_paths=[],
                        overlay_path=None,
                        warnings=call_warnings,
                    )
                )
                continue

            surrogate_path, surrogate_hash = self.artifact_store.write_bytes(
                run_dir=run_dir,
                stem=f"attempt_{idx:02d}_surrogate",
                payload=surrogate_png,
                suffix=".png",
                subdir="surrogate",
            )
            surrogate_bgr = _png_bytes_to_bgr(surrogate_png)

            candidates = self._extract_candidate_masks(
                surrogate_bgr=surrogate_bgr,
                task_profile=request.task_profile,
                input_shape=(h, w),
                attempt_mode=spec.attempt_mode,
            )

            scored_candidates: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
            intermediate_paths: List[str] = []
            for method_name, raw_mask, post_mask in candidates:
                raw_path, _ = self.artifact_store.write_bytes(
                    run_dir=run_dir,
                    stem=f"attempt_{idx:02d}_{method_name}_raw",
                    payload=_mask_to_png(raw_mask),
                    suffix=".png",
                    subdir="intermediate",
                )
                post_path, _ = self.artifact_store.write_bytes(
                    run_dir=run_dir,
                    stem=f"attempt_{idx:02d}_{method_name}_post",
                    payload=_mask_to_png(post_mask),
                    suffix=".png",
                    subdir="intermediate",
                )
                intermediate_paths.extend([str(raw_path), str(post_path)])

                metrics = compute_qc_metrics(
                    mask=post_mask,
                    surrogate=surrogate_bgr,
                    input_shape=(h, w),
                    attempt_mode=spec.attempt_mode,
                )
                candidate_qc_pass, candidate_failures = evaluate_qc(
                    metrics,
                    request.constraints,
                    attempt_mode=spec.attempt_mode,
                )
                candidate_score = score_attempt(metrics, qc_pass=candidate_qc_pass, failures=candidate_failures)
                scored_candidates.append((method_name, raw_mask, post_mask, candidate_score))

            selected_method, _, selected_mask, _ = self._choose_best_candidate(scored_candidates)

            metrics = compute_qc_metrics(
                mask=selected_mask,
                surrogate=surrogate_bgr,
                input_shape=(h, w),
                attempt_mode=spec.attempt_mode,
            )
            qc_pass, failures = evaluate_qc(metrics, request.constraints, attempt_mode=spec.attempt_mode)
            attempt_warnings = list(call_warnings)
            selected_mask_output, resized_mask = _resize_mask_to_shape(selected_mask, (h, w))
            if resized_mask:
                if "resolution_mismatch" not in failures:
                    failures = list(failures) + ["resolution_mismatch"]
                qc_pass = False
                if "resized_mask_to_input_shape" not in attempt_warnings:
                    attempt_warnings.append("resized_mask_to_input_shape")
                if "resized_mask_to_input_shape" not in warnings:
                    warnings.append("resized_mask_to_input_shape")
            score = score_attempt(metrics, qc_pass=qc_pass, failures=failures)

            overlay_path, _ = self.artifact_store.write_bytes(
                run_dir=run_dir,
                stem=f"attempt_{idx:02d}_overlay",
                payload=_overlay_png(input_rgb, selected_mask_output),
                suffix=".png",
                subdir="overlay",
            )

            attempts.append(
                AttemptResult(
                    attempt_index=idx,
                    prompt_id=spec.prompt_id,
                    prompt_text=spec.prompt_text,
                    attempt_mode=spec.attempt_mode,
                    extraction_method=selected_method,
                    qc_metrics=metrics,
                    qc_pass=qc_pass,
                    failure_reasons=failures,
                    score=score,
                    transport_retries=transport_retries,
                    transport_retry_events=retry_events,
                    thought_signature_present=model_result.thought_signature_present,
                    thought_signatures=model_result.thought_signatures,
                    thought_summaries=model_result.thought_summaries,
                    grounding=model_result.grounding,
                    raw_request_path=str(req_path),
                    raw_response_path=str(resp_path),
                    surrogate_image_path=str(surrogate_path),
                    intermediate_mask_paths=intermediate_paths,
                    overlay_path=str(overlay_path),
                    warnings=attempt_warnings,
                )
            )
            selected_masks[idx] = selected_mask_output.copy()

        if not attempts:
            raise RuntimeError("No attempts executed; check budget and retry settings")

        selected = sorted(attempts, key=lambda x: x.score, reverse=True)[0]
        selected_mask = selected_masks.get(selected.attempt_index, np.zeros((h, w), dtype=np.uint8))

        final_mask_path, final_mask_hash = self.artifact_store.write_bytes(
            run_dir=run_dir,
            stem="final_mask",
            payload=_mask_to_png(selected_mask),
            suffix=".png",
            subdir="final",
        )

        run_record = RunRecord(
            run_id=run_id,
            image_id=request.image_id,
            image_name=request.image_name,
            split=request.split,
            mode=request.tool_mode,
            replicate_idx=int(request.replicate_idx),
            model_config={
                "model_id": self.client.model_id,
                "thinking_level": request.thinking_level,
                "include_thoughts": request.include_thoughts,
                "api_surface": self.client.api_surface,
            },
            tool_config={
                "tool_mode": request.tool_mode,
                "query_policy": request.query_policy,
                "snapshot_policy": request.snapshot_policy,
                "scope_policy": request.scope_policy,
            },
            prompt={
                "template_version": "nanobanana_v1",
                "target": request.target,
                "task_profile": request.task_profile,
            },
            attempts=[
                {
                    "attempt_index": a.attempt_index,
                    "prompt_id": a.prompt_id,
                    "prompt_text": a.prompt_text,
                    "attempt_mode": a.attempt_mode,
                    "extraction_method": a.extraction_method,
                    "qc_metrics": asdict(a.qc_metrics),
                    "qc_pass": a.qc_pass,
                    "failure_reasons": a.failure_reasons,
                    "score": a.score,
                    "transport_retries": a.transport_retries,
                    "transport_retry_events": a.transport_retry_events,
                    "thought_signature_present": a.thought_signature_present,
                    "thought_signatures": a.thought_signatures,
                    "thought_summaries": a.thought_summaries,
                    "grounding": a.grounding,
                    "raw_request_path": a.raw_request_path,
                    "raw_response_path": a.raw_response_path,
                    "surrogate_image_path": a.surrogate_image_path,
                    "intermediate_mask_paths": a.intermediate_mask_paths,
                    "overlay_path": a.overlay_path,
                    "warnings": a.warnings,
                }
                for a in attempts
            ],
            outputs={
                "input_path": str(input_path),
                "input_hash": input_hash,
                "final_mask_path": str(final_mask_path),
                "final_mask_hash": final_mask_hash,
            },
            evaluation={},
            leakage={
                "retrieval_duplicate": False,
                "retrieval_mask_source": False,
                "audit_status": "pending",
            },
            warnings=warnings,
        )
        run_record_path = persist_run_record(self.artifact_store, run_dir=run_dir, record=run_record)

        return EngineResult(
            run_id=run_id,
            image_id=request.image_id,
            image_name=request.image_name,
            selected_attempt_index=selected.attempt_index,
            qc_pass=selected.qc_pass,
            warnings=warnings,
            attempts=attempts,
            mask=selected_mask,
            surrogate=None,
            mask_hash=final_mask_hash,
            surrogate_hash=None,
            run_record_path=str(run_record_path),
        )
