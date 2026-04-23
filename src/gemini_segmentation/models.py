from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from google import genai
from google.genai import types as genai_types
from google.genai.types import GenerateContentConfig, Part, SafetySetting, ThinkingConfig
import numpy as np
from PIL import Image

from .gemini_capabilities import (
    gemini_supports_explicit_cache,
    gemini_supports_structured_output,
)
from .io import encode_mask_to_b64, parse_segmentation_masks
from .prompts import GEMINI_SEGMENTATION_RESPONSE_JSON_SCHEMA
from .types import SegmentationMask

T = TypeVar("T")


def _run_with_timeout(
    func: Callable[[], T], timeout_s: float, timeout_msg: str
) -> tuple[T | None, bool]:
    """Execute a callable with a soft timeout using a daemon thread."""

    result_holder: List[T] = []
    error_holder: List[BaseException] = []

    def target() -> None:
        try:
            result_holder.append(func())
        except BaseException as exc:  # pragma: no cover - passthrough for caller handling
            error_holder.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        logging.error(timeout_msg, timeout_s)
        return None, True
    if error_holder:
        raise error_holder[0]
    return result_holder[0] if result_holder else None, False


def _usage_metadata_value(usage_metadata: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(usage_metadata, dict) and key in usage_metadata:
            return usage_metadata[key]
        value = getattr(usage_metadata, key, None)
        if value is not None:
            return value
    return None


def _build_code_execution_tools() -> list[object]:
    tool_cls = getattr(genai_types, "Tool", None)
    code_execution_cls = getattr(genai_types, "ToolCodeExecution", None)
    if tool_cls is None or code_execution_cls is None:
        raise RuntimeError(
            "The installed google-genai package does not expose Tool/ToolCodeExecution; "
            "upgrade google-genai to use Gemini agentic vision."
        )
    try:
        code_execution = code_execution_cls()
    except TypeError:
        code_execution = code_execution_cls
    return [tool_cls(code_execution=code_execution)]


class GeminiSegmenter:
    """Thin wrapper around the google-genai client used in the notebooks."""

    _prompt_cache_lock = threading.Lock()
    _prompt_cache_registry: Dict[str, str] = {}
    _prompt_cache_disabled: set[str] = set()

    def __init__(
        self,
        *,
        model_name: str,
        prompt: str,
        temperature: float = 0.5,
        thinking_budget: int = 0,
        timeout_s: float = 60.0,
        safety_settings: Optional[dict] = None,
        explicit_cache: bool = True,
        gemini_agentic_vision: bool = False,
        cache_ttl_s: int = 3600,
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.timeout_s = timeout_s
        self.safety_settings = safety_settings or {}
        self.explicit_cache = explicit_cache
        self.gemini_agentic_vision = gemini_agentic_vision
        self.cache_ttl_s = cache_ttl_s
        self.client = genai.Client()
        logging.info("GenAI backend: %s", "Vertex" if self.client.vertexai else "Dev API")
        self.cached_content_name: Optional[str] = self._get_or_create_prompt_cache()

    def _cache_key(self) -> str:
        payload = f"{self.model_name}|{self.prompt}|{self.cache_ttl_s}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_model_candidates(self) -> List[str]:
        if self.model_name.startswith("models/"):
            return [self.model_name]
        return [self.model_name, f"models/{self.model_name}"]

    def _get_or_create_prompt_cache(self) -> Optional[str]:
        if not self.explicit_cache:
            return None
        if not gemini_supports_explicit_cache(self.model_name):
            logging.info(
                "Explicit Gemini context caching is not supported for %s; continuing without it.",
                self.model_name,
            )
            return None

        cache_key = self._cache_key()
        with GeminiSegmenter._prompt_cache_lock:
            cached_name = GeminiSegmenter._prompt_cache_registry.get(cache_key)
            if cached_name:
                return cached_name
            if cache_key in GeminiSegmenter._prompt_cache_disabled:
                return None

        ttl = max(300, int(self.cache_ttl_s))
        config = {
            "display_name": f"seg-prompt-{cache_key[:8]}",
            "system_instruction": self.prompt,
            "ttl": f"{ttl}s",
        }
        last_error: BaseException | None = None
        for cache_model_name in self._cache_model_candidates():
            try:
                cached_content = self.client.caches.create(model=cache_model_name, config=config)
                cache_name = getattr(cached_content, "name", None)
                if cache_name:
                    with GeminiSegmenter._prompt_cache_lock:
                        GeminiSegmenter._prompt_cache_registry[cache_key] = cache_name
                    logging.info(
                        "Enabled explicit Gemini context cache for %s (ttl=%ss)", cache_model_name, ttl
                    )
                    return cache_name
            except Exception as exc:  # pragma: no cover - external API behavior
                last_error = exc
                continue

        if last_error:
            with GeminiSegmenter._prompt_cache_lock:
                GeminiSegmenter._prompt_cache_disabled.add(cache_key)
            logging.warning(
                "Unable to create explicit Gemini context cache for %s; proceeding without it: %s",
                self.model_name,
                last_error,
            )
        return None

    def _call_model(
        self, image_obj: Image.Image
    ) -> Tuple[list[SegmentationMask], float, bool, list[dict]]:
        original_width, original_height = image_obj.size

        img_for_api = image_obj.copy()
        max_dim = 1024
        if img_for_api.height > max_dim or img_for_api.width > max_dim:
            img_for_api.thumbnail((max_dim, max_dim))
            logging.info(
                "Resized image from %sx%s to %sx%s for API call.",
                original_width,
                original_height,
                img_for_api.width,
                img_for_api.height,
            )

        with io.BytesIO() as img_byte_arr:
            img_for_api.save(img_byte_arr, format="JPEG")
            img_bytes = img_byte_arr.getvalue()

        config_kwargs: Dict[str, Any] = {
            "thinking_config": ThinkingConfig(thinking_budget=self.thinking_budget),
            "temperature": self.temperature,
            "safety_settings": [SafetySetting(**s) for s in self.safety_settings.values()]
            if self.safety_settings
            else None,
        }
        if not self.gemini_agentic_vision and gemini_supports_structured_output(self.model_name):
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = GEMINI_SEGMENTATION_RESPONSE_JSON_SCHEMA
        if self.cached_content_name:
            config_kwargs["cached_content"] = self.cached_content_name
        if self.gemini_agentic_vision:
            config_kwargs["tools"] = _build_code_execution_tools()
        gen_config = GenerateContentConfig(**config_kwargs)

        image_part = Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        if self.cached_content_name:
            text_part = Part(text="Segment this image following the cached instructions. Return JSON.")
        else:
            text_part = Part(text=self.prompt)

        start_time = datetime.now()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image_part, text_part],
            config=gen_config,
        )
        latency = (datetime.now() - start_time).total_seconds()

        usage_metadata = getattr(response, "usage_metadata", None)
        cached_tokens = _usage_metadata_value(
            usage_metadata,
            "cached_content_token_count",
            "cachedContentTokenCount",
        )
        if cached_tokens:
            prompt_tokens = _usage_metadata_value(
                usage_metadata,
                "prompt_token_count",
                "promptTokenCount",
            )
            total_tokens = _usage_metadata_value(
                usage_metadata,
                "total_token_count",
                "totalTokenCount",
            )
            logging.info(
                "Gemini cache tokens used (model=%s): prompt=%s cached=%s total=%s",
                self.model_name,
                prompt_tokens,
                cached_tokens,
                total_tokens,
            )

        masks, parse_success, raw_items = parse_segmentation_masks(
            response, img_height=original_height, img_width=original_width
        )
        return masks, latency, parse_success, raw_items

    def segment(
        self, image_obj: Image.Image
    ) -> Tuple[list[SegmentationMask], float, bool, bool, list[dict]]:
        """
        Executes segmentation with an external timeout.

        Returns masks, latency, parse_success, timed_out.
        """

        result, timed_out = _run_with_timeout(
            lambda: self._call_model(image_obj), self.timeout_s, "Segmentation call exceeded timeout of %.1fs"
        )
        if timed_out or result is None:
            return [], 0.0, False, True, []
        masks, latency, parse_success, raw_items = result
        return masks, latency, parse_success, False, raw_items


def _require_cairosvg():
    try:
        import cairosvg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "cairosvg is required for rasterizing Moondream SVG masks. Install it via the environment.yml or pip."
        ) from exc
    return cairosvg


def _normalize_moondream_bbox(raw_bbox: Any) -> Dict[str, float] | None:
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        raw_bbox = {
            "x_min": raw_bbox[0],
            "y_min": raw_bbox[1],
            "x_max": raw_bbox[2],
            "y_max": raw_bbox[3],
        }

    if not isinstance(raw_bbox, dict):
        return None

    aliases = {
        "x_min": ("x_min", "xmin", "x0", "left"),
        "y_min": ("y_min", "ymin", "y0", "top"),
        "x_max": ("x_max", "xmax", "x1", "right"),
        "y_max": ("y_max", "ymax", "y1", "bottom"),
    }
    normalized: Dict[str, float] = {}
    for key, options in aliases.items():
        value = None
        for option in options:
            if option in raw_bbox:
                value = raw_bbox[option]
                break
        if value is None:
            return None
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            return None
    return normalized


def _extract_moondream_path_bbox(result: Any) -> Tuple[str | None, Dict[str, float] | None]:
    if isinstance(result, dict):
        raw_path = result.get("path") or result.get("svg_path")
        raw_bbox = result.get("bbox") or result.get("box") or result.get("bounding_box")
    else:
        raw_path = getattr(result, "path", None) or getattr(result, "svg_path", None)
        raw_bbox = (
            getattr(result, "bbox", None)
            or getattr(result, "box", None)
            or getattr(result, "bounding_box", None)
        )

    path_d = raw_path.strip() if isinstance(raw_path, str) else None
    bbox = _normalize_moondream_bbox(raw_bbox)
    return path_d, bbox


def _bbox_to_pixel_floats(bbox: Dict[str, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x0_raw = float(bbox.get("x_min", 0.0))
    y0_raw = float(bbox.get("y_min", 0.0))
    x1_raw = float(bbox.get("x_max", 0.0))
    y1_raw = float(bbox.get("y_max", 0.0))

    is_normalized = all(0.0 <= val <= 1.0 for val in (x0_raw, y0_raw, x1_raw, y1_raw))
    if is_normalized:
        x0 = x0_raw * width
        y0 = y0_raw * height
        x1 = x1_raw * width
        y1 = y1_raw * height
    else:
        x0 = x0_raw
        y0 = y0_raw
        x1 = x1_raw
        y1 = y1_raw

    x0 = max(0.0, min(float(width), x0))
    x1 = max(0.0, min(float(width), x1))
    y0 = max(0.0, min(float(height), y0))
    y1 = max(0.0, min(float(height), y1))
    return x0, y0, x1, y1


def _bbox_to_pixel_ints(bbox: Dict[str, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = _bbox_to_pixel_floats(bbox, width, height)
    x0_i, y0_i, x1_i, y1_i = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    x0_i = max(0, min(width, x0_i))
    x1_i = max(0, min(width, x1_i))
    y0_i = max(0, min(height, y0_i))
    y1_i = max(0, min(height, y1_i))
    return x0_i, y0_i, x1_i, y1_i


def _pixel_box_to_gemini_box(box: Tuple[int, int, int, int], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = box
    y0n = int(round((y0 / height) * 1000)) if height else 0
    x0n = int(round((x0 / width) * 1000)) if width else 0
    y1n = int(round((y1 / height) * 1000)) if height else 0
    x1n = int(round((x1 / width) * 1000)) if width else 0
    normalized = [y0n, x0n, y1n, x1n]
    return [max(0, min(1000, val)) for val in normalized]


def _rasterize_svg_path(
    path_d: str, bbox: Dict[str, float], width: int, height: int
) -> np.ndarray:
    """Render a Moondream SVG path into a grayscale mask sized to the provided canvas."""

    cairosvg = _require_cairosvg()
    x0, y0, x1, y1 = _bbox_to_pixel_floats(bbox, width, height)
    bw = max(1e-6, x1 - x0)
    bh = max(1e-6, y1 - y0)
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="black"/>
  <g transform="translate({x0},{y0}) scale({bw},{bh})">
    <path d="{path_d}" fill="white" stroke="none"/>
  </g>
</svg>
"""
    png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width, output_height=height)
    mask = Image.open(io.BytesIO(png_bytes)).convert("L")
    return np.array(mask, dtype=np.uint8)


def _invoke_moondream_segment(client: Any, image_obj: Image.Image, label: str, model_name: str) -> Any:
    # Prefer documented signatures first. Only attempt model override variants as fallback.
    attempts = [
        lambda: client.segment(image_obj, label),
        lambda: client.segment(image=image_obj, object=label),
        lambda: client.segment(image=image_obj, target=label),
    ]
    if model_name:
        attempts.extend(
            [
                lambda: client.segment(image_obj, label, model=model_name),
                lambda: client.segment(image=image_obj, object=label, model=model_name),
                lambda: client.segment(image=image_obj, target=label, model=model_name),
            ]
        )
    last_type_error: TypeError | None = None
    for invoke in attempts:
        try:
            return invoke()
        except TypeError as exc:
            last_type_error = exc
            continue
    if last_type_error:
        raise last_type_error
    raise RuntimeError("No Moondream client segment invocation strategy succeeded")


class MoondreamSegmenter:
    """Adapter for Moondream 3 segmentation returning Gemini-compatible masks."""

    def __init__(
        self,
        *,
        model_name: str,
        prompt: str,
        timeout_s: float = 60.0,
        targets: Optional[Sequence[str]] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        max_dim: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt
        self.timeout_s = timeout_s
        self.targets = list(targets) if targets else [prompt]
        self.endpoint = endpoint
        self.api_key = api_key
        self.max_dim = max_dim
        self.client = self._init_client()

    def _init_client(self) -> Any:
        try:
            import moondream as md
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "moondream is required for the Moondream provider. Install it via the environment.yml or pip."
            ) from exc

        kwargs: Dict[str, Any] = {}
        if self.endpoint:
            kwargs["endpoint"] = self.endpoint
        if not kwargs:
            api_key = self.api_key or os.environ.get("MOONDREAM_API_KEY")
            if not api_key:
                raise ValueError("MOONDREAM_API_KEY or --moondream-api-key is required for Moondream calls")
            kwargs["api_key"] = api_key
        return md.vl(**kwargs)

    def _prepare_image(self, image_obj: Image.Image) -> Tuple[Image.Image, Tuple[float, float]]:
        """Resize oversized images while tracking scale factors back to the original."""

        original_width, original_height = image_obj.size
        img_for_api = image_obj.copy()
        if img_for_api.height > self.max_dim or img_for_api.width > self.max_dim:
            img_for_api.thumbnail((self.max_dim, self.max_dim))
            logging.info(
                "Resized image from %sx%s to %sx%s for Moondream API call.",
                original_width,
                original_height,
                img_for_api.width,
                img_for_api.height,
            )
        scale_x = original_width / img_for_api.width
        scale_y = original_height / img_for_api.height
        return img_for_api, (scale_x, scale_y)

    def _segment_single(
        self,
        image_obj: Image.Image,
        label: str,
        scale: Tuple[float, float],
        original_size: Tuple[int, int],
    ) -> Tuple[SegmentationMask | None, Dict[str, Any] | None]:
        api_width, api_height = image_obj.size
        scale_x, scale_y = scale
        try:
            result = _invoke_moondream_segment(self.client, image_obj, label, self.model_name)
        except Exception:  # pragma: no cover - network/API failures
            logging.exception("Moondream segmentation call failed for label '%s'", label)
            return None, None

        path_d, bbox = _extract_moondream_path_bbox(result)
        if not path_d or bbox is None:
            logging.warning("Moondream response missing path or bbox for label '%s'", label)
            return None, None

        x0_api, y0_api, x1_api, y1_api = _bbox_to_pixel_ints(bbox, api_width, api_height)
        if x1_api <= x0_api or y1_api <= y0_api:
            logging.warning("Ignoring zero-area Moondream bbox for label '%s': %s", label, bbox)
            return None, None

        mask_api = _rasterize_svg_path(path_d, bbox, api_width, api_height)
        if scale_x != 1.0 or scale_y != 1.0:
            mask_pil = Image.fromarray(mask_api)
            mask_resized = mask_pil.resize(original_size, resample=Image.Resampling.BILINEAR)
            mask_full = np.array(mask_resized, dtype=np.uint8)
            x0 = int(round(x0_api * scale_x))
            y0 = int(round(y0_api * scale_y))
            x1 = int(round(x1_api * scale_x))
            y1 = int(round(y1_api * scale_y))
        else:
            mask_full = mask_api
            x0, y0, x1, y1 = x0_api, y0_api, x1_api, y1_api

        width, height = original_size
        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            logging.warning("Clamped Moondream bbox is invalid for label '%s'", label)
            return None, None

        crop = mask_full[y0:y1, x0:x1]
        raw_item = {
            "label": label,
            "box_2d": _pixel_box_to_gemini_box((x0, y0, x1, y1), width, height),
            "mask": encode_mask_to_b64(crop if crop.size else np.zeros((1, 1), dtype=np.uint8)),
        }
        seg_mask = SegmentationMask(y0=y0, x0=x0, y1=y1, x1=x1, mask=mask_full, label=label)
        return seg_mask, raw_item

    def _call_model(
        self, image_obj: Image.Image
    ) -> Tuple[list[SegmentationMask], float, bool, list[dict]]:
        original_size = image_obj.size
        api_image, scale = self._prepare_image(image_obj)
        start_time = datetime.now()
        masks: List[SegmentationMask] = []
        raw_items: List[Dict[str, Any]] = []
        parse_success = True
        for label in self.targets:
            mask, raw_item = self._segment_single(api_image, label, scale, original_size)
            if mask is None or raw_item is None:
                parse_success = False
                continue
            masks.append(mask)
            raw_items.append(raw_item)
        latency = (datetime.now() - start_time).total_seconds()
        return masks, latency, parse_success and bool(masks), raw_items

    def segment(
        self, image_obj: Image.Image
    ) -> Tuple[list[SegmentationMask], float, bool, bool, list[dict]]:
        result, timed_out = _run_with_timeout(
            lambda: self._call_model(image_obj),
            self.timeout_s,
            "Moondream segmentation call exceeded timeout of %.1fs",
        )
        if timed_out or result is None:
            return [], 0.0, False, True, []
        masks, latency, parse_success, raw_items = result
        return masks, latency, parse_success, False, raw_items


class ReplicateSegmenter:
    """Replicate-backed segmenter that mirrors the Gemini output schema."""

    def __init__(
        self,
        *,
        model_version: str,
        instruction: str,
        timeout_s: float = 60.0,
        targets: Optional[Sequence[str]] = None,
        instructions: Optional[Dict[str, str]] = None,
        cache_dir: Optional[Path] = None,
        max_dim: int = 1024,
    ) -> None:
        if not model_version:
            raise ValueError("A Replicate model version is required")
        self.model_version = model_version
        self.instruction = instruction
        self.timeout_s = timeout_s
        self.targets = list(targets) if targets else [""]
        self.instructions = dict(instructions) if instructions else {}
        self.cache_dir = cache_dir
        self.max_dim = max_dim

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        token = os.environ.get("REPLICATE_API_TOKEN")
        if not token:
            raise ValueError("REPLICATE_API_TOKEN is required for Replicate calls")
        try:
            import replicate
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "replicate is required for the Replicate provider. Install it via the environment.yml or pip."
            ) from exc

        self.client = replicate.Client(api_token=token)

    def _prepare_image(self, image_obj: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
        original_width, original_height = image_obj.size
        img_for_api = image_obj.copy()
        if img_for_api.height > self.max_dim or img_for_api.width > self.max_dim:
            img_for_api.thumbnail((self.max_dim, self.max_dim))
            logging.info(
                "Resized image from %sx%s to %sx%s for Replicate API call.",
                original_width,
                original_height,
                img_for_api.width,
                img_for_api.height,
            )
        return img_for_api, (original_width, original_height)

    def _download_mask(self, url: str) -> Image.Image | None:
        cache_path: Path | None = None
        if self.cache_dir:
            suffix = Path(urlparse(url).path).suffix or ".png"
            cache_path = self.cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}{suffix}"
            if cache_path.exists():
                try:
                    return Image.open(cache_path).convert("L")
                except OSError:
                    logging.warning("Failed to read cached mask at %s; re-downloading", cache_path)

        try:
            request = Request(url, headers={"User-Agent": "gemini-segmentation/replicate"})
            with urlopen(request, timeout=30) as resp:
                content = resp.read()
        except (URLError, TimeoutError, OSError, ValueError):  # pragma: no cover - network failures
            logging.exception("Failed to download Replicate mask from %s", url)
            return None

        if cache_path:
            try:
                cache_path.write_bytes(content)
            except OSError:
                logging.warning("Failed to write Replicate mask to cache at %s", cache_path)

        try:
            return Image.open(io.BytesIO(content)).convert("L")
        except OSError:
            logging.exception("Downloaded Replicate mask is not a valid image: %s", url)
            return None

    @staticmethod
    def _extract_img_url(result: Any) -> str | None:
        return ReplicateSegmenter._extract_img_url_inner(result, seen=set())

    @staticmethod
    def _is_url_like(value: str) -> bool:
        candidate = value.strip()
        if not candidate:
            return False
        if candidate.startswith("data:"):
            return True
        parsed = urlparse(candidate)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _extract_img_url_inner(result: Any, *, seen: set[int]) -> str | None:
        if result is None:
            return None
        if isinstance(result, str):
            value = result.strip()
            return value if ReplicateSegmenter._is_url_like(value) else None
        if isinstance(result, (bytes, bytearray)):
            return None

        obj_id = id(result)
        if obj_id in seen:
            return None
        seen.add(obj_id)

        if isinstance(result, dict):
            for key in ("img", "image", "url", "output"):
                if key in result:
                    candidate = ReplicateSegmenter._extract_img_url_inner(result[key], seen=seen)
                    if candidate:
                        return candidate
            for value in result.values():
                candidate = ReplicateSegmenter._extract_img_url_inner(value, seen=seen)
                if candidate:
                    return candidate
            return None

        if isinstance(result, (list, tuple, set)):
            for item in result:
                candidate = ReplicateSegmenter._extract_img_url_inner(item, seen=seen)
                if candidate:
                    return candidate
            return None

        url_attr = getattr(result, "url", None)
        if isinstance(url_attr, str):
            candidate = ReplicateSegmenter._extract_img_url_inner(url_attr, seen=seen)
            if candidate:
                return candidate
        elif callable(url_attr):
            try:
                candidate = ReplicateSegmenter._extract_img_url_inner(url_attr(), seen=seen)
                if candidate:
                    return candidate
            except Exception:
                pass

        for attr_name in ("img", "image", "output"):
            if hasattr(result, attr_name):
                candidate = ReplicateSegmenter._extract_img_url_inner(
                    getattr(result, attr_name),
                    seen=seen,
                )
                if candidate:
                    return candidate

        return None

    def _invoke_replicate(self, img_bytes: bytes, instruction: str) -> Any:
        with io.BytesIO(img_bytes) as image_input:
            # Replicate's Python client expects a file-like object (or URL/data URI), not raw bytes.
            image_input.name = "input.jpg"  # type: ignore[attr-defined]
            try:
                return self.client.run(
                    self.model_version,
                    input={"image": image_input, "instruction": instruction},
                )
            except TypeError as exc:
                if "not JSON serializable" not in str(exc):
                    raise
                logging.warning(
                    "Replicate client rejected file-like image payload; retrying with data URI fallback."
                )

        data_uri = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('ascii')}"
        return self.client.run(
            self.model_version,
            input={"image": data_uri, "instruction": instruction},
        )

    def _segment_single(
        self,
        img_bytes: bytes,
        api_size: Tuple[int, int],
        original_size: Tuple[int, int],
        label: str,
        instruction: str,
    ) -> Tuple[SegmentationMask | None, Dict[str, Any] | None, bool]:
        try:
            result = self._invoke_replicate(img_bytes, instruction)
            logging.debug("Replicate raw output for '%s': %s", label or instruction, result)
        except Exception:  # pragma: no cover - network/API failures
            logging.exception("Replicate segmentation call failed for label '%s'", label)
            return None, None, False

        img_url = self._extract_img_url(result)
        if not img_url:
            logging.warning("Replicate response missing mask URL for label '%s'", label)
            return None, None, False

        mask_img = self._download_mask(img_url)
        if mask_img is None:
            return None, None, False

        api_width, api_height = api_size
        if mask_img.size != api_size:
            logging.info(
                "Resizing Replicate mask from %sx%s to API dimensions %sx%s", *mask_img.size, api_width, api_height
            )
            mask_img = mask_img.resize(api_size, resample=Image.Resampling.NEAREST)

        mask_binary = (np.array(mask_img, dtype=np.uint8) > 127).astype(np.uint8) * 255
        if mask_binary.size == 0 or not mask_binary.any():
            logging.warning("Replicate mask is empty for label '%s'", label)
            return None, None, False

        if original_size != api_size:
            mask_pil = Image.fromarray(mask_binary)
            mask_full = np.array(mask_pil.resize(original_size, resample=Image.Resampling.NEAREST), dtype=np.uint8)
        else:
            mask_full = mask_binary

        coords = np.argwhere(mask_full > 0)
        if coords.size == 0:
            logging.warning("Replicate mask had no positive pixels after resizing for label '%s'", label)
            return None, None, False

        y0, x0 = coords.min(axis=0)[:2]
        y1, x1 = coords.max(axis=0)[:2]
        y1 += 1
        x1 += 1

        width, height = original_size
        box = _pixel_box_to_gemini_box((x0, y0, x1, y1), width, height)
        crop = mask_full[y0:y1, x0:x1]
        raw_item = {"label": label, "box_2d": box, "mask": encode_mask_to_b64(crop)}
        seg_mask = SegmentationMask(y0=y0, x0=x0, y1=y1, x1=x1, mask=mask_full, label=label)
        return seg_mask, raw_item, True

    def _call_model(
        self, image_obj: Image.Image
    ) -> Tuple[list[SegmentationMask], float, bool, list[dict]]:
        api_image, original_size = self._prepare_image(image_obj)
        with io.BytesIO() as img_byte_arr:
            api_image.save(img_byte_arr, format="JPEG")
            img_bytes = img_byte_arr.getvalue()

        start_time = datetime.now()
        masks: List[SegmentationMask] = []
        raw_items: List[Dict[str, Any]] = []
        parse_success = True
        api_size = api_image.size
        for target_label in self.targets:
            instr = self.instructions.get(target_label, self.instruction or target_label)
            seg_mask, raw_item, success = self._segment_single(
                img_bytes, api_size, original_size, target_label, instr
            )
            parse_success = parse_success and success
            if seg_mask is None or raw_item is None:
                continue
            masks.append(seg_mask)
            raw_items.append(raw_item)

        latency = (datetime.now() - start_time).total_seconds()
        if not masks:
            parse_success = False
        return masks, latency, parse_success, raw_items

    def segment(
        self, image_obj: Image.Image
    ) -> Tuple[list[SegmentationMask], float, bool, bool, list[dict]]:
        result, timed_out = _run_with_timeout(
            lambda: self._call_model(image_obj),
            self.timeout_s,
            "Replicate segmentation call exceeded timeout of %.1fs",
        )
        if timed_out or result is None:
            return [], 0.0, False, True, []
        masks, latency, parse_success, raw_items = result
        return masks, latency, parse_success, False, raw_items


class Sa2VAReplicateSegmenter(ReplicateSegmenter):
    """Compatibility wrapper for Sa2VA Replicate deployments."""

    def __init__(
        self,
        *,
        model_name: str,
        model_version: str,
        instruction: str,
        timeout_s: float = 60.0,
        targets: Optional[Sequence[str]] = None,
        instructions: Optional[Dict[str, str]] = None,
        cache_dir: Optional[Path] = None,
        max_dim: int = 1024,
    ) -> None:
        self.model_name = model_name
        super().__init__(
            model_version=model_version,
            instruction=instruction,
            timeout_s=timeout_s,
            targets=targets,
            instructions=instructions,
            cache_dir=cache_dir,
            max_dim=max_dim,
        )
