from __future__ import annotations

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
from google.genai.types import GenerateContentConfig, Part, SafetySetting, ThinkingConfig
import numpy as np
from PIL import Image

from .io import encode_mask_to_b64, parse_segmentation_masks
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


class GeminiSegmenter:
    """Thin wrapper around the google-genai client used in the notebooks."""

    def __init__(
        self,
        *,
        model_name: str,
        prompt: str,
        temperature: float = 0.5,
        thinking_budget: int = 0,
        timeout_s: float = 60.0,
        safety_settings: Optional[dict] = None,
    ) -> None:
        self.model_name = model_name
        self.prompt = prompt
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.timeout_s = timeout_s
        self.safety_settings = safety_settings or {}
        self.client = genai.Client()
        logging.info("GenAI backend: %s", "Vertex" if self.client.vertexai else "Dev API")

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

        gen_config = GenerateContentConfig(
            thinking_config=ThinkingConfig(thinking_budget=self.thinking_budget),
            temperature=self.temperature,
            safety_settings=[SafetySetting(**s) for s in self.safety_settings.values()] if self.safety_settings else None,
        )

        image_part = Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
        text_part = Part(text=self.prompt)

        start_time = datetime.now()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[image_part, text_part],
            config=gen_config,
        )
        latency = (datetime.now() - start_time).total_seconds()

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


def _bbox_to_pixel_floats(bbox: Dict[str, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x0 = float(bbox.get("x_min", 0.0)) * width
    y0 = float(bbox.get("y_min", 0.0)) * height
    x1 = float(bbox.get("x_max", 0.0)) * width
    y1 = float(bbox.get("y_max", 0.0)) * height
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
            call_kwargs: Dict[str, Any] = {"model": self.model_name} if self.model_name else {}
            try:
                result = self.client.segment(image_obj, label, **call_kwargs)
            except TypeError:
                # Older client versions may not accept model as a kwarg
                result = self.client.segment(image_obj, label)
        except Exception:  # pragma: no cover - network/API failures
            logging.exception("Moondream segmentation call failed for label '%s'", label)
            return None, None

        path_d = result.get("path") if isinstance(result, dict) else None
        bbox = result.get("bbox") if isinstance(result, dict) else None
        if not path_d or not bbox:
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
        except URLError:  # pragma: no cover - network failures
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
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get("img") or result.get("image")
        if isinstance(result, (list, tuple)):
            for item in result:
                candidate = ReplicateSegmenter._extract_img_url(item)
                if candidate:
                    return candidate
        return None

    def _segment_single(
        self,
        img_bytes: bytes,
        api_size: Tuple[int, int],
        original_size: Tuple[int, int],
        label: str,
        instruction: str,
    ) -> Tuple[SegmentationMask | None, Dict[str, Any] | None, bool]:
        try:
            result = self.client.run(self.model_version, input={"image": img_bytes, "instruction": instruction})
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
