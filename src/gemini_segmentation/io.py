from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, UnidentifiedImageError

from .types import PredictionRecord, SegmentationMask


def parse_json(predicted_str: str) -> str:
    """Extract a JSON blob even when wrapped in Markdown fences."""

    lines = predicted_str.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_content = "\n".join(lines[i + 1 :])
            closing_fence_index = json_content.find("```")
            if closing_fence_index != -1:
                json_content = json_content[:closing_fence_index]
            return json_content.strip()
    return predicted_str.strip()


def _extract_response_texts(response: object) -> List[str]:
    """Pull text parts from structured Gemini responses in order."""

    texts: List[str] = []
    candidates = getattr(response, "candidates", None)
    if candidates:
        for cand in candidates:
            parts = getattr(getattr(cand, "content", cand), "parts", []) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
    if texts:
        return texts

    fallback_text = getattr(response, "text", "")
    if fallback_text:
        return [fallback_text]
    return []


def _wrap_base64_mask(mask_str: str) -> List[dict]:
    """Allow single base64 mask responses by wrapping them into schema."""

    return [
        {
            "label": "",
            "box_2d": [0, 0, 1000, 1000],
            "mask": mask_str if mask_str.startswith("data:image") else f"data:image/png;base64,{mask_str}",
        }
    ]


def _looks_like_base64_mask_payload(candidate: str) -> bool:
    stripped = candidate.strip()
    if not stripped:
        return False
    if stripped.startswith("data:image"):
        return True
    if len(stripped) < 32 or any(ch.isspace() for ch in stripped):
        return False
    base64_chars = set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )
    return all(ch in base64_chars for ch in stripped)


def segmentation_masks_from_items(
    raw_items: Iterable[dict], *, img_height: int, img_width: int
) -> List[SegmentationMask]:
    """Convert schema-normalized mask items into full-image segmentation masks."""

    masks: List[SegmentationMask] = []
    for item in raw_items:
        try:
            abs_y0 = int(item["box_2d"][0] / 1000 * img_height)
            abs_x0 = int(item["box_2d"][1] / 1000 * img_width)
            abs_y1 = int(item["box_2d"][2] / 1000 * img_height)
            abs_x1 = int(item["box_2d"][3] / 1000 * img_width)
            if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
                logging.warning("Skipping invalid bounding box: %s", item.get("box_2d"))
                continue

            label = item.get("label", "")
            png_b64_str = item.get("mask", "")
            if not png_b64_str.startswith("data:image"):
                logging.warning("Skipping mask with unexpected format for label '%s'", label)
                continue

            png_bytes = base64.b64decode(png_b64_str.split(",", 1)[-1])
            partial_mask_img = Image.open(io.BytesIO(png_bytes))
            bbox_height = abs_y1 - abs_y0
            bbox_width = abs_x1 - abs_x0
            if bbox_height < 1 or bbox_width < 1:
                continue

            resized_mask = partial_mask_img.resize(
                (bbox_width, bbox_height), resample=Image.Resampling.BILINEAR
            )
            full_mask_np = np.zeros((img_height, img_width), dtype=np.uint8)
            full_mask_np[abs_y0:abs_y1, abs_x0:abs_x1] = np.array(resized_mask)
            masks.append(SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, full_mask_np, label))
        except (KeyError, IndexError, TypeError, base64.binascii.Error, UnidentifiedImageError):
            logging.exception("Skipping malformed mask entry: %s", item)
            continue
    return masks


def parse_segmentation_masks(
    response: object, *, img_height: int, img_width: int
) -> Tuple[List[SegmentationMask], bool, List[dict]]:
    """Parse model output into full-image masks with bounding boxes and raw entries."""

    text_candidates = _extract_response_texts(response)
    if not text_candidates:
        logging.warning("Empty segmentation response")
        return [], False, []

    raw_items: List[dict] | str = []
    parse_success = False
    last_parse_error: json.JSONDecodeError | None = None

    for raw_text in reversed(text_candidates):
        cleaned_json = parse_json(raw_text)
        if not cleaned_json:
            continue
        try:
            raw_items = json.loads(cleaned_json)
            parse_success = True
            break
        except json.JSONDecodeError as exc:
            last_parse_error = exc
            if _looks_like_base64_mask_payload(cleaned_json):
                raw_items = _wrap_base64_mask(cleaned_json)
                parse_success = True
                break

    if not parse_success and last_parse_error:
        logging.warning(
            "Failed to decode JSON response; attempting fallback handling: %s",
            last_parse_error,
        )

    # Normalize single-object dicts
    if isinstance(raw_items, dict):
        raw_items = [raw_items]

    normalized_items: List[dict] = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                normalized_items.append(item)
    masks = segmentation_masks_from_items(
        normalized_items, img_height=img_height, img_width=img_width
    )
    return masks, parse_success, normalized_items


def overlay_mask_on_img(img: Image.Image, mask: np.ndarray, color: str, alpha: float = 0.5) -> Image.Image:
    color_rgb = ImageColor.getrgb(color)
    img_rgba = img.convert("RGBA")
    overlay_color_rgba = color_rgb + (int(alpha * 255),)

    colored_mask_layer_np = np.zeros((img.height, img.width, 4), dtype=np.uint8)
    mask_binary = mask > 127
    colored_mask_layer_np[mask_binary] = overlay_color_rgba
    colored_mask_layer_pil = Image.fromarray(colored_mask_layer_np)
    return Image.alpha_composite(img_rgba, colored_mask_layer_pil)


def plot_segmentation_masks(img: Image.Image, segmentation_masks: List[SegmentationMask]) -> Image.Image:
    if not segmentation_masks:
        return img.convert("RGB")

    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except IOError:
        font = ImageFont.load_default()

    img_with_masks = img.copy()
    for i, seg_mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        img_with_masks = overlay_mask_on_img(img_with_masks, seg_mask.mask, color)

    draw = ImageDraw.Draw(img_with_masks)
    for i, seg_mask in enumerate(segmentation_masks):
        color = colors[i % len(colors)]
        box = ((seg_mask.x0, seg_mask.y0), (seg_mask.x1, seg_mask.y1))
        draw.rectangle(box, outline=color, width=3)
        if seg_mask.label:
            text_position = (seg_mask.x0 + 5, seg_mask.y0 + 5)
            draw.text(text_position, seg_mask.label, fill=color, font=font)
    return img_with_masks.convert("RGB")


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(path)


def encode_mask_to_b64(mask: np.ndarray) -> str:
    with io.BytesIO() as buffer:
        Image.fromarray(mask).save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"


def _normalize_record(rec: Any) -> dict:
    if is_dataclass(rec):
        rec_dict = asdict(rec)
    elif isinstance(rec, dict):
        rec_dict = dict(rec)
    elif hasattr(rec, "__dict__"):
        rec_dict = dict(rec.__dict__)
    else:
        rec_dict = {"value": rec}

    for key, val in list(rec_dict.items()):
        if isinstance(val, Path):
            rec_dict[key] = str(val)
    return rec_dict


def write_prediction_jsonl(records: Iterable[dict], path: Path, *, mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_normalize_record(rec)) for rec in records]
    with path.open(mode) as fh:
        for line in lines:
            fh.write(line + "\n")


def load_existing_predictions(path: Path) -> Dict[str, PredictionRecord]:
    if not path.exists():
        return {}

    records: Dict[str, PredictionRecord] = {}
    with path.open("r") as fh:
        for line in fh:
            try:
                payload = json.loads(line)
                image_name = payload.get("image_name")
                if not image_name:
                    continue
                records[image_name] = PredictionRecord(
                    image_name=image_name,
                    provider=payload.get("provider"),
                    prompt_family=payload.get("prompt_family"),
                    latency_s=float(payload.get("latency_s", 0.0)),
                    parse_success=bool(payload.get("parse_success", False)),
                    timed_out=bool(payload.get("timed_out", False)),
                    num_masks=int(payload.get("num_masks", 0)),
                    prediction_path=Path(payload.get("prediction_path", "")) if payload.get("prediction_path") else None,
                    overlay_path=Path(payload.get("overlay_path", "")) if payload.get("overlay_path") else None,
                    metrics_path=Path(payload.get("metrics_path", "")) if payload.get("metrics_path") else None,
                    legacy_json_path=Path(payload.get("legacy_json_path", "")) if payload.get("legacy_json_path") else None,
                    raw_response_path=Path(payload.get("raw_response_path", "")) if payload.get("raw_response_path") else None,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return records
