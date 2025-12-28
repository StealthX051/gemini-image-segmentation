from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from google import genai
from google.genai.types import GenerateContentConfig, Part, SafetySetting, ThinkingConfig
from PIL import Image

from .io import parse_segmentation_masks
from .types import SegmentationMask


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

        api_width, api_height = img_for_api.size

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
            response, img_height=api_height, img_width=api_width
        )

        if (api_height, api_width) != (original_height, original_width):
            y_scale = original_height / api_height
            x_scale = original_width / api_width
            scaled_masks: list[SegmentationMask] = []

            for mask in masks:
                resized_mask_img = Image.fromarray(mask.mask).resize(
                    (original_width, original_height), resample=Image.Resampling.NEAREST
                )
                scaled_masks.append(
                    SegmentationMask(
                        int(mask.y0 * y_scale),
                        int(mask.x0 * x_scale),
                        min(original_height, int(mask.y1 * y_scale)),
                        min(original_width, int(mask.x1 * x_scale)),
                        np.array(resized_mask_img),
                        mask.label,
                    )
                )

            masks = scaled_masks
        return masks, latency, parse_success, raw_items

    def segment(
        self, image_obj: Image.Image
    ) -> Tuple[list[SegmentationMask], float, bool, bool, list[dict]]:
        """
        Executes segmentation with an external timeout.

        Returns masks, latency, parse_success, timed_out.
        """

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._call_model, image_obj)
            try:
                masks, latency, parse_success, raw_items = future.result(timeout=self.timeout_s)
                return masks, latency, parse_success, False, raw_items
            except TimeoutError:
                logging.error("Segmentation call exceeded timeout of %.1fs", self.timeout_s)
                return [], 0.0, False, True, []
