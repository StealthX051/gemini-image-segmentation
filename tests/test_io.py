from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gemini_segmentation.io import encode_mask_to_b64, parse_segmentation_masks


def test_parse_segmentation_masks_prefers_final_json_text_part() -> None:
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="I'll zoom into the target before returning the result."),
                        SimpleNamespace(code_execution_result=SimpleNamespace(output="intermediate crop")),
                        SimpleNamespace(text="```json\n[]\n```"),
                    ]
                )
            )
        ]
    )

    masks, parse_success, raw_items = parse_segmentation_masks(
        response,
        img_height=8,
        img_width=8,
    )

    assert parse_success is True
    assert masks == []
    assert raw_items == []


def test_parse_segmentation_masks_handles_plain_json_text() -> None:
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="[]")]
                )
            )
        ]
    )

    masks, parse_success, raw_items = parse_segmentation_masks(
        response,
        img_height=8,
        img_width=8,
    )

    assert parse_success is True
    assert masks == []
    assert raw_items == []


def test_parse_segmentation_masks_handles_single_base64_mask_fallback() -> None:
    raw_mask = np.full((4, 4), 255, dtype=np.uint8)
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=encode_mask_to_b64(raw_mask))])
            )
        ]
    )

    masks, parse_success, raw_items = parse_segmentation_masks(
        response,
        img_height=8,
        img_width=8,
    )

    assert parse_success is True
    assert len(masks) == 1
    assert len(raw_items) == 1
    assert raw_items[0]["mask"].startswith("data:image/png;base64,")
