from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from nanobanana_segmentation.core.engine import NanoBananaClient, SegmentationEngine
from nanobanana_segmentation.core.logging.artifact_store import ArtifactStore
from nanobanana_segmentation.core.types import EngineRequest, ModelCallResult


class _FakeClient:
    def __init__(self, fail_first: bool = False, surrogate_shape: tuple[int, int] = (64, 64)):
        self.model_id = "gemini-3.1-flash-image-preview"
        self.api_surface = "generate_content"
        self._fail_first = fail_first
        self._calls = 0
        self._surrogate_shape = surrogate_shape

    def generate_mask_surrogate(self, **kwargs):
        self._calls += 1
        if self._fail_first and self._calls == 1:
            raise RuntimeError("429 retry")

        h, w = self._surrogate_shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 1] = 255
        y0, y1 = max(1, h // 4), max(2, (3 * h) // 4)
        x0, x1 = max(1, w // 4), max(2, (3 * w) // 4)
        img[y0:y1, x0:x1] = [255, 255, 255]
        ok, enc = cv2.imencode(".png", img)
        assert ok

        return ModelCallResult(
            surrogate_png=bytes(enc),
            raw_request={"prompt": kwargs.get("prompt", "")},
            raw_response={"candidates": []},
            thought_signature_present=False,
            thought_signatures=[],
            thought_summaries=[],
            grounding={"grounding_chunks": []},
        )


def test_engine_segment_once(tmp_path: Path) -> None:
    engine = SegmentationEngine(
        client=_FakeClient(),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        cost_per_attempt_usd=0.01,
    )

    input_img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", input_img)
    assert ok
    request = EngineRequest(
        image_bytes=bytes(enc),
        image_name="img.png",
        image_id="img",
        target="lesion",
        max_retries_semantic=1,
        max_retries_transport=0,
        run_id="testrun",
    )

    result = engine.segment_once(request)
    assert result.mask.shape == (64, 64)
    assert result.selected_attempt_index == 1
    assert result.run_record_path
    record = json.loads(Path(result.run_record_path).read_text(encoding="utf-8"))
    assert record["tool_config"]["query_policy"] == "model_generated"
    assert record["tool_config"]["snapshot_policy"] == "live_with_caching"
    assert record["tool_config"]["scope_policy"] == "open_web"


def test_engine_transport_retry(tmp_path: Path) -> None:
    engine = SegmentationEngine(
        client=_FakeClient(fail_first=True),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        cost_per_attempt_usd=0.01,
    )
    input_img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", input_img)
    assert ok

    request = EngineRequest(
        image_bytes=bytes(enc),
        image_name="img.png",
        image_id="img",
        target="lesion",
        max_retries_semantic=1,
        max_retries_transport=1,
        run_id="testrun2",
    )

    result = engine.segment_once(request)
    assert result.attempts[0].transport_retries == 1


def test_engine_resizes_mismatched_surrogate_for_output(tmp_path: Path) -> None:
    engine = SegmentationEngine(
        client=_FakeClient(surrogate_shape=(96, 112)),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        cost_per_attempt_usd=0.01,
    )
    input_img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", input_img)
    assert ok

    request = EngineRequest(
        image_bytes=bytes(enc),
        image_name="img.png",
        image_id="img",
        target="lesion",
        max_retries_semantic=1,
        max_retries_transport=0,
        run_id="testrun3",
    )

    result = engine.segment_once(request)
    assert result.mask.shape == (64, 64)
    assert any("resized_mask_to_input_shape" in a.warnings for a in result.attempts)


def test_tool_payload_mapping_with_supported_fields() -> None:
    tools, effective_mode, warnings = NanoBananaClient._resolve_tools_payload(
        tool_mode="text_image",
        tool_fields=("google_search",),
        google_search_fields=("search_types",),
        search_type_fields=("web_search", "image_search"),
    )
    assert effective_mode == "text_image"
    assert tools == [{"google_search": {"search_types": [{"web_search": {}}, {"image_search": {}}]}}]
    assert warnings == []


def test_tool_payload_falls_back_when_image_tool_missing() -> None:
    tools, effective_mode, warnings = NanoBananaClient._resolve_tools_payload(
        tool_mode="image",
        tool_fields=("google_search", "google_search_retrieval"),
        google_search_fields=(),
        search_type_fields=(),
    )
    assert effective_mode == "text"
    assert tools == [{"google_search": {}}]
    assert "tool_image_search_unavailable_in_sdk" in warnings
    assert "tool_mode_fallback:image->text" in warnings


def test_extract_surrogate_from_inline_bytes() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :] = [0, 128, 255]
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    payload = {
        "parts": [
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": bytes(enc),
                }
            }
        ]
    }
    extracted = NanoBananaClient._extract_surrogate_png(payload)
    assert extracted is not None
    decoded = cv2.imdecode(np.frombuffer(extracted, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (16, 16)


def test_extract_surrogate_from_inline_base64() -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :] = [255, 255, 255]
    ok, enc = cv2.imencode(".png", img)
    assert ok
    payload = {
        "parts": [
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(bytes(enc)).decode("utf-8"),
                }
            }
        ]
    }
    extracted = NanoBananaClient._extract_surrogate_png(payload)
    assert extracted is not None
    decoded = cv2.imdecode(np.frombuffer(extracted, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[:2] == (16, 16)
