from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nanobanana_segmentation.core.types import AttemptResult, EngineResult, QCMetrics
from nanobanana_segmentation.service import main as service_main


class _FakeEngine:
    class _Client:
        model_id = "gemini-3.1-flash-image-preview"

    client = _Client()

    def segment_once(self, request):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        attempt = AttemptResult(
            attempt_index=1,
            prompt_id="chromakey_v1",
            prompt_text="prompt",
            attempt_mode="chromakey",
            extraction_method="chromakey_hsv",
            qc_metrics=QCMetrics(
                resolution_match=True,
                mask_nonempty=True,
                mask_area_frac=0.25,
                component_count=1,
                largest_component_frac=1.0,
                speckle_score=0.0,
                border_touch=False,
                green_coverage=0.75,
                green_uniformity_proxy=1.0,
            ),
            qc_pass=True,
            failure_reasons=[],
            score=1000.0,
            transport_retries=0,
            transport_retry_events=[],
            thought_signature_present=False,
            thought_signatures=[],
            thought_summaries=[],
            grounding={"grounding_chunks": []},
            raw_request_path=None,
            raw_response_path=None,
            surrogate_image_path=None,
            intermediate_mask_paths=[],
            overlay_path=None,
        )
        return EngineResult(
            run_id="testrun",
            image_id="img",
            image_name="img.png",
            selected_attempt_index=1,
            qc_pass=True,
            warnings=[],
            attempts=[attempt],
            mask=mask,
            surrogate=None,
            mask_hash="abc",
            surrogate_hash=None,
            run_record_path="/tmp/run_record.json",
        )


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch):
    monkeypatch.setattr(service_main, "get_engine", lambda: _FakeEngine())


def test_segment_json_request() -> None:
    client = TestClient(service_main.app)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    import cv2

    ok, enc = cv2.imencode(".png", image)
    assert ok
    payload = {
        "image_base64": base64.b64encode(bytes(enc)).decode("utf-8"),
        "image_name": "img.png",
        "target": "lesion",
        "tool_mode": "closed",
        "output": "coco",
    }

    resp = client.post("/v1/segment", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "mask_png_base64" in body
    assert "meta" in body
    assert "mask_coco" in body


def test_health_and_metrics() -> None:
    client = TestClient(service_main.app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    m = client.get("/metrics")
    assert m.status_code == 200
