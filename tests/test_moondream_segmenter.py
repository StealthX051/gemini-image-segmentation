import os
import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


def _install_google_stubs() -> None:
    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_types_module = types.ModuleType("google.genai.types")

    class _Placeholder:
        def __init__(self, *args, **kwargs):
            pass

    genai_types_module.GenerateContentConfig = _Placeholder
    genai_types_module.Part = _Placeholder
    genai_types_module.SafetySetting = _Placeholder
    genai_types_module.ThinkingConfig = _Placeholder
    genai_module.types = genai_types_module
    genai_module.Client = _Placeholder
    google_module.genai = genai_module

    sys.modules.setdefault("google", google_module)
    sys.modules.setdefault("google.genai", genai_module)
    sys.modules.setdefault("google.genai.types", genai_types_module)


def _install_moondream_stub(client: object) -> mock.Mock:
    module = types.ModuleType("moondream")
    factory = mock.Mock(return_value=client)
    module.vl = factory
    sys.modules["moondream"] = module
    return factory


_install_google_stubs()

from gemini_segmentation.models import MoondreamSegmenter  # noqa: E402


def _small_image() -> Image.Image:
    return Image.new("RGB", (8, 8), color="black")


def test_init_client_uses_api_key_from_env() -> None:
    client = mock.Mock()
    factory = _install_moondream_stub(client)

    with mock.patch.dict(os.environ, {"MOONDREAM_API_KEY": "secret-key"}, clear=True):
        segmenter = MoondreamSegmenter(model_name="moondream-3", prompt="polyp")

    assert segmenter.client is client
    factory.assert_called_once_with(api_key="secret-key")


def test_init_client_uses_endpoint_without_api_key() -> None:
    client = mock.Mock()
    factory = _install_moondream_stub(client)

    with mock.patch.dict(os.environ, {}, clear=True):
        segmenter = MoondreamSegmenter(
            model_name="moondream-3",
            prompt="polyp",
            endpoint="http://localhost:2020",
        )

    assert segmenter.client is client
    factory.assert_called_once_with(endpoint="http://localhost:2020")


def test_init_client_missing_api_key_raises() -> None:
    client = mock.Mock()
    _install_moondream_stub(client)

    with mock.patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="MOONDREAM_API_KEY"):
            MoondreamSegmenter(model_name="moondream-3", prompt="polyp")


def test_segment_success_returns_gemini_compatible_payload() -> None:
    client = mock.Mock()
    client.segment.return_value = {
        "path": "M0 0 L1 0 L1 1 Z",
        "bbox": {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75},
    }
    _install_moondream_stub(client)

    with mock.patch.dict(os.environ, {"MOONDREAM_API_KEY": "secret-key"}, clear=True):
        segmenter = MoondreamSegmenter(
            model_name="moondream-3",
            prompt="polyp",
            targets=["polyp"],
            timeout_s=5.0,
        )

    mask_api = np.zeros((8, 8), dtype=np.uint8)
    mask_api[2:6, 2:6] = 255
    with mock.patch("gemini_segmentation.models._rasterize_svg_path", return_value=mask_api):
        masks, latency, parse_success, timed_out, raw_items = segmenter.segment(_small_image())

    assert latency >= 0.0
    assert not timed_out
    assert parse_success
    assert len(masks) == 1
    assert masks[0].label == "polyp"
    assert (masks[0].y0, masks[0].x0, masks[0].y1, masks[0].x1) == (2, 2, 6, 6)
    assert raw_items[0]["label"] == "polyp"
    assert raw_items[0]["box_2d"] == [250, 250, 750, 750]
    assert raw_items[0]["mask"].startswith("data:image/png;base64,")


def test_segment_supports_attribute_result_and_list_bbox() -> None:
    client = mock.Mock()

    class _Response:
        path = "M0 0 L1 0 L1 1 Z"
        bbox = [0.25, 0.25, 0.75, 0.75]

    client.segment.return_value = _Response()
    _install_moondream_stub(client)

    with mock.patch.dict(os.environ, {"MOONDREAM_API_KEY": "secret-key"}, clear=True):
        segmenter = MoondreamSegmenter(
            model_name="moondream-3",
            prompt="polyp",
            targets=["polyp"],
            timeout_s=5.0,
        )

    mask_api = np.zeros((8, 8), dtype=np.uint8)
    mask_api[2:6, 2:6] = 255
    with mock.patch("gemini_segmentation.models._rasterize_svg_path", return_value=mask_api):
        masks, _, parse_success, timed_out, raw_items = segmenter.segment(_small_image())

    assert not timed_out
    assert parse_success
    assert len(masks) == 1
    assert len(raw_items) == 1


def test_segment_falls_back_when_model_kwarg_is_unsupported() -> None:
    client = mock.Mock()
    client.segment.side_effect = [
        TypeError("unexpected keyword argument 'model'"),
        {
            "path": "M0 0 L1 0 L1 1 Z",
            "bbox": {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75},
        },
    ]
    _install_moondream_stub(client)

    with mock.patch.dict(os.environ, {"MOONDREAM_API_KEY": "secret-key"}, clear=True):
        segmenter = MoondreamSegmenter(
            model_name="moondream-3",
            prompt="polyp",
            targets=["polyp"],
            timeout_s=5.0,
        )

    mask_api = np.zeros((8, 8), dtype=np.uint8)
    mask_api[2:6, 2:6] = 255
    with mock.patch("gemini_segmentation.models._rasterize_svg_path", return_value=mask_api):
        masks, _, parse_success, timed_out, _ = segmenter.segment(_small_image())

    assert client.segment.call_count >= 2
    assert not timed_out
    assert parse_success
    assert len(masks) == 1


def test_segment_marks_parse_failure_for_missing_path_or_bbox() -> None:
    client = mock.Mock()
    client.segment.return_value = {"bbox": {"x_min": 0.2, "y_min": 0.2, "x_max": 0.8, "y_max": 0.8}}
    _install_moondream_stub(client)

    with mock.patch.dict(os.environ, {"MOONDREAM_API_KEY": "secret-key"}, clear=True):
        segmenter = MoondreamSegmenter(
            model_name="moondream-3",
            prompt="polyp",
            targets=["polyp"],
            timeout_s=5.0,
        )

    masks, _, parse_success, timed_out, raw_items = segmenter.segment(_small_image())
    assert not timed_out
    assert not parse_success
    assert masks == []
    assert raw_items == []


def test_segment_marks_parse_failure_for_zero_area_bbox() -> None:
    client = mock.Mock()
    client.segment.return_value = {
        "path": "M0 0 L1 0 L1 1 Z",
        "bbox": {"x_min": 0.5, "y_min": 0.25, "x_max": 0.5, "y_max": 0.75},
    }
    _install_moondream_stub(client)

    with mock.patch.dict(os.environ, {"MOONDREAM_API_KEY": "secret-key"}, clear=True):
        segmenter = MoondreamSegmenter(
            model_name="moondream-3",
            prompt="polyp",
            targets=["polyp"],
            timeout_s=5.0,
        )

    mask_api = np.zeros((8, 8), dtype=np.uint8)
    with mock.patch("gemini_segmentation.models._rasterize_svg_path", return_value=mask_api):
        masks, _, parse_success, timed_out, raw_items = segmenter.segment(_small_image())

    assert not timed_out
    assert not parse_success
    assert masks == []
    assert raw_items == []


def test_segment_timeout_returns_expected_flags() -> None:
    client = mock.Mock()
    _install_moondream_stub(client)

    with mock.patch.dict(os.environ, {"MOONDREAM_API_KEY": "secret-key"}, clear=True):
        segmenter = MoondreamSegmenter(
            model_name="moondream-3",
            prompt="polyp",
            targets=["polyp"],
            timeout_s=5.0,
        )

    with mock.patch("gemini_segmentation.models._run_with_timeout", return_value=(None, True)):
        masks, latency, parse_success, timed_out, raw_items = segmenter.segment(_small_image())

    assert masks == []
    assert latency == 0.0
    assert not parse_success
    assert timed_out
    assert raw_items == []
