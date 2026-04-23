import sys
import types
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


class _CaptureConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _CapturePart:
    def __init__(self, *, text=None, data=None, mime_type=None):
        self.text = text
        self.data = data
        self.mime_type = mime_type

    @staticmethod
    def from_bytes(*, data, mime_type):
        return _CapturePart(data=data, mime_type=mime_type)


class _CaptureTool:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _CaptureToolCodeExecution:
    pass


class _CaptureThinkingConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _CaptureSafetySetting:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_google_stubs() -> None:
    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_types_module = types.ModuleType("google.genai.types")

    genai_types_module.GenerateContentConfig = _CaptureConfig
    genai_types_module.Part = _CapturePart
    genai_types_module.SafetySetting = _CaptureSafetySetting
    genai_types_module.ThinkingConfig = _CaptureThinkingConfig
    genai_types_module.Tool = _CaptureTool
    genai_types_module.ToolCodeExecution = _CaptureToolCodeExecution
    genai_module.types = genai_types_module
    genai_module.Client = mock.Mock()
    google_module.genai = genai_module

    sys.modules["google"] = google_module
    sys.modules["google.genai"] = genai_module
    sys.modules["google.genai.types"] = genai_types_module


_install_google_stubs()

from gemini_segmentation import models as _models  # noqa: E402

models = importlib.reload(_models)


def _small_image() -> Image.Image:
    return Image.new("RGB", (8, 8), color="black")


def _response_with_text(text: str):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text=text)]))],
        usage_metadata=None,
    )


def _reset_prompt_cache_state() -> None:
    models.GeminiSegmenter._prompt_cache_registry.clear()
    models.GeminiSegmenter._prompt_cache_disabled.clear()


def test_gemini_segmenter_adds_code_execution_tools_when_agentic_vision_enabled() -> None:
    _reset_prompt_cache_state()
    client = mock.Mock()
    client.vertexai = False
    client.models.generate_content.return_value = _response_with_text("[]")

    with mock.patch.object(models.genai, "Client", return_value=client):
        segmenter = models.GeminiSegmenter(
            model_name="gemini-robotics-er-1.6-preview",
            prompt="Segment the lesion.",
            explicit_cache=False,
            gemini_agentic_vision=True,
        )
        segmenter.segment(_small_image())

    config = client.models.generate_content.call_args.kwargs["config"]
    assert "tools" in config.kwargs
    assert "response_mime_type" not in config.kwargs
    assert "response_json_schema" not in config.kwargs
    assert len(config.kwargs["tools"]) == 1
    tool = config.kwargs["tools"][0]
    assert isinstance(tool, _CaptureTool)
    assert isinstance(tool.kwargs["code_execution"], _CaptureToolCodeExecution)


def test_gemini_segmenter_omits_code_execution_tools_when_agentic_vision_disabled() -> None:
    _reset_prompt_cache_state()
    client = mock.Mock()
    client.vertexai = False
    client.models.generate_content.return_value = _response_with_text("[]")

    with mock.patch.object(models.genai, "Client", return_value=client):
        segmenter = models.GeminiSegmenter(
            model_name="gemini-robotics-er-1.6-preview",
            prompt="Segment the lesion.",
            explicit_cache=False,
            gemini_agentic_vision=False,
        )
        segmenter.segment(_small_image())

    config = client.models.generate_content.call_args.kwargs["config"]
    assert "tools" not in config.kwargs


def test_gemini_2_5_models_enable_structured_json_output() -> None:
    _reset_prompt_cache_state()
    client = mock.Mock()
    client.vertexai = False
    client.models.generate_content.return_value = _response_with_text("[]")

    with mock.patch.object(models.genai, "Client", return_value=client):
        segmenter = models.GeminiSegmenter(
            model_name="gemini-2.5-flash",
            prompt="Segment the lesion.",
            explicit_cache=False,
        )
        segmenter.segment(_small_image())

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.kwargs["response_mime_type"] == "application/json"
    assert isinstance(config.kwargs["response_json_schema"], dict)
    assert config.kwargs["response_json_schema"]["type"] == "array"


def test_robotics_er_1_6_plain_keeps_structured_output_disabled() -> None:
    _reset_prompt_cache_state()
    client = mock.Mock()
    client.vertexai = False
    client.models.generate_content.return_value = _response_with_text("[]")

    with mock.patch.object(models.genai, "Client", return_value=client):
        segmenter = models.GeminiSegmenter(
            model_name="gemini-robotics-er-1.6-preview",
            prompt="Segment the lesion.",
            explicit_cache=False,
        )
        segmenter.segment(_small_image())

    config = client.models.generate_content.call_args.kwargs["config"]
    assert "response_mime_type" not in config.kwargs
    assert "response_json_schema" not in config.kwargs


def test_gemini_segmenter_sends_image_before_text() -> None:
    _reset_prompt_cache_state()
    client = mock.Mock()
    client.vertexai = False
    client.models.generate_content.return_value = _response_with_text("[]")

    with mock.patch.object(models.genai, "Client", return_value=client):
        segmenter = models.GeminiSegmenter(
            model_name="gemini-2.5-flash",
            prompt="Segment the lesion.",
            explicit_cache=False,
        )
        segmenter.segment(_small_image())

    contents = client.models.generate_content.call_args.kwargs["contents"]
    assert len(contents) == 2
    assert contents[0].mime_type == "image/jpeg"
    assert contents[1].text == "Segment the lesion."


def test_robotics_er_1_6_allows_explicit_cache() -> None:
    _reset_prompt_cache_state()
    client = mock.Mock()
    client.vertexai = False
    client.caches.create.return_value = SimpleNamespace(name="cache-id")

    with mock.patch.object(models.genai, "Client", return_value=client):
        segmenter = models.GeminiSegmenter(
            model_name="gemini-robotics-er-1.6-preview",
            prompt="Segment the lesion.",
            explicit_cache=True,
        )

    assert segmenter.cached_content_name == "cache-id"
    client.caches.create.assert_called()


def test_robotics_er_1_5_skips_explicit_cache() -> None:
    _reset_prompt_cache_state()
    client = mock.Mock()
    client.vertexai = False

    with mock.patch.object(models.genai, "Client", return_value=client):
        segmenter = models.GeminiSegmenter(
            model_name="gemini-robotics-er-1.5-preview",
            prompt="Segment the lesion.",
            explicit_cache=True,
        )

    assert segmenter.cached_content_name is None
    client.caches.create.assert_not_called()
