import argparse
import base64
import hashlib
import io
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Dict
from unittest import mock

import numpy as np
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


def _install_pandas_stub() -> None:
    try:
        import pandas as pandas_module  # type: ignore
        if hasattr(pandas_module, "read_csv"):
            return
    except Exception:
        pass
    pandas_module = types.ModuleType("pandas")
    sys.modules.setdefault("pandas", pandas_module)


def _install_yaml_stub() -> None:
    try:
        import yaml as yaml_module  # type: ignore
        if hasattr(yaml_module, "safe_load") and hasattr(yaml_module, "safe_dump"):
            return
    except Exception:
        pass
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda stream: {}
    yaml_module.safe_dump = lambda payload: "{}\n"
    sys.modules.setdefault("yaml", yaml_module)


def _install_fairness_dep_stubs() -> None:
    scikit_module = types.ModuleType("scikit_posthocs")
    sys.modules.setdefault("scikit_posthocs", scikit_module)

    cliffs_module = types.ModuleType("cliffs_delta")
    cliffs_module.cliffs_delta = lambda *args, **kwargs: (0.0, "negligible")
    sys.modules.setdefault("cliffs_delta", cliffs_module)

    scipy_module = types.ModuleType("scipy")
    stats_module = types.ModuleType("scipy.stats")
    stats_module.rankdata = lambda *args, **kwargs: None
    stats_module.ttest_ind = lambda *args, **kwargs: (0.0, 1.0)
    scipy_module.stats = stats_module
    sys.modules.setdefault("scipy", scipy_module)
    sys.modules.setdefault("scipy.stats", stats_module)

    skimage_module = types.ModuleType("skimage")
    color_module = types.ModuleType("skimage.color")
    color_module.rgb2lab = lambda arr: arr
    skimage_module.color = color_module
    sys.modules.setdefault("skimage", skimage_module)
    sys.modules.setdefault("skimage.color", color_module)


def _install_replicate_stub(result: Dict[str, str]):
    client_mock = mock.Mock()
    client_mock.run.return_value = result
    replicate_module = types.ModuleType("replicate")
    replicate_module.Client = lambda api_token: client_mock
    sys.modules["replicate"] = replicate_module
    return client_mock


_install_google_stubs()
_install_pandas_stub()
_install_yaml_stub()
_install_fairness_dep_stubs()

from gemini_segmentation.cli import build_parser, command_segment  # noqa: E402
from gemini_segmentation.models import Sa2VAReplicateSegmenter  # noqa: E402


def _decode_mask(b64_mask: str) -> np.ndarray:
    prefix = "data:image/png;base64,"
    assert b64_mask.startswith(prefix)
    png_bytes = base64.b64decode(b64_mask.removeprefix(prefix))
    return np.array(Image.open(io.BytesIO(png_bytes)))


def test_sa2va_replicate_segmenter_uses_replicate_output() -> None:
    url = "http://example.com/mask.png"
    client_mock = _install_replicate_stub({"img": url})

    mask_array = np.zeros((3, 4), dtype=np.uint8)
    mask_array[1, 1:3] = 255
    mask_image = Image.fromarray(mask_array, mode="L")

    with mock.patch.dict(os.environ, {"REPLICATE_API_TOKEN": "token"}, clear=False):
        segmenter = Sa2VAReplicateSegmenter(
            model_name="sa2va/segmenter",
            model_version="sa2va/segmenter:1234",
            instruction="segment",
            timeout_s=5.0,
        )

    with mock.patch.object(Sa2VAReplicateSegmenter, "_download_mask", return_value=mask_image) as download_mock:
        image = Image.new("RGB", (4, 3), color="black")
        masks, latency, parse_success, timed_out, raw_items = segmenter.segment(image)

    client_mock.run.assert_called_once()
    download_mock.assert_called_once_with(url)
    submitted_input = client_mock.run.call_args.kwargs["input"]
    submitted_image = submitted_input["image"]
    assert submitted_input["instruction"] == "segment"
    assert not isinstance(submitted_image, (bytes, bytearray))
    assert hasattr(submitted_image, "read")

    assert not timed_out
    assert parse_success
    assert latency >= 0
    assert len(masks) == 1

    seg_mask = masks[0]
    assert seg_mask.y0 == 1
    assert seg_mask.x0 == 1
    assert seg_mask.y1 == 2
    assert seg_mask.x1 == 3
    np.testing.assert_array_equal(seg_mask.mask, mask_array)

    assert raw_items == [
        {
            "label": "",
            "box_2d": [333, 250, 667, 750],
            "mask": raw_items[0]["mask"],
        }
    ]
    decoded_mask = _decode_mask(raw_items[0]["mask"])
    np.testing.assert_array_equal(decoded_mask, mask_array[1:2, 1:3])


def test_replicate_segmenter_requires_token() -> None:
    _install_replicate_stub({"img": "http://example.com/mask.png"})
    with mock.patch.dict(os.environ, {}, clear=True):
        try:
            Sa2VAReplicateSegmenter(
                model_name="sa2va/segmenter",
                model_version="sa2va/segmenter:1234",
                instruction="segment",
            )
        except ValueError as exc:
            assert "REPLICATE_API_TOKEN is required" in str(exc)
        else:
            raise AssertionError("Expected ValueError when REPLICATE_API_TOKEN is missing")


def test_extract_img_url_supports_output_variants() -> None:
    class _FileOutputLike:
        def __init__(self, url: str) -> None:
            self.url = url

    url = "http://example.com/mask.png"
    cases = [
        {"img": url},
        {"image": url},
        [{"img": url}],
        {"img": {"url": url}},
        {"output": [{"image": _FileOutputLike(url)}]},
        _FileOutputLike(url),
    ]
    for payload in cases:
        assert Sa2VAReplicateSegmenter._extract_img_url(payload) == url
    assert Sa2VAReplicateSegmenter._extract_img_url({"response": "ok"}) is None


def test_replicate_segmenter_timeout_sets_timed_out_flag() -> None:
    _install_replicate_stub({"img": "http://example.com/mask.png"})
    with mock.patch.dict(os.environ, {"REPLICATE_API_TOKEN": "token"}, clear=False):
        segmenter = Sa2VAReplicateSegmenter(
            model_name="sa2va/segmenter",
            model_version="sa2va/segmenter:1234",
            instruction="segment",
            timeout_s=5.0,
        )

    with mock.patch("gemini_segmentation.models._run_with_timeout", return_value=(None, True)):
        masks, latency, parse_success, timed_out, raw_items = segmenter.segment(Image.new("RGB", (4, 4)))

    assert masks == []
    assert latency == 0.0
    assert parse_success is False
    assert timed_out is True
    assert raw_items == []


def test_replicate_segmenter_reports_parse_failure_on_missing_output_url() -> None:
    _install_replicate_stub({"response": "ok"})
    with mock.patch.dict(os.environ, {"REPLICATE_API_TOKEN": "token"}, clear=False):
        segmenter = Sa2VAReplicateSegmenter(
            model_name="sa2va/segmenter",
            model_version="sa2va/segmenter:1234",
            instruction="segment",
            timeout_s=5.0,
        )

    masks, _, parse_success, timed_out, raw_items = segmenter.segment(Image.new("RGB", (4, 4)))
    assert masks == []
    assert parse_success is False
    assert timed_out is False
    assert raw_items == []


def test_replicate_segmenter_reports_parse_failure_on_empty_mask() -> None:
    _install_replicate_stub({"img": "http://example.com/mask.png"})
    with mock.patch.dict(os.environ, {"REPLICATE_API_TOKEN": "token"}, clear=False):
        segmenter = Sa2VAReplicateSegmenter(
            model_name="sa2va/segmenter",
            model_version="sa2va/segmenter:1234",
            instruction="segment",
            timeout_s=5.0,
        )

    empty_mask = Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L")
    with mock.patch.object(Sa2VAReplicateSegmenter, "_download_mask", return_value=empty_mask):
        masks, _, parse_success, timed_out, raw_items = segmenter.segment(Image.new("RGB", (4, 4)))

    assert masks == []
    assert parse_success is False
    assert timed_out is False
    assert raw_items == []


def test_download_mask_prefers_cached_file() -> None:
    _install_replicate_stub({"img": "http://example.com/mask.png"})
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir) / "cache"
        with mock.patch.dict(os.environ, {"REPLICATE_API_TOKEN": "token"}, clear=False):
            segmenter = Sa2VAReplicateSegmenter(
                model_name="sa2va/segmenter",
                model_version="sa2va/segmenter:1234",
                instruction="segment",
                cache_dir=cache_dir,
            )

        url = "http://example.com/mask.png"
        cache_path = cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.png"
        expected = np.zeros((3, 3), dtype=np.uint8)
        expected[1, 1] = 255
        Image.fromarray(expected, mode="L").save(cache_path)

        with mock.patch("gemini_segmentation.models.urlopen", side_effect=AssertionError("network call not expected")):
            mask = segmenter._download_mask(url)

    assert mask is not None
    np.testing.assert_array_equal(np.array(mask), expected)


def test_download_mask_returns_none_for_invalid_image_bytes() -> None:
    _install_replicate_stub({"img": "http://example.com/mask.png"})
    with mock.patch.dict(os.environ, {"REPLICATE_API_TOKEN": "token"}, clear=False):
        segmenter = Sa2VAReplicateSegmenter(
            model_name="sa2va/segmenter",
            model_version="sa2va/segmenter:1234",
            instruction="segment",
        )

    class _DummyResponse:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return self.payload

    with mock.patch(
        "gemini_segmentation.models.urlopen",
        return_value=_DummyResponse(b"not-an-image"),
    ):
        assert segmenter._download_mask("http://example.com/mask.png") is None


def test_replicate_segmenter_uses_per_target_instruction_mapping() -> None:
    client_mock = _install_replicate_stub({"img": "http://example.com/mask.png"})
    client_mock.run.side_effect = [
        {"img": "http://example.com/mask1.png"},
        {"img": "http://example.com/mask2.png"},
    ]
    with mock.patch.dict(os.environ, {"REPLICATE_API_TOKEN": "token"}, clear=False):
        segmenter = Sa2VAReplicateSegmenter(
            model_name="sa2va/segmenter",
            model_version="sa2va/segmenter:1234",
            instruction="fallback",
            timeout_s=5.0,
            targets=["lesion", "instrument"],
            instructions={"lesion": "find lesion", "instrument": "find instrument"},
        )

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    with mock.patch.object(
        Sa2VAReplicateSegmenter,
        "_download_mask",
        side_effect=[Image.fromarray(mask, mode="L"), Image.fromarray(mask, mode="L")],
    ):
        masks, _, parse_success, timed_out, raw_items = segmenter.segment(Image.new("RGB", (4, 4)))

    assert parse_success is True
    assert timed_out is False
    assert len(masks) == 2
    assert [item["label"] for item in raw_items] == ["lesion", "instrument"]
    assert [call.kwargs["input"]["instruction"] for call in client_mock.run.call_args_list] == [
        "find lesion",
        "find instrument",
    ]


def test_cli_parses_replicate_args_into_run_config() -> None:
    url = "http://example.com/mask.png"
    _install_replicate_stub({"img": url})

    parser = build_parser()
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_root = Path(tmp_dir) / "data"
        dataset_root.mkdir(parents=True, exist_ok=True)
        args = parser.parse_args(
            [
                "segment",
                "polyp",
                str(dataset_root),
                "--provider",
                "replicate",
                "--replicate-model-version",
                "sa2va/segmenter:1234",
                "--replicate-target",
                "lesion",
                "--replicate-instruction",
                "find lesion",
                "--replicate-target",
                "instrument",
                "--replicate-instruction",
                "find instrument",
                "--replicate-cache-dir",
                str(dataset_root / "cache"),
                "--results-dir",
                str(dataset_root / "results"),
                "--dry-run",
            ]
        )

        manifest_path = dataset_root / "manifest.txt"
        masks_dir = dataset_root / "masks"

        with (
            mock.patch("gemini_segmentation.cli.dump_run_config") as mock_dump_run_config,
            mock.patch("gemini_segmentation.cli.build_run_config") as mock_build_run_config,
            mock.patch("gemini_segmentation.cli.load_metrics", return_value={}),
            mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={}),
            mock.patch("gemini_segmentation.cli.paired_masks", return_value=[]),
            mock.patch("gemini_segmentation.cli.sample_images", return_value=[]),
            mock.patch("gemini_segmentation.cli.read_manifest", return_value=[]),
            mock.patch("gemini_segmentation.cli._prepare_output_dirs") as mock_prepare_dirs,
            mock.patch("gemini_segmentation.cli.discover_dataset") as mock_discover,
        ):

            mock_discover.return_value = types.SimpleNamespace(
                manifest_path=manifest_path, masks_dir=masks_dir
            )
            run_dir = (
                Path(tmp_dir)
                / "results"
                / "polyp"
                / "sa2va/segmenter:1234"
                / "prompt-1234"
                / "run"
            )
            mock_prepare_dirs.return_value = {
                "run_dir": run_dir,
                "predictions_jsonl": Path(tmp_dir) / "predictions.jsonl",
                "masks": Path(tmp_dir) / "masks_out",
                "overlays": Path(tmp_dir) / "overlays",
                "metrics": Path(tmp_dir) / "metrics.csv",
                "summary": Path(tmp_dir) / "summary.csv",
                "fairness": Path(tmp_dir) / "fairness",
                "run_config": Path(tmp_dir) / "run_config.json",
                "raw_responses": Path(tmp_dir) / "raw_responses",
            }

            command_segment(args)

    mock_build_run_config.assert_called_once()
    kwargs = mock_build_run_config.call_args.kwargs

    assert kwargs["provider"] == "replicate"
    assert kwargs["replicate_model_version"] == "sa2va/segmenter:1234"
    assert kwargs["replicate_targets"] == ("lesion", "instrument")
    assert kwargs["replicate_instructions"] == {
        "lesion": "find lesion",
        "instrument": "find instrument",
    }
    assert kwargs["replicate_cache_dir"] == (dataset_root / "cache").resolve()

    assert mock_dump_run_config.called
