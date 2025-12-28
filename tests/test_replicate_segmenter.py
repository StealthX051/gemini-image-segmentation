import argparse
import base64
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
    pandas_module = types.ModuleType("pandas")
    sys.modules.setdefault("pandas", pandas_module)


def _install_yaml_stub() -> None:
    yaml_module = types.ModuleType("yaml")
    yaml_module.safe_load = lambda stream: {}
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
            mock_prepare_dirs.return_value = {
                "run_dir": Path(tmp_dir) / "run",
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
