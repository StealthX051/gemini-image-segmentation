import argparse
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

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

from gemini_segmentation.cli import build_parser, command_segment


class PresetBranchParserTests(TestCase):
    def test_parser_accepts_preset_branch(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "segment",
                "polyp",
                "/tmp/data",
                "--prompt-preset",
                "configs/prompts.yaml",
                "--preset-branch",
                "legacy",
            ]
        )
        self.assertEqual(args.preset_branch, "legacy")
        self.assertEqual(args.preset_name, "default")
        self.assertIsNone(args.prompt_family)


class CommandSegmentBranchTests(TestCase):
    @mock.patch("gemini_segmentation.cli.dump_run_config")
    @mock.patch("gemini_segmentation.cli.build_run_config")
    @mock.patch("gemini_segmentation.cli.load_metrics", return_value={})
    @mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={})
    @mock.patch("gemini_segmentation.cli.paired_masks", return_value=[])
    @mock.patch("gemini_segmentation.cli.sample_images", return_value=[])
    @mock.patch("gemini_segmentation.cli.read_manifest", return_value=[])
    @mock.patch("gemini_segmentation.cli._prepare_output_dirs")
    @mock.patch("gemini_segmentation.cli.load_preset")
    @mock.patch("gemini_segmentation.cli.resolve_preset_name")
    @mock.patch("gemini_segmentation.cli.discover_dataset")
    def test_resolves_branch_before_loading_preset(
        self,
        mock_discover,
        mock_resolve,
        mock_load_preset,
        mock_prepare_dirs,
        *_
    ) -> None:
        mock_resolve.return_value = "polyp"
        mock_load_preset.return_value = {
            "prompt_text": "prompt",
            "model": "gemini-2.5-flash",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            manifest_path = dataset_root / "manifest.txt"
            masks_dir = dataset_root / "masks"
            mock_discover.return_value = SimpleNamespace(
                manifest_path=manifest_path, masks_dir=masks_dir
            )
            run_dir = Path(tmp_dir) / "results" / "polyp" / "gemini-2.5-flash" / "label_v1" / "run"
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

            args = argparse.Namespace(
                command="segment",
                dataset_name="polyp",
                dataset_root=str(dataset_root),
                manifest=None,
                provider="gemini",
                model_name="gemini-2.5-flash",
                prompt="",
                prompt_file=None,
                prompt_preset="configs/prompts.yaml",
                preset_name="polyp",
                preset_branch="legacy",
                moondream_targets=None,
                moondream_endpoint=None,
                moondream_api_key=None,
                thinking_budget=0,
                temperature=0.5,
                timeout=1.0,
                workers=1,
                sample_size=None,
                results_dir=tmp_dir,
                run_id=None,
                rate_limit=None,
                legacy_predictions=False,
                success_threshold=0.5,
                bootstrap_method="bca",
                bootstrap_resamples=5000,
                dry_run=False,
                replicate_model_version=None,
                replicate_targets=None,
                replicate_instructions=None,
                replicate_cache_dir=None,
                prompt_family=None,
            )

            command_segment(args)

        mock_resolve.assert_called_once_with("polyp", "legacy")
        mock_load_preset.assert_called_once_with(Path("configs/prompts.yaml"), "polyp")


class PromptFamilyTests(TestCase):
    @mock.patch("gemini_segmentation.cli.build_prompt", return_value="rendered prompt")
    @mock.patch("gemini_segmentation.cli.dump_run_config")
    @mock.patch("gemini_segmentation.cli.build_run_config")
    @mock.patch("gemini_segmentation.cli.load_metrics", return_value={})
    @mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={})
    @mock.patch("gemini_segmentation.cli.paired_masks", return_value=[])
    @mock.patch("gemini_segmentation.cli.sample_images", return_value=[])
    @mock.patch("gemini_segmentation.cli.read_manifest", return_value=[])
    @mock.patch("gemini_segmentation.cli._prepare_output_dirs")
    @mock.patch("gemini_segmentation.cli.load_preset")
    @mock.patch("gemini_segmentation.cli.resolve_preset_name")
    @mock.patch("gemini_segmentation.cli.discover_dataset")
    def test_prompt_family_override(
        self,
        mock_discover,
        mock_resolve,
        mock_load_preset,
        mock_prepare_dirs,
        mock_read_manifest,
        mock_sample_images,
        mock_paired_masks,
        mock_load_existing_predictions,
        mock_load_metrics,
        mock_build_run_config,
        mock_dump_run_config,
        mock_build_prompt,
    ) -> None:
        mock_resolve.return_value = "polyp"
        mock_load_preset.return_value = {
            "prompt_task": "polyp",
            "prompt_family": "label_v1",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            manifest_path = dataset_root / "manifest.txt"
            masks_dir = dataset_root / "masks"
            mock_discover.return_value = SimpleNamespace(
                manifest_path=manifest_path, masks_dir=masks_dir
            )
            run_dir = Path(tmp_dir) / "results" / "polyp" / "gemini-2.5-flash" / "desc_neg_v1" / "run"
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

            args = argparse.Namespace(
                command="segment",
                dataset_name="polyp",
                dataset_root=str(dataset_root),
                manifest=None,
                provider="gemini",
                model_name="gemini-2.5-flash",
                prompt="",
                prompt_file=None,
                prompt_preset="configs/prompts.yaml",
                preset_name="polyp",
                preset_branch=None,
                moondream_targets=None,
                moondream_endpoint=None,
                moondream_api_key=None,
                thinking_budget=0,
                temperature=0.5,
                timeout=1.0,
                workers=1,
                sample_size=None,
                results_dir=tmp_dir,
                run_id=None,
                rate_limit=None,
                legacy_predictions=False,
                success_threshold=0.5,
                bootstrap_method="bca",
                bootstrap_resamples=5000,
                dry_run=False,
                replicate_model_version=None,
                replicate_targets=None,
                replicate_instructions=None,
                replicate_cache_dir=None,
                prompt_family="desc_neg_v1",
            )

            command_segment(args)

        mock_resolve.assert_called_once_with("polyp", None)
        mock_load_preset.assert_called_once_with(Path("configs/prompts.yaml"), "polyp")
        mock_build_prompt.assert_called_once_with("polyp", "desc_neg_v1")


class MultiPromptFamilyTests(TestCase):
    @mock.patch("gemini_segmentation.cli.build_prompt")
    @mock.patch("gemini_segmentation.cli.dump_run_config")
    @mock.patch("gemini_segmentation.cli.build_run_config")
    @mock.patch("gemini_segmentation.cli.load_metrics", return_value={})
    @mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={})
    @mock.patch("gemini_segmentation.cli.paired_masks", return_value=[])
    @mock.patch("gemini_segmentation.cli.sample_images", return_value=[])
    @mock.patch("gemini_segmentation.cli.read_manifest", return_value=[])
    @mock.patch("gemini_segmentation.cli._prepare_output_dirs")
    @mock.patch("gemini_segmentation.cli.discover_dataset")
    def test_runs_multiple_prompt_families(
        self,
        mock_discover,
        mock_prepare_dirs,
        mock_read_manifest,
        mock_sample_images,
        mock_paired_masks,
        mock_load_existing_predictions,
        mock_load_metrics,
        mock_build_run_config,
        mock_dump_run_config,
        mock_build_prompt,
    ) -> None:
        mock_build_prompt.side_effect = ["prompt_a", "prompt_b"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            manifest_path = dataset_root / "manifest.txt"
            masks_dir = dataset_root / "masks"
            mock_discover.return_value = SimpleNamespace(
                manifest_path=manifest_path, masks_dir=masks_dir
            )

            run_dir_label = (
                Path(tmp_dir) / "results" / "polyp" / "gemini-2.5-flash" / "label_v1" / "run"
            )
            run_dir_desc = (
                Path(tmp_dir) / "results" / "polyp" / "gemini-2.5-flash" / "desc_v1" / "run"
            )
            mock_prepare_dirs.side_effect = [
                {
                    "run_dir": run_dir_label,
                    "predictions_jsonl": Path(tmp_dir) / "predictions_a.jsonl",
                    "masks": Path(tmp_dir) / "masks_out",
                    "overlays": Path(tmp_dir) / "overlays",
                    "metrics": Path(tmp_dir) / "metrics.csv",
                    "summary": Path(tmp_dir) / "summary.csv",
                    "fairness": Path(tmp_dir) / "fairness",
                    "run_config": Path(tmp_dir) / "run_config.json",
                    "raw_responses": Path(tmp_dir) / "raw_responses",
                },
                {
                    "run_dir": run_dir_desc,
                    "predictions_jsonl": Path(tmp_dir) / "predictions_b.jsonl",
                    "masks": Path(tmp_dir) / "masks_out",
                    "overlays": Path(tmp_dir) / "overlays",
                    "metrics": Path(tmp_dir) / "metrics.csv",
                    "summary": Path(tmp_dir) / "summary.csv",
                    "fairness": Path(tmp_dir) / "fairness",
                    "run_config": Path(tmp_dir) / "run_config.json",
                    "raw_responses": Path(tmp_dir) / "raw_responses",
                },
            ]

            args = argparse.Namespace(
                command="segment",
                dataset_name="polyp",
                dataset_root=str(dataset_root),
                manifest=None,
                provider="gemini",
                model_name="gemini-2.5-flash",
                prompt="",
                prompt_file=None,
                prompt_preset=None,
                preset_name="default",
                preset_branch=None,
                moondream_targets=None,
                moondream_endpoint=None,
                moondream_api_key=None,
                thinking_budget=0,
                temperature=0.5,
                timeout=1.0,
                workers=1,
                sample_size=None,
                results_dir=tmp_dir,
                run_id=None,
                rate_limit=None,
                legacy_predictions=False,
                success_threshold=0.5,
                bootstrap_method="bca",
                bootstrap_resamples=5000,
                dry_run=True,
                replicate_model_version=None,
                replicate_targets=None,
                replicate_instructions=None,
                replicate_cache_dir=None,
                prompt_family=["label_v1", "desc_v1"],
            )

            command_segment(args)

        self.assertEqual(mock_build_prompt.call_count, 2)
        mock_build_prompt.assert_any_call("polyp", "label_v1")
        mock_build_prompt.assert_any_call("polyp", "desc_v1")
        self.assertEqual(mock_prepare_dirs.call_count, 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
