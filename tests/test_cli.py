import argparse
import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import pytest

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

from gemini_segmentation.cli import (
    _process_image_with_cache,
    _prompt_hash,
    _resolve_provider_prompt,
    build_parser,
    command_fairness,
    command_segment,
)
from gemini_segmentation.cache import DiskRequestCache, build_request_cache_key
from gemini_segmentation.io import encode_mask_to_b64
from gemini_segmentation.prompts import ProviderPrompt


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


class CacheFlagParserTests(TestCase):
    def test_parser_defaults_for_cache_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["segment", "polyp", "/tmp/data"])
        self.assertTrue(args.local_cache)
        self.assertIsNone(args.local_cache_dir)
        self.assertTrue(args.gemini_explicit_cache)
        self.assertEqual(args.gemini_cache_ttl, 3600)
        self.assertEqual(args.max_retries, 5)

    def test_parser_accepts_cache_flag_overrides(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "segment",
                "polyp",
                "/tmp/data",
                "--no-local-cache",
                "--local-cache-dir",
                "/tmp/request_cache",
                "--no-gemini-explicit-cache",
                "--gemini-cache-ttl",
                "7200",
                "--max-retries",
                "7",
            ]
        )
        self.assertFalse(args.local_cache)
        self.assertEqual(args.local_cache_dir, "/tmp/request_cache")
        self.assertFalse(args.gemini_explicit_cache)
        self.assertEqual(args.gemini_cache_ttl, 7200)
        self.assertEqual(args.max_retries, 7)


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
    @mock.patch(
        "gemini_segmentation.cli.build_prompt_for_provider",
        return_value=ProviderPrompt(prompt="rendered prompt"),
    )
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
        mock_build_prompt_for_provider,
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
        self.assertGreaterEqual(mock_build_prompt_for_provider.call_count, 1)
        mock_build_prompt_for_provider.assert_any_call(
            "polyp", "desc_neg_v1", "gemini", targets_override=None
        )


class MultiPromptFamilyTests(TestCase):
    @mock.patch("gemini_segmentation.cli.build_prompt_for_provider")
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
        mock_build_prompt_for_provider,
    ) -> None:
        mock_build_prompt_for_provider.side_effect = [
            ProviderPrompt(prompt="prompt_a"),
            ProviderPrompt(prompt="prompt_a"),
            ProviderPrompt(prompt="prompt_b"),
            ProviderPrompt(prompt="prompt_b"),
        ]
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
            first_paths = {
                "run_dir": run_dir_label,
                "predictions_jsonl": Path(tmp_dir) / "predictions_a.jsonl",
                "masks": Path(tmp_dir) / "masks_out",
                "overlays": Path(tmp_dir) / "overlays",
                "metrics": Path(tmp_dir) / "metrics.csv",
                "summary": Path(tmp_dir) / "summary.csv",
                "fairness": Path(tmp_dir) / "fairness",
                "run_config": Path(tmp_dir) / "run_config.json",
                "raw_responses": Path(tmp_dir) / "raw_responses",
            }
            second_paths = {
                "run_dir": run_dir_desc,
                "predictions_jsonl": Path(tmp_dir) / "predictions_b.jsonl",
                "masks": Path(tmp_dir) / "masks_out",
                "overlays": Path(tmp_dir) / "overlays",
                "metrics": Path(tmp_dir) / "metrics.csv",
                "summary": Path(tmp_dir) / "summary.csv",
                "fairness": Path(tmp_dir) / "fairness",
                "run_config": Path(tmp_dir) / "run_config.json",
                "raw_responses": Path(tmp_dir) / "raw_responses",
            }
            mock_prepare_dirs.side_effect = [first_paths, first_paths, second_paths, second_paths]

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

        self.assertEqual(mock_build_prompt_for_provider.call_count, 4)
        mock_build_prompt_for_provider.assert_any_call(
            "polyp", "label_v1", "gemini", targets_override=None
        )
        mock_build_prompt_for_provider.assert_any_call(
            "polyp", "desc_v1", "gemini", targets_override=None
        )
        self.assertEqual(mock_prepare_dirs.call_count, 4)


class PromptKeyTests(TestCase):
    @mock.patch(
        "gemini_segmentation.cli.build_prompt_for_provider",
        return_value=ProviderPrompt(prompt="rendered prompt"),
    )
    @mock.patch("gemini_segmentation.cli.dump_run_config")
    @mock.patch("gemini_segmentation.cli.build_run_config")
    @mock.patch("gemini_segmentation.cli.load_metrics", return_value={})
    @mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={})
    @mock.patch("gemini_segmentation.cli.paired_masks", return_value=[])
    @mock.patch("gemini_segmentation.cli.sample_images", return_value=[])
    @mock.patch("gemini_segmentation.cli.read_manifest", return_value=[])
    @mock.patch("gemini_segmentation.cli._prepare_output_dirs")
    @mock.patch("gemini_segmentation.cli.discover_dataset")
    def test_prompt_family_key_includes_hash(
        self,
        mock_discover,
        mock_prepare_dirs,
        *_,
    ) -> None:
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
                run_id="run",
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
                prompt_family="label_v1",
            )

            command_segment(args)

        prompt_key_arg = mock_prepare_dirs.call_args[0][3]
        payload = {"provider": "gemini", "family": "label_v1", "prompt": "rendered prompt"}
        expected_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8]
        assert prompt_key_arg == f"label_v1-{expected_hash}"


class PromptHashTests(TestCase):
    def test_prompt_hash_varies_by_provider(self) -> None:
        payload_gemini = {"provider": "gemini", "family": "label_v1", "prompt": "t"}
        payload_moondream = {"provider": "moondream", "family": "label_v1", "prompt": "t"}
        self.assertNotEqual(_prompt_hash(payload_gemini), _prompt_hash(payload_moondream))


class MoondreamCommandSegmentTests(TestCase):
    @mock.patch(
        "gemini_segmentation.cli._resolve_provider_prompt",
        return_value=ProviderPrompt(prompt="lesion", targets=("lesion",)),
    )
    @mock.patch("gemini_segmentation.cli.dump_run_config")
    @mock.patch("gemini_segmentation.cli.build_run_config")
    @mock.patch("gemini_segmentation.cli.load_metrics", return_value={})
    @mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={})
    @mock.patch("gemini_segmentation.cli.paired_masks", return_value=[])
    @mock.patch("gemini_segmentation.cli.sample_images", return_value=[])
    @mock.patch("gemini_segmentation.cli.read_manifest", return_value=[])
    @mock.patch("gemini_segmentation.cli._prepare_output_dirs")
    @mock.patch("gemini_segmentation.cli.discover_dataset")
    def test_moondream_provider_maps_default_model_and_preserves_cli_targets(
        self,
        mock_discover,
        mock_prepare_dirs,
        _mock_read_manifest,
        _mock_sample_images,
        _mock_paired_masks,
        _mock_load_existing_predictions,
        _mock_load_metrics,
        mock_build_run_config,
        _mock_dump_run_config,
        mock_resolve_provider_prompt,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            manifest_path = dataset_root / "manifest.txt"
            masks_dir = dataset_root / "masks"
            mock_discover.return_value = SimpleNamespace(
                manifest_path=manifest_path, masks_dir=masks_dir
            )
            run_dir = (
                Path(tmp_dir) / "results" / "polyp" / "moondream-3" / "label_v1-1234abcd" / "run"
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

            args = argparse.Namespace(
                command="segment",
                dataset_name="polyp",
                dataset_root=str(dataset_root),
                manifest=None,
                provider="moondream",
                model_name="gemini-2.5-flash",
                prompt="",
                prompt_file=None,
                prompt_preset=None,
                preset_name="default",
                preset_branch=None,
                moondream_targets=["lesion"],
                moondream_endpoint=None,
                moondream_api_key=None,
                thinking_budget=0,
                temperature=0.5,
                timeout=1.0,
                max_retries=5,
                workers=1,
                sample_size=None,
                results_dir=tmp_dir,
                run_id="run",
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
                prompt_family="label_v1",
                local_cache=True,
                local_cache_dir=None,
                gemini_explicit_cache=True,
                gemini_cache_ttl=3600,
            )

            command_segment(args)

        kwargs = mock_build_run_config.call_args.kwargs
        self.assertEqual(kwargs["provider"], "moondream")
        self.assertEqual(kwargs["model_name"], "moondream-3")
        self.assertEqual(kwargs["moondream_targets"], ["lesion"])
        mock_resolve_provider_prompt.assert_any_call(
            provider="moondream",
            prompt_family="label_v1",
            explicit_prompt=None,
            prompt_task="polyp",
            target_overrides=["lesion"],
            replicate_instruction_overrides=None,
        )

    @mock.patch(
        "gemini_segmentation.cli._resolve_provider_prompt",
        return_value=ProviderPrompt(prompt="optic disc", targets=("optic disc", "optic cup")),
    )
    @mock.patch("gemini_segmentation.cli.dump_run_config")
    @mock.patch("gemini_segmentation.cli.build_run_config")
    @mock.patch("gemini_segmentation.cli.load_metrics", return_value={})
    @mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={})
    @mock.patch("gemini_segmentation.cli.paired_masks", return_value=[])
    @mock.patch("gemini_segmentation.cli.sample_images", return_value=[])
    @mock.patch("gemini_segmentation.cli.read_manifest", return_value=[])
    @mock.patch("gemini_segmentation.cli._prepare_output_dirs")
    @mock.patch("gemini_segmentation.cli.discover_dataset")
    def test_moondream_targets_default_from_provider_prompt_when_not_passed(
        self,
        mock_discover,
        mock_prepare_dirs,
        _mock_read_manifest,
        _mock_sample_images,
        _mock_paired_masks,
        _mock_load_existing_predictions,
        _mock_load_metrics,
        mock_build_run_config,
        _mock_dump_run_config,
        mock_resolve_provider_prompt,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            manifest_path = dataset_root / "manifest.txt"
            masks_dir = dataset_root / "masks"
            mock_discover.return_value = SimpleNamespace(
                manifest_path=manifest_path, masks_dir=masks_dir
            )
            run_dir = (
                Path(tmp_dir) / "results" / "polyp" / "moondream-3" / "label_v1-1234abcd" / "run"
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

            args = argparse.Namespace(
                command="segment",
                dataset_name="polyp",
                dataset_root=str(dataset_root),
                manifest=None,
                provider="moondream",
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
                max_retries=5,
                workers=1,
                sample_size=None,
                results_dir=tmp_dir,
                run_id="run",
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
                prompt_family="label_v1",
                local_cache=True,
                local_cache_dir=None,
                gemini_explicit_cache=True,
                gemini_cache_ttl=3600,
            )

            command_segment(args)

        kwargs = mock_build_run_config.call_args.kwargs
        self.assertEqual(kwargs["provider"], "moondream")
        self.assertEqual(kwargs["model_name"], "moondream-3")
        self.assertEqual(kwargs["moondream_targets"], ["optic disc", "optic cup"])
        mock_resolve_provider_prompt.assert_any_call(
            provider="moondream",
            prompt_family="label_v1",
            explicit_prompt=None,
            prompt_task="polyp",
            target_overrides=None,
            replicate_instruction_overrides=None,
        )


class ProviderPromptResolutionTests(TestCase):
    @mock.patch(
        "gemini_segmentation.cli.build_prompt_for_provider",
        return_value=ProviderPrompt(
            prompt="Segment the colorectal polyp.",
            targets=("colorectal polyp",),
            instructions={"colorectal polyp": "Segment the colorectal polyp."},
        ),
    )
    def test_replicate_overrides_fill_missing_instructions(self, _mock_build_prompt) -> None:
        resolved = _resolve_provider_prompt(
            provider="replicate",
            prompt_family="label_v1",
            explicit_prompt=None,
            prompt_task="polyp",
            target_overrides=["new target"],
            replicate_instruction_overrides=None,
        )

        self.assertEqual(resolved.targets, ("new target",))
        self.assertEqual(resolved.prompt, "Segment the new target.")
        self.assertIn("new target", resolved.instructions)
        self.assertEqual(resolved.instructions["new target"], "Segment the new target.")


class LocalRequestCacheBehaviorTests(TestCase):
    def test_retries_parse_failure_before_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "img.png"
            from PIL import Image
            import numpy as np

            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(img_path)
            cache = DiskRequestCache(Path(tmp_dir) / "cache")

            class _FlakySegmenter:
                def __init__(self) -> None:
                    self.calls = 0

                def segment(self, _image_obj):
                    self.calls += 1
                    if self.calls == 1:
                        return [], 0.1, False, False, []
                    mask = np.zeros((8, 8), dtype=np.uint8)
                    mask[2:6, 2:6] = 255
                    raw_items = [
                        {
                            "label": "lesion",
                            "box_2d": [250, 250, 750, 750],
                            "mask": encode_mask_to_b64(mask[2:6, 2:6]),
                        }
                    ]
                    from gemini_segmentation.types import SegmentationMask

                    seg_mask = SegmentationMask(2, 2, 6, 6, mask, "lesion")
                    return [seg_mask], 0.2, True, False, raw_items

            flaky = _FlakySegmenter()

            _, masks, _, parse_success, timed_out, _, from_cache = _process_image_with_cache(
                lambda: flaky,
                img_path,
                provider="gemini",
                model_name="gemini-2.5-flash",
                prompt_hash="prompt-digest",
                prompt_family="label_v1",
                temperature=0.5,
                thinking_budget=0,
                request_cache=cache,
                max_retries=1,
            )

            self.assertFalse(from_cache)
            self.assertTrue(parse_success)
            self.assertFalse(timed_out)
            self.assertEqual(len(masks), 1)
            self.assertEqual(flaky.calls, 2)

    def test_parse_failure_is_not_written_to_local_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "img.png"
            from PIL import Image
            import numpy as np

            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(img_path)
            cache = DiskRequestCache(Path(tmp_dir) / "cache")

            class _FailingSegmenter:
                def segment(self, _image_obj):
                    raw_items = [{"label": "x", "box_2d": [0, 0, 1000, 1000], "mask": "bad"}]
                    return [], 0.1, False, False, raw_items

            _process_image_with_cache(
                lambda: _FailingSegmenter(),
                img_path,
                provider="gemini",
                model_name="gemini-2.5-flash",
                prompt_hash="prompt-digest",
                prompt_family="label_v1",
                temperature=0.5,
                thinking_budget=0,
                request_cache=cache,
            )

            key = build_request_cache_key(
                image_path=img_path,
                provider="gemini",
                model_name="gemini-2.5-flash",
                prompt_hash="prompt-digest",
                prompt_family="label_v1",
                temperature=0.5,
                thinking_budget=0,
                targets=None,
            )
            self.assertIsNone(cache.load(key))

    def test_cache_hit_bypasses_segmenter_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "img.png"
            from PIL import Image
            import numpy as np

            image_np = np.zeros((10, 10), dtype=np.uint8)
            Image.fromarray(image_np).save(img_path)
            cache = DiskRequestCache(Path(tmp_dir) / "cache")

            patch = np.zeros((4, 4), dtype=np.uint8)
            patch[:, :] = 255
            raw_items = [
                {
                    "label": "lesion",
                    "box_2d": [200, 200, 600, 600],
                    "mask": encode_mask_to_b64(patch),
                }
            ]
            key = build_request_cache_key(
                image_path=img_path,
                provider="gemini",
                model_name="gemini-2.5-flash",
                prompt_hash="prompt-digest",
                prompt_family="desc_v1",
                temperature=0.5,
                thinking_budget=0,
                targets=None,
            )
            cache.save(
                key,
                {
                    "latency_s": 0.2,
                    "parse_success": True,
                    "timed_out": False,
                    "raw_items": raw_items,
                },
            )

            class _UnexpectedSegmenter:
                def segment(self, _image_obj):
                    raise AssertionError("Segmenter should not be called on cache hit")

            _, masks, _, parse_success, timed_out, _, from_cache = _process_image_with_cache(
                lambda: _UnexpectedSegmenter(),
                img_path,
                provider="gemini",
                model_name="gemini-2.5-flash",
                prompt_hash="prompt-digest",
                prompt_family="desc_v1",
                temperature=0.5,
                thinking_budget=0,
                request_cache=cache,
            )

            self.assertTrue(from_cache)
            self.assertTrue(parse_success)
            self.assertFalse(timed_out)
            self.assertEqual(len(masks), 1)
            self.assertEqual(masks[0].label, "lesion")


class ReplicateValidationTests(TestCase):
    @mock.patch("gemini_segmentation.cli._resolve_provider_prompt", return_value=ProviderPrompt(prompt="p"))
    @mock.patch("gemini_segmentation.cli.load_metrics", return_value={})
    @mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={})
    @mock.patch("gemini_segmentation.cli.paired_masks", return_value=[])
    @mock.patch("gemini_segmentation.cli.sample_images", return_value=[])
    @mock.patch("gemini_segmentation.cli.read_manifest", return_value=[])
    @mock.patch("gemini_segmentation.cli._prepare_output_dirs")
    @mock.patch("gemini_segmentation.cli.discover_dataset")
    def test_instruction_and_target_counts_must_match(
        self,
        mock_discover,
        mock_prepare_dirs,
        *_,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "data"
            mock_discover.return_value = SimpleNamespace(
                manifest_path=dataset_root / "manifest.txt", masks_dir=dataset_root / "masks"
            )
            run_dir = Path(tmp_dir) / "results" / "polyp" / "replicate" / "label" / "run"
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
                provider="replicate",
                model_name="sa-1b",
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
                run_id="run",
                rate_limit=None,
                legacy_predictions=False,
                success_threshold=0.5,
                bootstrap_method="bca",
                bootstrap_resamples=5000,
                dry_run=False,
                replicate_model_version="1.0",
                replicate_targets=["a", "b"],
                replicate_instructions=["one"],
                replicate_cache_dir=None,
                prompt_family=None,
            )

            with self.assertRaisesRegex(
                ValueError,
                "The number of --replicate-instruction flags must match --replicate-target entries.",
            ):
                command_segment(args)

    @mock.patch("gemini_segmentation.cli._resolve_provider_prompt", return_value=ProviderPrompt(prompt="p"))
    @mock.patch("gemini_segmentation.cli.load_metrics", return_value={})
    @mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={})
    @mock.patch("gemini_segmentation.cli.paired_masks", return_value=[])
    @mock.patch("gemini_segmentation.cli.sample_images", return_value=[])
    @mock.patch("gemini_segmentation.cli.read_manifest", return_value=[])
    @mock.patch("gemini_segmentation.cli._prepare_output_dirs")
    @mock.patch("gemini_segmentation.cli.discover_dataset")
    def test_model_version_required_for_replicate_provider(
        self,
        mock_discover,
        mock_prepare_dirs,
        *_,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "data"
            mock_discover.return_value = SimpleNamespace(
                manifest_path=dataset_root / "manifest.txt", masks_dir=dataset_root / "masks"
            )
            run_dir = Path(tmp_dir) / "results" / "polyp" / "replicate" / "label" / "run"
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
                provider="replicate",
                model_name="sa-1b",
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
                run_id="run",
                rate_limit=None,
                legacy_predictions=False,
                success_threshold=0.5,
                bootstrap_method="bca",
                bootstrap_resamples=5000,
                dry_run=False,
                replicate_model_version=None,
                replicate_targets=["a"],
                replicate_instructions=None,
                replicate_cache_dir=None,
                prompt_family=None,
            )

            with self.assertRaisesRegex(
                ValueError, "--replicate-model-version is required when provider is 'replicate'"
            ):
                command_segment(args)

    @mock.patch("gemini_segmentation.cli.load_metrics", return_value={})
    @mock.patch("gemini_segmentation.cli.load_existing_predictions", return_value={})
    @mock.patch("gemini_segmentation.cli.paired_masks", return_value=[])
    @mock.patch("gemini_segmentation.cli.sample_images", return_value=[])
    @mock.patch("gemini_segmentation.cli.read_manifest", return_value=[])
    @mock.patch("gemini_segmentation.cli._prepare_output_dirs")
    @mock.patch("gemini_segmentation.cli.discover_dataset")
    @mock.patch("gemini_segmentation.cli.build_run_config")
    @mock.patch("gemini_segmentation.cli.dump_run_config")
    @mock.patch("gemini_segmentation.cli._resolve_provider_prompt")
    def test_replicate_resume_preserves_instruction_mapping_from_run_config(
        self,
        mock_resolve_provider_prompt,
        _mock_dump_run_config,
        mock_build_run_config,
        mock_discover,
        mock_prepare_dirs,
        *_,
    ) -> None:
        def _resolve_prompt(**kwargs):
            targets = tuple(kwargs.get("target_overrides") or ())
            instructions = dict(kwargs.get("replicate_instruction_overrides") or {})
            primary = (
                instructions.get(targets[0], kwargs.get("explicit_prompt") or "")
                if targets
                else kwargs.get("explicit_prompt") or ""
            )
            return ProviderPrompt(
                prompt=primary,
                targets=targets or None,
                instructions=instructions or None,
            )

        mock_resolve_provider_prompt.side_effect = _resolve_prompt

        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "data"
            dataset_root.mkdir(parents=True, exist_ok=True)
            manifest_path = dataset_root / "manifest.txt"
            masks_dir = dataset_root / "masks"
            mock_discover.return_value = SimpleNamespace(
                manifest_path=manifest_path, masks_dir=masks_dir
            )

            run_config = {
                "provider": "replicate",
                "replicate_model_version": "sa2va/segmenter:1234",
                "replicate_targets": ["lesion", "instrument"],
                "replicate_instructions": {
                    "lesion": "find lesion",
                    "instrument": "find instrument",
                },
            }
            run_config_path = Path(tmp_dir) / "run_config.json"
            run_config_path.write_text(json.dumps(run_config), encoding="utf-8")
            run_dir = Path(tmp_dir) / "results" / "polyp" / "sa2va/segmenter:1234" / "label" / "run"
            mock_prepare_dirs.return_value = {
                "run_dir": run_dir,
                "predictions_jsonl": Path(tmp_dir) / "predictions.jsonl",
                "masks": Path(tmp_dir) / "masks_out",
                "overlays": Path(tmp_dir) / "overlays",
                "metrics": Path(tmp_dir) / "metrics.csv",
                "summary": Path(tmp_dir) / "summary.csv",
                "fairness": Path(tmp_dir) / "fairness",
                "run_config": run_config_path,
                "raw_responses": Path(tmp_dir) / "raw_responses",
            }

            args = argparse.Namespace(
                command="segment",
                dataset_name="polyp",
                dataset_root=str(dataset_root),
                manifest=None,
                provider="replicate",
                model_name="sa2va/segmenter",
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
                run_id="run",
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
                prompt_family=None,
            )

            command_segment(args)

        kwargs = mock_build_run_config.call_args.kwargs
        self.assertEqual(kwargs["replicate_model_version"], "sa2va/segmenter:1234")
        self.assertEqual(kwargs["replicate_targets"], ("lesion", "instrument"))
        self.assertEqual(
            kwargs["replicate_instructions"],
            {"lesion": "find lesion", "instrument": "find instrument"},
        )


def test_command_fairness_invokes_analyze_with_expected_arguments(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    masks_dir = run_dir / "masks"
    masks_dir.mkdir()
    mask_path = masks_dir / "img1.png"
    mask_path.write_text("mask")

    metrics_path = run_dir / "metrics.csv"
    metrics_path.write_text("image_name,iou,dice,success\nimg1.png,0.5,0.6,True\n")

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest_path = dataset_root / "manifest.txt"
    manifest_path.write_text("img1.png\n")

    expected_pairs = [("img1.png", mask_path)]

    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "discover_dataset",
        lambda *_: SimpleNamespace(manifest_path=manifest_path, masks_dir=masks_dir),
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"], "read_manifest", lambda *_, **__: ["img1.png"]
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "sample_images",
        lambda images, sample_size: images,
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "paired_masks",
        lambda images, masks_dir_arg: expected_pairs,
    )

    analyze_mock = mock.Mock(return_value=("results", "summaries", "stats"))
    monkeypatch.setattr(sys.modules["gemini_segmentation.cli"], "analyze_fairness", analyze_mock)
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "write_fairness_results",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "write_fairness_summary",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "write_fairness_statistics",
        lambda *_, **__: None,
    )

    args = argparse.Namespace(
        dataset_name="dataset",
        dataset_root=str(dataset_root),
        run_dir=str(run_dir),
        manifest=None,
        bootstrap_method="abc",
        bootstrap_resamples=10,
        sample_size=None,
        success_threshold=0.7,
    )

    command_fairness(args)

    assert analyze_mock.call_count == 1
    called_kwargs = analyze_mock.call_args.kwargs
    assert called_kwargs["image_mask_pairs"] == expected_pairs
    assert called_kwargs["prediction_masks_dir"] == masks_dir
    assert called_kwargs["n_resamples"] == 10
    assert called_kwargs["method"] == "abc"
    assert called_kwargs["per_image_metrics"] == {"img1.png": (0.5, 0.6, True)}


def test_command_fairness_raises_when_metrics_missing(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest_path = dataset_root / "manifest.txt"
    manifest_path.write_text("")

    masks_dir = run_dir / "masks"
    masks_dir.mkdir()

    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "discover_dataset",
        lambda *_: SimpleNamespace(manifest_path=manifest_path, masks_dir=masks_dir),
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"], "read_manifest", lambda *_, **__: []
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "sample_images",
        lambda images, sample_size: images,
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"], "paired_masks", lambda *_, **__: []
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"], "analyze_fairness", mock.Mock()
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "write_fairness_results",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "write_fairness_summary",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        sys.modules["gemini_segmentation.cli"],
        "write_fairness_statistics",
        lambda *_, **__: None,
    )

    args = argparse.Namespace(
        dataset_name="dataset",
        dataset_root=str(dataset_root),
        run_dir=str(run_dir),
        manifest=None,
        bootstrap_method=None,
        bootstrap_resamples=None,
        sample_size=None,
        success_threshold=0.7,
    )

    with pytest.raises(FileNotFoundError):
        command_fairness(args)


if __name__ == "__main__":
    import unittest

    unittest.main()
