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
    _prompt_hash,
    _resolve_provider_prompt,
    build_parser,
    command_fairness,
    command_segment,
)
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
