from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Protocol, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .config import build_run_config, dump_run_config, load_preset, resolve_preset_name
from .data import (
    DEFAULT_MANIFEST_TEMPLATE,
    discover_dataset,
    load_image,
    paired_masks,
    read_manifest,
    sample_images,
)
from .fairness import (
    analyze_fairness,
    write_fairness_results,
    write_fairness_statistics,
    write_fairness_summary,
)
from .io import (
    encode_mask_to_b64,
    load_existing_predictions,
    plot_segmentation_masks,
    save_mask_png,
    write_prediction_jsonl,
)
from .metrics import (
    aggregate_from_map,
    combine_masks,
    compute_metrics_for_masks,
    load_metrics,
    upsert_metrics,
    write_summary,
)
from .models import GeminiSegmenter, MoondreamSegmenter
from .types import PerImageMetrics, SegmentationMask


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class RateLimiter:
    """Process-wide rate limiter shared across workers."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def wait(self) -> None:
        if self.min_interval_s is None:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
            self._next_allowed = time.monotonic() + self.min_interval_s


def _prepare_output_dirs(base_results: Path, dataset: str, model: str, run_id: str) -> Dict[str, Path]:
    run_dir = base_results / dataset / model / run_id
    paths = {
        "run_dir": run_dir,
        "predictions_jsonl": run_dir / "predictions.jsonl",
        "masks": run_dir / "masks",
        "overlays": run_dir / "overlays",
        "metrics": run_dir / "metrics.csv",
        "summary": run_dir / "summary.csv",
        "fairness": run_dir / "fairness",
        "run_config": run_dir / "run_config.json",
        "raw_responses": run_dir / "raw_responses",
    }
    for key, path in paths.items():
        if key == "run_dir":
            path.mkdir(parents=True, exist_ok=True)
        elif key in {"masks", "overlays", "fairness", "raw_responses"}:
            path.mkdir(parents=True, exist_ok=True)
        elif key != "run_config":
            path.parent.mkdir(parents=True, exist_ok=True)
    return paths


class SegmenterProtocol(Protocol):
    def segment(self, image_obj: Image.Image) -> Tuple[List[SegmentationMask], float, bool, bool, List[dict]]:
        ...


def _process_image(segmenter: SegmenterProtocol, img_path: Path, limiter: RateLimiter | None = None):
    if limiter:
        limiter.wait()
    image = load_image(img_path)
    masks, latency, parse_success, timed_out, raw_items = segmenter.segment(image)
    return img_path.name, masks, latency, parse_success, timed_out, raw_items


def _write_legacy_prediction(
    segmentation_masks: List[np.ndarray], path: Path, raw_items: List[dict] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_items:
        path.write_text(json.dumps(raw_items))
        return
    if not segmentation_masks:
        path.write_text(json.dumps([]))
        return
    payload = []
    for mask in segmentation_masks:
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            continue
        y0, x0 = coords.min(axis=0)[:2]
        y1, x1 = coords.max(axis=0)[:2]
        payload.append(
            {
                "mask": encode_mask_to_b64(mask),
                "box_2d": [int(y0 / mask.shape[0] * 1000), int(x0 / mask.shape[1] * 1000), int(y1 / mask.shape[0] * 1000), int(x1 / mask.shape[1] * 1000)],
                "label": "",
            }
        )
    path.write_text(json.dumps(payload))


def command_segment(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    manifest_name = args.manifest
    dataset_paths = discover_dataset(dataset_root, args.dataset_name, manifest_name)
    default_manifest = dataset_root / DEFAULT_MANIFEST_TEMPLATE.format(dataset=args.dataset_name)
    fallbacks = [] if manifest_name is None else [default_manifest]
    manifest_images = read_manifest(
        dataset_paths, fallbacks=fallbacks, regenerate_if_missing=manifest_name is None
    )
    images = sample_images(manifest_images, args.sample_size)
    image_mask_pairs = paired_masks(images, dataset_paths.masks_dir)

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
    if args.prompt_preset:
        resolved_preset_name = resolve_preset_name(args.preset_name, args.preset_branch)
        try:
            preset_cfg = load_preset(Path(args.prompt_preset), resolved_preset_name)
        except KeyError as exc:
            if args.preset_branch:
                raise KeyError(
                    f"Preset '{resolved_preset_name}' not found for branch '{args.preset_branch}'"
                ) from exc
            raise
        prompt = preset_cfg.get("prompt_text", prompt)
        if preset_cfg.get("model"):
            args.model_name = preset_cfg["model"]
        if preset_cfg.get("temperature") is not None:
            args.temperature = float(preset_cfg["temperature"])
        if preset_cfg.get("thinking_budget") is not None:
            args.thinking_budget = int(preset_cfg["thinking_budget"])

    if args.provider == "moondream" and args.model_name == "gemini-2.5-flash":
        args.model_name = "moondream-3"

    moondream_targets = args.moondream_targets or None

    run_id = args.run_id or _default_run_id()
    paths = _prepare_output_dirs(Path(args.results_dir), args.dataset_name, args.model_name, run_id)

    config = build_run_config(
        dataset_name=args.dataset_name,
        dataset_root=dataset_root,
        prompt=prompt,
        model_name=args.model_name,
        provider=args.provider,
        thinking_budget=args.thinking_budget,
        temperature=args.temperature,
        timeout_s=args.timeout,
        workers=args.workers,
        sample_size=args.sample_size,
        manifest_path=dataset_paths.manifest_path,
        rate_limit_s=args.rate_limit,
        legacy_predictions=args.legacy_predictions,
        run_id=run_id,
        bootstrap_method=args.bootstrap_method,
        bootstrap_resamples=args.bootstrap_resamples,
        moondream_targets=moondream_targets,
        moondream_endpoint=args.moondream_endpoint,
    )
    dump_run_config(config, paths["run_config"])

    predictions = load_existing_predictions(paths["predictions_jsonl"])
    metrics_map = load_metrics(paths["metrics"])
    gt_lookup = {img.name: gt for img, gt in image_mask_pairs}

    missing_artifacts = set()
    for name, record in predictions.items():
        if record.prediction_path and record.prediction_path.exists():
            if name not in metrics_map and name in gt_lookup:
                gt_array = np.array(Image.open(gt_lookup[name]))
                pred_array = np.array(Image.open(record.prediction_path))
                iou, dice, success = compute_metrics_for_masks(
                    gt_array, [pred_array], success_threshold=args.success_threshold
                )
                metrics_map = upsert_metrics(
                    metrics_map,
                    PerImageMetrics(image_name=name, iou=iou, dice=dice, success=success),
                    metrics_path=paths["metrics"],
                    summary_path=paths["summary"],
                    n_resamples=args.bootstrap_resamples,
                    method=args.bootstrap_method,
                )
        else:
            missing_artifacts.add(name)
        if record.overlay_path and not record.overlay_path.exists():
            missing_artifacts.add(name)

    pending_pairs: List[Tuple[Path, Path]] = []
    for img_path, gt_mask_path in image_mask_pairs:
        if img_path.name in missing_artifacts or img_path.name not in predictions:
            pending_pairs.append((img_path, gt_mask_path))

    if args.dry_run:
        logging.info("Dry run: %s images need processing", len(pending_pairs))
        for img_path, _ in pending_pairs:
            logging.info("Pending: %s", img_path.name)
        return

    rate_limiter = RateLimiter(args.rate_limit) if args.rate_limit else None

    from concurrent.futures import ThreadPoolExecutor, as_completed

    thread_local = threading.local()

    def get_segmenter() -> SegmenterProtocol:
        if not hasattr(thread_local, "segmenter"):
            if args.provider == "moondream":
                thread_local.segmenter = MoondreamSegmenter(
                    model_name=args.model_name,
                    prompt=prompt,
                    timeout_s=args.timeout,
                    targets=moondream_targets,
                    endpoint=args.moondream_endpoint,
                    api_key=args.moondream_api_key,
                )
            else:
                thread_local.segmenter = GeminiSegmenter(
                    model_name=args.model_name,
                    prompt=prompt,
                    temperature=args.temperature,
                    thinking_budget=args.thinking_budget,
                    timeout_s=args.timeout,
                )
        return thread_local.segmenter

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for img_path, gt_mask_path in pending_pairs:
            futures[executor.submit(lambda p=img_path: _process_image(get_segmenter(), p, rate_limiter))] = (
                img_path,
                gt_mask_path,
            )

        for future in as_completed(futures):
            img_path, gt_mask_path = futures[future]
            img_name, masks, latency, parse_success, timed_out, raw_items = future.result()
            mask_arrays = [m.mask for m in masks]
            mask_save_path = paths["masks"] / img_name
            gt_array = np.array(Image.open(gt_mask_path))
            combined_mask = combine_masks(mask_arrays) if mask_arrays else np.zeros_like(gt_array)
            save_mask_png(combined_mask, mask_save_path)

            overlay_path = paths["overlays"] / img_name
            overlay = plot_segmentation_masks(Image.open(img_path), masks)
            overlay.save(overlay_path)

            iou, dice, success = compute_metrics_for_masks(
                gt_array, mask_arrays, success_threshold=args.success_threshold
            )
            metrics_map = upsert_metrics(
                metrics_map,
                PerImageMetrics(image_name=img_name, iou=iou, dice=dice, success=success),
                metrics_path=paths["metrics"],
                summary_path=paths["summary"],
                n_resamples=args.bootstrap_resamples,
                method=args.bootstrap_method,
            )

            legacy_json_path = None
            if args.legacy_predictions:
                legacy_dir = dataset_root / f"predictions_{args.model_name}"
                legacy_json_path = legacy_dir / f"{img_name}.json"
                _write_legacy_prediction(mask_arrays, legacy_json_path, raw_items)

            raw_response_path = paths["raw_responses"] / f"{img_name}.json"
            raw_response_path.write_text(json.dumps(raw_items if raw_items is not None else []))

            predictions[img_name] = {
                "image_name": img_name,
                "latency_s": latency,
                "parse_success": parse_success,
                "timed_out": timed_out,
                "num_masks": len(mask_arrays),
                "prediction_path": str(mask_save_path),
                "overlay_path": str(overlay_path),
                "metrics": {"iou": iou, "dice": dice, "success": success},
                "legacy_json_path": str(legacy_json_path) if legacy_json_path else None,
                "raw_response_path": str(raw_response_path),
            }

            write_prediction_jsonl(predictions.values(), paths["predictions_jsonl"], mode="w")

    if metrics_map:
        summary = aggregate_from_map(
            metrics_map, n_resamples=args.bootstrap_resamples, method=args.bootstrap_method
        )
        write_summary(summary, paths["summary"])

    logging.info("Segmentation run complete. Outputs at %s", paths["run_dir"])


def command_fairness(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    run_config_path = Path(args.run_dir) / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}

    manifest_name = args.manifest or run_config.get("manifest_path")
    bootstrap_method = args.bootstrap_method or run_config.get("bootstrap_method", "bca")
    bootstrap_resamples = args.bootstrap_resamples or run_config.get("bootstrap_resamples", 5000)

    dataset_paths = discover_dataset(dataset_root, args.dataset_name, manifest_name)
    default_manifest = dataset_root / DEFAULT_MANIFEST_TEMPLATE.format(dataset=args.dataset_name)
    fallbacks = [] if manifest_name is None else [default_manifest]
    manifest_images = read_manifest(
        dataset_paths, fallbacks=fallbacks, regenerate_if_missing=manifest_name is None
    )
    images = sample_images(manifest_images, args.sample_size)
    image_mask_pairs = paired_masks(images, dataset_paths.masks_dir)

    metrics_path = Path(args.run_dir) / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Per-image metrics not found at {metrics_path}")
    per_image_df = pd.read_csv(metrics_path)
    per_image_metrics: Dict[str, Tuple[float, float, bool]] = {
        row.image_name: (float(row.iou), float(row.dice), bool(row.success)) for _, row in per_image_df.iterrows()
    }

    prediction_masks_dir = Path(args.run_dir) / "masks"
    results, summaries, stats_payload = analyze_fairness(
        image_mask_pairs=image_mask_pairs,
        prediction_masks_dir=prediction_masks_dir,
        per_image_metrics=per_image_metrics,
        success_threshold=args.success_threshold,
        n_resamples=int(bootstrap_resamples),
        method=str(bootstrap_method),
    )

    fairness_dir = Path(args.run_dir) / "fairness"
    write_fairness_results(results, fairness_dir / "fairness_results.csv")
    write_fairness_summary(summaries, fairness_dir / "fairness_summary.csv")
    write_fairness_statistics(stats_payload, fairness_dir / "fairness_stats.csv")
    logging.info("Fairness analysis complete. Outputs at %s", fairness_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini segmentation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seg = subparsers.add_parser("segment", help="Run segmentation")
    seg.add_argument("dataset_name", help="Dataset name (used for manifests and output paths)")
    seg.add_argument("dataset_root", help="Path to dataset root containing images/ and masks/")
    seg.add_argument("--manifest", help="Optional manifest filename or path (e.g., pilot list)")
    seg.add_argument(
        "--provider",
        choices=["gemini", "moondream"],
        default="gemini",
        help="Segmentation backend to use",
    )
    seg.add_argument("--model-name", default="gemini-2.5-flash", help="Gemini model name")
    seg.add_argument("--prompt", default="", help="Prompt text to send")
    seg.add_argument("--prompt-file", help="Path to a prompt text file")
    seg.add_argument("--prompt-preset", help="YAML file containing prompt presets")
    seg.add_argument("--preset-name", default="default", help="Preset key to use when loading prompt-preset")
    seg.add_argument(
        "--preset-branch",
        choices=["legacy", "hybrid"],
        help="Optional branch suffix (e.g., 'hybrid' selects <preset_name>_hybrid)",
    )
    seg.add_argument("--thinking-budget", type=int, default=0, help="Thinking budget tokens")
    seg.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    seg.add_argument("--timeout", type=float, default=60.0, help="Client-side timeout per image")
    seg.add_argument("--workers", type=int, default=1, help="Number of worker threads")
    seg.add_argument("--sample-size", type=int, help="Limit number of images")
    seg.add_argument("--results-dir", default="results", help="Root directory for outputs")
    seg.add_argument("--run-id", help="Override run id (timestamp by default)")
    seg.add_argument("--rate-limit", type=float, help="Seconds to sleep between calls")
    seg.add_argument("--legacy-predictions", action="store_true", help="Also save predictions_<model>/ JSONs")
    seg.add_argument("--success-threshold", type=float, default=0.5, help="IoU success threshold")
    seg.add_argument(
        "--moondream-target",
        action="append",
        dest="moondream_targets",
        help="One or more object labels to segment with Moondream (one API call per target)",
    )
    seg.add_argument("--moondream-endpoint", help="Optional Moondream Station endpoint URL")
    seg.add_argument("--moondream-api-key", help="Moondream API key (defaults to MOONDREAM_API_KEY)")
    seg.add_argument(
        "--bootstrap-method",
        choices=["bca", "percentile"],
        default="bca",
        help="Bootstrap confidence interval method",
    )
    seg.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples for summary stats",
    )
    seg.add_argument("--dry-run", action="store_true", help="List pending images without calling the API")
    seg.set_defaults(func=command_segment)

    fair = subparsers.add_parser("fairness", help="Run ITA-based fairness analysis")
    fair.add_argument("dataset_name", help="Dataset name")
    fair.add_argument("dataset_root", help="Dataset root containing images/ and masks/")
    fair.add_argument("run_dir", help="Segmentation run directory under results/")
    fair.add_argument("--manifest", help="Optional manifest filename or path")
    fair.add_argument("--sample-size", type=int, help="Limit number of images")
    fair.add_argument("--success-threshold", type=float, default=0.5)
    fair.add_argument(
        "--bootstrap-method",
        choices=["bca", "percentile"],
        help="Bootstrap method (defaults to run config or bca)",
    )
    fair.add_argument(
        "--bootstrap-resamples",
        type=int,
        help="Number of bootstrap resamples (defaults to run config or 5000)",
    )
    fair.set_defaults(func=command_fairness)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
