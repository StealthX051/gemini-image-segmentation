from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .config import build_run_config, dump_run_config, load_preset, resolve_preset_name
from .prompts import ProviderPrompt, PromptFamily, build_prompt_for_provider
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
from .cache import DiskRequestCache, build_request_cache_key
from .io import (
    encode_mask_to_b64,
    load_existing_predictions,
    plot_segmentation_masks,
    save_mask_png,
    segmentation_masks_from_items,
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
from .models import GeminiSegmenter, MoondreamSegmenter, Sa2VAReplicateSegmenter
from .types import PerImageMetrics, SegmentationMask


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _prompt_hash(prompt_payload: Dict[str, object]) -> str:
    serialized = json.dumps(prompt_payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


def _prompt_key(prompt_family: str | None = None, prompt_hash: str | None = None) -> str:
    digest = prompt_hash or ""
    if prompt_family:
        return f"{prompt_family}-{digest}"
    return f"prompt-{digest}"


def _safe_model_dir_name(model_name: str) -> str:
    token = model_name.strip().replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    token = re.sub(r"[^A-Za-z0-9_.-]", "_", token)
    return token or "model"


def _prompt_payload(provider: str, prompt_family: str | None, prompt: ProviderPrompt) -> Dict[str, object]:
    payload: Dict[str, object] = {"provider": provider, "prompt": prompt.prompt}
    if prompt_family:
        payload["family"] = prompt_family
    if prompt.targets:
        payload["targets"] = list(prompt.targets)
    if prompt.instructions:
        payload["instructions"] = dict(prompt.instructions)
    return payload


def _resolve_provider_prompt(
    *,
    provider: str,
    prompt_family: str | None,
    explicit_prompt: str | None,
    prompt_task: str,
    target_overrides: List[str] | None,
    replicate_instruction_overrides: Dict[str, str] | None,
) -> ProviderPrompt:
    if prompt_family:
        base_prompt = build_prompt_for_provider(
            prompt_task,
            prompt_family,
            provider,
            targets_override=target_overrides,
        )
    else:
        if provider == "gemini":
            base_prompt = ProviderPrompt(prompt=explicit_prompt or "")
        elif provider == "moondream":
            targets = tuple(target_overrides) if target_overrides else ()
            if not targets and explicit_prompt:
                targets = (explicit_prompt,)
            primary = targets[0] if targets else explicit_prompt or ""
            base_prompt = ProviderPrompt(prompt=primary, targets=targets or None)
        elif provider == "replicate":
            targets = tuple(target_overrides) if target_overrides else ()
            if not targets and explicit_prompt:
                targets = (explicit_prompt,)
            instructions: Dict[str, str] = {}
            if explicit_prompt and targets:
                instructions = {t: explicit_prompt for t in targets}
            primary = explicit_prompt or ""
            base_prompt = ProviderPrompt(prompt=primary, targets=targets or None, instructions=instructions)
        else:
            base_prompt = ProviderPrompt(prompt=explicit_prompt or "")

    if provider == "moondream":
        targets = tuple(target_overrides) if target_overrides else base_prompt.targets
        primary = targets[0] if targets else base_prompt.prompt
        return ProviderPrompt(prompt=primary, targets=targets)

    if provider == "replicate":
        targets = tuple(target_overrides) if target_overrides else base_prompt.targets or ()
        instructions = dict(base_prompt.instructions or {})
        if replicate_instruction_overrides:
            instructions.update(replicate_instruction_overrides)
        if targets:
            for target in targets:
                instructions.setdefault(target, f"Segment the {target}.")
        primary_instruction = base_prompt.prompt
        if targets:
            primary_instruction = instructions.get(targets[0], base_prompt.prompt)
        return ProviderPrompt(
            prompt=primary_instruction,
            targets=targets or None,
            instructions=instructions or None,
        )

    return base_prompt


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


def _prepare_output_dirs(
    base_results: Path, dataset: str, model: str, prompt_key: str, run_id: str
) -> Dict[str, Path]:
    run_dir = base_results / dataset / model / prompt_key / run_id
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


def _process_image_with_cache(
    segmenter_factory: Callable[[], SegmenterProtocol],
    img_path: Path,
    *,
    provider: str,
    model_name: str,
    prompt_hash: str,
    prompt_family: str | None,
    temperature: float | None,
    thinking_budget: int | None,
    limiter: RateLimiter | None = None,
    request_cache: Optional[DiskRequestCache] = None,
    targets: tuple[str, ...] | None = None,
    max_retries: int = 5,
):
    cache_key: str | None = None
    if request_cache:
        cache_key = build_request_cache_key(
            image_path=img_path,
            provider=provider,
            model_name=model_name,
            prompt_hash=prompt_hash,
            prompt_family=prompt_family,
            temperature=temperature,
            thinking_budget=thinking_budget,
            targets=targets,
        )
        cached_payload = request_cache.load(cache_key)
        if cached_payload:
            image = load_image(img_path)
            raw_items = cached_payload.get("raw_items") or []
            if isinstance(raw_items, list):
                masks = segmentation_masks_from_items(
                    raw_items, img_height=image.height, img_width=image.width
                )
                latency = float(cached_payload.get("latency_s", 0.0))
                parse_success = bool(cached_payload.get("parse_success", bool(masks)))
                timed_out = bool(cached_payload.get("timed_out", False))
                return img_path.name, masks, latency, parse_success, timed_out, raw_items, True

    image = load_image(img_path)
    retries = max(0, int(max_retries))
    attempt = 0
    masks: List[SegmentationMask] = []
    latency = 0.0
    parse_success = False
    timed_out = False
    raw_items: List[dict] = []

    while attempt <= retries:
        attempt += 1
        if limiter:
            limiter.wait()
        segmenter = segmenter_factory()
        try:
            masks, latency, parse_success, timed_out, raw_items = segmenter.segment(image)
        except Exception:
            if attempt <= retries:
                logging.warning(
                    "Segmentation call failed for %s (attempt %s/%s); retrying.",
                    img_path.name,
                    attempt,
                    retries + 1,
                )
                continue
            raise

        if (timed_out or not parse_success) and attempt <= retries:
            logging.warning(
                "Segmentation call returned timeout/parse failure for %s (attempt %s/%s); retrying.",
                img_path.name,
                attempt,
                retries + 1,
            )
            continue
        break

    if request_cache and cache_key and not timed_out and parse_success and isinstance(raw_items, list):
        request_cache.save(
            cache_key,
            {
                "provider": provider,
                "model_name": model_name,
                "prompt_hash": prompt_hash,
                "prompt_family": prompt_family,
                "latency_s": float(latency),
                "parse_success": bool(parse_success),
                "timed_out": bool(timed_out),
                "raw_items": raw_items,
            },
        )
    return img_path.name, masks, latency, parse_success, timed_out, raw_items, False


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

    prompt_text = args.prompt
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text()
    prompt_families: List[str] = []
    if args.prompt_family:
        if isinstance(args.prompt_family, str):
            prompt_families = [args.prompt_family]
        else:
            prompt_families = list(args.prompt_family)
    prompt_task = args.dataset_name

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
        preset_family = preset_cfg.get("prompt_family")
        prompt_task = preset_cfg.get("prompt_task", args.dataset_name)
        if preset_family and not prompt_families:
            if isinstance(preset_family, (list, tuple)):
                prompt_families.extend([str(fam) for fam in preset_family])
            else:
                prompt_families.append(str(preset_family))
        if not prompt_families:
            prompt_text = preset_cfg.get("prompt_text", prompt_text)
        if preset_cfg.get("model"):
            args.model_name = preset_cfg["model"]
        if preset_cfg.get("temperature") is not None:
            args.temperature = float(preset_cfg["temperature"])
        if preset_cfg.get("thinking_budget") is not None:
            args.thinking_budget = int(preset_cfg["thinking_budget"])

    prompt_runs: List[tuple[str | None, str | None]] = []
    seen_keys = set()
    if prompt_families:
        for family in prompt_families:
            prompt_key = _prompt_key(prompt_family=family, prompt_hash=family)
            if prompt_key in seen_keys:
                continue
            seen_keys.add(prompt_key)
            prompt_runs.append((family, None))
    else:
        prompt_runs.append((None, prompt_text))

    run_id = args.run_id or _default_run_id()
    base_model_name = args.model_name
    base_provider = args.provider
    base_replicate_model_version = args.replicate_model_version
    base_replicate_targets = args.replicate_targets
    base_replicate_instructions = args.replicate_instructions

    rate_limiter = RateLimiter(args.rate_limit) if args.rate_limit else None
    local_cache_enabled = bool(getattr(args, "local_cache", True))
    local_cache_dir_arg = getattr(args, "local_cache_dir", None)
    local_cache_dir = (
        Path(local_cache_dir_arg)
        if local_cache_dir_arg
        else Path(args.results_dir) / ".request_cache"
    ).expanduser().resolve()
    gemini_explicit_cache = bool(getattr(args, "gemini_explicit_cache", True))
    gemini_cache_ttl = int(getattr(args, "gemini_cache_ttl", 3600))
    base_local_cache_enabled = local_cache_enabled
    base_local_cache_dir = local_cache_dir
    base_gemini_explicit_cache = gemini_explicit_cache
    base_gemini_cache_ttl = gemini_cache_ttl
    base_max_retries = int(getattr(args, "max_retries", 5))

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    for prompt_family, explicit_prompt in prompt_runs:
        local_cache_enabled = base_local_cache_enabled
        local_cache_dir = base_local_cache_dir
        request_cache = DiskRequestCache(local_cache_dir) if local_cache_enabled else None
        gemini_explicit_cache = base_gemini_explicit_cache
        gemini_cache_ttl = base_gemini_cache_ttl
        max_retries = base_max_retries
        model_name = base_model_name
        run_provider = base_provider
        replicate_model_version = base_replicate_model_version
        replicate_targets_arg = base_replicate_targets
        replicate_instructions_arg = base_replicate_instructions

        provider_prompt = _resolve_provider_prompt(
            provider=run_provider,
            prompt_family=prompt_family,
            explicit_prompt=explicit_prompt,
            prompt_task=prompt_task,
            target_overrides=None,
            replicate_instruction_overrides=None,
        )
        prompt_payload = _prompt_payload(run_provider, prompt_family, provider_prompt)
        prompt_hash = _prompt_hash(prompt_payload)
        prompt_key = _prompt_key(prompt_family, prompt_hash)

        model_label = (
            (replicate_model_version or model_name)
            if run_provider == "replicate"
            else model_name
        )
        model_dir_name = _safe_model_dir_name(model_label)
        paths = _prepare_output_dirs(
            Path(args.results_dir), args.dataset_name, model_dir_name, prompt_key, run_id
        )

        existing_run_config = {}
        if paths["run_config"].exists():
            try:
                existing_run_config = json.loads(paths["run_config"].read_text())
            except json.JSONDecodeError:
                logging.warning(
                    "Failed to parse existing run_config.json; proceeding with CLI arguments"
                )

        run_provider = existing_run_config.get("provider", run_provider)
        gemini_explicit_cache = bool(
            existing_run_config.get("gemini_explicit_cache", gemini_explicit_cache)
        )
        gemini_cache_ttl = int(
            existing_run_config.get("gemini_cache_ttl_s", gemini_cache_ttl)
        )
        local_cache_enabled = bool(
            existing_run_config.get("local_cache_enabled", local_cache_enabled)
        )
        max_retries = int(existing_run_config.get("max_retries", max_retries))
        persisted_cache_dir = existing_run_config.get("local_cache_dir")
        if persisted_cache_dir:
            local_cache_dir = Path(persisted_cache_dir).expanduser().resolve()
        request_cache = DiskRequestCache(local_cache_dir) if local_cache_enabled else None

        if run_provider == "moondream" and model_name == "gemini-2.5-flash":
            model_name = "moondream-3"

        replicate_model_version = replicate_model_version or existing_run_config.get(
            "replicate_model_version"
        )
        replicate_targets_arg = replicate_targets_arg or existing_run_config.get(
            "replicate_targets"
        )
        replicate_instructions_arg = replicate_instructions_arg or existing_run_config.get(
            "replicate_instructions"
        )
        replicate_cache_dir = (
            Path(args.replicate_cache_dir).expanduser().resolve()
            if args.replicate_cache_dir
            else Path(existing_run_config["replicate_cache_dir"]).expanduser().resolve()
            if existing_run_config.get("replicate_cache_dir")
            else None
        )

        replicate_instruction_sequence = None
        replicate_instruction_map = None
        if isinstance(replicate_instructions_arg, dict):
            replicate_instruction_map = {
                str(target): str(instruction)
                for target, instruction in replicate_instructions_arg.items()
            }
        elif replicate_instructions_arg:
            replicate_instruction_sequence = [str(instruction) for instruction in replicate_instructions_arg]

        replicate_target_instructions = None
        if replicate_instruction_map:
            replicate_target_instructions = dict(replicate_instruction_map)
        elif replicate_instruction_sequence and replicate_targets_arg:
            replicate_target_instructions = {
                target: instruction
                for target, instruction in zip(replicate_targets_arg, replicate_instruction_sequence)
            }

        if run_provider == "replicate":
            if replicate_instruction_sequence and (
                not replicate_targets_arg
                or len(replicate_instruction_sequence) != len(replicate_targets_arg)
            ):
                raise ValueError(
                    "The number of --replicate-instruction flags must match --replicate-target entries."
                )
            if not replicate_model_version:
                raise ValueError(
                    "--replicate-model-version is required when provider is 'replicate'"
                )

        moondream_targets = (
            args.moondream_targets or existing_run_config.get("moondream_targets") or None
        )

        provider_prompt = _resolve_provider_prompt(
            provider=run_provider,
            prompt_family=prompt_family,
            explicit_prompt=explicit_prompt,
            prompt_task=prompt_task,
            target_overrides=moondream_targets
            if run_provider == "moondream"
            else replicate_targets_arg,
            replicate_instruction_overrides=replicate_target_instructions,
        )
        if run_provider == "moondream" and not moondream_targets:
            moondream_targets = list(provider_prompt.targets or []) or None
        if run_provider == "replicate":
            replicate_targets_arg = replicate_targets_arg or list(provider_prompt.targets or []) or None
            replicate_target_instructions = (
                dict(provider_prompt.instructions)
                if provider_prompt.instructions
                else replicate_target_instructions
            )

        prompt_payload = _prompt_payload(run_provider, prompt_family, provider_prompt)
        prompt_hash = _prompt_hash(prompt_payload)
        request_prompt_digest = hashlib.sha256(
            json.dumps(prompt_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        prompt_key = _prompt_key(prompt_family, prompt_hash)

        model_label = (
            (replicate_model_version or model_name)
            if run_provider == "replicate"
            else model_name
        )
        model_dir_name = _safe_model_dir_name(model_label)
        current_model_label = paths["run_dir"].parts[-3]
        current_prompt_key = paths["run_dir"].parts[-2]
        if model_dir_name != current_model_label or prompt_key != current_prompt_key:
            paths = _prepare_output_dirs(
                Path(args.results_dir), args.dataset_name, model_dir_name, prompt_key, run_id
            )

        resolved_prompt = provider_prompt.prompt

        config = build_run_config(
            dataset_name=args.dataset_name,
            dataset_root=dataset_root,
            prompt=resolved_prompt,
            prompt_family=prompt_family,
            prompt_hash=prompt_hash,
            model_name=model_label,
            provider=run_provider,
            thinking_budget=args.thinking_budget,
            temperature=args.temperature,
            timeout_s=args.timeout,
            max_retries=max_retries,
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
            replicate_model_version=replicate_model_version,
            replicate_targets=tuple(replicate_targets_arg) if replicate_targets_arg else None,
            replicate_instructions=replicate_target_instructions,
            replicate_cache_dir=replicate_cache_dir,
            local_cache_enabled=local_cache_enabled,
            local_cache_dir=local_cache_dir if local_cache_enabled else None,
            gemini_explicit_cache=gemini_explicit_cache,
            gemini_cache_ttl_s=gemini_cache_ttl,
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
            logging.info(
                "Dry run (%s): %s images need processing", prompt_key, len(pending_pairs)
            )
            for img_path, _ in pending_pairs:
                logging.info("Pending: %s", img_path.name)
            continue

        thread_local = threading.local()

        def get_segmenter() -> SegmenterProtocol:
            if not hasattr(thread_local, "segmenter"):
                if run_provider == "moondream":
                    thread_local.segmenter = MoondreamSegmenter(
                        model_name=model_name,
                        prompt=resolved_prompt,
                        timeout_s=args.timeout,
                        targets=moondream_targets,
                        endpoint=args.moondream_endpoint,
                        api_key=args.moondream_api_key,
                    )
                elif run_provider == "replicate":
                    replicate_model_name = replicate_model_version or model_name
                    thread_local.segmenter = Sa2VAReplicateSegmenter(
                        model_name=replicate_model_name,
                        model_version=replicate_model_version or model_name,
                        instruction=resolved_prompt,
                        timeout_s=args.timeout,
                        targets=replicate_targets_arg,
                        instructions=replicate_target_instructions,
                        cache_dir=replicate_cache_dir,
                    )
                else:
                    thread_local.segmenter = GeminiSegmenter(
                        model_name=model_name,
                        prompt=resolved_prompt,
                        temperature=args.temperature,
                        thinking_budget=args.thinking_budget,
                        timeout_s=args.timeout,
                        explicit_cache=gemini_explicit_cache,
                        cache_ttl_s=gemini_cache_ttl,
                    )
            return thread_local.segmenter

        cache_hits = 0
        cache_misses = 0
        cache_targets = tuple(provider_prompt.targets) if provider_prompt.targets else None
        futures = {}
        total_pending = len(pending_pairs)
        heartbeat_s = 30.0
        if total_pending:
            logging.info(
                "Starting %s pending images for %s (provider=%s, model=%s, workers=%s, max_retries=%s, rate_limit_s=%s)",
                total_pending,
                prompt_key,
                run_provider,
                model_label,
                args.workers,
                max_retries,
                args.rate_limit,
            )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for img_path, gt_mask_path in pending_pairs:
                futures[
                    executor.submit(
                        lambda p=img_path: _process_image_with_cache(
                            get_segmenter,
                            p,
                            provider=run_provider,
                            model_name=model_label,
                            prompt_hash=request_prompt_digest,
                            prompt_family=prompt_family,
                            temperature=args.temperature if run_provider == "gemini" else None,
                            thinking_budget=args.thinking_budget if run_provider == "gemini" else None,
                            limiter=rate_limiter,
                            request_cache=request_cache,
                            targets=cache_targets,
                            max_retries=max_retries,
                        )
                    )
                ] = (img_path, gt_mask_path)

            completed = 0
            pending_futures = set(futures.keys())
            while pending_futures:
                done, pending_futures = wait(
                    pending_futures, timeout=heartbeat_s, return_when=FIRST_COMPLETED
                )
                if not done:
                    logging.info(
                        "Progress heartbeat for %s: completed=%s/%s, remaining=%s, cache_hits=%s, cache_misses=%s",
                        prompt_key,
                        completed,
                        total_pending,
                        len(pending_futures),
                        cache_hits,
                        cache_misses,
                    )
                    continue

                for future in done:
                    img_path, gt_mask_path = futures[future]
                    try:
                        img_name, masks, latency, parse_success, timed_out, raw_items, from_cache = future.result()
                    except Exception:
                        logging.exception("Worker failed while processing image %s", img_path.name)
                        raise

                    if from_cache:
                        cache_hits += 1
                    else:
                        cache_misses += 1
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
                    raw_response_payload = raw_items or None
                    if args.legacy_predictions:
                        legacy_dir = dataset_root / f"predictions_{model_dir_name}"
                        legacy_json_path = legacy_dir / f"{img_name}.json"
                        _write_legacy_prediction(mask_arrays, legacy_json_path, raw_response_payload)

                    raw_response_path = None
                    if raw_response_payload is not None:
                        raw_response_path = paths["raw_responses"] / f"{img_name}.json"
                        raw_response_path.write_text(json.dumps(raw_response_payload))

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
                        "raw_response_path": str(raw_response_path) if raw_response_path else None,
                        "provider": run_provider,
                        "prompt_family": prompt_family,
                    }

                    write_prediction_jsonl(predictions.values(), paths["predictions_jsonl"], mode="w")
                    completed += 1
                    logging.info(
                        "Processed [%s/%s] %s (cache=%s, parse_success=%s, timed_out=%s, masks=%s, latency_s=%.3f, iou=%.4f, dice=%.4f)",
                        completed,
                        total_pending,
                        img_name,
                        "hit" if from_cache else "miss",
                        parse_success,
                        timed_out,
                        len(mask_arrays),
                        latency,
                        iou,
                        dice,
                    )

        if metrics_map:
            summary = aggregate_from_map(
                metrics_map, n_resamples=args.bootstrap_resamples, method=args.bootstrap_method
            )
            write_summary(summary, paths["summary"])

        logging.info(
            "Segmentation run complete for %s. Outputs at %s (cache hits=%s, cache misses=%s)",
            prompt_key,
            paths["run_dir"],
            cache_hits,
            cache_misses,
        )


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
    workers = max(1, int(getattr(args, "workers", 1)))
    logging.info(
        "Starting fairness analysis for %s image/mask pairs with workers=%s",
        len(image_mask_pairs),
        workers,
    )

    metrics_path = Path(args.run_dir) / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Per-image metrics not found at {metrics_path}")
    per_image_df = pd.read_csv(metrics_path)
    per_image_metrics: Dict[str, Tuple[float, float, bool]] = {
        row.image_name: (float(row.iou), float(row.dice), bool(row.success)) for _, row in per_image_df.iterrows()
    }

    prediction_masks_dir = Path(args.run_dir) / "masks"
    progress_interval = max(1, len(image_mask_pairs) // 20)
    progress_state = {"last_logged": 0}

    def _on_progress(done: int, total: int) -> None:
        if done == total or (done - progress_state["last_logged"]) >= progress_interval:
            logging.info("Fairness progress: %s/%s image pairs evaluated", done, total)
            progress_state["last_logged"] = done

    results, summaries, stats_payload = analyze_fairness(
        image_mask_pairs=image_mask_pairs,
        prediction_masks_dir=prediction_masks_dir,
        per_image_metrics=per_image_metrics,
        success_threshold=args.success_threshold,
        n_resamples=int(bootstrap_resamples),
        method=str(bootstrap_method),
        workers=workers,
        progress_callback=_on_progress,
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
        choices=["gemini", "moondream", "replicate"],
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
        choices=["legacy"],
        help="Optional branch suffix (e.g., 'legacy' selects the base preset)",
    )
    seg.add_argument(
        "--prompt-family",
        action="append",
        choices=[p.value for p in PromptFamily],
        help=(
            "Select one or more structured prompt families (repeat flag to evaluate multiple;"
            " overrides prompt_preset family if provided)"
        ),
    )
    seg.add_argument("--thinking-budget", type=int, default=0, help="Thinking budget tokens")
    seg.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    seg.add_argument("--timeout", type=float, default=60.0, help="Client-side timeout per image")
    seg.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Number of retries after the first attempt when timeout/parse failure occurs",
    )
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
    seg.add_argument("--replicate-model-version", help="Replicate model version to call")
    seg.add_argument(
        "--replicate-target",
        action="append",
        dest="replicate_targets",
        help="One or more labels to segment with Replicate (repeat per label)",
    )
    seg.add_argument(
        "--replicate-instruction",
        action="append",
        dest="replicate_instructions",
        help="Instruction text aligned with each --replicate-target",
    )
    seg.add_argument(
        "--replicate-cache-dir",
        help="Optional cache directory for Replicate assets (will be expanded)",
    )
    seg.add_argument(
        "--local-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable local request/response cache across runs",
    )
    seg.add_argument(
        "--local-cache-dir",
        help="Directory for local request/response cache records (default: <results-dir>/.request_cache)",
    )
    seg.add_argument(
        "--gemini-explicit-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Gemini API explicit context caching when supported",
    )
    seg.add_argument(
        "--gemini-cache-ttl",
        type=int,
        default=3600,
        help="Gemini explicit cache TTL in seconds",
    )
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
    fair.add_argument("--workers", type=int, default=1, help="Number of worker threads for fairness preprocessing")
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
