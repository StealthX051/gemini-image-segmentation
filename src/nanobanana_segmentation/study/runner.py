from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from nanobanana_segmentation.core.engine import NanoBananaClient, SegmentationEngine
from nanobanana_segmentation.core.logging.artifact_store import ArtifactStore
from nanobanana_segmentation.core.types import BudgetConfig, ConstraintConfig, EngineRequest
from nanobanana_segmentation.study.config import StudyConfig, load_study_config
from nanobanana_segmentation.study.dataset import load_dataset_items
from nanobanana_segmentation.study.eval import evaluate_segmentation
from nanobanana_segmentation.study.leakage import audit_retrieval
from nanobanana_segmentation.study.reports import build_reports


@dataclass(frozen=True)
class _RunTask:
    task_index: int
    image_path: Path
    mask_path: Path
    split: str
    tool_mode: str
    thinking_level: str
    replicate_idx: int


RESULT_COLUMNS: List[str] = [
    "run_id",
    "image_id",
    "image_name",
    "split",
    "tool_mode",
    "query_policy",
    "snapshot_policy",
    "scope_policy",
    "thinking_level",
    "replicate_idx",
    "iou",
    "dice",
    "precision",
    "recall",
    "qc_pass",
    "selected_attempt_index",
    "retrieval_duplicate",
    "retrieval_mask_source",
    "audit_unavailable",
    "analysis_primary",
    "analysis_sensitivity",
    "run_record_path",
    "mask_path",
]


def _stage_matrix(cfg: StudyConfig, stage: str) -> Tuple[List[str], List[str], int, int | None]:
    if stage == "stage0":
        return (
            list(cfg.matrix.tool_modes),
            ["minimal"],
            1,
            int(cfg.stages.stage0_sample_size),
        )
    if stage == "stage2":
        return (
            list(cfg.stages.stage2_tool_modes),
            list(cfg.stages.stage2_thinking_levels),
            int(cfg.stages.stage2_replicates),
            int(cfg.stages.stage2_sample_size),
        )
    return (
        list(cfg.matrix.tool_modes),
        list(cfg.matrix.thinking_levels),
        int(cfg.matrix.replicates),
        cfg.stages.stage1_sample_size,
    )


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _build_engine(cfg: StudyConfig) -> SegmentationEngine:
    artifact_store = ArtifactStore(Path(cfg.paths.artifacts_root).expanduser().resolve())
    client = NanoBananaClient(
        model_id=cfg.model.model_id,
        api_surface=cfg.model.api_surface,
        timeout_s=float(cfg.execution.client_timeout_s),
    )
    prompts_path = Path(cfg.execution.prompts_path).expanduser().resolve() if cfg.execution.prompts_path else None
    return SegmentationEngine(
        client=client,
        artifact_store=artifact_store,
        cost_per_attempt_usd=cfg.model.cost_per_attempt_usd,
        use_otsu_bw=cfg.model.use_otsu_bw,
        prompts_path=prompts_path,
        model_style=cfg.model.model_style,
    )


def _build_tasks(
    items,
    *,
    stage_tool_modes: List[str],
    stage_thinking_levels: List[str],
    stage_replicates: int,
) -> List[_RunTask]:
    tasks: List[_RunTask] = []
    task_index = 0
    for item in items:
        for tool_mode in stage_tool_modes:
            for thinking in stage_thinking_levels:
                for replicate_idx in range(stage_replicates):
                    tasks.append(
                        _RunTask(
                            task_index=task_index,
                            image_path=item.image_path,
                            mask_path=item.mask_path,
                            split=item.split,
                            tool_mode=tool_mode,
                            thinking_level=thinking,
                            replicate_idx=replicate_idx,
                        )
                    )
                    task_index += 1
    return tasks


def _task_descriptor(task: _RunTask) -> str:
    return (
        f"image={task.image_path.name} mode={task.tool_mode} "
        f"thinking={task.thinking_level} replicate={task.replicate_idx}"
    )


def _is_primary_eligible(
    *,
    retrieval_duplicate: bool,
    retrieval_mask_source: bool,
    audit_unavailable: bool,
    exclude_duplicates: bool,
    exclude_mask_source: bool,
    include_audit_unavailable_in_primary: bool,
) -> bool:
    if exclude_duplicates and retrieval_duplicate:
        return False
    if exclude_mask_source and retrieval_mask_source:
        return False
    if (not include_audit_unavailable_in_primary) and audit_unavailable:
        return False
    return True


def _update_run_record_with_eval_and_leakage(
    *,
    run_record_path: str | None,
    eval_metrics: Dict[str, float],
    leak,
    analysis_primary: bool,
) -> None:
    if not run_record_path:
        return
    path = Path(run_record_path)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        return
    payload["evaluation"] = {
        "iou": eval_metrics["iou"],
        "dice": eval_metrics["dice"],
        "precision": eval_metrics["precision"],
        "recall": eval_metrics["recall"],
    }
    payload["leakage"] = {
        "retrieval_duplicate": leak.retrieval_duplicate,
        "retrieval_mask_source": leak.retrieval_mask_source,
        "audit_unavailable": leak.audit_unavailable,
        "duplicate_reasons": leak.duplicate_reasons,
        "mask_source_reasons": leak.mask_source_reasons,
        "analysis_primary": analysis_primary,
        "analysis_sensitivity": True,
        "audit_status": "audited" if not leak.audit_unavailable else "audit_unavailable",
    }
    _atomic_write_json(path, payload)


def _resolve_selected_mask_path(cfg: StudyConfig, *, task: _RunTask, run_id: str) -> str:
    mask_dir = (
        Path(cfg.paths.artifacts_root).expanduser().resolve()
        / task.image_path.stem
        / task.tool_mode
        / f"replicate_{task.replicate_idx}"
        / run_id
        / "final"
    )
    mask_candidates = sorted(mask_dir.glob("final_mask_*.png"))
    return str(mask_candidates[-1]) if mask_candidates else ""


def _run_single_task(
    *,
    cfg: StudyConfig,
    task: _RunTask,
    run_id: str,
    engine: SegmentationEngine,
) -> Dict[str, object]:
    image_bytes = task.image_path.read_bytes()
    gt = np.array(Image.open(task.mask_path))

    request = EngineRequest(
        image_bytes=image_bytes,
        image_name=task.image_path.name,
        image_id=task.image_path.stem,
        target=cfg.target,
        mode=cfg.execution.mode,
        task_profile=cfg.execution.task_profile,
        tool_mode=task.tool_mode,
        query_policy=cfg.retrieval.query_policy,
        snapshot_policy=cfg.retrieval.snapshot_policy,
        scope_policy=cfg.retrieval.scope_policy,
        thinking_level=task.thinking_level,
        include_thoughts=cfg.model.include_thoughts,
        max_retries_semantic=cfg.execution.max_retries_semantic,
        max_retries_transport=cfg.execution.max_retries_transport,
        constraints=ConstraintConfig(**cfg.execution.constraints),
        budget=BudgetConfig(**cfg.execution.budget),
        output_format="png",
        return_debug=False,
        run_id=run_id,
        replicate_idx=task.replicate_idx,
        split=task.split,
    )
    result = engine.segment_once(request)
    eval_metrics = evaluate_segmentation(gt, result.mask)

    selected_attempt = [a for a in result.attempts if a.attempt_index == result.selected_attempt_index]
    selected_grounding = selected_attempt[0].grounding if selected_attempt else {}
    leak = audit_retrieval(
        input_image_path=task.image_path,
        grounding=selected_grounding,
        near_hamming_threshold=cfg.leakage.near_hamming_threshold,
    )
    analysis_primary = _is_primary_eligible(
        retrieval_duplicate=leak.retrieval_duplicate,
        retrieval_mask_source=leak.retrieval_mask_source,
        audit_unavailable=leak.audit_unavailable,
        exclude_duplicates=cfg.retrieval.primary_exclude_duplicates,
        exclude_mask_source=cfg.retrieval.primary_exclude_mask_source,
        include_audit_unavailable_in_primary=cfg.retrieval.include_audit_unavailable_in_primary,
    )
    _update_run_record_with_eval_and_leakage(
        run_record_path=result.run_record_path,
        eval_metrics=eval_metrics,
        leak=leak,
        analysis_primary=analysis_primary,
    )

    row = {
        "_task_index": task.task_index,
        "run_id": run_id,
        "image_id": task.image_path.stem,
        "image_name": task.image_path.name,
        "split": task.split,
        "tool_mode": task.tool_mode,
        "query_policy": cfg.retrieval.query_policy,
        "snapshot_policy": cfg.retrieval.snapshot_policy,
        "scope_policy": cfg.retrieval.scope_policy,
        "thinking_level": task.thinking_level,
        "replicate_idx": task.replicate_idx,
        "iou": eval_metrics["iou"],
        "dice": eval_metrics["dice"],
        "precision": eval_metrics["precision"],
        "recall": eval_metrics["recall"],
        "qc_pass": result.qc_pass,
        "selected_attempt_index": result.selected_attempt_index,
        "retrieval_duplicate": leak.retrieval_duplicate,
        "retrieval_mask_source": leak.retrieval_mask_source,
        "audit_unavailable": leak.audit_unavailable,
        "analysis_primary": analysis_primary,
        "analysis_sensitivity": True,
        "run_record_path": result.run_record_path,
        "mask_path": _resolve_selected_mask_path(cfg, task=task, run_id=run_id),
    }
    return row


def _run_tasks_serial(
    *,
    cfg: StudyConfig,
    stage: str,
    tasks: List[_RunTask],
    run_id: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    stall_events: List[Dict[str, object]] = []

    engine = _build_engine(cfg)
    last_progress_log = time.monotonic()
    log_every_s = max(1.0, float(cfg.execution.progress_log_interval_seconds))
    stall_warning_s = max(1.0, float(cfg.execution.stall_warning_seconds))

    for idx, task in enumerate(tasks, start=1):
        started = time.monotonic()
        try:
            rows.append(_run_single_task(cfg=cfg, task=task, run_id=run_id, engine=engine))
        except Exception as exc:
            failures.append(
                {
                    "task_index": task.task_index,
                    "task": _task_descriptor(task),
                    "error": str(exc),
                }
            )
            if cfg.execution.fail_fast:
                raise RuntimeError(f"Task failed ({_task_descriptor(task)}): {exc}") from exc
        finally:
            elapsed = time.monotonic() - started
            if elapsed >= stall_warning_s:
                event = {
                    "task_index": task.task_index,
                    "task": _task_descriptor(task),
                    "elapsed_seconds": round(elapsed, 3),
                }
                stall_events.append(event)
                print(f"[{stage}] long-running task detected: {event['task']} elapsed={elapsed:.1f}s")

            now = time.monotonic()
            if idx == len(tasks) or (now - last_progress_log) >= log_every_s:
                print(f"[{stage}] progress {idx}/{len(tasks)} completed")
                last_progress_log = now
    return rows, failures, stall_events


def _run_tasks_parallel(
    *,
    cfg: StudyConfig,
    stage: str,
    tasks: List[_RunTask],
    run_id: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    stall_events: List[Dict[str, object]] = []

    workers = max(1, int(cfg.execution.workers))
    poll_s = max(0.1, float(cfg.execution.progress_poll_seconds))
    log_every_s = max(poll_s, float(cfg.execution.progress_log_interval_seconds))
    stall_warning_s = max(1.0, float(cfg.execution.stall_warning_seconds))

    thread_local = threading.local()

    def _thread_engine() -> SegmentationEngine:
        engine = getattr(thread_local, "engine", None)
        if engine is None:
            engine = _build_engine(cfg)
            thread_local.engine = engine
        return engine

    def _worker(task: _RunTask) -> Dict[str, object]:
        return _run_single_task(cfg=cfg, task=task, run_id=run_id, engine=_thread_engine())

    future_to_task: Dict[Future, _RunTask] = {}
    future_started: Dict[Future, float] = {}
    warned_futures: set[Future] = set()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nanobanana-study") as pool:
        for task in tasks:
            fut = pool.submit(_worker, task)
            future_to_task[fut] = task
            future_started[fut] = time.monotonic()

        completed = 0
        last_progress_log = time.monotonic()
        total = len(tasks)

        while future_to_task:
            done, _ = wait(set(future_to_task.keys()), timeout=poll_s, return_when=FIRST_COMPLETED)
            now = time.monotonic()

            if done:
                for fut in done:
                    task = future_to_task.pop(fut)
                    started = future_started.pop(fut, now)
                    warned_futures.discard(fut)
                    completed += 1
                    try:
                        rows.append(fut.result())
                    except Exception as exc:
                        failures.append(
                            {
                                "task_index": task.task_index,
                                "task": _task_descriptor(task),
                                "error": str(exc),
                            }
                        )
                        if cfg.execution.fail_fast:
                            for other in future_to_task:
                                other.cancel()
                            raise RuntimeError(f"Task failed ({_task_descriptor(task)}): {exc}") from exc
                    finally:
                        elapsed = now - started
                        if elapsed >= stall_warning_s:
                            event = {
                                "task_index": task.task_index,
                                "task": _task_descriptor(task),
                                "elapsed_seconds": round(elapsed, 3),
                            }
                            stall_events.append(event)

            if completed == total or (now - last_progress_log) >= log_every_s:
                print(f"[{stage}] progress {completed}/{total} completed")
                last_progress_log = now

            for fut, task in future_to_task.items():
                if fut in warned_futures:
                    continue
                elapsed = now - future_started.get(fut, now)
                if elapsed >= stall_warning_s:
                    warned_futures.add(fut)
                    event = {
                        "task_index": task.task_index,
                        "task": _task_descriptor(task),
                        "elapsed_seconds": round(elapsed, 3),
                    }
                    stall_events.append(event)
                    print(f"[{stage}] long-running task detected: {event['task']} elapsed={elapsed:.1f}s")

    return rows, failures, stall_events


def _run_stage(cfg: StudyConfig, *, stage: str) -> Path:
    stage_tool_modes, stage_thinking_levels, stage_replicates, sample_size = _stage_matrix(cfg, stage)

    run_id = f"{cfg.study_id}_{stage}_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    results_root = Path(cfg.paths.results_root).expanduser().resolve()
    run_dir = results_root / cfg.dataset_name / cfg.model.model_id.replace("/", "_").replace(":", "_") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    items = load_dataset_items(
        dataset_name=cfg.dataset_name,
        dataset_root=cfg.dataset_root,
        manifest=cfg.manifest,
        sample_size=sample_size,
    )

    tasks = _build_tasks(
        items,
        stage_tool_modes=stage_tool_modes,
        stage_thinking_levels=stage_thinking_levels,
        stage_replicates=stage_replicates,
    )
    workers = max(1, int(cfg.execution.workers))
    print(f"[{stage}] starting run_id={run_id} tasks={len(tasks)} workers={workers}")

    if workers == 1:
        rows, failures, stall_events = _run_tasks_serial(cfg=cfg, stage=stage, tasks=tasks, run_id=run_id)
    else:
        rows, failures, stall_events = _run_tasks_parallel(cfg=cfg, stage=stage, tasks=tasks, run_id=run_id)

    rows.sort(key=lambda row: int(row.get("_task_index", 0)))
    for row in rows:
        row.pop("_task_index", None)

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_path = run_dir / "results.csv"
    tmp_results = results_path.with_suffix(".csv.tmp")
    results_df.to_csv(tmp_results, index=False)
    tmp_results.replace(results_path)

    report_paths = build_reports(
        results_df,
        run_dir / "reports",
        include_audit_unavailable_in_primary=cfg.retrieval.include_audit_unavailable_in_primary,
    )
    _atomic_write_json(
        run_dir / "run_summary.json",
        {
            "run_id": run_id,
            "stage": stage,
            "dataset_name": cfg.dataset_name,
            "dataset_root": str(cfg.dataset_root),
            "target": cfg.target,
            "model": asdict(cfg.model),
            "retrieval": asdict(cfg.retrieval),
            "matrix": {
                "tool_modes": stage_tool_modes,
                "thinking_levels": stage_thinking_levels,
                "replicates": stage_replicates,
            },
            "execution": asdict(cfg.execution),
            "n_tasks": int(len(tasks)),
            "n_rows": int(len(results_df)),
            "n_failures": int(len(failures)),
            "failures": failures,
            "stall_events": stall_events,
            "results_csv": str(results_path),
            "reports": report_paths,
        },
    )

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NanoBanana study runner")
    parser.add_argument("--config", required=True, help="Path to study YAML config")
    parser.add_argument(
        "--stage",
        choices=["stage0", "stage1", "stage2", "all"],
        default="stage1",
        help="Stage selection",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_study_config(Path(args.config).expanduser().resolve())
    stages: Iterable[str]
    if args.stage == "all":
        stages = ["stage0", "stage1"] + (["stage2"] if cfg.stages.stage2_enabled else [])
    elif args.stage == "stage2" and not cfg.stages.stage2_enabled:
        raise ValueError("stage2 requested but stage2_enabled=false in config")
    else:
        stages = [args.stage]

    for stage in stages:
        run_dir = _run_stage(cfg, stage=stage)
        print(f"[{stage}] outputs at {run_dir}")


if __name__ == "__main__":
    main()
