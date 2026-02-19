from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .prompts import PromptFamily


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
VALID_PROMPT_FAMILIES = {family.value for family in PromptFamily}


@dataclass(frozen=True)
class BatchJob:
    dataset_name: str
    dataset_root: Path
    model_name: str
    provider: str
    prompt_families: tuple[str, ...]
    manifest: Optional[str]
    timeout: float
    max_retries: int
    workers: int
    sample_size: Optional[int]
    rate_limit: Optional[float]
    local_cache: bool
    local_cache_dir: Path
    gemini_explicit_cache: bool
    gemini_cache_ttl: int
    thinking_budget: int
    temperature: float
    legacy_predictions: bool
    success_threshold: float
    bootstrap_method: str
    bootstrap_resamples: int

    @property
    def job_id(self) -> str:
        return f"{self.dataset_name}__{self.model_name}"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sanitize_token(value: str) -> str:
    token = value.strip().replace("/", "_").replace(":", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", token)


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _expand_env_placeholders(payload: Any, *, allow_missing: bool = False) -> Any:
    if isinstance(payload, dict):
        return {
            key: _expand_env_placeholders(value, allow_missing=allow_missing)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_expand_env_placeholders(item, allow_missing=allow_missing) for item in payload]
    if isinstance(payload, str):
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            value = os.getenv(key)
            if value is None:
                if allow_missing:
                    return match.group(0)
                raise ValueError(f"Missing environment variable '{key}' required by config placeholder.")
            return value

        return ENV_PATTERN.sub(replace, payload)
    return payload


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at root of {path}")
    return loaded


def load_batch_config(config_path: Path, overrides_path: Optional[Path] = None) -> Dict[str, Any]:
    base = _load_yaml(config_path)
    merged = base
    if overrides_path is not None:
        overrides = _load_yaml(overrides_path)
        merged = _deep_merge(base, overrides)
    expanded = _expand_env_placeholders(merged, allow_missing=True)
    return expanded


def _default_run_id(study_id: str) -> str:
    return f"{_sanitize_token(study_id)}_{_timestamp()}"


def _resolve_prompt_families(candidates: Optional[Iterable[str]]) -> tuple[str, ...]:
    if candidates is None:
        families = ("label_v1", "desc_v1", "desc_neg_v1")
    else:
        families = tuple(str(item) for item in candidates)
    if not families:
        raise ValueError("prompt_families must not be empty.")
    invalid = [family for family in families if family not in VALID_PROMPT_FAMILIES]
    if invalid:
        raise ValueError(
            f"Invalid prompt family values: {invalid}. Valid values: {sorted(VALID_PROMPT_FAMILIES)}"
        )
    return families


def build_jobs(
    config: Dict[str, Any],
    *,
    only_datasets: Optional[Iterable[str]] = None,
    only_models: Optional[Iterable[str]] = None,
) -> List[BatchJob]:
    schema_version = int(config.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version {schema_version}; expected 1.")

    datasets = config.get("datasets")
    models = config.get("models")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Config must define a non-empty datasets list.")
    if not isinstance(models, list) or not models:
        raise ValueError("Config must define a non-empty models list.")

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping.")

    dataset_filter = {item for item in (only_datasets or [])}
    model_filter = {item for item in (only_models or [])}

    jobs: List[BatchJob] = []
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise ValueError("Each dataset entry must be a mapping.")
        dataset_name = str(dataset.get("name", "")).strip()
        if not dataset_name:
            raise ValueError("Each dataset entry must include non-empty 'name'.")
        if dataset_filter and dataset_name not in dataset_filter:
            continue

        dataset_root_value = _expand_env_placeholders(dataset.get("root"))
        if not dataset_root_value:
            raise ValueError(f"Dataset '{dataset_name}' is missing required 'root'.")
        dataset_root = Path(str(dataset_root_value)).expanduser()

        for model in models:
            if not isinstance(model, dict):
                raise ValueError("Each model entry must be a mapping.")
            model_name = str(model.get("name", "")).strip()
            if not model_name:
                raise ValueError("Each model entry must include non-empty 'name'.")
            if model_filter and model_name not in model_filter:
                continue

            provider = str(
                model.get("provider", dataset.get("provider", defaults.get("provider", "gemini")))
            ).strip()
            if provider not in {"gemini", "moondream", "replicate"}:
                raise ValueError(
                    f"Unsupported provider '{provider}' in dataset '{dataset_name}' model '{model_name}'."
                )

            prompt_families = _resolve_prompt_families(
                model.get("prompt_families", dataset.get("prompt_families", defaults.get("prompt_families")))
            )

            local_cache_enabled = bool(
                model.get("local_cache", dataset.get("local_cache", defaults.get("local_cache", True)))
            )
            local_cache_dir = Path(
                str(
                    _expand_env_placeholders(
                        model.get(
                            "local_cache_dir",
                            dataset.get(
                                "local_cache_dir",
                                defaults.get("local_cache_dir", "results/.request_cache"),
                            ),
                        )
                    )
                )
            ).expanduser()

            gemini_explicit_cache = bool(
                model.get(
                    "gemini_explicit_cache",
                    dataset.get("gemini_explicit_cache", defaults.get("gemini_explicit_cache", True)),
                )
            )
            if provider == "gemini" and "robotics-er" in model_name:
                gemini_explicit_cache = False

            job = BatchJob(
                dataset_name=dataset_name,
                dataset_root=dataset_root,
                model_name=model_name,
                provider=provider,
                prompt_families=prompt_families,
                manifest=_expand_env_placeholders(
                    model.get("manifest", dataset.get("manifest", defaults.get("manifest")))
                ),
                timeout=float(model.get("timeout", dataset.get("timeout", defaults.get("timeout", 60.0)))),
                max_retries=int(
                    model.get("max_retries", dataset.get("max_retries", defaults.get("max_retries", 5)))
                ),
                workers=int(model.get("workers", dataset.get("workers", defaults.get("workers", 1)))),
                sample_size=model.get("sample_size", dataset.get("sample_size", defaults.get("sample_size"))),
                rate_limit=model.get("rate_limit", dataset.get("rate_limit", defaults.get("rate_limit"))),
                local_cache=local_cache_enabled,
                local_cache_dir=local_cache_dir,
                gemini_explicit_cache=gemini_explicit_cache,
                gemini_cache_ttl=int(
                    model.get(
                        "gemini_cache_ttl",
                        dataset.get("gemini_cache_ttl", defaults.get("gemini_cache_ttl", 3600)),
                    )
                ),
                thinking_budget=int(
                    model.get(
                        "thinking_budget",
                        dataset.get("thinking_budget", defaults.get("thinking_budget", 0)),
                    )
                ),
                temperature=float(
                    model.get("temperature", dataset.get("temperature", defaults.get("temperature", 0.5)))
                ),
                legacy_predictions=bool(
                    model.get(
                        "legacy_predictions",
                        dataset.get("legacy_predictions", defaults.get("legacy_predictions", False)),
                    )
                ),
                success_threshold=float(
                    model.get(
                        "success_threshold",
                        dataset.get("success_threshold", defaults.get("success_threshold", 0.5)),
                    )
                ),
                bootstrap_method=str(
                    model.get(
                        "bootstrap_method",
                        dataset.get("bootstrap_method", defaults.get("bootstrap_method", "bca")),
                    )
                ),
                bootstrap_resamples=int(
                    model.get(
                        "bootstrap_resamples",
                        dataset.get("bootstrap_resamples", defaults.get("bootstrap_resamples", 5000)),
                    )
                ),
            )
            jobs.append(job)

    if not jobs:
        raise ValueError("No jobs generated after applying filters.")
    return jobs


def preflight_jobs(jobs: Iterable[BatchJob], *, skip_env_checks: bool = False) -> None:
    providers = {job.provider for job in jobs}
    if not skip_env_checks:
        if "gemini" in providers and not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError("GOOGLE_API_KEY is required for Gemini jobs.")
        if "moondream" in providers and not os.getenv("MOONDREAM_API_KEY"):
            raise EnvironmentError("MOONDREAM_API_KEY is required for Moondream jobs.")
        if "replicate" in providers and not os.getenv("REPLICATE_API_TOKEN"):
            raise EnvironmentError("REPLICATE_API_TOKEN is required for Replicate jobs.")

    for job in jobs:
        images_dir = job.dataset_root / "images"
        masks_dir = job.dataset_root / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset '{job.dataset_name}' expected images/ and masks/ under {job.dataset_root}"
            )

        if job.manifest:
            manifest_path = Path(job.manifest)
            if not manifest_path.is_absolute():
                manifest_path = job.dataset_root / manifest_path
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Manifest for dataset '{job.dataset_name}' does not exist: {manifest_path}"
                )

        if job.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0 for job {job.job_id}")
        if job.workers < 1:
            raise ValueError(f"workers must be >= 1 for job {job.job_id}")
        if job.provider == "gemini" and "robotics-er" in job.model_name and job.gemini_explicit_cache:
            raise ValueError(
                f"Robotics ER job {job.job_id} must disable explicit Gemini cache."
            )


def build_segment_command(job: BatchJob, *, run_id: str, results_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "gemini_segmentation.cli",
        "segment",
        job.dataset_name,
        str(job.dataset_root),
        "--provider",
        job.provider,
        "--model-name",
        job.model_name,
    ]
    if job.manifest:
        cmd.extend(["--manifest", str(job.manifest)])
    for family in job.prompt_families:
        cmd.extend(["--prompt-family", family])
    if job.provider == "gemini":
        cmd.extend(["--thinking-budget", str(job.thinking_budget)])
        cmd.extend(["--temperature", str(job.temperature)])
    cmd.extend(["--timeout", str(job.timeout)])
    cmd.extend(["--max-retries", str(job.max_retries)])
    cmd.extend(["--workers", str(job.workers)])
    if job.sample_size is not None:
        cmd.extend(["--sample-size", str(job.sample_size)])
    cmd.extend(["--results-dir", str(results_dir)])
    cmd.extend(["--run-id", run_id])
    if job.rate_limit is not None:
        cmd.extend(["--rate-limit", str(job.rate_limit)])
    if job.legacy_predictions:
        cmd.append("--legacy-predictions")
    cmd.extend(["--success-threshold", str(job.success_threshold)])
    cmd.extend(["--bootstrap-method", job.bootstrap_method])
    cmd.extend(["--bootstrap-resamples", str(job.bootstrap_resamples)])
    if job.local_cache:
        cmd.extend(["--local-cache", "--local-cache-dir", str(job.local_cache_dir)])
    else:
        cmd.append("--no-local-cache")
    if job.provider == "gemini":
        if job.gemini_explicit_cache:
            cmd.append("--gemini-explicit-cache")
        else:
            cmd.append("--no-gemini-explicit-cache")
        cmd.extend(["--gemini-cache-ttl", str(job.gemini_cache_ttl)])
    return cmd


def build_fairness_command(job: BatchJob, *, run_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "gemini_segmentation.cli",
        "fairness",
        job.dataset_name,
        str(job.dataset_root),
        str(run_dir),
    ]
    if job.manifest:
        cmd.extend(["--manifest", str(job.manifest)])
    return cmd


def discover_prompt_run_dirs(
    *,
    results_dir: Path,
    dataset_name: str,
    model_name: str,
    run_id: str,
) -> List[Path]:
    model_dir = results_dir / dataset_name / model_name
    if not model_dir.is_dir():
        return []
    run_dirs: List[Path] = []
    for prompt_dir in sorted(model_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        candidate = prompt_dir / run_id
        if candidate.is_dir():
            run_dirs.append(candidate)
    return run_dirs


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _run_command(cmd: List[str], log_path: Path) -> tuple[int, float, str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _iso_now()
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"started_at: {started_at}\n")
        handle.write(f"command: {' '.join(cmd)}\n\n")
        handle.flush()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            process.stdout.close()
        return_code = process.wait()
    duration_s = time.monotonic() - start
    finished_at = _iso_now()
    return return_code, duration_s, started_at, finished_at


def run_batch(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    overrides_path = Path(args.overrides).expanduser().resolve() if args.overrides else None
    config = load_batch_config(config_path, overrides_path)

    study_id = str(config.get("study_id", "batch"))
    run_id = args.run_id or _default_run_id(study_id)
    results_dir = Path(config.get("results_dir", "results")).expanduser().resolve()
    batch_dir = results_dir / "batches" / run_id
    logs_dir = batch_dir / "logs"
    status_path = batch_dir / "job_status.jsonl"
    summary_path = batch_dir / "summary.json"
    resolved_path = batch_dir / "resolved_config.json"

    jobs = build_jobs(config, only_datasets=args.only_dataset, only_models=args.only_model)
    preflight_jobs(jobs, skip_env_checks=bool(args.dry_run))

    resolved_payload = {
        "schema_version": config.get("schema_version"),
        "study_id": study_id,
        "run_id": run_id,
        "results_dir": str(results_dir),
        "config_path": str(config_path),
        "overrides_path": str(overrides_path) if overrides_path else None,
        "auto_fairness": bool(args.auto_fairness),
        "stop_on_failure": bool(args.stop_on_failure),
        "dry_run": bool(args.dry_run),
        "jobs": [asdict(job) | {"dataset_root": str(job.dataset_root), "local_cache_dir": str(job.local_cache_dir)} for job in jobs],
    }
    _write_json(resolved_path, resolved_payload)

    if args.dry_run:
        for job in jobs:
            cmd = build_segment_command(job, run_id=run_id, results_dir=results_dir)
            _append_jsonl(
                status_path,
                {
                    "phase": "segment",
                    "status": "planned",
                    "job_id": job.job_id,
                    "dataset_name": job.dataset_name,
                    "model_name": job.model_name,
                    "command": cmd,
                    "run_id": run_id,
                },
            )
        _write_json(
            summary_path,
            {
                "study_id": study_id,
                "run_id": run_id,
                "dry_run": True,
                "planned_segment_jobs": len(jobs),
                "planned_fairness_jobs": 0,
                "failed_segment_jobs": 0,
                "failed_fairness_jobs": 0,
                "failed_job_ids": [],
            },
        )
        return 0

    failed_segment_jobs: List[str] = []
    failed_fairness_jobs: List[str] = []
    executed_segment_jobs = 0
    executed_fairness_jobs = 0
    started_at = _iso_now()
    run_start = time.monotonic()
    stop_requested = False

    for index, job in enumerate(jobs, start=1):
        segment_cmd = build_segment_command(job, run_id=run_id, results_dir=results_dir)
        segment_log = logs_dir / (
            f"{index:03d}_segment_{_sanitize_token(job.dataset_name)}__{_sanitize_token(job.model_name)}.log"
        )
        print(
            f"[{_iso_now()}] Starting segment job {index}/{len(jobs)}: {job.job_id} (log: {segment_log})",
            flush=True,
        )
        exit_code, duration_s, job_started, job_finished = _run_command(segment_cmd, segment_log)
        print(
            f"[{_iso_now()}] Finished segment job {job.job_id} with exit_code={exit_code} in {duration_s:.3f}s",
            flush=True,
        )
        executed_segment_jobs += 1
        record = {
            "phase": "segment",
            "job_id": job.job_id,
            "dataset_name": job.dataset_name,
            "model_name": job.model_name,
            "run_id": run_id,
            "command": segment_cmd,
            "log_path": str(segment_log),
            "started_at": job_started,
            "finished_at": job_finished,
            "duration_s": round(duration_s, 3),
            "exit_code": exit_code,
            "status": "succeeded" if exit_code == 0 else "failed",
        }
        _append_jsonl(status_path, record)

        if exit_code != 0:
            failed_segment_jobs.append(job.job_id)
            if args.stop_on_failure:
                stop_requested = True
                break
            continue

        if not args.auto_fairness:
            continue

        run_dirs = discover_prompt_run_dirs(
            results_dir=results_dir,
            dataset_name=job.dataset_name,
            model_name=job.model_name,
            run_id=run_id,
        )
        if not run_dirs:
            failed_fairness_jobs.append(job.job_id)
            _append_jsonl(
                status_path,
                {
                    "phase": "fairness",
                    "job_id": job.job_id,
                    "dataset_name": job.dataset_name,
                    "model_name": job.model_name,
                    "run_id": run_id,
                    "status": "failed",
                    "exit_code": None,
                    "duration_s": 0.0,
                    "reason": "No prompt-family run directories discovered for fairness.",
                },
            )
            if args.stop_on_failure:
                stop_requested = True
                break
            continue

        for fairness_idx, run_dir in enumerate(run_dirs, start=1):
            fairness_cmd = build_fairness_command(job, run_dir=run_dir)
            fairness_log = logs_dir / (
                f"{index:03d}_{fairness_idx:02d}_fairness_"
                f"{_sanitize_token(job.dataset_name)}__{_sanitize_token(job.model_name)}__"
                f"{_sanitize_token(run_dir.parent.name)}.log"
            )
            print(
                f"[{_iso_now()}] Starting fairness job {job.job_id} prompt={run_dir.parent.name} (log: {fairness_log})",
                flush=True,
            )
            fair_exit, fair_duration, fair_started, fair_finished = _run_command(
                fairness_cmd,
                fairness_log,
            )
            print(
                f"[{_iso_now()}] Finished fairness job {job.job_id} prompt={run_dir.parent.name} exit_code={fair_exit} in {fair_duration:.3f}s",
                flush=True,
            )
            executed_fairness_jobs += 1
            fair_record = {
                "phase": "fairness",
                "job_id": job.job_id,
                "dataset_name": job.dataset_name,
                "model_name": job.model_name,
                "prompt_key": run_dir.parent.name,
                "run_dir": str(run_dir),
                "run_id": run_id,
                "command": fairness_cmd,
                "log_path": str(fairness_log),
                "started_at": fair_started,
                "finished_at": fair_finished,
                "duration_s": round(fair_duration, 3),
                "exit_code": fair_exit,
                "status": "succeeded" if fair_exit == 0 else "failed",
            }
            _append_jsonl(status_path, fair_record)
            if fair_exit != 0:
                failed_fairness_jobs.append(f"{job.job_id}::{run_dir.parent.name}")
                if args.stop_on_failure:
                    stop_requested = True
                    break

        if stop_requested:
            break

    ended_at = _iso_now()
    total_duration_s = time.monotonic() - run_start

    summary = {
        "study_id": study_id,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": ended_at,
        "duration_s": round(total_duration_s, 3),
        "dry_run": False,
        "auto_fairness": bool(args.auto_fairness),
        "stop_on_failure": bool(args.stop_on_failure),
        "stop_requested": stop_requested,
        "segment_jobs_total": len(jobs),
        "segment_jobs_executed": executed_segment_jobs,
        "segment_jobs_failed": len(failed_segment_jobs),
        "fairness_jobs_executed": executed_fairness_jobs,
        "fairness_jobs_failed": len(failed_fairness_jobs),
        "failed_job_ids": failed_segment_jobs + failed_fairness_jobs,
    }
    _write_json(summary_path, summary)

    return 1 if failed_segment_jobs or failed_fairness_jobs else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch orchestration for segmentation benchmark matrices.")
    parser.add_argument("--config", required=True, help="Path to matrix YAML config.")
    parser.add_argument("--overrides", help="Optional local override YAML merged into base config.")
    parser.add_argument("--run-id", help="Override run id (default: <study_id>_<YYYYMMDD-HHMMSS>).")
    parser.add_argument(
        "--only-dataset",
        action="append",
        help="Repeat to restrict execution to selected dataset names.",
    )
    parser.add_argument(
        "--only-model",
        action="append",
        help="Repeat to restrict execution to selected model names.",
    )
    parser.add_argument(
        "--auto-fairness",
        action="store_true",
        help="Run fairness command for each generated prompt-family run directory after successful segmentation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and emit planned commands without executing jobs.")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately on the first failed segmentation or fairness command.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
