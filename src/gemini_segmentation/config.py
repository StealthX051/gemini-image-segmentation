from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .types import RunConfig


def resolve_preset_name(preset: str, branch: Optional[str] = None) -> str:
    """Return the preset key adjusted for an optional branch suffix."""

    if branch in (None, "", "legacy"):
        return preset
    suffix = f"_{branch}"
    if preset.endswith(suffix):
        return preset
    return f"{preset}{suffix}"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def load_preset(config_path: Path, preset: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    if preset not in cfg:
        raise KeyError(f"Preset '{preset}' not found in {config_path}")
    return cfg[preset]


def build_run_config(
    dataset_name: str,
    dataset_root: Path,
    prompt: str,
    model_name: str,
    *,
    thinking_budget: int = 0,
    temperature: float = 0.5,
    safety_settings: Optional[Dict[str, Any]] = None,
    timeout_s: float = 60.0,
    workers: int = 1,
    sample_size: Optional[int] = None,
    manifest_path: Optional[Path] = None,
    rate_limit_s: Optional[float] = None,
    legacy_predictions: bool = False,
    run_id: Optional[str] = None,
    bootstrap_method: str = "bca",
    bootstrap_resamples: int = 5000,
) -> RunConfig:
    return RunConfig(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        model_name=model_name,
        prompt=prompt,
        thinking_budget=thinking_budget,
        temperature=temperature,
        safety_settings=safety_settings,
        timeout_s=timeout_s,
        workers=workers,
        sample_size=sample_size,
        manifest_path=manifest_path,
        rate_limit_s=rate_limit_s,
        legacy_predictions=legacy_predictions,
        run_id=run_id,
        bootstrap_method=bootstrap_method,
        bootstrap_resamples=bootstrap_resamples,
    )


def dump_run_config(config: RunConfig, path: Path) -> None:
    data = asdict(config)
    # Convert Paths to strings for JSON/YAML friendliness
    data["dataset_root"] = str(config.dataset_root)
    if config.manifest_path:
        data["manifest_path"] = str(config.manifest_path)
    path.write_text(json.dumps(data, indent=2))
