from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class TableSpec:
    name: str
    title: str
    metric: str
    statistic: str
    group_by: List[str]
    description: Optional[str] = None


@dataclass
class FigureSpec:
    name: str
    title: str
    metric: str
    statistic: str
    x: str
    hue: Optional[str]
    group_by: List[str]
    description: Optional[str] = None


@dataclass
class PaperConfig:
    required_columns: List[str]
    models: Dict[str, str]
    prompt_strategies: Dict[str, str]
    tasks: Dict[str, str]
    tables: List[TableSpec]
    figures: List[FigureSpec]
    bootstrap_resamples: int = 5000
    bootstrap_method: str = "bca"


def _load_yaml(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def _parse_tables(raw_tables: Dict[str, dict]) -> List[TableSpec]:
    tables: List[TableSpec] = []
    for name, payload in raw_tables.items():
        tables.append(
            TableSpec(
                name=name,
                title=payload.get("title", name),
                metric=payload["metric"],
                statistic=payload.get("statistic", "mean"),
                group_by=list(payload.get("group_by", [])),
                description=payload.get("description"),
            )
        )
    return tables


def _parse_figures(raw_figures: Dict[str, dict]) -> List[FigureSpec]:
    figures: List[FigureSpec] = []
    for name, payload in raw_figures.items():
        figures.append(
            FigureSpec(
                name=name,
                title=payload.get("title", name),
                metric=payload["metric"],
                statistic=payload.get("statistic", "mean"),
                x=payload["x"],
                hue=payload.get("hue"),
                group_by=list(payload.get("group_by", [])),
                description=payload.get("description"),
            )
        )
    return figures


def load_paper_config(path: Path) -> PaperConfig:
    """Load and validate the paper registry YAML."""

    raw = _load_yaml(path)
    required_columns = list(raw.get("required_columns", []))
    if not required_columns:
        raise ValueError("Config must define required_columns")

    models = raw.get("models", {})
    prompt_strategies = raw.get("prompt_strategies", {})
    tasks = raw.get("tasks", {})
    tables = _parse_tables(raw.get("tables", {}))
    figures = _parse_figures(raw.get("figures", {}))

    if not tables:
        raise ValueError("Config must include at least one table spec")

    if not figures:
        raise ValueError("Config must include at least one figure spec")

    return PaperConfig(
        required_columns=required_columns,
        models=models,
        prompt_strategies=prompt_strategies,
        tasks=tasks,
        tables=tables,
        figures=figures,
        bootstrap_resamples=int(raw.get("bootstrap_resamples", 5000)),
        bootstrap_method=str(raw.get("bootstrap_method", "bca")),
    )
