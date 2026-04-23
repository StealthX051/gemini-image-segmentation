from __future__ import annotations

"""Render model-vs-prompt comparison reports from completed run directories."""

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
# Use a non-interactive backend so PDF rendering works in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "reports"

DEFAULT_PROMPT_ORDER = ("label_v1", "desc_v1", "desc_neg_v1")
DEFAULT_GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-robotics-er-1.6-preview",
    "gemini-robotics-er-1.6-preview-agentic",
)
DEFAULT_MOONDREAM_MODELS = ("moondream-3",)
DEFAULT_REPLICATE_BATCH_PATTERN = "replicate_sa2va_{dataset}_full_*"
MODEL_SORT_ORDER = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-robotics-er-1.5-preview",
    "gemini-robotics-er-1.6-preview",
    "gemini-robotics-er-1.6-preview-agentic",
    "moondream-3",
)

MODEL_DISPLAY_NAMES = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
    "gemini-robotics-er-1.5-preview": "Gemini Robotics-ER 1.5",
    "gemini-robotics-er-1.6-preview": "Gemini Robotics-ER 1.6",
    "gemini-robotics-er-1.6-preview-agentic": "Gemini Robotics-ER 1.6 + Agentic Vision",
    "moondream-3": "Moondream 3",
}
PROMPT_FAMILY_DISPLAY_NAMES = {
    "label_v1": "Label-Only",
    "desc_v1": "Descriptor",
    "desc_neg_v1": "Descriptor + Exclusions",
}


@dataclass(frozen=True)
class MetricRow:
    model: str
    prompt_family: str
    mean_iou: float
    median_iou: float
    ci_iou_lower: float
    ci_iou_upper: float
    mean_dice: float
    median_dice: float
    ci_dice_lower: float
    ci_dice_upper: float
    success_rate: float
    run_id: str
    source: str


@dataclass(frozen=True)
class ReportArtifacts:
    markdown: Path
    html: Path
    pdf: Path
    csv: Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _find_latest_successful_batch_run(results_dir: Path, pattern: str) -> Optional[str]:
    batch_root = results_dir / "batches"
    if not batch_root.exists():
        return None
    candidates = sorted(
        [path for path in batch_root.iterdir() if path.is_dir() and fnmatch(path.name, pattern)],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        summary_path = candidate / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = _load_json(summary_path)
        except Exception:
            continue
        if bool(summary.get("dry_run")):
            continue
        if int(summary.get("segment_jobs_failed", 0)) != 0:
            continue
        if int(summary.get("fairness_jobs_failed", 0)) != 0:
            continue
        return candidate.name
    return None


def _prompt_sort_key(prompt_family: str) -> tuple[int, str]:
    if prompt_family in DEFAULT_PROMPT_ORDER:
        return (DEFAULT_PROMPT_ORDER.index(prompt_family), prompt_family)
    return (len(DEFAULT_PROMPT_ORDER), prompt_family)


def _model_sort_key(model: str) -> tuple[int, str]:
    known_order = MODEL_SORT_ORDER
    if model in known_order:
        return (known_order.index(model), model)
    return (len(known_order), model)


def _display_model_name(model: str) -> str:
    normalized = model.lower()
    if "sa2va-26b-image" in normalized:
        return "SA2VA 26B (Replicate)"
    if "sa2va-4b-image" in normalized:
        return "SA2VA 4B (Replicate)"
    return MODEL_DISPLAY_NAMES.get(model, model)


def _display_prompt_family(prompt_family: str) -> str:
    return PROMPT_FAMILY_DISPLAY_NAMES.get(prompt_family, prompt_family)


def _merge_models(*model_groups: Iterable[str]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for models in model_groups:
        for model in models:
            if model in seen:
                continue
            merged.append(model)
            seen.add(model)
    return merged


def _iter_run_dirs(results_dir: Path, dataset: str, model: str, run_id: str) -> Iterable[Path]:
    model_dir = results_dir / dataset / model
    if not model_dir.exists():
        return []
    run_dirs: List[Path] = []
    for prompt_dir in sorted(model_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        run_dir = prompt_dir / run_id
        if run_dir.is_dir():
            run_dirs.append(run_dir)
    return run_dirs


def _discover_models_for_run_id(
    *,
    results_dir: Path,
    dataset: str,
    run_id: str,
    provider: str,
) -> List[str]:
    dataset_dir = results_dir / dataset
    if not dataset_dir.exists():
        return []

    discovered: set[str] = set()
    for model_dir in dataset_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for prompt_dir in model_dir.iterdir():
            if not prompt_dir.is_dir():
                continue
            run_dir = prompt_dir / run_id
            if not run_dir.is_dir():
                continue
            run_config_path = run_dir / "run_config.json"
            if not run_config_path.exists():
                continue
            try:
                run_config = _load_json(run_config_path)
            except Exception:
                continue
            if str(run_config.get("provider", "")).lower() != provider.lower():
                continue
            discovered.add(model_dir.name)
            break
    return sorted(discovered, key=_model_sort_key)


def _load_metric_row(run_dir: Path, model: str, run_id: str, source: str) -> Optional[MetricRow]:
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.csv"
    if not run_config_path.exists() or not summary_path.exists():
        return None
    run_config = _load_json(run_config_path)
    prompt_family = str(run_config.get("prompt_family", "unknown"))
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        return None
    row = summary_df.iloc[0]
    median_iou = float(row["median_iou"]) if "median_iou" in summary_df.columns else float(row["mean_iou"])
    median_dice = float(row["median_dice"]) if "median_dice" in summary_df.columns else float(row["mean_dice"])
    return MetricRow(
        model=model,
        prompt_family=prompt_family,
        mean_iou=float(row["mean_iou"]),
        median_iou=median_iou,
        ci_iou_lower=float(row["ci_iou_lower"]),
        ci_iou_upper=float(row["ci_iou_upper"]),
        mean_dice=float(row["mean_dice"]),
        median_dice=median_dice,
        ci_dice_lower=float(row["ci_dice_lower"]),
        ci_dice_upper=float(row["ci_dice_upper"]),
        success_rate=float(row["success_rate"]),
        run_id=run_id,
        source=source,
    )


def _collect_rows(
    *,
    results_dir: Path,
    dataset: str,
    run_id: str,
    models: Iterable[str],
    source: str,
) -> List[MetricRow]:
    rows: List[MetricRow] = []
    for model in models:
        for run_dir in _iter_run_dirs(results_dir, dataset, model, run_id):
            loaded = _load_metric_row(run_dir, model=model, run_id=run_id, source=source)
            if loaded is not None:
                rows.append(loaded)
    return rows


def _format_ci(mean: float, lower: float, upper: float) -> str:
    return f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"


def _format_success(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _auto_resize_table_columns(table: "plt.Table", col_labels: List[str], rows: List[List[str]]) -> None:
    num_cols = len(col_labels)
    try:
        table.auto_set_column_width(col=list(range(num_cols)))
        return
    except Exception:
        pass

    widths = [len(str(label)) for label in col_labels]
    for row in rows:
        for idx, value in enumerate(row):
            if idx < num_cols:
                widths[idx] = max(widths[idx], len(str(value)))
    total = sum(widths) or 1
    normalized = [width / total for width in widths]
    for (_r, col_idx), cell in table.get_celld().items():
        if 0 <= col_idx < num_cols:
            cell.set_width(max(0.05, normalized[col_idx] * 0.98))


def _group_rows(rows: List[MetricRow]) -> Dict[str, List[MetricRow]]:
    grouped: Dict[str, List[MetricRow]] = {}
    for row in rows:
        grouped.setdefault(row.model, []).append(row)
    for model in grouped:
        grouped[model] = sorted(grouped[model], key=lambda item: _prompt_sort_key(item.prompt_family))
    return grouped


def _render_markdown(
    rows: List[MetricRow],
    *,
    dataset: str,
    gemini_run_id: str,
    moondream_run_id: Optional[str],
    replicate_run_id: Optional[str],
    output_path: Path,
) -> None:
    grouped = _group_rows(rows)
    lines: List[str] = []
    lines.append(f"# Table 1. Prompt-Family Ablation Performance by Model ({dataset})")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Gemini run ID: `{gemini_run_id}`")
    if moondream_run_id:
        lines.append(f"- Moondream run ID: `{moondream_run_id}`")
    if replicate_run_id:
        lines.append(f"- Replicate run ID: `{replicate_run_id}`")
    lines.append("")
    lines.append("Metrics include mean with 95% CI, median values, and success rate.")
    lines.append("")

    for model in sorted(grouped, key=_model_sort_key):
        lines.append(f"## {_display_model_name(model)}")
        lines.append("")
        lines.append(
            "| Prompt Family | Mean IoU (95% CI) | Median IoU | Mean Dice (95% CI) | Median Dice | Success Rate |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in grouped[model]:
            lines.append(
                "| "
                f"{_display_prompt_family(row.prompt_family)} | "
                f"{_format_ci(row.mean_iou, row.ci_iou_lower, row.ci_iou_upper)} | "
                f"{_format_metric(row.median_iou)} | "
                f"{_format_ci(row.mean_dice, row.ci_dice_lower, row.ci_dice_upper)} | "
                f"{_format_metric(row.median_dice)} | "
                f"{_format_success(row.success_rate)} |"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _render_html(
    rows: List[MetricRow],
    *,
    dataset: str,
    gemini_run_id: str,
    moondream_run_id: Optional[str],
    replicate_run_id: Optional[str],
    output_path: Path,
) -> None:
    grouped = _group_rows(rows)
    html_parts: List[str] = []
    html_parts.append("<!doctype html>")
    html_parts.append("<html><head><meta charset='utf-8'>")
    html_parts.append("<title>Prompt-Family Ablation Performance by Model</title>")
    html_parts.append(
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111;}"
        "h1{margin:0 0 8px 0;} h2{margin-top:24px;}"
        "table{border-collapse:collapse;width:100%;margin-top:8px;}"
        "th,td{border:1px solid #d9d9d9;padding:8px 10px;text-align:left;}"
        "th{background:#f5f5f5;} .meta{color:#444;}"
        "</style>"
    )
    html_parts.append("</head><body>")
    html_parts.append(f"<h1>Table 1. Prompt-Family Ablation Performance by Model ({dataset})</h1>")
    html_parts.append("<div class='meta'>")
    html_parts.append(f"<div>Generated: {datetime.now().isoformat(timespec='seconds')}</div>")
    html_parts.append(f"<div>Gemini run ID: <code>{gemini_run_id}</code></div>")
    if moondream_run_id:
        html_parts.append(f"<div>Moondream run ID: <code>{moondream_run_id}</code></div>")
    if replicate_run_id:
        html_parts.append(f"<div>Replicate run ID: <code>{replicate_run_id}</code></div>")
    html_parts.append("<div>Metrics include mean with 95% CI, median values, and success rate.</div>")
    html_parts.append("</div>")

    for model in sorted(grouped, key=_model_sort_key):
        html_parts.append(f"<h2>{_display_model_name(model)}</h2>")
        html_parts.append("<table>")
        html_parts.append(
            "<thead><tr>"
            "<th>Prompt Family</th>"
            "<th>Mean IoU (95% CI)</th>"
            "<th>Median IoU</th>"
            "<th>Mean Dice (95% CI)</th>"
            "<th>Median Dice</th>"
            "<th>Success Rate</th>"
            "</tr></thead><tbody>"
        )
        for row in grouped[model]:
            html_parts.append(
                "<tr>"
                f"<td>{_display_prompt_family(row.prompt_family)}</td>"
                f"<td>{_format_ci(row.mean_iou, row.ci_iou_lower, row.ci_iou_upper)}</td>"
                f"<td>{_format_metric(row.median_iou)}</td>"
                f"<td>{_format_ci(row.mean_dice, row.ci_dice_lower, row.ci_dice_upper)}</td>"
                f"<td>{_format_metric(row.median_dice)}</td>"
                f"<td>{_format_success(row.success_rate)}</td>"
                "</tr>"
            )
        html_parts.append("</tbody></table>")

    html_parts.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


def _render_pdf(
    rows: List[MetricRow],
    *,
    dataset: str,
    gemini_run_id: str,
    moondream_run_id: Optional[str],
    replicate_run_id: Optional[str],
    output_path: Path,
) -> None:
    grouped = _group_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        cover = plt.figure(figsize=(11, 8.5))
        cover.text(0.05, 0.92, f"Table 1. Prompt-Family Ablation Performance by Model ({dataset})", fontsize=16, weight="bold")
        cover.text(0.05, 0.84, f"Generated: {datetime.now().isoformat(timespec='seconds')}", fontsize=10)
        cover.text(0.05, 0.80, f"Gemini run ID: {gemini_run_id}", fontsize=10)
        meta_y = 0.76
        if moondream_run_id:
            cover.text(0.05, meta_y, f"Moondream run ID: {moondream_run_id}", fontsize=10)
            meta_y -= 0.04
        if replicate_run_id:
            cover.text(0.05, meta_y, f"Replicate run ID: {replicate_run_id}", fontsize=10)
            meta_y -= 0.06
        else:
            meta_y -= 0.02
        cover.text(0.05, meta_y, "Metrics include mean with 95% CI, median values, and success rate.", fontsize=10)
        cover.gca().axis("off")
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        mega_df = pd.DataFrame(
            {
                "Model": [_display_model_name(row.model) for row in rows],
                "Prompt Family": [_display_prompt_family(row.prompt_family) for row in rows],
                "Mean IoU (95% CI)": [
                    _format_ci(row.mean_iou, row.ci_iou_lower, row.ci_iou_upper) for row in rows
                ],
                "Median IoU": [_format_metric(row.median_iou) for row in rows],
                "Mean Dice (95% CI)": [
                    _format_ci(row.mean_dice, row.ci_dice_lower, row.ci_dice_upper) for row in rows
                ],
                "Median Dice": [_format_metric(row.median_dice) for row in rows],
                "Success Rate": [_format_success(row.success_rate) for row in rows],
            }
        )

        mega_fig, mega_ax = plt.subplots(figsize=(11, 8.5))
        mega_ax.set_title("Table 1A. Consolidated Results Across All Models and Prompt Families", fontsize=12, pad=12)
        mega_ax.axis("off")
        mega_table = mega_ax.table(
            cellText=mega_df.values,
            colLabels=mega_df.columns,
            cellLoc="left",
            loc="center",
        )
        _auto_resize_table_columns(
            mega_table,
            mega_df.columns.tolist(),
            mega_df.values.tolist(),
        )
        mega_table.auto_set_font_size(False)
        mega_table.set_fontsize(7)
        mega_table.scale(1, 1.2)
        pdf.savefig(mega_fig, bbox_inches="tight")
        plt.close(mega_fig)

        for model in sorted(grouped, key=_model_sort_key):
            model_rows = grouped[model]
            display_df = pd.DataFrame(
                {
                    "Prompt Family": [_display_prompt_family(row.prompt_family) for row in model_rows],
                    "Mean IoU (95% CI)": [
                        _format_ci(row.mean_iou, row.ci_iou_lower, row.ci_iou_upper) for row in model_rows
                    ],
                    "Median IoU": [_format_metric(row.median_iou) for row in model_rows],
                    "Mean Dice (95% CI)": [
                        _format_ci(row.mean_dice, row.ci_dice_lower, row.ci_dice_upper) for row in model_rows
                    ],
                    "Median Dice": [_format_metric(row.median_dice) for row in model_rows],
                    "Success Rate": [_format_success(row.success_rate) for row in model_rows],
                }
            )

            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.set_title(f"Table 1B. {_display_model_name(model)} Prompt-Family Comparison", fontsize=12, pad=12)
            ax.axis("off")
            table = ax.table(
                cellText=display_df.values,
                colLabels=display_df.columns,
                cellLoc="left",
                loc="center",
            )
            _auto_resize_table_columns(
                table,
                display_df.columns.tolist(),
                display_df.values.tolist(),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.4)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def generate_prompt_comparison_report(
    *,
    results_dir: Path,
    output_dir: Path,
    dataset: str,
    gemini_run_id: Optional[str],
    moondream_run_id: Optional[str],
    replicate_run_id: Optional[str],
) -> ReportArtifacts:
    resolved_gemini_run_id = gemini_run_id or _find_latest_successful_batch_run(
        results_dir, f"{dataset}_full_3x3_w10_*"
    )
    if not resolved_gemini_run_id:
        raise FileNotFoundError(
            "Unable to auto-detect a successful Gemini full run ID. Pass --gemini-run-id explicitly."
        )

    resolved_moondream_run_id = moondream_run_id or _find_latest_successful_batch_run(
        results_dir, f"moondream3_{dataset}_full_*"
    )

    resolved_replicate_run_id = replicate_run_id or _find_latest_successful_batch_run(
        results_dir, DEFAULT_REPLICATE_BATCH_PATTERN.format(dataset=dataset)
    )

    rows: List[MetricRow] = []
    gemini_models = _merge_models(
        DEFAULT_GEMINI_MODELS,
        _discover_models_for_run_id(
            results_dir=results_dir,
            dataset=dataset,
            run_id=resolved_gemini_run_id,
            provider="gemini",
        ),
    )
    rows.extend(
        _collect_rows(
            results_dir=results_dir,
            dataset=dataset,
            run_id=resolved_gemini_run_id,
            models=gemini_models,
            source="gemini",
        )
    )
    if resolved_moondream_run_id:
        rows.extend(
            _collect_rows(
                results_dir=results_dir,
                dataset=dataset,
                run_id=resolved_moondream_run_id,
                models=DEFAULT_MOONDREAM_MODELS,
                source="moondream",
            )
        )
    if resolved_replicate_run_id:
        replicate_models = _discover_models_for_run_id(
            results_dir=results_dir,
            dataset=dataset,
            run_id=resolved_replicate_run_id,
            provider="replicate",
        )
        if replicate_run_id and not replicate_models:
            raise FileNotFoundError(
                f"No Replicate rows found for run ID '{replicate_run_id}'. "
                "Check --replicate-run-id and results directory structure."
            )
        rows.extend(
            _collect_rows(
                results_dir=results_dir,
                dataset=dataset,
                run_id=resolved_replicate_run_id,
                models=replicate_models,
                source="replicate",
            )
        )
    if not rows:
        raise FileNotFoundError(
            "No comparison rows were found. Check run IDs and results directory structure."
        )

    rows = sorted(rows, key=lambda item: (_model_sort_key(item.model), _prompt_sort_key(item.prompt_family)))
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}_prompt_family_comparison_{_timestamp()}"
    markdown_path = output_dir / f"{stem}.md"
    html_path = output_dir / f"{stem}.html"
    pdf_path = output_dir / f"{stem}.pdf"
    csv_path = output_dir / f"{stem}.csv"

    _render_markdown(
        rows,
        dataset=dataset,
        gemini_run_id=resolved_gemini_run_id,
        moondream_run_id=resolved_moondream_run_id,
        replicate_run_id=resolved_replicate_run_id,
        output_path=markdown_path,
    )
    _render_html(
        rows,
        dataset=dataset,
        gemini_run_id=resolved_gemini_run_id,
        moondream_run_id=resolved_moondream_run_id,
        replicate_run_id=resolved_replicate_run_id,
        output_path=html_path,
    )
    _render_pdf(
        rows,
        dataset=dataset,
        gemini_run_id=resolved_gemini_run_id,
        moondream_run_id=resolved_moondream_run_id,
        replicate_run_id=resolved_replicate_run_id,
        output_path=pdf_path,
    )

    pd.DataFrame(
        [
            {
                "model": row.model,
                "model_display": _display_model_name(row.model),
                "prompt_family": row.prompt_family,
                "prompt_family_display": _display_prompt_family(row.prompt_family),
                "mean_iou": row.mean_iou,
                "median_iou": row.median_iou,
                "ci_iou_lower": row.ci_iou_lower,
                "ci_iou_upper": row.ci_iou_upper,
                "mean_dice": row.mean_dice,
                "median_dice": row.median_dice,
                "ci_dice_lower": row.ci_dice_lower,
                "ci_dice_upper": row.ci_dice_upper,
                "success_rate": row.success_rate,
                "run_id": row.run_id,
                "source": row.source,
            }
            for row in rows
        ]
    ).to_csv(csv_path, index=False)

    LOGGER.info("Wrote comparison markdown: %s", markdown_path)
    LOGGER.info("Wrote comparison html: %s", html_path)
    LOGGER.info("Wrote comparison pdf: %s", pdf_path)
    LOGGER.info("Wrote comparison csv: %s", csv_path)
    return ReportArtifacts(
        markdown=markdown_path,
        html=html_path,
        pdf=pdf_path,
        csv=csv_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grouped model-vs-prompt comparison report (Markdown/HTML/PDF)."
    )
    parser.add_argument("--dataset", default="polyp", help="Dataset name under results/")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Root results directory containing run outputs and batch metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated report files.",
    )
    parser.add_argument(
        "--gemini-run-id",
        help="Gemini run ID to compare (defaults to latest successful '<dataset>_full_3x3_w10_*' batch run).",
    )
    parser.add_argument(
        "--moondream-run-id",
        help="Moondream run ID to compare (defaults to latest successful 'moondream3_<dataset>_full_*' batch run).",
    )
    parser.add_argument(
        "--replicate-run-id",
        help=(
            "Replicate run ID to compare "
            "(defaults to latest successful 'replicate_sa2va_<dataset>_full_*' batch run when available)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    artifacts = generate_prompt_comparison_report(
        results_dir=Path(args.results_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        dataset=args.dataset,
        gemini_run_id=args.gemini_run_id,
        moondream_run_id=args.moondream_run_id,
        replicate_run_id=args.replicate_run_id,
    )
    LOGGER.info("Comparison report ready under %s", artifacts.markdown.parent)


if __name__ == "__main__":
    main()
