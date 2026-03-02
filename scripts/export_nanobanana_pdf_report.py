from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _coerce_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _page_text(pdf: PdfPages, *, title: str, lines: Iterable[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=16, fontweight="bold", va="top", ha="left")

    y = 0.95
    for line in lines:
        ax.text(0.0, y, line, fontsize=10, va="top", ha="left", family="monospace")
        y -= 0.028
        if y < 0.03:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.axis("off")
            ax.text(0.0, 1.0, f"{title} (cont.)", fontsize=16, fontweight="bold", va="top", ha="left")
            y = 0.95

    pdf.savefig(fig)
    plt.close(fig)


def _chunked(df: pd.DataFrame, chunk_size: int) -> Iterable[pd.DataFrame]:
    if df.empty:
        yield df
        return
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]


def _page_table(
    pdf: PdfPages,
    *,
    title: str,
    df: pd.DataFrame,
    max_rows: int = 24,
    precision: int = 4,
) -> None:
    if df.empty:
        _page_text(pdf, title=title, lines=["No data available."])
        return

    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda v: "" if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else f"{float(v):.{precision}f}"
            )

    for idx, chunk in enumerate(_chunked(formatted, max_rows), start=1):
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.9])
        ax.axis("off")

        page_title = title if idx == 1 else f"{title} (cont. {idx})"
        ax.text(0.0, 1.02, page_title, fontsize=14, fontweight="bold", va="bottom", ha="left")

        table = ax.table(
            cellText=chunk.values.tolist(),
            colLabels=list(chunk.columns),
            loc="upper left",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.25)

        for (r, _), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f0f0f0")
            cell.set_edgecolor("#d0d0d0")

        pdf.savefig(fig)
        plt.close(fig)


def _page_image(pdf: PdfPages, *, title: str, image_path: Path, caption: str = "") -> None:
    if not image_path.exists():
        _page_text(pdf, title=title, lines=[f"Missing image: {image_path}"])
        return
    img = plt.imread(str(image_path))
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    ax.axis("off")
    ax.imshow(img)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    if caption:
        fig.text(0.05, 0.03, caption, fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def _delta_summary_table(paired_primary: pd.DataFrame) -> pd.DataFrame:
    if paired_primary.empty:
        return pd.DataFrame(columns=["tool_mode", "n", "mean_delta_iou", "mean_delta_dice"])
    subset = paired_primary[paired_primary["tool_mode"] != "closed"].copy()
    if subset.empty:
        return pd.DataFrame(columns=["tool_mode", "n", "mean_delta_iou", "mean_delta_dice"])
    return (
        subset.groupby("tool_mode", dropna=False)
        .agg(
            n=("tool_mode", "count"),
            mean_delta_iou=("delta_iou_vs_closed", "mean"),
            mean_delta_dice=("delta_dice_vs_closed", "mean"),
        )
        .reset_index()
        .sort_values("tool_mode")
    )


def _infer_run_title(run_summary: dict, run_dir: Path) -> str:
    run_id = str(run_summary.get("run_id") or run_dir.name)
    dataset = str(run_summary.get("dataset_name") or "dataset")
    stage = str(run_summary.get("stage") or "stage")
    return f"NanoBanana Study Report: {dataset} | {stage} | {run_id}"


def generate_pdf_report(run_dir: Path, output_pdf: Path) -> Path:
    run_dir = _coerce_path(run_dir)
    reports_dir = run_dir / "reports"
    run_summary = _read_json(run_dir / "run_summary.json")

    results_df = _read_csv(run_dir / "results.csv")
    summary_primary = _read_csv(reports_dir / "summary_by_mode_primary.csv")
    summary_sensitivity = _read_csv(reports_dir / "summary_by_mode_sensitivity.csv")
    qc_primary = _read_csv(reports_dir / "qc_failure_summary_primary.csv")
    retrieval = _read_csv(reports_dir / "retrieval_audit_summary.csv")
    partitions = _read_csv(reports_dir / "analysis_partition_counts.csv")
    paired_primary = _read_csv(reports_dir / "paired_delta_vs_closed_primary.csv")
    delta_summary = _delta_summary_table(paired_primary)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    title = _infer_run_title(run_summary, run_dir)

    with PdfPages(output_pdf) as pdf:
        overview_lines: List[str] = [
            f"run_dir: {run_dir}",
            f"run_id: {run_summary.get('run_id', run_dir.name)}",
            f"stage: {run_summary.get('stage', 'unknown')}",
            f"dataset: {run_summary.get('dataset_name', 'unknown')}",
            f"target: {run_summary.get('target', 'unknown')}",
            f"model: {run_summary.get('model', {}).get('model_id', 'unknown')}",
            f"n_tasks: {run_summary.get('n_tasks', len(results_df))}",
            f"n_rows: {run_summary.get('n_rows', len(results_df))}",
            f"n_failures: {run_summary.get('n_failures', 0)}",
            f"workers: {run_summary.get('execution', {}).get('workers', 'n/a')}",
            "",
            "Interpretation notes:",
            "- PRIMARY excludes duplicate/mask-source flagged retrieval runs per config.",
            "- SENSITIVITY includes all audited rows.",
            "- Delta tables compare each tool mode against CLOSED on paired rows.",
        ]
        _page_text(pdf, title=title, lines=overview_lines)

        if run_summary.get("failures"):
            failure_lines = [json.dumps(item, sort_keys=True) for item in run_summary.get("failures", [])[:300]]
            _page_text(pdf, title="Execution Failures", lines=failure_lines)

        if run_summary.get("stall_events"):
            stall_lines = [json.dumps(item, sort_keys=True) for item in run_summary.get("stall_events", [])[:500]]
            _page_text(pdf, title="Stall/Long-Running Events", lines=stall_lines)

        _page_table(pdf, title="Summary By Tool Mode (PRIMARY)", df=summary_primary)
        _page_table(pdf, title="Summary By Tool Mode (SENSITIVITY)", df=summary_sensitivity)
        _page_table(pdf, title="Delta Vs CLOSED (PRIMARY)", df=delta_summary)
        _page_table(pdf, title="QC Failure Summary (PRIMARY)", df=qc_primary)
        _page_table(pdf, title="Retrieval Audit Summary", df=retrieval)
        _page_table(pdf, title="Analysis Partition Counts", df=partitions)

        _page_image(
            pdf,
            title="Primary Performance Plot",
            image_path=reports_dir / "primary_delta_plot.png",
            caption="Mean IoU bars with Mean Dice line by tool mode.",
        )
        _page_image(
            pdf,
            title="Primary QC Failure Plot",
            image_path=reports_dir / "primary_qc_failure_bar.png",
            caption="QC failure rate by tool mode.",
        )
        _page_image(
            pdf,
            title="Sensitivity Performance Plot",
            image_path=reports_dir / "sensitivity_delta_plot.png",
            caption="Sensitivity-set mean IoU/Dice by tool mode.",
        )
        _page_image(
            pdf,
            title="Sensitivity QC Failure Plot",
            image_path=reports_dir / "sensitivity_qc_failure_bar.png",
            caption="Sensitivity-set QC failure rate by tool mode.",
        )

        if not results_df.empty:
            cols = [
                "image_name",
                "tool_mode",
                "iou",
                "dice",
                "precision",
                "recall",
                "qc_pass",
                "selected_attempt_index",
                "retrieval_duplicate",
                "retrieval_mask_source",
                "analysis_primary",
            ]
            cols = [c for c in cols if c in results_df.columns]
            _page_table(pdf, title="Per-Run Results (Excerpt)", df=results_df[cols].head(100), max_rows=28)

    return output_pdf


def _default_run_dir() -> Path:
    base = _coerce_path("results_nanobanana")
    if not base.exists():
        return base
    candidates = sorted([p for p in base.rglob("*") if p.is_dir() and (p / "run_summary.json").exists()], key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a NanoBanana study run into a PDF report.")
    parser.add_argument(
        "--run-dir",
        default=_safe_rel(_default_run_dir()),
        help="Run directory containing run_summary.json, results.csv, and reports/*.csv",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output PDF path (default: <run_dir>/reports/nanobanana_run_report.pdf)",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dir = _coerce_path(args.run_dir)
    if not (run_dir / "run_summary.json").exists():
        raise FileNotFoundError(f"Run summary not found under: {run_dir}")

    output_pdf = _coerce_path(args.output) if args.output else (run_dir / "reports" / "nanobanana_run_report.pdf")
    pdf_path = generate_pdf_report(run_dir=run_dir, output_pdf=output_pdf)
    print(f"PDF report written: {pdf_path}")


if __name__ == "__main__":
    main()
