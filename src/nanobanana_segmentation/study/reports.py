from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from gemini_segmentation.metrics import calculate_bootstrap_ci


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _ensure_analysis_flags(
    results_df: pd.DataFrame,
    *,
    include_audit_unavailable_in_primary: bool,
) -> pd.DataFrame:
    df = results_df.copy()
    if "analysis_primary" not in df.columns:
        primary = ~df["retrieval_duplicate"].astype(bool) & ~df["retrieval_mask_source"].astype(bool)
        if not include_audit_unavailable_in_primary:
            primary = primary & ~df["audit_unavailable"].astype(bool)
        df["analysis_primary"] = primary
    if "analysis_sensitivity" not in df.columns:
        df["analysis_sensitivity"] = True
    return df


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "tool_mode",
                "n",
                "mean_iou",
                "mean_dice",
                "mean_precision",
                "mean_recall",
                "qc_pass_rate",
                "iou_ci_lower",
                "iou_ci_upper",
                "dice_ci_lower",
                "dice_ci_upper",
            ]
        )

    summary = (
        df.groupby("tool_mode", dropna=False)
        .agg(
            n=("image_name", "count"),
            mean_iou=("iou", "mean"),
            mean_dice=("dice", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            qc_pass_rate=("qc_pass", "mean"),
        )
        .reset_index()
    )

    ci_rows = []
    for mode, group in df.groupby("tool_mode"):
        ci_iou = calculate_bootstrap_ci(group["iou"].tolist(), n_resamples=1000, method="percentile")
        ci_dice = calculate_bootstrap_ci(group["dice"].tolist(), n_resamples=1000, method="percentile")
        ci_rows.append(
            {
                "tool_mode": mode,
                "iou_ci_lower": ci_iou.lower,
                "iou_ci_upper": ci_iou.upper,
                "dice_ci_lower": ci_dice.lower,
                "dice_ci_upper": ci_dice.upper,
            }
        )
    ci_df = pd.DataFrame(ci_rows)
    return summary.merge(ci_df, on="tool_mode", how="left")


def _paired_delta_vs_closed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
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
                "closed_iou",
                "closed_dice",
                "delta_iou_vs_closed",
                "delta_dice_vs_closed",
            ]
        )

    closed = df[df["tool_mode"] == "closed"][["image_name", "replicate_idx", "iou", "dice"]].rename(
        columns={"iou": "closed_iou", "dice": "closed_dice"}
    )
    paired = df.merge(closed, on=["image_name", "replicate_idx"], how="left")
    paired["delta_iou_vs_closed"] = paired["iou"] - paired["closed_iou"]
    paired["delta_dice_vs_closed"] = paired["dice"] - paired["closed_dice"]
    return paired


def _qc_failure_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["tool_mode", "qc_fail_count", "total", "qc_fail_rate"])
    qc_failure = (
        df.assign(any_qc_fail=~df["qc_pass"])
        .groupby("tool_mode", dropna=False)
        .agg(qc_fail_count=("any_qc_fail", "sum"), total=("any_qc_fail", "count"))
        .reset_index()
    )
    qc_failure["qc_fail_rate"] = qc_failure["qc_fail_count"] / qc_failure["total"].clip(lower=1)
    return qc_failure


def _write_plots(summary: pd.DataFrame, qc_failure: pd.DataFrame, *, out_prefix: Path) -> Tuple[Path, Path]:
    delta_path = out_prefix.with_name(out_prefix.name + "_delta_plot.png")
    qc_path = out_prefix.with_name(out_prefix.name + "_qc_failure_bar.png")

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ordered = summary.sort_values("tool_mode")
    ax.bar(ordered["tool_mode"], ordered["mean_iou"], color="#2f7ed8", label="Mean IoU")
    ax.plot(ordered["tool_mode"], ordered["mean_dice"], marker="o", color="#d84f2f", label="Mean Dice")
    ax.set_title("NanoBanana Tool-Mode Performance")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(delta_path, dpi=180)
    plt.close(fig)

    fig2 = plt.figure(figsize=(8, 4))
    ax2 = fig2.add_subplot(111)
    qordered = qc_failure.sort_values("tool_mode")
    ax2.bar(qordered["tool_mode"], qordered["qc_fail_rate"], color="#d84f2f")
    ax2.set_title("QC Failure Rate by Tool Mode")
    ax2.set_ylabel("Failure Rate")
    ax2.set_ylim(0, 1)
    fig2.tight_layout()
    fig2.savefig(qc_path, dpi=180)
    plt.close(fig2)
    return delta_path, qc_path


def _write_partition_reports(df: pd.DataFrame, out_dir: Path, *, suffix: str) -> Dict[str, str]:
    summary = _summary_table(df)
    paired = _paired_delta_vs_closed(df)
    qc_failure = _qc_failure_table(df)

    summary_path = out_dir / f"summary_by_mode_{suffix}.csv"
    paired_path = out_dir / f"paired_delta_vs_closed_{suffix}.csv"
    qc_path = out_dir / f"qc_failure_summary_{suffix}.csv"
    _atomic_write_csv(summary, summary_path)
    _atomic_write_csv(paired, paired_path)
    _atomic_write_csv(qc_failure, qc_path)

    delta_plot, qc_plot = _write_plots(summary, qc_failure, out_prefix=out_dir / suffix)
    return {
        f"summary_{suffix}": str(summary_path),
        f"paired_delta_{suffix}": str(paired_path),
        f"qc_failure_{suffix}": str(qc_path),
        f"delta_plot_{suffix}": str(delta_plot),
        f"qc_plot_{suffix}": str(qc_plot),
    }


def build_reports(
    results_df: pd.DataFrame,
    out_dir: Path,
    *,
    include_audit_unavailable_in_primary: bool = True,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_df = _ensure_analysis_flags(
        results_df,
        include_audit_unavailable_in_primary=include_audit_unavailable_in_primary,
    )

    primary_df = full_df[full_df["analysis_primary"].astype(bool)].copy()
    sensitivity_df = full_df[full_df["analysis_sensitivity"].astype(bool)].copy()

    retrieval = (
        full_df.groupby("tool_mode", dropna=False)
        .agg(
            duplicate_rate=("retrieval_duplicate", "mean"),
            mask_source_rate=("retrieval_mask_source", "mean"),
            audit_unavailable_rate=("audit_unavailable", "mean"),
            audit_unavailable_count=("audit_unavailable", "sum"),
            n=("image_name", "count"),
        )
        .reset_index()
    )
    retrieval_path = out_dir / "retrieval_audit_summary.csv"
    _atomic_write_csv(retrieval, retrieval_path)

    partition_counts = (
        full_df.groupby("tool_mode", dropna=False)
        .agg(
            n_total=("image_name", "count"),
            n_primary=("analysis_primary", "sum"),
            n_sensitivity=("analysis_sensitivity", "sum"),
            n_audit_unavailable=("audit_unavailable", "sum"),
        )
        .reset_index()
    )
    partition_counts_path = out_dir / "analysis_partition_counts.csv"
    _atomic_write_csv(partition_counts, partition_counts_path)

    outputs: Dict[str, str] = {}
    outputs.update(_write_partition_reports(primary_df, out_dir, suffix="primary"))
    outputs.update(_write_partition_reports(sensitivity_df, out_dir, suffix="sensitivity"))
    outputs["retrieval"] = str(retrieval_path)
    outputs["analysis_partitions"] = str(partition_counts_path)

    # Backward-compatible aliases point to primary analysis outputs.
    outputs["summary"] = outputs["summary_primary"]
    outputs["paired_delta"] = outputs["paired_delta_primary"]
    outputs["qc_failure"] = outputs["qc_failure_primary"]
    outputs["delta_plot"] = outputs["delta_plot_primary"]
    outputs["qc_plot"] = outputs["qc_plot_primary"]
    return outputs
