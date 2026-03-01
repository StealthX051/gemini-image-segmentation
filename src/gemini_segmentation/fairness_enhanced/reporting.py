from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COVARIATE_LABELS = {
    "lesion_area_frac": "Lesion area fraction",
    "deltaE_lesion_skin": "Lesion-skin color contrast (Delta E)",
    "Lstar_skin_mean": "Perilesional brightness mean (L*)",
    "Lstar_skin_std": "Perilesional brightness variability (L* SD)",
    "sharpness_laplacian_var": "Image sharpness (variance of Laplacian)",
    "hair_frac": "Hair/occlusion fraction",
    "specular_frac": "Specular highlight fraction",
}


def _legend_if_labeled(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if any(str(label).strip() for label in labels):
        ax.legend(loc="best")


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    out = {
        "root": base_dir,
        "cache": base_dir / "cache",
        "covariates_qc": base_dir / "covariates_qc_plots",
        "duplicate_examples": base_dir / "duplicate_examples",
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def write_analysis_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_endpoint_effects(table: pd.DataFrame, payload: Dict[str, object], out_dir: Path) -> None:
    table.to_csv(out_dir / "endpoint_effects_table.csv", index=False)
    write_json(out_dir / "endpoint_effects.json", payload)


def write_ita_bins(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    grouped = (
        df.groupby("ita_bin6", dropna=False)
        .agg(
            n=("image_name", "count"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    grouped.to_csv(out_dir / "ita_bins_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(grouped["ita_bin6"], grouped["n"], color="#3f6ca8")
    ax.set_xlabel("ITA 6-bin")
    ax.set_ylabel("Count")
    ax.set_title("ITA Bin Distribution")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "ita_bins_plot.png", dpi=200)
    plt.close(fig)
    return grouped


def write_threshold_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_csv(out_dir / "threshold_sensitivity_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(df["threshold"], df["rd"], marker="o", color="#9c2f2f", label="RD")
    if {"rd_ci_lower", "rd_ci_upper"}.issubset(df.columns):
        ax.fill_between(df["threshold"], df["rd_ci_lower"], df["rd_ci_upper"], color="#9c2f2f", alpha=0.2)
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("IoU success threshold")
    ax.set_ylabel("Risk difference (Lower - Higher ITA)")
    ax.set_title("Threshold Sensitivity")
    ax.grid(alpha=0.25, linestyle=":")
    _legend_if_labeled(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_sensitivity.png", dpi=200)
    plt.close(fig)


def write_trend_outputs(
    *,
    success_df: pd.DataFrame,
    iou_df: pd.DataFrame,
    iou_summary: Dict[str, object],
    out_dir: Path,
) -> None:
    success_df.to_csv(out_dir / "ita_trend_success.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name, part in success_df.groupby("model"):
        ax.plot(part["ita_deg"], part["pred"], label=str(model_name))
        if {"ci_lower", "ci_upper"}.issubset(part.columns):
            ax.fill_between(part["ita_deg"], part["ci_lower"], part["ci_upper"], alpha=0.18)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("ITA (degrees)")
    ax.set_ylabel("Predicted success probability")
    ax.set_title("Success vs Continuous ITA")
    ax.grid(alpha=0.25, linestyle=":")
    _legend_if_labeled(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "ita_trend_success.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name, part in iou_df.groupby("model"):
        ax.plot(part["ita_deg"], part["pred"], label=str(model_name))
        if {"ci_lower", "ci_upper"}.issubset(part.columns):
            ax.fill_between(part["ita_deg"], part["ci_lower"], part["ci_upper"], alpha=0.18)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("ITA (degrees)")
    ax.set_ylabel("Predicted median IoU")
    ax.set_title("Median IoU vs Continuous ITA")
    ax.grid(alpha=0.25, linestyle=":")
    _legend_if_labeled(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "ita_trend_iou_median.png", dpi=200)
    plt.close(fig)

    write_json(out_dir / "ita_trend_iou_summary.json", iou_summary)


def write_covariate_qc(df: pd.DataFrame, qc_dir: Path) -> None:
    cov_cols = [
        "lesion_area_frac",
        "deltaE_lesion_skin",
        "Lstar_skin_mean",
        "Lstar_skin_std",
        "sharpness_laplacian_var",
        "hair_frac",
        "specular_frac",
    ]
    for col in cov_cols:
        if col not in df.columns:
            continue
        values = df[col].astype(float)
        label = _COVARIATE_LABELS.get(col, col)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.hist(values[~np.isnan(values)], bins=40, color="#5f7f9f", edgecolor="white")
        ax.set_title(f"Histogram: {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(qc_dir / f"{col}_hist.png", dpi=180)
        plt.close(fig)

    missing = []
    for col in cov_cols:
        if col in df.columns:
            missing.append({"column": col, "missing_frac": float(df[col].isna().mean())})
    pd.DataFrame(missing).to_csv(qc_dir / "covariate_missingness.csv", index=False)


def write_dedup_outputs(
    *,
    dedup_map_exact: pd.DataFrame,
    dedup_report: pd.DataFrame,
    dedup_map_near: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    dedup_map_exact.to_csv(out_dir / "dedup_map_exact.csv", index=False)
    dedup_report.to_csv(out_dir / "dedup_report.csv", index=False)
    if dedup_map_near is not None and not dedup_map_near.empty:
        dedup_map_near.to_csv(out_dir / "dedup_map_near.csv", index=False)
