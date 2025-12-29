from __future__ import annotations

"""Render fairness figures and tables for the manuscript."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

# Use a non-interactive backend so figure generation is reliable in headless runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from scipy.stats import bootstrap

from ..fairness import compute_fairness_statistics, summarize_groups
from ..types import FairnessResult, GroupSummary

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACTS_DIR = DEFAULT_ROOT / "artifacts" / "fairness"


@dataclass
class FigureArtifacts:
    pdf: Path
    png: Path


@dataclass
class TableArtifacts:
    csv: Path
    html: Path
    docx: Path


def _results_to_df(results: Iterable[FairnessResult]) -> pd.DataFrame:
    """Convert :class:`FairnessResult` records into a DataFrame."""

    return pd.DataFrame([r.__dict__ for r in results])


def _summaries_to_df(summaries: Iterable[GroupSummary]) -> pd.DataFrame:
    """Convert :class:`GroupSummary` records into a DataFrame."""

    rows = []
    for s in summaries:
        rows.append(
            {
                "group": s.group_name,
                "count": s.count,
                "mean_iou": s.mean_iou,
                "median_iou": s.median_iou,
                "ci_iou_lower": s.ci_iou.lower,
                "ci_iou_upper": s.ci_iou.upper,
                "mean_dice": s.mean_dice,
                "median_dice": s.median_dice,
                "ci_dice_lower": s.ci_dice.lower,
                "ci_dice_upper": s.ci_dice.upper,
                "success_rate": s.success_rate,
            }
        )
    return pd.DataFrame(rows)


def _load_csv(path: Path, description: str) -> pd.DataFrame:
    """Load a CSV file with a helpful error if it is missing."""

    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return pd.read_csv(path)


def _load_fairness_dir(fairness_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Load fairness CSV outputs from a run directory."""

    results_path = fairness_dir / "fairness_results.csv"
    summary_path = fairness_dir / "fairness_summary.csv"
    stats_path = fairness_dir / "fairness_stats.csv"

    results_df = _load_csv(results_path, "Fairness results")
    summary_df = _load_csv(summary_path, "Fairness summary")
    stats_df = _load_csv(stats_path, "Fairness statistics") if stats_path.exists() else pd.DataFrame()
    stats_payload: Dict[str, float] = stats_df.iloc[0].to_dict() if not stats_df.empty else {}
    return results_df, summary_df, stats_payload


def _tone_order(df: pd.DataFrame) -> List[str]:
    order = ["Light", "Dark"]
    present = [tone for tone in order if tone in df["tone_group"].unique().tolist()]
    remaining = [tone for tone in df["tone_group"].unique().tolist() if tone not in present]
    return present + remaining


def _render_histogram(ax: plt.Axes, df: pd.DataFrame) -> None:
    ita = df["ita"].dropna()
    ax.hist(ita, bins=20, color="#4c72b0", edgecolor="white")
    ax.axvline(28, color="red", linestyle="--", label="28° threshold")
    ax.set_title("ITA distribution")
    ax.set_xlabel("ITA (°)")
    ax.set_ylabel("Count")
    ax.legend()


def _render_success_bars(ax: plt.Axes, df: pd.DataFrame) -> None:
    order = _tone_order(df)
    counts = df.groupby(["tone_group", "success"]).size().unstack(fill_value=0)
    success = counts.get(True, pd.Series([0] * len(order), index=order)).reindex(order, fill_value=0)
    failure = counts.get(False, pd.Series([0] * len(order), index=order)).reindex(order, fill_value=0)

    ax.bar(order, success, color="#55a868", label="Success")
    ax.bar(order, failure, bottom=success, color="#c44e52", label="Failure")
    ax.set_ylabel("Images")
    ax.set_title("Success by tone group")
    ax.legend()


def _render_distribution(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    order = _tone_order(df)
    data = [df[df["tone_group"] == tone]["iou"].dropna().values for tone in order]
    positions = np.arange(len(order)) + 1

    if not any(len(values) for values in data):
        ax.text(0.5, 0.5, "No IoU data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks(positions)
        ax.set_xticklabels(order)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        return

    violin = ax.violinplot(data, positions=positions, showmeans=True, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#4c72b0")
        body.set_alpha(0.6)
    ax.boxplot(data, positions=positions, widths=0.2, patch_artist=True, boxprops={"facecolor": "white"})
    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.set_ylim(0, 1)


def render_figure2(df: pd.DataFrame, *, output_dir: Path, seed: int = 0) -> FigureArtifacts:
    """Render the four-panel fairness figure and write PNG/PDF outputs."""

    if df.empty:
        raise ValueError("Fairness results are empty; cannot render figure.")
    np.random.seed(seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _render_histogram(axes[0, 0], df)
    _render_success_bars(axes[0, 1], df)
    _render_distribution(axes[1, 0], df, "IoU by tone (all)")
    filtered = df[df["iou"] >= 0.5]
    _render_distribution(axes[1, 1], filtered, "IoU by tone (IoU ≥ 0.5)")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure2.pdf"
    png_path = output_dir / "figure2.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    LOGGER.info("Figure 2 saved to %s", output_dir)
    return FigureArtifacts(pdf=pdf_path, png=png_path)


def _format_ci(mean: float, lower: float, upper: float) -> str:
    return f"{mean:.3f} ({lower:.3f}, {upper:.3f})"


def _bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int = 0,
    n_resamples: int = 5000,
    method: str = "bca",
) -> Tuple[float, float]:
    """Compute a BCa confidence interval for a statistic with deterministic sampling."""

    clean = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if clean.size < 2:
        return float("nan"), float("nan")

    try:
        res = bootstrap(
            (clean,),
            statistic,
            vectorized=False,
            method=method,
            n_resamples=n_resamples,
            confidence_level=0.95,
            random_state=np.random.default_rng(seed),
        )
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.warning("Falling back to percentile CI for %s: %s", statistic.__name__, exc)
        rng = np.random.default_rng(seed)
        samples = [statistic(rng.choice(clean, size=clean.size, replace=True)) for _ in range(n_resamples)]
        return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _group_stats(results_df: pd.DataFrame, group: str, *, seed: int = 0) -> Mapping[str, Tuple[float, float, float]]:
    """Compute mean/median IoU and Dice metrics with BCa CIs for a tone group."""

    group_df = results_df[results_df["tone_group"] == group]
    if group_df.empty:
        raise ValueError(f"No data available for group '{group}'")

    ious = group_df["iou"].dropna().to_numpy()
    dices = group_df["dice"].dropna().to_numpy()

    stats: Dict[str, Tuple[float, float, float]] = {
        "mean_iou": (float(np.mean(ious)) if ious.size else float("nan"),)
        + _bootstrap_ci(ious, np.mean, seed=seed),
        "median_iou": (float(np.median(ious)) if ious.size else float("nan"),)
        + _bootstrap_ci(ious, np.median, seed=seed),
        "mean_dice": (float(np.mean(dices)) if dices.size else float("nan"),)
        + _bootstrap_ci(dices, np.mean, seed=seed),
        "median_dice": (float(np.median(dices)) if dices.size else float("nan"),)
        + _bootstrap_ci(dices, np.median, seed=seed),
    }
    return stats


def render_table4(
    *,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stats_payload: Dict[str, float],
    output_dir: Path,
    seed: int = 0,
) -> TableArtifacts:
    """Render Table 4 (summary statistics and tests) to CSV/HTML/DOCX."""

    if results_df.empty:
        raise ValueError("Fairness results are empty; cannot render table.")

    groups = set(results_df["tone_group"].unique())
    if not {"Light", "Dark"}.issubset(groups):
        raise ValueError("Both Light and Dark groups are required for Table 4.")

    light_stats = _group_stats(results_df, "Light", seed=seed)
    dark_stats = _group_stats(results_df, "Dark", seed=seed)

    rows = []
    rows.append(
        {
            "metric": "IoU mean",
            "Light": _format_ci(*light_stats["mean_iou"]),
            "Dark": _format_ci(*dark_stats["mean_iou"]),
            "kruskal_p": stats_payload.get("kruskal_iou_p", np.nan),
            "cliffs_delta": stats_payload.get("cliffs_delta_iou_light_dark", np.nan),
            "chi2": stats_payload.get("chi2_success", np.nan),
            "chi2_p": stats_payload.get("chi2_success_p", np.nan),
        }
    )
    rows.append(
        {
            "metric": "IoU median",
            "Light": _format_ci(*light_stats["median_iou"]),
            "Dark": _format_ci(*dark_stats["median_iou"]),
            "kruskal_p": stats_payload.get("kruskal_iou_p", np.nan),
            "cliffs_delta": stats_payload.get("cliffs_delta_iou_light_dark", np.nan),
            "chi2": stats_payload.get("chi2_success", np.nan),
            "chi2_p": stats_payload.get("chi2_success_p", np.nan),
        }
    )
    rows.append(
        {
            "metric": "Dice mean",
            "Light": _format_ci(*light_stats["mean_dice"]),
            "Dark": _format_ci(*dark_stats["mean_dice"]),
            "kruskal_p": stats_payload.get("kruskal_dice_p", np.nan),
            "cliffs_delta": np.nan,
            "chi2": stats_payload.get("chi2_success", np.nan),
            "chi2_p": stats_payload.get("chi2_success_p", np.nan),
        }
    )
    rows.append(
        {
            "metric": "Dice median",
            "Light": _format_ci(*light_stats["median_dice"]),
            "Dark": _format_ci(*dark_stats["median_dice"]),
            "kruskal_p": stats_payload.get("kruskal_dice_p", np.nan),
            "cliffs_delta": np.nan,
            "chi2": stats_payload.get("chi2_success", np.nan),
            "chi2_p": stats_payload.get("chi2_success_p", np.nan),
        }
    )

    table_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table4.csv"
    html_path = output_dir / "table4.html"
    docx_path = output_dir / "table4.docx"

    table_df.to_csv(csv_path, index=False)
    table_df.to_html(html_path, index=False)

    document = Document()
    document.add_heading("Table 4: Fairness metrics", 0)
    table = document.add_table(rows=1, cols=len(table_df.columns))
    for idx, col in enumerate(table_df.columns):
        table.rows[0].cells[idx].text = str(col)
    for _, row in table_df.iterrows():
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = str(value)
    document.save(docx_path)
    LOGGER.info("Table 4 saved to %s", output_dir)
    return TableArtifacts(csv=csv_path, html=html_path, docx=docx_path)


def generate_fairness_artifacts(
    *,
    fairness_dir: Optional[Path] = None,
    results: Optional[Sequence[FairnessResult]] = None,
    summaries: Optional[Sequence[GroupSummary]] = None,
    stats_payload: Optional[Dict[str, float]] = None,
    output_dir: Path = DEFAULT_ARTIFACTS_DIR,
    seed: int = 0,
) -> Tuple[FigureArtifacts, TableArtifacts]:
    """Build Figure 2 and Table 4 from fairness CSVs or in-memory results."""

    if fairness_dir is None and results is None:
        raise ValueError("Provide either fairness_dir or in-memory results.")

    if fairness_dir:
        LOGGER.info("Loading fairness artifacts from %s", fairness_dir)
        results_df, summary_df, stats_payload_from_disk = _load_fairness_dir(fairness_dir)
        stats_payload = stats_payload or stats_payload_from_disk
    else:
        if results is None:
            raise ValueError("Results are required when fairness_dir is not provided.")
        if summaries is None:
            summaries = summarize_groups(results)
        if stats_payload is None:
            stats_payload = compute_fairness_statistics(list(results), 0.5)
        results_df = _results_to_df(results)
        summary_df = _summaries_to_df(summaries)

    stats_payload = stats_payload or {}
    figure_artifacts = render_figure2(results_df, output_dir=output_dir, seed=seed)
    table_artifacts = render_table4(
        results_df=results_df,
        summary_df=summary_df,
        stats_payload=stats_payload,
        output_dir=output_dir,
        seed=seed,
    )
    return figure_artifacts, table_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fairness figures and tables")
    parser.add_argument("--fairness-dir", type=Path, required=True, help="Directory containing fairness CSV outputs")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR, help="Destination for generated artifacts"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for deterministic plots (affects violin jitter)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    generate_fairness_artifacts(
        fairness_dir=args.fairness_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
