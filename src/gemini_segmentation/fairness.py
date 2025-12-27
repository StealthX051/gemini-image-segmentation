from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from cliffs_delta import cliffs_delta
from PIL import Image
from scipy import stats
from skimage import color

from .metrics import calculate_bootstrap_ci
from .types import FairnessResult, GroupSummary


CHARDON_THRESHOLDS = [55, 41, 28, 10, -30]
CHARDON_LABELS = [
    "Very Light",  # >55
    "Light",  # >41 to 55
    "Intermediate",  # >28 to 41
    "Tan",  # >10 to 28
    "Dark",  # >-30 to 10
    "Very Dark",  # <= -30
]


def _load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def _perilesional_mask(gt_mask: np.ndarray) -> np.ndarray:
    inverted = np.logical_not(gt_mask > 127)
    return inverted.astype(np.uint8)


def compute_ita(image: Image.Image, peri_mask: np.ndarray) -> Tuple[float, int]:
    lab = color.rgb2lab(np.array(image))
    L = lab[:, :, 0]
    b = lab[:, :, 2]

    # apply luminance windowing
    luminance = L * peri_mask
    valid = luminance[luminance > 0]
    if valid.size == 0:
        return math.nan, 0

    low_floor = np.percentile(valid, 5)
    high = np.percentile(valid, 95)
    candidate_mask = (luminance >= low_floor) & (luminance <= high)
    candidate_mask &= peri_mask.astype(bool)

    candidate_count = int(candidate_mask.sum())
    if candidate_count < 200 or candidate_count < 0.02 * peri_mask.size:
        return math.nan, candidate_count

    ita_values = np.arctan2(L[candidate_mask] - 50, b[candidate_mask]) * 180 / np.pi
    if ita_values.size == 0:
        return math.nan, candidate_count
    return float(np.median(ita_values)), candidate_count


def label_chardon(ita: float) -> str:
    if math.isnan(ita):
        return "Unknown"
    thresholds = CHARDON_THRESHOLDS + [-math.inf]
    for boundary, label in zip(thresholds, CHARDON_LABELS):
        if ita > boundary:
            return label
    return CHARDON_LABELS[-1]


def tone_group(ita: float) -> str:
    if math.isnan(ita):
        return "Unknown"
    return "Light" if ita > 28 else "Dark"


def summarize_groups(
    results: Iterable[FairnessResult], *, n_resamples: int = 5000, method: str = "bca"
) -> List[GroupSummary]:
    grouped: Dict[str, List[FairnessResult]] = defaultdict(list)
    for res in results:
        grouped[res.tone_group].append(res)

    summaries: List[GroupSummary] = []
    for group_name, members in grouped.items():
        ious = [m.iou for m in members if not math.isnan(m.iou)]
        dices = [m.dice for m in members if not math.isnan(m.dice)]
        count = len(members)
        summaries.append(
            GroupSummary(
                group_name=group_name,
                count=count,
                mean_iou=float(np.mean(ious)) if ious else 0.0,
                median_iou=float(np.median(ious)) if ious else 0.0,
                ci_iou=
                calculate_bootstrap_ci(ious, n_resamples=n_resamples, method=method)
                if ious
                else calculate_bootstrap_ci([], n_resamples=n_resamples, method=method),
                mean_dice=float(np.mean(dices)) if dices else 0.0,
                median_dice=float(np.median(dices)) if dices else 0.0,
                ci_dice=
                calculate_bootstrap_ci(dices, n_resamples=n_resamples, method=method)
                if dices
                else calculate_bootstrap_ci([], n_resamples=n_resamples, method=method),
                success_rate=float(np.mean([m.success for m in members])) if members else 0.0,
            )
        )
    return summaries


def analyze_fairness(
    *,
    image_mask_pairs: Iterable[Tuple[Path, Path]],
    prediction_masks_dir: Path,
    per_image_metrics: Dict[str, Tuple[float, float, bool]],
    success_threshold: float = 0.5,
    n_resamples: int = 5000,
    method: str = "bca",
) -> Tuple[List[FairnessResult], List[GroupSummary], Dict[str, float]]:
    results: List[FairnessResult] = []
    for img_path, gt_mask_path in image_mask_pairs:
        pred_mask_file = prediction_masks_dir / img_path.name
        if not pred_mask_file.exists():
            continue
        image = Image.open(img_path)
        gt_mask = _load_mask(gt_mask_path)
        peri_mask = _perilesional_mask(gt_mask)
        ita, candidate_count = compute_ita(image, peri_mask)
        if math.isnan(ita) or candidate_count < 200:
            continue

        chardon = label_chardon(ita)
        tone = tone_group(ita)
        iou, dice, success = per_image_metrics.get(img_path.name, (math.nan, math.nan, False))
        results.append(
            FairnessResult(
                image_name=img_path.name,
                ita=ita,
                chardon_label=chardon,
                tone_group=tone,
                iou=iou,
                dice=dice,
                success=success,
                candidate_count=candidate_count,
            )
        )

    summaries = summarize_groups(results, n_resamples=n_resamples, method=method)
    stats_payload = compute_fairness_statistics(results, success_threshold)
    return results, summaries, stats_payload


def compute_fairness_statistics(results: List[FairnessResult], success_threshold: float) -> Dict[str, float]:
    if not results:
        return {}

    df = pd.DataFrame([r.__dict__ for r in results])
    payload: Dict[str, float] = {}

    if df["tone_group"].nunique() >= 2:
        try:
            kw_iou = stats.kruskal(*[group["iou"].dropna() for _, group in df.groupby("tone_group")])
            kw_dice = stats.kruskal(*[group["dice"].dropna() for _, group in df.groupby("tone_group")])
            payload["kruskal_iou_p"] = float(kw_iou.pvalue)
            payload["kruskal_dice_p"] = float(kw_dice.pvalue)
        except Exception:
            payload["kruskal_iou_p"] = math.nan
            payload["kruskal_dice_p"] = math.nan

        try:
            dunn = sp.posthoc_dunn(df, val_col="iou", group_col="tone_group", p_adjust="holm")
            if "Light" in dunn.columns and "Dark" in dunn.index:
                payload["dunn_iou_light_dark_p"] = float(dunn.loc["Dark", "Light"])
        except Exception:
            payload["dunn_iou_light_dark_p"] = math.nan

        try:
            delta, _ = cliffs_delta(df[df["tone_group"] == "Light"]["iou"], df[df["tone_group"] == "Dark"]["iou"])
            payload["cliffs_delta_iou_light_dark"] = float(delta)
        except Exception:
            payload["cliffs_delta_iou_light_dark"] = math.nan

    contingency = pd.crosstab(df["tone_group"], df["success"])
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 1:
        try:
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            payload["chi2_success"] = float(chi2)
            payload["chi2_success_p"] = float(p)
        except Exception:
            payload["chi2_success"] = math.nan
            payload["chi2_success_p"] = math.nan

    return payload


def write_fairness_results(results: List[FairnessResult], path: Path) -> None:
    rows = [r.__dict__ for r in results]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_fairness_summary(summaries: List[GroupSummary], path: Path) -> None:
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_fairness_statistics(stats_payload: Dict[str, float], path: Path) -> None:
    if not stats_payload:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats_payload]).to_csv(path, index=False)
