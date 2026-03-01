from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CIResult:
    estimate: float
    lower: float
    upper: float
    method_used: str
    warning: str | None = None


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return math.nan
    diff = a[:, None] - b[None, :]
    wins = np.sum(diff > 0)
    losses = np.sum(diff < 0)
    return float((wins - losses) / float(a.size * b.size))


def _percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.percentile(samples, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _bootstrap_two_group_percentile(
    left: np.ndarray,
    right: np.ndarray,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_resamples)):
        ls = rng.choice(left, size=len(left), replace=True)
        rs = rng.choice(right, size=len(right), replace=True)
        boot.append(float(statistic(ls, rs)))
    est = float(statistic(left, right))
    lo, hi = _percentile_ci(np.asarray(boot, dtype=float))
    return est, lo, hi


def bootstrap_two_group_ci(
    left: Iterable[float],
    right: Iterable[float],
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    method: str,
    fallback_method: str,
    n_resamples: int,
    seed: int,
) -> CIResult:
    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        est = float(statistic(a, b)) if a.size and b.size else math.nan
        return CIResult(est, math.nan, math.nan, fallback_method, "insufficient_sample")

    chosen = (method or "bca").strip().lower()

    if chosen == "bca":
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = stats.bootstrap(
                    (a, b),
                    statistic,
                    paired=False,
                    vectorized=False,
                    method="BCa",
                    n_resamples=int(n_resamples),
                    confidence_level=0.95,
                    random_state=int(seed),
                )
                warn_msg = None
                if caught:
                    warn_msg = "; ".join(str(w.message) for w in caught)
                est = float(statistic(a, b))
                return CIResult(
                    estimate=est,
                    lower=float(res.confidence_interval.low),
                    upper=float(res.confidence_interval.high),
                    method_used="bca",
                    warning=warn_msg,
                )
        except Exception as exc:
            est, lo, hi = _bootstrap_two_group_percentile(
                a,
                b,
                statistic=statistic,
                n_resamples=n_resamples,
                seed=seed,
            )
            return CIResult(est, lo, hi, fallback_method, f"bca_failed:{exc}")

    est, lo, hi = _bootstrap_two_group_percentile(
        a,
        b,
        statistic=statistic,
        n_resamples=n_resamples,
        seed=seed,
    )
    return CIResult(est, lo, hi, fallback_method)


def _risk_stats(left_success: np.ndarray, right_success: np.ndarray) -> Dict[str, float]:
    p_left = float(np.mean(left_success)) if left_success.size else math.nan
    p_right = float(np.mean(right_success)) if right_success.size else math.nan
    rd = p_left - p_right if not (math.isnan(p_left) or math.isnan(p_right)) else math.nan
    rr = (p_left / p_right) if p_right not in (0.0, math.nan) else math.nan

    a = float(np.sum(left_success))
    b = float(len(left_success) - a)
    c = float(np.sum(right_success))
    d = float(len(right_success) - c)
    # Haldane-Anscombe correction for zero cells.
    if min(a, b, c, d) == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
    odds_left = a / b
    odds_right = c / d
    or_val = odds_left / odds_right
    return {"rd": rd, "rr": rr, "or": or_val}


def compute_endpoint_effects(
    df: pd.DataFrame,
    *,
    group_col: str = "ita_binary",
    lower_label: str = "Lower ITA",
    higher_label: str = "Higher ITA",
    success_col: str = "success_t050",
    iou_col: str = "iou",
    bootstrap_method: str = "bca",
    bootstrap_fallback_method: str = "percentile",
    n_resamples: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object], List[str]]:
    warnings_out: List[str] = []

    working = df.copy()
    g_low = working[working[group_col] == lower_label]
    g_high = working[working[group_col] == higher_label]

    low_iou = g_low[iou_col].astype(float).to_numpy()
    high_iou = g_high[iou_col].astype(float).to_numpy()
    low_s = g_low[success_col].astype(bool).to_numpy()
    high_s = g_high[success_col].astype(bool).to_numpy()

    summary = {
        "lower_n": int(len(g_low)),
        "higher_n": int(len(g_high)),
        "lower_mean_iou": float(np.nanmean(low_iou)) if len(low_iou) else math.nan,
        "higher_mean_iou": float(np.nanmean(high_iou)) if len(high_iou) else math.nan,
        "lower_median_iou": float(np.nanmedian(low_iou)) if len(low_iou) else math.nan,
        "higher_median_iou": float(np.nanmedian(high_iou)) if len(high_iou) else math.nan,
        "lower_success_rate": float(np.mean(low_s)) if len(low_s) else math.nan,
        "higher_success_rate": float(np.mean(high_s)) if len(high_s) else math.nan,
    }

    cliffs = cliffs_delta(low_iou, high_iou)

    diff_median = bootstrap_two_group_ci(
        low_iou,
        high_iou,
        statistic=lambda a, b: float(np.nanmedian(a) - np.nanmedian(b)),
        method=bootstrap_method,
        fallback_method=bootstrap_fallback_method,
        n_resamples=n_resamples,
        seed=seed,
    )
    if diff_median.warning:
        warnings_out.append(f"median_diff:{diff_median.warning}")

    diff_mean = bootstrap_two_group_ci(
        low_iou,
        high_iou,
        statistic=lambda a, b: float(np.nanmean(a) - np.nanmean(b)),
        method=bootstrap_method,
        fallback_method=bootstrap_fallback_method,
        n_resamples=n_resamples,
        seed=seed + 1,
    )
    if diff_mean.warning:
        warnings_out.append(f"mean_diff:{diff_mean.warning}")

    rd_ci = bootstrap_two_group_ci(
        low_s.astype(float),
        high_s.astype(float),
        statistic=lambda a, b: float(np.mean(a) - np.mean(b)),
        method=bootstrap_method,
        fallback_method=bootstrap_fallback_method,
        n_resamples=n_resamples,
        seed=seed + 2,
    )
    rr_ci = bootstrap_two_group_ci(
        low_s.astype(float),
        high_s.astype(float),
        statistic=lambda a, b: float(np.mean(a) / np.mean(b)) if float(np.mean(b)) > 0 else math.nan,
        method=bootstrap_method,
        fallback_method=bootstrap_fallback_method,
        n_resamples=n_resamples,
        seed=seed + 3,
    )
    or_ci = bootstrap_two_group_ci(
        low_s.astype(float),
        high_s.astype(float),
        statistic=lambda a, b: _risk_stats(a > 0.5, b > 0.5)["or"],
        method=bootstrap_method,
        fallback_method=bootstrap_fallback_method,
        n_resamples=n_resamples,
        seed=seed + 4,
    )
    for key, payload in (("rd", rd_ci), ("rr", rr_ci), ("or", or_ci)):
        if payload.warning:
            warnings_out.append(f"{key}:{payload.warning}")

    mannwhitney_p = math.nan
    if len(low_iou) >= 2 and len(high_iou) >= 2:
        try:
            mannwhitney_p = float(stats.mannwhitneyu(low_iou, high_iou, alternative="two-sided").pvalue)
        except Exception:
            mannwhitney_p = math.nan

    contingency = pd.crosstab(
        pd.Series([lower_label] * len(low_s) + [higher_label] * len(high_s), name="group"),
        pd.Series(low_s.tolist() + high_s.tolist(), name="success"),
    )
    chi2_p = math.nan
    fisher_p = math.nan
    if contingency.shape == (2, 2):
        try:
            _, chi2_p, _, expected = stats.chi2_contingency(contingency)
            if np.any(expected < 5):
                _, fisher_p = stats.fisher_exact(contingency.to_numpy())
        except Exception:
            chi2_p = math.nan
            fisher_p = math.nan

    effect_rows = [
        {
            "metric": "cliffs_delta_iou_lower_vs_higher",
            "estimate": cliffs,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "ci_method": "na",
            "p_value": mannwhitney_p,
        },
        {
            "metric": "median_iou_diff_lower_minus_higher",
            "estimate": diff_median.estimate,
            "ci_lower": diff_median.lower,
            "ci_upper": diff_median.upper,
            "ci_method": diff_median.method_used,
            "p_value": mannwhitney_p,
        },
        {
            "metric": "mean_iou_diff_lower_minus_higher",
            "estimate": diff_mean.estimate,
            "ci_lower": diff_mean.lower,
            "ci_upper": diff_mean.upper,
            "ci_method": diff_mean.method_used,
            "p_value": mannwhitney_p,
        },
        {
            "metric": "success_risk_difference_lower_minus_higher",
            "estimate": rd_ci.estimate,
            "ci_lower": rd_ci.lower,
            "ci_upper": rd_ci.upper,
            "ci_method": rd_ci.method_used,
            "p_value": fisher_p if not math.isnan(fisher_p) else chi2_p,
        },
        {
            "metric": "success_relative_risk_lower_over_higher",
            "estimate": rr_ci.estimate,
            "ci_lower": rr_ci.lower,
            "ci_upper": rr_ci.upper,
            "ci_method": rr_ci.method_used,
            "p_value": fisher_p if not math.isnan(fisher_p) else chi2_p,
        },
        {
            "metric": "success_odds_ratio_lower_over_higher",
            "estimate": or_ci.estimate,
            "ci_lower": or_ci.lower,
            "ci_upper": or_ci.upper,
            "ci_method": or_ci.method_used,
            "p_value": fisher_p if not math.isnan(fisher_p) else chi2_p,
        },
    ]

    payload = {
        "grouping": {"group_col": group_col, "lower_label": lower_label, "higher_label": higher_label},
        "summary": summary,
        "tests": {
            "mannwhitney_iou_p": mannwhitney_p,
            "chi2_success_p": chi2_p,
            "fisher_success_p": fisher_p,
        },
        "warnings": warnings_out,
    }
    return pd.DataFrame(effect_rows), payload, warnings_out


def threshold_sensitivity_table(
    df: pd.DataFrame,
    *,
    thresholds: Iterable[float],
    group_col: str,
    lower_label: str,
    higher_label: str,
    iou_col: str,
    bootstrap_method: str,
    bootstrap_fallback_method: str,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for idx, threshold in enumerate(thresholds):
        working = df.copy()
        working["_success"] = working[iou_col].astype(float) >= float(threshold)
        g_low = working[working[group_col] == lower_label]["_success"].astype(float).to_numpy()
        g_high = working[working[group_col] == higher_label]["_success"].astype(float).to_numpy()

        rd_ci = bootstrap_two_group_ci(
            g_low,
            g_high,
            statistic=lambda a, b: float(np.mean(a) - np.mean(b)),
            method=bootstrap_method,
            fallback_method=bootstrap_fallback_method,
            n_resamples=n_resamples,
            seed=seed + idx,
        )
        rr_ci = bootstrap_two_group_ci(
            g_low,
            g_high,
            statistic=lambda a, b: float(np.mean(a) / np.mean(b)) if float(np.mean(b)) > 0 else math.nan,
            method=bootstrap_method,
            fallback_method=bootstrap_fallback_method,
            n_resamples=n_resamples,
            seed=seed + 100 + idx,
        )
        or_ci = bootstrap_two_group_ci(
            g_low,
            g_high,
            statistic=lambda a, b: _risk_stats(a > 0.5, b > 0.5)["or"],
            method=bootstrap_method,
            fallback_method=bootstrap_fallback_method,
            n_resamples=n_resamples,
            seed=seed + 200 + idx,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "rd": rd_ci.estimate,
                "rd_ci_lower": rd_ci.lower,
                "rd_ci_upper": rd_ci.upper,
                "rr": rr_ci.estimate,
                "rr_ci_lower": rr_ci.lower,
                "rr_ci_upper": rr_ci.upper,
                "or": or_ci.estimate,
                "or_ci_lower": or_ci.lower,
                "or_ci_upper": or_ci.upper,
            }
        )

    return pd.DataFrame(rows)
