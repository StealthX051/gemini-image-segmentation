from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer


def _clean_series(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr


def _nan_fill_matrix(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy().astype(float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    for col_idx in range(out.shape[1]):
        col = out[:, col_idx]
        mask = np.isnan(col)
        if np.any(mask):
            fill = float(np.nanmedian(col)) if np.any(~mask) else 0.0
            col[mask] = fill
            out[:, col_idx] = col
    return out


def _ita_grid(ita: np.ndarray, n_points: int = 120) -> np.ndarray:
    clean = ita[~np.isnan(ita)]
    if clean.size == 0:
        return np.linspace(-40.0, 80.0, n_points)
    lo = float(np.min(clean))
    hi = float(np.max(clean))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    return np.linspace(lo, hi, n_points)


def _fit_logistic_predictions(
    *,
    ita: np.ndarray,
    outcome: np.ndarray,
    grid: np.ndarray,
    knots: int,
    degree: int,
    covariates: np.ndarray | None = None,
) -> np.ndarray:
    spline = SplineTransformer(
        n_knots=max(3, int(knots)),
        degree=max(1, int(degree)),
        include_bias=False,
    )
    x_ita = ita.reshape(-1, 1)
    x_spline = spline.fit_transform(x_ita)

    if covariates is not None:
        x_cov = _nan_fill_matrix(covariates)
        x_train = np.hstack([x_spline, x_cov])
    else:
        x_train = x_spline

    y = (outcome > 0).astype(int)
    if y.size == 0:
        return np.full(len(grid), np.nan, dtype=float)

    unique = np.unique(y)
    if unique.size < 2:
        # Degenerate class labels can occur on tiny fixture subsets/resamples.
        # Return the observed empirical class probability instead of fitting.
        return np.full(len(grid), float(unique[0]), dtype=float)

    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=0,
    )
    model.fit(x_train, y)

    g_spline = spline.transform(grid.reshape(-1, 1))
    if covariates is not None:
        med = np.nanmedian(covariates, axis=0)
        g_cov = np.tile(med, (len(grid), 1))
        g_cov = _nan_fill_matrix(g_cov)
        g_input = np.hstack([g_spline, g_cov])
    else:
        g_input = g_spline

    return model.predict_proba(g_input)[:, 1]


def success_trend_with_bands(
    *,
    df: pd.DataFrame,
    ita_col: str,
    outcome_col: str,
    covariate_cols: List[str] | None,
    knots: int,
    degree: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    work = df[[ita_col, outcome_col] + (covariate_cols or [])].copy()
    work = work.dropna(subset=[ita_col, outcome_col])
    if work.empty:
        return pd.DataFrame(columns=["ita_deg", "pred", "ci_lower", "ci_upper", "model"])

    ita = work[ita_col].astype(float).to_numpy()
    outcome = work[outcome_col].astype(float).to_numpy()
    cov = work[covariate_cols].astype(float).to_numpy() if covariate_cols else None

    grid = _ita_grid(ita)
    base_pred = _fit_logistic_predictions(
        ita=ita,
        outcome=outcome,
        grid=grid,
        knots=knots,
        degree=degree,
        covariates=cov,
    )

    rng = np.random.default_rng(seed)
    boots: List[np.ndarray] = []
    for _ in range(max(0, int(n_bootstrap))):
        idx = rng.choice(np.arange(len(work)), size=len(work), replace=True)
        try:
            boot_pred = _fit_logistic_predictions(
                ita=ita[idx],
                outcome=outcome[idx],
                grid=grid,
                knots=knots,
                degree=degree,
                covariates=(cov[idx] if cov is not None else None),
            )
            boots.append(boot_pred)
        except Exception:
            continue

    if boots:
        band = np.vstack(boots)
        lower = np.percentile(band, 2.5, axis=0)
        upper = np.percentile(band, 97.5, axis=0)
    else:
        lower = np.full_like(base_pred, np.nan, dtype=float)
        upper = np.full_like(base_pred, np.nan, dtype=float)

    model_name = "ita_plus_covariates" if covariate_cols else "ita_only"
    return pd.DataFrame(
        {
            "ita_deg": grid,
            "pred": base_pred,
            "ci_lower": lower,
            "ci_upper": upper,
            "model": model_name,
        }
    )


def _fit_quantile_predictions(
    *,
    ita: np.ndarray,
    iou: np.ndarray,
    grid: np.ndarray,
    knots: int,
    degree: int,
    quantile: float,
    alpha: float,
    covariates: np.ndarray | None = None,
) -> np.ndarray:
    spline = SplineTransformer(
        n_knots=max(3, int(knots)),
        degree=max(1, int(degree)),
        include_bias=False,
    )
    x_ita = ita.reshape(-1, 1)
    x_spline = spline.fit_transform(x_ita)

    if covariates is not None:
        x_cov = _nan_fill_matrix(covariates)
        x_train = np.hstack([x_spline, x_cov])
    else:
        x_train = x_spline

    # Use an IRLS-style quantile fit (pinball-loss approximation via asymmetric
    # weighted least squares) to avoid SciPy HiGHS native backend instability on
    # some Windows environments.
    def _fit_quantile_irls(
        x: np.ndarray,
        y: np.ndarray,
        q: float,
        ridge_alpha: float,
        max_iter: int = 200,
        tol: float = 1e-7,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = x.shape[0]
        design = np.hstack([np.ones((n, 1), dtype=float), x])
        p = design.shape[1]

        # No regularization on intercept.
        reg = np.eye(p, dtype=float)
        reg[0, 0] = 0.0

        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        q = float(np.clip(q, 1e-6, 1.0 - 1e-6))
        ridge_alpha = max(0.0, float(ridge_alpha))

        for _ in range(max_iter):
            residual = y - (design @ beta)
            weights = np.where(residual >= 0.0, q, 1.0 - q).astype(float)
            weights = np.clip(weights, 1e-6, None)
            sqrt_w = np.sqrt(weights)
            design_w = design * sqrt_w[:, None]
            y_w = y * sqrt_w

            lhs = design_w.T @ design_w
            if ridge_alpha > 0.0:
                lhs = lhs + ridge_alpha * reg
            rhs = design_w.T @ y_w
            try:
                beta_new = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            if float(np.max(np.abs(beta_new - beta))) < tol:
                beta = beta_new
                break
            beta = beta_new

        return beta

    g_spline = spline.transform(grid.reshape(-1, 1))
    if covariates is not None:
        med = np.nanmedian(covariates, axis=0)
        g_cov = np.tile(med, (len(grid), 1))
        g_cov = _nan_fill_matrix(g_cov)
        g_input = np.hstack([g_spline, g_cov])
    else:
        g_input = g_spline

    beta = _fit_quantile_irls(
        x=x_train,
        y=iou,
        q=quantile,
        ridge_alpha=alpha,
    )
    g_design = np.hstack([np.ones((g_input.shape[0], 1), dtype=float), g_input])
    pred = g_design @ beta
    return np.clip(pred.astype(float), 0.0, 1.0)


def iou_trend_with_bands(
    *,
    df: pd.DataFrame,
    ita_col: str,
    iou_col: str,
    covariate_cols: List[str] | None,
    knots: int,
    degree: int,
    quantile: float,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    work = df[[ita_col, iou_col] + (covariate_cols or [])].copy()
    work = work.dropna(subset=[ita_col, iou_col])
    if work.empty:
        return (
            pd.DataFrame(columns=["ita_deg", "pred", "ci_lower", "ci_upper", "model"]),
            {
                "model": "quantile_regression",
                "status": "empty",
            },
        )

    ita = work[ita_col].astype(float).to_numpy()
    iou = work[iou_col].astype(float).to_numpy()
    cov = work[covariate_cols].astype(float).to_numpy() if covariate_cols else None
    grid = _ita_grid(ita)

    base_pred = _fit_quantile_predictions(
        ita=ita,
        iou=iou,
        grid=grid,
        knots=knots,
        degree=degree,
        quantile=quantile,
        alpha=alpha,
        covariates=cov,
    )

    rng = np.random.default_rng(seed)
    boots: List[np.ndarray] = []
    for _ in range(max(0, int(n_bootstrap))):
        idx = rng.choice(np.arange(len(work)), size=len(work), replace=True)
        try:
            boot_pred = _fit_quantile_predictions(
                ita=ita[idx],
                iou=iou[idx],
                grid=grid,
                knots=knots,
                degree=degree,
                quantile=quantile,
                alpha=alpha,
                covariates=(cov[idx] if cov is not None else None),
            )
            boots.append(boot_pred)
        except Exception:
            continue

    if boots:
        band = np.vstack(boots)
        lower = np.percentile(band, 2.5, axis=0)
        upper = np.percentile(band, 97.5, axis=0)
    else:
        lower = np.full_like(base_pred, np.nan, dtype=float)
        upper = np.full_like(base_pred, np.nan, dtype=float)

    model_name = "ita_plus_covariates" if covariate_cols else "ita_only"
    trend_df = pd.DataFrame(
        {
            "ita_deg": grid,
            "pred": base_pred,
            "ci_lower": lower,
            "ci_upper": upper,
            "model": model_name,
        }
    )

    summary = {
        "model": "quantile_regression_irls",
        "status": "ok",
        "quantile": float(quantile),
        "alpha": float(alpha),
        "knots": int(knots),
        "degree": int(degree),
        "covariates": list(covariate_cols or []),
    }
    return trend_df, summary


def build_success_trend_frames(
    *,
    df: pd.DataFrame,
    ita_col: str,
    outcome_col: str,
    covariate_cols: List[str],
    knots: int,
    degree: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    parts = [
        success_trend_with_bands(
            df=df,
            ita_col=ita_col,
            outcome_col=outcome_col,
            covariate_cols=None,
            knots=knots,
            degree=degree,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    ]
    if covariate_cols:
        try:
            parts.append(
                success_trend_with_bands(
                    df=df,
                    ita_col=ita_col,
                    outcome_col=outcome_col,
                    covariate_cols=covariate_cols,
                    knots=knots,
                    degree=degree,
                    n_bootstrap=n_bootstrap,
                    seed=seed + 1000,
                )
            )
        except Exception as exc:
            logging.warning("Covariate-adjusted success trend failed: %s", exc)
    return pd.concat(parts, ignore_index=True)


def build_iou_trend_frames(
    *,
    df: pd.DataFrame,
    ita_col: str,
    iou_col: str,
    covariate_cols: List[str],
    knots: int,
    degree: int,
    quantile: float,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    base_df, base_summary = iou_trend_with_bands(
        df=df,
        ita_col=ita_col,
        iou_col=iou_col,
        covariate_cols=None,
        knots=knots,
        degree=degree,
        quantile=quantile,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    frames = [base_df]
    summary = {
        "base": base_summary,
        "covariate_adjusted": None,
    }
    if covariate_cols:
        try:
            adj_df, adj_summary = iou_trend_with_bands(
                df=df,
                ita_col=ita_col,
                iou_col=iou_col,
                covariate_cols=covariate_cols,
                knots=knots,
                degree=degree,
                quantile=quantile,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                seed=seed + 1000,
            )
            frames.append(adj_df)
            summary["covariate_adjusted"] = adj_summary
        except Exception as exc:
            logging.warning("Covariate-adjusted iou trend failed: %s", exc)
            summary["covariate_adjusted"] = {"status": "failed", "error": str(exc)}

    return pd.concat(frames, ignore_index=True), summary
