from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class WritableArrayTransformer(BaseEstimator, TransformerMixin):
    """Force a writable ndarray copy while preserving feature-name passthrough."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(X, copy=True)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(list(input_features), dtype=object)


def _risk_effects(r_low: float, r_high: float) -> Dict[str, float]:
    rd = float(r_low - r_high)
    rr = float(r_low / r_high) if r_high > 0.0 else math.nan
    if 0.0 < r_low < 1.0 and 0.0 < r_high < 1.0:
        odds_low = r_low / (1.0 - r_low)
        odds_high = r_high / (1.0 - r_high)
        or_val = float(odds_low / odds_high)
    else:
        or_val = math.nan
    return {"rd_adj": rd, "rr_adj": rr, "or_adj": or_val}


def _safe_percentile_interval(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    return (
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    )


def _odds_ratio_from_coef(value: float) -> float:
    if not np.isfinite(value):
        return math.nan
    return float(math.exp(float(value)))


def _format_component_label(
    *,
    raw_feature: str,
    exposure_indicator_col: str,
    dataset_source_col: str,
) -> Tuple[str, str, str]:
    token = str(raw_feature)
    while "__" in token:
        token = token.split("__", 1)[1]

    if token == exposure_indicator_col:
        return (
            "Lower ITA indicator (vs Higher ITA)",
            "exposure",
            "binary indicator",
        )
    if token.startswith(f"{dataset_source_col}_"):
        level = token[len(dataset_source_col) + 1 :]
        return (
            f"Dataset source: {level} (vs reference)",
            "dataset_fixed_effect",
            "categorical contrast",
        )
    return (
        token,
        "covariate",
        "per 1 SD increase (standardized)",
    )


def _coef_significance(coef_lo: float, coef_hi: float) -> bool | None:
    if not np.isfinite(coef_lo) or not np.isfinite(coef_hi):
        return None
    return bool((coef_lo > 0.0) or (coef_hi < 0.0))


def _extract_feature_coefficients(model: Pipeline) -> Dict[str, float]:
    preprocess = model.named_steps.get("preprocess")
    clf = model.named_steps.get("model")
    if preprocess is None or clf is None or not hasattr(clf, "coef_"):
        return {}
    names = list(preprocess.get_feature_names_out())
    coef = np.asarray(clf.coef_, dtype=float)
    if coef.ndim != 2 or coef.shape[0] < 1:
        return {}
    coef_vec = coef[0]
    out: Dict[str, float] = {}
    for idx, name in enumerate(names):
        if idx >= len(coef_vec):
            break
        out[str(name)] = float(coef_vec[idx])
    return out


def _build_logistic_pipeline(
    *,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    seed: int,
    c_value: float,
    max_iter: int,
) -> Pipeline:
    # Some pandas backends can expose read-only views to sklearn transformers.
    # Force writable array copies before imputation/encoding to avoid transform-time failures.
    writable_copy = WritableArrayTransformer()
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("to_writable", writable_copy),
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_cols),
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("to_writable", writable_copy),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_cols),
            )
        )

    if not transformers:
        raise ValueError("No predictors available for covariate-adjusted logistic model.")

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=float(c_value),
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    return Pipeline([("preprocess", preprocess), ("model", clf)])


def _fit_predictive_margins(
    *,
    frame: pd.DataFrame,
    outcome_col: str,
    exposure_indicator_col: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    seed: int,
    c_value: float,
    max_iter: int,
) -> Tuple[Dict[str, float], Dict[str, object], Dict[str, float]]:
    y = pd.to_numeric(frame[outcome_col], errors="coerce").fillna(0.0).astype(float).to_numpy()
    y = (y > 0.0).astype(int)

    x_cols = list(numeric_cols) + list(categorical_cols)
    x = frame[x_cols].copy()

    unique = np.unique(y)
    if unique.size < 2:
        const_prob = float(unique[0]) if unique.size == 1 else math.nan
        margins = {"r_low_adj": const_prob, "r_high_adj": const_prob}
        margins.update(_risk_effects(const_prob, const_prob))
        spec = {
            "fit_status": "constant_class",
            "constant_outcome": int(unique[0]) if unique.size == 1 else None,
            "n_rows": int(len(frame)),
            "n_predictors": int(len(x_cols)),
        }
        return margins, spec, {}

    model = _build_logistic_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        seed=seed,
        c_value=c_value,
        max_iter=max_iter,
    )
    model.fit(x, y)
    coef_map = _extract_feature_coefficients(model)

    x_low = x.copy()
    x_high = x.copy()
    x_low[exposure_indicator_col] = 1.0
    x_high[exposure_indicator_col] = 0.0

    p_low = np.asarray(model.predict_proba(x_low)[:, 1], dtype=float)
    p_high = np.asarray(model.predict_proba(x_high)[:, 1], dtype=float)

    r_low = float(np.nanmean(p_low))
    r_high = float(np.nanmean(p_high))

    margins = {
        "r_low_adj": r_low,
        "r_high_adj": r_high,
    }
    margins.update(_risk_effects(r_low, r_high))
    spec = {
        "fit_status": "ok",
        "n_rows": int(len(frame)),
        "n_predictors": int(len(x_cols)),
        "n_numeric_predictors": int(len(numeric_cols)),
        "n_categorical_predictors": int(len(categorical_cols)),
        "n_model_features": int(len(coef_map)),
    }
    return margins, spec, coef_map


def compute_covariate_adjusted_success_effects(
    df: pd.DataFrame,
    *,
    outcome_col: str = "success_t050",
    exposure_col: str = "ita_binary",
    lower_label: str = "Lower ITA",
    higher_label: str = "Higher ITA",
    covariate_cols: Sequence[str] | None = None,
    include_dataset_source: bool = False,
    dataset_source_col: str = "dataset_source_primary",
    resample_unit_col: str | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
    c_value: float = 1_000_000.0,
    max_iter: int = 4000,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    covariates = [str(c) for c in (covariate_cols or []) if str(c) in df.columns]
    base_cols = [outcome_col, exposure_col] + covariates
    if include_dataset_source and dataset_source_col in df.columns:
        base_cols.append(dataset_source_col)
    if resample_unit_col and resample_unit_col in df.columns:
        base_cols.append(resample_unit_col)
    elif "dedup_group_id" in df.columns:
        resample_unit_col = "dedup_group_id"
        base_cols.append(resample_unit_col)
    elif "image_id" in df.columns:
        resample_unit_col = "image_id"
        base_cols.append(resample_unit_col)
    else:
        resample_unit_col = "image_name" if "image_name" in df.columns else None
        if resample_unit_col is not None:
            base_cols.append(resample_unit_col)

    if outcome_col not in df.columns or exposure_col not in df.columns:
        raise ValueError("Required columns missing for covariate-adjusted success effects.")

    work = df[base_cols].copy()
    work = work[work[exposure_col].isin([lower_label, higher_label])].copy()
    work = work.dropna(subset=[outcome_col])
    if work.empty:
        empty = pd.DataFrame(
            [
                {
                    "metric": "adjusted_rd_low_minus_high",
                    "estimate": math.nan,
                    "ci_lower": math.nan,
                    "ci_upper": math.nan,
                    "ci_method": "percentile_bootstrap",
                    "n_boot": 0,
                    "resample_unit": str(resample_unit_col or "none"),
                }
            ]
        )
        payload = {
            "status": "empty",
            "covariates_used": covariates,
            "resample_unit": str(resample_unit_col or "none"),
            "warnings": ["empty_input_after_filtering"],
        }
        return empty, payload, pd.DataFrame()

    for cov_col in covariates:
        work[cov_col] = pd.to_numeric(work[cov_col], errors="coerce")

    work["_exp_low"] = (work[exposure_col] == lower_label).astype(float)
    numeric_cols = ["_exp_low"] + covariates
    categorical_cols = [dataset_source_col] if include_dataset_source and dataset_source_col in work.columns else []

    margins, fit_spec, base_coef_map = _fit_predictive_margins(
        frame=work,
        outcome_col=outcome_col,
        exposure_indicator_col="_exp_low",
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        seed=seed,
        c_value=c_value,
        max_iter=max_iter,
    )

    y_low = pd.to_numeric(
        work.loc[work["_exp_low"] > 0.5, outcome_col],
        errors="coerce",
    ).fillna(0.0)
    y_high = pd.to_numeric(
        work.loc[work["_exp_low"] <= 0.5, outcome_col],
        errors="coerce",
    ).fillna(0.0)
    unadj_r_low = float((y_low > 0.0).mean()) if len(y_low) else math.nan
    unadj_r_high = float((y_high > 0.0).mean()) if len(y_high) else math.nan
    unadj_rd = float(unadj_r_low - unadj_r_high) if np.isfinite(unadj_r_low) and np.isfinite(unadj_r_high) else math.nan
    attenuation = (
        float(margins["rd_adj"] / unadj_rd)
        if np.isfinite(margins["rd_adj"]) and np.isfinite(unadj_rd) and abs(unadj_rd) > 1e-12
        else math.nan
    )

    boot_rows: List[Dict[str, float | int]] = []
    coef_boot_rows: List[Dict[str, float | int | str]] = []
    rng = np.random.default_rng(int(seed))
    units = None
    index_map: Dict[object, np.ndarray] = {}
    if resample_unit_col and resample_unit_col in work.columns:
        grouped = work.groupby(resample_unit_col, dropna=False).indices
        index_map = {k: np.asarray(v, dtype=int) for k, v in grouped.items()}
        units = np.array(list(index_map.keys()), dtype=object)

    for b in range(max(0, int(n_resamples))):
        if units is None or units.size == 0:
            sampled_idx = rng.choice(np.arange(len(work)), size=len(work), replace=True)
        else:
            sampled_units = rng.choice(units, size=len(units), replace=True)
            sampled_idx = np.concatenate([index_map[u] for u in sampled_units], axis=0)
        boot = work.iloc[sampled_idx].copy()
        try:
            boot_margins, _, boot_coef_map = _fit_predictive_margins(
                frame=boot,
                outcome_col=outcome_col,
                exposure_indicator_col="_exp_low",
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                seed=seed + 1000 + b,
                c_value=c_value,
                max_iter=max_iter,
            )
        except Exception:
            continue
        boot_rows.append(
            {
                "replicate": int(b),
                "r_low_adj": float(boot_margins.get("r_low_adj", math.nan)),
                "r_high_adj": float(boot_margins.get("r_high_adj", math.nan)),
                "rd_adj": float(boot_margins.get("rd_adj", math.nan)),
                "rr_adj": float(boot_margins.get("rr_adj", math.nan)),
                "or_adj": float(boot_margins.get("or_adj", math.nan)),
            }
        )
        for feat_name, feat_coef in boot_coef_map.items():
            coef_boot_rows.append(
                {
                    "replicate": int(b),
                    "feature": str(feat_name),
                    "coef": float(feat_coef),
                }
            )

    boot_df = pd.DataFrame(boot_rows)
    ci_lookup: Dict[str, Tuple[float, float]] = {}
    for metric in ("r_low_adj", "r_high_adj", "rd_adj", "rr_adj", "or_adj"):
        values = boot_df[metric].to_numpy(dtype=float) if metric in boot_df.columns else np.asarray([], dtype=float)
        ci_lookup[metric] = _safe_percentile_interval(values)

    coef_boot_df = pd.DataFrame(coef_boot_rows)
    component_rows: List[Dict[str, object]] = []
    base_feature_names = sorted(str(k) for k in base_coef_map.keys())
    for feature_name in base_feature_names:
        coef_est = float(base_coef_map.get(feature_name, math.nan))
        coef_samples = (
            pd.to_numeric(
                coef_boot_df.loc[coef_boot_df["feature"] == feature_name, "coef"],
                errors="coerce",
            ).to_numpy(dtype=float)
            if ("feature" in coef_boot_df.columns and "coef" in coef_boot_df.columns)
            else np.asarray([], dtype=float)
        )
        coef_ci_lo, coef_ci_hi = _safe_percentile_interval(coef_samples)
        or_est = _odds_ratio_from_coef(coef_est)
        or_ci_lo = _odds_ratio_from_coef(coef_ci_lo)
        or_ci_hi = _odds_ratio_from_coef(coef_ci_hi)
        label, comp_type, scale_note = _format_component_label(
            raw_feature=feature_name,
            exposure_indicator_col="_exp_low",
            dataset_source_col=dataset_source_col,
        )
        coef_sig = _coef_significance(coef_ci_lo, coef_ci_hi)
        or_sig = (
            None
            if not np.isfinite(or_ci_lo) or not np.isfinite(or_ci_hi)
            else bool((or_ci_lo > 1.0) or (or_ci_hi < 1.0))
        )
        p_boot = math.nan
        finite_coef = coef_samples[np.isfinite(coef_samples)]
        if finite_coef.size:
            p_pos = float(np.mean(finite_coef >= 0.0))
            p_neg = float(np.mean(finite_coef <= 0.0))
            p_boot = float(min(1.0, 2.0 * min(p_pos, p_neg)))
        direction = "uncertain"
        if np.isfinite(coef_est):
            if coef_est > 0:
                direction = "higher success odds"
            elif coef_est < 0:
                direction = "lower success odds"
            else:
                direction = "no directional effect"
        component_rows.append(
            {
                "feature": str(feature_name),
                "component": label,
                "component_type": comp_type,
                "scale": scale_note,
                "coef_estimate": coef_est,
                "coef_ci_lower": coef_ci_lo,
                "coef_ci_upper": coef_ci_hi,
                "coef_significant_95ci": (
                    "yes" if coef_sig is True else ("no" if coef_sig is False else "na")
                ),
                "or_estimate": or_est,
                "or_ci_lower": or_ci_lo,
                "or_ci_upper": or_ci_hi,
                "or_significant_95ci": (
                    "yes" if or_sig is True else ("no" if or_sig is False else "na")
                ),
                "direction": direction,
                "bootstrap_p_two_sided": p_boot,
                "n_boot_feature": int(np.isfinite(finite_coef).sum()),
            }
        )
    component_df = pd.DataFrame(component_rows)

    table = pd.DataFrame(
        [
            {
                "metric": "adjusted_risk_low",
                "estimate": float(margins["r_low_adj"]),
                "ci_lower": ci_lookup["r_low_adj"][0],
                "ci_upper": ci_lookup["r_low_adj"][1],
                "ci_method": "percentile_bootstrap",
            },
            {
                "metric": "adjusted_risk_high",
                "estimate": float(margins["r_high_adj"]),
                "ci_lower": ci_lookup["r_high_adj"][0],
                "ci_upper": ci_lookup["r_high_adj"][1],
                "ci_method": "percentile_bootstrap",
            },
            {
                "metric": "adjusted_rd_low_minus_high",
                "estimate": float(margins["rd_adj"]),
                "ci_lower": ci_lookup["rd_adj"][0],
                "ci_upper": ci_lookup["rd_adj"][1],
                "ci_method": "percentile_bootstrap",
            },
            {
                "metric": "adjusted_rr_low_over_high",
                "estimate": float(margins["rr_adj"]),
                "ci_lower": ci_lookup["rr_adj"][0],
                "ci_upper": ci_lookup["rr_adj"][1],
                "ci_method": "percentile_bootstrap",
            },
            {
                "metric": "adjusted_or_low_over_high",
                "estimate": float(margins["or_adj"]),
                "ci_lower": ci_lookup["or_adj"][0],
                "ci_upper": ci_lookup["or_adj"][1],
                "ci_method": "percentile_bootstrap",
            },
            {
                "metric": "unadjusted_rd_low_minus_high",
                "estimate": float(unadj_rd),
                "ci_lower": math.nan,
                "ci_upper": math.nan,
                "ci_method": "na",
            },
            {
                "metric": "rd_attenuation_adj_over_unadj",
                "estimate": float(attenuation),
                "ci_lower": math.nan,
                "ci_upper": math.nan,
                "ci_method": "na",
            },
        ]
    )
    table["n"] = int(len(work))
    table["n_units"] = int(len(units)) if units is not None else int(len(work))
    table["n_boot"] = int(len(boot_df))
    table["resample_unit"] = str(resample_unit_col or "row")

    payload: Dict[str, object] = {
        "status": "ok",
        "estimand": {
            "outcome": outcome_col,
            "exposure": exposure_col,
            "lower_label": lower_label,
            "higher_label": higher_label,
            "predictive_margins_definition": (
                "Average model-predicted success if all rows were set to lower-ITA vs higher-ITA, "
                "holding observed covariates fixed."
            ),
        },
        "summary": {
            "r_low_adj": float(margins["r_low_adj"]),
            "r_high_adj": float(margins["r_high_adj"]),
            "rd_adj": float(margins["rd_adj"]),
            "rr_adj": float(margins["rr_adj"]),
            "or_adj": float(margins["or_adj"]),
            "rd_ci_lower": ci_lookup["rd_adj"][0],
            "rd_ci_upper": ci_lookup["rd_adj"][1],
            "n": int(len(work)),
            "n_units": int(len(units)) if units is not None else int(len(work)),
            "n_boot": int(len(boot_df)),
            "resample_unit": str(resample_unit_col or "row"),
            "n_component_terms": int(len(component_df)),
            "n_component_terms_significant_or_95ci": int(
                (component_df.get("or_significant_95ci", pd.Series(dtype=str)) == "yes").sum()
                if not component_df.empty
                else 0
            ),
        },
        "unadjusted": {
            "risk_low": float(unadj_r_low),
            "risk_high": float(unadj_r_high),
            "rd": float(unadj_rd),
            "rd_attenuation_adj_over_unadj": float(attenuation),
        },
        "model_spec": {
            "type": "logistic_regression_predictive_margins",
            "solver": "lbfgs",
            "penalty": "l2",
            "C": float(c_value),
            "max_iter": int(max_iter),
            "imputer_numeric": "median",
            "scaler_numeric": "standard",
            "include_dataset_source": bool(include_dataset_source and dataset_source_col in work.columns),
            "dataset_source_col": dataset_source_col,
            "covariates_used": covariates,
            "numeric_predictors": numeric_cols,
            "categorical_predictors": categorical_cols,
            "fit": fit_spec,
            "component_scale_notes": {
                "continuous_covariates": "Coefficients correspond to 1 SD increase after standardization.",
                "exposure_indicator": "Binary contrast Lower ITA vs Higher ITA.",
                "dataset_fixed_effects": "One-hot contrasts vs reference dataset level.",
            },
            "bootstrap": {
                "n_resamples_requested": int(n_resamples),
                "n_resamples_successful": int(len(boot_df)),
                "ci_method": "percentile_bootstrap",
                "seed": int(seed),
                "resample_unit": str(resample_unit_col or "row"),
            },
        },
        "component_effects": (
            component_df.to_dict(orient="records")
            if not component_df.empty
            else []
        ),
        "warnings": (
            []
            if int(len(boot_df)) > 0
            else ["bootstrap_replicates_empty"]
        ),
    }
    return table, payload, boot_df
