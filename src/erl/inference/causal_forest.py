from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class ForestResult:
    ate: float
    cate: np.ndarray
    blp: pd.DataFrame
    calibration: pd.DataFrame
    moderators: list[str]
    n: int
    model: object


def _default_nuisance(random_state: int):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=20,
        max_depth=8,
        random_state=random_state,
        n_jobs=-1,
    )


def fit_causal_forest(
    panel: pd.DataFrame,
    outcome: str,
    treatment: str,
    moderators: list[str],
    controls: list[str] | None = None,
    cluster: str = "ticker",
    n_estimators: int = 1000,
    cv: int = 3,
    random_state: int = 7,
) -> ForestResult:
    from econml.dml import CausalForestDML
    from sklearn.model_selection import GroupKFold

    controls = controls or []
    columns = list(dict.fromkeys([outcome, treatment, *moderators, *controls, cluster]))
    frame = panel[columns].dropna().reset_index(drop=True)
    y = frame[outcome].to_numpy(dtype=float)
    t = frame[treatment].to_numpy(dtype=float)
    X = frame[moderators].to_numpy(dtype=float)
    W = frame[controls].to_numpy(dtype=float) if controls else None
    groups = frame[cluster].to_numpy()

    splitter = GroupKFold(n_splits=cv)
    estimator = CausalForestDML(
        model_y=_default_nuisance(random_state),
        model_t=_default_nuisance(random_state + 1),
        discrete_treatment=False,
        n_estimators=n_estimators,
        min_samples_leaf=20,
        cv=splitter,
        random_state=random_state,
    )
    estimator.fit(y, t, X=X, W=W, groups=groups)
    cate = np.asarray(estimator.effect(X)).ravel()
    ate = float(cate.mean())

    blp = best_linear_projection(cate, frame[moderators])
    calibration = cate_sort_test(frame, cate, outcome, treatment, controls)
    logger.info(
        "causal forest fit: n=%d, ate=%.5f, cate sd=%.5f", len(frame), ate, cate.std()
    )
    return ForestResult(
        ate=ate,
        cate=cate,
        blp=blp,
        calibration=calibration,
        moderators=moderators,
        n=len(frame),
        model=estimator,
    )


def best_linear_projection(cate: np.ndarray, moderators: pd.DataFrame) -> pd.DataFrame:
    X = moderators.to_numpy(dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds
    design = sm.add_constant(Z)
    fit = sm.OLS(cate, design).fit(cov_type="HC1")
    rows = [{"term": "intercept", "coef": float(fit.params[0]), "se": float(fit.bse[0])}]
    for i, name in enumerate(moderators.columns):
        rows.append(
            {
                "term": name,
                "coef": float(fit.params[1 + i]),
                "se": float(fit.bse[1 + i]),
            }
        )
    table = pd.DataFrame(rows)
    table["tstat"] = table["coef"] / table["se"]
    return table


def cate_sort_test(
    frame: pd.DataFrame,
    cate: np.ndarray,
    outcome: str,
    treatment: str,
    controls: list[str],
    quantiles: int = 4,
) -> pd.DataFrame:
    work = frame.copy()
    work["_cate"] = cate
    ranks = work["_cate"].rank(method="first")
    work["_bucket"] = pd.qcut(ranks, quantiles, labels=False, duplicates="drop") + 1
    rows = []
    for bucket, group in work.groupby("_bucket"):
        y = group[outcome].to_numpy(dtype=float)
        t = group[treatment].to_numpy(dtype=float)
        if controls:
            C = sm.add_constant(group[controls].to_numpy(dtype=float))
            y = y - sm.OLS(y, C).fit().predict(C)
            t = t - sm.OLS(t, C).fit().predict(C)
        design = sm.add_constant(t)
        fit = sm.OLS(y, design).fit(cov_type="HC1")
        rows.append(
            {
                "bucket": int(bucket),
                "n": len(group),
                "predicted_mean_cate": float(group["_cate"].mean()),
                "realized_effect": float(fit.params[1]),
                "se": float(fit.bse[1]),
            }
        )
    table = pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)
    if len(table) >= 2:
        spread = table["realized_effect"].iloc[-1] - table["realized_effect"].iloc[0]
        table.attrs["top_minus_bottom"] = float(spread)
        predicted = table["predicted_mean_cate"].to_numpy()
        realized = table["realized_effect"].to_numpy()
        if predicted.std() > 0 and realized.std() > 0:
            table.attrs["rank_correlation"] = float(np.corrcoef(predicted, realized)[0, 1])
    return table
