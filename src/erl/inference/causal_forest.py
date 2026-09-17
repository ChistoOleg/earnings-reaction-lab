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
    ate_se: float = float("nan")
    blp_naive: pd.DataFrame | None = None


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
    # cache_values keeps the nuisance residuals the BLP below needs.
    estimator.fit(y, t, X=X, W=W, groups=groups, cache_values=True)
    cate = np.asarray(estimator.effect(X)).ravel()
    ate = float(cate.mean())
    ate_se = float("nan")
    try:
        ate_inf = estimator.ate_inference(X=X)
        ate = float(np.ravel(ate_inf.mean_point)[0])
        ate_se = float(np.ravel(ate_inf.stderr_mean)[0])
    except Exception as exc:  # inference is optional; the point estimate stands
        logger.warning("ATE inference unavailable (%s); reporting mean CATE without SE", exc)

    y_res, t_res = _nuisance_residuals(estimator, y, t, X, W, groups, splitter, random_state)
    blp = best_linear_projection(y_res, t_res, frame[moderators], groups=groups)
    blp_naive = naive_cate_projection(cate, frame[moderators], groups=groups)
    calibration = cate_sort_test(frame, cate, outcome, treatment, controls, cluster=cluster)
    logger.info(
        "causal forest fit: n=%d, ate=%.5f (se %.5f), cate sd=%.5f, BLP SEs=%s",
        len(frame),
        ate,
        ate_se,
        cate.std(),
        blp.attrs.get("cov_type", "HC1"),
    )
    return ForestResult(
        ate=ate,
        cate=cate,
        blp=blp,
        calibration=calibration,
        moderators=moderators,
        n=len(frame),
        model=estimator,
        ate_se=ate_se,
        blp_naive=blp_naive,
    )


def _nuisance_residuals(estimator, y, t, X, W, groups, splitter, random_state: int):
    """Cross-fitted outcome and treatment residuals, from the fitted estimator if
    it cached them, otherwise recomputed on the same grouped folds."""
    try:
        y_res, t_res, _, _ = estimator.residuals_
        return np.asarray(y_res, dtype=float).ravel(), np.asarray(t_res, dtype=float).ravel()
    except (AttributeError, ValueError):
        pass
    from sklearn.model_selection import cross_val_predict

    XW = X if W is None else np.column_stack([X, W])
    y_hat = cross_val_predict(_default_nuisance(random_state), XW, y, cv=splitter, groups=groups)
    t_hat = cross_val_predict(_default_nuisance(random_state + 1), XW, t, cv=splitter, groups=groups)
    return y - y_hat, t - t_hat


def _standardize(moderators: pd.DataFrame) -> np.ndarray:
    X = moderators.to_numpy(dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0
    return (X - means) / stds


def _cluster_or_hc1(model, groups):
    if groups is not None and len(np.unique(groups)) > 1:
        return model.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)}), "cluster"
    return model.fit(cov_type="HC1"), "HC1"


def _coef_table(fit, names: list[str], cov_kind: str) -> pd.DataFrame:
    rows = [{"term": "intercept", "coef": float(fit.params[0]), "se": float(fit.bse[0])}]
    for i, name in enumerate(names):
        rows.append({"term": name, "coef": float(fit.params[1 + i]), "se": float(fit.bse[1 + i])})
    table = pd.DataFrame(rows)
    table["tstat"] = table["coef"] / table["se"]
    table.attrs["cov_type"] = cov_kind
    return table


def best_linear_projection(
    y_res: np.ndarray,
    t_res: np.ndarray,
    moderators: pd.DataFrame,
    groups: np.ndarray | None = None,
) -> pd.DataFrame:
    """Best linear projection of the treatment effect on standardised moderators.

    Under Y = theta(X)T + g(X,W) + e, Robinson residualisation gives
    Y_res = theta(X)T_res + e, so the projection is the regression
    Y_res = a*T_res + sum_j b_j*(T_res * Z_j) + u. Its standard errors reflect
    noise in the data; see ``naive_cate_projection`` for why the obvious
    alternative does not. Clustered by firm when ``groups`` is given.
    """
    Z = _standardize(moderators)
    t_res = np.asarray(t_res, dtype=float).ravel()
    y_res = np.asarray(y_res, dtype=float).ravel()
    interactions = Z * t_res[:, None]
    design = np.column_stack([t_res, interactions])
    # The constant absorbs any residual mean left by imperfect cross-fitting.
    # It is not a BLP coefficient.
    design = sm.add_constant(design, has_constant="add")
    fit, cov_kind = _cluster_or_hc1(sm.OLS(y_res, design), groups)
    names = ["treatment_mean"] + list(moderators.columns)
    table = _coef_table(fit, names, cov_kind)
    table.attrs["method"] = "residual_interaction"
    return table


def naive_cate_projection(
    cate: np.ndarray, moderators: pd.DataFrame, groups: np.ndarray | None = None
) -> pd.DataFrame:
    """OLS of fitted CATEs on standardised moderators. For comparison only: fitted
    CATEs are smooth functions of these same moderators, so the fit is nearly
    perfect and the standard errors measure the forest's smoothness rather than
    sampling uncertainty. Understates by 30x or more in practice."""
    design = sm.add_constant(_standardize(moderators))
    fit, cov_kind = _cluster_or_hc1(sm.OLS(cate, design), groups)
    table = _coef_table(fit, list(moderators.columns), cov_kind)
    table.attrs["method"] = "cate_on_moderators"
    return table


def cate_sort_test(
    frame: pd.DataFrame,
    cate: np.ndarray,
    outcome: str,
    treatment: str,
    controls: list[str],
    cluster: str | None = None,
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
        model = sm.OLS(y, design)
        if cluster is not None and group[cluster].nunique() > 1:
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": group[cluster].to_numpy()})
        else:
            fit = model.fit(cov_type="HC1")
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
