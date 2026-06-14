from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Lasso

logger = logging.getLogger(__name__)


def plugin_lasso_select(
    X: np.ndarray,
    y: np.ndarray,
    *,
    c: float = 1.1,
    gamma: float = 0.1,
    n_iterations: int = 3,
) -> list[int]:
    n, p = X.shape
    if p == 0:
        return []
    scale = X.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / scale
    residual = y - y.mean()
    lam = 2.0 * c * np.sqrt(n) * norm.ppf(1.0 - gamma / (2.0 * p))
    selected: list[int] = []
    for _ in range(n_iterations):
        loadings = np.sqrt(np.mean((Xs ** 2) * (residual[:, None] ** 2), axis=0))
        loadings[loadings == 0] = loadings[loadings > 0].min() if (loadings > 0).any() else 1.0
        Z = Xs / loadings
        alpha = lam / (2.0 * n)
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=20000)
        model.fit(Z, y)
        selected = [int(j) for j in np.flatnonzero(model.coef_)]
        residual = y - model.predict(Z)
    return selected


def _cluster_cov(results, groups: np.ndarray) -> np.ndarray:
    robust = results.get_robustcov_results(cov_type="cluster", groups=groups)
    return np.asarray(robust.cov_params())


def twoway_cluster_cov(results, groups_a: np.ndarray, groups_b: np.ndarray) -> np.ndarray:
    pair = pd.factorize(pd.Series(groups_a).astype(str) + "||" + pd.Series(groups_b).astype(str))[0]
    cov = (
        _cluster_cov(results, groups_a)
        + _cluster_cov(results, groups_b)
        - _cluster_cov(results, pair)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_min = 1.0
    for rank_from_end, idx in enumerate(order[::-1]):
        rank = m - rank_from_end
        value = p[idx] * m / rank
        running_min = min(running_min, value)
        adjusted[idx] = min(running_min, 1.0)
    return adjusted


@dataclass
class DoubleLassoResult:
    treatment: str
    coef: float
    se: float
    tstat: float
    pvalue: float
    n: int
    selected_controls: list[str] = field(default_factory=list)


def fit_double_lasso(
    panel: pd.DataFrame,
    outcome: str,
    treatment: str,
    controls: list[str],
    cluster_a: str = "ticker",
    cluster_b: str = "announce_quarter",
) -> DoubleLassoResult:
    columns = [outcome, treatment, *controls, cluster_a, cluster_b]
    frame = panel[columns].dropna().reset_index(drop=True)
    y = frame[outcome].to_numpy(dtype=float)
    d = frame[treatment].to_numpy(dtype=float)
    X = frame[controls].to_numpy(dtype=float)

    selected_y = plugin_lasso_select(X, y)
    selected_d = plugin_lasso_select(X, d)
    union = sorted(set(selected_y) | set(selected_d))
    selected_names = [controls[j] for j in union]

    design = sm.add_constant(np.column_stack([d, X[:, union]]) if union else d.reshape(-1, 1))
    fit = sm.OLS(y, design).fit()
    cov = twoway_cluster_cov(
        fit, frame[cluster_a].to_numpy(), frame[cluster_b].to_numpy()
    )
    coef = float(fit.params[1])
    se = float(np.sqrt(cov[1, 1]))
    tstat = coef / se if se > 0 else np.nan
    pvalue = float(2 * (1 - norm.cdf(abs(tstat)))) if np.isfinite(tstat) else np.nan
    logger.info(
        "double lasso: %s on %s | coef=%.5f se=%.5f | %d controls selected",
        outcome,
        treatment,
        coef,
        se,
        len(selected_names),
    )
    return DoubleLassoResult(
        treatment=treatment,
        coef=coef,
        se=se,
        tstat=tstat,
        pvalue=pvalue,
        n=len(frame),
        selected_controls=selected_names,
    )


def fit_interacted_double_lasso(
    panel: pd.DataFrame,
    outcome: str,
    treatment: str,
    moderators: list[str],
    controls: list[str],
    cluster_a: str = "ticker",
    cluster_b: str = "announce_quarter",
    alpha: float = 0.05,
) -> pd.DataFrame:
    columns = [outcome, treatment, *moderators, *controls, cluster_a, cluster_b]
    frame = panel[list(dict.fromkeys(columns))].dropna().reset_index(drop=True)
    y = frame[outcome].to_numpy(dtype=float)
    d = frame[treatment].to_numpy(dtype=float)

    standardized = {}
    for moderator in moderators:
        values = frame[moderator].to_numpy(dtype=float)
        spread = values.std(ddof=1)
        standardized[moderator] = (values - values.mean()) / (spread if spread > 0 else 1.0)

    focal_names = [treatment] + [f"{treatment}_x_{m}" for m in moderators]
    focal = np.column_stack([d] + [d * standardized[m] for m in moderators])

    control_matrix = np.column_stack(
        [frame[c].to_numpy(dtype=float) for c in controls]
        + [standardized[m] for m in moderators]
    )
    control_names = controls + [f"{m}_level" for m in moderators]

    union: set[int] = set(plugin_lasso_select(control_matrix, y))
    for k in range(focal.shape[1]):
        union |= set(plugin_lasso_select(control_matrix, focal[:, k]))
    forced = {control_names.index(f"{m}_level") for m in moderators}
    keep = sorted(union | forced)

    design = sm.add_constant(np.column_stack([focal, control_matrix[:, keep]]))
    fit = sm.OLS(y, design).fit()
    cov = twoway_cluster_cov(
        fit, frame[cluster_a].to_numpy(), frame[cluster_b].to_numpy()
    )

    rows = []
    for i, name in enumerate(focal_names):
        coef = float(fit.params[1 + i])
        se = float(np.sqrt(cov[1 + i, 1 + i]))
        tstat = coef / se if se > 0 else np.nan
        pvalue = float(2 * (1 - norm.cdf(abs(tstat)))) if np.isfinite(tstat) else np.nan
        rows.append({"term": name, "coef": coef, "se": se, "tstat": tstat, "pvalue": pvalue})
    table = pd.DataFrame(rows)
    interaction_mask = table["term"] != treatment
    adjusted = np.full(len(table), np.nan)
    adjusted[interaction_mask.to_numpy()] = benjamini_hochberg(
        table.loc[interaction_mask, "pvalue"].to_numpy(), alpha=alpha
    )
    table["pvalue_bh"] = adjusted
    table["significant_bh"] = table["pvalue_bh"] < alpha
    table.attrs["n"] = len(frame)
    table.attrs["selected_controls"] = [control_names[j] for j in keep]
    return table
