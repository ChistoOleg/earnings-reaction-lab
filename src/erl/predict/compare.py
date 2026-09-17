"""Is one model actually better out of sample, or is the gap noise?

Reading the ordering off the metrics table is tempting and close to meaningless:
at n = 400 the standard error of a Spearman correlation is about 0.05, so a
rank-IC gap of 0.02 says nothing. Claiming a winner *or* claiming the models are
indistinguishable needs a test, and since they predict the same events from the
same features their errors are correlated, so it has to be paired.

``loss_differential_test`` is Diebold-Mariano for a panel: the per-event
squared-error difference regressed on a constant, clustered by quarter because
same-quarter events share market shocks. ``rank_ic_difference_test`` block
bootstraps the rank-IC gap over whole quarters; resampling events independently
would understate the error for the same reason the naive CATE projection does.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def _blocks(dates: pd.Series) -> np.ndarray:
    return pd.PeriodIndex(pd.to_datetime(dates), freq="Q").astype(str).to_numpy()


def loss_differential_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    dates: pd.Series,
    loss: str = "squared",
) -> dict[str, float]:
    """Paired test of mean loss difference (a minus b). Negative favours ``a``."""
    y_true = np.asarray(y_true, dtype=float)
    err_a = y_true - np.asarray(pred_a, dtype=float)
    err_b = y_true - np.asarray(pred_b, dtype=float)
    if loss == "squared":
        d = err_a ** 2 - err_b ** 2
    elif loss == "absolute":
        d = np.abs(err_a) - np.abs(err_b)
    else:
        raise ValueError(f"loss must be 'squared' or 'absolute', got {loss!r}")

    blocks = _blocks(dates)
    design = np.ones((len(d), 1))
    model = sm.OLS(d, design)
    if len(np.unique(blocks)) > 1:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": blocks})
        cov_kind = "cluster_quarter"
    else:
        fit = model.fit(cov_type="HC1")
        cov_kind = "HC1"
    return {
        "mean_loss_diff": float(fit.params[0]),
        "se": float(fit.bse[0]),
        "tstat": float(fit.tvalues[0]),
        "pvalue": float(fit.pvalues[0]),
        "n": int(len(d)),
        "n_blocks": int(len(np.unique(blocks))),
        "cov_type": cov_kind,
        "loss": loss,
    }


def rank_ic_difference_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    dates: pd.Series,
    n_boot: int = 2000,
    random_state: int = 7,
) -> dict[str, float]:
    """Block bootstrap of the rank-IC difference (a minus b); positive favours
    ``a``. Quarters are resampled with replacement and events within a quarter
    travel together."""
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    blocks = _blocks(dates)
    unique = np.unique(blocks)
    index_by_block = {b: np.flatnonzero(blocks == b) for b in unique}

    def ic_gap(idx: np.ndarray) -> float:
        if len(idx) < 3:
            return np.nan
        a = spearmanr(y_true[idx], pred_a[idx]).statistic
        b = spearmanr(y_true[idx], pred_b[idx]).statistic
        return float(a - b)

    observed = ic_gap(np.arange(len(y_true)))
    rng = np.random.default_rng(random_state)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_block[b] for b in picked])
        draws[i] = ic_gap(idx)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return {"ic_diff": observed, "n_boot": 0}
    # Two-sided bootstrap p-value, centring the resampling distribution on zero.
    centred = draws - draws.mean()
    pvalue = float(np.mean(np.abs(centred) >= abs(observed)))
    return {
        "ic_diff": observed,
        "se": float(draws.std(ddof=1)),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "pvalue": pvalue,
        "n_boot": int(len(draws)),
        "n_blocks": int(len(unique)),
    }


def compare_models(
    frame: pd.DataFrame,
    reference: str,
    challengers: list[str],
    date_col: str = "announce_date",
    truth_col: str = "y_true",
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Paired comparisons of ``reference`` against each challenger. ``frame`` has
    one row per out-of-sample event, with a ``y_pred_<model>`` column each."""
    rows: list[dict] = []
    y = frame[truth_col].to_numpy(dtype=float)
    ref = frame[f"y_pred_{reference}"].to_numpy(dtype=float)
    for name in challengers:
        column = f"y_pred_{name}"
        if column not in frame.columns:
            logger.warning("no predictions for %s; skipping comparison", name)
            continue
        other = frame[column].to_numpy(dtype=float)
        mse = loss_differential_test(y, ref, other, frame[date_col], loss="squared")
        mae = loss_differential_test(y, ref, other, frame[date_col], loss="absolute")
        ic = rank_ic_difference_test(y, ref, other, frame[date_col], n_boot=n_boot)
        rows.append(
            {
                "model_a": reference,
                "model_b": name,
                "mse_diff": mse["mean_loss_diff"],
                "mse_tstat": mse["tstat"],
                "mse_pvalue": mse["pvalue"],
                "mae_diff": mae["mean_loss_diff"],
                "mae_pvalue": mae["pvalue"],
                "ic_diff": ic.get("ic_diff"),
                "ic_ci_low": ic.get("ci_low"),
                "ic_ci_high": ic.get("ci_high"),
                "ic_pvalue": ic.get("pvalue"),
                "n": mse["n"],
                "n_quarters": mse["n_blocks"],
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    verdicts = []
    for row in table.itertuples():
        wins = [
            p < 0.05
            for p in (row.mse_pvalue, row.mae_pvalue, row.ic_pvalue)
            if p is not None and np.isfinite(p)
        ]
        verdicts.append("differs" if any(wins) else "indistinguishable")
    table["verdict"] = verdicts
    logger.info("paired out-of-sample comparison:\n%s", table.to_string(index=False))
    if (table["verdict"] == "indistinguishable").all():
        logger.info(
            "no challenger is separated from %s on any paired test; the ordering of "
            "the raw metrics is not evidence of a difference",
            reference,
        )
    return table
