"""Is the pooled surprise effect one relationship or an average of several?

The sample spans the GFC, the zero-rate years, COVID and the 2022 repricing. A
coefficient pooled across a break is an average of different economies, and it
means less than its standard error suggests.

``regime_stability`` interacts the surprise with regime dummies and Wald-tests
the interactions jointly (a Chow test, two-way clustered so it assumes neither
homoskedasticity nor independence). ``rolling_effect`` shows whether a rejection
is a discrete break or a drift.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from erl.inference.double_lasso import twoway_cluster_cov

logger = logging.getLogger(__name__)

# Taken from the macro record, not chosen by looking at the outcome. Breaks the
# sample cannot support are dropped, so the same list works from 2005 or 2015.
DEFAULT_BREAKS = ("2008-09-15", "2009-07-01", "2020-03-01", "2022-01-01")

# Enough events for a clustered slope to mean anything.
MIN_REGIME_EVENTS = 150


def applicable_breaks(
    dates: pd.Series,
    breaks: tuple[str, ...] = DEFAULT_BREAKS,
    min_events: int = MIN_REGIME_EVENTS,
) -> tuple[str, ...]:
    """Drop breaks the sample cannot support.

    A break outside the sample range creates an empty regime and a singular
    design; one leaving fewer than ``min_events`` on either side gives a slope
    with no power. Both go, so what survives partitions the sample into
    estimable pieces.
    """
    stamps = pd.to_datetime(dates).sort_values()
    if stamps.empty:
        return ()
    kept: list[str] = []
    edges = [stamps.iloc[0]]
    for candidate in sorted(pd.Timestamp(b) for b in breaks):
        if candidate <= stamps.iloc[0] or candidate > stamps.iloc[-1]:
            continue
        left = int(((stamps >= edges[-1]) & (stamps < candidate)).sum())
        right = int((stamps >= candidate).sum())
        if left < min_events or right < min_events:
            continue
        kept.append(candidate.date().isoformat())
        edges.append(candidate)
    dropped = [str(b) for b in breaks if b not in kept]
    if dropped:
        logger.info(
            "regime breaks dropped as unsupported by the sample (%s to %s): %s",
            stamps.iloc[0].date(), stamps.iloc[-1].date(), ", ".join(dropped),
        )
    return tuple(kept)


def assign_regime(dates: pd.Series, breaks: tuple[str, ...] = DEFAULT_BREAKS) -> pd.Series:
    stamps = pd.to_datetime(dates)
    edges = (
        [pd.Timestamp.min]
        + sorted(pd.Timestamp(b) for b in breaks)
        + [pd.Timestamp.max]
    )
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lo_name = "start" if i == 0 else lo.date().isoformat()
        hi_name = "end" if i == len(edges) - 2 else hi.date().isoformat()
        labels.append(f"{lo_name}..{hi_name}")
    return pd.cut(stamps, bins=edges, labels=labels, right=False).astype(str)


def regime_stability(
    panel: pd.DataFrame,
    outcome: str = "car_reaction",
    treatment: str = "sue",
    controls: list[str] | None = None,
    breaks: tuple[str, ...] = DEFAULT_BREAKS,
    date_col: str = "announce_date",
    cluster_a: str = "ticker",
    cluster_b: str = "announce_quarter",
) -> pd.DataFrame:
    """Chow-style test for a break in the surprise effect.

    Fits ``outcome = a + b*T + sum_k c_k*(T x regime_k) + regime FE + controls``
    and Wald-tests the interactions jointly with two-way clustered errors. The
    table holds the per-regime effect; the joint test is in ``.attrs``.
    """
    controls = controls or []
    columns = list(
        dict.fromkeys([outcome, treatment, date_col, cluster_a, cluster_b, *controls])
    )
    frame = panel[columns].dropna().reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()
    usable = applicable_breaks(frame[date_col], breaks)
    if not usable:
        logger.warning(
            "regime_stability: none of the requested breaks %s is supported by a "
            "sample running %s to %s; nothing to test",
            list(breaks),
            pd.to_datetime(frame[date_col]).min().date(),
            pd.to_datetime(frame[date_col]).max().date(),
        )
        return pd.DataFrame()
    breaks = usable
    frame["regime"] = assign_regime(frame[date_col], breaks)
    regimes = sorted(frame["regime"].unique())
    if len(regimes) < 2:
        logger.warning("regime_stability: only one regime present, nothing to test")
        return pd.DataFrame()

    base, others = regimes[0], regimes[1:]
    d = frame[treatment].to_numpy(dtype=float)
    blocks = [d]
    names = [treatment]
    for regime in others:
        indicator = (frame["regime"] == regime).to_numpy(dtype=float)
        blocks.append(d * indicator)
        names.append(f"{treatment}_x_{regime}")
    for regime in others:  # regime fixed effects
        blocks.append((frame["regime"] == regime).to_numpy(dtype=float))
        names.append(f"fe_{regime}")
    for control in controls:
        blocks.append(frame[control].to_numpy(dtype=float))
        names.append(control)

    design = sm.add_constant(np.column_stack(blocks))
    fit = sm.OLS(frame[outcome].to_numpy(dtype=float), design).fit()
    cov = twoway_cluster_cov(fit, frame[cluster_a].to_numpy(), frame[cluster_b].to_numpy())

    idx = {name: 1 + i for i, name in enumerate(names)}
    interaction_idx = [idx[f"{treatment}_x_{r}"] for r in others]
    base_idx = idx[treatment]

    rows = []
    for regime in regimes:
        if regime == base:
            weights = np.zeros(design.shape[1])
            weights[base_idx] = 1.0
        else:
            weights = np.zeros(design.shape[1])
            weights[base_idx] = 1.0
            weights[idx[f"{treatment}_x_{regime}"]] = 1.0
        effect = float(weights @ fit.params)
        se = float(np.sqrt(weights @ cov @ weights))
        rows.append(
            {
                "regime": regime,
                "n": int((frame["regime"] == regime).sum()),
                "effect": effect,
                "se": se,
                "tstat": effect / se if se > 0 else np.nan,
                "is_base": regime == base,
            }
        )
    table = pd.DataFrame(rows)

    # Wald test that all interactions are zero.
    R = np.zeros((len(interaction_idx), design.shape[1]))
    for row, column in enumerate(interaction_idx):
        R[row, column] = 1.0
    middle = R @ cov @ R.T
    diff = R @ fit.params
    try:
        statistic = float(diff @ np.linalg.solve(middle, diff))
    except np.linalg.LinAlgError:
        statistic = np.nan
    dof = len(interaction_idx)
    from scipy.stats import chi2

    pvalue = float(chi2.sf(statistic, dof)) if np.isfinite(statistic) else np.nan
    table.attrs["wald_statistic"] = statistic
    table.attrs["wald_dof"] = dof
    table.attrs["wald_pvalue"] = pvalue
    table.attrs["n"] = len(frame)
    table.attrs["breaks"] = list(breaks)
    logger.info(
        "regime stability: chi2(%d)=%.2f p=%.4f | effects by regime:\n%s",
        dof,
        statistic,
        pvalue,
        table.to_string(index=False),
    )
    if np.isfinite(pvalue) and pvalue < 0.05:
        logger.warning(
            "surprise effect is NOT stable across regimes (p=%.4f); the pooled "
            "estimate averages different regimes and should be reported by period",
            pvalue,
        )
    return table


def rolling_effect(
    panel: pd.DataFrame,
    outcome: str = "car_reaction",
    treatment: str = "sue",
    date_col: str = "announce_date",
    window: int = 400,
    step: int = 50,
    cluster: str = "ticker",
) -> pd.DataFrame:
    """Surprise coefficient on a rolling window. Univariate by design: controls
    that are themselves regime-dependent would confound the time profile."""
    frame = (
        panel[[outcome, treatment, date_col, cluster]]
        .dropna()
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    if len(frame) < window:
        logger.warning(
            "rolling_effect: %d usable events is fewer than the %d-event window",
            len(frame),
            window,
        )
        return pd.DataFrame()
    rows = []
    for start in range(0, len(frame) - window + 1, step):
        chunk = frame.iloc[start : start + window]
        design = sm.add_constant(chunk[treatment].to_numpy(dtype=float))
        model = sm.OLS(chunk[outcome].to_numpy(dtype=float), design)
        if chunk[cluster].nunique() > 1:
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": chunk[cluster].to_numpy()})
        else:
            fit = model.fit(cov_type="HC1")
        rows.append(
            {
                "window_start": pd.to_datetime(chunk[date_col]).min(),
                "window_end": pd.to_datetime(chunk[date_col]).max(),
                "n": len(chunk),
                "effect": float(fit.params[1]),
                "se": float(fit.bse[1]),
            }
        )
    table = pd.DataFrame(rows)
    table["tstat"] = table["effect"] / table["se"]
    spread = float(table["effect"].max() - table["effect"].min())
    table.attrs["effect_range"] = spread
    mean_se = float(table["se"].mean())
    if mean_se > 0 and spread > 4 * mean_se:
        logger.warning(
            "rolling surprise effect ranges over %.4f, more than 4 average standard "
            "errors (%.4f); treat the pooled estimate as a period average",
            spread,
            mean_se,
        )
    return table
