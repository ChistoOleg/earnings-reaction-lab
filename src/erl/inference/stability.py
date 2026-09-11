"""Parameter stability of the surprise effect over time.

Every estimate elsewhere in this project pools 2015-2024 into one number. That
is only meaningful if the reaction-to-surprise relationship is stable over the
sample. It plausibly is not: the sample spans a zero-rate regime, the COVID
volatility shock, the 2022 rate repricing, and a period of very high index
concentration. A pooled coefficient across a break is an average of two
different economies, and the clustered standard error around it understates
nothing while the point estimate means less than it appears to.

Two tests are provided:

- ``regime_stability``: interact the surprise with regime dummies and run a Wald
  test that the interactions are jointly zero (a Chow test with two-way
  clustered errors, so it does not assume homoskedasticity or independence).
- ``rolling_effect``: the surprise coefficient on a rolling window, to see
  whether any rejection is a discrete break or a drift.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from erl.inference.double_lasso import twoway_cluster_cov

logger = logging.getLogger(__name__)

# Defaults chosen from the macro record, not from looking at the outcome:
# the GFC and its aftermath, the zero-rate years, the COVID shock, and the
# post-2022 higher-rate regime. Breaks that fall outside the sample, or that
# would leave a regime too small to estimate, are dropped automatically, so the
# same list works whether the panel starts in 2005 or in 2015.
DEFAULT_BREAKS = ("2008-09-15", "2009-07-01", "2020-03-01", "2022-01-01")

# A regime needs enough events for a clustered slope to mean anything.
MIN_REGIME_EVENTS = 150


def applicable_breaks(
    dates: pd.Series,
    breaks: tuple[str, ...] = DEFAULT_BREAKS,
    min_events: int = MIN_REGIME_EVENTS,
) -> tuple[str, ...]:
    """Drop breaks that the sample cannot support.

    A break before the first event or after the last one creates an empty
    regime, which makes the design matrix singular and the Wald test
    meaningless. A break that leaves fewer than ``min_events`` on either side
    gives a slope with no power. Both are dropped, in order, so the surviving
    breaks always partition the sample into estimable pieces.
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

    Estimates ``outcome = a + b*T + sum_k c_k*(T x regime_k) + regime FE +
    controls`` with the first regime as the base level, then Wald-tests
    ``c_1 = ... = c_K = 0`` using two-way (firm x quarter) clustered errors.
    The returned table carries the per-regime effect (base + interaction) and
    the joint test in ``.attrs``.
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
    for regime in others:  # regime fixed effects (level shifts)
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
    """Surprise coefficient on a rolling window of ``window`` events.

    Univariate by design: the point is the time profile of the slope, and adding
    controls that are themselves regime-dependent would confound it.
    """
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
