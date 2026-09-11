from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from erl.events.returns import ReturnContext

logger = logging.getLogger(__name__)


def add_beat_flags(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    frame["eps_beat"] = (frame["surprise"] > 0).astype(int)
    has_rev = frame["revenue_actual"].notna() & frame["revenue_estimate"].notna()
    rev_beat = np.where(
        has_rev, (frame["revenue_actual"] > frame["revenue_estimate"]).astype(float), np.nan
    )
    frame["rev_beat"] = rev_beat
    frame["both_beat"] = np.where(
        np.isnan(rev_beat), np.nan, ((frame["eps_beat"] == 1) & (rev_beat == 1)).astype(float)
    )
    return frame


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip to the [lower, upper] empirical quantiles, ignoring NaNs."""
    valid = series.dropna()
    if valid.empty:
        return series
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def winsorize_asof(
    values: pd.Series,
    dates: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
    min_prior: int = 100,
) -> pd.Series:
    """Point-in-time winsorisation: each event is clipped to quantiles computed
    from events that occurred strictly *earlier*.

    Pooled winsorisation uses the whole sample's quantiles, which means an event
    in 2017 is clipped using information from 2024. For the inference track that
    is harmless (it is explicitly a full-sample estimate), but the prediction
    track is evaluated out of time and must not see future data in any form,
    including a clipping threshold. Events before ``min_prior`` prior
    observations exist are left unclipped.
    """
    frame = pd.DataFrame({"value": values, "date": pd.to_datetime(dates)})
    order = frame.sort_values("date", kind="stable").index
    ordered = frame.loc[order, "value"]
    valid = ordered.notna()
    expanding = ordered.where(valid)
    lo = expanding.shift(1).expanding(min_periods=min_prior).quantile(lower)
    hi = expanding.shift(1).expanding(min_periods=min_prior).quantile(upper)
    clipped = ordered.clip(lower=lo, upper=hi)
    unclipped = (lo.isna() | hi.isna()) & valid
    clipped = clipped.where(~unclipped, ordered)
    n_unclipped = int(unclipped.sum())
    if n_unclipped:
        logger.info(
            "point-in-time winsorisation: first %d of %d events left unclipped "
            "(fewer than %d prior observations)",
            n_unclipped, int(valid.sum()), min_prior,
        )
        if n_unclipped == int(valid.sum()):
            logger.warning(
                "point-in-time winsorisation had no effect: the sample (%d events) is "
                "smaller than min_prior=%d, so outliers are unclipped. Use "
                'winsor_mode="pooled" for a small pilot and say so in the write-up.',
                int(valid.sum()), min_prior,
            )
    return clipped.reindex(values.index)


def add_sue(
    events: pd.DataFrame,
    min_history: int = 4,
    winsor: tuple[float, float] | None = (0.01, 0.99),
    winsor_mode: str = "asof",
) -> pd.DataFrame:
    """Standardised unexpected earnings: surprise divided by the standard
    deviation of the firm's *past* surprises (expanding window, at least
    ``min_history`` prior quarters).

    The denominator can be tiny for firms whose past surprises were all close to
    zero, which produces SUE values of +-20 or more. Quintile sorts are rank-based
    and immune to that, but the OLS slope, the double lasso and the causal forest
    all treat SUE as continuous, so a handful of such events would dominate them.
    ``sue`` is therefore winsorised at the given quantiles; the untouched value is
    kept in ``sue_raw``.

    ``winsor_mode="asof"`` (the default) computes the clipping quantiles from
    prior events only, so nothing downstream sees a threshold derived from future
    data. ``winsor_mode="pooled"`` uses full-sample quantiles, which is
    acceptable for the in-sample inference track but leaks into any out-of-time
    evaluation.
    """
    frame = events.sort_values(["ticker", "announce_date"]).copy()
    past_std = frame.groupby("ticker")["surprise"].transform(
        lambda s: s.shift(1).expanding(min_periods=min_history).std()
    )
    past_std = past_std.replace(0.0, np.nan)
    frame["sue_raw"] = frame["surprise"] / past_std
    if winsor is None:
        frame["sue"] = frame["sue_raw"]
        return frame
    if winsor_mode == "asof":
        frame["sue"] = winsorize_asof(frame["sue_raw"], frame["announce_date"], *winsor)
    elif winsor_mode == "pooled":
        frame["sue"] = winsorize(frame["sue_raw"], *winsor)
    else:
        raise ValueError(f"winsor_mode must be 'asof' or 'pooled', got {winsor_mode!r}")
    clipped = int(((frame["sue"] != frame["sue_raw"]) & frame["sue_raw"].notna()).sum())
    if clipped:
        logger.info(
            "SUE winsorised (%s) at %.0f%%/%.0f%%: %d of %d events clipped",
            winsor_mode, 100 * winsor[0], 100 * winsor[1], clipped,
            int(frame["sue_raw"].notna().sum()),
        )
    return frame


def _streak_values(beats: list[int]) -> list[int]:
    streak = 0
    out: list[int] = []
    for beat in beats:
        out.append(streak)
        streak = streak + 1 if beat == 1 else 0
    return out


def add_prior_streak(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.sort_values(["ticker", "announce_date"]).copy()
    frame["prior_streak"] = frame.groupby("ticker")["eps_beat"].transform(
        lambda s: pd.Series(_streak_values(list(s)), index=s.index)
    )
    return frame


def add_price_features(panel: pd.DataFrame, ctx: ReturnContext) -> pd.DataFrame:
    frame = panel.copy()
    runup20, runup60, momentum_raw, momentum_ab = [], [], [], []
    for row in frame.itertuples():
        if pd.isna(row.day0):
            runup20.append(np.nan)
            runup60.append(np.nan)
            momentum_raw.append(np.nan)
            momentum_ab.append(np.nan)
            continue
        runup20.append(ctx.car(row.ticker, row.day0, -20, -1))
        runup60.append(ctx.car(row.ticker, row.day0, -60, -1))
        momentum_raw.append(ctx.cumulative_raw(row.ticker, row.day0, -252, -21))
        momentum_ab.append(ctx.car(row.ticker, row.day0, -252, -21))
    frame["runup_20d"] = runup20
    frame["runup_60d"] = runup60
    # Two conventions, named explicitly. runup_20d/60d are market-adjusted
    # (abnormal) returns, so a momentum feature built from raw returns is not
    # comparable with them: "short-horizon run-up dampens, long-horizon momentum
    # amplifies" would be partly a statement about market beta rather than about
    # the firm. momentum_12_1 is the abnormal version, matching the run-ups;
    # momentum_12_1_raw keeps the total-return version for reference.
    frame["momentum_12_1_raw"] = momentum_raw
    frame["momentum_12_1"] = momentum_ab
    return frame


def add_idiosyncratic_vol(
    panel: pd.DataFrame,
    ctx,
    window: tuple[int, int] = (-60, -11),
    min_obs: int = 30,
) -> pd.DataFrame:
    """Pre-event idiosyncratic volatility: std of daily abnormal returns over
    ``window``, ending well before the announcement so the estimation window
    cannot contain any of the reaction or the run-up into it.

    This exists to test whether a change in the estimated surprise effect is a
    change in pricing or just a change in volatility. SUE is on a fixed scale
    but the reaction is not, so in a high-volatility period the same
    informational surprise produces a mechanically larger abnormal return and
    the regression coefficient rises without anything about price formation
    having changed. Dividing the reaction by this quantity puts every event on
    a comparable scale.
    """
    frame = panel.copy()
    start, end = window
    vols: list[float] = []
    for row in frame.itertuples():
        if pd.isna(row.day0):
            vols.append(np.nan)
            continue
        dates = ctx.window_dates(row.day0, start, end)
        series = ctx.ar.get(row.ticker)
        if dates is None or series is None:
            vols.append(np.nan)
            continue
        values = series.reindex(dates).dropna()
        vols.append(float(values.std(ddof=1)) if len(values) >= min_obs else np.nan)
    frame["idio_vol"] = vols
    return frame


def add_vol_adjusted_target(
    panel: pd.DataFrame, target: str = "car_reaction", floor_quantile: float = 0.01
) -> pd.DataFrame:
    """``<target>_vol_adj`` = reaction divided by pre-event idiosyncratic vol.

    The denominator is floored at its 1st percentile so a handful of unusually
    quiet stocks cannot generate enormous standardised reactions, which would
    reproduce the SUE-denominator problem one column over.
    """
    frame = panel.copy()
    if "idio_vol" not in frame.columns or target not in frame.columns:
        return frame
    vol = frame["idio_vol"]
    valid = vol.dropna()
    if valid.empty:
        frame[f"{target}_vol_adj"] = np.nan
        return frame
    floor = float(valid.quantile(floor_quantile))
    denom = vol.clip(lower=floor)
    frame[f"{target}_vol_adj"] = frame[target] / denom
    logger.info(
        "vol-adjusted target built: median idio_vol %.4f, denominator floored at %.4f",
        float(valid.median()), floor,
    )
    return frame


def _first_available(ctx, candidates, label: str) -> str:
    """First candidate symbol whose prices were actually harvested.

    Index symbols such as ^TNX and ^VIX are gated behind higher data plans and
    return 402 on lower ones, which silently leaves the corresponding feature
    entirely NaN. Falling back to a liquid ETF proxy (IEF/TLT for the long end,
    VIXY/VXX for volatility) keeps the feature alive on a restricted plan. The
    proxy is a different quantity from the index level, so which symbol was used
    is logged and recorded in the panel's attrs for the write-up.
    """
    if isinstance(candidates, str):
        candidates = (candidates,)
    for symbol in candidates:
        series = ctx.level.get(symbol)
        if series is not None and series.notna().any():
            if symbol != candidates[0]:
                logger.warning(
                    "%s series %s unavailable; using %s as a proxy. It is not the same "
                    "quantity as the index level, so describe it as a proxy",
                    label, candidates[0], symbol,
                )
            return symbol
    logger.warning(
        "no %s series available from %s; the corresponding feature will be all-NaN "
        "and dropped by the feature-coverage filter",
        label, list(candidates),
    )
    return candidates[0]


def add_market_state(
    panel: pd.DataFrame,
    ctx: ReturnContext,
    vix_symbol: str | tuple[str, ...] = ("^VIX", "VIXY", "VXX"),
    rate_symbol: str | tuple[str, ...] = ("^TNX", "^TYX", "IEF", "TLT"),
) -> pd.DataFrame:
    frame = panel.copy()
    vix_used = _first_available(ctx, vix_symbol, "volatility")
    rate_used = _first_available(ctx, rate_symbol, "interest rate")
    vix, rate = [], []
    for row in frame.itertuples():
        if pd.isna(row.day0):
            vix.append(np.nan)
            rate.append(np.nan)
            continue
        vix.append(ctx.level_on_prior_day(vix_used, row.day0))
        rate.append(ctx.level_on_prior_day(rate_used, row.day0))
    frame["vix_level"] = vix
    frame["rate_level"] = rate
    frame.attrs["vix_symbol_used"] = vix_used
    frame.attrs["rate_symbol_used"] = rate_used
    return frame


def merge_fundamentals(
    panel: pd.DataFrame,
    fundamentals: pd.DataFrame,
    publication_lag_days: int = 90,
) -> pd.DataFrame:
    """Attach the most recent fundamentals available *at the event*.

    The `date` on a key-metrics row is the fiscal period end, not the date the
    figures became public. Matching an event to the latest period-end on or
    before it therefore attaches numbers that had not yet been filed: a period
    ending 31 March is typically published 60-90 days later, so an event in
    mid-April would be given data from a report that did not exist. Every
    fundamental row is shifted forward by ``publication_lag_days`` before the
    as-of match; with annual data the cost is a staler figure, which is the
    correct direction to err.
    """
    if fundamentals is None or fundamentals.empty:
        frame = panel.copy()
        for column in ("pe_ratio", "pe_z", "market_cap"):
            frame[column] = np.nan
        return frame
    funda = fundamentals.sort_values(["ticker", "date"]).copy()
    funda["pe_ratio"] = pd.to_numeric(funda["pe_ratio"], errors="coerce")
    rolling_mean = funda.groupby("ticker")["pe_ratio"].transform(
        lambda s: s.rolling(12, min_periods=6).mean()
    )
    rolling_std = funda.groupby("ticker")["pe_ratio"].transform(
        lambda s: s.rolling(12, min_periods=6).std()
    )
    funda["pe_z"] = (funda["pe_ratio"] - rolling_mean) / rolling_std.replace(0.0, np.nan)

    left = panel.copy()
    left["asof_date"] = pd.to_datetime(left["announce_date"]) - pd.Timedelta(days=1)
    left = left.sort_values("asof_date")
    right = funda[["ticker", "date", "pe_ratio", "pe_z", "market_cap"]].copy()
    right["period_end"] = right["date"]
    right["date"] = right["date"] + pd.Timedelta(days=publication_lag_days)
    right = right.sort_values("date")
    merged = pd.merge_asof(
        left,
        right,
        left_on="asof_date",
        right_on="date",
        by="ticker",
        direction="backward",
    )
    return merged.drop(columns=["asof_date", "date", "period_end"]).sort_values(
        ["ticker", "announce_date"]
    ).reset_index(drop=True)


def add_mcap_decile(panel: pd.DataFrame, min_group: int = 10) -> pd.DataFrame:
    frame = panel.copy()
    frame["announce_quarter"] = pd.PeriodIndex(
        pd.to_datetime(frame["announce_date"]), freq="Q"
    ).astype(str)

    def decile(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) < min_group:
            return pd.Series(np.nan, index=series.index)
        ranks = series.rank(method="first")
        return pd.qcut(ranks, 10, labels=False, duplicates="drop") + 1

    frame["mcap_decile"] = frame.groupby("announce_quarter")["market_cap"].transform(decile)
    return frame
