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
    """Clip each event to quantiles computed from strictly earlier events.

    Pooled quantiles would clip a 2017 event using 2024 information. Harmless for
    a full-sample estimate, not for anything evaluated out of time. Events with
    fewer than ``min_prior`` predecessors are left unclipped.
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
    """Standardised unexpected earnings: the surprise over the standard deviation
    of the firm's own past surprises.

    That denominator goes near zero for firms whose past surprises were all tiny,
    giving SUE values past +-20. Rank-based sorts do not care, but the continuous
    estimators would be driven by a handful of events, so ``sue`` is winsorised
    and ``sue_raw`` keeps the original. ``winsor_mode="asof"`` takes the quantiles
    from prior events only; ``"pooled"`` uses the full sample, fine for in-sample
    inference but leaky for anything evaluated out of time.
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
    # Both horizons market-adjusted so they are comparable: a raw-return momentum
    # feature next to abnormal-return run-ups makes any contrast between them
    # partly a statement about beta. The total-return version is kept alongside.
    frame["momentum_12_1_raw"] = momentum_raw
    frame["momentum_12_1"] = momentum_ab
    return frame


def add_idiosyncratic_vol(
    panel: pd.DataFrame,
    ctx,
    window: tuple[int, int] = (-60, -11),
    min_obs: int = 30,
) -> pd.DataFrame:
    """Standard deviation of daily abnormal returns over ``window``, which ends
    well before day 0 so neither the reaction nor the run-up can inflate it.

    SUE is standardised by construction and the reaction is not, so a
    high-volatility period produces larger raw reactions with no change in
    pricing. Dividing by this puts every event on a comparable scale.
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
    """Reaction divided by pre-event idiosyncratic volatility, with the denominator
    floored at its 1st percentile so a few unusually quiet stocks do not recreate
    the SUE-denominator problem one column over."""
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
    """First candidate whose prices actually arrived.

    ^TNX and ^VIX are gated behind higher data plans and 402 on lower ones, which
    would leave the feature entirely NaN. An ETF proxy keeps it alive, but it is a
    different quantity from the index level, so the symbol used is logged and
    recorded for the write-up.
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
    """Attach the most recent fundamentals actually available at the event.

    The key-metrics `date` is the fiscal period end, not the filing date, so
    matching on it hands an April event figures from a report published in June.
    Rows are shifted forward by ``publication_lag_days`` first; the cost is a
    staler figure, which is the right direction to err.
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
