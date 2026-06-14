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


def add_sue(events: pd.DataFrame, min_history: int = 4) -> pd.DataFrame:
    frame = events.sort_values(["ticker", "announce_date"]).copy()
    past_std = frame.groupby("ticker")["surprise"].transform(
        lambda s: s.shift(1).expanding(min_periods=min_history).std()
    )
    past_std = past_std.replace(0.0, np.nan)
    frame["sue"] = frame["surprise"] / past_std
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
    runup20, runup60, momentum = [], [], []
    for row in frame.itertuples():
        if pd.isna(row.day0):
            runup20.append(np.nan)
            runup60.append(np.nan)
            momentum.append(np.nan)
            continue
        runup20.append(ctx.car(row.ticker, row.day0, -20, -1))
        runup60.append(ctx.car(row.ticker, row.day0, -60, -1))
        momentum.append(ctx.cumulative_raw(row.ticker, row.day0, -252, -21))
    frame["runup_20d"] = runup20
    frame["runup_60d"] = runup60
    frame["momentum_12_1"] = momentum
    return frame


def add_market_state(
    panel: pd.DataFrame,
    ctx: ReturnContext,
    vix_symbol: str = "^VIX",
    rate_symbol: str = "^TNX",
) -> pd.DataFrame:
    frame = panel.copy()
    vix, rate = [], []
    for row in frame.itertuples():
        if pd.isna(row.day0):
            vix.append(np.nan)
            rate.append(np.nan)
            continue
        vix.append(ctx.level_on_prior_day(vix_symbol, row.day0))
        rate.append(ctx.level_on_prior_day(rate_symbol, row.day0))
    frame["vix_level"] = vix
    frame["rate_level"] = rate
    return frame


def merge_fundamentals(panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
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
    right = funda[["ticker", "date", "pe_ratio", "pe_z", "market_cap"]].sort_values("date")
    merged = pd.merge_asof(
        left,
        right,
        left_on="asof_date",
        right_on="date",
        by="ticker",
        direction="backward",
    )
    return merged.drop(columns=["asof_date", "date"]).sort_values(
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
