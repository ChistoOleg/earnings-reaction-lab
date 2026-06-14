from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SAME_DAY_TIMES = {"bmo", "dmh"}
NEXT_DAY_TIMES = {"amc", "unknown"}

DEFAULT_WINDOWS: dict[str, tuple[int, int]] = {
    "car_reaction": (0, 1),
    "car_drift": (2, 21),
}


def trading_calendar(prices: pd.DataFrame, benchmark: str) -> pd.DatetimeIndex:
    dates = prices.loc[prices["ticker"] == benchmark, "date"]
    if dates.empty:
        raise ValueError(f"benchmark {benchmark} not found in prices")
    return pd.DatetimeIndex(sorted(pd.unique(dates)))


def align_day0(
    announce_date,
    announce_time: str,
    calendar: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    date = pd.Timestamp(announce_date).normalize()
    pos = int(calendar.searchsorted(date))
    if str(announce_time).lower() in SAME_DAY_TIMES:
        target = pos
    else:
        if pos < len(calendar) and calendar[pos] == date:
            target = pos + 1
        else:
            target = pos
    if target >= len(calendar):
        return None
    return calendar[target]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_values(["ticker", "date"]).copy()
    frame["ret"] = frame.groupby("ticker")["adj_close"].pct_change()
    return frame.dropna(subset=["ret"])[["ticker", "date", "ret"]]


class ReturnContext:
    def __init__(self, prices: pd.DataFrame, benchmark: str) -> None:
        self.benchmark = benchmark
        self.calendar = trading_calendar(prices, benchmark)
        self.pos = {date: i for i, date in enumerate(self.calendar)}
        returns = daily_returns(prices)
        bench = (
            returns.loc[returns["ticker"] == benchmark, ["date", "ret"]]
            .rename(columns={"ret": "bench_ret"})
        )
        merged = returns.merge(bench, on="date", how="left")
        merged["ar"] = merged["ret"] - merged["bench_ret"]
        self.ar = {
            ticker: group.set_index("date")["ar"]
            for ticker, group in merged.groupby("ticker")
        }
        self.ret = {
            ticker: group.set_index("date")["ret"]
            for ticker, group in returns.groupby("ticker")
        }
        self.level = {
            ticker: group.set_index("date")["adj_close"]
            for ticker, group in prices.sort_values(["ticker", "date"]).groupby("ticker")
        }

    def window_dates(self, day0: pd.Timestamp, start: int, end: int) -> pd.DatetimeIndex | None:
        pos0 = self.pos.get(pd.Timestamp(day0))
        if pos0 is None:
            return None
        lo, hi = pos0 + start, pos0 + end
        if lo < 0 or hi >= len(self.calendar):
            return None
        return self.calendar[lo : hi + 1]

    def car(self, ticker: str, day0, start: int, end: int) -> float:
        dates = self.window_dates(day0, start, end)
        series = self.ar.get(ticker)
        if dates is None or series is None:
            return np.nan
        values = series.reindex(dates)
        if values.isna().any():
            return np.nan
        return float(values.sum())

    def cumulative_raw(self, ticker: str, day0, start: int, end: int) -> float:
        dates = self.window_dates(day0, start, end)
        series = self.ret.get(ticker)
        if dates is None or series is None:
            return np.nan
        values = series.reindex(dates)
        if values.isna().any():
            return np.nan
        return float(np.prod(1.0 + values.to_numpy()) - 1.0)

    def level_on_prior_day(self, symbol: str, day0) -> float:
        pos0 = self.pos.get(pd.Timestamp(day0))
        series = self.level.get(symbol)
        if pos0 is None or pos0 == 0 or series is None:
            return np.nan
        prior = self.calendar[pos0 - 1]
        value = series.get(prior)
        return float(value) if value is not None and not pd.isna(value) else np.nan


def compute_cars(
    events: pd.DataFrame,
    ctx: ReturnContext,
    windows: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    windows = windows or DEFAULT_WINDOWS
    rows: list[dict] = []
    for event in events.itertuples():
        day0 = align_day0(event.announce_date, event.announce_time, ctx.calendar)
        row: dict = {"event_id": event.event_id, "day0": day0}
        for name, (start, end) in windows.items():
            row[name] = (
                ctx.car(event.ticker, day0, start, end) if day0 is not None else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
