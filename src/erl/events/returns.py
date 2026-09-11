from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Day-0 convention. Price data is close-to-close, so the first return that can
# contain the announcement is:
#   bmo / dmh  -> the announcement date itself (close(t-1) -> close(t))
#   amc        -> the next trading day (close(t) -> close(t+1))
#   unknown    -> the announcement date. The reaction window is (0, +1) precisely
#                 so that it covers both cases when timing is unknown; shifting
#                 unknown events forward instead makes the window start one day
#                 *after* a before-open reaction and miss it entirely. That was
#                 the source of the t-1 spike in the drift figure.
SAME_DAY_TIMES = {"bmo", "dmh", "unknown"}
NEXT_DAY_TIMES = {"amc"}

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
    if len(calendar) == 0 or date < calendar[0]:
        # The announcement predates the price history. searchsorted would return
        # position 0 and silently map the event onto the first available trading
        # day, which can be months or years later.
        return None
    pos = int(calendar.searchsorted(date))
    if str(announce_time).lower() in NEXT_DAY_TIMES:
        # After-close: reaction is in the next close-to-close return. If the
        # announcement fell on a non-trading day, searchsorted already points at
        # the next trading day and no further shift is needed.
        if pos < len(calendar) and calendar[pos] == date:
            target = pos + 1
        else:
            target = pos
    else:
        target = pos
    if target >= len(calendar):
        return None
    return calendar[target]


# Only whole-number ratios: 1.5:1 and 2.5:1 splits are rare and their gross
# ratios sit close enough to ordinary large moves in index levels (^VIX, ^TNX)
# to generate false positives.
SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
SPLIT_TOLERANCE = 0.015


def suspected_split_artifacts(
    returns: pd.DataFrame, threshold: float = 0.30, skip_prefix: str = "^"
) -> pd.DataFrame:
    """Daily returns whose gross ratio is close to a common split ratio.

    The price endpoint used by default (`historical-price-eod/full`) may return
    only `close`, in which case `adj_close` falls back to the unadjusted close
    and a 4:1 split shows up as a -75% one-day return. If such a day lands in an
    event window it becomes a fake earnings reaction. This screen flags the
    candidates so they can be inspected or the endpoint switched to the
    dividend-adjusted one.
    """
    candidates = returns.loc[returns["ret"].abs() >= threshold]
    if skip_prefix:
        # Index and volatility symbols are levels, not tradable prices: they are
        # never split-adjusted and routinely move this much.
        candidates = candidates.loc[~candidates["ticker"].str.startswith(skip_prefix)]
    frame = candidates.copy()
    if frame.empty:
        return frame.assign(gross=[], split_ratio=[])
    frame["gross"] = 1.0 + frame["ret"]
    def _match(gross: float) -> float | None:
        for ratio in SPLIT_RATIOS:
            for candidate in (1.0 / ratio, ratio):
                if abs(gross / candidate - 1.0) <= SPLIT_TOLERANCE:
                    return ratio
        return None
    frame["split_ratio"] = frame["gross"].map(_match)
    return frame.loc[frame["split_ratio"].notna()].reset_index(drop=True)


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_values(["ticker", "date"]).copy()
    frame["ret"] = frame.groupby("ticker")["adj_close"].pct_change()
    # A gap in a ticker's price history turns one pct_change into a multi-day
    # return. Flag the gaps rather than silently treating them as daily moves.
    gap = frame.groupby("ticker")["date"].diff().dt.days
    frame["stale_days"] = gap
    out = frame.dropna(subset=["ret"])[["ticker", "date", "ret", "stale_days"]]
    long_gaps = int((out["stale_days"] > 7).sum())
    if long_gaps:
        logger.warning(
            "%d daily returns span a price gap of more than 7 calendar days; "
            "these are multi-day returns, not daily ones",
            long_gaps,
        )
    suspects = suspected_split_artifacts(out)
    if not suspects.empty:
        logger.warning(
            "%d returns look like unadjusted split artifacts (e.g. %s on %s, %.1f%%); "
            "check whether the price endpoint returned adjClose",
            len(suspects),
            suspects["ticker"].iloc[0],
            suspects["date"].iloc[0].date(),
            100 * suspects["ret"].iloc[0],
        )
    return out.drop(columns=["stale_days"])


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
