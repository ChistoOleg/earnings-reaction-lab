"""Stock splits, and applying them to a price series.

FMP's price endpoints do not split-adjust. Both `historical-price-eod/full` and
`historical-price-eod/dividend-adjusted` return the raw traded price, so a 2:1
split reads as a -50% one-day return. Verified on AGN across its 2007-06-25
split: the dividend-adjusted endpoint goes 114.47 to 58.00 overnight.

For an event study that is not a cosmetic problem. An unadjusted split inside a
reaction window fabricates a +-50% earnings reaction; inside a run-up or
momentum window it corrupts the feature; and inside the 50-day window used for
pre-event idiosyncratic volatility it corrupts the denominator that scales every
reaction for that firm over a two-month stretch.

The adjustment itself is arithmetic. A split with numerator `n` and denominator
`d` (2:1 is n=2, d=1) multiplies the share count by n/d and divides the price by
the same, so every price *strictly before* the split date is multiplied by d/n
to put it on the post-split basis. Multiple splits compound.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from erl.fmp import FMPClient, FMPError

logger = logging.getLogger(__name__)

SPLITS_ENDPOINT = "/stable/splits"


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_splits(ticker: str, rows: list[dict], start_date: str) -> pd.DataFrame:
    """Split events at or after ``start_date``.

    A split before the sample start needs no handling: every price in the sample
    is already on the post-split basis.
    """
    start = pd.Timestamp(start_date).normalize()
    records: list[dict] = []
    for row in rows or []:
        when = pd.to_datetime(row.get("date"), errors="coerce")
        numerator = _to_float(row.get("numerator"))
        denominator = _to_float(row.get("denominator"))
        if pd.isna(when) or not numerator or not denominator:
            continue
        if when.normalize() < start:
            continue
        if numerator == denominator:
            continue  # a 1:1 "split" changes nothing
        records.append(
            {
                "ticker": ticker.upper(),
                "date": when.normalize(),
                "numerator": numerator,
                "denominator": denominator,
                "ratio": numerator / denominator,
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["ticker", "date"]).sort_values("date")
    return frame.reset_index(drop=True)


def harvest_splits(
    client: FMPClient,
    tickers: list[str],
    start_date: str,
    out_path=None,
    limit: int = 100,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for ticker in tickers:
        try:
            rows = client.get(SPLITS_ENDPOINT, {"symbol": ticker, "limit": limit})
        except FMPError as exc:
            logger.warning("split harvest failed for %s: %s", ticker, exc)
            failures.append(ticker)
            continue
        parsed = parse_splits(ticker, rows, start_date)
        if not parsed.empty:
            frames.append(parsed)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["ticker", "date", "numerator", "denominator", "ratio"]
    )
    if failures:
        logger.warning("split harvest finished with %d failures: %s", len(failures), failures[:20])
    if not combined.empty:
        logger.info(
            "%d split events across %d tickers in the sample window",
            len(combined), combined["ticker"].nunique(),
        )
    if out_path is not None:
        from erl.utils import write_parquet

        write_parquet(combined, out_path)
    return combined


# How far the observed price ratio may sit from its expected value and still be
# recognised. A split day also carries ordinary price movement, so this has to
# be loose enough to absorb that and tight enough to separate "halved" from
# "continuous".
RATIO_TOLERANCE = 0.15


def split_is_present(
    prices: pd.Series, split_date: pd.Timestamp, ratio: float
) -> tuple[bool, float | None]:
    """Is the split visible as a jump in this price series, or already adjusted?

    Vendors back-adjust inconsistently: FMP returns a raw traded price for some
    symbols (verified on AGN across its 2007 split) and an already-adjusted one
    for others, apparently depending on whether the name is still active.
    Applying the factor to a series that is already adjusted introduces an
    artifact instead of removing one, so each split event is checked against the
    prices before it is applied.

    Returns (present, observed_gross). ``present`` is True only when the jump is
    actually there; an ambiguous series (a genuine large move on the split date)
    returns False so that nothing is changed on a guess.
    """
    before = prices.loc[prices.index < split_date]
    after = prices.loc[prices.index >= split_date]
    if before.empty or after.empty:
        return False, None
    previous, current = before.iloc[-1], after.iloc[0]
    if not np.isfinite(previous) or not np.isfinite(current) or previous == 0:
        return False, None
    observed = float(current / previous)
    # Unadjusted: the price falls by the split ratio, so observed * ratio ~ 1.
    if abs(observed * ratio - 1.0) <= RATIO_TOLERANCE:
        return True, observed
    return False, observed


def split_factors(splits: pd.DataFrame, dates: pd.DatetimeIndex, ticker: str) -> np.ndarray:
    """Cumulative adjustment factor per date for one ticker.

    The factor for date t is the product of d/n over every split strictly after
    t, so prices before a 2:1 split are halved and prices from the split date
    onward are untouched.
    """
    factors = np.ones(len(dates), dtype=float)
    if splits.empty:
        return factors
    events = splits.loc[splits["ticker"] == ticker]
    for event in events.itertuples():
        before = dates < event.date
        factors[before] *= event.denominator / event.numerator
    return factors


def adjust_for_splits(prices: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Return ``prices`` with ``adj_close`` (and ``close``) on a split-adjusted
    basis, plus a ``split_adjusted`` flag naming which tickers were changed."""
    frame = prices.sort_values(["ticker", "date"]).copy()
    frame["split_adjusted"] = False
    if splits.empty:
        logger.warning(
            "no split data available; prices are left unadjusted and any split in the "
            "sample will read as a large one-day return"
        )
        return frame

    affected = set(splits["ticker"].unique())
    pieces: list[pd.DataFrame] = []
    changed = 0
    applied = 0
    already = 0
    ambiguous = 0
    for ticker, group in frame.groupby("ticker", sort=False):
        if ticker not in affected:
            pieces.append(group)
            continue
        series = pd.Series(
            group["adj_close"].to_numpy(dtype=float), index=pd.DatetimeIndex(group["date"])
        )
        events = splits.loc[splits["ticker"] == ticker]
        usable = []
        for event in events.itertuples():
            present, observed = split_is_present(series, event.date, event.ratio)
            if present:
                usable.append(event.Index)
                applied += 1
            elif observed is not None and abs(observed - 1.0) <= RATIO_TOLERANCE:
                already += 1  # the vendor already adjusted this one
            else:
                ambiguous += 1
                logger.debug(
                    "%s split on %s (ratio %.3f): observed gross %.3f, neither a visible "
                    "split nor a continuous series; left alone",
                    ticker, event.date.date(), event.ratio,
                    observed if observed is not None else float("nan"),
                )
        if not usable:
            pieces.append(group)
            continue
        dates = pd.DatetimeIndex(group["date"])
        factors = split_factors(splits.loc[usable], dates, ticker)
        if np.allclose(factors, 1.0):
            pieces.append(group)
            continue
        group = group.copy()
        group["adj_close"] = group["adj_close"].to_numpy(dtype=float) * factors
        if "close" in group.columns:
            group["close"] = group["close"].to_numpy(dtype=float) * factors
        # `volume` is deliberately left on its as-traded basis. A consistent
        # adjustment would divide it by the same factor, but nothing in the
        # project computes anything from volume, and rescaling it would make the
        # stored figure disagree with every external source for no benefit.
        group["split_adjusted"] = True
        changed += 1
        pieces.append(group)
    out = pd.concat(pieces, ignore_index=True).sort_values(["ticker", "date"])
    logger.info(
        "split adjustment: %d tickers changed; of %d split events, %d were visible in the "
        "prices and applied, %d were already adjusted by the vendor, %d ambiguous and left "
        "alone",
        changed, len(splits), applied, already, ambiguous,
    )
    if already:
        logger.info(
            "the price feed back-adjusts some symbols and not others, so each split is "
            "checked against the observed price before it is applied"
        )
    out.attrs["splits_applied"] = applied
    out.attrs["splits_already_adjusted"] = already
    out.attrs["splits_ambiguous"] = ambiguous
    return out.reset_index(drop=True)


def verify_adjustment(prices: pd.DataFrame) -> int:
    """Count remaining split-shaped daily returns after adjustment.

    Zero is the target. A non-zero count means the split feed missed events, and
    those returns are still fabricated reactions.
    """
    import logging as _logging

    from erl.events.returns import daily_returns, suspected_split_artifacts

    # daily_returns runs the same screen and would log its warning a second time.
    returns_logger = _logging.getLogger("erl.events.returns")
    previous = returns_logger.level
    returns_logger.setLevel(_logging.ERROR)
    try:
        returns = daily_returns(prices)
    finally:
        returns_logger.setLevel(previous)
    remaining = suspected_split_artifacts(returns)
    count = len(remaining)
    if count:
        logger.warning(
            "%d split-shaped returns remain after adjustment (e.g. %s on %s, %.1f%%); "
            "the split feed is incomplete for these names",
            count,
            remaining["ticker"].iloc[0],
            remaining["date"].iloc[0].date(),
            100 * remaining["ret"].iloc[0],
        )
    else:
        logger.info("no split-shaped returns remain after adjustment")
    return count


def unexplained_artifacts(prices: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Split-shaped returns that the splits feed does not account for.

    Distinguishes two cases that a naive ticker-level merge conflates: a ticker
    the feed covers but with no split near this date, and a ticker the feed does
    not cover at all. The second is the more common one, since coverage of
    delisted names is patchy.
    """
    from erl.events.returns import daily_returns, suspected_split_artifacts

    artifacts = suspected_split_artifacts(daily_returns(prices))
    if artifacts.empty:
        return artifacts.assign(feed_covers_ticker=[], near_feed_split=[])
    covered = set(splits["ticker"].unique()) if not splits.empty else set()
    artifacts = artifacts.copy()
    artifacts["feed_covers_ticker"] = artifacts["ticker"].isin(covered)
    near = []
    for row in artifacts.itertuples():
        if not row.feed_covers_ticker:
            near.append(False)
            continue
        events = splits.loc[splits["ticker"] == row.ticker, "date"]
        near.append(bool(((events - row.date).abs() <= pd.Timedelta("4D")).any()))
    artifacts["near_feed_split"] = near
    return artifacts


def suspicious_tickers(
    artifacts: pd.DataFrame, min_artifacts: int = 3
) -> pd.DataFrame:
    """Tickers whose history contains several unexplained split-shaped moves.

    One such move is usually a real crash or spike. Several on one ticker, at
    ratios like 2:1 then 3:1 then 10:1 within months, is the signature of a
    recycled ticker: a delisted company's price history spliced onto a different
    company that later took the symbol. Returns computed across the join are
    meaningless, and every feature built from them inherits that.

    Flagged rather than dropped automatically, because the decision to exclude a
    name from the sample belongs in the write-up, not in a silent filter.
    """
    if artifacts.empty:
        return pd.DataFrame(columns=["ticker", "artifacts", "first", "last", "ratios"])
    counts = (
        artifacts.groupby("ticker")
        .agg(
            artifacts=("ret", "size"),
            first=("date", "min"),
            last=("date", "max"),
            ratios=("split_ratio", lambda r: ",".join(sorted({f"{x:g}" for x in r}))),
        )
        .reset_index()
    )
    flagged = counts.loc[counts["artifacts"] >= min_artifacts].sort_values(
        "artifacts", ascending=False
    )
    if not flagged.empty:
        logger.warning(
            "%d ticker(s) show %d+ unexplained split-shaped moves, which usually means a "
            "recycled symbol with two companies' price history spliced together: %s. "
            "Consider excluding them and saying so in the write-up",
            len(flagged), min_artifacts, flagged["ticker"].tolist(),
        )
    return flagged.reset_index(drop=True)


def events_touched(
    panel: pd.DataFrame,
    artifacts: pd.DataFrame,
    lookback: int = 60,
    lookahead: int = 1,
) -> int:
    """How many events have an artifact inside any window used to build them.

    The reaction window is (0, +1) but `idio_vol` reaches back 60 trading days
    and the run-ups 20 and 60, so a single bad return contaminates far more than
    one event. This measures the exposure instead of guessing at it.
    """
    if panel.empty or artifacts.empty or "day0" not in panel.columns:
        return 0
    bad = artifacts.groupby("ticker")["date"].apply(list).to_dict()
    touched = 0
    for row in panel.itertuples():
        dates = bad.get(row.ticker)
        if not dates or pd.isna(row.day0):
            continue
        day0 = pd.Timestamp(row.day0)
        low = day0 - pd.Timedelta(days=int(lookback * 1.6))  # calendar padding
        high = day0 + pd.Timedelta(days=int(lookahead * 1.6) + 1)
        if any(low <= pd.Timestamp(d) <= high for d in dates):
            touched += 1
    return touched
