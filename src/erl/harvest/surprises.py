from __future__ import annotations

import logging

import pandas as pd

from erl.fmp import FMPClient, FMPError
from erl.utils import event_id

logger = logging.getLogger(__name__)

# Stable per-company earnings endpoint: actual vs estimated EPS/revenue + timing.
EARNINGS_ENDPOINT = "/stable/earnings"

_TIME_MAP = {"bmo": "bmo", "amc": "amc", "dmh": "dmh"}


def _normalize_time(value) -> str:
    if not value:
        return "unknown"
    return _TIME_MAP.get(str(value).strip().lower(), "unknown")


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_backfilled(
    eps_actual: float,
    eps_estimate: float,
    revenue_actual: float | None,
    revenue_estimate: float | None,
) -> bool:
    """True when the "estimate" is really the actual copied into the estimate field.

    Where no consensus existed, the feed fills the estimate with the reported
    figure, which makes ``surprise`` exactly zero for reasons that have nothing
    to do with the market being unsurprised. These rows cluster in the early
    years of the sample, so leaving them in would load the earliest regime with
    artificial zero-surprise events and corrupt any comparison across periods.

    Revenue is the discriminating test: a real consensus revenue estimate is a
    ten-digit number and will not equal the reported figure to the dollar, while
    a copied placeholder matches exactly. An exact EPS match is not evidence on
    its own, because analysts genuinely hit a cents-rounded EPS often, so it only
    counts when revenue is unavailable to check.
    """
    if revenue_actual is not None and revenue_estimate is not None:
        return revenue_estimate == revenue_actual
    return eps_estimate == eps_actual


def parse_calendar(ticker: str, rows: list[dict], start_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    records: list[dict] = []
    for row in rows or []:
        date_raw = row.get("date")
        if not date_raw:
            continue
        date = pd.to_datetime(date_raw, errors="coerce")
        if pd.isna(date) or date.normalize() < start:
            continue
        # Stable field names: epsActual/epsEstimated/revenueActual/revenueEstimated.
        # Fall back to legacy names so the parser tolerates either shape.
        eps_actual = _to_float(row.get("epsActual", row.get("eps")))
        eps_estimate = _to_float(row.get("epsEstimated"))
        if eps_actual is None or eps_estimate is None:
            continue
        surprise = eps_actual - eps_estimate
        surprise_pct = surprise / abs(eps_estimate) if eps_estimate else None
        date_str = date.date().isoformat()
        revenue_actual = _to_float(row.get("revenueActual", row.get("revenue")))
        revenue_estimate = _to_float(row.get("revenueEstimated"))
        backfilled = _looks_backfilled(
            eps_actual, eps_estimate, revenue_actual, revenue_estimate
        )
        records.append(
            {
                "event_id": event_id(ticker, date_str),
                "ticker": ticker.upper(),
                "announce_date": pd.Timestamp(date_str),
                "announce_time": _normalize_time(row.get("time")),
                "fiscal_date_ending": row.get("fiscalDateEnding") or row.get("date"),
                "eps_actual": eps_actual,
                "eps_estimate": eps_estimate,
                "surprise": surprise,
                "surprise_pct": surprise_pct,
                "revenue_actual": revenue_actual,
                "revenue_estimate": revenue_estimate,
                "estimate_backfilled": backfilled,
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.drop_duplicates(subset="event_id").sort_values("announce_date")
    return frame.reset_index(drop=True)


def harvest_surprises(
    client: FMPClient,
    tickers: list[str],
    start_date: str,
    out_path=None,
    limit: int = 1000,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for ticker in tickers:
        try:
            rows = client.get(EARNINGS_ENDPOINT, {"symbol": ticker, "limit": limit})
        except FMPError as exc:
            logger.warning("surprise harvest failed for %s: %s", ticker, exc)
            failures.append(ticker)
            continue
        parsed = parse_calendar(ticker, rows, start_date)
        if not parsed.empty:
            frames.append(parsed)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if failures:
        logger.warning("surprise harvest finished with %d failures: %s", len(failures), failures)
    if not combined.empty and "estimate_backfilled" in combined.columns:
        rates = backfill_rate_by_year(combined)
        logger.info(
            "share of events whose estimate is the actual copied over, by year:\n%s",
            rates.to_string(index=False),
        )
        overall = float(combined["estimate_backfilled"].mean())
        if overall > 0.05:
            logger.warning(
                "%.0f%% of harvested events have a placeholder estimate; check the "
                "by-year table before choosing a start date, since these cluster early",
                100 * overall,
            )
    if out_path is not None and not combined.empty:
        from erl.utils import write_parquet

        write_parquet(combined, out_path)
    return combined


def backfill_rate_by_year(events: pd.DataFrame) -> pd.DataFrame:
    """Share of events per year whose estimate is a copy of the actual.

    Read this before setting the start date: the first year where the rate is
    low is the first year in which a surprise variable means anything.
    """
    frame = events.copy()
    frame["year"] = pd.to_datetime(frame["announce_date"]).dt.year
    out = (
        frame.groupby("year")["estimate_backfilled"]
        .agg(events="size", placeholder="sum")
        .reset_index()
    )
    out["placeholder_share"] = out["placeholder"] / out["events"]
    return out
