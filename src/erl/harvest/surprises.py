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
                "revenue_actual": _to_float(row.get("revenueActual", row.get("revenue"))),
                "revenue_estimate": _to_float(row.get("revenueEstimated")),
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
    if out_path is not None and not combined.empty:
        from erl.utils import write_parquet

        write_parquet(combined, out_path)
    return combined
