from __future__ import annotations

import logging

import pandas as pd

from erl.fmp import FMPClient, FMPError

logger = logging.getLogger(__name__)

KEY_METRICS_ENDPOINT = "/stable/key-metrics"


def _pick(row: dict, *names):
    for name in names:
        if row.get(name) is not None:
            return row.get(name)
    return None


def parse_key_metrics(ticker: str, rows: list[dict], start_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    records: list[dict] = []
    for row in rows or []:
        raw_date = row.get("date")
        if not raw_date:
            continue
        when = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(when) or when.normalize() < start:
            continue
        records.append(
            {
                "ticker": ticker.upper(),
                "date": when.normalize(),
                # Stable key-metrics naming varies; accept several spellings and
                # degrade to NaN if a field is absent (pe_z just becomes NaN).
                "pe_ratio": _pick(row, "peRatio", "priceToEarningsRatio", "pe"),
                "pb_ratio": _pick(row, "pbRatio", "priceToBookRatio"),
                "market_cap": _pick(row, "marketCap", "marketCapitalization"),
                "ev_to_sales": _pick(row, "evToSales", "enterpriseValueOverSales"),
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["ticker", "date"]).sort_values("date")
    return frame.reset_index(drop=True)


def harvest_fundamentals(
    client: FMPClient,
    tickers: list[str],
    start_date: str,
    out_path=None,
    limit: int = 80,
    period: str = "annual",
) -> pd.DataFrame:
    # "annual" works on the Starter plan; "quarter" needs Premium. Annual is
    # coarser (merge_asof still attaches the latest available figure per event).
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for ticker in tickers:
        try:
            rows = client.get(
                KEY_METRICS_ENDPOINT,
                {"symbol": ticker, "period": period, "limit": limit},
            )
        except FMPError as exc:
            logger.warning("fundamentals harvest failed for %s: %s", ticker, exc)
            failures.append(ticker)
            continue
        parsed = parse_key_metrics(ticker, rows, start_date)
        if not parsed.empty:
            frames.append(parsed)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if failures:
        logger.warning(
            "fundamentals harvest finished with %d failures: %s", len(failures), failures
        )
    if out_path is not None and not combined.empty:
        from erl.utils import write_parquet

        write_parquet(combined, out_path)
    return combined
