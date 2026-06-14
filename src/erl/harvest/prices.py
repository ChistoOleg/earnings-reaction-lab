from __future__ import annotations

import logging
from datetime import date as date_type

import pandas as pd

from erl.fmp import FMPClient, FMPError

logger = logging.getLogger(__name__)

# FMP "stable" API (v3/v4 legacy endpoints were retired 2025-08-31).
# `full` is the broadest free-tier-friendly EOD endpoint and works for stocks,
# ETFs and indexes (^GSPC etc.). For dividend/split-clean returns on a paid tier,
# switch PRICE_ENDPOINT to "historical-price-eod/dividend-adjusted" (gives adjClose).
PRICE_ENDPOINT = "/stable/historical-price-eod/full"


def parse_prices(symbol: str, payload, start_date: str) -> pd.DataFrame:
    if isinstance(payload, dict):
        rows = payload.get("historical") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    start = pd.Timestamp(start_date).normalize()
    records: list[dict] = []
    for row in rows:
        raw_date = row.get("date")
        if not raw_date:
            continue
        when = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(when) or when.normalize() < start:
            continue
        adj = row.get("adjClose", row.get("close"))
        if adj is None:
            continue
        records.append(
            {
                "ticker": symbol.upper(),
                "date": when.normalize(),
                "adj_close": float(adj),
                "close": float(row["close"]) if row.get("close") is not None else None,
                "volume": float(row["volume"]) if row.get("volume") is not None else None,
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["ticker", "date"]).sort_values("date")
    return frame.reset_index(drop=True)


def harvest_prices(
    client: FMPClient,
    symbols: list[str],
    start_date: str,
    out_path=None,
    end_date: str | None = None,
) -> pd.DataFrame:
    to_date = end_date or date_type.today().isoformat()
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for symbol in symbols:
        try:
            payload = client.get(
                PRICE_ENDPOINT,
                {"symbol": symbol, "from": start_date, "to": to_date},
            )
        except FMPError as exc:
            logger.warning("price harvest failed for %s: %s", symbol, exc)
            failures.append(symbol)
            continue
        parsed = parse_prices(symbol, payload, start_date)
        if not parsed.empty:
            frames.append(parsed)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if failures:
        logger.warning("price harvest finished with %d failures: %s", len(failures), failures)
    if out_path is not None and not combined.empty:
        from erl.utils import write_parquet

        write_parquet(combined, out_path)
    return combined
