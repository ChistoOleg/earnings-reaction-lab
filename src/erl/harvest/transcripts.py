from __future__ import annotations

import logging

import pandas as pd

from erl.fmp import FMPClient, FMPError

logger = logging.getLogger(__name__)

# Stable transcript endpoint (Ultimate plan). Confirm exact params against your
# plan's docs when you reach Part 5; symbol/year (and sometimes quarter) are used.
TRANSCRIPT_ENDPOINT = "/stable/earning-call-transcript"


def parse_transcripts(ticker: str, rows: list[dict]) -> pd.DataFrame:
    records: list[dict] = []
    for row in rows or []:
        content = row.get("content")
        if not content:
            continue
        when = pd.to_datetime(row.get("date"), errors="coerce")
        records.append(
            {
                "ticker": ticker.upper(),
                "quarter": row.get("quarter") or row.get("period"),
                "year": row.get("year"),
                "call_date": None if pd.isna(when) else when.normalize(),
                "content": str(content),
            }
        )
    return pd.DataFrame(records)


def harvest_transcripts(
    client: FMPClient,
    tickers: list[str],
    years: list[int],
    out_path=None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        for year in years:
            try:
                rows = client.get(TRANSCRIPT_ENDPOINT, {"symbol": ticker, "year": year})
            except FMPError as exc:
                if exc.status in (402, 403):
                    raise FMPError(
                        "transcript endpoint rejected the request: earnings-call "
                        "transcripts require the FMP Ultimate plan. Subscribe for one "
                        "month, run this harvest to build the local cache, then "
                        "downgrade.",
                        status=exc.status,
                    ) from exc
                logger.warning("transcript harvest failed for %s %s: %s", ticker, year, exc)
                continue
            parsed = parse_transcripts(ticker, rows)
            if not parsed.empty:
                frames.append(parsed)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out_path is not None and not combined.empty:
        from erl.utils import write_parquet

        write_parquet(combined, out_path)
    return combined
