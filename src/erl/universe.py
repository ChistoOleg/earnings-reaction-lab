from __future__ import annotations

import logging

import pandas as pd

from erl.fmp import FMPClient

logger = logging.getLogger(__name__)

# Stable index-constituent endpoints (v3 legacy retired 2025-08-31).
CURRENT_ENDPOINT = "/stable/sp500-constituent"
HISTORY_ENDPOINT = "/stable/historical-sp500-constituent"


def _parse_date(record: dict) -> pd.Timestamp | None:
    for key in ("date", "dateAdded"):
        value = record.get(key)
        if not value:
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            return parsed.normalize()
    return None


def fetch_membership(client: FMPClient, start_date: str) -> pd.DataFrame:
    current_raw = client.get(CURRENT_ENDPOINT) or []
    history_raw = client.get(HISTORY_ENDPOINT) or []
    current = {str(r.get("symbol", "")).strip().upper() for r in current_raw if r.get("symbol")}
    events: list[tuple[pd.Timestamp, str, str]] = []
    for record in history_raw:
        date = _parse_date(record)
        if date is None:
            continue
        added = str(record.get("symbol") or "").strip().upper()
        removed = str(record.get("removedTicker") or "").strip().upper()
        events.append((date, added, removed))
    return build_membership(current, events, start_date)


def build_membership(
    current: set[str],
    events: list[tuple[pd.Timestamp, str, str]],
    start_date: str,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    events = sorted(events, key=lambda e: e[0])

    members = set(current)
    for date, added, removed in sorted(events, key=lambda e: e[0], reverse=True):
        if date <= start:
            break
        if added:
            members.discard(added)
        if removed:
            members.add(removed)

    open_spells: dict[str, pd.Timestamp | None] = {t: None for t in members}
    intervals: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]] = []

    for date, added, removed in events:
        if date <= start:
            continue
        if removed and removed in open_spells:
            intervals.append((removed, open_spells.pop(removed), date))
        if added and added not in open_spells:
            open_spells[added] = date

    for ticker, added_date in open_spells.items():
        intervals.append((ticker, added_date, None))

    frame = pd.DataFrame(intervals, columns=["ticker", "added_date", "removed_date"])
    frame["added_date"] = pd.to_datetime(frame["added_date"])
    frame["removed_date"] = pd.to_datetime(frame["removed_date"])
    frame = frame.sort_values(["ticker", "added_date"], na_position="first").reset_index(drop=True)
    logger.info(
        "membership built: %d spells, %d unique tickers", len(frame), frame["ticker"].nunique()
    )
    return frame


def members_on(membership: pd.DataFrame, date: str | pd.Timestamp) -> set[str]:
    when = pd.Timestamp(date).normalize()
    added_ok = membership["added_date"].isna() | (membership["added_date"] <= when)
    removed_ok = membership["removed_date"].isna() | (membership["removed_date"] > when)
    return set(membership.loc[added_ok & removed_ok, "ticker"])


def union_members(membership: pd.DataFrame) -> list[str]:
    return sorted(membership["ticker"].unique())


def current_members(client: FMPClient) -> list[str]:
    """Current S&P 500 constituents only (no history). Survivorship-biased."""
    rows = client.get(CURRENT_ENDPOINT) or []
    symbols = {str(r.get("symbol", "")).strip().upper() for r in rows if r.get("symbol")}
    return sorted(s for s in symbols if s)


def membership_from_current(symbols: list[str]) -> pd.DataFrame:
    """Build a membership table treating every symbol as a current member with
    open-ended spells. members_on() then returns the full set for any date."""
    frame = pd.DataFrame(
        {"ticker": [s.upper() for s in symbols], "added_date": pd.NaT, "removed_date": pd.NaT}
    )
    frame["added_date"] = pd.to_datetime(frame["added_date"])
    frame["removed_date"] = pd.to_datetime(frame["removed_date"])
    return frame
