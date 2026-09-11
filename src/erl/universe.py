from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from erl.fmp import FMPClient

logger = logging.getLogger(__name__)

# Stable index-constituent endpoints (v3 legacy retired 2025-08-31).
CURRENT_ENDPOINT = "/stable/sp500-constituent"
HISTORY_ENDPOINT = "/stable/historical-sp500-constituent"


def _parse_date(record: dict) -> pd.Timestamp | None:
    """Date of the membership change.

    Only ``date`` is the change date. ``dateAdded`` is the date the *added*
    security joined, which for a removal-only record is an unrelated, much
    earlier date, so it must never be used as a fallback for the change date:
    doing so would place a 2009 removal in 1957.
    """
    value = record.get("date")
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else parsed.normalize()


def parse_change(record: dict) -> tuple[str, str]:
    """(added_ticker, removed_ticker) for one change-log record.

    The payload uses ``symbol`` for the ticker that joined and
    ``removedTicker``/``removedSecurity`` for the one that left, with
    ``addedSecurity`` naming the joiner. A record can carry both (a straight
    swap), only an addition, or only a removal.

    The trap is a removal-only record: some rows repeat the departing ticker in
    ``symbol`` while leaving ``addedSecurity`` blank. Reading ``symbol`` as the
    joiner unconditionally then turns a removal into an addition, which in the
    backward pass cancels itself out (discard then add) and in the forward pass
    opens a fresh spell starting on the day the firm actually left the index.
    ``symbol`` is therefore only treated as an addition when ``addedSecurity``
    names something, or when there is no removal in the record at all.
    """
    symbol = str(record.get("symbol") or "").strip().upper()
    removed = str(record.get("removedTicker") or "").strip().upper()
    added_name = str(record.get("addedSecurity") or "").strip()
    if added_name:
        added = symbol
    elif removed:
        # removal-only row; symbol echoes the departing ticker (or is blank)
        added = "" if (not symbol or symbol == removed) else symbol
    else:
        added = symbol
    return added, removed


def fetch_membership(client: FMPClient, start_date: str) -> pd.DataFrame:
    current_raw = client.get(CURRENT_ENDPOINT) or []
    history_raw = client.get(HISTORY_ENDPOINT) or []
    current = {str(r.get("symbol", "")).strip().upper() for r in current_raw if r.get("symbol")}
    events: list[tuple[pd.Timestamp, str, str]] = []
    undated = 0
    for record in history_raw:
        date = _parse_date(record)
        if date is None:
            undated += 1
            continue
        added, removed = parse_change(record)
        if not added and not removed:
            continue
        events.append((date, added, removed))
    logger.info(
        "membership change log: %d dated records (%d skipped with no usable date), "
        "%d current constituents",
        len(events), undated, len(current),
    )
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

    unmatched_removals = 0
    for date, added, removed in events:
        if date <= start:
            continue
        if removed:
            if removed in open_spells:
                intervals.append((removed, open_spells.pop(removed), date))
            else:
                # A removal with no open spell means the backward pass and the
                # forward pass disagree about who was a member at `start`.
                unmatched_removals += 1
        if added and added not in open_spells:
            open_spells[added] = date

    for ticker, added_date in open_spells.items():
        intervals.append((ticker, added_date, None))

    frame = pd.DataFrame(intervals, columns=["ticker", "added_date", "removed_date"])
    frame["added_date"] = pd.to_datetime(frame["added_date"])
    frame["removed_date"] = pd.to_datetime(frame["removed_date"])
    frame = frame.sort_values(["ticker", "added_date"], na_position="first").reset_index(drop=True)
    ever_removed = int(frame["removed_date"].notna().sum())
    logger.info(
        "membership built: %d spells, %d unique tickers, %d spells ended inside the sample",
        len(frame), frame["ticker"].nunique(), ever_removed,
    )
    if unmatched_removals:
        logger.warning(
            "%d removal records had no open spell to close; the change log may be "
            "incomplete before %s, which would understate the delisted universe",
            unmatched_removals, start.date(),
        )
    if ever_removed == 0 and len(frame) > 0:
        logger.warning(
            "no spell ends inside the sample: every name in the universe is a current "
            "constituent, so the panel is survivorship-biased despite using the "
            "historical endpoint"
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


def membership_coverage(
    membership: pd.DataFrame, harvested: set[str] | list[str]
) -> pd.DataFrame:
    """Did the data actually arrive for the names the universe says existed?

    Reconstructing point-in-time membership correctly is only half of removing
    survivorship bias. If the price endpoint returns nothing for a firm that was
    delisted or acquired (Lehman, Washington Mutual, a firm taken private), that
    firm silently disappears at the panel stage and the bias is back, now harder
    to see because the universe step reported success. This compares the
    universe against what was actually harvested, split by whether the name left
    the index during the sample, and is the evidence for or against calling the
    panel survivorship-bias-free.
    """
    have = {str(t).strip().upper() for t in harvested}
    frame = membership.copy()
    frame["left_index"] = frame["removed_date"].notna()
    per_ticker = (
        frame.groupby("ticker")["left_index"].max().rename("left_index").reset_index()
    )
    per_ticker["harvested"] = per_ticker["ticker"].isin(have)
    out = (
        per_ticker.groupby("left_index")
        .agg(tickers=("ticker", "size"), harvested=("harvested", "sum"))
        .reset_index()
    )
    out["coverage"] = out["harvested"] / out["tickers"]
    out["group"] = np.where(out["left_index"], "left the index", "still a member")
    out = out[["group", "tickers", "harvested", "coverage"]]

    missing = sorted(per_ticker.loc[~per_ticker["harvested"], "ticker"])
    out.attrs["missing_tickers"] = missing
    out.attrs["by_era"] = coverage_by_era(membership, have).to_dict("records")
    logger.info("universe coverage after harvest:\n%s", out.to_string(index=False))

    left = out.loc[out["group"] == "left the index", "coverage"]
    stayed = out.loc[out["group"] == "still a member", "coverage"]
    if not left.empty and not stayed.empty:
        gap = float(stayed.iloc[0] - left.iloc[0])
        if gap > 0.10:
            logger.warning(
                "price data covers %.0f%% of current members but only %.0f%% of names "
                "that left the index: residual survivorship bias of %.0f pp. Report "
                "this number rather than claiming a bias-free panel",
                100 * stayed.iloc[0], 100 * left.iloc[0], 100 * gap,
            )
    if missing:
        logger.info("no data harvested for %d names, e.g. %s", len(missing), missing[:10])
    return out


def coverage_by_era(
    membership: pd.DataFrame, harvested: set[str] | list[str]
) -> pd.DataFrame:
    """Coverage of departed names by the year they left the index.

    A single headline coverage figure hides where the gap sits. Firms that left
    long ago are less likely to be retrievable, so the missing names concentrate
    in the early sample, which means the residual survivorship bias is not
    spread evenly across the panel: it falls hardest on exactly the regimes that
    the stability analysis compares. Report this table alongside the headline
    number so a reader can see which period is affected.
    """
    have = {str(t).strip().upper() for t in harvested}
    left = membership.loc[membership["removed_date"].notna()].copy()
    if left.empty:
        return pd.DataFrame(columns=["removal_year", "tickers", "harvested", "coverage"])
    left["removal_year"] = pd.to_datetime(left["removed_date"]).dt.year
    left["harvested"] = left["ticker"].isin(have)
    out = (
        left.groupby("removal_year")
        .agg(tickers=("ticker", "nunique"), harvested=("harvested", "sum"))
        .reset_index()
    )
    out["coverage"] = out["harvested"] / out["tickers"]
    return out
