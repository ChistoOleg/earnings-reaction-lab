from __future__ import annotations

import logging

import pandas as pd

from erl.events.features import (
    add_beat_flags,
    add_idiosyncratic_vol,
    add_market_state,
    add_vol_adjusted_target,
    add_mcap_decile,
    add_price_features,
    add_prior_streak,
    add_sue,
    merge_fundamentals,
)
from erl.events.returns import DEFAULT_WINDOWS, ReturnContext, compute_cars
from erl.inference.eventstudy import alignment_diagnostic

logger = logging.getLogger(__name__)


def timing_coverage(events: pd.DataFrame) -> dict[str, float]:
    """Share of events by announcement-time label. Logged because a source that
    omits the field entirely changes the alignment rule for the whole sample."""
    counts = events["announce_time"].fillna("unknown").astype(str).str.lower()
    shares = counts.value_counts(normalize=True).to_dict()
    return {k: float(v) for k, v in shares.items()}


def check_alignment(panel: pd.DataFrame, ctx: ReturnContext) -> pd.DataFrame:
    table = alignment_diagnostic(panel, ctx)
    if table.empty:
        return table
    peak = table.attrs.get("peak_rel_day")
    logger.info(
        "alignment diagnostic (mean |AR| by day relative to day0):\n%s",
        table.to_string(index=False),
    )
    if peak not in (0, 1):
        logger.warning(
            "alignment diagnostic: |AR| peaks at rel_day=%s, expected 0 or +1. "
            "day0 is probably misaligned; check the announce_time labels.",
            peak,
        )
    return table


def leakage_checks(panel: pd.DataFrame) -> None:
    if panel.empty:
        return
    day0 = pd.to_datetime(panel["day0"])
    announce = pd.to_datetime(panel["announce_date"])
    if (day0 < announce).any():
        raise AssertionError("leakage: day0 earlier than announcement date")
    amc = panel["announce_time"].isin(["amc"])
    if (day0[amc] <= announce[amc]).any():
        raise AssertionError("leakage: AMC events must react on a later trading day")


def build_panel(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    benchmark: str = "^GSPC",
    windows: dict[str, tuple[int, int]] | None = None,
    # Ordered fallbacks: index symbols are gated behind higher data plans, so an
    # ETF proxy keeps the feature alive. See _first_available in features.py.
    vix_symbol: str | tuple[str, ...] = ("^VIX", "VIXY", "VXX"),
    rate_symbol: str | tuple[str, ...] = ("^TNX", "^TYX", "IEF", "TLT"),
) -> pd.DataFrame:
    windows = windows or DEFAULT_WINDOWS
    ctx = ReturnContext(prices, benchmark)

    if "estimate_backfilled" in events.columns:
        flagged = int(events["estimate_backfilled"].fillna(False).astype(bool).sum())
        if flagged:
            events = events.loc[~events["estimate_backfilled"].fillna(False).astype(bool)].copy()
            logger.warning(
                "dropped %d events whose analyst estimate is the reported figure copied "
                "over (surprise would be a spurious zero); %d events remain",
                flagged, len(events),
            )

    coverage = timing_coverage(events)
    logger.info("announcement-time coverage: %s", coverage)
    if coverage.get("unknown", 0.0) > 0.5:
        logger.warning(
            "%.0f%% of events have unknown announcement time; day0 = announcement "
            "date for those and the (0, +1) window absorbs before/after-close timing",
            100 * coverage["unknown"],
        )

    enriched = add_prior_streak(add_sue(add_beat_flags(events)))
    cars = compute_cars(enriched, ctx, windows)
    panel = enriched.merge(cars, on="event_id", how="left")
    panel = add_price_features(panel, ctx)
    panel = add_idiosyncratic_vol(panel, ctx)
    panel = add_market_state(panel, ctx, vix_symbol, rate_symbol)
    # Capture now: pandas drops attrs across the merge below.
    market_symbols = {
        key: panel.attrs.get(key)
        for key in ("vix_symbol_used", "rate_symbol_used")
        if panel.attrs.get(key)
    }
    panel = merge_fundamentals(panel, fundamentals)
    panel = add_mcap_decile(panel)

    total = len(panel)
    target = next(iter(windows))
    usable = panel.dropna(subset=["day0", target]).reset_index(drop=True)
    dropped = total - len(usable)
    if dropped:
        logger.info(
            "panel attrition: %d of %d events dropped (no day0 or missing %s)",
            dropped,
            total,
            target,
        )
    usable = add_vol_adjusted_target(usable, next(iter(windows)))
    usable.attrs.update(market_symbols)

    leakage_checks(usable)
    diagnostic = check_alignment(usable, ctx)
    # Must stay JSON-serialisable: pandas writes attrs into parquet metadata.
    usable.attrs["alignment_diagnostic"] = diagnostic.to_dict("records")
    usable.attrs["alignment_peak_rel_day"] = diagnostic.attrs.get("peak_rel_day")
    return usable
