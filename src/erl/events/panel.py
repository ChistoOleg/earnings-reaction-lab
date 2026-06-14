from __future__ import annotations

import logging

import pandas as pd

from erl.events.features import (
    add_beat_flags,
    add_market_state,
    add_mcap_decile,
    add_price_features,
    add_prior_streak,
    add_sue,
    merge_fundamentals,
)
from erl.events.returns import DEFAULT_WINDOWS, ReturnContext, compute_cars

logger = logging.getLogger(__name__)


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
    vix_symbol: str = "^VIX",
    rate_symbol: str = "^TNX",
) -> pd.DataFrame:
    windows = windows or DEFAULT_WINDOWS
    ctx = ReturnContext(prices, benchmark)

    enriched = add_prior_streak(add_sue(add_beat_flags(events)))
    cars = compute_cars(enriched, ctx, windows)
    panel = enriched.merge(cars, on="event_id", how="left")
    panel = add_price_features(panel, ctx)
    panel = add_market_state(panel, ctx, vix_symbol, rate_symbol)
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
    leakage_checks(usable)
    return usable
