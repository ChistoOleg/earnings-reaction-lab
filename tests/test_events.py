from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erl.events.features import add_beat_flags, add_prior_streak, add_sue
from erl.events.panel import build_panel, leakage_checks
from erl.events.returns import ReturnContext, align_day0, compute_cars
from erl.utils import event_id


def make_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", "2024-12-31")
    frames = []
    bench = pd.DataFrame({"ticker": "^GSPC", "date": dates, "adj_close": 100.0})
    frames.append(bench)
    for symbol, level in (("^VIX", 15.0), ("^TNX", 4.2)):
        frames.append(pd.DataFrame({"ticker": symbol, "date": dates, "adj_close": level}))

    prices = np.full(len(dates), 100.0)
    day0 = pd.Timestamp("2024-05-02")
    idx0 = int(dates.searchsorted(day0))
    prices[idx0:] *= 1.02
    prices[idx0 + 1 :] *= 1.01
    frames.append(pd.DataFrame({"ticker": "TST", "date": dates, "adj_close": prices}))

    frame = pd.concat(frames, ignore_index=True)
    frame["close"] = frame["adj_close"]
    frame["volume"] = 1000.0
    return frame


def make_events() -> pd.DataFrame:
    rows = []
    for date, time in [
        ("2023-02-01", "bmo"),
        ("2023-05-03", "amc"),
        ("2023-08-02", "bmo"),
        ("2023-11-01", "amc"),
        ("2024-05-01", "amc"),
    ]:
        rows.append(
            {
                "event_id": event_id("TST", date),
                "ticker": "TST",
                "announce_date": pd.Timestamp(date),
                "announce_time": time,
                "eps_actual": 1.2,
                "eps_estimate": 1.0,
                "surprise": 0.2,
                "surprise_pct": 0.2,
                "revenue_actual": 100.0,
                "revenue_estimate": 95.0,
            }
        )
    return pd.DataFrame(rows)


CAL = pd.DatetimeIndex(pd.bdate_range("2024-04-29", "2024-05-10"))


def test_align_day0_bmo_same_day():
    assert align_day0("2024-05-01", "bmo", CAL) == pd.Timestamp("2024-05-01")


def test_align_day0_amc_next_trading_day():
    assert align_day0("2024-05-01", "amc", CAL) == pd.Timestamp("2024-05-02")


def test_align_day0_weekend_rolls_forward():
    assert align_day0("2024-05-04", "amc", CAL) == pd.Timestamp("2024-05-06")
    assert align_day0("2024-05-04", "bmo", CAL) == pd.Timestamp("2024-05-06")


def test_align_day0_unknown_treated_as_next_day():
    assert align_day0("2024-05-01", "unknown", CAL) == pd.Timestamp("2024-05-02")


def test_align_day0_beyond_calendar_returns_none():
    assert align_day0("2024-05-13", "bmo", CAL) is None


def test_car_windows_match_constructed_moves():
    prices = make_prices()
    ctx = ReturnContext(prices, "^GSPC")
    events = make_events()
    cars = compute_cars(events, ctx)
    row = cars[cars["event_id"] == event_id("TST", "2024-05-01")].iloc[0]
    assert row["day0"] == pd.Timestamp("2024-05-02")
    assert row["car_reaction"] == pytest.approx(0.03, abs=1e-6)
    assert row["car_drift"] == pytest.approx(0.0, abs=1e-9)


def test_car_missing_future_window_is_nan():
    prices = make_prices()
    prices = prices[prices["date"] <= "2024-05-03"]
    ctx = ReturnContext(prices, "^GSPC")
    events = make_events()
    cars = compute_cars(events, ctx)
    row = cars[cars["event_id"] == event_id("TST", "2024-05-01")].iloc[0]
    assert np.isnan(row["car_drift"])


def test_sue_uses_only_past_surprises():
    events = make_events()
    events["surprise"] = [1.0, -1.0, 1.0, -1.0, 2.0]
    enriched = add_sue(events, min_history=4)
    expected_std = np.std([1.0, -1.0, 1.0, -1.0], ddof=1)
    last = enriched.sort_values("announce_date").iloc[-1]
    assert last["sue"] == pytest.approx(2.0 / expected_std)
    assert enriched.sort_values("announce_date")["sue"].iloc[:4].isna().all()


def test_beat_flags_and_streak():
    events = make_events()
    events["surprise"] = [0.1, 0.1, -0.1, 0.1, 0.1]
    flagged = add_prior_streak(add_beat_flags(events))
    ordered = flagged.sort_values("announce_date")
    assert list(ordered["eps_beat"]) == [1, 1, 0, 1, 1]
    assert list(ordered["prior_streak"]) == [0, 1, 2, 0, 1]
    assert ordered["both_beat"].iloc[0] == 1.0


def test_build_panel_end_to_end():
    prices = make_prices()
    events = make_events()
    fundamentals = pd.DataFrame(
        {
            "ticker": "TST",
            "date": pd.to_datetime(["2024-03-31"]),
            "pe_ratio": [30.0],
            "pb_ratio": [5.0],
            "market_cap": [1.0e9],
            "ev_to_sales": [6.0],
        }
    )
    panel = build_panel(events, prices, fundamentals)
    assert not panel.empty
    target = panel[panel["event_id"] == event_id("TST", "2024-05-01")].iloc[0]
    assert target["car_reaction"] == pytest.approx(0.03, abs=1e-6)
    assert target["runup_20d"] == pytest.approx(0.0, abs=1e-9)
    assert target["momentum_12_1"] == pytest.approx(0.0, abs=1e-9)
    assert target["vix_level"] == pytest.approx(15.0)
    assert target["rate_level"] == pytest.approx(4.2)
    assert target["pe_ratio"] == pytest.approx(30.0)
    assert (pd.to_datetime(panel["day0"]) >= pd.to_datetime(panel["announce_date"])).all()


def test_leakage_check_raises_on_bad_day0():
    bad = pd.DataFrame(
        {
            "day0": [pd.Timestamp("2024-05-01")],
            "announce_date": [pd.Timestamp("2024-05-02")],
            "announce_time": ["bmo"],
        }
    )
    with pytest.raises(AssertionError):
        leakage_checks(bad)
