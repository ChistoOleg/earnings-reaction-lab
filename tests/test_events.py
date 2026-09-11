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


def test_align_day0_unknown_treated_as_same_day():
    # Unknown timing: day0 is the announcement date and the (0, +1) window covers
    # both a before-open reaction (day 0) and an after-close one (day +1).
    assert align_day0("2024-05-01", "unknown", CAL) == pd.Timestamp("2024-05-01")


def test_unknown_timing_window_covers_before_open_reaction():
    # A before-open reaction at the announcement date must land inside the
    # (0, +1) window even when the timing label is missing. Under the old rule
    # (unknown -> next day) the window started one day late and missed it.
    prices = make_prices()
    ctx = ReturnContext(prices, "^GSPC")
    events = make_events().iloc[[-1]].copy()
    events["announce_date"] = pd.Timestamp("2024-05-02")  # the day the price jumps
    events["announce_time"] = "unknown"
    cars = compute_cars(events, ctx)
    assert cars["day0"].iloc[0] == pd.Timestamp("2024-05-02")
    assert cars["car_reaction"].iloc[0] == pytest.approx(0.03, abs=1e-6)


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
    enriched = add_sue(events, min_history=4, winsor=None)
    expected_std = np.std([1.0, -1.0, 1.0, -1.0], ddof=1)
    last = enriched.sort_values("announce_date").iloc[-1]
    assert last["sue"] == pytest.approx(2.0 / expected_std)
    assert last["sue_raw"] == last["sue"]
    assert enriched.sort_values("announce_date")["sue"].iloc[:4].isna().all()


def test_sue_is_winsorised_but_raw_is_kept():
    rows = []
    rng = np.random.default_rng(3)
    for i in range(600):
        rows.append(
            {
                "event_id": f"e{i}",
                "ticker": f"T{i % 20}",
                "announce_date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=91 * (i // 20)),
                "surprise": float(rng.normal()),
            }
        )
    events = pd.DataFrame(rows)
    # one firm with a degenerate denominator: tiny past surprises, then a big one
    tiny = events["ticker"] == "T0"
    events.loc[tiny, "surprise"] = 1e-3 * rng.normal(size=int(tiny.sum()))
    last_t0 = events[tiny].index[-1]
    events.loc[last_t0, "surprise"] = 5.0
    pooled = add_sue(events, min_history=4, winsor_mode="pooled")
    raw = pooled["sue_raw"].dropna()
    win = pooled["sue"].dropna()
    assert raw.max() > 100
    assert win.max() == pytest.approx(raw.quantile(0.99))
    assert win.min() == pytest.approx(raw.quantile(0.01))
    assert (win.abs() <= raw.abs() + 1e-12).all()

    # the default point-in-time mode also clips, but never using future data
    asof = add_sue(events, min_history=4)
    assert asof["sue_raw"].equals(pooled["sue_raw"])
    assert asof["sue"].dropna().abs().max() < raw.abs().max()


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
            # published well before the 2024-05-01 event (90-day lag applies)
            "date": pd.to_datetime(["2023-12-31"]),
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
    assert "sue_raw" in panel.columns
    assert panel.attrs.get("alignment_diagnostic")
    # the constructed jump is entirely at day0 for the one event with returns
    assert panel.attrs["alignment_peak_rel_day"] == 0


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


def test_align_day0_before_price_history_returns_none():
    # An event that predates the price history must not be silently mapped onto
    # the first available trading day.
    assert align_day0("2015-03-04", "bmo", CAL) is None
    assert align_day0("2015-03-04", "amc", CAL) is None


def test_split_artifacts_are_flagged():
    from erl.events.returns import suspected_split_artifacts, daily_returns

    dates = pd.bdate_range("2024-01-02", "2024-03-29")
    px = np.full(len(dates), 400.0)
    px[int(dates.searchsorted("2024-02-15")) :] /= 4  # unadjusted 4:1 split
    prices = pd.concat(
        [
            pd.DataFrame({"ticker": "^GSPC", "date": dates, "adj_close": 100.0}),
            pd.DataFrame({"ticker": "SPLT", "date": dates, "adj_close": px}),
        ],
        ignore_index=True,
    )
    flagged = suspected_split_artifacts(daily_returns(prices))
    assert len(flagged) == 1
    assert flagged["ticker"].iloc[0] == "SPLT"
    assert flagged["split_ratio"].iloc[0] == pytest.approx(4.0)


def test_asof_winsorisation_uses_no_future_data():
    from erl.events.features import winsorize_asof

    rng = np.random.default_rng(7)
    n = 600
    values = pd.Series(rng.normal(size=n))
    dates = pd.Series(pd.date_range("2015-01-01", periods=n, freq="D"))
    # A huge outlier at the very end must not change how earlier events are clipped.
    with_outlier = values.copy()
    with_outlier.iloc[-1] = 500.0
    a = winsorize_asof(values, dates, min_prior=100)
    b = winsorize_asof(with_outlier, dates, min_prior=100)
    assert a.iloc[:-1].equals(b.iloc[:-1])
    # pooled winsorisation does not have that property
    from erl.events.features import winsorize

    assert not winsorize(values).iloc[:-1].equals(winsorize(with_outlier).iloc[:-1])


def test_price_features_use_one_return_convention():
    prices = make_prices()
    ctx = ReturnContext(prices, "^GSPC")
    events = make_events()
    panel = build_panel(events, prices)
    # both horizons are market-adjusted; the raw variant is kept separately
    assert "momentum_12_1" in panel.columns
    assert "momentum_12_1_raw" in panel.columns


def test_fundamentals_are_not_used_before_publication():
    prices = make_prices()
    events = make_events()
    # period ends 2024-03-31, so a 90-day publication lag makes it unavailable
    # to an event on 2024-05-01 and only the older row may be attached.
    fundamentals = pd.DataFrame(
        {
            "ticker": ["TST", "TST"],
            "date": pd.to_datetime(["2022-12-31", "2024-03-31"]),
            "pe_ratio": [20.0, 30.0],
            "pb_ratio": [4.0, 5.0],
            "market_cap": [5.0e8, 1.0e9],
            "ev_to_sales": [5.0, 6.0],
        }
    )
    panel = build_panel(events, prices, fundamentals)
    row = panel[panel["event_id"] == event_id("TST", "2024-05-01")].iloc[0]
    assert row["pe_ratio"] == pytest.approx(20.0)


def test_gated_index_symbol_falls_back_to_a_proxy():
    """^TNX returns 402 on restricted plans. The rate feature must fall back to
    an available proxy rather than being silently all-NaN."""
    from erl.events.features import _first_available

    prices = make_prices()
    # IEF present, ^TNX and ^TYX absent (as on a plan without index symbols)
    dates = prices.loc[prices["ticker"] == "^GSPC", "date"]
    proxy = pd.DataFrame({"ticker": "IEF", "date": dates, "adj_close": 95.0})
    ctx = ReturnContext(
        pd.concat([prices[prices["ticker"] != "^TNX"], proxy], ignore_index=True), "^GSPC"
    )
    assert _first_available(ctx, ("^TNX", "^TYX", "IEF", "TLT"), "interest rate") == "IEF"
    # ^VIX is in the fixture, so no fallback is taken
    assert _first_available(ctx, ("^VIX", "VIXY"), "volatility") == "^VIX"
    # nothing available: returns the first candidate, feature stays NaN
    assert _first_available(ctx, ("NOPE", "ALSONOPE"), "rates") == "NOPE"


def test_rate_feature_is_populated_from_the_proxy():
    prices = make_prices()
    dates = prices.loc[prices["ticker"] == "^GSPC", "date"]
    proxy = pd.DataFrame({"ticker": "IEF", "date": dates, "adj_close": 95.0})
    without_tnx = pd.concat(
        [prices[prices["ticker"] != "^TNX"], proxy], ignore_index=True
    )
    # Fundamentals must be supplied: merge_fundamentals is a merge, and pandas
    # drops DataFrame.attrs across merges, so a no-fundamentals call would pass
    # while the real pipeline path silently loses the recorded symbol.
    fundamentals = pd.DataFrame(
        {
            "ticker": ["TST"],
            "date": pd.to_datetime(["2023-12-31"]),
            "pe_ratio": [20.0],
            "pb_ratio": [4.0],
            "market_cap": [1.0e9],
            "ev_to_sales": [5.0],
        }
    )
    panel = build_panel(make_events(), without_tnx, fundamentals)
    row = panel[panel["event_id"] == event_id("TST", "2024-05-01")].iloc[0]
    assert row["rate_level"] == pytest.approx(95.0)
    assert panel.attrs["rate_symbol_used"] == "IEF"
    assert "vix_symbol_used" in panel.attrs


def test_idiosyncratic_vol_excludes_the_event_window():
    """The vol estimation window must end well before day 0, so a huge reaction
    cannot inflate the denominator that is meant to scale it."""
    from erl.events.features import add_idiosyncratic_vol

    dates = pd.bdate_range("2023-06-01", "2024-06-28")
    rng = np.random.default_rng(5)
    ret = rng.normal(0, 0.01, len(dates))
    i = int(dates.searchsorted("2024-05-01"))
    ret[i] = 0.40  # enormous reaction on day 0
    prices = pd.concat(
        [
            pd.DataFrame({"ticker": "^GSPC", "date": dates, "adj_close": 100.0}),
            pd.DataFrame(
                {"ticker": "TST", "date": dates, "adj_close": 50 * np.exp(np.cumsum(ret))}
            ),
        ],
        ignore_index=True,
    )
    ctx = ReturnContext(prices, "^GSPC")
    panel = pd.DataFrame(
        [{"ticker": "TST", "day0": pd.Timestamp("2024-05-01"), "event_id": "e1"}]
    )
    out = add_idiosyncratic_vol(panel, ctx)
    vol = out["idio_vol"].iloc[0]
    # ~1% daily noise, nowhere near the 40% day-0 move
    assert 0.005 < vol < 0.02


def test_vol_adjusted_target_floors_the_denominator():
    from erl.events.features import add_vol_adjusted_target

    # A realistic sample: the 1st-percentile floor is only meaningful when there
    # are enough observations for that quantile to sit in the bulk of the data.
    rng = np.random.default_rng(2)
    vol = np.abs(rng.normal(0.02, 0.004, 500))
    vol[:3] = 1e-6  # a few pathologically quiet stocks
    panel = pd.DataFrame({"car_reaction": np.full(500, 0.05), "idio_vol": vol})
    out = add_vol_adjusted_target(panel, "car_reaction")
    # unfloored, those three would be 50,000 standard deviations
    assert out["car_reaction_vol_adj"].max() < 20
    assert out["car_reaction_vol_adj"].iloc[10] == pytest.approx(0.05 / vol[10])


def test_vol_adjustment_removes_a_pure_volatility_shift():
    """If a 'regime change' is only higher volatility, the raw coefficient rises
    while the vol-adjusted one does not. This is the test that distinguishes a
    pricing change from a scaling artifact."""
    from erl.events.features import add_vol_adjusted_target

    rng = np.random.default_rng(9)
    n = 2000
    sue = rng.normal(size=n)
    vol = np.where(np.arange(n) < n // 2, 0.01, 0.03)  # vol triples in period 2
    car = 0.3 * sue * vol + rng.normal(scale=vol, size=n)
    panel = pd.DataFrame({"car_reaction": car, "sue": sue, "idio_vol": vol})
    panel = add_vol_adjusted_target(panel, "car_reaction")
    first, second = slice(0, n // 2), slice(n // 2, n)
    raw_ratio = (
        np.polyfit(panel["sue"][second], panel["car_reaction"][second], 1)[0]
        / np.polyfit(panel["sue"][first], panel["car_reaction"][first], 1)[0]
    )
    adj_ratio = (
        np.polyfit(panel["sue"][second], panel["car_reaction_vol_adj"][second], 1)[0]
        / np.polyfit(panel["sue"][first], panel["car_reaction_vol_adj"][first], 1)[0]
    )
    assert raw_ratio > 2.0          # raw coefficient tracks volatility
    assert 0.7 < adj_ratio < 1.4    # vol-adjusted coefficient does not
