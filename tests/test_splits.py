from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erl.harvest.splits import (
    adjust_for_splits,
    parse_splits,
    split_factors,
    verify_adjustment,
)


def test_parse_skips_pre_sample_and_trivial_splits():
    rows = [
        {"date": "2007-06-25", "numerator": 2, "denominator": 1},   # in sample
        {"date": "1998-01-05", "numerator": 2, "denominator": 1},   # before start
        {"date": "2010-03-01", "numerator": 1, "denominator": 1},   # no-op
        {"date": "2015-07-01", "numerator": 1, "denominator": 10},  # reverse split
    ]
    out = parse_splits("AGN", rows, "2005-01-01")
    assert list(out["date"].dt.date.astype(str)) == ["2007-06-25", "2015-07-01"]
    assert out["ratio"].tolist() == [2.0, 0.1]


def test_factors_halve_prices_before_a_two_for_one_split():
    splits = parse_splits("AGN", [{"date": "2007-06-25", "numerator": 2, "denominator": 1}],
                          "2005-01-01")
    dates = pd.DatetimeIndex(["2007-06-22", "2007-06-25", "2007-06-26"])
    assert split_factors(splits, dates, "AGN").tolist() == [0.5, 1.0, 1.0]


def test_factors_compound_across_multiple_splits():
    splits = parse_splits(
        "X",
        [
            {"date": "2010-01-04", "numerator": 2, "denominator": 1},
            {"date": "2015-01-05", "numerator": 4, "denominator": 1},
        ],
        "2005-01-01",
    )
    dates = pd.DatetimeIndex(["2009-01-02", "2012-01-03", "2016-01-04"])
    # before both: 1/8; between: 1/4; after both: 1
    assert split_factors(splits, dates, "X").tolist() == [0.125, 0.25, 1.0]


def test_reverse_split_scales_prices_up():
    splits = parse_splits("Y", [{"date": "2015-07-01", "numerator": 1, "denominator": 10}],
                          "2005-01-01")
    dates = pd.DatetimeIndex(["2015-06-30", "2015-07-01"])
    assert split_factors(splits, dates, "Y").tolist() == [10.0, 1.0]


def _agn_prices() -> pd.DataFrame:
    """The real AGN series across its 2007-06-25 split, as FMP returns it."""
    return pd.DataFrame(
        {
            "ticker": "AGN",
            "date": pd.to_datetime(
                ["2007-06-21", "2007-06-22", "2007-06-25", "2007-06-26"]
            ),
            "adj_close": [117.26, 114.47, 58.00, 58.67],
            "close": [117.26, 114.47, 58.00, 58.67],
            "volume": [862400.0, 1404600.0, 4482500.0, 4668300.0],
        }
    )


def test_adjustment_removes_the_fabricated_fifty_percent_return():
    from erl.events.returns import daily_returns

    prices = _agn_prices()
    raw = daily_returns(prices)
    assert raw["ret"].min() < -0.45  # the artifact is there before adjustment

    splits = parse_splits("AGN", [{"date": "2007-06-25", "numerator": 2, "denominator": 1}],
                          "2005-01-01")
    adjusted = adjust_for_splits(prices, splits)
    out = daily_returns(adjusted)
    # every return is now an ordinary daily move
    assert out["ret"].abs().max() < 0.05
    assert adjusted["split_adjusted"].all()
    # post-split prices are untouched; pre-split prices halved
    assert adjusted.loc[adjusted["date"] == "2007-06-26", "adj_close"].iloc[0] == pytest.approx(58.67)
    assert adjusted.loc[adjusted["date"] == "2007-06-22", "adj_close"].iloc[0] == pytest.approx(57.235)
    assert verify_adjustment(adjusted) == 0


def test_unaffected_tickers_are_left_alone():
    prices = pd.concat(
        [
            _agn_prices(),
            pd.DataFrame(
                {
                    "ticker": "CALM",
                    "date": pd.to_datetime(
                        ["2007-06-21", "2007-06-22", "2007-06-25", "2007-06-26"]
                    ),
                    "adj_close": [50.0, 50.5, 50.2, 50.9],
                    "close": [50.0, 50.5, 50.2, 50.9],
                    "volume": [1.0, 1.0, 1.0, 1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    splits = parse_splits("AGN", [{"date": "2007-06-25", "numerator": 2, "denominator": 1}],
                          "2005-01-01")
    adjusted = adjust_for_splits(prices, splits)
    untouched = adjusted.loc[adjusted["ticker"] == "CALM", "adj_close"].tolist()
    assert untouched == [50.0, 50.5, 50.2, 50.9]
    assert not adjusted.loc[adjusted["ticker"] == "CALM", "split_adjusted"].any()


def test_missing_split_feed_leaves_prices_alone_and_warns(caplog):
    prices = _agn_prices()
    with caplog.at_level("WARNING"):
        out = adjust_for_splits(prices, pd.DataFrame())
    assert out["adj_close"].tolist() == prices["adj_close"].tolist()
    assert not out["split_adjusted"].any()
    assert "left unadjusted" in caplog.text


def test_verify_reports_a_split_the_feed_missed():
    prices = _agn_prices()
    # adjust with an empty feed, so the artifact survives
    assert verify_adjustment(adjust_for_splits(prices, pd.DataFrame())) == 1


def test_etf_proxies_are_in_the_split_symbol_list():
    """VIXY reverse-splits repeatedly and is the volatility fallback, so it must
    be sent to the splits endpoint. Index symbols must not be."""
    from erl.config import Settings

    settings = Settings(fmp_api_key="x")
    tickers = ["AAPL", "MSFT"]
    symbols = [
        s for s in dict.fromkeys(tickers + settings.benchmark_symbols)
        if not s.startswith("^")
    ]
    assert "VIXY" in symbols and "IEF" in symbols and "XLF" in symbols
    assert not any(s.startswith("^") for s in symbols)
    assert len(symbols) == len(set(symbols))  # no duplicate requests


def test_reverse_split_on_a_proxy_is_corrected():
    from erl.events.returns import daily_returns

    dates = pd.to_datetime(["2021-04-30", "2021-05-03", "2021-05-04"])
    prices = pd.DataFrame(
        {"ticker": "VIXY", "date": dates, "adj_close": [10.0, 40.0, 39.5],
         "close": [10.0, 40.0, 39.5], "volume": [1.0, 1.0, 1.0]}
    )
    raw = daily_returns(prices)
    assert raw["ret"].max() > 2.0  # a fabricated +300% from a 1:4 reverse split
    splits = parse_splits("VIXY", [{"date": "2021-05-03", "numerator": 1, "denominator": 4}],
                          "2005-01-01")
    adjusted = adjust_for_splits(prices, splits)
    assert adjusted["adj_close"].tolist() == pytest.approx([40.0, 40.0, 39.5])
    assert daily_returns(adjusted)["ret"].abs().max() < 0.05


def test_already_adjusted_series_is_not_adjusted_again():
    """The feed back-adjusts some symbols. Applying the factor to a series that
    is already continuous across the split introduces an artifact rather than
    removing one, so it must be detected and skipped."""
    from erl.events.returns import daily_returns
    from erl.harvest.splits import split_is_present

    dates = pd.to_datetime(["2020-08-27", "2020-08-28", "2020-08-31", "2020-09-01"])
    # continuous: the vendor already divided the pre-split prices by 4
    prices = pd.DataFrame(
        {"ticker": "AAPL", "date": dates, "adj_close": [125.0, 124.8, 129.0, 134.2],
         "close": [125.0, 124.8, 129.0, 134.2], "volume": [1.0] * 4}
    )
    splits = parse_splits("AAPL", [{"date": "2020-08-31", "numerator": 4, "denominator": 1}],
                          "2005-01-01")
    series = pd.Series(prices["adj_close"].to_numpy(), index=pd.DatetimeIndex(prices["date"]))
    present, observed = split_is_present(series, pd.Timestamp("2020-08-31"), 4.0)
    assert not present
    assert observed == pytest.approx(129.0 / 124.8, abs=0.01)

    out = adjust_for_splits(prices, splits)
    assert out["adj_close"].tolist() == prices["adj_close"].tolist()
    assert not out["split_adjusted"].any()
    assert out.attrs["splits_already_adjusted"] == 1
    assert out.attrs["splits_applied"] == 0
    assert daily_returns(out)["ret"].abs().max() < 0.05


def test_visible_split_is_still_detected_and_applied():
    from erl.harvest.splits import split_is_present

    prices = _agn_prices()
    series = pd.Series(prices["adj_close"].to_numpy(), index=pd.DatetimeIndex(prices["date"]))
    present, observed = split_is_present(series, pd.Timestamp("2007-06-25"), 2.0)
    assert present
    assert observed == pytest.approx(58.00 / 114.47, abs=0.01)
    splits = parse_splits("AGN", [{"date": "2007-06-25", "numerator": 2, "denominator": 1}],
                          "2005-01-01")
    out = adjust_for_splits(prices, splits)
    assert out.attrs["splits_applied"] == 1
    assert out.attrs["splits_already_adjusted"] == 0


def test_mixed_feed_adjusts_only_the_unadjusted_ticker():
    """The realistic case: one delisted name unadjusted, one active name already
    adjusted, in the same price frame."""
    agn = _agn_prices()
    dates = pd.to_datetime(["2007-06-21", "2007-06-22", "2007-06-25", "2007-06-26"])
    active = pd.DataFrame(
        {"ticker": "ACTV", "date": dates, "adj_close": [30.0, 30.2, 30.1, 30.5],
         "close": [30.0, 30.2, 30.1, 30.5], "volume": [1.0] * 4}
    )
    prices = pd.concat([agn, active], ignore_index=True)
    splits = pd.concat(
        [
            parse_splits("AGN", [{"date": "2007-06-25", "numerator": 2, "denominator": 1}],
                         "2005-01-01"),
            parse_splits("ACTV", [{"date": "2007-06-25", "numerator": 2, "denominator": 1}],
                         "2005-01-01"),
        ],
        ignore_index=True,
    )
    out = adjust_for_splits(prices, splits)
    assert out.attrs["splits_applied"] == 1
    assert out.attrs["splits_already_adjusted"] == 1
    assert out.loc[out["ticker"] == "ACTV", "adj_close"].tolist() == [30.0, 30.2, 30.1, 30.5]
    assert out.loc[out["ticker"] == "AGN", "adj_close"].iloc[1] == pytest.approx(57.235)
    assert verify_adjustment(out) == 0


def test_genuine_large_move_on_a_split_date_is_left_alone():
    """An ambiguous series (a real crash coinciding with the split date) must not
    be 'corrected' on a guess."""
    from erl.harvest.splits import split_is_present

    dates = pd.to_datetime(["2010-01-04", "2010-01-05"])
    series = pd.Series([100.0, 12.0], index=pd.DatetimeIndex(dates))
    present, observed = split_is_present(series, pd.Timestamp("2010-01-05"), 2.0)
    assert not present                      # 0.12 is nowhere near 0.5
    assert observed == pytest.approx(0.12)  # and nowhere near continuous either


def _artifact_prices(ticker: str, dates: list[str], moves: list[float]) -> pd.DataFrame:
    """A price series with a given gross move on each listed date."""
    all_dates = pd.bdate_range("2005-01-03", "2026-09-11")
    px = np.full(len(all_dates), 100.0)
    factor = np.ones(len(all_dates))
    for date, gross in zip(dates, moves):
        i = int(all_dates.searchsorted(pd.Timestamp(date)))
        factor[i:] *= gross
    px = px * factor
    return pd.DataFrame(
        {"ticker": ticker, "date": all_dates, "adj_close": px, "close": px,
         "volume": np.ones(len(all_dates))}
    )


def test_unexplained_distinguishes_uncovered_tickers_from_covered_ones():
    """A naive ticker merge hides tickers the feed does not cover at all, which
    is the more common case for delisted names."""
    from erl.harvest.splits import unexplained_artifacts

    covered = _artifact_prices("COV", ["2010-06-01"], [0.5])
    uncovered = _artifact_prices("UNCOV", ["2012-06-01"], [0.5])
    prices = pd.concat([covered, uncovered], ignore_index=True)
    # the feed knows COV but reports a split on an unrelated date
    splits = parse_splits("COV", [{"date": "2018-01-03", "numerator": 2, "denominator": 1}],
                          "2005-01-01")
    out = unexplained_artifacts(prices, splits)
    cov = out.loc[out["ticker"] == "COV"].iloc[0]
    unc = out.loc[out["ticker"] == "UNCOV"].iloc[0]
    assert cov["feed_covers_ticker"] and not cov["near_feed_split"]
    assert not unc["feed_covers_ticker"] and not unc["near_feed_split"]


def test_recycled_ticker_is_flagged_but_a_single_crash_is_not():
    from erl.harvest.splits import suspicious_tickers, unexplained_artifacts

    # one genuine collapse: a real price move, must not be flagged
    crash = _artifact_prices("FRC", ["2023-04-25"], [0.5])
    # a recycled symbol: repeated implausible ratios within months
    recycled = _artifact_prices(
        "CPWR",
        ["2025-01-17", "2025-01-23", "2025-05-19", "2025-06-11", "2025-07-09"],
        [2.0, 0.5, 0.2, 1 / 3, 0.1],
    )
    prices = pd.concat([crash, recycled], ignore_index=True)
    artifacts = unexplained_artifacts(prices, pd.DataFrame())
    flagged = suspicious_tickers(artifacts, min_artifacts=3)
    assert flagged["ticker"].tolist() == ["CPWR"]
    assert flagged["artifacts"].iloc[0] >= 5
    assert "FRC" not in flagged["ticker"].tolist()


def test_events_touched_counts_the_volatility_window_not_just_the_reaction():
    """idio_vol reaches back 60 trading days, so one bad return contaminates far
    more than the single event whose reaction window contains it."""
    from erl.harvest.splits import events_touched

    artifacts = pd.DataFrame({"ticker": ["X"], "date": [pd.Timestamp("2015-06-01")]})
    panel = pd.DataFrame(
        {
            "ticker": ["X", "X", "X", "Y"],
            "day0": pd.to_datetime(
                ["2015-06-01", "2015-07-15", "2016-06-01", "2015-06-01"]
            ),
        }
    )
    # the event on the artifact date and the one ~30 trading days later are both
    # exposed; the one a year later and the other ticker are not
    assert events_touched(panel, artifacts) == 2
    assert events_touched(panel, pd.DataFrame()) == 0
