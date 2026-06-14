from __future__ import annotations

import pandas as pd
import pytest

from erl.fmp import FMPError
from erl.harvest.fundamentals import harvest_fundamentals, parse_key_metrics
from erl.harvest.prices import harvest_prices, parse_prices
from erl.harvest.surprises import harvest_surprises, parse_calendar
from erl.harvest.transcripts import harvest_transcripts, parse_transcripts


class CannedClient:
    """Keys responses by (path, symbol) so per-symbol query params resolve."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, path, params=None, refresh=False):
        params = params or {}
        self.calls.append((path, params))
        symbol = params.get("symbol")
        for key in ((path, symbol), path):
            if key in self.responses:
                result = self.responses[key]
                if isinstance(result, Exception):
                    raise result
                return result
        return None


# Stable /stable/earnings rows: epsActual / epsEstimated / revenueActual / revenueEstimated.
CAL_ROWS = [
    {"date": "2024-05-02", "epsActual": 1.50, "epsEstimated": 1.40, "time": "amc",
     "revenueActual": 100.0, "revenueEstimated": 95.0, "fiscalDateEnding": "2024-03-31"},
    {"date": "2024-02-01", "epsActual": 1.10, "epsEstimated": 1.20, "time": "bmo",
     "revenueActual": 90.0, "revenueEstimated": 92.0},
    {"date": "2024-08-01", "epsActual": None, "epsEstimated": 1.55, "time": "amc"},
    {"date": "2010-01-01", "epsActual": 0.5, "epsEstimated": 0.4, "time": "bmo"},
    {"date": "2024-05-02", "epsActual": 1.50, "epsEstimated": 1.40, "time": "amc"},
]


def test_parse_calendar_filters_and_computes():
    frame = parse_calendar("AAPL", CAL_ROWS, "2015-01-01")
    assert len(frame) == 2
    beat = frame[frame["announce_date"] == pd.Timestamp("2024-05-02")].iloc[0]
    assert beat["announce_time"] == "amc"
    assert beat["surprise"] == pytest.approx(0.10)
    assert beat["surprise_pct"] == pytest.approx(0.10 / 1.40)
    miss = frame[frame["announce_date"] == pd.Timestamp("2024-02-01")].iloc[0]
    assert miss["surprise"] < 0
    assert frame["event_id"].is_unique


def test_parse_calendar_tolerates_legacy_field_names():
    rows = [{"date": "2024-05-02", "eps": 2.0, "epsEstimated": 1.5, "time": "amc"}]
    frame = parse_calendar("AAPL", rows, "2015-01-01")
    assert frame.iloc[0]["eps_actual"] == pytest.approx(2.0)


def test_harvest_surprises_skips_failed_tickers():
    client = CannedClient(
        {
            ("/stable/earnings", "AAPL"): CAL_ROWS,
            ("/stable/earnings", "BAD"): FMPError("nope", status=404),
        }
    )
    frame = harvest_surprises(client, ["AAPL", "BAD"], "2015-01-01")
    assert set(frame["ticker"]) == {"AAPL"}


# Stable /stable/historical-price-eod/full returns a bare list of daily rows.
PRICE_ROWS_AAPL = [
    {"date": "2024-05-03", "close": 183.4, "adjClose": 183.4, "volume": 1000},
    {"date": "2024-05-02", "close": 173.0, "adjClose": 172.8, "volume": 900},
    {"date": "2012-01-01", "close": 14.0, "adjClose": 12.0, "volume": 100},
]
PRICE_ROWS_GSPC = [
    {"date": "2024-05-02", "close": 5064.2, "volume": 0},
]


def test_parse_prices_long_format_and_filtering():
    frame = parse_prices("AAPL", PRICE_ROWS_AAPL, "2015-01-01")
    assert list(frame["date"]) == [pd.Timestamp("2024-05-02"), pd.Timestamp("2024-05-03")]
    assert frame.iloc[0]["adj_close"] == pytest.approx(172.8)
    assert set(frame.columns) == {"ticker", "date", "adj_close", "close", "volume"}


def test_parse_prices_index_without_adjclose_uses_close():
    frame = parse_prices("^GSPC", PRICE_ROWS_GSPC, "2015-01-01")
    assert frame.iloc[0]["adj_close"] == pytest.approx(5064.2)


def test_harvest_prices_combines_symbols(tmp_path):
    client = CannedClient(
        {
            ("/stable/historical-price-eod/full", "AAPL"): PRICE_ROWS_AAPL,
            ("/stable/historical-price-eod/full", "^GSPC"): PRICE_ROWS_GSPC,
        }
    )
    out = tmp_path / "prices.parquet"
    frame = harvest_prices(client, ["AAPL", "^GSPC"], "2015-01-01", out_path=out)
    assert set(frame["ticker"]) == {"AAPL", "^GSPC"}
    assert out.exists()


def test_parse_key_metrics():
    rows = [
        {"date": "2024-03-31", "peRatio": 28.5, "pbRatio": 40.1,
         "marketCap": 2.8e12, "evToSales": 7.5},
        {"date": "2009-12-31", "peRatio": 15.0},
    ]
    frame = parse_key_metrics("AAPL", rows, "2015-01-01")
    assert len(frame) == 1
    assert frame.iloc[0]["pe_ratio"] == pytest.approx(28.5)


def test_harvest_fundamentals_resilient():
    client = CannedClient(
        {
            ("/stable/key-metrics", "AAPL"): [
                {"date": "2024-03-31", "peRatio": 28.5, "marketCap": 2.8e12}
            ],
            ("/stable/key-metrics", "BAD"): FMPError("nope", status=404),
        }
    )
    frame = harvest_fundamentals(client, ["AAPL", "BAD"], "2015-01-01")
    assert set(frame["ticker"]) == {"AAPL"}


def test_parse_transcripts_skips_empty_content():
    rows = [
        {"quarter": 1, "year": 2024, "date": "2024-05-02 21:00:00", "content": "Good evening."},
        {"quarter": 4, "year": 2023, "date": "2024-02-01", "content": ""},
    ]
    frame = parse_transcripts("AAPL", rows)
    assert len(frame) == 1
    assert frame.iloc[0]["call_date"] == pd.Timestamp("2024-05-02")


def test_harvest_transcripts_plan_gate_message():
    client = CannedClient(
        {("/stable/earning-call-transcript", "AAPL"): FMPError("denied", status=403)}
    )
    with pytest.raises(FMPError) as excinfo:
        harvest_transcripts(client, ["AAPL"], [2024])
    assert "Ultimate" in str(excinfo.value)
