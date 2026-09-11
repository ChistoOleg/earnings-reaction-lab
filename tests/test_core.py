from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest

from erl.config import Settings
from erl.fmp import FMPClient, FMPError
from erl.utils import event_id


def test_settings_defaults_and_benchmarks(tmp_path):
    settings = Settings(fmp_api_key="k", data_dir=str(tmp_path / "d"))
    assert settings.benchmark_symbols[0] == "^GSPC"
    assert "^VIX" in settings.benchmark_symbols
    assert "XLK" in settings.benchmark_symbols
    settings.ensure_dirs()
    assert settings.raw_dir.exists() and settings.processed_dir.exists()


def test_settings_rejects_bad_start_date():
    with pytest.raises(Exception):
        Settings(fmp_api_key="k", start_date="01-2015")


def test_event_id_stable_and_distinct():
    a = event_id("AAPL", "2024-05-02")
    assert a == event_id("aapl ", "2024-05-02")
    assert a != event_id("AAPL", "2024-08-01")
    assert len(a) == 16


def _make_client(tmp_path, handler, retries=2):
    transport = httpx.MockTransport(handler)
    return FMPClient(
        "test-key",
        tmp_path / "cache",
        interval_seconds=0.0,
        max_retries=retries,
        transport=transport,
    )


def test_client_requires_key(tmp_path):
    with pytest.raises(FMPError):
        FMPClient("", tmp_path)


def test_client_fetches_caches_and_appends_key(tmp_path):
    calls = {"n": 0, "apikey": None}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        calls["apikey"] = request.url.params.get("apikey")
        return httpx.Response(200, json=[{"ok": True}])

    client = _make_client(tmp_path, handler)
    first = client.get("/api/v3/thing", {"a": 1})
    second = client.get("/api/v3/thing", {"a": 1})
    third = client.get("/api/v3/thing", {"a": 2})
    assert first == second == [{"ok": True}]
    assert third == [{"ok": True}]
    assert calls["n"] == 2
    assert calls["apikey"] == "test-key"
    cached = list((tmp_path / "cache").glob("*.json"))
    assert len(cached) == 2
    record = json.loads(cached[0].read_text())
    assert "apikey" not in json.dumps(record["params"])


def test_client_refresh_bypasses_cache(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"v": calls["n"]})

    client = _make_client(tmp_path, handler)
    assert client.get("/p")["v"] == 1
    assert client.get("/p")["v"] == 1
    assert client.get("/p", refresh=True)["v"] == 2


def test_client_retries_then_succeeds(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"ok": 1})

    client = _make_client(tmp_path, handler, retries=3)
    assert client.get("/flaky") == {"ok": 1}
    assert calls["n"] == 3


def test_client_plan_gate_raises_clearly(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="upgrade required")

    client = _make_client(tmp_path, handler)
    with pytest.raises(FMPError) as excinfo:
        client.get("/api/v4/premium_thing")
    assert excinfo.value.status == 403
    assert "plan" in str(excinfo.value)


def test_client_gives_up_after_retries(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    client = _make_client(tmp_path, handler, retries=1)
    with pytest.raises(FMPError) as excinfo:
        client.get("/always429")
    assert excinfo.value.status == 429


def test_removal_only_record_is_not_read_as_an_addition():
    """A removal-only row that echoes the departing ticker in `symbol` must not
    open a new spell on the day the firm left the index."""
    from erl.universe import parse_change

    swap = {
        "symbol": "NEW",
        "addedSecurity": "New Co",
        "removedTicker": "OLD",
        "removedSecurity": "Old Co",
    }
    assert parse_change(swap) == ("NEW", "OLD")

    removal_only = {
        "symbol": "OLD",
        "addedSecurity": "",
        "removedTicker": "OLD",
        "removedSecurity": "Old Co",
    }
    assert parse_change(removal_only) == ("", "OLD")

    addition_only = {
        "symbol": "NEW",
        "addedSecurity": "New Co",
        "removedTicker": "",
        "removedSecurity": "",
    }
    assert parse_change(addition_only) == ("NEW", "")


def test_membership_closes_the_spell_of_a_delisted_name():
    from erl.universe import build_membership, members_on

    # GONE was a member at the start and was removed in 2009; SURV is current.
    events = [
        (pd.Timestamp("2009-06-01"), "", "GONE"),
        (pd.Timestamp("2012-01-01"), "LATER", ""),
    ]
    membership = build_membership({"SURV", "LATER"}, events, "2005-01-01")
    assert "GONE" in members_on(membership, "2008-01-01")
    assert "GONE" not in members_on(membership, "2010-01-01")
    assert "LATER" not in members_on(membership, "2010-01-01")
    assert "LATER" in members_on(membership, "2013-01-01")
    gone = membership[membership["ticker"] == "GONE"].iloc[0]
    assert gone["removed_date"] == pd.Timestamp("2009-06-01")


def test_coverage_flags_missing_delisted_names():
    from erl.universe import build_membership, membership_coverage

    events = [
        (pd.Timestamp("2009-06-01"), "", "GONE"),
        (pd.Timestamp("2010-06-01"), "", "ALSOGONE"),
    ]
    membership = build_membership({"SURV"}, events, "2005-01-01")
    # prices arrived for the survivor only
    coverage = membership_coverage(membership, {"SURV"})
    left = coverage[coverage["group"] == "left the index"].iloc[0]
    stayed = coverage[coverage["group"] == "still a member"].iloc[0]
    assert left["coverage"] == 0.0
    assert stayed["coverage"] == 1.0
    assert set(coverage.attrs["missing_tickers"]) == {"GONE", "ALSOGONE"}


def test_coverage_by_era_locates_the_survivorship_gap():
    """A single coverage figure hides which period the missing names sit in."""
    from erl.universe import coverage_by_era

    membership = pd.DataFrame(
        {
            "ticker": ["OLD1", "OLD2", "NEW1", "NEW2", "STAY"],
            "added_date": pd.NaT,
            "removed_date": pd.to_datetime(
                ["2008-03-01", "2008-09-01", "2023-01-01", "2023-06-01", None]
            ),
        }
    )
    # the two old delistings are unavailable, the recent ones are fine
    out = coverage_by_era(membership, {"NEW1", "NEW2", "STAY"}).set_index("removal_year")
    assert out.loc[2008, "coverage"] == 0.0
    assert out.loc[2023, "coverage"] == 1.0
    assert "STAY" not in out.index  # current members are not in this table
