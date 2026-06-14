from __future__ import annotations

import pandas as pd

from erl.universe import build_membership, members_on, union_members


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def test_membership_reconstruction():
    current = {"AAA", "BBB", "DDD"}
    events = [
        (ts("2018-06-01"), "CCC", "ZZZ"),
        (ts("2020-03-15"), "DDD", "CCC"),
        (ts("2014-01-01"), "AAA", "OLD"),
    ]
    table = build_membership(current, events, "2015-01-01")

    at_start = members_on(table, "2015-01-02")
    assert at_start == {"AAA", "BBB", "ZZZ"}

    mid = members_on(table, "2019-01-01")
    assert mid == {"AAA", "BBB", "CCC"}

    today = members_on(table, "2024-01-01")
    assert today == {"AAA", "BBB", "DDD"}

    assert set(union_members(table)) == {"AAA", "BBB", "CCC", "DDD", "ZZZ"}

    ccc = table[table["ticker"] == "CCC"].iloc[0]
    assert ccc["added_date"] == ts("2018-06-01")
    assert ccc["removed_date"] == ts("2020-03-15")

    zzz = table[table["ticker"] == "ZZZ"].iloc[0]
    assert pd.isna(zzz["added_date"])
    assert zzz["removed_date"] == ts("2018-06-01")


def test_membership_survivorship_includes_departed_names():
    current = {"AAA"}
    events = [(ts("2019-05-01"), "AAA", "GONE")]
    table = build_membership(current, events, "2015-01-01")
    assert "GONE" in union_members(table)
    assert "GONE" in members_on(table, "2016-01-01")
    assert "GONE" not in members_on(table, "2019-05-01")
