from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erl.inference.double_lasso import (
    benjamini_hochberg,
    fit_double_lasso,
    fit_interacted_double_lasso,
    plugin_lasso_select,
)
from erl.inference.eventstudy import car_by_quantile

RNG = np.random.default_rng(11)


def simulated_panel(n: int = 1200, p_controls: int = 30) -> pd.DataFrame:
    X = RNG.normal(size=(n, p_controls))
    d = 0.8 * X[:, 0] + RNG.normal(size=n)
    m1 = RNG.normal(size=n)
    m2 = RNG.normal(size=n)
    y = 0.5 * d + 1.0 * d * m1 + 2.0 * X[:, 0] + 1.0 * X[:, 1] + RNG.normal(size=n)
    frame = pd.DataFrame(X, columns=[f"c{i}" for i in range(p_controls)])
    frame["sue"] = d
    frame["m1"] = m1
    frame["m2"] = m2
    frame["car_reaction"] = y
    frame["ticker"] = [f"T{i % 60}" for i in range(n)]
    frame["announce_quarter"] = [f"20{15 + (i % 9)}Q{1 + (i % 4)}" for i in range(n)]
    return frame


def test_plugin_lasso_selects_true_support():
    n, p = 800, 40
    X = RNG.normal(size=(n, p))
    y = 3.0 * X[:, 2] - 2.0 * X[:, 7] + RNG.normal(size=n)
    selected = plugin_lasso_select(X, y)
    assert 2 in selected and 7 in selected
    assert len(selected) <= 10


def test_double_lasso_removes_omitted_variable_bias():
    frame = simulated_panel()
    controls = [f"c{i}" for i in range(30)]
    result = fit_double_lasso(frame, "car_reaction", "sue", controls)
    assert result.coef == pytest.approx(0.5, abs=0.15)
    assert "c0" in result.selected_controls
    assert result.se > 0

    naive = float(
        np.polyfit(frame["sue"].to_numpy(), frame["car_reaction"].to_numpy(), 1)[0]
    )
    assert abs(naive - 0.5) > 0.5


def test_interacted_double_lasso_finds_true_moderator():
    frame = simulated_panel()
    controls = [f"c{i}" for i in range(30)]
    table = fit_interacted_double_lasso(
        frame, "car_reaction", "sue", ["m1", "m2"], controls
    )
    m1_row = table[table["term"] == "sue_x_m1"].iloc[0]
    m2_row = table[table["term"] == "sue_x_m2"].iloc[0]
    assert bool(m1_row["significant_bh"]) is True
    assert m1_row["coef"] > 0.5
    assert abs(m2_row["coef"]) < 0.3
    assert bool(m2_row["significant_bh"]) is False


def test_benjamini_hochberg_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.2, 0.8])
    adjusted = benjamini_hochberg(p)
    assert np.all(adjusted >= p - 1e-12)
    assert np.all(adjusted <= 1.0)
    assert adjusted[0] < 0.01


def test_event_study_recovers_monotone_relation():
    frame = simulated_panel(n=2000, p_controls=5)
    frame["car_reaction"] = 0.02 * frame["sue"] + RNG.normal(scale=0.01, size=len(frame))
    table = car_by_quantile(frame, bins=5)
    assert len(table) == 5
    assert table["mean_car"].iloc[-1] > table["mean_car"].iloc[0]
    assert table["tstat"].iloc[-1] > 2


def test_causal_forest_recovers_heterogeneity():
    econml = pytest.importorskip("econml")
    n = 1500
    X1 = RNG.normal(size=n)
    X2 = RNG.normal(size=n)
    t = RNG.normal(size=n)
    tau = 1.0 + 2.0 * (X1 > 0)
    y = tau * t + 0.5 * X2 + RNG.normal(scale=0.5, size=n)
    frame = pd.DataFrame(
        {
            "car_reaction": y,
            "sue": t,
            "m_x1": X1,
            "m_x2": X2,
            "ticker": [f"T{i % 50}" for i in range(n)],
        }
    )
    from erl.inference.causal_forest import fit_causal_forest

    result = fit_causal_forest(
        frame,
        "car_reaction",
        "sue",
        moderators=["m_x1", "m_x2"],
        controls=["m_x2"],
        n_estimators=400,
        cv=3,
    )
    assert result.ate == pytest.approx(2.0, abs=0.5)
    true_tau = tau
    correlation = float(np.corrcoef(result.cate, true_tau)[0, 1])
    assert correlation > 0.6

    blp_x1 = result.blp[result.blp["term"] == "m_x1"].iloc[0]
    blp_x2 = result.blp[result.blp["term"] == "m_x2"].iloc[0]
    assert blp_x1["coef"] > 3 * abs(blp_x2["coef"])
    assert abs(blp_x1["tstat"]) > 3
    assert abs(blp_x2["tstat"]) < 3
    assert np.isfinite(result.ate_se) and result.ate_se > 0

    # The naive projection (CATE on moderators) must not be trusted for
    # inference: its standard errors are mechanically smaller than the
    # residual-based BLP because it ignores estimation error in the CATEs.
    naive_x2 = result.blp_naive[result.blp_naive["term"] == "m_x2"].iloc[0]
    assert naive_x2["se"] < blp_x2["se"]

    calib = result.calibration
    assert calib.attrs["top_minus_bottom"] > 0.8
    assert calib.attrs["rank_correlation"] > 0.7


def test_blp_standard_errors_are_honest_under_no_heterogeneity():
    """With a constant effect, the residual-based BLP should reject at roughly
    the nominal rate. The naive CATE projection over-rejects badly."""
    from erl.inference.causal_forest import best_linear_projection, naive_cate_projection

    rng = np.random.default_rng(5)
    rejections_blp, rejections_naive = 0, 0
    reps = 30
    for _ in range(reps):
        n = 600
        Z = pd.DataFrame(rng.normal(size=(n, 2)), columns=["z1", "z2"])
        t_res = rng.normal(size=n)
        y_res = 1.0 * t_res + rng.normal(size=n)
        # a smooth "fitted CATE" that is pure noise around the true constant 1.0
        fake_cate = 1.0 + 0.05 * Z["z1"].to_numpy() + 0.02 * rng.normal(size=n)
        blp = best_linear_projection(y_res, t_res, Z)
        naive = naive_cate_projection(fake_cate, Z)
        rejections_blp += int(abs(blp.loc[blp["term"] == "z1", "tstat"].iloc[0]) > 1.96)
        rejections_naive += int(abs(naive.loc[naive["term"] == "z1", "tstat"].iloc[0]) > 1.96)
    assert rejections_blp <= 6  # ~5% nominal, generous bound for 30 reps
    assert rejections_naive >= 25


def _stability_panel(n: int = 1600, break_effect: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    dates = pd.date_range("2015-01-01", "2024-12-01", periods=n)
    sue = rng.normal(size=n)
    post = (dates >= pd.Timestamp("2022-01-01")).astype(float)
    car = (0.01 + break_effect * post) * sue + rng.normal(scale=0.03, size=n)
    return pd.DataFrame(
        {
            "car_reaction": car,
            "sue": sue,
            "announce_date": dates,
            "ticker": [f"T{i % 70}" for i in range(n)],
            "announce_quarter": pd.PeriodIndex(dates, freq="Q").astype(str),
        }
    )


def test_regime_stability_detects_a_real_break():
    from erl.inference.stability import regime_stability

    table = regime_stability(_stability_panel(break_effect=0.03))
    assert len(table) == 3
    assert table.attrs["wald_pvalue"] < 0.01
    late = table[table["regime"].str.startswith("2022")].iloc[0]
    early = table[table["is_base"]].iloc[0]
    assert late["effect"] > early["effect"]


def test_regime_stability_does_not_invent_a_break():
    from erl.inference.stability import regime_stability

    table = regime_stability(_stability_panel(break_effect=0.0))
    assert table.attrs["wald_pvalue"] > 0.05


def test_rolling_effect_tracks_the_slope():
    from erl.inference.stability import rolling_effect

    table = rolling_effect(_stability_panel(break_effect=0.04), window=400, step=100)
    assert len(table) >= 3
    assert table["effect"].iloc[-1] > table["effect"].iloc[0]


def test_breaks_outside_the_sample_are_dropped():
    from erl.inference.stability import applicable_breaks, regime_stability

    panel = _stability_panel()  # runs 2015-2024
    # the GFC breaks predate this sample and must not create empty regimes
    kept = applicable_breaks(panel["announce_date"])
    assert "2008-09-15" not in kept
    assert "2020-03-01" in kept and "2022-01-01" in kept
    table = regime_stability(panel)
    assert len(table) == len(kept) + 1
    assert (table["n"] > 0).all()


def test_breaks_are_kept_when_the_sample_reaches_back():
    from erl.inference.stability import applicable_breaks

    # ~500 names reporting quarterly over 2005-2024 is roughly 40k events, so
    # even the 10-month GFC crisis window holds enough to estimate a slope.
    dates = pd.Series(pd.date_range("2005-01-01", "2024-12-01", periods=40000))
    assert applicable_breaks(dates) == (
        "2008-09-15", "2009-07-01", "2020-03-01", "2022-01-01",
    )


def test_narrow_window_is_dropped_in_a_thin_sample():
    from erl.inference.stability import applicable_breaks

    # A 30-name pilot over the same span cannot support the crisis window; the
    # surrounding breaks survive so the partition stays estimable.
    dates = pd.Series(pd.date_range("2005-01-01", "2024-12-01", periods=2400))
    kept = applicable_breaks(dates)
    assert "2009-07-01" not in kept
    assert "2008-09-15" in kept and "2020-03-01" in kept


def test_short_sample_yields_no_test_rather_than_a_broken_one():
    from erl.inference.stability import regime_stability

    panel = _stability_panel(n=200)
    panel["announce_date"] = pd.date_range("2024-01-01", periods=200, freq="D")
    assert regime_stability(panel).empty
