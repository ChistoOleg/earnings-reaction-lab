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

    calib = result.calibration
    assert calib.attrs["top_minus_bottom"] > 0.8
    assert calib.attrs["rank_correlation"] > 0.7
