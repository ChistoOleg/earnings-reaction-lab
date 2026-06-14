from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erl.predict.cv import PurgedWalkForwardCV

RNG = np.random.default_rng(23)


def make_dates(n: int) -> pd.Series:
    return pd.Series(
        pd.date_range("2015-01-05", "2024-12-20", periods=n).normalize()
    )


def test_splitter_respects_purge_and_order():
    dates = make_dates(1000)
    cv = PurgedWalkForwardCV(n_splits=4, purge_days=35, embargo_days=5)
    folds = list(cv.split(dates))
    assert len(folds) == 4
    previous_test_start = pd.Timestamp.min
    for train_idx, test_idx in folds:
        train_dates = dates.iloc[train_idx]
        test_dates = dates.iloc[test_idx]
        test_start = test_dates.min()
        assert test_start > previous_test_start
        previous_test_start = test_start
        assert train_dates.max() < test_start - pd.Timedelta(days=35)
        assert len(set(train_idx) & set(test_idx)) == 0


def test_splitter_purges_boundary_events():
    dates = pd.Series(
        pd.to_datetime(
            ["2015-01-01"] * 50
            + ["2019-12-25"] * 5
            + ["2020-01-10"] * 50
            + ["2021-01-10"] * 50
        )
    )
    cv = PurgedWalkForwardCV(n_splits=2, purge_days=35, embargo_days=0)
    folds = list(cv.split(dates))
    for train_idx, test_idx in folds:
        test_start = dates.iloc[test_idx].min()
        cutoff = test_start - pd.Timedelta(days=35)
        assert (dates.iloc[train_idx] < cutoff).all()


def test_splitter_rejects_tiny_samples():
    with pytest.raises(ValueError):
        list(PurgedWalkForwardCV(n_splits=5).split(make_dates(8)))


def make_predictable_panel(n: int = 2500) -> pd.DataFrame:
    X = RNG.normal(size=(n, 5))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 1.0 * X[:, 0] * X[:, 1] + RNG.normal(
        scale=0.5, size=n
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    frame["car_reaction"] = y
    frame["announce_date"] = make_dates(n)
    frame["ticker"] = [f"T{i % 80}" for i in range(n)]
    frame["event_id"] = [f"e{i}" for i in range(n)]
    return frame


def test_gbm_recovers_signal_out_of_time():
    from erl.predict.gbm import train_gbm

    panel = make_predictable_panel()
    result = train_gbm(
        panel,
        "car_reaction",
        [f"f{i}" for i in range(5)],
        cv=PurgedWalkForwardCV(n_splits=3, purge_days=10),
        n_trials=8,
    )
    assert result.oos_metrics["r2"] > 0.5
    assert result.oos_metrics["rank_ic"] > 0.6
    assert len(result.fold_metrics) == 3
    assert result.fold_metrics["role"].iloc[-1] == "oos_final"
    assert {"y_true", "y_pred", "event_id", "ticker"} <= set(
        result.oos_predictions.columns
    )
    assert len(result.oos_predictions) == result.fold_metrics["n_test"].iloc[-1]


def test_shap_ranks_true_drivers_first():
    shap = pytest.importorskip("shap")
    from erl.predict.gbm import shap_importance, train_gbm

    panel = make_predictable_panel()
    features = [f"f{i}" for i in range(5)]
    result = train_gbm(
        panel,
        "car_reaction",
        features,
        cv=PurgedWalkForwardCV(n_splits=2, purge_days=10),
        n_trials=5,
    )
    table = shap_importance(result.model, panel[features])
    top_two = set(table["feature"].iloc[:2])
    assert top_two == {"f0", "f1"}
