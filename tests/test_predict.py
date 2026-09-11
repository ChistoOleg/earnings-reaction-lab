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


def test_gbm_does_not_collapse_on_decimal_scale_target():
    """Regression test for the degenerate-model bug: a weak signal on a
    decimal-return scale (sd ~ 0.03) must still produce a model that uses more
    than one feature and has non-trivial prediction dispersion."""
    from erl.predict.gbm import train_gbm

    n = 2500
    X = RNG.normal(size=(n, 6))
    y = 0.006 * X[:, 0] + 0.005 * X[:, 1] + 0.004 * X[:, 0] * X[:, 2] + RNG.normal(
        scale=0.03, size=n
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    frame["car_reaction"] = y
    frame["announce_date"] = make_dates(n)
    frame["ticker"] = [f"T{i % 80}" for i in range(n)]
    frame["event_id"] = [f"e{i}" for i in range(n)]
    result = train_gbm(
        frame,
        "car_reaction",
        [f"f{i}" for i in range(6)],
        cv=PurgedWalkForwardCV(n_splits=3, purge_days=10),
        n_trials=8,
    )
    usage = result.feature_usage
    assert usage is not None
    assert int((usage["n_splits"] > 0).sum()) >= 2
    assert result.oos_predictions["y_pred"].std() > 0.002
    assert result.oos_metrics["rank_ic"] > 0.1
    # predictions are returned on the original (decimal) scale
    assert result.oos_predictions["y_pred"].abs().max() < 0.5


def test_oos_r2_uses_the_training_mean_as_benchmark():
    from erl.predict.gbm import regression_metrics

    rng = np.random.default_rng(0)
    y_train = rng.normal(0.0, 0.03, 800)
    y_test = rng.normal(0.02, 0.03, 200)  # the test period has a different mean
    predict_train_mean = np.full(200, y_train.mean())
    metrics = regression_metrics(y_test, predict_train_mean, y_train=y_train)
    # predicting the training mean is exactly the benchmark, so r2 is ~0
    assert abs(metrics["r2"]) < 0.02
    assert metrics["r2_benchmark"] == "train_mean"
    # the test-mean version penalises the same model for the mean shift
    assert metrics["r2_within"] < -0.3


def _paired_frame(n=800, gap=0.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.Series(pd.date_range("2020-01-01", periods=n, freq="D"))
    y = rng.normal(scale=0.03, size=n)
    # model a sees a fraction of the signal; b sees that fraction minus `gap`
    signal = y + rng.normal(scale=0.03, size=n)
    noise = rng.normal(scale=0.03, size=n)
    return pd.DataFrame(
        {
            "announce_date": dates,
            "y_true": y,
            "y_pred_a": 0.3 * signal + (0.3 - gap) * 0.0 + 0.0 * noise,
            "y_pred_b": (0.3 - gap) * signal + gap * noise,
        }
    )


def test_paired_tests_do_not_separate_equally_good_models():
    from erl.predict.compare import compare_models

    frame = _paired_frame(gap=0.0)
    frame["y_pred_b"] = frame["y_pred_a"] + np.random.default_rng(1).normal(
        scale=1e-6, size=len(frame)
    )
    table = compare_models(frame, "a", ["b"], n_boot=300)
    assert table["verdict"].iloc[0] == "indistinguishable"
    assert abs(table["ic_diff"].iloc[0]) < 0.05


def test_paired_tests_detect_a_genuinely_better_model():
    from erl.predict.compare import compare_models

    rng = np.random.default_rng(3)
    n = 1200
    dates = pd.Series(pd.date_range("2018-01-01", periods=n, freq="D"))
    y = rng.normal(scale=0.03, size=n)
    good = y + rng.normal(scale=0.02, size=n)      # informative
    bad = rng.normal(scale=0.03, size=n)           # pure noise
    frame = pd.DataFrame(
        {"announce_date": dates, "y_true": y, "y_pred_a": 0.5 * good, "y_pred_b": 0.5 * bad}
    )
    table = compare_models(frame, "a", ["b"], n_boot=300)
    row = table.iloc[0]
    assert row["verdict"] == "differs"
    assert row["mse_diff"] < 0          # a has lower squared error
    assert row["ic_diff"] > 0           # a ranks better
    assert row["mse_pvalue"] < 0.05


def test_block_bootstrap_respects_quarterly_clustering():
    """Resampling whole quarters must give a wider standard error than
    resampling events independently would, when the gap is driven by a few
    quarters rather than spread evenly."""
    from erl.predict.compare import rank_ic_difference_test

    rng = np.random.default_rng(11)
    n = 1200
    dates = pd.Series(pd.date_range("2015-01-01", periods=n, freq="D"))
    y = rng.normal(size=n)
    pred_a = rng.normal(size=n)
    pred_b = rng.normal(size=n)
    # only 2015 carries any signal, so the gap is concentrated in a few quarters
    early = (dates.dt.year == 2015).to_numpy()
    pred_a[early] = y[early] + rng.normal(scale=0.5, size=early.sum())
    out = rank_ic_difference_test(y, pred_a, pred_b, dates, n_boot=400)
    assert out["ic_diff"] > 0
    assert out["se"] > 0
    assert out["ci_low"] < out["ic_diff"] < out["ci_high"]
    assert out["n_blocks"] >= 8
