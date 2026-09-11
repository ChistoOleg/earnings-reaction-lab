from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from erl.predict.cv import PurgedWalkForwardCV

logger = logging.getLogger(__name__)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray | None = None
) -> dict[str, float]:
    """Out-of-sample metrics.

    ``r2`` compares the model against the only benchmark actually available at
    prediction time: the mean of the *training* data. Using the test fold's own
    mean instead (the textbook in-sample formula) hands the benchmark
    information the model never had, and on a period whose mean differs from the
    training period it penalises a correct model. Where the training mean is not
    supplied it falls back to the test mean and ``r2_benchmark`` records which
    was used. ``r2_within`` always reports the test-mean version so the two are
    comparable across runs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    ss_res = float(np.sum(residual ** 2))
    ss_within = float(np.sum((y_true - y_true.mean()) ** 2))
    if y_train is not None and len(np.asarray(y_train)) > 0:
        benchmark = float(np.mean(y_train))
        kind = "train_mean"
    else:
        benchmark = float(y_true.mean())
        kind = "test_mean"
    ss_tot = float(np.sum((y_true - benchmark) ** 2))
    mae = float(np.mean(np.abs(residual)))
    ic = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 2 else np.nan
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "r2_within": 1.0 - ss_res / ss_within if ss_within > 0 else np.nan,
        "r2_benchmark": kind,
        "mae": mae,
        "rank_ic": ic,
    }


# The target is a two-day abnormal return stored as a decimal (0.01 = 1%).
# LightGBM's leaf penalties (reg_alpha, reg_lambda) act on the gradient sums,
# which are on the same scale as the target: with a target whose standard
# deviation is ~0.03, an L1 penalty of 1 zeroes every leaf and the model
# collapses to a near-constant. Fitting on percentage points keeps the penalty
# search space meaningful. Predictions are converted back to decimals.
TARGET_SCALE = 100.0


@dataclass
class GBMResult:
    best_params: dict
    fold_metrics: pd.DataFrame
    oos_metrics: dict[str, float]
    oos_predictions: pd.DataFrame
    features: list[str]
    model: object = field(repr=False, default=None)
    feature_usage: pd.DataFrame | None = None


def _make_model(params: dict, random_state: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        # subsample (bagging_fraction) is silently ignored unless bagging runs
        # at least every subsample_freq iterations.
        subsample_freq=1,
        **params,
    )


def _suggest_params(trial) -> dict:
    # Search space sized for a panel of a few thousand events with ~10 features:
    # shallow trees, leaves that must hold a meaningful number of events, and
    # penalties on a percentage-point target scale.
    return {
        "num_leaves": trial.suggest_int("num_leaves", 4, 31),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }


def feature_usage(model, features: list[str]) -> pd.DataFrame:
    """How many splits and how much gain each feature received in the fitted
    booster. A feature with zero splits was never used; a model where only one
    feature has splits is a step function of that feature, not an ML model."""
    booster = model.booster_
    splits = booster.feature_importance(importance_type="split")
    gain = booster.feature_importance(importance_type="gain")
    table = pd.DataFrame({"feature": features, "n_splits": splits, "gain": gain})
    total_gain = float(table["gain"].sum())
    table["gain_share"] = table["gain"] / total_gain if total_gain > 0 else np.nan
    return table.sort_values("gain", ascending=False).reset_index(drop=True)


def train_gbm(
    panel: pd.DataFrame,
    target: str,
    features: list[str],
    date_col: str = "announce_date",
    cv: PurgedWalkForwardCV | None = None,
    n_trials: int = 25,
    random_state: int = 7,
    target_scale: float = TARGET_SCALE,
) -> GBMResult:
    import optuna

    cv = cv or PurgedWalkForwardCV()
    frame = (
        panel.dropna(subset=[target, date_col, *features])
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    dropped = len(panel) - len(frame)
    if dropped:
        logger.info(
            "gbm sample: %d of %d panel rows kept; %d dropped for missing features "
            "(effective start %s vs panel start %s)",
            len(frame), len(panel), dropped,
            pd.to_datetime(frame[date_col]).min().date(),
            pd.to_datetime(panel[date_col]).min().date(),
        )
        if dropped / max(len(panel), 1) > 0.10:
            logger.warning(
                "gbm dropped %.0f%% of the panel to missing features; the linear "
                "baselines use the same rows, so the comparison stays like-for-like, "
                "but the estimation sample is not the full panel",
                100 * dropped / len(panel),
            )
    X = frame[features]
    y = frame[target].to_numpy(dtype=float)
    y_fit = y * target_scale
    folds = list(cv.split(frame[date_col]))
    if len(folds) < 2:
        raise ValueError("need at least 2 folds: interior for tuning, final for OOS")
    tuning_folds, final_fold = folds[:-1], folds[-1]

    def objective(trial) -> float:
        params = _suggest_params(trial)
        losses = []
        for train_idx, test_idx in tuning_folds:
            model = _make_model(params, random_state)
            model.fit(X.iloc[train_idx], y_fit[train_idx])
            pred = model.predict(X.iloc[test_idx]) / target_scale
            losses.append(float(np.mean((y[test_idx] - pred) ** 2)))
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    fold_rows = []
    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        model = _make_model(best_params, random_state)
        model.fit(X.iloc[train_idx], y_fit[train_idx])
        pred = model.predict(X.iloc[test_idx]) / target_scale
        metrics = regression_metrics(y[test_idx], pred, y_train=y[train_idx])
        fold_rows.append(
            {
                "fold": i,
                "role": "oos_final" if i == len(folds) else "tuning",
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                **metrics,
            }
        )

    train_idx, test_idx = final_fold
    final_model = _make_model(best_params, random_state)
    final_model.fit(X.iloc[train_idx], y_fit[train_idx])
    oos_pred = final_model.predict(X.iloc[test_idx]) / target_scale
    oos_metrics = regression_metrics(y[test_idx], oos_pred, y_train=y[train_idx])

    usage = feature_usage(final_model, features)
    used = int((usage["n_splits"] > 0).sum())
    logger.info(
        "gbm final model: %d trees, %d of %d features used, prediction sd=%.4f "
        "(target sd=%.4f)",
        final_model.booster_.num_trees(),
        used,
        len(features),
        float(np.std(oos_pred)),
        float(np.std(y[test_idx])),
    )
    if used <= 1:
        logger.warning(
            "gbm degenerate: only %d feature(s) received any split; the model is a "
            "step function of %s, not evidence about ML vs linear",
            used,
            usage["feature"].iloc[0],
        )

    predictions = frame.iloc[test_idx][[date_col]].copy()
    if "event_id" in frame.columns:
        predictions["event_id"] = frame.iloc[test_idx]["event_id"].to_numpy()
    if "ticker" in frame.columns:
        predictions["ticker"] = frame.iloc[test_idx]["ticker"].to_numpy()
    predictions["y_true"] = y[test_idx]
    predictions["y_pred"] = oos_pred

    logger.info(
        "gbm trained: %d folds, final OOS r2=%.4f rank_ic=%.4f",
        len(folds),
        oos_metrics["r2"],
        oos_metrics["rank_ic"],
    )
    return GBMResult(
        best_params=best_params,
        fold_metrics=pd.DataFrame(fold_rows),
        oos_metrics=oos_metrics,
        oos_predictions=predictions.reset_index(drop=True),
        features=features,
        model=final_model,
        feature_usage=usage,
    )


def shap_importance(
    model, X: pd.DataFrame, max_rows: int = 2000, target_scale: float = TARGET_SCALE
) -> pd.DataFrame:
    """Mean |SHAP| per feature, reported in the target's *original* units (the
    model is fitted on target * target_scale, so raw SHAP values are divided
    back)."""
    import shap

    sample = X if len(X) <= max_rows else X.sample(max_rows, random_state=7)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    importance = np.mean(np.abs(values), axis=0) / target_scale
    table = pd.DataFrame(
        {"feature": list(X.columns), "mean_abs_shap": importance}
    ).sort_values("mean_abs_shap", ascending=False)
    return table.reset_index(drop=True)
