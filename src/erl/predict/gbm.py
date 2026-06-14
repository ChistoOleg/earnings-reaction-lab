from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from erl.predict.cv import PurgedWalkForwardCV

logger = logging.getLogger(__name__)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    total = y_true - y_true.mean()
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum(total ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(residual)))
    ic = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 2 else np.nan
    return {"r2": r2, "mae": mae, "rank_ic": ic}


@dataclass
class GBMResult:
    best_params: dict
    fold_metrics: pd.DataFrame
    oos_metrics: dict[str, float]
    oos_predictions: pd.DataFrame
    features: list[str]
    model: object = field(repr=False, default=None)


def _make_model(params: dict, random_state: int):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )


def _suggest_params(trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def train_gbm(
    panel: pd.DataFrame,
    target: str,
    features: list[str],
    date_col: str = "announce_date",
    cv: PurgedWalkForwardCV | None = None,
    n_trials: int = 25,
    random_state: int = 7,
) -> GBMResult:
    import optuna

    cv = cv or PurgedWalkForwardCV()
    frame = (
        panel.dropna(subset=[target, date_col, *features])
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    X = frame[features]
    y = frame[target].to_numpy(dtype=float)
    folds = list(cv.split(frame[date_col]))
    if len(folds) < 2:
        raise ValueError("need at least 2 folds: interior for tuning, final for OOS")
    tuning_folds, final_fold = folds[:-1], folds[-1]

    def objective(trial) -> float:
        params = _suggest_params(trial)
        losses = []
        for train_idx, test_idx in tuning_folds:
            model = _make_model(params, random_state)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
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
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        metrics = regression_metrics(y[test_idx], pred)
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
    final_model.fit(X.iloc[train_idx], y[train_idx])
    oos_pred = final_model.predict(X.iloc[test_idx])
    oos_metrics = regression_metrics(y[test_idx], oos_pred)

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
    )


def shap_importance(model, X: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    import shap

    sample = X if len(X) <= max_rows else X.sample(max_rows, random_state=7)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    importance = np.mean(np.abs(values), axis=0)
    table = pd.DataFrame(
        {"feature": list(X.columns), "mean_abs_shap": importance}
    ).sort_values("mean_abs_shap", ascending=False)
    return table.reset_index(drop=True)
