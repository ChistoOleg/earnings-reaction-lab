"""Simple linear baselines for the prediction track.

A gradient-boosted model is only worth its complexity if it beats a simple
linear regression out of sample (a point made by Prof. Kazak). These baselines
are evaluated on exactly the same purged/embargoed walk-forward folds and the
same metrics as ``train_gbm``, so the comparison is like-for-like.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from erl.predict.cv import PurgedWalkForwardCV
from erl.predict.gbm import regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    oos_metrics: dict[str, dict[str, float]]
    fold_metrics: pd.DataFrame
    oos_predictions: pd.DataFrame


def _build_models(random_state: int):
    from sklearn.base import clone
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Standardise first: OLS is scale-invariant but Ridge is not, and scaling
    # keeps the penalty meaningful across features on different units.
    models = {
        "ols": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))
        ),
    }
    return {name: clone(model) for name, model in models.items()}, clone


def train_linear_baselines(
    panel: pd.DataFrame,
    target: str,
    features: list[str],
    date_col: str = "announce_date",
    cv: PurgedWalkForwardCV | None = None,
    random_state: int = 7,
) -> BaselineResult:
    from sklearn.base import clone

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

    prototypes, _ = _build_models(random_state)

    fold_rows: list[dict] = []
    oos_metrics: dict[str, dict[str, float]] = {}
    final_train, final_test = folds[-1]
    oos_frame = frame.iloc[final_test][[date_col]].copy()
    if "ticker" in frame.columns:
        oos_frame["ticker"] = frame.iloc[final_test]["ticker"].to_numpy()
    oos_frame["y_true"] = y[final_test]

    for name, proto in prototypes.items():
        for i, (train_idx, test_idx) in enumerate(folds, start=1):
            model = clone(proto)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            fold_rows.append(
                {
                    "model": name,
                    "fold": i,
                    "role": "oos_final" if i == len(folds) else "tuning",
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **regression_metrics(y[test_idx], pred),
                }
            )
        model = clone(proto)
        model.fit(X.iloc[final_train], y[final_train])
        final_pred = model.predict(X.iloc[final_test])
        oos_metrics[name] = regression_metrics(y[final_test], final_pred)
        oos_frame[f"y_pred_{name}"] = final_pred
        logger.info(
            "baseline %-5s OOS r2=%.4f rank_ic=%.4f",
            name,
            oos_metrics[name]["r2"],
            oos_metrics[name]["rank_ic"],
        )

    return BaselineResult(
        oos_metrics=oos_metrics,
        fold_metrics=pd.DataFrame(fold_rows),
        oos_predictions=oos_frame.reset_index(drop=True),
    )
