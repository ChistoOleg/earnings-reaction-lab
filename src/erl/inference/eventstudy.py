from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from erl.events.returns import ReturnContext, align_day0

logger = logging.getLogger(__name__)


def car_by_quantile(
    panel: pd.DataFrame,
    target: str = "car_reaction",
    by: str = "sue",
    bins: int = 5,
    cluster: str = "announce_quarter",
) -> pd.DataFrame:
    frame = panel.dropna(subset=[target, by, cluster]).copy()
    if frame.empty:
        return pd.DataFrame()
    ranks = frame[by].rank(method="first")
    frame["bin"] = pd.qcut(ranks, bins, labels=False, duplicates="drop") + 1
    dummies = pd.get_dummies(frame["bin"], prefix="q").astype(float)
    model = sm.OLS(frame[target].to_numpy(), dummies.to_numpy())
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": frame[cluster].to_numpy()})
    rows = []
    for i, column in enumerate(dummies.columns):
        bin_id = int(column.split("_")[1])
        subset = frame[frame["bin"] == bin_id]
        rows.append(
            {
                "bin": bin_id,
                "n": len(subset),
                "mean_" + by: float(subset[by].mean()),
                "mean_car": float(fit.params[i]),
                "se": float(fit.bse[i]),
                "tstat": float(fit.tvalues[i]),
                "pvalue": float(fit.pvalues[i]),
            }
        )
    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)


def ar_path(
    events: pd.DataFrame,
    ctx: ReturnContext,
    rel_days: tuple[int, int] = (-5, 25),
    split_by_sign: bool = True,
) -> pd.DataFrame:
    start, end = rel_days
    rows: list[dict] = []
    for event in events.itertuples():
        day0 = align_day0(event.announce_date, event.announce_time, ctx.calendar)
        if day0 is None:
            continue
        sign = "beat" if getattr(event, "surprise", 0) > 0 else "miss"
        for offset in range(start, end + 1):
            value = ctx.car(event.ticker, day0, offset, offset)
            if np.isnan(value):
                continue
            rows.append({"rel_day": offset, "group": sign, "ar": value})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keys = ["rel_day", "group"] if split_by_sign else ["rel_day"]
    out = frame.groupby(keys)["ar"].agg(["mean", "count"]).reset_index()
    out = out.rename(columns={"mean": "mean_ar", "count": "n"})
    pivot_keys = keys
    out = out.sort_values(pivot_keys).reset_index(drop=True)
    cumulative = []
    for _, group in out.groupby("group") if split_by_sign else [(None, out)]:
        cumulative.append(group.assign(cum_ar=group["mean_ar"].cumsum()))
    return pd.concat(cumulative, ignore_index=True)
