from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from erl.events.returns import ReturnContext
from erl.inference.eventstudy import ar_path, car_by_quantile

logger = logging.getLogger(__name__)

BLUE, GREEN, RED, GREY = "#4C72B0", "#55A868", "#C44E52", "#888888"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(
    panel: pd.DataFrame,
    prices: pd.DataFrame | None,
    processed_dir,
    benchmark: str = "^GSPC",
) -> list[Path]:
    figdir = Path(processed_dir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. Reaction by surprise quintile (bar + 95% CI)
    try:
        q = car_by_quantile(panel, bins=5)
        if not q.empty:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.bar(q["bin"], q["mean_car"] * 100, yerr=q["se"] * 100 * 1.96,
                   capsize=4, color=BLUE)
            ax.axhline(0, color=GREY, lw=0.8)
            ax.set_xlabel("SUE quintile  (1 = biggest miss  →  5 = biggest beat)")
            ax.set_ylabel("Mean abnormal return, day 0–1 (%)")
            ax.set_title("Earnings reaction by surprise quintile")
            written.append(_save(fig, figdir / "01_car_by_sue_quintile.png"))
    except Exception as exc:
        logger.warning("quintile figure skipped: %s", exc)

    # 2. Surprise vs reaction scatter — the heterogeneity picture
    try:
        d = panel.dropna(subset=["sue", "car_reaction"])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(d["sue"], d["car_reaction"] * 100, s=10, alpha=0.35, color=GREEN)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel("Standardized surprise (SUE)")
        ax.set_ylabel("Abnormal return, day 0–1 (%)")
        ax.set_title("Reaction vs. surprise — wide scatter is the puzzle")
        written.append(_save(fig, figdir / "02_surprise_vs_reaction.png"))
    except Exception as exc:
        logger.warning("scatter figure skipped: %s", exc)

    # 3. Distribution of reactions
    try:
        d = panel.dropna(subset=["car_reaction"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(d["car_reaction"] * 100, bins=60, color=BLUE, alpha=0.85)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel("Abnormal return, day 0–1 (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of earnings reactions (n={len(d)})")
        written.append(_save(fig, figdir / "03_reaction_distribution.png"))
    except Exception as exc:
        logger.warning("histogram skipped: %s", exc)

    # 4. Average abnormal-return path, beats vs misses
    if prices is not None:
        try:
            ctx = ReturnContext(prices, benchmark)
            path = ar_path(panel, ctx, rel_days=(-5, 25))
            if not path.empty:
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                colors = {"beat": GREEN, "miss": RED}
                for grp, g in path.groupby("group"):
                    ax.plot(g["rel_day"], g["cum_ar"] * 100, marker="o", ms=3,
                            label=grp, color=colors.get(grp))
                ax.axvline(0, color=GREY, lw=0.8, ls="--")
                ax.axhline(0, color=GREY, lw=0.6)
                ax.set_xlabel("Trading days relative to announcement")
                ax.set_ylabel("Cumulative abnormal return (%)")
                ax.set_title("Drift around earnings: beats vs. misses")
                ax.legend()
                written.append(_save(fig, figdir / "04_drift_beats_vs_misses.png"))
        except Exception as exc:
            logger.warning("drift path skipped: %s", exc)

    # 5. Out-of-sample predicted vs actual (if predict stage saved them)
    pred_csv = Path(processed_dir) / "gbm_oos_predictions.csv"
    if pred_csv.exists():
        try:
            p = pd.read_csv(pred_csv)
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.scatter(p["y_pred"] * 100, p["y_true"] * 100, s=10, alpha=0.4, color=BLUE)
            lim = np.nanpercentile(np.abs(p[["y_true", "y_pred"]].to_numpy()) * 100, 99)
            ax.plot([-lim, lim], [-lim, lim], color=RED, lw=1)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("Predicted reaction (%)")
            ax.set_ylabel("Actual reaction (%)")
            ax.set_title("Out-of-sample: predicted vs. actual")
            written.append(_save(fig, figdir / "05_oos_predicted_vs_actual.png"))
        except Exception as exc:
            logger.warning("pred-vs-actual skipped: %s", exc)

    # 6. SHAP importance (if available)
    shap_csv = Path(processed_dir) / "gbm_shap.csv"
    if shap_csv.exists():
        try:
            t = pd.read_csv(shap_csv).head(12).iloc[::-1]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(t["feature"], t["mean_abs_shap"], color=BLUE)
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title("Feature importance (prediction model)")
            written.append(_save(fig, figdir / "06_shap_importance.png"))
        except Exception as exc:
            logger.warning("shap figure skipped: %s", exc)

    # 7. Causal-forest best linear projection (if available)
    blp_csv = Path(processed_dir) / "forest_blp.csv"
    if blp_csv.exists():
        try:
            t = pd.read_csv(blp_csv)
            t = t[t["term"] != "intercept"]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            colors = [GREEN if c >= 0 else RED for c in t["coef"]]
            ax.barh(t["term"], t["coef"], xerr=t["se"] * 1.96, color=colors, capsize=3)
            ax.axvline(0, color=GREY, lw=0.8)
            ax.set_xlabel("Effect on the reaction's sensitivity to surprise")
            ax.set_title("Which conditions moderate the reaction (causal forest)")
            written.append(_save(fig, figdir / "07_forest_moderators.png"))
        except Exception as exc:
            logger.warning("forest figure skipped: %s", exc)

    # 8. Out-of-sample comparison: ML vs linear baselines (if available)
    cmp_csv = Path(processed_dir) / "prediction_comparison.csv"
    if cmp_csv.exists():
        try:
            t = pd.read_csv(cmp_csv)
            fig, ax = plt.subplots(figsize=(6.5, 4))
            bars = ax.bar(t["model"], t["rank_ic"], color=[GREY, GREY, BLUE][: len(t)])
            ax.axhline(0, color=GREY, lw=0.8)
            ax.set_ylabel("Out-of-sample rank IC")
            ax.set_title("Does ML beat a linear baseline? (same fold, same metric)")
            for bar, val in zip(bars, t["rank_ic"]):
                ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
            written.append(_save(fig, figdir / "08_model_comparison.png"))
        except Exception as exc:
            logger.warning("model comparison figure skipped: %s", exc)

    logger.info("wrote %d figures to %s", len(written), figdir)
    return written
