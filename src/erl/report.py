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
            ax.set_xlabel("SUE quintile (1 = biggest miss, 5 = biggest beat)")
            ax.set_ylabel("Mean abnormal return, days 0 to +1 (%)")
            ax.set_title(f"Earnings reaction by surprise quintile (n={int(q['n'].sum())}, 95% CI, clustered by quarter)")
            written.append(_save(fig, figdir / "01_car_by_sue_quintile.png"))
    except Exception as exc:
        logger.warning("quintile figure skipped: %s", exc)

    # 2. Surprise vs reaction scatter: the heterogeneity picture
    try:
        d = panel.dropna(subset=["sue", "car_reaction"])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(d["sue"], d["car_reaction"] * 100, s=10, alpha=0.35, color=GREEN)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel("Standardized surprise (SUE, winsorised 1%/99%)")
        ax.set_ylabel("Abnormal return, days 0 to +1 (%)")
        ax.set_title(f"Reaction vs. surprise (n={len(d)}): the wide scatter is the puzzle")
        written.append(_save(fig, figdir / "02_surprise_vs_reaction.png"))
    except Exception as exc:
        logger.warning("scatter figure skipped: %s", exc)

    # 3. Distribution of reactions
    try:
        d = panel.dropna(subset=["car_reaction"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(d["car_reaction"] * 100, bins=60, color=BLUE, alpha=0.85)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel("Abnormal return, days 0 to +1 (%)")
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
                ax.set_xlabel("Trading days relative to day 0 (first return that can contain the announcement)")
                ax.set_ylabel("Cumulative abnormal return (%)")
                ax.set_title("Cumulative abnormal return around earnings: beats vs. misses")
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
            ax.set_title(f"Out-of-sample: predicted vs. actual (n={len(p)}, final walk-forward fold)")
            written.append(_save(fig, figdir / "05_oos_predicted_vs_actual.png"))
        except Exception as exc:
            logger.warning("pred-vs-actual skipped: %s", exc)

    # 6. SHAP importance (if available)
    shap_csv = Path(processed_dir) / "gbm_shap.csv"
    if shap_csv.exists():
        try:
            t = pd.read_csv(shap_csv).head(12).iloc[::-1]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.barh(t["feature"], t["mean_abs_shap"] * 100, color=BLUE)
            ax.set_xlabel("Mean |SHAP| (percentage points of two-day abnormal return)")
            ax.set_title("Feature importance in the prediction model (LightGBM)")
            written.append(_save(fig, figdir / "06_shap_importance.png"))
        except Exception as exc:
            logger.warning("shap figure skipped: %s", exc)

    # 7. Causal-forest best linear projection (if available)
    blp_csv = Path(processed_dir) / "forest_blp.csv"
    if blp_csv.exists():
        try:
            t = pd.read_csv(blp_csv)
            t = t[~t["term"].isin(["intercept", "treatment_mean"])]
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            colors = [GREEN if c >= 0 else RED for c in t["coef"]]
            ax.barh(t["term"], t["coef"] * 100, xerr=t["se"] * 100 * 1.96,
                    color=colors, capsize=3)
            ax.axvline(0, color=GREY, lw=0.8)
            ax.set_xlabel("Change in reaction per unit of SUE (pp) for a +1 SD move in the moderator")
            ax.set_title("Best linear projection of the surprise effect (95% CI, clustered by firm)")
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

    # 9. Parameter stability: is the pooled surprise effect one relationship?
    roll_csv = Path(processed_dir) / "rolling_effect.csv"
    if roll_csv.exists():
        try:
            t = pd.read_csv(roll_csv, parse_dates=["window_start", "window_end"])
            fig, ax = plt.subplots(figsize=(7.5, 4))
            mid = t["window_start"] + (t["window_end"] - t["window_start"]) / 2
            ax.plot(mid, t["effect"] * 100, color=BLUE, marker="o", ms=3)
            ax.fill_between(
                mid,
                (t["effect"] - 1.96 * t["se"]) * 100,
                (t["effect"] + 1.96 * t["se"]) * 100,
                color=BLUE, alpha=0.18,
            )
            ax.axhline(0, color=GREY, lw=0.8)
            ax.set_xlabel(f"Centre of a rolling {int(t['n'].iloc[0])}-event window")
            ax.set_ylabel("Reaction per unit of SUE (pp)")
            ax.set_title("Is the surprise effect stable? (95% CI, clustered by firm)")
            written.append(_save(fig, figdir / "09_effect_stability.png"))
        except Exception as exc:
            logger.warning("stability figure skipped: %s", exc)

    written += diagnostic_figures(processed_dir, figdir)

    logger.info("wrote %d figures to %s", len(written), figdir)
    return written


def _read(processed_dir, name: str) -> pd.DataFrame | None:
    path = Path(processed_dir) / name
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        logger.warning("could not read %s: %s", name, exc)
        return None
    return None if frame.empty else frame


def diagnostic_figures(processed_dir, figdir: Path) -> list[Path]:
    """Figures for the data-quality and robustness work: that day 0 is aligned,
    how far the honest standard errors sit from the naive ones, whether the
    regime break survives volatility standardisation, and where the survivorship
    gap falls."""
    written: list[Path] = []

    # 10. Event-day alignment
    align = _read(processed_dir, "alignment_diagnostic.csv")
    if align is not None and {"rel_day", "mean_abs_ar"} <= set(align.columns):
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = [GREEN if d in (0, 1) else GREY for d in align["rel_day"]]
            ax.bar(align["rel_day"], align["mean_abs_ar"] * 100, color=colors)
            pre = align.loc[align["rel_day"] < 0, "mean_abs_ar"].mean() * 100
            ax.axhline(pre, color=RED, ls="--", lw=1,
                       label=f"pre-event baseline ({pre:.2f} pp)")
            ax.set_xlabel("Trading days relative to day 0")
            ax.set_ylabel("Mean |abnormal return| (pp)")
            ax.set_title("Event-day alignment: the reaction sits in the (0, +1) window")
            ax.legend(frameon=False)
            written.append(_save(fig, figdir / "10_alignment_diagnostic.png"))
        except Exception as exc:
            logger.warning("alignment figure skipped: %s", exc)

    # 11. Honest vs naive confidence intervals
    blp, naive = _read(processed_dir, "forest_blp.csv"), _read(processed_dir, "forest_blp_naive.csv")
    if blp is not None and naive is not None and "term" in blp.columns:
        try:
            merged = blp.loc[~blp["term"].isin(["intercept", "treatment_mean"])].merge(
                naive, on="term", suffixes=("_honest", "_naive")
            )
            if not merged.empty:
                y = np.arange(len(merged))
                fig, ax = plt.subplots(figsize=(7.5, 0.8 * len(merged) + 2.2))
                ax.errorbar(merged["coef_naive"] * 100, y + 0.16,
                            xerr=merged["se_naive"] * 100 * 1.96, fmt="o", ms=4,
                            color=RED, capsize=3, label="naive (CATEs on moderators)")
                ax.errorbar(merged["coef_honest"] * 100, y - 0.16,
                            xerr=merged["se_honest"] * 100 * 1.96, fmt="o", ms=4,
                            color=BLUE, capsize=3, label="honest (residual BLP, clustered)")
                ax.axvline(0, color=GREY, lw=0.8)
                ax.set_yticks(y)
                ax.set_yticklabels(merged["term"])
                ax.set_xlabel("Change in reaction per unit of SUE (pp), per +1 SD of the moderator")
                ax.set_title("Why the standard errors matter (95% CI)")
                ax.legend(frameon=False, fontsize=9)
                written.append(_save(fig, figdir / "11_honest_vs_naive_ci.png"))
        except Exception as exc:
            logger.warning("honest-vs-naive figure skipped: %s", exc)

    # 12. Regime effects, raw and standardised
    raw = _read(processed_dir, "regime_stability.csv")
    adj = _read(processed_dir, "regime_stability_voladj.csv")
    if raw is not None and "effect" in raw.columns:
        try:
            panels = [("Raw reaction (pp per unit SUE)", raw, 100.0)]
            if adj is not None and "effect" in adj.columns:
                panels.append(("Standardised by pre-event volatility", adj, 1.0))
            fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.2))
            axes = np.atleast_1d(axes)
            for ax, (label, table, scale) in zip(axes, panels):
                order = table.sort_values("regime")
                y = np.arange(len(order))
                ax.barh(y, order["effect"] * scale,
                        xerr=order["se"] * scale * 1.96, color=BLUE, capsize=3)
                ax.axvline(0, color=GREY, lw=0.8)
                ax.set_yticks(y)
                ax.set_yticklabels(
                    [f"{r}\n(n={int(n)})" for r, n in zip(order["regime"], order["n"])],
                    fontsize=8,
                )
                ax.set_xlabel(label)
            fig.suptitle("Surprise effect by regime (95% CI, two-way clustered)")
            written.append(_save(fig, figdir / "12_regime_effects.png"))
        except Exception as exc:
            logger.warning("regime figure skipped: %s", exc)

    # 13. Rolling effect on a shared time axis
    roll = _read(processed_dir, "rolling_effect.csv")
    roll_adj = _read(processed_dir, "rolling_effect_voladj.csv")
    if roll is not None and roll_adj is not None:
        try:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            for ax, table, label, scale in (
                (axes[0], roll, "Raw (pp per unit SUE)", 100.0),
                (axes[1], roll_adj, "Per pre-event SD", 1.0),
            ):
                table = table.copy()
                for column in ("window_start", "window_end"):
                    table[column] = pd.to_datetime(table[column])
                mid = table["window_start"] + (table["window_end"] - table["window_start"]) / 2
                ax.plot(mid, table["effect"] * scale, color=BLUE, lw=1.4)
                ax.fill_between(mid, (table["effect"] - 1.96 * table["se"]) * scale,
                                (table["effect"] + 1.96 * table["se"]) * scale,
                                color=BLUE, alpha=0.18)
                ax.axhline(0, color=GREY, lw=0.8)
                ax.set_ylabel(label)
            axes[1].set_xlabel(
                "Window centre (overlapping windows: adjacent points are not independent)"
            )
            axes[0].set_title("Is the effect stable? Raw vs volatility-standardised")
            written.append(_save(fig, figdir / "13_stability_raw_vs_voladj.png"))
        except Exception as exc:
            logger.warning("stability comparison figure skipped: %s", exc)

    # 14. Survivorship coverage by era
    era = _read(processed_dir, "universe_coverage_by_era.csv")
    if era is not None and {"removal_year", "coverage"} <= set(era.columns):
        try:
            fig, ax = plt.subplots(figsize=(7.5, 4))
            colors = [GREEN if c >= 0.95 else (BLUE if c >= 0.8 else RED)
                      for c in era["coverage"]]
            ax.bar(era["removal_year"], era["coverage"] * 100, color=colors)
            ax.axhline(100, color=GREY, lw=0.8)
            ax.set_xlabel("Year the firm left the index")
            ax.set_ylabel("Price data available (%)")
            ax.set_title("Residual survivorship bias concentrates in the early sample")
            written.append(_save(fig, figdir / "14_survivorship_by_era.png"))
        except Exception as exc:
            logger.warning("survivorship figure skipped: %s", exc)

    # 15. Placeholder analyst estimates by year
    backfill = _read(processed_dir, "estimate_backfill_by_year.csv")
    if backfill is not None and "placeholder_share" in backfill.columns:
        try:
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            ax.bar(backfill["year"], backfill["placeholder_share"] * 100, color=RED)
            ax.set_xlabel("Year")
            ax.set_ylabel("Share of events (%)")
            ax.set_title("Events whose 'estimate' is the reported figure copied over (dropped)")
            written.append(_save(fig, figdir / "15_placeholder_estimates.png"))
        except Exception as exc:
            logger.warning("placeholder figure skipped: %s", exc)

    # 16. Paired model comparison
    significance = _read(processed_dir, "prediction_significance.csv")
    if significance is not None and {"ic_diff", "ic_ci_low"} <= set(significance.columns):
        try:
            labels = [f"{a} vs {b}" for a, b in
                      zip(significance["model_a"], significance["model_b"])]
            y = np.arange(len(significance))
            low = significance["ic_diff"] - significance["ic_ci_low"]
            high = significance["ic_ci_high"] - significance["ic_diff"]
            fig, ax = plt.subplots(figsize=(7, 0.9 * len(significance) + 2))
            ax.errorbar(significance["ic_diff"], y,
                        xerr=[low, high], fmt="o", ms=5, color=BLUE, capsize=4)
            ax.axvline(0, color=RED, ls="--", lw=1)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Rank-IC advantage (block-bootstrapped 95% CI)")
            ax.set_title("Does boosting beat the linear baseline out of sample?")
            written.append(_save(fig, figdir / "16_model_comparison_ci.png"))
        except Exception as exc:
            logger.warning("model comparison figure skipped: %s", exc)

    return written
