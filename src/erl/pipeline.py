from __future__ import annotations

import argparse
import logging

import pandas as pd

from erl.config import get_settings
from erl.events.panel import build_panel
from erl.fmp import FMPClient
from erl.harvest.fundamentals import harvest_fundamentals
from erl.harvest.prices import harvest_prices
from erl.harvest.surprises import harvest_surprises
from erl.pilot import OUT_OF_UNIVERSE_EXTRAS, PILOT_TICKERS
from erl.universe import fetch_membership, union_members
from erl.utils import read_parquet, write_parquet

logger = logging.getLogger(__name__)

# momentum_12_1 is market-adjusted to match the run-ups; the total-return
# version is built but kept out so the horizons are comparable.
TABULAR_FEATURES = [
    "sue", "eps_beat", "both_beat", "prior_streak",
    "runup_20d", "runup_60d", "momentum_12_1",
    "vix_level", "rate_level", "pe_z", "mcap_decile",
]
MODERATORS = ["runup_20d", "pe_z", "rate_level", "mcap_decile", "momentum_12_1"]


def make_client() -> FMPClient:
    settings = get_settings()
    return FMPClient(
        settings.fmp_api_key,
        settings.raw_dir / "fmp_cache",
        interval_seconds=settings.request_interval_seconds,
        max_retries=settings.max_retries,
        timeout=settings.http_timeout_seconds,
    )


def resolve_universe(client: FMPClient, mode: str) -> list[str]:
    settings = get_settings()
    if mode == "pilot":
        return PILOT_TICKERS + OUT_OF_UNIVERSE_EXTRAS

    # sp500 mode adapts to the plan: point-in-time (Premium) -> current (Starter)
    # -> curated subset (no special endpoint). Each fallback is logged.
    from erl.fmp import FMPError
    from erl.pilot import SP500_SUBSET
    from erl.universe import (
        current_members,
        fetch_membership,
        membership_from_current,
    )

    try:
        membership = fetch_membership(client, settings.start_date)
        if not membership.empty:
            write_parquet(membership, settings.interim_dir / "membership.parquet")
            logger.info(
                "point-in-time S&P 500 membership: %d names", membership["ticker"].nunique()
            )
            return union_members(membership)
    except FMPError as exc:
        logger.warning("historical constituents unavailable (%s); trying current list", exc)

    try:
        symbols = current_members(client)
        if symbols:
            write_parquet(
                membership_from_current(symbols), settings.interim_dir / "membership.parquet"
            )
            logger.warning(
                "using CURRENT S&P 500 constituents (%d names) - survivorship bias present, "
                "documented in README",
                len(symbols),
            )
            return symbols
    except FMPError as exc:
        logger.warning("current constituents unavailable (%s); using curated subset", exc)

    logger.warning(
        "using curated %d-name S&P 500 subset (survivorship bias present)", len(SP500_SUBSET)
    )
    write_parquet(
        membership_from_current(SP500_SUBSET), settings.interim_dir / "membership.parquet"
    )
    return SP500_SUBSET


def usable_features(panel: pd.DataFrame, features: list[str], min_coverage: float = 0.6) -> list[str]:
    """Keep only features present and non-null for at least min_coverage of rows,
    so sparse fields (e.g. annual-only pe_z on Starter) can't collapse the sample
    via dropna in the estimators."""
    n = len(panel)
    if n == 0:
        return []
    return [
        f for f in features
        if f in panel.columns and float(panel[f].notna().mean()) >= min_coverage
    ]


def stage_harvest(mode: str = "pilot") -> None:
    settings = get_settings()
    settings.ensure_dirs()
    client = make_client()
    tickers = resolve_universe(client, mode)
    logger.info("harvesting %d tickers (%s)", len(tickers), mode)

    surprises = harvest_surprises(client, tickers, settings.start_date,
                                  out_path=settings.interim_dir / "surprises.parquet")
    pd.DataFrame([{
        "stage": "harvest",
        "universe": mode,
        "tickers_requested": len(tickers),
        "events": len(surprises),
        "tickers_with_events": int(surprises["ticker"].nunique()) if not surprises.empty else 0,
        "start_date": settings.start_date,
        "date_min": str(surprises["announce_date"].min()) if not surprises.empty else "",
        "date_max": str(surprises["announce_date"].max()) if not surprises.empty else "",
    }]).to_csv(settings.processed_dir / "harvest_manifest.csv", index=False)
    if not surprises.empty and "estimate_backfilled" in surprises.columns:
        from erl.harvest.surprises import backfill_rate_by_year

        backfill_rate_by_year(surprises).to_csv(
            settings.processed_dir / "estimate_backfill_by_year.csv", index=False
        )
    prices = harvest_prices(client, tickers + settings.benchmark_symbols, settings.start_date)
    # Neither price endpoint is split-adjusted, so adjust before anything
    # downstream computes a return.
    from erl.harvest.splits import adjust_for_splits, harvest_splits, verify_adjustment

    # Everything in the price frame, not just the equities: VIXY has
    # reverse-split repeatedly. Index symbols never split, so they are skipped
    # rather than sent as doomed requests.
    split_symbols = [
        s for s in dict.fromkeys(tickers + settings.benchmark_symbols)
        if not s.startswith("^")
    ]
    splits = harvest_splits(client, split_symbols, settings.start_date,
                            out_path=settings.interim_dir / "splits.parquet")
    if prices.empty:
        logger.error(
            "no prices harvested; leaving any existing prices.parquet in place, which "
            "means downstream stages would run on stale data. Fix the harvest first."
        )
    else:
        prices = adjust_for_splits(prices, splits)
        remaining = verify_adjustment(prices)
        from erl.harvest.splits import suspicious_tickers, unexplained_artifacts

        artifacts = unexplained_artifacts(prices, splits)
        if not artifacts.empty:
            artifacts.to_csv(
                settings.processed_dir / "residual_artifacts.csv", index=False
            )
            flagged = suspicious_tickers(artifacts)
            if not flagged.empty:
                flagged.to_csv(
                    settings.processed_dir / "suspicious_tickers.csv", index=False
                )
        pd.DataFrame([{
            "split_events": len(splits),
            "splits_applied": prices.attrs.get("splits_applied"),
            "splits_already_adjusted": prices.attrs.get("splits_already_adjusted"),
            "splits_ambiguous": prices.attrs.get("splits_ambiguous"),
            "tickers_adjusted": int(prices["split_adjusted"].groupby(
                prices["ticker"]).any().sum()) if "split_adjusted" in prices.columns else 0,
            "split_shaped_returns_remaining": remaining,
            "artifacts_on_tickers_the_feed_covers": int(
                artifacts["feed_covers_ticker"].sum()) if not artifacts.empty else 0,
            "suspicious_tickers": int(len(flagged)) if not artifacts.empty else 0,
        }]).to_csv(settings.processed_dir / "split_adjustment.csv", index=False)
        write_parquet(prices, settings.interim_dir / "prices.parquet")
    harvest_fundamentals(client, tickers, settings.start_date,
                         out_path=settings.interim_dir / "fundamentals.parquet")
    client.close()

    # Did the delisted names actually arrive, or only the survivors?
    membership_path = settings.interim_dir / "membership.parquet"
    prices_path = settings.interim_dir / "prices.parquet"
    if membership_path.exists() and prices_path.exists():
        from erl.universe import membership_coverage

        membership = pd.read_parquet(membership_path)
        harvested = set(pd.read_parquet(prices_path, columns=["ticker"])["ticker"].unique())
        coverage = membership_coverage(membership, harvested)
        coverage.to_csv(settings.processed_dir / "universe_coverage.csv", index=False)
        by_era = coverage.attrs.get("by_era") or []
        if by_era:
            era = pd.DataFrame(by_era)
            era.to_csv(settings.processed_dir / "universe_coverage_by_era.csv", index=False)
            logger.info(
                "coverage of departed names by removal year:\n%s", era.to_string(index=False)
            )
        missing = coverage.attrs.get("missing_tickers") or []
        if missing:
            pd.DataFrame({"ticker": missing}).to_csv(
                settings.processed_dir / "universe_missing.csv", index=False
            )


def stage_panel() -> pd.DataFrame:
    settings = get_settings()
    events = read_parquet(settings.interim_dir / "surprises.parquet")
    prices = read_parquet(settings.interim_dir / "prices.parquet")
    fundamentals_path = settings.interim_dir / "fundamentals.parquet"
    if fundamentals_path.exists():
        fundamentals = read_parquet(fundamentals_path)
    else:
        fundamentals = None
        logger.warning(
            "no fundamentals.parquet found; building panel without valuation "
            "features (pe_z, mcap_decile will be NaN)"
        )
    panel = build_panel(events, prices, fundamentals, benchmark=settings.benchmark_symbol)
    write_parquet(panel, settings.processed_dir / "event_panel.parquet")
    artifacts_path = settings.processed_dir / "residual_artifacts.csv"
    if artifacts_path.exists():
        from erl.harvest.splits import events_touched

        artifacts = pd.read_csv(artifacts_path, parse_dates=["date"])
        touched = events_touched(panel, artifacts)
        share = touched / max(len(panel), 1)
        logger.info(
            "%d of %d events (%.2f%%) have an unexplained split-shaped return inside a "
            "feature window", touched, len(panel), 100 * share,
        )
        if share > 0.02:
            logger.warning(
                "%.1f%% of events are exposed to a residual price artifact; report this "
                "or exclude the affected tickers", 100 * share,
            )
        pd.DataFrame([{"events_touched": touched, "events": len(panel), "share": share}]).to_csv(
            settings.processed_dir / "artifact_exposure.csv", index=False
        )

    pd.DataFrame([{
        "stage": "panel",
        "events": len(panel),
        "tickers": int(panel["ticker"].nunique()) if "ticker" in panel.columns else 0,
        "date_min": str(pd.to_datetime(panel["announce_date"]).min()),
        "date_max": str(pd.to_datetime(panel["announce_date"]).max()),
    }]).to_csv(settings.processed_dir / "panel_manifest.csv", index=False)

    used = {k: panel.attrs.get(k) for k in ("vix_symbol_used", "rate_symbol_used")}
    if any(used.values()):
        logger.info("market-state series used: %s", used)
        pd.DataFrame([used]).to_csv(
            settings.processed_dir / "market_state_symbols.csv", index=False
        )

    diagnostic = panel.attrs.get("alignment_diagnostic")
    if diagnostic:
        pd.DataFrame(diagnostic).to_csv(
            settings.processed_dir / "alignment_diagnostic.csv", index=False
        )
    logger.info("panel built: %d events, %d columns", len(panel), panel.shape[1])
    return panel


def stage_inference() -> None:
    settings = get_settings()
    from erl.inference.causal_forest import fit_causal_forest
    from erl.inference.double_lasso import fit_double_lasso, fit_interacted_double_lasso

    panel = read_parquet(settings.processed_dir / "event_panel.parquet")
    controls = usable_features(panel, [c for c in TABULAR_FEATURES if c != "sue"])
    moderators = usable_features(panel, MODERATORS)
    logger.info("inference controls: %s", controls)
    logger.info("inference moderators: %s", moderators)

    baseline = fit_double_lasso(panel, "car_reaction", "sue", controls)
    logger.info(
        "baseline surprise effect: %.4f (se %.4f, t %.2f, n %d)",
        baseline.coef, baseline.se, baseline.tstat, baseline.n,
    )
    pd.DataFrame([{
        "coef": baseline.coef, "se": baseline.se, "tstat": baseline.tstat,
        "pvalue": baseline.pvalue, "n": baseline.n,
        "selected_controls": ";".join(baseline.selected_controls),
    }]).to_csv(settings.processed_dir / "double_lasso_baseline.csv", index=False)

    # One relationship, or an average across regimes?
    from erl.inference.stability import regime_stability, rolling_effect

    regimes = regime_stability(panel, "car_reaction", "sue", controls)
    if not regimes.empty:
        regimes.to_csv(settings.processed_dir / "regime_stability.csv", index=False)
        pd.DataFrame([{
            "wald_statistic": regimes.attrs.get("wald_statistic"),
            "wald_dof": regimes.attrs.get("wald_dof"),
            "wald_pvalue": regimes.attrs.get("wald_pvalue"),
            "n": regimes.attrs.get("n"),
            "breaks": ";".join(regimes.attrs.get("breaks", [])),
        }]).to_csv(settings.processed_dir / "regime_stability_test.csv", index=False)
    rolling = rolling_effect(panel, "car_reaction", "sue")
    if not rolling.empty:
        rolling.to_csv(settings.processed_dir / "rolling_effect.csv", index=False)

    # A change in pricing, or just in volatility? If the break survives
    # standardisation it is not an artifact of scale.
    if "car_reaction_vol_adj" in panel.columns:
        adj_regimes = regime_stability(panel, "car_reaction_vol_adj", "sue", controls)
        if not adj_regimes.empty:
            adj_regimes.to_csv(
                settings.processed_dir / "regime_stability_voladj.csv", index=False
            )
            pd.DataFrame([{
                "wald_statistic": adj_regimes.attrs.get("wald_statistic"),
                "wald_dof": adj_regimes.attrs.get("wald_dof"),
                "wald_pvalue": adj_regimes.attrs.get("wald_pvalue"),
                "n": adj_regimes.attrs.get("n"),
                "target": "car_reaction_vol_adj",
            }]).to_csv(
                settings.processed_dir / "regime_stability_voladj_test.csv", index=False
            )
        adj_rolling = rolling_effect(panel, "car_reaction_vol_adj", "sue")
        if not adj_rolling.empty:
            adj_rolling.to_csv(
                settings.processed_dir / "rolling_effect_voladj.csv", index=False
            )
    if not moderators:
        logger.warning(
            "no moderators with sufficient coverage on this plan; skipping interaction "
            "and causal-forest analysis (upgrade for quarterly fundamentals to enable them)"
        )
        return
    interacted = fit_interacted_double_lasso(panel, "car_reaction", "sue", moderators, controls)
    interacted.to_csv(settings.processed_dir / "interacted_lasso.csv", index=False)
    forest = fit_causal_forest(panel, "car_reaction", "sue", moderators,
                               controls=controls, cluster="ticker")
    forest.blp.to_csv(settings.processed_dir / "forest_blp.csv", index=False)
    if forest.blp_naive is not None:
        forest.blp_naive.to_csv(settings.processed_dir / "forest_blp_naive.csv", index=False)
    forest.calibration.to_csv(settings.processed_dir / "forest_calibration.csv", index=False)
    pd.DataFrame([{"ate": forest.ate, "se": forest.ate_se, "n": forest.n}]).to_csv(
        settings.processed_dir / "forest_ate.csv", index=False
    )


def stage_predict() -> None:
    settings = get_settings()
    from erl.predict.baseline import train_linear_baselines
    from erl.predict.gbm import shap_importance, train_gbm

    panel = read_parquet(settings.processed_dir / "event_panel.parquet")
    features = usable_features(panel, TABULAR_FEATURES)
    logger.info("prediction features: %s", features)

    # Linear baselines first: ML must beat a simple linear model to be justified.
    baseline = train_linear_baselines(panel, "car_reaction", features)
    baseline.fold_metrics.to_csv(settings.processed_dir / "baseline_folds.csv", index=False)
    baseline.oos_predictions.to_csv(
        settings.processed_dir / "baseline_oos_predictions.csv", index=False
    )

    result = train_gbm(panel, "car_reaction", features)
    result.fold_metrics.to_csv(settings.processed_dir / "gbm_folds.csv", index=False)
    result.oos_predictions.to_csv(settings.processed_dir / "gbm_oos_predictions.csv", index=False)
    if result.feature_usage is not None:
        result.feature_usage.to_csv(settings.processed_dir / "gbm_feature_usage.csv", index=False)
    pd.DataFrame([result.best_params]).to_csv(
        settings.processed_dir / "gbm_best_params.csv", index=False
    )
    logger.info("gbm OOS: %s", result.oos_metrics)

    # Single like-for-like comparison on the same final out-of-sample fold.
    comparison = pd.DataFrame(
        [
            {"model": "ols", **baseline.oos_metrics["ols"]},
            {"model": "ridge", **baseline.oos_metrics["ridge"]},
            {"model": "lightgbm", **result.oos_metrics},
        ]
    )
    comparison.to_csv(settings.processed_dir / "prediction_comparison.csv", index=False)
    logger.info(
        "OOS prediction comparison (same fold, same metrics):\n%s",
        comparison.to_string(index=False),
    )

    # The metrics alone cannot say whether the ordering is real.
    from erl.predict.compare import compare_models

    paired = baseline.oos_predictions.copy()
    gbm_pred = result.oos_predictions
    key = "event_id" if "event_id" in gbm_pred.columns and "event_id" in paired.columns else None
    if key is None:
        # Both frames come from the same final fold in the same row order.
        if len(paired) == len(gbm_pred):
            paired["y_pred_lightgbm"] = gbm_pred["y_pred"].to_numpy()
        else:
            logger.warning(
                "cannot align baseline (%d rows) and gbm (%d rows) predictions; "
                "skipping the paired significance test",
                len(paired), len(gbm_pred),
            )
            paired = None
    else:
        paired = paired.merge(
            gbm_pred[[key, "y_pred"]].rename(columns={"y_pred": "y_pred_lightgbm"}),
            on=key, how="inner",
        )
    if paired is not None and not paired.empty:
        significance = compare_models(paired, "lightgbm", ["ols", "ridge"])
        if not significance.empty:
            significance.to_csv(
                settings.processed_dir / "prediction_significance.csv", index=False
            )

    try:
        shap_importance(result.model, panel[result.features].dropna()).to_csv(
            settings.processed_dir / "gbm_shap.csv", index=False
        )
    except Exception as exc:  # SHAP is a nice-to-have; never let it fail the stage
        logger.warning("SHAP importance skipped: %s", exc)


def stage_plots() -> None:
    settings = get_settings()
    from erl.report import make_figures

    panel = read_parquet(settings.processed_dir / "event_panel.parquet")
    prices_path = settings.interim_dir / "prices.parquet"
    prices = read_parquet(prices_path) if prices_path.exists() else None
    written = make_figures(panel, prices, settings.processed_dir, settings.benchmark_symbol)
    for path in written:
        logger.info("figure: %s", path)


def stage_export() -> None:
    settings = get_settings()
    from erl.export import export_results

    written = export_results(settings.processed_dir)
    for path in written:
        logger.info("export: %s", path)


STAGES = {
    "harvest": lambda args: stage_harvest(args.universe),
    "panel": lambda args: stage_panel(),
    "inference": lambda args: stage_inference(),
    "predict": lambda args: stage_predict(),
    "plots": lambda args: stage_plots(),
    "export": lambda args: stage_export(),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # httpx logs full request URLs at INFO and the key travels as a query param,
    # so a redirected log would carry the secret on every line.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Earnings Reaction Lab pipeline")
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--universe", choices=["pilot", "sp500"], default="pilot")
    args = parser.parse_args()
    order = ["harvest", "panel", "inference", "predict", "plots", "export"]
    todo = order if args.stage == "all" else [args.stage]
    for stage in todo:
        logger.info("=== stage: %s ===", stage)
        STAGES[stage](args)


if __name__ == "__main__":
    main()
