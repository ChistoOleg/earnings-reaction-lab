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

    harvest_surprises(client, tickers, settings.start_date,
                      out_path=settings.interim_dir / "surprises.parquet")
    harvest_prices(client, tickers + settings.benchmark_symbols, settings.start_date,
                   out_path=settings.interim_dir / "prices.parquet")
    harvest_fundamentals(client, tickers, settings.start_date,
                         out_path=settings.interim_dir / "fundamentals.parquet")
    client.close()


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
    forest.calibration.to_csv(settings.processed_dir / "forest_calibration.csv", index=False)


def stage_predict() -> None:
    settings = get_settings()
    from erl.predict.gbm import shap_importance, train_gbm

    panel = read_parquet(settings.processed_dir / "event_panel.parquet")
    features = usable_features(panel, TABULAR_FEATURES)
    logger.info("prediction features: %s", features)
    result = train_gbm(panel, "car_reaction", features)
    result.fold_metrics.to_csv(settings.processed_dir / "gbm_folds.csv", index=False)
    result.oos_predictions.to_csv(settings.processed_dir / "gbm_oos_predictions.csv", index=False)
    logger.info("gbm OOS: %s", result.oos_metrics)
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


STAGES = {
    "harvest": lambda args: stage_harvest(args.universe),
    "panel": lambda args: stage_panel(),
    "inference": lambda args: stage_inference(),
    "predict": lambda args: stage_predict(),
    "plots": lambda args: stage_plots(),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Earnings Reaction Lab pipeline")
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--universe", choices=["pilot", "sp500"], default="pilot")
    args = parser.parse_args()
    order = ["harvest", "panel", "inference", "predict", "plots"]
    todo = order if args.stage == "all" else [args.stage]
    for stage in todo:
        logger.info("=== stage: %s ===", stage)
        STAGES[stage](args)


if __name__ == "__main__":
    main()
