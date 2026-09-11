"""Collect every computed result into one workbook and one text summary.

The pipeline scatters its output across two dozen CSVs, which is right for
reproducibility and wrong for reading. This module gathers them into a single
`results.xlsx` (one sheet per CSV, plus a Summary sheet that pulls out the
headline numbers) and a `results.md` that can be read in a terminal or pasted
into a write-up.

Nothing here computes anything. Every figure is read back from the CSVs the
earlier stages wrote, and each Summary row names the file it came from, so a
number in the workbook can always be traced to the stage that produced it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sheet order: the things a reader wants first, then supporting detail. Files
# not listed still get a sheet, appended alphabetically after these.
PREFERRED_ORDER = [
    "provenance",
    "split_adjustment",
    "harvest_manifest",
    "panel_manifest",
    "double_lasso_baseline",
    "forest_ate",
    "forest_blp",
    "forest_blp_naive",
    "forest_calibration",
    "regime_stability_test",
    "regime_stability",
    "regime_stability_voladj_test",
    "regime_stability_voladj",
    "rolling_effect",
    "rolling_effect_voladj",
    "prediction_comparison",
    "prediction_significance",
    "gbm_feature_usage",
    "gbm_best_params",
    "gbm_shap",
    "interacted_lasso",
    "alignment_diagnostic",
    "universe_coverage",
    "universe_coverage_by_era",
    "estimate_backfill_by_year",
    "market_state_symbols",
]

INVALID_SHEET_CHARS = set(r"[]:*?/\\")


def sheet_name(stem: str, taken: set[str]) -> str:
    """Excel sheet names: 31 characters, no []:*?/\\, unique within a workbook."""
    clean = "".join("_" if c in INVALID_SHEET_CHARS else c for c in stem)[:31]
    candidate = clean or "sheet"
    i = 2
    while candidate.lower() in {t.lower() for t in taken}:
        suffix = f"_{i}"
        candidate = clean[: 31 - len(suffix)] + suffix
        i += 1
    return candidate


def collect_results(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Every CSV in the processed directory, keyed by filename stem."""
    directory = Path(processed_dir)
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*.csv")):
        try:
            tables[path.stem] = pd.read_csv(path)
        except Exception as exc:  # a half-written CSV should not kill the export
            logger.warning("could not read %s: %s", path.name, exc)
    ordered = {k: tables[k] for k in PREFERRED_ORDER if k in tables}
    ordered.update({k: v for k, v in tables.items() if k not in ordered})
    logger.info("collected %d result tables from %s", len(ordered), directory)
    return ordered


def _cell(table: pd.DataFrame | None, column: str, row: int = 0):
    if table is None or table.empty or column not in table.columns:
        return None
    try:
        value = table[column].iloc[row]
    except IndexError:
        return None
    return None if pd.isna(value) else value


def _row_where(table: pd.DataFrame | None, column: str, value) -> pd.DataFrame | None:
    if table is None or table.empty or column not in table.columns:
        return None
    hit = table.loc[table[column] == value]
    return None if hit.empty else hit.reset_index(drop=True)


def provenance_check(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Do the harvest-derived and analysis-derived tables come from one run?

    The processed directory is just a folder: a full-universe harvest followed by
    an analysis that was never re-run leaves data-quality figures describing
    54,000 events sitting next to results estimated on 2,000. Every number is
    individually correct and the table as a whole is misleading. This compares
    the ticker counts the harvest and panel stages recorded and reports a
    mismatch rather than presenting the mixture silently.
    """
    harvest = tables.get("harvest_manifest")
    panel = tables.get("panel_manifest")
    if harvest is None or panel is None or harvest.empty or panel.empty:
        return {"status": "unknown", "detail":
                "no run manifests found; cannot confirm the tables describe one run"}
    h_tickers = _cell(harvest, "tickers_with_events") or 0
    p_tickers = _cell(panel, "tickers") or 0
    universe = _cell(harvest, "universe") or "?"
    if h_tickers and p_tickers and p_tickers < 0.8 * h_tickers:
        return {
            "status": "MISMATCH",
            "detail": (
                f"harvest covers {int(h_tickers)} tickers (universe '{universe}') but the "
                f"panel was built on {int(p_tickers)}; the analysis tables below are from "
                "an older, smaller run. Re-run panel, inference, predict and export."
            ),
        }
    return {
        "status": "consistent",
        "detail": (
            f"harvest {int(h_tickers)} tickers (universe '{universe}'), "
            f"panel {int(p_tickers)} tickers"
        ),
    }


def headline_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The numbers a reader should see first, each tagged with its source file."""
    rows: list[dict] = []

    def add(metric: str, value, source: str, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "source": source, "note": note})

    # Provenance first: if the tables describe different runs, nothing below is
    # safe to read as a single set of results.
    provenance = provenance_check(tables)
    add("Provenance of the tables below", provenance["status"],
        "harvest_manifest.csv, panel_manifest.csv", str(provenance["detail"]))
    if provenance["status"] == "MISMATCH":
        logger.warning("provenance: %s", provenance["detail"])
    panel_manifest = tables.get("panel_manifest")
    if panel_manifest is not None and not panel_manifest.empty:
        add("Analysis sample", f"{int(_cell(panel_manifest, 'events') or 0)} events, "
            f"{int(_cell(panel_manifest, 'tickers') or 0)} tickers", "panel_manifest.csv",
            f"{_cell(panel_manifest, 'date_min')} to {_cell(panel_manifest, 'date_max')}")

    # --- sample and data quality
    align = tables.get("alignment_diagnostic")
    if align is not None and not align.empty and "mean_abs_ar" in align.columns:
        peak = align.loc[align["mean_abs_ar"].idxmax(), "rel_day"]
        baseline = align.loc[align["rel_day"] < 0, "mean_abs_ar"].mean()
        peak_value = align["mean_abs_ar"].max()
        add("Event-day alignment: |AR| peak at rel_day", peak, "alignment_diagnostic.csv",
            "must be 0 or +1; a peak at -1 means day0 is late")
        if baseline and baseline > 0:
            add("Peak |AR| / pre-event baseline", round(float(peak_value / baseline), 2),
                "alignment_diagnostic.csv", "how far the reaction stands out from ordinary days")

    coverage = tables.get("universe_coverage")
    if coverage is not None and "group" in coverage.columns:
        left = _row_where(coverage, "group", "left the index")
        stayed = _row_where(coverage, "group", "still a member")
        if left is not None:
            add("Price coverage of names that left the index",
                round(float(left["coverage"].iloc[0]), 3), "universe_coverage.csv",
                f"{int(left['harvested'].iloc[0])} of {int(left['tickers'].iloc[0])} firms; "
                "residual survivorship bias, do not claim a bias-free panel")
        if stayed is not None:
            add("Price coverage of current members",
                round(float(stayed["coverage"].iloc[0]), 3), "universe_coverage.csv", "")

    backfill = tables.get("estimate_backfill_by_year")
    if backfill is not None and "placeholder_share" in backfill.columns:
        add("Events with a placeholder analyst estimate",
            int(backfill["placeholder"].sum()), "estimate_backfill_by_year.csv",
            "estimate equals the reported figure, so the surprise is a spurious zero; dropped")
        worst = backfill.loc[backfill["placeholder_share"].idxmax()]
        add("Worst year for placeholder estimates",
            f"{int(worst['year'])} ({worst['placeholder_share']:.1%})",
            "estimate_backfill_by_year.csv", "these cluster early; check before extending the start date")

    splits = tables.get("split_adjustment")
    if splits is not None and not splits.empty:
        remaining = _cell(splits, "split_shaped_returns_remaining")
        add("Split events adjusted for", _cell(splits, "split_events"),
            "split_adjustment.csv",
            f"{_cell(splits, 'tickers_adjusted')} tickers; FMP prices are not "
            "split-adjusted, so this is applied from the splits feed")
        add("Split-shaped returns remaining", remaining, "split_adjustment.csv",
            "zero is the target; any remainder is a fabricated reaction the feed missed")

    symbols = tables.get("market_state_symbols")
    if symbols is not None and not symbols.empty:
        rate = _cell(symbols, "rate_symbol_used")
        if rate and rate != "^TNX":
            add("Interest-rate series used", rate, "market_state_symbols.csv",
                "a proxy, not the index level; an ETF price moves opposite to yields")

    # --- the effect itself
    lasso = tables.get("double_lasso_baseline")
    if lasso is not None and not lasso.empty:
        add("Surprise effect, double lasso", _cell(lasso, "coef"),
            "double_lasso_baseline.csv",
            f"se {_cell(lasso, 'se')}, t {_cell(lasso, 'tstat')}, n {_cell(lasso, 'n')}")

    ate = tables.get("forest_ate")
    if ate is not None and not ate.empty:
        coef, se = _cell(ate, "ate"), _cell(ate, "se")
        note = f"se {se}" if se is not None else ""
        if coef is not None and se:
            note += f", t {float(coef) / float(se):.2f}"
        add("Surprise effect, causal forest ATE", coef, "forest_ate.csv", note)

    blp, naive = tables.get("forest_blp"), tables.get("forest_blp_naive")
    if blp is not None and "term" in blp.columns:
        mean_row = _row_where(blp, "term", "treatment_mean")
        if mean_row is not None:
            add("Surprise effect, BLP mean", mean_row["coef"].iloc[0], "forest_blp.csv",
                f"se {mean_row['se'].iloc[0]:.6f}, t {mean_row['tstat'].iloc[0]:.2f}, "
                "residual-based with firm-clustered errors")
        moderators = blp.loc[~blp["term"].isin(["intercept", "treatment_mean"])]
        significant = moderators.loc[moderators["tstat"].abs() > 1.96, "term"].tolist()
        add("Moderators significant at 5% (honest SEs)",
            ", ".join(significant) if significant else "none", "forest_blp.csv",
            f"of {len(moderators)} tested")
        if naive is not None and "term" in naive.columns:
            merged = moderators.merge(naive, on="term", suffixes=("_honest", "_naive"))
            if not merged.empty and (merged["se_naive"] > 0).all():
                ratio = (merged["se_honest"] / merged["se_naive"]).max()
                add("Largest SE understatement by the naive projection",
                    f"{ratio:.1f}x", "forest_blp.csv vs forest_blp_naive.csv",
                    "regressing fitted CATEs on moderators ignores their estimation error")
            naive_sig = naive.loc[
                (~naive["term"].isin(["intercept"])) & (naive["tstat"].abs() > 1.96), "term"
            ].tolist()
            add("Moderators significant at 5% (naive SEs)",
                ", ".join(naive_sig) if naive_sig else "none", "forest_blp_naive.csv",
                "shown only for contrast; not a valid inference")

    # --- stability
    for key, label in (
        ("regime_stability_test", "raw reaction"),
        ("regime_stability_voladj_test", "vol-adjusted reaction"),
    ):
        test = tables.get(key)
        if test is not None and not test.empty:
            p = _cell(test, "wald_pvalue")
            verdict = "rejects stability" if p is not None and p < 0.05 else "does not reject"
            add(f"Regime-stability Wald p ({label})", p, f"{key}.csv",
                f"chi2({_cell(test, 'wald_dof')}) = {_cell(test, 'wald_statistic')}; {verdict}")

    for key, label in (("regime_stability", "raw"), ("regime_stability_voladj", "vol-adjusted")):
        regimes = tables.get(key)
        if regimes is not None and "effect" in regimes.columns and len(regimes) > 1:
            hi = regimes.loc[regimes["effect"].idxmax()]
            lo = regimes.loc[regimes["effect"].idxmin()]
            if lo["effect"] != 0:
                add(f"Largest / smallest regime effect ({label})",
                    f"{float(hi['effect'] / lo['effect']):.2f}x", f"{key}.csv",
                    f"{hi['regime']} vs {lo['regime']}")

    for key, label in (("rolling_effect", "raw"), ("rolling_effect_voladj", "vol-adjusted")):
        rolling = tables.get(key)
        if rolling is not None and "effect" in rolling.columns and not rolling.empty:
            lo, hi = float(rolling["effect"].min()), float(rolling["effect"].max())
            add(f"Rolling effect range ({label})", f"{lo:.4f} to {hi:.4f}", f"{key}.csv",
                f"{len(rolling)} overlapping windows, so not independent observations")

    # --- prediction
    comparison = tables.get("prediction_comparison")
    if comparison is not None and "model" in comparison.columns:
        for _, row in comparison.iterrows():
            add(f"Out-of-sample rank-IC, {row['model']}", round(float(row["rank_ic"]), 4),
                "prediction_comparison.csv",
                f"R2 {row['r2']:.4f} vs the training mean, MAE {row['mae']:.4f}")

    significance = tables.get("prediction_significance")
    if significance is not None and "verdict" in significance.columns:
        for _, row in significance.iterrows():
            add(f"{row['model_a']} vs {row['model_b']}", row["verdict"],
                "prediction_significance.csv",
                f"MSE p {row['mse_pvalue']:.3f}, rank-IC gap {row['ic_diff']:.3f} "
                f"[{row['ic_ci_low']:.3f}, {row['ic_ci_high']:.3f}], n {int(row['n'])}")

    usage = tables.get("gbm_feature_usage")
    if usage is not None and "n_splits" in usage.columns:
        used = int((usage["n_splits"] > 0).sum())
        add("Features the boosted model actually split on", f"{used} of {len(usage)}",
            "gbm_feature_usage.csv",
            "one feature would mean a step function, not a model; in-sample gain, "
            "not evidence of out-of-sample skill")

    return pd.DataFrame(rows)


def write_workbook(
    tables: dict[str, pd.DataFrame], path: str | Path, summary: pd.DataFrame | None = None
) -> Path:
    """One sheet per result table, Summary first. Values only, no formulas:
    these are computed research outputs, not a model to recalculate."""
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if summary is not None and not summary.empty:
            name = sheet_name("Summary", taken)
            taken.add(name)
            summary.to_excel(writer, sheet_name=name, index=False)
        for stem, table in tables.items():
            name = sheet_name(stem, taken)
            taken.add(name)
            table.to_excel(writer, sheet_name=name, index=False)

        book = writer.book
        for worksheet in book.worksheets:
            worksheet.freeze_panes = "A2"
            for cell in worksheet[1]:
                cell.font = Font(name="Arial", size=11, bold=True)
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            for column_cells in worksheet.iter_cols(min_row=2):
                for cell in column_cells:
                    cell.font = Font(name="Arial", size=10)
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
            for i, column_cells in enumerate(worksheet.iter_cols(), start=1):
                widest = max(
                    (len(str(c.value)) for c in column_cells if c.value is not None),
                    default=10,
                )
                worksheet.column_dimensions[get_column_letter(i)].width = min(
                    max(widest + 2, 10), 60
                )
    logger.info("wrote %s (%d sheets)", path, len(taken))
    return path


def write_text_summary(
    tables: dict[str, pd.DataFrame], path: str | Path, summary: pd.DataFrame | None = None
) -> Path:
    """Markdown version of the same content, for reading in a terminal or
    pasting into a write-up."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Results",
        "",
        f"Generated from the CSVs in `{Path(path).parent.name}`. Every figure below is "
        "read back from a pipeline output; nothing is recomputed here.",
        "",
    ]
    if summary is not None and not summary.empty:
        lines += ["## Headline numbers", "", summary.to_markdown(index=False), ""]
    lines += ["## Full tables", ""]
    for stem, table in tables.items():
        lines += [f"### {stem}", ""]
        if len(table) > 60:
            lines += [
                f"{len(table)} rows; first 30 and last 10 shown. Full data in "
                f"`{stem}.csv` or the workbook.",
                "",
                pd.concat([table.head(30), table.tail(10)]).to_markdown(index=False),
            ]
        else:
            lines.append(table.to_markdown(index=False))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def export_results(processed_dir: str | Path) -> list[Path]:
    tables = collect_results(processed_dir)
    if not tables:
        logger.warning("no result CSVs found in %s; run the earlier stages first", processed_dir)
        return []
    summary = headline_summary(tables)
    directory = Path(processed_dir)
    provenance = provenance_check(tables)
    pd.DataFrame([provenance]).to_csv(directory / "provenance.csv", index=False)
    tables = {"provenance": pd.DataFrame([provenance]), **tables}
    written = [
        write_workbook(tables, directory / "results.xlsx", summary),
        write_text_summary(tables, directory / "results.md", summary),
    ]
    if not summary.empty:
        summary.to_csv(directory / "results_summary.csv", index=False)
        written.append(directory / "results_summary.csv")
        logger.info("headline summary:\n%s", summary.to_string(index=False))
    return written
