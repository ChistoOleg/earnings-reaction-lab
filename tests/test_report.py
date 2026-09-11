from __future__ import annotations

import pandas as pd
import pytest

from erl.report import diagnostic_figures


def _write(directory) -> None:
    pd.DataFrame({"rel_day": [-3, -2, -1, 0, 1, 2, 3],
                  "mean_abs_ar": [0.011, 0.011, 0.011, 0.0215, 0.0364, 0.0138, 0.0127],
                  "n": [2236] * 7}).to_csv(directory / "alignment_diagnostic.csv", index=False)
    pd.DataFrame({"term": ["intercept", "treatment_mean", "runup_20d", "mcap_decile"],
                  "coef": [0.0, 0.00505, -0.00055, 0.00199],
                  "se": [0.0018, 0.00133, 0.00116, 0.00093],
                  "tstat": [0.0, 3.81, -0.47, 2.14]}).to_csv(
        directory / "forest_blp.csv", index=False)
    pd.DataFrame({"term": ["intercept", "runup_20d", "mcap_decile"],
                  "coef": [0.005, -0.00016, 0.00087],
                  "se": [0.00008, 0.0000335, 0.0000947],
                  "tstat": [60.4, -4.91, 9.2]}).to_csv(
        directory / "forest_blp_naive.csv", index=False)
    for name, scale in (("regime_stability.csv", 0.003), ("regime_stability_voladj.csv", 0.44)):
        pd.DataFrame({"regime": ["a..b", "b..c", "c..end"], "n": [1168, 209, 587],
                      "effect": [scale, scale * 0.8, scale * 3.0],
                      "se": [scale * 0.6] * 3, "tstat": [1.5, 1.0, 5.4],
                      "is_base": [True, False, False]}).to_csv(directory / name, index=False)
    for name, scale in (("rolling_effect.csv", 0.004), ("rolling_effect_voladj.csv", 0.5)):
        pd.DataFrame({"window_start": pd.date_range("2010-01-01", periods=8, freq="180D"),
                      "window_end": pd.date_range("2014-01-01", periods=8, freq="180D"),
                      "n": [400] * 8, "effect": [scale] * 8, "se": [scale * 0.3] * 8,
                      "tstat": [3.0] * 8}).to_csv(directory / name, index=False)
    pd.DataFrame({"removal_year": [2005, 2015, 2025], "tickers": [18, 25, 19],
                  "harvested": [11, 13, 19], "coverage": [0.61, 0.52, 1.0]}).to_csv(
        directory / "universe_coverage_by_era.csv", index=False)
    pd.DataFrame({"year": [2005, 2015, 2025], "events": [2000] * 3,
                  "placeholder": [146, 99, 2],
                  "placeholder_share": [0.073, 0.049, 0.001]}).to_csv(
        directory / "estimate_backfill_by_year.csv", index=False)
    pd.DataFrame([{"model_a": "lightgbm", "model_b": "ols", "ic_diff": 0.009,
                   "ic_ci_low": 0.001, "ic_ci_high": 0.018}]).to_csv(
        directory / "prediction_significance.csv", index=False)


def test_all_diagnostic_figures_render(tmp_path):
    _write(tmp_path)
    figdir = tmp_path / "figures"
    figdir.mkdir()
    written = diagnostic_figures(tmp_path, figdir)
    names = sorted(p.name for p in written)
    assert names == [
        "10_alignment_diagnostic.png",
        "11_honest_vs_naive_ci.png",
        "12_regime_effects.png",
        "13_stability_raw_vs_voladj.png",
        "14_survivorship_by_era.png",
        "15_placeholder_estimates.png",
        "16_model_comparison_ci.png",
    ]
    assert all(p.stat().st_size > 5000 for p in written)


def test_missing_inputs_are_skipped_not_fatal(tmp_path):
    """A partial run must still produce whatever figures it can."""
    figdir = tmp_path / "figures"
    figdir.mkdir()
    assert diagnostic_figures(tmp_path, figdir) == []

    pd.DataFrame({"rel_day": [0, 1], "mean_abs_ar": [0.02, 0.03], "n": [10, 10]}).to_csv(
        tmp_path / "alignment_diagnostic.csv", index=False)
    written = diagnostic_figures(tmp_path, figdir)
    assert [p.name for p in written] == ["10_alignment_diagnostic.png"]


def test_regime_figure_falls_back_to_one_panel_without_voladj(tmp_path):
    _write(tmp_path)
    (tmp_path / "regime_stability_voladj.csv").unlink()
    (tmp_path / "rolling_effect_voladj.csv").unlink()
    figdir = tmp_path / "figures"
    figdir.mkdir()
    names = {p.name for p in diagnostic_figures(tmp_path, figdir)}
    assert "12_regime_effects.png" in names          # single panel still drawn
    assert "13_stability_raw_vs_voladj.png" not in names  # comparison needs both


def test_corrupt_csv_does_not_kill_the_stage(tmp_path):
    _write(tmp_path)
    (tmp_path / "forest_blp_naive.csv").write_text("not,a\nvalid")
    figdir = tmp_path / "figures"
    figdir.mkdir()
    names = {p.name for p in diagnostic_figures(tmp_path, figdir)}
    assert "10_alignment_diagnostic.png" in names
    assert "16_model_comparison_ci.png" in names
