# %% [markdown]
# # 04 - Prediction (LightGBM + SHAP)
# **Question:** can we *predict* the reaction out of sample? (Distinct from the inference
# notebooks - here we care about OOS accuracy, not unbiased coefficients.)
#
# **Why this setup:** trees regularize internally, so no variable selection is needed -
# "choosing parameters" here means hyperparameter tuning, done with Optuna on interior
# folds only, leaving the final fold untouched as an honest out-of-time test. The CV is
# purged and embargoed (Lopez de Prado) because the drift label spans ~a month; ordinary
# k-fold would let the model peek across the label horizon.
#
# **Expectation:** real OOS R-squared will be low single digits and rank IC ~0.05-0.10.
# That is a genuine result, not a failure - reactions are mostly idiosyncratic.

# %%
from erl.config import get_settings
from erl.predict.gbm import shap_importance, train_gbm
from erl.utils import read_parquet

settings = get_settings()
panel = read_parquet(settings.processed_dir / "event_panel.parquet")
features = ["sue", "eps_beat", "both_beat", "prior_streak", "runup_20d", "runup_60d",
            "momentum_12_1", "vix_level", "rate_level", "pe_z", "mcap_decile"]

result = train_gbm(panel, "car_reaction", features, n_trials=40)
print("OOS metrics:", result.oos_metrics)
result.fold_metrics

# %% [markdown]
# ## SHAP: which features drive predictions, and in which direction

# %%
shap_importance(result.model, panel[features].dropna())
