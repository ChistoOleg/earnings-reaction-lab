# %% [markdown]
# # 03 - Causal Forest (heterogeneity discovery)
# **Question:** which conditions moderate the reaction, *without* us pre-specifying them?
#
# **Why this tool:** notebook 02 tests interactions we chose in advance. A causal forest
# (CausalForestDML) discovers heterogeneity nonparametrically, then we validate it. It is
# itself a DML method - it residualizes outcome and treatment with ML nuisance models
# before splitting - so we use GroupKFold by ticker for cluster-aware cross-fitting.
# Crucially we do NOT trust the forest until it passes calibration: with ~20k events it
# will find noise heterogeneity unless checked.

# %%
from erl.config import get_settings
from erl.inference.causal_forest import fit_causal_forest
from erl.utils import read_parquet

settings = get_settings()
panel = read_parquet(settings.processed_dir / "event_panel.parquet")
controls = ["eps_beat", "both_beat", "prior_streak", "runup_60d",
            "vix_level", "momentum_12_1"]
moderators = ["runup_20d", "pe_z", "rate_level", "mcap_decile", "momentum_12_1"]

forest = fit_causal_forest(panel, "car_reaction", "sue", moderators,
                           controls=controls, cluster="ticker", n_estimators=1500)
print("ATE:", forest.ate, " CATE sd:", forest.cate.std())

# %% [markdown]
# ## Best linear projection of CATEs onto moderators (which dimensions drive variation)

# %%
forest.blp

# %% [markdown]
# ## Calibration / sort test - is the discovered heterogeneity real?
# Bucket events by predicted CATE, re-estimate the realized effect within each bucket.
# A monotone predicted->realized relationship (positive top-minus-bottom, high rank
# correlation) means the forest found signal, not noise.

# %%
print("top-minus-bottom:", forest.calibration.attrs.get("top_minus_bottom"))
print("rank correlation:", forest.calibration.attrs.get("rank_correlation"))
forest.calibration
