# %% [markdown]
# # 02 - Double-Selection Lasso
# **Question:** what is the effect of surprise size on the reaction, with honest standard
# errors, when there are many candidate controls?
#
# **Why this tool:** with dozens of correlated firm/market controls, kitchen-sink OLS
# overfits and stepwise selection invalidates inference. Post-double-selection lasso
# (Belloni-Chernozhukov-Hansen) selects controls from *both* the outcome and treatment
# equations, so the focal coefficient stays valid after selection. We use the plug-in
# penalty (not CV lambda) because CV over-selects and breaks the inference guarantee,
# and two-way clustered SEs (firm x quarter) because earnings events share shocks in
# both dimensions.

# %%
import pandas as pd
from erl.config import get_settings
from erl.inference.double_lasso import fit_double_lasso, fit_interacted_double_lasso
from erl.utils import read_parquet

settings = get_settings()
panel = read_parquet(settings.processed_dir / "event_panel.parquet")
controls = ["eps_beat", "both_beat", "prior_streak", "runup_20d", "runup_60d",
            "momentum_12_1", "vix_level", "rate_level", "pe_z", "mcap_decile"]
moderators = ["runup_20d", "pe_z", "rate_level", "mcap_decile", "momentum_12_1"]

# %% [markdown]
# ## Baseline: average effect of SUE on the 0-1 day reaction

# %%
baseline = fit_double_lasso(panel, "car_reaction", "sue", controls)
print(f"effect={baseline.coef:.4f}  se={baseline.se:.4f}  t={baseline.tstat:.2f}  n={baseline.n}")
print("selected controls:", baseline.selected_controls)

# %% [markdown]
# ## Which conditions *moderate* the reaction?
# Pre-specified interactions (SUE x moderator), Benjamini-Hochberg correction applied to
# the interaction family only. This is the interpretable, communicable answer to the
# Meta-vs-Shopify question.

# %%
interactions = fit_interacted_double_lasso(panel, "car_reaction", "sue", moderators, controls)
interactions
