# %% [markdown]
# # 06 - Findings
# Pulls together the saved outputs of notebooks 01-05 into the headline story:
# why identical earnings beats produce different price reactions.
#
# Fill these in after a full run. The narrative the README promises lives here:
# 1. Average effect of SUE on the reaction (from 02).
# 2. The moderators that survive BH correction (from 02) and the forest BLP (from 03).
# 3. Whether discovered heterogeneity is calibrated (from 03).
# 4. Out-of-sample predictability ceiling (from 04), and whether DL or text moved it (05).
# 5. The Meta-vs-Shopify cases re-examined through the fitted moderators.

# %%
import pandas as pd
from erl.config import get_settings

settings = get_settings()
p = settings.processed_dir

interacted = pd.read_csv(p / "interacted_lasso.csv")
blp = pd.read_csv(p / "forest_blp.csv")
calibration = pd.read_csv(p / "forest_calibration.csv")
gbm_folds = pd.read_csv(p / "gbm_folds.csv")
gbm_shap = pd.read_csv(p / "gbm_shap.csv")

# %% [markdown]
# ## Significant moderators (BH-adjusted interactions)

# %%
interacted[interacted["significant_bh"]]

# %% [markdown]
# ## Forest agreement: do BLP directions match the lasso interactions?

# %%
blp.merge(
    interacted.assign(term=interacted["term"].str.replace("sue_x_", "", regex=False)),
    on="term", how="outer", suffixes=("_forest", "_lasso"),
)
