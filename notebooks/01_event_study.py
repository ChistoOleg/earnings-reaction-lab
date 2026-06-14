# %% [markdown]
# # 01 - Event Study
# **Question:** is the data right, and what is the *average* reaction to an earnings surprise?
#
# Before any fancy estimator, we establish the basic stylized fact: bigger standardized
# surprises (SUE) should map to larger announcement returns, and beats should drift.
# If this monotone relationship is absent, something upstream is broken - this notebook
# is the data sanity check as much as the first result.

# %%
import pandas as pd
from erl.config import get_settings
from erl.events.returns import ReturnContext
from erl.inference.eventstudy import ar_path, car_by_quantile
from erl.utils import read_parquet

settings = get_settings()
panel = read_parquet(settings.processed_dir / "event_panel.parquet")
prices = read_parquet(settings.interim_dir / "prices.parquet")

# %% [markdown]
# ## CAR by SUE quintile (cluster-robust t-stats, clustered by quarter)

# %%
quintiles = car_by_quantile(panel, target="car_reaction", by="sue", bins=5)
quintiles

# %% [markdown]
# ## Average abnormal-return path: beats vs misses (the PEAD picture)

# %%
ctx = ReturnContext(prices, settings.benchmark_symbol)
path = ar_path(panel, ctx, rel_days=(-5, 25))
path.pivot_table(index="rel_day", columns="group", values="cum_ar").plot()

# %% [markdown]
# ## The motivating anecdote
# Pull META and SHOP events with similar EPS surprise % but opposite reactions -
# the concrete puzzle the rest of the repo tries to explain.

# %%
anecdote = panel[panel["ticker"].isin(["META", "SHOP"])][
    ["ticker", "announce_date", "surprise_pct", "car_reaction", "car_drift"]
]
anecdote.sort_values("announce_date").tail(12)
