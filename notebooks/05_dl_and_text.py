# %% [markdown]
# # 05 - Deep Learning & Transcript Text (RTX 4090)
# Two GPU workloads. Run this notebook on the PC with `requirements-gpu.txt` installed.
#
# **(a) FT-Transformer tabular benchmark.** Honest question: does a deep net beat
# LightGBM on ~20k tabular events? Usually no - and showing that rigorously (same purged
# CV, same OOS fold, same metrics) is the point. A negative result here is informative.
#
# **(b) Transcript embeddings.** This is where the 4090 earns its place and where the
# Meta-vs-Shopify answer often hides (guidance tone, capex commentary). Embed call
# transcripts with a FinBERT-class model, cache vectors to parquet, reduce with
# leakage-safe PCA, and add them to both the inference and prediction tracks.

# %%
from erl.config import get_settings
from erl.predict.cv import PurgedWalkForwardCV
from erl.predict.tabular_dl import train_ft_transformer
from erl.utils import read_parquet

settings = get_settings()
panel = read_parquet(settings.processed_dir / "event_panel.parquet")
features = ["sue", "eps_beat", "both_beat", "prior_streak", "runup_20d", "runup_60d",
            "momentum_12_1", "vix_level", "rate_level", "pe_z", "mcap_decile"]

dl = train_ft_transformer(panel, "car_reaction", features, epochs=60)
print("FT-Transformer OOS:", dl.oos_metrics, "| device:", dl.config["device"])

# %% [markdown]
# ## Transcript embeddings -> parquet cache (resumable)

# %%
from erl.text.embed import TransformerEncoder, embed_frame
from erl.text.features import link_transcripts_to_events

transcripts = read_parquet(settings.interim_dir / "transcripts.parquet")
linked = link_transcripts_to_events(transcripts, panel, max_days=7)
encoder = TransformerEncoder("ProsusAI/finbert")  # GPU auto-detected
embeddings = embed_frame(linked, encoder, out_path=settings.processed_dir / "embeddings.parquet")
embeddings.head()

# %% [markdown]
# ## Add reduced embeddings to the panel, then re-run 04 to measure the text lift

# %%
from erl.text.features import EmbeddingReducer, embeddings_matrix

matrix = embeddings_matrix(panel, embeddings)
# Fit the reducer inside each CV training fold in practice; shown here on all rows for brevity.
reducer = EmbeddingReducer(n_components=16).fit(matrix[~__import__("numpy").isnan(matrix).any(axis=1)])
