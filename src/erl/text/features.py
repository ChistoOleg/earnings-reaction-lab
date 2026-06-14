from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def link_transcripts_to_events(
    transcripts: pd.DataFrame,
    events: pd.DataFrame,
    max_days: int = 7,
) -> pd.DataFrame:
    left = transcripts.dropna(subset=["call_date"]).copy()
    left["call_date"] = pd.to_datetime(left["call_date"])
    left = left.sort_values("call_date")
    right = events[["event_id", "ticker", "announce_date"]].copy()
    right["announce_date"] = pd.to_datetime(right["announce_date"])
    right = right.sort_values("announce_date")
    merged = pd.merge_asof(
        left,
        right,
        left_on="call_date",
        right_on="announce_date",
        by="ticker",
        direction="nearest",
        tolerance=pd.Timedelta(days=max_days),
    )
    linked = merged.dropna(subset=["event_id"]).reset_index(drop=True)
    logger.info(
        "linked %d of %d transcripts to events (tol=%dd)",
        len(linked),
        len(transcripts),
        max_days,
    )
    return linked


def embeddings_matrix(
    panel: pd.DataFrame,
    embeddings: pd.DataFrame,
    id_col: str = "event_id",
) -> np.ndarray:
    if embeddings.empty:
        return np.empty((len(panel), 0))
    lookup = {row[id_col]: np.asarray(row["embedding"], dtype=float)
              for _, row in embeddings.iterrows()}
    dim = len(next(iter(lookup.values())))
    rows = [lookup.get(eid, np.full(dim, np.nan)) for eid in panel[id_col]]
    return np.vstack(rows)


class EmbeddingReducer:
    """Leakage-safe PCA: fit on training rows only, then transform any split."""

    def __init__(self, n_components: int = 16, random_state: int = 7) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self._pca = None
        self._mean = None

    def fit(self, matrix: np.ndarray) -> "EmbeddingReducer":
        from sklearn.decomposition import PCA

        clean = matrix[~np.isnan(matrix).any(axis=1)]
        n_components = min(self.n_components, clean.shape[1], max(1, clean.shape[0] - 1))
        self._mean = clean.mean(axis=0)
        self._pca = PCA(n_components=n_components, random_state=self.random_state).fit(clean)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("EmbeddingReducer must be fit before transform")
        filled = np.where(np.isnan(matrix), self._mean, matrix)
        return self._pca.transform(filled)

    def feature_names(self) -> list[str]:
        if self._pca is None:
            raise RuntimeError("EmbeddingReducer must be fit before naming features")
        return [f"emb_pc{i}" for i in range(self._pca.n_components_)]
