from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from erl.utils import read_parquet, write_parquet

logger = logging.getLogger(__name__)


def chunk_text(text: str, max_words: int = 350, overlap: int = 50) -> list[str]:
    words = str(text).split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max(1, max_words - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + max_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + max_words >= len(words):
            break
    return chunks


@runtime_checkable
class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class FakeEncoder:
    """Deterministic, offline encoder for tests and dry runs."""

    def __init__(self, dim: int = 16, name: str = "fake-encoder") -> None:
        self.dim = dim
        self.name = name
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, text in enumerate(texts):
            seed = int(hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:8], 16)
            out[i] = np.random.default_rng(seed).normal(size=self.dim)
        return out


class TransformerEncoder:
    """FinBERT-class encoder. Lazily loads torch + transformers (GPU path)."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).type_as(output)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy())
        return np.vstack(vectors)


def embed_documents(
    texts: Sequence[str],
    encoder: Encoder,
    max_words: int = 350,
    overlap: int = 50,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for text in texts:
        chunks = chunk_text(text, max_words, overlap) or [""]
        chunk_vectors = encoder.encode(chunks)
        rows.append(chunk_vectors.mean(axis=0))
    return np.vstack(rows) if rows else np.empty((0, encoder.dim))


def embed_frame(
    frame: pd.DataFrame,
    encoder: Encoder,
    id_col: str = "event_id",
    text_col: str = "content",
    out_path=None,
    max_words: int = 350,
    overlap: int = 50,
) -> pd.DataFrame:
    done: set = set()
    existing: pd.DataFrame | None = None
    if out_path is not None and Path(out_path).exists():
        existing = read_parquet(out_path)
        done = set(existing[id_col].tolist())

    todo = frame[~frame[id_col].isin(done)].reset_index(drop=True)
    if todo.empty:
        logger.info("embed_frame: nothing new to embed (%d cached)", len(done))
        return existing if existing is not None else pd.DataFrame()

    vectors = embed_documents(todo[text_col].tolist(), encoder, max_words, overlap)
    fresh = pd.DataFrame(
        {
            id_col: todo[id_col].to_numpy(),
            "model": encoder.name,
            "dim": encoder.dim,
        }
    )
    fresh["embedding"] = list(vectors)
    combined = pd.concat([existing, fresh], ignore_index=True) if existing is not None else fresh
    if out_path is not None:
        write_parquet(combined, out_path)
    logger.info("embed_frame: embedded %d new documents (%d total)", len(todo), len(combined))
    return combined
