from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def event_id(ticker: str, announce_date: str) -> str:
    basis = f"{ticker.strip().upper()}|{announce_date}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
