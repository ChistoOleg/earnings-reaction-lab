from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erl.text.embed import FakeEncoder, chunk_text, embed_documents, embed_frame
from erl.text.features import EmbeddingReducer, embeddings_matrix, link_transcripts_to_events

RNG = np.random.default_rng(31)


def test_chunk_text_short_and_long():
    assert chunk_text("a b c", max_words=10) == ["a b c"]
    assert chunk_text("", max_words=10) == []
    words = " ".join(str(i) for i in range(1000))
    chunks = chunk_text(words, max_words=350, overlap=50)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 350 for c in chunks)


def test_fake_encoder_deterministic():
    enc = FakeEncoder(dim=8)
    a = enc.encode(["hello world"])
    b = enc.encode(["hello world"])
    assert a.shape == (1, 8)
    assert np.allclose(a, b)
    assert not np.allclose(enc.encode(["different"]), a)


def test_embed_documents_averages_chunks():
    enc = FakeEncoder(dim=12)
    long_doc = " ".join(str(i) for i in range(900))
    out = embed_documents(["short text", long_doc], enc, max_words=300, overlap=30)
    assert out.shape == (2, 12)


def test_embed_frame_roundtrip_and_resume(tmp_path):
    enc = FakeEncoder(dim=10)
    frame = pd.DataFrame(
        {"event_id": ["e1", "e2"], "content": ["alpha beta", "gamma delta"]}
    )
    out = tmp_path / "emb.parquet"
    first = embed_frame(frame, enc, out_path=out)
    assert len(first) == 2
    assert out.exists()
    calls_after_first = enc.calls

    bigger = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "content": ["alpha beta", "gamma delta", "epsilon zeta"],
        }
    )
    second = embed_frame(bigger, enc, out_path=out)
    assert len(second) == 3
    assert enc.calls == calls_after_first + 1
    assert set(second["event_id"]) == {"e1", "e2", "e3"}
    assert len(np.asarray(second.iloc[0]["embedding"])) == 10


def test_link_transcripts_to_events_within_tolerance():
    transcripts = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "call_date": pd.to_datetime(["2023-02-02", "2023-05-04", "2023-02-20"]),
            "content": ["x", "y", "z"],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["a1", "a2", "b1"],
            "ticker": ["AAA", "AAA", "BBB"],
            "announce_date": pd.to_datetime(["2023-02-01", "2023-05-03", "2023-01-01"]),
        }
    )
    linked = link_transcripts_to_events(transcripts, events, max_days=7)
    assert set(linked["event_id"]) == {"a1", "a2"}
    mapping = dict(zip(linked["call_date"].dt.strftime("%Y-%m-%d"), linked["event_id"]))
    assert mapping["2023-02-02"] == "a1"
    assert mapping["2023-05-04"] == "a2"


def test_embeddings_matrix_and_reducer():
    ids = [f"e{i}" for i in range(40)]
    vectors = list(RNG.normal(size=(40, 20)))
    embeddings = pd.DataFrame({"event_id": ids, "embedding": vectors})
    panel = pd.DataFrame({"event_id": ids + ["missing"]})
    matrix = embeddings_matrix(panel, embeddings)
    assert matrix.shape == (41, 20)
    assert np.isnan(matrix[-1]).all()

    train = matrix[:30]
    reducer = EmbeddingReducer(n_components=5).fit(train)
    reduced = reducer.transform(matrix[30:40])
    assert reduced.shape == (10, 5)
    assert len(reducer.feature_names()) == 5


def test_ft_transformer_runs_and_learns():
    pytest.importorskip("torch")
    from erl.predict.cv import PurgedWalkForwardCV
    from erl.predict.tabular_dl import train_ft_transformer

    n = 1500
    X = RNG.normal(size=(n, 5))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 1.0 * X[:, 0] * X[:, 1] + RNG.normal(scale=0.5, size=n)
    panel = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    panel["car_reaction"] = y
    panel["announce_date"] = pd.date_range("2015-01-05", "2024-12-20", periods=n).normalize()
    panel["event_id"] = [f"e{i}" for i in range(n)]
    panel["ticker"] = [f"T{i % 80}" for i in range(n)]

    result = train_ft_transformer(
        panel,
        "car_reaction",
        [f"f{i}" for i in range(5)],
        cv=PurgedWalkForwardCV(n_splits=2, purge_days=10),
        epochs=25,
        prefer_gpu=False,
    )
    assert np.isfinite(result.oos_metrics["r2"])
    assert result.oos_metrics["rank_ic"] > 0.3
    assert len(result.oos_predictions) == result.fold_metrics["n_test"].iloc[-1]
