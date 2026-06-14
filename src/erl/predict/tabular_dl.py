from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from erl.predict.cv import PurgedWalkForwardCV
from erl.predict.gbm import regression_metrics

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only where torch is absent
    _HAS_TORCH = False


if _HAS_TORCH:

    class FTTransformer(nn.Module):
        """Feature-tokenizer transformer for tabular regression.

        Each numerical feature x_j is embedded as a learnable token
        e_j = x_j * W_j + b_j; a [CLS] token is prepended; the CLS output
        feeds a small MLP regression head.
        """

        def __init__(
            self,
            n_features: int,
            d_token: int = 32,
            n_layers: int = 3,
            n_heads: int = 4,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
            self.bias = nn.Parameter(torch.zeros(n_features, d_token))
            self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=d_token * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Linear(d_token, d_token),
                nn.GELU(),
                nn.Linear(d_token, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            tokens = x.unsqueeze(-1) * self.weight + self.bias
            cls = self.cls.expand(x.shape[0], -1, -1)
            sequence = torch.cat([cls, tokens], dim=1)
            encoded = self.encoder(sequence)
            return self.head(encoded[:, 0]).squeeze(-1)


@dataclass
class DLResult:
    fold_metrics: pd.DataFrame
    oos_metrics: dict[str, float]
    oos_predictions: pd.DataFrame
    features: list[str]
    config: dict
    model: object = field(default=None, repr=False)


def select_device(prefer_gpu: bool = True) -> str:
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required: pip install -r requirements-gpu.txt")
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (other - mean) / std


def _train_network(model, X, y, *, epochs, lr, batch_size, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    n = len(X)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx]), y_t[idx])
            loss.backward()
            optimizer.step()
    return model


def _predict(model, X, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        return model(X_t).cpu().numpy()


def train_ft_transformer(
    panel: pd.DataFrame,
    target: str,
    features: list[str],
    date_col: str = "announce_date",
    cv: PurgedWalkForwardCV | None = None,
    d_token: int = 32,
    n_layers: int = 3,
    n_heads: int = 4,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 256,
    prefer_gpu: bool = True,
    random_state: int = 7,
) -> DLResult:
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required: pip install -r requirements-gpu.txt")
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = select_device(prefer_gpu)

    cv = cv or PurgedWalkForwardCV()
    frame = (
        panel.dropna(subset=[target, date_col, *features])
        .sort_values(date_col)
        .reset_index(drop=True)
    )
    X_all = frame[features].to_numpy(dtype=float)
    y_all = frame[target].to_numpy(dtype=float)
    folds = list(cv.split(frame[date_col]))
    if len(folds) < 2:
        raise ValueError("need at least 2 folds")

    config = {
        "d_token": d_token,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "device": device,
    }

    fold_rows = []
    final_model = None
    oos_predictions = pd.DataFrame()
    oos_metrics: dict[str, float] = {}
    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        X_tr, X_te = _standardize(X_all[train_idx], X_all[test_idx])
        model = FTTransformer(len(features), d_token, n_layers, n_heads)
        model.to(device)
        _train_network(
            model, X_tr, y_all[train_idx],
            epochs=epochs, lr=lr, batch_size=batch_size, device=device,
        )
        pred = _predict(model, X_te, device)
        metrics = regression_metrics(y_all[test_idx], pred)
        is_final = i == len(folds)
        fold_rows.append(
            {
                "fold": i,
                "role": "oos_final" if is_final else "tuning",
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                **metrics,
            }
        )
        if is_final:
            final_model = model
            oos_metrics = metrics
            block = frame.iloc[test_idx]
            cols = [c for c in ("event_id", "ticker") if c in frame.columns]
            oos_predictions = block[[date_col, *cols]].copy()
            oos_predictions["y_true"] = y_all[test_idx]
            oos_predictions["y_pred"] = pred
            oos_predictions = oos_predictions.reset_index(drop=True)

    logger.info(
        "ft-transformer trained on %s: OOS r2=%.4f rank_ic=%.4f",
        device,
        oos_metrics.get("r2", float("nan")),
        oos_metrics.get("rank_ic", float("nan")),
    )
    return DLResult(
        fold_metrics=pd.DataFrame(fold_rows),
        oos_metrics=oos_metrics,
        oos_predictions=oos_predictions,
        features=features,
        config=config,
        model=final_model,
    )
