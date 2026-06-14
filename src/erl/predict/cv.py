from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PurgedWalkForwardCV:
    n_splits: int = 4
    purge_days: int = 35
    embargo_days: int = 5

    def split(self, dates) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        stamps = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        order = np.argsort(stamps.to_numpy(), kind="stable")
        n = len(stamps)
        if self.n_splits < 1 or n < (self.n_splits + 1) * 2:
            raise ValueError("not enough observations for the requested number of splits")
        boundaries = np.linspace(0, n, self.n_splits + 2, dtype=int)

        for fold in range(1, self.n_splits + 1):
            test_positions = order[boundaries[fold] : boundaries[fold + 1]]
            test_dates = stamps.iloc[test_positions]
            test_start = test_dates.min()
            test_end = test_dates.max()
            purge_cutoff = test_start - pd.Timedelta(days=self.purge_days)
            embargo_end = test_end + pd.Timedelta(days=self.embargo_days)

            before = stamps < purge_cutoff
            inside_or_embargoed = (stamps >= test_start) & (stamps <= embargo_end)
            train_mask = before & ~inside_or_embargoed
            train_positions = np.flatnonzero(train_mask.to_numpy())
            if len(train_positions) == 0:
                continue
            logger.debug(
                "fold %d: train=%d test=%d test_start=%s purge_cutoff=%s",
                fold,
                len(train_positions),
                len(test_positions),
                test_start.date(),
                purge_cutoff.date(),
            )
            yield train_positions, np.sort(test_positions)
