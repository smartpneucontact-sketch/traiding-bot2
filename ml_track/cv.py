"""PurgedAnchoredWF — anchored walk-forward with purge + embargo.

Folds test on calendar years 2018..2022 (decision dates in-year).
Train = all rows with decision_date < test_start,
  MINUS rows with label_end_date >= test_start (purge),
  MINUS rows with decision_date in the EMBARGO (21) trading days immediately
  before test_start.
Early-stopping eval set = rows whose decision_date is in the last 15% of
unique training DATES (date-sliced; never split by row position). Since the
2026-07 audit the same label-end purge is applied at the core/ES boundary
(core rows whose 21d label window reaches into the ES slice are dropped) —
before that, boundary-spanning labels leaked into early stopping.

HONESTY NOTE (2026-07 audit): the pre-registered EMBARGO is a no-op here.
Any row with decision_date within 21 trading days of test_start has
label_end_date (t+22) >= test_start and is already removed by the purge, so
the embargo mask removes nothing. Removing (or widening) it would change the
pre-registered fold boundaries, so it is kept as registered and disclosed.
"""
from __future__ import annotations

import pandas as pd

from ml_track.config import EMBARGO, TEST_YEARS


class PurgedAnchoredWF:
    def __init__(self, trading_index: pd.Index,
                 test_years: list[int] = TEST_YEARS,
                 embargo: int = EMBARGO):
        self.trading_index = pd.DatetimeIndex(trading_index)
        self.test_years = test_years
        self.embargo = embargo

    def folds(self, dates: pd.Series, label_end: pd.Series):
        """Yield (year, core_mask, es_mask, test_mask) boolean masks over the
        row frame. `dates` = decision_date per row, `label_end` per row."""
        for year in self.test_years:
            test_start = pd.Timestamp(f"{year}-01-01")
            test_end = pd.Timestamp(f"{year}-12-31")

            # embargo window: last `embargo` trading days strictly before
            pre = self.trading_index[self.trading_index < test_start]
            emb_start = pre[-self.embargo] if len(pre) >= self.embargo else pre[0]

            train = (dates < test_start)
            train &= ~(label_end.notna() & (label_end >= test_start))  # purge
            # embargo: a no-op given the purge above (see module docstring);
            # kept because it is pre-registered.
            train &= ~((dates >= emb_start) & (dates < test_start))
            test = (dates >= test_start) & (dates <= test_end)

            tr_dates = sorted(dates[train].unique())
            n_es = max(1, int(round(len(tr_dates) * 0.15)))
            es_cut = tr_dates[-n_es]
            es = train & (dates >= es_cut)
            # core/ES boundary purge (2026-07 audit): drop core rows whose
            # label window reaches into the ES slice, else ES labels leak
            # into training via early stopping.
            core = (train & (dates < es_cut)
                    & ~(label_end.notna() & (label_end >= es_cut)))
            yield year, core, es, test
