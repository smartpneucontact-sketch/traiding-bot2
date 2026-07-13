"""Candidate-set construction for the ML re-ranker.

Candidate set per decision date = union of
  - top-CAND_POOL names by 12-1 momentum (px[t-21]/px[t-252] - 1), and
  - top-CAND_POOL names by 6m total return (126d), POSITIVE only,
among names with >= 273 valid closes (cumulative) and a valid close today.

Decision grids:
  - TRAINING rows: every TRAIN_GRID (5) trading days,
  - PORTFOLIO decisions: every REBAL (21) trading days — anchored on the
    canonical sleeve grid (dual_momentum sleeve decision dates) so the I2
    blend lines up exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_track.config import CAND_POOL, MAX_LOOKBACK_BARS, TRAIN_GRID


def decision_grids(close_index: pd.Index, sleeve_dates: pd.Index,
                   dev_end: str, grid: int = TRAIN_GRID
                   ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """(training_dates, portfolio_dates), both <= dev_end and with enough
    history (>= MAX_LOOKBACK_BARS bars before them)."""
    cutoff = pd.Timestamp(dev_end)
    idx = close_index
    train_dates = pd.DatetimeIndex(
        [idx[i] for i in range(MAX_LOOKBACK_BARS, len(idx))
         if i % grid == 0 and idx[i] <= cutoff])
    port_dates = pd.DatetimeIndex(
        [d for d in sleeve_dates
         if d <= cutoff and d in idx and idx.get_loc(d) >= MAX_LOOKBACK_BARS])
    return train_dates, port_dates


def candidate_mask(close: pd.DataFrame, dates: pd.DatetimeIndex,
                   pool: int = CAND_POOL) -> pd.DataFrame:
    """Boolean frame (dates x symbols): True where symbol is a candidate."""
    valid = close.notna()
    cum_valid = valid.cumsum()
    mom_12_1 = close.shift(21) / close.shift(252) - 1.0
    ret_6m = close.pct_change(126, fill_method=None)

    rows = {}
    for dt in dates:
        eligible = valid.loc[dt] & (cum_valid.loc[dt] >= MAX_LOOKBACK_BARS)
        m = mom_12_1.loc[dt].where(eligible)
        r6 = ret_6m.loc[dt].where(eligible)
        top_mom = m.dropna().nlargest(pool).index
        r6_pos = r6[r6 > 0].dropna()
        top_6m = r6_pos.nlargest(pool).index
        sel = pd.Series(False, index=close.columns)
        sel.loc[top_mom.union(top_6m)] = True
        rows[dt] = sel
    return pd.DataFrame(rows).T
