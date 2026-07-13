"""Scores -> top-30 portfolio with hysteresis -> sparse 21d decision frame.

Hysteresis: incumbents stay while model rank <= BUF_OUT (45); new entries
need rank <= BUF_IN (25); book size capped at TOP_N (30). Weights equal or
inverse-vol60, normalized to sum = 1 (pre-leverage; callers blend/leverage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_track.config import BUF_IN, BUF_OUT, TOP_N


def build_decision_frame(scores: pd.DataFrame, columns: pd.Index,
                         weighting: str = "EW") -> pd.DataFrame:
    """scores: MultiIndex (date, sym) frame with cols score, sigma60 —
    portfolio decision dates only. Returns sparse frame (dates x columns)."""
    rows = {}
    held: list[str] = []
    for dt, g in scores.groupby(level=0):
        g = g.droplevel(0).dropna(subset=["score"])
        rank = g["score"].rank(ascending=False, method="first")

        keep = [s for s in held if s in rank.index and rank[s] <= BUF_OUT]
        entries = rank[(rank <= BUF_IN) & ~rank.index.isin(keep)]
        entries = entries.sort_values().index.tolist()
        book = keep + entries[: max(0, TOP_N - len(keep))]
        if not book:
            held = []
            rows[dt] = pd.Series(0.0, index=columns)
            continue

        if weighting == "IV":
            iv = 1.0 / g.loc[book, "sigma60"].replace(0.0, np.nan)
            iv = iv.fillna(iv.median() if np.isfinite(iv.median()) else 1.0)
            w = iv / iv.sum()
        else:
            w = pd.Series(1.0 / len(book), index=book)
        row = pd.Series(0.0, index=columns)
        row.loc[w.index] = w.values
        rows[dt] = row
        held = list(book)
    return pd.DataFrame(rows).T.sort_index()
