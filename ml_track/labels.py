"""Labels for the ML re-ranker.

r_fwd21 measured from the EXECUTION day (corrected-engine convention):
decision at close t -> trade at open t+1 -> first full holding mark is close
t+1; horizon close is t+22. r_fwd21 = close[t+22]/close[t+1] - 1.

L1 = pctrank of r_fwd21 within the candidate set on date t.
L2 = pctrank of (r_fwd21 / (sigma60 * sqrt(21))) within the candidate set,
     sigma60 = trailing 60d daily-return std known at t.
L3 = 1{r_fwd21 > candidate-set median r_fwd21}.
L3v = 1{vol-scaled r_fwd21 > candidate-set median} (the L2-analog for B_cls).
Each row carries label_end_date = the calendar date at position t+22.

DISCLOSURE (2026-07 audit, dev/val boundary bleed): dev decision dates run
through DEV_END (2022-12-31) while each label consumes closes through t+22,
so late-2022 rows read prices as far as 2023-02-02 — inside the LOCKED
validation window. Model/candidate selection on dev metrics therefore saw
~1 month of validation-period prices via these labels. Purging labels at
DEV_END would change the pre-registered dev set, so the bleed is disclosed
rather than fixed; treat 2023 validation results as marginally contaminated
through 2023-02-02.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_track.config import HORIZON


def build_labels(close: pd.DataFrame, keep_idx: pd.MultiIndex) -> pd.DataFrame:
    """Long label frame aligned to (date, sym) candidate rows."""
    idx = close.index
    r_fwd = close.shift(-(HORIZON + 1)) / close.shift(-1) - 1.0
    sigma60 = close.pct_change(fill_method=None).rolling(60).std()

    end_pos = np.arange(len(idx)) + HORIZON + 1
    end_date = pd.Series(
        [idx[p] if p < len(idx) else pd.NaT for p in end_pos], index=idx)

    dates = keep_idx.get_level_values(0).unique()
    long = pd.DataFrame({
        "r_fwd21": r_fwd.loc[dates].stack(future_stack=True).reindex(keep_idx),
        "sigma60": sigma60.loc[dates].stack(future_stack=True).reindex(keep_idx),
    })
    long["r_fwd21_vadj"] = long["r_fwd21"] / (
        long["sigma60"] * np.sqrt(HORIZON) + 1e-8)

    grp = long.groupby(level=0)
    long["L1"] = grp["r_fwd21"].rank(pct=True)
    long["L2"] = grp["r_fwd21_vadj"].rank(pct=True)
    med = grp["r_fwd21"].transform("median")
    med_v = grp["r_fwd21_vadj"].transform("median")
    long["L3"] = (long["r_fwd21"] > med).astype(float)
    long["L3v"] = (long["r_fwd21_vadj"] > med_v).astype(float)
    long.loc[long["r_fwd21"].isna(), ["L1", "L3"]] = np.nan
    long.loc[long["r_fwd21_vadj"].isna(), ["L2", "L3v"]] = np.nan

    date_lvl = long.index.get_level_values(0)
    long["label_end_date"] = end_date.reindex(date_lvl).to_numpy()
    long.index.names = ["date", "sym"]
    return long
