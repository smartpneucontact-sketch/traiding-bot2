"""Per-year breakdown + drawdown timeline + correlation matrix.

Identifies which strategies are uncorrelated enough to combine, and which
years each strategy struggles in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import max_drawdown, sharpe

OUT = Path(__file__).parent / "results"


def main() -> None:
    eq = pd.read_parquet(OUT / "equity_curves.parquet")
    eq.index = pd.to_datetime(eq.index)

    # ── 1. Year-by-year monthly returns table ────────────────────────────
    daily = eq.pct_change().dropna(how="all")
    yearly_ret = (1 + daily).resample("YE").apply(lambda s: s.prod() - 1)

    # Pick top strategies by Sharpe over full window
    sharpe_rank = pd.Series({c: sharpe(eq[c].dropna()) for c in eq.columns}).sort_values(ascending=False)
    top_strats = sharpe_rank.head(10).index.tolist()
    print("──── TOP 10 BY SHARPE ────")
    print(sharpe_rank.head(10).round(2).to_string())
    print()
    print("──── YEARLY RETURNS (%) — TOP 10 STRATEGIES ────")
    print((yearly_ret[top_strats] * 100).round(1).to_string())
    print()

    # ── 2. Yearly Sharpe for each top strategy ───────────────────────────
    print("──── YEARLY SHARPE — TOP 10 STRATEGIES ────")
    yearly_sharpe = pd.DataFrame(index=yearly_ret.index, columns=top_strats, dtype=float)
    for year_end in yearly_ret.index:
        start = year_end - pd.offsets.YearBegin(1)
        for c in top_strats:
            sub = eq[c].loc[start:year_end].dropna()
            if len(sub) > 30:
                yearly_sharpe.loc[year_end, c] = sharpe(sub)
    print(yearly_sharpe.round(2).to_string())
    print()

    # ── 3. Correlation between daily returns ─────────────────────────────
    print("──── DAILY-RETURN CORRELATION — TOP 10 ────")
    corr = daily[top_strats].corr()
    print(corr.round(2).to_string())
    print()

    # ── 4. Max drawdown windows ──────────────────────────────────────────
    print("──── PEAK-TO-TROUGH DRAWDOWN DATES — TOP STRATEGIES ────")
    rows = []
    for c in top_strats:
        s = eq[c].dropna()
        peak = s.cummax()
        dd = s / peak - 1
        trough = dd.idxmin()
        peak_dt = s.loc[:trough].idxmax()
        recover = s.loc[trough:].ge(s.loc[peak_dt])
        recover_dt = recover[recover].index[0] if recover.any() else "ongoing"
        rows.append({
            "strategy": c,
            "max_dd": float(dd.min()),
            "peak_date": str(peak_dt.date()),
            "trough_date": str(trough.date()),
            "recover_date": str(recover_dt) if recover_dt != "ongoing" else "ongoing",
            "dd_days": (trough - peak_dt).days,
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
