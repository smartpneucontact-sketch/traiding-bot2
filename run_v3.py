"""Wave 3: time-series multi-asset momentum, fast 3-1 weekly momentum, and
a 4-strategy combined system that diversifies across signal types.

Aim: find a path to a clean 3%/mo with Sharpe > 1.1 by COMBINING uncorrelated
signals — not by squeezing more juice out of a single signal.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BTConfig, run_backtest
from ensemble import fast_drawdown_gate, vol_target_overlay
from metrics import fmt_summary
from strategies import (
    dual_momentum_voltarget,
    ts_momentum_multiasset,
    union_prices,
    xs_momentum,
    xs_momentum_fast,
)

CACHE = Path(__file__).parent / "data_cache.pkl"
OUT = Path(__file__).parent / "results"


def project_macro_weights(w_macro: pd.DataFrame, full_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=w_macro.index, columns=full_cols)
    for c in w_macro.columns:
        if c in full_cols:
            out[c] = w_macro[c].values
    return out


def project_stock_weights(w_stock: pd.DataFrame, full_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=w_stock.index, columns=full_cols)
    common = [c for c in w_stock.columns if c in full_cols]
    out[common] = w_stock[common].values
    return out


def main() -> None:
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    panel = cache["panel"]
    macro = cache["macro"]
    px = panel["close"]
    vol = panel.get("volume")
    pricing_universe = union_prices(px, macro)
    cols = list(pricing_universe.columns)

    cfg = BTConfig(tc_bps=5.0, leverage_cap=1.5, allow_shorts=False)
    cfg_lev = BTConfig(tc_bps=5.0, leverage_cap=2.5, allow_shorts=False)

    # Build a 4-way diversified combo
    def combo_4way(lev: float = 1.0):
        idx = pricing_universe.index
        w_mom = xs_momentum(px, macro, n_long=30).reindex(idx).ffill().fillna(0.0)
        w_fast = xs_momentum_fast(px, macro, volume=vol, n_long=20).reindex(idx).ffill().fillna(0.0)
        w_dm = dual_momentum_voltarget(px, macro, n_long=30).reindex(idx).ffill().fillna(0.0)
        w_ts = ts_momentum_multiasset(macro).reindex(idx).ffill().fillna(0.0)

        full = (
            0.30 * project_stock_weights(w_mom, cols)
            + 0.25 * project_stock_weights(w_fast, cols)
            + 0.20 * project_stock_weights(w_dm, cols)
            + 0.25 * project_macro_weights(w_ts, cols)
        )
        full = fast_drawdown_gate(full, macro, lookback=60, full_dd=0.08, cash_dd=0.18)
        full = vol_target_overlay(full, pricing_universe, target_vol=0.22,
                                  max_leverage=1.0, min_leverage=0.4, lookback=60)
        full = full * lev
        return full.iloc[::5]  # weekly trade

    candidates = [
        ("ts_momentum_multiasset",   lambda: project_macro_weights(ts_momentum_multiasset(macro), cols), cfg),
        ("xs_momentum_fast",         lambda: xs_momentum_fast(px, macro, volume=vol, n_long=20), cfg),
        ("xs_momentum_fast_top10",   lambda: xs_momentum_fast(px, macro, volume=vol, n_long=10), cfg),
        ("combo_4way_1x",            lambda: combo_4way(1.0), cfg),
        ("combo_4way_1.5x",          lambda: combo_4way(1.5), cfg_lev),
        ("combo_4way_2x",            lambda: combo_4way(2.0), cfg_lev),
    ]

    with open(OUT / "results.json") as f:
        results = json.load(f)
    prev_eq = pd.read_parquet(OUT / "equity_curves.parquet")
    eq_curves = {c: prev_eq[c] for c in prev_eq.columns}

    for name, fn, c in candidates:
        print(f"→ {name}")
        t0 = time.time()
        w = fn()
        print(f"   built in {time.time()-t0:.1f}s  shape={w.shape}")
        bt = run_backtest(w, pricing_universe, c, name=name)
        print(f"   {fmt_summary(bt['summary'])}\n")
        results[name] = bt["summary"]
        eq_curves[name] = bt["equity"]

    eq_df = pd.DataFrame(eq_curves)
    eq_df.to_parquet(OUT / "equity_curves.parquet")
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    df = pd.DataFrame(results).T.sort_values("sharpe", ascending=False)
    cols_disp = ["cagr","mean_monthly","ann_vol","sharpe","max_drawdown",
                 "calmar","hit_rate_monthly","worst_month","avg_n_positions","turnover_annualized"]
    print("══════ FULL RANKING ══════")
    print(df[cols_disp].round(4).to_string())
    df[cols_disp].round(4).to_csv(OUT / "summary.csv")


if __name__ == "__main__":
    main()
