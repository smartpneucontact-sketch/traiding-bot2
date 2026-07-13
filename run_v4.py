"""Wave 4: Kelly-fraction leveraged combos + push to 3%/mo via diversification.

Goal: identify a strategy that backtests to ≥3.5%/mo so that even after a 15%
live-friction haircut, the live target of 3%/mo is met.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd

from backtest import BTConfig, run_backtest
from ensemble import combo_4way_kelly
from metrics import fmt_summary
from strategies import (
    dual_momentum_voltarget,
    ts_momentum_multiasset,
    union_prices,
    xs_momentum,
    xs_momentum_fast,
)
from ensemble import fast_drawdown_gate, vol_target_overlay

CACHE = Path(__file__).parent / "data_cache.pkl"
OUT = Path(__file__).parent / "results"


def main() -> None:
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    panel = cache["panel"]
    macro = cache["macro"]
    px = panel["close"]
    vol = panel.get("volume")
    pricing_universe = union_prices(px, macro)
    cols = list(pricing_universe.columns)

    cfg_lev = BTConfig(tc_bps=5.0, leverage_cap=3.0, allow_shorts=False)

    def project_stock(w):
        out = pd.DataFrame(0.0, index=pricing_universe.index, columns=cols)
        w2 = w.reindex(pricing_universe.index).ffill().fillna(0.0)
        common = [c for c in w2.columns if c in cols]
        out[common] = w2[common].values
        return out

    def project_macro(w):
        out = pd.DataFrame(0.0, index=pricing_universe.index, columns=cols)
        w2 = w.reindex(pricing_universe.index).ffill().fillna(0.0)
        for c in w2.columns:
            if c in cols:
                out[c] = w2[c].values
        return out

    def combo_4way(lev: float):
        idx = pricing_universe.index
        w_mom = xs_momentum(px, macro, n_long=30).reindex(idx).ffill().fillna(0.0)
        w_fast = xs_momentum_fast(px, macro, volume=vol, n_long=20).reindex(idx).ffill().fillna(0.0)
        w_dm = dual_momentum_voltarget(px, macro, n_long=30).reindex(idx).ffill().fillna(0.0)
        w_ts = ts_momentum_multiasset(macro).reindex(idx).ffill().fillna(0.0)

        full = (
            0.30 * project_stock(w_mom)
            + 0.25 * project_stock(w_fast)
            + 0.20 * project_stock(w_dm)
            + 0.25 * project_macro(w_ts)
        )
        full = fast_drawdown_gate(full, macro, lookback=60, full_dd=0.08, cash_dd=0.18)
        full = vol_target_overlay(full, pricing_universe, target_vol=0.22,
                                  max_leverage=1.0, min_leverage=0.4, lookback=60)
        return (full * lev).iloc[::5]

    candidates = [
        ("combo_4way_2.5x",         lambda: combo_4way(2.5), cfg_lev),
        ("combo_4way_kelly_0.4",    lambda: combo_4way_kelly(px, macro, pricing_universe, kelly_fraction=0.4, max_leverage=2.5), cfg_lev),
        ("combo_4way_kelly_0.6",    lambda: combo_4way_kelly(px, macro, pricing_universe, kelly_fraction=0.6, max_leverage=3.0), cfg_lev),
        ("combo_4way_kelly_0.25",   lambda: combo_4way_kelly(px, macro, pricing_universe, kelly_fraction=0.25, max_leverage=2.0), cfg_lev),
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

    df = pd.DataFrame(results).T.sort_values("mean_monthly", ascending=False)
    cols_disp = ["cagr","mean_monthly","median_monthly","ann_vol","sharpe",
                 "max_drawdown","calmar","hit_rate_monthly","worst_month",
                 "best_month","avg_n_positions","turnover_annualized"]
    print("══════ TOP 10 BY MONTHLY RETURN ══════")
    print(df[cols_disp].head(10).round(4).to_string())
    df[cols_disp].round(4).to_csv(OUT / "summary.csv")


if __name__ == "__main__":
    main()
