"""Second wave: multi-horizon momentum, concentrated top-15, and a leveraged
multi-horizon ensemble to try to *cleanly* clear 3%/mo with manageable DD."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd

from backtest import BTConfig, run_backtest
from ensemble import fast_drawdown_gate, vol_target_overlay
from metrics import fmt_summary
from strategies import (
    union_prices,
    xs_momentum,
    xs_momentum_concentrated,
    xs_momentum_multi,
)

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

    cfg = BTConfig(tc_bps=5.0, leverage_cap=1.0, allow_shorts=False)
    cfg_lev = BTConfig(tc_bps=5.0, leverage_cap=2.0, allow_shorts=False)

    def multi_lev(lev: float):
        w = xs_momentum_multi(px, macro, n_long=30)
        w = fast_drawdown_gate(w, macro, lookback=60, full_dd=0.08, cash_dd=0.18)
        return w * lev

    def conc_lev(lev: float):
        w = xs_momentum_concentrated(px, macro, volume=vol, n_long=15)
        w = fast_drawdown_gate(w, macro, lookback=60, full_dd=0.08, cash_dd=0.18)
        return w * lev

    candidates = [
        ("xs_momentum_multi",         lambda: xs_momentum_multi(px, macro, n_long=30), cfg),
        ("xs_momentum_concentrated",  lambda: xs_momentum_concentrated(px, macro, volume=vol, n_long=15), cfg),
        ("multi_dd_lev_1.3x",         lambda: multi_lev(1.3), cfg_lev),
        ("conc_dd_lev_1.3x",          lambda: conc_lev(1.3), cfg_lev),
        ("conc_dd_lev_1.5x",          lambda: conc_lev(1.5), cfg_lev),
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
    cols = ["cagr","mean_monthly","ann_vol","sharpe","max_drawdown",
            "calmar","hit_rate_monthly","worst_month","avg_n_positions","turnover_annualized"]
    print(df[cols].round(4).to_string())
    df[cols].round(4).to_csv(OUT / "summary.csv")


if __name__ == "__main__":
    main()
