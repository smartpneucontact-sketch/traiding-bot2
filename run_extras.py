"""Run the L/S momentum + clean ML strategies."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd

from backtest import BTConfig, run_backtest
from metrics import fmt_summary
from strategies import ml_cross_sectional, union_prices, xs_momentum_ls

CACHE = Path(__file__).parent / "data_cache.pkl"
OUT_DIR = Path(__file__).parent / "results"


def main() -> None:
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    panel = cache["panel"]
    macro = cache["macro"]
    px = panel["close"]
    vol = panel.get("volume")
    pricing_universe = union_prices(px, macro)

    # Load existing
    with open(OUT_DIR / "results.json") as f:
        results = json.load(f)
    prev_eq = pd.read_parquet(OUT_DIR / "equity_curves.parquet")
    eq_curves = {c: prev_eq[c] for c in prev_eq.columns}

    cfg_ls = BTConfig(tc_bps=7.0, leverage_cap=1.5, allow_shorts=True, borrow_bps_annual=80.0)
    cfg_long = BTConfig(tc_bps=5.0, leverage_cap=1.0, allow_shorts=False)

    candidates = [
        ("xs_momentum_ls",        lambda: xs_momentum_ls(px, macro, n_each_side=50, long_weight=1.0, short_weight=0.5), cfg_ls),
        ("xs_momentum_ls_neutral",lambda: xs_momentum_ls(px, macro, n_each_side=50, long_weight=1.0, short_weight=1.0), cfg_ls),
        ("ml_lgb_xs",             lambda: ml_cross_sectional(px, macro, volume=vol, n_long=30), cfg_long),
    ]

    for name, fn, cfg in candidates:
        print(f"→ {name}")
        t0 = time.time()
        try:
            w = fn()
        except Exception as e:
            print(f"   FAILED build: {e}")
            continue
        print(f"   weights built in {time.time()-t0:.1f}s  shape={w.shape}")
        bt = run_backtest(w, pricing_universe, cfg, name=name)
        print(f"   {fmt_summary(bt['summary'])}")
        results[name] = bt["summary"]
        eq_curves[name] = bt["equity"]
        print()

    eq_df = pd.DataFrame(eq_curves)
    eq_df.to_parquet(OUT_DIR / "equity_curves.parquet")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n══════ ALL RESULTS ranked by Sharpe ══════")
    df = pd.DataFrame(results).T.sort_values("sharpe", ascending=False)
    cols = ["cagr","mean_monthly","ann_vol","sharpe","max_drawdown",
            "calmar","hit_rate_monthly","worst_month","avg_n_positions","turnover_annualized"]
    print(df[cols].round(4).to_string())


if __name__ == "__main__":
    main()
