"""Run the FINAL candidate strategies and the leveraged variant.

Also produces a `summary.csv` with all strategies sorted by Sharpe.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd

from backtest import BTConfig, run_backtest
from ensemble import final_3pct_leveraged, final_3pct_target
from metrics import fmt_summary
from strategies import union_prices

CACHE = Path(__file__).parent / "data_cache.pkl"
OUT = Path(__file__).parent / "results"


def main() -> None:
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    panel = cache["panel"]
    macro = cache["macro"]
    px = panel["close"]
    pricing_universe = union_prices(px, macro)

    cfg = BTConfig(tc_bps=5.0, leverage_cap=1.5, allow_shorts=False)
    cfg_lev = BTConfig(tc_bps=5.0, leverage_cap=2.0, allow_shorts=False)

    with open(OUT / "results.json") as f:
        results = json.load(f)
    prev_eq = pd.read_parquet(OUT / "equity_curves.parquet")
    eq_curves = {c: prev_eq[c] for c in prev_eq.columns}

    candidates = [
        ("FINAL_3pct_target",     lambda: final_3pct_target(px, macro, pricing_universe), cfg),
        ("FINAL_3pct_lev_1.3x",   lambda: final_3pct_leveraged(px, macro, pricing_universe, 1.3), cfg_lev),
        ("FINAL_3pct_lev_1.5x",   lambda: final_3pct_leveraged(px, macro, pricing_universe, 1.5), cfg_lev),
    ]

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
    df[cols].round(4).to_csv(OUT / "summary.csv")
    print("══════ FULL SUMMARY ══════")
    print(df[cols].round(4).to_string())
    print(f"\nsaved → {OUT/'summary.csv'}")


if __name__ == "__main__":
    main()
