"""Run the risk-managed ensemble strategies and append to results."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd

from backtest import BTConfig, run_backtest
from ensemble import (
    ensemble_top3,
    fast_dd_momentum,
    fast_dd_voltarget_momentum,
    regime_gated_momentum,
    regime_voltarget_momentum,
    vol_target_momentum,
)
from metrics import fmt_summary
from strategies import (
    align_to_stock_cols,
    buy_hold_spy,
    union_prices,
    xs_momentum,
)

CACHE = Path(__file__).parent / "data_cache.pkl"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    panel = cache["panel"]
    macro = cache["macro"]
    px = panel["close"]
    pricing_universe = union_prices(px, macro)
    cfg = BTConfig(tc_bps=5.0, leverage_cap=1.0, allow_shorts=False)

    print(f"stocks: {px.shape[1]} | dates: {px.shape[0]}\n")

    candidates = [
        ("regime_gated_momentum",    lambda: regime_gated_momentum(px, macro)),
        ("voltarget_momentum",       lambda: vol_target_momentum(px, macro)),
        ("regime_voltarget_momentum",lambda: regime_voltarget_momentum(px, macro)),
        ("fast_dd_momentum",         lambda: fast_dd_momentum(px, macro)),
        ("fast_dd_voltarget_momentum",lambda: fast_dd_voltarget_momentum(px, macro)),
        ("ensemble_top3",            lambda: ensemble_top3(px, macro, pricing_universe)),
    ]

    results = {}
    equity_curves: dict[str, pd.Series] = {}
    # Re-load equity curves from prior runs if any
    prior_eq = OUT_DIR / "equity_curves.parquet"
    if prior_eq.exists():
        prev = pd.read_parquet(prior_eq)
        for c in prev.columns:
            equity_curves[c] = prev[c]
        print(f"loaded {len(equity_curves)} prior equity curves")
    prior_res = OUT_DIR / "results.json"
    if prior_res.exists():
        with open(prior_res) as f:
            results = json.load(f)
        print(f"loaded {len(results)} prior summaries\n")

    for name, fn in candidates:
        print(f"→ {name}")
        t0 = time.time()
        try:
            w = fn()
        except Exception as e:
            print(f"   FAILED build: {e}")
            continue
        print(f"   weights built in {time.time()-t0:.1f}s  shape={w.shape}")
        try:
            bt = run_backtest(w, pricing_universe, cfg, name=name)
        except Exception as e:
            print(f"   FAILED backtest: {e}")
            continue
        print(f"   {fmt_summary(bt['summary'])}")
        results[name] = bt["summary"]
        equity_curves[name] = bt["equity"]
        print()

    # Save
    eq_df = pd.DataFrame(equity_curves)
    eq_df.to_parquet(OUT_DIR / "equity_curves.parquet")
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print ordered summary
    print("\n══════ ALL RESULTS — ranked by Sharpe ══════")
    df = pd.DataFrame(results).T.sort_values("sharpe", ascending=False)
    cols = ["cagr", "mean_monthly", "ann_vol", "sharpe",
            "max_drawdown", "calmar", "hit_rate_monthly",
            "worst_month", "avg_n_positions", "turnover_annualized"]
    print(df[cols].round(4).to_string())


if __name__ == "__main__":
    main()
