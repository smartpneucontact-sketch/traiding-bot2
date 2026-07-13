"""Run all strategies, collect equity curves, and save a summary table.

Usage:
    python run_all.py                    # full 10-yr backtest, all strategies
    python run_all.py --skip ml         # skip the slow LightGBM strategy
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BTConfig, run_backtest
from metrics import fmt_summary
from strategies import (
    align_to_stock_cols,
    buy_hold_spy,
    dual_momentum_voltarget,
    market_regime,
    mean_reversion,
    ml_cross_sectional,
    sector_rotation,
    trend_following,
    union_prices,
    xs_momentum,
)

CACHE = Path(__file__).parent / "data_cache.pkl"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def load() -> dict:
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def run_one(name: str, weights: pd.DataFrame, prices: pd.DataFrame, cfg: BTConfig) -> dict:
    t0 = time.time()
    bt = run_backtest(weights, prices, cfg, name=name)
    t1 = time.time()
    print(f"  {fmt_summary(bt['summary'])} | runtime {t1-t0:5.1f}s")
    return bt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="Strategy names to skip")
    ap.add_argument("--only", nargs="*", default=None, help="Run only these names")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    print("loading cached panel...")
    cache = load()
    panel = cache["panel"]
    macro = cache["macro"]

    px = panel["close"]
    vol = panel.get("volume")
    pricing_universe = union_prices(px, macro)

    print(f"  stocks: {px.shape[1]} | dates: {px.shape[0]} | range: {px.index[0].date()} → {px.index[-1].date()}")
    print()

    cfg = BTConfig(tc_bps=5.0, leverage_cap=1.0, allow_shorts=False)

    candidates = [
        ("buy_hold_spy",       lambda: align_to_stock_cols(buy_hold_spy(px, macro), pricing_universe.columns, [])),
        ("xs_momentum_12_1",   lambda: xs_momentum(px, macro, n_long=50)),
        ("xs_momentum_top30",  lambda: xs_momentum(px, macro, n_long=30)),
        ("mean_reversion_5d",  lambda: mean_reversion(px, macro, volume=vol, n_long=30)),
        ("trend_following",    lambda: trend_following(px, macro, n_long=30)),
        ("dual_momentum_vol",  lambda: dual_momentum_voltarget(px, macro, n_long=30)),
        ("sector_rotation",    lambda: align_to_stock_cols(sector_rotation(px, macro), pricing_universe.columns, [])),
        ("ml_lgb_xs",          lambda: ml_cross_sectional(px, macro, volume=vol, n_long=30)),
    ]
    if args.only:
        candidates = [(n, fn) for (n, fn) in candidates if n in args.only]
    for skip in args.skip:
        candidates = [(n, fn) for (n, fn) in candidates if skip not in n]

    print(f"running {len(candidates)} strategies...\n")

    results: dict[str, dict] = {}
    equity_curves: dict[str, pd.Series] = {}
    for name, fn in candidates:
        print(f"→ {name}")
        t0 = time.time()
        try:
            w = fn()
        except Exception as e:
            print(f"   FAILED building weights: {e}")
            continue
        print(f"   weights built in {time.time()-t0:.1f}s  shape={w.shape}")
        bt = run_one(name, w, pricing_universe, cfg)
        results[name] = bt["summary"]
        equity_curves[name] = bt["equity"]
        print()

    # Persist equity curves + summary
    eq_df = pd.DataFrame(equity_curves)
    eq_df.to_parquet(OUT_DIR / "equity_curves.parquet")
    with open(OUT_DIR / args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"saved → {OUT_DIR/'equity_curves.parquet'} and {OUT_DIR/args.out}")

    # Pretty summary
    print("\n══════ FINAL SUMMARY ══════")
    df = pd.DataFrame(results).T.sort_values("mean_monthly", ascending=False)
    cols = ["cagr", "mean_monthly", "median_monthly", "ann_vol",
            "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate_monthly",
            "worst_month", "best_month", "avg_n_positions", "turnover_annualized"]
    print(df[cols].round(4).to_string())


if __name__ == "__main__":
    main()
