"""run_v5 — back-test the 10 NEW alternative strategy classes added 2026-06-04.

Runs each over the cached 10-year window, same TC (5 bp/side), and writes
the results into results/summary_v5.csv + results.json (appended).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import pandas as pd

from backtest import BTConfig, run_backtest
from metrics import fmt_summary
from strategies import (
    acceleration_momentum,
    adaptive_voltarget_momentum,
    align_to_stock_cols,
    calendar_tom,
    donchian_breakout,
    low_vol_quality,
    momentum_quality_blend,
    multifactor_mvr,
    risk_parity_etf,
    sector_momentum_rotation,
    union_prices,
    vix_gated_trend,
)

CACHE = Path(__file__).parent / "data_cache.pkl"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("loading cached panel...")
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    panel = cache["panel"]
    macro = cache["macro"]

    px = panel["close"]
    pricing_universe = union_prices(px, macro)
    print(f"  stocks: {px.shape[1]} | dates: {px.shape[0]} | range: {px.index[0].date()} → {px.index[-1].date()}")
    print(f"  macro:  {macro.shape[1]} columns")
    print()

    cfg = BTConfig(tc_bps=5.0, leverage_cap=2.0, allow_shorts=False)
    # leverage_cap=2.0 lets the adaptive_voltarget_momentum strategy use its
    # full 1.5x leverage and the risk_parity_etf's modest scaling. The 1.0x
    # strategies are unaffected since the cap is per-day gross.

    candidates = [
        ("low_vol_quality", lambda: low_vol_quality(px, macro, n_long=30)),
        ("acceleration_momentum", lambda: acceleration_momentum(px, macro, n_long=30)),
        ("donchian_breakout", lambda: donchian_breakout(px, macro, n_long=30)),
        ("multifactor_mvr", lambda: multifactor_mvr(px, macro, n_long=30)),
        ("risk_parity_etf", lambda: align_to_stock_cols(
            risk_parity_etf(macro, target_vol=0.12),
            pricing_universe.columns, [])),
        ("vix_gated_trend", lambda: vix_gated_trend(px, macro, n_long=30)),
        ("calendar_tom", lambda: calendar_tom(px, macro, n_long=30)),
        ("sector_momentum_rotation", lambda: align_to_stock_cols(
            sector_momentum_rotation(macro, n_sectors=3),
            pricing_universe.columns, [])),
        ("adaptive_voltarget_momentum", lambda: adaptive_voltarget_momentum(px, macro, n_long=30)),
        ("momentum_quality_blend", lambda: momentum_quality_blend(px, macro, n_long=30)),
    ]

    print(f"running {len(candidates)} new strategies...\n")
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
        try:
            bt = run_backtest(w, pricing_universe, cfg, name=name)
        except Exception as e:
            print(f"   FAILED backtest: {e}")
            continue
        print(f"   {fmt_summary(bt['summary'])}")
        results[name] = bt["summary"]
        equity_curves[name] = bt["equity"]
        print()

    if not results:
        print("No strategies completed.")
        return

    # Persist v5 results
    eq_df = pd.DataFrame(equity_curves)
    eq_df.to_parquet(OUT_DIR / "equity_curves_v5.parquet")
    with open(OUT_DIR / "results_v5.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    df_new = pd.DataFrame(results).T

    # Merge into the existing summary.csv so the comparison is full
    summary_path = OUT_DIR / "summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path, index_col=0)
        merged = pd.concat([existing.drop(index=df_new.index, errors="ignore"), df_new])
        merged = merged.sort_values("mean_monthly", ascending=False)
        merged.to_csv(summary_path)
        print(f"merged into {summary_path} ({len(merged)} strategies total)")
    else:
        df_new.sort_values("mean_monthly", ascending=False).to_csv(summary_path)

    # Final v5 ranking
    print("\n══════ V5 RESULTS (new strategies only) ══════")
    cols = ["cagr", "mean_monthly", "median_monthly", "ann_vol",
            "sharpe", "max_drawdown", "calmar", "hit_rate_monthly",
            "worst_month", "best_month", "avg_n_positions",
            "turnover_annualized"]
    print(df_new[cols].sort_values("mean_monthly", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
