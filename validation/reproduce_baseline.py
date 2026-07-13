"""Fidelity check: reproduce the published combo_v2_2x baseline from data_cache.pkl
using the SAME sleeve functions and engine (imported from strategies.py / backtest.py,
mirroring run_v6._build_combo_v2_baseline). Must match results_v6.json
combo_v2_2x_baseline within ~0.2pp/mo and Sharpe ~0.05, else STOP.

Saves the post-processed daily levered target weights (the exact `target` frame the
engine trades) + the reproduced equity to validation/ for the stop-grid overlay.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # ".../Traiding 11"
sys.path.insert(0, str(ROOT))

from backtest import BTConfig, run_backtest  # noqa: E402
from metrics import fmt_summary  # noqa: E402
from strategies import (  # noqa: E402
    adaptive_voltarget_momentum,
    dual_momentum_voltarget,
    union_prices,
    xs_momentum,
)

OUT = ROOT / "validation"
OUT.mkdir(exist_ok=True)


def build_combo_v2_base(px, macro):
    """Byte-identical to run_v6._build_combo_v2_baseline."""
    w_xs = xs_momentum(px, macro, n_long=30)
    w_dual = dual_momentum_voltarget(px, macro, n_long=30)
    w_adapt = adaptive_voltarget_momentum(px, macro, n_long=30)
    dates = sorted(set(w_xs.index) | set(w_dual.index) | set(w_adapt.index))
    cols = sorted(set(w_xs.columns) | set(w_dual.columns) | set(w_adapt.columns))
    w_xs = w_xs.reindex(index=dates, columns=cols, fill_value=0.0)
    w_dual = w_dual.reindex(index=dates, columns=cols, fill_value=0.0)
    w_adapt = w_adapt.reindex(index=dates, columns=cols, fill_value=0.0)
    return (w_xs + w_dual + w_adapt) / 3.0


def main():
    with open(ROOT / "data_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    panel, macro = cache["panel"], cache["macro"]
    px = panel["close"]
    pu = union_prices(px, macro)

    print("building combo_v2 base weights (3 sleeves, n_long=30, rebal 21d)...")
    base = build_combo_v2_base(px, macro)
    print(f"  decision dates: {len(base.index)} | first {base.index[0].date()} last {base.index[-1].date()}")

    cfg = BTConfig(tc_bps=5.0, leverage_cap=2.0, allow_shorts=False)
    bt = run_backtest(base * 2.0, pu, cfg, name="combo_v2_2x_repro")
    s = bt["summary"]
    print(fmt_summary(s))

    with open(ROOT / "results" / "results_v6.json") as f:
        pub = json.load(f)["combo_v2_2x_baseline"]

    checks = {
        "mean_monthly": (s["mean_monthly"], pub["mean_monthly"], 0.002),
        "sharpe": (s["sharpe"], pub["sharpe"], 0.05),
        "max_drawdown": (s["max_drawdown"], pub["max_drawdown"], 0.02),
    }
    ok = True
    for k, (mine, theirs, tol) in checks.items():
        d = abs(mine - theirs)
        flag = "OK " if d <= tol else "FAIL"
        ok &= d <= tol
        print(f"  {flag} {k:14s} repro={mine:+.6f} published={theirs:+.6f} diff={d:.6f} tol={tol}")

    if not ok:
        print("FIDELITY CHECK FAILED — do not run the grid.")
        sys.exit(1)

    # Persist artifacts for the stop grid: exact traded daily target (levered,
    # capped, shifted) and reproduced equity / daily net returns.
    target = bt["weights"]            # daily levered target, post shift/clip/cap
    target.to_parquet(OUT / "target_weights_daily.parquet")
    bt["equity"].rename("equity").to_frame().to_parquet(OUT / "repro_equity.parquet")
    bt["returns"].rename("ret").to_frame().to_parquet(OUT / "repro_daily_net.parquet")
    # decision dates (signal dates) for the period structure
    pd.Series(base.index, name="decision_date").to_frame().to_parquet(OUT / "decision_dates.parquet")
    with open(OUT / "repro_summary.json", "w") as f:
        json.dump({k: (v if not isinstance(v, float) else float(v)) for k, v in s.items()}, f, indent=2, default=str)
    print(f"FIDELITY CHECK PASSED — artifacts written to {OUT}")


if __name__ == "__main__":
    main()
