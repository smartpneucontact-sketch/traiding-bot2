"""V7 experiment harness: cache loading, sleeve caching, trial running and
ledger logging, with MECHANICAL enforcement of the dev/validation split.

Protocol (binding — see plans/V7):
- All selection happens on the dev window (≤ 2022-12-31). `run_trial`
  truncates every input at DEV_END unless window="val"/"full" is requested
  WITH final=True (reserved for the one-shot validation phase).
- Every run appends a row to results/v7/trials/{family}.csv — including
  discarded configs. The union of these files is the trial count for
  deflated Sharpe.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine_v2 import BTConfigV2, run_backtest_v2
from metrics_v2 import DEV_END, crash_table

ROOT = Path(__file__).resolve().parent
TRIALS_DIR = ROOT / "results" / "v7" / "trials"
WEIGHTS_STORE = ROOT / "weights_store"
TRIALS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_STORE.mkdir(exist_ok=True)

_CACHE = {}


def load_cache() -> tuple[dict, pd.DataFrame, dict]:
    """(panel dict, macro df, sector_map) — memoized."""
    if "cache" not in _CACHE:
        with open(ROOT / "data_cache.pkl", "rb") as f:
            _CACHE["cache"] = pickle.load(f)
    c = _CACHE["cache"]
    return c["panel"], c["macro"], c["sector_map"]


def union_prices_cached() -> pd.DataFrame:
    if "pu" not in _CACHE:
        from strategies import union_prices
        panel, macro, _ = load_cache()
        _CACHE["pu"] = union_prices(panel["close"], macro)
    return _CACHE["pu"]


def cached_weights(key: str, builder) -> pd.DataFrame:
    """Parquet-cache a (slow) sleeve/strategy weight frame by key."""
    p = WEIGHTS_STORE / f"{key}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    w = builder()
    w.to_parquet(p)
    return w


def run_trial(
    weights: pd.DataFrame,
    *,
    name: str,
    family: str,
    params: dict,
    window: str = "dev",
    exec_model: str = "next_open",
    tc_bps: float = 5.0,
    leverage_cap: float = 2.0,
    weights_are_daily: bool = False,
    final: bool = False,
    notes: str = "",
) -> dict:
    """Score a weight frame under the corrected engine and log the trial.

    `weights` are PRE-leverage decision weights ONLY if the caller already
    multiplied leverage in — this function does NOT apply leverage; pass
    post-leverage frames exactly as backtest.py callers do.
    """
    if window in ("val", "full") and not final:
        raise PermissionError(
            f"window={window!r} requires final=True — the validation window "
            f"is locked until the one-shot evaluation phase."
        )

    panel, macro, _ = load_cache()
    pu = union_prices_cached()
    opn = panel["open"]

    if window == "dev":
        cutoff = pd.Timestamp(DEV_END)
        pu_w = pu.loc[:cutoff]
        opn_w = opn.loc[:cutoff]
        w = weights.loc[:cutoff]
    elif window == "val":
        # Burn-in: weights may begin before VAL_START; equity is sliced after.
        pu_w, opn_w, w = pu, opn, weights
    else:
        pu_w, opn_w, w = pu, opn, weights

    cfg = BTConfigV2(tc_bps=tc_bps, leverage_cap=leverage_cap,
                     exec_model=exec_model)
    t0 = time.time()
    bt = run_backtest_v2(w, pu_w, cfg, name=name,
                         open_prices=opn_w if exec_model == "next_open" else None,
                         weights_are_daily=weights_are_daily)
    eq = bt["equity"]
    if window == "val":
        eq = eq.loc["2023-01-01":]
        eq = eq / eq.iloc[0] * 100_000.0
        from metrics import summary as msum
        s = msum(eq, name=name)
        s["turnover_annualized"] = bt["summary"].get("turnover_annualized")
        s["avg_gross_exposure"] = bt["summary"].get("avg_gross_exposure")
    else:
        s = bt["summary"]

    row = {
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "family": family, "name": name, "window": window,
        "exec_model": exec_model, "tc_bps": tc_bps, "leverage_cap": leverage_cap,
        "params_json": json.dumps(params, default=str),
        "mean_monthly": s.get("mean_monthly"), "median_monthly": s.get("median_monthly"),
        "sharpe": s.get("sharpe"), "sortino": s.get("sortino"),
        "max_drawdown": s.get("max_drawdown"), "calmar": s.get("calmar"),
        "cagr": s.get("cagr"), "ann_vol": s.get("ann_vol"),
        "worst_month": s.get("worst_month"), "hit_rate": s.get("hit_rate_monthly"),
        "turnover_ann": s.get("turnover_annualized"),
        "avg_gross": s.get("avg_gross_exposure"),
        "runtime_s": round(time.time() - t0, 2),
        "notes": notes,
    }
    for ep, st in crash_table(eq).items():
        row[f"ep_{ep}_ret"] = st["ret"]
        row[f"ep_{ep}_dd"] = st["max_dd"]

    ledger = TRIALS_DIR / f"{family}.csv"
    pd.DataFrame([row]).to_csv(ledger, mode="a", header=not ledger.exists(),
                               index=False)
    return {"summary": s, "equity": eq, "returns": bt["returns"],
            "weights": bt["weights"], "row": row}


def trial_count() -> int:
    """Total logged trials across all families (deflated-Sharpe input)."""
    n = 0
    for f in TRIALS_DIR.glob("*.csv"):
        n += sum(1 for _ in open(f)) - 1
    return n
