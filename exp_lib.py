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

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine_v2 import BTConfigV2, run_backtest_v2
from metrics_v2 import DEV_END, VAL_START, crash_table

ROOT = Path(__file__).resolve().parent
TRIALS_DIR = ROOT / "results" / "v7" / "trials"
WEIGHTS_STORE = ROOT / "weights_store"
TRIALS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_STORE.mkdir(exist_ok=True)

# Cache version tag: sha1 over the strategy-defining sources, so cached
# weight frames are invalidated automatically when strategy code changes.
_CODE_HASH = hashlib.sha1(
    b"".join((ROOT / f).read_bytes()
             for f in ("strategies.py", "strategies_v2.py", "ensemble.py"))
).hexdigest()[:10]

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
    """Parquet-cache a (slow) sleeve/strategy weight frame by key.

    The cache path embeds `_CODE_HASH`: keying by name alone was unsafe
    because an edit to strategies.py/strategies_v2.py/ensemble.py would
    silently serve weights built by the OLD code. Pre-existing unversioned
    {key}.parquet files are left on disk untouched — they are published-record
    artifacts that other scripts read by explicit path.
    """
    p = WEIGHTS_STORE / f"{key}__{_CODE_HASH}.parquet"
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
    margin_bps_annual: float = 0.0,
    weights_are_daily: bool = False,
    final: bool = False,
    notes: str = "",
) -> dict:
    """Score a weight frame under the corrected engine and log the trial.

    `weights` are PRE-leverage decision weights ONLY if the caller already
    multiplied leverage in — this function does NOT apply leverage; pass
    post-leverage frames exactly as backtest.py callers do.

    `margin_bps_annual` is forwarded to BTConfigV2 (financing drag on long
    gross > 1.0x NAV). When non-zero it is also embedded in params_json and
    logged as a `margin_bps` ledger column, so margin-on and margin-off runs
    of the same config dedupe as DISTINCT trials. The default 0.0 keeps
    existing calls byte-identical (no new key/column), because (a) ledgers
    are append-mode CSVs and cannot change schema mid-file, and (b) injecting
    the key at 0.0 would mutate params_json of already-logged configs and
    inflate the deduped trial count. Consequence: callers turning margin ON
    must use a NEW family name — the margin_bps column only exists in family
    CSVs created after margin was first requested (enforced below).
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
                     exec_model=exec_model,
                     margin_bps_annual=margin_bps_annual)
    if margin_bps_annual != 0.0:
        params = {**params, "margin_bps_annual": margin_bps_annual}
    t0 = time.time()
    bt = run_backtest_v2(w, pu_w, cfg, name=name,
                         open_prices=opn_w if exec_model == "next_open" else None,
                         weights_are_daily=weights_are_daily)
    eq = bt["equity"]
    if window == "val":
        eq = eq.loc[VAL_START:]
        eq = eq / eq.iloc[0] * 100_000.0
        from metrics import summary as msum
        s = msum(eq, name=name)
        # Exposure/turnover stats recomputed on the val slice (the full-run
        # summary mixes in the dev burn-in). Slicing convention: effective
        # daily weights from VAL_START on; diff() row 1 is NaN (no prior row
        # inside the slice) and would sum to a synthetic 0-turnover day, so
        # drop it via iloc[1:]. The book itself was established during the
        # dev burn-in, so its entry cost is charged to dev, not val; a trade
        # landing exactly on the first val day is invisible to this stat
        # (its cost still hits val equity) — accepted, sub-bp effect.
        eff = bt["weights"].loc[VAL_START:]
        turnover = eff.diff().abs().sum(axis=1).iloc[1:]
        s["turnover_annualized"] = float(turnover.mean() * 252.0)
        s["avg_gross_exposure"] = float(eff.abs().sum(axis=1).mean())
        s["avg_n_positions"] = float((eff.abs() > 1e-6).sum(axis=1).mean())
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
    if margin_bps_annual != 0.0:
        row["margin_bps"] = margin_bps_annual
    for ep, st in crash_table(eq).items():
        row[f"ep_{ep}_ret"] = st["ret"]
        row[f"ep_{ep}_dd"] = st["max_dd"]

    ledger = TRIALS_DIR / f"{family}.csv"
    row_df = pd.DataFrame([row])
    if ledger.exists():
        # Append-mode CSV: to_csv(mode="a", header=False) writes POSITIONALLY,
        # so the row must be aligned to the on-file header in both content and
        # order. Two failure modes guarded here: (a) a margin row into a
        # pre-margin family (new column the header lacks -> hard error, use a
        # new family); (b) a margin=0 row into a margin family (header has
        # margin_bps, row lacks it -> fill 0.0 so columns stay aligned).
        with open(ledger) as f:
            header = f.readline().strip().split(",")
        extra = [c for c in row_df.columns if c not in header]
        if extra:
            raise ValueError(
                f"family {family!r} ledger header lacks column(s) {extra} — "
                f"new columns require a NEW family name (append-mode CSVs "
                f"cannot change schema mid-file)."
            )
        if "margin_bps" in header and "margin_bps" not in row_df.columns:
            row_df["margin_bps"] = 0.0
        row_df = row_df.reindex(columns=header)
    row_df.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
    return {"summary": s, "equity": eq, "returns": bt["returns"],
            "weights": bt["weights"], "row": row}


def trial_count(dedupe: bool = True) -> int:
    """Logged trials across all families (deflated-Sharpe input).

    Ledgers are append-only, so re-running an experiment script re-logs the
    same configs. Re-runs are not new hypotheses: counting raw rows deflates
    the Sharpe too aggressively AND makes the count depend on how many times
    scripts were executed. Default counts UNIQUE trials by the identifying
    columns; dedupe=False gives the old raw row count.
    """
    key_cols = ["family", "name", "window", "exec_model", "tc_bps",
                "leverage_cap", "params_json"]
    if not dedupe:
        return sum(sum(1 for _ in open(f)) - 1 for f in TRIALS_DIR.glob("*.csv"))
    frames = [pd.read_csv(f, usecols=key_cols) for f in TRIALS_DIR.glob("*.csv")]
    if not frames:
        return 0
    return int(len(pd.concat(frames, ignore_index=True).drop_duplicates()))
