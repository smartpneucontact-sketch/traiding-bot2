"""E2 — DYNAMIC SLEEVE ALLOCATION (V7 program).

Replaces the static 1/3 blend of the combo_v2 champion with allocations
driven by trailing sleeve risk/performance, computed STRICTLY from sleeve
returns before each decision date (no look-ahead).

Pre-registered grid (18 configs = 9 methods x 2 floors):
  (a) inverse-vol, lookback in {63, 126}d
  (b) Sharpe-tilt, lookback in {126, 252}d x mapping in
      {softmax(s/tau=1.0), rank 50/33/17, winner-take-most 60/30/10}
  (c) ERC on the 60d sleeve covariance
  All with per-sleeve floor in {0.10, 0.20} (waterfall renormalization).

Fallback: equal weight 1/3 whenever fewer than `lookback` joint daily sleeve
returns exist after the all-sleeves-active date (2017-05-02) strictly before
the decision date — this exactly reproduces the baseline blend on early dates.

Sleeve inputs are simulated ONCE at 1x via engine_v2 (next_open, 5bp) on
prices truncated at `end` (DEV_END for this experiment). Only blended
candidates go through exp_lib.run_trial (family="E2_alloc", window="dev").

Rebuild a winner with, e.g.:
    from exp_e2_alloc import build_e2_blend_1x
    blend = build_e2_blend_1x(method="sharpe", mapping="rank", lookback=126,
                              floor=0.10, end=None)   # 1x; multiply by 2.0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine_v2 import BTConfigV2, run_backtest_v2          # noqa: E402
from exp_lib import load_cache, run_trial, union_prices_cached  # noqa: E402
from metrics_v2 import DEV_END                              # noqa: E402

SLEEVE_KEYS = ["xs", "dual", "adapt"]
SLEEVE_FILES = {
    "xs": "sleeve_xs_momentum_n30",
    "dual": "sleeve_dual_momentum_voltarget_n30",
    "adapt": "sleeve_adaptive_voltarget_n30",
}
FAMILY = "E2_alloc"


# ───────────────────────────────────────────────────────────────────────────
# Inputs
# ───────────────────────────────────────────────────────────────────────────

def load_sleeves() -> dict[str, pd.DataFrame]:
    return {k: pd.read_parquet(ROOT / "weights_store" / f"{f}.parquet")
            for k, f in SLEEVE_FILES.items()}


def align_union(sleeves: dict[str, pd.DataFrame]):
    """Union dates/cols with zero fill — byte-identical alignment to
    reproduce_baseline.build_combo_v2_base."""
    dates = sorted(set().union(*[s.index for s in sleeves.values()]))
    cols = sorted(set().union(*[s.columns for s in sleeves.values()]))
    return ({k: s.reindex(index=dates, columns=cols, fill_value=0.0)
             for k, s in sleeves.items()}, dates, cols)


def sleeve_daily_returns(sleeves: dict[str, pd.DataFrame],
                         end: str | None) -> pd.DataFrame:
    """Per-sleeve 1x daily NET returns via the corrected engine
    (next_open, 5bp), prices truncated at `end`. Direct engine calls —
    inputs only, not logged to the ledger."""
    panel, _, _ = load_cache()
    pu = union_prices_cached()
    opn = panel["open"]
    if end is not None:
        pu, opn = pu.loc[:end], opn.loc[:end]
    cfg = BTConfigV2(tc_bps=5.0, leverage_cap=2.0, exec_model="next_open")
    out = {}
    for k, w in sleeves.items():
        ww = w.loc[:end] if end is not None else w
        bt = run_backtest_v2(ww, pu, cfg, name=f"sleeve_{k}_1x",
                             open_prices=opn)
        out[k] = bt["returns"]
    return pd.DataFrame(out)[SLEEVE_KEYS]


# ───────────────────────────────────────────────────────────────────────────
# Allocation mappings (all operate on a trailing returns window, strictly < d)
# ───────────────────────────────────────────────────────────────────────────

EQ = np.full(3, 1.0 / 3.0)


def apply_floor(a: np.ndarray, floor: float) -> np.ndarray:
    """Waterfall floor: fix sleeves below the floor AT the floor, renormalize
    the rest over the remaining mass. Exact floors, sums to 1."""
    a = np.clip(np.asarray(a, dtype=float), 0.0, None)
    if not np.isfinite(a).all() or a.sum() <= 0:
        return EQ.copy()
    a = a / a.sum()
    fixed = np.zeros(len(a), dtype=bool)
    for _ in range(len(a)):
        below = (a < floor - 1e-12) & ~fixed
        if not below.any():
            break
        fixed |= below
        a[fixed] = floor
        free = ~fixed
        rem = 1.0 - fixed.sum() * floor
        s = a[free].sum()
        a[free] = a[free] / s * rem if s > 0 else rem / max(free.sum(), 1)
    return a


def _sharpes(win: pd.DataFrame) -> np.ndarray | None:
    mu, sd = win.mean().to_numpy(), win.std().to_numpy()
    if (sd <= 1e-12).any():
        return None
    return mu / sd * np.sqrt(252.0)


def alloc_inverse_vol(win: pd.DataFrame, **_) -> np.ndarray:
    sd = win.std().to_numpy()
    if (sd <= 1e-12).any():
        return EQ.copy()
    return (1.0 / sd) / (1.0 / sd).sum()


def alloc_sharpe_softmax(win: pd.DataFrame, tau: float = 1.0, **_) -> np.ndarray:
    s = _sharpes(win)
    if s is None:
        return EQ.copy()
    z = np.exp((s - s.max()) / tau)
    return z / z.sum()


def _rank_map(win: pd.DataFrame, w_by_rank: np.ndarray) -> np.ndarray:
    s = _sharpes(win)
    if s is None:
        return EQ.copy()
    order = np.argsort(-s)          # best first; ties broken by sleeve order
    a = np.empty(3)
    a[order] = w_by_rank
    return a / a.sum()


def alloc_sharpe_rank(win: pd.DataFrame, **_) -> np.ndarray:
    return _rank_map(win, np.array([0.50, 0.33, 0.17]))


def alloc_sharpe_wtm(win: pd.DataFrame, **_) -> np.ndarray:
    return _rank_map(win, np.array([0.60, 0.30, 0.10]))


def alloc_erc(win: pd.DataFrame, **_) -> np.ndarray:
    """Equal-risk-contribution on the window covariance (multiplicative
    fixed-point; 3 assets, long-only)."""
    cov = win.cov().to_numpy()
    if not np.isfinite(cov).all() or (np.diag(cov) <= 1e-16).any():
        return EQ.copy()
    a = EQ.copy()
    for _ in range(500):
        rc = a * (cov @ a)
        if (rc <= 0).any():
            return EQ.copy()
        a_new = a * (rc.mean() / rc) ** 0.5
        a_new = np.clip(a_new, 1e-8, None)
        a_new /= a_new.sum()
        if np.abs(a_new - a).max() < 1e-10:
            a = a_new
            break
        a = a_new
    return a


METHODS = {
    "invvol": alloc_inverse_vol,
    "sharpe_softmax": alloc_sharpe_softmax,
    "sharpe_rank": alloc_sharpe_rank,
    "sharpe_wtm": alloc_sharpe_wtm,
    "erc": alloc_erc,
}


# ───────────────────────────────────────────────────────────────────────────
# Blend builder
# ───────────────────────────────────────────────────────────────────────────

def compute_allocations(sleeve_rets: pd.DataFrame, dates: list,
                        method: str, lookback: int, floor: float,
                        tau: float = 1.0) -> pd.DataFrame:
    """Allocation a_i(d) per decision date from sleeve returns STRICTLY
    before d, restricted to the all-sleeves-active period."""
    sleeves = load_sleeves()
    active_start = max(s.index[0] for s in sleeves.values())
    r = sleeve_rets.loc[sleeve_rets.index >= active_start]
    fn = METHODS[method]
    rows = {}
    for d in dates:
        win = r.loc[r.index < d]
        if len(win) < lookback:
            rows[d] = EQ.copy()                       # baseline 1/3 fallback
        else:
            rows[d] = apply_floor(fn(win.iloc[-lookback:], tau=tau), floor)
    return pd.DataFrame(rows, index=SLEEVE_KEYS).T


def build_e2_blend_1x(method: str, lookback: int, floor: float,
                      tau: float = 1.0, end: str | None = DEV_END,
                      _cache: dict = {}) -> pd.DataFrame:
    """Sparse 1x blended decision frame Σ a_i(d)·w_i(d) on union dates/cols.
    Multiply by 2.0 before scoring. `end` truncates BOTH the sleeve return
    inputs and the decision dates (DEV_END for selection; None = full)."""
    ck = ("inputs", end)
    if ck not in _cache:
        sleeves = load_sleeves()
        sl_u, dates, cols = align_union(sleeves)
        if end is not None:
            dates = [d for d in dates if d <= pd.Timestamp(end)]
            sl_u = {k: v.loc[dates] for k, v in sl_u.items()}
        _cache[ck] = (sl_u, dates, sleeve_daily_returns(sleeves, end))
    sl_u, dates, srets = _cache[ck]
    alloc = compute_allocations(srets, dates, method, lookback, floor, tau)
    blend = sum(sl_u[k].mul(alloc[k], axis=0) for k in SLEEVE_KEYS)
    blend.attrs["alloc"] = alloc
    return blend


# ───────────────────────────────────────────────────────────────────────────
# Pre-registered grid + G1
# ───────────────────────────────────────────────────────────────────────────

def grid() -> list[dict]:
    cfgs = []
    for fl in (0.10, 0.20):
        for lb in (63, 126):
            cfgs.append(dict(method="invvol", lookback=lb, floor=fl))
        for lb in (126, 252):
            for m in ("sharpe_softmax", "sharpe_rank", "sharpe_wtm"):
                cfgs.append(dict(method=m, lookback=lb, floor=fl, tau=1.0))
        cfgs.append(dict(method="erc", lookback=60, floor=fl))
    return cfgs


def cfg_name(c: dict) -> str:
    return f"e2_{c['method']}_lb{c['lookback']}_f{int(c['floor'] * 100)}"


def neighbors(c: dict, all_cfgs: list[dict]) -> list[str]:
    """±1-grid-step neighbors: floor toggled; adjacent lookback (same
    method/floor); for Sharpe-tilt, the other two mappings (same lb/floor)."""
    out = []
    for o in all_cfgs:
        if o == c:
            continue
        same_m = o["method"] == c["method"]
        if same_m and o["lookback"] == c["lookback"] and o["floor"] != c["floor"]:
            out.append(cfg_name(o))
        elif same_m and o["floor"] == c["floor"] and o["lookback"] != c["lookback"]:
            out.append(cfg_name(o))
        elif (c["method"].startswith("sharpe") and o["method"].startswith("sharpe")
              and not same_m and o["lookback"] == c["lookback"]
              and o["floor"] == c["floor"]):
            out.append(cfg_name(o))
    return out


def main():
    cfgs = grid()
    print(f"E2 grid: {len(cfgs)} configs")
    results, alloc_stats = {}, {}
    for c in cfgs:
        name = cfg_name(c)
        blend = build_e2_blend_1x(c["method"], c["lookback"], c["floor"],
                                  tau=c.get("tau", 1.0), end=DEV_END)
        alloc = blend.attrs["alloc"]
        res = run_trial(blend * 2.0, name=name, family=FAMILY, params=c,
                        window="dev", exec_model="next_open", tc_bps=5.0,
                        leverage_cap=2.0,
                        notes="dynamic sleeve allocation, 1/3 fallback pre-burn-in")
        s = res["summary"]
        results[name] = {"cfg": c, "summary": s}
        dyn = alloc[(alloc - 1 / 3).abs().max(axis=1) > 1e-9]
        alloc_stats[name] = {
            "n_dates": len(alloc), "n_dynamic": len(dyn),
            "avg": alloc.mean().round(4).to_dict(),
            "avg_dynamic": (dyn.mean().round(4).to_dict() if len(dyn) else None),
            "min": alloc.min().round(4).to_dict(),
            "max": alloc.max().round(4).to_dict(),
            "mean_abs_dev_from_third": float((alloc - 1 / 3).abs().mean().mean()),
        }
        print(f"  {name:28s} mm={s['mean_monthly']*100:+.3f}% sharpe={s['sharpe']:.3f} "
              f"dd={s['max_drawdown']*100:.1f}% calmar={s['calmar']:.3f}")

    # G1 evaluation
    rows = []
    for name, r in results.items():
        s, c = r["summary"], r["cfg"]
        nb = neighbors(c, cfgs)
        nb_cal = [results[n]["summary"]["calmar"] for n in nb if n in results]
        med_nb = float(np.median(nb_cal)) if nb_cal else float("nan")
        g1 = (s["calmar"] >= 0.944 and s["mean_monthly"] >= 0.04
              and s["max_drawdown"] > -0.50
              and np.isfinite(med_nb) and med_nb >= 0.85 * s["calmar"])
        rows.append({
            "name": name, **c,
            "mean_monthly_pct": s["mean_monthly"] * 100, "sharpe": s["sharpe"],
            "max_dd_pct": s["max_drawdown"] * 100, "calmar": s["calmar"],
            "turnover_ann": s.get("turnover_annualized"),
            "avg_gross": s.get("avg_gross_exposure"),
            "neighbors": ";".join(nb), "median_neighbor_calmar": med_nb,
            "stability_ok": bool(np.isfinite(med_nb) and med_nb >= 0.85 * s["calmar"]),
            "G1": bool(g1),
        })
    df = pd.DataFrame(rows).sort_values("calmar", ascending=False)
    out = ROOT / "results" / "v7" / "E2_grid_results.csv"
    df.to_csv(out, index=False)
    with open(ROOT / "results" / "v7" / "E2_alloc_stats.json", "w") as f:
        json.dump(alloc_stats, f, indent=2)
    print("\n" + df.drop(columns=["neighbors"]).to_string(index=False))
    print(f"\nsaved {out}")
    return df, alloc_stats


if __name__ == "__main__":
    main()
