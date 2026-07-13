"""E5 — OVERLAY RECOMBINATION on the corrected champion (combo_v2_base_1x x 2.0).

Pre-registered grid: the 2^4 = 16 on/off combinations of four FIXED-parameter
overlays, plus the dd_v2 alternate for every freeze-inclusive combo (8 cells)
= 24 scored configurations. NO parameter re-gridding.

Overlays (params fixed):
  BOOK (B): engine_v2.book_drawdown_gate(W2_daily, close, 0.12, 0.30, 60).
            Multiplicative daily gate. Note: the gate is invariant to per-day
            scalar multipliers (num and den both scale), so it is computed
            ONCE on the base 2x daily frame.
  TIER (T): portfolio tiered stop, P=-0.08 (t1=-8% -> 60%, t2=-13.33% -> 30%,
            t3=-18.67% -> liquidate), re-entry delay = 2 trading days, phi=0.
            Adapted from validation/stop_grid.py's self-contained P&L loop,
            driven by the CORRECTED daily frame. Close-to-close daily-bar
            approximation (see report) — scored via a manual ledger, with the
            untiered loop AND the untiered next_open run_trial as comparators.
  FREEZE (F): strategies.freeze_signal_spy_drawdown, both published variants:
            dd_v1 = (21, 0.12, 0.08, 10), dd_v2 = (30, 0.10, 0.05, 10).
            Daily 0/1 multiplier (0 on frozen days).
  VOLMGMT (V): E1 winner multiplier: vol_managed_multiplier(pass-1 unscaled-2x
            dev net returns, target_vol=0.30, lookback=21, cap=1.0, floor=0.3).
            Built from DEV returns for dev scoring (no out-of-window info).

Composition: W = clip&cap_2.0( expand_daily(base*2) * m_V * g_B * f_F ),
engine exec next_open, 5bp, leverage_cap 2.0, dev window only.

Tier combos: the same composed frame drives the adapted stop_grid loop
(tier-on and tier-off); tier-off reproduces engine next_close to machine
precision (asserted). Manual rows -> results/v7/trials/E5_overlays_tier.csv.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from engine_v2 import (BTConfigV2, run_backtest_v2, expand_daily,
                       vol_managed_multiplier, book_drawdown_gate,
                       apply_gate, _clip_and_cap)
from exp_lib import load_cache, union_prices_cached, run_trial
from metrics import summary as metrics_summary
from metrics_v2 import DEV_END, crash_table
from strategies import freeze_signal_spy_drawdown

ROOT = Path(__file__).resolve().parent
FAMILY = "E5_overlays"
TIER_LEDGER = ROOT / "results" / "v7" / "trials" / "E5_overlays_tier.csv"
TC = 5.0 / 10_000.0
INIT = 100_000.0
TIER_F = {1: 0.6, 2: 0.3, 3: 0.0}
P_TIER, DELAY = -0.08, 2

CFG = dict(exec_model="next_open", tc_bps=5.0, leverage_cap=2.0)

# ── Data (dev window only) ────────────────────────────────────────────────
panel, macro, _ = load_cache()
pu = union_prices_cached()
cutoff = pd.Timestamp(DEV_END)
pu_dev = pu.loc[:cutoff]
opn_dev = panel["open"].loc[:cutoff]
idx = pu_dev.index
base = pd.read_parquet(ROOT / "weights_store" / "combo_v2_base_1x.parquet").loc[:cutoff]

cfg_engine = BTConfigV2(tc_bps=5.0, leverage_cap=2.0, exec_model="next_open")
cfg_close = BTConfigV2(tc_bps=5.0, leverage_cap=2.0, exec_model="next_close")

W2_daily = expand_daily(base * 2.0, idx)          # decision-side daily frame

# ── Overlay multipliers (daily series on idx) ─────────────────────────────
# VOLMGMT: pass-1 unscaled 2x run on DEV ONLY -> multiplier from dev returns.
bt0 = run_backtest_v2(base * 2.0, pu_dev, cfg_engine, name="unscaled2x_dev",
                      open_prices=opn_dev)
m_vol = vol_managed_multiplier(bt0["returns"], 0.30, lookback=21,
                               cap=1.0, floor=0.3).reindex(idx).fillna(1.0)

# BOOK: computed once on the base daily frame (invariant to scalar gates).
g_book = book_drawdown_gate(W2_daily, panel["close"].loc[:cutoff],
                            full_dd=0.12, cash_dd=0.30, lookback=60)
g_book = g_book.reindex(idx).fillna(1.0).clip(0.0, 1.0)

# FREEZE: 0/1 daily multipliers, both published parameterizations.
fz = {
    "v1": freeze_signal_spy_drawdown(macro, peak_lookback=21, freeze_dd_pct=0.12,
                                     unfreeze_within_pct=0.08, min_freeze_days=10),
    "v2": freeze_signal_spy_drawdown(macro, peak_lookback=30, freeze_dd_pct=0.10,
                                     unfreeze_within_pct=0.05, min_freeze_days=10),
}
f_mult = {k: (1.0 - v.astype(float)).reindex(idx).ffill().fillna(1.0)
          for k, v in fz.items()}

# ── Tier-loop market data (held names only; conventions = stop_grid.py) ───
held = list(base.columns[(base.abs() > 1e-12).any()])
close_u = pu_dev[held]                                  # same closes the engine uses
close_ff = close_u.ffill()
prevc = close_ff.shift(1)
rets_h = close_u.pct_change(fill_method=None).fillna(0.0)
open_h = panel["open"][held].reindex(idx)
open_f = open_h.where(open_h.notna(), close_ff)         # NaN open -> today's ffilled close
gap = (open_f / prevc - 1.0).where(prevc.notna(), 0.0).fillna(0.0)

R_np, G_np = rets_h.to_numpy(), gap.to_numpy()
N, ndays = len(held), len(idx)

# Scheduled-rebalance days (base book switch days, EFFECTIVE side) — used to
# reset the persistent tier scale, mirroring stop_grid's "until next rebalance".
E_base = _clip_and_cap(W2_daily, cfg_engine)[held].shift(1).fillna(0.0).to_numpy()
chg = np.abs(np.diff(E_base, axis=0)).sum(axis=1)
base_exec_set = set(int(i) for i in np.where(chg > 1e-15)[0] + 1)


def tier_loop(W_decision: pd.DataFrame, P: float | None, delay: int = DELAY):
    """Adapted stop_grid.simulate: portfolio tiered stop only (T=H=None, phi=0)
    on an arbitrary (possibly daily-varying) corrected decision frame.

    Conventions: book held during day d = clip&cap(W_decision)[d-1] (engine
    effective side); daily P&L close-to-close; tier thresholds on the levered
    book's day return checked at the open gap (actual opens) and at the close
    (phi=0 checkpoint walk); tier scale persists until the next BASE scheduled
    rebalance; tier3 -> cash `delay` days, re-enter at close to that day's
    target; 5bp on all traded notional. With P=None this reproduces engine
    next_close net returns exactly (asserted in main).
    """
    W_eff = _clip_and_cap(W_decision, cfg_engine)[held].shift(1).fillna(0.0).to_numpy()
    t1 = t2 = t3 = None
    if P is not None:
        t1, t2, t3 = P, P * 5.0 / 3.0, P * 7.0 / 3.0
    w = np.zeros(N)
    tier_mult, cash_cd = 1.0, -1
    net = np.zeros(ndays)
    traded = np.zeros(ndays)
    gross_held = np.zeros(ndays)
    ev = dict(tier1=0, tier2=0, tier3=0)

    for d in range(ndays):
        cost = 0.0
        if cash_cd >= 0:                                   # tier3 cash state
            cash_cd -= 1
            if cash_cd < 0:                                # re-enter at close
                tier_mult = 1.0
                w = W_eff[d].copy()
                traded[d] = np.abs(w).sum()
                cost = traded[d] * TC
                net[d] = -cost
            continue
        if d in base_exec_set:
            tier_mult = 1.0                                # scheduled rebalance resets tier scale
        w_new = W_eff[d] * tier_mult
        traded[d] += np.abs(w_new - w).sum()
        cost += np.abs(w_new - w).sum() * TC
        w = w_new
        glev = np.abs(w).sum()
        gross_held[d] = glev
        if P is None or glev <= 0:
            net[d] = float(w @ R_np[d]) - cost
            continue
        r_open = float(w @ G_np[d])
        if r_open <= t3:                                   # gap through tier3
            traded[d] += glev
            net[d] = r_open - cost - glev * TC
            ev["tier3"] += 1
            w = np.zeros(N)
            cash_cd = delay
            continue
        full = float(w @ R_np[d])
        if r_open <= t2:
            open_tier, f = 2, 0.3
        elif r_open <= t1:
            open_tier, f = 1, 0.6
        else:
            open_tier, f = 0, 1.0
        if open_tier:                                      # sold (1-f) at OPEN
            traded[d] += (1.0 - f) * glev
            cost += (1.0 - f) * glev * TC
            R0, dU = r_open, full - r_open
        else:
            R0, dU = 0.0, full
        deepest = open_tier
        for k in range(open_tier + 1, 4):                  # phi=0 checkpoint walk
            tk = (t1, t2, t3)[k - 1]
            u = (tk - R0) / f
            if dU <= u:
                fk = TIER_F[k]
                traded[d] += (f - fk) * glev
                cost += (f - fk) * glev * TC
                R0, dU, f, deepest = tk, dU - u, fk, k
                if k == 3:
                    break
            else:
                break
        if deepest:
            ev[f"tier{deepest}"] += 1
        net[d] = R0 + f * dU - cost
        if deepest == 3:
            w = np.zeros(N)
            cash_cd = delay
        else:
            w = w * f
            tier_mult *= f
    return net, ev, traded, gross_held


def tier_summary(name, net, ev, traded, gross_held, params, notes):
    eq = pd.Series((1.0 + pd.Series(net, index=idx)).cumprod() * INIT, index=idx)
    s = metrics_summary(eq, name=name)
    row = {
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "family": "E5_overlays_tier", "name": name, "window": "dev",
        "exec_model": "tier_loop_cc", "tc_bps": 5.0, "leverage_cap": 2.0,
        "params_json": json.dumps(params, default=str),
        "mean_monthly": s["mean_monthly"], "median_monthly": s["median_monthly"],
        "sharpe": s["sharpe"], "sortino": s["sortino"],
        "max_drawdown": s["max_drawdown"], "calmar": s["calmar"],
        "cagr": s["cagr"], "ann_vol": s["ann_vol"],
        "worst_month": s["worst_month"], "hit_rate": s["hit_rate_monthly"],
        "turnover_ann": float(traded.mean() * 252.0),
        "avg_gross": float(gross_held.mean()),
        "runtime_s": np.nan,
        "notes": notes + f" events={ev}",
    }
    for ep, st in crash_table(eq).items():
        row[f"ep_{ep}_ret"] = st["ret"]
        row[f"ep_{ep}_dd"] = st["max_dd"]
    pd.DataFrame([row]).to_csv(TIER_LEDGER, mode="a",
                               header=not TIER_LEDGER.exists(), index=False)
    return row


def combo_frame(B, F, V):
    W = W2_daily.copy()
    if V:
        W = W.mul(m_vol, axis=0)
    if B:
        W = apply_gate(W, g_book)
    if F:
        W = apply_gate(W, f_mult[F])
    return W


def combo_name(B, T, F, V):
    parts = []
    if B: parts.append("B")
    if T: parts.append("T")
    if F: parts.append("F1" if F == "v1" else "F2")
    if V: parts.append("V")
    return "E5_" + ("+".join(parts) if parts else "none")


def main():
    # ── Fidelity check: tier loop with P=None == engine next_close ────────
    bt_nc = run_backtest_v2(W2_daily, pu_dev, cfg_close, name="nc_check",
                            weights_are_daily=True)
    net_loop, _, _, _ = tier_loop(W2_daily, None)
    diff = float(np.abs(net_loop - bt_nc["returns"].to_numpy()).max())
    print(f"tier loop (P=None) vs engine next_close: max|daily diff| = {diff:.2e}")
    assert diff < 1e-12, "tier loop does not reproduce the next_close engine"

    fixed = {"book": {"full_dd": 0.12, "cash_dd": 0.30, "lookback": 60},
             "tier": {"pstop_eff": P_TIER, "tiers": [-0.08, -0.13333, -0.18667],
                      "fractions": [0.6, 0.3, 0.0], "reentry_delay": DELAY, "phi": 0.0},
             "freeze": {"v1": [21, 0.12, 0.08, 10], "v2": [30, 0.10, 0.05, 10]},
             "volmgmt": {"base_lev": 2.0, "target_vol": 0.30, "lookback": 21,
                         "cap": 1.0, "floor": 0.3}}

    rows = []
    for B, T, V in itertools.product([0, 1], [0, 1], [0, 1]):
        for F in ([None, "v1", "v2"]):
            name = combo_name(B, T, F, V)
            params = {"book": bool(B), "tier": bool(T), "freeze": F,
                      "volmgmt": bool(V), "fixed": fixed}
            W = combo_frame(B, F, V)
            rec = {"name": name, "B": B, "T": T, "F": F or "", "V": V}
            if not T:
                r = run_trial(W, name=name, family=FAMILY, params=params,
                              window="dev", weights_are_daily=True, **CFG,
                              notes="E5 overlay combo, no tier")
                rec.update({k: r["row"][k] for k in
                            ("mean_monthly", "sharpe", "max_drawdown", "calmar",
                             "turnover_ann", "avg_gross", "ep_covid_dd",
                             "ep_2018Q4_dd", "ep_2022_ret", "ep_2022_dd")})
                rec["track"] = "next_open"
            else:
                net_t, ev, traded, gross = tier_loop(W, P_TIER, DELAY)
                row_t = tier_summary(name + "_tier", net_t, ev, traded, gross, params,
                                     notes="manual tier loop, close-to-close daily-bar approx;")
                # untiered loop on the SAME frame (same approximation) for the
                # tier marginal; logged for transparency.
                net_u, ev_u, traded_u, gross_u = tier_loop(W, None)
                row_u = tier_summary(name + "_loop_untiered", net_u, ev_u, traded_u,
                                     gross_u, params,
                                     notes="untiered comparator under the same cc loop;")
                rec.update({k: row_t[k] for k in
                            ("mean_monthly", "sharpe", "max_drawdown", "calmar",
                             "turnover_ann", "avg_gross", "ep_covid_dd",
                             "ep_2018Q4_dd", "ep_2022_ret", "ep_2022_dd")})
                rec["track"] = "tier_loop_cc"
                rec["untiered_cc_calmar"] = row_u["calmar"]
                rec["untiered_cc_mm"] = row_u["mean_monthly"]
                rec["untiered_cc_dd"] = row_u["max_drawdown"]
                rec["tier_events"] = json.dumps(ev)
            rows.append(rec)
            print(f"{name:18s} mm={rec['mean_monthly']*100:6.3f}% "
                  f"sharpe={rec['sharpe']:.3f} dd={rec['max_drawdown']*100:6.1f}% "
                  f"calmar={rec['calmar']:.3f} [{rec['track']}]")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "v7" / "E5_grid_results.csv", index=False)
    print(f"\n{len(df)} combos scored -> results/v7/E5_grid_results.csv")
    print(df.sort_values("calmar", ascending=False)
            [["name", "track", "mean_monthly", "sharpe", "max_drawdown", "calmar"]]
            .head(12).to_string(index=False))


if __name__ == "__main__":
    main()
