"""Daily-bar backtest of the live intraday cut-loss layer on top of the
reproduced combo_v2_2x book (see reproduce_baseline.py — must be run first).

Mechanics (all on the LEVERED book, i.e. the exact daily target weights the
vector engine trades, gross ~1.8-2.0):

  Holding period p: decision date t_p (signal at close), book switches at the
  close of t_p (the engine's weights.shift(1) means day t_p+1 earns the new
  book's close-to-close return). Entry price = close[t_p]. Days (t_p, t_{p+1}]
  hold the book; stopped positions go to CASH until the next rebalance
  (no redistribution).

  TRAILING stop T: peak = running max of daily closes since entry (incl. entry
  close, through yesterday). Stop if today's LOW <= peak*(1+T).
  Fill = min(today's OPEN, peak*(1+T))  — conservative on overnight gaps.

  HARD stop H: same vs entry price. If both armed, the binding intraday level
  is the HIGHER one (hit first); event classified by that level.

  PORTFOLIO tiered stop, thresholds on LEVERED equity vs yesterday's close:
  t1 = pstop_eff, t2 = pstop_eff*5/3, t3 = pstop_eff*7/3 (as in the live code).
  Intraday equity approximated at two checkpoints:
    (a) the open gap  r_open = sum(w_i * (open_i/prevclose_i - 1))
        breach at open  -> act at OPEN prices (gap blows through the level)
    (b) the close path -> threshold-level fill approximation: walk the
        (assumed monotone) unscaled path to the close; each tier crossing
        scales the book to 60% / 30% / 0% of the morning book, slowing the
        subsequent equity slope; a tier only fires if the SCALED path still
        reaches its threshold by the close.
  Tier1/2: sell pro-rata, scaled book persists until the next rebalance.
  Tier3: liquidate, sit in cash REENTRY_DELAY trading days, then re-buy the
  scheduled target at that day's close (entry prices reset).

  Costs: 5bp one-way on every traded notional (rebalance turnover, stop sells,
  tier sells, tier3 liquidation, re-entry) — identical cost model to the
  baseline engine; with all stops off this reproduces the baseline equity
  to machine precision (asserted below).
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from metrics import summary as metrics_summary  # noqa: E402

V = ROOT / "validation"
TC = 5.0 / 10_000.0
INIT = 100_000.0

# ── Load artifacts ───────────────────────────────────────────────────────
target = pd.read_parquet(V / "target_weights_daily.parquet")
held_cols = list(target.columns[(target.abs() > 1e-12).any()])
target = target[held_cols]
idx = target.index
repro_net = pd.read_parquet(V / "repro_daily_net.parquet")["ret"].reindex(idx).fillna(0.0)
dec_dates = pd.read_parquet(V / "decision_dates.parquet")["decision_date"]

with open(ROOT / "data_cache.pkl", "rb") as f:
    cache = pickle.load(f)
panel = cache["panel"]
close = panel["close"][held_cols].reindex(idx)
open_ = panel["open"][held_cols].reindex(idx)
low = panel["low"][held_cols].reindex(idx)

close_ff = close.ffill()
prevc = close_ff.shift(1)
rets = close.pct_change(fill_method=None).fillna(0.0)          # == engine returns
open_f = open_.where(open_.notna(), close_ff)                   # NaN open -> ffilled close
gap = (open_f / prevc - 1.0).where(prevc.notna(), 0.0).fillna(0.0)
low_f = low.where(low.notna(), np.inf)                          # NaN low -> never triggers
# per-ticker intraday-low return for the PESSIMISTIC basket-low bound
low_b = low.where(low.notna(), close_ff)
gap_low = (low_b / prevc - 1.0).where(prevc.notna(), 0.0).fillna(0.0)

T_np = target.to_numpy()
C_np = close_ff.to_numpy()
PC_np = prevc.to_numpy()
R_np = rets.to_numpy()
G_np = gap.to_numpy()
GL_np = gap_low.to_numpy()
O_np = open_f.to_numpy()
L_np = low_f.to_numpy()
N = len(held_cols)
ndays = len(idx)

# NOTE (fidelity finding): the published engine's `weights.shift(1)` acts on the
# SPARSE 113-row decision-date frame, so each decision is applied one decision
# ROW later = one full 21-trading-day period of execution lag (not 1 day as the
# docstring claims). The book switches ON decision dates, holding the weights
# decided at the PREVIOUS decision date. We derive the true switch days from
# the saved daily target frame itself so the overlay matches the published
# baseline exactly. Entry (close-basis) = close of the day before the switch.
chg = np.abs(np.diff(T_np, axis=0)).sum(axis=1)
exec_set = set(int(i) for i in np.where(chg > 1e-15)[0] + 1)
first_exec = min(exec_set)
gross_target = np.abs(T_np).sum(axis=1)
active_years = (idx[-1] - idx[first_exec]).days / 365.25
TIER_F = {1: 0.6, 2: 0.3, 3: 0.0}


def simulate(T, H, P, delay, phi=0.0):
    """Return (daily_net np.array over idx, event dict, time-in-market).

    phi: intraday tier-breach stress. The true intraday basket low lies
    between the daily-bar checkpoint bottom B_cp = min(r_open, r_close)
    (phi=0, the spec's open/close approximation — optimistic, misses
    intraday dips that recover by the close) and the SIMULTANEOUS
    per-ticker-lows basket B_sl = sum(w_i*(low_i/prevclose_i-1)) (phi=1,
    every holding at its low at the same minute — extreme worst case).
    Tier breaches on the descent are tested against
    B = B_cp - phi*(B_cp - B_sl); the book then recovers to the close
    with the post-tier scaled exposure.
    """
    w = np.zeros(N)
    entry = np.full(N, np.nan)
    peak = np.full(N, np.nan)
    cash_cd = -1                       # >=0: in cash, counting down to re-entry
    net = np.zeros(ndays)
    ev = dict(trail=0, hard=0, tier1=0, tier2=0, tier3=0)
    tim_sum, tim_n = 0.0, 0
    t1 = t2 = t3 = None
    if P is not None:
        t1, t2, t3 = P, P * 5.0 / 3.0, P * 7.0 / 3.0

    for d in range(first_exec, ndays):
        cost = 0.0
        # ── scheduled rebalance (skipped while in tier3 cash) ──
        if d in exec_set and cash_cd < 0:
            w_new = T_np[d]
            cost += np.abs(w_new - w).sum() * TC
            w = w_new.copy()
            entry = PC_np[d].copy()    # close of decision date t_p
            peak = entry.copy()

        # ── tier3 cash state ──
        if cash_cd >= 0:
            cash_cd -= 1
            if cash_cd < 0:            # re-enter at today's close
                w = T_np[d].copy()
                cost = w.sum() * TC
                entry = C_np[d].copy()
                peak = entry.copy()
                net[d] = -cost
            else:
                net[d] = 0.0
            gt = gross_target[d]
            tim_sum += (w.sum() / gt) if gt > 0 else 0.0
            tim_n += 1
            continue

        glev = w.sum()
        r = R_np[d]

        # ── individual stops ──
        trig = None
        if (T is not None or H is not None) and glev > 0:
            level = np.full(N, -np.inf)
            lt = lh = None
            if T is not None:
                lt = peak * (1.0 + T)
                level = np.fmax(level, np.where(np.isnan(lt), -np.inf, lt))
            if H is not None:
                lh = entry * (1.0 + H)
                level = np.fmax(level, np.where(np.isnan(lh), -np.inf, lh))
            trig = (w > 0) & np.isfinite(level) & (L_np[d] <= level)
            if trig.any():
                fill = np.minimum(O_np[d], level)
                ok = trig & np.isfinite(PC_np[d])
                r = r.copy()
                r[ok] = fill[ok] / PC_np[d][ok] - 1.0
                if T is not None and H is not None:
                    is_trail = lt >= lh                  # higher level hit first
                    ev["trail"] += int((trig & is_trail).sum())
                    ev["hard"] += int((trig & ~is_trail).sum())
                elif T is not None:
                    ev["trail"] += int(trig.sum())
                else:
                    ev["hard"] += int(trig.sum())
            else:
                trig = None

        # ── portfolio tiered stop ──
        if P is not None and glev > 0:
            r_open = float(w @ G_np[d])
            if r_open <= t3:                              # gap through tier3
                net[d] = r_open - cost - glev * TC
                ev["tier3"] += 1
                w = np.zeros(N)
                cash_cd = delay
                gt = gross_target[d]
                tim_sum += 0.0
                tim_n += 1
                continue
            full = float(w @ r)                           # day P&L incl. stop fills
            if r_open <= t2:
                open_tier, f = 2, 0.3
            elif r_open <= t1:
                open_tier, f = 1, 0.6
            else:
                open_tier, f = 0, 1.0
            if open_tier:                                 # sold (1-f) at OPEN prices
                cost += (1.0 - f) * glev * TC
                R0, dU = r_open, full - r_open
            else:
                R0, dU = 0.0, full
            # descent leg: blend of checkpoint bottom and simultaneous-lows bottom
            start = r_open if open_tier else 0.0
            if phi > 0.0:
                b_cp = min(r_open, full)
                b_sl = min(float(w @ GL_np[d]), b_cp)     # simultaneous lows
                B = b_cp - phi * (b_cp - b_sl)
                dU_desc = min(B - start, 0.0)
                dU_rec = (full - start) - dU_desc         # recovery into the close
            else:
                dU_desc, dU_rec = dU, 0.0
            deepest = open_tier
            for k in range(open_tier + 1, 4):             # walk remaining tiers
                tk = (t1, t2, t3)[k - 1]
                u = (tk - R0) / f
                if dU_desc <= u:                          # scaled path reaches tier k
                    fk = TIER_F[k]
                    cost += (f - fk) * glev * TC
                    R0, dU_desc, f, deepest = tk, dU_desc - u, fk, k
                    if k == 3:
                        break
                else:
                    break
            dU = dU_desc + dU_rec
            if deepest:
                ev[f"tier{deepest}"] += 1
            if trig is not None:
                # stop sells happen on the post-open-scaled book
                f0 = 1.0 if not open_tier else TIER_F[open_tier]
                cost += f0 * w[trig].sum() * TC
            day_gross = R0 + f * dU
            if deepest == 3:
                w = np.zeros(N)
                cash_cd = delay
            else:
                w = w * f
                if trig is not None:
                    w[trig] = 0.0
        else:
            day_gross = float(w @ r)
            if trig is not None:
                cost += w[trig].sum() * TC
                w = w.copy()
                w[trig] = 0.0

        net[d] = day_gross - cost
        peak = np.fmax(peak, C_np[d])
        gt = gross_target[d]
        tim_sum += (w.sum() / gt) if gt > 0 else 0.0
        tim_n += 1

    return net, ev, tim_sum / max(tim_n, 1)


def evaluate(name, T, H, P, delay, phi=0.0):
    net, ev, tim = simulate(T, H, P, delay, phi=phi)
    eq = pd.Series((1.0 + pd.Series(net, index=idx)).cumprod() * INIT, index=idx)
    s = metrics_summary(eq, name=name)
    act = net[first_exec:]
    out = {
        "config": name,
        "trailing": T, "hard": H, "pstop_eff": P, "reentry_delay": delay if P is not None else None,
        "mean_monthly": s["mean_monthly"], "sharpe": s["sharpe"],
        "max_drawdown": s["max_drawdown"], "calmar": s["calmar"],
        "cagr": s["cagr"], "ann_vol": s["ann_vol"],
        "worst_day": float(act.min()), "worst_month": s["worst_month"],
        "final_equity": s["final_equity"], "hit_rate_monthly": s["hit_rate_monthly"],
        "events_per_year": {k: round(v / active_years, 2) for k, v in ev.items()},
        "events_total": ev, "avg_time_in_market": tim,
    }
    return out, eq


def cname(T, H, P, delay):
    f = lambda x: "none" if x is None else f"{x*100:g}"
    base = f"T={f(T)}|H={f(H)}|P={f(P)}"
    return base + (f"|d={delay}" if P is not None else "")


def main():
    t0 = time.time()
    # ── fidelity: all-stops-off must equal the reproduced baseline exactly ──
    net0, _, _ = simulate(None, None, None, 1)
    diff = float(np.abs(net0 - repro_net.to_numpy()).max())
    print(f"no-stop overlay vs reproduced baseline: max |daily ret diff| = {diff:.2e}")
    assert diff < 1e-12, "overlay engine does not reproduce the baseline"

    T_grid = [None, -0.08, -0.10, -0.12, -0.15, -0.20]
    H_grid = [None, -0.15, -0.20, -0.25]
    # assignment grid {none,-4,-6,-8} + intermediate refinement points found
    # by a fine 1-D scan of the portfolio-only family (-4.5/-5 are local optima)
    P_grid = [None, -0.04, -0.045, -0.05, -0.055, -0.06, -0.07, -0.08]
    results = []
    for T, H, P in itertools.product(T_grid, H_grid, P_grid):
        for delay in ([1, 2, 3] if P is not None else [1]):
            name = cname(T, H, P, delay)
            out, eq = evaluate(name, T, H, P, delay)
            results.append(out)
    # AS-DEPLOYED config
    out, eq = evaluate("AS_DEPLOYED " + cname(-0.05, -0.08, -0.03, 1), -0.05, -0.08, -0.03, 1)
    out["as_deployed"] = True
    results.append(out)

    # ── intraday-path stress for the leading candidates + as-deployed:
    # phi=0.5 heuristic mid-case, phi=1.0 worst-case (simultaneous lows)
    for lbl, (T, H, P, dly) in {
        "P=-4.5": (None, None, -0.045, 1),
        "P=-5": (None, None, -0.05, 1),
        "P=-6|d=2": (None, None, -0.06, 2),
        "P=-8": (None, None, -0.08, 1),
        "AS_DEPLOYED": (-0.05, -0.08, -0.03, 1),
    }.items():
        for phi in (0.5, 1.0):
            out, eq = evaluate(f"STRESS phi={phi:g} {lbl}", T, H, P, dly, phi=phi)
            out["phi"] = phi
            results.append(out)

    with open(V / "stop_grid_results.json", "w") as f:
        json.dump({"baseline": cname(None, None, None, 1),
                   "active_years": active_years,
                   "results": results}, f, indent=2, default=str)
    print(f"{len(results)} configs in {time.time()-t0:.1f}s -> {V/'stop_grid_results.json'}")

    df = pd.DataFrame(results)
    base_row = df[df.config == "T=none|H=none|P=none"].iloc[0]
    floor = base_row.mean_monthly - 0.003
    plain = ~df.config.str.startswith(("AS_DEPLOYED", "STRESS"))
    elig = df[(df.mean_monthly >= floor) & plain]
    best = elig.sort_values("calmar", ascending=False).head(15)
    cols = ["config", "mean_monthly", "sharpe", "max_drawdown", "calmar", "worst_day", "avg_time_in_market"]
    print(f"\nbaseline: mo={base_row.mean_monthly:.4%} sharpe={base_row.sharpe:.3f} dd={base_row.max_drawdown:.2%} calmar={base_row.calmar:.3f}")
    print(f"eligibility floor mean_monthly >= {floor:.4%}\nTop by Calmar among eligible:")
    with pd.option_context("display.width", 200):
        print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
