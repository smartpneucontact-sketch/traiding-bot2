"""ASSEMBLY — combine V7 family winners into final rules-track candidates.

Pre-registered product (12 candidates, no degenerate duplicates):
  BASE sleeve set:
    A = champion 3-sleeve  (xs_momentum_n30, dual_momentum_voltarget_n30,
                            adaptive_voltarget_n30)
    B = E3 winner          (residual_mom_market_sector_raw_n30 REPLACING xs,
                            + dual + adapt)            [e3_resid_ms_raw_replxs]
  ALLOCATION:
    st = static 1/3 (champion convention: union dates/cols, sum/3)
    sx = E2 winner  (sharpe_softmax, lookback=126, floor=0.10, tau=1.0,
                     1/3 fallback pre-burn-in, allocations strictly ex-ante)
  OVERLAY stack (E5 family, fixed params):
    TF1 = E5 winner: portfolio tier stop (P=-0.08 -> -8/-13.33/-18.67% ->
          60/30/0%, re-entry delay 2, phi=0) + SPY-DD freeze dd_v1
          (21, 0.12, 0.08, 10)
    T   = E5 runner-up: tier stop only
    F1  = freeze dd_v1 only (floor; pure next_open run_trial track)

Scoring (dev window only, <= 2022-12-31):
  - F1-only candidates: exp_lib.run_trial (family="ASSEMBLY", next_open,
    leverage_cap=2.0) at tc_bps in {5, 10, 20}.
  - Tier candidates: generalized E5 tier loop (close-to-close daily-bar
    approximation, identical conventions to exp_e5_overlays.tier_loop but
    re-instantiated per base frame: held set, gaps, scheduled-rebalance
    reset days from the candidate's OWN un-gated base) at tc in {5,10,20};
    manual ledger results/v7/trials/ASSEMBLY_tier.csv (E5 precedent).
    Fidelity assertion per base frame: loop with stops off == engine
    next_close daily net returns to <1e-12.
  - Un-gated base of each (base, alloc) pair run once at 5bp via run_trial
    as anchor/comparator (A_st must reproduce baseline.json dev, A_sx the
    E2 winner, B_st the E3 winner; A?_T/TF1/F1 must reproduce E5 rows).

Gate G2 (5bp, native track): mean>=4.5%/mo, MaxDD>=-40%, Calmar>=1.3,
turnover<=30x/yr, 20bp Calmar drop <25%, covid/2026Q1 episode DD not worse
than champion's by >5pp (2026Q1 is outside dev — flagged for validation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine_v2 import (BTConfigV2, run_backtest_v2, expand_daily,  # noqa: E402
                       apply_gate, _clip_and_cap)
from exp_lib import load_cache, union_prices_cached, run_trial, trial_count  # noqa: E402
from metrics import summary as metrics_summary  # noqa: E402
from metrics_v2 import DEV_END, crash_table  # noqa: E402
from strategies import freeze_signal_spy_drawdown  # noqa: E402
from exp_e2_alloc import (build_e2_blend_1x, apply_floor,  # noqa: E402
                          alloc_sharpe_softmax)

FAMILY = "ASSEMBLY"
TIER_LEDGER = ROOT / "results" / "v7" / "trials" / "ASSEMBLY_tier.csv"
INIT = 100_000.0
P_TIER, DELAY = -0.08, 2
TIER_F = {1: 0.6, 2: 0.3, 3: 0.0}
TCS = (5.0, 10.0, 20.0)

SLEEVE_FILES = {
    "xs": "sleeve_xs_momentum_n30",
    "res": "sleeve_residual_mom_market_sector_raw_n30",
    "dual": "sleeve_dual_momentum_voltarget_n30",
    "adapt": "sleeve_adaptive_voltarget_n30",
}

# ── Data (dev window only) ────────────────────────────────────────────────
panel, macro, _ = load_cache()
pu = union_prices_cached()
cutoff = pd.Timestamp(DEV_END)
pu_dev = pu.loc[:cutoff]
opn_dev = panel["open"].loc[:cutoff]
idx = pu_dev.index

cfg_open = BTConfigV2(tc_bps=5.0, leverage_cap=2.0, exec_model="next_open")
cfg_close = BTConfigV2(tc_bps=5.0, leverage_cap=2.0, exec_model="next_close")

# FREEZE dd_v1: 0/1 daily multiplier (identical construction to exp_e5_overlays)
_fz = freeze_signal_spy_drawdown(macro, peak_lookback=21, freeze_dd_pct=0.12,
                                 unfreeze_within_pct=0.08, min_freeze_days=10)
F1_MULT = (1.0 - _fz.astype(float)).reindex(idx).ffill().fillna(1.0)


def load_sleeve(k: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "weights_store" / f"{SLEEVE_FILES[k]}.parquet")


def align_union(sleeves: dict[str, pd.DataFrame]):
    """Union dates/cols, zero fill — byte-identical to E2/E3 conventions."""
    dates = sorted(set().union(*[s.index for s in sleeves.values()]))
    cols = sorted(set().union(*[s.columns for s in sleeves.values()]))
    return ({k: s.reindex(index=dates, columns=cols, fill_value=0.0)
             for k, s in sleeves.items()}, dates, cols)


def static_blend_1x(sleeves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    su, _, _ = align_union(sleeves)
    return (sum(su.values()) / len(su)).loc[:cutoff]


def sleeve_rets_1x(sleeves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-sleeve 1x daily NET returns (next_open, 5bp), dev-truncated —
    engine inputs only (not ledger-logged), exactly as exp_e2_alloc does."""
    out = {}
    for k, w in sleeves.items():
        bt = run_backtest_v2(w.loc[:cutoff], pu_dev, cfg_open,
                             name=f"sleeve_{k}_1x_dev", open_prices=opn_dev)
        out[k] = bt["returns"]
    return pd.DataFrame(out)[list(sleeves)]


def softmax_blend_1x(sleeves: dict[str, pd.DataFrame], lookback: int = 126,
                     floor: float = 0.10, tau: float = 1.0) -> pd.DataFrame:
    """Generalized E2-winner allocation for an arbitrary 3-sleeve set.
    Allocation at decision date d from sleeve net returns STRICTLY before d,
    restricted to the all-sleeves-active period; 1/3 fallback pre-burn-in."""
    su, dates, _ = align_union(sleeves)
    dates = [d for d in dates if d <= cutoff]
    su = {k: v.loc[dates] for k, v in su.items()}
    srets = sleeve_rets_1x(sleeves)
    active_start = max(s.index[0] for s in sleeves.values())
    r = srets.loc[srets.index >= active_start]
    keys = list(sleeves)
    eq = np.full(len(keys), 1.0 / len(keys))
    rows = {}
    for d in dates:
        win = r.loc[r.index < d]
        if len(win) < lookback:
            rows[d] = eq.copy()
        else:
            rows[d] = apply_floor(
                alloc_sharpe_softmax(win.iloc[-lookback:], tau=tau), floor)
    alloc = pd.DataFrame(rows, index=keys).T
    blend = sum(su[k].mul(alloc[k], axis=0) for k in keys)
    blend.attrs["alloc"] = alloc
    return blend


# ── Generalized tier environment (per base×alloc frame) ──────────────────
class TierEnv:
    """Re-instantiates exp_e5_overlays' tier-loop data section for an
    arbitrary sparse 1x blend: held names, close-to-close returns, open
    gaps, and scheduled-rebalance reset days from the UN-GATED base."""

    def __init__(self, blend_1x: pd.DataFrame):
        self.W2_daily = expand_daily(blend_1x * 2.0, idx)
        self.held = list(blend_1x.columns[(blend_1x.abs() > 1e-12).any()])
        close_u = pu_dev[self.held]
        close_ff = close_u.ffill()
        prevc = close_ff.shift(1)
        self.R = close_u.pct_change(fill_method=None).fillna(0.0).to_numpy()
        open_h = panel["open"][self.held].reindex(idx)
        open_f = open_h.where(open_h.notna(), close_ff)
        self.G = (open_f / prevc - 1.0).where(prevc.notna(), 0.0).fillna(0.0).to_numpy()
        self.N, self.ndays = len(self.held), len(idx)
        e_base = _clip_and_cap(self.W2_daily, cfg_open)[self.held] \
            .shift(1).fillna(0.0).to_numpy()
        chg = np.abs(np.diff(e_base, axis=0)).sum(axis=1)
        self.base_exec_set = set(int(i) for i in np.where(chg > 1e-15)[0] + 1)

    def tier_loop(self, W_decision: pd.DataFrame, P: float | None,
                  delay: int = DELAY, tc_bps: float = 5.0):
        """Identical conventions to exp_e5_overlays.tier_loop (close-to-close
        daily-bar approx, phi=0, 5bp->tc_bps on all traded notional); with
        P=None reproduces engine next_close exactly (asserted in main)."""
        tc = tc_bps / 10_000.0
        W_eff = _clip_and_cap(W_decision, cfg_open)[self.held] \
            .shift(1).fillna(0.0).to_numpy()
        t1 = t2 = t3 = None
        if P is not None:
            t1, t2, t3 = P, P * 5.0 / 3.0, P * 7.0 / 3.0
        w = np.zeros(self.N)
        tier_mult, cash_cd = 1.0, -1
        net = np.zeros(self.ndays)
        traded = np.zeros(self.ndays)
        gross_held = np.zeros(self.ndays)
        ev = dict(tier1=0, tier2=0, tier3=0)
        R_np, G_np = self.R, self.G

        for d in range(self.ndays):
            cost = 0.0
            if cash_cd >= 0:
                cash_cd -= 1
                if cash_cd < 0:
                    tier_mult = 1.0
                    w = W_eff[d].copy()
                    traded[d] = np.abs(w).sum()
                    cost = traded[d] * tc
                    net[d] = -cost
                continue
            if d in self.base_exec_set:
                tier_mult = 1.0
            w_new = W_eff[d] * tier_mult
            traded[d] += np.abs(w_new - w).sum()
            cost += np.abs(w_new - w).sum() * tc
            w = w_new
            glev = np.abs(w).sum()
            gross_held[d] = glev
            if P is None or glev <= 0:
                net[d] = float(w @ R_np[d]) - cost
                continue
            r_open = float(w @ G_np[d])
            if r_open <= t3:
                traded[d] += glev
                net[d] = r_open - cost - glev * tc
                ev["tier3"] += 1
                w = np.zeros(self.N)
                cash_cd = delay
                continue
            full = float(w @ R_np[d])
            if r_open <= t2:
                open_tier, f = 2, 0.3
            elif r_open <= t1:
                open_tier, f = 1, 0.6
            else:
                open_tier, f = 0, 1.0
            if open_tier:
                traded[d] += (1.0 - f) * glev
                cost += (1.0 - f) * glev * tc
                R0, dU = r_open, full - r_open
            else:
                R0, dU = 0.0, full
            deepest = open_tier
            for k in range(open_tier + 1, 4):
                tk = (t1, t2, t3)[k - 1]
                u = (tk - R0) / f
                if dU <= u:
                    fk = TIER_F[k]
                    traded[d] += (f - fk) * glev
                    cost += (f - fk) * glev * tc
                    R0, dU, f, deepest = tk, dU - u, fk, k
                    if k == 3:
                        break
                else:
                    break
            if deepest:
                ev[f"tier{deepest}"] += 1
            net[d] = R0 + f * dU - cost
            if deepest == 3:
                w = np.zeros(self.N)
                cash_cd = delay
            else:
                w = w * f
                tier_mult *= f
        return net, ev, traded, gross_held


def tier_row(name, net, ev, traded, gross_held, params, tc_bps, notes):
    """Manual ledger row mirroring exp_lib.run_trial's schema (E5 precedent)."""
    eq = pd.Series((1.0 + pd.Series(net, index=idx)).cumprod() * INIT, index=idx)
    s = metrics_summary(eq, name=name)
    row = {
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "family": "ASSEMBLY_tier", "name": name, "window": "dev",
        "exec_model": "tier_loop_cc", "tc_bps": tc_bps, "leverage_cap": 2.0,
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


# ── Candidate construction specs ──────────────────────────────────────────
BASES = {
    "A": ["xs", "dual", "adapt"],     # champion 3-sleeve
    "B": ["res", "dual", "adapt"],    # E3 winner: residual ms_raw replaces xs
}
ALLOCS = ("st", "sx")
OVERLAYS = ("TF1", "T", "F1")

KEEP = ("mean_monthly", "sharpe", "max_drawdown", "calmar", "turnover_ann",
        "avg_gross", "ep_2018Q4_ret", "ep_2018Q4_dd", "ep_covid_ret",
        "ep_covid_dd", "ep_2022_ret", "ep_2022_dd", "ep_2026Q1_ret",
        "ep_2026Q1_dd")


def build_blend(bkey: str, akey: str) -> pd.DataFrame:
    if bkey == "A" and akey == "st":
        return pd.read_parquet(ROOT / "weights_store" /
                               "combo_v2_base_1x.parquet").loc[:cutoff]
    if bkey == "A" and akey == "sx":
        # E2 winner module, byte-identical to its G1 run.
        return build_e2_blend_1x(method="sharpe_softmax", lookback=126,
                                 floor=0.10, tau=1.0, end=DEV_END)
    sleeves = {k: load_sleeve(k) for k in BASES[bkey]}
    if akey == "st":
        return static_blend_1x(sleeves)
    return softmax_blend_1x(sleeves, lookback=126, floor=0.10, tau=1.0)


def cand_params(bkey, akey, ov, tc):
    return {
        "base": {"A": "champ3 (xs+dual+adapt n30)",
                 "B": "E3 resid ms_raw replxs (res+dual+adapt n30)"}[bkey],
        "alloc": {"st": "static 1/3",
                  "sx": "sharpe_softmax lb126 floor0.10 tau1.0"}[akey],
        "overlay": {"TF1": "tier(P-0.08,2d,phi0) + freeze dd_v1",
                    "T": "tier(P-0.08,2d,phi0)",
                    "F1": "freeze dd_v1"}[ov],
        "tier": ov in ("TF1", "T"),
        "freeze_dd_v1": ov in ("TF1", "F1"),
        "leverage": 2.0, "tc_bps": tc, "exec":
            "tier_loop_cc" if ov in ("TF1", "T") else "next_open",
    }


def main():
    rows = []
    for bkey in BASES:
        for akey in ALLOCS:
            tag = f"{bkey}{akey}"
            blend = build_blend(bkey, akey)
            env = TierEnv(blend)

            # Fidelity: stops-off loop == engine next_close on un-gated base.
            bt_nc = run_backtest_v2(env.W2_daily, pu_dev, cfg_close,
                                    name=f"nc_{tag}", weights_are_daily=True)
            net0, _, _, _ = env.tier_loop(env.W2_daily, None)
            diff = float(np.abs(net0 - bt_nc["returns"].to_numpy()).max())
            print(f"[{tag}] tier loop (P=None) vs next_close: {diff:.2e}")
            assert diff < 1e-12, f"fidelity FAIL for {tag}"

            # Anchor/comparator: un-gated base at 5bp (next_open, run_trial).
            r = run_trial(env.W2_daily, name=f"ASM_{tag}_base", family=FAMILY,
                          params=cand_params(bkey, akey, "F1", 5.0) | {
                              "overlay": "none (un-gated base anchor)"},
                          window="dev", exec_model="next_open", tc_bps=5.0,
                          leverage_cap=2.0, weights_are_daily=True,
                          notes="assembly anchor: un-gated base x alloc")
            rows.append({"cand": f"ASM_{tag}_base", "base": bkey,
                         "alloc": akey, "overlay": "none", "tc": 5.0,
                         "track": "next_open",
                         **{k: r["row"].get(k) for k in KEEP}})
            print(f"  ASM_{tag}_base       mm={r['row']['mean_monthly']*100:6.3f}% "
                  f"dd={r['row']['max_drawdown']*100:6.1f}% "
                  f"calmar={r['row']['calmar']:.3f}")

            W_f1 = apply_gate(env.W2_daily, F1_MULT)
            for ov in OVERLAYS:
                W = env.W2_daily if ov == "T" else W_f1
                for tc in TCS:
                    name = f"ASM_{tag}_{ov}" + ("" if tc == 5.0 else f"_tc{int(tc)}")
                    params = cand_params(bkey, akey, ov, tc)
                    if ov == "F1":
                        r = run_trial(W, name=name, family=FAMILY,
                                      params=params, window="dev",
                                      exec_model="next_open", tc_bps=tc,
                                      leverage_cap=2.0, weights_are_daily=True,
                                      notes="assembly candidate, freeze-only")
                        row = r["row"]
                        track = "next_open"
                        ev = None
                    else:
                        net, ev, traded, gross = env.tier_loop(W, P_TIER,
                                                               DELAY, tc)
                        row = tier_row(name, net, ev, traded, gross, params,
                                       tc, "assembly candidate, cc tier loop;")
                        track = "tier_loop_cc"
                    rec = {"cand": name, "base": bkey, "alloc": akey,
                           "overlay": ov, "tc": tc, "track": track,
                           **{k: row.get(k) for k in KEEP}}
                    if ev is not None:
                        rec["tier_events"] = json.dumps(ev)
                    rows.append(rec)
                    print(f"  {name:22s} mm={row['mean_monthly']*100:6.3f}% "
                          f"sharpe={row['sharpe']:.3f} "
                          f"dd={row['max_drawdown']*100:6.1f}% "
                          f"calmar={row['calmar']:.3f} [{track}]")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "v7" / "ASSEMBLY_grid_results.csv"
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {out}")

    # ── Gate G2 ──────────────────────────────────────────────────────────
    champ_covid = float(df.loc[df["cand"] == "ASM_Ast_base", "ep_covid_dd"].iloc[0])
    c5 = df[(df["tc"] == 5.0) & (df["overlay"] != "none")].set_index("cand")
    verdicts = []
    for cand, r in c5.iterrows():
        c20 = df[(df["cand"].str.startswith(cand + "_tc20")) |
                 ((df["cand"] == cand + "_tc20"))]
        cal20 = float(c20["calmar"].iloc[0])
        drop = 1.0 - cal20 / r["calmar"]
        v = {
            "cand": cand,
            "mean_ok": r["mean_monthly"] >= 0.045,
            "dd_ok": r["max_drawdown"] >= -0.40,
            "calmar_ok": r["calmar"] >= 1.3,
            "turn_ok": r["turnover_ann"] <= 30.0,
            "tc20_calmar": cal20, "tc20_drop": drop,
            "tc20_ok": drop < 0.25,
            "covid_ok": (r["ep_covid_dd"] >= champ_covid - 0.05),
        }
        v["G2"] = all(v[k] for k in
                      ("mean_ok", "dd_ok", "calmar_ok", "turn_ok",
                       "tc20_ok", "covid_ok"))
        verdicts.append(v)
    vdf = pd.DataFrame(verdicts).set_index("cand")
    vout = ROOT / "results" / "v7" / "ASSEMBLY_G2_verdicts.csv"
    vdf.to_csv(vout)
    print(f"\nG2 verdicts -> {vout}")
    print(vdf.to_string())
    print(f"\ntotal logged trials (all families): {trial_count()}")


if __name__ == "__main__":
    main()
