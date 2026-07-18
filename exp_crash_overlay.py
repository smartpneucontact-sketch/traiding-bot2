"""exp_crash_overlay.py — S6: CRASH1 dev grid (results/v7/crash/CRASH1_PREREG.md).

Runs EXACTLY the 8 pre-registered cells (2 bear defs x 2 responses x 2
vol-thresholds) — no extras, no extensions. Family CRASH1, dev-only
(<= 2022-12-31), margin_bps_annual=600, tc=5bp, leverage_cap 2.0.

Base frame (prereg "Base frame"): the candidate-X construction
(ASM_Bst_BTF1's Bst blend + book gate + tier) with the FREEZE LAYER REMOVED
and the crash overlay in its place via engine_v2.apply_gate:

    W_cell = apply_gate( apply_gate(W2_daily, g_book), crash_mult )
    -> generalized tier loop (assemble_candidates.TierEnv, P=-0.08, delay 2)
    -> margin-600 drag replicated post-loop (exp_rescore600.margin_net —
       IDENTICAL convention to the comparator run, like-for-like).

Grid factors (prereg "Grid"):
  bear (2):  b24m = SPY trailing 24-month total return < 0
             b200 = SPY close < its 200-day moving average
  resp (2):  bin  = de-lever to 0.5x while bear
             dm   = continuous DM-style scale while bear:
                    clip(target_vol / trailing 126d realized vol, 0, 1)
  vth  (2):  v0   = off (bear def alone)
             v1   = bear additionally requires trailing 63d SPY realized vol
                    > its rolling 75th percentile

Implementation fixations (chosen BEFORE any cell ran, not tuned; documented
because the prereg leaves them open — no other values were tried):
  - "24-month" trailing return = 504 trading days on daily closes
    (24 x 21, the program's rebalance-calendar convention).
  - 200dma = rolling(200, min_periods=200).mean() of ffilled SPY closes.
  - DM response vol = trailing 126d realized vol (prereg-fixed lookback) of
    the PASS-1 dev net returns of the freeze-removed base itself (book-gated
    frame, engine next_open, tc=5bp, margin 0 — signal input only, E1/E5
    precedent of building multipliers from pass-1 strategy returns);
    target_vol = 0.30 (the E1-winner / E5 "V" convention for this 2x book,
    ledgered precedent); clip to [0, 1] (floor=None), one-day lag inside
    engine_v2.vol_managed_multiplier. Multiplier = 1.0 outside bear state.
  - Vol threshold percentile: "rolling 75th percentile" implemented as the
    expanding (all trailing history) 75th percentile, min_periods=252 — the
    parameter-free reading; NaN (burn-in) -> threshold not met.
  - All signals are computed on close-t data and applied to the DECISION-side
    frame; the engine/tier-loop shift(1) lags them one day (same convention
    as the book gate and freeze layers). NaN bear signals -> not bear.

Comparator (prereg "Dev gates"): ASM_Bst_BTF1 at margin 600, read from the
S1 ledger results/v7/trials/ASM_rescore600.csv (run exp_rescore600.py first).

Dev gates (frozen in the prereg): winner = top dev Calmar cell;
  G-cal: dev Calmar >= 1.05 x comparator dev Calmar
  G-mm : dev mean_monthly >= comparator mean − 0.5pp/mo
  G-nbr: median Calmar of the winner's 3 adjacent cells (Hamming distance 1
         in the bear/resp/vth cube) >= 0.85 x winner Calmar
Winner-or-none: the gates are evaluated on the top-Calmar cell only (no
cell shopping). If it fails -> family closed, no val shot.

HARD STOP: dev-only. No final=True, no window='val'. If the winner passes
all dev gates this script STOPS and reports — the one-shot validation is a
separate orchestrator decision (prereg "Consequences").
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine_v2 import (BTConfigV2, run_backtest_v2, apply_gate,  # noqa: E402
                       vol_managed_multiplier)
from assemble_candidates import (P_TIER, DELAY, macro, cutoff, idx,  # noqa: E402
                                 pu_dev, opn_dev)
from exp_rescore600 import (build_candidate_env, verify_reproduction,  # noqa: E402
                            margin_net, tier_row, file_sha256, MARGIN_BPS)

FAMILY = "CRASH1"
LEDGER = ROOT / "results" / "v7" / "trials" / f"{FAMILY}.csv"
PREREG = ROOT / "results" / "v7" / "crash" / "CRASH1_PREREG.md"
OUT_MD = ROOT / "results" / "v7" / "crash" / "CRASH1_DEV_RESULTS.md"
COMP_LEDGER = ROOT / "results" / "v7" / "trials" / "ASM_rescore600.csv"

DM_TARGET_VOL = 0.30   # E1-winner/E5 "V" convention for the 2x book (fixed)
DM_LOOKBACK = 126      # prereg-fixed
BIN_LEVEL = 0.5        # prereg-fixed
VOL_LB = 63            # prereg-fixed
VOL_PCTL = 0.75        # prereg-fixed
RET_24M_DAYS = 504     # 24 x 21 trading days (fixation, see docstring)
DMA_DAYS = 200         # prereg-fixed

BEARS = ("b24m", "b200")
RESPS = ("bin", "dm")
VTHS = ("v0", "v1")


def build_signals(env, g_book):
    """All overlay input series on the dev index (see fixations above)."""
    spy = macro["SPY"].ffill()
    bear = {
        "b24m": (spy / spy.shift(RET_24M_DAYS) - 1.0) < 0.0,
        "b200": spy < spy.rolling(DMA_DAYS, min_periods=DMA_DAYS).mean(),
    }
    bear = {k: v.reindex(idx).ffill().fillna(False).astype(bool)
            for k, v in bear.items()}

    spy_ret = spy.pct_change(fill_method=None)
    rv63 = spy_ret.rolling(VOL_LB).std() * np.sqrt(252.0)
    thresh = rv63.expanding(min_periods=252).quantile(VOL_PCTL)
    vhigh = (rv63 > thresh).reindex(idx).ffill().fillna(False).astype(bool)

    # PASS-1: freeze-removed base (book-gated) dev net returns -> DM vol scale.
    W_bt = apply_gate(env.W2_daily, g_book)
    cfg = BTConfigV2(tc_bps=5.0, leverage_cap=2.0, exec_model="next_open")
    bt = run_backtest_v2(W_bt, pu_dev, cfg, name="crash_pass1_bookgated_dev",
                         open_prices=opn_dev, weights_are_daily=True)
    m_dm = vol_managed_multiplier(bt["returns"], DM_TARGET_VOL,
                                  lookback=DM_LOOKBACK, cap=1.0, floor=None)
    m_dm = m_dm.reindex(idx).fillna(1.0)
    return bear, vhigh, m_dm, W_bt


def crash_multiplier(bear_eff: pd.Series, resp: str, m_dm: pd.Series) -> pd.Series:
    if resp == "bin":
        return pd.Series(np.where(bear_eff, BIN_LEVEL, 1.0), index=idx)
    return pd.Series(np.where(bear_eff, m_dm.to_numpy(), 1.0), index=idx)


def neighbors(cell: tuple[str, str, str]) -> list[tuple[str, str, str]]:
    """The 3 cells at Hamming distance 1 in the (bear, resp, vth) cube."""
    b, r, v = cell
    out = [(b2, r, v) for b2 in BEARS if b2 != b]
    out += [(b, r2, v) for r2 in RESPS if r2 != r]
    out += [(b, r, v2) for v2 in VTHS if v2 != v]
    return out


def main():
    prereg_sha = file_sha256(PREREG)
    print(f"CRASH1_PREREG.md sha256 = {prereg_sha}")

    if not COMP_LEDGER.exists():
        raise SystemExit("comparator ledger missing — run exp_rescore600.py (S1) first")
    comp_df = pd.read_csv(COMP_LEDGER)
    comp = comp_df[comp_df["name"] == "ASM_Bst_BTF1"].iloc[-1]
    assert float(comp["margin_bps"]) == MARGIN_BPS, "comparator is not margin600"
    print(f"[comparator] ASM_Bst_BTF1 @margin600: mm={comp['mean_monthly']*100:.3f}% "
          f"dd={comp['max_drawdown']*100:.1f}% calmar={comp['calmar']:.3f}")

    # Construction guard: the freeze-INCLUDED frame must still reproduce the
    # ledgered ASM_Bst_BTF1 margin-0 row (proves env + g_book are exact).
    env, g_book, W_btf1 = build_candidate_env()
    verify_reproduction(env, W_btf1)

    bear, vhigh, m_dm, W_bt = build_signals(env, g_book)
    for k, s in bear.items():
        print(f"[signal] {k}: {int(s.sum())} bear days "
              f"({int((s & vhigh).sum())} with vol-threshold)")
    print(f"[signal] vhigh: {int(vhigh.sum())} days; "
          f"m_dm on bear-capable days: min={m_dm.min():.3f} "
          f"mean={m_dm.mean():.3f}")

    results = {}
    for b, r, v in itertools.product(BEARS, RESPS, VTHS):
        name = f"CRASH1_{b}_{r}_{v}"
        bear_eff = bear[b] & vhigh if v == "v1" else bear[b]
        mult = crash_multiplier(bear_eff, r, m_dm)
        W = apply_gate(W_bt, mult)
        net, ev, traded, gross = env.tier_loop(W, P_TIER, DELAY, 5.0)
        net_m = margin_net(net, gross)
        params = {
            "base": "ASM_Bst_BTF1 minus freeze (Bst blend + book(0.12,0.30,60) "
                    "+ tier(P-0.08,2d,phi0))",
            "bear": {"b24m": f"SPY {RET_24M_DAYS}d trailing ret < 0",
                     "b200": f"SPY < {DMA_DAYS}dma"}[b],
            "resp": {"bin": f"binary {BIN_LEVEL}x while bear",
                     "dm": f"clip({DM_TARGET_VOL}/rv{DM_LOOKBACK}(pass1 net),0,1) "
                           f"while bear"}[r],
            "vol_threshold": v == "v1",
            "vol_threshold_def": f"SPY rv{VOL_LB} > expanding p{int(VOL_PCTL*100)} "
                                 f"(min_periods=252)",
            "leverage": 2.0, "tc_bps": 5.0, "exec": "tier_loop_cc",
            "margin_bps_annual": MARGIN_BPS,
            "bear_days_effective": int(bear_eff.sum()),
        }
        row = tier_row(name, net_m, ev, traded, gross, params, 5.0,
                       notes=(f"CRASH1 prereg cell (CRASH1_PREREG.md sha256="
                              f"{prereg_sha[:16]}..); freeze removed, crash "
                              f"overlay in its place; margin600 post-loop;"),
                       family=FAMILY, ledger=LEDGER)
        row["bear_days"] = int(bear_eff.sum())
        row["tier_events"] = json.dumps(ev)
        results[(b, r, v)] = row
        print(f"{name:22s} mm={row['mean_monthly']*100:6.3f}% "
              f"dd={row['max_drawdown']*100:6.1f}% calmar={row['calmar']:.3f} "
              f"turn={row['turnover_ann']:5.1f} bear_days={row['bear_days']:4d} "
              f"events={ev}")

    # ── Frozen dev gates on the top-Calmar cell ──────────────────────────
    comp_cal, comp_mm = float(comp["calmar"]), float(comp["mean_monthly"])
    win_key = max(results, key=lambda k: results[k]["calmar"])
    win = results[win_key]
    nbr_cals = [results[n]["calmar"] for n in neighbors(win_key)]
    med_nbr = float(np.median(nbr_cals))

    g_cal = win["calmar"] >= 1.05 * comp_cal
    g_mm = win["mean_monthly"] >= comp_mm - 0.005
    g_nbr = med_nbr >= 0.85 * win["calmar"]
    dev_pass = g_cal and g_mm and g_nbr
    win_name = f"CRASH1_{'_'.join(win_key)}"

    print("\n" + "=" * 72)
    print(f"winner (top dev Calmar): {win_name}")
    print(f"  G-cal: {win['calmar']:.3f} vs 1.05x{comp_cal:.3f}={1.05*comp_cal:.3f}"
          f" -> {'PASS' if g_cal else 'FAIL'}")
    print(f"  G-mm : {win['mean_monthly']*100:.3f}%/mo vs "
          f"{(comp_mm-0.005)*100:.3f}%/mo -> {'PASS' if g_mm else 'FAIL'}")
    print(f"  G-nbr: median nbr Calmar {med_nbr:.3f} vs "
          f"0.85x{win['calmar']:.3f}={0.85*win['calmar']:.3f}"
          f" -> {'PASS' if g_nbr else 'FAIL'}")
    verdict_txt = ("PASS — STOP; one-shot val is a separate orchestrator "
                   "decision" if dev_pass
                   else "FAIL — family closed, no val shot")
    print(f"CRASH1 DEV VERDICT: {verdict_txt}")
    print("=" * 72)

    # ── Results artifact ─────────────────────────────────────────────────
    lines = [
        f"# CRASH1 dev grid results ({pd.Timestamp.now().date()})",
        "",
        f"Prereg: `results/v7/crash/CRASH1_PREREG.md` sha256 `{prereg_sha}`.",
        "Exactly the 8 pre-registered cells; family `CRASH1`, dev-only,",
        "tier_loop_cc track, tc=5bp, leverage_cap 2.0, margin 600bp/yr",
        "(post-loop drag, identical convention to the comparator — see",
        "`exp_rescore600.py`). Ledger: `results/v7/trials/CRASH1.csv`.",
        "",
        "Base: ASM_Bst_BTF1 construction with the freeze layer REMOVED and the",
        "crash overlay in its place via `engine_v2.apply_gate`. Construction",
        "guard: the freeze-included frame reproduced the ledgered ASM_Bst_BTF1",
        "margin-0 row before any cell ran.",
        "",
        "Implementation fixations (chosen before any cell ran, not tuned —",
        "see `exp_crash_overlay.py` docstring): 24mo = 504 trading days;",
        f"DM scale = clip({DM_TARGET_VOL}/rv126(pass-1 book-gated base net),0,1),",
        "1.0 outside bear; vol threshold = SPY rv63 > expanding 75th pctile",
        "(min_periods=252); all gates decision-side (one-day engine lag).",
        "",
        f"Comparator (S1, `results/v7/trials/ASM_rescore600.csv`): ASM_Bst_BTF1",
        f"@margin600 — mean {comp_mm*100:.3f}%/mo, MaxDD "
        f"{comp['max_drawdown']*100:.1f}%, Calmar {comp_cal:.3f}.",
        "",
        "## Cells",
        "",
        "| cell | bear def | resp | vth | bear days | mean/mo | MaxDD | Calmar |"
        " turnover | avg_gross | tier events |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (b, r, v), row in results.items():
        star = " **(winner)**" if (b, r, v) == win_key else ""
        lines.append(
            f"| CRASH1_{b}_{r}_{v}{star} | {b} | {r} | {v} | {row['bear_days']} "
            f"| {row['mean_monthly']*100:.3f}% | {row['max_drawdown']*100:.1f}% "
            f"| {row['calmar']:.3f} | {row['turnover_ann']:.1f} "
            f"| {row['avg_gross']:.3f} | {row['tier_events']} |")
    lines += [
        "",
        "## Frozen dev gates (evaluated on the top-Calmar cell only)",
        "",
        f"Winner: **{win_name}** (dev Calmar {win['calmar']:.3f}, mean "
        f"{win['mean_monthly']*100:.3f}%/mo, MaxDD {win['max_drawdown']*100:.1f}%).",
        "",
        "| gate | requirement | value | verdict |",
        "|---|---|---|---|",
        f"| G-cal | Calmar >= 1.05 x {comp_cal:.3f} = {1.05*comp_cal:.3f} "
        f"| {win['calmar']:.3f} | {'PASS' if g_cal else 'FAIL'} |",
        f"| G-mm | mean >= {comp_mm*100:.3f} − 0.5 = {(comp_mm-0.005)*100:.3f}%/mo "
        f"| {win['mean_monthly']*100:.3f}%/mo | {'PASS' if g_mm else 'FAIL'} |",
        f"| G-nbr | median neighbor Calmar >= 0.85 x {win['calmar']:.3f} = "
        f"{0.85*win['calmar']:.3f} | {med_nbr:.3f} (neighbors: "
        + ", ".join(f"{c:.3f}" for c in nbr_cals) + f") | {'PASS' if g_nbr else 'FAIL'} |",
        "",
        f"## Verdict: {'WINNER — ' + win_name if dev_pass else 'NONE'}",
        "",
    ]
    if dev_pass:
        lines += [
            f"`{win_name}` passes all three frozen dev gates. STOPPING HERE:",
            "the validation window has NOT been touched. The one-shot",
            "validation (gates frozen in CRASH1_PREREG.md: val Calmar >= 1.62,",
            "val MaxDD >= −35%, 2026Q1 DD <= −27.2%) is a separate",
            "orchestrator decision; exactly one winner may take it, and",
            "failing it burns the family permanently.",
        ]
    else:
        lines += [
            "The top-Calmar cell fails the frozen dev gates (winner-or-none;",
            "no cell shopping). Per the prereg: fail dev -> no val shot;",
            "family CRASH1 closed. The freeze layer remains in candidate X v1",
            "as DISPUTED under its pre-registered ablation adjudication.",
        ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_MD}")
    return results, win_key, dev_pass


if __name__ == "__main__":
    main()
