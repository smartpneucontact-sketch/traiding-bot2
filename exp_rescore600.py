"""exp_rescore600.py — S1: margin-600 re-score of candidate X (DECISION_RULE R6).

Executes rule R6 of results/v7/candidate_x/DECISION_RULE.md (FROZEN 2026-07-18):
re-score the selected construction ASM_Bst_BTF1 dev-only at
margin_bps_annual=600, family ASM_rescore600, before any bundle build.

Runs (all dev-only, window <= 2022-12-31, margin 600bp/yr):
  (a) ASM_Bst_BTF1        — the R2 selection, EXACT ledgered construction
                            (assemble_added_cell.py: Bst blend x book gate x
                            freeze dd_v1 -> generalized tier loop, tc=5bp).
                            This same row is the CRASH1 comparator
                            (CRASH1_PREREG.md "Dev gates" section) — one run
                            serves both S1 and S6.
  (b) combo_v2_2x         — champion dev context row (next_open, run_trial).

R6 fall-through check (reported PASS/FAIL loudly, nothing else changes):
  margin600 dev Calmar >= 1.0 AND dev MaxDD >= -45%.

Margin convention on the tier track
-----------------------------------
The tier loop (assemble_candidates.TierEnv.tier_loop, close-to-close
daily-bar approximation) does not run through engine_v2, so the engine's
margin leg is replicated post-loop, additively and without touching the
frozen loop code:

    drag_d = max(gross_held_d - 1.0, 0.0) * (600/10_000) / 252
    net_margin_d = net_d - drag_d

where gross_held_d is the loop's held-book gross during day d (0 on tier-3
cash days and on the close re-entry day). This mirrors engine_v2's charge on
the EFFECTIVE (held) book's long gross above 1.0x NAV, ACT/252. Long-only
book (shorts clipped) so long gross == gross. On intraday tier-cut days the
full start-of-day gross is charged for the whole day — a conservative
(cost-overstating) approximation, identical across every row scored on this
track (candidate X here and all CRASH1 cells), so comparisons are like-for-
like.

Ledger: results/v7/trials/ASM_rescore600.csv (NEW family — exp_lib mandates a
new family name for margin-on rows; the ASSEMBLY/E5 rows predate the margin
convention). Manual tier rows mirror exp_lib.run_trial's schema + margin_bps
column and are header-aligned on append. Reproduction of the ledgered
margin-0 row is asserted before the margin row is written.

Dev-only: no final=True anywhere; the validation window is never touched.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine_v2 import book_drawdown_gate, apply_gate  # noqa: E402
from assemble_candidates import (TierEnv, build_blend, cand_params,  # noqa: E402
                                 F1_MULT, P_TIER, DELAY, panel, cutoff, idx)
from exp_lib import run_trial  # noqa: E402
from metrics import summary as metrics_summary  # noqa: E402
from metrics_v2 import crash_table  # noqa: E402

FAMILY = "ASM_rescore600"
MARGIN_BPS = 600.0
INIT = 100_000.0
LEDGER = ROOT / "results" / "v7" / "trials" / f"{FAMILY}.csv"
OUT_MD = ROOT / "results" / "v7" / "candidate_x" / "RESCORE600_RESULTS.md"
DECISION_RULE = ROOT / "results" / "v7" / "candidate_x" / "DECISION_RULE.md"

# Ledgered margin-0 reference row (results/v7/trials/ASSEMBLY_tier.csv,
# name=ASM_Bst_BTF1, tc=5) — reproduction is asserted before margin applies.
REF_M0 = {"mean_monthly": 0.042758, "calmar": 1.233776, "max_drawdown": -0.417449}

# R6 fall-through bars (DECISION_RULE.md, frozen 2026-07-18).
R6_CALMAR_MIN = 1.0
R6_MAXDD_MIN = -0.45


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def margin_net(net: np.ndarray, gross_held: np.ndarray,
               margin_bps: float = MARGIN_BPS) -> np.ndarray:
    """engine_v2 margin leg replicated on tier-loop outputs (see module doc)."""
    drag = np.clip(gross_held - 1.0, 0.0, None) * (margin_bps / 10_000.0) / 252.0
    return net - drag


def tier_metrics(net: np.ndarray) -> tuple[pd.Series, dict]:
    eq = pd.Series((1.0 + pd.Series(net, index=idx)).cumprod() * INIT, index=idx)
    return eq, metrics_summary(eq)


def ledger_append(row: dict, ledger: Path = LEDGER) -> None:
    """Append a manual row mirroring exp_lib.run_trial's append conventions:
    header-aligned, hard error on columns the on-file header lacks."""
    row_df = pd.DataFrame([row])
    if ledger.exists():
        with open(ledger) as f:
            header = f.readline().strip().split(",")
        extra = [c for c in row_df.columns if c not in header]
        if extra:
            raise ValueError(f"ledger {ledger.name} header lacks {extra}")
        row_df = row_df.reindex(columns=header)
    row_df.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)


def tier_row(name: str, net_m: np.ndarray, ev: dict, traded: np.ndarray,
             gross_held: np.ndarray, params: dict, tc_bps: float,
             notes: str, family: str = FAMILY,
             ledger: Path = LEDGER) -> dict:
    """Manual margin-track ledger row, schema = run_trial(margin!=0) rows."""
    eq, s = tier_metrics(net_m)
    row = {
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "family": family, "name": name, "window": "dev",
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
        "margin_bps": MARGIN_BPS,
    }
    for ep, st in crash_table(eq).items():
        row[f"ep_{ep}_ret"] = st["ret"]
        row[f"ep_{ep}_dd"] = st["max_dd"]
    ledger_append(row, ledger)
    return row


def build_candidate_env() -> tuple[TierEnv, pd.Series, pd.DataFrame]:
    """EXACT ledgered ASM_Bst_BTF1 construction (assemble_added_cell.py):
    Bst blend -> TierEnv; book gate on the un-gated base daily frame; the
    fully-composed frame W = base x g_book x freeze_dd_v1 feeds the tier loop.
    Returns (env, g_book, W_btf1)."""
    blend = build_blend("B", "st")
    env = TierEnv(blend)
    g_book = book_drawdown_gate(env.W2_daily, panel["close"].loc[:cutoff],
                                full_dd=0.12, cash_dd=0.30, lookback=60)
    g_book = g_book.reindex(idx).fillna(1.0).clip(0.0, 1.0)
    W = apply_gate(apply_gate(env.W2_daily, g_book), F1_MULT)
    return env, g_book, W


def verify_reproduction(env: TierEnv, W: pd.DataFrame):
    """Assert the margin-0 tier run reproduces the ledgered ASM_Bst_BTF1 row
    (trials/ASSEMBLY_tier.csv) before any margin row is written. Returns the
    loop outputs so the margin row reuses the identical run."""
    net, ev, traded, gross = env.tier_loop(W, P_TIER, DELAY, 5.0)
    _, s0 = tier_metrics(net)
    print(f"[repro] margin-0: mm={s0['mean_monthly']*100:.4f}% "
          f"calmar={s0['calmar']:.6f} dd={s0['max_drawdown']*100:.4f}% "
          f"events={ev}")
    for k, ref in REF_M0.items():
        got = s0[k]
        assert abs(got - ref) < 5e-6, \
            f"reproduction FAIL: {k} = {got:.6f}, ledgered {ref:.6f}"
    assert ev == {"tier1": 14, "tier2": 0, "tier3": 0}, f"tier events differ: {ev}"
    print("[repro] PASS — ledgered ASM_Bst_BTF1 (margin 0) reproduced")
    return net, ev, traded, gross


def main():
    dr_sha = file_sha256(DECISION_RULE)
    print(f"DECISION_RULE.md sha256 = {dr_sha}")

    env, _g_book, W = build_candidate_env()
    net0, ev, traded, gross = verify_reproduction(env, W)

    # (a) ASM_Bst_BTF1 at margin 600 — selection re-score AND CRASH1 comparator.
    net_m = margin_net(net0, gross)
    params = cand_params("B", "st", "TF1", 5.0)
    params["overlay"] = "book(0.12,0.30,60) + tier(P-0.08,2d,phi0) + freeze dd_v1"
    params["added_cell"] = True
    params["margin_bps_annual"] = MARGIN_BPS
    notes = (f"R6 margin re-score of the R2 selection (DECISION_RULE.md "
             f"sha256={dr_sha[:16]}..); margin drag replicated post-loop on "
             f"gross_held (engine_v2 convention); doubles as CRASH1 comparator;")
    row_x = tier_row("ASM_Bst_BTF1", net_m, ev, traded, gross, params, 5.0, notes)
    print(f"\nASM_Bst_BTF1 @margin600  mm={row_x['mean_monthly']*100:6.3f}% "
          f"dd={row_x['max_drawdown']*100:6.1f}% calmar={row_x['calmar']:.3f} "
          f"turn={row_x['turnover_ann']:.1f} gross={row_x['avg_gross']:.3f}")

    # (b) champion combo_v2_2x dev context row (next_open engine, margin 600).
    champ_w = pd.read_parquet(ROOT / "weights_store" /
                              "combo_v2_base_1x.parquet") * 2.0
    r_ch = run_trial(
        champ_w, name="combo_v2_2x", family=FAMILY,
        params={"lev": 2.0, "source": "run_v6",
                "role": "champion dev context row (DECISION_RULE R6)"},
        window="dev", exec_model="next_open", tc_bps=5.0, leverage_cap=2.0,
        margin_bps_annual=MARGIN_BPS,
        notes=(f"context row per DECISION_RULE.md R6 (sha256={dr_sha[:16]}..); "
               f"expect ~= catalog_v2 dev margin600 row"))
    row_ch = r_ch["row"]
    print(f"combo_v2_2x  @margin600  mm={row_ch['mean_monthly']*100:6.3f}% "
          f"dd={row_ch['max_drawdown']*100:6.1f}% calmar={row_ch['calmar']:.3f}")
    # Cross-check against the already-ledgered catalog_v2 dev margin600 row.
    cat = pd.read_csv(ROOT / "results" / "v7" / "trials" / "catalog_v2.csv")
    cref = cat[(cat["name"] == "combo_v2_2x") & (cat["window"] == "dev")].iloc[-1]
    dmm = abs(row_ch["mean_monthly"] - cref["mean_monthly"])
    print(f"[check] vs catalog_v2 dev margin600: |d mean_monthly| = {dmm:.2e}")
    assert dmm < 5e-6, "champion context row does not match catalog_v2 ledger"

    # ── R6 fall-through check ────────────────────────────────────────────
    calmar_ok = row_x["calmar"] >= R6_CALMAR_MIN
    dd_ok = row_x["max_drawdown"] >= R6_MAXDD_MIN
    verdict = "PASS" if (calmar_ok and dd_ok) else "FAIL"
    banner = (f"R6 FALL-THROUGH CHECK: {verdict}  "
              f"(Calmar {row_x['calmar']:.3f} {'>=' if calmar_ok else '<'} "
              f"{R6_CALMAR_MIN:.1f}; MaxDD {row_x['max_drawdown']*100:.1f}% "
              f"{'>=' if dd_ok else '<'} {R6_MAXDD_MIN*100:.0f}%)")
    print("\n" + "=" * 72 + f"\n{banner}\n" + "=" * 72)
    if verdict == "FAIL":
        print("R6: selection degrades below the margin600 bars — fall to the "
              "next-ranked eligible row (orchestrator decision; the DECISION_"
              "RULE application table shows every other row already fails "
              "eligibility).")

    # ── Results artifact ─────────────────────────────────────────────────
    m0 = tier_metrics(net0)[1]
    md = f"""# ASM_rescore600 — R6 margin re-score results ({pd.Timestamp.now().date()})

Executes R6 of DECISION_RULE.md (sha256 `{dr_sha}`), dev-only, family
`ASM_rescore600`, ledger `results/v7/trials/ASM_rescore600.csv`.
Margin convention on the tier track: engine_v2 leg replicated post-loop,
`drag = max(gross_held-1,0) * 600bp/252` daily (see exp_rescore600.py
docstring; conservative on intraday tier-cut days, identical for all rows
on this track including the CRASH1 cells).

| row | track | margin | dev mean/mo | dev MaxDD | dev Calmar | turnover | avg_gross |
|---|---|---|---|---|---|---|---|
| ASM_Bst_BTF1 (ledgered ref) | tier_loop_cc | 0 | {REF_M0['mean_monthly']*100:.3f}% | {REF_M0['max_drawdown']*100:.1f}% | {REF_M0['calmar']:.3f} | 20.8 | 1.414 |
| ASM_Bst_BTF1 (repro, this run) | tier_loop_cc | 0 | {m0['mean_monthly']*100:.3f}% | {m0['max_drawdown']*100:.1f}% | {m0['calmar']:.3f} | {float(traded.mean()*252):.1f} | {float(gross.mean()):.3f} |
| **ASM_Bst_BTF1 (selection)** | tier_loop_cc | **600** | **{row_x['mean_monthly']*100:.3f}%** | **{row_x['max_drawdown']*100:.1f}%** | **{row_x['calmar']:.3f}** | {row_x['turnover_ann']:.1f} | {row_x['avg_gross']:.3f} |
| combo_v2_2x (champion context) | next_open | 600 | {row_ch['mean_monthly']*100:.3f}% | {row_ch['max_drawdown']*100:.1f}% | {row_ch['calmar']:.3f} | {row_ch['turnover_ann']:.1f} | {row_ch['avg_gross']:.3f} |

Margin-0 reproduction asserted against `trials/ASSEMBLY_tier.csv` (tolerance
5e-6 on mean/Calmar/MaxDD; tier events identical). Champion context row
asserted equal to the `catalog_v2` dev margin600 row.

## {banner}

- Calmar bar (>= {R6_CALMAR_MIN:.1f}): {row_x['calmar']:.3f} -> {'PASS' if calmar_ok else 'FAIL'}
- MaxDD bar (>= {R6_MAXDD_MIN*100:.0f}%): {row_x['max_drawdown']*100:.1f}% -> {'PASS' if dd_ok else 'FAIL'}

Per R6, no other outcome of this re-score alters the selection.
`backtest_reference` in the bundle must use ONLY the margin600 row above.
The margin600 ASM_Bst_BTF1 row is the frozen CRASH1 comparator
(CRASH1_PREREG.md, "Dev gates").

Episodes (margin600 selection): 2018Q4 {row_x['ep_2018Q4_dd']*100:.1f}%, covid \
{row_x['ep_covid_dd']*100:.1f}%, 2022 {row_x['ep_2022_dd']*100:.1f}% (max in-window DD).
"""
    OUT_MD.write_text(md)
    print(f"\nwrote {OUT_MD}")
    return row_x, row_ch, verdict


if __name__ == "__main__":
    main()
