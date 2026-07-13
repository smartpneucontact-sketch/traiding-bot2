"""FINAL VALIDATION — one-shot evaluation of the locked window (V7 program).

This is the ONLY evaluation of 2023-01-01..2026-03-27. Protocol (binding):

  1. FIDELITY ANCHORS: rebuild ASM_Bst_TF1 / ASM_Bst_BTF1 / ASM_Bst_T and the
     corrected champion (combo_v2_base_1x x 2.0) on the FULL index, slice to
     dev (<= 2022-12-31), reproduce the assembly's dev numbers (full-precision
     anchor values pulled from results/v7/trials/ASSEMBLY_tier.csv and
     results/v7/baseline.json). Tolerance 1e-3 per metric. HARD STOP on fail.
  2. VALIDATION WINDOW: equity built on the full index (lookback burn-in),
     sliced at 2023-01-01 and renormalized to 100k (run_trial "val"
     convention), tc=5bp. + 2026Q1 episode.
  3. FULL WINDOW at tc in {5,10,20}bp.
  4. TIER PHI-STRESS (full window, tc=5): phi in {0.5, 1.0}, convention ported
     verbatim from validation/stop_grid.py simulate() (phi blends the day's
     per-ticker lows into the intraday tier-breach check: simultaneous-lows
     basket bound).
  5. STATISTICS: paired monthly diffs vs champion -> metrics_v2.newey_west_tstat
     (lag 3); deflated Sharpe (n_trials=450, cross-trial SR variance from the
     pre-existing ledger's annual-Sharpe column converted to daily scale).
  6. GATES G3 (validation) / G4 (full), reported PASS/FAIL with no re-tuning.
  7. OUTPUTS: results/v7/final_validation.json,
     results/v7/FINAL_VALIDATION_REPORT.md (rendered from the json by
     write_report() — regenerable standalone via `--report-only`, which
     exits before the module-level data load and never recomputes; the
     report carries the selection-contamination disclosure: the champion
     benchmark anchoring the G3 gates was itself selected on the full
     window incl. 2023-2026, so val-window results are quasi-out-of-sample
     at the sleeve level); ledger rows: champion through
     exp_lib.run_trial (family=FINAL_VALIDATION, final=True for val/full);
     tier candidates through the manual ledger
     results/v7/trials/FINAL_VALIDATION_tier.csv mirroring run_trial's schema
     (E5/ASSEMBLY precedent — tier P&L cannot be expressed as a weight frame).

Machinery: assemble_candidates.py and exp_e5_overlays.py are DEV-BOUND at
module level, so this script re-instantiates the SAME logic on the full index:
TierEnv/align_union/static blend are copied verbatim from assemble_candidates
(only the module-level dev slicing removed); the tier loop body is the
assembly loop with the phi descent/recovery leg ported verbatim from
validation/stop_grid.py simulate() (phi=0 reduces exactly to the assembly
loop) plus a per-day event-depth record (P&L unchanged) so window slices of
the event counts are available.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine_v2 import (BTConfigV2, run_backtest_v2, expand_daily,  # noqa: E402
                       apply_gate, _clip_and_cap, book_drawdown_gate)
from exp_lib import load_cache, union_prices_cached, run_trial, trial_count  # noqa: E402
from metrics import summary as metrics_summary, monthly_returns, daily_returns  # noqa: E402
from metrics_v2 import (DEV_END, VAL_START, crash_table, episode_stats,  # noqa: E402
                        deflated_sharpe, newey_west_tstat)
from strategies import freeze_signal_spy_drawdown  # noqa: E402

FAMILY = "FINAL_VALIDATION"
TIER_LEDGER = ROOT / "results" / "v7" / "trials" / "FINAL_VALIDATION_tier.csv"
OUT_JSON = ROOT / "results" / "v7" / "final_validation.json"
OUT_MD = ROOT / "results" / "v7" / "FINAL_VALIDATION_REPORT.md"
INIT = 100_000.0
P_TIER, DELAY = -0.08, 2
TIER_F = {1: 0.6, 2: 0.3, 3: 0.0}
TCS = (5.0, 10.0, 20.0)
VAL_END = "2026-03-27"
N_TRIALS_DSR = 450  # protocol-fixed deflated-Sharpe trial count
ANCHOR_TOL = 1e-3

SLEEVE_FILES = ["sleeve_residual_mom_market_sector_raw_n30",
                "sleeve_dual_momentum_voltarget_n30",
                "sleeve_adaptive_voltarget_n30"]


# ── Report renderer (json -> md, NO recomputation) ────────────────────────
def _pc(x: float) -> str:
    """Percent with 2dp; '—' for missing."""
    return "—" if x is None else f"{float(x) * 100:.2f}%"


def _f3(x: float) -> str:
    return "—" if x is None else f"{float(x):.3f}"


def _ev(ev: dict | None) -> str:
    if not ev:
        return "—"
    return "/".join(str(int(ev[f"tier{k}"])) for k in (1, 2, 3))


def write_report(results: dict | None = None) -> Path:
    """Render results/v7/FINAL_VALIDATION_REPORT.md from the persisted
    results/v7/final_validation.json. Pure formatting — every number comes
    from the artifact; nothing is recomputed (the one-shot protocol forbids
    re-evaluating the locked window)."""
    if results is None:
        results = json.loads(OUT_JSON.read_text())
    meta = results["meta"]
    L: list[str] = []
    L.append("# FINAL VALIDATION REPORT — V7 program")
    L.append("")
    L.append(f"Rendered from `{OUT_JSON.relative_to(ROOT)}` "
             f"(run generated {meta['generated']}); no recomputation.")
    L.append("")
    L.append(f"- Protocol: {meta['protocol']}")
    L.append(f"- Full index: {meta['full_index'][0]} .. {meta['full_index'][1]} "
             f"({meta['n_days_full']} days); tc=5bp unless stated; "
             f"leverage cap 2.0")
    L.append(f"- Deflated-Sharpe trial count: {meta['dsr_n_trials']} "
             f"(ledger rows pre-run: {meta['sr_variance_n_ledger_rows_pre_run']})")
    L.append("- Metric provenance: numbers in the artifact predate the "
             "2026-07-12 metric fixes (deflated-Sharpe kurtosis term, LPM2 "
             "sortino, deduped trial counts) and are rendered as recorded; "
             "deltas are immaterial at skew=0/excess_kurt=0 inputs but the "
             "values are not regenerable byte-identically from current code.")
    if "HARD_STOP" in meta:
        L.append("")
        L.append(f"**HARD STOP: {meta['HARD_STOP']}**")

    L.append("")
    L.append("## Selection contamination (disclosure)")
    L.append("")
    L.append("The champion benchmark (`combo_v2_base_1x` × 2.0) that anchors "
             "the G3 relative gates (val Calmar ≥ champion, 2026Q1 DD ≤ "
             "champion) was itself selected on the FULL window — including "
             "2023-2026, the very window used here as 'validation'. The "
             "candidate sleeves' parameters were tuned on dev only, but the "
             "bar they are measured against has seen the answer key. "
             "Val-window results are therefore quasi-out-of-sample at the "
             "sleeve level, not a clean out-of-sample test of the "
             "candidate-vs-champion comparison.")

    # Step 1 — fidelity anchors
    s1 = results["step1_fidelity_anchors"]
    L.append("")
    L.append(f"## Step 1 — Fidelity anchors "
             f"({'PASS' if s1['pass'] else 'FAIL'}, tol {meta['anchor_tolerance']})")
    L.append("")
    L.append("| name | max abs diff | tier events dev (ref) | pass |")
    L.append("|---|---|---|---|")
    for nm, a in s1.items():
        if nm == "pass":
            continue
        md = max(float(v) for v in a["abs_diffs"].values())
        ev = (f"{tuple(a['tier_events_dev'])} ({tuple(a['tier_events_ref'])})"
              if "tier_events_dev" in a else "—")
        L.append(f"| {nm} | {md:.2e} | {ev} | "
                 f"{'PASS' if a['pass'] else 'FAIL'} |")

    # Step 2 — validation window
    # Steps 2-6 are absent from a HARD_STOP artifact — render what exists.
    s2 = results.get("step2_validation_window", {})
    if s2:
        L.append("")
        L.append(f"## Step 2 — Validation window {VAL_START}..{VAL_END} "
                 f"(tc=5bp)")
        L.append("")
        L.append("| name | mean/mo | sharpe | maxDD | calmar | worst mo | "
                 "hit | turn | gross | 26Q1 ret | 26Q1 dd | tier ev |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for nm, w in s2.items():
            L.append(f"| {nm} | {_pc(w['mean_monthly'])} | {_f3(w['sharpe'])} | "
                     f"{_pc(w['max_drawdown'])} | {_f3(w['calmar'])} | "
                     f"{_pc(w['worst_month'])} | {_pc(w['hit_rate'])} | "
                     f"{w['turnover_ann']:.1f} | {w['avg_gross']:.2f} | "
                     f"{_pc(w['ep_2026Q1_ret'])} | {_pc(w['ep_2026Q1_dd'])} | "
                     f"{_ev(w.get('tier_events'))} |")

    # Step 3 — full window at tc 5/10/20
    s3 = results.get("step3_full_window", {})
    if s3:
        L.append("")
        L.append("## Step 3 — Full window at tc 5/10/20bp")
        L.append("")
        L.append("| name | tc | mean/mo | sharpe | maxDD | calmar | "
                 "2018Q4 dd | covid dd | 2022 dd | 26Q1 dd |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for nm, d in s3.items():
            for tck, w in d.items():
                eps = w.get("episodes", {})
                epc = [_pc(eps.get(k, {}).get("max_dd"))
                       for k in ("2018Q4", "covid", "2022", "2026Q1")]
                L.append(f"| {nm} | {tck[2:]} | {_pc(w['mean_monthly'])} | "
                         f"{_f3(w['sharpe'])} | {_pc(w['max_drawdown'])} | "
                         f"{_f3(w['calmar'])} | {' | '.join(epc)} |")

    # Step 4 — phi stress
    s4 = results.get("step4_phi_stress", {})
    if s4:
        L.append("")
        L.append("## Step 4 — Tier phi-stress (full window, tc=5bp)")
        L.append("")
        L.append(f"Champion full-window Calmar (tc=5): "
                 f"{_f3(s4['champion_full_calmar_tc5'])}. "
                 f"Promotion risk (TF1 phi=1.0 Calmar < champion): "
                 f"{'YES' if s4['PROMOTION_RISK_TF1_phi1_below_champion'] else 'no'}.")
        L.append("")
        L.append("| name | phi | mean/mo | sharpe | maxDD | calmar | "
                 "calmar drop vs phi0 | dd widening (pp) | tier ev |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for nm, d in s4["tables"].items():
            for pk, w in d.items():
                drop = w.get("calmar_drop_vs_phi0")
                wide = w.get("dd_widening_pp")
                L.append(f"| {nm} | {pk[3:]} | {_pc(w['mean_monthly'])} | "
                         f"{_f3(w['sharpe'])} | {_pc(w['max_drawdown'])} | "
                         f"{_f3(w['calmar'])} | "
                         f"{_pc(drop) if drop is not None else '—'} | "
                         f"{f'{wide:+.2f}' if wide is not None else '—'} | "
                         f"{_ev(w.get('tier_events'))} |")

    # Step 5 — statistics
    s5 = results.get("step5_statistics", {})
    if s5:
        L.append("")
        L.append("## Step 5 — Statistics")
        L.append("")
        L.append("Paired monthly diffs vs champion (full window), "
                 "Newey-West lag 3:")
        L.append("")
        L.append("| name | mean diff /mo | NW t | n months |")
        L.append("|---|---|---|---|")
        for nm, w in s5["newey_west"].items():
            L.append(f"| {nm} | {float(w['mean_monthly_diff'])*100:+.3f}% | "
                     f"{float(w['nw_tstat_lag3']):+.3f} | {w['n_months']} |")
        L.append("")
        L.append(f"Deflated Sharpe (n_trials={meta['dsr_n_trials']}, "
                 f"cross-trial SR var daily="
                 f"{meta['sr_variance_daily_scale']:.3e}):")
        L.append("")
        L.append("| name | SR (ann) | skew | ex.kurt | P(true SR>0) | primary |")
        L.append("|---|---|---|---|---|---|")
        for nm, w in s5["deflated_sharpe"].items():
            L.append(f"| {nm} | {_f3(w['observed_sharpe_annual'])} | "
                     f"{float(w['skew']):+.2f} | "
                     f"{float(w['excess_kurtosis']):.1f} | "
                     f"{float(w['dsr_probability']):.4f} | "
                     f"{'yes' if w['primary'] else ''} |")

    # Step 6 — gates
    s6 = results.get("step6_gates", {})
    if s6:
        L.append("")
        L.append("## Step 6 — Gates (no re-tuning)")
        L.append("")
        L.append("G3 anchors (val Calmar, 2026Q1 DD) inherit the selection "
                 "contamination disclosed above.")
        for nm, g in s6.items():
            L.append("")
            L.append(f"### {nm} — G3 {'PASS' if g['G3_pass'] else 'FAIL'} / "
                     f"G4 {'PASS' if g['G4_pass'] else 'FAIL'}")
            L.append("")
            L.append("| gate | criterion | result |")
            L.append("|---|---|---|")
            for k, v in g["G3_validation"].items():
                L.append(f"| G3 | {k} | {'PASS' if v else 'FAIL'} |")
            for k, v in g["G4_full"].items():
                L.append(f"| G4 | {k} | {'PASS' if v else 'FAIL'} |")

    # Per-year returns
    pyr = results.get("per_year_returns", {})
    if pyr:
        names = list(pyr)
        years = sorted(set().union(*[set(v) for v in pyr.values()]))
        L.append("")
        L.append("## Per-year returns (full window, tc=5bp)")
        L.append("")
        L.append("| year | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for y in years:
            L.append(f"| {y} | " +
                     " | ".join(_pc(pyr[n].get(y)) for n in names) + " |")

    L.append("")
    L.append(f"Ledger trial count after run: "
             f"{meta.get('ledger_trial_count_post_run', '—')}. "
             f"Tier ledger: `{TIER_LEDGER.relative_to(ROOT)}`.")
    L.append("")
    OUT_MD.write_text("\n".join(L))
    print(f"report -> {OUT_MD}")
    return OUT_MD


# --report-only: render the MD from the persisted json and exit BEFORE the
# module-level data load below (no cache load, no recomputation).
if __name__ == "__main__" and "--report-only" in sys.argv:
    write_report()
    sys.exit(0)

# ── Cross-trial SR variance: computed BEFORE any new rows are logged ──────
_sharpes = []
for f in sorted(glob.glob(str(ROOT / "results" / "v7" / "trials" / "*.csv"))):
    try:
        _sharpes.append(pd.read_csv(f)["sharpe"])
    except Exception:
        pass
_sh = pd.concat(_sharpes).dropna().astype(float)
SR_VAR_DAILY = float((_sh / np.sqrt(252.0)).var())   # ddof=1, daily scale
N_LEDGER_PRE = int(len(_sh))

# ── Data (FULL index — the one deviation from the dev-bound modules) ──────
panel, macro, _ = load_cache()
pu = union_prices_cached()
idx = pu.index                                   # 2016-04-01 .. 2026-03-27
opn = panel["open"]

cfg_open = BTConfigV2(tc_bps=5.0, leverage_cap=2.0, exec_model="next_open")
cfg_close = BTConfigV2(tc_bps=5.0, leverage_cap=2.0, exec_model="next_close")

# FREEZE dd_v1 (causal: rolling trailing peak + forward state machine) on the
# full macro — identical values on dev days to the dev-truncated build.
_fz = freeze_signal_spy_drawdown(macro, peak_lookback=21, freeze_dd_pct=0.12,
                                 unfreeze_within_pct=0.08, min_freeze_days=10)
F1_MULT = (1.0 - _fz.astype(float)).reindex(idx).ffill().fillna(1.0)


def align_union(sleeves: dict[str, pd.DataFrame]):
    """Verbatim from assemble_candidates.align_union."""
    dates = sorted(set().union(*[s.index for s in sleeves.values()]))
    cols = sorted(set().union(*[s.columns for s in sleeves.values()]))
    return ({k: s.reindex(index=dates, columns=cols, fill_value=0.0)
             for k, s in sleeves.items()}, dates, cols)


def build_blend_Bst_full() -> pd.DataFrame:
    """assemble_candidates.build_blend('B','st') without the dev slice."""
    sleeves = {k: pd.read_parquet(ROOT / "weights_store" / f"{k}.parquet")
               for k in SLEEVE_FILES}
    su, _, _ = align_union(sleeves)
    return sum(su.values()) / len(su)


class TierEnvFull:
    """assemble_candidates.TierEnv re-instantiated on the FULL index, plus
    (a) per-ticker-low gap matrix GL for the stop_grid phi convention and
    (b) per-day tier-event depth record (P&L identical at phi=0)."""

    def __init__(self, blend_1x: pd.DataFrame):
        self.W2_daily = expand_daily(blend_1x * 2.0, idx)
        self.held = list(blend_1x.columns[(blend_1x.abs() > 1e-12).any()])
        close_u = pu[self.held]
        close_ff = close_u.ffill()
        prevc = close_ff.shift(1)
        self.R = close_u.pct_change(fill_method=None).fillna(0.0).to_numpy()
        open_h = panel["open"][self.held].reindex(idx)
        open_f = open_h.where(open_h.notna(), close_ff)
        self.G = (open_f / prevc - 1.0).where(prevc.notna(), 0.0).fillna(0.0).to_numpy()
        # stop_grid.py convention: NaN low -> ffilled close (no extra dip)
        low_h = panel["low"][self.held].reindex(idx)
        low_b = low_h.where(low_h.notna(), close_ff)
        self.GL = (low_b / prevc - 1.0).where(prevc.notna(), 0.0).fillna(0.0).to_numpy()
        self.N, self.ndays = len(self.held), len(idx)
        e_base = _clip_and_cap(self.W2_daily, cfg_open)[self.held] \
            .shift(1).fillna(0.0).to_numpy()
        chg = np.abs(np.diff(e_base, axis=0)).sum(axis=1)
        self.base_exec_set = set(int(i) for i in np.where(chg > 1e-15)[0] + 1)

    def tier_loop(self, W_decision: pd.DataFrame, P: float | None,
                  delay: int = DELAY, tc_bps: float = 5.0, phi: float = 0.0):
        """Assembly tier loop (close-to-close daily-bar approx) with the
        phi descent/recovery leg ported verbatim from
        validation/stop_grid.py simulate(). phi=0 -> assembly loop exactly;
        P=None -> engine next_close exactly (asserted in main)."""
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
        ev_day = np.zeros(self.ndays, dtype=np.int8)
        R_np, G_np, GL_np = self.R, self.G, self.GL

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
                ev_day[d] = 3
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
            # descent leg: blend of checkpoint bottom and simultaneous-lows
            # bottom — ported verbatim from validation/stop_grid.py simulate()
            start = r_open if open_tier else 0.0
            if phi > 0.0:
                b_cp = min(r_open, full)
                b_sl = min(float(w @ GL_np[d]), b_cp)     # simultaneous lows
                B = b_cp - phi * (b_cp - b_sl)
                dU_desc = min(B - start, 0.0)
                dU_rec = (full - start) - dU_desc         # recovery into close
            else:
                dU_desc, dU_rec = dU, 0.0
            deepest = open_tier
            for k in range(open_tier + 1, 4):
                tk = (t1, t2, t3)[k - 1]
                u = (tk - R0) / f
                if dU_desc <= u:
                    fk = TIER_F[k]
                    traded[d] += (f - fk) * glev
                    cost += (f - fk) * glev * tc
                    R0, dU_desc, f, deepest = tk, dU_desc - u, fk, k
                    if k == 3:
                        break
                else:
                    break
            dU = dU_desc + dU_rec
            if deepest:
                ev[f"tier{deepest}"] += 1
                ev_day[d] = deepest
            net[d] = R0 + f * dU - cost
            if deepest == 3:
                w = np.zeros(self.N)
                cash_cd = delay
            else:
                w = w * f
                tier_mult *= f
        return net, ev, traded, gross_held, ev_day


# ── Window metric helpers (run_trial conventions) ─────────────────────────
def eq_from_net(net: np.ndarray) -> pd.Series:
    return (1.0 + pd.Series(net, index=idx)).cumprod() * INIT


def window_stats(eq_full: pd.Series, traded: np.ndarray, gross: np.ndarray,
                 ev_day: np.ndarray, window: str) -> dict:
    """Metrics for dev/val/full slices of a full-index tier run.
    val: sliced at VAL_START and renormalized (run_trial 'val' convention).
    turnover/avg_gross/events computed on the window's own days."""
    tr = pd.Series(traded, index=idx)
    gr = pd.Series(gross, index=idx)
    evd = pd.Series(ev_day, index=idx)
    if window == "dev":
        eq = eq_full.loc[:DEV_END]
        tr, gr, evd = tr.loc[:DEV_END], gr.loc[:DEV_END], evd.loc[:DEV_END]
    elif window == "val":
        eq = eq_full.loc[VAL_START:]
        eq = eq / eq.iloc[0] * INIT
        tr, gr, evd = tr.loc[VAL_START:], gr.loc[VAL_START:], evd.loc[VAL_START:]
    else:
        eq = eq_full
    s = metrics_summary(eq)
    return {
        "mean_monthly": s["mean_monthly"], "sharpe": s["sharpe"],
        "sortino": s["sortino"], "max_drawdown": s["max_drawdown"],
        "calmar": s["calmar"], "cagr": s["cagr"], "ann_vol": s["ann_vol"],
        "worst_month": s["worst_month"], "hit_rate": s["hit_rate_monthly"],
        "turnover_ann": float(tr.mean() * 252.0),
        "avg_gross": float(gr.mean()),
        "tier_events": {f"tier{k}": int((evd == k).sum()) for k in (1, 2, 3)},
        "_equity": eq,
    }


def tier_ledger_row(name: str, window: str, tc_bps: float, phi: float,
                    ws: dict, eq_for_episodes: pd.Series, params: dict,
                    notes: str) -> None:
    """Manual ledger row mirroring exp_lib.run_trial's schema."""
    row = {
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "family": "FINAL_VALIDATION_tier", "name": name, "window": window,
        "exec_model": "tier_loop_cc", "tc_bps": tc_bps, "leverage_cap": 2.0,
        "params_json": json.dumps(params | {"phi": phi}, default=str),
        "mean_monthly": ws["mean_monthly"], "median_monthly": np.nan,
        "sharpe": ws["sharpe"], "sortino": ws["sortino"],
        "max_drawdown": ws["max_drawdown"], "calmar": ws["calmar"],
        "cagr": ws["cagr"], "ann_vol": ws["ann_vol"],
        "worst_month": ws["worst_month"], "hit_rate": ws["hit_rate"],
        "turnover_ann": ws["turnover_ann"], "avg_gross": ws["avg_gross"],
        "runtime_s": np.nan,
        "notes": notes + f" events={ws['tier_events']}",
    }
    for ep, st in crash_table(eq_for_episodes).items():
        row[f"ep_{ep}_ret"] = st["ret"]
        row[f"ep_{ep}_dd"] = st["max_dd"]
    pd.DataFrame([row]).to_csv(TIER_LEDGER, mode="a",
                               header=not TIER_LEDGER.exists(), index=False)


def per_year_returns(eq: pd.Series) -> dict:
    ye = eq.resample("YE").last()
    out = {str(ye.index[0].year): float(ye.iloc[0] / eq.iloc[0] - 1.0)}
    for i in range(1, len(ye)):
        out[str(ye.index[i].year)] = float(ye.iloc[i] / ye.iloc[i - 1] - 1.0)
    return out


def main():
    results: dict = {"meta": {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "protocol": "V7 one-shot final validation; locked window "
                    f"{VAL_START}..{VAL_END} evaluated exactly once",
        "full_index": [str(idx[0].date()), str(idx[-1].date())],
        "n_days_full": int(len(idx)),
        "dsr_n_trials": N_TRIALS_DSR,
        "sr_variance_daily_scale": SR_VAR_DAILY,
        "sr_variance_n_ledger_rows_pre_run": N_LEDGER_PRE,
        "anchor_tolerance": ANCHOR_TOL,
    }}

    # ── Build candidate frames on the FULL index ──────────────────────────
    blend = build_blend_Bst_full()
    env = TierEnvFull(blend)
    print(f"blend Bst full: {blend.index[0].date()}..{blend.index[-1].date()} "
          f"({len(blend)} decisions), held={len(env.held)}")

    # engine fidelity: loop with P=None == engine next_close on full index
    bt_nc = run_backtest_v2(env.W2_daily, pu, cfg_close, name="nc_full",
                            weights_are_daily=True)
    net0, _, _, _, _ = env.tier_loop(env.W2_daily, None)
    diff_nc = float(np.abs(net0 - bt_nc["returns"].to_numpy()).max())
    print(f"tier loop (P=None) vs engine next_close (full idx): {diff_nc:.2e}")
    assert diff_nc < 1e-12, "tier loop does not reproduce next_close engine"
    results["meta"]["loop_vs_next_close_max_diff"] = diff_nc

    g_book = book_drawdown_gate(env.W2_daily, panel["close"],
                                full_dd=0.12, cash_dd=0.30, lookback=60)
    g_book = g_book.reindex(idx).fillna(1.0).clip(0.0, 1.0)

    W_TF1 = apply_gate(env.W2_daily, F1_MULT)
    W_BTF1 = apply_gate(apply_gate(env.W2_daily, g_book), F1_MULT)
    W_T = env.W2_daily

    CAND_FRAMES = {"ASM_Bst_TF1": W_TF1, "ASM_Bst_BTF1": W_BTF1,
                   "ASM_Bst_T": W_T}
    CAND_PARAMS = {
        "ASM_Bst_TF1": {"base": "E3 resid ms_raw replxs (res+dual+adapt n30)",
                        "alloc": "static 1/3", "leverage": 2.0,
                        "overlay": "tier(P-0.08,2d) + freeze dd_v1(21,0.12,0.08,10)"},
        "ASM_Bst_BTF1": {"base": "E3 resid ms_raw replxs (res+dual+adapt n30)",
                         "alloc": "static 1/3", "leverage": 2.0,
                         "overlay": "book(0.12,0.30,60) + tier(P-0.08,2d) + freeze dd_v1"},
        "ASM_Bst_T": {"base": "E3 resid ms_raw replxs (res+dual+adapt n30)",
                      "alloc": "static 1/3", "leverage": 2.0,
                      "overlay": "tier(P-0.08,2d) only"},
    }

    # ── STEP 1: FIDELITY ANCHORS (hard stop) ──────────────────────────────
    print("\n══ STEP 1: fidelity anchors (full-index build, dev slice) ══")
    asm = pd.read_csv(ROOT / "results" / "v7" / "trials" / "ASSEMBLY_tier.csv")
    anchor_ref = {}
    for nm in CAND_FRAMES:
        r = asm[(asm["name"] == nm) & (asm["tc_bps"] == 5.0)].iloc[-1]
        anchor_ref[nm] = {k: float(r[k]) for k in
                          ("mean_monthly", "sharpe", "max_drawdown", "calmar")}
    anchor_events = {"ASM_Bst_TF1": (15, 0, 0), "ASM_Bst_BTF1": (14, 0, 0),
                     "ASM_Bst_T": (15, 1, 0)}
    with open(ROOT / "results" / "v7" / "baseline.json") as f:
        baseline = json.load(f)

    full_runs: dict = {}      # name -> {tc -> run dict}
    anchors_out, anchors_ok = {}, True
    for nm, W in CAND_FRAMES.items():
        net, ev, traded, gross, ev_day = env.tier_loop(W, P_TIER, DELAY, 5.0)
        eq_full = eq_from_net(net)
        full_runs[nm] = {5.0: dict(net=net, ev=ev, traded=traded, gross=gross,
                                   ev_day=ev_day, eq=eq_full)}
        dev = window_stats(eq_full, traded, gross, ev_day, "dev")
        ref = anchor_ref[nm]
        diffs = {k: abs(dev[k] - ref[k]) for k in ref}
        ev_dev = tuple(dev["tier_events"][f"tier{k}"] for k in (1, 2, 3))
        ok = all(d <= ANCHOR_TOL for d in diffs.values()) \
            and ev_dev == anchor_events[nm]
        anchors_ok &= ok
        anchors_out[nm] = {
            "reproduced": {k: dev[k] for k in ref}, "reference": ref,
            "abs_diffs": diffs, "tier_events_dev": list(ev_dev),
            "tier_events_ref": list(anchor_events[nm]), "pass": bool(ok)}
        print(f"  {nm:14s} mm={dev['mean_monthly']*100:6.3f}% "
              f"sharpe={dev['sharpe']:.4f} dd={dev['max_drawdown']*100:6.2f}% "
              f"calmar={dev['calmar']:.4f} ev={ev_dev} "
              f"maxdiff={max(diffs.values()):.2e} -> {'PASS' if ok else 'FAIL'}")
        tier_ledger_row(f"FV_{nm}_devanchor", "dev", 5.0, 0.0, dev,
                        dev["_equity"], CAND_PARAMS[nm],
                        "final-validation fidelity anchor (full-index build, "
                        "dev slice);")

    # champion dev anchor through run_trial (the assembly/baseline path)
    base_champ = pd.read_parquet(ROOT / "weights_store" /
                                 "combo_v2_base_1x.parquet")
    rc_dev = run_trial(base_champ * 2.0, name="FV_champion_dev", family=FAMILY,
                       params={"strategy": "combo_v2_2x (combo_v2_base_1x x 2.0)",
                               "role": "fidelity anchor"},
                       window="dev", exec_model="next_open", tc_bps=5.0,
                       leverage_cap=2.0,
                       notes="final-validation champion dev anchor")
    champ_ref = baseline["dev"]
    champ_diffs = {k: abs(rc_dev["row"][k] - champ_ref[k]) for k in
                   ("mean_monthly", "sharpe", "max_drawdown", "calmar")}
    champ_ok = all(d <= ANCHOR_TOL for d in champ_diffs.values())
    anchors_ok &= champ_ok
    anchors_out["champion_combo_v2_2x"] = {
        "reproduced": {k: rc_dev["row"][k] for k in champ_diffs},
        "reference": {k: champ_ref[k] for k in champ_diffs},
        "abs_diffs": champ_diffs, "pass": bool(champ_ok)}
    print(f"  champion       mm={rc_dev['row']['mean_monthly']*100:6.3f}% "
          f"sharpe={rc_dev['row']['sharpe']:.4f} "
          f"dd={rc_dev['row']['max_drawdown']*100:6.2f}% "
          f"calmar={rc_dev['row']['calmar']:.4f} "
          f"maxdiff={max(champ_diffs.values()):.2e} -> "
          f"{'PASS' if champ_ok else 'FAIL'}")

    results["step1_fidelity_anchors"] = {
        "pass": bool(anchors_ok), **anchors_out}
    if not anchors_ok:
        results["meta"]["HARD_STOP"] = ("FIDELITY ANCHORS FAILED — validation "
                                        "window NOT evaluated.")
        OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
        print("\nHARD STOP: anchors failed; aborting before validation window.")
        sys.exit(1)
    print("  ALL ANCHORS PASS — proceeding to the locked window.")

    # ── Champion val/full runs (final=True — authorized one-shot) ─────────
    rc_val = run_trial(base_champ * 2.0, name="FV_champion_val", family=FAMILY,
                       params={"strategy": "combo_v2_2x", "role": "reference"},
                       window="val", exec_model="next_open", tc_bps=5.0,
                       leverage_cap=2.0, final=True,
                       notes="one-shot validation window, champion reference")
    rc_full = {}
    for tc in TCS:
        rc_full[tc] = run_trial(
            base_champ * 2.0, name=f"FV_champion_full_tc{int(tc)}",
            family=FAMILY, params={"strategy": "combo_v2_2x",
                                   "role": "reference", "tc_bps": tc},
            window="full", exec_model="next_open", tc_bps=tc,
            leverage_cap=2.0, final=True,
            notes="one-shot full-window record, champion reference")
    champ_eq_full = rc_full[5.0]["equity"]
    # champion window turnover/gross from the effective daily frame
    champ_W = rc_full[5.0]["weights"]
    champ_delta = champ_W.diff().abs().sum(axis=1)
    champ_delta.iloc[0] = champ_W.iloc[0].abs().sum()
    champ_gross = champ_W.abs().sum(axis=1)

    def champ_window(window: str) -> dict:
        if window == "val":
            eq = champ_eq_full.loc[VAL_START:]
            eq = eq / eq.iloc[0] * INIT
            d, g = champ_delta.loc[VAL_START:], champ_gross.loc[VAL_START:]
        elif window == "dev":
            eq = champ_eq_full.loc[:DEV_END]
            d, g = champ_delta.loc[:DEV_END], champ_gross.loc[:DEV_END]
        else:
            eq, d, g = champ_eq_full, champ_delta, champ_gross
        s = metrics_summary(eq)
        return {"mean_monthly": s["mean_monthly"], "sharpe": s["sharpe"],
                "sortino": s["sortino"], "max_drawdown": s["max_drawdown"],
                "calmar": s["calmar"], "cagr": s["cagr"],
                "ann_vol": s["ann_vol"], "worst_month": s["worst_month"],
                "hit_rate": s["hit_rate_monthly"],
                "turnover_ann": float(d.mean() * 252.0),
                "avg_gross": float(g.mean()), "_equity": eq}

    # ── STEP 2: VALIDATION WINDOW (tc=5bp) ────────────────────────────────
    print("\n══ STEP 2: validation window 2023-01-01..2026-03-27 (tc=5bp) ══")
    val_table = {}
    for nm in CAND_FRAMES:
        run = full_runs[nm][5.0]
        ws = window_stats(run["eq"], run["traded"], run["gross"],
                          run["ev_day"], "val")
        ep = episode_stats(run["eq"], "2026-01-01", "2026-03-27")
        ws["ep_2026Q1_ret"], ws["ep_2026Q1_dd"] = ep["ret"], ep["max_dd"]
        val_table[nm] = ws
        tier_ledger_row(f"FV_{nm}_val", "val", 5.0, 0.0, ws, run["eq"],
                        CAND_PARAMS[nm],
                        "ONE-SHOT validation window (sliced+renormalized at "
                        "2023-01-01);")
    cw = champ_window("val")
    ep = episode_stats(champ_eq_full, "2026-01-01", "2026-03-27")
    cw["ep_2026Q1_ret"], cw["ep_2026Q1_dd"] = ep["ret"], ep["max_dd"]
    cw["tier_events"] = None
    val_table["champion_combo_v2_2x"] = cw
    for nm, ws in val_table.items():
        print(f"  {nm:22s} mm={ws['mean_monthly']*100:6.3f}% "
              f"sharpe={ws['sharpe']:.3f} dd={ws['max_drawdown']*100:6.2f}% "
              f"calmar={ws['calmar']:.3f} worst_mo={ws['worst_month']*100:6.2f}% "
              f"hit={ws['hit_rate']*100:4.1f}% turn={ws['turnover_ann']:5.1f} "
              f"26Q1 ret={ws['ep_2026Q1_ret']*100:6.2f}% "
              f"dd={ws['ep_2026Q1_dd']*100:6.2f}%")
    results["step2_validation_window"] = {
        nm: {k: v for k, v in ws.items() if k != "_equity"}
        for nm, ws in val_table.items()}

    # ── STEP 3: FULL WINDOW at 5/10/20bp ──────────────────────────────────
    print("\n══ STEP 3: full window 2016-04..2026-03-27 at tc 5/10/20bp ══")
    full_table = {}
    for nm, W in CAND_FRAMES.items():
        full_table[nm] = {}
        for tc in TCS:
            if tc not in full_runs[nm]:
                net, ev, traded, gross, ev_day = env.tier_loop(W, P_TIER,
                                                               DELAY, tc)
                full_runs[nm][tc] = dict(net=net, ev=ev, traded=traded,
                                         gross=gross, ev_day=ev_day,
                                         eq=eq_from_net(net))
            run = full_runs[nm][tc]
            ws = window_stats(run["eq"], run["traded"], run["gross"],
                              run["ev_day"], "full")
            ws["episodes"] = crash_table(run["eq"])
            full_table[nm][f"tc{int(tc)}"] = ws
            tier_ledger_row(f"FV_{nm}_full_tc{int(tc)}", "full", tc, 0.0, ws,
                            run["eq"], CAND_PARAMS[nm],
                            "one-shot full-window record;")
            print(f"  {nm:14s} tc{int(tc):2d} mm={ws['mean_monthly']*100:6.3f}% "
                  f"sharpe={ws['sharpe']:.3f} dd={ws['max_drawdown']*100:6.2f}% "
                  f"calmar={ws['calmar']:.3f} ev={ws['tier_events']}")
    full_table["champion_combo_v2_2x"] = {}
    for tc in TCS:
        s = rc_full[tc]["row"]
        ws = {k: s[k] for k in ("mean_monthly", "sharpe", "max_drawdown",
                                "calmar", "worst_month", "hit_rate",
                                "turnover_ann", "avg_gross", "ann_vol",
                                "cagr", "sortino")}
        ws["episodes"] = crash_table(rc_full[tc]["equity"])
        full_table["champion_combo_v2_2x"][f"tc{int(tc)}"] = ws
        print(f"  champion       tc{int(tc):2d} mm={ws['mean_monthly']*100:6.3f}% "
              f"sharpe={ws['sharpe']:.3f} dd={ws['max_drawdown']*100:6.2f}% "
              f"calmar={ws['calmar']:.3f}")
    # consistency: champion full 5bp vs baseline.json full record
    bf = baseline["full"]
    champ_full_diffs = {k: abs(full_table["champion_combo_v2_2x"]["tc5"][k] - bf[k])
                        for k in ("mean_monthly", "sharpe", "max_drawdown",
                                  "calmar")}
    print(f"  champion full vs baseline.json full: "
          f"maxdiff={max(champ_full_diffs.values()):.2e}")
    results["meta"]["champion_full_vs_baseline_json_diffs"] = champ_full_diffs
    results["step3_full_window"] = {
        nm: {tck: {k: v for k, v in ws.items() if k != "_equity"}
             for tck, ws in d.items()} for nm, d in full_table.items()}

    # ── STEP 4: TIER PHI-STRESS (full window, tc=5bp) ─────────────────────
    print("\n══ STEP 4: tier phi-stress (full window, tc=5bp) ══")
    champ_full_calmar = float(full_table["champion_combo_v2_2x"]["tc5"]["calmar"])
    phi_table = {}
    for nm, W in CAND_FRAMES.items():
        phi_table[nm] = {}
        base_ws = full_table[nm]["tc5"]
        phi_table[nm]["phi0.0"] = {
            "mean_monthly": base_ws["mean_monthly"], "sharpe": base_ws["sharpe"],
            "max_drawdown": base_ws["max_drawdown"], "calmar": base_ws["calmar"],
            "tier_events": base_ws["tier_events"]}
        for phi in (0.5, 1.0):
            net, ev, traded, gross, ev_day = env.tier_loop(W, P_TIER, DELAY,
                                                           5.0, phi=phi)
            eq = eq_from_net(net)
            ws = window_stats(eq, traded, gross, ev_day, "full")
            ws["episodes"] = crash_table(eq)
            phi_table[nm][f"phi{phi}"] = {
                "mean_monthly": ws["mean_monthly"], "sharpe": ws["sharpe"],
                "max_drawdown": ws["max_drawdown"], "calmar": ws["calmar"],
                "tier_events": ws["tier_events"],
                "calmar_drop_vs_phi0": float(1.0 - ws["calmar"] /
                                             base_ws["calmar"]),
                "dd_widening_pp": float((ws["max_drawdown"] -
                                         base_ws["max_drawdown"]) * 100.0)}
            tier_ledger_row(f"FV_{nm}_full_phi{phi}", "full", 5.0, phi, ws,
                            eq, CAND_PARAMS[nm],
                            f"phi-stress (stop_grid convention), phi={phi};")
            print(f"  {nm:14s} phi={phi:3.1f} mm={ws['mean_monthly']*100:6.3f}% "
                  f"dd={ws['max_drawdown']*100:6.2f}% calmar={ws['calmar']:.3f} "
                  f"ev={ws['tier_events']}")
    promo_risk = phi_table["ASM_Bst_TF1"]["phi1.0"]["calmar"] < champ_full_calmar
    results["step4_phi_stress"] = {
        "tables": phi_table,
        "champion_full_calmar_tc5": champ_full_calmar,
        "PROMOTION_RISK_TF1_phi1_below_champion": bool(promo_risk)}
    if promo_risk:
        print(f"  *** PROMOTION RISK: ASM_Bst_TF1 phi=1.0 full Calmar "
              f"{phi_table['ASM_Bst_TF1']['phi1.0']['calmar']:.3f} < champion "
              f"{champ_full_calmar:.3f} ***")

    # ── STEP 5: STATISTICS ────────────────────────────────────────────────
    print("\n══ STEP 5: statistics ══")
    champ_mo = monthly_returns(champ_eq_full)
    stats_out = {"newey_west": {}, "deflated_sharpe": {}}
    for nm in CAND_FRAMES:
        cand_mo = monthly_returns(full_runs[nm][5.0]["eq"])
        joined = pd.concat([cand_mo, champ_mo], axis=1, join="inner",
                           keys=["cand", "champ"]).dropna()
        d = joined["cand"] - joined["champ"]
        t = newey_west_tstat(d, lags=3)
        stats_out["newey_west"][nm] = {
            "mean_monthly_diff": float(d.mean()), "n_months": int(len(d)),
            "nw_tstat_lag3": float(t)}
        print(f"  NW t (vs champion, full, lag3) {nm:14s}: "
              f"mean diff={d.mean()*100:+.3f}%/mo  t={t:+.3f}  n={len(d)}")
    for nm in CAND_FRAMES:
        eq = full_runs[nm][5.0]["eq"]
        r = daily_returns(eq)
        sr_ann = float(full_table[nm]["tc5"]["sharpe"])
        sk = float(sps.skew(r))
        ek = float(sps.kurtosis(r))          # excess kurtosis (fisher)
        dsr = deflated_sharpe(sr_ann, int(len(r)), N_TRIALS_DSR,
                              SR_VAR_DAILY, skew=sk, excess_kurtosis=ek)
        stats_out["deflated_sharpe"][nm] = {
            "observed_sharpe_annual": sr_ann, "n_obs_daily": int(len(r)),
            "n_trials": N_TRIALS_DSR, "sr_variance_daily": SR_VAR_DAILY,
            "skew": sk, "excess_kurtosis": ek, "dsr_probability": float(dsr),
            "primary": nm == "ASM_Bst_TF1"}
        print(f"  DSR {nm:14s}: SR={sr_ann:.3f} skew={sk:+.2f} exkurt={ek:.1f} "
              f"-> P(true SR>0)={dsr:.4f}")
    results["step5_statistics"] = stats_out

    # ── STEP 6: GATES ─────────────────────────────────────────────────────
    print("\n══ STEP 6: gates ══")
    champ_val_calmar = float(val_table["champion_combo_v2_2x"]["calmar"])
    champ_q1_dd = float(val_table["champion_combo_v2_2x"]["ep_2026Q1_dd"])
    gates = {}
    for nm in CAND_FRAMES:
        v = val_table[nm]
        fw = full_table[nm]["tc5"]
        dsr = stats_out["deflated_sharpe"][nm]["dsr_probability"]
        g3 = {
            "mean>=4.5%/mo": bool(v["mean_monthly"] >= 0.045),
            "val_MaxDD>=-35%": bool(v["max_drawdown"] >= -0.35),
            f"calmar>=champ_val({champ_val_calmar:.3f})":
                bool(v["calmar"] >= champ_val_calmar),
            f"2026Q1_dd<=champ({champ_q1_dd*100:.2f}%)":
                bool(v["ep_2026Q1_dd"] >= champ_q1_dd),
        }
        g4_dd = bool(fw["max_drawdown"] >= -0.40 or
                     (fw["max_drawdown"] >= -0.45 and fw["calmar"] >= 1.5))
        g4 = {
            "mean>=5.0%/mo": bool(fw["mean_monthly"] >= 0.050),
            "MaxDD>=-40%(-45% if Calmar>=1.5)": g4_dd,
            "calmar>=1.4": bool(fw["calmar"] >= 1.4),
            "deflated_sharpe>0": bool(dsr > 0.0),
            "deflated_sharpe>0.5 (supplementary)": bool(dsr > 0.5),
        }
        gates[nm] = {
            "G3_validation": g3, "G3_pass": bool(all(g3.values())),
            "G4_full": g4,
            "G4_pass": bool(all(v for k, v in g4.items()
                                if "supplementary" not in k)),
        }
        print(f"  {nm:14s} G3={'PASS' if gates[nm]['G3_pass'] else 'FAIL'} "
              f"{g3}")
        print(f"  {'':14s} G4={'PASS' if gates[nm]['G4_pass'] else 'FAIL'} "
              f"{g4}")
    results["step6_gates"] = gates

    # per-year returns: primary vs champion (full window)
    results["per_year_returns"] = {
        "ASM_Bst_TF1": per_year_returns(full_runs["ASM_Bst_TF1"][5.0]["eq"]),
        "champion_combo_v2_2x": per_year_returns(champ_eq_full)}

    results["meta"]["ledger_trial_count_post_run"] = trial_count()
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults -> {OUT_JSON}")
    print(f"ledger total after run: {results['meta']['ledger_trial_count_post_run']}")
    # render the MD from the just-persisted json (round-trip: report always
    # reflects the artifact on disk, never in-memory state)
    write_report(json.loads(OUT_JSON.read_text()))
    return results


if __name__ == "__main__":
    main()
