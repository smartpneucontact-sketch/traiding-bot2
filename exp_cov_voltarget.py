"""Stream 2 — the never-run vol_est="cov" grid (family=cov_voltarget,
DEV-ONLY, margin 600). Completes E4's book-vol-target direction on the
corrected cost basis.

Motivating prior — WITH A LEDGER CORRECTION
-------------------------------------------
The Roadmap v2 plan cites "E4's vt35 dev Calmar 0.968" as the motivating
prior. That number is a planning-memory error (same class as the E5 ledger
correction in DECISION_RULE.md): the ledgered row e4_vt35_eq_2x
(results/v7/trials/E4_weighting.csv, 2026-06) is dev SHARPE 0.967 and dev
Calmar 0.639 vs the 0.858 anchor — the plan's "Calmar 0.968" is the Sharpe
misfiled. The honest motivating prior is therefore weaker but real:
  (1) E4 scored its grid MARGIN-FREE; at the corrected 600bp/yr financing
      basis, low-gross books are taxed less than the ~1.6x-gross baseline,
      which shifts the ranking toward vol-targeted cells (this experiment's
      whole point: E4's comparison was run on the wrong cost basis).
  (2) The vol_est="cov" infrastructure (strategies._est_portfolio_vol,
      engine_v2.ex_ante_book_vol) shipped in the 2026-07-12 audit and was
      NEVER trialed as a sleeve-internal fix at targets other than the
      binding-92%-of-dates 0.15 (e4_dualfix, Calmar 0.815).
  (3) The book-level target 0.20 was never run at any cost basis.

PRE-COMMITTED GATE (written before any cell ran; template per the
Improvement Roadmap v2, Stream 2):
    dev DSR >= 0.95 at the current deduped trial count
    AND dev Calmar >= baseline + 10%   (baseline = catalog_v2 combo_v2_2x
                                        dev row at margin600)
    AND geo give-up <= 0.2pp/mo        (dev geo >= baseline dev geo - 0.2pp)
    -> eligible for ONE-SHOT validation LATER (no final=True / val runs in
       this phase; dev verdicts only).

Grid (exactly 8 cells, all ledgered, family cov_voltarget, margin 600):
  sleeve-level: dual_momentum_voltarget(vol_est="cov", target_vol=t),
      t in {0.20, 0.25, 0.30, 0.35}, blended with the cached xs30 + adapt30
      catalog sleeves, x2.0 lev                                        [4]
  book-level:   exp_e4_weighting.true_book_voltarget on the standard blend
      (60d covariance of the FULL levered book at each decision date),
      target in {0.20, 0.25, 0.30, 0.35}, x2.0 lev                     [4]

CACHE-KEY NOTE (binding): building sleeves with vol_est="cov" flips NO cache
key automatically beyond exp_lib._CODE_HASH — the legacy key scheme encodes
only the sleeve name, so a "cov" build under a legacy key would silently
shadow the diag_legacy frames the published record depends on. All new
sleeves here use NEW keys with the `cov__` prefix.

Engine: next_open, tc=5bp, margin_bps_annual=600, leverage_cap=2.0,
window="dev". Base-blend fidelity asserted against combo_v2_base_1x.parquet
(the E4 sanity anchor) before any cell runs.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python exp_cov_voltarget.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import strategies as S
from exp_lib import (_CODE_HASH, TRIALS_DIR, cached_weights, load_cache,
                     run_trial, trial_count, union_prices_cached)
from exp_e4_weighting import blend3, true_book_voltarget
from exp_ws4_vixterm import baseline_row, cached_frame_or_die, dsr_inputs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v7"
FAMILY = "cov_voltarget"
LEDGER = TRIALS_DIR / f"{FAMILY}.csv"
MARGIN_BPS = 600.0
LEV = 2.0
TARGETS = (0.20, 0.25, 0.30, 0.35)


def main() -> None:
    t0 = time.time()
    base = baseline_row()
    base_geo_dev = float((1.0 + base["cagr"]) ** (1.0 / 12.0) - 1.0)
    print(f"baseline (catalog_v2 combo_v2_2x dev, margin600): "
          f"calmar {base['calmar']:.4f} | geo {base_geo_dev:.4%}/mo")

    panel, macro, _ = load_cache()
    px = panel["close"]
    pu = union_prices_cached()

    w_xs = cached_frame_or_die("sleeve_xs_momentum_n30")
    w_dual = cached_frame_or_die("sleeve_dual_momentum_voltarget_n30")
    w_adapt = cached_frame_or_die("sleeve_adaptive_voltarget_n30")

    b_eq = blend3(w_xs, w_dual, w_adapt)
    base_ref = pd.read_parquet(ROOT / "weights_store" / "combo_v2_base_1x.parquet")
    chk = (b_eq.reindex_like(base_ref).fillna(0.0) - base_ref).abs().max().max()
    print(f"sanity: |blend3(sleeves) - combo_v2_base_1x|_max = {chk:.2e}")
    assert chk < 1e-9, "cached sleeves do not reproduce combo_v2_base_1x"

    rets_pu = pu.pct_change(fill_method=None)
    cols_b = [c for c in b_eq.columns if c in rets_pu.columns]
    rets_b = rets_pu[cols_b]

    done: set[str] = set()
    if LEDGER.exists():
        done = set(pd.read_csv(LEDGER)["name"])
        print(f"[resume] {len(done)} cells already ledgered")

    def cell(name: str, w: pd.DataFrame, params: dict, notes: str) -> None:
        if name in done:
            return
        r = run_trial(w, name=name, family=FAMILY, params=params,
                      window="dev", exec_model="next_open", tc_bps=5.0,
                      leverage_cap=2.0, margin_bps_annual=MARGIN_BPS,
                      final=False, notes=notes)
        row = r["row"]
        print(f"  {name:24s} mm={row['mean_monthly']*100:5.2f}% "
              f"sharpe={row['sharpe']:.3f} dd={row['max_drawdown']*100:5.1f}% "
              f"calmar={row['calmar']:.3f} gross={row['avg_gross']:.2f}")

    print("\n[1/2] sleeve-level: dual sleeve vol_est='cov' x 4 targets")
    for tgt in TARGETS:
        key = f"cov__sleeve_dual_cov{int(tgt*100)}_n30"
        print(f"  building/loading {key} ...")
        w_dual_cov = cached_weights(
            key,
            lambda tgt=tgt: S.dual_momentum_voltarget(
                px, macro, n_long=30, target_vol=tgt, vol_est="cov"))
        b = blend3(w_xs, w_dual_cov, w_adapt)
        cell(f"cov_dual{int(tgt*100)}_2x", b * LEV,
             {"variant": "sleeve_dual_cov", "target_vol": tgt,
              "vol_est": "cov", "vol_window": 60, "lev": LEV,
              "cache_key": key},
             notes=f"dual sleeve ex-ante covariance vol target {tgt:g}; "
                   f"new cov__ cache key")

    print("\n[2/2] book-level: true_book_voltarget (E4 construction) x 4 "
          "targets, margin600")
    for tgt in TARGETS:
        wf, sc = true_book_voltarget(b_eq, rets_b, tgt, lev=LEV, window=60)
        sc_dev = sc.loc[:"2022-12-31"]
        frac = float((sc_dev < 0.9999).mean())
        cell(f"cov_bookvt{int(tgt*100)}_2x", wf,
             {"variant": "book_voltarget", "target_vol": tgt,
              "vol_window": 60, "basis": "levered_2x", "lev": LEV,
              "frac_binding_dev": round(frac, 4)},
             notes=f"E4 true book voltarget {tgt:g} at margin600 "
                   f"(E4 scored margin-free); binds {frac:.0%} of dev "
                   f"decision dates")

    # ── assemble + render-time DSR + gate verdicts ──
    led = pd.read_csv(LEDGER).drop_duplicates("name", keep="last")
    led["geo_monthly"] = (1.0 + led["cagr"]) ** (1.0 / 12.0) - 1.0
    n_trials, sr_var = dsr_inputs()
    from metrics_v2 import deflated_sharpe
    n_days_dev = int(pu.loc[:"2022-12-31"].shape[0])
    led["dsr_dev"] = [deflated_sharpe(sh, n_days_dev, n_trials,
                                      sr_variance_across_trials=sr_var)
                      for sh in led["sharpe"]]

    calmar_bar = 1.10 * float(base["calmar"])
    geo_bar = base_geo_dev - 0.002
    led["gate_dsr"] = led["dsr_dev"] >= 0.95
    led["gate_calmar"] = led["calmar"] >= calmar_bar
    led["gate_geo"] = led["geo_monthly"] >= geo_bar
    led["gate_all"] = led["gate_dsr"] & led["gate_calmar"] & led["gate_geo"]

    cols = ["name", "geo_monthly", "mean_monthly", "sharpe", "max_drawdown",
            "calmar", "avg_gross", "dsr_dev", "gate_dsr", "gate_calmar",
            "gate_geo", "gate_all"]
    tbl = led[cols].sort_values("calmar", ascending=False).reset_index(drop=True)
    tbl.to_csv(OUT / "cov_voltarget_results.csv", index=False)

    best = tbl.iloc[0]
    verdict = {
        "family": FAMILY,
        "n_cells": int(len(led)),
        "baseline": {"calmar_dev": float(base["calmar"]),
                     "geo_dev": base_geo_dev,
                     "source": "results/v7/catalog_v2.csv combo_v2_2x dev"},
        "gates": {"dsr_min": 0.95, "calmar_min": calmar_bar,
                  "geo_min": geo_bar, "n_trials_at_eval": n_trials},
        "best_cell": {k: (float(best[k]) if isinstance(best[k], (int, float, np.floating))
                          else bool(best[k]) if isinstance(best[k], (bool, np.bool_))
                          else str(best[k])) for k in cols},
        "n_pass": int(tbl["gate_all"].sum()),
        "verdict": ("ELIGIBLE for one-shot val LATER"
                    if bool(tbl["gate_all"].any()) else
                    "FAIL dev gates — no val shot"),
        "e4_prior_correction": "plan cited 'vt35 dev Calmar 0.968'; ledger "
                               "e4_vt35_eq_2x is Sharpe 0.967 / Calmar 0.639 "
                               "(margin-free) — see docstring",
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "code_hash": _CODE_HASH,
    }
    with open(OUT / "cov_voltarget_verdicts.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n== {FAMILY} dev table (calmar bar {calmar_bar:.3f}, "
          f"geo bar {geo_bar:.4%}, DSR bar 0.95 at n={n_trials}) ==")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nVERDICT: {verdict['verdict']} "
          f"(best cell {best['name']}, calmar {best['calmar']:.3f})")
    print(f"wrote cov_voltarget_results.csv + cov_voltarget_verdicts.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
