"""WS4(c) — breadth-regime gate on the combo_v2 book
(Stream 2, family=regime_breadth, DEV-ONLY).

Hypothesis: collapsing market breadth (few names above their 200dma; deeply
negative 63d advance-decline) marks broad-regime stress in which the momentum
book should de-risk. Signals come from breadth.py, computed on the EXISTING
survivors-only close panel — see the survivorship warning in that module:
LEVELS are biased, so ONLY rolling-percentile transforms are used here
(pre-committed; no raw-level thresholds, no alternative transforms).

PRE-COMMITTED GATE (written before any cell ran; template per the
Improvement Roadmap v2, Stream 2):
    dev DSR >= 0.95 at the current deduped trial count
    AND dev Calmar >= baseline + 10%   (baseline = catalog_v2 combo_v2_2x
                                        dev row at margin600)
    AND geo give-up <= 0.2pp/mo        (dev geo >= baseline dev geo - 0.2pp)
    -> eligible for ONE-SHOT validation LATER (no final=True / val runs in
       this phase; dev verdicts only).

Grid (exactly 8 cells, all ledgered, family regime_breadth, margin 600):
    feature in {frac_above_200dma, ad_net_63d}   (rolling pctile, w=756,
                                                  min_periods=252)
  x percentile threshold in {0.10, 0.20}         (de-risk when pct < thr)
  x exposure multiplier in {0.3, 0.5}
No in-family anchor cell: the identical daily-expanded base was fidelity-
checked in family regime_vixterm (cell vixterm_base_2x reproduced the
catalog_v2 combo_v2_2x dev row at 0 error, 2026-07-18) — re-ledgering it
here would only inflate the trial count.

Construction identical to exp_ws4_vixterm: cached catalog sleeves (cache-hit
only), blend /3, x2.0, daily-expanded, engine_v2.apply_gate (decision-side;
engine shift adds the execution lag), next_open, tc=5bp, margin 600bp/yr,
cap 2.0, window="dev".

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python exp_ws4_breadth.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from breadth import adv_decline_net, frac_above_ma, rolling_percentile
from engine_v2 import apply_gate
from exp_lib import _CODE_HASH, TRIALS_DIR, load_cache, run_trial, trial_count
from exp_ws4_vixterm import baseline_row, build_base_daily, dsr_inputs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v7"
FAMILY = "regime_breadth"
LEDGER = TRIALS_DIR / f"{FAMILY}.csv"
MARGIN_BPS = 600.0
LEV = 2.0

PCT_WINDOW = 756
PCT_MINP = 252
THRESHOLDS = (0.10, 0.20)
MULTS = (0.3, 0.5)


def main() -> None:
    t0 = time.time()
    base = baseline_row()
    base_geo_dev = float((1.0 + base["cagr"]) ** (1.0 / 12.0) - 1.0)
    print(f"baseline (catalog_v2 combo_v2_2x dev, margin600): "
          f"calmar {base['calmar']:.4f} | geo {base_geo_dev:.4%}/mo")

    panel, _, _ = load_cache()
    close = panel["close"]
    print("computing breadth features (survivors-only panel — see "
          "breadth.py warning; percentile transforms only)...")
    feats = {
        "above200": rolling_percentile(frac_above_ma(close, 200),
                                       PCT_WINDOW, PCT_MINP),
        "ad63": rolling_percentile(adv_decline_net(close, 63),
                                   PCT_WINDOW, PCT_MINP),
    }
    for k, f in feats.items():
        fd = f.loc[:"2022-12-31"].dropna()
        print(f"  {k}: {f.notna().sum()} defined days; dev p10 share "
              f"{(fd < 0.10).mean():.1%}, p20 share {(fd < 0.20).mean():.1%}")

    w_daily = build_base_daily()

    done: set[str] = set()
    if LEDGER.exists():
        done = set(pd.read_csv(LEDGER)["name"])
        print(f"[resume] {len(done)} cells already ledgered")

    print("\n[grid] 2 features x 2 thresholds x 2 multipliers")
    for feat_name, pct in feats.items():
        for thr in THRESHOLDS:
            active = pct < thr  # NaN comparisons are False -> gate 1.0 in warmup
            frac_dev = float(active.loc[:"2022-12-31"].mean())
            for mult in MULTS:
                name = f"breadth_{feat_name}_p{int(thr*100)}_m{mult:g}"
                if name in done:
                    continue
                gate = pd.Series(np.where(active, mult, 1.0), index=pct.index)
                wg = apply_gate(w_daily, gate)
                r = run_trial(wg, name=name, family=FAMILY,
                              params={"feature": feat_name,
                                      "pct_window": PCT_WINDOW,
                                      "pct_threshold": thr, "mult": mult,
                                      "lev": LEV,
                                      "frac_active_dev": round(frac_dev, 4)},
                              window="dev", exec_model="next_open",
                              tc_bps=5.0, leverage_cap=2.0,
                              margin_bps_annual=MARGIN_BPS,
                              weights_are_daily=True, final=False,
                              notes=f"breadth pctile gate {feat_name}<p"
                                    f"{int(thr*100)} -> {mult:g}x; active "
                                    f"{frac_dev:.1%} of dev days; survivors-"
                                    f"biased panel (breadth.py warning)")
                row = r["row"]
                print(f"  {name:28s} mm={row['mean_monthly']*100:5.2f}% "
                      f"sharpe={row['sharpe']:.3f} "
                      f"dd={row['max_drawdown']*100:5.1f}% "
                      f"calmar={row['calmar']:.3f}")

    # ── assemble + render-time DSR + gate verdicts ──
    led = pd.read_csv(LEDGER).drop_duplicates("name", keep="last")
    led["geo_monthly"] = (1.0 + led["cagr"]) ** (1.0 / 12.0) - 1.0
    n_trials, sr_var = dsr_inputs()
    from metrics_v2 import deflated_sharpe
    n_days_dev = int(w_daily.loc[:"2022-12-31"].shape[0])
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
            "calmar", "dsr_dev", "gate_dsr", "gate_calmar", "gate_geo",
            "gate_all"]
    tbl = led[cols].sort_values("calmar", ascending=False).reset_index(drop=True)
    tbl.to_csv(OUT / "ws4_breadth_results.csv", index=False)

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
        "survivorship_note": "breadth features from survivors-only panel; "
                             "percentile transforms only (breadth.py)",
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "code_hash": _CODE_HASH,
    }
    with open(OUT / "ws4_breadth_verdicts.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n== {FAMILY} dev table (calmar bar {calmar_bar:.3f}, "
          f"geo bar {geo_bar:.4%}, DSR bar 0.95 at n={n_trials}) ==")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nVERDICT: {verdict['verdict']} "
          f"(best cell {best['name']}, calmar {best['calmar']:.3f})")
    print(f"wrote ws4_breadth_results.csv + ws4_breadth_verdicts.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
