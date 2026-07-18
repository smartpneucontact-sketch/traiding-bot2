"""WS4(b) — VIX term-structure regime gate on the combo_v2 book
(Stream 2, family=regime_vixterm, DEV-ONLY).

Hypothesis: short-dated implied vol trading above medium-dated implied vol
(^VIX9D / ^VIX3M backwardation) marks stress regimes in which the momentum
book should de-risk; a daily exposure gate keyed on the ratio may cut
drawdown at acceptable return cost — the mechanism class E4 said is right
(daily overlays act BETWEEN rebalances) applied to a signal the E1 realized-
vol overlays did not test (implied term structure is forward-looking).

PRE-COMMITTED GATE (written before any cell ran; template per the
Improvement Roadmap v2, Stream 2):
    dev DSR >= 0.95 at the current deduped trial count
    AND dev Calmar >= baseline + 10%   (baseline = catalog_v2 combo_v2_2x
                                        dev row at margin600)
    AND geo give-up <= 0.2pp/mo        (dev geo >= baseline dev geo - 0.2pp)
    -> eligible for ONE-SHOT validation LATER (no final=True / val runs in
       this phase; dev verdicts only).

Grid (10 cells <= 12, all ledgered, family regime_vixterm, margin 600):
    anchor: ungated base (fidelity cell — must reproduce the catalog_v2
            combo_v2_2x dev row; also the in-family comparator)      [1]
    ratio threshold s = ^VIX9D/^VIX3M in {0.95, 1.00, 1.05}
      x exposure multiplier while s > threshold in {0.0, 0.3, 0.5}   [9]

Construction: the three cached combo_v2 sleeves (cache-hit ONLY under the
current code hash — provably the catalog frames), blended /3, x2.0 leverage,
expanded to a daily frame, gated via engine_v2.apply_gate (the gate scales
the decision-side frame; the engine's shift(1) then delays it one day, so a
close-t signal trades at open t+1 — no lookahead), scored next_open, tc=5bp,
margin_bps_annual=600, leverage_cap=2.0, window="dev".

Data: ^VIX9D and ^VIX3M through the provenance-frozen data_free layer.
^VIX9D history begins 2011 (CBOE backfill), fully covering the dev window;
days without both quotes inherit the prior gate value (ffill inside
apply_gate) and pre-history defaults to gate=1.0.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python exp_ws4_vixterm.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_free import fetch_yf_series
from engine_v2 import apply_gate, expand_daily
from exp_lib import (_CODE_HASH, TRIALS_DIR, WEIGHTS_STORE, run_trial,
                     trial_count, union_prices_cached)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v7"
FAMILY = "regime_vixterm"
LEDGER = TRIALS_DIR / f"{FAMILY}.csv"
MARGIN_BPS = 600.0
LEV = 2.0

THRESHOLDS = (0.95, 1.00, 1.05)
MULTS = (0.0, 0.3, 0.5)

SLEEVE_KEYS = ("sleeve_xs_momentum_n30", "sleeve_dual_momentum_voltarget_n30",
               "sleeve_adaptive_voltarget_n30")

# Anchor fidelity targets = catalog_v2 combo_v2_2x dev row (margin600).
ANCHOR_TOL = 1e-6


def cached_frame_or_die(key: str) -> pd.DataFrame:
    p = WEIGHTS_STORE / f"{key}__{_CODE_HASH}.parquet"
    if not p.exists():
        raise SystemExit(f"cache miss: {p.name} — sleeves are not the "
                         f"catalog frames; aborting.")
    return pd.read_parquet(p)


def build_base_daily() -> pd.DataFrame:
    """combo_v2 blend x2 as a DAILY frame on the union-price index."""
    frames = [cached_frame_or_die(k) for k in SLEEVE_KEYS]
    dates = sorted(set().union(*[f.index for f in frames]))
    cols = sorted(set().union(*[f.columns for f in frames]))
    blend = sum(f.reindex(index=dates, columns=cols, fill_value=0.0)
                for f in frames) / 3.0
    pu = union_prices_cached()
    return expand_daily(blend * LEV, pu.index)


def vix_ratio() -> pd.Series:
    v9 = fetch_yf_series("^VIX9D")
    v3m = fetch_yf_series("^VIX3M")
    s = (v9 / v3m).dropna()
    s.name = "vix9d_over_vix3m"
    return s


def baseline_row() -> pd.Series:
    cat = pd.read_csv(OUT / "catalog_v2.csv")
    r = cat[(cat["name"] == "combo_v2_2x") & (cat["window"] == "dev")]
    if r.empty:
        raise SystemExit("catalog_v2.csv lacks the combo_v2_2x dev row")
    return r.iloc[0]


def dsr_inputs() -> tuple[int, float]:
    from exp_catalog_v2 import ledger_sr_variance_daily
    return trial_count(dedupe=True), ledger_sr_variance_daily()


def main() -> None:
    t0 = time.time()
    base = baseline_row()
    base_geo_dev = float((1.0 + base["cagr"]) ** (1.0 / 12.0) - 1.0)
    print(f"baseline (catalog_v2 combo_v2_2x dev, margin600): "
          f"calmar {base['calmar']:.4f} | geo {base_geo_dev:.4%}/mo")

    s = vix_ratio()
    print(f"^VIX9D/^VIX3M: {len(s)} rows, {s.index[0].date()} -> "
          f"{s.index[-1].date()}")

    w_daily = build_base_daily()

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
                      weights_are_daily=True, final=False, notes=notes)
        row = r["row"]
        print(f"  {name:28s} geo? mm={row['mean_monthly']*100:5.2f}% "
              f"sharpe={row['sharpe']:.3f} dd={row['max_drawdown']*100:5.1f}% "
              f"calmar={row['calmar']:.3f}")

    print("\n[anchor] ungated base (fidelity cell)")
    cell("vixterm_base_2x", w_daily,
         {"gate": None, "lev": LEV, "sleeves": "xs30+dual30+adapt30"},
         notes="anchor: ungated combo_v2 blend, daily-expanded; must "
               "reproduce catalog_v2 combo_v2_2x dev (margin600)")

    print("\n[grid] 3 thresholds x 3 multipliers")
    for thr in THRESHOLDS:
        active = (s > thr)
        frac_dev = float(active.loc[:"2022-12-31"].mean())
        for mult in MULTS:
            gate = pd.Series(np.where(active, mult, 1.0), index=s.index)
            wg = apply_gate(w_daily, gate)
            cell(f"vixterm_thr{thr:g}_m{mult:g}", wg,
                 {"gate": "vix9d_over_vix3m", "threshold": thr,
                  "mult": mult, "lev": LEV,
                  "frac_active_dev": round(frac_dev, 4)},
                 notes=f"backwardation gate s>{thr:g} -> {mult:g}x; "
                       f"active {frac_dev:.1%} of dev days")

    # ── assemble + render-time DSR + gate verdicts ──
    led = pd.read_csv(LEDGER).drop_duplicates("name", keep="last")
    led["geo_monthly"] = (1.0 + led["cagr"]) ** (1.0 / 12.0) - 1.0
    anchor = led[led["name"] == "vixterm_base_2x"].iloc[0]
    d_calmar = abs(float(anchor["calmar"]) - float(base["calmar"]))
    d_sharpe = abs(float(anchor["sharpe"]) - float(base["sharpe"]))
    print(f"\nanchor fidelity vs catalog dev row: |dCalmar|={d_calmar:.2e} "
          f"|dSharpe|={d_sharpe:.2e}")
    if d_calmar > ANCHOR_TOL or d_sharpe > ANCHOR_TOL:
        raise SystemExit("ANCHOR FIDELITY FAILED — gated cells are not "
                         "overlays on the catalog book; results void.")

    n_trials, sr_var = dsr_inputs()
    from metrics_v2 import deflated_sharpe
    # dev window day count from the anchor run (same for all cells).
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
    tbl.to_csv(OUT / "ws4_vixterm_results.csv", index=False)

    grid = tbl[tbl["name"] != "vixterm_base_2x"]
    best = grid.iloc[0]
    verdict = {
        "family": FAMILY,
        "n_cells": int(len(led)),
        "baseline": {"calmar_dev": float(base["calmar"]),
                     "geo_dev": base_geo_dev,
                     "source": "results/v7/catalog_v2.csv combo_v2_2x dev"},
        "gates": {"dsr_min": 0.95, "calmar_min": calmar_bar,
                  "geo_min": geo_bar,
                  "n_trials_at_eval": n_trials},
        "best_cell": {k: (float(best[k]) if isinstance(best[k], (int, float, np.floating))
                          else bool(best[k]) if isinstance(best[k], (bool, np.bool_))
                          else str(best[k]))
                      for k in cols},
        "n_pass": int(grid["gate_all"].sum()),
        "verdict": ("ELIGIBLE for one-shot val LATER"
                    if bool(grid["gate_all"].any()) else
                    "FAIL dev gates — no val shot"),
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "code_hash": _CODE_HASH,
    }
    with open(OUT / "ws4_vixterm_verdicts.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n== {FAMILY} dev table (calmar bar {calmar_bar:.3f}, "
          f"geo bar {geo_bar:.4%}, DSR bar 0.95 at n={n_trials}) ==")
    print(tbl.to_string(index=False,
                        float_format=lambda v: f"{v:.4f}"))
    print(f"\nVERDICT: {verdict['verdict']} "
          f"(best cell {best['name']}, calmar {best['calmar']:.3f})")
    print(f"wrote ws4_vixterm_results.csv + ws4_vixterm_verdicts.json "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
