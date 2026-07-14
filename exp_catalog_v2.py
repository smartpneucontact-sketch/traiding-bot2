"""WS2 — Corrected catalog regeneration (plan: what-can-we-do roadmap, Track B).

Re-scores the full exp_rescore.REGISTRY (48 entries) under the CORRECTED
reporting configuration the 2026-07-12 audit established:

    exec_model = next_open          (realistic execution)
    margin_bps_annual = 600         (6%/yr financing on long gross > 1.0x NAV)
    headline metric = geo_monthly   ((1+CAGR)^(1/12)-1 — what an account earns)

EXCEPTION GRANTED (same precedent as exp_rescore.py): window="full",
final=True — this is a CORRECTION of the published record under fixed,
pre-declared settings, not a search. Every run is still ledgered
(family="catalog_v2") and counts toward the deflated-Sharpe trial count.

Margin note: run_trial embeds margin_bps_annual in params_json when non-zero,
so these rows dedupe as distinct trials from the margin-free "rescore" family,
and the family CSV is new so the margin_bps column is in its header from row 1.

Verification built in (run after the sweep):
  - champion cross-check: combo_v2 2x full-window row must reproduce the
    REPORT.md corrected headline (mean_monthly 4.80%, geo 3.68%, sharpe 1.06,
    maxDD -60.3%) — tolerances below; failing hard-stops the assembly.

Importing exp_rescore triggers its module-level data load + sleeve cache
(re)build; after a strategies.py edit flips exp_lib._CODE_HASH the first run
rebuilds all sleeves (~30-45 min). Resume-safe: completed (name, window) pairs
are skipped on re-run.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python exp_catalog_v2.py
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from exp_lib import TRIALS_DIR, cached_weights, run_trial, trial_count
from metrics_v2 import deflated_sharpe


def ledger_sr_variance_daily() -> float:
    """Cross-trial Sharpe variance (daily scale) from the actual ledgers.

    deflated_sharpe's fallback (v = observed daily SR²) makes the expected
    max-SR benchmark ~3x the observed SR at ~500 trials — DSR degenerates to
    0.000 for every strategy regardless of merit. The Bailey-LdP estimator
    wants the variance of SR estimates ACROSS the trials actually run, which
    the ledgers record; deduplicate the same way trial_count does so re-runs
    don't shrink the variance.
    """
    srs = []
    for f in TRIALS_DIR.glob("*.csv"):
        df = pd.read_csv(f)
        keys = [c for c in ("family", "name", "window", "exec_model",
                            "tc_bps", "leverage_cap", "params_json")
                if c in df.columns]
        df = df.drop_duplicates(subset=keys, keep="last")
        srs.append(df["sharpe"].dropna())
    all_sr = pd.concat(srs) / np.sqrt(252.0)  # annual -> daily scale
    return float(all_sr.var())

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v7"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = TRIALS_DIR / "catalog_v2.csv"

MARGIN_BPS = 600.0
FAMILY = "catalog_v2"
NOTE = "catalog_v2 correction exception (exp_rescore precedent): fixed settings, not a search"

# Champion cross-check targets = REPORT.md Validity Notice corrected headline.
CHAMPION = "combo_v2_2x"
CHAMPION_TOL = {"mean_monthly": (0.0480, 0.0015), "sharpe": (1.06, 0.03),
                "max_drawdown": (-0.603, 0.01)}


def main() -> None:
    # Import here so the (slow) module-level data load + sleeve builds happen
    # inside main, after argparse-free startup prints.
    print("importing exp_rescore (data load + sleeve cache — may rebuild after code changes)...")
    t0 = time.time()
    from exp_rescore import REGISTRY, SHORT_CLIPPED  # noqa: E402
    print(f"  ready in {time.time()-t0:.0f}s — {len(REGISTRY)} registry entries")

    done: set[tuple[str, str]] = set()
    if LEDGER.exists():
        led = pd.read_csv(LEDGER)
        done = set(zip(led["name"], led["window"]))
        print(f"[resume] {len(done)} runs already in ledger")

    errors = []
    for name, builder, tc, cap, daily, source, params in REGISTRY:
        t0 = time.time()
        try:
            w = cached_weights(f"catalog_{name}", builder)
        except Exception:
            errors.append((name, "build", traceback.format_exc()))
            print(f"[FAIL build] {name}")
            continue
        note = NOTE + ("; shorts_clipped" if name in SHORT_CLIPPED else "")
        p = dict(params)
        p.update({"source": source, "tc_bps": tc, "leverage_cap": cap})
        for window in ("full", "dev"):
            if (name, window) in done:
                continue
            try:
                run_trial(w, name=name, family=FAMILY, params=p, window=window,
                          exec_model="next_open", tc_bps=tc, leverage_cap=cap,
                          margin_bps_annual=MARGIN_BPS,
                          weights_are_daily=daily, final=(window == "full"),
                          notes=note)
            except Exception:
                errors.append((name, window, traceback.format_exc()))
                print(f"[FAIL run] {name} {window}")
        print(f"[done] {name:34s} ({source})  {time.time()-t0:6.1f}s")

    assemble(errors)


def assemble(errors: list) -> None:
    led = pd.read_csv(LEDGER)
    led = led[led["family"] == FAMILY].copy()
    led = led.drop_duplicates(subset=["name", "window"], keep="last")

    # geo_monthly at render time from the logged CAGR (run_trial rows do not
    # carry summary()'s geo_monthly key).
    led["geo_monthly"] = (1.0 + led["cagr"]) ** (1.0 / 12.0) - 1.0

    # Render-time DSR: deflates as the program-wide deduped trial count grows.
    # Cross-trial SR variance measured from the ledgers (see helper above).
    n_trials = trial_count(dedupe=True)
    sr_var = ledger_sr_variance_daily()
    full = led[led["window"] == "full"].copy()
    full["dsr"] = [
        deflated_sharpe(s, int(n), n_trials, sr_variance_across_trials=sr_var)
        if np.isfinite(s) and np.isfinite(n) else float("nan")
        for s, n in zip(full["sharpe"], full.get("n_days", pd.Series(2512, index=full.index)).fillna(2512))
    ]
    led = led.merge(full[["name", "dsr"]], on="name", how="left")

    led = led.sort_values(["window", "geo_monthly"], ascending=[True, False])
    out_csv = OUT / "catalog_v2.csv"
    led.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(led)} rows; n_trials for DSR = {n_trials})")

    # ── champion cross-check (hard gate) ──
    ch = full[full["name"] == CHAMPION]
    if ch.empty:
        print(f"WARNING: champion {CHAMPION!r} not in registry results — cross-check skipped")
    else:
        row = ch.iloc[0]
        ok = True
        print(f"\n══ champion cross-check ({CHAMPION}, full window) ══")
        for k, (target, tol) in CHAMPION_TOL.items():
            d = abs(float(row[k]) - target)
            flag = "OK " if d <= tol else "FAIL"
            ok &= d <= tol
            print(f"  {flag} {k:14s} got={float(row[k]):+.4f} target={target:+.4f} tol={tol}")
        geo = float(row["geo_monthly"])
        flag = "OK " if abs(geo - 0.0368) <= 0.002 else "FAIL"
        ok &= abs(geo - 0.0368) <= 0.002
        print(f"  {flag} geo_monthly    got={geo:+.4f} target=+0.0368 tol=0.002")
        if not ok:
            raise SystemExit("CHAMPION CROSS-CHECK FAILED — do not publish catalog_v2")

    top = led[led["window"] == "full"].head(15)
    print("\n══ top 15 by geo_monthly (full, next_open, 6%/yr margin) ══")
    for _, r in top.iterrows():
        print(f"  {r['name']:34s} geo {r['geo_monthly']*100:+.2f}%/mo | arith {r['mean_monthly']*100:+.2f}% | "
              f"sharpe {r['sharpe']:.2f} | maxDD {r['max_drawdown']*100:.1f}% | DSR {r['dsr']:.3f}")

    if errors:
        print(f"\n{len(errors)} ERRORS (details in {OUT/'catalog_v2_errors.txt'}):")
        for n, stage, tb in errors:
            print(f"  {n} [{stage}]: {tb.splitlines()[-1]}")
    with open(OUT / "catalog_v2_errors.txt", "w") as f:
        for n, stage, tb in errors:
            f.write(f"--- {n} [{stage}] ---\n{tb}\n")


if __name__ == "__main__":
    main()
