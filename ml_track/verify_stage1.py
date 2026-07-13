"""Post-hoc audit of the Stage 1 CORE-12 run (no ledger writes).

Checks:
  V1. Determinism — re-run the purged WF for S1_01 (A_rank/L1) and S1_05
      (B_cls/L1) in a fresh process and compare signal metrics to the ledger.
  V2. Look-ahead test — rebuild the wide features on a panel TRUNCATED at a
      sample decision date and confirm the feature row at that date is
      identical to the full-panel build (no future data enters any feature).
  V3. Label convention — for a sample (date, sym), recompute
      close[t+22]/close[t+1]-1 by hand and compare to the cached r_fwd21;
      check label_end_date == idx[t+22].
  V4. Gate-1 arithmetic — recompute g1_pass for all 12 ledger rows from the
      logged columns and compare to the logged verdict.
  V5. Raw-score (pre-isotonic) check for B_cls — reproduces the ad-hoc
      verification rows: isotonic ties can change Spearman IC; report both.

NOTE (2026-07 audit): the audit fixes post-date the Stage-1 ledger — the
core/ES boundary purge in cv.py changes fold training rows, train.py now
sets deterministic/force_col_wise, and B_cls defaults to RAW scores (the
pooled-OOF isotonic was a leak; S1_05 rank_ic 0.0326 calibrated vs 0.0246
raw). V1 therefore re-runs B_cls with legacy_calibrated_metrics=True and is
reported as an INFORMATIONAL drift check, excluded from the final verdict:
re-runs of the fixed pipeline are not expected to bit-match the pre-fix
ledger.

Run:  /opt/anaconda3/bin/python3 -m ml_track.verify_stage1
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import json

import numpy as np
import pandas as pd

import exp_lib
from ml_track import config as C
from ml_track.experiments import LEDGER, get_dataset, run_walkforward
from ml_track.features import build_wide_features
from ml_track.train import signal_metrics

TOL = 1e-9


def v1_determinism():
    print("── V1 determinism (fresh-process WF re-run vs ledger) ──")
    print("  [informational since the 2026-07 audit fixes — see module "
          "docstring; drift vs the pre-fix ledger is expected]")
    led = pd.read_csv(LEDGER)
    ok = True
    for exp_id in ("S1_01_Arank_L1", "S1_05_Bcls_L1"):
        spec = C.STAGE1_EXPERIMENTS[exp_id]
        wf = run_walkforward(spec["arch"], spec["label"], spec["pool"],
                             spec["grid"], spec["monotone"], spec["hp"],
                             legacy_calibrated_metrics=(spec["arch"] == "B_cls"))
        m = wf["metrics"]
        row = led[led.exp_id == exp_id].iloc[-1]
        for col, key, nd in (("rank_ic", "rank_ic", 5),
                             ("prior_ic", "prior_ic", 5),
                             ("hit30", "hit30", 4)):
            got, ref = round(m[key], nd), row[col]
            match = abs(got - ref) < 10 ** (-nd) / 2 + 1e-12
            ok &= match
            print(f"  {exp_id} {col}: rerun={got} ledger={ref} "
                  f"{'OK' if match else 'MISMATCH'}")
    return ok


def v2_lookahead():
    print("── V2 look-ahead (truncated-panel feature equality) ──")
    panel, macro, sector_map = exp_lib.load_cache()
    idx = panel["close"].index
    t = idx[800]  # a mid-dev decision date with full history
    full = build_wide_features(panel, macro, sector_map)
    panel_tr = {k: v.loc[:t] for k, v in panel.items()}
    trunc = build_wide_features(panel_tr, macro.loc[:t], sector_map)
    ok = True
    for name, w in full.items():
        a = (w.loc[t] if isinstance(w, pd.DataFrame) else
             pd.Series([w.loc[t]]))
        b = (trunc[name].loc[t] if isinstance(w, pd.DataFrame) else
             pd.Series([trunc[name].loc[t]]))
        diff = (a - b).abs()
        bad = float(diff.max(skipna=True)) if diff.notna().any() else 0.0
        same_nan = bool((a.isna() == b.isna()).all())
        if bad > 1e-10 or not same_nan:
            ok = False
            print(f"  {name}: LEAK? max|diff|={bad:.3e} nan_match={same_nan}")
    print(f"  all 36 features identical at {t.date()} on truncated panel: "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def v3_label():
    print("── V3 label convention (hand recompute) ──")
    panel, _, _ = exp_lib.load_cache()
    close = panel["close"]
    idx = close.index
    ds = get_dataset(80)
    samp = ds.dropna(subset=["r_fwd21"]).sample(25, random_state=0)
    ok = True
    for (dt, sym), row in samp.iterrows():
        p = idx.get_loc(dt)
        manual = close[sym].iloc[p + C.HORIZON + 1] / close[sym].iloc[p + 1] - 1
        end_ok = pd.Timestamp(row["label_end_date"]) == idx[p + C.HORIZON + 1]
        if not (abs(manual - row["r_fwd21"]) < 1e-6 and end_ok):
            ok = False
            print(f"  {dt.date()} {sym}: manual={manual:.6f} "
                  f"cached={row['r_fwd21']:.6f} end_ok={end_ok}  MISMATCH")
    print(f"  25/25 sampled labels match close[t+22]/close[t+1]-1: "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def v4_gate():
    print("── V4 Gate-1 arithmetic on all 12 core rows ──")
    led = pd.read_csv(LEDGER)
    core = led[led.exp_id.str.match(r"S1_\d\d_") &
               ~led.exp_id.str.contains("diag")]
    ok = True
    for _, r in core.iterrows():
        expect = (r.rank_ic >= C.G1_RANK_IC
                  and r.ic_vs_prior >= C.G1_IC_EDGE_OVER_PRIOR
                  and r.pct_pos_years >= C.G1_PCT_POS_YEARS
                  and r.hit30 >= C.G1_HIT30)
        if bool(r.g1_pass) != expect:
            ok = False
            print(f"  {r.exp_id}: logged={r.g1_pass} recomputed={expect}")
    print(f"  12/12 logged verdicts match recomputed Gate-1: "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def v5_rawscore():
    print("── V5 B_cls raw (pre-isotonic) scores — anomaly reproduction ──")
    # Reproduce the V_05/V_06/V_08 ledger rows: run the WF but compute
    # metrics on the raw predict_proba scores (no isotonic step).
    from ml_track.cv import PurgedAnchoredWF
    from ml_track.features import F1_COLS
    from ml_track.train import fit_fold, predict
    out = {}
    for tag, (label, pool) in {"V_05": ("L1", 80), "V_06": ("L2", 80),
                               "V_08": ("L1", 120)}.items():
        ds = get_dataset(pool).copy()
        panel, _, _ = exp_lib.load_cache()
        from ml_track.experiments import _grid_sets
        d5, d21, port = _grid_sets()
        lvl = ds.index.get_level_values(0)
        ds["_is_train_grid"] = lvl.isin(d5)
        ds["_is_eval_grid"] = lvl.isin(d5)
        ycol = {"L1": "L3", "L2": "L3v"}[label]
        dser = pd.Series(lvl, index=ds.index)
        cv = PurgedAnchoredWF(panel["close"].index)
        parts = []
        for year, core, es, test in cv.folds(dser, ds["label_end_date"]):
            tr = ds[core & ds["_is_train_grid"] & ds[ycol].notna()]
            ev = ds[es & ds["_is_train_grid"] & ds[ycol].notna()]
            te = ds[test]
            if len(tr) < 500 or len(ev) < 50 or len(te) == 0:
                continue
            mdl = fit_fold("B_cls", C.HP_S, tr[F1_COLS], tr[ycol],
                           pd.Series(tr.index.get_level_values(0), index=tr.index),
                           ev[F1_COLS], ev[ycol],
                           pd.Series(ev.index.get_level_values(0), index=ev.index))
            sc = te[["r_fwd21", "sigma60", "mom_12_1_r", "_is_eval_grid"]].copy()
            sc["score"] = predict("B_cls", mdl, te[F1_COLS])
            parts.append(sc)
        oof = pd.concat(parts).sort_index()
        m = signal_metrics(oof[oof["_is_eval_grid"]])
        out[tag] = m
        print(f"  {tag} ({label}, p{pool}) raw rank_ic={m['rank_ic']:+.4f} "
              f"edge={m['ic_minus_prior']:+.4f} pos_yrs={m['pct_pos_years']:.0%} "
              f"hit30={m['hit30']:.4f}")
    return out


if __name__ == "__main__":
    r1 = v1_determinism()
    r2 = v2_lookahead()
    r3 = v3_label()
    r4 = v4_gate()
    r5 = v5_rawscore()
    # V1 excluded from the verdict since the 2026-07 audit fixes (expected
    # drift vs the pre-fix ledger); reported above as informational.
    print("\nVERDICT:", "ALL CHECKS PASSED" if (r2 and r3 and r4)
          else "CHECK FAILURES — see above")
