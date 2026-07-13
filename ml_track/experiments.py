"""Stage 1 experiment machinery: dataset build/cache, purged-WF OOF,
Gate 1, sleeve construction, I2 integration, ledger logging."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import exp_lib
from ml_track import config as C
from ml_track.candidates import candidate_mask
from ml_track.cv import PurgedAnchoredWF
from ml_track.features import F1_COLS, build_long_features
from ml_track.labels import build_labels
from ml_track.portfolio import build_decision_frame
from ml_track.train import (calibrate_isotonic, fit_fold, predict,
                            signal_metrics)

PKG = Path(__file__).resolve().parent
CACHE = PKG / "cache"
RESULTS = PKG / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

LEDGER = RESULTS / "ledger.csv"
LEDGER_COLS = [
    "ts", "exp_id", "arch", "label", "features", "hp", "pool", "weighting",
    "grid", "monotone", "n_train_rows", "rank_ic", "rank_ic_t", "prior_ic",
    "ic_vs_prior", "pct_pos_years", "hit30", "g1_pass", "ic_by_year",
    "port_mean_monthly", "port_sharpe", "port_max_dd", "port_calmar",
    "port_turnover", "port_avg_gross", "runtime_s", "notes",
]

_MEM: dict = {}


# ───────────────────────────── dataset ─────────────────────────────

def _grid_sets():
    """(dates_5d, dates_21d, portfolio_dates) all <= DEV_END, i >= 273."""
    panel, _, _ = exp_lib.load_cache()
    idx = panel["close"].index
    cutoff = pd.Timestamp(C.DEV_END)
    pos_ok = np.arange(len(idx)) >= C.MAX_LOOKBACK_BARS
    in_dev = idx <= cutoff
    d5 = idx[pos_ok & in_dev & (np.arange(len(idx)) % C.TRAIN_GRID == 0)]
    d21 = idx[pos_ok & in_dev & (np.arange(len(idx)) % C.REBAL == 0)]
    dual = pd.read_parquet(
        Path(exp_lib.ROOT) / "weights_store" /
        "sleeve_dual_momentum_voltarget_n30.parquet")
    port = pd.DatetimeIndex(
        [d for d in dual.index
         if d <= cutoff and d in idx
         and idx.get_loc(d) >= C.MAX_LOOKBACK_BARS])
    return d5, d21, port


def get_dataset(pool: int) -> pd.DataFrame:
    """Long frame indexed (date, sym): 36 F1 cols + label cols + sigma60 +
    label_end_date, for the union grid (5d ∪ 21d ∪ portfolio dates)."""
    key = f"ds_p{pool}"
    if key in _MEM:
        return _MEM[key]
    fpath = CACHE / ("features_F1.parquet" if pool == C.CAND_POOL
                     else f"features_F1_p{pool}.parquet")
    lpath = CACHE / f"labels_p{pool}.parquet"

    if fpath.exists() and lpath.exists():
        feats = pd.read_parquet(fpath)
        labs = pd.read_parquet(lpath)
    else:
        panel, macro, sector_map = exp_lib.load_cache()
        d5, d21, port = _grid_sets()
        dates = d5.union(d21).union(port)
        t0 = time.time()
        mask = candidate_mask(panel["close"], dates, pool=pool)
        feats = build_long_features(panel, macro, sector_map, mask)
        labs = build_labels(panel["close"], feats.index)
        feats.to_parquet(fpath)
        labs.to_parquet(lpath)
        print(f"[dataset p{pool}] {len(feats):,} rows, "
              f"{len(dates)} dates, {time.time()-t0:.1f}s")
    ds = feats.join(labs)
    _MEM[key] = ds
    return ds


# ───────────────────────────── walk-forward ─────────────────────────────

def run_walkforward(arch: str, label: str, pool: int, grid: int,
                    monotone: bool, hp_name: str = "HP_S",
                    legacy_calibrated_metrics: bool = False) -> dict:
    """Purged anchored WF -> pooled OOF scores + signal metrics. Memoized on
    the model key so weighting-only variants reuse the same OOF.

    legacy_calibrated_metrics: reproduce the pre-2026-07 (leaky) B_cls
    isotonic step for ledger reproduction/audits only — see the comment at
    the calibration block below. Default False = raw scores everywhere."""
    mkey = (f"wf_{arch}_{label}_p{pool}_g{grid}_m{int(monotone)}_{hp_name}"
            + ("_legacyiso" if legacy_calibrated_metrics else ""))
    if mkey in _MEM:
        return _MEM[mkey]
    t0 = time.time()
    hp = getattr(C, hp_name)
    ds = get_dataset(pool).copy()
    panel, _, _ = exp_lib.load_cache()
    d5, d21, port = _grid_sets()
    train_dates = d5 if grid == 5 else d21

    dates_lvl = ds.index.get_level_values(0)
    ds["_is_train_grid"] = dates_lvl.isin(train_dates)
    ds["_is_eval_grid"] = dates_lvl.isin(d5)       # common eval rows
    ds["_is_port"] = dates_lvl.isin(port)

    ycol = {"A_rank": label, "A_reg": label,
            "B_cls": {"L1": "L3", "L2": "L3v"}[label]}[arch]

    dser = pd.Series(dates_lvl, index=ds.index)
    cv = PurgedAnchoredWF(panel["close"].index)
    oof_parts = []
    n_train_rows = 0
    for year, core, es, test in cv.folds(dser, ds["label_end_date"]):
        tr = ds[core & ds["_is_train_grid"] & ds[ycol].notna()]
        ev = ds[es & ds["_is_train_grid"] & ds[ycol].notna()]
        te = ds[test]
        if len(tr) < 500 or len(ev) < 50 or len(te) == 0:
            print(f"  fold {year}: SKIPPED (rows tr={len(tr)} es={len(ev)})")
            continue
        n_train_rows += len(tr) + len(ev)
        mdl = fit_fold(arch, hp,
                       tr[F1_COLS], tr[ycol],
                       pd.Series(tr.index.get_level_values(0), index=tr.index),
                       ev[F1_COLS], ev[ycol],
                       pd.Series(ev.index.get_level_values(0), index=ev.index),
                       monotone=monotone)
        sc = te[["r_fwd21", "sigma60", "mom_12_1_r",
                 "_is_eval_grid", "_is_port"]].copy()
        sc["score"] = predict(arch, mdl, te[F1_COLS])
        sc["fold_year"] = year
        oof_parts.append(sc)
    oof = pd.concat(oof_parts).sort_index()

    # B_cls isotonic calibration — REMOVED from the default path (2026-07
    # audit, leakage). The original code fit IsotonicRegression on POOLED
    # out-of-fold scores vs REALIZED labels across all test years, then
    # re-scored those same rows: the fit is monotone but its tie-flattening
    # is informed by future outcomes and altered the Spearman-based Gate-1
    # metrics and the I2 decision frame. Measured impact: S1_05_Bcls_L1
    # rank_ic 0.0326 calibrated vs 0.0246 raw — the leak pushed it over the
    # 0.03 gate (ledger rows logged before this fix carry calibrated values).
    # Gate-1 metrics and build_decision_frame are rank-based, so calibration
    # is unnecessary; raw predict_proba scores are used. If calibrated
    # probabilities are ever genuinely consumed downstream, fit the
    # calibrator per-fold on that fold's TRAINING rows only.
    if arch == "B_cls" and legacy_calibrated_metrics:
        ev_rows = oof["_is_eval_grid"] & oof["r_fwd21"].notna()
        ycol_bin = {"L1": "L3", "L2": "L3v"}[label]
        ybin = get_dataset(pool).loc[oof.index[ev_rows], ycol_bin].to_numpy()
        iso = calibrate_isotonic(oof.loc[ev_rows, "score"].to_numpy(), ybin)
        oof["score"] = iso.predict(oof["score"].to_numpy())

    mets = signal_metrics(oof[oof["_is_eval_grid"]])
    mets["n_train_rows"] = n_train_rows
    mets["runtime_s"] = round(time.time() - t0, 1)
    out = {"oof": oof, "metrics": mets}
    _MEM[mkey] = out
    return out


def gate1(m: dict) -> bool:
    return (m["rank_ic"] >= C.G1_RANK_IC
            and m["ic_minus_prior"] >= C.G1_IC_EDGE_OVER_PRIOR
            and m["pct_pos_years"] >= C.G1_PCT_POS_YEARS
            and m["hit30"] >= C.G1_HIT30)


# ───────────────────────────── experiment driver ─────────────────────────────

def run_experiment(exp_id: str, spec: dict, force_portfolio: bool = False) -> dict:
    print(f"\n=== {exp_id}: {spec} ===")
    wf = run_walkforward(spec["arch"], spec["label"], spec["pool"],
                         spec["grid"], spec["monotone"], spec["hp"])
    m = wf["metrics"]
    g1 = gate1(m)
    print(f"  rank_ic={m['rank_ic']:+.4f} prior={m['prior_ic']:+.4f} "
          f"edge={m['ic_minus_prior']:+.4f} pos_yrs={m['pct_pos_years']:.0%} "
          f"hit30={m['hit30']:.3f} -> G1 {'PASS' if g1 else 'FAIL'}")

    row = {
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "exp_id": exp_id, "arch": spec["arch"], "label": spec["label"],
        "features": "F1", "hp": spec["hp"], "pool": spec["pool"],
        "weighting": spec["weighting"], "grid": spec["grid"],
        "monotone": spec["monotone"], "n_train_rows": m["n_train_rows"],
        "rank_ic": round(m["rank_ic"], 5), "rank_ic_t": round(m["rank_ic_t"], 2),
        "prior_ic": round(m["prior_ic"], 5),
        "ic_vs_prior": round(m["ic_minus_prior"], 5),
        "pct_pos_years": m["pct_pos_years"], "hit30": round(m["hit30"], 4),
        "g1_pass": g1, "ic_by_year": json.dumps(m["ic_by_year"]),
        "port_mean_monthly": None, "port_sharpe": None, "port_max_dd": None,
        "port_calmar": None, "port_turnover": None, "port_avg_gross": None,
        "runtime_s": m["runtime_s"], "notes": "",
    }

    if g1 or force_portfolio:
        from ml_track.backtest_adapter import run_ml_sleeve
        oof = wf["oof"]
        psc = oof[oof["_is_port"]][["score", "sigma60"]]
        panel, _, _ = exp_lib.load_cache()
        frame = build_decision_frame(psc, panel["close"].columns,
                                     weighting=spec["weighting"])
        res = run_ml_sleeve(
            frame, name=f"I2_{exp_id}",
            params={**spec, "exp_id": exp_id},
            notes=("G1 pass" if g1 else "G1 FAIL (diagnostic portfolio run)"))
        s = res["summary"]
        row.update(port_mean_monthly=s.get("mean_monthly"),
                   port_sharpe=s.get("sharpe"),
                   port_max_dd=s.get("max_drawdown"),
                   port_calmar=s.get("calmar"),
                   port_turnover=s.get("turnover_annualized"),
                   port_avg_gross=s.get("avg_gross_exposure"))
        print(f"  I2: mm={s['mean_monthly']:.4%} sharpe={s['sharpe']:.3f} "
              f"dd={s['max_drawdown']:.1%} calmar={s['calmar']:.3f}")

    pd.DataFrame([row])[LEDGER_COLS].to_csv(
        LEDGER, mode="a", header=not LEDGER.exists(), index=False)
    return row
