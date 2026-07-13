"""ML-2 rescue follow-ups: generalized walk-forward (feature set / label /
HP / pool / seed) reusing the tested Stage-1 machinery.

Additions over ml_track.experiments:
  - F2 feature set = F1 (36) + 17 new columns. Formulas ported from
    Trading bot 6 deploy/core/features.py (compute_stock_features /
    compute_stock_features_v6) — the same source F1 was ported from:
      rsi_14, sma_20/50/200 dists, sma_50_200_cross, macd_hist, bb_20,
      atr_14, ret_zscore_20, sharpe_60, trend_dev_20, kurt_60,
      vol_ratio_5_20, day_range, + cs-ranks vol_20_r / sharpe_60_r /
      rsi_14_r (within the candidate set per date, like F1's _r cols).
    NOTE on lookback cap: macd_hist and trend_dev_20 use EMAs (pandas ewm
    from series start — backward-only, so the look-ahead audit property is
    preserved). Their EFFECTIVE lookback is < 273 bars (weight on bar 273
    back < 1e-9 for span 26; < 4e-12 for span 20); declared at 273.
  - L4 label = per-date cross-sectional OLS residual of L1 (= pctrank of
    r_fwd21) on mom_12_1_r (= pctrank of 12-1 momentum within the candidate
    set), re-pct-ranked within date so the ranker's quintile bucketing
    (label*5) stays valid. Rows with NaN L1 stay NaN. L4b = median split
    of L4 (the L3-analog binary target, so B_cls x L4 is runnable).
  - run_walkforward_x: same purged anchored WF as Stage 1 (memoized), but
    parameterized by feature set and seed (for Stage-3 seed jitter).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import exp_lib
from ml_track import config as C
from ml_track import features as FT
from ml_track.candidates import candidate_mask
from ml_track.cv import PurgedAnchoredWF
from ml_track.experiments import (CACHE, LEDGER, LEDGER_COLS, _grid_sets,
                                  get_dataset, gate1)
from ml_track.labels import build_labels
from ml_track.train import fit_fold, predict, signal_metrics

EPS = 1e-8

F2_NEW = [
    "rsi_14", "sma_20_dist", "sma_50_dist", "sma_200_dist",
    "sma_50_200_cross", "macd_hist", "bb_20", "atr_14", "ret_zscore_20",
    "sharpe_60", "trend_dev_20", "kurt_60", "vol_ratio_5_20", "day_range",
    "vol_20_r", "sharpe_60_r", "rsi_14_r",
]
F2_COLS = FT.F1_COLS + F2_NEW
assert len(F2_COLS) == 53

F2_RANK_COLS = {"vol_20": "vol_20_r", "sharpe_60": "sharpe_60_r",
                "rsi_14": "rsi_14_r"}

_MEM: dict = {}


# ───────────────────────────── F2 features ─────────────────────────────

def build_wide_f2(panel: dict) -> dict[str, pd.DataFrame]:
    """The 15 new wide builders (raw cols; _r ranks added at long-build)."""
    c, h, lo = panel["close"], panel["high"], panel["low"]
    ret = c.pct_change(fill_method=None)
    out: dict[str, pd.DataFrame] = {}

    delta = c.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    rs = up.rolling(14).mean() / (dn.rolling(14).mean() + EPS)
    out["rsi_14"] = 100 - 100 / (1 + rs)

    for p, name in [(20, "sma_20_dist"), (50, "sma_50_dist"),
                    (200, "sma_200_dist")]:
        sma = c.rolling(p).mean()
        out[name] = (c - sma) / (sma + EPS) * 100
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    out["sma_50_200_cross"] = (sma50 - sma200) / (sma200 + EPS) * 100

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - sig) / (c + EPS) * 100

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    out["bb_20"] = (c - (mid - 2 * sd)) / (4 * sd + EPS)

    cp = c.shift(1)
    tr = pd.DataFrame(
        np.fmax((h - lo).to_numpy(),
                np.fmax((h - cp).abs().to_numpy(),
                        (lo - cp).abs().to_numpy())),
        index=c.index, columns=c.columns)
    out["atr_14"] = tr.rolling(14).mean() / (c + EPS) * 100

    rm20 = ret.rolling(20)
    out["ret_zscore_20"] = (ret - rm20.mean()) / (rm20.std() + EPS)
    rm60 = ret.rolling(60)
    out["sharpe_60"] = rm60.mean() / (rm60.std() + EPS) * np.sqrt(252)

    ema20 = c.ewm(span=20, adjust=False).mean()
    out["trend_dev_20"] = (c - ema20) / (ema20 + EPS) * 100

    out["kurt_60"] = ret.rolling(60).kurt()
    out["vol_ratio_5_20"] = ret.rolling(5).std() / (ret.rolling(20).std() + EPS)
    out["day_range"] = (h - lo) / (c + EPS) * 100
    out["vol_20"] = ret.rolling(20).std() * np.sqrt(252)
    return out


def build_long_features_f2(panel, macro, sector_map, cand_mask) -> pd.DataFrame:
    """Long F2 frame — same mechanics as features.build_long_features."""
    wide = FT.build_wide_features(panel, macro, sector_map)
    wide.update(build_wide_f2(panel))
    dates = cand_mask.index
    mask_long = cand_mask.stack(future_stack=True)
    keep_idx = mask_long[mask_long.fillna(False)].index

    cols = {}
    for raw, w in wide.items():
        if isinstance(w, pd.Series):
            continue
        cols[raw] = w.loc[dates].stack(future_stack=True).reindex(keep_idx)
    long = pd.DataFrame(cols)

    grp = long.groupby(level=0)
    ranks = dict(FT.RANK_COLS)
    ranks.update(F2_RANK_COLS)
    for raw, rname in ranks.items():
        long[rname] = grp[raw].rank(pct=True)

    date_lvl = long.index.get_level_values(0)
    for name in ("vix_pct_252", "spy_dist_200sma", "hyg_dist_50sma"):
        long[name] = wide[name].reindex(date_lvl).to_numpy()

    long = long[F2_COLS].astype(np.float32)
    long.index.names = ["date", "sym"]
    return long


def get_dataset_x(pool: int, fset: str) -> pd.DataFrame:
    """Dataset with the requested feature set (+ labels, + L4)."""
    key = f"dsx_p{pool}_{fset}"
    if key in _MEM:
        return _MEM[key]
    if fset == "F1":
        ds = get_dataset(pool).copy()
    else:
        fpath = CACHE / f"features_F2_p{pool}.parquet"
        lpath = CACHE / f"labels_p{pool}.parquet"
        if fpath.exists():
            feats = pd.read_parquet(fpath)
        else:
            panel, macro, sector_map = exp_lib.load_cache()
            d5, d21, port = _grid_sets()
            dates = d5.union(d21).union(port)
            t0 = time.time()
            mask = candidate_mask(panel["close"], dates, pool=pool)
            feats = build_long_features_f2(panel, macro, sector_map, mask)
            feats.to_parquet(fpath)
            print(f"[dataset F2 p{pool}] {len(feats):,} rows, "
                  f"{time.time()-t0:.1f}s")
        if not lpath.exists():  # labels are feature-set independent
            panel, _, _ = exp_lib.load_cache()
            labs = build_labels(panel["close"], feats.index)
            labs.to_parquet(lpath)
        labs = pd.read_parquet(lpath)
        ds = feats.join(labs)
    ds = add_l4(ds)
    _MEM[key] = ds
    return ds


# ───────────────────────────── L4 label ─────────────────────────────

def add_l4(ds: pd.DataFrame) -> pd.DataFrame:
    """L4 = pctrank-within-date of the residual from per-date OLS of L1 on
    mom_12_1_r (both already in [0,1]). Also adds L4b = median-split binary
    analog of L4 (mirrors L3/L3v) so B_cls x L4 has a target."""
    if "L4" in ds.columns and "L4b" in ds.columns:
        return ds
    x = ds["mom_12_1_r"].astype(float)
    y = ds["L1"].astype(float)
    ok = x.notna() & y.notna()
    xv = x.where(ok)
    yv = y.where(ok)
    g0 = lambda s: s.groupby(level=0)
    mx = g0(xv).transform("mean")
    my = g0(yv).transform("mean")
    xc = xv - mx
    yc = yv - my
    num = g0(xc * yc).transform("sum")
    den = g0(xc * xc).transform("sum")
    b = num / den.replace(0.0, np.nan)
    resid = yc - b * xc
    ds = ds.copy()
    ds["L4"] = resid.groupby(level=0).rank(pct=True)
    ds["L4b"] = (ds["L4"] > 0.5).astype(float)
    ds.loc[ds["L4"].isna(), "L4b"] = np.nan
    return ds


# ───────────────────────────── generalized WF ─────────────────────────────

def run_walkforward_x(arch: str, label: str, pool: int, grid: int,
                      monotone: bool, hp_name: str = "HP_S",
                      fset: str = "F1", seed: int = 7) -> dict:
    """Stage-1 purged anchored WF, generalized (fset/label L4/seed)."""
    mkey = f"wfx_{arch}_{label}_p{pool}_g{grid}_m{int(monotone)}_{hp_name}_{fset}_s{seed}"
    if mkey in _MEM:
        return _MEM[mkey]
    t0 = time.time()
    hp = getattr(C, hp_name)
    fcols = FT.F1_COLS if fset == "F1" else F2_COLS
    ds = get_dataset_x(pool, fset).copy()
    panel, _, _ = exp_lib.load_cache()
    d5, d21, port = _grid_sets()
    train_dates = d5 if grid == 5 else d21

    dates_lvl = ds.index.get_level_values(0)
    ds["_is_train_grid"] = dates_lvl.isin(train_dates)
    ds["_is_eval_grid"] = dates_lvl.isin(d5)
    ds["_is_port"] = dates_lvl.isin(port)

    if arch in ("A_rank", "A_reg"):
        ycol = label
    else:
        # B_cls needs a binary target: L3 (for L1), L3v (for L2), L4b (for
        # L4; built by add_l4). A bare {L1,L2}-only lookup here used to
        # crash B_cls x L4 runs with KeyError.
        bmap = {"L1": "L3", "L2": "L3v", "L4": "L4b"}
        if label not in bmap:
            raise ValueError(
                f"B_cls has no registered binary analog for label {label!r} "
                f"(known: {sorted(bmap)})")
        ycol = bmap[label]

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
                       tr[fcols], tr[ycol],
                       pd.Series(tr.index.get_level_values(0), index=tr.index),
                       ev[fcols], ev[ycol],
                       pd.Series(ev.index.get_level_values(0), index=ev.index),
                       monotone=monotone, seed=seed)
        sc = te[["r_fwd21", "sigma60", "mom_12_1_r",
                 "_is_eval_grid", "_is_port"]].copy()
        sc["score"] = predict(arch, mdl, te[fcols])
        sc["fold_year"] = year
        oof_parts.append(sc)
    oof = pd.concat(oof_parts).sort_index()

    mets = signal_metrics(oof[oof["_is_eval_grid"]])
    mets["n_train_rows"] = n_train_rows
    mets["runtime_s"] = round(time.time() - t0, 1)
    out = {"oof": oof, "metrics": mets}
    _MEM[mkey] = out
    return out


def log_signal_row(exp_id: str, spec: dict, m: dict, g1: bool,
                   notes: str = "", port: dict | None = None) -> dict:
    row = {
        "ts": pd.Timestamp.now().isoformat(timespec="seconds"),
        "exp_id": exp_id, "arch": spec["arch"], "label": spec["label"],
        "features": spec.get("fset", "F1"), "hp": spec["hp"],
        "pool": spec["pool"], "weighting": spec["weighting"],
        "grid": spec["grid"], "monotone": spec["monotone"],
        "n_train_rows": m["n_train_rows"],
        "rank_ic": round(m["rank_ic"], 5), "rank_ic_t": round(m["rank_ic_t"], 2),
        "prior_ic": round(m["prior_ic"], 5),
        "ic_vs_prior": round(m["ic_minus_prior"], 5),
        "pct_pos_years": m["pct_pos_years"], "hit30": round(m["hit30"], 4),
        "g1_pass": g1, "ic_by_year": json.dumps(m["ic_by_year"]),
        "port_mean_monthly": (port or {}).get("mean_monthly"),
        "port_sharpe": (port or {}).get("sharpe"),
        "port_max_dd": (port or {}).get("max_drawdown"),
        "port_calmar": (port or {}).get("calmar"),
        "port_turnover": (port or {}).get("turnover_annualized"),
        "port_avg_gross": (port or {}).get("avg_gross_exposure"),
        "runtime_s": m["runtime_s"], "notes": notes,
    }
    pd.DataFrame([row])[LEDGER_COLS].to_csv(
        LEDGER, mode="a", header=not LEDGER.exists(), index=False)
    return row
