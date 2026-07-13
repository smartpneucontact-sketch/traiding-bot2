"""Model training (LightGBM) + OOF signal metrics for the ML re-ranker.

Architectures:
  A_rank — LGBMRanker (lambdarank), group = decision date, label = quintile
           bucket 0-4 of the chosen pct-rank label (L1 or L2).
  A_reg  — LGBMRegressor on the pct-rank label (L1 or L2).
  B_cls  — LGBMClassifier on the binary label (L3 for "L1", L3v for "L2")
           + isotonic calibration fitted on pooled OOF probabilities
           (monotone -> rank metrics unchanged; calibrated scores stored).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from ml_track.config import TOP_N
from ml_track.features import F1_COLS

# map HP dict (lgb native names) -> sklearn API names
_HP_MAP = {"feature_fraction": "colsample_bytree",
           "bagging_fraction": "subsample",
           "bagging_freq": "subsample_freq"}


def _sk_params(hp: dict) -> dict:
    return {_HP_MAP.get(k, k): v for k, v in hp.items()}


def _group_sizes(dates: pd.Series) -> list[int]:
    return dates.groupby(dates.values).size().reindex(
        pd.unique(dates.values)).tolist()


def _quintile(label: pd.Series) -> np.ndarray:
    return np.minimum((label.to_numpy() * 5).astype(int), 4)


def fit_fold(arch: str, hp: dict, X_tr, y_tr, d_tr, X_es, y_es, d_es,
             monotone: bool = False, seed: int = 7):
    params = _sk_params(hp)
    params.update(random_state=seed, n_jobs=-1, verbose=-1)
    if monotone:
        mono = [1 if c == "mom_12_1_r" else 0 for c in F1_COLS]
        params["monotone_constraints"] = mono
    cb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]

    if arch == "A_rank":
        mdl = lgb.LGBMRanker(objective="lambdarank", **params)
        mdl.fit(X_tr, _quintile(y_tr), group=_group_sizes(d_tr),
                eval_set=[(X_es, _quintile(y_es))],
                eval_group=[_group_sizes(d_es)],
                eval_at=[TOP_N], callbacks=cb)
    elif arch == "A_reg":
        mdl = lgb.LGBMRegressor(objective="regression", **params)
        mdl.fit(X_tr, y_tr, eval_set=[(X_es, y_es)],
                eval_metric="l2", callbacks=cb)
    elif arch == "B_cls":
        mdl = lgb.LGBMClassifier(objective="binary", **params)
        mdl.fit(X_tr, y_tr.astype(int), eval_set=[(X_es, y_es.astype(int))],
                eval_metric="binary_logloss", callbacks=cb)
    else:
        raise ValueError(arch)
    return mdl


def predict(arch: str, mdl, X) -> np.ndarray:
    if arch == "B_cls":
        return mdl.predict_proba(X)[:, 1]
    return mdl.predict(X)


def calibrate_isotonic(oof_scores: np.ndarray, oof_binary: np.ndarray):
    """Isotonic calibration on pooled OOF (B_cls only)."""
    m = np.isfinite(oof_scores) & np.isfinite(oof_binary)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_scores[m], oof_binary[m])
    return iso


def signal_metrics(oof: pd.DataFrame) -> dict:
    """OOF frame: MultiIndex (date,sym) with cols score, r_fwd21, mom_12_1_r,
    fold_year. Returns rank-IC stats, the momentum-prior IC on the SAME rows,
    %positive fold-years, top-30 hit-rate vs candidate median."""
    df = oof.dropna(subset=["score", "r_fwd21"])

    def _date_ic(g, col):
        if len(g) < 10 or g[col].nunique() < 3:
            return np.nan
        return spearmanr(g[col], g["r_fwd21"]).statistic

    by_date = df.groupby(level=0)
    ic = by_date.apply(lambda g: _date_ic(g, "score"))
    ic_prior = by_date.apply(lambda g: _date_ic(g, "mom_12_1_r"))

    yr = pd.Series(ic.index.year, index=ic.index)
    ic_by_year = ic.groupby(yr).mean()
    pos_years = float((ic_by_year > 0).mean())

    def _hit30(g):
        top = g.nlargest(TOP_N, "score")
        med = g["r_fwd21"].median()
        return float((top["r_fwd21"] > med).mean())
    hit30 = float(by_date.apply(_hit30).mean())

    return {
        "rank_ic": float(ic.mean()),
        "rank_ic_t": float(ic.mean() / (ic.std() / np.sqrt(ic.notna().sum()))),
        "prior_ic": float(ic_prior.mean()),
        "ic_minus_prior": float(ic.mean() - ic_prior.mean()),
        "pct_pos_years": pos_years,
        "hit30": hit30,
        "n_dates": int(ic.notna().sum()),
        "ic_by_year": {int(k): round(float(v), 4) for k, v in ic_by_year.items()},
    }
