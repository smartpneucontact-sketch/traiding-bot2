"""F1 feature set (36 features) — vectorized wide-panel builders -> long frame
of candidate rows only.

Port notes (formulas from train_ml_v5.py / train_ml_v6.py where referenced):
  - drawdown_60/120, recovery_speed_60, consec_up_5, move_vs_atr (ATR14),
    close_loc, gap, dollar_vol_ratio: ported verbatim from
    Trading bot 6/trading_system/scripts/train_ml_v5.py & train_ml_v6.py.
  - obv_trend / ad_trend: v5 uses an inception-anchored cumsum, which has
    UNBOUNDED lookback. To honor MAX_LOOKBACK_BARS=273 the cumulative lines
    here are 252-bar windowed sums (rolling OBV / rolling A-D), then the same
    (x - sma20(x)) / (|sma20(x)| + eps) transform. Total lookback 252+20=272.
  - vol_accel: spec text "(vol_10/vol_60 change)" -> volume-based:
    ratio = v10/v60, vol_accel = ratio - ratio.shift(5).
All lookbacks <= 273 bars — asserted via the DECLARED_LOOKBACKS table.

Cross-sectional ranks (suffix _r) are computed WITHIN the candidate set per
decision date (pct ranks), after the candidate filter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_track.config import MAX_LOOKBACK_BARS

EPS = 1e-8

SECTOR_TO_ETF = {
    "Basic Materials": "XLB", "Communication Services": "XLC",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Energy": "XLE", "Financial Services": "XLF", "Healthcare": "XLV",
    "Industrials": "XLI", "Real Estate": "XLRE", "Technology": "XLK",
    "Utilities": "XLU", "Unknown": "SPY",  # SPY fallback for Unknown
}
SPDRS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
         "XLV", "XLY"]

# Declared max lookback (bars) per wide builder — the MAX_LOOKBACK assert.
DECLARED_LOOKBACKS = {
    "mom_12_1": 253, "mom_6m": 127, "mom_3m": 64, "mom_1m": 22,
    "mom_12_1_volscaled": 253, "info_discrete_252": 253,
    "mom_consistency_12": 253, "mom_accel": 253, "efficiency_20": 21,
    "move_vs_atr": 15, "consec_up_5": 6, "dist_hi_252": 252,
    "channel_252": 252, "vol_60": 61, "vol_ratio_10_60": 61, "ivol_60": 61,
    "beta_spy_120": 121, "skew_60": 61, "max_ret_21": 22,
    "drawdown_60": 60, "drawdown_120": 120, "recovery_speed_60": 65,
    "vol_accel": 65, "obv_trend": 272, "dollar_vol_ratio": 21,
    "ad_trend": 272, "ret_5d": 6, "gap": 2, "close_loc": 1,
    "rel_sector_63": 64, "rel_sector_21": 22, "sector_mom_rank": 64,
    "corr_spy_60": 61, "vix_pct_252": 252, "spy_dist_200sma": 200,
    "hyg_dist_50sma": 50,
}
assert max(DECLARED_LOOKBACKS.values()) <= MAX_LOOKBACK_BARS, \
    "feature lookback exceeds MAX_LOOKBACK_BARS"

# columns cross-sectionally ranked within the candidate set (raw -> _r)
RANK_COLS = {"mom_12_1": "mom_12_1_r", "mom_6m": "mom_6m_r",
             "mom_3m": "mom_3m_r", "mom_1m": "mom_1m_r",
             "vol_60": "vol_60_r", "ivol_60": "ivol_60_r",
             "ret_5d": "ret_5d_r"}

F1_COLS = [
    "mom_12_1_r", "mom_6m_r", "mom_3m_r", "mom_1m_r", "mom_12_1_volscaled",
    "info_discrete_252", "mom_consistency_12", "mom_accel", "efficiency_20",
    "move_vs_atr", "consec_up_5", "dist_hi_252", "channel_252",
    "vol_60_r", "vol_ratio_10_60", "ivol_60_r", "beta_spy_120", "skew_60",
    "max_ret_21", "drawdown_60", "drawdown_120", "recovery_speed_60",
    "vol_accel", "obv_trend", "dollar_vol_ratio", "ad_trend",
    "ret_5d_r", "gap", "close_loc",
    "rel_sector_63", "rel_sector_21", "sector_mom_rank", "corr_spy_60",
    "vix_pct_252", "spy_dist_200sma", "hyg_dist_50sma",
]
assert len(F1_COLS) == 36


def _rolling_cov_var(rets: pd.DataFrame, mkt: pd.Series, win: int):
    """Population rolling cov(stock, mkt) and var(mkt), var(stock)."""
    mxy = rets.mul(mkt, axis=0).rolling(win).mean()
    mx = rets.rolling(win).mean()
    my = mkt.rolling(win).mean()
    cov = mxy - mx.mul(my, axis=0)
    var_m = (mkt ** 2).rolling(win).mean() - my ** 2
    var_s = (rets ** 2).rolling(win).mean() - mx ** 2
    return cov, var_m, var_s


def build_wide_features(panel: dict, macro: pd.DataFrame,
                        sector_map: dict) -> dict[str, pd.DataFrame | pd.Series]:
    """All 36 features as wide frames (dates x symbols) or Series (per-date)."""
    c = panel["close"]
    h = panel["high"]
    lo = panel["low"]
    op = panel["open"]
    v = panel["volume"]
    ret = c.pct_change(fill_method=None)
    out: dict[str, pd.DataFrame | pd.Series] = {}

    # ── momentum family ──
    mom_12_1 = c.shift(21) / c.shift(252) - 1.0
    out["mom_12_1"] = mom_12_1
    out["mom_6m"] = c.pct_change(126, fill_method=None)
    out["mom_3m"] = c.pct_change(63, fill_method=None)
    out["mom_1m"] = c.pct_change(21, fill_method=None)
    vol_252_ann = ret.rolling(252).std() * np.sqrt(252.0)
    out["mom_12_1_volscaled"] = mom_12_1 / (vol_252_ann + EPS)

    up = (ret > 0).astype(float)
    dn = (ret < 0).astype(float)
    pct_up = up.rolling(252).mean()
    pct_dn = dn.rolling(252).mean()
    out["info_discrete_252"] = np.sign(mom_12_1) * (pct_dn - pct_up)

    pos_month = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for k in range(1, 13):
        blk = c.shift(21 * (k - 1)) / c.shift(21 * k) - 1.0
        pos_month += (blk > 0).astype(float)
    out["mom_consistency_12"] = pos_month

    r_2_6 = c.shift(21) / c.shift(126) - 1.0     # months 2-6 (105d)
    r_7_12 = c.shift(126) / c.shift(252) - 1.0   # months 7-12 (126d)
    ann_2_6 = (1.0 + r_2_6) ** (252.0 / 105.0) - 1.0
    ann_7_12 = (1.0 + r_7_12) ** (252.0 / 126.0) - 1.0
    out["mom_accel"] = ann_2_6 - ann_7_12

    out["efficiency_20"] = c.diff(20).abs() / (c.diff().abs().rolling(20).sum() + EPS)

    cp = c.shift(1)
    tr = pd.DataFrame(
        np.fmax((h - lo).to_numpy(),
                np.fmax((h - cp).abs().to_numpy(), (lo - cp).abs().to_numpy())),
        index=c.index, columns=c.columns)
    atr_14 = tr.rolling(14).mean()
    out["move_vs_atr"] = c.diff(1).abs() / (atr_14 + EPS)
    out["consec_up_5"] = up.rolling(5).sum()

    hi_252 = h.rolling(252).max()
    lo_252 = lo.rolling(252).min()
    out["dist_hi_252"] = (c - hi_252) / (c + EPS) * 100.0
    out["channel_252"] = (c - lo_252) / (hi_252 - lo_252 + EPS)

    # ── risk family ──
    vol_60 = ret.rolling(60).std()
    out["vol_60"] = vol_60
    out["vol_ratio_10_60"] = ret.rolling(10).std() / (vol_60 + EPS)

    spy_ret = macro["SPY"].pct_change(fill_method=None).reindex(c.index)
    cov60, varm60, vars60 = _rolling_cov_var(ret, spy_ret, 60)
    ivol_var = (vars60 - cov60.pow(2).div(varm60 + EPS, axis=0)).clip(lower=0.0)
    out["ivol_60"] = np.sqrt(ivol_var)
    cov120, varm120, _ = _rolling_cov_var(ret, spy_ret, 120)
    out["beta_spy_120"] = cov120.div(varm120 + EPS, axis=0)
    out["skew_60"] = ret.rolling(60).skew()
    out["max_ret_21"] = ret.rolling(21).max()
    out["corr_spy_60"] = cov60 / (np.sqrt(vars60.clip(lower=0.0))
                                  .mul(np.sqrt(varm60.clip(lower=0.0)), axis=0) + EPS)

    # ── drawdown family (v6 port) ──
    rmax_60 = c.rolling(60).max()
    rmax_120 = c.rolling(120).max()
    out["drawdown_60"] = (c - rmax_60) / (rmax_60 + EPS) * 100.0
    out["drawdown_120"] = (c - rmax_120) / (rmax_120 + EPS) * 100.0
    dd_60 = (c - rmax_60) / (rmax_60 + EPS)
    out["recovery_speed_60"] = dd_60 - dd_60.shift(5)

    # ── volume family (v5/v6 port; windowed cum-lines, see module docstring) ──
    v10, v60m = v.rolling(10).mean(), v.rolling(60).mean()
    vr = v10 / (v60m + EPS)
    out["vol_accel"] = vr - vr.shift(5)
    obv_inc = v * np.sign(ret.fillna(0.0))
    obv = obv_inc.rolling(252).sum()
    obv_sma = obv.rolling(20).mean()
    out["obv_trend"] = (obv - obv_sma) / (obv_sma.abs() + EPS)
    dv = c * v
    out["dollar_vol_ratio"] = dv / (dv.rolling(20).mean() + EPS)
    clv = ((c - lo) - (h - c)) / (h - lo + EPS)
    ad = (clv * v).rolling(252).sum()
    ad_sma = ad.rolling(20).mean()
    out["ad_trend"] = (ad - ad_sma) / (ad_sma.abs() + EPS)

    # ── short-term price action ──
    out["ret_5d"] = c.pct_change(5, fill_method=None)
    out["gap"] = op / c.shift(1) - 1.0
    out["close_loc"] = (c - lo) / (h - lo + EPS)

    # ── sector relatives ──
    etf_for = {s: SECTOR_TO_ETF.get(sector_map.get(s, "Unknown"), "SPY")
               for s in c.columns}
    etf_cols = [etf_for[s] for s in c.columns]
    m63 = macro.pct_change(63, fill_method=None).reindex(c.index)
    m21 = macro.pct_change(21, fill_method=None).reindex(c.index)
    etf63 = m63[etf_cols].set_axis(c.columns, axis=1)
    etf21 = m21[etf_cols].set_axis(c.columns, axis=1)
    out["rel_sector_63"] = out["mom_3m"] - etf63
    out["rel_sector_21"] = out["mom_1m"] - etf21
    spdr_rank = m63[SPDRS].rank(axis=1, pct=True)
    sect_rank = pd.DataFrame(
        {s: (spdr_rank[etf_for[s]] if etf_for[s] in SPDRS else np.nan)
         for s in c.columns}, index=c.index)
    out["sector_mom_rank"] = sect_rank

    # ── per-date macro features (Series, same value per date) ──
    vix = macro["VIX"].reindex(c.index)
    out["vix_pct_252"] = vix.rolling(252).rank(pct=True)
    spy = macro["SPY"].reindex(c.index)
    out["spy_dist_200sma"] = spy / spy.rolling(200).mean() - 1.0
    hyg = macro["HYG"].reindex(c.index)
    out["hyg_dist_50sma"] = hyg / hyg.rolling(50).mean() - 1.0

    return out


def build_long_features(panel: dict, macro: pd.DataFrame, sector_map: dict,
                        cand_mask: pd.DataFrame) -> pd.DataFrame:
    """Long frame (date, sym, 36 F1 cols, f32) for candidate rows only.
    cs-rank features are pct-ranked WITHIN the candidate set per date."""
    wide = build_wide_features(panel, macro, sector_map)
    dates = cand_mask.index
    mask_long = cand_mask.stack(future_stack=True)
    keep_idx = mask_long[mask_long.fillna(False)].index  # MultiIndex (date, sym)

    cols = {}
    for raw, w in wide.items():
        if isinstance(w, pd.Series):           # per-date macro feature
            continue
        cols[raw] = w.loc[dates].stack(future_stack=True).reindex(keep_idx)
    long = pd.DataFrame(cols)

    # cs-ranks within candidate set
    grp = long.groupby(level=0)
    for raw, rname in RANK_COLS.items():
        long[rname] = grp[raw].rank(pct=True)
    long = long.drop(columns=[r for r in RANK_COLS if RANK_COLS[r] != r])

    # broadcast per-date macro features
    date_lvl = long.index.get_level_values(0)
    for name in ("vix_pct_252", "spy_dist_200sma", "hyg_dist_50sma"):
        long[name] = wide[name].reindex(date_lvl).to_numpy()

    long = long[F1_COLS].astype(np.float32)
    long.index.names = ["date", "sym"]
    return long
