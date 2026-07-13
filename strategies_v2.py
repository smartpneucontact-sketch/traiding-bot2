"""V7 new sleeves (E3+). Residual (idiosyncratic) momentum — Blitz/Huij/
Martens (2011).

Rank stocks by 12-1 momentum of RESIDUAL returns (after removing market /
market+sector beta) instead of raw returns. Literature: similar premium,
much smaller factor crashes because the factor's market-beta tilt (the
crash mechanism) is stripped out.

Conventions mirror strategies.py exactly:
- Sparse decision frame: one row per `rebal_freq` (21) trading days, emitted
  at positions i where ``i % rebal_freq == 0 and i >= lookback_long +
  lookback_skip`` over the price index (same warm-up as xs_momentum: first
  decision at position 273 of the panel).
- Equal-weight 1/n_long, long-only, pre-leverage (row sum = 1).
- All information as of the close of the decision date; execution lag is the
  engine's job.

Spec (binding, E3):
- Window = trailing 252+21 daily returns ending at the decision date d.
- beta_mode="market": per-stock OLS beta vs SPY over the window
  (beta = cov(r_i, r_spy)/var(r_spy)); residual = r_i − beta_i·r_spy
  (alpha stays IN the residual — the idiosyncratic drift is the signal).
- beta_mode="market_sector": two-factor OLS on [SPY, own-sector XL* ETF]
  (demeaned normal equations, 2×2 solve per sector group); residual =
  r_i − b_mkt·r_spy − b_sec·r_sec. Stocks without a mapped sector, sectors
  whose ETF has missing data in the window (e.g. XLC before 2018-06), or
  singular/collinear factor windows fall back to market-only.
- Signal = SUM of residuals over the window EXCLUDING the most recent 21
  days (12-1 convention; 252 formation days).
- scale_mode="sharpe": divide by the std (ddof=0) of the residuals over the
  same 252-day formation window (the paper's iMom variant). "raw": no scale.

Eligibility (deterministic — identical in backtest and live twin):
finite close at the window start and at d, AND ≥ 95% finite daily returns
inside the window. Missing returns are zero-filled for the regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Yahoo-style sector names (data_cache sector_map) → SPDR sector ETF in macro.
SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

MIN_OBS_FRAC = 0.95  # share of finite returns required inside the window


def _residual_momentum_scores(
    px_hist: pd.DataFrame,
    macro_hist: pd.DataFrame,
    sector_map: dict,
    beta_mode: str = "market",
    scale_mode: str = "raw",
    lookback_long: int = 252,
    lookback_skip: int = 21,
) -> pd.Series:
    """Residual-momentum scores as of the LAST row of `px_hist`.

    Shared core for the backtest frame builder and the stateless live twin —
    both call this, so parity is structural. Returns a Series over eligible
    tickers (others absent). Empty Series if not enough history.
    """
    T = lookback_long + lookback_skip  # 273 return days
    if len(px_hist) < T + 1:
        return pd.Series(dtype=float)
    pxw = px_hist.iloc[-(T + 1):]
    rets = pxw.pct_change(fill_method=None).iloc[1:]  # T × N

    if "SPY" not in macro_hist.columns:
        raise RuntimeError("SPY missing from macro panel")
    spy_px = macro_hist["SPY"].reindex(pxw.index)
    spy = spy_px.pct_change(fill_method=None).iloc[1:].to_numpy(dtype=float)
    if not np.isfinite(spy).all():
        raise RuntimeError("SPY return window contains NaN")

    # Eligibility: finite close at window start & end, ≥95% finite returns.
    first_ok = pxw.iloc[0].notna().to_numpy()
    last_ok = pxw.iloc[-1].notna().to_numpy()
    finite_cnt = np.isfinite(rets.to_numpy(dtype=float)).sum(axis=0)
    eligible = first_ok & last_ok & (finite_cnt >= int(np.ceil(MIN_OBS_FRAC * T)))

    R = rets.to_numpy(dtype=float)
    R = np.where(np.isfinite(R), R, 0.0)  # zero-fill missing returns
    cols = np.asarray(rets.columns)

    # Market betas (always needed: market mode + sector fallback).
    xc = spy - spy.mean()
    denom = float(xc @ xc)
    Rc = R - R.mean(axis=0)
    beta_mkt = (xc @ Rc) / denom
    resid = R - np.outer(spy, beta_mkt)  # alpha kept in the residual

    if beta_mode == "market_sector":
        sec_px = {}
        for sec, etf in SECTOR_TO_ETF.items():
            if etf in macro_hist.columns:
                s = macro_hist[etf].reindex(pxw.index)
                sr = s.pct_change(fill_method=None).iloc[1:].to_numpy(dtype=float)
                if np.isfinite(sr).all():
                    sec_px[sec] = sr
        # group columns by sector
        sec_of = np.asarray([sector_map.get(c) for c in cols], dtype=object)
        for sec, sr in sec_px.items():
            g = np.where(sec_of == sec)[0]
            if g.size == 0:
                continue
            X = np.column_stack([spy, sr])               # 273 × 2, raw
            Xc = X - X.mean(axis=0)
            G = Xc.T @ Xc                                # 2 × 2
            if abs(np.linalg.det(G)) < 1e-18:
                continue  # collinear → keep market-only residual
            B = np.linalg.solve(G, Xc.T @ Rc[:, g])      # 2 × n_g
            resid[:, g] = R[:, g] - X @ B
    elif beta_mode != "market":
        raise ValueError(f"unknown beta_mode: {beta_mode}")

    form = resid[:T - lookback_skip]  # first 252 rows = excl. last 21 days
    sig = form.sum(axis=0)
    if scale_mode == "sharpe":
        sd = form.std(axis=0, ddof=0)
        bad = sd < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = sig / sd
        sig[bad] = np.nan
    elif scale_mode != "raw":
        raise ValueError(f"unknown scale_mode: {scale_mode}")

    out = pd.Series(sig, index=cols)
    return out[eligible & np.isfinite(sig)]


def residual_momentum(
    px: pd.DataFrame,
    macro: pd.DataFrame,
    sector_map: dict,
    n_long: int = 30,
    beta_mode: str = "market",
    scale_mode: str = "raw",
    lookback_long: int = 252,
    lookback_skip: int = 21,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Sparse decision frame (same grid convention as strategies.xs_momentum):
    long top-`n_long` by residual 12-1 momentum, equal weight 1/n, monthly."""
    dates = px.index
    w_list = []
    for i in range(len(dates)):
        if i % rebal_freq != 0 or i < lookback_long + lookback_skip:
            continue
        scores = _residual_momentum_scores(
            px.iloc[: i + 1], macro.iloc[: i + 1], sector_map,
            beta_mode=beta_mode, scale_mode=scale_mode,
            lookback_long=lookback_long, lookback_skip=lookback_skip,
        )
        if len(scores) < n_long:
            continue
        top = scores.nlargest(n_long).index
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[top] = 1.0 / n_long
        w_list.append(wrow.rename(dates[i]))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def _residual_momentum_weights_live(
    px_window: pd.DataFrame,
    macro_window: pd.DataFrame,
    sector_map: dict,
    n_long: int = 30,
    beta_mode: str = "market",
    scale_mode: str = "raw",
    lookback_long: int = 252,
    lookback_skip: int = 21,
) -> dict[str, float]:
    """Stateless live twin (pattern: combo_strategy._xs_momentum_weights).

    `px_window` / `macro_window`: daily close history ending at the decision
    date (last row = today's close), ≥ lookback_long+lookback_skip+1 rows.
    Returns {symbol: weight} (equal-weight 1/n_long) or {} if not feasible.
    """
    if px_window.empty or len(px_window) < lookback_long + lookback_skip + 1:
        return {}
    scores = _residual_momentum_scores(
        px_window, macro_window, sector_map,
        beta_mode=beta_mode, scale_mode=scale_mode,
        lookback_long=lookback_long, lookback_skip=lookback_skip,
    )
    if len(scores) < n_long:
        return {}
    top = scores.nlargest(n_long).index
    return {sym: 1.0 / n_long for sym in top}
