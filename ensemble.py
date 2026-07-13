"""Risk-managed ensemble strategies.

These combine the base strategies in `strategies.py` with:
  - Market regime gating (cut exposure in bear markets)
  - Vol targeting (cap leverage so realized vol ≤ target)
  - Multi-strategy blending (uncorrelated alpha sources)

Goal: hit ≥3%/month with max-drawdown ≤ 25%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# AUDIT NOTE: align_to_stock_cols, buy_hold_spy, mean_reversion and
# trend_following are imported but never used in this module. Kept as-is —
# this file is part of the published-record rebuild chain; do not delete.
from strategies import (
    align_to_stock_cols,      # AUDIT: unused (referenced only in a comment)
    buy_hold_spy,             # AUDIT: unused
    dual_momentum_voltarget,
    market_regime,
    mean_reversion,           # AUDIT: unused
    sector_rotation,
    trend_following,          # AUDIT: unused
    xs_momentum,
)


def regime_gate(weights: pd.DataFrame, macro: pd.DataFrame,
                neutral_exposure: float = 0.7,
                min_exposure: float = 0.3) -> pd.DataFrame:
    """Scale weights row-by-row based on market regime.

    regime >  0.3  → 100% exposure
    regime in  0..0.3 → linear ramp from neutral_exposure → 100%
    regime in -0.3..0 → linear ramp from min_exposure → neutral_exposure
    regime < -0.3  → min_exposure
    """
    regime = market_regime(macro)
    # Reindex on weights' index
    r = regime.reindex(weights.index).ffill().fillna(0.0)
    # AUDIT: dead assignment — unconditionally overwritten by the np.where
    # below. Kept as-is (published-record rebuild chain).
    exp = pd.Series(min_exposure, index=r.index, dtype=float)
    exp = np.where(r > 0.3, 1.0,
          np.where(r > 0.0, neutral_exposure + (1.0 - neutral_exposure) * r / 0.3,
          np.where(r > -0.3, min_exposure + (neutral_exposure - min_exposure) * (r + 0.3) / 0.3,
          min_exposure)))
    exp = pd.Series(exp, index=r.index)
    return weights.mul(exp, axis=0)


def vol_target_overlay(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol: float = 0.20,
    lookback: int = 60,
    max_leverage: float = 1.0,
    min_leverage: float = 0.3,
) -> pd.DataFrame:
    """Scale weights so realized portfolio vol ≈ target_vol.

    Uses a rolling estimate of realized portfolio vol assuming the current
    weights had been held over the lookback. Updated only on rebalance days
    (rows that already exist in `weights`).
    """
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    w_full = weights.reindex(rets.index).ffill().fillna(0.0)
    # Daily portfolio return at current weights
    port_ret = (w_full * rets).sum(axis=1)
    realised_vol = port_ret.rolling(lookback).std() * np.sqrt(252)
    scale = (target_vol / realised_vol.replace(0.0, np.nan)).clip(min_leverage, max_leverage)
    scale = scale.ffill().fillna(1.0)
    # Apply only on the weight-update rows
    scale_on_weights = scale.reindex(weights.index).ffill().fillna(1.0)
    return weights.mul(scale_on_weights, axis=0)


def fast_drawdown_gate(weights: pd.DataFrame, macro: pd.DataFrame,
                       lookback: int = 60,
                       full_dd: float = 0.05,
                       cash_dd: float = 0.15) -> pd.DataFrame:
    """Fast risk-off gate based on SPY drawdown from `lookback`-day high.

    SPY DD <= full_dd → 100% exposure
    SPY DD ∈ (full_dd, cash_dd) → linear ramp to 0% exposure
    SPY DD >= cash_dd → 0% exposure (full cash)
    """
    if "SPY" not in macro.columns:
        return weights
    spy = macro["SPY"]
    rolling_high = spy.rolling(lookback, min_periods=10).max()
    dd = (spy / rolling_high - 1.0)
    # Map DD (negative number) → exposure in [0, 1]
    exp = ((cash_dd + dd) / (cash_dd - full_dd)).clip(0.0, 1.0)
    # AUDIT NOTE: the daily exposure series is sampled ONLY on `weights` rows
    # (the strategy's rebalance dates); the backtester then ffills the gated
    # weights between rebalances. The gate therefore reacts at the caller's
    # rebalance frequency, not daily — "fast" is misleading for the monthly
    # momentum variants below (same sampling applies to vol_target_overlay).
    exp = exp.reindex(weights.index).ffill().fillna(1.0)
    return weights.mul(exp, axis=0)


def regime_gated_momentum(prices: pd.DataFrame, macro: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """xs_momentum_top30 + regime gate."""
    w = xs_momentum(prices, macro, n_long=30, **kwargs)
    return regime_gate(w, macro)


def fast_dd_momentum(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """xs_momentum_top30 + fast SPY-drawdown-based gate.

    AUDIT NOTE: xs_momentum rebalances monthly, and fast_drawdown_gate samples
    its exposure only on those rows — between rebalances the gate does not
    update, so the "fast" 60-day DD gate effectively updates monthly here."""
    w = xs_momentum(prices, macro, n_long=30)
    return fast_drawdown_gate(w, macro, lookback=60, full_dd=0.04, cash_dd=0.12)


def fast_dd_voltarget_momentum(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Best-of-three: momentum + fast DD gate + vol target.

    AUDIT NOTE: both the DD gate and the vol-target overlay are sampled only
    on xs_momentum's monthly rebalance rows (see fast_drawdown_gate /
    vol_target_overlay) — neither risk control updates between rebalances."""
    w = xs_momentum(prices, macro, n_long=30)
    w = fast_drawdown_gate(w, macro, lookback=60, full_dd=0.04, cash_dd=0.12)
    return vol_target_overlay(w, prices, target_vol=0.22, max_leverage=1.0, min_leverage=0.2)


def final_3pct_target(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    pricing_universe: pd.DataFrame,
) -> pd.DataFrame:
    """The final candidate aimed at 3%/month with managed drawdown.

    Architecture:
      - 60% allocation: xs_momentum_top30 (highest raw alpha, ~2.9%/mo standalone)
      - 30% allocation: dual_momentum_vol (vol-targeted absolute-momentum, defends 2018/2022)
      - 10% allocation: SHY (short-term Treasury — pure cash ballast for COVID-style shocks)
      - Overlay 1: SPY drawdown circuit breaker — when SPY is > 8% below its
        60-day high, scale stock weights by 0.5; > 15% below, scale by 0.0
        (full cash).
      - Overlay 2: portfolio vol target ≈ 20% annual.

    Trades weekly to keep turnover manageable. Long-only.

    AUDIT NOTE — the description of Overlay 1 above contradicts the code:
      * Actual gate params are full_dd=0.08, cash_dd=0.18: a LINEAR ramp from
        100% exposure at 8% SPY DD down to 0% at 18% DD. There is no 0.5 step
        at 8% and full cash arrives at 18%, not 15%.
      * The gate multiplies EVERY column, including the 10% SHY sleeve — the
        "pure cash ballast for COVID-style shocks" is scaled toward zero by
        the DD gate exactly during crashes, so it provides no ballast when it
        is supposed to. (Vol-target overlay rescales it too.)
    Behavior kept as-is: published-record rebuild chain.
    """
    cols = pricing_universe.columns
    union_idx = pricing_universe.index

    w_mom = xs_momentum(prices, macro, n_long=30).reindex(union_idx).ffill().fillna(0.0)
    w_dm = dual_momentum_voltarget(prices, macro, n_long=30).reindex(union_idx).ffill().fillna(0.0)

    full = pd.DataFrame(0.0, index=union_idx, columns=cols)
    common_mom = [c for c in w_mom.columns if c in cols]
    full[common_mom] = full[common_mom].values + 0.60 * w_mom[common_mom].values
    common_dm = [c for c in w_dm.columns if c in cols]
    full[common_dm] = full[common_dm].values + 0.30 * w_dm[common_dm].values

    # 10% SHY for the cash buffer (if present in macro/columns)
    # AUDIT: this "buffer" is NOT exempt from the DD gate / vol target below —
    # it is scaled away together with the stock sleeves during crashes.
    if "SHY" in cols:
        full["SHY"] = full["SHY"].values + 0.10
    elif "TLT" in cols:
        full["TLT"] = full["TLT"].values + 0.10
    else:
        full["SPY"] = full["SPY"].values + 0.10  # last resort

    # Fast DD gate — linear ramp 8%→18% DD (docstring's 0.5-step/15% is wrong)
    full = fast_drawdown_gate(full, macro, lookback=60, full_dd=0.08, cash_dd=0.18)

    # Vol-target the whole thing (cap leverage at 1.0)
    full = vol_target_overlay(full, pricing_universe, target_vol=0.20,
                              max_leverage=1.0, min_leverage=0.4, lookback=60)

    # Weekly rebal
    return full.iloc[::5]


def final_3pct_leveraged(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    pricing_universe: pd.DataFrame,
    leverage: float = 1.3,
) -> pd.DataFrame:
    """Same as `final_3pct_target` but with a modest constant leverage.
    For brokers like Alpaca that offer 2x intraday / 4x daytrade — this is the
    practical maximum you'd take with a Sharpe ~1 system. Backtest tells you
    where the DD goes when you scale up."""
    w = final_3pct_target(prices, macro, pricing_universe)
    return w * leverage


def kelly_voltarget_overlay(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int = 60,
    target_vol: float = 0.30,    # high target since we'll Kelly-clip
    kelly_fraction: float = 0.4,  # 40 % of full Kelly (typical practitioner choice)
    max_leverage: float = 3.0,
    min_leverage: float = 0.2,
) -> pd.DataFrame:
    """Vol-target overlay that uses a Kelly-fraction style scaling. The
    realized Sharpe of the underlying weights is estimated rolling, and the
    leverage is set to `kelly_fraction × Sharpe / vol`. Clipped between
    [min_leverage, max_leverage].

    Theoretical Kelly leverage for a Sharpe-1 strategy at 20 % vol is 1/0.20 ≈
    5x. Practitioners typically run at 0.25–0.5 Kelly, so 1.25–2.5x.
    """
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    w_full = weights.reindex(rets.index).ffill().fillna(0.0)
    port_ret = (w_full * rets).sum(axis=1)
    realised_mu = port_ret.rolling(lookback).mean() * 252.0
    realised_sd = port_ret.rolling(lookback).std() * np.sqrt(252.0)
    kelly_lev = (kelly_fraction * realised_mu / (realised_sd ** 2).replace(0.0, np.nan)).clip(min_leverage, max_leverage)
    # Also clip via vol target
    vol_lev = (target_vol / realised_sd.replace(0.0, np.nan)).clip(min_leverage, max_leverage)
    lev = pd.concat([kelly_lev, vol_lev], axis=1).min(axis=1).ffill().fillna(min_leverage)
    lev = lev.reindex(weights.index).ffill().fillna(min_leverage)
    return weights.mul(lev, axis=0)


def combo_4way_kelly(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    pricing_universe: pd.DataFrame,
    kelly_fraction: float = 0.4,
    max_leverage: float = 2.5,
) -> pd.DataFrame:
    """Diversified 4-way combo with Kelly-fraction leverage instead of fixed."""
    from strategies import dual_momentum_voltarget, ts_momentum_multiasset, xs_momentum, xs_momentum_fast
    cols = list(pricing_universe.columns)
    idx = pricing_universe.index

    def project_stock(w):
        out = pd.DataFrame(0.0, index=idx, columns=cols)
        w2 = w.reindex(idx).ffill().fillna(0.0)
        common = [c for c in w2.columns if c in cols]
        out[common] = w2[common].values
        return out

    def project_macro(w):
        out = pd.DataFrame(0.0, index=idx, columns=cols)
        w2 = w.reindex(idx).ffill().fillna(0.0)
        for c in w2.columns:
            if c in cols:
                out[c] = w2[c].values
        return out

    w_mom = xs_momentum(prices, macro, n_long=30)
    w_fast = xs_momentum_fast(prices, macro, n_long=20)
    w_dm = dual_momentum_voltarget(prices, macro, n_long=30)
    w_ts = ts_momentum_multiasset(macro)

    full = (
        0.30 * project_stock(w_mom)
        + 0.25 * project_stock(w_fast)
        + 0.20 * project_stock(w_dm)
        + 0.25 * project_macro(w_ts)
    )
    full = fast_drawdown_gate(full, macro, lookback=60, full_dd=0.08, cash_dd=0.18)
    full = kelly_voltarget_overlay(full, pricing_universe,
                                   kelly_fraction=kelly_fraction,
                                   max_leverage=max_leverage,
                                   target_vol=0.30,
                                   lookback=60)
    return full.iloc[::5]


def vol_target_momentum(prices: pd.DataFrame, macro: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """xs_momentum_top30 + vol target."""
    w = xs_momentum(prices, macro, n_long=30, **kwargs)
    return vol_target_overlay(w, prices, target_vol=0.20)


def regime_voltarget_momentum(prices: pd.DataFrame, macro: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """xs_momentum_top30 + regime gate + vol target. Belt and braces."""
    w = xs_momentum(prices, macro, n_long=30, **kwargs)
    w = regime_gate(w, macro, neutral_exposure=0.7, min_exposure=0.3)
    return vol_target_overlay(w, prices, target_vol=0.18, max_leverage=1.0)


def ensemble_top3(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    pricing_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Equal-weighted ensemble of three uncorrelated alpha sources:
      - 40% xs_momentum_top30 (return-seeking)
      - 30% dual_momentum_vol (vol-targeted, lower DD in fast moves)
      - 30% sector_rotation (cleaner regime exposure, slower turnover)
    All gated by the market regime score.
    """
    cols = pricing_universe.columns
    union_idx = pricing_universe.index

    w_mom = xs_momentum(prices, macro, n_long=30).reindex(union_idx).ffill().fillna(0.0)
    w_dm = dual_momentum_voltarget(prices, macro, n_long=30).reindex(union_idx).ffill().fillna(0.0)
    w_sec = sector_rotation(prices, macro).reindex(union_idx).ffill().fillna(0.0)

    # Re-project sector weights onto the union (already done by align_to_stock_cols outside)
    w_sec_full = pd.DataFrame(0.0, index=union_idx, columns=cols)
    for c in w_sec.columns:
        if c in cols:
            w_sec_full[c] = w_sec[c].values

    w_mom_full = pd.DataFrame(0.0, index=union_idx, columns=cols)
    common_mom = [c for c in w_mom.columns if c in cols]
    w_mom_full[common_mom] = w_mom[common_mom].values

    w_dm_full = pd.DataFrame(0.0, index=union_idx, columns=cols)
    common_dm = [c for c in w_dm.columns if c in cols]
    w_dm_full[common_dm] = w_dm[common_dm].values

    blended = 0.40 * w_mom_full + 0.30 * w_dm_full + 0.30 * w_sec_full
    blended = regime_gate(blended, macro, neutral_exposure=0.65, min_exposure=0.25)
    blended = vol_target_overlay(blended, pricing_universe, target_vol=0.16, max_leverage=1.0)

    # Re-sample to a weekly grid so we don't trade every day
    if not blended.empty:
        weekly = blended.iloc[::5]
        return weekly
    return blended
