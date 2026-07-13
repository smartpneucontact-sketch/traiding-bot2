"""Corrected vectorised daily backtester (V7 research program).

Why this exists
---------------
`backtest.py` applies ``weights.shift(1)`` to the SPARSE decision-date frame
(one row per ~21-trading-day rebalance), so every decision executes one full
rebalance PERIOD late, not one day. All pre-V7 published numbers (including
combo_v2_2x's 4.96%/mo) carry that 21-day-stale execution. This engine fixes
the lag and additionally models the live bot's actual execution (market
orders at the next OPEN, 9:35 ET, off prior-close signals).

`backtest.py` is intentionally left untouched: validation scripts assert
machine-precision reproduction of the published record through it. This
engine's ``legacy_period`` mode reproduces it bit-for-bit — that equivalence
is the regression test proving v2 is a superset.

Execution models
----------------
- ``next_open``  : signal at close t → trade at open of t+1. Day-t+1 return =
                   w_old·(open/close_prev − 1) borne by the OLD book overnight
                   plus w_new·(close/open − 1) borne by the NEW book intraday.
                   Costs on |Δw| at the open. NaN/absent opens fall back to the
                   prior close (the switch then happens at prior close — same
                   convention as validation/stop_grid.py).
- ``next_close`` : signal at close t → trade at close of t+1 (true 1-day lag
                   on the DAILY frame). Close-to-close returns.
- ``legacy_period``: byte-reproduces backtest.py (shift on the sparse frame;
                   one full period of lag). For regression/comparison only.

Conventions match backtest.py: weights rows are decision dates with
POST-leverage values (e.g. blend × 2.0); shorts clipped unless allowed;
per-day gross capped at ``leverage_cap``; 5bp one-way costs on turnover;
linear (non-compounded) daily cross-asset aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metrics import summary as compute_summary


@dataclass
class BTConfigV2:
    init_equity: float = 100_000.0
    tc_bps: float = 5.0
    borrow_bps_annual: float = 50.0
    leverage_cap: float = 2.0
    allow_shorts: bool = False
    exec_model: str = "next_open"  # "next_open" | "next_close" | "legacy_period"


def expand_daily(weights_sparse: pd.DataFrame, daily_index: pd.Index) -> pd.DataFrame:
    """Sparse decision-date frame → daily frame (ffill, NO shift here)."""
    return weights_sparse.reindex(daily_index).ffill().fillna(0.0)


def _clip_and_cap(target: pd.DataFrame, cfg: BTConfigV2) -> pd.DataFrame:
    """Short clip + per-day gross leverage cap (identical to backtest.py)."""
    if not cfg.allow_shorts:
        target = target.clip(lower=0.0)
    gross = target.abs().sum(axis=1)
    scale = np.minimum(1.0, cfg.leverage_cap / gross.replace(0.0, np.nan))
    return target.mul(scale.fillna(1.0), axis=0)


def run_backtest_v2(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BTConfigV2 = BTConfigV2(),
    name: str = "strategy",
    open_prices: pd.DataFrame | None = None,
    weights_are_daily: bool = False,
) -> dict:
    """Run a backtest under the configured execution model.

    `weights`: decision-date frame (default) or an already-daily frame
    (``weights_are_daily=True``), post-leverage values.
    `prices`: daily close panel (stocks + any macro tickers traded).
    `open_prices`: daily open panel; tickers absent from it (e.g. macro ETFs,
    close-only) fall back to prior-close execution and are counted in
    ``summary["n_close_fallback_names"]``. Only used by ``next_open``.
    """
    if weights.empty:
        return {"equity": pd.Series(dtype=float), "name": name}

    rets = prices.pct_change(fill_method=None).fillna(0.0)

    if cfg.exec_model == "legacy_period":
        # Bit-identical to backtest.py: shift on the sparse frame FIRST.
        target = weights.shift(1).reindex(rets.index).ffill().fillna(0.0)
        common = [c for c in target.columns if c in rets.columns]
        target, r = target[common], rets[common]
        target = _clip_and_cap(target, cfg)
        daily_port_ret = (target * r).sum(axis=1)
        turnover = target.diff().abs().sum(axis=1).fillna(target.iloc[0].abs().sum())
        effective = target
    else:
        daily = weights if weights_are_daily else expand_daily(weights, rets.index)
        daily = daily.reindex(rets.index).ffill().fillna(0.0)
        common = [c for c in daily.columns if c in rets.columns]
        daily, r = daily[common], rets[common]
        daily = _clip_and_cap(daily, cfg)
        # Effective book during day t: decided at close t-1, in force from
        # day t's execution point (open or close per model).
        effective = daily.shift(1).fillna(0.0)
        delta = effective.diff().fillna(effective.iloc[0])
        turnover = delta.abs().sum(axis=1)

        if cfg.exec_model == "next_close":
            daily_port_ret = (effective * r).sum(axis=1)
        elif cfg.exec_model == "next_open":
            close = prices[common]
            if open_prices is not None:
                op = open_prices.reindex(index=rets.index)
                op = op.reindex(columns=common)
            else:
                op = pd.DataFrame(np.nan, index=rets.index, columns=common)
            n_fallback = int(op.isna().all(axis=0).sum())
            # NaN open → prior close (switch effectively at prior close).
            op = op.where(op.notna(), close.shift(1))
            # Intraday return close_t / open_t − 1 (0 where unavailable).
            r_intra = (close / op - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Base: previous book carries the full close-to-close move;
            # on switch days the CHANGE in book carries the intraday leg
            # instead. r = w_prev·r_cc + Δw·r_intra  (exact for held names,
            # overnight gap borne by the old book, intraday by the new).
            prev_book = effective.shift(1).fillna(0.0)
            daily_port_ret = (prev_book * r).sum(axis=1) + (delta * r_intra).sum(axis=1)
        else:
            raise ValueError(f"unknown exec_model: {cfg.exec_model}")

    tc = turnover * (cfg.tc_bps / 10_000.0)
    short_notional = effective.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * (cfg.borrow_bps_annual / 10_000.0) / 252.0

    daily_net = daily_port_ret - tc - borrow
    equity = (1 + daily_net).cumprod() * cfg.init_equity

    s = compute_summary(equity, name=name)
    s["turnover_annualized"] = float(turnover.mean() * 252.0)
    s["avg_gross_exposure"] = float(effective.abs().sum(axis=1).mean())
    s["avg_n_positions"] = float((effective.abs() > 1e-6).sum(axis=1).mean())
    s["exec_model"] = cfg.exec_model
    if cfg.exec_model == "next_open":
        s["n_close_fallback_names"] = n_fallback
    return {
        "equity": equity, "summary": s, "name": name,
        "weights": effective, "returns": daily_net,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Overlay utilities (daily-frame, vectorized)
# ═══════════════════════════════════════════════════════════════════════════

def apply_gate(weights_daily: pd.DataFrame, gate: pd.Series) -> pd.DataFrame:
    """Multiply a daily weights frame by a daily exposure multiplier in [0,1].
    The gate inherits the engine's execution lag automatically (it scales the
    decision-side frame, which the engine then shifts)."""
    g = gate.reindex(weights_daily.index).ffill().fillna(1.0).clip(0.0, 1.0)
    return weights_daily.mul(g, axis=0)


def vol_managed_multiplier(
    daily_net: pd.Series,
    target_vol: float,
    lookback: int = 126,
    cap: float = 1.0,
    floor: float | None = 0.2,
) -> pd.Series:
    """Barroso/Santa-Clara style scaling: m_t = clip(target_vol / realized_vol),
    where realized_vol is the annualized std of the UNSCALED strategy's daily
    net returns over the trailing `lookback` days, lagged one day (no
    same-day information). Returns the daily multiplier series."""
    rv = daily_net.rolling(lookback).std() * np.sqrt(252.0)
    m = (target_vol / rv).shift(1)
    lo = floor if floor is not None else 0.0
    return m.clip(lower=lo, upper=cap).fillna(1.0)


def book_drawdown_gate(
    weights_daily: pd.DataFrame,
    close: pd.DataFrame,
    full_dd: float = 0.12,
    cash_dd: float = 0.30,
    lookback: int = 60,
) -> pd.Series:
    """Daily exposure multiplier from the weighted drawdown of the CURRENT
    book's names vs their trailing `lookback`-day close highs (the validated
    12→30 gate from validation/BOOK_GATE_REPORT.md, generalized).
    Lagged one day by the engine's execution shift when applied pre-engine."""
    w = weights_daily.abs()
    rolling_hi = close.rolling(lookback, min_periods=2).max()
    dd = (close / rolling_hi - 1.0).clip(upper=0.0)
    dd = dd.reindex(index=w.index, columns=w.columns)
    num = (w * dd).sum(axis=1)
    den = w.sum(axis=1).replace(0.0, np.nan)
    book_dd = (num / den).fillna(0.0)
    mag = -book_dd
    gate = (cash_dd - mag) / (cash_dd - full_dd)
    return gate.clip(0.0, 1.0).where(mag > full_dd, 1.0)


def ex_ante_book_vol(
    weights_row: pd.Series,
    rets_window: pd.DataFrame,
) -> float:
    """Annualized ex-ante portfolio vol sqrt(w'Σw·252) with the ACTUAL sample
    covariance — the fix for the correlation-free formula in strategies.py:306
    / combo_strategy.py:103 that never binds."""
    syms = [s for s in weights_row.index if abs(weights_row[s]) > 1e-9
            and s in rets_window.columns]
    if not syms:
        return 0.0
    w = weights_row[syms].to_numpy(dtype=float)
    cov = rets_window[syms].cov().to_numpy(dtype=float)
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0) * 252.0))
