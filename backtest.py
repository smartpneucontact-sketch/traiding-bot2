"""Vectorised daily backtester.

DEPRECATED — DO NOT USE FOR NEW RESEARCH.

Known execution flaw: `weights.shift(1)` in run_backtest shifts the SPARSE weights
frame (rows = rebalance dates only) BEFORE reindexing to the daily calendar,
so each decision row takes effect one full rebalance period late (a month for
monthly strategies), not one trading day as the docstrings below claim. This
module is retained BIT-FROZEN solely so the validation scripts
(validation/reproduce_baseline.py, validation/test_engine_v2.py) can reproduce
the published record in results/results_v6.json to machine precision. All new
research must use engine_v2.run_backtest_v2 (whose default exec_model fixes
the lag; its `legacy_period` mode reproduces this module exactly).

Each strategy returns a `weights` DataFrame: rows = decision dates (signal
formed at close), columns = ticker, values = target weights (post-leverage,
sum ≤ leverage_cap; negatives allowed for shorts).

We then:
  1. Forward-shift weights by 1 trading day to avoid lookahead — the trade
     happens at next day's open, but we approximate with close-to-close.
  2. Multiply by daily returns of the same tickers to get gross P&L per row.
  3. Sum across tickers → portfolio daily return.
  4. Apply a per-trade transaction cost on weight changes.
  5. Compound to build the equity curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BTConfig:
    init_equity: float = 100_000.0
    tc_bps: float = 5.0  # 5bp one-way per turnover dollar (slippage + commission for Alpaca)
    borrow_bps_annual: float = 50.0  # for shorts
    leverage_cap: float = 1.0
    allow_shorts: bool = False


def _align_weights(weights: pd.DataFrame, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align weights & returns on the same trading-day index, forward-fill weights."""
    weights = weights.reindex(returns.index).ffill().fillna(0.0)
    # Drop any tickers in weights that aren't in returns
    common = [c for c in weights.columns if c in returns.columns]
    weights = weights[common]
    returns = returns[common]
    return weights, returns


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BTConfig = BTConfig(),
    name: str = "strategy",
) -> dict:
    """Run a backtest. Returns a dict with equity curve and summary.

    `weights` rows are the *decision date* (signal computed at close of date t).
    They take effect at close of date t+1 (one-day execution lag).
    """
    if weights.empty:
        return {"equity": pd.Series(dtype=float), "name": name}

    rets = prices.pct_change(fill_method=None).fillna(0.0)
    # Shift weights by 1 day: signal formed at t → traded by close of t+1
    target = weights.shift(1).reindex(rets.index).ffill().fillna(0.0)
    target, rets = _align_weights(target, rets)

    if not cfg.allow_shorts:
        target = target.clip(lower=0.0)
    # Enforce per-day leverage cap on the gross
    gross = target.abs().sum(axis=1)
    scale = np.minimum(1.0, cfg.leverage_cap / gross.replace(0.0, np.nan))
    target = target.mul(scale.fillna(1.0), axis=0)

    # Daily portfolio return = sum(weight * stock_return)
    daily_port_ret = (target * rets).sum(axis=1)

    # Turnover cost
    turnover = target.diff().abs().sum(axis=1).fillna(target.iloc[0].abs().sum())
    tc = turnover * (cfg.tc_bps / 10_000.0)
    # Borrow cost on short notional (annual → daily)
    short_notional = target.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * (cfg.borrow_bps_annual / 10_000.0) / 252.0

    daily_net = daily_port_ret - tc - borrow
    equity = (1 + daily_net).cumprod() * cfg.init_equity

    from metrics import summary as compute_summary
    s = compute_summary(equity, name=name)
    s["turnover_annualized"] = float(turnover.mean() * 252.0)
    s["avg_gross_exposure"] = float(target.abs().sum(axis=1).mean())
    s["avg_n_positions"] = float((target.abs() > 1e-6).sum(axis=1).mean())
    return {"equity": equity, "summary": s, "name": name, "weights": target, "returns": daily_net}
