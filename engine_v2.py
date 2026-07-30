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
- ``next_close`` : signal at close t; the book earns the full close t →
                   close t+1 move. Because ``effective = daily.shift(1)``, the
                   fill is AT the decision close itself (the standard
                   vectorized close-to-close convention) — NOT a true 1-day
                   execution lag, and mildly optimistic since a real order off
                   the close print cannot fill at that print. ``next_open`` is
                   the realistic benchmark.
- ``legacy_period``: byte-reproduces backtest.py (shift on the sparse frame;
                   one full period of lag). For regression/comparison only.

Financing
---------
Shorts pay ``borrow_bps_annual`` on short notional. Longs pay
``margin_bps_annual`` on gross long exposure ABOVE 1.0x NAV, charged daily at
rate/252 (ACT/252). WHY: the published record implicitly assumed free
leverage — at ~6%/yr broker margin a 2x long book pays ~0.3–0.6%/mo, material
next to combo_v2_2x's 4.96%/mo headline. Both default to legacy values
(margin 0.0) so every existing result reproduces bit-for-bit; charging margin
is opt-in, mirroring the engine_v2-supersedes-backtest.py pattern.

Delisting (opt-in, Norgate plan Phase N0 / N1 prereg)
------------------------------------------------------
Default behavior when a name stops trading (prices go NaN): pct_change →
NaN → ``fillna(0.0)``, so the ffilled position earns exactly zero forever
while still consuming gross exposure in ``_clip_and_cap`` and in the
margin/borrow notionals — no exit is ever booked. With
``BTConfigV2(delist_exit=True)`` and ``last_quote`` (symbol → last quoted
date), each delisted position instead EXITS AT ITS LAST QUOTED CLOSE
(zipline-norgatedata ``auto_close_date`` convention): the decision-side
weight is forced to 0.0 from the last-quote date onward, so the effective
book (``shift(1)``) drops the name the day AFTER — turnover + tc are
charged on the exit like any sell, and the name leaves the gross/cap math.
Sensitivity arm ``delist_haircut_pct``: bankruptcy-class names (terminal
adjusted close < 20% of the close 252 trading days earlier) get an extra
one-day return of −haircut × exited weight on the exit day. Both default
off → the published record chain stays bit-identical
(validation/test_delisting_hook.py proves all of this).

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
    margin_bps_annual: float = 0.0  # on long gross > 1.0x NAV; 0.0 = published record
    delist_exit: bool = False       # exit at last quoted close (needs last_quote=)
    delist_haircut_pct: float = 0.0  # extra -haircut on bankruptcy-class exit days


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


# ───── delisting hook helpers (opt-in; see module docstring) ──────────────

BANKRUPTCY_TERMINAL_FRAC = 0.20  # terminal close < 20% of the 252d-ago close
BANKRUPTCY_LOOKBACK_TD = 252     # trading days (N1 prereg classification)


def _delist_masks(
    daily: pd.DataFrame, last_quote: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Force delisted names to 0 on the DECISION-side daily frame.

    Zeroing from each symbol's last-quote date ONWARD means the effective
    book (``daily.shift(1)``) still holds the name through its last quoted
    close and is flat from the day after — i.e. an exit fill AT the last
    quoted close. Composes exactly like ``apply_gate`` (pre-shift, so the
    downstream turnover/cost/gross logic prices the exit like any sell), and
    overrides any later decision that re-selects a dead name.

    Returns ``(masked_daily, exit_flag)`` where ``exit_flag`` (decision-side,
    delisted columns only) marks each symbol's first forced date; shifted one
    day it locates the exit day in the effective book. ``None`` if no
    delisted symbol intersects the frame.
    """
    lq = pd.to_datetime(pd.Series(last_quote).dropna())
    syms = [c for c in daily.columns if c in lq.index]
    if not syms:
        return daily, None
    idx = daily.index.values.astype("datetime64[ns]")
    cut = lq[syms].values.astype("datetime64[ns]")
    dead = pd.DataFrame(idx[:, None] >= cut[None, :],
                        index=daily.index, columns=syms)
    daily = daily.mask(dead.reindex(columns=daily.columns, fill_value=False), 0.0)
    exit_flag = dead & ~dead.shift(1, fill_value=False)
    return daily, exit_flag


def _bankruptcy_class(
    prices: pd.DataFrame, last_quote: pd.Series, syms: list[str]
) -> list[str]:
    """Symbols whose terminal adjusted close is < ``BANKRUPTCY_TERMINAL_FRAC``
    of their own close ``BANKRUPTCY_LOOKBACK_TD`` quoted trading days earlier
    (computed on each symbol's OWN quoted series up to its last-quote date).
    Names with insufficient history are not classified (no haircut) — the
    conservative direction for the sensitivity arm."""
    lq = pd.to_datetime(pd.Series(last_quote).dropna())
    out = []
    for s in syms:
        if s not in prices.columns or s not in lq.index:
            continue
        px = prices[s].loc[: lq[s]].dropna()
        if len(px) <= BANKRUPTCY_LOOKBACK_TD:
            continue
        if px.iloc[-1] < BANKRUPTCY_TERMINAL_FRAC * px.iloc[-1 - BANKRUPTCY_LOOKBACK_TD]:
            out.append(s)
    return out


def run_backtest_v2(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BTConfigV2 = BTConfigV2(),
    name: str = "strategy",
    open_prices: pd.DataFrame | None = None,
    weights_are_daily: bool = False,
    last_quote: pd.Series | None = None,
) -> dict:
    """Run a backtest under the configured execution model.

    `weights`: decision-date frame (default) or an already-daily frame
    (``weights_are_daily=True``), post-leverage values.
    `prices`: daily close panel (stocks + any macro tickers traded).
    `open_prices`: daily open panel; tickers absent from it (e.g. macro ETFs,
    close-only) fall back to prior-close execution and are counted in
    ``summary["n_close_fallback_names"]``. Only used by ``next_open``.
    `last_quote`: symbol → last quoted date (unique index; e.g. Norgate
    ``last_quoted_date``). Only read when ``cfg.delist_exit`` — positions in
    those names exit at the last quoted close (see module docstring).
    Ignored entirely when ``cfg.delist_exit=False`` (bit-identical default).
    """
    if weights.empty:
        return {"equity": pd.Series(dtype=float), "name": name}

    if cfg.delist_exit:
        if last_quote is None:
            raise ValueError(
                "delist_exit=True requires last_quote (symbol -> last quoted date)")
        if cfg.exec_model == "legacy_period":
            raise ValueError(
                "delist_exit is not supported under legacy_period "
                "(record-reproduction mode must stay byte-identical)")

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
        exit_flag = None
        if cfg.delist_exit:
            # Decision-side delisting mask BEFORE clip/cap: dead names leave
            # the gross/cap math from the last-quote date on (effective book
            # exits the day after — fill at the last quoted close).
            daily, exit_flag = _delist_masks(daily, last_quote)
        daily = _clip_and_cap(daily, cfg)
        # Effective book during day t: decided at close t-1. next_close earns
        # the full close t-1 → close t move (fill AT the decision close, not
        # a lagged fill); next_open splits it at the open via the delta leg.
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

        if exit_flag is not None:
            # Exit accounting: on each symbol's exit day (effective side =
            # decision-side first-forced date shifted 1) the book change is
            # exactly −(held weight); tc flows through the normal turnover
            # path above. Haircut arm: bankruptcy-class exits get an extra
            # −haircut × exited weight one-day return on that exit day.
            exit_eff = exit_flag.shift(1, fill_value=False)
            exited = (-delta[exit_flag.columns]).where(exit_eff, 0.0)
            n_delist_exits = int((exited.abs() > 1e-12).to_numpy().sum())
            delist_bk: list[str] = []
            if cfg.delist_haircut_pct != 0.0:
                delist_bk = _bankruptcy_class(prices, last_quote,
                                              list(exit_flag.columns))
                if delist_bk:
                    daily_port_ret = daily_port_ret - (
                        cfg.delist_haircut_pct * exited[delist_bk].sum(axis=1))

    tc = turnover * (cfg.tc_bps / 10_000.0)
    short_notional = effective.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * (cfg.borrow_bps_annual / 10_000.0) / 252.0
    # Margin financing on long gross above 1.0x NAV (ACT/252); applies to
    # every exec_model including legacy_period. Default 0.0 → no-op.
    long_notional = effective.clip(lower=0.0).sum(axis=1)
    margin = (long_notional - 1.0).clip(lower=0.0) * (cfg.margin_bps_annual / 10_000.0) / 252.0

    daily_net = daily_port_ret - tc - borrow - margin
    equity = (1 + daily_net).cumprod() * cfg.init_equity

    s = compute_summary(equity, name=name)
    s["turnover_annualized"] = float(turnover.mean() * 252.0)
    s["avg_gross_exposure"] = float(effective.abs().sum(axis=1).mean())
    s["avg_n_positions"] = float((effective.abs() > 1e-6).sum(axis=1).mean())
    s["exec_model"] = cfg.exec_model
    s["margin_bps_annual"] = cfg.margin_bps_annual
    if cfg.exec_model == "next_open":
        s["n_close_fallback_names"] = n_fallback
    if cfg.delist_exit:  # provenance keys only when the hook is active
        s["delist_exit"] = True
        s["delist_haircut_pct"] = cfg.delist_haircut_pct
        s["n_delist_exits"] = n_delist_exits if exit_flag is not None else 0
        s["n_delist_bankruptcy"] = len(delist_bk) if exit_flag is not None else 0
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
