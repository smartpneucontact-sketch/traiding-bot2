"""Performance metrics computed from a daily equity curve.

Conventions
-----------
- `equity` is a pd.Series indexed by trading-date, equity[t] is the close-of-day
  portfolio value.
- All metrics are net of whatever cost is baked into the equity curve.
- Annualization assumes 252 trading days/year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def monthly_returns(equity: pd.Series) -> pd.Series:
    return equity.resample("ME").last().pct_change().dropna()


def mean_monthly_return(equity: pd.Series) -> float:
    m = monthly_returns(equity)
    return float(m.mean()) if len(m) else float("nan")


def geo_monthly_return(equity: pd.Series) -> float:
    """Geometric monthly return: (1+CAGR)**(1/12) - 1.

    This is the correct metric for "X%/month" compounding targets. The
    arithmetic `mean_monthly` overstates the compounded rate by roughly
    vol^2/2 per period (~1.2pp/mo at 59% annualized vol); use this instead
    when quoting a monthly rate. NaN-safe: returns nan when cagr is nan.
    """
    c = cagr(equity)
    if np.isnan(c):
        return float("nan")
    return float((1.0 + c) ** (1.0 / 12.0) - 1.0)


def median_monthly_return(equity: pd.Series) -> float:
    m = monthly_returns(equity)
    return float(m.median()) if len(m) else float("nan")


def annualized_vol(equity: pd.Series) -> float:
    r = daily_returns(equity)
    return float(r.std() * np.sqrt(ANN)) if len(r) else float("nan")


def _excess_daily(r: pd.Series, rf: float | pd.Series) -> pd.Series:
    """Daily excess returns for a scalar OR series risk-free rate.

    `rf` convention (both forms): ANNUALIZED decimal rate (e.g. 0.052 for
    5.2%); the daily deduction is rf/252. A Series rf (e.g. ^IRX/100) is
    aligned to the return index by reindex + ffill (T-bill quotes gap on
    market holidays), with any leading unquoted days treated as rf=0 — the
    conservative choice for pre-history, and irrelevant for ^IRX (quoted
    since 1960). The scalar path is byte-identical to the pre-2026-07-18
    code: the published-record fidelity chain depends on the rf=0.0 default
    producing unchanged numbers.
    """
    if isinstance(rf, pd.Series):
        rf_ann = rf.reindex(r.index).ffill().fillna(0.0)
        return r - rf_ann / ANN
    return r - rf / ANN


def sharpe(equity: pd.Series, rf: float | pd.Series = 0.0) -> float:
    r = daily_returns(equity)
    if len(r) == 0:
        return float("nan")
    excess = _excess_daily(r, rf)
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(ANN))


def sortino(equity: pd.Series, rf: float | pd.Series = 0.0) -> float:
    """Sortino ratio using the standard downside deviation (LPM2).

    downside_dev = sqrt(mean(min(excess, 0)**2)) over ALL observations —
    not the sample std of negative returns around their own mean, which the
    pre-2026-07-12 version used. Values are NOT comparable to ledger rows
    recorded before 2026-07-12.
    """
    r = daily_returns(equity)
    if len(r) == 0:
        return float("nan")
    excess = _excess_daily(r, rf)
    downside_dev = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    if downside_dev == 0:
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(ANN))


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min()) if len(dd) else float("nan")


def calmar(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return cagr(equity) / mdd


def hit_rate(equity: pd.Series) -> float:
    """Fraction of months with positive return."""
    m = monthly_returns(equity)
    return float((m > 0).mean()) if len(m) else float("nan")


def worst_month(equity: pd.Series) -> float:
    m = monthly_returns(equity)
    return float(m.min()) if len(m) else float("nan")


def best_month(equity: pd.Series) -> float:
    m = monthly_returns(equity)
    return float(m.max()) if len(m) else float("nan")


def yearly_returns(equity: pd.Series) -> pd.Series:
    return equity.resample("YE").last().pct_change().dropna()


def summary(equity: pd.Series, name: str = "strategy",
            rf: float | pd.Series = 0.0) -> dict:
    return {
        "name": name,
        "start": str(equity.index[0].date()) if len(equity) else None,
        "end": str(equity.index[-1].date()) if len(equity) else None,
        "n_days": int(len(equity)),
        "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
        "cagr": cagr(equity),
        "ann_vol": annualized_vol(equity),
        "sharpe": sharpe(equity, rf=rf),
        "sortino": sortino(equity, rf=rf),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity),
        "mean_monthly": mean_monthly_return(equity),
        "geo_monthly": geo_monthly_return(equity),
        "median_monthly": median_monthly_return(equity),
        "hit_rate_monthly": hit_rate(equity),
        "worst_month": worst_month(equity),
        "best_month": best_month(equity),
    }


def fmt_summary(s: dict) -> str:
    """One-line pretty-print of a summary dict."""
    return (
        f"{s['name']:>28} | "
        f"CAGR {s['cagr']*100:6.2f}% | "
        f"Mo {s['mean_monthly']*100:5.2f}% | "
        f"Sharpe {s['sharpe']:5.2f} | "
        f"MaxDD {s['max_drawdown']*100:6.2f}% | "
        f"Calmar {s['calmar']:5.2f} | "
        f"HitRate {s['hit_rate_monthly']*100:4.0f}%"
    )
