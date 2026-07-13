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


def median_monthly_return(equity: pd.Series) -> float:
    m = monthly_returns(equity)
    return float(m.median()) if len(m) else float("nan")


def annualized_vol(equity: pd.Series) -> float:
    r = daily_returns(equity)
    return float(r.std() * np.sqrt(ANN)) if len(r) else float("nan")


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    r = daily_returns(equity)
    if r.std() == 0 or len(r) == 0:
        return float("nan")
    excess = r - rf / ANN
    return float(excess.mean() / r.std() * np.sqrt(ANN))


def sortino(equity: pd.Series, rf: float = 0.0) -> float:
    r = daily_returns(equity) - rf / ANN
    downside = r[r < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(r.mean() / downside.std() * np.sqrt(ANN))


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


def summary(equity: pd.Series, name: str = "strategy") -> dict:
    return {
        "name": name,
        "start": str(equity.index[0].date()) if len(equity) else None,
        "end": str(equity.index[-1].date()) if len(equity) else None,
        "n_days": int(len(equity)),
        "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
        "cagr": cagr(equity),
        "ann_vol": annualized_vol(equity),
        "sharpe": sharpe(equity),
        "sortino": sortino(equity),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity),
        "mean_monthly": mean_monthly_return(equity),
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
