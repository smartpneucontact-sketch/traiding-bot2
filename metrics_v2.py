"""V7 metrics extensions: crash-episode tables, deflated Sharpe, window
slicing for the dev/validation split, rolling stability.

Builds on metrics.py (same conventions). Episode windows match
validation/book_gate.py EPISODES so all reports are comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from metrics import daily_returns, summary  # noqa: F401  (re-exported)

# Binding window split for the V7 program (selection vs locked validation).
DEV_END = "2022-12-31"
VAL_START = "2023-01-01"

# Same windows as validation/book_gate.py EPISODES.
EPISODES = {
    "2018Q4":   ("2018-10-01", "2018-12-31"),
    "covid":    ("2020-02-19", "2020-04-07"),
    "2022":     ("2022-01-01", "2022-12-31"),
    "2026Q1":   ("2026-01-01", "2026-03-27"),
}


def episode_stats(equity: pd.Series, start: str, end: str) -> dict:
    """Window return + in-window max drawdown (book_gate.py convention)."""
    eq = equity.loc[start:end]
    if len(eq) < 2:
        return {"ret": float("nan"), "max_dd": float("nan")}
    peak = eq.cummax()
    return {
        "ret": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "max_dd": float((eq / peak - 1.0).min()),
    }


def crash_table(equity: pd.Series) -> dict:
    return {name: episode_stats(equity, a, b) for name, (a, b) in EPISODES.items()}


def split_windows(equity: pd.Series) -> dict[str, pd.Series]:
    """dev (≤2022-12-31) / val (≥2023-01-01) / full slices of an equity curve."""
    return {
        "dev": equity.loc[:DEV_END],
        "val": equity.loc[VAL_START:],
        "full": equity,
    }


def window_summaries(equity: pd.Series, name: str = "strategy") -> dict:
    out = {}
    for win, eq in split_windows(equity).items():
        if len(eq) > 30:
            s = summary(eq, name=f"{name}[{win}]")
            s["window"] = win
            out[win] = s
    out["episodes"] = crash_table(equity)
    return out


def deflated_sharpe(
    observed_sr_annual: float,
    n_obs_daily: int,
    n_trials: int,
    sr_variance_across_trials: float | None = None,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Bailey & López de Prado deflated Sharpe probability.

    Returns P(true SR > 0 | multiple testing). `observed_sr_annual` is the
    annualized Sharpe; internally converted to the daily (non-annualized)
    scale at which the PSR formula operates. If the cross-trial SR variance
    is unknown, a conservative default equal to the squared observed daily
    SR is used.
    """
    if n_obs_daily < 30 or n_trials < 1:
        return float("nan")
    sr = observed_sr_annual / np.sqrt(252.0)  # daily-scale SR
    v = sr_variance_across_trials
    if v is None or not np.isfinite(v) or v <= 0:
        v = max(sr ** 2, 1e-8)
    emc = 0.5772156649015329
    max_z = ((1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials)
             + emc * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr0 = np.sqrt(v) * max_z  # expected max SR under H0 across trials
    denom = np.sqrt(max(1e-12,
                        1 - skew * sr + (excess_kurtosis - 1) / 4.0 * sr ** 2))
    z = (sr - sr0) * np.sqrt(n_obs_daily - 1) / denom
    return float(stats.norm.cdf(z))


def rolling_calmar(equity: pd.Series, window_days: int = 756) -> pd.Series:
    """Rolling 3-year Calmar for stability reporting."""
    out = {}
    idx = equity.index
    for i in range(window_days, len(idx)):
        eq = equity.iloc[i - window_days:i]
        years = window_days / 252.0
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
        mdd = abs((eq / eq.cummax() - 1.0).min())
        out[idx[i]] = cagr / mdd if mdd > 0 else np.nan
    return pd.Series(out)


def newey_west_tstat(diff_monthly: pd.Series, lags: int = 3) -> float:
    """Newey-West t-stat for the mean of paired monthly return differences."""
    x = diff_monthly.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 12:
        return float("nan")
    mu = x.mean()
    e = x - mu
    s = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return float(mu / se) if se > 0 else float("nan")
