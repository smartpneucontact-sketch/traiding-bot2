"""Market-breadth features computed from the EXISTING local close panel.

SURVIVORSHIP WARNING (binding caveat for every consumer)
--------------------------------------------------------
The bar cache is survivors-only: names that delisted, blew up, or were
acquired have no bars, and the panel's cross-section at date t is "names that
exist TODAY and already traded at t", not the true tradable universe at t.
Breadth LEVELS built on such a panel are upwardly biased (the dead cohort —
disproportionately names that were crashing — is absent from the denominator
exactly when breadth signals matter), and the bias is worst early in the
window where the missing cohort is largest (24.8% of true 2016 S&P members
have no bars; see results/v7/pit/PIT_REPORT.md). For that reason this module
deliberately exposes ROLLING-PERCENTILE transforms as the only intended
regime signal: ranking a biased level against its own trailing history
removes the slow level bias, though it cannot remove the cross-sectional
composition bias inside fast crashes. Consumers must not threshold raw
levels.

All features use only same-day-or-earlier closes (rolling windows end at t);
the engine's decision-side shift adds the execution lag on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def frac_above_ma(close: pd.DataFrame, window: int = 200) -> pd.Series:
    """Fraction of names trading above their `window`-day simple moving
    average, among names with a defined MA that day (min_periods=window)."""
    ma = close.rolling(window, min_periods=window).mean()
    above = (close > ma) & ma.notna()
    valid = ma.notna() & close.notna()
    n_valid = valid.sum(axis=1).replace(0, np.nan)
    out = above.sum(axis=1) / n_valid
    out.name = f"frac_above_{window}dma"
    return out


def adv_decline_net(close: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Net advance-decline breadth over `lookback` days:
    (advancers - decliners) / valid names, in [-1, 1]."""
    r = close.pct_change(lookback, fill_method=None)
    adv = (r > 0).sum(axis=1)
    dec = (r < 0).sum(axis=1)
    n_valid = r.notna().sum(axis=1).replace(0, np.nan)
    out = (adv - dec) / n_valid
    out.name = f"ad_net_{lookback}d"
    return out


def rolling_percentile(s: pd.Series, window: int = 756,
                       min_periods: int = 252) -> pd.Series:
    """Percentile rank of s_t within its own trailing `window` values
    (inclusive of t): fraction of the window <= current value, in (0, 1].
    Uses only information up to t. NaN until `min_periods` observations."""
    def _pct(x: np.ndarray) -> float:
        c = x[-1]
        if np.isnan(c):
            return np.nan
        v = x[~np.isnan(x)]
        return float((v <= c).mean()) if len(v) else np.nan
    out = s.rolling(window, min_periods=min_periods).apply(_pct, raw=True)
    out.name = f"{s.name}_pct{window}"
    return out
