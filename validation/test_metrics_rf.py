"""Tests for the 2026-07-18 additive rf-Series extension of metrics.sharpe /
metrics.sortino (WS4a, exp_ws4_rf.py).

The extension is ADDITIVE: rf may now be a daily-aligned pd.Series of
ANNUALIZED decimal rates (excess = r - rf/252). The binding contract tested
here is that the scalar path — in particular the rf=0.0 default the entire
published-record fidelity chain depends on — is byte-identical to the
pre-change code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import ANN, daily_returns, sharpe, sortino, summary  # noqa: E402


def _equity(n: int = 500, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    r = rng.normal(0.0006, 0.012, size=n)
    return pd.Series(100_000.0 * np.cumprod(1.0 + r), index=idx)


def _legacy_sharpe(equity: pd.Series, rf: float) -> float:
    """Verbatim pre-2026-07-18 implementation (scalar rf only)."""
    r = daily_returns(equity)
    excess = r - rf / ANN
    return float(excess.mean() / excess.std() * np.sqrt(ANN))


def _legacy_sortino(equity: pd.Series, rf: float) -> float:
    r = daily_returns(equity)
    excess = r - rf / ANN
    dd = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    return float(excess.mean() / dd * np.sqrt(ANN))


def test_scalar_default_unchanged():
    eq = _equity()
    assert sharpe(eq) == _legacy_sharpe(eq, 0.0)
    assert sortino(eq) == _legacy_sortino(eq, 0.0)
    assert sharpe(eq, rf=0.03) == _legacy_sharpe(eq, 0.03)
    assert sortino(eq, rf=0.03) == _legacy_sortino(eq, 0.03)


def test_constant_series_matches_scalar():
    eq = _equity()
    r_idx = daily_returns(eq).index
    rf_s = pd.Series(0.05, index=r_idx)
    assert abs(sharpe(eq, rf=rf_s) - sharpe(eq, rf=0.05)) < 1e-12
    assert abs(sortino(eq, rf=rf_s) - sortino(eq, rf=0.05)) < 1e-12


def test_series_alignment_ffill_and_leading_zero():
    eq = _equity(300)
    r_idx = daily_returns(eq).index
    # rf quoted only on a sparse grid inside the window, starting late:
    # missing leading days -> 0.0; gaps -> forward-fill.
    rf_sparse = pd.Series(0.04, index=r_idx[100::5])
    got = sharpe(eq, rf=rf_sparse)
    rf_full = rf_sparse.reindex(r_idx).ffill().fillna(0.0)
    want = sharpe(eq, rf=rf_full)
    assert abs(got - want) < 1e-12
    # And a positive rf must lower the Sharpe of a long-only-ish curve.
    assert got < sharpe(eq)


def test_summary_accepts_series_rf():
    eq = _equity()
    rf_s = pd.Series(0.05, index=eq.index)
    s = summary(eq, rf=rf_s)
    assert abs(s["sharpe"] - sharpe(eq, rf=0.05)) < 1e-12
    s0 = summary(eq)
    assert s0["sharpe"] == sharpe(eq)


if __name__ == "__main__":
    test_scalar_default_unchanged()
    test_constant_series_matches_scalar()
    test_series_alignment_ffill_and_leading_zero()
    test_summary_accepts_series_rf()
    print("test_metrics_rf: all OK")
