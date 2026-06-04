"""Combo-v2 strategy — V5 winning blend (2026-06-04).

10-year backtest (2016-04 → 2026-03, 1,040 stocks, 5 bp/side TC):

    mean monthly return   4.96 %        ← highest of 45 strategies tested
    Sharpe                1.06
    Calmar                0.87
    max drawdown          -65 %         (2022 momentum reversal)
    leverage              2.0x          (fits Alpaca paper ≈ 2.37x bp)

Unlike combo_v1 (4 momentum-family sleeves), combo_v2 blends three
uncorrelated *winners* discovered in the V5 expansion:

  1. xs_momentum_top30  (12-1 cross-sectional, monthly rebal)
     Long top-30 stocks by 12-month return minus most recent month.
     Equal-weight 1/30 each. Best single-signal Calmar (0.85).

  2. dual_momentum_voltarget  (6-mo abs-mom + inverse-vol, vol-target 15%)
     Top-30 by 6-month total return (must be > 0, else cash). Inverse-vol
     weights, scaled to 15% portfolio realised vol. Best Sharpe (1.10).

  3. adaptive_voltarget_momentum  (xs_momentum + VIX-percentile leverage)
     Same xs_momentum top-30 picks, inverse-vol weighted, with leverage
     adapting to the rolling-252d VIX percentile: 1.5x in calm regimes
     (bottom tercile), 1.0x neutral, 0.5x in stressed (top tercile).
     Second-best Calmar across all 45 strategies (0.80).

After blending (1/3 each) we apply:
  - SPY drawdown gate: ≥8% below 60-day high ramps exposure toward 0;
                       ≥18% drawdown → full cash.
  - Gross-exposure cap: weights sum to ≤ max_gross_exposure (default 1.0).
  - target_leverage at the order layer (config.py default 2.0x) scales
    the final allocation to match the backtest behaviour.

Reference: /Users/arsenkhanguieldyan/Documents/Trading/Traiding 11/REPORT.md
(V5 section) and results/summary.csv (45 strategies).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_MULTI_ASSET_TICKERS = (
    "SPY", "QQQ", "IWM", "TLT", "SHY", "HYG", "GLD", "USO", "UUP",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
)


def _to_close_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """dict[symbol → OHLCV df] → wide close-price DataFrame."""
    if not data:
        return pd.DataFrame()
    closes = {}
    for sym, df in data.items():
        if "close" in df.columns and len(df) > 0:
            closes[sym] = df["close"]
    return pd.DataFrame(closes).sort_index() if closes else pd.DataFrame()


# ───── Sleeve 1: xs_momentum_top30 (12-1, monthly) ────────────────────────
def _xs_momentum_weights(px: pd.DataFrame, n_long: int = 30,
                         lookback_long: int = 252,
                         lookback_skip: int = 21) -> dict[str, float]:
    if px.empty or len(px) < lookback_long + lookback_skip + 5:
        return {}
    prev = px.iloc[-(lookback_skip + 1)]
    base = px.iloc[-(lookback_long + 1)]
    last = px.iloc[-1]
    mom = (prev / base - 1.0)
    mom = mom[(prev.notna()) & (base.notna()) & (last.notna())]
    mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
    if len(mom) < n_long:
        return {}
    top = mom.nlargest(n_long).index
    return {sym: 1.0 / n_long for sym in top}


# ───── Sleeve 2: dual_momentum_voltarget (6-mo + inv-vol + vol target 15%)
def _dual_momentum_voltarget_weights(
    px: pd.DataFrame, n_long: int = 30,
    lookback: int = 126, vol_lookback: int = 60,
    target_vol: float = 0.15,
) -> dict[str, float]:
    if px.empty or len(px) < lookback + 5:
        return {}
    last = px.iloc[-1]
    base = px.iloc[-(lookback + 1)]
    daily = px.pct_change()
    vol = daily.iloc[-vol_lookback:].std() * np.sqrt(252)
    mom = (last / base - 1.0)
    mom = mom[(last.notna()) & (base.notna())].replace([np.inf, -np.inf], np.nan).dropna()
    pos = mom[mom > 0]
    if pos.empty:
        return {}
    top = pos.nlargest(min(n_long, len(pos))).index
    v = vol[top].replace(0, np.nan).dropna()
    if v.empty:
        return {}
    inv_v = 1.0 / v
    raw_w = inv_v / inv_v.sum()
    est_vol = float(np.sqrt(np.sum((raw_w * vol[raw_w.index]) ** 2)))
    scale = min(1.0, target_vol / max(est_vol, 1e-6))
    return {sym: float(w * scale) for sym, w in raw_w.items()}


# ───── Sleeve 3: adaptive_voltarget_momentum (xs_mom + VIX-percentile lev)
def _adaptive_voltarget_momentum_weights(
    stock_px: pd.DataFrame, macro_close: pd.DataFrame,
    n_long: int = 30, vol_lookback: int = 60,
    lookback_long: int = 252, lookback_skip: int = 21,
    calm_pctile: float = 0.33, stress_pctile: float = 0.67,
    calm_leverage: float = 1.5, neutral_leverage: float = 1.0,
    stress_leverage: float = 0.5,
) -> dict[str, float]:
    """Top-N by 12-1 momentum, inverse-vol weighted, leverage based on
    where today's VIX sits in its rolling-252-day percentile distribution.
    """
    if stock_px.empty or len(stock_px) < lookback_long + lookback_skip + 5:
        return {}
    prev = stock_px.iloc[-(lookback_skip + 1)]
    base = stock_px.iloc[-(lookback_long + 1)]
    last = stock_px.iloc[-1]
    mom = (prev / base - 1.0)
    mom = mom[(prev.notna()) & (base.notna()) & (last.notna())]
    mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
    if len(mom) < n_long:
        return {}
    top = mom.nlargest(n_long).index
    daily = stock_px.pct_change()
    vol = daily.iloc[-vol_lookback:].std() * np.sqrt(252)
    v = vol[top].replace(0, np.nan).dropna()
    if v.empty:
        return {}
    inv = 1.0 / v
    base_w = inv / inv.sum()

    # VIX-percentile leverage. Falls back to neutral_leverage when VIX
    # data is missing or insufficient.
    lev = neutral_leverage
    if not macro_close.empty and "VIX" in macro_close.columns:
        vix = macro_close["VIX"].dropna()
        if len(vix) >= 252:
            recent = vix.iloc[-252:]
            pctile = float((recent <= recent.iloc[-1]).sum()) / len(recent)
            if pctile < calm_pctile:
                lev = calm_leverage
            elif pctile > stress_pctile:
                lev = stress_leverage
            else:
                lev = neutral_leverage
    return {sym: float(w * lev) for sym, w in base_w.items()}


# ───── Risk overlay ──────────────────────────────────────────────────────
def _spy_drawdown_gate(macro_close: pd.DataFrame, lookback: int = 60,
                       full_dd: float = 0.08, cash_dd: float = 0.18) -> float:
    """Exposure multiplier in [0, 1]. ≥18% SPY drawdown → 0 (cash);
    ≤8% drawdown → 1 (full exposure); linear ramp between.
    """
    if "SPY" not in macro_close.columns or len(macro_close) < lookback + 1:
        return 1.0
    spy = macro_close["SPY"].dropna()
    if len(spy) < lookback + 1:
        return 1.0
    rolling_high = spy.iloc[-lookback:].max()
    dd = float(spy.iloc[-1] / rolling_high - 1.0)
    if dd >= -full_dd:
        return 1.0
    if dd <= -cash_dd:
        return 0.0
    return float((dd + cash_dd) / (cash_dd - full_dd))


# ───── Config ────────────────────────────────────────────────────────────
@dataclass
class ComboConfig:
    """All knobs for ComboStrategy. Pickle-safe."""

    # Equal-weight blend of 3 sleeves (must sum to 1.0)
    xs_mom_weight: float = 1.0 / 3.0
    xs_mom_top_n: int = 30
    dual_mom_weight: float = 1.0 / 3.0
    dual_mom_top_n: int = 30
    dual_mom_vol_target: float = 0.15
    adaptive_weight: float = 1.0 / 3.0
    adaptive_top_n: int = 30
    adaptive_calm_leverage: float = 1.5
    adaptive_neutral_leverage: float = 1.0
    adaptive_stress_leverage: float = 0.5

    # SPY drawdown gate
    spy_dd_lookback: int = 60
    spy_full_dd: float = 0.08
    spy_cash_dd: float = 0.18

    # Cap on pre-leverage gross exposure
    max_gross_exposure: float = 1.0

    # Drop dust positions
    min_position_weight: float = 0.003

    def __post_init__(self):
        total = self.xs_mom_weight + self.dual_mom_weight + self.adaptive_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ComboConfig sleeve weights sum to {total:.6f}, expected 1.0. "
                f"Got xs={self.xs_mom_weight}, dual={self.dual_mom_weight}, "
                f"adapt={self.adaptive_weight}."
            )


# ───── Strategy ──────────────────────────────────────────────────────────
class ComboStrategy:
    """V5 winning combo_v2 strategy. Drops into the runner's
    direct_weights path via `compute_weights(stock_data, macro_data)`.
    """

    def __init__(self, config: ComboConfig | None = None):
        self.config = config or ComboConfig()

    def compute_weights(
        self,
        stock_data: dict[str, pd.DataFrame],
        macro_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, float]:
        c = self.config
        stock_px = _to_close_panel(stock_data) if stock_data else pd.DataFrame()
        macro_px = _to_close_panel(macro_data) if macro_data else pd.DataFrame()

        # Sleeves
        w_xs = _xs_momentum_weights(stock_px, n_long=c.xs_mom_top_n)
        w_dual = _dual_momentum_voltarget_weights(
            stock_px, n_long=c.dual_mom_top_n,
            target_vol=c.dual_mom_vol_target,
        )
        w_adapt = _adaptive_voltarget_momentum_weights(
            stock_px, macro_px,
            n_long=c.adaptive_top_n,
            calm_leverage=c.adaptive_calm_leverage,
            neutral_leverage=c.adaptive_neutral_leverage,
            stress_leverage=c.adaptive_stress_leverage,
        )

        combined: dict[str, float] = {}
        for sleeve, sleeve_weight in (
            (w_xs, c.xs_mom_weight),
            (w_dual, c.dual_mom_weight),
            (w_adapt, c.adaptive_weight),
        ):
            for sym, w in sleeve.items():
                combined[sym] = combined.get(sym, 0.0) + sleeve_weight * w

        if not combined:
            return {}

        # SPY drawdown gate
        exposure = _spy_drawdown_gate(
            macro_px, lookback=c.spy_dd_lookback,
            full_dd=c.spy_full_dd, cash_dd=c.spy_cash_dd,
        )
        combined = {sym: w * exposure for sym, w in combined.items()}

        # Gross-exposure cap (pre-leverage)
        gross = sum(abs(w) for w in combined.values())
        if gross > c.max_gross_exposure:
            scale = c.max_gross_exposure / gross
            combined = {sym: w * scale for sym, w in combined.items()}

        # Dust filter
        combined = {sym: w for sym, w in combined.items()
                    if abs(w) >= c.min_position_weight}
        return combined

    def predict(self, X):
        """Intentionally fails — combo_v2 is direct_weights only."""
        raise NotImplementedError(
            "ComboStrategy.predict() is not implemented. The bundle must "
            "set strategy_type='direct_weights' so the runner dispatches "
            "to compute_weights() instead."
        )
