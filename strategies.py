"""All trading strategies. Each function returns a weights DataFrame
(index = decision date, columns = ticker, values = target weight).

Conventions
-----------
- Long-only unless noted. Sum of |weights| ≤ leverage_cap (enforced in backtest).
- "Rebalance every N days" means we emit a new row every N trading days; the
  backtester forward-fills weights between decisions, so transaction costs only
  hit on rebalance dates.
- All decisions use information available *at the close of the decision date*;
  the backtester applies a 1-day execution lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Strategy 1: Buy & hold SPY (benchmark) ──────────────────────────────
def buy_hold_spy(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """SPY = 100% of equity, never rebalanced."""
    if "SPY" not in macro.columns:
        raise RuntimeError("SPY missing from macro panel")
    dates = prices.index
    w = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    if "SPY" in prices.columns:
        w["SPY"] = 1.0
    else:
        # SPY only in macro — add it as an extra column for the BT
        w["SPY"] = 1.0
    return w


# ─── Strategy 2: Cross-sectional momentum (12-1) ────────────────────────
def xs_momentum(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    lookback_long: int = 252,   # 12 months
    lookback_skip: int = 21,    # skip last month
    n_long: int = 50,
    rebal_freq: int = 21,       # monthly
) -> pd.DataFrame:
    """Buy top-N stocks by (252-day return excluding most recent 21 days).
    Equal-weight, monthly rebalance."""
    px = prices
    # 12-1 momentum: return from t-252 to t-21
    mom = px.shift(lookback_skip) / px.shift(lookback_long) - 1.0
    # Filter: require ≥ 252 trading days of history (drop newly listed)
    valid = mom.notna()

    w_list = []
    dates = px.index
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < lookback_long + lookback_skip:
            continue
        row_mom = mom.loc[dt][valid.loc[dt]]
        if len(row_mom) < n_long:
            continue
        top = row_mom.nlargest(n_long).index
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[top] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    w = pd.concat(w_list, axis=1).T
    return w


# ─── Strategy 2a: Multi-horizon momentum (1m, 3m, 6m, 12-1) ─────────────
def xs_momentum_multi(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    n_long: int = 30,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Average rank across four momentum horizons: 21d, 63d, 126d, 252-21d.
    Then long top-N. Historically more robust than single-horizon momentum
    because it doesn't get fooled by a single-window anomaly."""
    px = prices
    m1 = px.pct_change(21, fill_method=None)
    m3 = px.pct_change(63, fill_method=None)
    m6 = px.pct_change(126, fill_method=None)
    m12 = px.shift(21) / px.shift(252) - 1.0

    # Per-day cross-sectional rank for each horizon (0–1), then average
    def rank(df):
        return df.rank(axis=1, pct=True)
    avg_rank = (rank(m1) + rank(m3) + rank(m6) + rank(m12)) / 4

    w_list = []
    dates = px.index
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < 280:
            continue
        row = avg_rank.loc[dt].dropna()
        if len(row) < n_long:
            continue
        top = row.nlargest(n_long).index
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[top] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ─── Strategy 2c: Concentrated top-15 momentum with quality screen ──
def xs_momentum_concentrated(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    n_long: int = 15,
    rebal_freq: int = 10,         # bi-weekly rebal — faster reaction
    min_price: float = 5.0,
    min_dollar_vol: float = 1e7,  # $10M/day average → liquid enough to scale
) -> pd.DataFrame:
    """High-conviction: top-15 by 12-1 momentum, with quality screens:
      - Price ≥ $5 (avoid penny stocks)
      - 20d avg dollar volume ≥ $10M
      - TOTAL realised vol over last 60d in the bottom 80% (drop the wildest;
        not idiosyncratic vol — no factor model is fitted)
    Bi-weekly rebalance for tighter capture."""
    px = prices
    mom = px.shift(21) / px.shift(252) - 1.0
    daily_ret = px.pct_change(fill_method=None)
    vol60 = daily_ret.rolling(60).std()
    price_ok = px > min_price
    if volume is not None:
        adv = (px * volume).rolling(20).mean()
        liq_ok = adv > min_dollar_vol
    else:
        liq_ok = pd.DataFrame(True, index=px.index, columns=px.columns)
    # Per-day vol percentile (drop top 20%)
    vol_pct = vol60.rank(axis=1, pct=True)
    not_too_volatile = vol_pct <= 0.80
    valid = mom.notna() & price_ok & liq_ok & not_too_volatile

    w_list = []
    dates = px.index
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < 280:
            continue
        row = mom.loc[dt][valid.loc[dt]]
        if len(row) < n_long:
            continue
        top = row.nlargest(n_long).index
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[top] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ─── Strategy 2b: Long-short cross-sectional momentum ─────────────────
def xs_momentum_ls(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    lookback_long: int = 252,
    lookback_skip: int = 21,
    n_each_side: int = 50,
    rebal_freq: int = 21,
    long_weight: float = 1.0,
    short_weight: float = 0.5,
) -> pd.DataFrame:
    """Long top-N, short bottom-N by 12-1 momentum. Reduces market beta
    and (historically) cuts drawdowns vs long-only."""
    px = prices
    mom = px.shift(lookback_skip) / px.shift(lookback_long) - 1.0
    valid = mom.notna()

    w_list = []
    dates = px.index
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < lookback_long + lookback_skip:
            continue
        row = mom.loc[dt][valid.loc[dt]]
        if len(row) < 2 * n_each_side:
            continue
        top = row.nlargest(n_each_side).index
        bot = row.nsmallest(n_each_side).index
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[top] = long_weight / n_each_side
        wrow.loc[bot] = -short_weight / n_each_side
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ─── Strategy 3: Short-term mean reversion ──────────────────────────────
def mean_reversion(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    lookback: int = 5,
    n_long: int = 30,
    rebal_freq: int = 5,
    min_dollar_vol: float = 5e6,
    volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Buy most-oversold stocks (worst N-day return), liquid only, weekly rebal."""
    ret = prices.pct_change(lookback, fill_method=None)
    # Liquidity filter: 20-day average dollar volume ≥ threshold
    if volume is not None:
        adv = (prices * volume).rolling(20).mean()
        liquid = adv > min_dollar_vol
    else:
        liquid = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    valid = ret.notna() & liquid

    w_list = []
    dates = prices.index
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < lookback + 20:
            continue
        row = ret.loc[dt][valid.loc[dt]]
        if len(row) < n_long:
            continue
        # Most-negative N: buying the dip
        bottom = row.nsmallest(n_long).index
        wrow = pd.Series(0.0, index=prices.columns)
        wrow.loc[bottom] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=prices.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ─── Strategy 4: Trend following with regime gate ───────────────────────
def trend_following(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    sma_fast: int = 50,
    sma_slow: int = 200,
    breakout_lookback: int = 100,
    n_long: int = 30,
    rebal_freq: int = 21,
    cash_when_no_signal: bool = False,
) -> pd.DataFrame:
    """Long stocks that are above their 200-SMA AND making new 100-day highs.
    Rank by relative strength (50-SMA over 200-SMA gap). Equal-weight top N.

    Legacy default: when NO stock qualifies on a decision date, no row is
    emitted, so the backtester's forward-fill HOLDS the previous book (it does
    not go to cash). Pass cash_when_no_signal=True to emit an explicit
    all-zero row on those dates instead."""
    px = prices
    sma50 = px.rolling(sma_fast).mean()
    sma200 = px.rolling(sma_slow).mean()
    high_n = px.rolling(breakout_lookback).max()

    is_uptrend = (sma50 > sma200) & (px > sma200)
    new_high = px >= high_n * 0.98  # within 2% of N-day high
    rel_strength = (sma50 / sma200 - 1.0)

    w_list = []
    dates = px.index
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < sma_slow + 10:
            continue
        eligible = is_uptrend.loc[dt] & new_high.loc[dt]
        scores = rel_strength.loc[dt][eligible].dropna()
        if len(scores) == 0:
            if cash_when_no_signal:
                w_list.append(pd.Series(0.0, index=px.columns).rename(dt))
            continue
        k = min(n_long, len(scores))
        top = scores.nlargest(k).index
        wrow = pd.Series(0.0, index=px.columns)
        if k > 0:
            wrow.loc[top] = 1.0 / k
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ─── Helper: portfolio vol estimate for vol-targeting sleeves ────────────
def _est_portfolio_vol(
    w: pd.Series,
    name_vol: pd.Series,
    daily: pd.DataFrame,
    dt: pd.Timestamp,
    vol_lookback: int,
    vol_est: str,
) -> float:
    """Annualised portfolio-vol estimate for weights `w` (index = tickers held).

    vol_est="diag_legacy": sqrt(sum((w_i * sigma_i)^2)) — assumes ZERO
        cross-correlation. For a diversified long book of correlated names this
        understates true vol so badly that min(1, target/est) never binds:
        the vol-targeting is a NO-OP. Retained only to reproduce the published
        record bit-for-bit.
    vol_est="cov": sqrt(w' Sigma w * 252) with Sigma the sample covariance of
        the same trailing `vol_lookback` daily-return window already used for
        the per-name sigmas. Use this for new research.

    `name_vol` is the per-name annualised vol Series at `dt` (used only by
    diag_legacy); `daily` is the full daily-returns frame (used only by cov).

    If the covariance is undefined at `dt` (a held name has <2 observations in
    the window), the cov path falls back to the diagonal estimate for that
    decision date — explicit, rather than letting NaN leak through the
    caller's min()/max() (whose result depends on argument order).
    """
    if vol_est == "diag_legacy":
        return float(np.sqrt(np.sum((w * name_vol[w.index]) ** 2)))
    if vol_est == "cov":
        window = daily.loc[:dt, w.index].tail(vol_lookback)
        sigma = window.cov()
        var = float(w.values @ sigma.values @ w.values)
        if not np.isfinite(var):
            return float(np.sqrt(np.sum((w * name_vol[w.index]) ** 2)))
        return float(np.sqrt(max(var, 0.0) * 252.0))
    raise ValueError(f"unknown vol_est: {vol_est!r}")


# ─── Strategy 5: Volatility-targeted dual momentum ──────────────────────
def dual_momentum_voltarget(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    lookback: int = 126,          # 6-month total return
    n_long: int = 30,
    target_vol: float = 0.15,     # 15% annualized
    vol_lookback: int = 60,
    rebal_freq: int = 21,
    vol_est: str = "diag_legacy",
    cash_when_no_signal: bool = False,
) -> pd.DataFrame:
    """Dual momentum: pick top-N by 6-month return *and* require absolute return > 0.
    Size each holding inversely to its 60-day volatility, then scale the
    portfolio to target_vol.

    Caveats of the legacy defaults (kept for published-record reproduction):
    - vol_est="diag_legacy" ignores cross-correlation, so the estimated
      portfolio vol is far below target_vol and the scale never binds in
      practice — the vol-targeting is a NO-OP. New research should pass
      vol_est="cov" (full sample-covariance estimate; see _est_portfolio_vol).
    - When no name qualifies on a decision date the legacy code emits NO row,
      so the backtester's forward-fill silently HOLDS the previous book — it
      does NOT go to cash despite the "else hold cash" intent. Pass
      cash_when_no_signal=True to emit an explicit all-zero row instead.
    """
    px = prices
    ret = px.pct_change(lookback, fill_method=None)
    daily = px.pct_change(fill_method=None)
    vol = daily.rolling(vol_lookback).std() * np.sqrt(252)

    w_list = []
    dates = px.index
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < max(lookback, vol_lookback) + 5:
            continue
        row = ret.loc[dt].dropna()
        # Absolute filter: must be positive
        row = row[row > 0]
        if len(row) == 0:
            if cash_when_no_signal:
                w_list.append(pd.Series(0.0, index=px.columns).rename(dt))
            continue
        top = row.nlargest(min(n_long, len(row))).index
        # Inverse-vol weights inside the top set
        v = vol.loc[dt][top].replace(0.0, np.nan).dropna()
        if v.empty:
            if cash_when_no_signal:
                w_list.append(pd.Series(0.0, index=px.columns).rename(dt))
            continue
        raw_w = 1.0 / v
        raw_w = raw_w / raw_w.sum()  # normalise to sum=1
        # Portfolio realized vol at these weights
        port_vol_est = _est_portfolio_vol(raw_w, vol.loc[dt], daily, dt, vol_lookback, vol_est)
        scale = min(1.0, target_vol / max(port_vol_est, 1e-6))
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[raw_w.index] = raw_w.values * scale
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ─── Strategy 5b: Time-series momentum across assets ────────────────────
def ts_momentum_multiasset(
    macro: pd.DataFrame,
    lookback: int = 252,
    rebal_freq: int = 21,
    target_vol: float = 0.20,
    vol_lookback: int = 60,
    vol_est: str = "diag_legacy",
) -> pd.DataFrame:
    """Classic AQR-style time-series momentum applied to ALL the macro/sector
    ETFs (22 of them). For each ETF: long if 12-month return > 0, else cash.
    Inverse-vol weights, then scaled to target_vol.

    The default vol_est="diag_legacy" assumes zero cross-correlation, so the
    scale never binds in practice (a no-op, retained only to reproduce the
    published record). New research should pass vol_est="cov".

    Returns weights indexed by macro tickers — caller must project onto the
    full pricing universe.
    """
    px = macro
    ret_12 = px.pct_change(lookback, fill_method=None)
    daily = px.pct_change(fill_method=None)
    vol_d = daily.rolling(vol_lookback).std() * np.sqrt(252)

    w_list = []
    dates = px.index
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < lookback + vol_lookback:
            continue
        signal = ret_12.loc[dt]
        long_only = signal[signal > 0].dropna()
        if len(long_only) == 0:
            wrow = pd.Series(0.0, index=px.columns)
        else:
            inv_v = 1.0 / vol_d.loc[dt][long_only.index].replace(0, np.nan).dropna()
            if inv_v.empty:
                continue
            inv_v = inv_v / inv_v.sum()
            est_vol = _est_portfolio_vol(inv_v, vol_d.loc[dt], daily, dt, vol_lookback, vol_est)
            scale = min(1.0, target_vol / max(est_vol, 1e-6))
            wrow = pd.Series(0.0, index=px.columns)
            wrow.loc[inv_v.index] = inv_v.values * scale
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ─── Strategy 2d: Fast cross-sectional momentum (1-3 month, weekly rebal)─
def xs_momentum_fast(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    lookback_long: int = 63,    # 3-month
    lookback_skip: int = 5,
    n_long: int = 20,
    rebal_freq: int = 5,
    volume: pd.DataFrame | None = None,
    min_dollar_vol: float = 5e6,
) -> pd.DataFrame:
    """Top-N by 63-day (~3-month) momentum skipping the most recent 5 days,
    weekly rebal. (Despite the "fast" name this is a 3-month lookback, not
    "3-1 week".) Faster turnover; literature shows shorter momentum lookbacks
    can capture more rapid trend rotations."""
    px = prices
    mom = px.shift(lookback_skip) / px.shift(lookback_long) - 1.0
    if volume is not None:
        adv = (px * volume).rolling(20).mean()
        liq = adv > min_dollar_vol
    else:
        liq = pd.DataFrame(True, index=px.index, columns=px.columns)
    valid = mom.notna() & liq

    w_list = []
    dates = px.index
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < lookback_long + lookback_skip + 20:
            continue
        row = mom.loc[dt][valid.loc[dt]]
        if len(row) < n_long:
            continue
        top = row.nlargest(n_long).index
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[top] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ─── Strategy 6: Sector rotation ────────────────────────────────────────
def sector_rotation(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    lookback: int = 63,    # 3-month return
    n_long: int = 4,       # top 4 sectors of 11
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Buy top-N sector ETFs by 3-month return. Equal-weight."""
    sectors = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
    sectors = [s for s in sectors if s in macro.columns]
    sec_prices = macro[sectors].dropna(how="all")
    ret = sec_prices.pct_change(lookback)
    dates = sec_prices.index
    w_list = []
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < lookback + 5:
            continue
        row = ret.loc[dt].dropna()
        if len(row) == 0:
            continue
        top = row.nlargest(min(n_long, len(row))).index
        wrow = pd.Series(0.0, index=sectors)
        wrow.loc[top] = 1.0 / len(top)
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=sectors).fillna(0.0)
    w_panel = pd.concat(w_list, axis=1).T
    # Project onto the stock universe by adding sector columns explicitly
    return w_panel


# ─── Strategy 7: ML cross-sectional ranking (LightGBM) ──────────────────
def _compute_ml_features(
    px: pd.DataFrame,
    volume: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Cross-sectional + time-series features computed in a vectorised way."""
    ret = px.pct_change()
    log_ret = np.log1p(ret)
    feats: dict[str, pd.DataFrame] = {}
    for lb in (5, 10, 21, 63, 126, 252):
        feats[f"ret_{lb}"] = px.pct_change(lb)
        feats[f"vol_{lb}"] = ret.rolling(lb).std()
    for lb in (50, 200):
        feats[f"px_over_sma_{lb}"] = px / px.rolling(lb).mean() - 1
    feats["high_252"] = px / px.rolling(252).max() - 1
    feats["low_252"] = px / px.rolling(252).min() - 1
    # Volume ratio: 20d / 60d
    if volume is not None:
        feats["dv_20_60"] = (px * volume).rolling(20).mean() / (px * volume).rolling(60).mean()
    return feats


def ml_cross_sectional(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    horizon: int = 5,
    n_long: int = 30,
    rebal_freq: int = 5,
    train_window_days: int = 252 * 2,    # 2-year rolling train
    retrain_every: int = 63,             # quarterly retrain
    seed: int = 42,
) -> pd.DataFrame:
    """Walk-forward LightGBM regression: predict `horizon`-day forward return
    on cross-sectional + time-series features. Long top-N by predicted return,
    equal-weight, rebalanced every `rebal_freq` days. Retrains every quarter."""
    import lightgbm as lgb

    px = prices
    feats = _compute_ml_features(px, volume)
    feature_names = list(feats.keys())
    fwd = px.pct_change(horizon).shift(-horizon)  # forward h-day return aligned at decision date

    dates = px.index
    feat_panel = np.stack([feats[n].values for n in feature_names], axis=-1)  # (T, N, F)
    fwd_arr = fwd.values  # (T, N)

    T, N, F = feat_panel.shape
    w_list = []
    last_train_i = -10**9
    model = None
    rng = np.random.default_rng(seed)
    # Winsorize forward returns to suppress crazy outliers (kept for safety even
    # after data-level outlier filter — covers e.g. surprise pre-earnings moves).
    fwd_arr = np.clip(fwd_arr, -0.5, 0.5)

    for i, dt in enumerate(dates):
        if i < train_window_days + 50:
            continue
        # Retrain every `retrain_every` days
        if i - last_train_i >= retrain_every:
            last_train_i = i
            tr_start = max(0, i - train_window_days)
            tr_end = i - horizon  # last index with valid forward return is i-h
            X_blocks = feat_panel[tr_start:tr_end].reshape(-1, F)
            y_blocks = fwd_arr[tr_start:tr_end].reshape(-1)
            mask = np.isfinite(X_blocks).all(axis=1) & np.isfinite(y_blocks)
            X = X_blocks[mask]
            y = y_blocks[mask]
            if len(y) < 5000:
                continue
            # Sub-sample for speed
            if len(y) > 400_000:
                idx = rng.choice(len(y), size=400_000, replace=False)
                X = X[idx]; y = y[idx]
            model = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, num_leaves=63,
                min_data_in_leaf=200, feature_fraction=0.85, bagging_fraction=0.85,
                bagging_freq=5, random_state=seed, verbose=-1,
            )
            model.fit(X, y)
        if model is None or (i % rebal_freq != 0):
            continue
        Xpred = feat_panel[i]
        mask = np.isfinite(Xpred).all(axis=1)
        if mask.sum() < n_long:
            continue
        yhat = np.full(N, -np.inf)
        yhat[mask] = model.predict(Xpred[mask])
        # Also require stock to have valid 21d/63d returns (filters newly listed / illiquid)
        mom_filter = (np.isfinite(feat_panel[i, :, feature_names.index("ret_21")]) &
                      np.isfinite(feat_panel[i, :, feature_names.index("ret_63")]))
        yhat[~mom_filter] = -np.inf
        top_idx = np.argpartition(-yhat, n_long)[:n_long]
        top_idx = top_idx[np.argsort(-yhat[top_idx])]
        wrow = pd.Series(0.0, index=px.columns)
        wrow.iloc[top_idx] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ─── Regime detector ────────────────────────────────────────────────────
def market_regime(macro: pd.DataFrame) -> pd.Series:
    """Per-day regime score in [-1, +1] (-1 = full risk-off, +1 = full risk-on).

    Inputs (all available in `macro`):
      - SPY: trend (above/below 200-SMA, 50/200 cross)
      - VIX: scaled (50 = full risk-off, 10 = full risk-on; (30-VIX)/20 clipped)
      - HYG: high-yield trend (above 200-SMA = risk-on)
    """
    if "SPY" not in macro.columns:
        return pd.Series(0.0, index=macro.index)
    spy = macro["SPY"]
    sma50 = spy.rolling(50).mean()
    sma200 = spy.rolling(200).mean()
    trend = ((spy > sma200).astype(float) + (sma50 > sma200).astype(float) - 1.0)  # -1..+1
    vix = macro.get("VIX", pd.Series(15.0, index=macro.index))
    vix_score = ((30.0 - vix) / 20.0).clip(-1, 1)
    hyg = macro.get("HYG", spy)
    hyg_trend = ((hyg > hyg.rolling(200).mean()).astype(float) * 2 - 1)
    score = 0.4 * trend + 0.3 * vix_score + 0.3 * hyg_trend
    return score.clip(-1, 1).fillna(0.0)


# ─── Helper: convert sector_rotation weights into the stock universe ────
def align_to_stock_cols(weights: pd.DataFrame, stock_cols: list[str], macro_cols: list[str]) -> pd.DataFrame:
    """Re-project a weights matrix that uses macro/sector columns onto the
    union (stock_cols ∪ macro_cols). Used by sector_rotation and buy_hold_spy."""
    union = list(dict.fromkeys(list(stock_cols) + list(macro_cols)))
    out = pd.DataFrame(0.0, index=weights.index, columns=union)
    for c in weights.columns:
        if c in out.columns:
            out[c] = weights[c].values
    return out


def union_prices(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Stock-close panel plus macro-close columns. Used when a strategy holds
    macro/sector ETFs (SPY, sector rotation)."""
    extras = [c for c in macro.columns if c not in prices.columns]
    return pd.concat([prices, macro[extras]], axis=1).sort_index()


# ════════════════════════════════════════════════════════════════════════
# v5 — Alternative strategy classes
# Added 2026-06-04 to widen the search beyond pure momentum. Same calling
# convention as the originals (returns weights DataFrame; long-only unless
# noted). Each is intentionally distinct so the comparison spans many
# strategy archetypes.
# ════════════════════════════════════════════════════════════════════════


def low_vol_quality(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    vol_lookback: int = 60,
    n_long: int = 30,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Defensive low-vol factor. Buy bottom-N stocks by 60-day realised vol,
    inverse-vol weighted then renormalised to sum=1. Academic literature
    (Baker/Haugen) shows low-vol stocks earn premia in many regimes.
    """
    daily = prices.pct_change(fill_method=None)
    vol = daily.rolling(vol_lookback).std() * np.sqrt(252)
    dates = prices.index
    w_list = []
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < vol_lookback + 5:
            continue
        row = vol.loc[dt].replace(0, np.nan).dropna()
        if len(row) < n_long:
            continue
        # Lowest vol = "defensive"
        bottom = row.nsmallest(n_long).index
        inv_v = 1.0 / row.loc[bottom]
        wnorm = inv_v / inv_v.sum()
        wrow = pd.Series(0.0, index=prices.columns)
        wrow.loc[wnorm.index] = wnorm.values
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=prices.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def acceleration_momentum(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    short_lb: int = 63,
    long_lb: int = 252,
    skip: int = 21,
    n_long: int = 30,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Buy stocks whose momentum is ACCELERATING — top-N by (3-mo return
    minus 12-1 mo return). Rewards stocks where the recent trend is stronger
    than the long trend (positive 2nd derivative).
    """
    px = prices
    short_mom = px.shift(skip) / px.shift(skip + short_lb) - 1.0
    long_mom = px.shift(skip) / px.shift(long_lb) - 1.0
    accel = short_mom - long_mom
    dates = px.index
    w_list = []
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < long_lb + skip + 5:
            continue
        row = accel.loc[dt].dropna()
        # Require positive long momentum too (don't catch falling knives)
        lm = long_mom.loc[dt]
        row = row[lm.reindex(row.index) > 0]
        if len(row) < n_long:
            continue
        top = row.nlargest(n_long).index
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[top] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def donchian_breakout(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    breakout_lookback: int = 100,
    exit_lookback: int = 50,
    n_long: int = 30,
    rebal_freq: int = 5,
) -> pd.DataFrame:
    """Turtle-style channel breakout. Long stocks at new 100-day high; hold
    until they touch the 50-day low. Position selection: 30 stocks with the
    largest (close / 100d_high - 1) score among those currently above their
    breakout level (so we pick the freshest breakouts).
    """
    px = prices
    high_n = px.rolling(breakout_lookback).max()
    low_n = px.rolling(exit_lookback).min()
    above_high = (px >= high_n * 0.995)
    below_exit = (px <= low_n * 1.005)
    # "Freshness" — how close to the top of the channel
    freshness = (px / high_n - 1.0)  # ~0 at entry; negative as price drifts down
    dates = px.index
    held = set()
    w_list = []
    for i, dt in enumerate(dates):
        # Update held positions first: drop those at/below exit channel
        held = {t for t in held if t in below_exit.columns and not below_exit.loc[dt, t]}
        if i % rebal_freq == 0 and i >= breakout_lookback + 5:
            # Add freshest breakouts
            candidates = above_high.loc[dt]
            scores = freshness.loc[dt][candidates].dropna()
            if len(scores) > 0:
                top = scores.nlargest(n_long).index
                held = (held | set(top))
                # Cap at 2 × n_long to avoid runaway turnover. Truncation is
                # deterministic: keep the names freshest today, ties (and
                # missing freshness) broken by ticker. NOTE: the published
                # donchian number was produced by the old `list(set)` slice —
                # nondeterministic hash order — and is not exactly reproducible.
                if len(held) > 2 * n_long:
                    fresh_today = freshness.loc[dt]

                    def _trunc_key(t: str) -> tuple[float, str]:
                        f = fresh_today.get(t, np.nan)
                        return (-f if pd.notna(f) else np.inf, t)

                    held = set(sorted(held, key=_trunc_key)[:2 * n_long])
            wrow = pd.Series(0.0, index=px.columns)
            if held:
                w = 1.0 / len(held)
                for t in held:
                    if t in wrow.index:
                        wrow.loc[t] = w
            w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def multifactor_mvr(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    n_long: int = 30,
    vol_lookback: int = 60,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Composite-rank long-only factor model. For each stock, compute three z-
    scored ranks across the cross-section:
      - Momentum (12-1 month return), higher = better
      - Volatility (60-day realised vol), lower = better
      - 1-month reversal (-1 × last 21d return), lower = better → invert
    Sum the three ranks; pick top N.
    """
    px = prices
    mom = px.shift(21) / px.shift(252) - 1.0
    daily = px.pct_change(fill_method=None)
    vol = daily.rolling(vol_lookback).std() * np.sqrt(252)
    rev = px.pct_change(21, fill_method=None)
    dates = px.index
    w_list = []
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < 260:
            continue
        m = mom.loc[dt].dropna()
        v = vol.loc[dt].loc[m.index].dropna()
        r = rev.loc[dt].loc[m.index].dropna()
        common = m.index.intersection(v.index).intersection(r.index)
        if len(common) < n_long:
            continue
        m = m.loc[common]; v = v.loc[common]; r = r.loc[common]
        # ranks: higher mom good, lower vol good, lower rev (mean-reversion) good
        z_m = (m.rank() - 0.5 * len(m)) / len(m)
        z_v = -(v.rank() - 0.5 * len(v)) / len(v)
        z_r = -(r.rank() - 0.5 * len(r)) / len(r)
        composite = z_m + z_v + z_r
        top = composite.nlargest(n_long).index
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[top] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def risk_parity_etf(
    macro: pd.DataFrame,
    tickers: tuple = ("SPY", "TLT", "GLD", "HYG", "IWM"),
    vol_lookback: int = 60,
    target_vol: float = 0.12,
    rebal_freq: int = 21,
    vol_est: str = "diag_legacy",
) -> pd.DataFrame:
    """Equal-risk-contribution across a fixed multi-asset basket. Approximated
    by inverse-vol weights then scaled to target_vol. Classic Bridgewater
    all-weather idea without the leverage.

    The default vol_est="diag_legacy" assumes zero cross-correlation, so the
    scale never binds in practice (a no-op, retained only to reproduce the
    published record). New research should pass vol_est="cov".
    """
    available = [t for t in tickers if t in macro.columns]
    if not available:
        return pd.DataFrame(index=macro.index)
    px = macro[available]
    daily = px.pct_change(fill_method=None)
    vol = daily.rolling(vol_lookback).std() * np.sqrt(252)
    dates = px.index
    w_list = []
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < vol_lookback + 5:
            continue
        v = vol.loc[dt].replace(0, np.nan).dropna()
        if v.empty:
            continue
        inv = 1.0 / v
        wnorm = inv / inv.sum()
        est_vol = _est_portfolio_vol(wnorm, v, daily, dt, vol_lookback, vol_est)
        scale = min(1.0, target_vol / max(est_vol, 1e-6))
        wrow = pd.Series(0.0, index=macro.columns)
        wrow.loc[wnorm.index] = wnorm.values * scale
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=macro.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def vix_gated_trend(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    sma_fast: int = 50,
    sma_slow: int = 200,
    breakout_lookback: int = 100,
    n_long: int = 30,
    rebal_freq: int = 21,
    vix_threshold: float = 28.0,
) -> pd.DataFrame:
    """trend_following but with an explicit VIX gate. When VIX closes above
    the threshold on the rebalance day, go to cash. Otherwise run the
    standard trend filter (above 200-SMA, near 100-day high).
    """
    px = prices
    sma50 = px.rolling(sma_fast).mean()
    sma200 = px.rolling(sma_slow).mean()
    high_n = px.rolling(breakout_lookback).max()
    is_up = (sma50 > sma200) & (px > sma200)
    near_high = px >= high_n * 0.98
    rs = (sma50 / sma200 - 1.0)
    vix = macro.get("VIX")
    dates = px.index
    w_list = []
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < sma_slow + 10:
            continue
        vix_now = float(vix.loc[dt]) if vix is not None and dt in vix.index else 15.0
        wrow = pd.Series(0.0, index=px.columns)
        if vix_now < vix_threshold:
            eligible = is_up.loc[dt] & near_high.loc[dt]
            scores = rs.loc[dt][eligible].dropna()
            if len(scores) > 0:
                top = scores.nlargest(min(n_long, len(scores))).index
                wrow.loc[top] = 1.0 / len(top)
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def calendar_tom(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    n_long: int = 30,
) -> pd.DataFrame:
    """Turn-of-month + first-week-of-month seasonality. Long top-N by 63d
    momentum on the last 3 trading days of each month and the first 3 of
    the next; cash otherwise. Captures the well-documented TOM effect.
    """
    px = prices
    mom = px.pct_change(63, fill_method=None)
    dates = px.index
    df_dt = pd.DataFrame(index=dates)
    df_dt["month"] = df_dt.index.month
    df_dt["is_last3"] = False
    df_dt["is_first3"] = False
    # Mark last 3 and first 3 of each month
    for mkey, group in df_dt.groupby([df_dt.index.year, df_dt["month"]]):
        idx = group.index
        last_three = idx[-3:]
        first_three = idx[:3]
        df_dt.loc[last_three, "is_last3"] = True
        df_dt.loc[first_three, "is_first3"] = True
    in_window = df_dt["is_last3"] | df_dt["is_first3"]
    w_list = []
    for i, dt in enumerate(dates):
        if i < 65:
            continue
        wrow = pd.Series(0.0, index=px.columns)
        if in_window.iloc[i]:
            row = mom.loc[dt].dropna()
            if len(row) >= n_long:
                top = row.nlargest(n_long).index
                wrow.loc[top] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def sector_momentum_rotation(
    macro: pd.DataFrame,
    n_sectors: int = 3,
    lookback: int = 126,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Pick top-N sector ETFs (XL* family) by 6-month return; equal-weight.
    Uses macro_data so the strategy is sector-direct, not stock-pick.
    Simpler than the existing sector_rotation but with a momentum filter.
    """
    sector_etfs_local = ["XLB","XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY"]
    sectors = [s for s in sector_etfs_local if s in macro.columns]
    if not sectors:
        return pd.DataFrame(index=macro.index)
    px = macro[sectors]
    mom = px.pct_change(lookback, fill_method=None)
    dates = px.index
    w_list = []
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < lookback + 5:
            continue
        row = mom.loc[dt].dropna()
        if row.empty:
            continue
        # Only long if positive momentum
        row = row[row > 0]
        if row.empty:
            wrow = pd.Series(0.0, index=macro.columns)
        else:
            k = min(n_sectors, len(row))
            top = row.nlargest(k).index
            wrow = pd.Series(0.0, index=macro.columns)
            wrow.loc[top] = 1.0 / k
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=macro.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def adaptive_voltarget_momentum(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    n_long: int = 30,
    vol_lookback: int = 60,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """xs_momentum_top30 but with a leverage that adapts to VIX percentile.
    When VIX is in its bottom-tercile (calm), target 1.5x leverage; when
    in top-tercile (stressed), target 0.5x; middle = 1.0x. Self-adjusting
    risk-on / risk-off without hard cash flips.
    """
    SECTOR_ETFS_LOCAL = []  # not used here but keeps namespace clean
    px = prices
    mom = px.shift(21) / px.shift(252) - 1.0
    daily = px.pct_change(fill_method=None)
    vol = daily.rolling(vol_lookback).std() * np.sqrt(252)
    vix = macro.get("VIX")
    if vix is None:
        # Without VIX, fall back to constant 1.0x
        vix = pd.Series(15.0, index=px.index)
    vix_252 = vix.rolling(252).rank(pct=True)
    dates = px.index
    w_list = []
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < 260:
            continue
        row = mom.loc[dt].dropna()
        if len(row) < n_long:
            continue
        top = row.nlargest(n_long).index
        v = vol.loc[dt][top].replace(0, np.nan).dropna()
        if v.empty:
            continue
        inv = 1.0 / v
        wnorm = inv / inv.sum()
        # Adaptive leverage
        pctile = float(vix_252.loc[dt]) if dt in vix_252.index and pd.notna(vix_252.loc[dt]) else 0.5
        if pctile < 0.33:
            lev = 1.5
        elif pctile > 0.67:
            lev = 0.5
        else:
            lev = 1.0
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[wnorm.index] = wnorm.values * lev
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


def momentum_quality_blend(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    n_long: int = 30,
    vol_lookback: int = 60,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Momentum + low-vol intersection. Top 100 by 12-1 momentum; among those,
    pick the 30 with the lowest 60d volatility. Combines two academically
    validated factors that are negatively correlated in their tracking error.
    """
    px = prices
    mom = px.shift(21) / px.shift(252) - 1.0
    daily = px.pct_change(fill_method=None)
    vol = daily.rolling(vol_lookback).std() * np.sqrt(252)
    dates = px.index
    w_list = []
    for i, dt in enumerate(dates):
        if i % rebal_freq != 0 or i < 260:
            continue
        m = mom.loc[dt].dropna()
        if len(m) < 100:
            continue
        top100 = m.nlargest(100).index
        v = vol.loc[dt][top100].dropna()
        if len(v) < n_long:
            continue
        bottom30 = v.nsmallest(n_long).index
        # Equal-weight inside the chosen 30
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[bottom30] = 1.0 / n_long
        w_list.append(wrow.rename(dt))
    if not w_list:
        return pd.DataFrame(index=dates, columns=px.columns).fillna(0.0)
    return pd.concat(w_list, axis=1).T


# ════════════════════════════════════════════════════════════════════════
# v6 — Two-channel "bad regime" freeze (added 2026-06-09)
#
# Returns boolean Series indexed by date (True = bot should be frozen).
# Both channels use sticky persistence to avoid whipsaw — that's the part
# the old composite-score gate (regime_gated_momentum) got wrong.
# ════════════════════════════════════════════════════════════════════════


def freeze_signal_vix_spike(
    macro: pd.DataFrame,
    vix_col: str = "VIX",
    spike_level: float = 25.0,
    spike_pctile: float = 0.95,
    pctile_lookback: int = 252,
    unfreeze_level: float = 20.0,
    unfreeze_pctile: float = 0.60,
    unfreeze_consecutive_days: int = 3,
    min_freeze_days: int = 10,
) -> pd.Series:
    """Channel A — VIX-spike freeze with sticky persistence.

    Trigger: VIX > spike_level AND VIX > spike_pctile of trailing-252d VIX.
    Stay frozen ≥ min_freeze_days. Unfreeze only when VIX < unfreeze_level
    AND VIX < unfreeze_pctile pctile for unfreeze_consecutive_days in a row.
    Returns a boolean Series (True = frozen). Index = macro.index.
    """
    if vix_col not in macro.columns:
        return pd.Series(False, index=macro.index)
    vix = macro[vix_col].ffill()

    # Rolling pctile rank
    roll_pctile = vix.rolling(pctile_lookback, min_periods=60).rank(pct=True)

    trigger = (vix > spike_level) & (roll_pctile > spike_pctile)
    unfreeze_ok = (vix < unfreeze_level) & (roll_pctile < unfreeze_pctile)

    out = pd.Series(False, index=macro.index)
    frozen = False
    frozen_at = None
    unfreeze_streak = 0
    for i, dt in enumerate(macro.index):
        if not frozen:
            if bool(trigger.iloc[i]) if i < len(trigger) and pd.notna(trigger.iloc[i]) else False:
                frozen = True
                frozen_at = i
                unfreeze_streak = 0
        else:
            days_frozen = i - (frozen_at if frozen_at is not None else 0)
            ok_today = bool(unfreeze_ok.iloc[i]) if i < len(unfreeze_ok) and pd.notna(unfreeze_ok.iloc[i]) else False
            if ok_today:
                unfreeze_streak += 1
            else:
                unfreeze_streak = 0
            if days_frozen >= min_freeze_days and unfreeze_streak >= unfreeze_consecutive_days:
                frozen = False
                frozen_at = None
                unfreeze_streak = 0
        out.iloc[i] = frozen
    return out


def freeze_signal_spy_drawdown(
    macro: pd.DataFrame,
    spy_col: str = "SPY",
    peak_lookback: int = 30,
    freeze_dd_pct: float = 0.12,
    unfreeze_within_pct: float = 0.08,
    min_freeze_days: int = 5,
) -> pd.Series:
    """Channel B (backtest proxy) — SPY drawdown ≥ freeze_dd_pct from
    trailing peak_lookback peak.

    In live, Channel B uses the bot's OWN equity (precisely measures
    strategy-specific failure). In backtest, we use SPY drawdown as a
    proxy. Approximate but tractable: the strategy correlates ~0.6-0.8
    with SPY, so SPY DD episodes overlap with bot-DD episodes most of
    the time. The proxy will MISS pure factor reversals (2022 momentum
    in a flat market) — Channel A or a finer proxy might catch those.
    """
    if spy_col not in macro.columns:
        return pd.Series(False, index=macro.index)
    spy = macro[spy_col].ffill()
    roll_peak = spy.rolling(peak_lookback, min_periods=5).max()
    dd = spy / roll_peak - 1.0  # ≤ 0

    trigger = dd <= -freeze_dd_pct
    unfreeze_ok = dd >= -unfreeze_within_pct

    out = pd.Series(False, index=macro.index)
    frozen = False
    frozen_at = None
    for i, dt in enumerate(macro.index):
        if not frozen:
            if bool(trigger.iloc[i]) if i < len(trigger) and pd.notna(trigger.iloc[i]) else False:
                frozen = True
                frozen_at = i
        else:
            days_frozen = i - (frozen_at if frozen_at is not None else 0)
            ok_today = bool(unfreeze_ok.iloc[i]) if i < len(unfreeze_ok) and pd.notna(unfreeze_ok.iloc[i]) else False
            if days_frozen >= min_freeze_days and ok_today:
                frozen = False
                frozen_at = None
        out.iloc[i] = frozen
    return out


def combo_v2_with_freeze(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    use_vix_channel: bool = True,
    use_dd_channel: bool = True,
    return_freeze_series: bool = False,
    daily_freeze: bool = True,
) -> pd.DataFrame:
    """combo_v2 3-sleeve blend (xs_momentum + dual_momentum_voltarget +
    adaptive_voltarget_momentum) with the v6 freeze applied on top.

    Blend warm-up caveat (known property of the published record): the blend
    always divides by 3, but the three sleeves start emitting weights at
    different dates, so for roughly the first ~6 months the combo runs at
    reduced exposure (1/3 or 2/3 of full) until all sleeves are live.

    Freeze granularity: `daily_freeze=True` (default, corrected) expands the
    blend onto the freeze signal's daily calendar (reindex+ffill), zeroes the
    frozen days, and returns a DAILY weights frame — this matches how the
    published V6 numbers were produced (run_v6.py builds the base blend itself
    and applies its own daily `apply_freeze`; it never called this function).
    `daily_freeze=False` reproduces this function's old buggy behavior — the
    freeze was applied only on the sparse monthly decision index, so the stale
    book was held through frozen days between decisions (the exact bug
    REPORT.md V6 describes). That sparse path produced NO published number;
    it is kept only for forensic comparison.

    Callers passing the daily frame to engine_v2.run_backtest_v2 should set
    weights_are_daily=True (backtest.run_backtest handles either shape).

    Returns weights DataFrame. When `return_freeze_series=True`, also
    returns the boolean Series of frozen dates (same index as the returned
    frame) as the second tuple element.
    """
    # Sleeves
    w_xs = xs_momentum(prices, macro, n_long=30)
    w_dual = dual_momentum_voltarget(prices, macro, n_long=30)
    w_adapt = adaptive_voltarget_momentum(prices, macro, n_long=30)

    # Align and average
    dates = sorted(set(w_xs.index) | set(w_dual.index) | set(w_adapt.index))
    cols = sorted(set(w_xs.columns) | set(w_dual.columns) | set(w_adapt.columns))
    w_xs = w_xs.reindex(index=dates, columns=cols, fill_value=0.0)
    w_dual = w_dual.reindex(index=dates, columns=cols, fill_value=0.0)
    w_adapt = w_adapt.reindex(index=dates, columns=cols, fill_value=0.0)
    blended = (w_xs + w_dual + w_adapt) / 3.0

    # Freeze channels live on macro's daily calendar.
    channel_freeze = pd.Series(False, index=macro.index)
    if use_vix_channel:
        channel_freeze = channel_freeze | freeze_signal_vix_spike(macro)
    if use_dd_channel:
        channel_freeze = channel_freeze | freeze_signal_spy_drawdown(macro)

    if daily_freeze:
        # Expand the sparse decision-date blend to the daily calendar, then
        # zero the frozen days — the live bot skips its rebalance cron on
        # every frozen day, which is what a daily zero simulates.
        all_dates = sorted(set(blended.index) | set(channel_freeze.index))
        blended = blended.reindex(all_dates).ffill().fillna(0.0)
        freeze_total = channel_freeze.reindex(all_dates).fillna(False).astype(bool)
        if freeze_total.any():
            blended.loc[freeze_total[freeze_total].index] = 0.0
    else:
        # Legacy sparse bug path: zeroes weights only on the monthly decision
        # index; the ffill in the backtester holds the stale book through
        # frozen days between decisions.
        freeze_total = (
            channel_freeze.reindex(blended.index, method="ffill").fillna(False).astype(bool)
        )
        if freeze_total.any():
            blended.loc[freeze_total[freeze_total].index] = 0.0

    if return_freeze_series:
        return blended, freeze_total
    return blended
