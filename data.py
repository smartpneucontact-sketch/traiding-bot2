"""Data layer for the multi-strategy backtest.

Reads daily OHLCV parquet bars from the prior Trading bot 6 cache. Builds a
panel (date x ticker) of closing prices, returns, and volume. Macro ETFs and
sector ETFs are loaded separately for regime / sector-rotation strategies.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BARS_DIR = Path(
    "/Users/arsenkhanguieldyan/Documents/Trading/Trading bot after 4/"
    "Trading bot 6/trading_system/data/bars_daily"
)
MACRO_DIR = BARS_DIR / "macro"
SECTOR_MAP_PATH = Path(
    "/Users/arsenkhanguieldyan/Documents/Trading/Trading bot after 4/"
    "Trading bot 6/trading_system/data/sector_map.json"
)

SECTOR_ETFS = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]


def _read_one(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


@lru_cache(maxsize=1)
def list_stock_tickers() -> tuple[str, ...]:
    files = sorted(BARS_DIR.glob("*.parquet"))
    return tuple(p.stem for p in files)


@lru_cache(maxsize=1)
def load_panel(
    fields: tuple[str, ...] = ("close", "open", "high", "low", "volume"),
    start: str = "2016-04-01",
    end: str = "2026-03-27",
    min_obs: int = 500,
    max_daily_abs_ret: float = 0.80,
) -> dict[str, pd.DataFrame]:
    """Return dict[field] -> DataFrame indexed by date, columns = ticker.

    Drops tickers with:
      - fewer than `min_obs` valid closes inside the window (filters out
        symbols that started trading mid-window).
      - any single-day close-to-close return exceeding `max_daily_abs_ret`
        (likely a corporate-action discontinuity / data error). Default 80%
        rules out CHRD-style +25733% blips while leaving real meme-stock
        moves like GME +90% in place.
    """
    tickers = list_stock_tickers()
    panels: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}
    dropped_outlier = []
    for t in tickers:
        df = _read_one(BARS_DIR / f"{t}.parquet")
        if df is None:
            continue
        df = df.loc[(df.index >= start) & (df.index <= end)]
        if len(df) < min_obs:
            continue
        if "close" in df.columns:
            daily = df["close"].pct_change(fill_method=None)
            if (daily.abs() > max_daily_abs_ret).any():
                dropped_outlier.append(t)
                continue
        for f in fields:
            if f in df.columns:
                panels[f][t] = df[f]
    if dropped_outlier:
        print(f"[load_panel] dropped {len(dropped_outlier)} tickers with |daily ret| > {max_daily_abs_ret:.0%}: "
              f"{dropped_outlier[:10]}{'...' if len(dropped_outlier)>10 else ''}")
    out = {f: pd.DataFrame(series_dict).sort_index() for f, series_dict in panels.items()}
    # Align all field DataFrames to the union of dates and intersection of tickers
    common_tickers = sorted(set.intersection(*[set(df.columns) for df in out.values()]))
    out = {f: df[common_tickers].sort_index() for f, df in out.items()}
    return out


@lru_cache(maxsize=1)
def load_macro(
    start: str = "2016-04-01",
    end: str = "2026-03-27",
) -> pd.DataFrame:
    """Return a wide DataFrame: date x macro_close_<ticker>."""
    if not MACRO_DIR.is_dir():
        return pd.DataFrame()
    out: dict[str, pd.Series] = {}
    for p in sorted(MACRO_DIR.glob("*.parquet")):
        df = _read_one(p)
        if df is None or "close" not in df.columns:
            continue
        df = df.loc[(df.index >= start) & (df.index <= end)]
        out[p.stem] = df["close"]
    return pd.DataFrame(out).sort_index()


@lru_cache(maxsize=1)
def load_sector_map() -> dict[str, str]:
    """Read sector map from prior bot's cache. Falls back to empty dict."""
    import json
    if not SECTOR_MAP_PATH.exists():
        return {}
    try:
        with open(SECTOR_MAP_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def trading_days(start: str = "2016-04-01", end: str = "2026-03-27") -> pd.DatetimeIndex:
    """Return SPY's trading days as the canonical calendar."""
    spy_path = MACRO_DIR / "SPY.parquet"
    if not spy_path.exists():
        # Fallback: union over a few core stocks
        df = _read_one(BARS_DIR / "AAPL.parquet")
    else:
        df = _read_one(spy_path)
    df = df.loc[(df.index >= start) & (df.index <= end)]
    return df.index


if __name__ == "__main__":
    panel = load_panel()
    print("close shape:", panel["close"].shape)
    print("date range:", panel["close"].index.min(), "->", panel["close"].index.max())
    print("# tickers after filter:", panel["close"].shape[1])
    macro = load_macro()
    print("macro shape:", macro.shape, "cols:", list(macro.columns)[:6], "...")
    sm = load_sector_map()
    print("sector map entries:", len(sm))
    if sm:
        print("  sample:", dict(list(sm.items())[:3]))
