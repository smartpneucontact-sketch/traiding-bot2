"""Data layer for the multi-strategy backtest.

Reads daily OHLCV parquet bars from the prior Trading bot 6 cache. Builds a
panel (date x ticker) of closing prices, returns, and volume. Macro ETFs and
sector ETFs are loaded separately for regime / sector-rotation strategies.

Known limitations / survivorship bias
-------------------------------------
The local bar cache is survivors-only. An audit found that 0 of the ~1,040
tickers stop trading before the window end, and known in-window delistings
(SIVB, FRC, TWTR, BBBY, ATVI, CERN, ...) are entirely absent: the universe
was snapshotted at cache-build time, so every name in it is a company that
survived (or was never removed from) the listing through ~2026. This is NOT
correctable from this cache.

Why it matters: momentum backtests are structurally inflated by this bias.
Names that blew up, were acquired at a discount, or delisted mid-window are
exactly the ones a long-momentum book would have held into the drawdown (or
a short book would have profited from); removing them retroactively deletes
their losses from the sample. Over a 10-year window the inflation compounds
and affects the whole strategy chain built on load_panel.

A real fix requires (a) point-in-time index/universe constituent lists and
(b) bars for delisted names including terminal delisting returns (e.g. CRSP
delisting-adjusted returns). Neither is available in the local cache, so
results built on this data layer must be read as upper bounds.

A related — and correctable — issue is the outlier filter: see the
`outlier_mode` parameter of load_panel. The published-record default
("full_window") drops a ticker from the whole window if ANY single-day
|return| > threshold occurs ANYWHERE in the window, i.e. future information
conditions the past. "point_in_time" instead truncates the series at the
first offending date, like a delisting.
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


def _apply_outlier_filter(
    df: pd.DataFrame,
    mode: str,
    threshold: float,
) -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    """Outlier handling for one ticker's bar frame.

    Returns (frame, first_event_date):
      - "full_window": (df, None) if no single-day |close-to-close return|
        exceeds `threshold`; otherwise (None, first_event_date) and the
        caller drops the ticker for the whole window (published-record
        behavior — note this conditions the past on future information).
      - "point_in_time": keeps all rows strictly BEFORE the first offending
        date and drops everything from that date onward, as a delisting
        would appear; returns the truncated frame and the event date
        (or (df, None) if clean). NOTE: the event-day return itself is
        excluded — a real delisting delivers the crash to the holder, so
        this is optimistic for genuine blow-ups and correct only for data
        errors; the two are indistinguishable from bars alone. Backtests in
        PIT mode still understate blow-up losses for held names.
    Frames without a "close" column pass through untouched.
    """
    if mode not in ("full_window", "point_in_time"):
        raise ValueError(f"unknown outlier_mode: {mode!r}")
    if "close" not in df.columns:
        return df, None
    daily = df["close"].pct_change(fill_method=None)
    bad = daily.abs() > threshold
    if not bad.any():
        return df, None
    first_bad = bad.idxmax()
    if mode == "full_window":
        return None, first_bad
    return df.loc[df.index < first_bad], first_bad


@lru_cache(maxsize=1)
def load_panel(
    fields: tuple[str, ...] = ("close", "open", "high", "low", "volume"),
    start: str = "2016-04-01",
    end: str = "2026-03-27",
    min_obs: int = 500,
    max_daily_abs_ret: float = 0.80,
    outlier_mode: str = "full_window",
) -> dict[str, pd.DataFrame]:
    """Return dict[field] -> DataFrame indexed by date, columns = ticker.

    NOTE: the universe is survivors-only regardless of parameters — see the
    "Known limitations / survivorship bias" section of the module docstring.
    Results are structurally inflated for momentum-style strategies.

    Drops tickers with:
      - fewer than `min_obs` valid closes inside the window (filters out
        symbols that started trading mid-window).
      - any single-day close-to-close return exceeding `max_daily_abs_ret`
        (likely a corporate-action discontinuity / data error). Default 80%
        rules out CHRD-style +25733% blips while leaving real meme-stock
        moves like GME +90% in place.

    outlier_mode:
      - "full_window" (default, reproduces the published record): a ticker
        with any offending return anywhere in the window is dropped entirely,
        which lets future events remove a name from the past.
      - "point_in_time": the ticker's series is instead truncated at the
        first offending date (all fields), as if it delisted there; the
        `min_obs` check is re-applied after truncation.
    """
    tickers = list_stock_tickers()
    panels: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}
    dropped_outlier = []
    truncated: list[tuple[str, str]] = []
    for t in tickers:
        df = _read_one(BARS_DIR / f"{t}.parquet")
        if df is None:
            continue
        df = df.loc[(df.index >= start) & (df.index <= end)]
        if len(df) < min_obs:
            continue
        df, event_date = _apply_outlier_filter(df, outlier_mode, max_daily_abs_ret)
        if df is None:
            dropped_outlier.append(t)
            continue
        if event_date is not None:
            truncated.append((t, event_date.date().isoformat()))
            if len(df) < min_obs:  # re-check after truncation
                continue
        for f in fields:
            if f in df.columns:
                panels[f][t] = df[f]
    if dropped_outlier:
        print(f"[load_panel] dropped {len(dropped_outlier)} tickers with |daily ret| > {max_daily_abs_ret:.0%}: "
              f"{dropped_outlier[:10]}{'...' if len(dropped_outlier)>10 else ''}")
    if truncated:
        preview = ", ".join(f"{t}@{d}" for t, d in truncated[:10])
        print(f"[load_panel] point-in-time truncated {len(truncated)} tickers at first |daily ret| > "
              f"{max_daily_abs_ret:.0%}: {preview}{'...' if len(truncated) > 10 else ''}")
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
