"""Tradeable universe — S&P 500 + Nasdaq 100 + Russell 1000.

Scrapes Wikipedia (which is the cheapest reliable source for the
constituent lists), caches the result on the Railway persistent volume,
and falls back to the cache if every scrape fails. The cache is the
*entire* reason the bot can still run on a day Wikipedia is being
slow/blocked.

Symbols are normalized to Alpaca's dash form (BRK.B → BRK-B). Tickers
in `EXCLUDED_SYMBOLS` are blacklisted from the final set as a safety net
(very illiquid / known data-quality issues).
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.run_report import RunReport


_DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
SYMBOL_CACHE_PATH = _DATA_DIR / "universe_cache.json"


# Symbols excluded from trading. Re-evaluate annually — names that have
# accumulated 4+ years of post-IPO history can be removed. Kept here as
# a safety net for the pipeline; not authoritative.
EXCLUDED_SYMBOLS: set[str] = {
    "VFS",   # very illiquid, halted often
    "SMCI",  # data quality / restated financials
}


def _wiki_read_html(url: str, logger, max_attempts: int = 3) -> list:
    """Read HTML tables from Wikipedia with retry + exponential backoff."""
    import requests as _req
    headers = {
        "User-Agent": "MLTradingBot/1.0 (educational paper trading project)",
    }
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _req.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            return pd.read_html(io.StringIO(resp.text))
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    f"  scrape attempt {attempt}/{max_attempts} failed for {url}: "
                    f"{e} — retrying in {wait}s"
                )
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


def _load_symbol_cache(logger) -> list[str]:
    """Load cached symbol list from the last successful scrape."""
    if SYMBOL_CACHE_PATH.exists():
        try:
            cache = json.loads(SYMBOL_CACHE_PATH.read_text())
            syms = cache.get("symbols", [])
            cached_at = cache.get("cached_at", "unknown")
            try:
                cached_dt = datetime.fromisoformat(cached_at)
                age_days = (datetime.now() - cached_dt).days
                logger.info(
                    f"  Loaded {len(syms)} cached symbols "
                    f"(from {cached_at}, {age_days}d old)"
                )
            except Exception:
                logger.info(f"  Loaded {len(syms)} cached symbols (from {cached_at})")
            return syms
        except Exception as e:
            logger.warning(f"  Cache load failed: {e}")
    return []


def _save_symbol_cache(symbols: list[str], logger) -> None:
    """Save the symbol list to cache for future scrape-failure fallback."""
    try:
        SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYMBOL_CACHE_PATH.write_text(json.dumps({
            "symbols": symbols,
            "cached_at": datetime.now().isoformat(),
            "count": len(symbols),
        }, indent=2))
        logger.info(f"  Saved {len(symbols)} symbols to cache")
    except Exception as e:
        logger.warning(f"  Cache save failed: {e}")


# A healthy full scrape yields 500 (S&P) + ~1000 (Russell) ≈ 1000+ unique
# names. If the union comes back below this, at least one source silently
# returned garbage (renamed table, empty parse) — trade on the cache, and
# do NOT overwrite a good cache with the degraded set.
MIN_HEALTHY_UNIVERSE = 400

# Sentinel large-caps that must appear in any sane US large/mid-cap scrape.
# Their absence means the parse returned the wrong table entirely.
_UNIVERSE_SENTINELS = ("AAPL", "MSFT", "NVDA")


def _clean_symbols(series) -> list[str]:
    """Extract a clean ticker list from a scraped pandas column.

    Guards the exact failure that took the live bot down: a blank/footnote
    cell in a Wikipedia table becomes a float NaN, which then poisons the
    downstream ``sorted(set(...))`` with a ``float < str`` TypeError. We drop
    NaN, coerce to str, strip whitespace, normalise to Alpaca's dash form,
    and discard anything that is empty or obviously not a ticker.
    """
    out: list[str] = []
    for raw in series.dropna().astype(str).tolist():
        sym = raw.strip().replace(".", "-")
        # Real tickers are short alphanumerics (plus dash); this also drops
        # stray header rows, footnote markers, and "nan"/"None" leftovers.
        if sym and sym.upper() not in ("NAN", "NONE") and len(sym) <= 8:
            out.append(sym)
    return out


def get_tradeable_symbols(logger, report: "RunReport") -> list[str]:
    """Get S&P 500 + Nasdaq 100 + Russell 1000 symbols from Wikipedia.

    Uses proper User-Agent to avoid 403 blocks. Falls back to cached
    symbol list if all scrapes fail.
    """
    report.start_step("get_universe")
    sp500: list[str] = []
    ndx_syms: list[str] = []
    russell_syms: list[str] = []

    try:
        tables = _wiki_read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", logger
        )
        sp500 = _clean_symbols(tables[0]["Symbol"])
        logger.info(f"  S&P 500: {len(sp500)} symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"  S&P 500 scrape failed: {e}")
        report.add_warning(f"S&P 500 scrape failed: {e}")

    try:
        ndx = _wiki_read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100", logger
        )
        for table in ndx:
            if "Ticker" in table.columns:
                ndx_syms = _clean_symbols(table["Ticker"])
                break
            elif "Symbol" in table.columns:
                ndx_syms = _clean_symbols(table["Symbol"])
                break
        if not ndx_syms:
            logger.warning(
                "  Nasdaq 100: no Ticker/Symbol column matched — page layout "
                "may have changed (non-critical; S&P 500 + Russell 1000 cover it)"
            )
            report.add_warning("Nasdaq 100 scrape returned 0 symbols")
        logger.info(f"  Nasdaq 100: {len(ndx_syms)} symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"  Nasdaq 100 scrape failed: {e}")
        report.add_warning(f"Nasdaq 100 scrape failed: {e}")

    # Russell 1000 — adds ~400-500 mid-cap stocks not in S&P 500
    try:
        r1k_tables = _wiki_read_html(
            "https://en.wikipedia.org/wiki/Russell_1000_Index", logger
        )
        for table in r1k_tables:
            if "Ticker" in table.columns:
                russell_syms = _clean_symbols(table["Ticker"])
                break
            elif "Symbol" in table.columns:
                russell_syms = _clean_symbols(table["Symbol"])
                break
        logger.info(f"  Russell 1000: {len(russell_syms)} symbols from Wikipedia")
    except Exception as e:
        logger.warning(f"  Russell 1000 scrape failed (non-critical): {e}")
        report.add_warning(f"Russell 1000 scrape failed: {e}")

    # Belt-and-suspenders: even after _clean_symbols, only keep real str
    # tickers before sorting so a stray non-string can never re-introduce the
    # `float < str` crash that this whole path is guarding against.
    merged = {s for s in (sp500 + ndx_syms + russell_syms)
              if isinstance(s, str) and s}
    all_syms = sorted(merged - EXCLUDED_SYMBOLS)

    # Decide whether this scrape is trustworthy. A healthy union has 400+
    # names and contains the mega-cap sentinels. A degraded scrape (one
    # source silently returned an empty/wrong table) must NOT overwrite a
    # good cache — we'd rather trade yesterday's known-good universe.
    has_sentinels = any(s in merged for s in _UNIVERSE_SENTINELS)
    healthy = len(all_syms) >= MIN_HEALTHY_UNIVERSE and has_sentinels

    if healthy:
        _save_symbol_cache(all_syms, logger)
    else:
        cached = _load_symbol_cache(logger)
        if cached:
            logger.warning(
                f"  Scrape degraded ({len(all_syms)} names, sentinels "
                f"present={has_sentinels}) — falling back to cached universe "
                f"({len(cached)} names); NOT overwriting cache"
            )
            report.add_warning(
                f"Degraded universe scrape ({len(all_syms)} names); used cache"
            )
            all_syms = cached
        elif all_syms:
            logger.warning(
                f"  Scrape degraded ({len(all_syms)} names) and no cache "
                f"available — proceeding with the reduced universe"
            )
            report.add_warning(
                f"Degraded universe scrape ({len(all_syms)} names), no cache"
            )
        else:
            logger.error("  All scrapes failed and no cache — universe is empty")
            report.add_error("Universe empty: all scrapes failed, no cache")

    logger.info(
        f"  Total universe: {len(all_syms)} unique symbols "
        f"(excluded {len(EXCLUDED_SYMBOLS)} blacklisted)"
    )
    report.set("universe_size", len(all_syms))
    report.end_step("get_universe")
    return all_syms
