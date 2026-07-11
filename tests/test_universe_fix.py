"""Regression tests for the universe-scrape crash that took the live bot down.

Root cause: a blank/footnote cell in a scraped Wikipedia table became a float
NaN, which poisoned ``sorted(set(...))`` with a
``'<' not supported between instances of 'float' and 'str'`` TypeError — so
every daily rebalance aborted in the shared data phase.

Run:  python3 -m pytest tests/test_universe_fix.py -v
(any interpreter with pandas + pytest)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_universe_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import numpy as np
import pandas as pd
import pytest

import core.universe as u

logger = logging.getLogger("test")


def _report():
    """Minimal RunReport stand-in capturing warnings/errors."""
    rec = {"warnings": [], "errors": []}
    return SimpleNamespace(
        start_step=lambda *a, **k: None,
        end_step=lambda *a, **k: None,
        add_warning=lambda m: rec["warnings"].append(m),
        add_error=lambda m: rec["errors"].append(m),
        set=lambda *a, **k: None,
        _rec=rec,
    )


def test_clean_symbols_drops_nan_and_normalises():
    s = pd.Series(["AAPL", "MSFT", np.nan, "BRK.B", "  GOOG ", "", "nan"])
    out = u._clean_symbols(s)
    assert out == ["AAPL", "MSFT", "BRK-B", "GOOG"]
    assert all(isinstance(x, str) for x in out)


def test_get_tradeable_symbols_survives_nan_cell(monkeypatch):
    """A NaN ticker in any source must NOT crash the sorted() union."""
    sp = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "NVDA", "BRK.B"]})
    ndx = pd.DataFrame({"Ticker": ["AAPL", "AMD"]})
    # The poisoned table: a blank cell parsed as NaN (the live failure).
    russ = pd.DataFrame({"Ticker": ["AAPL", "F", np.nan, "GM", "CAT"]})

    def fake_scrape(url, _logger, **kw):
        if "S%26P_500" in url:
            return [sp]
        if "Nasdaq-100" in url:
            return [ndx]
        if "Russell_1000" in url:
            return [russ]
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(u, "_wiki_read_html", fake_scrape)
    # Force the healthy-scrape path regardless of sentinel count in this fixture.
    monkeypatch.setattr(u, "MIN_HEALTHY_UNIVERSE", 3)

    rep = _report()
    syms = u.get_tradeable_symbols(logger, rep)  # must not raise

    assert syms == sorted(syms)
    assert all(isinstance(x, str) for x in syms)
    assert not any((isinstance(x, float) and np.isnan(x)) for x in syms)
    assert {"AAPL", "MSFT", "NVDA", "BRK-B", "F", "GM", "CAT", "AMD"} <= set(syms)


def test_degraded_scrape_does_not_poison_cache(monkeypatch, tmp_path):
    """A tiny/garbage scrape must fall back to cache, not overwrite it."""
    good = ["AAPL", "MSFT", "NVDA"] + [f"SYM{i}" for i in range(500)]
    cache_file = tmp_path / "universe_cache.json"
    monkeypatch.setattr(u, "SYMBOL_CACHE_PATH", cache_file)
    u._save_symbol_cache(good, logger)

    # Now every scrape returns almost nothing (below MIN_HEALTHY_UNIVERSE).
    empty = pd.DataFrame({"Symbol": ["AAPL"]})
    monkeypatch.setattr(u, "_wiki_read_html", lambda *a, **k: [empty])

    rep = _report()
    syms = u.get_tradeable_symbols(logger, rep)

    # Fell back to the good cache, and did NOT overwrite it with the degraded set.
    assert len(syms) == len(good)
    import json
    still_cached = json.loads(cache_file.read_text())["symbols"]
    assert len(still_cached) == len(good)
    assert any("degraded" in w.lower() for w in rep._rec["warnings"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
