"""Live-twin parity: bot residual sleeve == research residual_momentum.

The bot's `core.combo_strategy._residual_momentum_weights` (Stream-1 port,
2026-07-18) must match the research implementation in
`Traiding 11/strategies_v2.py` — both the stateless live twin
(`_residual_momentum_weights_live`) and the sparse backtest frame builder
(`residual_momentum`) — to 1e-9 on seeded decision dates, in the B-base
config (beta_mode="market_sector", scale_mode="raw", n_long=30) and in the
other spec modes. Seeding follows the E3 convention
(np.random.default_rng(7), exp_e3_residual.parity_check).

DATA-PATH CAVEAT (documented per plan risk #4): the research side prices
come from `Traiding 11/data_cache.pkl` (yfinance auto-adjusted closes,
snapshotted 2026-05-20). The live bot fetches yfinance closes with
`auto_adjust=True` (core/data.py) — the SAME adjustment convention — but
adjusted closes for PAST dates get rescaled whenever a new dividend/split
posts, so a fresh fetch can differ from the snapshot by small factors.
`test_data_path_adjustment_convention` verifies the conventions agree
within a dividend-drift tolerance when the network is available (and skips
offline). The 1e-9 numeric parity is therefore asserted on the RESEARCH
frame passed through the bot's function: it proves the CODE is identical;
the convention test covers the data path.

Run:  /opt/anaconda3/bin/python -m pytest tests/test_residual_parity.py -v
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

RESEARCH = Path("/Users/arsenkhanguieldyan/Documents/Trading/Traiding 11")
CACHE = RESEARCH / "data_cache.pkl"

TOL = 1e-9
N_LONG = 30
GRID_WARMUP = 273   # lookback_long + lookback_skip (research grid)
REBAL = 21

pytestmark = pytest.mark.skipif(
    not CACHE.exists(), reason=f"research data_cache.pkl not found at {CACHE}"
)

if str(RESEARCH) not in sys.path:
    sys.path.append(str(RESEARCH))  # append (not insert): bot modules win

from core.combo_strategy import (  # noqa: E402
    ComboConfig, ComboStrategy, _residual_momentum_weights,
)

_cache: dict | None = None


def load_research():
    """Cached (px, macro, sector_map) from the research data cache."""
    global _cache
    if _cache is None:
        with open(CACHE, "rb") as fh:
            c = pickle.load(fh)
        _cache = {"px": c["panel"]["close"], "macro": c["macro"],
                  "sector_map": c["sector_map"]}
    return _cache["px"], _cache["macro"], _cache["sector_map"]


def _seeded_grid_positions(n_rows: int, k: int, seed: int = 7) -> list[int]:
    """k seeded positions from the research decision grid
    (i % 21 == 0 and i >= 273) — E3 seeding convention (rng seed 7)."""
    grid = [i for i in range(n_rows) if i % REBAL == 0 and i >= GRID_WARMUP]
    rng = np.random.default_rng(seed)
    return sorted(int(grid[j]) for j in rng.choice(len(grid), size=k, replace=False))


def _as_series(weights: dict, columns) -> pd.Series:
    row = pd.Series(0.0, index=columns)
    for s, w in weights.items():
        row.loc[s] = w
    return row


# ─── 1. bot vs research live twin, B-base config, seeded dates ───────────

def test_parity_live_twin_bbase():
    import strategies_v2 as sv2

    px, macro, sector_map = load_research()
    for i in _seeded_grid_positions(len(px), k=4):
        pxi, mci = px.iloc[: i + 1], macro.iloc[: i + 1]
        want = sv2._residual_momentum_weights_live(
            pxi, mci, sector_map,
            n_long=N_LONG, beta_mode="market_sector", scale_mode="raw",
        )
        got = _residual_momentum_weights(
            pxi, mci, sector_map,
            n_long=N_LONG, beta_mode="market_sector", scale_mode="raw",
        )
        d = px.index[i].date()
        assert len(want) == N_LONG, f"research twin empty at {d} — bad fixture"
        assert set(got) == set(want), (
            f"pick mismatch at {d}: only_bot={sorted(set(got) - set(want))} "
            f"only_research={sorted(set(want) - set(got))}"
        )
        diff = float((_as_series(got, px.columns)
                      - _as_series(want, px.columns)).abs().max())
        assert diff <= TOL, f"PARITY FAIL @ {d}: max|diff|={diff:.2e} > {TOL}"


# ─── 2. bot vs research sparse frame builder (residual_momentum) ─────────

def test_parity_backtest_frame_bbase():
    """The bot live twin must reproduce rows of the research
    `residual_momentum` decision frame (the function the backtests use),
    at 1e-9, on seeded rows of a 600-session slice."""
    import strategies_v2 as sv2

    px, macro, sector_map = load_research()
    a = px.index.get_indexer([pd.Timestamp("2019-01-02")], method="bfill")[0]
    pxs, mcs = px.iloc[a: a + 600], macro.iloc[a: a + 600]

    frame = sv2.residual_momentum(
        pxs, mcs, sector_map, n_long=N_LONG,
        beta_mode="market_sector", scale_mode="raw",
    )
    live_rows = frame.index[frame.abs().sum(axis=1) > 0]
    assert len(live_rows) >= 10, "slice produced too few decision rows"

    rng = np.random.default_rng(7)
    for k in rng.choice(len(live_rows), size=3, replace=False):
        d = live_rows[int(k)]
        i = pxs.index.get_loc(d)
        got = _residual_momentum_weights(
            pxs.iloc[: i + 1], mcs.iloc[: i + 1], sector_map,
            n_long=N_LONG, beta_mode="market_sector", scale_mode="raw",
        )
        diff = float((_as_series(got, pxs.columns) - frame.loc[d]).abs().max())
        assert diff <= TOL, (
            f"FRAME PARITY FAIL @ {d.date()}: max|diff|={diff:.2e} > {TOL}"
        )


# ─── 3. other spec modes stay in parity too ──────────────────────────────

@pytest.mark.parametrize("beta_mode,scale_mode", [
    ("market", "raw"),
    ("market_sector", "sharpe"),
])
def test_parity_other_modes(beta_mode, scale_mode):
    import strategies_v2 as sv2

    px, macro, sector_map = load_research()
    i = _seeded_grid_positions(len(px), k=1, seed=11)[0]
    pxi, mci = px.iloc[: i + 1], macro.iloc[: i + 1]
    want = sv2._residual_momentum_weights_live(
        pxi, mci, sector_map,
        n_long=N_LONG, beta_mode=beta_mode, scale_mode=scale_mode,
    )
    got = _residual_momentum_weights(
        pxi, mci, sector_map,
        n_long=N_LONG, beta_mode=beta_mode, scale_mode=scale_mode,
    )
    assert set(got) == set(want)
    diff = float((_as_series(got, px.columns)
                  - _as_series(want, px.columns)).abs().max())
    assert diff <= TOL, f"{beta_mode}/{scale_mode}: max|diff|={diff:.2e}"


# ─── 4. compute_weights integration: B-base blend wiring ─────────────────

def test_compute_weights_bbase_blend():
    """B base (candidate X): residual REPLACES xs at static 1/3. The
    residual sleeve's pre-gate contribution inside compute_weights must be
    (1/3) x the research live twin's weights; xs must be off."""
    import strategies_v2 as sv2

    px, macro, sector_map = load_research()
    i = _seeded_grid_positions(len(px), k=1, seed=13)[0]
    # Bot-format dicts from the research slice (close-only frames).
    window = px.iloc[max(0, i - 300): i + 1]
    stock_data = {sym: pd.DataFrame({"close": window[sym]})
                  for sym in window.columns[:400]}
    mwin = macro.iloc[max(0, i - 300): i + 1]
    macro_data = {sym: pd.DataFrame({"close": mwin[sym]}) for sym in mwin.columns}

    cfg = ComboConfig(
        xs_mom_weight=0.0, residual_weight=1.0 / 3.0,
        dual_mom_weight=1.0 / 3.0, adaptive_weight=1.0 / 3.0,
        residual_beta_mode="market_sector", residual_scale_mode="raw",
        residual_top_n=N_LONG, residual_sector_map=sector_map,
    )
    strat = ComboStrategy(cfg)
    weights = strat.compute_weights(stock_data, macro_data)
    assert weights, "B-base blend produced an empty book"

    diag = strat.last_diagnostics
    assert "residual_momentum" in diag["sleeves"], diag["sleeves"].keys()
    assert diag["sleeves"]["xs_momentum"] == {}, "xs sleeve should be off"

    sub_px = pd.DataFrame({s: stock_data[s]["close"] for s in stock_data})
    want = sv2._residual_momentum_weights_live(
        sub_px, mwin, sector_map,
        n_long=N_LONG, beta_mode="market_sector", scale_mode="raw",
    )
    contrib = diag["sleeves"]["residual_momentum"]
    assert set(contrib) == set(want)
    for sym, w in want.items():
        # contribs are journal-rounded to 6dp — compare at that precision
        assert abs(contrib[sym] - w / 3.0) <= 5e-7, (sym, contrib[sym], w / 3.0)


# ─── 5. back-compat: pre-port pickles lack the residual fields ───────────

def test_old_pickle_attr_fallback():
    """Simulate a ComboConfig unpickled from a pre-2026-07-18 bundle:
    instance __dict__ has only the old fields. Every residual access path
    must fall back to the class-level dataclass default and reproduce
    no-residual behavior (compute_weights runs, no residual sleeve)."""
    cfg = ComboConfig()
    for f in ("residual_weight", "residual_top_n", "residual_beta_mode",
              "residual_scale_mode", "residual_lookback_long",
              "residual_lookback_skip", "residual_sector_map"):
        del cfg.__dict__[f]  # pickle restores __dict__; old ones lack these
        assert not f in cfg.__dict__
    assert cfg.residual_weight == 0.0  # class-attribute fallback

    px, macro, _ = load_research()
    i = _seeded_grid_positions(len(px), k=1, seed=17)[0]
    window = px.iloc[i - 300: i + 1]
    stock_data = {sym: pd.DataFrame({"close": window[sym]})
                  for sym in window.columns[:200]}
    mwin = macro.iloc[i - 300: i + 1]
    macro_data = {sym: pd.DataFrame({"close": mwin[sym]}) for sym in mwin.columns}

    strat = ComboStrategy(cfg)
    weights = strat.compute_weights(stock_data, macro_data)
    assert weights, "old-config path should still trade"
    assert "residual_momentum" not in strat.last_diagnostics["sleeves"]


# ─── 6. data-path adjustment convention (network-dependent) ──────────────

def test_data_path_adjustment_convention():
    """Both paths use yfinance AUTO-ADJUSTED closes (research data_cache
    snapshot vs the bot's live auto_adjust=True fetch). Verify on a sample
    ticker that overlapping closes agree within a dividend-rescale drift
    tolerance (2%): distributions posted after the 2026-05-20 snapshot
    rescale PAST adjusted closes by their (small) dividend factors, so
    exact equality is not expected. Skips offline."""
    px, _, _ = load_research()
    sym = "AAPL" if "AAPL" in px.columns else px.columns[0]
    try:
        import yfinance as yf
        fresh = yf.download(sym, period="2y", auto_adjust=True,
                            progress=False, timeout=20)
    except Exception as exc:  # offline / rate-limited → documented skip
        pytest.skip(f"yfinance unavailable ({exc}); adjustment-convention "
                    "check needs network")
    if fresh is None or len(fresh) == 0:
        pytest.skip("yfinance returned no data; likely offline")
    close = fresh["Close"]
    if isinstance(close, pd.DataFrame):  # MultiIndex layout
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    cached = px[sym].dropna()
    common = cached.index.intersection(close.index)
    assert len(common) > 100, "insufficient overlap between cache and fetch"
    rel = (close.loc[common] / cached.loc[common] - 1.0).abs()
    med = float(rel.median())
    assert med < 0.02, (
        f"{sym}: median |rel diff| {med:.4%} — adjustment conventions of "
        "the bot data path and the research cache disagree materially; "
        "residual-sleeve parity holds on code, but the DATA paths diverge "
        "beyond dividend drift and must be investigated"
    )
