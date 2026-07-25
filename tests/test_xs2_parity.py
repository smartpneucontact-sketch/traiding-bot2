"""Live-twin parity: bot second xs sleeve (xs2, top-50) == research
`strategies.xs_momentum(n_long=50)` — the catalog's `xs_momentum_12_1`.

The Stream-3 process slot's 2026 selection is
[dual_momentum_vol, xs_momentum_top30, xs_momentum_12_1].  xs_momentum_12_1
is the SAME 12-1 cross-sectional momentum as the primary xs sleeve, just
top-50 (verified: research exp_rescore.py REGISTRY entry
`("xs_momentum_12_1", lambda: _orig["xs_momentum"](px, macro, n_long=50),
..., {"n_long": 50})`).  The port (2026-07-25) therefore reuses the bot's
EXISTING `_xs_momentum_weights` at `n_long=ComboConfig.xs2_top_n` — no new
numerics — blended at `ComboConfig.xs2_weight` with diagnostics sleeve key
"xs_momentum_2" (only when active, like the residual sleeve).

Parity is asserted at 1e-9 against the research `strategies.xs_momentum`
decision frame (the function the walk-forward scoring engine consumed), on
the SAME seeded dates / panel convention as tests/test_residual_parity.py
(research grid i % 21 == 0 and i >= 273, np.random.default_rng(7)).  The
same DATA-PATH CAVEAT as the residual parity module applies: parity on the
research frame proves the CODE is identical; the adjustment-convention
test in test_residual_parity.py covers the data path.

Known warmup divergence (documented, not asserted): the bot's
`_xs_momentum_weights` warmup guard requires 252+21+5 rows while the
research decision loop starts at row 273 — the bot returns {} on the first
grid date of a fresh panel.  All seeded positions land years past warmup.

Run:  /opt/anaconda3/bin/python -m pytest tests/test_xs2_parity.py -v
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import pytest

RESEARCH = Path("/Users/arsenkhanguieldyan/Documents/Trading/Traiding 11")
CACHE = RESEARCH / "data_cache.pkl"
PROCESS_PKL = _REPO / "model" / "combo_v2_process" / "model.pkl"

TOL = 1e-9
N_LONG_XS2 = 50     # xs_momentum_12_1 = xs_momentum(n_long=50)
GRID_WARMUP = 273   # lookback_long + lookback_skip (research grid)
REBAL = 21

pytestmark = pytest.mark.skipif(
    not CACHE.exists(), reason=f"research data_cache.pkl not found at {CACHE}"
)

if str(RESEARCH) not in sys.path:
    sys.path.append(str(RESEARCH))  # append (not insert): bot modules win

from core.combo_strategy import (  # noqa: E402
    ComboConfig, ComboStrategy, _xs_momentum_weights,
)

_cache: dict | None = None


def load_research():
    """Cached (px, macro) from the research data cache (same fixture
    convention as tests/test_residual_parity.py)."""
    global _cache
    if _cache is None:
        with open(CACHE, "rb") as fh:
            c = pickle.load(fh)
        _cache = {"px": c["panel"]["close"], "macro": c["macro"]}
    return _cache["px"], _cache["macro"]


def _seeded_grid_positions(n_rows: int, k: int, seed: int = 7) -> list[int]:
    """k seeded positions from the research decision grid
    (i % 21 == 0 and i >= 273) — E3 seeding convention (rng seed 7),
    byte-identical to tests/test_residual_parity.py."""
    grid = [i for i in range(n_rows) if i % REBAL == 0 and i >= GRID_WARMUP]
    rng = np.random.default_rng(seed)
    return sorted(int(grid[j]) for j in rng.choice(len(grid), size=k, replace=False))


def _as_series(weights: dict, columns) -> pd.Series:
    row = pd.Series(0.0, index=columns)
    for s, w in weights.items():
        row.loc[s] = w
    return row


# ─── 1. bot xs sleeve @ top_n=50 vs research xs_momentum frame ───────────

def test_parity_xs2_vs_research_frame():
    """The bot's xs sleeve at n_long=50 must reproduce rows of the research
    `strategies.xs_momentum(px, macro, n_long=50)` decision frame — the
    exact object the wf scoring engine consumed for xs_momentum_12_1 — at
    1e-9 on the seeded dates."""
    import strategies as S

    px, macro = load_research()
    frame = S.xs_momentum(px, macro, n_long=N_LONG_XS2)
    for i in _seeded_grid_positions(len(px), k=4):
        d = px.index[i]
        assert d in frame.index, f"{d.date()} not a research decision row — bad fixture"
        row = frame.loc[d]
        want = row[row > 0]
        assert len(want) == N_LONG_XS2, f"research frame not top-{N_LONG_XS2} at {d.date()}"
        got = _xs_momentum_weights(px.iloc[: i + 1], n_long=N_LONG_XS2)
        assert set(got) == set(want.index), (
            f"pick mismatch at {d.date()}: "
            f"only_bot={sorted(set(got) - set(want.index))} "
            f"only_research={sorted(set(want.index) - set(got))}"
        )
        diff = float((_as_series(got, px.columns) - row).abs().max())
        assert diff <= TOL, f"PARITY FAIL @ {d.date()}: max|diff|={diff:.2e} > {TOL}"


# ─── 2. compute_weights integration: 2026 process blend wiring ───────────

def test_compute_weights_process_blend():
    """The 2026 process construction: xs (top-30) 1/3 + dual 1/3 + xs2
    (top-50) 1/3.  Each xs instance's pre-gate contribution inside
    compute_weights must be (1/3) x its own sleeve weights, published
    under distinct diagnostics keys — proving the two parameterizations
    coexist without touching each other."""
    px, macro = load_research()
    i = _seeded_grid_positions(len(px), k=1, seed=13)[0]
    window = px.iloc[max(0, i - 300): i + 1]
    stock_data = {sym: pd.DataFrame({"close": window[sym]})
                  for sym in window.columns[:400]}
    mwin = macro.iloc[max(0, i - 300): i + 1]
    macro_data = {sym: pd.DataFrame({"close": mwin[sym]}) for sym in mwin.columns}

    cfg = ComboConfig(
        xs_mom_weight=1.0 / 3.0, xs_mom_top_n=30,
        dual_mom_weight=1.0 / 3.0,
        adaptive_weight=0.0, residual_weight=0.0,
        xs2_weight=1.0 / 3.0, xs2_top_n=50,
        enable_spy_dd_gate=False, enable_book_dd_gate=False,
    )
    strat = ComboStrategy(cfg)
    weights = strat.compute_weights(stock_data, macro_data)
    assert weights, "process blend produced an empty book"

    diag = strat.last_diagnostics
    assert "xs_momentum_2" in diag["sleeves"], diag["sleeves"].keys()
    assert "residual_momentum" not in diag["sleeves"]

    sub_px = pd.DataFrame({s: stock_data[s]["close"] for s in stock_data})
    want_xs = _xs_momentum_weights(sub_px, n_long=30)
    want_xs2 = _xs_momentum_weights(sub_px, n_long=50)
    assert len(want_xs) == 30 and len(want_xs2) == 50

    for key, want in (("xs_momentum", want_xs), ("xs_momentum_2", want_xs2)):
        contrib = diag["sleeves"][key]
        assert set(contrib) == set(want), key
        for sym, w in want.items():
            # contribs are journal-rounded to 6dp — compare at that precision
            assert abs(contrib[sym] - w / 3.0) <= 5e-7, (key, sym)
    # top-30 picks are a subset of top-50 picks (same signal, same ranking)
    assert set(want_xs) <= set(want_xs2)


# ─── 3. back-compat: pre-port pickles lack the xs2 fields ────────────────

def test_old_pickle_attr_fallback_xs2():
    """Simulate a ComboConfig unpickled from a pre-2026-07-25 bundle:
    instance __dict__ lacks the xs2 fields.  Every xs2 access path must
    fall back to the class-level dataclass default (weight 0.0) and
    reproduce pre-port behavior (no xs_momentum_2 sleeve computed or
    published)."""
    cfg = ComboConfig()
    for f in ("xs2_weight", "xs2_top_n"):
        del cfg.__dict__[f]  # pickle restores __dict__; old ones lack these
        assert f not in cfg.__dict__
    assert cfg.xs2_weight == 0.0  # class-attribute fallback
    assert cfg.xs2_top_n == 50

    px, macro = load_research()
    i = _seeded_grid_positions(len(px), k=1, seed=17)[0]
    window = px.iloc[i - 300: i + 1]
    stock_data = {sym: pd.DataFrame({"close": window[sym]})
                  for sym in window.columns[:200]}
    mwin = macro.iloc[i - 300: i + 1]
    macro_data = {sym: pd.DataFrame({"close": mwin[sym]}) for sym in mwin.columns}

    strat = ComboStrategy(cfg)
    weights = strat.compute_weights(stock_data, macro_data)
    assert weights, "old-config path should still trade"
    assert "xs_momentum_2" not in strat.last_diagnostics["sleeves"]


# ─── 4. the built process bundle carries the 2026 construction ───────────

@pytest.mark.skipif(not PROCESS_PKL.exists(),
                    reason=f"process bundle not built yet: {PROCESS_PKL}")
def test_process_bundle_2026_construction():
    """model/combo_v2_process/model.pkl (built by scripts/build_variant.py
    --variant process from the real selection_2026.json) must carry the
    measured WS3 object: plain equal 1/3 blend of xs(top30) + dual +
    xs2(top50), residual OFF, and NO in-strategy overlays (no SPY gate, no
    book gate, no freeze — process_spec_v2)."""
    from core.runner import load_model_bundle

    bundle = load_model_bundle(PROCESS_PKL)
    assert bundle["strategy_type"] == "direct_weights"
    assert bundle["version"].startswith("combo_v2_process.2026."), bundle["version"]

    c = bundle["combo_config"]
    third = 1.0 / 3.0
    assert abs(c.xs_mom_weight - third) < 1e-12
    assert abs(c.dual_mom_weight - third) < 1e-12
    assert abs(c.xs2_weight - third) < 1e-12
    assert c.adaptive_weight == 0.0
    assert c.residual_weight == 0.0
    assert c.xs_mom_top_n == 30      # xs_momentum_top30
    assert c.xs2_top_n == 50         # xs_momentum_12_1
    # NO in-strategy overlays: the measured process curve is the plain blend.
    assert c.enable_spy_dd_gate is False
    assert c.enable_book_dd_gate is False
    assert c.enable_drawdown_freeze is False

    # The pickled strategy's config is the same construction.
    sc = bundle["model"].config
    assert abs(sc.xs2_weight - third) < 1e-12 and sc.xs2_top_n == 50

    prov = bundle["provenance"]
    assert prov["selection"]["picks"] == [
        "dual_momentum_vol", "xs_momentum_top30", "xs_momentum_12_1"]
    oc = prov["overlays_config"]
    assert oc["enable_spy_dd_gate"] is False
    assert oc["enable_book_dd_gate"] is False
    assert oc["enable_drawdown_freeze"] is False
    assert prov["sleeve_params"]["xs_momentum_12_1"] == {"xs2_top_n": 50}
    assert prov["sleeve_params"]["xs_momentum_top30"] == {"xs_mom_top_n": 30}
