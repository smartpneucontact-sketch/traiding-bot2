"""THE golden test — the deployed PRIMARY combo_v2 bundle is frozen.

Loads the deployed primary `model/combo_v2/model.pkl` through the runner's
real unpickle path (`core.runner.load_model_bundle`) and runs
`compute_weights` on two fixed, seeded synthetic panels ("calm" and
"crash" — the crash panel engages the book-drawdown gate ramp). The
resulting weights must be BYTE-identical (IEEE-754 bit patterns, compared
via `float.hex()`) to the golden snapshot captured from the PRE-CHANGE
code on 2026-07-18, before the residual-sleeve port (Roadmap v2 Stream 1).

This test gates every future deploy: any refactor of core/combo_strategy.py
(new sleeves, new ComboConfig fields, blend changes) must leave the frozen
primary bundle's output bit-for-bit unchanged. If this test fails, the
change ALTERED THE FROZEN PRIMARY and must not ship.

Snapshot provenance:
  - captured with:  /opt/anaconda3/bin/python -m tests.test_primary_bundle_frozen --capture
  - stored at:      tests/data/golden_primary_weights.json
  - captured BEFORE any Stream-1 edits, from the pristine 2026-07-13
    combo_v2.3_nofreeze code path. NEVER regenerate the snapshot to make
    a red test green — a red test here means the primary changed.

Run:  /opt/anaconda3/bin/python -m pytest tests/test_primary_bundle_frozen.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
MODEL_PKL = _REPO / "model" / "combo_v2" / "model.pkl"
GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "golden_primary_weights.json"

PANEL_SEED = 20260718
N_STOCKS = 60
N_DAYS = 320
END_DATE = "2026-06-30"

MACRO_SYMS = (
    "SPY", "VIX", "QQQ", "IWM", "TLT",
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
)


def _bar_df(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    """OHLCV frame in the bot's per-symbol dict format (only `close` is
    consumed by ComboStrategy, but ship the full column set the data layer
    produces)."""
    close = pd.Series(close, index=index)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": pd.Series(1_000_000.0, index=index),
    })


def build_panels(scenario: str) -> tuple[dict, dict]:
    """Deterministic synthetic (stock_data, macro_data) dicts.

    scenario="calm":  plain seeded geometric walks (gates fully open).
    scenario="crash": same walks, but the last 25 sessions decay linearly
                      to -20% — lands inside the book-DD gate's linear
                      ramp (full_dd 12% → cash_dd 30%), exercising the
                      partial-exposure path and the dust filter.

    Everything derives from a fixed rng seed + fixed calendar, so the
    panels are bit-identical on every run (numpy PCG64 is reproducible).
    """
    rng = np.random.default_rng(PANEL_SEED)
    idx = pd.bdate_range(end=END_DATE, periods=N_DAYS)

    stock_data: dict[str, pd.DataFrame] = {}
    for i in range(N_STOCKS):
        sym = f"S{i:02d}"
        drift = rng.uniform(-0.0006, 0.0014)
        vol = rng.uniform(0.010, 0.035)
        rets = rng.normal(drift, vol, size=N_DAYS)
        px = rng.uniform(20.0, 200.0) * np.exp(np.cumsum(rets))
        if scenario == "crash":
            ramp = np.ones(N_DAYS)
            ramp[-25:] = np.linspace(1.0, 0.80, 25)
            px = px * ramp
        close = px.copy()
        if i >= N_STOCKS - 3:  # 3 short-history names exercise notna paths
            close[: N_DAYS - 100] = np.nan
        stock_data[sym] = _bar_df(idx, close)

    macro_data: dict[str, pd.DataFrame] = {}
    for sym in MACRO_SYMS:
        if sym == "VIX":
            steps = rng.normal(0.0, 0.8, size=N_DAYS)
            close = np.clip(16.0 + np.cumsum(steps) * 0.1 + np.abs(steps), 9.0, 80.0)
        else:
            rets = rng.normal(0.0003, 0.011, size=N_DAYS)
            close = 100.0 * np.exp(np.cumsum(rets))
            if scenario == "crash" and sym == "SPY":
                ramp = np.ones(N_DAYS)
                ramp[-25:] = np.linspace(1.0, 0.85, 25)
                close = close * ramp
        macro_data[sym] = _bar_df(idx, close)

    return stock_data, macro_data


def compute_snapshot() -> dict:
    """Load the deployed primary bundle and run both scenarios.

    Weights are serialized as float.hex() — exact IEEE-754 bit patterns,
    so equality below is byte-identity of every weight.
    """
    from core.runner import load_model_bundle

    bundle = load_model_bundle(MODEL_PKL)
    strategy = bundle["model"]
    snap: dict = {
        "bundle_version": bundle.get("version"),
        "strategy_type": bundle.get("strategy_type"),
        "panel_seed": PANEL_SEED,
    }
    for scenario in ("calm", "crash"):
        stock_data, macro_data = build_panels(scenario)
        weights = strategy.compute_weights(stock_data, macro_data)
        snap[scenario] = {
            "weights_hex": {sym: float(w).hex() for sym, w in sorted(weights.items())},
            "n_positions": len(weights),
            "gross": float(sum(abs(w) for w in weights.values())).hex(),
        }
    return snap


# ─── tests ───────────────────────────────────────────────────────────────

def test_golden_snapshot_exists():
    assert MODEL_PKL.exists(), f"primary bundle missing: {MODEL_PKL}"
    assert GOLDEN_PATH.exists(), (
        f"golden snapshot missing: {GOLDEN_PATH}. It must be captured from "
        "PRE-CHANGE code and committed — never regenerated to silence a "
        "failure. See module docstring."
    )


def test_primary_bundle_weights_byte_identical():
    """The non-negotiable deploy gate: frozen primary output is bit-frozen."""
    golden = json.loads(GOLDEN_PATH.read_text())
    snap = compute_snapshot()

    assert snap["bundle_version"] == golden["bundle_version"], (
        f"primary bundle version changed: {snap['bundle_version']!r} != "
        f"{golden['bundle_version']!r} — the PRIMARY slot is frozen."
    )
    assert snap["strategy_type"] == golden["strategy_type"]

    for scenario in ("calm", "crash"):
        g, s = golden[scenario], snap[scenario]
        assert s["n_positions"] == g["n_positions"], (
            f"[{scenario}] position count drifted: {s['n_positions']} != "
            f"{g['n_positions']}"
        )
        assert set(s["weights_hex"]) == set(g["weights_hex"]), (
            f"[{scenario}] symbol set drifted: "
            f"only_new={sorted(set(s['weights_hex']) - set(g['weights_hex']))} "
            f"only_golden={sorted(set(g['weights_hex']) - set(s['weights_hex']))}"
        )
        diffs = {
            sym: (s["weights_hex"][sym], g["weights_hex"][sym])
            for sym in g["weights_hex"]
            if s["weights_hex"][sym] != g["weights_hex"][sym]
        }
        assert not diffs, (
            f"[{scenario}] weights not byte-identical to the golden snapshot "
            f"({len(diffs)} drifted): {dict(list(diffs.items())[:5])} — the "
            "frozen primary's behavior changed. Do NOT regenerate the "
            "snapshot; fix the code."
        )
        assert s["gross"] == g["gross"], f"[{scenario}] gross exposure drifted"


def test_crash_scenario_engages_book_gate():
    """Sanity that the snapshot actually covers the gated code path:
    the crash panel must produce lower gross than the calm panel."""
    golden = json.loads(GOLDEN_PATH.read_text())
    calm = float.fromhex(golden["calm"]["gross"])
    crash = float.fromhex(golden["crash"]["gross"])
    assert calm > 0.0
    assert crash < calm, (
        f"crash gross {crash} not below calm gross {calm} — the golden "
        "panels no longer exercise the book-DD gate ramp"
    )


def test_primary_config_defaults_unpickle_cleanly():
    """Old pickle + new ComboConfig class: every documented access path
    must resolve (class-level dataclass defaults cover missing attrs)."""
    from core.runner import load_model_bundle

    bundle = load_model_bundle(MODEL_PKL)
    cfg = bundle["model"].config
    # Fields that pre-date this bundle resolve from the instance...
    assert abs(cfg.xs_mom_weight + cfg.dual_mom_weight + cfg.adaptive_weight - 1.0) < 1e-9
    # ...and any field added AFTER the bundle was pickled must resolve via
    # the class-level default AND reproduce pre-change behavior:
    assert getattr(cfg, "residual_weight", 0.0) == 0.0, (
        "frozen primary must never grow a residual sleeve allocation"
    )


if __name__ == "__main__":
    if "--capture" in sys.argv:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        if GOLDEN_PATH.exists() and "--force" not in sys.argv:
            raise SystemExit(
                f"refusing to overwrite existing golden snapshot {GOLDEN_PATH} "
                "(pass --force only if you are deliberately re-baselining, "
                "which requires orchestrator approval)"
            )
        snap = compute_snapshot()
        GOLDEN_PATH.write_text(json.dumps(snap, indent=2, sort_keys=True))
        print(f"golden snapshot written: {GOLDEN_PATH}")
        for scenario in ("calm", "crash"):
            print(f"  {scenario}: n={snap[scenario]['n_positions']} "
                  f"gross={float.fromhex(snap[scenario]['gross']):.4f}")
    else:
        print(__doc__)
