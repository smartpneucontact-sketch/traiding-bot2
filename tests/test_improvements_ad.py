"""Tests for the 2026-08-30 improvements pass (Tiers A + D).

A1  Daily gate-cadence tracking (core/gate_update.py): the validated
    backtests apply the drawdown gates daily; live sampled them only at
    21-day rebalances. Covers: the recomputed multiplier matches
    compute_weights' own diagnostics, band/no-op behavior, pro-rata
    scale math both directions, the tier scaler's multiplier write, and
    the state-merge contract for the new scanner-owned keys.
A2  Universe churn monitor (core/universe_monitor.py).
D10 Spec-manifest invariant checker (core/invariants.py).
D12 Program calendar (program_calendar.json + dashboard view).

Run:  /opt/anaconda3/bin/python3 -m pytest tests/test_improvements_ad.py -v
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.config import ModelConfig
from core.combo_strategy import ComboConfig, ComboStrategy
import core.gate_update as gu
import core.invariants as inv_mod
import core.universe_monitor as um

logger = logging.getLogger("test_improvements_ad")


def make_mc(tmp_path, **kw) -> ModelConfig:
    defaults = dict(
        name="combo_v2",
        model_path=Path("/nonexistent/model.pkl"),
        feature_version="combo",
        alpaca_key="k",
        alpaca_secret="s",
        state_path=tmp_path / "state.json",
        enable_cutloss=True,
        target_leverage=2.0,
    )
    defaults.update(kw)
    return ModelConfig(**defaults)


class StubJournal:
    def __init__(self, *a, **kw):
        self.records = []

    def log_trade(self, record):
        self.records.append(record)


def _synthetic_stock_data(n_days=400, n_syms=40, crash_last=0.0, seed=7):
    """Symbols with mild upward drift; optionally a crash over the last
    30 days of `crash_last` (e.g. 0.2 = -20% from the running high)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    data = {}
    for i in range(n_syms):
        base = 100 * np.cumprod(1 + rng.normal(0.0005, 0.001, n_days))
        if crash_last > 0:
            ramp = np.linspace(0.0, crash_last, 30)
            base[-30:] = base[-30] * (1 - ramp)
        data[f"S{i:03d}"] = pd.DataFrame({"close": base}, index=idx)
    return data


# ───────────────────────── A1: gate computation ─────────────────────────

def test_gate_multiplier_matches_compute_weights_diagnostics():
    """The daily pass must see the SAME multiplier compute_weights would
    publish for the same market state (same functions, same fields)."""
    cfg = ComboConfig()  # book gate on, spy gate off (calibrated default)
    strat = ComboStrategy(cfg)
    stock_data = _synthetic_stock_data(crash_last=0.20)
    weights = strat.compute_weights(stock_data, {})
    diag_mult = strat.last_diagnostics["exposure_multiplier"]
    assert weights, "synthetic book must be non-empty"
    # Held book = the (post-gate) live positions; _book_drawdown
    # normalizes so the scale doesn't matter.
    held = {s: abs(w) * 100_000 for s, w in weights.items()}
    gate = gu.compute_live_gate_multiplier(cfg, stock_data, {}, held)
    assert gate is not None
    # Not exactly equal: compute_weights keys the gate to its own PRE-GATE
    # combined book, the daily pass to the held book — same names, same
    # relative weights here, so they must agree tightly.
    assert gate["multiplier"] == pytest.approx(diag_mult, abs=1e-6)
    # A -20% dip must actually engage the 12%→30% ramp.
    assert 0.0 <= gate["multiplier"] < 1.0


def test_gate_multiplier_is_one_in_calm_markets():
    cfg = ComboConfig()
    stock_data = _synthetic_stock_data(crash_last=0.0)
    held = {s: 1.0 for s in list(stock_data)[:30]}
    gate = gu.compute_live_gate_multiplier(cfg, stock_data, {}, held)
    assert gate is not None
    assert gate["multiplier"] == pytest.approx(1.0)


def test_ungated_config_returns_none():
    """Process-slot semantics: both gates off → the daily pass no-ops."""
    cfg = ComboConfig(enable_spy_dd_gate=False, enable_book_dd_gate=False)
    assert not gu.config_has_gates(cfg)
    assert gu.compute_live_gate_multiplier(
        cfg, _synthetic_stock_data(), {}, {"S000": 1.0}) is None


def test_config_has_gates_defaults():
    assert gu.config_has_gates(ComboConfig())          # book gate default on
    assert gu.config_has_gates(None) is False


# ───────────────────────── A1: scale math ───────────────────────────────

def _positions(mvs: dict[str, float]) -> list[dict]:
    return [{"symbol": s, "market_value": str(mv), "current_price": "100"}
            for s, mv in mvs.items()]


def _wire_scale(monkeypatch, orders: list, bp: float = 1e9):
    def fake_alpaca(method, endpoint, mc, data=None, logger=None, **kw):
        if endpoint == "v2/account":
            return {"buying_power": str(bp)}
        orders.append(data)
        return {"id": f"o{len(orders)}", "status": "accepted"}
    monkeypatch.setattr(gu, "alpaca_request", fake_alpaca)
    # Sell instrumentation lives in core.risk; neutralize the network parts.
    import core.risk as risk_mod
    monkeypatch.setattr(risk_mod, "fetch_snapshots",
                        lambda *a, **kw: {}, raising=False)
    monkeypatch.setattr(risk_mod, "poll_order_status",
                        lambda *a, **kw: None, raising=False)


def test_scale_book_up_pro_rata(tmp_path, monkeypatch):
    orders: list = []
    _wire_scale(monkeypatch, orders)
    mc = make_mc(tmp_path)
    out = gu._scale_book(mc, _positions({"A": 30_000, "B": 10_000}),
                         factor=1.5, journal=StubJournal(), logger=logger,
                         dry_run=False)
    assert out["orders"] == 2 and out["failed"] == 0
    by_sym = {o["symbol"]: o for o in orders if o.get("side")}
    assert by_sym["A"]["side"] == "buy"
    assert float(by_sym["A"]["notional"]) == pytest.approx(15_000)
    assert float(by_sym["B"]["notional"]) == pytest.approx(5_000)


def test_scale_book_down_pro_rata(tmp_path, monkeypatch):
    orders: list = []
    _wire_scale(monkeypatch, orders)
    mc = make_mc(tmp_path)
    journal = StubJournal()
    monkeypatch.setattr("core.risk.TradeJournal", lambda name: journal,
                        raising=False)
    out = gu._scale_book(mc, _positions({"A": 30_000, "B": 10_000}),
                         factor=0.5, journal=journal, logger=logger,
                         dry_run=False)
    assert out["orders"] == 2
    by_sym = {o["symbol"]: o for o in orders if o.get("side")}
    assert by_sym["A"]["side"] == "sell"
    assert float(by_sym["A"]["notional"]) == pytest.approx(15_000)


def test_scale_book_caps_buys_to_buying_power(tmp_path, monkeypatch):
    orders: list = []
    _wire_scale(monkeypatch, orders, bp=5_000)  # want $20k, only $5k bp
    mc = make_mc(tmp_path)
    out = gu._scale_book(mc, _positions({"A": 40_000}), factor=1.5,
                         journal=StubJournal(), logger=logger, dry_run=False)
    assert out.get("factor_capped") is not None
    total_bought = sum(float(o["notional"]) for o in orders if o.get("side") == "buy")
    assert total_bought <= 5_000 * 1.01


def test_scale_book_skips_dust_delta(tmp_path, monkeypatch):
    orders: list = []
    _wire_scale(monkeypatch, orders)
    mc = make_mc(tmp_path)
    out = gu._scale_book(mc, _positions({"A": 1_000}), factor=1.01,
                         journal=StubJournal(), logger=logger, dry_run=False)
    assert out.get("skipped") == "delta_below_minimum"
    assert not [o for o in orders if o.get("side")]


def test_scale_book_dry_run_places_no_orders(tmp_path, monkeypatch):
    orders: list = []
    _wire_scale(monkeypatch, orders)
    mc = make_mc(tmp_path)
    out = gu._scale_book(mc, _positions({"A": 30_000}), factor=1.5,
                         journal=StubJournal(), logger=logger, dry_run=True)
    assert out["dry_run"] and not [o for o in orders if o.get("side")]


# ─────────────────── A1: applied-multiplier bookkeeping ─────────────────

def test_set_applied_multiplier_roundtrip(tmp_path):
    mc = make_mc(tmp_path)
    gu.set_applied_multiplier(mc, 0.4321, logger)
    from core.state import load_state
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(0.4321)
    gu.set_applied_multiplier(mc, None, logger)
    assert "applied_exposure_multiplier" not in load_state(mc)


def test_tier_scale_updates_applied_multiplier(tmp_path, monkeypatch):
    """A Tier-1 cut to 60% of the day book must multiply the applied gate
    multiplier by the fraction it kept — otherwise tomorrow's daily gate
    pass would scale against a book it thinks is bigger than it is."""
    import core.risk as risk_mod
    from core.state import load_state, save_state
    mc = make_mc(tmp_path)
    state = load_state(mc)
    state["applied_exposure_multiplier"] = 0.80
    save_state(state, mc)
    monkeypatch.setattr(
        risk_mod, "alpaca_request",
        lambda *a, **kw: {"id": "o1", "status": "accepted"})
    monkeypatch.setattr(risk_mod, "fetch_snapshots", lambda *a, **kw: {},
                        raising=False)
    monkeypatch.setattr(risk_mod, "poll_order_status", lambda *a, **kw: None,
                        raising=False)
    monkeypatch.setattr(risk_mod, "TradeJournal", lambda name: StubJournal())
    positions = [{"symbol": "A", "market_value": "100000",
                  "current_price": "100"}]
    n = risk_mod._soft_scale_portfolio(
        mc, positions, daily_book_start=100_000, book_fraction=0.60,
        tier="Tier1", dd_pct=-8.0, logger=logger)
    assert n == 1
    # kept 60% of the day book → applied 0.80 × 0.60 = 0.48
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(0.48)


def test_tier_scale_multiplier_update_under_scanner_lock(tmp_path, monkeypatch):
    """Regression for the 2026-08-30 review catch: the scanner's tier calls
    happen INSIDE _cutloss_state_lock (non-reentrant) — the multiplier
    update must not re-acquire it (deadlock) and must mutate the caller's
    state dict in place so the scanner's later saves keep it."""
    import threading
    import core.risk as risk_mod
    from core.state import load_state, save_state, state_lock
    mc = make_mc(tmp_path)
    state = load_state(mc)
    state["applied_exposure_multiplier"] = 0.80
    save_state(state, mc)
    monkeypatch.setattr(
        risk_mod, "alpaca_request",
        lambda *a, **kw: {"id": "o1", "status": "accepted"})
    monkeypatch.setattr(risk_mod, "fetch_snapshots", lambda *a, **kw: {},
                        raising=False)
    monkeypatch.setattr(risk_mod, "poll_order_status", lambda *a, **kw: None,
                        raising=False)
    monkeypatch.setattr(risk_mod, "TradeJournal", lambda name: StubJournal())
    positions = [{"symbol": "A", "market_value": "100000",
                  "current_price": "100"}]
    done = threading.Event()

    def scanner_style_call():
        with state_lock:  # what _cutloss_scan_model holds at the call site
            risk_mod._soft_scale_portfolio(
                mc, positions, daily_book_start=100_000, book_fraction=0.60,
                tier="Tier1", dd_pct=-8.0, logger=logger, state=state)
        done.set()

    t = threading.Thread(target=scanner_style_call, daemon=True)
    t.start()
    assert done.wait(20), \
        "DEADLOCK: _soft_scale_portfolio re-acquired the scanner lock"
    assert state["applied_exposure_multiplier"] == pytest.approx(0.48)
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(0.48)


def test_new_keys_are_scanner_owned():
    """Regression: the merged pipeline save must adopt these from disk or
    every gate/tier write gets clobbered at the end of a pipeline run."""
    from core.state import SCANNER_OWNED_KEYS
    assert "applied_exposure_multiplier" in SCANNER_OWNED_KEYS
    assert "gate_update_history" in SCANNER_OWNED_KEYS


def test_maybe_daily_gate_update_holds_within_band(tmp_path, monkeypatch):
    """applied=1.0, calm market (target 1.0) → 'hold', no orders."""
    from core.state import load_state, save_state
    mc = make_mc(tmp_path)
    state = load_state(mc)
    state["applied_exposure_multiplier"] = 1.0
    save_state(state, mc)
    stock_data = _synthetic_stock_data(crash_last=0.0)
    monkeypatch.setattr(gu, "is_market_open", lambda: True)
    monkeypatch.setattr(
        gu, "get_positions",
        lambda mc, logger: {"S000": {"symbol": "S000",
                                     "market_value": "50000",
                                     "current_price": "100"}})
    orders: list = []
    _wire_scale(monkeypatch, orders)
    bundle = {"combo_config": ComboConfig()}
    info = gu.maybe_daily_gate_update(
        mc, bundle, "direct_weights", stock_data, {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["checked"] and info["action"] == "hold"
    assert not [o for o in orders if o.get("side")]
    # The daily check leaves a trace for the dashboard.
    assert load_state(mc)["gate_update_history"][-1]["action"] == "hold"


def test_maybe_daily_gate_update_relevers_after_recovery(tmp_path, monkeypatch):
    """The whipsaw fix itself: book pinned at 0.34 from a crash-day
    rebalance, market now calm (target 1.0) → pro-rata BUYS scale the
    book back up and the applied multiplier moves to the target."""
    from core.state import load_state, save_state
    mc = make_mc(tmp_path)
    state = load_state(mc)
    state["applied_exposure_multiplier"] = 0.34
    save_state(state, mc)
    stock_data = _synthetic_stock_data(crash_last=0.0)
    monkeypatch.setattr(gu, "is_market_open", lambda: True)
    monkeypatch.setattr(
        gu, "get_positions",
        lambda mc, logger: {"S000": {"symbol": "S000",
                                     "market_value": "34000",
                                     "current_price": "100"}})
    orders: list = []
    _wire_scale(monkeypatch, orders)
    called = {}
    monkeypatch.setattr(
        "core.risk.refresh_daily_book_anchor_after_rebalance",
        lambda mc, rb, logger=None: called.setdefault("anchor", rb))
    bundle = {"combo_config": ComboConfig()}
    info = gu.maybe_daily_gate_update(
        mc, bundle, "direct_weights", stock_data, {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["action"] == "scaled"
    buys = [o for o in orders if o.get("side") == "buy"]
    assert buys, "recovery must produce re-lever buys"
    # factor = 1.0 / 0.34 → buy ≈ $66k against the $34k position
    assert sum(float(o["notional"]) for o in buys) == pytest.approx(66_000, rel=0.01)
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(1.0)
    assert called.get("anchor", {}).get("target_gross_usd") == pytest.approx(100_000, rel=0.01)


def test_maybe_daily_gate_update_skips_thin_data(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    monkeypatch.setattr(gu, "is_market_open", lambda: True)
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        {"S000": pd.DataFrame()}, {}, StubJournal(), logger,
        report=None, min_stocks_required=400, dry_run=False)
    assert str(info.get("skipped", "")).startswith("insufficient_data")


def test_maybe_daily_gate_update_bootstraps_from_history(tmp_path, monkeypatch):
    """No applied multiplier in state (pre-fix slots): adopt the last
    rebalance's published exposure_multiplier instead of trading blind."""
    from core.state import load_state, save_state
    mc = make_mc(tmp_path)
    state = load_state(mc)
    state["history"] = [{"strategy_diagnostics": {"exposure_multiplier": 0.337}}]
    save_state(state, mc)
    stock_data = _synthetic_stock_data(crash_last=0.0)
    monkeypatch.setattr(gu, "is_market_open", lambda: True)
    monkeypatch.setattr(
        gu, "get_positions",
        lambda mc, logger: {"S000": {"symbol": "S000",
                                     "market_value": "34000",
                                     "current_price": "100"}})
    orders: list = []
    _wire_scale(monkeypatch, orders)
    monkeypatch.setattr(
        "core.risk.refresh_daily_book_anchor_after_rebalance",
        lambda *a, **kw: None)
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        stock_data, {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["applied"] == pytest.approx(0.337)
    assert info["action"] == "scaled"


# ───────────────────────── A2: universe monitor ─────────────────────────

def test_universe_churn_records_and_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(um, "CHURN_PATH", tmp_path / "churn.json")
    monkeypatch.setattr(um, "WARN_THRESHOLD", 450)
    snap1 = um.record_universe_snapshot([f"S{i}" for i in range(500)], logger)
    assert snap1["count"] == 500 and not snap1["warn"]
    # Next day: 60 names gone, 5 new → warn fires below 450.
    names2 = [f"S{i}" for i in range(60, 500)] + [f"N{i}" for i in range(5)]
    # Force a different date so the same-day dedupe doesn't collapse them.
    payload = json.loads((tmp_path / "churn.json").read_text())
    payload["history"][0]["date"] = "2000-01-01"
    (tmp_path / "churn.json").write_text(json.dumps(payload))
    snap2 = um.record_universe_snapshot(names2, logger)
    assert snap2["count"] == 445 and snap2["warn"]
    assert snap2["n_removed"] == 60 and snap2["n_added"] == 5
    status = um.load_universe_status()
    assert status["count"] == 445 and status["warn"]
    assert len(status["trend"]) == 2


# ───────────────────────── D10: invariant checker ───────────────────────

def _write_manifest(tmp_path, slots: dict, runtime: dict | None = None):
    p = tmp_path / "spec_manifest.json"
    p.write_text(json.dumps({"slots": slots, "runtime": runtime or {}}))
    return p


def test_invariants_pass_on_matching_config(tmp_path, monkeypatch):
    mc = make_mc(tmp_path, name="combo_v2", exec_style="marketable_limit",
                 model_path=tmp_path / "m.pkl")
    (tmp_path / "m.pkl").write_bytes(b"x")
    monkeypatch.setattr(inv_mod, "MANIFEST_PATH", _write_manifest(
        tmp_path,
        {"combo_v2": {"enabled": True, "target_leverage": 2.0,
                      "exec_style": "marketable_limit"}},
        {"tier_fractions": {"1": 0.6, "2": 0.3, "3": 0.0}},
    ))
    monkeypatch.setattr(inv_mod, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr("core.config.get_active_models", lambda: [mc])
    res = inv_mod.run_invariant_checks(logger)
    assert res["pass"], res["failures"]
    assert (tmp_path / "status.json").exists()


def test_invariants_catch_leverage_divergence(tmp_path, monkeypatch):
    """The exact exp-slot incident: spec says 2.0, runtime has 1.0."""
    mc = make_mc(tmp_path, target_leverage=1.0, model_path=tmp_path / "m.pkl")
    (tmp_path / "m.pkl").write_bytes(b"x")
    monkeypatch.setattr(inv_mod, "MANIFEST_PATH", _write_manifest(
        tmp_path, {"combo_v2": {"enabled": True, "target_leverage": 2.0}}))
    monkeypatch.setattr(inv_mod, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr("core.config.get_active_models", lambda: [mc])
    res = inv_mod.run_invariant_checks(logger)
    assert not res["pass"]
    assert any("target_leverage" in f for f in res["failures"])


def test_invariants_catch_missing_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(inv_mod, "MANIFEST_PATH", _write_manifest(
        tmp_path, {"combo_v2_exp": {"enabled": True}}))
    monkeypatch.setattr(inv_mod, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr("core.config.get_active_models", lambda: [])
    res = inv_mod.run_invariant_checks(logger)
    assert not res["pass"]
    assert any("combo_v2_exp" in f and "ACTIVE" in f for f in res["failures"])


def test_invariants_catch_protocol_sha_mismatch(tmp_path, monkeypatch):
    from core.state import load_state, save_state
    mc = make_mc(tmp_path, model_path=tmp_path / "m.pkl")
    (tmp_path / "m.pkl").write_bytes(b"x")
    st = load_state(mc)
    st["forward_test_start"] = "2026-07-20"
    st["protocol_sha256"] = "0" * 64  # will not match the real file hash
    save_state(st, mc)
    monkeypatch.setattr(inv_mod, "MANIFEST_PATH", _write_manifest(
        tmp_path, {"combo_v2": {"enabled": True}}))
    monkeypatch.setattr(inv_mod, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr("core.config.get_active_models", lambda: [mc])
    res = inv_mod.run_invariant_checks(logger)
    assert not res["pass"]
    assert any("MODIFIED_AFTER_START" in f or "protocol" in f
               for f in res["failures"])


def test_repo_manifest_matches_code_constants():
    """The committed spec_manifest.json must agree with the constants the
    trading code actually imports (tier fractions, guard ordering)."""
    manifest = json.loads(
        (Path(__file__).resolve().parent.parent / "spec_manifest.json")
        .read_text())
    from core.risk import TIER_FRACTIONS
    wanted = {int(k): float(v)
              for k, v in manifest["runtime"]["tier_fractions"].items()}
    assert {int(k): float(v) for k, v in TIER_FRACTIONS.items()} == wanted
    from core.runner import MIN_STOCKS_REQUIRED
    from core.universe_monitor import WARN_THRESHOLD
    assert WARN_THRESHOLD > MIN_STOCKS_REQUIRED


# ───────────────────────── D12: program calendar ────────────────────────

def test_program_calendar_file_is_valid():
    cal = json.loads(
        (Path(__file__).resolve().parent.parent / "program_calendar.json")
        .read_text())
    events = cal["events"]
    assert len(events) >= 10
    for e in events:
        assert e["kind"] in ("checkpoint", "verdict", "decision", "measurement")
        assert e["severity"] in ("info", "high")
        # ISO dates sort correctly as strings.
        assert len(e["date"]) == 10 and e["date"][4] == "-"
    # The three 12-month verdicts must be present.
    verdicts = [e for e in events if e["kind"] == "verdict"]
    assert len(verdicts) == 3


def test_dashboard_calendar_view(monkeypatch):
    import dashboard
    view = dashboard._program_calendar_view()
    assert view is not None
    # The Norgate decision (2026-08-30, unresolved) is overdue or due today.
    all_dates = [e["date"] for e in view["overdue"] + view["upcoming"]]
    assert view["upcoming"] == sorted(view["upcoming"], key=lambda e: e["date"])
    assert len(view["upcoming"]) <= 5
