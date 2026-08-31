"""Tests for the 2026-08-30 improvements pass (Tiers A + D), reworked
2026-08-31 after the adversarial audit.

Audit lesson baked in here: the original mocks injected a 'symbol' key
into get_positions() values — a shape the real function never returns —
which masked a KeyError that left the daily gate pass dead in
production. Every get_positions mock below now mirrors the REAL contract
(core/alpaca.py): {alpaca_symbol: {qty, market_value, current_price,
...}} with NO 'symbol' key inside the values.

Run:  /opt/anaconda3/bin/python3 -m pytest tests/test_improvements_ad.py -v
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime
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


def real_positions(mvs: dict[str, float]) -> dict[str, dict]:
    """The REAL get_positions() shape: keyed by symbol, values without a
    'symbol' key (core/alpaca.py:187-224)."""
    return {s: {"qty": mv / 100.0, "market_value": mv, "unrealized_pl": 0.0,
                "unrealized_pl_pct": 0.0, "avg_entry": 100.0,
                "current_price": 100.0, "side": "long"}
            for s, mv in mvs.items()}


def _seed_state(mc, **keys):
    from core.state import load_state, save_state
    st = load_state(mc)
    st.update(keys)
    save_state(st, mc)
    return st


def _synthetic_stock_data(n_days=400, n_syms=40, crash_last=0.0, seed=7):
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


def _wire(monkeypatch, orders: list, bp: float = 1e9,
          positions: dict | None = None):
    def fake_alpaca(method, endpoint, mc, data=None, logger=None, **kw):
        if endpoint == "v2/account":
            return {"buying_power": str(bp)}
        orders.append(data)
        return {"id": f"o{len(orders)}", "status": "accepted"}
    monkeypatch.setattr(gu, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(gu, "is_market_open", lambda: True)
    if positions is not None:
        monkeypatch.setattr(gu, "get_positions",
                            lambda mc, logger: positions)
    import core.risk as risk_mod
    monkeypatch.setattr(risk_mod, "fetch_snapshots", lambda *a, **kw: {},
                        raising=False)
    monkeypatch.setattr(risk_mod, "poll_order_status", lambda *a, **kw: None,
                        raising=False)
    monkeypatch.setattr(
        "core.risk.refresh_daily_book_anchor_after_rebalance",
        lambda mc, rb, logger=None: orders.append({"_anchor": rb}))


def _anchor_calls(orders):
    return [o["_anchor"] for o in orders if isinstance(o, dict) and "_anchor" in o]


def _placed(orders):
    return [o for o in orders if isinstance(o, dict) and o.get("side")]


# ───────────────────────── A1: gate computation ─────────────────────────

def test_gate_multiplier_matches_compute_weights_diagnostics():
    cfg = ComboConfig()  # book gate on, spy gate off (calibrated default)
    strat = ComboStrategy(cfg)
    stock_data = _synthetic_stock_data(crash_last=0.20)
    weights = strat.compute_weights(stock_data, {})
    diag_mult = strat.last_diagnostics["exposure_multiplier"]
    assert weights, "synthetic book must be non-empty"
    held = {s: abs(w) * 100_000 for s, w in weights.items()}
    gate = gu.compute_live_gate_multiplier(cfg, stock_data, {}, held)
    assert gate is not None
    assert gate["multiplier"] == pytest.approx(diag_mult, abs=1e-6)
    assert 0.0 <= gate["multiplier"] < 1.0


def test_gate_multiplier_is_one_in_calm_markets():
    cfg = ComboConfig()
    stock_data = _synthetic_stock_data(crash_last=0.0)
    held = {s: 1.0 for s in list(stock_data)[:30]}
    gate = gu.compute_live_gate_multiplier(cfg, stock_data, {}, held)
    assert gate is not None and gate["multiplier"] == pytest.approx(1.0)


def test_ungated_config_returns_none():
    cfg = ComboConfig(enable_spy_dd_gate=False, enable_book_dd_gate=False)
    assert not gu.config_has_gates(cfg)
    assert gu.compute_live_gate_multiplier(
        cfg, _synthetic_stock_data(), {}, {"S000": 1.0}) is None


# ───────────── A1: the production get_positions shape (audit) ───────────

def test_gate_pass_works_with_real_get_positions_shape(tmp_path, monkeypatch):
    """THE critical audit regression: values carry no 'symbol' key. The
    pass must inject it from the dict key and trade normally."""
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=0.34)
    orders: list = []
    _wire(monkeypatch, orders,
          positions=real_positions({"S000": 34_000.0}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(crash_last=0.0), {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert "error" not in info, info
    assert info["action"] == "scaled"
    buys = [o for o in _placed(orders) if o["side"] == "buy"]
    assert buys and sum(float(o["notional"]) for o in buys) == pytest.approx(66_000, rel=0.01)
    from core.state import load_state
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(1.0, abs=0.01)


def test_gate_pass_maps_alpaca_dot_symbols_to_universe_dash_form(tmp_path, monkeypatch):
    """BRK.B position must land on the BRK-B price column in book_dd."""
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=1.0)
    stock_data = _synthetic_stock_data(crash_last=0.0)
    # Rename one synthetic name to a dash-form class share.
    stock_data["BRK-B"] = stock_data.pop("S000")
    seen = {}
    orig = gu.compute_live_gate_multiplier
    def spy(config, sd, md, held):
        seen["held"] = dict(held)
        return orig(config, sd, md, held)
    monkeypatch.setattr(gu, "compute_live_gate_multiplier", spy)
    orders: list = []
    _wire(monkeypatch, orders, positions=real_positions({"BRK.B": 50_000.0}))
    gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        stock_data, {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert "BRK-B" in seen["held"] and "BRK.B" not in seen["held"]


# ───────────────────────── A1: scale math ───────────────────────────────

def _pos_list(mvs: dict[str, float]) -> list[dict]:
    # What maybe_daily_gate_update hands _scale_book: symbol injected.
    return [{"symbol": s, "market_value": mv, "current_price": 100.0}
            for s, mv in mvs.items()]


def test_scale_book_up_pro_rata(tmp_path, monkeypatch):
    orders: list = []
    _wire(monkeypatch, orders)
    out = gu._scale_book(make_mc(tmp_path), _pos_list({"A": 30_000, "B": 10_000}),
                         factor=1.5, journal=StubJournal(), logger=logger,
                         dry_run=False)
    assert out["orders"] == 2 and out["failed"] == 0
    by_sym = {o["symbol"]: o for o in _placed(orders)}
    assert by_sym["A"]["side"] == "buy"
    assert float(by_sym["A"]["notional"]) == pytest.approx(15_000)
    assert out["submitted_usd"] == pytest.approx(20_000)


def test_scale_book_down_pro_rata(tmp_path, monkeypatch):
    orders: list = []
    _wire(monkeypatch, orders)
    journal = StubJournal()
    monkeypatch.setattr("core.risk.TradeJournal", lambda name: journal,
                        raising=False)
    out = gu._scale_book(make_mc(tmp_path), _pos_list({"A": 30_000, "B": 10_000}),
                         factor=0.5, journal=journal, logger=logger,
                         dry_run=False)
    assert out["orders"] == 2
    assert out["submitted_usd"] == pytest.approx(-20_000)


def test_scale_book_cap_updates_delta_and_reports_submitted(tmp_path, monkeypatch):
    """Audit fix: the pre-cap delta must never leak into delta_usd (it
    inflated the tier anchor and disabled the soft stop for the day)."""
    orders: list = []
    _wire(monkeypatch, orders, bp=5_000)  # want $20k, only $5k bp
    out = gu._scale_book(make_mc(tmp_path), _pos_list({"A": 40_000}), factor=1.5,
                         journal=StubJournal(), logger=logger, dry_run=False)
    assert out.get("factor_capped") is not None
    assert out["delta_usd"] <= 5_000 * 1.01          # capped, not 20k
    assert out["factor"] == out["factor_capped"]
    assert out["submitted_usd"] <= 5_000 * 1.01
    total_bought = sum(float(o["notional"]) for o in _placed(orders))
    assert total_bought <= 5_000 * 1.01


def test_scale_book_skips_dust_delta(tmp_path, monkeypatch):
    orders: list = []
    _wire(monkeypatch, orders)
    out = gu._scale_book(make_mc(tmp_path), _pos_list({"A": 1_000}), factor=1.01,
                         journal=StubJournal(), logger=logger, dry_run=False)
    assert out.get("skipped") == "delta_below_minimum" and not _placed(orders)


def test_scale_book_dry_run_places_no_orders(tmp_path, monkeypatch):
    orders: list = []
    _wire(monkeypatch, orders)
    out = gu._scale_book(make_mc(tmp_path), _pos_list({"A": 30_000}), factor=1.5,
                         journal=StubJournal(), logger=logger, dry_run=True)
    assert out["dry_run"] and not _placed(orders)
    assert out["submitted_usd"] == 0.0


# ─────────────── A1: commit-from-submitted semantics (audit) ────────────

def test_capped_relever_commits_only_what_was_bought(tmp_path, monkeypatch):
    """bp-capped batch: applied must move by the ACHIEVED factor and the
    tier anchor must reflect submitted notional, not the intention."""
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=0.34)
    orders: list = []
    _wire(monkeypatch, orders, bp=10_000,
          positions=real_positions({"S000": 34_000.0}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(crash_last=0.0), {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["action"] == "scaled"
    submitted = info["scale"]["submitted_usd"]
    assert submitted <= 10_000 * 1.01
    expected = 0.34 * (1 + submitted / 34_000.0)
    from core.state import load_state
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(expected, abs=0.005)
    anchors = _anchor_calls(orders)
    assert anchors and anchors[-1]["target_gross_usd"] == pytest.approx(34_000 + submitted, rel=0.01)


def test_total_order_failure_commits_nothing(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=0.34)
    orders: list = []
    _wire(monkeypatch, orders, positions=real_positions({"S000": 34_000.0}))
    def failing_alpaca(method, endpoint, mc, data=None, logger=None, **kw):
        if endpoint == "v2/account":
            return {"buying_power": "1000000"}
        raise RuntimeError("exchange down")
    monkeypatch.setattr(gu, "alpaca_request", failing_alpaca)
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(crash_last=0.0), {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["action"] == "no_orders"
    from core.state import load_state
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(0.34)
    assert not _anchor_calls(orders)


# ─────────────── A1: sticky tier floor + gross cap (audit) ──────────────

def test_tier_floor_caps_the_relever_target(tmp_path, monkeypatch):
    """stop_grid semantics: a tier cut persists to the next rebalance.
    Gate recovered to 1.0 but floor=0.6 → target 0.6, not 1.0."""
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=0.40,
                tier_floor_multiplier=0.60)
    orders: list = []
    _wire(monkeypatch, orders, positions=real_positions({"S000": 40_000.0}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(crash_last=0.0), {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["target"] == pytest.approx(0.60)
    from core.state import load_state
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(0.60, abs=0.01)


def test_gate_ref_gross_cap_bounds_target(tmp_path, monkeypatch):
    """Rebalance-time gross_pre_gate 1.25 with cap 1.0: realized scaling
    can never exceed 0.8 — the raw gate 1.0 must be capped."""
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=0.50,
                gate_ref={"gross_pre_gate": 1.25, "max_gross_exposure": 1.0})
    orders: list = []
    _wire(monkeypatch, orders, positions=real_positions({"S000": 50_000.0}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(crash_last=0.0), {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["target"] == pytest.approx(0.80)


def test_full_delever_liquidates_and_requests_reentry(tmp_path, monkeypatch):
    """Raw gate ~0 (validated g=0): sell everything, applied→0, and clear
    last_rebalance so re-entry happens through a full rebalance."""
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=0.80,
                last_rebalance="2026-08-15T13:39:00")
    orders: list = []
    _wire(monkeypatch, orders, positions=real_positions({"S000": 40_000.0,
                                                         "S001": 40_000.0}))
    monkeypatch.setattr(gu, "compute_live_gate_multiplier",
                        lambda *a, **kw: {"multiplier": 0.0, "spy_gate": 1.0,
                                          "book_gate": 0.0, "book_dd": -0.35})
    journal = StubJournal()
    monkeypatch.setattr("core.risk.TradeJournal", lambda name: journal,
                        raising=False)
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(), {}, journal, logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["action"] == "scaled"
    sells = [o for o in _placed(orders) if o["side"] == "sell"]
    assert sum(float(o["notional"]) for o in sells) == pytest.approx(80_000, rel=0.01)
    from core.state import load_state
    st = load_state(mc)
    assert st["applied_exposure_multiplier"] == pytest.approx(0.0, abs=0.01)
    assert st["last_rebalance"] is None


def test_essentially_cash_book_requests_reentry_rebalance(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=0.01,
                last_rebalance="2026-08-15T13:39:00")
    orders: list = []
    _wire(monkeypatch, orders, positions=real_positions({}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(), {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["skipped"] == "book_essentially_cash"
    assert info.get("action") == "requested_reentry_rebalance"
    from core.state import load_state
    assert load_state(mc)["last_rebalance"] is None


def test_portfolio_stop_today_aborts_before_trading(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=0.50,
                portfolio_stop_tripped_date=datetime.now().strftime("%Y-%m-%d"))
    orders: list = []
    _wire(monkeypatch, orders, positions=real_positions({"S000": 50_000.0}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(crash_last=0.0), {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["skipped"] == "portfolio_stop_tripped_meanwhile"
    assert not _placed(orders)


def test_no_bootstrap_from_history(tmp_path, monkeypatch):
    """Audit: the history bootstrap ignored tier cuts made since that
    rebalance — removed. applied=None must skip, even with history."""
    mc = make_mc(tmp_path)
    _seed_state(mc, history=[{"strategy_diagnostics":
                              {"exposure_multiplier": 0.337}}])
    orders: list = []
    _wire(monkeypatch, orders, positions=real_positions({"S000": 34_000.0}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(crash_last=0.0), {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["skipped"] == "no_applied_multiplier"
    assert not _placed(orders)


def test_hold_within_band_and_trace(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=1.0)
    orders: list = []
    _wire(monkeypatch, orders, positions=real_positions({"S000": 50_000.0}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights",
        _synthetic_stock_data(crash_last=0.0), {}, StubJournal(), logger,
        report=None, min_stocks_required=5, dry_run=False)
    assert info["checked"] and info["action"] == "hold"
    assert not _placed(orders)
    from core.state import load_state
    assert load_state(mc)["gate_update_history"][-1]["action"] == "hold"


def test_thin_data_counts_consecutive_skips(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=1.0)
    monkeypatch.setattr(gu, "is_market_open", lambda: True)
    for expected in (1, 2):
        info = gu.maybe_daily_gate_update(
            mc, {"combo_config": ComboConfig()}, "direct_weights",
            {"S000": pd.DataFrame()}, {}, StubJournal(), logger,
            report=None, min_stocks_required=400, dry_run=False)
        assert str(info.get("skipped", "")).startswith("insufficient_data")
        assert info["consecutive_data_skips"] == expected
    from core.state import load_state
    assert load_state(mc)["gate_data_skips"] == 2


# ────────────── A1: rebalance-time gate state (audit semantics) ─────────

def test_set_gate_state_records_realized_multiplier(tmp_path):
    """gross cap bound at rebalance: applied = gross_final/gross_pre_gate
    (0.55/1.2), NOT the raw gate 0.9; floor and skip counter reset."""
    mc = make_mc(tmp_path)
    _seed_state(mc, tier_floor_multiplier=0.6, gate_data_skips=4)
    gu.set_gate_state_at_rebalance(
        mc, {"exposure_multiplier": 0.9, "gross_pre_gate": 1.2,
             "gross_final": 0.55}, ComboConfig(), logger)
    from core.state import load_state
    st = load_state(mc)
    assert st["applied_exposure_multiplier"] == pytest.approx(0.55 / 1.2, abs=1e-4)
    assert st["tier_floor_multiplier"] == 1.0
    assert st["gate_ref"]["gross_pre_gate"] == pytest.approx(1.2)
    assert "gate_data_skips" not in st


def test_set_gate_state_falls_back_to_raw_exposure(tmp_path):
    mc = make_mc(tmp_path)
    gu.set_gate_state_at_rebalance(
        mc, {"exposure_multiplier": 0.7922}, ComboConfig(), logger)
    from core.state import load_state
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(0.7922)


# ─────────────── A1: tier scaler bookkeeping + lock safety ──────────────

def test_tier_scale_updates_multiplier_and_floor(tmp_path, monkeypatch):
    import core.risk as risk_mod
    from core.state import load_state, save_state
    mc = make_mc(tmp_path)
    _seed_state(mc, applied_exposure_multiplier=0.80)
    monkeypatch.setattr(risk_mod, "alpaca_request",
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
    st = load_state(mc)
    assert st["applied_exposure_multiplier"] == pytest.approx(0.48)
    assert st["tier_floor_multiplier"] == pytest.approx(0.60)


def test_tier_scale_multiplier_update_under_scanner_lock(tmp_path, monkeypatch):
    """Regression: the scanner's tier calls hold the non-reentrant lock —
    the update must not re-acquire it and must mutate the caller's dict."""
    import threading
    import core.risk as risk_mod
    from core.state import load_state, save_state, state_lock
    mc = make_mc(tmp_path)
    state = _seed_state(mc, applied_exposure_multiplier=0.80)
    monkeypatch.setattr(risk_mod, "alpaca_request",
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
        with state_lock:
            risk_mod._soft_scale_portfolio(
                mc, positions, daily_book_start=100_000, book_fraction=0.60,
                tier="Tier1", dd_pct=-8.0, logger=logger, state=state)
        done.set()

    t = threading.Thread(target=scanner_style_call, daemon=True)
    t.start()
    assert done.wait(20), \
        "DEADLOCK: _soft_scale_portfolio re-acquired the scanner lock"
    assert state["applied_exposure_multiplier"] == pytest.approx(0.48)
    assert state["tier_floor_multiplier"] == pytest.approx(0.60)
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(0.48)


def test_new_keys_are_scanner_owned():
    from core.state import SCANNER_OWNED_KEYS
    for key in ("applied_exposure_multiplier", "tier_floor_multiplier",
                "gate_ref", "gate_data_skips", "gate_update_history"):
        assert key in SCANNER_OWNED_KEYS, key


def test_save_state_is_atomic(tmp_path):
    """Write goes through a temp file + os.replace — no .tmp leftover,
    valid JSON on disk."""
    from core.state import load_state, save_state
    mc = make_mc(tmp_path)
    save_state({"x": 1}, mc)
    assert json.loads(mc.state_path.read_text())["x"] == 1
    assert not list(tmp_path.glob("*.tmp"))


# ───────────────────────── A2: universe monitor ─────────────────────────

def test_universe_churn_records_and_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(um, "CHURN_PATH", tmp_path / "churn.json")
    monkeypatch.setattr(um, "WARN_THRESHOLD", 450)
    snap1 = um.record_universe_snapshot([f"S{i}" for i in range(500)], logger)
    assert snap1["count"] == 500 and not snap1["warn"]
    # Simulate "next day": age the stored baseline date.
    payload = json.loads((tmp_path / "churn.json").read_text())
    payload["last_names_date"] = "2000-01-01"
    payload["history"][0]["date"] = "2000-01-01"
    (tmp_path / "churn.json").write_text(json.dumps(payload))
    names2 = [f"S{i}" for i in range(60, 500)] + [f"N{i}" for i in range(5)]
    snap2 = um.record_universe_snapshot(names2, logger)
    assert snap2["count"] == 445 and snap2["warn"]
    assert snap2["n_removed"] == 60 and snap2["n_added"] == 5
    status = um.load_universe_status()
    assert status["count"] == 445 and status["warn"]


def test_universe_same_day_rerun_keeps_yesterday_baseline(tmp_path, monkeypatch):
    """Audit fix: a same-day re-run must diff against YESTERDAY's names,
    not this morning's run (which erased the day's churn record)."""
    monkeypatch.setattr(um, "CHURN_PATH", tmp_path / "churn.json")
    monkeypatch.setattr(um, "WARN_THRESHOLD", 450)
    um.record_universe_snapshot([f"S{i}" for i in range(500)], logger)
    payload = json.loads((tmp_path / "churn.json").read_text())
    payload["last_names_date"] = "2000-01-01"
    (tmp_path / "churn.json").write_text(json.dumps(payload))
    # First run today: 10 names lost vs yesterday.
    names_a = [f"S{i}" for i in range(10, 500)]
    snap_a = um.record_universe_snapshot(names_a, logger)
    assert snap_a["n_removed"] == 10
    # Re-run same day with the same names: still 10 lost vs YESTERDAY,
    # not 0 lost vs this morning.
    snap_b = um.record_universe_snapshot(names_a, logger)
    assert snap_b["n_removed"] == 10, "rerun clobbered the day's baseline"


def test_universe_corrupt_file_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(um, "CHURN_PATH", tmp_path / "churn.json")
    (tmp_path / "churn.json").write_text("{corrupt json")
    snap = um.record_universe_snapshot([f"S{i}" for i in range(500)], logger)
    assert snap["count"] == 500
    assert json.loads((tmp_path / "churn.json").read_text())["last_names"]


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
    _seed_state(mc, forward_test_start="2026-07-20",
                protocol_sha256="0" * 64)
    monkeypatch.setattr(inv_mod, "MANIFEST_PATH", _write_manifest(
        tmp_path, {"combo_v2": {"enabled": True}}))
    monkeypatch.setattr(inv_mod, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr("core.config.get_active_models", lambda: [mc])
    res = inv_mod.run_invariant_checks(logger)
    assert not res["pass"]
    assert any("MODIFIED_AFTER_START" in f or "protocol" in f
               for f in res["failures"])


def test_invariants_catch_fts_without_bound_sha(tmp_path, monkeypatch):
    """Audit fix: the July half-bound state (clock running, no sha) was
    silently skipped by the very check built to catch it."""
    mc = make_mc(tmp_path, model_path=tmp_path / "m.pkl")
    (tmp_path / "m.pkl").write_bytes(b"x")
    _seed_state(mc, forward_test_start="2026-07-20")
    monkeypatch.setattr(inv_mod, "MANIFEST_PATH", _write_manifest(
        tmp_path, {"combo_v2": {"enabled": True}}))
    monkeypatch.setattr(inv_mod, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr("core.config.get_active_models", lambda: [mc])
    res = inv_mod.run_invariant_checks(logger)
    assert not res["pass"]
    assert any("NO protocol sha256" in f for f in res["failures"])


def test_invariants_gate_enablement_check(tmp_path, monkeypatch):
    """The one config the gate layer keys on: spec gated=false vs a
    bundle whose config has gates on → failure."""
    import core.runner as runner_mod
    mc = make_mc(tmp_path, model_path=tmp_path / "m.pkl")
    (tmp_path / "m.pkl").write_bytes(b"x")
    monkeypatch.setattr(inv_mod, "MANIFEST_PATH", _write_manifest(
        tmp_path, {"combo_v2": {"enabled": True, "gated": False}}))
    monkeypatch.setattr(inv_mod, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr("core.config.get_active_models", lambda: [mc])
    monkeypatch.setattr(runner_mod, "load_model_bundle",
                        lambda p: {"combo_config": ComboConfig()})
    res = inv_mod.run_invariant_checks(logger)
    assert not res["pass"]
    assert any("gates are ON" in f for f in res["failures"])


def test_invariants_checker_errors_do_not_fail_the_sweep(tmp_path, monkeypatch):
    """Transient introspection errors are 'degraded', never a 503."""
    monkeypatch.setattr(inv_mod, "MANIFEST_PATH", _write_manifest(tmp_path, {}))
    monkeypatch.setattr(inv_mod, "STATUS_PATH", tmp_path / "status.json")
    def boom():
        raise RuntimeError("config loader down")
    monkeypatch.setattr("core.config.get_active_models", boom)
    res = inv_mod.run_invariant_checks(logger)
    assert res["pass"] and res["degraded"]
    assert res["checker_errors"]


def test_repo_manifest_matches_code_constants():
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
    # Gate flags present for all three slots (audit: previously unpinned).
    for slot, gated in (("combo_v2", True), ("combo_v2_exp", True),
                        ("combo_v2_process", False)):
        assert manifest["slots"][slot]["gated"] is gated


# ─────────────── execution stats: risk-layer exclusion (audit) ──────────

def test_slippage_stats_exclude_gate_and_risk_layer_rows():
    from core.execution_stats import compute_slippage_stats
    rebal = {"action": "exit_position", "side": "sell", "slippage_bps": 10.0,
             "notional_usd": 1000.0, "run_id": "r1"}
    gate = {"action": "gate_relevel", "side": "buy", "slippage_bps": None,
            "notional_usd": 5000.0, "run_id": "g1"}
    tier = {"action": "soft_scale_tier1", "side": "sell",
            "slippage_bps": 400.0, "notional_usd": 9999.0, "run_id": "t1"}
    stats = compute_slippage_stats([rebal, gate, tier])
    assert stats["n_trades"] == 1
    assert stats["n_excluded_risk_layer"] == 2
    assert stats["calibrated_cost_bps_per_side"] == pytest.approx(10.0)
    assert [r["run_id"] for r in stats["by_run"]] == ["r1"]


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
        assert len(e["date"]) == 10 and e["date"][4] == "-"
    assert len([e for e in events if e["kind"] == "verdict"]) == 3


def test_dashboard_calendar_view_overdue_covers_all_kinds(monkeypatch):
    import dashboard
    view = dashboard._program_calendar_view()
    assert view is not None
    # The unresolved Norgate decision (2026-08-30) must be overdue.
    assert any("Norgate" in e.get("label", "") for e in view["overdue"])
    # Resolved events never appear.
    for e in view["overdue"] + view["upcoming"]:
        assert not e.get("resolved")
    # A past unresolved NON-decision event must also count as overdue
    # (audit fix) — verified by logic: every overdue entry is simply a
    # past unresolved event, no kind filter.
    assert view["upcoming"] == sorted(view["upcoming"], key=lambda e: e["date"])
    assert len(view["upcoming"]) <= 5
