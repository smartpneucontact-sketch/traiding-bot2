"""Unit tests for the 2026-06 risk-layer fixes.

Covers:
  - leverage scaling of the portfolio-stop tiers
  - Tier-3 re-entry cool-down scheduling
  - redistribute: proceeds-based budget, per-name weight cap, ranked
    selection from the "weights" key, same-day-stop exclusion
  - orders.liquidate_all_positions (the freeze-liquidation path that used
    to be a guaranteed no-op)
  - honest realized_leverage reporting
  - book-drawdown gate + diagnostics, incl. old-pickle compatibility

Run:  /opt/anaconda3/bin/python3 -m pytest tests/ -v
(system python3 lacks pandas; core.runner is not imported here because it
pulls yfinance via core.data)
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.config import ModelConfig
import core.orders as orders_mod
import core.risk as risk_mod
from core.risk import (
    _add_trading_days, _effective_portfolio_stop, _redistribute_after_cutloss,
)

logger = logging.getLogger("test")


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


# ───── threshold scaling ─────────────────────────────────────────────────

def test_portfolio_stop_scales_by_leverage(tmp_path):
    mc = make_mc(tmp_path, cutloss_portfolio_stop=-3.0, target_leverage=2.0)
    assert _effective_portfolio_stop(mc) == pytest.approx(-6.0)


def test_portfolio_stop_unscaled_when_opted_out(tmp_path):
    mc = make_mc(tmp_path, cutloss_portfolio_stop=-3.0, target_leverage=2.0,
                 cutloss_scale_by_leverage=False)
    assert _effective_portfolio_stop(mc) == pytest.approx(-3.0)


def test_portfolio_stop_never_scaled_below_1x(tmp_path):
    mc = make_mc(tmp_path, cutloss_portfolio_stop=-3.0, target_leverage=0.5)
    assert _effective_portfolio_stop(mc) == pytest.approx(-3.0)


def test_portfolio_stop_disabled_and_sign_safe(tmp_path):
    # None / 0 disable the tiers entirely (0 would otherwise fire on every
    # scan); a positive misconfig is treated as its negative.
    assert _effective_portfolio_stop(
        make_mc(tmp_path, cutloss_portfolio_stop=None)) is None
    assert _effective_portfolio_stop(
        make_mc(tmp_path, cutloss_portfolio_stop=0.0)) is None
    assert _effective_portfolio_stop(
        make_mc(tmp_path, cutloss_portfolio_stop=4.0, target_leverage=2.0)
    ) == pytest.approx(-8.0)


def test_add_trading_days_skips_weekend():
    # 2026-06-05 is a Friday: +2 trading days → Tuesday 06-09.
    assert _add_trading_days("2026-06-05", 2) == "2026-06-09"
    assert _add_trading_days("2026-06-05", 1) == "2026-06-08"


# ───── Tier-3 sets the re-entry cool-down ────────────────────────────────

def test_tier3_sets_cooldown_and_trip_flag(tmp_path, monkeypatch):
    mc = make_mc(tmp_path, cutloss_portfolio_stop=-3.0, target_leverage=2.0,
                 cutloss_reentry_delay_days=2)
    saved = {}
    liquidated = {}

    def fake_alpaca(method, path, mc_, logger=None, data=None):
        if path == "v2/positions":
            return [{"symbol": "AAA", "qty": "10", "avg_entry_price": "100",
                     "current_price": "85", "market_value": "850"}]
        if path == "v2/account":
            # -15% on the day ≤ Tier-3 (-6% × 7/3 = -14%) → full liquidation
            return {"equity": "85000", "last_equity": "100000"}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(risk_mod, "load_state", lambda mc_: {})
    monkeypatch.setattr(risk_mod, "save_state",
                        lambda st, mc_: saved.update(st))
    monkeypatch.setattr(risk_mod, "_liquidate_all",
                        lambda mc_, pos, reason, lg: liquidated.update(n=len(pos)))

    risk_mod._cutloss_scan_model(mc, logger)

    assert liquidated["n"] == 1
    assert saved["portfolio_stop_tripped_date"]
    assert saved["reentry_cooldown_until"] == _add_trading_days(
        saved["portfolio_stop_tripped_date"], 2)
    assert saved["last_rebalance"] is None


def test_no_tier_fires_below_scaled_threshold(tmp_path, monkeypatch):
    """-3.5% daily DD tripped Tier-1 under the old unscaled -3% stop;
    with leverage scaling (-6% effective) nothing should fire."""
    mc = make_mc(tmp_path, cutloss_portfolio_stop=-3.0, target_leverage=2.0)
    scaled = {}

    def fake_alpaca(method, path, mc_, logger=None, data=None):
        if path == "v2/positions":
            return [{"symbol": "AAA", "qty": "10", "avg_entry_price": "100",
                     "current_price": "96.5", "market_value": "965"}]
        if path == "v2/account":
            return {"equity": "96500", "last_equity": "100000"}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(risk_mod, "load_state", lambda mc_: {})
    monkeypatch.setattr(risk_mod, "save_state", lambda st, mc_: None)
    monkeypatch.setattr(risk_mod, "_soft_scale_portfolio",
                        lambda *a, **kw: scaled.update(hit=True) or 0)
    monkeypatch.setattr(risk_mod, "_liquidate_all",
                        lambda *a, **kw: scaled.update(hit=True))

    risk_mod._cutloss_scan_model(mc, logger)
    assert "hit" not in scaled


# ───── disabled per-position stops (2026-06 calibration default) ────────

def test_disabled_position_stops_never_fire(tmp_path, monkeypatch):
    """hard/trailing = None (the calibrated default) must never sell, even
    on a -90% position. A 0 threshold must also mean disabled, not
    fire-on-every-tick."""
    for disabled_value in (None, 0, 0.0):
        mc = make_mc(tmp_path, cutloss_hard_stop=disabled_value,
                     cutloss_trailing_stop=disabled_value,
                     cutloss_portfolio_stop=-4.0, target_leverage=2.0)
        sells = []

        def fake_alpaca(method, path, mc_, logger=None, data=None):
            if path == "v2/positions":
                return [{"symbol": "AAA", "qty": "10", "avg_entry_price": "100",
                         "current_price": "10", "market_value": "100"}]
            if path == "v2/account":
                # account flat on the day → no portfolio tier
                return {"equity": "100000", "last_equity": "100000"}
            raise AssertionError(f"unexpected call {method} {path}")

        monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
        monkeypatch.setattr(risk_mod, "load_state", lambda mc_: {})
        monkeypatch.setattr(risk_mod, "save_state", lambda st, mc_: None)
        # The scan places stop sells via _place_position_close (the legacy
        # _execute_cutloss_sell is no longer on the scan path).
        monkeypatch.setattr(risk_mod, "_place_position_close",
                            lambda *a, **kw: sells.append(a))
        monkeypatch.setattr(risk_mod, "_instrument_and_journal_sells",
                            lambda *a, **kw: None)

        risk_mod._cutloss_scan_model(mc, logger)
        assert sells == [], f"stop fired with disabled value {disabled_value!r}"


def test_parse_stop_threshold():
    from core.config import _parse_stop_threshold
    assert _parse_stop_threshold(None) is None
    assert _parse_stop_threshold("") is None
    assert _parse_stop_threshold(0) is None
    assert _parse_stop_threshold(0.0) is None
    assert _parse_stop_threshold(5.0) is None      # positive is nonsense
    assert _parse_stop_threshold("-12") == -12.0
    assert _parse_stop_threshold(-15.0) == -15.0


# ───── redistribute: sizing, selection, exclusion ────────────────────────

def _setup_redistribute(monkeypatch, tmp_path, *, equity=100000.0,
                        cash=80000.0, proceeds=4000.0,
                        held=("WBD", "OGN"), sold=("RVMD",),
                        history=None, stopped_today=()):
    mc = make_mc(tmp_path, cutloss_portfolio_stop=-3.0, target_leverage=2.0)
    buys = []

    def fake_alpaca(method, path, mc_, logger=None, data=None):
        if path == "v2/positions":
            return [{"symbol": s} for s in held]
        if path == "v2/account":
            return {"equity": str(equity), "last_equity": str(equity),
                    "cash": str(cash), "buying_power": str(cash * 4)}
        raise AssertionError(f"unexpected call {method} {path}")

    state = {"history": history or []}
    if stopped_today:
        from datetime import datetime
        state["cutloss_sold_today"] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "symbols": list(stopped_today),
        }

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(risk_mod, "load_state", lambda mc_: state)
    monkeypatch.setattr(risk_mod, "TradeJournal", StubJournal)
    monkeypatch.setattr(risk_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        risk_mod, "_place_redistribute_buy",
        lambda mc_, sym, alloc, action, score, journal, lg:
            buys.append((sym, alloc)) or 1,
    )
    return mc, buys, list(sold), proceeds


def test_redistribute_budget_is_proceeds_not_cash_pile(tmp_path, monkeypatch):
    """The 2026-06-10 bug: $3.9k stop sale → $26.4k buy from the cash pile.
    The budget must now be the sale proceeds, split across picks and capped
    by each name's target weight."""
    history = [{
        "target_symbols": ["MTSI", "WBD", "OGN"],
        "weights": {"MTSI": 0.03, "WBD": 0.07, "OGN": 0.01},
        "conviction_weights": {"MTSI": 0.03, "WBD": 0.07, "OGN": 0.01},
    }]
    mc, buys, sold, proceeds = _setup_redistribute(
        monkeypatch, tmp_path, proceeds=4000.0, history=history)

    _redistribute_after_cutloss(mc, sold, logger, proceeds=proceeds)

    assert len(buys) == 1
    sym, alloc = buys[0]
    assert sym == "MTSI"
    # ≤ proceeds and ≤ equity × weight × leverage = 100k × 0.03 × 2 = $6k
    assert alloc <= proceeds
    assert alloc <= 6000.0
    assert alloc == pytest.approx(4000.0)


def test_redistribute_caps_at_target_weight(tmp_path, monkeypatch):
    """Large proceeds must still not build a position bigger than the
    strategy's own target allocation for that name."""
    history = [{
        "target_symbols": ["MTSI", "WBD", "OGN"],
        "weights": {"MTSI": 0.012, "WBD": 0.07, "OGN": 0.01},
        "conviction_weights": {"MTSI": 0.012, "WBD": 0.07, "OGN": 0.01},
    }]
    mc, buys, sold, _ = _setup_redistribute(
        monkeypatch, tmp_path, proceeds=50000.0, history=history)

    _redistribute_after_cutloss(mc, sold, logger, proceeds=50000.0)

    assert len(buys) == 1
    sym, alloc = buys[0]
    assert sym == "MTSI"
    # cap = 100k × 0.012 × 2 = $2.4k — nowhere near the $26.4k of the incident
    assert alloc == pytest.approx(2400.0)


def test_redistribute_reads_weights_key_for_direct_strategies(tmp_path, monkeypatch):
    """direct_weights runs store rankings under "weights"; the old code read
    only "predictions" and fell back to arbitrary set-order picks."""
    history = [{
        "target_symbols": ["AAA", "BBB", "WBD", "OGN"],
        "weights": {"AAA": 0.02, "BBB": 0.05, "WBD": 0.07, "OGN": 0.01},
    }]
    mc, buys, sold, proceeds = _setup_redistribute(
        monkeypatch, tmp_path, proceeds=3000.0, history=history)

    _redistribute_after_cutloss(mc, sold, logger, proceeds=proceeds)

    # BBB outranks AAA (0.05 > 0.02): ranked selection, not set order.
    assert [s for s, _ in buys] == ["BBB"]


def test_redistribute_never_rebuys_names_stopped_today(tmp_path, monkeypatch):
    """On 2026-06-09 seven just-stopped names were re-bought the same day and
    re-stopped within two hours."""
    history = [{
        "target_symbols": ["AAA", "BBB", "WBD", "OGN"],
        "weights": {"AAA": 0.02, "BBB": 0.05, "WBD": 0.07, "OGN": 0.01},
    }]
    mc, buys, sold, proceeds = _setup_redistribute(
        monkeypatch, tmp_path, proceeds=3000.0, history=history,
        stopped_today=("BBB",))

    _redistribute_after_cutloss(mc, sold, logger, proceeds=proceeds)

    assert [s for s, _ in buys] == ["AAA"]


def test_redistribute_skips_names_absent_from_ranking(tmp_path, monkeypatch):
    """Never buy a name the model's last run didn't rank."""
    history = [{
        "target_symbols": ["WBD", "OGN"],
        "weights": {"WBD": 0.07, "OGN": 0.01},
    }]
    mc, buys, sold, proceeds = _setup_redistribute(
        monkeypatch, tmp_path, proceeds=3000.0, history=history)

    _redistribute_after_cutloss(mc, sold, logger, proceeds=proceeds)
    assert buys == []


# ───── integration: scan → stop sell → ledger + proceeds → redistribute ──

def test_scan_wires_proceeds_and_ledger_into_redistribute(tmp_path, monkeypatch):
    """End-to-end through _cutloss_scan_model with a real state file:
    a trailing stop must (a) persist the sold name in cutloss_sold_today on
    disk and (b) pass the actual fill notional as the redistribute budget.
    Guards the scan→redistribute wiring the unit tests stub out."""
    import json

    mc = make_mc(tmp_path, cutloss_trailing_stop=-5.0, cutloss_hard_stop=None,
                 cutloss_portfolio_stop=-4.0, target_leverage=2.0)
    redistribute_calls = []

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "GET" and path == "v2/positions":
            return [{"symbol": "AAA", "qty": "10", "avg_entry_price": "100",
                     "current_price": "94", "market_value": "940"}]
        if method == "GET" and path == "v2/account":
            return {"equity": "100000", "last_equity": "100000"}
        if method == "DELETE" and path == "v2/positions/AAA":
            return {"id": "oid1", "status": "accepted"}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(risk_mod, "poll_order_status",
                        lambda oid, mc_, lg, **kw: {"status": "filled",
                                                    "filled_qty": "10",
                                                    "filled_avg_price": "94.5"})
    # Post-placement snapshot (instrumentation only) — keep the test offline.
    monkeypatch.setattr(risk_mod, "fetch_snapshots",
                        lambda syms, mc_, lg=None: {})
    monkeypatch.setattr(risk_mod, "TradeJournal", StubJournal)
    monkeypatch.setattr(
        risk_mod, "_redistribute_after_cutloss",
        lambda mc_, sold, lg, top_n=20, proceeds=None:
            redistribute_calls.append((list(sold), proceeds)),
    )

    risk_mod._cutloss_scan_model(mc, logger)

    # (a) ledger persisted to the real state file
    persisted = json.loads(mc.state_path.read_text())
    ledger = persisted.get("cutloss_sold_today", {})
    assert ledger.get("symbols") == ["AAA"]
    # (b) redistribute received the polled fill notional (10 × $94.50)
    assert redistribute_calls == [(["AAA"], pytest.approx(945.0))]


# ───── liquidate_all_positions (freeze path) ─────────────────────────────

def test_liquidate_all_closes_every_position(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    deleted = []

    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {
        "AAA": {"qty": 10.0, "market_value": 1000.0, "avg_entry": 100.0,
                "current_price": 100.0, "unrealized_pl": 0.0,
                "unrealized_pl_pct": 0.0},
        "BBB": {"qty": 5.0, "market_value": 500.0, "avg_entry": 100.0,
                "current_price": 100.0, "unrealized_pl": 0.0,
                "unrealized_pl_pct": 0.0},
    })

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        assert method == "DELETE"
        deleted.append(path)
        return {"id": "oid", "status": "accepted"}

    monkeypatch.setattr(orders_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, mc_, lg: {"status": "filled",
                                              "filled_qty": "10",
                                              "filled_avg_price": "100"})
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)

    journal = StubJournal()
    result = orders_mod.liquidate_all_positions(mc, journal, logger, None,
                                                reason="freeze_liquidation")

    assert result["n_closed"] == 2 and result["n_failed"] == 0
    assert sorted(deleted) == ["v2/positions/AAA", "v2/positions/BBB"]
    assert len(journal.records) == 2
    assert all(r.action == "freeze_liquidation" for r in journal.records)


def test_liquidate_all_is_idempotent_with_no_positions(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {})
    monkeypatch.setattr(
        orders_mod, "alpaca_request",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no orders expected")))

    result = orders_mod.liquidate_all_positions(mc, StubJournal(), logger)
    assert result == {"n_closed": 0, "n_failed": 0, "closed": [], "failed": []}


# ───── realized_leverage reporting ───────────────────────────────────────

def test_realized_leverage_is_measured_not_echoed(tmp_path, monkeypatch):
    """Weights summing to 0.8333 at target 2.0x must report ~1.67x, not 2.0
    (the June-8 dashboard bug)."""
    from core.run_report import RunReport

    mc = make_mc(tmp_path, target_leverage=2.0)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": "90000", "cash": "90000", "buying_power": "360000",
    })
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {})

    weights = {"AAA": 0.41665, "BBB": 0.41665}  # sums to 0.8333
    rb = orders_mod.rebalance_portfolio(
        target_symbols=list(weights), rankings=list(weights.items()),
        mc=mc, journal=StubJournal(), logger=logger, report=RunReport(),
        dry_run=True, target_weights=weights,
    )
    assert rb["realized_leverage"] == pytest.approx(1.6666, abs=0.001)


# ───── book-drawdown gate ────────────────────────────────────────────────

def test_book_dd_gate_cuts_exposure_in_sector_crash():
    import numpy as np
    import pandas as pd
    from core.combo_strategy import _book_drawdown, _linear_dd_ramp

    idx = pd.date_range("2026-01-01", periods=80, freq="B")
    flat = pd.Series(100.0, index=idx)
    crashed = pd.Series(100.0, index=idx)
    crashed.iloc[-10:] = np.linspace(100, 75, 10)  # -25% from its high
    px = pd.DataFrame({"FLAT": flat, "CRSH": crashed})

    dd = _book_drawdown(px, {"FLAT": 0.5, "CRSH": 0.5}, lookback=60)
    assert dd == pytest.approx(-0.125, abs=0.01)
    # ramp 15%→30%: -12.5% book DD is above the 15% trigger → no cut yet
    assert _linear_dd_ramp(dd, 0.15, 0.30) == 1.0
    # all-in on the crashed name: -25% → deep cut. Literal expectations
    # (not the implementation's own formula): -25% on a 15→30 ramp is 1/3
    # of the way from cash back to full.
    dd_concentrated = _book_drawdown(px, {"CRSH": 1.0}, lookback=60)
    assert dd_concentrated == pytest.approx(-0.25, abs=1e-9)
    assert _linear_dd_ramp(-0.25, 0.15, 0.30) == pytest.approx(1 / 3, abs=1e-6)
    assert _linear_dd_ramp(-0.12, 0.12, 0.30) == 1.0
    assert _linear_dd_ramp(-0.30, 0.12, 0.30) == 0.0
    assert _linear_dd_ramp(-0.21, 0.12, 0.30) == pytest.approx(0.5, abs=1e-6)


def test_compute_weights_diagnostics_and_old_pickle_compat():
    """A config object lacking the new book-gate fields (old pickle) must
    still work, and diagnostics must be published."""
    import numpy as np
    import pandas as pd
    from core.combo_strategy import ComboStrategy

    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    stock_data = {}
    for i in range(35):
        drift = 0.0005 + 0.002 * (i / 35)
        px = 100 * np.exp(np.cumsum(rng.normal(drift, 0.02, len(idx))))
        stock_data[f"S{i:02d}"] = pd.DataFrame({"close": px}, index=idx)
    spy = pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))},
        index=idx)
    vix = pd.DataFrame({"close": rng.uniform(12, 35, len(idx))}, index=idx)
    macro = {"SPY": spy, "VIX": vix}

    # Old-pickle simulation: config namespace without any book-gate fields.
    old_cfg = SimpleNamespace(
        xs_mom_weight=1/3, xs_mom_top_n=10,
        dual_mom_weight=1/3, dual_mom_top_n=10, dual_mom_vol_target=0.15,
        adaptive_weight=1/3, adaptive_top_n=10,
        adaptive_calm_leverage=1.5, adaptive_neutral_leverage=1.0,
        adaptive_stress_leverage=0.5,
        spy_dd_lookback=60, spy_full_dd=0.08, spy_cash_dd=0.18,
        max_gross_exposure=1.0, min_position_weight=0.003,
    )
    strat = ComboStrategy.__new__(ComboStrategy)
    strat.config = old_cfg
    weights = strat.compute_weights(stock_data, macro)

    assert weights, "old-style config must still produce a book"
    diag = strat.last_diagnostics
    assert diag["book_gate"] is None          # gate off for old configs
    assert 0.0 <= diag["exposure_multiplier"] <= 1.0
    assert diag["n_positions"] == len(weights)
    assert sum(abs(w) for w in weights.values()) <= 1.0 + 1e-9
