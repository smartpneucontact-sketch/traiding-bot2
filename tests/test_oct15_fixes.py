"""Tests for the 2026-10-15 fixes after the Oct 7-15 drawdown review.

1. Rebalance pre-flight sizes on INCREMENTAL buys, not the whole book
   (the Sep 29 bug that left the gated slots at ~1.6x instead of 2x).
2. Rebalance gate-state records the REALIZED multiplier and a sticky
   floor when the broker funded less than the design book.
3. The daily gate pass reconciles pre-fix bookkeeping once, from the
   last rebalance's own record, without trading.
4. Relative symmetric gate band; today-row guard; target-weight keying.
5. Scanner heartbeat, session minimum drawdown, near-miss warning.
6. /ready flags a stale scanner heartbeat during market hours.
7. Config migration enables the process slot's tiers.

Run:  /opt/anaconda3/bin/python3 -m pytest tests/test_oct15_fixes.py -v
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("DATA_DIR", os.environ.get("DATA_DIR") or tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.config import ModelConfig
from core.combo_strategy import ComboConfig
import core.gate_update as gu
import core.orders as orders_mod
import core.risk as risk_mod
from core.run_report import RunReport
from core.state import load_state, save_state

logger = logging.getLogger("test_oct15")


def make_mc(tmp_path, **kw) -> ModelConfig:
    defaults = dict(name="combo_v2", model_path=Path("/nonexistent/model.pkl"),
                    feature_version="combo", alpaca_key="k", alpaca_secret="s",
                    state_path=tmp_path / "state.json", enable_cutloss=True,
                    target_leverage=2.0, cutloss_portfolio_stop=-4.0,
                    cutloss_scale_by_leverage=True)
    defaults.update(kw)
    return ModelConfig(**defaults)


class StubJournal:
    jsonl_path = "/dev/null"

    def __init__(self, *a, **kw):
        self.records = []

    def log_trade(self, record):
        self.records.append(record)


def _seed(mc, **keys):
    st = load_state(mc); st.update(keys); save_state(st, mc); return st


# ───────────────────── 1. pre-flight on incremental buys ────────────────

def _run_rebalance(tmp_path, monkeypatch, *, equity, bp, positions, weights):
    """Market-order rebalance with a recording fake broker."""
    posted = []

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        if method == "POST" and endpoint == "v2/orders":
            posted.append(dict(data))
            return {"id": f"o{len(posted)}", "status": "accepted"}
        if endpoint == "v2/positions":
            return []
        return {}

    mc = make_mc(tmp_path)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": str(equity), "cash": "0", "buying_power": str(bp)})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {
        s: {"qty": mv / 100.0, "market_value": float(mv), "avg_entry": 100.0,
            "current_price": 100.0, "unrealized_pl": 0.0, "unrealized_pl_pct": 0.0}
        for s, mv in positions.items()})
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, *a, **kw: {"id": oid, "status": "filled",
                                               "filled_qty": "1", "filled_avg_price": "100"})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {})
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)
    rb = orders_mod.rebalance_portfolio(
        target_symbols=list(weights), rankings=[(s, 1.0) for s in weights],
        mc=mc, journal=StubJournal(), logger=logger, report=RunReport(),
        dry_run=False, target_weights=weights)
    return rb, posted


def test_preflight_does_not_scale_a_full_book_when_buys_fit(tmp_path, monkeypatch):
    """The Sep 29 scenario: 2x book already held (190k on 100k equity),
    buying power 60k. Only the NEW 20k position needs funding -> no
    auto-scale. (The old check compared the whole 200k target to 60k.)"""
    rb, posted = _run_rebalance(
        tmp_path, monkeypatch, equity=100_000, bp=60_000,
        positions={"A": 95_000, "B": 95_000},
        weights={"A": 0.5, "B": 0.4, "C": 0.1})
    assert rb["auto_scaled_to_buying_power"] is False
    assert rb["buy_need_usd"] == pytest.approx(20_000)       # C only (A drift 5% < 10%)
    assert rb["target_gross_usd"] == pytest.approx(200_000)
    c_buy = [o for o in posted if o.get("symbol") == "C" and o.get("side") == "buy"]
    assert c_buy and float(c_buy[0]["notional"]) == pytest.approx(20_000)


def test_preflight_scales_only_the_buy_side(tmp_path, monkeypatch):
    rb, posted = _run_rebalance(
        tmp_path, monkeypatch, equity=100_000, bp=10_000,
        positions={"A": 95_000, "B": 95_000},
        weights={"A": 0.5, "B": 0.4, "C": 0.1})
    assert rb["auto_scaled_to_buying_power"] is True
    assert rb["buy_scale"] == pytest.approx(0.5)
    c_buy = [o for o in posted if o.get("symbol") == "C" and o.get("side") == "buy"]
    assert float(c_buy[0]["notional"]) == pytest.approx(10_000)
    b_trim = [o for o in posted if o.get("symbol") == "B" and o.get("side") == "sell"]
    assert b_trim and float(b_trim[0]["notional"]) == pytest.approx(15_000)  # trims untouched
    assert rb["target_gross_usd"] == pytest.approx(190_000)   # 100k + 80k + 10k


# ───────────────── 2. realized multiplier + sticky floor ────────────────

def test_rebalance_shortfall_becomes_sticky_floor(tmp_path):
    mc = make_mc(tmp_path)
    diag = {"exposure_multiplier": 1.0, "gross_pre_gate": 1.1667, "gross_final": 1.0}
    gu.set_gate_state_at_rebalance(
        mc, diag, ComboConfig(), logger,
        result={"target_gross_usd": 130_745.93, "portfolio_value": 87_562.0,
                "auto_scaled_to_buying_power": True},
        leverage=2.0)
    st = load_state(mc)
    m_real = 130_745.93 / (87_562.0 * 2.0 * 1.1667)
    assert st["applied_exposure_multiplier"] == pytest.approx(m_real, abs=1e-3)
    assert st["tier_floor_multiplier"] == pytest.approx(m_real / (1.0 / 1.1667), abs=1e-3)
    assert st["gate_ref"]["rebalance_shortfall"] == pytest.approx(0.7466, abs=2e-3)


def test_rebalance_at_design_keeps_floor_one(tmp_path):
    mc = make_mc(tmp_path)
    diag = {"exposure_multiplier": 1.0, "gross_pre_gate": 1.1667, "gross_final": 1.0}
    gu.set_gate_state_at_rebalance(
        mc, diag, ComboConfig(), logger,
        result={"reconciliation": {"book": {"achieved_gross_usd": 199_000.0}},
                "target_gross_usd": 200_000.0, "portfolio_value": 100_000.0},
        leverage=2.0)
    st = load_state(mc)
    assert st["applied_exposure_multiplier"] == pytest.approx(1 / 1.1667, abs=1e-3)
    assert st["tier_floor_multiplier"] == 1.0
    assert st["gate_ref"]["rebalance_shortfall"] == pytest.approx(0.995, abs=0.01)


# ─────────────── 3. one-time reconciliation in the daily pass ───────────

def _synthetic(n_days=400, n_syms=40, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n_days)
    return {f"S{i:03d}": pd.DataFrame({"close": 100 * np.cumprod(1 + rng.normal(0.0005, 0.001, n_days))}, index=idx)
            for i in range(n_syms)}


def _real_positions(mvs):
    return {s: {"qty": mv / 100.0, "market_value": mv, "current_price": 100.0,
                "unrealized_pl": 0.0, "unrealized_pl_pct": 0.0, "avg_entry": 100.0,
                "side": "long"} for s, mv in mvs.items()}


def _wire(monkeypatch, orders, positions):
    monkeypatch.setattr(gu, "alpaca_request", lambda m, e, mc, data=None, logger=None, **kw:
                        {"buying_power": "1000000"} if e == "v2/account"
                        else (orders.append(data) or {"id": "o", "status": "accepted"}))
    monkeypatch.setattr(gu, "is_market_open", lambda: True)
    monkeypatch.setattr(gu, "get_positions", lambda mc, logger: positions)
    monkeypatch.setattr(risk_mod, "fetch_snapshots", lambda *a, **kw: {}, raising=False)
    monkeypatch.setattr(risk_mod, "poll_order_status", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr("core.risk.refresh_daily_book_anchor_after_rebalance",
                        lambda mc, rb, logger=None: None)


def test_daily_pass_reconciles_prefix_bookkeeping_once(tmp_path, monkeypatch):
    """Live situation on 2026-10-15: applied 0.8571 recorded by the pre-fix
    code while the Sep 29 rebalance funded ~75% of design. The pass must
    adopt the realized multiplier + floor from the rebalance record, trade
    nothing, and not repeat the correction."""
    mc = make_mc(tmp_path)
    _seed(mc, applied_exposure_multiplier=0.8571, tier_floor_multiplier=1.0,
          gate_ref={"gross_pre_gate": 1.1667, "max_gross_exposure": 1.0},
          history=[{"strategy_diagnostics": {"exposure_multiplier": 1.0,
                                             "gross_pre_gate": 1.1667, "gross_final": 1.0},
                    "result": {"target_gross_usd": 130_745.93, "portfolio_value": 87_562.0,
                               "auto_scaled_to_buying_power": True}}])
    orders: list = []
    _wire(monkeypatch, orders, _real_positions({"S000": 130_000.0}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights", _synthetic(), {},
        StubJournal(), logger, report=None, min_stocks_required=5, dry_run=False)
    assert info["reconciled_shortfall"] == pytest.approx(0.7466, abs=2e-3)
    st = load_state(mc)
    assert st["applied_exposure_multiplier"] == pytest.approx(0.640, abs=2e-3)
    assert st["tier_floor_multiplier"] == pytest.approx(0.7466, abs=2e-3)
    assert st["gate_ref"]["rebalance_shortfall"] == pytest.approx(0.7466, abs=2e-3)
    # Calm market: target = cap/g0 x floor = 0.857 x 0.7466 = applied -> hold, no orders.
    assert info["action"] == "hold" and not [o for o in orders if o and o.get("side")]
    # Second pass: nothing to reconcile again.
    info2 = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights", _synthetic(), {},
        StubJournal(), logger, report=None, min_stocks_required=5, dry_run=False)
    assert "reconciled_shortfall" not in info2
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(0.640, abs=2e-3)


# ─────────────── 4. relative band / today-row guard / target weights ────

def _force_gate(monkeypatch, mult):
    monkeypatch.setattr(gu, "compute_live_gate_multiplier",
                        lambda *a, **kw: {"multiplier": mult, "spy_gate": 1.0,
                                          "book_gate": mult, "book_dd": -0.15})


def test_relative_band_is_symmetric(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    for target, expect in ((0.835, "hold"), (0.82, "scaled"), (0.88, "hold"), (0.90, "scaled")):
        _seed(mc, applied_exposure_multiplier=0.857, tier_floor_multiplier=1.0,
              gate_ref={"gross_pre_gate": 1.0, "max_gross_exposure": 1.0,
                        "rebalance_shortfall": 1.0})
        orders: list = []
        _wire(monkeypatch, orders, _real_positions({"S000": 85_700.0}))
        _force_gate(monkeypatch, target)
        info = gu.maybe_daily_gate_update(
            mc, {"combo_config": ComboConfig()}, "direct_weights", _synthetic(), {},
            StubJournal(), logger, report=None, min_stocks_required=5, dry_run=False)
        assert info["action"] == expect, (target, info)
        assert info["band_abs"] == pytest.approx(0.035 * 0.857, abs=1e-4)


def test_today_row_is_ignored_by_the_gate_read():
    """A non-NaN partial bar dated today (ET) must not open or close the
    gate: the read is prior closes only."""
    cfg = ComboConfig()
    data = _synthetic()
    from zoneinfo import ZoneInfo
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    for sym, df in data.items():
        df.loc[today] = df["close"].iloc[-1] * 0.5     # fake -50% intraday print
    held = {s: 1.0 for s in list(data)[:30]}
    gate = gu.compute_live_gate_multiplier(cfg, data, {}, held)
    assert gate is not None and gate["multiplier"] == pytest.approx(1.0)


def test_book_dd_keys_to_last_rebalance_target_weights(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    _seed(mc, applied_exposure_multiplier=1.0, tier_floor_multiplier=1.0,
          gate_ref={"gross_pre_gate": 1.0, "max_gross_exposure": 1.0, "rebalance_shortfall": 1.0},
          history=[{"weights": {"S000": 0.5, "S001": 0.5},
                    "result": {"target_gross_usd": 100_000.0, "portfolio_value": 50_000.0}}])
    seen = {}
    orig = gu.compute_live_gate_multiplier
    def spy(config, sd, md, held):
        seen["held"] = dict(held); return orig(config, sd, md, held)
    monkeypatch.setattr(gu, "compute_live_gate_multiplier", spy)
    orders: list = []
    _wire(monkeypatch, orders, _real_positions({"S000": 90_000.0, "S001": 10_000.0}))
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights", _synthetic(), {},
        StubJournal(), logger, report=None, min_stocks_required=5, dry_run=False)
    assert info["book_dd_weights"] == "target"
    assert seen["held"] == {"S000": 0.5, "S001": 0.5}


# ───────────── 5. scanner heartbeat / min DD / near-miss warning ────────

def _wire_scanner(monkeypatch, equity, last_equity, positions):
    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        if endpoint == "v2/positions":
            return positions
        if endpoint == "v2/account":
            return {"equity": str(equity), "last_equity": str(last_equity)}
        return {"id": "o", "status": "accepted"}
    monkeypatch.setattr(risk_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(risk_mod, "fetch_snapshots", lambda *a, **kw: {}, raising=False)
    monkeypatch.setattr(risk_mod, "poll_order_status", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(risk_mod, "TradeJournal", lambda name: StubJournal())


def test_scanner_writes_heartbeat_min_dd_and_near_miss(tmp_path, monkeypatch, caplog):
    mc = make_mc(tmp_path)
    positions = [{"symbol": "A", "qty": "100", "avg_entry_price": "100",
                  "current_price": "93.5", "market_value": "9350"}]
    caplog.set_level(logging.INFO)
    _wire_scanner(monkeypatch, equity=93_500, last_equity=100_000, positions=positions)
    risk_mod._cutloss_scan_model(mc, logger)          # -6.5% vs anchor: near miss, no tier
    st = load_state(mc)
    assert st["daily_portfolio_start"] == pytest.approx(100_000)
    assert st["cutloss_last_tick_ts"]
    assert st["cutloss_daily_min_dd_pct"] == pytest.approx(-6.5, abs=1e-6)
    assert st["cutloss_near_miss_warned_date"] == st["daily_portfolio_start_date"]
    assert any("NEAR MISS" in r.message for r in caplog.records)
    assert any("session anchor" in r.message for r in caplog.records)
    assert not any("SOFT SCALE" in r.message or "TIER" in r.message
                   for r in caplog.records if r.levelno >= logging.WARNING and "NEAR" not in r.message)
    # Recovery tick: heartbeat advances, session minimum stays.
    _wire_scanner(monkeypatch, equity=97_000, last_equity=100_000, positions=positions)
    risk_mod._cutloss_scan_model(mc, logger)
    st2 = load_state(mc)
    assert st2["cutloss_daily_min_dd_pct"] == pytest.approx(-6.5, abs=1e-6)
    assert st2["cutloss_last_tick_ts"] >= st["cutloss_last_tick_ts"]


def test_scanner_heartbeat_on_empty_book(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    _wire_scanner(monkeypatch, equity=100_000, last_equity=100_000, positions=[])
    risk_mod._cutloss_scan_model(mc, logger)
    assert load_state(mc)["cutloss_last_tick_ts"]


def test_new_scanner_keys_are_scanner_owned():
    from core.state import SCANNER_OWNED_KEYS
    for k in ("cutloss_last_tick_ts", "cutloss_daily_min_dd_pct",
              "cutloss_daily_min_dd_date", "cutloss_near_miss_warned_date"):
        assert k in SCANNER_OWNED_KEYS


# ───────────────────── 6. /ready heartbeat staleness ────────────────────

def test_ready_flags_stale_scanner_heartbeat_in_market_hours(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path, name="combo_v2")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr("core.market.is_market_open", lambda: True)
    monkeypatch.setattr(dashboard, "STATE_DIR", tmp_path)
    monkeypatch.setitem(dashboard.bot_status, "started_at",
                        (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())
    monkeypatch.setitem(dashboard.bot_status, "last_run_status", "success")
    monkeypatch.setitem(dashboard.bot_status, "last_run_at", datetime.now(timezone.utc).isoformat())
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: load_state(mc))
    monkeypatch.setattr(dashboard, "DASHBOARD_AUTH_TOKEN", None)
    # Pin the dashboard clock at 11:00 ET (the open-grace window is clock
    # based — review round 6: this test used to fail before 09:35 ET).
    from zoneinfo import ZoneInfo
    base = datetime.now(ZoneInfo("America/New_York")).replace(hour=11, minute=0, second=0, microsecond=0)

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return base.astimezone(tz) if tz else base.replace(tzinfo=None)
    monkeypatch.setattr(dashboard, "datetime", FakeDT)
    monkeypatch.setitem(dashboard.bot_status, "started_at",
                        (base.astimezone(timezone.utc) - timedelta(hours=2)).isoformat())
    c = dashboard.app.test_client()
    stale = (base.astimezone(timezone.utc) - timedelta(minutes=12)).isoformat()
    _seed(mc, cutloss_last_tick_ts=stale)
    (tmp_path / "x.json").write_text("{}")   # non-empty state dir
    r = c.get("/ready")
    assert r.status_code == 503
    assert any("heartbeat" in p for p in r.get_json()["problems"])
    _seed(mc, cutloss_last_tick_ts=(base.astimezone(timezone.utc) - timedelta(seconds=10)).isoformat())
    r2 = c.get("/ready")
    assert not any("heartbeat" in p for p in r2.get_json()["problems"])


# ───────────────────── 7. config migration (process tiers) ──────────────

def test_migration_enables_process_tiers(monkeypatch):
    import core.config as cfg
    saved = {}
    monkeypatch.setattr(cfg, "save_model_config", lambda c: saved.update(c))
    config = {"cutloss_calibration": "2026-06-riskfix-v1", "slots": [
        {"slot_id": 1, "model": "combo_v2", "enabled": True, "enable_cutloss": True,
         "cutloss_portfolio_stop": -4.0, "cutloss_scale_by_leverage": True, "cutloss_reentry_delay_days": 2},
        {"slot_id": 3, "model": "combo_v2_process", "enabled": True, "enable_cutloss": False,
         "cutloss_portfolio_stop": -4, "cutloss_scale_by_leverage": True, "cutloss_reentry_delay_days": 2},
        {"slot_id": 6, "model": "combo_v2_process", "enabled": False, "enable_cutloss": False,
         "cutloss_scale_by_leverage": True, "cutloss_reentry_delay_days": 2},
    ]}
    out = cfg._migrate_cutloss_calibration(config)
    by_id = {s["slot_id"]: s for s in out["slots"]}
    assert by_id[3]["enable_cutloss"] is True
    assert by_id[6]["enable_cutloss"] is False          # disabled slots untouched
    assert by_id[1]["enable_cutloss"] is True
    assert out["cutloss_calibration"] == cfg.CUTLOSS_CALIBRATION_VERSION == "2026-10-riskfix-v2"
    assert saved
    # Idempotent.
    assert cfg._migrate_cutloss_calibration(out) is out



# ───────────── review round 2 (2026-10-15): regression tests ────────────

def test_reconciliation_composes_onto_a_tier_cut(tmp_path, monkeypatch):
    """MAJOR review finding: a Tier-1 cut after the Sep-29-style rebalance
    lives in BOTH applied (0.857*0.6=0.5143) and the floor (0.6). The
    reconciliation must compose the shortfall onto both, land on 'hold',
    and sell nothing — not overwrite applied and sell the cut twice."""
    mc = make_mc(tmp_path)
    _seed(mc, applied_exposure_multiplier=0.5143, tier_floor_multiplier=0.6,
          gate_ref={"gross_pre_gate": 1.1667, "max_gross_exposure": 1.0},
          history=[{"strategy_diagnostics": {"exposure_multiplier": 1.0,
                                             "gross_pre_gate": 1.1667, "gross_final": 1.0},
                    "result": {"target_gross_usd": 130_745.93, "portfolio_value": 87_562.0,
                               "auto_scaled_to_buying_power": True}}])
    orders: list = []
    _wire(monkeypatch, orders, _real_positions({"S000": 50_000.0}))
    _force_gate(monkeypatch, 1.0)
    info = gu.maybe_daily_gate_update(
        mc, {"combo_config": ComboConfig()}, "direct_weights", _synthetic(), {},
        StubJournal(), logger, report=None, min_stocks_required=5, dry_run=False)
    st = load_state(mc)
    assert st["applied_exposure_multiplier"] == pytest.approx(0.5143 * 0.7466, abs=2e-3)
    assert st["tier_floor_multiplier"] == pytest.approx(0.6 * 0.7466, abs=2e-3)
    assert info["action"] == "hold"
    assert not [o for o in orders if o and o.get("side") == "sell"]


def test_drift_inside_hold_band_moves_applied_but_not_the_floor(tmp_path):
    """Achieved 96% of design with NO funding evidence (no auto-scale,
    fills complete): applied follows the truth, floor stays 1.0 so the
    daily pass may top up toward design (validated behaviour)."""
    mc = make_mc(tmp_path)
    diag = {"exposure_multiplier": 1.0, "gross_pre_gate": 1.1667, "gross_final": 1.0}
    gu.set_gate_state_at_rebalance(
        mc, diag, ComboConfig(), logger,
        result={"reconciliation": {"book": {"achieved_gross_usd": 192_000.0},
                                   "flow": {"fill_rate_notional": 1.0, "n_rejected": 0,
                                            "n_unfilled_cancelled": 0}},
                "target_gross_usd": 200_000.0, "portfolio_value": 100_000.0,
                "auto_scaled_to_buying_power": False},
        leverage=2.0)
    st = load_state(mc)
    assert st["applied_exposure_multiplier"] == pytest.approx(0.96 / 1.1667, abs=2e-3)
    assert st["tier_floor_multiplier"] == 1.0
    assert st["gate_ref"]["rebalance_shortfall"] == pytest.approx(0.96, abs=1e-3)


def test_last_funded_rebalance_skips_dry_and_closed_entries():
    hist = [
        {"result": {"target_gross_usd": 100.0, "portfolio_value": 50.0}, "weights": {"A": 1.0}},
        {"result": {"dry_run": True}, "weights": {"B": 1.0}},
        {"result": {"dry_run": False, "skipped_market_closed": True}, "weights": {"C": 1.0}},
    ]
    assert gu._last_funded_rebalance(hist)["weights"] == {"A": 1.0}
    assert gu._last_funded_rebalance([]) == {}


def test_preflight_forced_upsize_is_still_placed(tmp_path, monkeypatch):
    """Reviewer probe: A 90k->101k (12% up), B 95k->79k, C new 20k, bp 10k.
    buy_need 31k -> scale 0.3226. A's scaled increment (3,548) is under the
    10% band but its budget was reserved: it must still be bought."""
    rb, posted = _run_rebalance(
        tmp_path, monkeypatch, equity=100_000, bp=10_000,
        positions={"A": 90_000, "B": 95_000},
        weights={"A": 0.505, "B": 0.395, "C": 0.10})
    assert rb["auto_scaled_to_buying_power"] is True
    buys = {o["symbol"]: float(o["notional"]) for o in posted if o.get("side") == "buy"}
    assert "A" in buys and buys["A"] == pytest.approx(11_000 * rb["buy_scale"], rel=0.01)
    assert sum(buys.values()) == pytest.approx(10_000, rel=0.01)


def test_scanner_heartbeat_on_trip_flag_day(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = datetime.now().strftime("%Y-%m-%d")
    _seed(mc, portfolio_stop_tripped_date=today)
    _wire_scanner(monkeypatch, equity=100_000, last_equity=100_000, positions=[])
    risk_mod._cutloss_scan_model(mc, logger)
    assert load_state(mc)["cutloss_last_tick_ts"]


def test_ready_grace_after_session_open(tmp_path, monkeypatch):
    """At 09:31 ET the last heartbeat is legitimately ~17h old: no alarm
    within 5 minutes of the open."""
    import dashboard
    from zoneinfo import ZoneInfo
    mc = make_mc(tmp_path, name="combo_v2")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr("core.market.is_market_open", lambda: True)
    monkeypatch.setattr(dashboard, "STATE_DIR", tmp_path)
    monkeypatch.setitem(dashboard.bot_status, "started_at",
                        (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())
    monkeypatch.setitem(dashboard.bot_status, "last_run_status", "success")
    monkeypatch.setitem(dashboard.bot_status, "last_run_at", datetime.now(timezone.utc).isoformat())
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: load_state(mc))
    monkeypatch.setattr(dashboard, "DASHBOARD_AUTH_TOKEN", None)

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime.now(ZoneInfo("America/New_York")).replace(hour=9, minute=31, second=0, microsecond=0)
            return base.astimezone(tz) if tz else base.replace(tzinfo=None)
    monkeypatch.setattr(dashboard, "datetime", FakeDT)
    (tmp_path / "x.json").write_text("{}")
    _seed(mc, cutloss_last_tick_ts=(datetime.now(timezone.utc) - timedelta(hours=17)).isoformat())
    r = dashboard.app.test_client().get("/ready")
    assert not any("heartbeat" in p for p in r.get_json()["problems"])
