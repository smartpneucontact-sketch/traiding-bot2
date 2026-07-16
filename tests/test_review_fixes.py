"""Unit tests for the 2026-07 review fixes.

Covers:
  1. (MAJOR) day-start book anchor refreshed after a funded rebalance:
     pre-rebalance anchor X, rebalance achieves gross Y → the tier
     scaler's anchor becomes Y for the same day (stop_grid.py scales
     TIER_F off the POST-rebalance book), with the target-notional
     fallback when reconciliation failed; the anchor keys survive the
     pipeline's merged state save; the scanner never re-anchors same-day.
  2. hard/trailing stop path places every sell BEFORE any snapshot/poll
     instrumentation call (risk reduction first), and still wires fill
     proceeds + the sold-today ledger into redistribution.
  3. `_instrument_and_journal_sells` enforces its wall-clock budget HARD
     (slow polls can't stretch it; every record is still journalled) and
     `poll_order_status` honors max_wait as wall-clock time.
  4. reconciliation book diff normalizes symbol forms (BRK-B vs BRK.B)
     so class shares aren't flagged missing + unexpected simultaneously.

Run:  /opt/anaconda3/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.config import ModelConfig
from core.journal import TradeRecord
import core.orders as orders_mod
import core.risk as risk_mod
import core.state as state_mod

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
        cutloss_portfolio_stop=-3.0,   # → -6% effective at 2x
        cutloss_hard_stop=None,
        cutloss_trailing_stop=None,
    )
    defaults.update(kw)
    return ModelConfig(**defaults)


class StubJournal:
    def __init__(self, *a, **kw):
        self.records = []
        self.jsonl_path = Path("/dev/null")

    def log_trade(self, record):
        self.records.append(record)


def make_record(sym: str, order_id: str | None = None,
                status: str = "accepted") -> TradeRecord:
    return TradeRecord(
        trade_id=f"t_{sym}", run_id="r1", model="combo_v2",
        timestamp="2026-07-16T13:35:00+00:00", symbol=sym, side="sell",
        action="portfolio_stop", order_type="market", time_in_force="day",
        notional_usd=0.0, order_id=order_id or f"oid_{sym}",
        order_status=status,
    )


TODAY = datetime.now().strftime("%Y-%m-%d")


# ═════ 1. day-start book anchor refresh after rebalance (MAJOR) ═══════════

def _seed_anchor(mc, anchor: float) -> None:
    mc.state_path.write_text(json.dumps({
        "daily_book_start": anchor,
        "daily_book_start_date": TODAY,
        "history": [],
    }))


def test_anchor_becomes_achieved_gross_after_rebalance(tmp_path):
    """Pre-rebalance anchor X=$150k (the 09:30 overnight book); the 09:35
    rebalance achieves gross Y=$90k → the same-day anchor must become Y,
    so Tier-1/2 targets (f × day-start book) match stop_grid.py's
    post-rebalance-book semantics."""
    mc = make_mc(tmp_path)
    _seed_anchor(mc, 150000.0)

    rb_data = {
        "target_gross_usd": 80000.0,  # must NOT win over the measured book
        "reconciliation": {"book": {"achieved_gross_usd": 90000.0}},
    }
    out = risk_mod.refresh_daily_book_anchor_after_rebalance(mc, rb_data, logger)

    assert out == pytest.approx(90000.0)
    persisted = json.loads(mc.state_path.read_text())
    assert persisted["daily_book_start"] == pytest.approx(90000.0)
    assert persisted["daily_book_start_date"] == TODAY


def test_anchor_falls_back_to_target_notional_sum(tmp_path):
    """Book-level reconciliation failed (book=None) → fall back to the
    summed target notionals rb_data["target_gross_usd"]."""
    mc = make_mc(tmp_path)
    _seed_anchor(mc, 150000.0)

    for rb_data in (
        {"reconciliation": {"book": None}, "target_gross_usd": 80000.0},
        {"target_gross_usd": 80000.0},                       # no recon at all
        {"reconciliation": {"book": {"achieved_gross_usd": 0.0}},
         "target_gross_usd": 80000.0},                       # zero gross
    ):
        _seed_anchor(mc, 150000.0)
        out = risk_mod.refresh_daily_book_anchor_after_rebalance(
            mc, rb_data, logger)
        assert out == pytest.approx(80000.0)
        persisted = json.loads(mc.state_path.read_text())
        assert persisted["daily_book_start"] == pytest.approx(80000.0)
        assert persisted["daily_book_start_date"] == TODAY


def test_anchor_kept_when_rebalance_data_unusable(tmp_path):
    """No usable gross anywhere → keep the morning anchor untouched
    (never write garbage into the tier base)."""
    mc = make_mc(tmp_path)
    for rb_data in (None, {}, {"reconciliation": {}},
                    {"target_gross_usd": "not-a-number"}):
        _seed_anchor(mc, 150000.0)
        out = risk_mod.refresh_daily_book_anchor_after_rebalance(
            mc, rb_data, logger)
        assert out is None
        persisted = json.loads(mc.state_path.read_text())
        assert persisted["daily_book_start"] == pytest.approx(150000.0)
        assert persisted["daily_book_start_date"] == TODAY


def test_scanner_does_not_reanchor_after_rebalance_refresh(tmp_path, monkeypatch):
    """After the post-rebalance refresh sets Y, the 60s scanner's same-day
    guard must keep Y (not re-anchor to the drifting current book)."""
    mc = make_mc(tmp_path)
    _seed_anchor(mc, 150000.0)
    risk_mod.refresh_daily_book_anchor_after_rebalance(
        mc, {"reconciliation": {"book": {"achieved_gross_usd": 90000.0}}},
        logger)

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "GET" and path == "v2/positions":
            return [{"symbol": "AAA", "qty": "880", "avg_entry_price": "100",
                     "current_price": "100", "market_value": "88000"}]
        if method == "GET" and path == "v2/account":
            return {"equity": "100000", "last_equity": "100000"}  # flat day
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
    risk_mod._cutloss_scan_model(mc, logger)

    persisted = json.loads(mc.state_path.read_text())
    assert persisted["daily_book_start"] == pytest.approx(90000.0)
    assert persisted["daily_book_start_date"] == TODAY


def test_anchor_keys_survive_pipeline_merged_save(tmp_path):
    """The pipeline saves a state dict it loaded before the scanner (or the
    post-rebalance refresh) wrote the book anchor. merge_scanner_state must
    adopt the disk anchor instead of dropping it — daily_book_start /
    daily_book_start_date are scanner-owned keys."""
    assert "daily_book_start" in state_mod.SCANNER_OWNED_KEYS
    assert "daily_book_start_date" in state_mod.SCANNER_OWNED_KEYS

    mc = make_mc(tmp_path)
    mc.state_path.write_text(json.dumps({
        "daily_book_start": 90000.0,
        "daily_book_start_date": TODAY,
        "last_rebalance": None,
    }))
    pipeline_state = {"history": [], "run_count": 5}  # pre-anchor snapshot
    merged = state_mod.merge_scanner_state(pipeline_state, mc)
    assert merged["daily_book_start"] == pytest.approx(90000.0)
    assert merged["daily_book_start_date"] == TODAY
    assert merged["run_count"] == 5


# ═════ 2. hard/trailing stops: sells placed before instrumentation ════════

def test_stop_path_places_all_sells_before_snapshot_and_poll(tmp_path, monkeypatch):
    """The per-position stop path used to fetch snapshots BEFORE placing
    the sells (up to ~15s of data-API latency on a risk-reduction path).
    Now: every close order first, snapshot + poll strictly after — and
    fill proceeds + the sold-today ledger still reach redistribution."""
    mc = make_mc(tmp_path, cutloss_trailing_stop=-5.0)
    events: list[tuple] = []
    journal = StubJournal()
    redistribute_calls: list[tuple] = []

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "GET" and path == "v2/positions":
            return [
                {"symbol": s, "qty": "10", "avg_entry_price": "100",
                 "current_price": "94", "market_value": "940"}
                for s in ("AAA", "BBB")
            ]
        if method == "GET" and path == "v2/account":
            return {"equity": "100000", "last_equity": "100000"}  # no tier
        if method == "DELETE" and path.startswith("v2/positions/"):
            sym = path.rsplit("/", 1)[1]
            events.append(("order", sym))
            return {"id": f"oid_{sym}", "status": "accepted"}
        raise AssertionError(f"unexpected call {method} {path}")

    def fake_snapshots(syms, mc_, lg=None):
        events.append(("snapshot", tuple(sorted(syms))))
        return {s: {"arrival": 94.0, "day_open": 95.0, "prev_close": 96.0}
                for s in syms}

    def fake_poll(oid, mc_, lg, **kw):
        events.append(("poll", oid))
        return {"status": "filled", "filled_qty": "10",
                "filled_avg_price": "94.5"}

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(risk_mod, "fetch_snapshots", fake_snapshots)
    monkeypatch.setattr(risk_mod, "poll_order_status", fake_poll)
    monkeypatch.setattr(risk_mod, "TradeJournal", lambda name: journal)
    monkeypatch.setattr(
        risk_mod, "_redistribute_after_cutloss",
        lambda mc_, sold, lg, top_n=20, proceeds=None:
            redistribute_calls.append((sorted(sold), proceeds)),
    )

    risk_mod._cutloss_scan_model(mc, logger)

    # Risk reduction first: both DELETE closes before ANY instrumentation.
    kinds = [e[0] for e in events]
    assert kinds[:2] == ["order", "order"], events
    assert "snapshot" in kinds and "poll" in kinds
    assert kinds.index("snapshot") >= 2

    # Instrumentation still lands: fills, backfilled notional, slippage
    # (post-placement reference prices — a few seconds of skew, sell sign).
    assert len(journal.records) == 2
    for r in journal.records:
        assert r.action == "trailing_stop"
        assert r.order_status == "filled"
        assert r.fill_price == 94.5
        assert r.notional_usd == pytest.approx(945.0)
        assert r.reference_open == 95.0
        assert r.slippage_bps == pytest.approx(52.63, abs=0.01)

    # Proceeds from actual fills (2 × $945) and the persisted ledger.
    assert redistribute_calls == [(["AAA", "BBB"], pytest.approx(1890.0))]
    persisted = json.loads(mc.state_path.read_text())
    assert persisted["cutloss_sold_today"]["symbols"] == ["AAA", "BBB"]


def test_stop_path_uses_mv_estimate_when_fill_unknown(tmp_path, monkeypatch):
    """A poll that never resolves a fill must not zero the redistribution
    budget — fall back to the position's last market value."""
    mc = make_mc(tmp_path, cutloss_trailing_stop=-5.0)
    redistribute_calls: list[tuple] = []

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "GET" and path == "v2/positions":
            return [{"symbol": "AAA", "qty": "10", "avg_entry_price": "100",
                     "current_price": "94", "market_value": "940"}]
        if method == "GET" and path == "v2/account":
            return {"equity": "100000", "last_equity": "100000"}
        if method == "DELETE" and path == "v2/positions/AAA":
            return {"id": "oid_AAA", "status": "accepted"}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(risk_mod, "fetch_snapshots",
                        lambda syms, mc_, lg=None: {})
    monkeypatch.setattr(risk_mod, "poll_order_status",
                        lambda oid, mc_, lg, **kw: {"status": "accepted"})
    monkeypatch.setattr(risk_mod, "TradeJournal", StubJournal)
    monkeypatch.setattr(
        risk_mod, "_redistribute_after_cutloss",
        lambda mc_, sold, lg, top_n=20, proceeds=None:
            redistribute_calls.append((list(sold), proceeds)),
    )

    risk_mod._cutloss_scan_model(mc, logger)
    assert redistribute_calls == [(["AAA"], pytest.approx(940.0))]


# ═════ 3. hard instrumentation budget ═════════════════════════════════════

def test_instrument_budget_is_hard_and_all_records_journalled(tmp_path, monkeypatch):
    """30 records × 0.2s-slow polls against a 1.0s budget: wall time must
    stay under budget + one-poll slack, polling must stop early, and every
    record must still be journalled (with its submit-time status)."""
    mc = make_mc(tmp_path)
    journal = StubJournal()
    polls = []

    monkeypatch.setattr(risk_mod, "fetch_snapshots",
                        lambda syms, mc_, lg=None: {})

    def slow_poll(oid, mc_, lg, **kw):
        polls.append(oid)
        time.sleep(0.2)  # a poll that ignores max_wait (slow HTTP)
        return {"status": "filled", "filled_qty": "1",
                "filled_avg_price": "100"}

    monkeypatch.setattr(risk_mod, "poll_order_status", slow_poll)

    records = [make_record(f"S{i:02d}") for i in range(30)]
    budget = 1.0
    t0 = time.monotonic()
    risk_mod._instrument_and_journal_sells(mc, records, journal, logger,
                                           total_budget=budget)
    wall = time.monotonic() - t0

    # Hard budget: overshoot bounded by the one poll in flight at deadline.
    assert wall < budget + 0.5, f"budget not enforced: wall={wall:.2f}s"
    assert 1 <= len(polls) < 30          # stopped polling at the deadline
    assert len(journal.records) == 30    # ...but journalled everything
    # Un-polled records keep their submit-time status; polled ones filled.
    statuses = {r.order_status for r in journal.records}
    assert "filled" in statuses and "accepted" in statuses


def test_poll_order_status_max_wait_is_wall_clock(tmp_path, monkeypatch):
    """max_wait must include HTTP time (the old accounting summed only the
    sleeps), and final_fetch=False must not add a trailing request."""
    mc = make_mc(tmp_path)
    calls = []

    def slow_request(method, endpoint, mc_, data=None, logger=None):
        calls.append(endpoint)
        time.sleep(0.25)
        return {"status": "new", "id": "oid1"}

    monkeypatch.setattr(orders_mod, "alpaca_request", slow_request)

    t0 = time.monotonic()
    out = orders_mod.poll_order_status("oid1", mc, logger, max_wait=0.6,
                                       interval=0.1, final_fetch=False)
    wall = time.monotonic() - t0

    assert out["status"] == "new"        # last polled state, not "unknown"
    assert wall < 1.5, f"wall-clock max_wait not honored: {wall:.2f}s"
    n_without_final = len(calls)

    # Default final_fetch=True adds exactly one trailing status request.
    calls.clear()
    orders_mod.poll_order_status("oid1", mc, logger, max_wait=0.6,
                                 interval=0.1)
    assert len(calls) == n_without_final + 1


def test_poll_order_status_returns_terminal_immediately(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    calls = []

    def fast_request(method, endpoint, mc_, data=None, logger=None):
        calls.append(endpoint)
        return {"status": "filled", "id": "oid1", "filled_qty": "10"}

    monkeypatch.setattr(orders_mod, "alpaca_request", fast_request)
    out = orders_mod.poll_order_status("oid1", mc, logger, max_wait=8.0)
    assert out["status"] == "filled"
    assert len(calls) == 1


# ═════ 4. reconciliation symbol-form normalization ════════════════════════

def test_reconciliation_normalizes_class_share_symbol_forms(tmp_path, monkeypatch):
    """Target BRK-B (universe dash form) achieved as BRK.B (Alpaca dot
    form) must reconcile as the SAME position — not one missing plus one
    unexpected."""
    from core.run_report import RunReport

    mc = make_mc(tmp_path)
    get_positions_calls = {"n": 0}

    def fake_get_positions(mc_, lg):
        get_positions_calls["n"] += 1
        if get_positions_calls["n"] == 1:
            return {}  # nothing held before the rebalance
        # Reconciliation call: Alpaca reports the fill in dot form.
        return {"BRK.B": {"market_value": 5000.0}}

    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": "100000", "cash": "100000",
        "buying_power": "400000",
    })
    monkeypatch.setattr(orders_mod, "get_positions", fake_get_positions)

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "POST" and path == "v2/orders":
            assert data["symbol"] == "BRK.B"  # dot form on the wire
            return {"id": "oid_buy", "status": "accepted"}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(orders_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, mc_, lg, **kw: {"status": "filled",
                                                    "filled_qty": "12",
                                                    "filled_avg_price": "410"})
    monkeypatch.setattr(orders_mod, "fetch_snapshots",
                        lambda syms, mc_, lg=None: {})
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)

    rb = orders_mod.rebalance_portfolio(
        target_symbols=["BRK-B"], rankings=[("BRK-B", 1.0)],
        mc=mc, journal=StubJournal(), logger=logger, report=RunReport(),
        dry_run=False, target_weights={"BRK-B": 0.05},
    )

    book = rb["reconciliation"]["book"]
    assert book["missing_positions"] == []
    assert book["unexpected_positions"] == []
    assert book["achieved_gross_usd"] == pytest.approx(5000.0)
    # One deviation row, keyed by the target's own spelling.
    assert len(book["top_deviations"]) == 1
    dev = book["top_deviations"][0]
    assert dev["symbol"] == "BRK-B"
    assert dev["target_usd"] == pytest.approx(10000.0)   # 100k × 0.05 × 2x
    assert dev["achieved_usd"] == pytest.approx(5000.0)
    # target_gross_usd is exposed for the runner's anchor-refresh fallback.
    assert rb["target_gross_usd"] == pytest.approx(10000.0)


def test_reconciliation_still_flags_genuinely_missing_and_unexpected(tmp_path, monkeypatch):
    """Normalization must not mask real book divergence: an unfilled
    target stays missing; a stray holding stays unexpected."""
    from core.run_report import RunReport

    mc = make_mc(tmp_path)
    get_positions_calls = {"n": 0}

    def fake_get_positions(mc_, lg):
        get_positions_calls["n"] += 1
        if get_positions_calls["n"] == 1:
            return {}
        return {"STRAY": {"market_value": 1234.0}}  # target AAA never filled

    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": "100000", "cash": "100000",
        "buying_power": "400000",
    })
    monkeypatch.setattr(orders_mod, "get_positions", fake_get_positions)
    monkeypatch.setattr(orders_mod, "alpaca_request",
                        lambda method, path, mc_, data=None, logger=None:
                        {"id": "oid", "status": "accepted"})
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, mc_, lg, **kw: {"status": "accepted"})
    monkeypatch.setattr(orders_mod, "fetch_snapshots",
                        lambda syms, mc_, lg=None: {})
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)

    rb = orders_mod.rebalance_portfolio(
        target_symbols=["AAA"], rankings=[("AAA", 1.0)],
        mc=mc, journal=StubJournal(), logger=logger, report=RunReport(),
        dry_run=False, target_weights={"AAA": 0.05},
    )
    book = rb["reconciliation"]["book"]
    assert book["missing_positions"] == ["AAA"]
    assert book["unexpected_positions"] == ["STRAY"]
