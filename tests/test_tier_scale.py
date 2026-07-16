"""Unit tests for the 2026-07 tier-scale base fix + stop instrumentation.

The 2026-07-15 incident: `_soft_scale_portfolio` targeted
`equity × target_exposure`, so Tier 1 on a ~$150k gross / ~$79k equity
book sold ~$98k where the validated rule (validation/stop_grid.py,
TIER_F={1:0.6, 2:0.3, 3:0.0} — fractions of the MORNING book) sells
~$60k. The same day's journal rows were blind (order_status=pending_new,
fill_price=None).

Covers:
  - Tier 1 sells down to 60% of the DAY-START book, not 60% of equity
  - sequential Tier1→Tier2 same day: total sold = 0.7 × day-start book
    (each tier anchored to the morning book, not the already-scaled one)
  - mid-day restart: missing daily_book_start initializes from current
    positions; an existing same-day anchor is never overwritten
  - fill-poll + snapshot instrumentation lands on journal rows; snapshot
    failure leaves fields None but orders are still placed
  - risk reduction first: every sell order is placed before any
    snapshot/poll instrumentation call (soft-scale and liquidation)

Run:  /opt/anaconda3/bin/python3 -m pytest tests/ -v
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

import pytest

from core.config import ModelConfig
import core.risk as risk_mod

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

    def log_trade(self, record):
        self.records.append(record)


def make_positions(mvs: dict[str, float]) -> list[dict]:
    """Synthetic Alpaca position rows; price 100 so qty = mv / 100."""
    return [{"symbol": s, "qty": str(mv / 100.0), "avg_entry_price": "100",
             "current_price": "100", "market_value": str(mv)}
            for s, mv in mvs.items()]


def _wire_soft_scale(monkeypatch, *, events, orders, journal,
                     snapshots=None, snapshot_raises=False,
                     fill_price="99.5"):
    """Mock the network surface of _soft_scale_portfolio / _liquidate_all,
    recording every call into `events` to assert ordering."""

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "POST" and path == "v2/orders":
            events.append(("order", data["symbol"]))
            orders.append(dict(data))
            return {"id": f"oid_{data['symbol']}", "status": "accepted"}
        if method == "DELETE" and path.startswith("v2/positions/"):
            sym = path.rsplit("/", 1)[1]
            events.append(("order", sym))
            orders.append({"symbol": sym, "side": "sell", "close": True})
            return {"id": f"oid_{sym}", "status": "accepted"}
        raise AssertionError(f"unexpected call {method} {path}")

    def fake_snapshots(syms, mc_, lg=None):
        events.append(("snapshot", tuple(syms)))
        if snapshot_raises:
            raise RuntimeError("data API down")
        return dict(snapshots or {})

    def fake_poll(oid, mc_, lg, max_wait=8.0, interval=0.5,
                  final_fetch=True):
        events.append(("poll", oid))
        return {"status": "filled", "filled_qty": "10",
                "filled_avg_price": fill_price}

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(risk_mod, "fetch_snapshots", fake_snapshots)
    monkeypatch.setattr(risk_mod, "poll_order_status", fake_poll)
    monkeypatch.setattr(risk_mod, "TradeJournal", lambda name: journal)


# ───── (a) Tier 1 scales to 0.6 × DAY-START BOOK, not 0.6 × equity ───────

def test_tier1_sells_60k_of_150k_book_not_98k(tmp_path, monkeypatch):
    """The 2026-07-15 replay through the full scan: book $150k gross,
    equity $79k, DD in the Tier-1 band. Target = 0.6 × 150k = $90k, so
    $60k is sold pro-rata — NOT the ~$98k the equity-based rule sold
    (150k − 0.6 × 79k ≈ $102.6k excess, ~$98k filled live)."""
    mc = make_mc(tmp_path)
    events, orders = [], []
    journal = StubJournal()

    positions = make_positions({"AAA": 75000.0, "BBB": 45000.0,
                                "CCC": 30000.0})

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "GET" and path == "v2/positions":
            return positions
        if method == "GET" and path == "v2/account":
            # DD = 79/85 − 1 = −7.06%: inside Tier 1 (−6%), above Tier 2
            # (−10%) at the −3% configured / 2x-levered thresholds.
            return {"equity": "79000", "last_equity": "85000"}
        if method == "POST" and path == "v2/orders":
            orders.append(dict(data))
            return {"id": f"oid_{data['symbol']}", "status": "accepted"}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(risk_mod, "fetch_snapshots",
                        lambda syms, mc_, lg=None: {})
    monkeypatch.setattr(risk_mod, "poll_order_status",
                        lambda oid, mc_, lg, **kw: {"status": "filled"})
    monkeypatch.setattr(risk_mod, "TradeJournal", lambda name: journal)

    risk_mod._cutloss_scan_model(mc, logger)

    total_sold = sum(float(o["notional"]) for o in orders)
    assert total_sold == pytest.approx(60000.0, rel=1e-6)
    assert total_sold < 90000.0  # the equity-based rule would sell ~$98-102k
    # pro-rata: 40% of every position
    by_sym = {o["symbol"]: float(o["notional"]) for o in orders}
    assert by_sym["AAA"] == pytest.approx(30000.0)
    assert by_sym["BBB"] == pytest.approx(18000.0)
    assert by_sym["CCC"] == pytest.approx(12000.0)
    # first scan of the day anchored the book at the pre-sale gross
    persisted = json.loads(mc.state_path.read_text())
    assert persisted["daily_book_start"] == pytest.approx(150000.0)


# ───── (b) sequential same-day Tier1 → Tier2 = 0.7 × morning book ────────

def test_sequential_tiers_scale_against_morning_book(tmp_path, monkeypatch):
    """stop_grid crossings do `w = w_morning * f`: Tier 1 sells 0.4 × book,
    a later Tier 2 sells down to 0.3 × the SAME morning book (another
    0.3 × book) — total 0.7 × book, not 0.3 of the already-scaled book."""
    mc = make_mc(tmp_path)
    events, orders = [], []
    journal = StubJournal()
    _wire_soft_scale(monkeypatch, events=events, orders=orders,
                     journal=journal)

    book = 150000.0
    morning = make_positions({"AAA": 75000.0, "BBB": 45000.0,
                              "CCC": 30000.0})
    n1 = risk_mod._soft_scale_portfolio(
        mc, morning, book, book_fraction=0.60,
        tier="Tier1", dd_pct=-7.0, logger=logger)
    tier1_sold = sum(float(o["notional"]) for o in orders)
    assert n1 == 3
    assert tier1_sold == pytest.approx(60000.0)

    # book now scaled to 0.6 × morning; Tier 2 fires later the same day
    scaled = make_positions({"AAA": 45000.0, "BBB": 27000.0,
                             "CCC": 18000.0})
    orders.clear()
    n2 = risk_mod._soft_scale_portfolio(
        mc, scaled, book, book_fraction=0.30,
        tier="Tier2", dd_pct=-11.0, logger=logger)
    tier2_sold = sum(float(o["notional"]) for o in orders)
    assert n2 == 3
    # 0.6×book − 0.3×book = $45k (an already-scaled base would sell $63k)
    assert tier2_sold == pytest.approx(45000.0)
    assert tier1_sold + tier2_sold == pytest.approx(0.7 * book)

    # idempotence: re-scan at the same tier target sells nothing more
    post_tier2 = make_positions({"AAA": 22500.0, "BBB": 13500.0,
                                 "CCC": 9000.0})
    orders.clear()
    assert risk_mod._soft_scale_portfolio(
        mc, post_tier2, book, book_fraction=0.30,
        tier="Tier2", dd_pct=-11.0, logger=logger) == 0
    assert orders == []


# ───── (c) mid-day restart: book anchor init + no same-day overwrite ─────

def test_restart_midday_initializes_book_anchor_from_positions(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    positions = make_positions({"AAA": 90000.0, "BBB": 60000.0})

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "GET" and path == "v2/positions":
            return positions
        if method == "GET" and path == "v2/account":
            return {"equity": "100000", "last_equity": "100000"}  # flat day
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(risk_mod, "alpaca_request", fake_alpaca)

    # deploy restart mid-day: state file has today's equity anchor but no
    # book anchor → first scan initializes it from current positions
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    mc.state_path.write_text(json.dumps({
        "daily_portfolio_start": 100000.0,
        "daily_portfolio_start_date": today,
    }))
    risk_mod._cutloss_scan_model(mc, logger)
    persisted = json.loads(mc.state_path.read_text())
    assert persisted["daily_book_start"] == pytest.approx(150000.0)
    assert persisted["daily_book_start_date"] == today

    # a same-day anchor is never overwritten by a shrunken (post-tier) book
    positions[:] = make_positions({"AAA": 54000.0, "BBB": 36000.0})
    risk_mod._cutloss_scan_model(mc, logger)
    persisted = json.loads(mc.state_path.read_text())
    assert persisted["daily_book_start"] == pytest.approx(150000.0)


# ───── (d) fill-poll + snapshot instrumentation on journal rows ──────────

def test_soft_scale_rows_carry_fill_and_slippage(tmp_path, monkeypatch):
    """The 2026-07-15 rows logged pending_new / fill_price=None. Rows must
    now carry the polled fill and the snapshot decision prices, with the
    sell sign convention (fill below the open = positive bps = cost)."""
    mc = make_mc(tmp_path)
    events, orders = [], []
    journal = StubJournal()
    _wire_soft_scale(
        monkeypatch, events=events, orders=orders, journal=journal,
        snapshots={"AAA": {"arrival": 100.0, "day_open": 101.0,
                           "prev_close": 99.0}},
        fill_price="99.5",
    )

    positions = make_positions({"AAA": 150000.0})
    n = risk_mod._soft_scale_portfolio(
        mc, positions, 150000.0, book_fraction=0.60,
        tier="Tier1", dd_pct=-7.0, logger=logger)

    assert n == 1 and len(journal.records) == 1
    r = journal.records[0]
    assert r.order_status == "filled"
    assert r.fill_price == 99.5
    assert r.shares == 10.0
    assert r.decision_price == 100.0
    assert r.reference_open == 101.0
    assert r.prev_close == 99.0
    # sell sign −1: 99.5 vs 101 open → −1×(99.5/101 − 1)×1e4 = +148.51 cost
    assert r.slippage_bps == pytest.approx(148.51, abs=0.01)
    assert r.slippage_vs_arrival_bps == pytest.approx(50.0, abs=0.01)


def test_snapshot_failure_still_places_orders_fields_none(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    events, orders = [], []
    journal = StubJournal()
    _wire_soft_scale(monkeypatch, events=events, orders=orders,
                     journal=journal, snapshot_raises=True)

    positions = make_positions({"AAA": 90000.0, "BBB": 60000.0})
    n = risk_mod._soft_scale_portfolio(
        mc, positions, 150000.0, book_fraction=0.60,
        tier="Tier1", dd_pct=-7.0, logger=logger)

    assert n == 2
    assert len(orders) == 2                     # orders still placed
    assert len(journal.records) == 2            # rows still journalled
    for r in journal.records:
        assert r.fill_price == 99.5             # poll still ran
        for f in ("decision_price", "reference_open", "prev_close",
                  "slippage_bps", "slippage_vs_arrival_bps"):
            assert getattr(r, f) is None        # snapshot fields degrade


# ───── (e) risk reduction first: orders before any instrumentation ───────

def _assert_orders_first(events, n_orders):
    kinds = [e[0] for e in events]
    assert kinds[:n_orders] == ["order"] * n_orders, (
        f"instrumentation ran before all orders were placed: {events}")
    assert "poll" in kinds and "snapshot" in kinds


def test_soft_scale_places_all_orders_before_poll_and_snapshot(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    events, orders = [], []
    _wire_soft_scale(monkeypatch, events=events, orders=orders,
                     journal=StubJournal())
    positions = make_positions({"AAA": 60000.0, "BBB": 50000.0,
                                "CCC": 40000.0})
    risk_mod._soft_scale_portfolio(
        mc, positions, 150000.0, book_fraction=0.60,
        tier="Tier1", dd_pct=-7.0, logger=logger)
    _assert_orders_first(events, 3)


def test_liquidate_all_places_all_orders_before_poll_and_snapshot(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    events, orders = [], []
    journal = StubJournal()
    _wire_soft_scale(monkeypatch, events=events, orders=orders,
                     journal=journal)
    positions = make_positions({"AAA": 60000.0, "BBB": 50000.0})

    risk_mod._liquidate_all(mc, positions, "portfolio_stop", logger)

    _assert_orders_first(events, 2)
    assert len(journal.records) == 2
    for r in journal.records:
        assert r.action == "portfolio_stop"
        assert r.fill_price == 99.5
        # DELETE /v2/positions returns no notional — backfilled from fill
        assert r.notional_usd == pytest.approx(10 * 99.5)
