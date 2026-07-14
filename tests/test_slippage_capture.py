"""Unit tests for Phase A0 decision-price capture (slippage measurement).

Covers:
  - alpaca.fetch_snapshots: parsing, class-share symbol mapping, missing
    symbols/pieces, 100-symbol chunking, error -> {} contract
  - TradeRecord: the five new Optional fields roundtrip None through JSONL
  - TradeJournal CSV rollover: stale header renamed to .pre-slippage
    (numeric suffix when taken), matching header appends cleanly
  - rebalance_portfolio: snapshot prices land on journal rows; slippage
    sign convention (+ = cost for both sides); snapshot failure is non-fatal

Run:  /opt/anaconda3/bin/python3 -m pytest tests/ -v
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import fields as dc_fields
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import requests

import core.alpaca as alpaca_mod
import core.orders as orders_mod
from core.config import ModelConfig
from core.journal import TradeJournal, TradeRecord

logger = logging.getLogger("test")

NEW_FIELDS = ["decision_price", "reference_open", "prev_close",
              "slippage_bps", "slippage_vs_arrival_bps"]


def make_mc(tmp_path, **kw) -> ModelConfig:
    defaults = dict(
        name="combo_v2",
        model_path=Path("/nonexistent/model.pkl"),
        feature_version="combo",
        alpaca_key="k",
        alpaca_secret="s",
        state_path=tmp_path / "state.json",
        target_leverage=2.0,
    )
    defaults.update(kw)
    return ModelConfig(**defaults)


def make_record(model: str, **kw) -> TradeRecord:
    defaults = dict(
        trade_id="t1", run_id="r1", model=model,
        timestamp="2026-07-14T13:35:00+00:00", symbol="AAA", side="buy",
        action="new_position", order_type="market", time_in_force="day",
        notional_usd=1000.0,
    )
    defaults.update(kw)
    return TradeRecord(**defaults)


class StubJournal:
    def __init__(self, *a, **kw):
        self.records = []
        self.jsonl_path = Path("/dev/null")  # rebalance summary logs it

    def log_trade(self, record):
        self.records.append(record)


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


# ───── fetch_snapshots ───────────────────────────────────────────────────

def test_fetch_snapshots_parses_and_maps_class_shares(tmp_path, monkeypatch):
    """Response keys arrive in Alpaca dot form; result must be keyed by the
    caller's dash-form symbols. Missing pieces -> None; a symbol absent
    from the response is simply absent from the result."""
    mc = make_mc(tmp_path)
    urls = []
    payload = {
        "AAPL": {"latestTrade": {"p": 201.5}, "dailyBar": {"o": 200.0},
                 "prevDailyBar": {"c": 199.0}},
        "BRK.B": {"latestTrade": {"p": 410.25}},  # pre-open: no bars yet
    }

    def fake_get(url, headers=None, timeout=None):
        urls.append(url)
        assert headers["APCA-API-KEY-ID"] == "k"
        return FakeResp(payload=payload)

    monkeypatch.setattr(requests, "get", fake_get)
    snaps = alpaca_mod.fetch_snapshots(["AAPL", "BRK-B", "GONE"], mc, logger)

    assert snaps["AAPL"] == {"arrival": 201.5, "day_open": 200.0,
                             "prev_close": 199.0}
    assert snaps["BRK-B"] == {"arrival": 410.25, "day_open": None,
                              "prev_close": None}
    assert "GONE" not in snaps
    assert len(urls) == 1
    assert "feed=iex" in urls[0]
    assert "BRK.B" in urls[0] and "BRK-B" not in urls[0]


def test_fetch_snapshots_chunks_at_100_symbols(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    batch_sizes = []

    def fake_get(url, headers=None, timeout=None):
        syms = url.split("symbols=")[1].split("&")[0].split(",")
        batch_sizes.append(len(syms))
        return FakeResp(payload={
            s: {"latestTrade": {"p": 1.0}} for s in syms
        })

    monkeypatch.setattr(requests, "get", fake_get)
    symbols = [f"S{i:04d}" for i in range(250)]
    snaps = alpaca_mod.fetch_snapshots(symbols, mc, logger)

    assert batch_sizes == [100, 100, 50]
    assert len(snaps) == 250
    assert snaps["S0000"]["arrival"] == 1.0


def test_fetch_snapshots_returns_empty_dict_on_any_error(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)

    # HTTP error status
    monkeypatch.setattr(requests, "get",
                        lambda *a, **kw: FakeResp(status_code=500, text="boom"))
    assert alpaca_mod.fetch_snapshots(["AAPL"], mc, logger) == {}

    # Mid-batch transport failure keeps the chunks already captured —
    # chunks are arbitrary slices of the book (no bias vector), missing
    # symbols surface as absent keys handled via .get(sym), and the stats
    # layer counts unmeasured rows explicitly. Partial beats none.
    calls = {"n": 0}

    def flaky_get(url, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] > 1:
            raise requests.exceptions.ConnectionError("net down")
        syms = url.split("symbols=")[1].split("&")[0].split(",")
        return FakeResp(payload={s: {"latestTrade": {"p": 1.0}} for s in syms})

    monkeypatch.setattr(requests, "get", flaky_get)
    symbols = [f"S{i:04d}" for i in range(150)]  # 2 batches, 2nd fails
    partial = alpaca_mod.fetch_snapshots(symbols, mc, logger)
    assert len(partial) == 100  # first chunk retained
    assert partial["S0000"]["arrival"] == 1.0
    assert "S0149" not in partial  # failed chunk absent, not None-filled

    # Empty input short-circuits without a request
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("no call expected")))
    assert alpaca_mod.fetch_snapshots([], mc, logger) == {}


# ───── TradeRecord roundtrip ─────────────────────────────────────────────

def test_traderecord_new_fields_default_none_and_roundtrip_jsonl(tmp_path):
    journal = TradeJournal("sliptest_roundtrip")
    try:
        journal.log_trade(make_record(journal.model_name, trade_id="t_none"))
        journal.log_trade(make_record(
            journal.model_name, trade_id="t_vals", fill_price=101.0,
            decision_price=100.5, reference_open=100.0, prev_close=99.0,
            slippage_bps=100.0, slippage_vs_arrival_bps=49.75,
        ))
        trades = journal.get_trades()
        assert len(trades) == 2
        by_id = {t["trade_id"]: t for t in trades}
        for f in NEW_FIELDS:
            assert by_id["t_none"][f] is None       # old-row degradation
        assert by_id["t_vals"]["decision_price"] == 100.5
        assert by_id["t_vals"]["reference_open"] == 100.0
        assert by_id["t_vals"]["prev_close"] == 99.0
        assert by_id["t_vals"]["slippage_bps"] == 100.0
        assert by_id["t_vals"]["slippage_vs_arrival_bps"] == 49.75
    finally:
        journal.jsonl_path.unlink(missing_ok=True)
        journal.csv_path.unlink(missing_ok=True)


# ───── CSV header rollover ───────────────────────────────────────────────

def _old_header() -> str:
    """The pre-slippage CSV header: current schema minus the 5 new fields."""
    return ",".join(f.name for f in dc_fields(TradeRecord)
                    if f.name not in NEW_FIELDS)


def test_csv_rollover_renames_stale_header(tmp_path):
    journal = TradeJournal("sliptest_roll")
    rolled = journal.csv_path.with_name(journal.csv_path.name + ".pre-slippage")
    try:
        old_content = _old_header() + "\nold_row_payload\n"
        journal.csv_path.write_text(old_content)

        journal.log_trade(make_record(journal.model_name))

        # Old CSV preserved byte-for-byte under .pre-slippage
        assert rolled.exists()
        assert rolled.read_text() == old_content
        # Fresh CSV starts with the full current header + the new row
        lines = journal.csv_path.read_text().splitlines()
        assert lines[0] == ",".join(f.name for f in dc_fields(TradeRecord))
        assert len(lines) == 2
    finally:
        for p in (journal.jsonl_path, journal.csv_path, rolled):
            p.unlink(missing_ok=True)


def test_csv_rollover_uses_numeric_suffix_when_taken(tmp_path):
    journal = TradeJournal("sliptest_roll2")
    rolled = journal.csv_path.with_name(journal.csv_path.name + ".pre-slippage")
    rolled1 = journal.csv_path.with_name(journal.csv_path.name + ".pre-slippage.1")
    try:
        rolled.write_text("earlier rollover artifact\n")
        journal.csv_path.write_text(_old_header() + "\nrow\n")

        journal.log_trade(make_record(journal.model_name))

        assert rolled.read_text() == "earlier rollover artifact\n"  # untouched
        assert rolled1.exists()
        assert rolled1.read_text().splitlines()[0] == _old_header()
    finally:
        for p in (journal.jsonl_path, journal.csv_path, rolled, rolled1):
            p.unlink(missing_ok=True)


def test_csv_matching_header_appends_without_rollover(tmp_path):
    journal = TradeJournal("sliptest_append")
    rolled = journal.csv_path.with_name(journal.csv_path.name + ".pre-slippage")
    try:
        journal.log_trade(make_record(journal.model_name, trade_id="t1"))
        journal.log_trade(make_record(journal.model_name, trade_id="t2"))

        assert not rolled.exists()
        lines = journal.csv_path.read_text().splitlines()
        assert len(lines) == 3  # one header + two rows, no repeated header
        assert lines[0] == ",".join(f.name for f in dc_fields(TradeRecord))
        # JSONL untouched by the CSV logic
        assert len(journal.get_trades()) == 2
    finally:
        for p in (journal.jsonl_path, journal.csv_path, rolled):
            p.unlink(missing_ok=True)


# ───── rebalance integration: capture + slippage sign ────────────────────

def _run_rebalance(tmp_path, monkeypatch, snapshots, fill_price="101.0"):
    """Drive rebalance_portfolio with one exit (AAA) and one new buy (BBB),
    both filling at `fill_price`, with `fetch_snapshots` stubbed."""
    from core.run_report import RunReport

    mc = make_mc(tmp_path)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": "100000", "cash": "100000",
        "buying_power": "400000",
    })
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {
        "AAA": {"qty": 10.0, "market_value": 1000.0, "avg_entry": 90.0,
                "current_price": 100.0, "unrealized_pl": 100.0,
                "unrealized_pl_pct": 11.1},
    })

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "DELETE" and path == "v2/positions/AAA":
            return {"id": "oid_sell", "status": "accepted"}
        if method == "POST" and path == "v2/orders":
            return {"id": "oid_buy", "status": "accepted"}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(orders_mod, "alpaca_request", fake_alpaca)
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, mc_, lg: {"status": "filled",
                                              "filled_qty": "10",
                                              "filled_avg_price": fill_price})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", snapshots)
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)

    journal = StubJournal()
    rb = orders_mod.rebalance_portfolio(
        target_symbols=["BBB"], rankings=[("BBB", 1.0)],
        mc=mc, journal=journal, logger=logger, report=RunReport(),
        dry_run=False, target_weights={"BBB": 0.05},
    )
    return rb, {r.symbol: r for r in journal.records}


def test_slippage_sign_convention(tmp_path, monkeypatch):
    """Both fills at 101 against a 100 open: the buy paid up (+100 bps
    cost) while the sell got MORE than the reference (favorable = -100
    bps). Positive must always mean cost."""
    snap_data = {
        "AAA": {"arrival": 100.5, "day_open": 100.0, "prev_close": 99.0},
        "BBB": {"arrival": 100.5, "day_open": 100.0, "prev_close": 99.0},
    }
    rb, recs = _run_rebalance(
        tmp_path, monkeypatch, lambda syms, mc_, lg: snap_data)

    buy, sell = recs["BBB"], recs["AAA"]
    assert buy.side == "buy" and sell.side == "sell"
    for r in (buy, sell):
        assert r.fill_price == 101.0
        assert r.decision_price == 100.5
        assert r.reference_open == 100.0
        assert r.prev_close == 99.0

    assert buy.slippage_bps == pytest.approx(100.0)           # cost
    assert sell.slippage_bps == pytest.approx(-100.0)         # gain
    # vs arrival: 101/100.5 - 1 = +49.75 bps raw
    assert buy.slippage_vs_arrival_bps == pytest.approx(49.75, abs=0.01)
    assert sell.slippage_vs_arrival_bps == pytest.approx(-49.75, abs=0.01)
    assert rb["executed"] == 2 and rb["failed"] == 0


def test_snapshot_failure_is_nonfatal_and_leaves_fields_none(tmp_path, monkeypatch):
    """fetch_snapshots raising (or returning {}) must not touch order
    execution — journal rows just carry None for every capture field."""
    def exploding(syms, mc_, lg):
        raise RuntimeError("data API down")

    rb, recs = _run_rebalance(tmp_path, monkeypatch, exploding)

    assert rb["executed"] == 2 and rb["failed"] == 0
    for r in recs.values():
        assert r.fill_price == 101.0                 # fills still recorded
        for f in NEW_FIELDS:
            assert getattr(r, f) is None


def test_missing_symbol_in_snapshots_yields_none_slippage(tmp_path, monkeypatch):
    """A symbol the snapshot endpoint didn't return gets None capture
    fields and no slippage, while covered symbols still compute."""
    snap_data = {"BBB": {"arrival": None, "day_open": 100.0,
                         "prev_close": None}}
    rb, recs = _run_rebalance(
        tmp_path, monkeypatch, lambda syms, mc_, lg: snap_data)

    aaa, bbb = recs["AAA"], recs["BBB"]
    for f in NEW_FIELDS:
        assert getattr(aaa, f) is None
    assert bbb.reference_open == 100.0
    assert bbb.slippage_bps == pytest.approx(100.0)
    # arrival missing -> vs-arrival slippage stays None
    assert bbb.decision_price is None
    assert bbb.slippage_vs_arrival_bps is None
