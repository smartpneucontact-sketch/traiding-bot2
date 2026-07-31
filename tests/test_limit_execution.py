"""Task B — marketable-limit execution for REBALANCE orders.

Covers:
  - _limit_price_for: buffer arithmetic both sides, penny rounding TOWARD
    marketability (buys up / sells down), $0.01 floor
  - ModelConfig: new exec_* fields default to the legacy market behavior
  - default path byte-identical: exec_style="market" journals plain
    TradeRecord rows (no exec fields), places the same market orders, and
    the reconciliation flow carries no limit counters
  - limit placement: buys at arrival*(1+buffer), sells at *(1-buffer),
    qty-based limit for full exits, notional for the rest
  - snapshot-missing fallback: a symbol without an arrival price gets a
    market order; limit-POST failure also falls back per symbol
  - timeout-replace flow (mocked): unfilled at exec_fill_timeout_s →
    cancel + market-replace remainder; journal rows carry
    timed_out/replaced_to_market/replacement_order_id; reconciliation
    gains n_timeout_replaced
  - cancel flow: exec_timeout_action="cancel" leaves the shortfall
    unfilled and reconciliation reports n_unfilled_cancelled
  - partial-fill remainder arithmetic in _cancel_and_market_replace

Run:  /opt/anaconda3/bin/python -m pytest tests/test_limit_execution.py -v
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import asdict, fields as dc_fields
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import core.orders as orders_mod
from core.config import ModelConfig
from core.journal import TradeRecord

logger = logging.getLogger("test")

EXEC_FIELDS = ["exec_style", "limit_price", "timed_out",
               "replaced_to_market", "replacement_order_id"]


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


class StubJournal:
    def __init__(self):
        self.records = []
        self.jsonl_path = Path("/dev/null")

    def log_trade(self, record):
        self.records.append(record)


class FakeBroker:
    """Scriptable Alpaca stub.

    `limit_poll_state`: dict merged into an open limit order's state on
    GET (e.g. {"status": "filled", ...}); None = the limit order stays
    "new" forever (timeout scenarios). Market/close orders always fill at
    `market_fill` when polled via the poll_order_status stub.
    """

    def __init__(self, limit_poll_state=None, market_fill=("10", "101.0"),
                 fail_limit_post=False):
        self.calls: list[tuple] = []
        self.orders: dict[str, dict] = {}
        self._n = 0
        self.limit_poll_state = limit_poll_state
        self.market_fill = market_fill
        self.fail_limit_post = fail_limit_post

    def _mk(self, kind, payload=None):
        self._n += 1
        oid = f"oid{self._n}"
        self.orders[oid] = {
            "id": oid, "kind": kind, "status": "accepted",
            "filled_qty": "0", "filled_avg_price": "0",
            "payload": payload or {},
        }
        return oid

    def request(self, method, path, mc, data=None, logger=None):
        self.calls.append((method, path, data))
        if method == "POST" and path == "v2/orders":
            if (data or {}).get("type") == "limit":
                if self.fail_limit_post:
                    raise Exception("limit orders rejected by stub")
                oid = self._mk("limit", data)
                return {"id": oid, "status": "new"}
            oid = self._mk("market", data)
            return {"id": oid, "status": "accepted"}
        if method == "DELETE" and path.startswith("v2/positions/"):
            oid = self._mk("close", {"symbol": path.rsplit("/", 1)[-1]})
            return {"id": oid, "status": "accepted"}
        if method == "DELETE" and path.startswith("v2/orders/"):
            oid = path.rsplit("/", 1)[-1]
            o = self.orders.get(oid)
            if o and o["status"] != "filled":
                o["status"] = "canceled"
            return {}
        if method == "GET" and path.startswith("v2/orders/"):
            oid = path.rsplit("/", 1)[-1]
            o = self.orders.get(oid)
            if o is None:
                return {"id": oid, "status": "unknown"}
            if o["kind"] == "limit" and o["status"] not in ("canceled",):
                if self.limit_poll_state:
                    o.update(self.limit_poll_state)
                else:
                    o["status"] = "new"
            return dict(o)
        raise AssertionError(f"unexpected call {method} {path}")

    def poll(self, oid, *a, **kw):
        """poll_order_status stand-in: market/close orders fill; limit
        orders report their stored state (incl. cancel + partial fills)."""
        o = self.orders.get(oid)
        if o is None:
            return {"id": oid, "status": "unknown"}
        if o["kind"] in ("market", "close"):
            fq, fp = self.market_fill
            o.update({"status": "filled", "filled_qty": fq,
                      "filled_avg_price": fp})
        return dict(o)

    def posted(self, otype):
        return [d for (m, p, d) in self.calls
                if m == "POST" and p == "v2/orders"
                and (d or {}).get("type") == otype]


SNAPS = {
    "AAA": {"arrival": 100.5, "day_open": 100.0, "prev_close": 99.0},
    "BBB": {"arrival": 100.5, "day_open": 100.0, "prev_close": 99.0},
}


def run_rebalance(tmp_path, monkeypatch, broker, snap_data=SNAPS, **mc_kw):
    """One exit (AAA, qty 10) + one new buy (BBB, $10k = 100k * 0.05 * 2x)."""
    from core.run_report import RunReport

    mc = make_mc(tmp_path, **mc_kw)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": "100000", "cash": "100000",
        "buying_power": "400000",
    })
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {
        "AAA": {"qty": 10.0, "market_value": 1000.0, "avg_entry": 90.0,
                "current_price": 100.0, "unrealized_pl": 100.0,
                "unrealized_pl_pct": 11.1},
    })
    monkeypatch.setattr(orders_mod, "alpaca_request", broker.request)
    monkeypatch.setattr(orders_mod, "poll_order_status", broker.poll)
    monkeypatch.setattr(orders_mod, "fetch_snapshots",
                        lambda syms, mc_, lg: snap_data)
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)

    journal = StubJournal()
    rb = orders_mod.rebalance_portfolio(
        target_symbols=["BBB"], rankings=[("BBB", 1.0)],
        mc=mc, journal=journal, logger=logger, report=RunReport(),
        dry_run=False, target_weights={"BBB": 0.05},
    )
    return rb, {r.symbol: r for r in journal.records}, broker


# ───── limit price arithmetic ────────────────────────────────────────────

def test_limit_price_buy_and_sell_buffer():
    # 10bp on 200.00: buy crosses UP, sell crosses DOWN
    assert orders_mod._limit_price_for("buy", 200.0, 10.0) == 200.20
    assert orders_mod._limit_price_for("sell", 200.0, 10.0) == 199.80


def test_limit_price_rounds_toward_marketability():
    # 123.456 * 1.001 = 123.579456 → buy rounds UP to 123.58
    assert orders_mod._limit_price_for("buy", 123.456, 10.0) == 123.58
    # 123.456 * 0.999 = 123.332544 → sell rounds DOWN to 123.33
    assert orders_mod._limit_price_for("sell", 123.456, 10.0) == 123.33
    # zero buffer: rounding alone must not cross AGAINST marketability
    assert orders_mod._limit_price_for("buy", 100.004, 0.0) == 100.01
    assert orders_mod._limit_price_for("sell", 100.004, 0.0) == 100.00
    # exact cents pass through unchanged either side
    assert orders_mod._limit_price_for("buy", 100.10, 0.0) == 100.10
    assert orders_mod._limit_price_for("sell", 100.10, 0.0) == 100.10


def test_limit_price_floors_at_one_cent():
    assert orders_mod._limit_price_for("sell", 0.005, 10.0) == 0.01
    assert orders_mod._limit_price_for("sell", 0.05, 10.0) == 0.04


# ───── config defaults ───────────────────────────────────────────────────

def test_modelconfig_exec_defaults_are_market(tmp_path):
    mc = make_mc(tmp_path)
    assert mc.exec_style == "market"
    assert mc.exec_limit_buffer_bps == 10.0
    assert mc.exec_fill_timeout_s == 120
    assert mc.exec_timeout_action == "market"


# ───── default market path byte-identical ────────────────────────────────

def test_default_market_path_unchanged(tmp_path, monkeypatch):
    """exec_style='market' (default): plain TradeRecord rows with the
    legacy field set, legacy order placement (DELETE position + market
    POST), no limit counters in reconciliation, no exec_style in rb."""
    rb, recs, broker = run_rebalance(tmp_path, monkeypatch, FakeBroker())

    assert rb["executed"] == 2 and rb["failed"] == 0
    assert "exec_style" not in rb
    for r in recs.values():
        assert type(r) is TradeRecord  # NOT ExecTradeRecord
        assert set(asdict(r).keys()) == {f.name for f in dc_fields(TradeRecord)}
        assert r.order_type == "market"
    # legacy placement calls: exit via DELETE position, buy via market POST
    assert ("DELETE", "v2/positions/AAA", None) in broker.calls
    assert len(broker.posted("market")) == 1
    assert broker.posted("limit") == []
    flow = rb["reconciliation"]["flow"]
    assert "n_timeout_replaced" not in flow
    assert "n_unfilled_cancelled" not in flow
    assert flow["n_filled"] == 2


# ───── limit placement + fill ────────────────────────────────────────────

def test_limit_orders_placed_and_filled(tmp_path, monkeypatch):
    broker = FakeBroker(limit_poll_state={
        "status": "filled", "filled_qty": "10", "filled_avg_price": "100.55"})
    rb, recs, broker = run_rebalance(
        tmp_path, monkeypatch, broker, exec_style="marketable_limit",
        exec_fill_timeout_s=5)

    assert rb["exec_style"] == "marketable_limit"
    assert rb["executed"] == 2 and rb["failed"] == 0

    limits = broker.posted("limit")
    assert len(limits) == 2
    by_side = {d["side"]: d for d in limits}
    # buy: 100.5 * 1.001 = 100.6005 → ceil to 100.61; notional order
    assert by_side["buy"]["symbol"] == "BBB"
    assert by_side["buy"]["limit_price"] == 100.61
    assert by_side["buy"]["notional"] == 10000.0
    assert by_side["buy"]["time_in_force"] == "day"
    # sell (full exit): 100.5 * 0.999 = 100.3995 → floor to 100.39; qty order
    assert by_side["sell"]["symbol"] == "AAA"
    assert by_side["sell"]["limit_price"] == 100.39
    assert by_side["sell"]["qty"] == 10.0
    assert "notional" not in by_side["sell"]
    # no market orders and no position DELETE were needed
    assert broker.posted("market") == []
    assert not any(m == "DELETE" and p.startswith("v2/positions/")
                   for m, p, d in broker.calls)

    for sym, lp in (("BBB", 100.61), ("AAA", 100.39)):
        r = recs[sym]
        assert isinstance(r, orders_mod.ExecTradeRecord)
        assert r.exec_style == "marketable_limit"
        assert r.order_type == "limit"
        assert r.limit_price == lp
        assert r.order_status == "filled"
        assert r.fill_price == 100.55 and r.shares == 10.0
        assert r.timed_out is None and r.replaced_to_market is None
    # slippage vs arrival still computed after resolution (+cost for buy,
    # favorable for the sell filling above arrival)
    assert recs["BBB"].slippage_vs_arrival_bps == pytest.approx(4.98, abs=0.01)
    assert recs["AAA"].slippage_vs_arrival_bps == pytest.approx(-4.98, abs=0.01)

    flow = rb["reconciliation"]["flow"]
    assert flow["n_filled"] == 2
    assert flow["n_timeout_replaced"] == 0
    assert flow["n_unfilled_cancelled"] == 0


# ───── snapshot-missing / limit-error fallbacks ──────────────────────────

def test_missing_arrival_snapshot_falls_back_to_market(tmp_path, monkeypatch):
    """AAA has no arrival price → its exit goes out as the legacy market
    DELETE; BBB (covered) still goes out as a limit order."""
    broker = FakeBroker(limit_poll_state={
        "status": "filled", "filled_qty": "10", "filled_avg_price": "100.55"})
    snaps = {"BBB": {"arrival": 100.5, "day_open": 100.0, "prev_close": 99.0}}
    rb, recs, broker = run_rebalance(
        tmp_path, monkeypatch, broker, snap_data=snaps,
        exec_style="marketable_limit", exec_fill_timeout_s=5)

    assert rb["executed"] == 2 and rb["failed"] == 0
    assert ("DELETE", "v2/positions/AAA", None) in broker.calls
    assert len(broker.posted("limit")) == 1  # BBB only
    aaa, bbb = recs["AAA"], recs["BBB"]
    assert aaa.exec_style == "market" and aaa.order_type == "market"
    assert aaa.limit_price is None
    assert aaa.order_status == "filled"  # legacy quick-poll path filled it
    assert bbb.exec_style == "marketable_limit" and bbb.limit_price == 100.61


def test_limit_post_failure_falls_back_to_market_per_symbol(tmp_path, monkeypatch):
    """Broker rejects every limit POST → each symbol independently falls
    back to its legacy market path; nothing fails, nothing is pending."""
    broker = FakeBroker(fail_limit_post=True)
    rb, recs, broker = run_rebalance(
        tmp_path, monkeypatch, broker, exec_style="marketable_limit",
        exec_fill_timeout_s=5)

    assert rb["executed"] == 2 and rb["failed"] == 0
    assert ("DELETE", "v2/positions/AAA", None) in broker.calls
    assert len(broker.posted("market")) == 1
    for r in recs.values():
        assert r.exec_style == "market"
        assert r.order_type == "market"
        assert r.order_status == "filled"
    flow = rb["reconciliation"]["flow"]
    assert flow["n_filled"] == 2
    assert flow["n_timeout_replaced"] == 0
    assert flow["n_unfilled_cancelled"] == 0


# ───── timeout → cancel-and-market-replace ───────────────────────────────

def test_timeout_replaces_unfilled_with_market(tmp_path, monkeypatch):
    """Limit orders never fill (poll state stays 'new'); at timeout both
    are cancelled and re-sent as market orders which fill. Journal rows
    carry timed_out/replaced_to_market/replacement ids; reconciliation
    counts n_timeout_replaced."""
    broker = FakeBroker(limit_poll_state=None)  # stays "new" forever
    rb, recs, broker = run_rebalance(
        tmp_path, monkeypatch, broker, exec_style="marketable_limit",
        exec_fill_timeout_s=0, exec_timeout_action="market")

    # both limit orders went out, then were cancelled
    limit_ids = [oid for oid, o in broker.orders.items() if o["kind"] == "limit"]
    assert len(limit_ids) == 2
    for oid in limit_ids:
        assert ("DELETE", f"v2/orders/{oid}", None) in broker.calls
        assert broker.orders[oid]["status"] == "canceled"
    # replacements: exit remainder via DELETE position, buy via market POST
    assert ("DELETE", "v2/positions/AAA", None) in broker.calls
    market_posts = broker.posted("market")
    assert len(market_posts) == 1
    assert market_posts[0]["symbol"] == "BBB"
    assert market_posts[0]["notional"] == 10000.0  # zero limit fills → full remainder

    for r in recs.values():
        assert r.timed_out is True
        assert r.replaced_to_market is True
        assert r.replacement_order_id is not None
        assert r.order_status == "filled"          # market replacement filled
        assert r.fill_price == 101.0 and r.shares == 10.0
    flow = rb["reconciliation"]["flow"]
    assert flow["n_filled"] == 2
    assert flow["n_timeout_replaced"] == 2
    assert flow["n_unfilled_cancelled"] == 0


def test_partial_fill_market_replace_remainder_arithmetic(tmp_path, monkeypatch):
    """20 of $10k filled at 100 on the limit → cancel keeps the partial
    fill, market-replace goes out for the $8k remainder, and the journal
    row blends both fills."""
    broker = FakeBroker(market_fill=("80", "101.0"))
    broker.orders["L1"] = {
        "id": "L1", "kind": "limit", "status": "partially_filled",
        "filled_qty": "20", "filled_avg_price": "100.0", "payload": {},
    }
    monkeypatch.setattr(orders_mod, "alpaca_request", broker.request)
    monkeypatch.setattr(orders_mod, "poll_order_status", broker.poll)
    mc = make_mc(tmp_path)
    trade = orders_mod.ExecTradeRecord(
        trade_id="t", run_id="r", model="combo_v2", timestamp="ts",
        symbol="BBB", side="buy", action="new_position", order_type="limit",
        time_in_force="day", notional_usd=10000.0, order_id="L1",
        exec_style="marketable_limit", limit_price=100.61)
    order = {"action": "buy_notional", "symbol": "BBB", "side": "buy",
             "notional": 10000.0, "trade_action": "new_position"}

    replaced = orders_mod._cancel_and_market_replace(trade, order, mc, logger)

    assert replaced is True
    posts = broker.posted("market")
    assert len(posts) == 1
    assert posts[0]["notional"] == pytest.approx(8000.0)  # 10000 - 20*100
    assert posts[0]["side"] == "buy" and posts[0]["symbol"] == "BBB"
    assert trade.replaced_to_market is True
    assert trade.replacement_order_id is not None
    assert trade.shares == 100.0                       # 20 limit + 80 market
    assert trade.fill_price == pytest.approx(100.8)    # (20*100 + 80*101)/100
    assert trade.order_status == "filled"


def test_cancel_race_fill_skips_replacement(tmp_path, monkeypatch):
    """Order fills while we're cancelling → no replacement is sent."""
    broker = FakeBroker()
    broker.orders["L1"] = {
        "id": "L1", "kind": "limit", "status": "filled",
        "filled_qty": "99", "filled_avg_price": "100.5", "payload": {},
    }
    monkeypatch.setattr(orders_mod, "alpaca_request", broker.request)
    monkeypatch.setattr(orders_mod, "poll_order_status", broker.poll)
    mc = make_mc(tmp_path)
    trade = orders_mod.ExecTradeRecord(
        trade_id="t", run_id="r", model="combo_v2", timestamp="ts",
        symbol="BBB", side="buy", action="new_position", order_type="limit",
        time_in_force="day", notional_usd=10000.0, order_id="L1",
        exec_style="marketable_limit")
    order = {"action": "buy_notional", "symbol": "BBB", "side": "buy",
             "notional": 10000.0, "trade_action": "new_position"}

    replaced = orders_mod._cancel_and_market_replace(trade, order, mc, logger)

    assert replaced is False
    assert broker.posted("market") == []
    assert trade.replaced_to_market is None
    assert trade.order_status == "filled"
    assert trade.shares == 99.0 and trade.fill_price == 100.5


# ───── timeout → cancel (leave unfilled) ─────────────────────────────────

def test_timeout_cancel_leaves_shortfall_and_reports_it(tmp_path, monkeypatch):
    """exec_timeout_action='cancel': unfilled orders are cancelled, NOT
    replaced; the journal shows the shortfall and reconciliation reports
    n_unfilled_cancelled + the cancelled symbols."""
    broker = FakeBroker(limit_poll_state=None)  # never fills
    rb, recs, broker = run_rebalance(
        tmp_path, monkeypatch, broker, exec_style="marketable_limit",
        exec_fill_timeout_s=0, exec_timeout_action="cancel")

    # cancels went out, but NO replacement of any kind
    limit_ids = [oid for oid, o in broker.orders.items() if o["kind"] == "limit"]
    assert len(limit_ids) == 2
    for oid in limit_ids:
        assert ("DELETE", f"v2/orders/{oid}", None) in broker.calls
    assert broker.posted("market") == []
    assert not any(m == "DELETE" and p.startswith("v2/positions/")
                   for m, p, d in broker.calls)

    for r in recs.values():
        assert r.timed_out is True
        assert r.replaced_to_market is None
        assert r.order_status == "canceled"
        assert r.shares is None and r.fill_price is None
    flow = rb["reconciliation"]["flow"]
    assert flow["n_filled"] == 0
    assert flow["n_rejected"] == 2  # canceled counts as a rejected state
    assert sorted(flow["rejected_symbols"]) == ["AAA", "BBB"]
    assert flow["n_timeout_replaced"] == 0
    assert flow["n_unfilled_cancelled"] == 2


# ───── journal schema isolation ──────────────────────────────────────────

def test_exec_fields_only_on_subclass():
    """The exec telemetry fields live on ExecTradeRecord, NOT on the base
    TradeRecord — the frozen market path's journal schema is untouched."""
    base = {f.name for f in dc_fields(TradeRecord)}
    ext = {f.name for f in dc_fields(orders_mod.ExecTradeRecord)}
    for f in EXEC_FIELDS:
        assert f not in base
        assert f in ext
    assert ext == base | set(EXEC_FIELDS)


def test_slot_config_exec_style_passthrough(tmp_path, monkeypatch):
    """model_config.json slot fields MUST reach ModelConfig — setting
    exec_style in the config was silently ignored before 2026-07-31 (the
    same fault class as the 07-25 target_leverage incident)."""
    import json
    import core.config as cfg
    slots = {"slots": [{
        "slot_id": 1, "model": "combo_v2", "enabled": True,
        "alpaca_key": "k", "alpaca_secret": "s",
        "target_leverage": 2,
        "exec_style": "marketable_limit",
        "exec_limit_buffer_bps": 12.5,
        "exec_fill_timeout_s": 90,
        "exec_timeout_action": "cancel",
    }]}
    p = tmp_path / "model_config.json"
    p.write_text(json.dumps(slots))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)
    models = cfg.get_active_models()
    assert len(models) == 1
    mc = models[0]
    assert mc.exec_style == "marketable_limit"
    assert mc.exec_limit_buffer_bps == 12.5
    assert mc.exec_fill_timeout_s == 90
    assert mc.exec_timeout_action == "cancel"


def test_slot_config_exec_style_defaults_market(tmp_path, monkeypatch):
    import json
    import core.config as cfg
    slots = {"slots": [{
        "slot_id": 1, "model": "combo_v2", "enabled": True,
        "alpaca_key": "k", "alpaca_secret": "s",
    }]}
    p = tmp_path / "model_config.json"
    p.write_text(json.dumps(slots))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)
    mc = cfg.get_active_models()[0]
    assert mc.exec_style == "market"
    assert mc.exec_timeout_action == "market"
