"""Tests for exec_style="auction" (2026-10-15, reworked after the adversarial
review): pre-open submission with yesterday's OFFICIAL close as the anchor,
market-on-open sells / limit-on-open buys, polled finalization, one run_id
per rebalance day, plan reuse by the 09:35 completion, marketable-limit
completion, stage-aware dashboard bookkeeping.

Run:  /opt/anaconda3/bin/python3 -m pytest tests/test_auction.py -v
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.config import ModelConfig
import core.orders as orders_mod
import core.market as market_mod
from core.run_report import RunReport
from core.state import load_state, save_state

logger = logging.getLogger("test_auction")
NY = ZoneInfo("America/New_York")


def make_mc(tmp_path, **kw) -> ModelConfig:
    defaults = dict(name="combo_v2_process", model_path=Path("/nonexistent/model.pkl"),
                    feature_version="combo", alpaca_key="k", alpaca_secret="s",
                    state_path=tmp_path / "state.json", enable_cutloss=True,
                    target_leverage=2.0, exec_style="auction")
    defaults.update(kw)
    return ModelConfig(**defaults)


class StubJournal:
    jsonl_path = "/dev/null"

    def __init__(self, *a, **kw):
        self.records = []

    def log_trade(self, record):
        self.records.append(record)


# ───────────────────────── window ───────────────────────────────────────

def test_auction_window_bounds():
    d = datetime(2026, 10, 15, tzinfo=NY)   # a Thursday
    assert market_mod.is_auction_window(d.replace(hour=9, minute=6))
    assert market_mod.is_auction_window(d.replace(hour=9, minute=26, second=59))
    assert not market_mod.is_auction_window(d.replace(hour=9, minute=27))
    assert not market_mod.is_auction_window(d.replace(hour=8, minute=59))
    assert not market_mod.is_auction_window(d.replace(hour=9, minute=35))
    assert not market_mod.is_auction_window(datetime(2026, 10, 17, 9, 10, tzinfo=NY))


# ───────────────────────── config passthrough ───────────────────────────

def test_config_passthrough_auction_fields(tmp_path, monkeypatch):
    import core.config as cfg
    monkeypatch.setattr(cfg, "load_model_config", lambda: {
        "cutloss_calibration": cfg.CUTLOSS_CALIBRATION_VERSION,
        "shadow_slots": cfg.SHADOW_SLOTS_VERSION,
        "slots": [{"slot_id": 1, "model": "combo_v2", "enabled": True,
                   "alpaca_key": "k", "alpaca_secret": "s",
                   "exec_style": "auction", "exec_auction_buffer_bps": 80}]})
    monkeypatch.setattr(cfg, "_ensure_shadow_slots", lambda c: c)
    monkeypatch.setattr(cfg, "_migrate_cutloss_calibration", lambda c: c)
    monkeypatch.setattr(cfg.Path, "exists", lambda self: True)
    models = cfg.get_active_models()
    assert models and models[0].exec_style == "auction"
    assert models[0].exec_auction_buffer_bps == 80.0


# ───────────────────────── stage 1: submission ──────────────────────────

def _broker(posted, *, reject=("REJ",)):
    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        if method == "POST" and endpoint == "v2/orders":
            posted.append(dict(data))
            if data["symbol"] in reject:
                return {"status": "rejected", "reject_reason": "no_bid"}
            return {"id": f"o{len(posted)}", "status": "accepted"}
        return {}
    return fake_request


def _run_auction(tmp_path, monkeypatch, *, positions, weights, last_close,
                 equity=100_000, bp=1e9, run_id=None):
    posted = []
    mc = make_mc(tmp_path)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": str(equity), "cash": "0", "buying_power": str(bp)})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {
        s: {"qty": q, "market_value": mv, "avg_entry": 100.0, "current_price": mv / q,
            "unrealized_pl": 0.0, "unrealized_pl_pct": 0.0}
        for s, (q, mv) in positions.items()})
    monkeypatch.setattr(orders_mod, "alpaca_request", _broker(posted))
    # Pre-open snapshots are deliberately NOT consulted for the anchor —
    # feed a stale/odd snapshot to prove it.
    monkeypatch.setattr(orders_mod, "fetch_snapshots",
                        lambda syms, mc_, lg: {s: {"prev_close": 1.0} for s in syms})
    journal = StubJournal()
    rb = orders_mod.rebalance_portfolio(
        target_symbols=list(weights), rankings=[(s, 1.0) for s in weights],
        mc=mc, journal=journal, logger=logger, report=RunReport(),
        dry_run=False, target_weights=weights, auction=True,
        last_close=last_close, run_id=run_id)
    return rb, posted, journal


def test_auction_uses_official_close_moo_sells_loo_buys(tmp_path, monkeypatch):
    rb, posted, journal = _run_auction(
        tmp_path, monkeypatch,
        positions={"AAA": (10.5, 1_050.0),      # full exit, fractional position
                   "CCC": (100.0, 10_000.0)},   # trimmed 10k -> 8.75k (12.5%)
        weights={"BBB": 0.05, "CCC": 0.04375, "DDD": 0.0025, "EEE": 0.05},
        last_close={"AAA": 100.0, "BBB": 100.0, "CCC": 50.0, "DDD": 900.0},
        run_id="combo_v2_process_20261028_130600_000000")
    by = {o["symbol"]: o for o in posted}
    for o in posted:
        assert o["time_in_force"] == "opg" and "notional" not in o and float(o["qty"]).is_integer()
    # Sells are MARKET-on-open (risk-reducing, never left hanging).
    assert by["AAA"] == {"symbol": "AAA", "side": "sell", "type": "market", "qty": "10", "time_in_force": "opg"}
    assert by["CCC"]["type"] == "market" and by["CCC"]["qty"] == "25"
    # Buys are LIMIT-on-open at last_close + 100bp (not the snapshot's 1.0).
    assert by["BBB"]["type"] == "limit" and by["BBB"]["qty"] == "100"
    assert by["BBB"]["limit_price"] == pytest.approx(101.0)
    assert "DDD" not in by and "EEE" not in by
    a = rb["auction"]
    assert a["n_submitted"] == 3 and a["n_skipped"] == 2
    reasons = {s["symbol"]: s["reason"] for s in a["skipped"]}
    assert reasons["DDD"] == "sub_share" and reasons["EEE"] == "no_prev_close"
    assert rb["run_id"] == "combo_v2_process_20261028_130600_000000"
    assert all(p["run_id"] == rb["run_id"] for p in a["pending"])
    assert {p["order_type"] for p in a["pending"]} == {"market", "limit"}
    assert rb["exec_style"] == "auction_loo"
    assert not journal.records


def test_auction_rejected_order_is_skipped_not_fatal(tmp_path, monkeypatch):
    rb, posted, _ = _run_auction(
        tmp_path, monkeypatch, positions={}, weights={"REJ": 0.05, "BBB": 0.05},
        last_close={"REJ": 10.0, "BBB": 100.0})
    assert rb["auction"]["n_submitted"] == 1
    assert any(s["symbol"] == "REJ" and s["reason"].startswith("rejected") for s in rb["auction"]["skipped"])


# ───────────────────────── stage 2: finalization ────────────────────────

def test_finalize_polls_fills_cancels_leftovers_and_shares_run_id(tmp_path, monkeypatch):
    calls = []
    states = {"o1": {"id": "o1", "status": "filled", "filled_qty": "100", "filled_avg_price": "100.40"},
              "o2": {"id": "o2", "status": "canceled", "filled_qty": "0"},
              "o3": {"id": "o3", "status": "new", "filled_qty": "0"}}

    def fake_poll(order_id, mc, lg, max_wait=8.0, **kw):
        calls.append(("POLL", order_id))
        return dict(states[order_id])

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        calls.append((method, endpoint))
        if method == "DELETE":
            states["o3"] = {"id": "o3", "status": "canceled", "filled_qty": "0"}
            return {}
        return {}

    monkeypatch.setattr(orders_mod, "poll_order_status", fake_poll)
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {
        "BBB": {"day_open": 100.0, "arrival": 100.9, "prev_close": 99.0},
        "CCC": {"day_open": 50.0}, "DDD": {"day_open": 20.0}})
    mc = make_mc(tmp_path)
    # Same-session finalize: the submission date is TODAY (an earlier date
    # is a late finalize and carries no reference open — see below).
    today = datetime.now(NY).strftime("%Y-%m-%d")
    base = {"submitted_at": f"{today}T13:06:00+00:00", "run_id": "RID", "order_status": "accepted"}
    pending = [
        {**base, "symbol": "BBB", "side": "buy", "trade_action": "new_position", "order_type": "limit",
         "qty": 100, "limit_price": 100.0, "prev_close": 99.0, "notional_est": 9_900.0, "order_id": "o1"},
        {**base, "symbol": "CCC", "side": "sell", "trade_action": "rebalance_down", "order_type": "market",
         "qty": 25, "limit_price": None, "prev_close": 50.0, "notional_est": 1_250.0, "order_id": "o2"},
        {**base, "symbol": "DDD", "side": "buy", "trade_action": "new_position", "order_type": "limit",
         "qty": 10, "limit_price": 20.2, "prev_close": 20.0, "notional_est": 200.0, "order_id": "o3"},
    ]
    journal = StubJournal()
    stats = orders_mod.finalize_auction_orders(mc, pending, journal, logger)
    assert stats["n"] == 3 and stats["n_filled"] == 1 and stats["n_unfilled"] == 2
    assert stats["n_cancelled_open"] == 1 and ("DELETE", "v2/orders/o3") in calls
    assert stats["submitted_notional_usd"] == pytest.approx(11_350.0)
    rows = {r.symbol: r for r in journal.records}
    assert all(r.run_id == "RID" and r.exec_style == "auction_loo" and r.time_in_force == "opg"
               for r in rows.values())
    assert rows["BBB"].order_type == "limit" and rows["CCC"].order_type == "market"
    assert rows["BBB"].fill_price == pytest.approx(100.40)
    assert rows["BBB"].slippage_bps == pytest.approx(40.0)
    assert rows["BBB"].decision_price is None and rows["BBB"].slippage_vs_arrival_bps is None
    assert rows["BBB"].notional_usd == pytest.approx(10_040.0)
    assert rows["DDD"].order_status == "canceled"          # final state after the cancel, not 'new'
    # Anchor-priced: 100 sh x 99.0 prev close over 11,350 planned.
    assert stats["fill_rate_notional"] == pytest.approx(9_900.0 / 11_350.0, abs=1e-3)
    assert stats["filled_notional_usd"] == pytest.approx(10_040.0)
    assert stats["vw_cost_vs_open_bps"] == pytest.approx(40.0)


# ──────────────── completion leg of an auction slot (09:35) ─────────────

def test_auction_slot_completion_uses_marketable_limit_and_tags_rows(tmp_path, monkeypatch):
    posted = []
    polled = {}

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        if method == "POST" and endpoint == "v2/orders":
            posted.append(dict(data))
            oid = f"o{len(posted)}"
            polled[oid] = {"id": oid, "status": "filled", "filled_qty": "1", "filled_avg_price": "100.5"}
            return {"id": oid, "status": "accepted"}
        if method == "GET" and endpoint.startswith("v2/orders/"):
            return polled.get(endpoint.rsplit("/", 1)[-1], {})
        if endpoint == "v2/positions":
            return []
        return {}

    mc = make_mc(tmp_path)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": "100000", "cash": "0", "buying_power": "400000"})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {})
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, *a, **kw: polled.get(oid, {"status": "filled", "filled_qty": "1", "filled_avg_price": "100.5"}))
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {
        "BBB": {"arrival": 100.5, "day_open": 100.0, "prev_close": 99.0}})
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)
    journal = StubJournal()
    rb = orders_mod.rebalance_portfolio(
        target_symbols=["BBB"], rankings=[("BBB", 1.0)], mc=mc, journal=journal,
        logger=logger, report=RunReport(), dry_run=False, target_weights={"BBB": 0.05},
        run_id="RID")
    assert posted and posted[0]["type"] == "limit" and posted[0]["time_in_force"] == "day"
    assert rb["exec_style"] == "auction_completion"
    assert journal.records and journal.records[0].exec_style == "auction_completion"
    assert journal.records[0].run_id == "RID"


# ───────────────────────── runner glue ──────────────────────────────────

def test_finalize_helper_defers_pre_open_and_finalizes_old_days(tmp_path, monkeypatch):
    import core.runner as runner_mod
    mc = make_mc(tmp_path)
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda mc_, pending, journal, lg: {"n": len(pending), "n_filled": 1})
    # Today's list before the open: deferred, untouched.
    st = {"auction_stage": {"date": "2026-10-28", "pending": [{"order_id": "o1"}]}}
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: False)
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), "2026-10-28")
    assert st["auction_stage"]["pending"] == [{"order_id": "o1"}]
    # After the open: finalized.
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), "2026-10-28")
    assert st["auction_stage"]["pending"] == [] and st["auction_stage"]["summary"]["n_filled"] == 1
    # An orphaned earlier day finalizes regardless of the clock, flagged late.
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: False)
    st2 = {"auction_stage": {"date": "2026-10-27", "pending": [{"order_id": "x"}]}}
    runner_mod._finalize_auction_if_any(mc, st2, StubJournal(), logger, RunReport(), "2026-10-28")
    assert st2["auction_stage"]["pending"] == [] and st2["auction_stage"]["finalized_late"] is True


def test_todays_plan_and_merge_into_result():
    import core.runner as runner_mod
    st = {"auction_stage": {"date": "2026-10-28", "n_submitted": 3, "outcome": "submitted",
                            "plan": {"target_symbols": ["A"], "rankings": [["A", 1.0]], "target_weights": {"A": 1.0}},
                            "summary": {"n": 3, "n_filled": 2, "submitted_notional_usd": 30_000.0,
                                        "filled_notional_usd": 20_000.0}}}
    assert runner_mod._todays_auction_plan(st, "2026-10-28")["target_symbols"] == ["A"]
    assert runner_mod._todays_auction_plan(st, "2026-10-29") is None
    # Auction: 30k submitted, 20k filled; completion re-submits the 10k
    # residue and fills it -> the PLAN executed 100%: rate must be 1.0, not
    # 30k/40k (the double count that would have fired a false K3a).
    result = {"reconciliation": {"flow": {"n_orders": 2, "n_filled": 2, "n_unconfirmed": 0,
                                          "fill_rate_notional": 1.0,
                                          "orders_notional_usd": 10_000.0, "filled_notional_usd": 10_000.0},
                                 "book": None}}
    runner_mod._merge_auction_into_result(result, st["auction_stage"], "2026-10-28")
    flow = result["reconciliation"]["flow"]
    assert flow["n_orders"] == 5 and flow["n_filled"] == 4 and flow["auction_unfilled"] == 1
    assert flow["n_unconfirmed"] == 0
    assert flow["fill_rate_notional"] == pytest.approx(1.0)
    assert flow["fill_rate_notional_completion"] == 1.0
    assert result["reconciliation"]["auction"]["n"] == 3
    # Partial completion: 30k submitted / 20k filled, completion submits 10k but fills 5k -> 25/30.
    result2 = {"reconciliation": {"flow": {"n_orders": 2, "n_filled": 1, "n_unconfirmed": 1,
                                           "fill_rate_notional": 0.5,
                                           "orders_notional_usd": 10_000.0, "filled_notional_usd": 5_000.0},
                                  "book": None}}
    runner_mod._merge_auction_into_result(result2, st["auction_stage"], "2026-10-28")
    assert result2["reconciliation"]["flow"]["fill_rate_notional"] == pytest.approx(25_000.0 / 30_000.0, abs=1e-3)


def test_pipeline_auction_stage_only_runs_auction_slots(tmp_path, monkeypatch):
    import core.runner as runner_mod
    ran = []
    a = make_mc(tmp_path, name="combo_v2_process", exec_style="auction")
    b = make_mc(tmp_path, name="combo_v2", exec_style="marketable_limit")
    monkeypatch.setattr(runner_mod, "get_active_models", lambda: [a, b])
    monkeypatch.setattr("core.invariants.run_invariant_checks", lambda lg: None)
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: True)

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 10, 28, 9, 8, tzinfo=NY)
            return base.astimezone(tz) if tz else base.replace(tzinfo=None)
    monkeypatch.setattr(runner_mod, "datetime", FakeDT)
    monkeypatch.setattr(runner_mod, "get_tradeable_symbols", lambda lg, rp: ["X"])
    monkeypatch.setattr(runner_mod, "download_bars", lambda syms, days, lg, rp: {"X": object()})
    monkeypatch.setattr(runner_mod, "download_macro", lambda days, lg, rp: {})
    monkeypatch.setattr(runner_mod, "compute_macro_features", lambda m: __import__("pandas").DataFrame())
    monkeypatch.setattr(runner_mod, "run_single_model",
                        lambda mc, *args, **kw: ran.append((mc.name, kw.get("stage"))))
    runner_mod.run_pipeline(stage="auction")
    assert ran == [("combo_v2_process", "auction")]
    ran.clear()
    monkeypatch.setattr(runner_mod, "get_active_models", lambda: [b])
    runner_mod.run_pipeline(stage="auction")
    assert ran == []


def test_dashboard_auction_stage_keeps_main_run_fields(monkeypatch):
    import dashboard
    seen = {}
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", lambda **kw: seen.update(kw))
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {})
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["last_run_at"] = "KEEP"
        before_total = dashboard.bot_status["total_runs"]
    dashboard.run_trading_pipeline(stage="auction")
    assert seen.get("stage") == "auction"
    with dashboard.status_lock:
        assert dashboard.bot_status["last_run_at"] == "KEEP"
        assert dashboard.bot_status["total_runs"] == before_total
        assert dashboard.bot_status["auction_last_run_status"] == "success"
        assert dashboard.bot_status["state"] == "idle"



# ───────────── review round 2 regressions ───────────────────────────────

def test_finalize_keeps_unreadable_orders_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)
    """A failed read must never cancel or journal an order: it stays pending."""
    def fake_poll(order_id, mc, lg, max_wait=8.0, **kw):
        if order_id == "o1":
            return {"id": "o1", "status": "filled", "filled_qty": "10", "filled_avg_price": "50.0"}
        raise RuntimeError("api down")
    deletes = []
    monkeypatch.setattr(orders_mod, "poll_order_status", fake_poll)
    monkeypatch.setattr(orders_mod, "alpaca_request",
                        lambda m, e, mc, data=None, logger=None, **kw: deletes.append(e) if m == "DELETE" else {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {"AAA": {"day_open": 50.0}, "BBB": {"day_open": 1.0}})
    mc = make_mc(tmp_path)
    base = {"submitted_at": "2026-10-28T13:06:00+00:00", "run_id": "RID", "order_status": "accepted"}
    pending = [{**base, "symbol": "AAA", "side": "buy", "order_type": "limit", "qty": 10, "limit_price": 50.5,
                "prev_close": 50.0, "notional_est": 500.0, "order_id": "o1"},
               {**base, "symbol": "BBB", "side": "buy", "order_type": "limit", "qty": 10, "limit_price": 1.01,
                "prev_close": 1.0, "notional_est": 10.0, "order_id": "o2"}]
    journal = StubJournal()
    stats = orders_mod.finalize_auction_orders(mc, pending, journal, logger)
    assert stats["n"] == 1 and stats["n_filled"] == 1 and stats["n_remaining_unread"] == 1
    assert [p["order_id"] for p in stats["remaining"]] == ["o2"]
    assert not deletes and [r.symbol for r in journal.records] == ["AAA"]
    assert journal.records[0].trade_id.endswith("_20261028_buy_AAA")


def test_stage1_skipped_day_never_inherits_previous_plan(tmp_path, monkeypatch):
    import core.runner as runner_mod
    mc = make_mc(tmp_path)
    prev = {"date": "2026-10-27", "outcome": "submitted", "n_submitted": 30, "run_id": "OLD",
            "plan": {"target_symbols": ["OLD"], "rankings": [["OLD", 1.0]], "target_weights": {"OLD": 1.0}},
            "pending": [{"order_id": "o9", "symbol": "OLD", "side": "buy", "order_type": "limit",
                         "qty": 1, "prev_close": 10.0, "notional_est": 10.0, "run_id": "OLD",
                         "submitted_at": "2026-10-27T13:06:00+00:00", "order_status": "accepted"}]}
    st = {"auction_stage": prev}
    finalized = []
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda mc_, pending, journal, lg: (finalized.append(len(pending)) or {"n": len(pending), "n_filled": 1}))
    # Simulate the stage-1 pre-amble: orphan finalize happens regardless of today's outcome.
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), "2026-10-28")
    assert finalized == [1] and st["auction_stage"]["pending"] == [] and st["auction_stage"]["finalized_late"]
    # A skipped outcome writes a FRESH record; the old plan must not be visible as today's.
    st["auction_stage"] = {"date": "2026-10-28", "outcome": "skipped:window", "n_submitted": 0,
                           "n_skipped": 0, "pending": [], "plan": None}
    assert runner_mod._todays_auction_plan(st, "2026-10-28") is None


def test_dashboard_guard_is_atomic_and_clears_skip_flag(monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", lambda **kw: None)
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {})
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["skipped_main_run_at"] = "2026-10-28T13:35:00+00:00"
    dashboard.run_trading_pipeline(stage="main")
    with dashboard.status_lock:
        assert "skipped_main_run_at" not in dashboard.bot_status
        assert dashboard.bot_status["state"] == "idle" and dashboard.bot_status["running_stage"] is None


# ───────────── review round 3 regressions ───────────────────────────────

def test_unreadable_after_open_does_not_block_completion(tmp_path, monkeypatch):
    """Round 3: after 09:31 every opg order is terminal and the completion
    is delta-based — an unreadable leftover must not defer the day."""
    import core.runner as runner_mod
    st = {"auction_stage": {"date": "2026-10-28", "outcome": "submitted", "n_submitted": 3,
                            "run_id": "RID", "plan": {"target_symbols": ["A"], "rankings": [["A", 1.0]],
                                                      "target_weights": {"A": 1.0}},
                            "pending": [{"order_id": "o7"}]}}
    assert runner_mod._todays_auction_plan(st, "2026-10-28") is not None
    # completed_at spends the plan.
    st["auction_stage"]["completed_at"] = "x"
    assert runner_mod._todays_auction_plan(st, "2026-10-28") is None


def test_finalize_per_order_isolation_and_anchor_fill_rate(tmp_path, monkeypatch):
    """A malformed order payload (filled, no price) must not drop the
    others; the fill rate is measured at the anchor price."""
    def fake_poll(order_id, mc, lg, max_wait=8.0, **kw):
        return {"o1": {"id": "o1", "status": "filled", "filled_qty": "10", "filled_avg_price": "49.0"},
                "o2": {"id": "o2", "status": "filled", "filled_qty": "7", "filled_avg_price": None},
                "o3": {"id": "o3", "status": "canceled", "filled_qty": "0"}}[order_id]
    monkeypatch.setattr(orders_mod, "poll_order_status", fake_poll)
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {s: {"day_open": 49.0} for s in syms})
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)
    mc = make_mc(tmp_path)
    today = datetime.now(NY).strftime("%Y-%m-%d")
    base = {"submitted_at": f"{today}T13:06:00+00:00", "run_id": "RID", "order_status": "accepted"}
    pending = [{**base, "symbol": "AAA", "side": "buy", "order_type": "limit", "qty": 10, "limit_price": 50.5,
                "prev_close": 50.0, "notional_est": 500.0, "order_id": "o1"},
               {**base, "symbol": "BBB", "side": "buy", "order_type": "limit", "qty": 7, "limit_price": 50.5,
                "prev_close": 50.0, "notional_est": 350.0, "order_id": "o2"},
               {**base, "symbol": "CCC", "side": "buy", "order_type": "limit", "qty": 4, "limit_price": 50.5,
                "prev_close": 50.0, "notional_est": 200.0, "order_id": "o3"}]
    journal = StubJournal()
    stats = orders_mod.finalize_auction_orders(mc, pending, journal, logger)
    # o2: status filled, no price -> fill priced at prev_close fallback, still counted.
    assert stats["n_filled"] == 2 and stats["n_unfilled"] == 1 and stats["n_remaining_unread"] == 0
    # Anchor-priced fill rate: (10 + 7) * 50 / 1050 = 0.8095 even though AAA filled at 49 (gap down).
    assert stats["fill_rate_notional"] == pytest.approx(850.0 / 1050.0, abs=1e-3)
    assert stats["filled_notional_usd"] == pytest.approx(10 * 49.0 + 7 * 50.0)
    assert {r.symbol for r in journal.records} == {"AAA", "BBB", "CCC"}


def test_late_finalize_has_no_reference_open(tmp_path, monkeypatch):
    """An earlier session's orders finalized today must not be scored
    against TODAY's open (round 3): reference_open None, excluded from
    the vw cost, but the fill itself is still journaled and counted."""
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, mc, lg, max_wait=8.0, **kw:
                        {"id": oid, "status": "filled", "filled_qty": "10", "filled_avg_price": "101.0"})
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {"AAA": {"day_open": 95.0}})
    mc = make_mc(tmp_path)
    pending = [{"submitted_at": "2026-10-28T13:06:00+00:00", "run_id": "RID", "order_status": "accepted",
                "symbol": "AAA", "side": "buy", "order_type": "limit", "qty": 10, "limit_price": 101.0,
                "prev_close": 100.0, "notional_est": 1000.0, "order_id": "o1"}]
    journal = StubJournal()
    stats = orders_mod.finalize_auction_orders(mc, pending, journal, logger)
    row = journal.records[0]
    assert row.reference_open is None and row.slippage_bps is None
    assert row.fill_price == pytest.approx(101.0) and row.trade_id.endswith("_auction_20261028_buy_AAA")
    assert stats["n_filled"] == 1 and stats["vw_cost_vs_open_bps"] is None
    assert stats["fill_rate_notional"] == pytest.approx(1.0)


def test_finalize_helper_drains_orphans_and_keeps_pending_on_error(tmp_path, monkeypatch):
    import core.runner as runner_mod
    mc = make_mc(tmp_path)
    calls = []

    def fake_finalize(mc_, pending, journal, lg):
        calls.append([p["order_id"] for p in pending])
        if pending and pending[0]["order_id"] == "boom":
            raise RuntimeError("api")
        return {"n": len(pending), "n_filled": len(pending), "remaining": []}
    monkeypatch.setattr("core.orders.finalize_auction_orders", fake_finalize)
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    st = {"auction_stage": {"date": "2026-10-28", "pending": [{"order_id": "boom"}],
                            "orphans": [{"order_id": "old1"}]}}
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), "2026-10-28")
    assert calls == [["old1"], ["boom"]]
    assert st["auction_stage"]["orphans"] == []                      # drained
    assert st["auction_stage"]["pending"] == [{"order_id": "boom"}]   # kept for retry
    assert st["auction_stage"]["summary"]["error"] == "api"


def test_merge_uses_anchor_priced_fills_and_skips_empty(tmp_path):
    import core.runner as runner_mod
    st = {"date": "2026-10-28", "summary": {"n": 2, "n_filled": 2, "submitted_notional_usd": 1000.0,
                                            "filled_notional_anchor_usd": 1000.0,
                                            "filled_notional_usd": 950.0}}     # gap-down fills
    result = {"reconciliation": {"flow": {"n_orders": 0, "n_filled": 0, "fill_rate_notional": None,
                                          "orders_notional_usd": 0.0, "filled_notional_usd": 0.0}, "book": None}}
    runner_mod._merge_auction_into_result(result, st, "2026-10-28")
    assert result["reconciliation"]["flow"]["fill_rate_notional"] == pytest.approx(1.0)
    empty = {"date": "2026-10-28", "summary": {"n": 0}}
    r2 = {"reconciliation": {"flow": {"fill_rate_notional": 0.9}, "book": None}}
    runner_mod._merge_auction_into_result(r2, empty, "2026-10-28")
    assert r2["reconciliation"]["flow"]["fill_rate_notional"] == 0.9 and "auction" not in r2["reconciliation"]


def test_rebalance_rows_tagged_completion_only_when_completing_a_plan(tmp_path, monkeypatch):
    """An 'auction' slot rebalancing WITHOUT a stored plan (stage skipped,
    forced re-run) is an ordinary marketable-limit rebalance (round 3)."""
    posted = []
    polled = {}

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        if method == "POST" and endpoint == "v2/orders":
            posted.append(dict(data))
            oid = f"o{len(posted)}"
            polled[oid] = {"id": oid, "status": "filled", "filled_qty": "1", "filled_avg_price": "100.0"}
            return {"id": oid, "status": "accepted"}
        if method == "GET" and endpoint.startswith("v2/orders/"):
            return polled.get(endpoint.rsplit("/", 1)[-1], {})
        return {}
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc, lg, rp: {"portfolio_value": "100000",
                                                                       "cash": "100000", "buying_power": "200000",
                                                                       "equity": "100000"})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc, lg: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {s: {"day_open": 100.0, "arrival": 100.0,
                                                                                 "prev_close": 100.0, "ask": 100.05,
                                                                                 "bid": 99.95} for s in syms})
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, *a, **kw: polled.get(oid, {"status": "filled", "filled_qty": "1",
                                                               "filled_avg_price": "100.0"}))
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)
    mc = make_mc(tmp_path, exec_style="auction")
    journal = StubJournal()
    rb = orders_mod.rebalance_portfolio(["AAA"], [("AAA", 1.0)], mc, journal, logger, RunReport(),
                                        target_weights={"AAA": 0.05})
    assert rb["exec_style"] == "marketable_limit"
    assert journal.records and all(r.exec_style == "marketable_limit" for r in journal.records)


def test_auction_stage_skips_data_phase_when_no_slot_is_due(tmp_path, monkeypatch):
    import core.runner as runner_mod
    a = make_mc(tmp_path, name="combo_v2_process", exec_style="auction")
    save_state({"last_rebalance": datetime.now().isoformat()}, a)      # rebalanced today -> not due
    monkeypatch.setattr(runner_mod, "get_active_models", lambda: [a])
    monkeypatch.setattr("core.invariants.run_invariant_checks", lambda lg: None)
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: True)

    def boom(*a, **kw):
        raise AssertionError("data phase must not run when no auction slot is due")
    monkeypatch.setattr(runner_mod, "get_tradeable_symbols", boom)
    monkeypatch.setattr(runner_mod, "download_bars", boom)
    runner_mod.run_pipeline(stage="auction")
    assert load_state(a).get("auction_last_run")                         # stamped as a no-op stage run
    # A forced run still goes through (the due-check is bypassed); the
    # start-window check needs a 09:08 ET clock.
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 10, 28, 9, 8, tzinfo=NY)
            return base.astimezone(tz) if tz else base.replace(tzinfo=None)
    monkeypatch.setattr(runner_mod, "datetime", FakeDT)
    called = []
    monkeypatch.setattr(runner_mod, "get_tradeable_symbols", lambda lg, rp: called.append(1) or [])
    runner_mod.run_pipeline(stage="auction", force=True)
    assert called == [1]


# ───────────── enablement: migration + manifest pin ─────────────────────

def test_exec_style_migration_flips_only_the_enabled_process_slot(monkeypatch):
    import core.config as config_mod
    saved = []
    monkeypatch.setattr(config_mod, "save_model_config", lambda cfg: saved.append(dict(cfg)))
    cfg = {"slots": [
        {"slot_id": 1, "model": "combo_v2", "enabled": True, "exec_style": "marketable_limit"},
        {"slot_id": 2, "model": "combo_v2_exp", "enabled": True, "exec_style": "marketable_limit"},
        {"slot_id": 3, "model": "combo_v2_process", "enabled": True, "exec_style": "marketable_limit",
         "exec_limit_buffer_bps": 10.0},
        {"slot_id": 4, "model": "combo_v2_process", "enabled": False, "exec_style": "marketable_limit"},
    ]}
    out = config_mod._migrate_exec_style(cfg)
    by_id = {s["slot_id"]: s for s in out["slots"]}
    assert by_id[3]["exec_style"] == "auction"
    assert by_id[3]["exec_auction_buffer_bps"] == 100.0 and by_id[3]["exec_limit_buffer_bps"] == 10.0
    assert by_id[1]["exec_style"] == "marketable_limit" and by_id[2]["exec_style"] == "marketable_limit"
    assert by_id[4]["exec_style"] == "marketable_limit"          # disabled slot untouched
    assert out["exec_style_migration"] == config_mod.EXEC_STYLE_VERSION and len(saved) == 1
    # Idempotent: a second call neither rewrites nor re-saves.
    config_mod._migrate_exec_style(out)
    assert len(saved) == 1
    # One-shot: an operator REVERT after the stamp is never re-applied.
    out["slots"][2]["exec_style"] = "marketable_limit"
    config_mod._migrate_exec_style(out)
    assert out["slots"][2]["exec_style"] == "marketable_limit" and len(saved) == 1


def test_manifest_pins_auction_on_the_process_slot_only():
    import json
    from pathlib import Path
    import core.invariants as inv
    m = json.loads((Path(inv.MANIFEST_PATH)).read_text())
    assert m["slots"]["combo_v2_process"]["exec_style"] == "auction"
    assert m["slots"]["combo_v2_process"]["exec_auction_buffer_bps"] == 100.0
    assert m["slots"]["combo_v2"]["exec_style"] == "marketable_limit"
    assert m["slots"]["combo_v2_exp"]["exec_style"] == "marketable_limit"
    assert "exec_auction_buffer_bps" in inv._SLOT_FIELDS
