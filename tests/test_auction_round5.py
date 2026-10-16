"""Opening-auction execution — review round 5 regressions (2026-10-15/16).

Covers: equity in the rebalance result (shortfall logic alive), limit-priced
buy sizing, dot-form symbol lookups, terminal-state discipline after a
cancel, vw-cost weights across finalize passes, journal newline guard and
glued-line id recovery, venue-side recovery of unpersisted orders, the
provisional 'submitting' record + incremental persistence, deferred
completion before the market-closed test, dry runs never driving the
outage counter, unmeasured-auction merge + history repair, typed slot
failures in the dashboard, boot-time catch-up, the auction stage waiting
for a concurrent run, the /run route guard, the plan-not-completed alarm,
the typed Alpaca error through the real request path, ET session-date
keying under a UTC container clock, and the /api/status auction block.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATA_DIR", os.environ.get("DATA_DIR") or tempfile.mkdtemp(prefix="combo_test_r5_"))

import pandas as pd
import pytest

from core.config import ModelConfig
import core.alpaca as alpaca_mod
import core.orders as orders_mod
import core.runner as runner_mod
from core.run_report import RunReport
from core.state import load_state, save_state

NY = ZoneInfo("America/New_York")
logger = logging.getLogger("test_auction_r5")


def make_mc(tmp_path, **kw) -> ModelConfig:
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_file = tmp_path / "model.pkl"
    model_file.write_bytes(b"")
    defaults = dict(name="combo_v2_process", model_path=model_file,
                    feature_version="combo", alpaca_key="k", alpaca_secret="s",
                    state_path=tmp_path / "state.json", enable_cutloss=True,
                    target_leverage=2.0, exec_style="auction")
    defaults.update(kw)
    return ModelConfig(**defaults)


class StubJournal:
    def __init__(self, path=None):
        self.records = []
        self.jsonl_path = path or "/dev/null"

    def log_trade(self, record):
        self.records.append(record)


def _today_ny() -> str:
    return datetime.now(NY).strftime("%Y-%m-%d")


def _pending(order_id, sym="AAA", qty=10, prev_close=50.0, side="buy", **extra):
    d = {"submitted_at": f"{_today_ny()}T13:06:00+00:00", "run_id": "RID", "order_status": "accepted",
         "symbol": sym, "side": side, "order_type": "limit" if side == "buy" else "market", "qty": qty,
         "limit_price": round(prev_close * 1.01, 2) if side == "buy" else None,
         "prev_close": prev_close, "notional_est": qty * prev_close, "order_id": order_id}
    d.update(extra)
    return d


def _no_sleep(monkeypatch):
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)


# ───────────────────────── orders layer ──────────────────────────────────

def test_rebalance_result_carries_equity_so_shortfall_logic_is_alive(tmp_path, monkeypatch):
    from core.gate_update import _shortfall_from_result
    posted, polled = [], {}

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
    monkeypatch.setattr(orders_mod, "get_account", lambda mc, lg, rp: {"portfolio_value": "100000", "cash": "100000",
                                                                       "buying_power": "200000", "equity": "100000"})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc, lg: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {s: {"day_open": 100.0, "arrival": 100.0,
                                                                                 "prev_close": 100.0, "ask": 100.05,
                                                                                 "bid": 99.95} for s in syms})
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, *a, **kw: polled.get(oid, {"status": "filled", "filled_qty": "1",
                                                               "filled_avg_price": "100.0"}))
    _no_sleep(monkeypatch)
    mc = make_mc(tmp_path, exec_style="marketable_limit")
    rb = orders_mod.rebalance_portfolio(["AAA"], [("AAA", 1.0)], mc, StubJournal(), logger, RunReport(),
                                        target_weights={"AAA": 0.05})
    assert rb["portfolio_value"] == pytest.approx(100_000.0)
    m_real, shortfall, _ = _shortfall_from_result(rb, gross_pre=1.0, applied_design=1.0, leverage=2.0)
    assert m_real is not None and shortfall is not None       # was (None, None, False) in production


def test_auction_buy_qty_is_sized_at_the_limit_price():
    buy = {"symbol": "AAA", "side": "buy", "action": "buy", "notional": 10_000.0}
    assert orders_mod._auction_qty(buy, 100.0) == 100.0
    assert orders_mod._auction_qty(buy, 100.0, buy_limit_mult=1.01) == 99.0      # 99 x 101 <= 10,000
    sell = {"symbol": "AAA", "side": "sell", "action": "sell", "qty": 10.7}
    assert orders_mod._auction_qty(sell, 100.0, buy_limit_mult=1.01) == 10.0


def test_dot_form_exit_matches_stale_and_panel_lookups(tmp_path, monkeypatch):
    posted = []

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        posted.append(dict(data or {}))
        return {"id": f"o{len(posted)}", "status": "accepted"}
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    orders = [{"symbol": "BRK.B", "side": "sell", "action": "sell", "qty": 10}]
    summ = orders_mod.submit_auction_orders(orders, make_mc(tmp_path), {"BRK-B": 480.0}, logger, {},
                                            run_id="R", stale_close={"BRK-B"})
    assert summ["n_submitted"] == 0 and summ["skipped"][0]["reason"] == "stale_close"
    # finalize: a dot-form row finds the dash-keyed panel open.
    _no_sleep(monkeypatch)
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, mc, lg, max_wait=8.0, **kw:
                        {"id": oid, "status": "filled", "filled_qty": "10", "filled_avg_price": "481.0"})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {})
    j = StubJournal()
    stats = orders_mod.finalize_auction_orders(make_mc(tmp_path), [_pending("o1", "BRK.B", 10, 480.0, side="sell")],
                                               j, logger, panel_open={"BRK-B": 482.0})
    assert j.records[0].reference_open == 482.0 and stats["reference_sources"]["panel"] == 1


def test_non_terminal_state_after_cancel_stays_pending_then_counts_the_cancel(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    reads = []

    def fake_poll(oid, mc, lg, max_wait=8.0, **kw):
        reads.append(oid)
        if len(reads) <= 2:       # first read AND post-cancel read: still partially filled
            return {"id": oid, "status": "partially_filled", "filled_qty": "60", "filled_avg_price": "100.2"}
        return {"id": oid, "status": "canceled", "filled_qty": "60", "filled_avg_price": "100.2"}
    monkeypatch.setattr(orders_mod, "poll_order_status", fake_poll)
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {"AAA": {"day_open": 100.0, "day_bar_date": _today_ny()}})
    mc = make_mc(tmp_path)
    j = StubJournal()
    stats = orders_mod.finalize_auction_orders(mc, [_pending("o1", "AAA", 100, 100.0)], j, logger)
    assert stats["n"] == 0 and j.records == [] and stats["remaining"][0]["cancel_sent"] is True
    # Second pass: terminal now; the cancel sent earlier is counted.
    stats2 = orders_mod.finalize_auction_orders(mc, stats["remaining"], j, logger)
    assert stats2["n"] == 1 and stats2["n_filled"] == 1 and stats2["n_cancelled_open"] == 1
    assert j.records[0].shares == 60 and j.records[0].order_status == "canceled"
    assert stats2["fill_rate_notional"] == pytest.approx(0.6)


def test_vw_cost_is_weighted_by_referenced_dollars_across_passes(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, mc, lg, max_wait=8.0, **kw:
                        {"id": oid, "status": "filled", "filled_qty": "100", "filled_avg_price": "101.0"})
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    # AAA has a same-day reference (cost +100bp); BBB has none.
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {"AAA": {"day_open": 100.0, "day_bar_date": _today_ny()}})
    stats = orders_mod.finalize_auction_orders(make_mc(tmp_path),
                                               [_pending("o1", "AAA", 100, 100.0), _pending("o2", "BBB", 100, 100.0)],
                                               StubJournal(), logger)
    assert stats["vw_cost_vs_open_bps"] == pytest.approx(100.0) and stats["vw_cost_weight_usd"] == pytest.approx(10_100.0)
    assert stats["filled_notional_usd"] == pytest.approx(20_200.0)
    new = {"n": 1, "n_filled": 1, "filled_notional_usd": 10_100.0, "vw_cost_vs_open_bps": -200.0,
           "vw_cost_weight_usd": 10_100.0, "submitted_notional_usd": 10_000.0,
           "filled_notional_anchor_usd": 10_000.0}
    out = runner_mod._accumulate_auction_summary(stats, new)
    assert out["vw_cost_vs_open_bps"] == pytest.approx(-50.0)        # (100 - 200) / 2, not weighted by BBB's dollars
    assert out["vw_cost_weight_usd"] == pytest.approx(20_200.0)


def test_journal_append_never_glues_onto_a_truncated_line(tmp_path, monkeypatch):
    from core.journal import TradeJournal, TradeRecord
    import core.journal as journal_mod
    monkeypatch.setattr(journal_mod, "_TRADE_DIR", tmp_path)
    j = TradeJournal("rtest")
    j.jsonl_path.write_text('{"trade_id": "rtest_auction_20261028_buy_AAA", "symbol": "AAA"}')   # no newline
    rec = TradeRecord(trade_id="t2", run_id="r", model="rtest", timestamp="x", symbol="BBB", side="buy",
                      action="new_position", order_type="limit", time_in_force="opg", notional_usd=1.0)
    j.log_trade(rec)
    lines = [l for l in j.jsonl_path.read_text().split("\n") if l.strip()]
    assert len(lines) == 2 and all(json.loads(l) for l in lines)
    # And the dedupe scan recovers ids even from a glued line.
    glued = tmp_path / "glued.jsonl"
    glued.write_text('{"trade_id": "m_auction_20261028_buy_AAA"}{"trade_id": "m_auction_20261028_buy_BBB"}\n')
    ids = orders_mod._existing_auction_trade_ids(StubJournal(path=str(glued)), "m_auction_")
    assert ids == {"m_auction_20261028_buy_AAA", "m_auction_20261028_buy_BBB"}


def test_recover_unpersisted_auction_orders_adopts_todays_prefix_only(tmp_path, monkeypatch):
    day = _today_ny().replace("-", "")
    venue = [
        {"id": "v1", "client_order_id": f"a{day}_BRK.B_abc", "time_in_force": "opg", "side": "buy",
         "type": "limit", "limit_price": "101", "qty": "10", "status": "filled", "symbol": "BRK.B",
         "submitted_at": f"{_today_ny()}T13:07:00Z"},
        {"id": "v2", "client_order_id": f"a{day}_CCC_abc", "time_in_force": "opg", "side": "sell",
         "type": "market", "qty": "5", "status": "filled", "symbol": "CCC"},
        {"id": "known", "client_order_id": f"a{day}_DDD_abc", "time_in_force": "opg", "side": "buy",
         "type": "limit", "limit_price": "50.5", "qty": "1", "status": "filled", "symbol": "DDD"},
        {"id": "other", "client_order_id": "manual-123", "time_in_force": "day", "side": "buy",
         "type": "market", "qty": "1", "status": "filled", "symbol": "EEE"},
    ]
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda m, e, mc, data=None, logger=None, **kw: venue)
    found = orders_mod.recover_unpersisted_auction_orders(make_mc(tmp_path), _today_ny(), {"known"}, logger)
    by = {f["symbol"]: f for f in found}
    assert set(by) == {"BRK-B", "CCC"}
    assert by["BRK-B"]["prev_close"] == pytest.approx(100.0) and by["BRK-B"]["notional_est"] == pytest.approx(1000.0)
    assert by["BRK-B"]["recovered"] is True and by["CCC"]["prev_close"] is None


# ───────────────────────── runner ────────────────────────────────────────

class FakeModel:
    def __init__(self, weights):
        self._w = weights
        self.last_diagnostics = {"exposure_multiplier": 1.0}

    def compute_weights(self, stock_data, macro_data):
        return dict(self._w)


def _wire_runner(monkeypatch, weights, rebalance_result=None, calls=None, persist=True):
    calls = calls if calls is not None else []
    monkeypatch.setattr(runner_mod, "load_model_bundle", lambda p: {
        "model": FakeModel(weights), "feature_cols": [], "strategy_type": "direct_weights",
        "horizon": 21, "task": "w", "saved_at": "x", "combo_config": None})
    monkeypatch.setattr(runner_mod, "MIN_STOCKS_REQUIRED", 1)
    monkeypatch.setattr(runner_mod, "MIN_MACRO_FEATURES", 0)
    monkeypatch.setattr(runner_mod, "is_market_open", lambda now=None: True)
    monkeypatch.setattr(runner_mod, "fetch_inactive_assets", lambda syms, mc, lg, top_n=20: set())
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: True)
    monkeypatch.setattr("core.risk.refresh_daily_book_anchor_after_rebalance", lambda *a, **kw: None)
    monkeypatch.setattr("core.protocol.bind_forward_test_start", lambda *a, **kw: False)

    def fake_rebalance(target_symbols, rankings, mc, journal, lg, report, **kw):
        calls.append({"target_symbols": list(target_symbols), **kw})
        if rebalance_result is not None:
            return dict(rebalance_result)
        pend = [{"order_id": "o1", "symbol": "AAA", "notional_est": 500.0},
                {"order_id": "o2", "symbol": "BBB", "notional_est": 500.0}]
        cb = kw.get("on_submitted")
        if cb is not None and persist:
            for e in pend:
                cb(e, "RID")
        return {"dry_run": False, "auction": {"n_submitted": 2, "n_skipped": 0, "skipped": [],
                                               "submitted_notional_usd": 1000.0, "run_id": "RID", "pending": pend},
                "run_id": "RID"}
    monkeypatch.setattr(runner_mod, "rebalance_portfolio", fake_rebalance)
    return calls


def _panel(dates, close=50.0):
    idx = pd.to_datetime(dates)
    return {"AAA": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx),
            "BBB": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx)}


def _plan_state(today, pending=(), outcome="submitted", **extra):
    st = {"last_rebalance": (datetime.now() - timedelta(days=1)).isoformat(), "history": [],
          "auction_stage": {"date": today, "outcome": outcome, "n_submitted": 2 if outcome == "submitted" else 0,
                            "run_id": "RID",
                            "plan": {"target_symbols": ["AAA", "BBB"], "rankings": [["AAA", 0.6], ["BBB", 0.4]],
                                     "target_weights": {"AAA": 0.6, "BBB": 0.4},
                                     "diagnostics": {"exposure_multiplier": 0.7}},
                            "pending": list(pending)}}
    st.update(extra)
    return st


def test_stage_persists_each_order_before_the_loop_returns(tmp_path, monkeypatch):
    """The provisional 'submitting' record exists before the first POST and
    every accepted order is appended on disk by the callback."""
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev = runner_mod._previous_session_iso(today)
    seen_during = []

    def spy_rebalance(target_symbols, rankings, mc_, journal, lg, report, **kw):
        seen_during.append(dict(load_state(mc_)["auction_stage"]))          # BEFORE any POST
        kw["on_submitted"]({"order_id": "o1", "symbol": "AAA", "notional_est": 500.0}, "RID")
        seen_during.append(dict(load_state(mc_)["auction_stage"]))          # after the first POST
        return {"dry_run": False, "auction": {"n_submitted": 1, "n_skipped": 0, "skipped": [],
                                               "submitted_notional_usd": 500.0, "run_id": "RID",
                                               "pending": [{"order_id": "o1", "symbol": "AAA", "notional_est": 500.0}]},
                "run_id": "RID"}
    _wire_runner(monkeypatch, {"AAA": 0.5, "BBB": 0.5})
    monkeypatch.setattr(runner_mod, "rebalance_portfolio", spy_rebalance)
    save_state({"last_rebalance": None, "history": []}, mc)
    runner_mod.run_single_model(mc, _panel([prev]), pd.DataFrame(), macro_data={}, stage="auction")
    assert seen_during[0]["outcome"] == "submitting" and seen_during[0]["pending"] == [] and seen_during[0]["plan"]
    assert seen_during[1]["pending"][0]["order_id"] == "o1" and seen_during[1]["run_id"] == "RID"
    final = load_state(mc)["auction_stage"]
    assert final["outcome"] == "submitted" and final["n_submitted"] == 1 and final["run_id"] == "RID"


def test_stage_never_resubmits_after_a_run_died_while_submitting(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev = runner_mod._previous_session_iso(today)
    calls = _wire_runner(monkeypatch, {"AAA": 0.5, "BBB": 0.5})
    save_state(_plan_state(today, outcome="submitting", last_rebalance=None), mc)
    runner_mod.run_single_model(mc, _panel([prev]), pd.DataFrame(), macro_data={}, stage="auction")
    assert calls == [] and load_state(mc)["auction_stage"]["outcome"] == "skipped:prior_submitting"


def test_main_adopts_a_dead_submitting_record_and_completes_it(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    result = {"dry_run": False, "reconciliation": {"flow": {"n_orders": 1, "n_filled": 1, "fill_rate_notional": 1.0,
                                                            "orders_notional_usd": 100.0, "filled_notional_usd": 100.0},
                                                   "book": None}}
    calls = _wire_runner(monkeypatch, {"ZZZ": 1.0}, rebalance_result=result)
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    recovered = [{"order_id": "v9", "symbol": "BBB", "side": "buy", "qty": 1.0, "prev_close": 50.0,
                  "notional_est": 50.0, "submitted_at": f"{today}T13:07:00+00:00", "recovered": True}]
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda mc_, s, known, lg: recovered)
    fin = []
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda mc_, pending, journal, lg, **kw: fin.append([p["order_id"] for p in pending]) or
                        {"n": len(pending), "n_filled": len(pending), "n_unfilled": 0, "n_cancelled_open": 0,
                         "submitted_notional_usd": 550.0, "filled_notional_anchor_usd": 550.0,
                         "filled_notional_usd": 550.0, "fill_rate_notional": 1.0, "vw_cost_vs_open_bps": 0.0,
                         "vw_cost_weight_usd": 550.0, "n_remaining_unread": 0, "remaining": []})
    save_state(_plan_state(today, outcome="submitting",
                           pending=[{"order_id": "o1", "symbol": "AAA", "notional_est": 500.0,
                                     "submitted_at": f"{today}T13:06:00+00:00"}]), mc)
    runner_mod.run_single_model(mc, {}, pd.DataFrame(), macro_data={}, stage="main")
    st = load_state(mc)
    assert fin == [["o1", "v9"]]                                       # persisted + venue-recovered, finalized together
    assert st["auction_stage"]["recovered_from_submitting"] is True and st["auction_stage"]["n_recovered"] == 1
    assert calls and calls[0]["run_id"] == "RID" and st["auction_stage"]["completed_at"]
    assert st["history"][-1]["result"]["reconciliation"]["auction"]["n"] == 2


def test_pre_open_manual_run_on_a_plan_day_defers_instead_of_market_closed(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    calls = _wire_runner(monkeypatch, {"AAA": 1.0})
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: False)
    monkeypatch.setattr(runner_mod, "is_market_open", lambda now=None: False)         # 09:28 ET
    save_state(_plan_state(today, pending=[{"order_id": "o1", "submitted_at": f"{today}T13:06:00+00:00"}]), mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(today)]), pd.DataFrame(),
                                macro_data={}, stage="main")
    st = load_state(mc)
    assert calls == [] and st["auction_completion_deferred"] == today
    assert st["history"][-1]["result"].get("skipped_auction_pending") is True
    assert not st["history"][-1]["result"].get("skipped_market_closed")


def test_dry_runs_never_drive_the_outage_counter(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    _wire_runner(monkeypatch, {"AAA": 1.0})
    monkeypatch.setattr(runner_mod, "MIN_STOCKS_REQUIRED", 400)
    st0 = _plan_state(today)
    st0["data_guard_aborts"] = 1
    save_state(st0, mc)
    ran = runner_mod._run_plan_only_slots([mc], logger, dry_run=True, force=True, top_n=20, horizon=21, reason="x")
    assert ran == [] and load_state(mc)["data_guard_aborts"] == 1
    runner_mod.run_single_model(mc, {}, pd.DataFrame(), macro_data={}, dry_run=True, stage="main")
    assert load_state(mc)["data_guard_aborts"] == 1


def test_unmeasured_auction_leg_reads_as_na_not_as_the_completion_rate():
    flow = {"n_orders": 6, "n_filled": 6, "fill_rate_notional": 1.0, "orders_notional_usd": 600.0,
            "filled_notional_usd": 600.0}
    result = {"reconciliation": {"flow": dict(flow), "book": None}}
    st = {"date": "D", "n_submitted": 40, "pending": [{"order_id": "x"}]}          # finalize failed: no summary
    runner_mod._merge_auction_into_result(result, st, "D")
    f = result["reconciliation"]["flow"]
    assert f["fill_rate_notional"] is None and f["auction_unmeasured"] is True and f["fill_rate_notional_completion"] == 1.0
    st2 = {"date": "D", "n_submitted": 40, "summary": {"n": 0}}                     # every entry unreadable
    r2 = {"reconciliation": {"flow": dict(flow), "book": None}}
    runner_mod._merge_auction_into_result(r2, st2, "D")
    assert r2["reconciliation"]["flow"]["fill_rate_notional"] is None
    st3 = {"date": "D", "n_submitted": 2, "summary": {"n": 2, "n_filled": 2, "submitted_notional_usd": 200.0,
                                                      "filled_notional_anchor_usd": 200.0, "filled_notional_usd": 200.0}}
    runner_mod._merge_auction_into_result(result, st3, "D")                         # measured later: repaired
    assert "auction_unmeasured" not in result["reconciliation"]["flow"]
    assert result["reconciliation"]["flow"]["fill_rate_notional"] == pytest.approx(1.0)


def test_late_first_measurement_repairs_the_completed_runs_history(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda mc_, pending, journal, lg, **kw: {"n": 2, "n_filled": 2, "n_unfilled": 0,
                                                                 "n_cancelled_open": 0, "submitted_notional_usd": 200.0,
                                                                 "filled_notional_anchor_usd": 200.0,
                                                                 "filled_notional_usd": 200.0, "fill_rate_notional": 1.0,
                                                                 "vw_cost_vs_open_bps": 3.0, "vw_cost_weight_usd": 200.0,
                                                                 "n_remaining_unread": 0, "remaining": []})
    hist = {"auction_run_id": "RID", "result": {"reconciliation": {"flow": {"fill_rate_notional": None,
                                                                            "auction_unmeasured": True,
                                                                            "fill_rate_notional_completion": 1.0,
                                                                            "orders_notional_usd": 100.0,
                                                                            "filled_notional_usd": 100.0,
                                                                            "n_orders": 1, "n_filled": 1},
                                                                   "book": None}}}
    st = {"history": [hist], "auction_stage": {"date": today, "outcome": "submitted", "n_submitted": 2, "run_id": "RID",
                                               "completed_at": "x", "summary_error": "api",
                                               "pending": [{"order_id": "o1", "symbol": "AAA"},
                                                           {"order_id": "o2", "symbol": "BBB"}]}}
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    f = st["history"][0]["result"]["reconciliation"]["flow"]
    assert "auction_unmeasured" not in f and f["fill_rate_notional"] == pytest.approx(1.0) and f["n_orders"] == 3
    assert "summary_error" not in st["auction_stage"] and st["auction_stage"]["summary"]["n"] == 2


def test_pipeline_raises_typed_slot_failures(tmp_path, monkeypatch):
    a = make_mc(tmp_path / "a", exec_style="auction")
    b = make_mc(tmp_path / "b", name="combo_v2", exec_style="marketable_limit")
    monkeypatch.setattr(runner_mod, "get_active_models", lambda: [a, b])
    monkeypatch.setattr("core.invariants.run_invariant_checks", lambda lg: None)
    monkeypatch.setattr(runner_mod, "get_tradeable_symbols", lambda lg, rp: ["X"])
    monkeypatch.setattr(runner_mod, "download_bars", lambda syms, days, lg, rp: {"X": object()})
    monkeypatch.setattr(runner_mod, "download_macro", lambda days, lg, rp: {})
    monkeypatch.setattr(runner_mod, "compute_macro_features", lambda m: pd.DataFrame())

    def rsm(mc, *a_, **kw):
        if mc.name == "combo_v2_process":
            raise RuntimeError("positions call failed")
    monkeypatch.setattr(runner_mod, "run_single_model", rsm)
    with pytest.raises(runner_mod.SlotFailures) as ei:
        runner_mod.run_pipeline(stage="main")
    assert set(ei.value.failed) == {"combo_v2_process"}


def test_utc_container_clock_keys_the_auction_record_on_the_et_session_date(tmp_path, monkeypatch):
    """At 22:30 ET the container's UTC date is already tomorrow: the stage
    record must carry the New York date and be found by the same key."""
    mc = make_mc(tmp_path)
    base = datetime(2026, 10, 27, 22, 30, tzinfo=NY)                      # 02:30 UTC Oct 28

    class UTCClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return base.astimezone(tz) if tz else base.astimezone(timezone.utc).replace(tzinfo=None)
    monkeypatch.setattr(runner_mod, "datetime", UTCClock)
    calls = _wire_runner(monkeypatch, {"AAA": 0.5, "BBB": 0.5})
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: True)   # window check is faked
    save_state({"last_rebalance": None, "history": []}, mc)
    prev = runner_mod._previous_session_iso("2026-10-27")
    runner_mod.run_single_model(mc, _panel([prev]), pd.DataFrame(), macro_data={}, stage="auction")
    st = load_state(mc)
    assert st["auction_stage"]["date"] == "2026-10-27"                       # ET session date, not the UTC date
    assert runner_mod._todays_auction_plan(st, "2026-10-27") is not None
    assert runner_mod._todays_auction_plan(st, "2026-10-28") is None


# ───────────────────────── Alpaca client ─────────────────────────────────

def test_alpaca_request_raises_typed_error_and_poll_maps_404(tmp_path, monkeypatch):
    import requests

    class Resp:
        status_code = 404
        text = "order not found"

        def json(self):
            return {}
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=15: Resp())
    mc = make_mc(tmp_path)
    with pytest.raises(alpaca_mod.AlpacaHTTPError) as ei:
        alpaca_mod.alpaca_request("GET", "v2/orders/x", mc)
    assert ei.value.status_code == 404
    assert orders_mod.poll_order_status("x", mc, logger, max_wait=0.2)["status"] == "not_found"


# ───────────────────────── dashboard ─────────────────────────────────────

def _dash_clock(monkeypatch, dashboard, h, m, d=28):
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 10, d, h, m, tzinfo=NY)
            return base.astimezone(tz) if tz else base.replace(tzinfo=None)
    monkeypatch.setattr(dashboard, "datetime", FakeDT)


def test_dashboard_wait_loop_uses_the_new_york_cutoff(monkeypatch):
    import dashboard
    seen, slept = [], []
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", lambda **kw: seen.append(kw))
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {})

    def fake_sleep(s):
        slept.append(s)
        with dashboard.status_lock:
            dashboard.bot_status["state"] = "idle"
            dashboard.bot_status["running_stage"] = None
    monkeypatch.setattr(dashboard.time, "sleep", fake_sleep)
    for (h, m), expect_wait in (((13, 0), True), ((15, 29), True), ((15, 30), False)):
        seen.clear(); slept.clear()
        _dash_clock(monkeypatch, dashboard, h, m)
        with dashboard.status_lock:
            dashboard.bot_status["state"] = "running"
            dashboard.bot_status["running_stage"] = "auction"
        dashboard.run_trading_pipeline(stage="main")
        assert (slept == [15]) is expect_wait and bool(seen) is expect_wait, (h, m)
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["running_stage"] = None
        dashboard.bot_status.pop("skipped_main_run_at", None)
        dashboard.bot_status.pop("skipped_main_run_blocker", None)


def test_auction_stage_waits_for_a_concurrent_run_then_records_due_slots(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path, exec_style="auction")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {"last_rebalance": None})
    seen, slept = [], []
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", lambda **kw: seen.append(kw))

    def fake_sleep(s):
        slept.append(s)
        with dashboard.status_lock:
            dashboard.bot_status["state"] = "idle"
            dashboard.bot_status["running_stage"] = None
    monkeypatch.setattr(dashboard.time, "sleep", fake_sleep)
    _dash_clock(monkeypatch, dashboard, 9, 7)
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "running"
        dashboard.bot_status["running_stage"] = "dry"
    dashboard.run_trading_pipeline(stage="auction")
    assert slept == [15] and seen and seen[0]["stage"] == "auction"        # waited, then ran
    # Past the 09:20 start cutoff it yields and records WHICH slots were due.
    seen.clear(); slept.clear()
    _dash_clock(monkeypatch, dashboard, 9, 21)
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "running"
        dashboard.bot_status["running_stage"] = "main"
    dashboard.run_trading_pipeline(stage="auction")
    with dashboard.status_lock:
        assert seen == [] and dashboard.bot_status["auction_last_run_status"] == "skipped"
        assert dashboard.bot_status["auction_skipped_due"] == ["combo_v2_process"]
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["running_stage"] = None
        dashboard.bot_status["auction_last_run_status"] = None
        dashboard.bot_status["auction_last_run_at"] = None
        dashboard.bot_status.pop("auction_skipped_due", None)


def test_slot_failures_mark_only_the_failed_slot(monkeypatch):
    import dashboard

    def boom(**kw):
        raise runner_mod.SlotFailures({"combo_v2_process": "positions call failed"})
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", boom)
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {"run_count": 7})
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["models"]["combo_v2"] = {"state": "idle"}
        dashboard.bot_status["models"]["combo_v2_process"] = {"state": "idle"}
        dashboard.bot_status.pop("last_full_run_at", None)
    dashboard.run_trading_pipeline(stage="main")
    with dashboard.status_lock:
        m = dashboard.bot_status["models"]
        assert m["combo_v2"]["state"] == "idle" and m["combo_v2"]["last_run_status"] == "success"
        assert m["combo_v2"]["run_count"] == 7
        assert m["combo_v2_process"]["state"] == "error"
        assert dashboard.bot_status["last_run_status"] == "error" and dashboard.bot_status.get("last_full_run_at")
        for k in ("combo_v2", "combo_v2_process"):
            dashboard.bot_status["models"].pop(k, None)
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["last_run_status"] = None


def test_boot_catchup_plan():
    import dashboard
    today = datetime(2026, 10, 28, 10, 0, tzinfo=NY)
    yday = "2026-10-27T13:35:00"
    # Saturday: nothing.
    assert dashboard._boot_catchup_plan(datetime(2026, 10, 31, 10, 0, tzinfo=NY), {"a": {}}, set()) == \
        {"main": False, "auction": False, "reasons": []}
    # 10:00 ET, a slot has not run today -> main.
    p = dashboard._boot_catchup_plan(today, {"a": {"last_run": yday}}, set())
    assert p["main"] is True and not p["auction"]
    # Already ran today -> nothing.
    p = dashboard._boot_catchup_plan(today, {"a": {"last_run": "2026-10-28T13:36:00"}}, set())
    assert p["main"] is False
    # Ran today but today's plan is still open -> main.
    p = dashboard._boot_catchup_plan(today, {"a": {"last_run": "2026-10-28T13:00:00",
                                                   "auction_stage": {"date": "2026-10-28", "outcome": "submitted",
                                                                     "n_submitted": 3}}}, set())
    assert p["main"] is True and "open auction plan" in p["reasons"][0]
    # 09:10 ET, auction slot due with no stage record -> auction; with a record -> nothing.
    nine_ten = datetime(2026, 10, 28, 9, 10, tzinfo=NY)
    p = dashboard._boot_catchup_plan(nine_ten, {"a": {"last_rebalance": None}}, {"a"})
    assert p["auction"] is True and p["main"] is False
    p = dashboard._boot_catchup_plan(nine_ten, {"a": {"last_rebalance": None,
                                                      "auction_stage": {"date": "2026-10-28", "outcome": "submitted"}}}, {"a"})
    assert p["auction"] is False
    # 15:50 ET: too late for a main catch-up.
    assert dashboard._boot_catchup_plan(datetime(2026, 10, 28, 15, 50, tzinfo=NY), {"a": {"last_run": yday}}, set())["main"] is False


def _ready_client(monkeypatch, tmp_path, mc, state):
    import dashboard
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr("core.market.is_market_open", lambda now=None: False)
    monkeypatch.setattr(dashboard, "STATE_DIR", tmp_path)
    (tmp_path / "x.json").write_text("{}")
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: dict(state))
    monkeypatch.setattr("core.invariants.load_last_status", lambda: None)
    monkeypatch.setitem(dashboard.bot_status, "last_run_status", "success")
    monkeypatch.setitem(dashboard.bot_status, "last_run_at", datetime.now(timezone.utc).isoformat())
    monkeypatch.setitem(dashboard.bot_status, "last_error", None)
    for k in ("skipped_main_run_at", "skipped_main_run_blocker", "auction_skipped_due", "last_full_run_at"):
        monkeypatch.delitem(dashboard.bot_status, k, raising=False)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", None)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_at", None)
    return dashboard, dashboard.app.test_client()


def test_ready_flags_a_plan_not_completed_after_ten_et(tmp_path, monkeypatch):
    mc = make_mc(tmp_path, exec_style="auction")
    today = _today_ny()
    state = {"auction_stage": {"date": today, "outcome": "submitted", "n_submitted": 3}}
    dashboard, c = _ready_client(monkeypatch, tmp_path, mc, state)
    now_et = datetime.now(NY)

    def clock(h, m):
        class FakeDT(datetime):
            @classmethod
            def now(cls, tz=None):
                base = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
                return base.astimezone(tz) if tz else base.replace(tzinfo=None)
        monkeypatch.setattr(dashboard, "datetime", FakeDT)
    clock(9, 50)
    assert c.get("/ready").status_code == 200
    clock(10, 5)
    r = c.get("/ready")
    assert r.status_code == 503 and any("not completed by a main run" in p for p in r.get_json()["problems"])
    state["auction_stage"]["completed_at"] = "x"
    assert c.get("/ready").status_code == 200


def test_run_route_refuses_in_window_when_due_and_allows_after_the_stage_ran(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path, exec_style="auction")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", lambda **kw: None)
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: True)
    monkeypatch.setattr(dashboard, "DASHBOARD_AUTH_TOKEN", None)
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {"last_rebalance": None})
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "idle"
    c = dashboard.app.test_client()
    r = c.get("/run")
    assert r.status_code == 409 and "submission window" in r.get_json()["error"]
    assert c.get("/run?dry=1").status_code == 409
    monkeypatch.setattr(dashboard, "_load_model_state",
                        lambda name: {"last_rebalance": None,
                                      "auction_stage": {"date": _today_ny(), "outcome": "skipped:stale_panel"}})
    r = c.get("/run?dry=1")
    assert r.status_code == 200 and r.get_json()["status"] == "triggered"
    import time as _t
    _t.sleep(0.2)
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["running_stage"] = None


def test_api_status_auction_block_exposes_counts_never_the_raw_pending_list(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path, exec_style="auction")
    today = _today_ny()
    state = {"auction_stage": {"date": today, "outcome": "submitted", "n_submitted": 3, "completed_at": "x",
                               "pending": [{"order_id": "a"}, {"order_id": "b"}], "orphans": [{"order_id": "c"}],
                               "plan": {"target_symbols": ["AAA"]}},
             "auction_completion_deferred": today, "history": []}
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: dict(state))
    monkeypatch.setattr(dashboard, "DASHBOARD_AUTH_TOKEN", None)
    body = dashboard.app.test_client().get("/api/status").get_json()
    a = body["models"]["combo_v2_process"]["auction"]
    assert a["n_pending"] == 2 and a["n_orphans"] == 1 and a["completed_at"] == "x"
    assert a["completion_deferred"] == today and "pending" not in a and "plan" not in a


def test_shortfall_recovers_equity_from_the_books_leverage_for_old_results():
    """Sep 29 history results carry no portfolio_value; the book's
    achieved_gross / achieved_leverage gives the equity back."""
    from core.gate_update import _shortfall_from_result
    res = {"target_gross_usd": 176_496.0, "auto_scaled_to_buying_power": True,
           "reconciliation": {"book": {"achieved_gross_usd": 140_970.0, "achieved_leverage": 1.5974},
                              "flow": {"fill_rate_notional": 0.99}}}
    m_real, shortfall, funding = _shortfall_from_result(res, gross_pre=1.1667, applied_design=0.857, leverage=2.0)
    equity = 140_970.0 / 1.5974
    assert m_real == pytest.approx(140_970.0 / (equity * 2.0 * 1.1667), rel=1e-4)
    assert shortfall == pytest.approx(m_real / 0.857, rel=1e-3) and funding is True
    assert _shortfall_from_result({"target_gross_usd": 1.0, "reconciliation": {"book": {}}}, 1.0, 1.0, 2.0) == (None, None, False)
