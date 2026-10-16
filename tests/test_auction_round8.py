"""Opening-auction execution — review round 7 regressions (2026-10-16).

Covers: the resting-order sweep aborting instead of sizing on top of live
orders (still-open limits, unsettled in-flight market orders, a failed
settle listing retried), terminal-state cancel counts, the aborted
rebalance never stamped as done and surfaced in /ready, the gate pass's
last_rebalance=None surviving the not-due merged save, the early-close
afternoon in _after_open_et, the window check before the venue check and
the dead 'submitting' record kept, the persistent order-id set stopping
re-adoption after a drained finalize, date-checked recovered anchors, the
idempotent merge (repeat merges, an order adopted after the completion),
the precise missed-stage marker, the boot listener filtered to its own
jobs, and lost-stage detection after 09:50.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATA_DIR", os.environ.get("DATA_DIR") or tempfile.mkdtemp(prefix="combo_test_r8_"))

import pandas as pd
import pytest

from core.config import ModelConfig
import core.orders as orders_mod
import core.runner as runner_mod
from core.run_report import RunReport
from core.state import load_state, save_state

NY = ZoneInfo("America/New_York")
logger = logging.getLogger("test_auction_r8")


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


def _no_sleep(monkeypatch):
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)


def _fake_clock(monkeypatch, module, h, m, month=10, day=28):
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, month, day, h, m, tzinfo=NY)
            return base.astimezone(tz) if tz else base.astimezone(timezone.utc).replace(tzinfo=None)
    monkeypatch.setattr(module, "datetime", FakeDT)


# ───────────────────────── resting-order sweep ───────────────────────────

def _sweep_broker(listings, final_states=None, calls=None, delete_raises=False):
    calls = calls if calls is not None else []
    final_states = final_states or {}

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        calls.append((method, endpoint))
        if method == "GET" and endpoint.startswith("v2/orders?status=open"):
            nxt = listings.pop(0) if listings else []
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        if method == "GET" and endpoint.startswith("v2/orders/"):
            return final_states.get(endpoint.rsplit("/", 1)[-1], {})
        if method == "DELETE":
            if delete_raises:
                raise RuntimeError("500 server error")
            return {}
        if method == "POST":
            return {"id": f"o{len(calls)}", "status": "accepted"}
        return {}
    return fake_request, calls


def _wire_rebalance(monkeypatch):
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {"portfolio_value": "100000",
                                                                        "cash": "100000", "buying_power": "200000"})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {})
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, *a, **kw: {"id": oid, "status": "filled",
                                                                                "filled_qty": "1", "filled_avg_price": "100"})
    _no_sleep(monkeypatch)


def test_sweep_aborts_the_rebalance_when_a_resting_order_will_not_cancel(tmp_path, monkeypatch):
    rest = {"id": "rest1", "symbol": "AAA", "time_in_force": "day", "type": "limit"}

    class Stuck(list):          # the order never leaves the open listing
        def pop(self, i=-1):
            return [dict(rest)]
    fake, calls = _sweep_broker(Stuck([None]), delete_raises=True)
    monkeypatch.setattr(orders_mod, "alpaca_request", fake)
    _wire_rebalance(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(orders_mod.time, "monotonic", lambda: clock.__setitem__(0, clock[0] + 5.0) or clock[0])
    report = RunReport()
    rb = orders_mod.rebalance_portfolio(["AAA"], [("AAA", 1.0)], make_mc(tmp_path, exec_style="market"),
                                        StubJournal(), logger, report, target_weights={"AAA": 0.05})
    assert rb["aborted_open_orders"] is True and rb["open_orders_cancelled"]["n_still_open"] == 1
    assert not [c for c in calls if c[0] == "POST"]                      # never sized on top of the live order


def test_sweep_waits_for_inflight_market_orders_and_aborts_if_unsettled(tmp_path, monkeypatch):
    mkt = {"id": "scan1", "symbol": "CCC", "time_in_force": "day", "type": "market"}
    # Settles on the second re-listing -> the rebalance proceeds; the market order is NEVER cancelled.
    fake, calls = _sweep_broker([[mkt], [mkt], []])
    monkeypatch.setattr(orders_mod, "alpaca_request", fake)
    _wire_rebalance(monkeypatch)
    rb = orders_mod.rebalance_portfolio(["AAA"], [("AAA", 1.0)], make_mc(tmp_path, exec_style="market"),
                                        StubJournal(), logger, RunReport(), target_weights={"AAA": 0.05})
    assert not rb.get("aborted_open_orders") and ("DELETE", "v2/orders/scan1") not in calls
    assert rb["open_orders_cancelled"]["n_inflight_market"] == 1
    # Never settles inside the wait -> abort.
    clock = [0.0]
    monkeypatch.setattr(orders_mod.time, "monotonic", lambda: clock.__setitem__(0, clock[0] + 5.0) or clock[0])
    fake2, calls2 = _sweep_broker([[mkt]] * 50)
    monkeypatch.setattr(orders_mod, "alpaca_request", fake2)
    rb2 = orders_mod.rebalance_portfolio(["AAA"], [("AAA", 1.0)], make_mc(tmp_path, exec_style="market"),
                                         StubJournal(), logger, RunReport(), target_weights={"AAA": 0.05})
    assert rb2["aborted_open_orders"] is True and rb2["open_orders_cancelled"]["n_inflight_unsettled"] == 1


def test_sweep_settle_listing_failure_is_retried_and_counts_terminal_states(tmp_path, monkeypatch):
    rest1 = {"id": "r1", "symbol": "AAA", "time_in_force": "day", "type": "limit"}
    rest2 = {"id": "r2", "symbol": "BBB", "time_in_force": "day", "type": "limit"}
    fake, calls = _sweep_broker([[rest1, rest2], TimeoutError("read timed out"), []],
                                final_states={"r1": {"status": "canceled"}, "r2": {"status": "filled"}})
    monkeypatch.setattr(orders_mod, "alpaca_request", fake)
    _no_sleep(monkeypatch)
    out = orders_mod.cancel_open_orders(make_mc(tmp_path), logger)
    assert out["n_still_open"] == 0 and out["settle_unverified"] is False
    assert out["n_cancel_requested"] == 2 and out["n_cancelled"] == 1 and out["n_filled_in_race"] == 1


def test_sweep_listing_failure_proceeds_like_before(tmp_path, monkeypatch):
    fake, calls = _sweep_broker([RuntimeError("x"), RuntimeError("x"), RuntimeError("x")])
    monkeypatch.setattr(orders_mod, "alpaca_request", fake)
    _no_sleep(monkeypatch)
    out = orders_mod.cancel_open_orders(make_mc(tmp_path), logger)
    assert out["listing_failed"] is True and out["n_still_open"] == 0
    assert sum(1 for c in calls if c[1].startswith("v2/orders?status=open")) == 3


# ───────────────────────── runner ────────────────────────────────────────

class FakeModel:
    def __init__(self, weights):
        self._w = weights
        self.last_diagnostics = {"exposure_multiplier": 1.0}

    def compute_weights(self, stock_data, macro_data):
        return dict(self._w)


def _wire_runner(monkeypatch, weights, rebalance_result=None, calls=None, combo_config=None):
    calls = calls if calls is not None else []
    monkeypatch.setattr(runner_mod, "load_model_bundle", lambda p: {
        "model": FakeModel(weights), "feature_cols": [], "strategy_type": "direct_weights",
        "horizon": 21, "task": "w", "saved_at": "x", "combo_config": combo_config})
    monkeypatch.setattr(runner_mod, "MIN_STOCKS_REQUIRED", 1)
    monkeypatch.setattr(runner_mod, "MIN_MACRO_FEATURES", 0)
    monkeypatch.setattr(runner_mod, "is_market_open", lambda now=None: True)
    monkeypatch.setattr(runner_mod, "fetch_inactive_assets", lambda syms, mc, lg, top_n=20: set())
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: True)
    monkeypatch.setattr("core.risk.refresh_daily_book_anchor_after_rebalance", lambda *a, **kw: None)
    monkeypatch.setattr("core.protocol.bind_forward_test_start", lambda *a, **kw: False)
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: None)

    def fake_rebalance(target_symbols, rankings, mc, journal, lg, report, **kw):
        calls.append({"target_symbols": list(target_symbols), **kw})
        return dict(rebalance_result or {"dry_run": False})
    monkeypatch.setattr(runner_mod, "rebalance_portfolio", fake_rebalance)
    return calls


def _panel(dates, close=50.0):
    idx = pd.to_datetime(dates)
    return {"AAA": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx)}


def test_aborted_rebalance_is_not_stamped_and_is_flagged(tmp_path, monkeypatch):
    mc = make_mc(tmp_path, exec_style="marketable_limit", name="combo_v2")
    _wire_runner(monkeypatch, {"AAA": 1.0}, rebalance_result={"dry_run": False, "aborted_open_orders": True})
    save_state({"last_rebalance": None, "history": []}, mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(_today_ny())]), pd.DataFrame(),
                                macro_data={}, stage="main")
    st = load_state(mc)
    assert st.get("last_rebalance") is None and st["rebalance_aborted_open_orders"] == _today_ny()
    # A later successful rebalance clears the flag.
    _wire_runner(monkeypatch, {"AAA": 1.0}, rebalance_result={"dry_run": False})
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(_today_ny())]), pd.DataFrame(),
                                macro_data={}, stage="main")
    st = load_state(mc)
    assert st.get("last_rebalance") and "rebalance_aborted_open_orders" not in st


def test_gate_pass_reentry_request_survives_the_not_due_save(tmp_path, monkeypatch):
    mc = make_mc(tmp_path, exec_style="marketable_limit", name="combo_v2")
    _wire_runner(monkeypatch, {"AAA": 1.0})
    recent = (datetime.now() - timedelta(days=1)).isoformat()
    save_state({"last_rebalance": recent, "history": []}, mc)

    def fake_gate_pass(mc_, *a, **kw):
        # What core.gate_update does when the book is gated ~flat.
        d = load_state(mc_)
        d["last_rebalance"] = None
        save_state(d, mc_)
        return {"action": "requested_reentry_rebalance"}
    monkeypatch.setattr("core.gate_update.maybe_daily_gate_update", fake_gate_pass)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(_today_ny())]), pd.DataFrame(),
                                macro_data={}, stage="main")
    assert load_state(mc).get("last_rebalance") is None


def test_after_open_on_an_early_close_afternoon(monkeypatch):
    for (h, m), expect in (((14, 0), True), ((12, 59), True), ((9, 30), False), ((8, 0), False)):
        _fake_clock(monkeypatch, runner_mod, h, m, month=11, day=27)
        assert runner_mod._after_open_et() is expect, (h, m)
    _fake_clock(monkeypatch, runner_mod, 14, 0, month=11, day=27)
    st = {"auction_stage": {"date": "2026-11-27", "pending": [{"order_id": "o1"}]}}
    assert runner_mod._auction_pending_live(st, "2026-11-27") is False


def test_stage_checks_the_window_before_the_venue_and_keeps_the_dead_record(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev = runner_mod._previous_session_iso(today)
    calls = _wire_runner(monkeypatch, {"AAA": 1.0})
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: False)
    venue_calls = []
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders",
                        lambda *a, **kw: venue_calls.append(1) or [])
    dead = {"date": today, "outcome": "submitting", "pending": [], "n_submitted": 0, "run_id": None,
            "plan": {"target_symbols": ["AAA"], "rankings": [["AAA", 1.0]], "target_weights": {"AAA": 1.0}}}
    save_state({"last_rebalance": None, "history": [], "auction_stage": dict(dead)}, mc)
    runner_mod.run_single_model(mc, _panel([prev]), pd.DataFrame(), macro_data={}, stage="auction")
    st = load_state(mc)["auction_stage"]
    assert venue_calls == [] and calls == []                             # no venue GET outside the window
    assert st["outcome"] == "submitting" and st["plan"]["target_weights"] == {"AAA": 1.0}


def test_venue_reconciliation_never_readopts_finalized_orders(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    venue = [{"order_id": "o1", "symbol": "AAA", "side": "buy", "qty": 10.0, "prev_close": 50.0,
              "notional_est": 500.0, "submitted_at": f"{today}T13:06:00+00:00", "recovered": True}]
    replies = [None, list(venue)]           # pass 1: listing fails; pass 2: the venue returns o1 again

    def fake_recover(mc_, s, known, lg):
        r = replies.pop(0)
        return None if r is None else [e for e in r if e["order_id"] not in known]
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", fake_recover)
    fin = []

    def fake_finalize(mc_, pending, journal, lg, **kw):
        fin.append([p["order_id"] for p in pending])
        return {"n": len(pending), "n_filled": len(pending), "n_unfilled": 0, "n_cancelled_open": 0,
                "submitted_notional_usd": 500.0 * len(pending), "filled_notional_anchor_usd": 500.0 * len(pending),
                "filled_notional_usd": 500.0 * len(pending), "fill_rate_notional": 1.0,
                "vw_cost_vs_open_bps": 1.0, "vw_cost_weight_usd": 500.0, "n_remaining_unread": 0, "remaining": []}
    monkeypatch.setattr("core.orders.finalize_auction_orders", fake_finalize)
    st = {"auction_stage": {"date": today, "outcome": "submitted", "n_submitted": 1, "run_id": "RID",
                            "submitted_notional_usd": 500.0, "order_ids": ["o1"],
                            "pending": [{"order_id": "o1", "symbol": "AAA", "notional_est": 500.0,
                                         "submitted_at": f"{today}T13:06:00+00:00", "run_id": "RID"}]}}
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    a = st["auction_stage"]
    assert fin == [["o1"]] and a["summary"]["n"] == 1 and a["n_recovered"] == 0
    assert a["n_submitted"] == 1 and a["submitted_notional_usd"] == 500.0 and a["recovery_checked"]


def test_journal_order_ids_back_up_the_persistent_set(tmp_path):
    jl = tmp_path / "trades.jsonl"
    jl.write_text('{"trade_id": "combo_v2_process_auction_20261028_buy_AAA", "order_id": "o1"}\n'
                  '{"trade_id": "combo_v2_process_auction_20261027_buy_BBB", "order_id": "old"}\n'
                  '{"trade_id": "combo_v2_process_20261028_x", "order_id": "ml1"}\n')
    ids = runner_mod._journal_auction_order_ids(StubJournal(path=str(jl)), "combo_v2_process", "2026-10-28")
    assert ids == {"o1"}


def test_recovered_anchor_snapshot_only_for_its_own_session(tmp_path, monkeypatch):
    def fake_snaps(syms, mc_, lg):
        return {s: {"prev_close": 33.0, "day_bar_date": "2026-10-15"} for s in syms}
    monkeypatch.setattr("core.orders.fetch_snapshots", fake_snaps)
    early = [{"symbol": "AAA", "qty": 2.0, "prev_close": None, "notional_est": 0.0}]
    runner_mod._fill_recovered_anchors(early, {}, "2026-10-14", make_mc(tmp_path), logger)
    assert early[0]["prev_close"] is None and early[0]["notional_est"] == 0.0
    same = [{"symbol": "AAA", "qty": 2.0, "prev_close": None, "notional_est": 0.0}]
    runner_mod._fill_recovered_anchors(same, {}, "2026-10-15", make_mc(tmp_path), logger)
    assert same[0]["prev_close"] == 33.0 and same[0]["notional_est"] == 66.0


def test_merge_is_idempotent_and_absorbs_a_later_adopted_order():
    flow = {"n_orders": 6, "n_filled": 6, "fill_rate_notional": 0.98, "orders_notional_usd": 600.0,
            "filled_notional_usd": 588.0, "n_rejected": 0}
    result = {"reconciliation": {"flow": dict(flow), "book": None}}
    st = {"date": "D", "n_submitted": 2, "summary": {"n": 2, "n_filled": 2, "submitted_notional_usd": 200.0,
                                                     "filled_notional_anchor_usd": 200.0, "filled_notional_usd": 200.0}}
    runner_mod._merge_auction_into_result(result, st, "D")
    first = dict(result["reconciliation"]["flow"])
    runner_mod._merge_auction_into_result(result, st, "D")                  # repeat: no double count
    assert result["reconciliation"]["flow"] == first and first["n_orders"] == 8
    # A venue-adopted order is accumulated into the summary later: re-merge reflects it exactly once.
    st["summary"] = {"n": 3, "n_filled": 3, "submitted_notional_usd": 300.0,
                     "filled_notional_anchor_usd": 300.0, "filled_notional_usd": 300.0}
    runner_mod._merge_auction_into_result(result, st, "D")
    f = result["reconciliation"]["flow"]
    assert f["n_orders"] == 9 and f["fill_rate_notional"] == pytest.approx(888.0 / 900.0, abs=1e-3)
    assert f["fill_rate_notional_completion"] == 0.98 and f["n_rejected"] == 0
    assert result["reconciliation"]["flow_completion"]["n_orders"] == 6


def test_repair_runs_on_every_measuring_pass_after_completion(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: [])
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda mc_, pending, journal, lg, **kw: {"n": 1, "n_filled": 1, "n_unfilled": 0,
                                                                 "n_cancelled_open": 0, "submitted_notional_usd": 100.0,
                                                                 "filled_notional_anchor_usd": 100.0,
                                                                 "filled_notional_usd": 100.0, "fill_rate_notional": 1.0,
                                                                 "vw_cost_vs_open_bps": 0.0, "vw_cost_weight_usd": 100.0,
                                                                 "n_remaining_unread": 0, "remaining": []})
    result = {"reconciliation": {"flow": {"n_orders": 6, "n_filled": 6, "fill_rate_notional": 1.0,
                                          "orders_notional_usd": 600.0, "filled_notional_usd": 600.0}, "book": None}}
    rec = {"date": today, "outcome": "submitted", "n_submitted": 3, "run_id": "RID", "completed_at": "x",
           "recovery_checked": "x",
           "summary": {"n": 2, "n_filled": 2, "submitted_notional_usd": 200.0,
                       "filled_notional_anchor_usd": 200.0, "filled_notional_usd": 200.0,
                       "vw_cost_vs_open_bps": 0.0, "vw_cost_weight_usd": 200.0},
           "pending": [{"order_id": "late", "symbol": "CCC", "submitted_at": f"{today}T13:06:00+00:00"}]}
    runner_mod._merge_auction_into_result(result, rec, today)                # what the 09:35 completion merged
    st = {"history": [{"auction_run_id": "RID", "result": result}], "auction_stage": rec}
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    f = st["history"][0]["result"]["reconciliation"]["flow"]
    assert st["auction_stage"]["summary"]["n"] == 3 and f["n_orders"] == 9   # 6 completion + 3 auction, once


def _plan_state(today, **extra):
    st = {"last_rebalance": None, "history": [],
          "auction_stage": {"date": today, "outcome": "submitted", "n_submitted": 2, "run_id": "RID",
                            "plan": {"target_symbols": ["AAA"], "rankings": [["AAA", 1.0]],
                                     "target_weights": {"AAA": 1.0}, "diagnostics": {}},
                            "pending": []}}
    st["auction_stage"].update(extra)
    return st


def test_rerun_after_a_completed_plan_is_not_a_missed_stage(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    _wire_runner(monkeypatch, {"AAA": 1.0}, rebalance_result={"dry_run": False})
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    save_state(_plan_state(today, completed_at="x", recovery_checked="x"), mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(today)]), pd.DataFrame(),
                                macro_data={}, force=True, stage="main")
    res = load_state(mc)["history"][-1]["result"]
    assert res.get("auction_rerun_after_completion") is True and not res.get("auction_stage_missed")


def test_stage_outcome_rides_along_with_a_missed_stage(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    _wire_runner(monkeypatch, {"AAA": 1.0}, rebalance_result={"dry_run": False})
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    save_state({"last_rebalance": None, "history": [],
                "auction_stage": {"date": today, "outcome": "skipped:stale_panel", "n_submitted": 0,
                                  "pending": [], "plan": None}}, mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(today)]), pd.DataFrame(),
                                macro_data={}, stage="main")
    res = load_state(mc)["history"][-1]["result"]
    assert res["auction_stage_missed"] is True and res["auction_stage_outcome"] == "skipped:stale_panel"


# ───────────────────────── dashboard ─────────────────────────────────────

def test_boot_listener_ignores_other_jobs_and_is_not_registered_when_idle(tmp_path, monkeypatch):
    import dashboard
    from apscheduler.events import (EVENT_JOB_EXECUTED, EVENT_JOB_MISSED, JobExecutionEvent)
    from apscheduler.schedulers.background import BackgroundScheduler
    mc = make_mc(tmp_path, exec_style="auction")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {"last_run": "2026-10-27T13:35:00",
                                                                      "last_rebalance": None})
    sched = BackgroundScheduler(timezone="America/New_York")
    added = []
    monkeypatch.setattr(sched, "add_listener", lambda cb, mask=None: added.append(cb))
    # 05:00 ET: nothing to replay -> no listener at all.
    dashboard._schedule_boot_catchup(sched, datetime(2026, 10, 28, 5, 0, tzinfo=NY), logging.getLogger("t"))
    assert added == []
    # 10:00 ET: a main replay -> listener registered; scanner events are ignored, the replay is recorded once.
    sched2 = BackgroundScheduler(timezone="America/New_York")
    monkeypatch.setattr(sched2, "add_listener", lambda cb, mask=None: added.append(cb))
    dashboard._schedule_boot_catchup(sched2, datetime(2026, 10, 28, 10, 0, tzinfo=NY), logging.getLogger("t"))
    assert len(added) == 1
    cb = added[0]
    now = datetime(2026, 10, 28, 10, 1, tzinfo=NY)
    cb(JobExecutionEvent(EVENT_JOB_EXECUTED, "cutloss_scanner", "default", now))
    cb(JobExecutionEvent(EVENT_JOB_MISSED, "cutloss_scanner", "default", now))
    cb(JobExecutionEvent(EVENT_JOB_EXECUTED, "catchup_main", "default", now))
    cb(JobExecutionEvent(EVENT_JOB_EXECUTED, "catchup_main", "default", now))
    with dashboard.status_lock:
        bc = dashboard.bot_status["boot_catchup"]
        assert bc["fired"] == ["catchup_main"] and bc["missed"] == []
        for k in ("auction_last_run_status", "auction_last_run_at", "auction_skipped_due", "boot_catchup"):
            dashboard.bot_status.pop(k, None)


def test_lost_stage_detection_after_nine_fifty():
    import dashboard
    at = lambda h, m: datetime(2026, 10, 28, h, m, tzinfo=NY)   # noqa: E731
    # 10:05: no live run today (replay will rebalance the due slot without a plan) -> lost.
    p = dashboard._boot_catchup_plan(at(10, 5), {"a": {"last_run": "2026-10-27T13:35:00", "last_rebalance": None}}, {"a"})
    assert p["main"] is True and p["lost_auction"] == ["a"]
    # 10:05: the main run already ran today (no replay) and the slot is still due -> not lost (stage tomorrow).
    p = dashboard._boot_catchup_plan(at(10, 5), {"a": {"last_run": "2026-10-28T13:36:00", "last_rebalance": None}}, {"a"})
    assert p["main"] is False and p["lost_auction"] == []
    # 09:25: the 09:35 cron will rebalance it -> lost.
    p = dashboard._boot_catchup_plan(at(9, 25), {"a": {"last_rebalance": None}}, {"a"})
    assert p["lost_auction"] == ["a"]


def test_ready_flags_an_aborted_rebalance(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path, exec_style="marketable_limit", name="combo_v2")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr("core.market.is_market_open", lambda now=None: False)
    monkeypatch.setattr(dashboard, "STATE_DIR", tmp_path)
    (tmp_path / "x.json").write_text("{}")
    monkeypatch.setattr(dashboard, "_load_model_state",
                        lambda name: {"rebalance_aborted_open_orders": _today_ny()})
    monkeypatch.setattr("core.invariants.load_last_status", lambda: None)
    monkeypatch.setitem(dashboard.bot_status, "last_run_status", "success")
    monkeypatch.setitem(dashboard.bot_status, "last_run_at", datetime.now(timezone.utc).isoformat())
    for k in ("skipped_main_run_at", "auction_skipped_due", "last_full_run_at", "boot_catchup"):
        monkeypatch.delitem(dashboard.bot_status, k, raising=False)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", None)
    r = dashboard.app.test_client().get("/ready")
    assert r.status_code == 503 and any("rebalance aborted" in p for p in r.get_json()["problems"])
