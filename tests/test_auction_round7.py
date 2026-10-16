"""Opening-auction execution — review round 6 regressions (2026-10-16).

Covers: the resting-order sweep before a rebalance (catch-up / manual
re-run double-submit), dry runs never stamping last_run, the boot
catch-up plan (dry stamps, lost stage, half-day window, dead provisional
record, edges), the scheduling helper (grace, boundary, lost-stage trace),
provisional record preservation + venue-clear resubmission, adoption with a
synthesized run_id and panel anchors, recovery failure keeping the record
open, venue reconciliation for 'submitted' records, merge/repair guards,
yfinance single-ticker shape, cancel accounting, early closes and the
wait cutoff, auction-aware full-run marker, readiness traces, the gate
reconciliation retry and same-time equity, the on_submitted seam, and
the 'submitting' plan acceptance.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATA_DIR", os.environ.get("DATA_DIR") or tempfile.mkdtemp(prefix="combo_test_r7_"))

import pandas as pd
import pytest

from core.config import ModelConfig
import core.market as market_mod
import core.orders as orders_mod
import core.runner as runner_mod
from core.run_report import RunReport
from core.state import load_state, save_state

NY = ZoneInfo("America/New_York")
logger = logging.getLogger("test_auction_r7")


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


# ───────────────────────── orders layer ──────────────────────────────────

def test_rebalance_sweeps_resting_non_opg_orders_before_sizing(tmp_path, monkeypatch):
    calls = []
    open_orders = [{"id": "rest1", "symbol": "AAA", "time_in_force": "day", "type": "limit"},
                   {"id": "opg1", "symbol": "BBB", "time_in_force": "opg", "type": "limit"},
                   {"id": "scan1", "symbol": "CCC", "time_in_force": "day", "type": "market"}]   # scanner in flight
    listings = [list(open_orders), []]

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        calls.append((method, endpoint))
        if method == "GET" and endpoint.startswith("v2/orders?status=open"):
            return listings.pop(0) if listings else []
        if method == "GET" and endpoint == "v2/orders/rest1":
            return {"id": "rest1", "status": "canceled"}
        if method == "DELETE":
            return {}
        if method == "POST" and endpoint == "v2/orders":
            return {"id": f"o{len(calls)}", "status": "accepted"}
        return {}
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {"portfolio_value": "100000", "cash": "100000",
                                                                        "buying_power": "200000"})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {})
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, *a, **kw: {"id": oid, "status": "filled",
                                                                                "filled_qty": "1", "filled_avg_price": "100"})
    _no_sleep(monkeypatch)
    mc = make_mc(tmp_path, exec_style="market")
    rb = orders_mod.rebalance_portfolio(["AAA"], [("AAA", 1.0)], mc, StubJournal(), logger, RunReport(),
                                        target_weights={"AAA": 0.05})
    sw = rb["open_orders_cancelled"]
    assert sw["n_open"] == 1 and sw["n_cancel_requested"] == 1 and sw["n_cancelled"] == 1
    assert sw["n_inflight_market"] == 1 and sw["n_inflight_unsettled"] == 0 and not rb.get("aborted_open_orders")
    assert ("DELETE", "v2/orders/rest1") in calls and ("DELETE", "v2/orders/opg1") not in calls
    assert ("DELETE", "v2/orders/scan1") not in calls                     # market orders are never swept
    # The sweep precedes the sizing reads and the first POST.
    assert calls.index(("DELETE", "v2/orders/rest1")) < min(i for i, c in enumerate(calls) if c[0] == "POST")


def test_rebalance_result_carries_same_time_equity(tmp_path, monkeypatch):
    from core.gate_update import _shortfall_from_result
    reads = []

    def fake_account(mc_, lg, rp):
        reads.append(1)
        return {"portfolio_value": "100000" if len(reads) == 1 else "98000", "cash": "100000",
                "buying_power": "200000"}
    def fake_request(m, e, mc, data=None, logger=None, **kw):
        if m == "POST":
            return {"id": "o1", "status": "accepted"}
        if m == "GET" and e == "v2/account":
            return {"portfolio_value": "98000", "cash": "1", "buying_power": "1"}
        return []
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(orders_mod, "get_account", fake_account)
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {})
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, *a, **kw: {"id": oid, "status": "filled",
                                                                                "filled_qty": "1", "filled_avg_price": "100"})
    _no_sleep(monkeypatch)
    rb = orders_mod.rebalance_portfolio(["AAA"], [("AAA", 1.0)], make_mc(tmp_path, exec_style="market"),
                                        StubJournal(), logger, RunReport(), target_weights={"AAA": 0.05})
    # The post-trade read never overwrites the PRE-trade account fields (round 7).
    assert rb["portfolio_value"] == 100_000.0 and rb["portfolio_value_after"] == 98_000.0
    assert len(reads) == 1
    res = {"target_gross_usd": 50_000.0, "portfolio_value": 100_000.0, "portfolio_value_after": 98_000.0}
    m_real, _, _ = _shortfall_from_result(res, gross_pre=1.0, applied_design=1.0, leverage=2.0)
    assert m_real == pytest.approx(50_000.0 / (98_000.0 * 2.0))          # same-time equity preferred


def test_recovery_returns_none_when_the_venue_cannot_be_listed_and_uses_the_slot_buffer(tmp_path, monkeypatch):
    def raising(*a, **kw):
        raise RuntimeError("timeout")
    monkeypatch.setattr(orders_mod, "alpaca_request", raising)
    assert orders_mod.recover_unpersisted_auction_orders(make_mc(tmp_path), _today_ny(), set(), logger) is None
    seen = {}

    def listing(m, endpoint, mc, data=None, logger=None, **kw):
        seen["endpoint"] = endpoint
        day = "20260115"
        return [{"id": "v1", "client_order_id": f"a{day}_AAA_x", "time_in_force": "opg", "side": "buy",
                 "type": "limit", "limit_price": "108.0", "qty": "10", "status": "filled", "symbol": "AAA"}]
    monkeypatch.setattr(orders_mod, "alpaca_request", listing)
    found = orders_mod.recover_unpersisted_auction_orders(make_mc(tmp_path, exec_auction_buffer_bps=80.0),
                                                          "2026-01-15", set(), logger)
    assert "after=2026-01-15T08:00:00-05:00" in seen["endpoint"]           # EST offset in January
    assert found[0]["prev_close"] == pytest.approx(108.0 / 1.008, abs=1e-3)


def test_cancel_counts_only_when_it_took_effect(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    reads = []

    def fake_poll(oid, mc, lg, max_wait=8.0, **kw):
        reads.append(oid)
        if oid == "race":          # open at first read, FILLED in the cancel race
            return ({"id": oid, "status": "accepted", "filled_qty": "0"} if reads.count(oid) == 1
                    else {"id": oid, "status": "filled", "filled_qty": "10", "filled_avg_price": "50.0"})
        return {"id": oid, "status": "filled", "filled_qty": "10", "filled_avg_price": "50.0"}
    monkeypatch.setattr(orders_mod, "poll_order_status", fake_poll)
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {})
    base = {"submitted_at": f"{_today_ny()}T13:06:00+00:00", "run_id": "RID", "order_status": "accepted",
            "side": "buy", "order_type": "limit", "qty": 10, "limit_price": 50.5, "prev_close": 50.0,
            "notional_est": 500.0}
    pending = [{**base, "symbol": "AAA", "order_id": "race"},
               {**base, "symbol": "BBB", "order_id": "prior", "cancel_sent": True}]   # filled despite an earlier cancel
    stats = orders_mod.finalize_auction_orders(make_mc(tmp_path), pending, StubJournal(), logger)
    assert stats["n_filled"] == 2 and stats["n_cancelled_open"] == 0


def test_on_submitted_seam_through_the_real_rebalance(tmp_path, monkeypatch):
    posted = []

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        if method == "POST" and endpoint == "v2/orders":
            posted.append(dict(data))
            return {"id": f"o{len(posted)}", "status": "accepted"}
        return {}
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {"portfolio_value": "100000", "cash": "100000",
                                                                        "buying_power": "200000"})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {})
    spy = []
    rb = orders_mod.rebalance_portfolio(["AAA", "BBB"], [("AAA", 1.0), ("BBB", 1.0)], make_mc(tmp_path),
                                        StubJournal(), logger, RunReport(), target_weights={"AAA": 0.05, "BBB": 0.05},
                                        auction=True, last_close={"AAA": 100.0, "BBB": 50.0},
                                        on_submitted=lambda e, rid: spy.append((e["order_id"], rid)))
    assert [o for o, _ in spy] == ["o1", "o2"] and all(rid == rb["run_id"] for _, rid in spy)


# ───────────────────────── market calendar ───────────────────────────────

def test_early_close_sessions():
    half = datetime(2026, 11, 27, 12, 59, tzinfo=NY)
    assert market_mod.is_market_open(half) and not market_mod.is_market_open(half.replace(hour=13, minute=1))
    assert market_mod.session_close(half).hour == 13
    full = datetime(2026, 10, 28, 15, 59, tzinfo=NY)
    assert market_mod.is_market_open(full) and market_mod.session_close(full).hour == 16


# ───────────────────────── runner ────────────────────────────────────────

class FakeModel:
    def __init__(self, weights):
        self._w = weights
        self.last_diagnostics = {"exposure_multiplier": 1.0}

    def compute_weights(self, stock_data, macro_data):
        return dict(self._w)


def _wire_runner(monkeypatch, weights, rebalance_result=None, calls=None):
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
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: None)

    def fake_rebalance(target_symbols, rankings, mc, journal, lg, report, **kw):
        calls.append({"target_symbols": list(target_symbols), **kw})
        if rebalance_result is not None:
            return dict(rebalance_result)
        pend = [{"order_id": "o1", "symbol": "AAA", "notional_est": 500.0}]
        cb = kw.get("on_submitted")
        if cb is not None:
            for e in pend:
                cb(e, "RID")
        return {"dry_run": False, "auction": {"n_submitted": 1, "n_skipped": 0, "skipped": [], "n_stale_close": 0,
                                               "submitted_notional_usd": 500.0, "run_id": "RID", "pending": pend},
                "run_id": "RID"}
    monkeypatch.setattr(runner_mod, "rebalance_portfolio", fake_rebalance)
    return calls


def _panel(dates, close=50.0):
    idx = pd.to_datetime(dates)
    return {"AAA": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx),
            "BBB": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx)}


def _plan_state(today, pending=(), outcome="submitted", run_id="RID", **extra):
    st = {"last_rebalance": (datetime.now() - timedelta(days=1)).isoformat(), "history": [],
          "auction_stage": {"date": today, "outcome": outcome, "n_submitted": 2 if outcome == "submitted" else 0,
                            "run_id": run_id,
                            "plan": {"target_symbols": ["AAA", "BBB"], "rankings": [["AAA", 0.6], ["BBB", 0.4]],
                                     "target_weights": {"AAA": 0.6, "BBB": 0.4},
                                     "diagnostics": {"exposure_multiplier": 0.7}},
                            "pending": list(pending)}}
    st.update(extra)
    return st


_COMPLETION = {"dry_run": False, "reconciliation": {"flow": {"n_orders": 1, "n_filled": 1, "fill_rate_notional": 1.0,
                                                             "orders_notional_usd": 100.0, "filled_notional_usd": 100.0},
                                                    "book": None}}


def test_dry_run_stamps_last_dry_run_not_last_run(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    _wire_runner(monkeypatch, {"AAA": 1.0})
    save_state({"last_rebalance": None, "history": []}, mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(_today_ny())]), pd.DataFrame(),
                                macro_data={}, dry_run=True, force=True, stage="main")
    st = load_state(mc)
    assert st.get("last_run") is None and st.get("last_dry_run")


def test_adoption_synthesizes_run_id_fills_anchors_and_closes_the_plan(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev = runner_mod._previous_session_iso(today)
    calls = _wire_runner(monkeypatch, {"ZZZ": 1.0}, rebalance_result=_COMPLETION)
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    # The dead run persisted nothing: run_id None, pending []. The venue holds one filled SELL (no prev_close).
    recovered = [{"order_id": "v9", "symbol": "BBB", "side": "sell", "qty": 2.0, "prev_close": None,
                  "notional_est": 0.0, "submitted_at": f"{today}T13:07:00+00:00", "recovered": True, "run_id": None}]
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda mc_, s, known, lg: list(recovered))
    fin = []

    def fake_finalize(mc_, pending, journal, lg, **kw):
        fin.append([(p["order_id"], p.get("prev_close"), p.get("notional_est"), p.get("run_id")) for p in pending])
        return {"n": len(pending), "n_filled": len(pending), "n_unfilled": 0, "n_cancelled_open": 0,
                "submitted_notional_usd": 100.0, "filled_notional_anchor_usd": 100.0, "filled_notional_usd": 100.0,
                "fill_rate_notional": 1.0, "vw_cost_vs_open_bps": 0.0, "vw_cost_weight_usd": 100.0,
                "n_remaining_unread": 0, "remaining": []}
    monkeypatch.setattr("core.orders.finalize_auction_orders", fake_finalize)
    save_state(_plan_state(today, outcome="submitting", run_id=None), mc)
    runner_mod.run_single_model(mc, _panel([prev], close=50.0), pd.DataFrame(), macro_data={}, stage="main")
    st = load_state(mc)
    rid = f"combo_v2_process_auction_{today.replace('-', '')}_recovered"
    assert fin == [[("v9", 50.0, 100.0, rid)]]                       # anchor from the panel's prior-session close
    assert st["auction_stage"]["run_id"] == rid and st["auction_stage"]["completed_at"]
    assert calls[0]["run_id"] == rid
    assert st["history"][-1]["auction_run_id"] == rid
    assert st["history"][-1]["result"]["reconciliation"]["auction"]["n"] == 1
    assert runner_mod._todays_auction_plan(st, today) is None            # spent: no re-completion all day


def test_recovery_failure_keeps_the_record_open_then_adopts(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: None)
    st = _plan_state(today, outcome="submitting", run_id=None)
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    a = st["auction_stage"]
    assert a["outcome"] == "submitting" and a["recovery_error"] and "recovery_checked" not in a
    # Next run: the venue answers (empty) -> the record closes as nothing submitted.
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: [])
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    a = st["auction_stage"]
    assert a["outcome"] == "skipped:nothing_submitted" and a["recovery_checked"] and "recovery_error" not in a


def test_submitted_record_is_reconciled_with_the_venue_once(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    n_calls = []
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders",
                        lambda mc_, s, known, lg: n_calls.append(sorted(known)) or
                        [{"order_id": "lost", "symbol": "CCC", "side": "buy", "qty": 1.0, "prev_close": 40.0,
                          "notional_est": 40.0, "submitted_at": f"{today}T13:06:30+00:00", "recovered": True}])
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda mc_, pending, journal, lg, **kw: {"n": len(pending), "n_filled": len(pending),
                                                                 "n_unfilled": 0, "n_cancelled_open": 0,
                                                                 "submitted_notional_usd": 540.0,
                                                                 "filled_notional_anchor_usd": 540.0,
                                                                 "filled_notional_usd": 540.0, "fill_rate_notional": 1.0,
                                                                 "vw_cost_vs_open_bps": 0.0, "vw_cost_weight_usd": 540.0,
                                                                 "n_remaining_unread": 0, "remaining": []})
    st = _plan_state(today, pending=[{"order_id": "o1", "symbol": "AAA", "notional_est": 500.0,
                                      "submitted_at": f"{today}T13:06:00+00:00", "run_id": "RID"}])
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    a = st["auction_stage"]
    assert n_calls == [["o1"]] and a["n_recovered"] == 1 and a["summary"]["n"] == 2 and a["recovery_checked"]
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    assert n_calls == [["o1"]]                                              # once per record


def test_merge_guards_unmeasured_without_anchor_and_preserves_completion_rate():
    flow = {"n_orders": 6, "n_filled": 6, "fill_rate_notional": 0.98, "orders_notional_usd": 600.0,
            "filled_notional_usd": 588.0}
    result = {"reconciliation": {"flow": dict(flow), "book": None}}
    # Rows without an anchor (recovered sells only): measured n but zero anchor dollars -> NA.
    st = {"date": "D", "n_submitted": 1, "summary": {"n": 1, "n_filled": 1, "submitted_notional_usd": 0.0,
                                                     "filled_notional_anchor_usd": 0.0, "filled_notional_usd": 50.0}}
    runner_mod._merge_auction_into_result(result, st, "D")
    f = result["reconciliation"]["flow"]
    assert f["fill_rate_notional"] is None and f["auction_unmeasured"] is True and f["fill_rate_notional_completion"] == 0.98
    # Later repair with a real summary keeps the completion rate captured the first time.
    st2 = {"date": "D", "n_submitted": 2, "summary": {"n": 2, "n_filled": 2, "submitted_notional_usd": 200.0,
                                                      "filled_notional_anchor_usd": 200.0, "filled_notional_usd": 200.0}}
    runner_mod._merge_auction_into_result(result, st2, "D")
    assert f["fill_rate_notional_completion"] == 0.98 and "auction_unmeasured" not in f
    assert f["fill_rate_notional"] == pytest.approx(788.0 / 800.0, abs=1e-3)


def test_history_repair_works_the_next_day(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    yday = (datetime.now(NY) - timedelta(days=1)).strftime("%Y-%m-%d")
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: [])
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda mc_, pending, journal, lg, **kw: {"n": 2, "n_filled": 2, "n_unfilled": 0,
                                                                 "n_cancelled_open": 0, "submitted_notional_usd": 200.0,
                                                                 "filled_notional_anchor_usd": 200.0,
                                                                 "filled_notional_usd": 200.0, "fill_rate_notional": 1.0,
                                                                 "vw_cost_vs_open_bps": 3.0, "vw_cost_weight_usd": 200.0,
                                                                 "n_remaining_unread": 0, "remaining": []})
    hist = {"auction_run_id": "RID", "result": {"reconciliation": {"flow": {"fill_rate_notional": None,
                                                                            "auction_unmeasured": True,
                                                                            "fill_rate_notional_completion": 0.97,
                                                                            "orders_notional_usd": 100.0,
                                                                            "filled_notional_usd": 97.0,
                                                                            "n_orders": 1, "n_filled": 1},
                                                                   "book": None}}}
    st = {"history": [hist], "auction_stage": {"date": yday, "outcome": "submitted", "n_submitted": 2, "run_id": "RID",
                                               "completed_at": "x", "summary_error": "api",
                                               "pending": [{"order_id": "o1", "symbol": "AAA",
                                                            "submitted_at": f"{yday}T13:06:00+00:00"}]}}
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    f = st["history"][0]["result"]["reconciliation"]["flow"]
    assert "auction_unmeasured" not in f and f["fill_rate_notional"] == pytest.approx(297.0 / 300.0, abs=1e-3)
    assert f["fill_rate_notional_completion"] == 0.97 and st["auction_stage"]["finalized_late"] is True


def test_fetch_today_opens_handles_single_ticker_multiindex_and_dot_form(monkeypatch):
    import yfinance
    today = _today_ny()

    def fake_download(tickers, **kw):
        idx = pd.to_datetime([today])
        cols = pd.MultiIndex.from_tuples([(t, f) for t in tickers for f in ("Open", "Close")],
                                         names=["Ticker", "Price"])
        data = [[101.0, 102.0] * len(tickers)]
        return pd.DataFrame(data, index=idx, columns=cols)
    monkeypatch.setattr(yfinance, "download", fake_download)
    assert runner_mod._fetch_today_opens(["AAA"], today, logger) == {"AAA": 101.0}
    out = runner_mod._fetch_today_opens(["BRK.B", "AAA"], today, logger)
    assert out == {"BRK.B": 101.0, "AAA": 101.0}                          # dot-form queried as BRK-B, mapped back


def test_todays_auction_plan_accepts_submitting_with_persisted_orders():
    plan = {"target_symbols": ["AAA"]}
    st = {"auction_stage": {"date": "D", "outcome": "submitting", "n_submitted": 0,
                            "pending": [{"order_id": "o1"}], "plan": plan}}
    assert runner_mod._todays_auction_plan(st, "D") is plan
    st["auction_stage"]["pending"] = []
    assert runner_mod._todays_auction_plan(st, "D") is None


def test_auction_slot_without_a_plan_marks_the_missed_stage(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    calls = _wire_runner(monkeypatch, {"AAA": 1.0}, rebalance_result=_COMPLETION)
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    save_state({"last_rebalance": None, "history": []}, mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(_today_ny())]), pd.DataFrame(),
                                macro_data={}, stage="main")
    st = load_state(mc)
    assert calls and st["history"][-1]["result"]["auction_stage_missed"] is True


# ───────────────────────── gate reconciliation ───────────────────────────

def test_gate_reconciliation_retries_instead_of_stamping_none(tmp_path, monkeypatch):
    from test_oct15_fixes import _seed, _wire, _real_positions, _synthetic, make_mc as gate_mc
    from core.combo_strategy import ComboConfig
    import core.gate_update as gu
    mc = gate_mc(tmp_path)
    # Sep 29-style record WITHOUT equity and WITHOUT a usable book: cannot derive.
    _seed(mc, applied_exposure_multiplier=0.8571, tier_floor_multiplier=1.0,
          gate_ref={"gross_pre_gate": 1.1667, "max_gross_exposure": 1.0},
          history=[{"strategy_diagnostics": {"exposure_multiplier": 1.0, "gross_pre_gate": 1.1667, "gross_final": 1.0},
                    "result": {"target_gross_usd": 130_745.93, "auto_scaled_to_buying_power": True,
                               "reconciliation": {"book": None, "flow": {}}}}])
    orders: list = []
    _wire(monkeypatch, orders, _real_positions({"S000": 130_000.0}))
    for attempt in (1, 2):
        gu.maybe_daily_gate_update(mc, {"combo_config": ComboConfig()}, "direct_weights", _synthetic(), {},
                                   StubJournal(), logger, report=None, min_stocks_required=5, dry_run=False)
        ref = load_state(mc)["gate_ref"]
        assert "rebalance_shortfall" not in ref and ref["reconcile_attempts"] == attempt
    gu.maybe_daily_gate_update(mc, {"combo_config": ComboConfig()}, "direct_weights", _synthetic(), {},
                               StubJournal(), logger, report=None, min_stocks_required=5, dry_run=False)
    ref = load_state(mc)["gate_ref"]
    assert ref["reconcile_gave_up"] is True and ref["rebalance_shortfall"] is None
    assert load_state(mc)["applied_exposure_multiplier"] == pytest.approx(0.8571)   # never traded or re-levered


def test_gate_reconciliation_uses_the_books_leverage_when_equity_is_missing(tmp_path, monkeypatch):
    from test_oct15_fixes import _seed, _wire, _real_positions, _synthetic, make_mc as gate_mc
    from core.combo_strategy import ComboConfig
    import core.gate_update as gu
    mc = gate_mc(tmp_path)
    _seed(mc, applied_exposure_multiplier=0.8571, tier_floor_multiplier=1.0,
          gate_ref={"gross_pre_gate": 1.1667, "max_gross_exposure": 1.0},
          history=[{"strategy_diagnostics": {"exposure_multiplier": 1.0, "gross_pre_gate": 1.1667, "gross_final": 1.0},
                    "result": {"target_gross_usd": 176_496.0, "auto_scaled_to_buying_power": True,
                               "reconciliation": {"book": {"achieved_gross_usd": 140_970.0, "achieved_leverage": 1.5974},
                                                  "flow": {"fill_rate_notional": 0.99}}}}])
    orders: list = []
    _wire(monkeypatch, orders, _real_positions({"S000": 140_000.0}))
    info = gu.maybe_daily_gate_update(mc, {"combo_config": ComboConfig()}, "direct_weights", _synthetic(), {},
                                      StubJournal(), logger, report=None, min_stocks_required=5, dry_run=False)
    equity = 140_970.0 / 1.5974
    expect = (140_970.0 / (equity * 2.0 * 1.1667)) / 0.8571
    assert info["reconciled_shortfall"] == pytest.approx(expect, abs=2e-3)
    st = load_state(mc)
    assert st["gate_ref"]["rebalance_shortfall"] == pytest.approx(expect, abs=2e-3)
    assert st["tier_floor_multiplier"] == pytest.approx(expect, abs=2e-3)    # funding evidence -> sticky floor


# ───────────────────────── dashboard ─────────────────────────────────────

def _dash_clock(monkeypatch, dashboard, h, m, month=10, day=28):
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, month, day, h, m, tzinfo=NY)
            return base.astimezone(tz) if tz else base.replace(tzinfo=None)
    monkeypatch.setattr(dashboard, "datetime", FakeDT)


def test_boot_catchup_plan_round6_cases():
    import dashboard
    at = lambda h, m, month=10, day=28: datetime(2026, month, day, h, m, tzinfo=NY)   # noqa: E731
    # A dry run stamps last_dry_run, not last_run: still a missed main run.
    p = dashboard._boot_catchup_plan(at(10, 0), {"a": {"last_dry_run": "2026-10-28T12:40:00"}}, set())
    assert p["main"] is True
    # Boot after the window on a due day: lost, not re-fired.
    p = dashboard._boot_catchup_plan(at(9, 25), {"a": {"last_rebalance": None}}, {"a"})
    assert p["auction"] is False and p["lost_auction"] == ["a"]
    p = dashboard._boot_catchup_plan(at(9, 19), {"a": {"last_rebalance": None}}, {"a"})
    assert p["auction"] is False and p["lost_auction"] == ["a"]
    # A provisional record that never got an order counts as no record -> re-fire inside the window.
    dead = {"last_rebalance": None, "auction_stage": {"date": "2026-10-28", "outcome": "submitting",
                                                      "pending": [], "n_submitted": 0}}
    assert dashboard._boot_catchup_plan(at(9, 10), {"a": dead}, {"a"})["auction"] is True
    # Half-day: the main window ends 40 min before the 13:00 close.
    assert dashboard._boot_catchup_plan(at(12, 10, 11, 27), {"a": {"last_run": "2026-11-26T13:35:00"}}, set())["main"] is True
    assert dashboard._boot_catchup_plan(at(12, 30, 11, 27), {"a": {"last_run": "2026-11-26T13:35:00"}}, set())["main"] is False
    assert dashboard._boot_catchup_plan(at(15, 45), {"a": {"last_run": "2026-10-27T13:35:00"}}, set())["main"] is False
    # A slot state that is not a dict -> the check errors loudly, not silently.
    assert any("check errored" in r for r in dashboard._boot_catchup_plan(at(10, 0), {"a": None}, set())["reasons"])


def test_schedule_boot_catchup_wiring(tmp_path, monkeypatch):
    import dashboard
    from apscheduler.schedulers.background import BackgroundScheduler
    mc = make_mc(tmp_path, exec_style="auction")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {"last_run": "2026-10-27T13:35:00",
                                                                      "last_rebalance": None})
    sched = BackgroundScheduler(timezone="America/New_York")
    now = datetime(2026, 10, 28, 9, 10, tzinfo=NY)
    for k in ("auction_last_run_status", "auction_skipped_due"):
        monkeypatch.delitem(dashboard.bot_status, k, raising=False)
    plan = dashboard._schedule_boot_catchup(sched, now, logging.getLogger("t"))
    job = sched.get_job("catchup_auction")
    assert plan["auction"] is True and job.kwargs == {"stage": "auction"} and job.misfire_grace_time == 300
    assert sched.get_job("catchup_main") is None                         # 09:10 is before the main window
    # 10:00 ET, no live run today -> main replay with a real grace; a cron about to fire suppresses it.
    sched2 = BackgroundScheduler(timezone="America/New_York")
    plan2 = dashboard._schedule_boot_catchup(sched2, datetime(2026, 10, 28, 10, 0, tzinfo=NY), logging.getLogger("t"))
    assert plan2["main"] is True and sched2.get_job("catchup_main").misfire_grace_time == 900
    sched3 = BackgroundScheduler(timezone="America/New_York")

    class CronStub:
        next_run_time = datetime(2026, 10, 28, 9, 35, 30, tzinfo=NY)
    monkeypatch.setattr(sched3, "get_job", lambda jid: CronStub() if jid == "trading_pipeline" else None)
    plan3 = dashboard._schedule_boot_catchup(sched3, datetime(2026, 10, 28, 9, 35, tzinfo=NY), logging.getLogger("t"))
    assert plan3["main"] is False and any("cron fires" in r for r in plan3["reasons"])
    assert sched3.get_job("catchup_main") is None
    # Lost stage after the window -> status trace for /ready.
    sched4 = BackgroundScheduler(timezone="America/New_York")
    plan4 = dashboard._schedule_boot_catchup(sched4, datetime(2026, 10, 28, 9, 25, tzinfo=NY), logging.getLogger("t"))
    with dashboard.status_lock:
        assert plan4["lost_auction"] == ["combo_v2_process"]
        assert dashboard.bot_status["auction_last_run_status"] == "skipped"
        assert dashboard.bot_status["auction_skipped_due"] == ["combo_v2_process"]
        dashboard.bot_status["auction_last_run_status"] = None
        dashboard.bot_status["auction_last_run_at"] = None
        dashboard.bot_status.pop("auction_skipped_due", None)
        dashboard.bot_status.pop("boot_catchup", None)


def test_half_day_wait_cutoff(monkeypatch):
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
    _dash_clock(monkeypatch, dashboard, 12, 35, month=11, day=27)          # 25 min before a 13:00 close
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "running"
        dashboard.bot_status["running_stage"] = "auction"
    dashboard.run_trading_pipeline(stage="main")
    with dashboard.status_lock:
        assert seen == [] and slept == [] and dashboard.bot_status["skipped_main_run_blocker"] == "auction"
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["running_stage"] = None
        dashboard.bot_status.pop("skipped_main_run_at", None)
        dashboard.bot_status.pop("skipped_main_run_blocker", None)


def test_slot_failures_of_the_auction_slot_do_not_count_as_a_full_run(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path, exec_style="auction")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])

    def boom(**kw):
        raise runner_mod.SlotFailures({"combo_v2_process": "down"})
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", boom)
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {})
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["models"]["combo_v2_process"] = {"state": "idle"}
        dashboard.bot_status.pop("last_full_run_at", None)
    dashboard.run_trading_pipeline(stage="main")
    with dashboard.status_lock:
        assert "last_full_run_at" not in dashboard.bot_status
        assert dashboard.bot_status["last_failed_slots"] == ["combo_v2_process"]
        dashboard.bot_status["models"].pop("combo_v2_process", None)
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["last_run_status"] = None
        dashboard.bot_status.pop("last_failed_slots", None)


def test_skip_time_due_list_exempts_a_stage_that_already_ran(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path, exec_style="auction")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    # The clock is pinned to 2026-10-28 09:21 ET, so the stage record must carry THAT date.
    monkeypatch.setattr(dashboard, "_load_model_state",
                        lambda name: {"last_rebalance": None,
                                      "auction_stage": {"date": "2026-10-28", "outcome": "skipped:stale_panel"}})
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", lambda **kw: None)
    monkeypatch.setattr(dashboard.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("real sleep")))
    _dash_clock(monkeypatch, dashboard, 9, 21)
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "running"
        dashboard.bot_status["running_stage"] = "main"
    dashboard.run_trading_pipeline(stage="auction")
    with dashboard.status_lock:
        assert dashboard.bot_status["auction_skipped_due"] == []
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["running_stage"] = None
        dashboard.bot_status["auction_last_run_status"] = None
        dashboard.bot_status["auction_last_run_at"] = None
        dashboard.bot_status.pop("auction_skipped_due", None)


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
    for k in ("skipped_main_run_at", "skipped_main_run_blocker", "auction_skipped_due", "last_full_run_at", "boot_catchup"):
        monkeypatch.delitem(dashboard.bot_status, k, raising=False)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", None)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_at", None)
    return dashboard, dashboard.app.test_client()


def test_ready_lost_stage_traces(tmp_path, monkeypatch):
    mc = make_mc(tmp_path, exec_style="auction")
    today = _today_ny()
    state = {"auction_stage": {"date": today, "outcome": "skipped:nothing_submitted", "n_submitted": 0,
                               "recovered_from_submitting": True, "n_recovered": 0}}
    dashboard, c = _ready_client(monkeypatch, tmp_path, mc, state)
    r = c.get("/ready")
    assert r.status_code == 503 and any("died before submitting" in p for p in r.get_json()["problems"])
    state.clear()
    # last_rebalance is a naive UTC stamp in production (container clock UTC).
    state.update({"last_rebalance": datetime.now(timezone.utc).replace(tzinfo=None).isoformat()})
    r = c.get("/ready")
    assert r.status_code == 200 and any("no auction stage record" in w for w in r.get_json()["warnings"])
    monkeypatch.setitem(dashboard.bot_status, "boot_catchup", {"missed": ["catchup_main"]})
    r = c.get("/ready")
    assert r.status_code == 503 and any("catch-up" in p for p in r.get_json()["problems"])
