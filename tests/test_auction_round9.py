"""Opening-auction execution — review round 8 regressions (2026-10-16).

Covers: per-symbol blocking by the resting-order sweep (unblocked names
still trade, unknown order types block their symbol), the runner's
blocked-symbol bookkeeping and /ready warning, no auction merge onto an
aborted completion (history, protocol and gate readers), the dead
record's running notional and outcome after a drained finalize, the dead
record replaced when the venue was proven clear, the persisted
missed-stage date surviving a restart, and the venue-reconciliation
failure warning.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATA_DIR", os.environ.get("DATA_DIR") or tempfile.mkdtemp(prefix="combo_test_r9_"))

import pandas as pd
import pytest

from core.config import ModelConfig
import core.orders as orders_mod
import core.runner as runner_mod
from core.run_report import RunReport
from core.state import load_state, save_state

NY = ZoneInfo("America/New_York")
logger = logging.getLogger("test_auction_r9")


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


# ───────────────────────── orders layer ──────────────────────────────────

def test_unknown_order_types_block_their_symbol_instead_of_being_ignored(tmp_path, monkeypatch):
    trail = {"id": "t1", "symbol": "BRK.B", "time_in_force": "gtc", "type": "trailing_stop"}
    clock = [0.0]
    monkeypatch.setattr(orders_mod.time, "monotonic", lambda: clock.__setitem__(0, clock[0] + 5.0) or clock[0])
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)
    calls = []

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        calls.append((method, endpoint))
        if method == "GET" and endpoint.startswith("v2/orders?status=open"):
            return [dict(trail)]
        return {}
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    out = orders_mod.cancel_open_orders(make_mc(tmp_path), logger)
    assert out["n_inflight_market"] == 1 and out["blocked_symbols"] == ["BRK-B"]
    assert not any(c[0] == "DELETE" for c in calls)                     # never cancels a non-limit order


def test_blocked_names_are_left_untouched_and_the_rest_trades(tmp_path, monkeypatch):
    rest = {"id": "rest1", "symbol": "AAA", "time_in_force": "day", "type": "limit"}
    clock = [0.0]
    monkeypatch.setattr(orders_mod.time, "monotonic", lambda: clock.__setitem__(0, clock[0] + 5.0) or clock[0])
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)
    posted = []

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        if method == "GET" and endpoint.startswith("v2/orders?status=open"):
            return [dict(rest)]                                          # the cancel never takes effect
        if method == "DELETE":
            raise RuntimeError("422 order is not cancelable")
        if method == "POST" and endpoint == "v2/orders":
            posted.append(dict(data))
            return {"id": f"o{len(posted)}", "status": "accepted"}
        return {}
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {"portfolio_value": "100000", "cash": "100000",
                                                                        "buying_power": "200000"})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: {})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {})
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, *a, **kw: {"id": oid, "status": "filled",
                                                                                "filled_qty": "1", "filled_avg_price": "100"})
    report = RunReport()
    rb = orders_mod.rebalance_portfolio(["AAA", "BBB"], [("AAA", 1.0), ("BBB", 1.0)],
                                        make_mc(tmp_path, exec_style="market"), StubJournal(), logger, report,
                                        target_weights={"AAA": 0.05, "BBB": 0.05})
    assert not rb.get("aborted_open_orders")
    assert rb["blocked_symbols"] == ["AAA"] and rb["n_orders_blocked"] == 1
    assert [p["symbol"] for p in posted] == ["BBB"]


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
        return dict(rebalance_result or {"dry_run": False})
    monkeypatch.setattr(runner_mod, "rebalance_portfolio", fake_rebalance)
    return calls


def _panel(dates, close=50.0):
    idx = pd.to_datetime(dates)
    return {"AAA": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx)}


def _plan_state(today, **extra):
    st = {"last_rebalance": None, "history": [],
          "auction_stage": {"date": today, "outcome": "submitted", "n_submitted": 2, "run_id": "RID",
                            "recovery_checked": "x",
                            "plan": {"target_symbols": ["AAA"], "rankings": [["AAA", 1.0]],
                                     "target_weights": {"AAA": 1.0}, "diagnostics": {}},
                            "summary": {"n": 2, "n_filled": 1, "submitted_notional_usd": 200.0,
                                        "filled_notional_anchor_usd": 100.0, "filled_notional_usd": 100.0},
                            "pending": []}}
    st["auction_stage"].update(extra)
    return st


def test_runner_records_blocked_symbols_for_ready(tmp_path, monkeypatch):
    mc = make_mc(tmp_path, exec_style="marketable_limit", name="combo_v2")
    _wire_runner(monkeypatch, {"AAA": 1.0},
                 rebalance_result={"dry_run": False, "blocked_symbols": ["XYZ"], "n_orders_blocked": 1})
    save_state({"last_rebalance": None, "history": []}, mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(_today_ny())]), pd.DataFrame(),
                                macro_data={}, stage="main")
    st = load_state(mc)
    assert st["last_rebalance"] and st["rebalance_blocked_symbols"] == {"date": _today_ny(), "symbols": ["XYZ"]}


def test_aborted_completion_gets_no_auction_merge_and_no_run_id(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    _wire_runner(monkeypatch, {"ZZZ": 1.0}, rebalance_result={"dry_run": False, "aborted_open_orders": True})
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    save_state(_plan_state(today), mc)
    runner_mod.run_single_model(mc, {}, pd.DataFrame(), macro_data={}, stage="main")
    st = load_state(mc)
    h = st["history"][-1]
    assert "reconciliation" not in h["result"] and "auction_run_id" not in h
    assert st.get("last_rebalance") is None and not st["auction_stage"].get("completed_at")
    # The plan is still today's and open: a later same-day main run can complete it.
    assert runner_mod._todays_auction_plan(st, today) is not None
    # Readers never count it as a funded rebalance.
    from core.gate_update import _last_funded_rebalance
    assert _last_funded_rebalance([h]) == {}
    assert _last_funded_rebalance([{"result": {"skipped_auction_pending": True, "reconciliation": {"flow": {}}}}]) == {}


def test_dead_record_keeps_its_notional_and_outcome_after_a_drained_finalize(tmp_path, monkeypatch):
    """The 09:06 stage dies after persisting o1/o2; at 09:35 the venue listing
    fails but the persisted rows finalize; the next run's listing succeeds
    with nothing extra. The record must stay 'submitted' with its notional."""
    mc = make_mc(tmp_path)
    today = _today_ny()
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    replies = [None, []]
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: replies.pop(0))
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda mc_, pending, journal, lg, **kw: {"n": len(pending), "n_filled": len(pending),
                                                                 "n_unfilled": 0, "n_cancelled_open": 0,
                                                                 "submitted_notional_usd": 700.0,
                                                                 "filled_notional_anchor_usd": 700.0,
                                                                 "filled_notional_usd": 700.0,
                                                                 "fill_rate_notional": 1.0, "vw_cost_vs_open_bps": 0.0,
                                                                 "vw_cost_weight_usd": 700.0, "n_remaining_unread": 0,
                                                                 "remaining": []})
    st = {"auction_stage": {"date": today, "outcome": "submitting", "run_id": "RID", "n_submitted": 2,
                            "submitted_notional_usd": 700.0, "order_ids": ["o1", "o2"],
                            "plan": {"target_symbols": ["AAA"]},
                            "pending": [{"order_id": "o1", "symbol": "AAA", "notional_est": 400.0,
                                         "submitted_at": f"{today}T13:06:00+00:00", "run_id": "RID"},
                                        {"order_id": "o2", "symbol": "BBB", "notional_est": 300.0,
                                         "submitted_at": f"{today}T13:06:01+00:00", "run_id": "RID"}]}}
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    assert st["auction_stage"]["outcome"] == "submitting" and st["auction_stage"]["pending"] == []
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today)
    a = st["auction_stage"]
    assert a["outcome"] == "submitted" and a["submitted_notional_usd"] == 700.0 and a["n_submitted"] == 2
    assert a["recovered_from_submitting"] is True and a["n_recovered"] == 0


def test_persist_callback_keeps_a_running_notional(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev = runner_mod._previous_session_iso(today)
    seen = []

    def spy_rebalance(target_symbols, rankings, mc_, journal, lg, report, **kw):
        kw["on_submitted"]({"order_id": "o1", "symbol": "AAA", "notional_est": 400.0}, "RID")
        kw["on_submitted"]({"order_id": "o2", "symbol": "BBB", "notional_est": 300.0}, "RID")
        seen.append(dict(load_state(mc_)["auction_stage"]))
        raise RuntimeError("container killed mid-loop")
    _wire_runner(monkeypatch, {"AAA": 0.5, "BBB": 0.5})
    monkeypatch.setattr(runner_mod, "rebalance_portfolio", spy_rebalance)
    save_state({"last_rebalance": None, "history": []}, mc)
    with pytest.raises(RuntimeError):
        runner_mod.run_single_model(mc, {"AAA": _panel([prev])["AAA"], "BBB": _panel([prev])["AAA"]},
                                    pd.DataFrame(), macro_data={}, stage="auction")
    rec = load_state(mc)["auction_stage"]
    assert rec["outcome"] == "submitting" and rec["submitted_notional_usd"] == 700.0
    assert rec["order_ids"] == ["o1", "o2"] and len(rec["pending"]) == 2


def test_venue_proven_clear_then_stage_skip_replaces_the_dead_record(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev2 = runner_mod._previous_session_iso(runner_mod._previous_session_iso(today))
    _wire_runner(monkeypatch, {"AAA": 1.0})
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: [])   # venue empty
    dead = {"date": today, "outcome": "submitting", "pending": [], "n_submitted": 0, "run_id": None,
            "plan": {"target_symbols": ["AAA"], "rankings": [["AAA", 1.0]], "target_weights": {"AAA": 1.0}}}
    save_state({"last_rebalance": None, "history": [], "auction_stage": dict(dead)}, mc)
    # A stale panel makes the re-run skip after the venue was proven empty.
    runner_mod.run_single_model(mc, _panel([prev2]), pd.DataFrame(), macro_data={}, stage="auction")
    st = load_state(mc)["auction_stage"]
    assert st["outcome"] == "skipped:stale_panel" and st["replaced_dead_record"] is True


def test_missed_stage_date_is_persisted_and_survives_a_restart(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    _wire_runner(monkeypatch, {"AAA": 1.0}, rebalance_result={"dry_run": False})
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    save_state({"last_rebalance": None, "history": []}, mc)                # no stage record today
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(today)]), pd.DataFrame(),
                                macro_data={}, stage="main")
    st = load_state(mc)
    assert st["auction_stage_missed_date"] == today
    # /ready on a FRESH process (empty bot_status) still reports the lost day.
    import dashboard
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr("core.market.is_market_open", lambda now=None: False)
    monkeypatch.setattr(dashboard, "STATE_DIR", tmp_path)
    (tmp_path / "x.json").write_text("{}")
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: load_state(mc))
    monkeypatch.setattr("core.invariants.load_last_status", lambda: None)
    monkeypatch.setitem(dashboard.bot_status, "last_run_status", "success")
    monkeypatch.setitem(dashboard.bot_status, "last_run_at", datetime.now(timezone.utc).isoformat())
    for k in ("skipped_main_run_at", "auction_skipped_due", "last_full_run_at", "boot_catchup"):
        monkeypatch.delitem(dashboard.bot_status, k, raising=False)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", None)
    r = dashboard.app.test_client().get("/ready")
    assert r.status_code == 503 and any("WITHOUT its opening-auction plan" in p for p in r.get_json()["problems"])
    # A completed plan on a later day clears it.
    save_state(_plan_state(today, completed_at=None), mc)
    _wire_runner(monkeypatch, {"AAA": 1.0}, rebalance_result={"dry_run": False})
    st2 = load_state(mc)
    st2["auction_stage_missed_date"] = "2026-01-01"
    save_state(st2, mc)
    runner_mod.run_single_model(mc, {}, pd.DataFrame(), macro_data={}, stage="main")
    assert "auction_stage_missed_date" not in load_state(mc)


def test_ready_warns_on_a_failed_venue_reconciliation_and_blocked_names(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path)
    state = {"auction_stage": {"date": _today_ny(), "outcome": "submitting", "recovery_error": "x"},
             "rebalance_blocked_symbols": {"date": _today_ny(), "symbols": ["XYZ"]}}
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr("core.market.is_market_open", lambda now=None: False)
    monkeypatch.setattr(dashboard, "STATE_DIR", tmp_path)
    (tmp_path / "x.json").write_text("{}")
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: dict(state))
    monkeypatch.setattr("core.invariants.load_last_status", lambda: None)
    monkeypatch.setitem(dashboard.bot_status, "last_run_status", "success")
    monkeypatch.setitem(dashboard.bot_status, "last_run_at", datetime.now(timezone.utc).isoformat())
    for k in ("skipped_main_run_at", "auction_skipped_due", "last_full_run_at", "boot_catchup"):
        monkeypatch.delitem(dashboard.bot_status, k, raising=False)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", None)
    w = dashboard.app.test_client().get("/ready").get_json()["warnings"]
    assert any("venue reconciliation failed" in x for x in w) and any("left untouched" in x for x in w)


def test_protocol_recons_skip_non_trading_entries(tmp_path, monkeypatch):
    import dashboard
    src = Path(dashboard.__file__).read_text()
    # Both history readers filter aborted / deferred completions (review round 8).
    assert src.count('aborted_open_orders") or res_.get("skipped_auction_pending")') == 1
    assert 'if not ((e.get("result") or {}).get("aborted_open_orders")' in src
