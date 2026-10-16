"""Opening-auction execution — review round 9 regressions (2026-10-16).

Covers: blocked names removed BEFORE the buying-power pre-flight (no
spurious auto-scale / funding floor; a held blocked name pinned at its
value), an all-blocked completion still completes the plan, failed
cancels re-sent inside the settle loop, a missed-stage rebalance spending
today's plan, the missed-stage date not cleared by a same-day completion,
and the blocked-residue warning persisting until a clean rebalance.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATA_DIR", os.environ.get("DATA_DIR") or tempfile.mkdtemp(prefix="combo_test_r10_"))

import pandas as pd
import pytest

from core.config import ModelConfig
import core.orders as orders_mod
import core.runner as runner_mod
from core.run_report import RunReport
from core.state import load_state, save_state

NY = ZoneInfo("America/New_York")
logger = logging.getLogger("test_auction_r10")


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


def _broker(open_orders, posted, delete_fails=0, deletes=None):
    deletes = deletes if deletes is not None else []
    state = {"fails": delete_fails}

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        if method == "GET" and endpoint.startswith("v2/orders?status=open"):
            return [dict(o) for o in open_orders]
        if method == "GET" and endpoint.startswith("v2/orders/"):
            return {"status": "canceled"}
        if method == "DELETE":
            deletes.append(endpoint)
            if state["fails"] > 0:
                state["fails"] -= 1
                raise RuntimeError("503 service unavailable")
            oid = endpoint.rsplit("/", 1)[-1]
            open_orders[:] = [o for o in open_orders if o["id"] != oid]
            return {}
        if method == "POST" and endpoint == "v2/orders":
            posted.append(dict(data))
            return {"id": f"o{len(posted)}", "status": "accepted"}
        return {}
    return fake_request


def _wire(monkeypatch, fake_request, bp="200000", positions=None):
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {"portfolio_value": "100000",
                                                                        "cash": "100000", "buying_power": bp})
    monkeypatch.setattr(orders_mod, "get_positions", lambda mc_, lg: dict(positions or {}))
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: {})
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, *a, **kw: {"id": oid, "status": "filled",
                                                                                "filled_qty": "1", "filled_avg_price": "100"})
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)
    clock = [0.0]
    monkeypatch.setattr(orders_mod.time, "monotonic", lambda: clock.__setitem__(0, clock[0] + 5.0) or clock[0])


def test_blocked_buy_is_excluded_before_the_buying_power_preflight(tmp_path, monkeypatch):
    """XXX has a resting buy limit that will not cancel (its reservation is
    already out of buying_power). BBB alone fits the free buying power, so
    there must be NO auto-scale and no funding evidence."""
    rest = [{"id": "r1", "symbol": "XXX", "time_in_force": "day", "type": "limit"}]
    posted = []
    _wire(monkeypatch, _broker(rest, posted, delete_fails=99), bp="40000")
    rb = orders_mod.rebalance_portfolio(["XXX", "BBB"], [("XXX", 1.0), ("BBB", 1.0)],
                                        make_mc(tmp_path, exec_style="market"), StubJournal(), logger, RunReport(),
                                        target_weights={"XXX": 0.2, "BBB": 0.2})
    assert rb["blocked_symbols"] == ["XXX"] and not rb.get("auto_scaled_to_buying_power")
    assert [p["symbol"] for p in posted] == ["BBB"]
    assert float(posted[0].get("notional") or 0) == pytest.approx(40_000.0, rel=1e-3)
    assert rb["target_gross_usd"] == pytest.approx(40_000.0)


def test_held_blocked_name_is_pinned_at_its_market_value(tmp_path, monkeypatch):
    mkt = [{"id": "m1", "symbol": "AAA", "time_in_force": "day", "type": "market"}]
    posted = []
    pos = {"AAA": {"qty": 300.0, "market_value": 30_000.0, "avg_entry": 100.0, "current_price": 100.0,
                   "unrealized_pl": 0.0, "unrealized_pl_pct": 0.0}}
    _wire(monkeypatch, _broker(mkt, posted), positions=pos)
    rb = orders_mod.rebalance_portfolio(["AAA", "BBB"], [("AAA", 1.0), ("BBB", 1.0)],
                                        make_mc(tmp_path, exec_style="market"), StubJournal(), logger, RunReport(),
                                        target_weights={"AAA": 0.05, "BBB": 0.05})
    assert rb["blocked_symbols"] == ["AAA"] and [p["symbol"] for p in posted] == ["BBB"]
    # AAA counted at its current $30k (not its $10k target) in the book target.
    assert rb["target_gross_usd"] == pytest.approx(30_000.0 + 10_000.0)


def test_all_blocked_completion_still_completes(tmp_path, monkeypatch):
    mkt = [{"id": "m1", "symbol": "AAA", "time_in_force": "day", "type": "market"}]
    posted = []
    _wire(monkeypatch, _broker(mkt, posted))
    rb = orders_mod.rebalance_portfolio(["AAA"], [("AAA", 1.0)], make_mc(tmp_path), StubJournal(), logger,
                                        RunReport(), target_weights={"AAA": 0.05}, run_id="RID")
    assert not rb.get("aborted_open_orders") and rb["completion_residue_blocked"] == ["AAA"]
    # Nothing posted, but the book-level reconciliation still ran, so the
    # runner can merge the auction leg and stamp the completion.
    assert posted == [] and "reconciliation" in rb
    # The same plan as an ordinary rebalance aborts.
    rb2 = orders_mod.rebalance_portfolio(["AAA"], [("AAA", 1.0)], make_mc(tmp_path, exec_style="market"),
                                         StubJournal(), logger, RunReport(), target_weights={"AAA": 0.05})
    assert rb2["aborted_open_orders"] is True


def test_failed_cancel_is_resent_inside_the_settle_loop(tmp_path, monkeypatch):
    rest = [{"id": "r1", "symbol": "AAA", "time_in_force": "day", "type": "limit"}]
    deletes = []
    _wire(monkeypatch, _broker(rest, [], delete_fails=1, deletes=deletes))
    out = orders_mod.cancel_open_orders(make_mc(tmp_path), logger)
    assert deletes == ["v2/orders/r1", "v2/orders/r1"]          # first try 503, second accepted
    assert out["n_still_open"] == 0 and out["n_cancel_requested"] == 1 and out["blocked_symbols"] == []
    # Never more than 3 attempts for a cancel that keeps failing.
    rest2 = [{"id": "r2", "symbol": "BBB", "time_in_force": "day", "type": "limit"}]
    deletes2 = []
    _wire(monkeypatch, _broker(rest2, [], delete_fails=99, deletes=deletes2))
    out2 = orders_mod.cancel_open_orders(make_mc(tmp_path), logger)
    assert len(deletes2) == 3 and out2["blocked_symbols"] == ["BBB"]


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

    def fake_rebalance(target_symbols, rankings, mc, journal, lg, report, **kw):
        calls.append({"target_symbols": list(target_symbols), **kw})
        r = rebalance_result.pop(0) if isinstance(rebalance_result, list) else rebalance_result
        return dict(r or {"dry_run": False})
    monkeypatch.setattr(runner_mod, "rebalance_portfolio", fake_rebalance)
    return calls


def _panel(dates, close=50.0):
    idx = pd.to_datetime(dates)
    return {"AAA": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx),
            "BBB": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx)}


def test_missed_stage_rebalance_spends_todays_plan_and_keeps_the_alarm(tmp_path, monkeypatch):
    """The 09:06 stage died between an accepted POST and its persist
    callback; the 09:35 listing fails; the run re-plans. A later run that
    adopts the venue order must NOT trade the book back to the morning plan
    nor clear today's missed-stage alarm."""
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev = runner_mod._previous_session_iso(today)
    calls = _wire_runner(monkeypatch, {"BBB": 1.0}, rebalance_result={"dry_run": False})
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    venue = [None, [{"order_id": "v1", "symbol": "AAA", "side": "buy", "qty": 1.0, "prev_close": 50.0,
                     "notional_est": 50.0, "submitted_at": f"{today}T13:06:00+00:00", "recovered": True}]]
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: venue.pop(0))
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda mc_, pending, journal, lg, **kw: {"n": len(pending), "n_filled": len(pending),
                                                                 "n_unfilled": 0, "n_cancelled_open": 0,
                                                                 "submitted_notional_usd": 50.0,
                                                                 "filled_notional_anchor_usd": 50.0,
                                                                 "filled_notional_usd": 50.0, "fill_rate_notional": 1.0,
                                                                 "vw_cost_vs_open_bps": 0.0, "vw_cost_weight_usd": 50.0,
                                                                 "n_remaining_unread": 0, "remaining": []})
    dead = {"date": today, "outcome": "submitting", "n_submitted": 0, "pending": [], "run_id": None,
            "plan": {"target_symbols": ["AAA"], "rankings": [["AAA", 1.0]], "target_weights": {"AAA": 1.0}}}
    save_state({"last_rebalance": None, "history": [], "auction_stage": dict(dead)}, mc)
    runner_mod.run_single_model(mc, _panel([prev]), pd.DataFrame(), macro_data={}, stage="main")
    st = load_state(mc)
    assert calls[-1]["target_symbols"] == ["BBB"] and st["auction_stage_missed_date"] == today
    assert st["auction_stage"]["superseded_by_missed_rebalance"] is True and st["auction_stage"]["completed_at"]
    # A later run adopts the venue order but never resurrects the plan.
    runner_mod.run_single_model(mc, _panel([prev]), pd.DataFrame(), macro_data={}, force=True, stage="main")
    st = load_state(mc)
    assert calls[-1]["target_symbols"] == ["BBB"] and st["auction_stage_missed_date"] == today
    assert st["auction_stage"]["n_recovered"] == 1


def test_blocked_residue_warning_persists_until_a_clean_rebalance(tmp_path, monkeypatch):
    mc = make_mc(tmp_path, exec_style="marketable_limit", name="combo_v2")
    _wire_runner(monkeypatch, {"AAA": 1.0},
                 rebalance_result=[{"dry_run": False, "blocked_symbols": ["XYZ"]}, {"dry_run": False}])
    save_state({"last_rebalance": None, "history": []}, mc)
    prev = runner_mod._previous_session_iso(_today_ny())
    runner_mod.run_single_model(mc, _panel([prev]), pd.DataFrame(), macro_data={}, stage="main")
    assert load_state(mc)["rebalance_blocked_symbols"]["symbols"] == ["XYZ"]
    runner_mod.run_single_model(mc, _panel([prev]), pd.DataFrame(), macro_data={}, force=True, stage="main")
    assert "rebalance_blocked_symbols" not in load_state(mc)


# ───────────── review round 10 regressions ──────────────────────────────

def test_unplanned_live_order_is_reported_and_keeps_the_alarm(tmp_path, monkeypatch):
    rest = [{"id": "z1", "symbol": "ZZZ", "time_in_force": "day", "type": "limit"}]
    posted = []
    _wire(monkeypatch, _broker(rest, posted, delete_fails=99))
    rb = orders_mod.rebalance_portfolio(["AAA", "BBB"], [("AAA", 1.0), ("BBB", 1.0)],
                                        make_mc(tmp_path, exec_style="market"), StubJournal(), logger, RunReport(),
                                        target_weights={"AAA": 0.05, "BBB": 0.05})
    assert rb["live_orders_unplanned"] == ["ZZZ"] and "blocked_symbols" not in rb
    assert sorted(p["symbol"] for p in posted) == ["AAA", "BBB"]
    # The runner keeps the alarm while the unplanned order is live.
    mc = make_mc(tmp_path / "r", exec_style="marketable_limit", name="combo_v2")
    _wire_runner(monkeypatch, {"AAA": 1.0}, rebalance_result={"dry_run": False, "live_orders_unplanned": ["ZZZ"]})
    save_state({"last_rebalance": None, "history": [],
                "rebalance_blocked_symbols": {"date": "2026-10-15", "symbols": ["XYZ"]}}, mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(_today_ny())]), pd.DataFrame(),
                                macro_data={}, stage="main")
    assert load_state(mc)["rebalance_blocked_symbols"]["symbols"] == ["ZZZ"]


def test_unreconciled_dead_record_defers_a_main_run_until_the_open(tmp_path, monkeypatch):
    today = _today_ny()
    dead = {"auction_stage": {"date": today, "outcome": "submitting", "pending": [], "n_submitted": 0}}
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: False)        # 09:30:30 ET
    assert runner_mod._auction_pending_live(dead, today) is True
    dead["auction_stage"]["recovery_checked"] = "x"
    assert runner_mod._auction_pending_live(dead, today) is False
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    dead["auction_stage"].pop("recovery_checked")
    assert runner_mod._auction_pending_live(dead, today) is False
    # End to end: the 09:30 main run does not trade on top of the hidden orders.
    mc = make_mc(tmp_path)
    calls = _wire_runner(monkeypatch, {"BBB": 1.0})
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: False)
    monkeypatch.setattr("core.orders.recover_unpersisted_auction_orders", lambda *a, **kw: None)
    save_state({"last_rebalance": None, "history": [],
                "auction_stage": {"date": today, "outcome": "submitting", "pending": [], "n_submitted": 0,
                                  "plan": {"target_symbols": ["AAA"]}}}, mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(today)]), pd.DataFrame(),
                                macro_data={}, stage="main")
    st = load_state(mc)
    assert calls == [] and st["auction_completion_deferred"] == today
    assert not st["auction_stage"].get("completed_at") and not st.get("auction_stage_missed_date")


def test_blocked_warning_never_advises_a_forced_rerun_during_a_cooldown(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path, exec_style="marketable_limit", name="combo_v2")
    state = {"rebalance_blocked_symbols": {"date": "2026-10-14", "symbols": ["ABC"]},
             "reentry_cooldown_until": "2099-01-01"}
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
    c = dashboard.app.test_client()
    w = [x for x in c.get("/ready").get_json()["warnings"] if "ABC" in x]
    assert w and "do NOT force a re-run" in w[0] and "re-run the slot once" not in w[0]
    state.pop("reentry_cooldown_until")
    w = [x for x in c.get("/ready").get_json()["warnings"] if "ABC" in x]
    assert w and "re-run the slot once they settle" in w[0]
