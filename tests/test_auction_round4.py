"""Opening-auction execution — review round 4 regressions (2026-10-15).

Covers: finalize robustness (post-cancel unreadable, 404 terminal, journal
failure, idempotent re-journal, budget, per-order isolation with a raising
payload, anchor-priced fill rate with a partial fill, reference-open source
selection), submission idempotency keys + recovery, stale-close guard,
summary accumulation on retry, orphan age-out, the plan short-circuit in
run_single_model (completion with NO panel), the plan-only fallback and
loud slot failures in run_pipeline, the auction stage's no-log-file
no-op path, and the dashboard's wait-to-cutoff / alarm semantics.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_r4_"))

import pandas as pd
import pytest

from core.config import ModelConfig
import core.orders as orders_mod
import core.runner as runner_mod
from core.run_report import RunReport
from core.state import load_state, save_state

NY = ZoneInfo("America/New_York")
logger = logging.getLogger("test_auction_r4")


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


def _pending(order_id, sym="AAA", qty=10, prev_close=50.0, side="buy", submitted_at=None):
    return {"submitted_at": submitted_at or f"{_today_ny()}T13:06:00+00:00", "run_id": "RID",
            "order_status": "accepted", "symbol": sym, "side": side,
            "order_type": "limit" if side == "buy" else "market", "qty": qty,
            "limit_price": round(prev_close * 1.01, 2) if side == "buy" else None,
            "prev_close": prev_close, "notional_est": qty * prev_close, "order_id": order_id}


def _no_sleep(monkeypatch):
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)


def _snap(monkeypatch, mapping):
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda syms, mc_, lg: mapping)


# ───────────────────────── finalize robustness ──────────────────────────

def test_poll_order_status_404_is_terminal_not_unknown(tmp_path, monkeypatch):
    from core.alpaca import AlpacaHTTPError

    def raising(method, endpoint, mc, data=None, logger=None, **kw):
        raise AlpacaHTTPError("Alpaca GET v2/orders/x: 404 order not found", status_code=404)
    monkeypatch.setattr(orders_mod, "alpaca_request", raising)
    st = orders_mod.poll_order_status("x", make_mc(tmp_path), logger, max_wait=0.2)
    assert st["status"] == "not_found"


def test_finalize_not_found_is_journaled_and_dropped_from_pending(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, mc, lg, max_wait=8.0, **kw: {"status": "not_found", "id": oid})
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    _snap(monkeypatch, {"AAA": {"day_open": 50.0, "day_bar_date": _today_ny()}})
    j = StubJournal()
    stats = orders_mod.finalize_auction_orders(make_mc(tmp_path), [_pending("o1")], j, logger)
    assert stats["n_not_found"] == 1 and stats["n_unfilled"] == 1 and stats["remaining"] == []
    assert j.records[0].order_status == "not_found" and j.records[0].shares is None


def test_finalize_post_cancel_unreadable_keeps_entry_pending(tmp_path, monkeypatch):
    """A partial fill whose post-cancel read fails must never be journaled
    as unfilled (the 60 executed shares would vanish) — kept pending."""
    _no_sleep(monkeypatch)
    calls = []

    def fake_poll(oid, mc, lg, max_wait=8.0, **kw):
        calls.append(oid)
        if len(calls) == 1:
            return {"id": oid, "status": "partially_filled", "filled_qty": "60", "filled_avg_price": "100.2"}
        return {"status": "unknown", "id": oid}          # every read after the cancel fails
    monkeypatch.setattr(orders_mod, "poll_order_status", fake_poll)
    deletes = []
    monkeypatch.setattr(orders_mod, "alpaca_request",
                        lambda m, e, mc, data=None, logger=None, **kw: deletes.append(e) if m == "DELETE" else {})
    _snap(monkeypatch, {"AAA": {"day_open": 100.0, "day_bar_date": _today_ny()}})
    j = StubJournal()
    stats = orders_mod.finalize_auction_orders(make_mc(tmp_path), [_pending("o1", qty=100, prev_close=100.0)], j, logger)
    assert deletes == ["v2/orders/o1"]
    assert stats["n"] == 0 and stats["n_remaining_unread"] == 1 and stats["remaining"][0]["order_id"] == "o1"
    assert j.records == [] and stats["submitted_notional_usd"] == 0.0


def test_finalize_journal_failure_keeps_entry_pending_and_uncounted(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, mc, lg, max_wait=8.0, **kw:
                        {"id": oid, "status": "filled", "filled_qty": "10", "filled_avg_price": "50.0"})
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    _snap(monkeypatch, {"AAA": {"day_open": 50.0, "day_bar_date": _today_ny()}})

    class BadJournal(StubJournal):
        def log_trade(self, record):
            raise OSError("disk full")
    stats = orders_mod.finalize_auction_orders(make_mc(tmp_path), [_pending("o1")], BadJournal(), logger)
    assert stats["n_filled"] == 0 and stats["filled_notional_usd"] == 0.0 and stats["submitted_notional_usd"] == 0.0
    assert [p["order_id"] for p in stats["remaining"]] == ["o1"]


def test_finalize_is_idempotent_after_a_restart(tmp_path, monkeypatch):
    """Rows already in the journal (restart between appends and the state
    save) are counted but never appended twice."""
    _no_sleep(monkeypatch)
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, mc, lg, max_wait=8.0, **kw:
                        {"id": oid, "status": "filled", "filled_qty": "10", "filled_avg_price": "50.0"})
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    _snap(monkeypatch, {"AAA": {"day_open": 50.0, "day_bar_date": _today_ny()}, "BBB": {"day_open": 50.0, "day_bar_date": _today_ny()}})
    mc = make_mc(tmp_path)
    sub_day = _today_ny().replace("-", "")
    jl = tmp_path / "trades.jsonl"
    jl.write_text('{"trade_id": "%s_auction_%s_buy_AAA", "symbol": "AAA"}\n' % (mc.name, sub_day))
    j = StubJournal(path=str(jl))
    stats = orders_mod.finalize_auction_orders(mc, [_pending("o1", "AAA"), _pending("o2", "BBB")], j, logger)
    assert stats["n"] == 2 and stats["n_filled"] == 2 and stats["n_already_journaled"] == 1
    assert [r.symbol for r in j.records] == ["BBB"]


def test_finalize_wall_clock_budget_defers_untouched_entries(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(orders_mod.time, "monotonic", lambda: clock[0])

    def fake_poll(oid, mc, lg, max_wait=8.0, **kw):
        clock[0] += 100.0                                   # each read burns the whole budget
        return {"id": oid, "status": "filled", "filled_qty": "10", "filled_avg_price": "50.0"}
    monkeypatch.setattr(orders_mod, "poll_order_status", fake_poll)
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    _snap(monkeypatch, {})
    j = StubJournal()
    stats = orders_mod.finalize_auction_orders(make_mc(tmp_path), [_pending("o1", "AAA"), _pending("o2", "BBB")],
                                               j, logger, budget_s=5.0)
    assert stats["n"] == 1 and stats["n_budget_deferred"] == 1
    assert [p["order_id"] for p in stats["remaining"]] == ["o2"]


def test_finalize_per_order_isolation_with_a_raising_payload(tmp_path, monkeypatch):
    """A malformed fill price raises inside _apply_order_fill: that entry is
    kept pending and EXCLUDED from the submitted notional; the others are
    journaled normally."""
    _no_sleep(monkeypatch)
    states = {"o1": {"id": "o1", "status": "filled", "filled_qty": "10", "filled_avg_price": "abc"},
              "o2": {"id": "o2", "status": "filled", "filled_qty": "7", "filled_avg_price": "50.0"},
              "o3": {"id": "o3", "status": "canceled", "filled_qty": "0"}}
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, mc, lg, max_wait=8.0, **kw: dict(states[oid]))
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    _snap(monkeypatch, {s: {"day_open": 49.0, "day_bar_date": _today_ny()} for s in ("AAA", "BBB", "CCC")})
    j = StubJournal()
    stats = orders_mod.finalize_auction_orders(
        make_mc(tmp_path), [_pending("o1", "AAA", 10), _pending("o2", "BBB", 7), _pending("o3", "CCC", 4)], j, logger)
    assert [p["order_id"] for p in stats["remaining"]] == ["o1"] and stats["n_remaining_unread"] == 1
    assert stats["submitted_notional_usd"] == pytest.approx(7 * 50.0 + 4 * 50.0)
    assert stats["fill_rate_notional"] == pytest.approx(350.0 / 550.0, abs=1e-3)
    assert {r.symbol for r in j.records} == {"BBB", "CCC"}


def test_finalize_partial_fill_keeps_submitted_qty_and_anchor_rate(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, mc, lg, max_wait=8.0, **kw:
                        {"id": oid, "status": "canceled", "filled_qty": "4", "filled_avg_price": "49.0"})
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    _snap(monkeypatch, {"AAA": {"day_open": 49.0, "day_bar_date": _today_ny()}})
    j = StubJournal()
    stats = orders_mod.finalize_auction_orders(make_mc(tmp_path), [_pending("o1", "AAA", 10)], j, logger)
    row = j.records[0]
    assert row.qty_submitted == 10 and row.shares == 4 and row.target_weight_usd == pytest.approx(500.0)
    assert row.notional_usd == pytest.approx(4 * 49.0)
    assert stats["n_filled"] == 1 and stats["fill_rate_notional"] == pytest.approx(0.4)


def test_finalize_reference_open_sources(tmp_path, monkeypatch):
    """Panel open (today's bar) beats the snapshot; a snapshot bar dated
    another day is rejected; an earlier session's order gets none."""
    _no_sleep(monkeypatch)
    monkeypatch.setattr(orders_mod, "poll_order_status", lambda oid, mc, lg, max_wait=8.0, **kw:
                        {"id": oid, "status": "filled", "filled_qty": "10", "filled_avg_price": "50.5"})
    monkeypatch.setattr(orders_mod, "alpaca_request", lambda *a, **kw: {})
    yday = (datetime.now(NY) - timedelta(days=1)).strftime("%Y-%m-%d")
    _snap(monkeypatch, {"AAA": {"day_open": 60.0, "day_bar_date": _today_ny()},
                        "BBB": {"day_open": 60.0, "day_bar_date": _today_ny()},
                        "CCC": {"day_open": 60.0, "day_bar_date": yday},
                        "DDD": {"day_open": 60.0, "day_bar_date": _today_ny()}})
    j = StubJournal()
    pending = [_pending("o1", "AAA"), _pending("o2", "BBB"), _pending("o3", "CCC"),
               _pending("o4", "DDD", submitted_at="2026-01-05T14:06:00+00:00")]
    stats = orders_mod.finalize_auction_orders(make_mc(tmp_path), pending, j, logger, panel_open={"AAA": 50.0})
    rows = {r.symbol: r for r in j.records}
    assert rows["AAA"].reference_open == 50.0 and rows["AAA"].slippage_bps == pytest.approx(100.0)
    assert rows["BBB"].reference_open == 60.0
    assert rows["CCC"].reference_open is None and rows["CCC"].slippage_bps is None
    assert rows["DDD"].reference_open is None
    assert stats["reference_sources"] == {"panel": 1, "snapshot": 1, "none": 2}


# ───────────────────────── submission ────────────────────────────────────

def _submit(tmp_path, monkeypatch, fake_request, orders, last_close, stale=None):
    monkeypatch.setattr(orders_mod, "alpaca_request", fake_request)
    rb = {}
    summ = orders_mod.submit_auction_orders(orders, make_mc(tmp_path), last_close, logger, rb,
                                            run_id="combo_v2_process_20261028_130600_000000",
                                            stale_close=stale)
    return summ


def test_submit_sends_idempotency_key_and_recovers_a_lost_post(tmp_path, monkeypatch):
    posted, gets = [], []

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        if method == "POST":
            posted.append(dict(data))
            if data["symbol"] == "BBB":
                raise TimeoutError("read timed out")          # accepted server-side, response lost
            return {"id": f"o{len(posted)}", "status": "accepted"}
        if method == "GET" and endpoint.startswith("v2/orders:by_client_order_id"):
            gets.append(endpoint)
            return {"id": "recovered-1", "status": "accepted"}
        return {}
    orders = [{"symbol": "AAA", "side": "buy", "action": "buy", "notional": 1000.0, "trade_action": "new_position"},
              {"symbol": "BBB", "side": "buy", "action": "buy", "notional": 1000.0, "trade_action": "new_position"}]
    summ = _submit(tmp_path, monkeypatch, fake_request, orders, {"AAA": 100.0, "BBB": 100.0})
    coids = [p["client_order_id"] for p in posted]
    assert all(len(c) <= 48 for c in coids) and len(set(coids)) == 2
    assert summ["n_submitted"] == 2 and summ["n_skipped"] == 0
    by = {p["symbol"]: p for p in summ["pending"]}
    assert by["BBB"]["order_id"] == "recovered-1" and by["BBB"]["recovered"] is True
    assert gets and coids[1] in gets[0]
    # Deterministic: the same run_id/symbol always yields the same key.
    assert orders_mod._auction_client_order_id("combo_v2_process_20261028_130600_000000", "AAA") == coids[0]


def test_submit_skips_stale_close_names(tmp_path, monkeypatch):
    posted = []

    def fake_request(method, endpoint, mc, data=None, logger=None, **kw):
        posted.append(dict(data))
        return {"id": f"o{len(posted)}", "status": "accepted"}
    orders = [{"symbol": "AAA", "side": "buy", "action": "buy", "notional": 1000.0},
              {"symbol": "BBB", "side": "buy", "action": "buy", "notional": 1000.0}]
    summ = _submit(tmp_path, monkeypatch, fake_request, orders, {"AAA": 100.0, "BBB": 100.0}, stale={"BBB"})
    assert summ["n_submitted"] == 1 and [s["reason"] for s in summ["skipped"]] == ["stale_close"]


# ───────────────────────── runner helpers ────────────────────────────────

def test_previous_session_skips_weekends_and_holidays():
    assert runner_mod._previous_session_iso("2026-10-19") == "2026-10-16"   # Monday -> Friday
    assert runner_mod._previous_session_iso("2026-11-27") == "2026-11-25"   # Thanksgiving skipped
    assert runner_mod._previous_session_iso("2026-12-28") == "2026-12-24"   # Christmas (Fri) skipped


def test_accumulate_auction_summary_sums_and_reweights():
    prev = {"n": 39, "n_filled": 37, "n_unfilled": 2, "n_cancelled_open": 0, "submitted_notional_usd": 390_000.0,
            "filled_notional_anchor_usd": 370_000.0, "filled_notional_usd": 371_000.0,
            "fill_rate_notional": 0.9487, "vw_cost_vs_open_bps": 10.0, "n_remaining_unread": 1,
            "reference_sources": {"panel": 39, "snapshot": 0, "none": 0}}
    new = {"n": 1, "n_filled": 1, "n_unfilled": 0, "n_cancelled_open": 0, "submitted_notional_usd": 10_000.0,
           "filled_notional_anchor_usd": 10_000.0, "filled_notional_usd": 9_000.0,
           "fill_rate_notional": 1.0, "vw_cost_vs_open_bps": 50.0, "n_remaining_unread": 0,
           "reference_sources": {"panel": 0, "snapshot": 0, "none": 1}}
    out = runner_mod._accumulate_auction_summary(prev, new)
    assert out["n"] == 40 and out["n_filled"] == 38 and out["submitted_notional_usd"] == 400_000.0
    assert out["fill_rate_notional"] == pytest.approx(380_000.0 / 400_000.0)
    assert out["vw_cost_vs_open_bps"] == pytest.approx((10.0 * 371_000 + 50.0 * 9_000) / 380_000, abs=0.01)
    assert out["n_remaining_unread"] == 0 and out["n_finalize_passes"] == 2
    assert out["reference_sources"] == {"panel": 39, "snapshot": 0, "none": 1}
    assert runner_mod._accumulate_auction_summary(None, new) == new


def test_finalize_helper_retry_accumulates_and_ages_out_orphans(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    seen = []

    def fake_finalize(mc_, pending, journal, lg, **kw):
        seen.append(([p["order_id"] for p in pending], kw.get("panel_open")))
        return {"n": len(pending), "n_filled": len(pending), "n_unfilled": 0, "n_cancelled_open": 0,
                "submitted_notional_usd": 100.0 * len(pending), "filled_notional_anchor_usd": 100.0 * len(pending),
                "filled_notional_usd": 100.0 * len(pending), "fill_rate_notional": 1.0,
                "vw_cost_vs_open_bps": 5.0, "n_remaining_unread": 0, "remaining": []}
    monkeypatch.setattr("core.orders.finalize_auction_orders", fake_finalize)
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    today = _today_ny()
    old = (datetime.now(NY) - timedelta(days=9)).strftime("%Y-%m-%d")
    fresh = (datetime.now(NY) - timedelta(days=2)).strftime("%Y-%m-%d")
    st = {"auction_stage": {
        "date": today, "outcome": "submitted", "n_submitted": 40,
        "summary": {"n": 39, "n_filled": 37, "n_unfilled": 2, "n_cancelled_open": 0,
                    "submitted_notional_usd": 3900.0, "filled_notional_anchor_usd": 3700.0,
                    "filled_notional_usd": 3700.0, "fill_rate_notional": 0.9487, "vw_cost_vs_open_bps": 10.0,
                    "n_remaining_unread": 1},
        "pending": [{"order_id": "x", "submitted_at": f"{today}T13:06:00+00:00"}],
        "orphans": [{"order_id": "old1", "symbol": "OLD", "submitted_at": f"{old}T13:06:00+00:00"},
                    {"order_id": "new1", "symbol": "NEW", "submitted_at": f"{fresh}T13:06:00+00:00"}]}}
    idx = pd.to_datetime([today])
    panel = {"AAA": pd.DataFrame({"open": [50.0], "close": [51.0]}, index=idx)}
    runner_mod._finalize_auction_if_any(mc, st, StubJournal(), logger, RunReport(), today, stock_data=panel)
    a = st["auction_stage"]
    assert [o["order_id"] for o in a["orphans_dropped"]] == ["old1"] and a["orphans"] == []
    assert seen[0] == (["new1"], {"AAA": 50.0}) and seen[1][0] == ["x"]
    assert a["summary"]["n"] == 40 and a["summary"]["n_filled"] == 38 and a["summary"]["n_finalize_passes"] == 2
    assert a["last_retry_summary"]["n"] == 1 and "finalized_late" not in a and a["pending"] == []


def test_auction_pending_live_helper(monkeypatch):
    today = _today_ny()
    st = {"auction_stage": {"date": today, "pending": [{"order_id": "o1"}]}}
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: False)
    assert runner_mod._auction_pending_live(st, today) is True
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    assert runner_mod._auction_pending_live(st, today) is False
    st["auction_stage"]["pending"] = []
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: False)
    assert runner_mod._auction_pending_live(st, today) is False


def test_after_open_et_clock(monkeypatch):
    def at(h, m, d=15):
        class FakeDT(datetime):
            @classmethod
            def now(cls, tz=None):
                base = datetime(2026, 10, d, h, m, tzinfo=NY)
                return base.astimezone(tz) if tz else base.replace(tzinfo=None)
        monkeypatch.setattr(runner_mod, "datetime", FakeDT)
        return runner_mod._after_open_et()
    assert at(9, 30) is False and at(9, 31) is True and at(16, 0) is True and at(8, 0) is False
    assert at(10, 0, d=17) is False          # Saturday morning: not open, before "the close"


# ───────────────── driving run_single_model end to end ──────────────────

class FakeModel:
    def __init__(self, weights):
        self._w = weights
        self.last_diagnostics = {"exposure_multiplier": 1.0, "spy_gate": 1.0, "book_gate": 1.0}

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
        if rebalance_result is not None:
            return dict(rebalance_result)
        return {"dry_run": False, "auction": {"n_submitted": 2, "n_skipped": 0, "skipped": [],
                                               "submitted_notional_usd": 1000.0, "run_id": "RID",
                                               "pending": [{"order_id": "o1", "symbol": "AAA"}]},
                "run_id": "RID"}
    monkeypatch.setattr(runner_mod, "rebalance_portfolio", fake_rebalance)
    return calls


def _panel(dates, close=50.0):
    idx = pd.to_datetime(dates)
    return {"AAA": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx),
            "BBB": pd.DataFrame({"open": [close] * len(idx), "close": [close] * len(idx)}, index=idx)}


def test_stage1_submits_off_prior_session_close_and_carries_orphans(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev = runner_mod._previous_session_iso(today)
    calls = _wire_runner(monkeypatch, {"AAA": 0.5, "BBB": 0.5})
    finalized = []
    monkeypatch.setattr("core.orders.finalize_auction_orders",
                        lambda *a, **kw: finalized.append(1) or {"n": 0, "remaining": []})
    yday_rec = {"date": prev, "outcome": "submitted", "n_submitted": 1, "run_id": "OLD",
                "pending": [{"order_id": "left", "submitted_at": f"{prev}T13:06:00+00:00"}],
                "orphans": [{"order_id": "older", "submitted_at": f"{prev}T13:06:00+00:00"}]}
    save_state({"last_rebalance": None, "history": [], "auction_stage": yday_rec}, mc)
    # Panel: prior-session bar plus a today-dated pre-market row that must be ignored.
    panel = _panel([prev, today], close=50.0)
    panel["AAA"].loc[pd.Timestamp(today), "close"] = 999.0
    runner_mod.run_single_model(mc, panel, pd.DataFrame(), macro_data={}, stage="auction")
    st = load_state(mc)["auction_stage"]
    assert st["date"] == today and st["outcome"] == "submitted" and st["prev_session"] == prev
    assert calls[0]["auction"] is True and calls[0]["last_close"] == {"AAA": 50.0, "BBB": 50.0}
    assert calls[0]["stale_close"] == set() and st["n_stale_close"] == 0
    assert [o["order_id"] for o in st["orphans"]] == ["older", "left"]      # carried, NOT drained in the stage
    assert finalized == [] and load_state(mc).get("last_rebalance") is None
    assert st["plan"]["target_weights"] == {"AAA": 0.5, "BBB": 0.5}


def test_stage1_skips_on_a_stale_panel(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev = runner_mod._previous_session_iso(today)
    prev2 = runner_mod._previous_session_iso(prev)
    calls = _wire_runner(monkeypatch, {"AAA": 0.5, "BBB": 0.5})
    save_state({"last_rebalance": None, "history": []}, mc)
    runner_mod.run_single_model(mc, _panel([prev2]), pd.DataFrame(), macro_data={}, stage="auction")
    st = load_state(mc)["auction_stage"]
    assert st["outcome"] == "skipped:stale_panel" and st["n_submitted"] == 0 and calls == []


def test_stage1_flags_individual_stale_names(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    prev = runner_mod._previous_session_iso(today)
    prev2 = runner_mod._previous_session_iso(prev)
    calls = _wire_runner(monkeypatch, {"AAA": 0.5, "BBB": 0.5})
    save_state({"last_rebalance": None, "history": []}, mc)
    panel = _panel([prev2, prev])
    panel["BBB"] = panel["BBB"].iloc[:1]                     # BBB's last bar is two sessions old
    runner_mod.run_single_model(mc, panel, pd.DataFrame(), macro_data={}, stage="auction")
    assert calls[0]["stale_close"] == {"BBB"} and load_state(mc)["auction_stage"]["n_stale_close"] == 1


def _plan_state(today, pending=()):
    return {"last_rebalance": (datetime.now() - timedelta(days=1)).isoformat(),   # NOT due by count
            "history": [],
            "auction_stage": {"date": today, "outcome": "submitted", "n_submitted": 2, "run_id": "RID",
                              "plan": {"target_symbols": ["AAA", "BBB"], "rankings": [["AAA", 0.6], ["BBB", 0.4]],
                                       "target_weights": {"AAA": 0.6, "BBB": 0.4},
                                       "diagnostics": {"exposure_multiplier": 0.7}},
                              "pending": list(pending),
                              "summary": {"n": 2, "n_filled": 2, "submitted_notional_usd": 1000.0,
                                          "filled_notional_anchor_usd": 1000.0, "filled_notional_usd": 990.0}}}


def test_main_completes_stored_plan_without_any_panel(tmp_path, monkeypatch):
    """The plan-only path (shared data phase failed): empty panel, empty
    macro — the completion still runs on the stored plan, with the stored
    run_id and weights, and spends the plan."""
    mc = make_mc(tmp_path)
    today = _today_ny()
    result = {"dry_run": False, "reconciliation": {"flow": {"n_orders": 1, "n_filled": 1, "fill_rate_notional": 1.0,
                                                            "orders_notional_usd": 100.0, "filled_notional_usd": 100.0},
                                                   "book": None}}
    calls = _wire_runner(monkeypatch, {"ZZZ": 1.0}, rebalance_result=result)    # a re-plan would pick ZZZ
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: True)
    save_state(_plan_state(today), mc)
    runner_mod.run_single_model(mc, {}, pd.DataFrame(), macro_data={}, stage="main")
    assert len(calls) == 1 and calls[0]["run_id"] == "RID" and calls[0]["target_symbols"] == ["AAA", "BBB"]
    assert calls[0]["target_weights"] == {"AAA": 0.6, "BBB": 0.4} and not calls[0].get("auction")
    st = load_state(mc)
    assert st["auction_stage"]["completed_at"] and str(st["last_rebalance"])[:10] == datetime.now().strftime("%Y-%m-%d")
    assert "auction_completion_deferred" not in st
    assert st["history"][-1]["regime_exposure"] == pytest.approx(0.7)
    assert st["history"][-1]["strategy_diagnostics"]["exposure_multiplier"] == 0.7
    assert st["history"][-1]["result"]["reconciliation"]["auction"]["n"] == 2
    # Spent: a forced same-day re-run never reuses it.
    assert runner_mod._todays_auction_plan(st, today) is None


def test_main_defers_completion_while_orders_are_live_pre_open(tmp_path, monkeypatch):
    mc = make_mc(tmp_path)
    today = _today_ny()
    calls = _wire_runner(monkeypatch, {"AAA": 1.0})
    monkeypatch.setattr(runner_mod, "_after_open_et", lambda: False)
    monkeypatch.setattr("core.orders.finalize_auction_orders", lambda *a, **kw: {"n": 0, "remaining": []})
    before = _plan_state(today, pending=[{"order_id": "o1", "submitted_at": f"{today}T13:06:00+00:00"}])
    save_state(before, mc)
    runner_mod.run_single_model(mc, _panel([runner_mod._previous_session_iso(today)]), pd.DataFrame(),
                                macro_data={}, stage="main")
    st = load_state(mc)
    assert calls == [] and st["auction_completion_deferred"] == today
    assert st["last_rebalance"] == before["last_rebalance"] and "completed_at" not in st["auction_stage"]


# ───────────────────────── run_pipeline ──────────────────────────────────

def _fake_clock(monkeypatch, h, m, d=28):
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 10, d, h, m, tzinfo=NY)
            return base.astimezone(tz) if tz else base.replace(tzinfo=None)
    monkeypatch.setattr(runner_mod, "datetime", FakeDT)


def test_pipeline_runs_plan_only_slots_when_the_data_phase_fails(tmp_path, monkeypatch):
    a = make_mc(tmp_path / "a", exec_style="auction")
    b = make_mc(tmp_path / "b", name="combo_v2", exec_style="marketable_limit")
    today = _today_ny()
    save_state(_plan_state(today), a)
    save_state({"last_rebalance": None, "history": []}, b)
    monkeypatch.setattr(runner_mod, "get_active_models", lambda: [a, b])
    monkeypatch.setattr("core.invariants.run_invariant_checks", lambda lg: None)
    monkeypatch.setattr(runner_mod, "get_tradeable_symbols", lambda lg, rp: [])      # Wikipedia scrape failed
    ran = []
    monkeypatch.setattr(runner_mod, "run_single_model",
                        lambda mc, sd, mf, **kw: ran.append((mc.name, len(sd), kw.get("stage"))))
    runner_mod.run_pipeline(stage="main")
    assert ran == [("combo_v2_process", 0, "main")]


def test_pipeline_flags_a_failed_plan_only_completion_for_ready(tmp_path, monkeypatch):
    a = make_mc(tmp_path, exec_style="auction")
    today = _today_ny()
    save_state(_plan_state(today), a)
    monkeypatch.setattr(runner_mod, "get_active_models", lambda: [a])
    monkeypatch.setattr("core.invariants.run_invariant_checks", lambda lg: None)
    monkeypatch.setattr(runner_mod, "get_tradeable_symbols", lambda lg, rp: (_ for _ in ()).throw(RuntimeError("yf down")))

    def boom(mc, sd, mf, **kw):
        raise RuntimeError("alpaca down too")
    monkeypatch.setattr(runner_mod, "run_single_model", boom)
    with pytest.raises(RuntimeError):
        runner_mod.run_pipeline(stage="main")
    assert load_state(a)["auction_completion_deferred"] == today


def test_pipeline_raises_when_a_slot_fails(tmp_path, monkeypatch):
    a = make_mc(tmp_path / "a", exec_style="auction")
    b = make_mc(tmp_path / "b", name="combo_v2", exec_style="marketable_limit")
    monkeypatch.setattr(runner_mod, "get_active_models", lambda: [a, b])
    monkeypatch.setattr("core.invariants.run_invariant_checks", lambda lg: None)
    monkeypatch.setattr(runner_mod, "get_tradeable_symbols", lambda lg, rp: ["X"])
    monkeypatch.setattr(runner_mod, "download_bars", lambda syms, days, lg, rp: {"X": object()})
    monkeypatch.setattr(runner_mod, "download_macro", lambda days, lg, rp: {})
    monkeypatch.setattr(runner_mod, "compute_macro_features", lambda m: pd.DataFrame())
    ran = []

    def rsm(mc, *a_, **kw):
        ran.append(mc.name)
        if mc.name == "combo_v2_process":
            raise RuntimeError("positions call failed")
    monkeypatch.setattr(runner_mod, "run_single_model", rsm)
    with pytest.raises(RuntimeError, match="combo_v2_process"):
        runner_mod.run_pipeline(stage="main")
    assert ran == ["combo_v2_process", "combo_v2"]          # the other slot still ran


def test_auction_stage_no_op_creates_no_log_file_and_respects_cutoff(tmp_path, monkeypatch):
    a = make_mc(tmp_path, exec_style="auction")
    monkeypatch.setattr(runner_mod, "get_active_models", lambda: [a])
    monkeypatch.setattr("core.invariants.run_invariant_checks", lambda lg: None)
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: True)
    names = []
    monkeypatch.setattr(runner_mod, "setup_logging",
                        lambda name="main": names.append(name) or (logging.getLogger("t"), tmp_path / "x.log"))
    fetched = []
    monkeypatch.setattr(runner_mod, "get_tradeable_symbols", lambda lg, rp: fetched.append(1) or [])
    # 1. Not due -> no log file at all, no data phase.
    save_state({"last_rebalance": datetime.now().isoformat()}, a)
    _fake_clock(monkeypatch, 9, 8)
    runner_mod.run_pipeline(stage="auction")
    assert names == [] and fetched == []
    # 2. Due (unparsable last_rebalance is treated as due) -> 'auction' log + data phase.
    save_state({"last_rebalance": "garbage"}, a)
    runner_mod.run_pipeline(stage="auction")
    assert names == ["auction"] and fetched == [1]
    # 3. Due but the 09:20 start cutoff has passed -> skipped before the data phase.
    save_state({"last_rebalance": None}, a)
    _fake_clock(monkeypatch, 9, 21)
    runner_mod.run_pipeline(stage="auction")
    assert fetched == [1]


# ───────────────────────── dashboard ─────────────────────────────────────

def _dash_clock(monkeypatch, dashboard, h, m):
    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 10, 28, h, m, tzinfo=NY)
            return base.astimezone(tz) if tz else base.replace(tzinfo=None)
    monkeypatch.setattr(dashboard, "datetime", FakeDT)


def test_dashboard_main_run_waits_for_the_auction_stage_until_the_cutoff(monkeypatch):
    import dashboard
    seen = []
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", lambda **kw: seen.append(kw))
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {})
    slept = []

    def fake_sleep(s):
        slept.append(s)
        with dashboard.status_lock:
            dashboard.bot_status["state"] = "idle"
            dashboard.bot_status["running_stage"] = None
    monkeypatch.setattr(dashboard.time, "sleep", fake_sleep)
    _dash_clock(monkeypatch, dashboard, 9, 35)
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "running"
        dashboard.bot_status["running_stage"] = "auction"
        dashboard.bot_status.pop("skipped_main_run_at", None)
    dashboard.run_trading_pipeline(stage="main")
    with dashboard.status_lock:
        assert seen and seen[0]["stage"] == "main" and slept == [15]
        assert dashboard.bot_status.get("main_run_waited_s") == 15
        assert "skipped_main_run_at" not in dashboard.bot_status
        # After the cutoff the main run is dropped and flagged instead.
        dashboard.bot_status["state"] = "running"
        dashboard.bot_status["running_stage"] = "auction"
    _dash_clock(monkeypatch, dashboard, 15, 31)
    seen.clear(); slept.clear()
    dashboard.run_trading_pipeline(stage="main")
    with dashboard.status_lock:
        assert seen == [] and slept == []
        assert dashboard.bot_status["skipped_main_run_blocker"] == "auction"
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["running_stage"] = None
        dashboard.bot_status.pop("skipped_main_run_at", None)
        dashboard.bot_status.pop("skipped_main_run_blocker", None)


def test_dashboard_single_slot_manual_run_keeps_the_skipped_alarm(monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", lambda **kw: None)
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {})
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["models"]["combo_v2"] = {"state": "idle"}
        dashboard.bot_status["skipped_main_run_at"] = "2026-10-28T13:35:00+00:00"
    dashboard.run_trading_pipeline(stage="main", model_filter="combo_v2")
    with dashboard.status_lock:
        assert "skipped_main_run_at" in dashboard.bot_status          # two slots never ran
    dashboard.run_trading_pipeline(stage="main")
    with dashboard.status_lock:
        assert "skipped_main_run_at" not in dashboard.bot_status
        dashboard.bot_status["models"].pop("combo_v2", None)


def test_dashboard_auction_stage_yielding_to_a_manual_run_leaves_a_trace(monkeypatch):
    import dashboard
    monkeypatch.setattr(dashboard.pipeline, "run_pipeline", lambda **kw: None)
    with dashboard.status_lock:
        dashboard.bot_status["state"] = "running"
        dashboard.bot_status["running_stage"] = "main"
    dashboard.run_trading_pipeline(stage="auction")
    with dashboard.status_lock:
        assert dashboard.bot_status["auction_last_run_status"] == "skipped"
        assert "main" in dashboard.bot_status["auction_last_error"]
        dashboard.bot_status["state"] = "idle"
        dashboard.bot_status["running_stage"] = None
        dashboard.bot_status["auction_last_run_status"] = None
        dashboard.bot_status["auction_last_run_at"] = None


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
    for k in ("skipped_main_run_at", "skipped_main_run_blocker"):
        monkeypatch.delitem(dashboard.bot_status, k, raising=False)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", None)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_at", None)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_error", None)
    return dashboard, dashboard.app.test_client()


def test_ready_auction_alarms(tmp_path, monkeypatch):
    mc = make_mc(tmp_path, exec_style="auction")
    dashboard, c = _ready_client(monkeypatch, tmp_path, mc, {"last_rebalance": None})
    now = datetime.now(timezone.utc)
    # (a) stage errored today, no main run since -> 503
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", "error")
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_at", now.isoformat())
    monkeypatch.setitem(dashboard.bot_status, "auction_last_error", "boom")
    monkeypatch.setitem(dashboard.bot_status, "last_run_at", (now - timedelta(hours=1)).isoformat())
    r = c.get("/ready")
    assert r.status_code == 503 and any("auction stage errored" in p for p in r.get_json()["problems"])
    # (b) a later successful main run demotes it to a warning
    monkeypatch.setitem(dashboard.bot_status, "last_run_at", (now + timedelta(minutes=1)).isoformat())
    r = c.get("/ready")
    assert r.status_code == 200 and any("auction stage errored" in w for w in r.get_json()["warnings"])
    # (c) a stage skipped today while an auction slot was due (last_rebalance None) -> 503
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", "skipped")
    monkeypatch.setitem(dashboard.bot_status, "last_run_at", (now - timedelta(hours=1)).isoformat())
    r = c.get("/ready")
    assert r.status_code == 503 and any("auction stage skipped" in p for p in r.get_json()["problems"])
    # (d) same, but no slot was due -> warning only
    monkeypatch.setattr(dashboard, "_load_model_state",
                        lambda name: {"last_rebalance": datetime.now().isoformat()})
    r = c.get("/ready")
    assert r.status_code == 200 and any("no auction slot was due" in w for w in r.get_json()["warnings"])
    # (e) ET session date: a stamp from 02:00 UTC today is YESTERDAY evening in New York -> ignored
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", "error")
    stamp = datetime.now(NY).replace(hour=22, minute=0, second=0, microsecond=0) - timedelta(days=1)
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_at", stamp.astimezone(timezone.utc).isoformat())
    r = c.get("/ready")
    assert r.status_code == 200 and not any("auction stage" in p for p in r.get_json()["problems"])
    # (f) deferred completion today -> 503
    monkeypatch.setitem(dashboard.bot_status, "auction_last_run_status", None)
    monkeypatch.setattr(dashboard, "_load_model_state",
                        lambda name: {"auction_completion_deferred": _today_ny()})
    r = c.get("/ready")
    assert r.status_code == 503 and any("completion deferred" in p for p in r.get_json()["problems"])


def test_run_endpoint_refuses_manual_runs_in_the_submission_window_when_due(tmp_path, monkeypatch):
    import dashboard
    mc = make_mc(tmp_path, exec_style="auction")
    monkeypatch.setattr(dashboard.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: True)
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {"last_rebalance": None})
    assert dashboard._auction_window_block_reason(dry=False)
    assert dashboard._auction_window_block_reason(dry=True) is None
    monkeypatch.setattr(dashboard, "_load_model_state",
                        lambda name: {"last_rebalance": None,
                                      "auction_stage": {"date": _today_ny(), "outcome": "submitted"}})
    assert dashboard._auction_window_block_reason(dry=False) is None     # already submitted
    monkeypatch.setattr("core.market.is_auction_window", lambda now=None: False)
    monkeypatch.setattr(dashboard, "_load_model_state", lambda name: {"last_rebalance": None})
    assert dashboard._auction_window_block_reason(dry=False) is None     # outside the window
