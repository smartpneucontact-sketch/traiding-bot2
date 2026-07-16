"""Unit tests for Phase A1 measurement endpoints (read-only analytics).

Covers:
  - core.execution_stats.compute_slippage_stats: synthetic journals with
    mixed measured/unmeasured rows, buy/sell sign handling, hand-checked
    notional weighting, vs-arrival view, per-run grouping, drag arithmetic
    (+10 bp/side ~= -0.16 pp/mo at ~1.6x equity traded/mo)
  - core.orders.rebalance_portfolio reconciliation block: flow counts from
    synthetic order results (filled + rejected + unconfirmed), book-level
    target-vs-achieved from a stubbed second get_positions call, and the
    non-fatal contract when that call fails
  - dashboard._compute_gate_state: freeze pill + cutloss tiers from a
    synthetic state dict with an injected equity stub (no network)

Run:  /opt/anaconda3/bin/python -m pytest tests/ -v
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import core.orders as orders_mod
from core.config import ModelConfig
from core.execution_stats import compute_slippage_stats
from core.run_report import RunReport

logger = logging.getLogger("test")


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


def make_trade(**kw) -> dict:
    """A journal row as TradeJournal.get_trades() returns it."""
    row = dict(
        trade_id="t", run_id="r1", model="combo_v2",
        timestamp="2026-07-15T13:35:00+00:00", symbol="AAA", side="buy",
        action="new_position", order_type="market", time_in_force="day",
        notional_usd=1000.0, order_status="filled",
        slippage_bps=None, slippage_vs_arrival_bps=None,
    )
    row.update(kw)
    return row


# ───── compute_slippage_stats ────────────────────────────────────────────

def test_slippage_stats_hand_checked_notional_weighting():
    """4 measured rows + 1 unmeasured. Weighted mean vs open:
    (10*1000 + 20*3000 - 10*1000 + 30*1000) / 6000 = 15.0 bps.
    vs-arrival covers only 3 rows: (4*1000 + 8*3000 - 2*1000) / 5000 = 5.2.
    """
    trades = [
        make_trade(trade_id="b1", side="buy", notional_usd=1000.0,
                   slippage_bps=10.0, slippage_vs_arrival_bps=4.0),
        make_trade(trade_id="b2", side="buy", notional_usd=3000.0,
                   slippage_bps=20.0, slippage_vs_arrival_bps=8.0),
        make_trade(trade_id="s1", side="sell", notional_usd=1000.0,
                   slippage_bps=-10.0, slippage_vs_arrival_bps=-2.0),
        make_trade(trade_id="s2", side="sell", notional_usd=1000.0,
                   slippage_bps=30.0, slippage_vs_arrival_bps=None),
        make_trade(trade_id="u1", side="buy", notional_usd=500.0,
                   slippage_bps=None),  # pre-A0 row: unmeasured, never zero
    ]
    stats = compute_slippage_stats(trades)

    assert stats["n_trades"] == 5
    assert stats["n_measured"] == 4
    assert stats["n_unmeasured"] == 1

    buy = stats["by_side"]["buy"]
    assert buy["n"] == 2
    assert buy["mean"] == pytest.approx(15.0)
    assert buy["median"] == pytest.approx(15.0)
    assert buy["p75"] == pytest.approx(17.5)   # 10 + 0.75*(20-10)
    assert buy["p95"] == pytest.approx(19.5)
    sell = stats["by_side"]["sell"]
    assert sell["n"] == 2
    assert sell["mean"] == pytest.approx(10.0)

    assert stats["notional_weighted_mean_bps"] == pytest.approx(15.0)
    assert stats["calibrated_cost_bps_per_side"] == pytest.approx(15.0)
    assert stats["research_assumption_bps"] == 5.0

    va = stats["vs_arrival"]
    assert va["n_measured"] == 3
    assert va["notional_weighted_mean_bps"] == pytest.approx(5.2)
    assert va["by_side"]["buy"]["n"] == 2
    assert va["by_side"]["sell"]["n"] == 1

    # Drag: 15 bp/side * 1.6x equity traded/mo = -0.24 pp/mo
    assert stats["monthly_drag_estimate_pp"] == pytest.approx(-0.24)


def test_slippage_drag_arithmetic_ten_bps_is_016pp():
    """The documented anchor: +10 bp/side ~= -0.16 pp/mo at 1.6x/mo."""
    stats = compute_slippage_stats([make_trade(slippage_bps=10.0)])
    assert stats["calibrated_cost_bps_per_side"] == pytest.approx(10.0)
    assert stats["monthly_drag_estimate_pp"] == pytest.approx(-0.16)


def test_slippage_stats_all_unmeasured_yields_none_not_zero():
    """A journal with no captured slippage must NOT report 0 bps cost."""
    trades = [make_trade(trade_id=f"t{i}", slippage_bps=None)
              for i in range(3)]
    stats = compute_slippage_stats(trades)
    assert stats["n_measured"] == 0
    assert stats["n_unmeasured"] == 3
    assert stats["notional_weighted_mean_bps"] is None
    assert stats["calibrated_cost_bps_per_side"] is None
    assert stats["monthly_drag_estimate_pp"] is None
    for side in ("buy", "sell"):
        assert stats["by_side"][side] == {
            "n": 0, "mean": None, "median": None, "p75": None, "p95": None}
    assert stats["vs_arrival"]["n_measured"] == 0
    # Runs still enumerated, with explicit unmeasured counts
    assert stats["by_run"][0]["n_orders"] == 3
    assert stats["by_run"][0]["n_measured"] == 0
    assert stats["by_run"][0]["notional_weighted_mean_bps"] is None


def test_slippage_stats_empty_journal():
    stats = compute_slippage_stats([])
    assert stats["n_trades"] == 0
    assert stats["n_measured"] == 0
    assert stats["calibrated_cost_bps_per_side"] is None
    assert stats["by_run"] == []


def test_slippage_stats_by_run_grouping():
    trades = [
        make_trade(trade_id="a", run_id="r1", notional_usd=1000.0,
                   slippage_bps=10.0, timestamp="2026-07-01T13:35:00+00:00"),
        make_trade(trade_id="b", run_id="r1", notional_usd=1000.0,
                   slippage_bps=None, timestamp="2026-07-01T13:35:01+00:00"),
        make_trade(trade_id="c", run_id="r2", notional_usd=1000.0,
                   slippage_bps=20.0, timestamp="2026-07-02T13:35:00+00:00"),
        make_trade(trade_id="d", run_id="r2", notional_usd=3000.0,
                   slippage_bps=40.0, timestamp="2026-07-02T13:35:01+00:00"),
    ]
    by_run = compute_slippage_stats(trades)["by_run"]
    assert [r["run_id"] for r in by_run] == ["r1", "r2"]
    r1, r2 = by_run
    assert r1["n_orders"] == 2 and r1["n_measured"] == 1
    assert r1["notional_weighted_mean_bps"] == pytest.approx(10.0)
    assert r1["timestamp"] == "2026-07-01T13:35:00+00:00"
    assert r2["n_orders"] == 2 and r2["n_measured"] == 2
    # (20*1000 + 40*3000) / 4000 = 35.0
    assert r2["notional_weighted_mean_bps"] == pytest.approx(35.0)
    assert r2["median_bps"] == pytest.approx(30.0)


# ───── rebalance reconciliation ──────────────────────────────────────────

class StubJournal:
    def __init__(self):
        self.records = []
        self.jsonl_path = Path("/dev/null")

    def log_trade(self, record):
        self.records.append(record)


def _run_recon_rebalance(tmp_path, monkeypatch, positions_after,
                         after_raises=False):
    """Drive rebalance_portfolio through 4 synthetic order outcomes:
      AAA exit sell  -> filled ($1,000)
      BBB new buy    -> filled ($10,000)
      DDD new buy    -> rejected at submit ($10,000)
      EEE new buy    -> accepted but never confirmed filled ($10,000)
    CCC is held (drift < 10%, no order). Targets: 4 names x $10,000.
    The SECOND get_positions call feeds the book-level reconciliation.
    """
    mc = make_mc(tmp_path)
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": "100000", "cash": "100000",
        "buying_power": "400000",
    })

    pos_calls = {"n": 0}
    positions_before = {
        "AAA": {"qty": 10.0, "market_value": 1000.0, "avg_entry": 90.0,
                "current_price": 100.0, "unrealized_pl": 100.0,
                "unrealized_pl_pct": 11.1},
        "CCC": {"qty": 100.0, "market_value": 10500.0, "avg_entry": 100.0,
                "current_price": 105.0, "unrealized_pl": 500.0,
                "unrealized_pl_pct": 5.0},
    }

    def fake_get_positions(mc_, lg):
        pos_calls["n"] += 1
        if pos_calls["n"] == 1:
            return positions_before
        if after_raises:
            raise RuntimeError("positions endpoint down")
        return positions_after

    monkeypatch.setattr(orders_mod, "get_positions", fake_get_positions)

    def fake_alpaca(method, path, mc_, data=None, logger=None):
        if method == "DELETE" and path == "v2/positions/AAA":
            return {"id": "oid_aaa", "status": "accepted"}
        if method == "POST" and path == "v2/orders":
            sym = data["symbol"]
            if sym == "DDD":
                return {"status": "rejected",
                        "reject_reason": "insufficient buying power"}
            return {"id": f"oid_{sym.lower()}", "status": "accepted"}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(orders_mod, "alpaca_request", fake_alpaca)

    def fake_poll(order_id, mc_, lg):
        if order_id == "oid_eee":
            return {"status": "accepted"}  # never reaches terminal state
        return {"status": "filled", "filled_qty": "10",
                "filled_avg_price": "100.0"}

    monkeypatch.setattr(orders_mod, "poll_order_status", fake_poll)
    monkeypatch.setattr(orders_mod, "fetch_snapshots",
                        lambda syms, mc_, lg: {})
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)

    weights = {"BBB": 0.05, "CCC": 0.05, "DDD": 0.05, "EEE": 0.05}
    rb = orders_mod.rebalance_portfolio(
        target_symbols=list(weights), rankings=[(s, w) for s, w in weights.items()],
        mc=mc, journal=StubJournal(), logger=logger, report=RunReport(),
        dry_run=False, target_weights=weights,
    )
    assert pos_calls["n"] == 2  # exactly ONE extra call for the book check
    return rb


def test_reconciliation_flow_counts_rejected_and_unconfirmed(tmp_path, monkeypatch):
    rb = _run_recon_rebalance(tmp_path, monkeypatch, positions_after={})
    flow = rb["reconciliation"]["flow"]

    assert flow["n_orders"] == 4
    assert flow["n_filled"] == 2          # AAA exit + BBB buy
    assert flow["n_rejected"] == 1        # DDD rejected at submit
    assert flow["n_unconfirmed"] == 1     # EEE accepted, never filled
    assert flow["rejected_symbols"] == ["DDD"]
    assert flow["unconfirmed_symbols"] == ["EEE"]
    # filled 1,000 + 10,000 of 31,000 total order notional
    assert flow["fill_rate_notional"] == pytest.approx(11000 / 31000, abs=1e-4)


def test_reconciliation_book_targets_vs_achieved(tmp_path, monkeypatch):
    positions_after = {
        "BBB": {"qty": 100.0, "market_value": 9990.0, "avg_entry": 100.0,
                "current_price": 99.9, "unrealized_pl": -10.0,
                "unrealized_pl_pct": -0.1},
        "CCC": {"qty": 100.0, "market_value": 10500.0, "avg_entry": 100.0,
                "current_price": 105.0, "unrealized_pl": 500.0,
                "unrealized_pl_pct": 5.0},
        "ZZZ": {"qty": 5.0, "market_value": 500.0, "avg_entry": 100.0,
                "current_price": 100.0, "unrealized_pl": 0.0,
                "unrealized_pl_pct": 0.0},
    }
    rb = _run_recon_rebalance(tmp_path, monkeypatch, positions_after)
    book = rb["reconciliation"]["book"]

    # Targets: 4 names x $10,000 (0.05 x $100k x 2x leverage)
    assert book["target_gross_usd"] == pytest.approx(40000.0)
    assert book["achieved_gross_usd"] == pytest.approx(20990.0)
    assert book["gross_achieved_pct_of_target"] == pytest.approx(52.48, abs=0.01)
    assert book["target_leverage"] == pytest.approx(0.4)   # realized_leverage
    assert book["achieved_leverage"] == pytest.approx(0.2099)
    assert book["positions_after"] == 3

    # Rejected/unconfirmed names missing from the book; stray ZZZ flagged
    assert book["missing_positions"] == ["DDD", "EEE"]
    assert book["unexpected_positions"] == ["ZZZ"]

    devs = book["top_deviations"]
    assert len(devs) == 5
    # The two $10k shortfalls dominate
    assert {devs[0]["symbol"], devs[1]["symbol"]} == {"DDD", "EEE"}
    assert devs[0]["deviation_usd"] == pytest.approx(-10000.0)
    by_sym = {d["symbol"]: d for d in devs}
    assert by_sym["CCC"]["deviation_usd"] == pytest.approx(500.0)
    assert by_sym["ZZZ"]["deviation_usd"] == pytest.approx(500.0)
    assert by_sym["BBB"]["deviation_usd"] == pytest.approx(-10.0)


def test_reconciliation_book_failure_is_nonfatal(tmp_path, monkeypatch):
    """A failing get_positions must not kill the rebalance return path —
    flow-level reconciliation survives, book is None."""
    rb = _run_recon_rebalance(tmp_path, monkeypatch, positions_after={},
                              after_raises=True)
    assert rb["executed"] == 3 and rb["failed"] == 1
    recon = rb["reconciliation"]
    assert recon["flow"]["n_orders"] == 4
    assert recon["book"] is None


def test_reconciliation_lands_in_run_report_summary(tmp_path, monkeypatch):
    """format_summary renders the RECONCILIATION block from rb_data."""
    positions_after = {
        "BBB": {"qty": 100.0, "market_value": 9990.0, "avg_entry": 100.0,
                "current_price": 99.9, "unrealized_pl": -10.0,
                "unrealized_pl_pct": -0.1},
    }
    mc = make_mc(tmp_path)
    report = RunReport()
    # Reuse the full rebalance driver but with our own report object
    monkeypatch.setattr(orders_mod, "get_account", lambda mc_, lg, rp: {
        "portfolio_value": "100000", "cash": "100000",
        "buying_power": "400000"})
    calls = {"n": 0}

    def fake_pos(mc_, lg):
        calls["n"] += 1
        return {} if calls["n"] == 1 else positions_after

    monkeypatch.setattr(orders_mod, "get_positions", fake_pos)
    monkeypatch.setattr(orders_mod, "alpaca_request",
                        lambda m, p, mc_, data=None, logger=None:
                        {"id": "oid_bbb", "status": "accepted"})
    monkeypatch.setattr(orders_mod, "poll_order_status",
                        lambda oid, mc_, lg: {"status": "filled",
                                              "filled_qty": "100",
                                              "filled_avg_price": "99.9"})
    monkeypatch.setattr(orders_mod, "fetch_snapshots", lambda s, m, l: {})
    monkeypatch.setattr(orders_mod.time, "sleep", lambda s: None)

    orders_mod.rebalance_portfolio(
        target_symbols=["BBB"], rankings=[("BBB", 1.0)],
        mc=mc, journal=StubJournal(), logger=logger, report=report,
        dry_run=False, target_weights={"BBB": 0.05},
    )
    summary = report.format_summary()
    assert "RECONCILIATION:" in summary
    assert "1 filled, 0 rejected, 0 unconfirmed" in summary
    assert "Fill rate (notional):  100.0%" in summary
    assert "Achieved gross:        $9,990.00" in summary


# ───── dashboard._compute_gate_state (no network) ────────────────────────

@pytest.fixture(scope="module")
def dash():
    import dashboard
    return dashboard


TODAY = datetime.now().strftime("%Y-%m-%d")


def _cutloss_state(tmp_path, dash, equity, start=100000.0):
    """Tier from a synthetic state via an injected equity stub."""
    mc = make_mc(tmp_path, enable_cutloss=True,
                 cutloss_portfolio_stop=-4.0, target_leverage=2.0)
    # Effective pstop = -4.0 x 2x leverage = -8.0% equity drawdown
    state = {
        "daily_portfolio_start_date": TODAY,
        "daily_portfolio_start": start,
    }
    return dash._compute_gate_state(mc, state, fetch_equity=lambda: equity)


def test_gate_state_tiers_from_synthetic_equity(tmp_path, dash):
    # dd -5% > tier1 (-8%) -> normal, but dd still reported
    g = _cutloss_state(tmp_path, dash, equity=95000.0)
    assert g["cutloss_state"] == "normal"
    assert g["cutloss_daily_dd_pct"] == pytest.approx(-5.0)
    # dd -9% <= -8% -> tier1
    assert _cutloss_state(tmp_path, dash, 91000.0)["cutloss_state"] == "tier1"
    # dd -14% <= -13.33% -> tier2
    assert _cutloss_state(tmp_path, dash, 86000.0)["cutloss_state"] == "tier2"
    # dd -19% <= -18.67% -> tier3
    assert _cutloss_state(tmp_path, dash, 81000.0)["cutloss_state"] == "tier3"


def test_gate_state_tripped_flag_and_freeze(tmp_path, dash):
    mc = make_mc(tmp_path, enable_cutloss=True)
    state = {
        "portfolio_stop_tripped_date": TODAY,
        "freeze_state": {"active": True, "since_iso": "2026-07-10"},
        # stale anchor date -> live tier read must NOT fire
        "daily_portfolio_start_date": "2000-01-01",
        "daily_portfolio_start": 100000.0,
    }

    def boom():
        raise AssertionError("no equity fetch expected on stale anchor")

    g = dash._compute_gate_state(mc, state, fetch_equity=boom)
    assert g["cutloss_state"] == "tripped"
    assert g["freeze_state"] == "frozen"
    assert g["freeze_since"] == "2026-07-10"
    assert "cutloss_daily_dd_pct" not in g


def test_gate_state_inactive_model_and_defaults(tmp_path, dash):
    # mc=None (model not assigned to a slot): state-only answer, no network
    g = dash._compute_gate_state(None, {}, fetch_equity=None)
    assert g == {"freeze_state": "normal", "cutloss_state": "normal"}


def test_gate_state_equity_fetch_failure_is_swallowed(tmp_path, dash):
    mc = make_mc(tmp_path, enable_cutloss=True)
    state = {
        "daily_portfolio_start_date": TODAY,
        "daily_portfolio_start": 100000.0,
    }

    def broken():
        raise RuntimeError("alpaca down")

    g = dash._compute_gate_state(mc, state, fetch_equity=broken)
    assert g["cutloss_state"] == "normal"
    assert "cutloss_daily_dd_pct" not in g


def test_gate_state_cutloss_disabled_skips_live_read(tmp_path, dash):
    mc = make_mc(tmp_path, enable_cutloss=False)
    state = {
        "daily_portfolio_start_date": TODAY,
        "daily_portfolio_start": 100000.0,
    }

    def boom():
        raise AssertionError("no equity fetch expected when cutloss disabled")

    g = dash._compute_gate_state(mc, state, fetch_equity=boom)
    assert g["cutloss_state"] == "normal"
