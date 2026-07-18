"""Unit tests for Phase A2 — tracking, sleeve attribution, protocol scoring.

Covers:
  - core.tracking.compute_tracking: a synthetic equity curve compounding
    exactly 3.68%/mo tracks the reference with ~0 diff and sits inside the
    ±1σ band; the sigma derivation reproduces the pre-registered ~0.157;
    clean not-started / insufficient-data states; padding trim
  - core.attribution.compute_attribution: the accounting invariant
    Σ sleeve contributions + residual ≈ live book return on synthetic
    data; data gaps degrade to a partial result, never an exception
  - core.protocol.score_protocol: everything NA before forward_test_start,
    KILL on a −46% drawdown, MODIFIED_AFTER_START on hash mismatch,
    K3a/K3b twice-consecutive collapse, M6-BAND AND-conditions
  - dashboard endpoints /api/tracking|sleeves|protocol return 200 with a
    clean payload on the empty / not-started state (no Alpaca keys)

Run:  /opt/anaconda3/bin/python -m pytest tests/test_a2.py -v
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_a2_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.attribution import compute_attribution
from core.protocol import load_protocol, score_protocol
from core.run_report import RunReport
from core.tracking import (
    TRADING_DAYS_PER_MONTH, compute_tracking, derive_sigma_monthly,
    trim_equity_padding,
)

REF = {
    "geo_monthly_return": 0.0368,
    "mean_monthly_return": 0.0480,
    "sharpe": 1.06,
    "max_drawdown": -0.603,
}

START = datetime(2026, 7, 20, tzinfo=timezone.utc)  # a Monday


def _pts(equities, start=START):
    """(ts_ms, equity) points on consecutive days from `start`."""
    return [(int((start + timedelta(days=i)).timestamp() * 1000), e)
            for i, e in enumerate(equities)]


# ───── compute_tracking ──────────────────────────────────────────────────

def test_sigma_derivation_matches_preregistered_0157():
    """σ_m = mean_m·√12 / Sharpe = 0.048 × 3.464 / 1.06 ≈ 0.157 — the
    number written into protocol.json / PROTOCOL.md."""
    sigma = derive_sigma_monthly(REF)
    assert sigma == pytest.approx(0.157, abs=0.001)
    assert derive_sigma_monthly({}) is None
    assert derive_sigma_monthly({"mean_monthly_return": 0.05, "sharpe": 0}) is None


def test_tracking_on_exact_reference_curve_has_zero_diff():
    """Equity compounding exactly 3.68%/mo → live geo == expected, cum
    tracking diff ≈ 0, and the curve sits inside the ±1σ band."""
    g_daily = (1.0368) ** (1.0 / TRADING_DAYS_PER_MONTH)
    n = 126  # 6 "months" of trading days
    equities = [100_000.0 * g_daily ** i for i in range(n + 1)]
    tr = compute_tracking(_pts(equities), REF, "2026-07-20")

    assert tr["status"] == "ok"
    assert tr["n_days"] == n
    assert tr["months_elapsed"] == pytest.approx(6.0)
    assert tr["live_geo_monthly"] == pytest.approx(0.0368, abs=1e-9)
    assert tr["expected_geo_monthly"] == 0.0368
    assert tr["cum_tracking_diff_pp"] == pytest.approx(0.0, abs=1e-6)
    assert tr["within_1sigma"] is True
    assert tr["cum_log_return"] == pytest.approx(6 * math.log(1.0368), abs=1e-6)
    # Band matches the M6-BAND arithmetic: expected ± σ·√6
    band = tr["band_1sigma_log"]
    sigma = derive_sigma_monthly(REF)
    assert band["hi"] - band["lo"] == pytest.approx(2 * sigma * math.sqrt(6), abs=1e-4)
    # Smooth exponential curve: no drawdown, ~zero realized vol
    assert tr["max_drawdown_pct"] == pytest.approx(0.0)
    assert tr["realized_vol_monthly_pct"] == pytest.approx(0.0, abs=1e-6)
    # SPY series not supplied → context fields null, not fabricated
    assert tr["spy"]["cum_return_pct"] is None
    assert tr["spy"]["live_minus_2x_spy_pp"] is None
    # Path arrays aligned for charting
    p = tr["path"]
    assert (len(p["dates"]) == len(p["cum_log_live"])
            == len(p["cum_log_expected"]) == len(p["band_lo"]) == n + 1)


def test_tracking_flat_account_falls_below_expected_band_at_6mo():
    """A flat (0%/mo) account after 6 months: cum diff ≈ −(1.0368^6−1) and
    it drops OUT of the 1σ band (expected 0.2168, half-width 0.157·√6≈0.384
    → lo ≈ −0.168 > 0? no — 0.2168−0.3845 = −0.1677 < 0 = live). So a flat
    account is still *inside* at 6mo; check the numbers instead."""
    n = 126
    tr = compute_tracking(_pts([100_000.0] * (n + 1)), REF, "2026-07-20")
    assert tr["live_geo_monthly"] == pytest.approx(0.0)
    expected_cum_pct = (1.0368 ** 6 - 1) * 100
    assert tr["cum_tracking_diff_pp"] == pytest.approx(-expected_cum_pct, abs=1e-4)
    assert tr["band_1sigma_log"]["lo"] == pytest.approx(
        6 * math.log(1.0368) - derive_sigma_monthly(REF) * math.sqrt(6), abs=1e-4)
    assert tr["within_1sigma"] is True  # power disclosure in action


def test_tracking_spy_context_when_series_supplied():
    n = 21
    g = 1.05 ** (1 / 21)
    equities = [100_000.0 * g ** i for i in range(n + 1)]
    spy = _pts([100.0 * (1.02 ** (i / 21)) for i in range(n + 1)])
    tr = compute_tracking(_pts(equities), REF, "2026-07-20", spy_points=spy)
    assert tr["spy"]["cum_return_pct"] == pytest.approx(2.0, abs=1e-6)
    assert tr["spy"]["live_minus_2x_spy_pp"] == pytest.approx(5.0 - 4.0, abs=1e-4)


def test_tracking_not_started_and_insufficient_data():
    tr = compute_tracking([], REF, None)
    assert tr["status"] == "not_started"
    assert tr["forward_test_start"] is None
    assert tr["sigma_monthly_pct"] == pytest.approx(15.68, abs=0.05)

    tr = compute_tracking(_pts([100_000.0]), REF, "2026-07-20")
    assert tr["status"] == "insufficient_data"

    # Points before the start date are excluded from the window
    pre = _pts([90_000.0, 95_000.0], start=START - timedelta(days=30))
    tr = compute_tracking(pre, REF, "2026-07-20")
    assert tr["status"] == "insufficient_data"


def test_trim_equity_padding_drops_leading_unfunded_days():
    ts = [1000, 1001, 1002, 1003, 1004]
    eq = [0, None, 100000.0, 0, 101000.0]
    out_ts, out_eq = trim_equity_padding(ts, eq)
    # Leading 0/None dropped; interior zero kept (downstream math skips it)
    assert out_ts == [1002000, 1003000, 1004000]
    assert out_eq == [100000.0, 0.0, 101000.0]


def test_run_report_renders_live_vs_backtest_block():
    report = RunReport()
    g_daily = (1.0368) ** (1.0 / 21)
    equities = [100_000.0 * g_daily ** i for i in range(43)]
    report.set("tracking", compute_tracking(_pts(equities), REF, "2026-07-20"))
    summary = report.format_summary()
    assert "LIVE VS BACKTEST:" in summary
    assert "+3.68%" in summary
    assert "Within +/-1 sigma band: yes" in summary


# ───── compute_attribution ───────────────────────────────────────────────

def _synthetic_book():
    """One sleeved rebalance, 3 symbols, 3 trading days of closes with
    hand-computable returns, equity consistent with the modeled book."""
    weights = {"AAA": 0.30, "BBB": 0.20, "CCC": 0.10}
    sleeves = {
        "xs_momentum":   {"AAA": 0.20, "BBB": 0.20},
        "dual_momentum": {"AAA": 0.10, "CCC": 0.10},
        "adaptive":      {},
    }
    history = [{
        "date": "2026-07-20T09:35:00",
        "weights": weights,
        "strategy_diagnostics": {"sleeves": sleeves},
        # Funded rebalance marker: orders actually reached Alpaca
        # (core/orders.py writes `executed` only on the live path).
        "result": {"dry_run": False, "n_buys": 3, "n_sells": 0,
                   "executed": 3, "failed": 0},
    }]
    dates = ["2026-07-20", "2026-07-21", "2026-07-22", "2026-07-23"]
    closes = {
        "AAA": dict(zip(dates, [100.0, 102.0, 101.0, 103.0])),
        "BBB": dict(zip(dates, [50.0, 50.5, 51.0, 50.0])),
        "CCC": dict(zip(dates, [200.0, 198.0, 202.0, 202.0])),
    }
    leverage = 2.0
    # Equity path = exactly the modeled book return each day → residual ≈ 0
    equity = {dates[0]: 100_000.0}
    for prev_d, cur_d in zip(dates, dates[1:]):
        r_book = sum(
            w * leverage * (closes[s][cur_d] / closes[s][prev_d] - 1.0)
            for s, w in weights.items())
        equity[cur_d] = equity[prev_d] * (1.0 + r_book)
    return history, closes, equity, leverage


def test_attribution_invariant_sleeves_plus_residual_equals_live():
    history, closes, equity, lev = _synthetic_book()
    out = compute_attribution(history, closes, equity, leverage=lev)

    assert out["status"] == "ok"
    assert out["n_days"] == 3
    assert out["n_live_days"] == 3
    sleeves_sum = sum(v["cum_contribution_pp"] for v in out["sleeves"].values())
    # Live equity == modeled book → residual ~0 and Σ sleeves == book == live
    assert out["unattributed_residual_pp"] == pytest.approx(0.0, abs=1e-6)
    assert sleeves_sum == pytest.approx(out["book_cum_pp"], abs=1e-6)
    assert sleeves_sum + out["unattributed_residual_pp"] == pytest.approx(
        out["live_cum_pp"], abs=1e-6)
    # AAA splits 2/3 xs, 1/3 dual; BBB fully xs; CCC fully dual — the empty
    # adaptive sleeve contributes exactly zero.
    assert out["sleeves"]["adaptive"]["cum_contribution_pp"] == 0.0
    day1 = out["daily_tail"][0]
    assert day1["date"] == "2026-07-21"
    # Hand check day 1: AAA +2%, BBB +1%, CCC −1% at 2x leverage
    r_aaa, r_bbb, r_ccc = 0.02, 0.01, -0.01
    xs_pp = (0.30 * 2 * r_aaa * (2 / 3) + 0.20 * 2 * r_bbb) * 100
    dual_pp = (0.30 * 2 * r_aaa * (1 / 3) + 0.10 * 2 * r_ccc) * 100
    assert day1["sleeves_pp"]["xs_momentum"] == pytest.approx(xs_pp, abs=1e-4)
    assert day1["sleeves_pp"]["dual_momentum"] == pytest.approx(dual_pp, abs=1e-4)


def test_attribution_residual_catches_costs():
    """Equity lagging the modeled book (costs) shows up as a NEGATIVE
    residual — never smeared into the sleeves."""
    history, closes, equity, lev = _synthetic_book()
    lagged = {d: e * (1.0 - 0.001) ** i
              for i, (d, e) in enumerate(sorted(equity.items()))}
    out = compute_attribution(history, closes, lagged, leverage=lev)
    sleeves_sum = sum(v["cum_contribution_pp"] for v in out["sleeves"].values())
    assert out["unattributed_residual_pp"] < -0.25  # ~3 × −0.1 pp/day
    assert sleeves_sum + out["unattributed_residual_pp"] == pytest.approx(
        out["live_cum_pp"], abs=1e-6)
    # Sleeve numbers identical to the cost-free case: costs → residual only
    clean = compute_attribution(history, closes, equity, leverage=lev)
    for name in out["sleeves"]:
        assert out["sleeves"][name]["cum_contribution_pp"] == pytest.approx(
            clean["sleeves"][name]["cum_contribution_pp"], abs=1e-9)


def test_attribution_partial_data_lists_gaps_never_raises():
    history, closes, equity, lev = _synthetic_book()
    del closes["CCC"]                     # symbol with no bars at all
    del closes["BBB"]["2026-07-22"]       # one missing close
    out = compute_attribution(history, closes, equity, leverage=lev)
    assert out["status"] == "ok"
    gaps = " | ".join(out["data_gaps"])
    assert "CCC: no close data" in gaps
    assert "BBB: missing close" in gaps
    # No equity at all → residual is explicitly None, not zero
    out2 = compute_attribution(history, closes, None, leverage=lev)
    assert out2["unattributed_residual_pp"] is None
    assert out2["live_cum_pp"] is None


def test_attribution_not_started_without_sleeved_history():
    out = compute_attribution([], {}, {}, leverage=2.0)
    assert out["status"] == "not_started"
    out = compute_attribution(
        [{"date": "2026-07-01", "weights": {"AAA": 1.0}}], {}, {})
    assert out["status"] == "not_started"  # pre-A0 entry: no sleeves diag


def test_attribution_window_starts_at_funded_not_dry_run():
    """The runbook prescribes a dry-run smoke BEFORE the first funded
    rebalance; that entry carries sleeves diagnostics too but placed no
    orders. The attribution window must start at the FUNDED entry, and
    skipped / all-orders-failed entries must never anchor or segment it."""
    history, closes, equity, lev = _synthetic_book()
    funded = history[0]  # dated 2026-07-20, executed=3
    dry = {
        "date": "2026-07-15T14:05:00",
        "weights": dict(funded["weights"]),
        "strategy_diagnostics": funded["strategy_diagnostics"],
        "result": {"dry_run": True, "n_buys": 3, "n_sells": 0},
    }
    skipped_closed = {
        "date": "2026-07-16T09:35:00",
        "weights": dict(funded["weights"]),
        "strategy_diagnostics": funded["strategy_diagnostics"],
        "result": {"dry_run": False, "skipped_market_closed": True},
    }
    skipped_stop = {
        "date": "2026-07-17T09:35:00",
        "weights": dict(funded["weights"]),
        "strategy_diagnostics": funded["strategy_diagnostics"],
        "result": {"dry_run": False, "skipped_portfolio_stop": True},
    }
    all_failed = {  # live path reached, but zero orders made it to Alpaca
        "date": "2026-07-18T09:35:00",
        "weights": dict(funded["weights"]),
        "strategy_diagnostics": funded["strategy_diagnostics"],
        "result": {"dry_run": False, "n_buys": 3, "n_sells": 0,
                   "executed": 0, "failed": 3},
    }
    # Closes exist before the funded date too — the window must ignore them.
    closes_wide = {
        sym: {"2026-07-15": vals["2026-07-20"],
              "2026-07-16": vals["2026-07-20"],
              "2026-07-17": vals["2026-07-20"], **vals}
        for sym, vals in closes.items()
    }
    out = compute_attribution(
        [dry, skipped_closed, skipped_stop, all_failed, funded],
        closes_wide, equity, leverage=lev)

    assert out["status"] == "ok"
    assert out["window_start"] == "2026-07-20"   # the funded entry only
    assert out["n_rebalances"] == 1
    assert out["n_days"] == 3                    # 07-20 → 07-23
    # Same numbers as the clean funded-only history: the pre-funded
    # entries contribute nothing.
    clean = compute_attribution([funded], closes, equity, leverage=lev)
    for name in out["sleeves"]:
        assert out["sleeves"][name]["cum_contribution_pp"] == pytest.approx(
            clean["sleeves"][name]["cum_contribution_pp"], abs=1e-9)
    assert out["book_cum_pp"] == pytest.approx(clean["book_cum_pp"], abs=1e-9)

    # Dry/skipped-only history → still not_started.
    out2 = compute_attribution(
        [dry, skipped_closed, skipped_stop, all_failed], closes_wide, equity,
        leverage=lev)
    assert out2["status"] == "not_started"


# ───── score_protocol ────────────────────────────────────────────────────

PROTO = load_protocol()


def _ok_tracking(**over):
    tr = {
        "status": "ok", "months_elapsed": 1.0,
        "live_geo_monthly_pct": 3.0, "realized_vol_monthly_pct": 14.0,
        "cum_log_return": 0.03, "cum_return_pct": 3.0,
        "max_drawdown_pct": -8.0,
        "spy": {"live_minus_2x_spy_pp": 1.0},
    }
    tr.update(over)
    return tr


def _started_state():
    return {"forward_test_start": "2026-07-20",
            "protocol_sha256": PROTO["_sha256"]}


def test_score_protocol_all_na_before_start():
    score = score_protocol(None, None, None, {}, PROTO)
    assert score["overall"] == "NOT_STARTED"
    assert score["binding"] == "not_bound_yet"
    assert score["checkpoint"] == "pre-start"
    assert score["forward_test_start"] is None
    assert len(score["criteria"]) == len(PROTO["criteria"])
    assert all(c["status"] == "NA" for c in score["criteria"])


def test_score_protocol_kill_on_drawdown_minus_46():
    tr = _ok_tracking(max_drawdown_pct=-46.0)
    score = score_protocol(tr, None, None, _started_state(), PROTO)
    by_id = {c["id"]: c for c in score["criteria"]}
    assert by_id["K1"]["status"] == "KILL"
    assert by_id["K1"]["value"] == -46.0
    assert score["overall"] == "KILL"
    assert score["binding"] == "intact"
    # 3mo/6mo/12mo criteria still NA at 1 month elapsed
    assert by_id["M3-GEO"]["status"] == "NA"
    assert by_id["M12-GEO"]["status"] == "NA"


def test_score_protocol_healthy_continuous_passes():
    score = score_protocol(_ok_tracking(), None, None, _started_state(), PROTO)
    by_id = {c["id"]: c for c in score["criteria"]}
    assert by_id["K1"]["status"] == "PASS"
    # No slippage / reconciliation data yet → K2/K3 stay NA, not PASS
    assert by_id["K2"]["status"] == "NA"
    assert by_id["K3a"]["status"] == "NA"
    assert score["overall"] == "PASS"
    assert score["checkpoint"] == "continuous"


def test_score_protocol_binding_mismatch_flags_modified():
    state = {"forward_test_start": "2026-07-20",
             "protocol_sha256": "deadbeef" * 8}
    score = score_protocol(_ok_tracking(), None, None, state, PROTO)
    assert score["binding"] == "MODIFIED_AFTER_START"


def test_score_protocol_k2_k3_consecutive_collapse():
    slip = {
        "calibrated_cost_bps_per_side": 30.0,
        "by_run": [{"notional_weighted_mean_bps": m}
                   for m in (30.0, 28.0, 27.0)],
    }
    recons = [
        {"flow": {"fill_rate_notional": 0.85},
         "book": {"gross_achieved_pct_of_target": 80.0}},
        {"flow": {"fill_rate_notional": 0.88},
         "book": {"gross_achieved_pct_of_target": 82.0}},
    ]
    score = score_protocol(_ok_tracking(), slip, recons, _started_state(), PROTO)
    by_id = {c["id"]: c for c in score["criteria"]}
    # K2: min of last 3 runs = 27 > 25 → all three above → KILL
    assert by_id["K2"]["value"] == pytest.approx(27.0)
    assert by_id["K2"]["status"] == "KILL"
    # K3a: max fill rate 88% < 90 twice consecutively → operational pause (WARN)
    assert by_id["K3a"]["value"] == pytest.approx(88.0)
    assert by_id["K3a"]["status"] == "WARN"
    # K3b: min gross deviation 18% > 15 twice consecutively → WARN
    assert by_id["K3b"]["value"] == pytest.approx(18.0)
    assert by_id["K3b"]["status"] == "WARN"

    # One healthy rebalance breaks the consecutive window → NA/PASS again
    recons_ok = recons[:1] + [
        {"flow": {"fill_rate_notional": 0.99},
         "book": {"gross_achieved_pct_of_target": 99.0}}]
    score2 = score_protocol(_ok_tracking(), slip, recons_ok,
                            _started_state(), PROTO)
    assert {c["id"]: c for c in score2["criteria"]}["K3a"]["status"] == "PASS"


def test_score_protocol_m6_band_needs_and_conditions_to_kill():
    # Band breached (cum log −0.30 < −0.2754) but live beats 2×SPY → WARN
    tr = _ok_tracking(months_elapsed=6.0, cum_log_return=-0.30,
                      cum_return_pct=-25.9,
                      spy={"live_minus_2x_spy_pp": 5.0})
    score = score_protocol(tr, None, None, _started_state(), PROTO)
    m6 = {c["id"]: c for c in score["criteria"]}["M6-BAND"]
    assert m6["status"] == "WARN"
    # At exactly t=6 the elapsed-time band reproduces the registered
    # −0.2754 — the 6mo checkpoint review reads the committed number.
    assert m6["threshold"] == pytest.approx(-0.2754)          # registered
    assert m6["effective_threshold"] == pytest.approx(-0.2754, abs=2e-4)
    assert m6["evaluated_at_months"] == pytest.approx(6.0)

    # Band breached AND live − 2×SPY < −15 AND live < 0 → KILL
    tr = _ok_tracking(months_elapsed=6.0, cum_log_return=-0.30,
                      cum_return_pct=-25.9,
                      spy={"live_minus_2x_spy_pp": -20.0})
    score = score_protocol(tr, None, None, _started_state(), PROTO)
    m6 = {c["id"]: c for c in score["criteria"]}["M6-BAND"]
    assert m6["status"] == "KILL"
    assert all(a["holds"] for a in m6["and_conditions"])


def test_score_protocol_m6_band_evaluated_at_actual_elapsed_time():
    """Past month 6 the band is expected(t) − 1.28σ√t at t=months_elapsed,
    not the frozen 6-month number — a 12-month window is held to the
    12-month band."""
    band_12 = (12 * math.log(1.0368)
               - 1.28 * (PROTO["reference"]["sigma_monthly_pct"] / 100.0)
               * math.sqrt(12))
    assert band_12 == pytest.approx(-0.2625, abs=2e-4)  # above −0.2754

    # cum log −0.27 at t=12: INSIDE the frozen 6mo band (−0.27 > −0.2754)
    # but BELOW the 12-month band (−0.27 < −0.2625) → breached; with both
    # AND-conditions holding it must KILL.
    tr = _ok_tracking(months_elapsed=12.0, cum_log_return=-0.27,
                      cum_return_pct=-23.7,
                      spy={"live_minus_2x_spy_pp": -20.0})
    score = score_protocol(tr, None, None, _started_state(), PROTO)
    m6 = {c["id"]: c for c in score["criteria"]}["M6-BAND"]
    assert m6["status"] == "KILL"
    assert m6["threshold"] == pytest.approx(-0.2754)          # registered
    assert m6["effective_threshold"] == pytest.approx(band_12, abs=1e-4)
    assert m6["evaluated_at_months"] == pytest.approx(12.0)

    # Same window sitting above the 12-month band → PASS.
    tr = _ok_tracking(months_elapsed=12.0, cum_log_return=-0.25,
                      cum_return_pct=-22.1,
                      spy={"live_minus_2x_spy_pp": -20.0})
    score = score_protocol(tr, None, None, _started_state(), PROTO)
    m6 = {c["id"]: c for c in score["criteria"]}["M6-BAND"]
    assert m6["status"] == "PASS"


def test_score_protocol_12mo_success_and_inconclusive():
    tr = _ok_tracking(months_elapsed=12.0, live_geo_monthly_pct=2.1,
                      realized_vol_monthly_pct=16.0, max_drawdown_pct=-30.0)
    slip = {"calibrated_cost_bps_per_side": 8.0,
            "by_run": [{"notional_weighted_mean_bps": 8.0}] * 4}
    score = score_protocol(tr, slip, None, _started_state(), PROTO)
    by_id = {c["id"]: c for c in score["criteria"]}
    for cid in ("M12-GEO", "M12-DD", "M12-SLIP", "M12-VOL", "M12-FAIL"):
        assert by_id[cid]["status"] == "PASS", cid
    assert by_id["M12-INCONCLUSIVE"]["status"] == "PASS"  # not in [0, 1.8)
    assert score["checkpoint"] == "12mo"

    # geo 1.0%/mo → INCONCLUSIVE fires (WARN) and M12-GEO fails (WARN)
    tr = _ok_tracking(months_elapsed=12.0, live_geo_monthly_pct=1.0,
                      realized_vol_monthly_pct=16.0, max_drawdown_pct=-30.0)
    score = score_protocol(tr, slip, None, _started_state(), PROTO)
    by_id = {c["id"]: c for c in score["criteria"]}
    assert by_id["M12-GEO"]["status"] == "WARN"
    assert by_id["M12-INCONCLUSIVE"]["status"] == "WARN"
    assert by_id["M12-FAIL"]["status"] == "PASS"  # geo >= 0


def test_score_protocol_geo_exactly_at_success_floor_is_success():
    """Boundary: geo == 1.8%/mo is the SUCCESS floor. The INCONCLUSIVE
    band is half-open [0, 1.8) — exactly 1.8 must score SUCCESS
    (M12-GEO PASS) and must NOT fire INCONCLUSIVE."""
    slip = {"calibrated_cost_bps_per_side": 8.0,
            "by_run": [{"notional_weighted_mean_bps": 8.0}] * 4}
    tr = _ok_tracking(months_elapsed=12.0, live_geo_monthly_pct=1.8,
                      realized_vol_monthly_pct=16.0, max_drawdown_pct=-30.0)
    score = score_protocol(tr, slip, None, _started_state(), PROTO)
    by_id = {c["id"]: c for c in score["criteria"]}
    assert by_id["M12-GEO"]["status"] == "PASS"           # >= 1.8 holds
    assert by_id["M12-INCONCLUSIVE"]["status"] == "PASS"  # 1.8 not in [0,1.8)
    assert by_id["M12-FAIL"]["status"] == "PASS"

    # Just below the floor → INCONCLUSIVE fires and M12-GEO fails.
    tr = _ok_tracking(months_elapsed=12.0, live_geo_monthly_pct=1.7999,
                      realized_vol_monthly_pct=16.0, max_drawdown_pct=-30.0)
    score = score_protocol(tr, slip, None, _started_state(), PROTO)
    by_id = {c["id"]: c for c in score["criteria"]}
    assert by_id["M12-GEO"]["status"] == "WARN"
    assert by_id["M12-INCONCLUSIVE"]["status"] == "WARN"

    # Lower edge stays INCLUSIVE: geo == 0 is inside the band (fires) and
    # M12-FAIL (geo < 0) does not.
    tr = _ok_tracking(months_elapsed=12.0, live_geo_monthly_pct=0.0,
                      realized_vol_monthly_pct=16.0, max_drawdown_pct=-30.0)
    score = score_protocol(tr, slip, None, _started_state(), PROTO)
    by_id = {c["id"]: c for c in score["criteria"]}
    assert by_id["M12-INCONCLUSIVE"]["status"] == "WARN"
    assert by_id["M12-FAIL"]["status"] == "PASS"


# ───── dashboard endpoints on the empty / not-started state ──────────────

@pytest.fixture(scope="module")
def client():
    import dashboard
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


def test_api_tracking_not_started_returns_200(client):
    r = client.get("/api/tracking/combo_v2")
    assert r.status_code == 200
    data = r.get_json()
    assert data["model"] == "combo_v2"
    assert data["status"] in ("not_started", "insufficient_data")
    assert data["forward_test_start"] is None


def test_api_sleeves_not_started_returns_200(client):
    r = client.get("/api/sleeves/combo_v2")
    assert r.status_code == 200
    data = r.get_json()
    assert data["model"] == "combo_v2"
    assert data["status"] == "not_started"
    assert data["sleeves"] == {}


def test_api_protocol_not_started_returns_200(client):
    r = client.get("/api/protocol/combo_v2")
    assert r.status_code == 200
    data = r.get_json()
    assert data["model"] == "combo_v2"
    assert data["overall"] == "NOT_STARTED"
    assert data["binding"] == "not_bound_yet"
    assert data["checkpoint"] == "pre-start"
    assert all(c["status"] == "NA" for c in data["criteria"])
    # The current protocol hash is always reported for the tamper check
    assert data["protocol_sha256_current"] == PROTO["_sha256"]


def test_api_endpoints_unknown_model_404(client):
    for ep in ("tracking", "sleeves", "protocol"):
        r = client.get(f"/api/{ep}/nope")
        assert r.status_code == 404, ep


def test_v2_evidence_panels_send_bearer_token(client):
    """The /v2 Protocol + Sleeves panels hit auth-gated internal /api/
    endpoints. They must go through the Bearer-header helper (reusing the
    token the original dashboard stashes under 'dashboard_auth_token'),
    not a bare fetch, and keep a graceful note when no token is stashed."""
    html = client.get("/v2").get_data(as_text=True)
    assert "localStorage.getItem('dashboard_auth_token')" in html
    assert "'Authorization': 'Bearer ' + t" in html
    assert "apiGet(`/api/protocol/${name}`)" in html
    assert "apiGet(`/api/sleeves/${name}`)" in html
    # Graceful fallback note still present for the no-token case.
    assert "authNote" in html


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
