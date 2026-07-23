"""Regression tests for the main-dashboard display bugs (2026-07 screenshot).

BUG 1 — "DAY P&L" showed the SUM OF EQUITIES (+Infinity%): Alpaca's
  last_equity came back 0/missing, the backend coerced it to 0.0 and the JS
  computed equity - 0 = equity as the daily delta. Covers the server-side
  _day_pl_fields guard (day_pl/day_pl_pct are None — never a fallback to
  equity — when last_equity <= 0 or missing) and the /api/account payload.

BUG 2 — "NEXT RUN" rendered in the past / "LAST RUN" confusion:
  bot_status["next_run_at"] was stamped ONCE at boot and never refreshed, so
  after that fire passed the tile kept showing a stale (past) next-run.
  last_run_at was a NAIVE datetime.now() string the JS had to guess was UTC.
  Covers: /api/status live next-run from the scheduler, offset-aware UTC
  emission, UTC->ET round-trip, and the tripwire invariants
  (last-run <= now, next-run > now).

Run:  /opt/anaconda3/bin/python -m pytest tests/test_dashboard_display.py -v
"""
from __future__ import annotations

import copy
import os
import re
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Must be set before dashboard/core imports — DATA_DIR resolves at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

ET = ZoneInfo("America/New_York")

# Replica of the dashboard JS _toDate() offset-detection regex. Strings the
# status API emits MUST already match it, so the JS never has to guess
# ("append Z") what timezone a timestamp is in.
JS_TZ_RE = re.compile(r"[Z+]|[+-]\d\d:?\d\d$")


@pytest.fixture(scope="module")
def dash():
    import dashboard
    return dashboard


@pytest.fixture(scope="module")
def client(dash):
    dash.app.config["TESTING"] = True
    with dash.app.test_client() as c:
        yield c


@pytest.fixture
def status_snapshot(dash):
    """Snapshot/restore bot_status around tests that mutate it."""
    with dash.status_lock:
        before = copy.deepcopy(dash.bot_status)
    yield
    with dash.status_lock:
        dash.bot_status.clear()
        dash.bot_status.update(before)


# ───── BUG 1: _day_pl_fields guard ───────────────────────────────────────

def test_day_pl_normal_positive(dash):
    f = dash._day_pl_fields(100396.79, 100000.0)
    assert f["day_pl"] == pytest.approx(396.79)
    assert f["day_pl_pct"] == pytest.approx(0.39679, abs=1e-4)


def test_day_pl_normal_negative(dash):
    f = dash._day_pl_fields(79777.05, 80500.0)
    assert f["day_pl"] == pytest.approx(-722.95)
    assert f["day_pl_pct"] == pytest.approx(-722.95 / 80500.0 * 100, abs=1e-4)


def test_day_pl_flat_is_zero_not_none(dash):
    f = dash._day_pl_fields(100000.0, 100000.0)
    assert f["day_pl"] == 0.0
    assert f["day_pl_pct"] == 0.0


def test_day_pl_last_equity_zero_is_none_not_equity(dash):
    """THE screenshot bug: last_equity=0 must yield None, never equity
    (which summed to $180,173.84 'Day P&L') and never a /0 Infinity pct."""
    f = dash._day_pl_fields(79777.05, 0)
    assert f == {"day_pl": None, "day_pl_pct": None}
    f2 = dash._day_pl_fields(100396.79, 0.0)
    assert f2 == {"day_pl": None, "day_pl_pct": None}


def test_day_pl_last_equity_missing_or_negative_is_none(dash):
    assert dash._day_pl_fields(79777.05, None) == \
        {"day_pl": None, "day_pl_pct": None}
    assert dash._day_pl_fields(79777.05, -5.0) == \
        {"day_pl": None, "day_pl_pct": None}


def test_day_pl_accepts_alpaca_string_fields(dash):
    """Alpaca returns numeric fields as strings."""
    f = dash._day_pl_fields("100396.79", "100000")
    assert f["day_pl"] == pytest.approx(396.79)
    assert f["day_pl_pct"] == pytest.approx(0.39679, abs=1e-4)


def test_day_pl_garbage_and_nan_are_none(dash):
    assert dash._day_pl_fields("oops", "100000") == \
        {"day_pl": None, "day_pl_pct": None}
    assert dash._day_pl_fields(100000.0, "n/a") == \
        {"day_pl": None, "day_pl_pct": None}
    assert dash._day_pl_fields(float("nan"), 100000.0) == \
        {"day_pl": None, "day_pl_pct": None}
    assert dash._day_pl_fields(100000.0, float("nan")) == \
        {"day_pl": None, "day_pl_pct": None}


# ───── BUG 1: /api/account payload ───────────────────────────────────────

def _fake_account_route(dash, monkeypatch, model_name, acct_payload):
    """Route /api/account/<model_name> against a canned Alpaca answer."""
    mc = types.SimpleNamespace(name=model_name)
    monkeypatch.setattr(dash, "_get_model_config",
                        lambda name: mc if name == model_name else None)
    monkeypatch.setattr(dash.pipeline, "alpaca_request",
                        lambda method, path, mc_, logger=None: acct_payload)
    with dash._cache_lock:
        dash._api_cache.pop(f"account_{model_name}", None)


def test_api_account_day_pl_computed(dash, client, monkeypatch):
    _fake_account_route(dash, monkeypatch, "m_daypl_ok", {
        "equity": "100396.79", "last_equity": "100000",
        "portfolio_value": "100396.79", "cash": "5000",
        "buying_power": "200000", "long_market_value": "95396.79",
        "short_market_value": "0", "initial_margin": "0",
        "status": "ACTIVE",
    })
    r = client.get("/api/account/m_daypl_ok")
    assert r.status_code == 200
    d = r.get_json()
    assert d["day_pl"] == pytest.approx(396.79)
    assert d["day_pl_pct"] == pytest.approx(0.39679, abs=1e-4)
    assert d["last_equity"] == pytest.approx(100000.0)


def test_api_account_last_equity_zero_yields_null_day_pl(dash, client, monkeypatch):
    """Regression for the screenshot: equity $79,777.05, last_equity 0.
    day_pl must be null — NOT 79777.05 — and day_pl_pct null (no Infinity)."""
    _fake_account_route(dash, monkeypatch, "m_daypl_zero", {
        "equity": "79777.05", "last_equity": "0",
        "portfolio_value": "79777.05", "cash": "1000",
        "buying_power": "0", "long_market_value": "78777.05",
        "short_market_value": "0", "initial_margin": "0",
        "status": "ACTIVE",
    })
    d = client.get("/api/account/m_daypl_zero").get_json()
    assert d["equity"] == pytest.approx(79777.05)
    assert d["day_pl"] is None
    assert d["day_pl_pct"] is None


def test_api_account_last_equity_missing_yields_null_day_pl(dash, client, monkeypatch):
    _fake_account_route(dash, monkeypatch, "m_daypl_missing", {
        "equity": "100396.79",   # no last_equity key at all
        "portfolio_value": "100396.79", "cash": "1000",
        "buying_power": "0", "long_market_value": "99396.79",
        "short_market_value": "0", "initial_margin": "0",
        "status": "ACTIVE",
    })
    d = client.get("/api/account/m_daypl_missing").get_json()
    assert d["day_pl"] is None
    assert d["day_pl_pct"] is None
    # last_equity is None (not 0.0) when no baseline resolves from Alpaca OR
    # portfolio history — serving 0.0 as if it were real is the lie behind
    # the original Infinity% bug.
    assert d["last_equity"] is None
    assert d["day_pl_source"] == "unavailable"


# ───── BUG 2: timestamp emission (UTC-aware, JS never guesses) ───────────

def test_utc_now_iso_is_offset_aware_and_not_in_future(dash):
    s = dash._utc_now_iso()
    assert JS_TZ_RE.search(s), f"no explicit offset in {s!r}"
    dt = datetime.fromisoformat(s)
    assert dt.tzinfo is not None and dt.utcoffset() == timedelta(0)
    # Tripwire: a freshly emitted last-run timestamp can never be later
    # than now (the screenshot showed LAST RUN two days in the future).
    assert dt <= datetime.now(timezone.utc) + timedelta(seconds=2)


def test_utc_to_et_round_trip_summer_and_winter(dash):
    """The display convention: UTC instant -> ET wall clock (what the JS
    Intl formatter does). 13:36Z in July == 09:36 ET (EDT);
    14:36Z in January == 09:36 ET (EST)."""
    summer = datetime.fromisoformat("2026-07-21T13:36:45+00:00").astimezone(ET)
    assert (summer.month, summer.day, summer.hour, summer.minute) == (7, 21, 9, 36)
    winter = datetime.fromisoformat("2026-01-15T14:36:00+00:00").astimezone(ET)
    assert (winter.month, winter.day, winter.hour, winter.minute) == (1, 15, 9, 36)
    # Round-trip back to UTC is lossless (no double-shifting).
    assert summer.astimezone(timezone.utc).isoformat() == "2026-07-21T13:36:45+00:00"


def test_run_completion_stamps_aware_last_run_and_refreshes_next_run(
        dash, monkeypatch, status_snapshot):
    """End-to-end tripwire around run_trading_pipeline:
    after a run, last_run_at <= now < next_run_at (both offset-aware)."""
    fired = datetime.now(ET) + timedelta(days=1)
    stub_sched = types.SimpleNamespace(
        get_job=lambda job_id: types.SimpleNamespace(next_run_time=fired))
    monkeypatch.setattr(dash, "_scheduler", stub_sched)
    monkeypatch.setattr(dash.pipeline, "run_pipeline",
                        lambda dry_run=False, force=False, model_filter=None: None)
    with dash.status_lock:
        dash.bot_status["state"] = "idle"
        # the stale boot-time value that produced "NEXT RUN in the past"
        dash.bot_status["next_run_at"] = "2026-07-21T09:35:00-04:00"

    dash.run_trading_pipeline(force=True)

    now = datetime.now(timezone.utc)
    last = datetime.fromisoformat(dash.bot_status["last_run_at"])
    nxt = datetime.fromisoformat(dash.bot_status["next_run_at"])
    assert last.tzinfo is not None
    assert last <= now + timedelta(seconds=2), "last run rendered in the future"
    assert nxt > now, "next run rendered in the past"
    assert nxt == fired


# ───── BUG 2: /api/status serves the LIVE next fire time ────────────────

def test_api_status_next_run_is_live_not_boot_frozen(
        dash, client, monkeypatch, status_snapshot):
    """The stored next_run_at is a stale PAST value (frozen at boot); the
    scheduler's live job says tomorrow. /api/status must serve the live one."""
    live = datetime.now(ET).replace(microsecond=0) + timedelta(days=1)
    stub_sched = types.SimpleNamespace(
        get_job=lambda job_id: types.SimpleNamespace(next_run_time=live))
    monkeypatch.setattr(dash, "_scheduler", stub_sched)
    with dash.status_lock:
        dash.bot_status["next_run_at"] = "2026-07-21T09:35:00-04:00"  # stale

    s = client.get("/api/status").get_json()
    assert s["next_run_at"] == live.isoformat()
    nxt = datetime.fromisoformat(s["next_run_at"])
    assert JS_TZ_RE.search(s["next_run_at"])
    assert nxt > datetime.now(timezone.utc), "next run rendered in the past"


def test_api_status_falls_back_to_stored_when_no_scheduler(
        dash, client, monkeypatch, status_snapshot):
    monkeypatch.setattr(dash, "_scheduler", None)
    with dash.status_lock:
        dash.bot_status["next_run_at"] = "2026-07-22T09:35:00-04:00"
    s = client.get("/api/status").get_json()
    assert s["next_run_at"] == "2026-07-22T09:35:00-04:00"


def test_cron_trigger_next_fire_always_in_future(dash):
    """Parity with __main__'s CronTrigger: the next fire it computes is
    offset-aware and strictly after now — the value the dashboard renders
    as NEXT RUN can therefore never be in the past unless it goes stale."""
    from apscheduler.triggers.cron import CronTrigger
    trig = CronTrigger(hour=9, minute=35, day_of_week="mon-fri",
                       timezone="America/New_York")
    now = datetime.now(ET)
    nxt = trig.get_next_fire_time(None, now)
    assert nxt is not None and nxt.tzinfo is not None
    assert nxt > now
    assert JS_TZ_RE.search(nxt.isoformat())
    assert (nxt.hour, nxt.minute) == (9, 35)
    assert nxt.weekday() < 5


# ───── BUG 2: /ready staleness check survives aware timestamps ──────────

def test_ready_staleness_flags_old_aware_last_run(dash, client, status_snapshot):
    """Aware last_run_at used to raise (naive-vs-aware subtraction) and
    silently DISABLE the staleness alert. It must flag a 10-day-old run."""
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    with dash.status_lock:
        dash.bot_status["last_run_at"] = old
        dash.bot_status["last_run_status"] = "success"
    r = client.get("/ready")
    problems = r.get_json()["problems"]
    assert any("stale" in p for p in problems), problems
    assert r.status_code == 503


def test_ready_fresh_aware_last_run_not_stale(dash, client, status_snapshot):
    with dash.status_lock:
        dash.bot_status["last_run_at"] = dash._utc_now_iso()
        dash.bot_status["last_run_status"] = "success"
    problems = client.get("/ready").get_json()["problems"]
    assert not any("stale" in p for p in problems), problems


# ───── Frontend bindings (source-level tripwires, test_a2 style) ─────────

def test_main_page_day_pl_binding_uses_server_fields(client):
    html = client.get("/").get_data(as_text=True)
    # The raw-fallback delta computation is gone from the main page JS…
    assert "acct.equity - acct.last_equity" not in html
    # …replaced by the server-computed, null-guarded fields.
    assert "acct.day_pl" in html
    assert "day_pl_pct" in html
    # Top tile aggregates the delta over models with a KNOWN baseline.
    assert "totalPL / totalLastEq" in html
    # The old Infinity-producing denominator is gone.
    assert "totalEquity - totalPL" not in html


def test_v2_page_day_pnl_guarded(client):
    html = client.get("/v2").get_data(as_text=True)
    # last_equity no longer silently defaults to equity (delta-of-zero lie)…
    assert "a.last_equity ?? equity" not in html
    assert "totalDayPnl / (totalEquity - totalDayPnl)" not in html
    # …the slot + hero use the guarded day_pl path.
    assert "a.day_pl" in html
    assert "haveDayPnl" in html


def test_main_page_renders_run_tiles(client):
    html = client.get("/").get_data(as_text=True)
    assert "status.last_run_at" in html
    assert "status.next_run_at" in html


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ───── BUG 1b: last_equity baseline fallback (Alpaca returns '0') ────────
# Observed live 2026-07-23: both ACTIVE funded paper accounts returned
# last_equity='0'. The guard correctly refused it, but the tile then showed
# a permanent "—". _resolve_last_equity derives the baseline from portfolio
# history (last daily mark strictly before today, ET) instead.

def _et_epoch(days_ago: int) -> int:
    d = datetime.now(ET) - timedelta(days=days_ago)
    return int(d.replace(hour=16, minute=0, second=0, microsecond=0).timestamp())


def test_resolve_prefers_alpaca_value_no_history_call(dash, monkeypatch):
    def boom(mc, logger):
        raise AssertionError("history must not be fetched when alpaca value is good")
    monkeypatch.setattr(dash, "_fetch_equity_history", boom)
    mc = types.SimpleNamespace(name="m_res_alpaca")
    le, src = dash._resolve_last_equity(mc, "100000.5", None)
    assert le == pytest.approx(100000.5) and src == "alpaca"


def test_resolve_falls_back_to_history_yesterday_bar(dash, monkeypatch):
    ts = [_et_epoch(2), _et_epoch(1), _et_epoch(0)]
    eq = [78000.0, 79500.0, 80425.84]
    monkeypatch.setattr(dash, "_fetch_equity_history", lambda mc, lg: (ts, eq))
    mc = types.SimpleNamespace(name="m_res_hist")
    le, src = dash._resolve_last_equity(mc, "0", None)
    assert le == pytest.approx(79500.0) and src == "history"  # newest pre-today bar


def test_resolve_handles_ms_epochs(dash, monkeypatch):
    ts = [_et_epoch(1) * 1000, _et_epoch(0) * 1000]
    eq = [79500.0, 80000.0]
    monkeypatch.setattr(dash, "_fetch_equity_history", lambda mc, lg: (ts, eq))
    mc = types.SimpleNamespace(name="m_res_ms")
    le, src = dash._resolve_last_equity(mc, None, None)
    assert le == pytest.approx(79500.0) and src == "history"


def test_resolve_unavailable_when_only_today(dash, monkeypatch):
    monkeypatch.setattr(dash, "_fetch_equity_history",
                        lambda mc, lg: ([_et_epoch(0)], [100000.0]))
    mc = types.SimpleNamespace(name="m_res_today")
    le, src = dash._resolve_last_equity(mc, "0", None)
    assert le is None and src == "unavailable"


def test_resolve_caches_per_day(dash, monkeypatch):
    calls = {"n": 0}
    def counted(mc, lg):
        calls["n"] += 1
        return ([_et_epoch(1)], [79500.0])
    monkeypatch.setattr(dash, "_fetch_equity_history", counted)
    mc = types.SimpleNamespace(name="m_res_cache")
    a = dash._resolve_last_equity(mc, "0", None)
    b = dash._resolve_last_equity(mc, "0", None)
    assert a == b == (79500.0, "history") and calls["n"] == 1


def test_api_account_history_fallback_end_to_end(dash, client, monkeypatch):
    """last_equity '0' from Alpaca + valid history ⇒ real day_pl, not '—'."""
    _fake_account_route(dash, monkeypatch, "m_daypl_histfb", {
        "equity": "80425.84", "last_equity": "0",
        "portfolio_value": "80425.84", "cash": "34251.08",
        "buying_power": "34251.08", "long_market_value": "46174.76",
        "short_market_value": "0", "initial_margin": "0",
        "status": "ACTIVE",
    })
    monkeypatch.setattr(dash, "_fetch_equity_history",
                        lambda mc, lg: ([_et_epoch(1)], [79839.25]))
    r = client.get("/api/account/m_daypl_histfb")
    d = r.get_json()
    assert d["day_pl_source"] == "history"
    assert d["day_pl"] == pytest.approx(80425.84 - 79839.25)
    assert d["day_pl_pct"] == pytest.approx((80425.84 - 79839.25) / 79839.25 * 100, abs=1e-3)


# ───── BUG 2b: LAST RUN "Never" after deploy (in-memory reset) ───────────

def test_status_last_run_falls_back_to_persisted_state(dash, client, monkeypatch,
                                                       status_snapshot):
    with dash.status_lock:
        dash.bot_status["last_run_at"] = None
    mc = types.SimpleNamespace(name="m_status_fb")
    monkeypatch.setattr(dash.pipeline, "get_active_models", lambda: [mc])
    monkeypatch.setattr(dash, "_load_model_state", lambda name: {
        "run_count": 3, "history": [],
        "last_run": "2026-07-21T13:37:29.991337",  # legacy naive-UTC form
    })
    d = client.get("/api/status").get_json()
    assert d["last_run_at"] == "2026-07-21T13:37:29.991337+00:00"
    assert d["last_run_source"] == "persisted_state"
    assert JS_TZ_RE.search(d["last_run_at"])  # JS never guesses the offset
