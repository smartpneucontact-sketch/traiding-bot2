"""Live-vs-backtest tracking for the forward paper test (Phase A2).

`compute_tracking()` is the PURE core: it takes an equity curve (trading-day
points), the bundle's `backtest_reference`, and the `forward_test_start`
date, and answers "is the live account inside the statistical band the
backtest predicts?". Everything network-y (Alpaca portfolio history, bundle
loading) lives in small, best-effort wrappers so the pure math is unit-
testable and safe to call from the dashboard, the runner, and offline tools.

Key conventions:
  - The reference monthly rate is `geo_monthly_return` (0.0368 for the
    v2.3_nofreeze bundle) — the compounded rate an account actually earns.
  - `sigma_monthly` is DERIVED, not measured: the research repo never
    published a monthly sigma, so we back it out of the annualized Sharpe:
        Sharpe = (mean_monthly * 12) / (sigma_monthly * sqrt(12))
        => sigma_monthly = mean_monthly * sqrt(12) / Sharpe
    With mean_monthly=0.0480 and Sharpe=1.06 this gives ~0.157 (15.7%/mo),
    the same number pre-registered in protocol.json / PROTOCOL.md.
  - The +/-1 sigma band is drawn on CUMULATIVE LOG RETURN:
        expected(m) = m * ln(1 + geo_monthly)
        band(m)     = expected(m) +/- sigma_monthly * sqrt(m)
    using the simple-return sigma as an approximation of the log-return
    sigma (documented approximation; the difference is second-order at
    monthly scale).
  - A "month" is 21 trading days — the strategy's rebalance cadence and
    the unit every reference number was computed at.
  - Alpaca pads pre-funding days of `portfolio/history` with synthetic /
    zero equity. `trim_equity_padding()` is the ONE shared trim used both
    here and by dashboard._compute_performance so the two can never
    disagree about where the account's history starts.
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timezone
from typing import Optional

# One "month" = one rebalance cycle = 21 trading days (the cadence every
# backtest reference number was computed at).
TRADING_DAYS_PER_MONTH = 21.0


# ── Equity-curve plumbing (shared with the dashboard) ─────────────────────

def trim_equity_padding(timestamps: list, equity: list) -> tuple[list, list]:
    """Drop leading unfunded padding from an Alpaca portfolio-history pull.

    Alpaca pads days before the account was funded with 0/None (and, on
    fixed-period queries, synthetic $100k) equity. Keep everything from the
    first strictly-positive snapshot on; interior zero days (data hiccups)
    are kept — downstream return math already skips non-positive values.

    Returns (ts_ms, equity) as parallel lists of ints (milliseconds) and
    floats. Rows whose timestamp or equity fails to parse are dropped.
    """
    ts_ms: list[int] = []
    eq: list[float] = []
    started = False
    for t, e in zip(timestamps or [], equity or []):
        try:
            f = float(e) if e is not None else 0.0
            ms = int(t) * 1000
        except (TypeError, ValueError):
            continue
        if not started and f <= 0:
            continue
        started = True
        ts_ms.append(ms)
        eq.append(f)
    return ts_ms, eq


def fetch_forward_equity(mc, logger=None) -> list[tuple[int, float]]:
    """Trimmed (ts_ms, equity) points from Alpaca, `period=all`.

    `period=all` (not 1A) so Alpaca only returns days the account actually
    existed — the same choice dashboard._compute_performance documents.
    Raises on network/auth failure; callers decide how loudly.
    """
    from core.alpaca import alpaca_request
    hist = alpaca_request(
        "GET",
        "v2/account/portfolio/history?period=all&timeframe=1D&extended_hours=false",
        mc, logger=logger,
    )
    ts_ms, eq = trim_equity_padding(
        hist.get("timestamp") or [], hist.get("equity") or [])
    return list(zip(ts_ms, eq))


# ── Bundle reference cache ────────────────────────────────────────────────
# Keyed (path, mtime) like dashboard._load_bundle_meta_cached so a redeploy
# or rebuild invalidates automatically without a restart.

_ref_cache: dict[tuple[str, float], Optional[dict]] = {}
_ref_cache_lock = threading.Lock()


def get_backtest_reference(mc) -> Optional[dict]:
    """The bundle's `backtest_reference` dict for a model slot, or None.

    None on any miss (mc is None, bundle missing/unreadable, no reference
    key) — tracking is evidence tooling and must never take down a caller.
    """
    if mc is None:
        return None
    try:
        path = mc.model_path
        key = (str(path), path.stat().st_mtime)
    except Exception:
        return None
    with _ref_cache_lock:
        if key in _ref_cache:
            return _ref_cache[key]
    try:
        from core.runner import load_model_bundle
        ref = load_model_bundle(path).get("backtest_reference")
        if not isinstance(ref, dict):
            ref = None
    except Exception:
        ref = None
    with _ref_cache_lock:
        _ref_cache[key] = ref
        # Old (path, mtime) generations accumulate one entry per redeploy;
        # cap to keep this bounded over a long-lived process.
        while len(_ref_cache) > 16:
            _ref_cache.pop(next(iter(_ref_cache)))
    return ref


# ── Pure math ─────────────────────────────────────────────────────────────

def derive_sigma_monthly(ref: dict) -> Optional[float]:
    """sigma_monthly = mean_monthly * sqrt(12) / Sharpe (see module doc).

    Uses the ARITHMETIC mean (0.0480), matching how the annualized Sharpe
    was computed in the research repo. 0.0480 * sqrt(12) / 1.06 ~= 0.157.
    """
    try:
        mean_m = float(ref.get("mean_monthly_return"))
        sharpe = float(ref.get("sharpe"))
    except (TypeError, ValueError):
        return None
    if sharpe <= 0 or mean_m <= 0:
        return None
    return mean_m * math.sqrt(12.0) / sharpe


def _date_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d")


def compute_tracking(
    equity_points: list[tuple[int, float]],
    ref: dict,
    start_iso: Optional[str],
    spy_points: Optional[list[tuple[int, float]]] = None,
) -> dict:
    """Live-vs-backtest tracking stats. PURE — no I/O.

    Args:
      equity_points: [(ts_ms, equity)] trading-day points, ascending,
        already padding-trimmed (see trim_equity_padding).
      ref: the bundle's backtest_reference (geo_monthly_return,
        mean_monthly_return, sharpe at minimum).
      start_iso: forward_test_start ("YYYY-MM-DD" or full ISO). None →
        status "not_started" (the clean pre-first-rebalance state).
      spy_points: OPTIONAL [(ts_ms, close)] SPY series over the same
        window for regime context. None → the spy fields stay null.

    Live geo monthly = (E_T / E_0) ** (21 / n_days) - 1 where n_days is the
    number of trading-day steps between the first and last point in the
    forward-test window.
    """
    ref = ref or {}
    geo_ref = ref.get("geo_monthly_return")
    sigma_m = derive_sigma_monthly(ref)

    base = {
        "status": "not_started",
        "forward_test_start": start_iso[:10] if start_iso else None,
        "expected_geo_monthly": geo_ref,
        "expected_geo_monthly_pct": (
            round(geo_ref * 100, 4) if geo_ref is not None else None),
        "sigma_monthly": round(sigma_m, 6) if sigma_m is not None else None,
        "sigma_monthly_pct": (
            round(sigma_m * 100, 4) if sigma_m is not None else None),
        "sigma_derivation": (
            "sigma_m = mean_monthly * sqrt(12) / sharpe_annualized "
            "(back-derived; the research repo published no monthly sigma)"),
    }
    if not start_iso:
        base["reason"] = ("forward_test_start not set — no funded rebalance "
                          "has started the clock yet")
        return base

    start_date = start_iso[:10]
    pts = [(int(t), float(e)) for t, e in (equity_points or [])
           if e is not None and float(e) > 0 and _date_iso(int(t)) >= start_date]
    if len(pts) < 2 or geo_ref is None:
        base["status"] = "insufficient_data"
        base["n_points"] = len(pts)
        base["reason"] = (
            "bundle has no geo_monthly_return reference" if geo_ref is None
            else "need >= 2 equity snapshots on/after forward_test_start")
        return base

    ts = [t for t, _ in pts]
    eq = [e for _, e in pts]
    n_days = len(pts) - 1
    months = n_days / TRADING_DAYS_PER_MONTH

    live_geo = (eq[-1] / eq[0]) ** (TRADING_DAYS_PER_MONTH / n_days) - 1.0
    cum_log = math.log(eq[-1] / eq[0])
    expected_cum_log = months * math.log1p(geo_ref)
    cum_simple_pct = (math.exp(cum_log) - 1.0) * 100.0
    expected_cum_simple_pct = (math.exp(expected_cum_log) - 1.0) * 100.0

    band_half = (sigma_m * math.sqrt(months)) if sigma_m is not None else None
    band_lo = expected_cum_log - band_half if band_half is not None else None
    band_hi = expected_cum_log + band_half if band_half is not None else None
    within = (band_lo <= cum_log <= band_hi) if band_half is not None else None

    # Realized monthly vol from daily log returns (sqrt-21 scaling).
    dlogs = [math.log(b / a) for a, b in zip(eq, eq[1:]) if a > 0 and b > 0]
    realized_vol_m = None
    if len(dlogs) >= 2:
        mu = sum(dlogs) / len(dlogs)
        var = sum((x - mu) ** 2 for x in dlogs) / (len(dlogs) - 1)
        realized_vol_m = math.sqrt(var) * math.sqrt(TRADING_DAYS_PER_MONTH)

    # Drawdown from the forward-test equity peak.
    peak = eq[0]
    max_dd = 0.0
    for e in eq:
        peak = max(peak, e)
        max_dd = min(max_dd, e / peak - 1.0)
    cur_dd = eq[-1] / peak - 1.0

    # Expected path + band, sampled at every live equity point (for charts
    # and for eyeballing where the curve sits inside the cone).
    path_dates, path_live, path_exp, path_lo, path_hi = [], [], [], [], []
    for i, (t, e) in enumerate(pts):
        m_i = i / TRADING_DAYS_PER_MONTH
        exp_i = m_i * math.log1p(geo_ref)
        half_i = sigma_m * math.sqrt(m_i) if sigma_m is not None else None
        path_dates.append(_date_iso(t))
        path_live.append(round(math.log(e / eq[0]), 6))
        path_exp.append(round(exp_i, 6))
        path_lo.append(round(exp_i - half_i, 6) if half_i is not None else None)
        path_hi.append(round(exp_i + half_i, 6) if half_i is not None else None)

    # SPY regime context — only when the caller supplied a series.
    spy = {"cum_return_pct": None, "two_x_cum_return_pp": None,
           "live_minus_2x_spy_pp": None}
    spy_pts = [(int(t), float(c)) for t, c in (spy_points or [])
               if c is not None and float(c) > 0
               and _date_iso(int(t)) >= start_date]
    if len(spy_pts) >= 2:
        spy_ret_pct = (spy_pts[-1][1] / spy_pts[0][1] - 1.0) * 100.0
        spy["cum_return_pct"] = round(spy_ret_pct, 4)
        spy["two_x_cum_return_pp"] = round(2.0 * spy_ret_pct, 4)
        spy["live_minus_2x_spy_pp"] = round(cum_simple_pct - 2.0 * spy_ret_pct, 4)

    base.update({
        "status": "ok",
        "as_of": _date_iso(ts[-1]),
        "n_days": n_days,
        "months_elapsed": round(months, 4),
        "equity_start": round(eq[0], 2),
        "equity_last": round(eq[-1], 2),
        "live_geo_monthly": round(live_geo, 6),
        "live_geo_monthly_pct": round(live_geo * 100, 4),
        "cum_log_return": round(cum_log, 6),
        "expected_cum_log_return": round(expected_cum_log, 6),
        "band_1sigma_log": {
            "lo": round(band_lo, 6) if band_lo is not None else None,
            "hi": round(band_hi, 6) if band_hi is not None else None,
        },
        "within_1sigma": within,
        "cum_return_pct": round(cum_simple_pct, 4),
        "expected_cum_return_pct": round(expected_cum_simple_pct, 4),
        "cum_tracking_diff_pp": round(
            cum_simple_pct - expected_cum_simple_pct, 4),
        "realized_vol_monthly": (
            round(realized_vol_m, 6) if realized_vol_m is not None else None),
        "realized_vol_monthly_pct": (
            round(realized_vol_m * 100, 4)
            if realized_vol_m is not None else None),
        "live_drawdown_pct": round(cur_dd * 100, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "spy": spy,
        "path": {
            "dates": path_dates,
            "cum_log_live": path_live,
            "cum_log_expected": path_exp,
            "band_lo": path_lo,
            "band_hi": path_hi,
        },
    })
    return base


# ── Best-effort snapshot for the runner ───────────────────────────────────

def tracking_snapshot(mc, state: dict, logger=None) -> Optional[dict]:
    """Fetch equity + reference and compute tracking; None on ANY failure.

    Called best-effort from core/runner.py after a funded rebalance so the
    run report can print the LIVE VS BACKTEST block. Must never raise into
    the live path.
    """
    try:
        state = state or {}
        ref = get_backtest_reference(mc)
        if not ref:
            return None
        pts = fetch_forward_equity(mc, logger=logger)
        return compute_tracking(pts, ref, state.get("forward_test_start"))
    except Exception as e:
        if logger:
            logger.warning(f"  tracking snapshot failed (non-fatal): {e}")
        return None
