"""Pure execution-cost statistics over trade-journal dicts (Phase A1).

`compute_slippage_stats(trades)` consumes the dicts returned by
`TradeJournal.get_trades()` and produces the calibrated per-side cost the
research repo needs as its TC parameter — directly comparable against the
5 bp/side assumption baked into every backtest.

Design rules:
  - Pure functions, no I/O, no Alpaca — safe to call from any endpoint.
  - A row with a missing/None `slippage_bps` is UNMEASURED (snapshot fetch
    failed, pre-A0 row, or no fill). It is counted in `n_unmeasured` and
    excluded from every statistic — never treated as zero slippage, which
    would silently bias the calibration toward the assumption.
  - Sign convention matches core/orders.py: positive bps = execution cost
    for BOTH sides (a sell filling above its reference is negative).

Drag arithmetic (the `monthly_drag_estimate_pp` field):
  the combo_v2 book trades roughly 1.6x equity per month in total order
  notional (2.0x-levered gross book, 21-trading-day rebalance cadence,
  ~40% turnover per rebalance, buys + sells both paying the cost). Every
  traded dollar pays the per-side cost once, so the monthly return drag in
  percentage points is:

      drag_pp = -(cost_bps / 1e4) * EQUITY_TRADED_PER_MONTH * 100
              = -cost_bps * 0.016

  e.g. +10 bp/side ~= -0.16 pp/mo. This is the number to hold against the
  3.68 %/mo backtest headline when judging whether live costs matter.
"""

from __future__ import annotations

from typing import Optional

# Research repo's transaction-cost assumption (bps per side) used across
# the combo_v2 backtests; the calibrated live number validates/replaces it.
RESEARCH_ASSUMPTION_BPS = 5.0

# ~1.6x equity traded per month in total order notional (see module
# docstring for the derivation). Used only for the drag estimate.
EQUITY_TRADED_PER_MONTH = 1.6


def _num(v) -> Optional[float]:
    """Coerce a journal value to float; None/non-numeric -> None (unmeasured)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # NaN guard (NaN != NaN) — a NaN slippage is unmeasured, not zero.
    if f != f:
        return None
    return f


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default method) on an
    already-sorted list. q in [0, 1]."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = (n - 1) * q
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _side_stats(vals: list[float]) -> dict:
    """{n, mean, median, p75, p95} over measured bps values; all-None when
    the side has no measured rows."""
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p75": None, "p95": None}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 2),
        "median": round(_percentile(s, 0.50), 2),
        "p75": round(_percentile(s, 0.75), 2),
        "p95": round(_percentile(s, 0.95), 2),
    }


def _notional_weighted_mean(pairs: list[tuple[float, float]]) -> Optional[float]:
    """Weighted mean of (bps, notional_usd) pairs. Rows with zero/None
    notional get zero weight; if ALL weights vanish, fall back to the
    unweighted mean so a degenerate journal still yields a number."""
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs if w and w > 0)
    if total_w > 0:
        return round(sum(b * w for b, w in pairs if w and w > 0) / total_w, 2)
    return round(sum(b for b, _ in pairs) / len(pairs), 2)


def _measured_rows(trades: list[dict], field: str) -> list[tuple[dict, float]]:
    """(trade, bps) for every row where `field` is a real number."""
    out: list[tuple[dict, float]] = []
    for t in trades:
        bps = _num(t.get(field))
        if bps is not None:
            out.append((t, bps))
    return out


def _stats_block(rows: list[tuple[dict, float]]) -> dict:
    """Shared shape for the headline (vs open) and vs-arrival views."""
    return {
        "n_measured": len(rows),
        "by_side": {
            side: _side_stats([b for t, b in rows if t.get("side") == side])
            for side in ("buy", "sell")
        },
        "notional_weighted_mean_bps": _notional_weighted_mean(
            [(b, _num(t.get("notional_usd")) or 0.0) for t, b in rows]
        ),
    }


def _by_run(trades: list[dict]) -> list[dict]:
    """Per-rebalance slippage summary, in journal (chronological) order."""
    order: list[str] = []
    groups: dict[str, list[dict]] = {}
    for t in trades:
        run_id = str(t.get("run_id") or "unknown")
        if run_id not in groups:
            groups[run_id] = []
            order.append(run_id)
        groups[run_id].append(t)

    out: list[dict] = []
    for run_id in order:
        rows = groups[run_id]
        measured = _measured_rows(rows, "slippage_bps")
        vals = sorted(b for _, b in measured)
        out.append({
            "run_id": run_id,
            "timestamp": rows[0].get("timestamp"),
            "n_orders": len(rows),
            "n_measured": len(measured),
            "median_bps": round(_percentile(vals, 0.50), 2) if vals else None,
            "notional_weighted_mean_bps": _notional_weighted_mean(
                [(b, _num(t.get("notional_usd")) or 0.0) for t, b in measured]
            ),
        })
    return out


def compute_slippage_stats(trades: list[dict]) -> dict:
    """Slippage statistics over journal rows (see module docstring).

    Headline numbers use `slippage_bps` (fill vs same-day open — the
    backtest's assumed fill, so `calibrated_cost_bps_per_side` is directly
    the research repo's TC parameter). `vs_arrival` mirrors the same shape
    on `slippage_vs_arrival_bps` (fill vs decision-time arrival price).
    """
    measured = _measured_rows(trades, "slippage_bps")
    headline = _stats_block(measured)
    calibrated = headline["notional_weighted_mean_bps"]

    return {
        "n_trades": len(trades),
        "n_measured": len(measured),
        "n_unmeasured": len(trades) - len(measured),
        "by_side": headline["by_side"],
        "notional_weighted_mean_bps": calibrated,
        "vs_arrival": _stats_block(
            _measured_rows(trades, "slippage_vs_arrival_bps")),
        "by_run": _by_run(trades),
        "calibrated_cost_bps_per_side": calibrated,
        "research_assumption_bps": RESEARCH_ASSUMPTION_BPS,
        # -cost_bps * 1.6/100 pp/mo (module docstring): +10 bp/side of
        # measured cost drags ~0.16 pp off the monthly return at the
        # book's ~1.6x equity-traded-per-month flow.
        "monthly_drag_estimate_pp": (
            round(-calibrated * EQUITY_TRADED_PER_MONTH / 100.0, 4)
            if calibrated is not None else None
        ),
        "equity_traded_per_month_assumption": EQUITY_TRADED_PER_MONTH,
    }
