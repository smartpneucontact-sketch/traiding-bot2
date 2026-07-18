"""Daily weight-based sleeve attribution for the forward test (Phase A2).

Orders are netted per symbol across the three sleeves, so per-fill sleeve
attribution is impossible. The weight-based approximation used here is the
same measurement the backtest's per-sleeve references were computed with:

  - Each rebalance's `strategy_diagnostics.sleeves` (recorded PRE-gate by
    ComboStrategy.compute_weights) gives every symbol's per-sleeve blend
    contribution. Gates/caps scale all sleeves identically, so a symbol's
    LIVE weight splits across sleeves in the same proportions:
        frac_sleeve(sym) = contrib_sleeve[sym] / sum_sleeves contrib[sym]
  - Daily sleeve return contribution (on account equity):
        r_sleeve(t) = sum_sym  w_live[sym] * leverage * frac_sleeve(sym)
                               * r_sym(t)
    with r_sym(t) the close-to-close return and w_live the post-gate
    combined weight from state.history (fraction of equity, pre-leverage).
  - `unattributed_residual = live daily return - sum_sleeves r_sleeve` —
    the honesty line: costs, cash drag, intraday timing, missing closes,
    and anything the sleeves diagnostics didn't cover all land here
    explicitly instead of being silently smeared across sleeves.

The attribution window starts at the FIRST history entry recording a
FUNDED rebalance that carries sleeves diagnostics (older entries pre-date
Phase A0 and cannot be split; dry-run smokes and skipped runs also record
diagnostics but placed no orders, so they can neither anchor nor segment
the window — see sleeve_entries). Everything here is pure — the dashboard
endpoint supplies closes (Alpaca bars, cached ~1h) and the equity-by-date
map; any data failure produces a partial result with `data_gaps`, never
an exception.
"""

from __future__ import annotations

from typing import Optional

# Contributions are summed in simple percentage points per day; cumulative
# numbers are arithmetic sums of the daily pp (attribution identity holds
# exactly day-by-day; compounding cross-terms land in the residual note).
MAX_GAPS_LISTED = 30


def _is_funded_rebalance(entry: dict) -> bool:
    """True only when the entry records a FUNDED rebalance — the run went
    down the live order-submission path and at least one order reached
    Alpaca. Dry-run smokes (result.dry_run) and skipped runs
    (result.skipped_market_closed / skipped_portfolio_stop) also append
    history entries carrying sleeves diagnostics + weights (core/runner.py
    records them for every run), but they placed no orders — letting one
    anchor the attribution window would attribute returns to a book that
    never existed (the deployment runbook prescribes a dry-run smoke
    BEFORE the first funded rebalance). `executed` (count of orders
    actually submitted) is written by core/orders.py ONLY on the non-dry,
    non-skipped path, so `executed > 0` is the reliable placed-orders
    marker; it also correctly excludes a funded attempt whose every order
    failed (the prior book still stands, and the prior funded entry still
    describes it)."""
    result = entry.get("result")
    if not isinstance(result, dict):
        return False
    if (result.get("dry_run")
            or result.get("skipped_market_closed")
            or result.get("skipped_portfolio_stop")):
        return False
    return (result.get("executed") or 0) > 0


def sleeve_entries(history: list[dict]) -> list[dict]:
    """History entries (chronological) attribution can be computed for:
    FUNDED rebalances (see _is_funded_rebalance) that carry sleeves
    diagnostics and a live weights dict. Shared with the dashboard's
    /api/sleeves endpoint so the bar-fetch window and the attribution
    window can never disagree."""
    out = []
    for entry in history or []:
        diag = entry.get("strategy_diagnostics") or {}
        if (diag.get("sleeves") and entry.get("weights")
                and _is_funded_rebalance(entry)):
            out.append(entry)
    return out


def _entry_date(entry: dict) -> Optional[str]:
    d = entry.get("date")
    return str(d)[:10] if d else None


def compute_attribution(
    history: list[dict],
    closes_by_symbol: dict[str, dict[str, float]],
    equity_by_date: Optional[dict[str, float]] = None,
    leverage: float = 1.0,
) -> dict:
    """Daily weight-based sleeve attribution. PURE — no I/O.

    Args:
      history: state["history"] (chronological run entries).
      closes_by_symbol: {symbol: {"YYYY-MM-DD": close}} daily closes for
        the held names (endpoint fetches via Alpaca bars). Missing symbols
        / dates become data_gaps, their P&L stays in the residual.
      equity_by_date: {"YYYY-MM-DD": account_equity} for the live daily
        return; None → residual fields stay null.
      leverage: slot target_leverage (weights are pre-leverage fractions).

    Returns a dict with per-sleeve cumulative contributions (pp), the
    book-model sum, the live return, the explicit unattributed residual,
    and `data_gaps`. status "not_started" when no history entry carries
    sleeves diagnostics yet.
    """
    entries = sleeve_entries(history)
    if not entries:
        return {
            "status": "not_started",
            "reason": ("no funded rebalance with sleeves diagnostics yet — "
                       "attribution starts at the first funded rebalance "
                       "(orders actually placed; dry runs and skipped runs "
                       "don't count) that records "
                       "strategy_diagnostics.sleeves"),
            "sleeves": {},
            "data_gaps": [],
        }

    closes_by_symbol = closes_by_symbol or {}
    equity_by_date = equity_by_date or {}
    window_start = _entry_date(entries[0])

    # Trading-day axis: union of all close dates on/after the window start.
    all_dates: set[str] = set()
    for date_map in closes_by_symbol.values():
        all_dates.update(d for d in date_map if d >= window_start)
    dates = sorted(all_dates)

    sleeve_names: list[str] = []
    for entry in entries:
        for name in (entry["strategy_diagnostics"]["sleeves"] or {}):
            if name not in sleeve_names:
                sleeve_names.append(name)

    cum = {name: 0.0 for name in sleeve_names}
    sleeve_days = {name: 0 for name in sleeve_names}
    book_cum = 0.0
    live_cum = 0.0
    residual_cum = 0.0
    n_days = 0
    n_live_days = 0
    daily: list[dict] = []
    gaps: list[str] = []
    unattributed_syms: set[str] = set()

    def _gap(msg: str) -> None:
        if msg not in gaps and len(gaps) < MAX_GAPS_LISTED:
            gaps.append(msg)

    # Per-day walk: day-return p -> t is attributed to the book established
    # by the latest rebalance whose date <= p (the rebalance trades near
    # the open of its own date, so its book already owns the NEXT full
    # close-to-close return; the rebalance day itself is a mixed day whose
    # slippage lands in the residual).
    seg_idx = 0
    for prev_d, cur_d in zip(dates, dates[1:]):
        while (seg_idx + 1 < len(entries)
               and _entry_date(entries[seg_idx + 1]) <= prev_d):
            seg_idx += 1
        entry = entries[seg_idx]
        if _entry_date(entry) > prev_d:
            continue  # before the first sleeved rebalance actually held

        weights: dict = entry.get("weights") or {}
        sleeves: dict = entry["strategy_diagnostics"]["sleeves"] or {}
        contrib_total = {
            sym: sum(abs(sleeves[name].get(sym, 0.0)) for name in sleeves)
            for sym in weights
        }

        day_sleeve = {name: 0.0 for name in sleeve_names}
        day_book = 0.0
        for sym, w in weights.items():
            cmap = closes_by_symbol.get(sym)
            if not cmap:
                _gap(f"{sym}: no close data")
                continue
            c_prev, c_cur = cmap.get(prev_d), cmap.get(cur_d)
            if not c_prev or not c_cur or c_prev <= 0:
                _gap(f"{sym}: missing close {prev_d if not c_prev else cur_d}")
                continue
            r_sym = c_cur / c_prev - 1.0
            pos_ret = float(w) * float(leverage) * r_sym  # on equity
            day_book += pos_ret
            total = contrib_total.get(sym, 0.0)
            if total <= 0:
                unattributed_syms.add(sym)
                continue  # in the live book but in no sleeve → residual
            for name in sleeves:
                frac = abs(sleeves[name].get(sym, 0.0)) / total
                if frac:
                    day_sleeve[name] += pos_ret * frac

        n_days += 1
        day_book_pp = day_book * 100.0
        book_cum += day_book_pp
        for name in sleeve_names:
            pp = day_sleeve[name] * 100.0
            cum[name] += pp
            if day_sleeve[name] != 0.0:
                sleeve_days[name] += 1

        e_prev, e_cur = equity_by_date.get(prev_d), equity_by_date.get(cur_d)
        live_pp = resid_pp = None
        if e_prev and e_cur and e_prev > 0:
            live_pp = (e_cur / e_prev - 1.0) * 100.0
            sleeve_sum_pp = sum(day_sleeve.values()) * 100.0
            resid_pp = live_pp - sleeve_sum_pp
            live_cum += live_pp
            residual_cum += resid_pp
            n_live_days += 1

        daily.append({
            "date": cur_d,
            "sleeves_pp": {n: round(day_sleeve[n] * 100.0, 4)
                           for n in sleeve_names},
            "book_pp": round(day_book_pp, 4),
            "live_pp": round(live_pp, 4) if live_pp is not None else None,
            "residual_pp": round(resid_pp, 4) if resid_pp is not None else None,
        })

    if n_days == 0:
        _gap("no daily closes in the attribution window (bars fetch "
             "failed or window has no completed trading day yet)")
    if n_live_days == 0 and n_days > 0:
        _gap("no equity history overlapping the window — residual "
             "unavailable, sleeve sums are model-only")
    for sym in sorted(unattributed_syms):
        _gap(f"{sym}: held live but absent from sleeves diagnostics — "
             f"its P&L stays in the residual")

    return {
        "status": "ok" if n_days > 0 else "no_data",
        "window_start": window_start,
        "as_of": dates[-1] if dates else None,
        "n_days": n_days,
        "n_live_days": n_live_days,
        "n_rebalances": len(entries),
        "leverage": leverage,
        "sleeves": {
            name: {
                "cum_contribution_pp": round(cum[name], 4),
                "n_days": sleeve_days[name],
            } for name in sleeve_names
        },
        "book_cum_pp": round(book_cum, 4),
        "live_cum_pp": round(live_cum, 4) if n_live_days else None,
        "unattributed_residual_pp": (
            round(residual_cum, 4) if n_live_days else None),
        "residual_note": (
            "residual = live daily return - sum of sleeve contributions, "
            "summed over days with equity data: costs, cash drag, intraday "
            "timing, data gaps. Cumulative numbers are arithmetic sums of "
            "daily pp (no compounding)."),
        "daily_tail": daily[-21:],
        "data_gaps": gaps,
    }
