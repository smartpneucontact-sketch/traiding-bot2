"""Intraday cut-loss scanner — runs every minute during market hours.

Three stop layers, applied in order per slot every 60s:

  1. Portfolio-level **soft tiered** stop (replaces the legacy hard
     "liquidate all at -3% DD" rule):
       Tier 1 (DD ≤ pstop):       scale to 60% of the DAY-START gross book
       Tier 2 (DD ≤ pstop·5/3):   scale to 30% of the DAY-START gross book
       Tier 3 (DD ≤ pstop·7/3):   liquidate to 0% + trip flag
     Tier targets are fractions of the MORNING book (the validated
     `validation/stop_grid.py` TIER_F={1:0.6, 2:0.3, 3:0.0} semantics),
     NOT of current equity: on 2026-07-15 the old `equity × f` target
     over-sold a ~$150k gross book against ~$79k equity at the Tier-1
     trigger — ~$98k sold where the validated rule sells ~$60k.
     `pstop` is the configured `cutloss_portfolio_stop` scaled by
     `target_leverage` (unless `cutloss_scale_by_leverage=False`), so the
     thresholds keep their *underlying-move* meaning at any leverage —
     a -3% configured stop trips at -6% levered equity on a 2x book.
     After Tier 3, re-entry waits `cutloss_reentry_delay_days` trading
     days (the 2026-06-05 cascade re-bought the full book the very next
     morning and was stopped out again the day after).

  2. Per-position **hard stop**: sell if down `cutloss_hard_stop`%
     from average entry price.

  3. Per-position **trailing stop**: sell if down `cutloss_trailing_stop`%
     from the position's peak price since entry.

After hard/trailing stops fire, `_redistribute_after_cutloss` replaces
sold positions with the model's next-best picks, sized from the actual
stop-sale proceeds and capped at each name's own target weight (the
2026-06-10 incident put 30% of equity into one stock because sizing
used the account's whole cash pile / a shrinking divisor). Names already
stopped out today are never re-bought the same day. The "topup" pattern
(adding freed cash pro-rata to surviving positions, which caused the
2026-05-01 / 2026-05-07 redistribute death spirals) remains removed —
leftover cash sits as cash until the next scheduled rebalance.

State (peak prices, daily equity + book anchors, trip flag, today's
stop-sales, re-entry cool-down) is persisted per-slot in `mc.state_path`
so it survives Railway redeploys.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from core.alpaca import _to_alpaca_symbol, alpaca_request, fetch_snapshots
from core.config import get_active_models
from core.journal import TradeJournal, TradeRecord
from core.logging_setup import get_cutloss_logger
from core.market import is_market_open
from core.orders import poll_order_status
from core.state import load_state, save_state, state_lock

if TYPE_CHECKING:
    from core.config import ModelConfig


# A single lock serialises read-modify-write windows on the state file —
# shared with the daily pipeline via core.state so a 9:35 ET rebalance
# can't clobber scanner-written keys. The old name is kept because
# pipeline.py re-exports it.
_cutloss_state_lock = state_lock


def _effective_portfolio_stop(mc: "ModelConfig") -> float | None:
    """Tier-1 threshold on *levered* equity drawdown, in negative percent.
    None means the portfolio stop is disabled.

    The configured `cutloss_portfolio_stop` is expressed as an underlying
    market move (the v7-era tuning assumed a ~1x book). On a levered book
    the same underlying move produces `leverage`× the equity drawdown, so
    the equity-space threshold is scaled by `target_leverage` unless the
    slot opts out via `cutloss_scale_by_leverage=False`.

    Sign-safe: a positive configured value is treated as its negative (a
    positive threshold would otherwise liquidate on every scan); 0/None
    disables the tiers.
    """
    raw = getattr(mc, "cutloss_portfolio_stop", None)
    if not raw:
        return None
    pstop = -abs(float(raw))
    if getattr(mc, "cutloss_scale_by_leverage", True):
        pstop *= max(1.0, float(getattr(mc, "target_leverage", 1.0) or 1.0))
    return pstop


class _DebugTickLogger:
    """Proxy that demotes `.info()` to `.debug()`.

    Used for the routine positions/account fetches the 60s scan makes on
    every tick, so a no-action scan writes nothing at INFO (apscheduler
    already logs the job firing). Triggers, tier events, warnings and
    errors keep their real levels — only `.info` is demoted.
    """

    def __init__(self, logger):
        self._logger = logger

    def info(self, *args, **kwargs):
        self._logger.debug(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._logger, name)


def _add_trading_days(start_iso: str, n: int) -> str:
    """ISO date `n` Mon–Fri days after `start_iso` (holidays not subtracted,
    matching core.state.trading_days_between — overshooting is harmless)."""
    d = date.fromisoformat(start_iso[:10])
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d.isoformat()


def refresh_daily_book_anchor_after_rebalance(mc: "ModelConfig",
                                              rb_data: dict | None,
                                              logger=None) -> float | None:
    """Re-anchor the tier scaler's day-start book to the POST-REBALANCE gross.

    stop_grid.py semantics: the validated backtest applies the scheduled
    rebalance at the top of the day (`w = w_new` BEFORE any tier check),
    so the TIER_F targets ({1: 0.6, 2: 0.3, 3: 0.0} × book) are fractions
    of THAT day's post-rebalance book. Live, the scanner initializes
    `daily_book_start` on its first in-hours tick (~09:30) — the
    overnight, PRE-rebalance book — and the same-day guard in
    `_cutloss_scan_model` deliberately never re-anchors. On rebalance
    days (09:35 pipeline) the tiers would therefore scale against the
    WRONG base, so `core.runner` calls this right after a successful
    funded rebalance.

    Anchor source, in order:
      1. rb_data["reconciliation"]["book"]["achieved_gross_usd"] — the
         measured post-rebalance gross (one get_positions call made by
         the reconciliation block in core.orders.rebalance_portfolio);
      2. rb_data["target_gross_usd"] — the sum of target notionals
         (post auto-scale), when the book-level reconciliation failed.

    Returns the new anchor, or None when no usable gross was found (the
    morning anchor is kept and a warning is logged). Writes go through
    the same locked load→mutate→save path the scanner uses, so the two
    writers can never interleave, and the keys are in
    core.state.SCANNER_OWNED_KEYS so the pipeline's merged save adopts
    the disk value instead of clobbering it.
    """
    achieved: float | None = None
    try:
        book = ((rb_data or {}).get("reconciliation") or {}).get("book") or {}
        val = book.get("achieved_gross_usd")
        if val is not None and float(val) > 0:
            achieved = float(val)
    except (TypeError, ValueError):
        achieved = None
    if achieved is None:
        try:
            val = (rb_data or {}).get("target_gross_usd")
            if val is not None and float(val) > 0:
                achieved = float(val)
        except (TypeError, ValueError):
            achieved = None
    if achieved is None:
        if logger:
            logger.warning(
                f"[CUTLOSS] {mc.name}: day-start book anchor NOT refreshed "
                f"after rebalance — no achieved/target gross in rebalance "
                f"data; tier targets keep the morning (pre-rebalance) "
                f"anchor for today."
            )
        return None

    today_iso = datetime.now().strftime("%Y-%m-%d")
    with _cutloss_state_lock:
        state = load_state(mc)
        prev = state.get("daily_book_start")
        state["daily_book_start"] = achieved
        state["daily_book_start_date"] = today_iso
        save_state(state, mc)
    if logger:
        logger.info(
            f"[CUTLOSS] {mc.name}: day-start book anchor refreshed after "
            f"rebalance: ${float(prev or 0):,.2f} → ${achieved:,.2f} "
            f"(stop_grid TIER_F tiers scale against the post-rebalance book)."
        )
    return achieved


# ═══════════════════════════════════════════════════════════════════════════
# Top-level scan entry point — called every 60s by the dashboard scheduler.
# ═══════════════════════════════════════════════════════════════════════════

def cutloss_scan() -> None:
    """Scan all cutloss-enabled models and execute stops if triggered.

    No-ops outside market hours. Each model's exceptions are caught
    individually so one slot's API failure doesn't disable scans for the
    others.
    """
    if not is_market_open():
        return

    logger = get_cutloss_logger()
    models = get_active_models()
    cutloss_models = [mc for mc in models if mc.enable_cutloss]
    if not cutloss_models:
        return

    for mc in cutloss_models:
        try:
            _cutloss_scan_model(mc, logger)
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: scan error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Per-model scan: portfolio-tier scaler + per-position stops + redistribute
# ═══════════════════════════════════════════════════════════════════════════

def _cutloss_scan_model(mc: "ModelConfig", logger, top_n: int = 20) -> None:
    """Run cut-loss checks for a single model.

    `top_n` is used only in the redistribute summary log and is passed
    through to `_redistribute_after_cutloss`.
    """
    today_iso = datetime.now().strftime("%Y-%m-%d")
    # Routine per-tick API fetches log at DEBUG (no-action ticks used to
    # write INFO lines every 60s); triggers keep their real levels.
    tick_logger = _DebugTickLogger(logger)

    with _cutloss_state_lock:
        state = load_state(mc)

        # Trip-flag short-circuit: portfolio_stop already fired today.
        # Don't return blindly — if the Tier-3 liquidation partially
        # failed (per-position close errors are swallowed), the leftovers
        # would otherwise sit unprotected all session with every stop
        # layer disabled. Retry until flat.
        if state.get("portfolio_stop_tripped_date") == today_iso:
            try:
                leftovers = alpaca_request("GET", "v2/positions", mc,
                                           logger=tick_logger)
            except Exception:
                return
            if leftovers:
                logger.warning(
                    f"[CUTLOSS] {mc.name}: Tier-3 tripped today but "
                    f"{len(leftovers)} positions remain — retrying liquidation."
                )
                _liquidate_all(mc, leftovers, "portfolio_stop", logger)
            return

        try:
            positions = alpaca_request("GET", "v2/positions", mc,
                                       logger=tick_logger)
        except Exception as e:
            logger.warning(f"[CUTLOSS] {mc.name}: failed to fetch positions: {e}")
            return

        if not positions:
            return

        n_positions = len(positions)

        try:
            acct = alpaca_request("GET", "v2/account", mc, logger=tick_logger)
            current_equity = float(acct.get("equity", 0))
            last_equity = float(acct.get("last_equity", 0))
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: failed to fetch account: {e}")
            return

        # Initialise / refresh the daily anchor on date change. last_equity
        # is yesterday's close from Alpaca and is stable all day, so any
        # intraday restart still measures drawdown from the same baseline.
        state_dirty = False
        if state.get("daily_portfolio_start_date") != today_iso:
            anchor = last_equity if last_equity > 0 else current_equity
            if anchor > 0:
                state["daily_portfolio_start"] = anchor
                state["daily_portfolio_start_date"] = today_iso
                state_dirty = True
        start_equity = state.get("daily_portfolio_start", last_equity)

        # Day-anchored gross book (sum of position market values): the
        # base the validated tier rule scales against — stop_grid.py's
        # TIER_F targets are fractions of the MORNING book, not of
        # equity (the 2026-07-15 over-sale). Re-anchored on date change;
        # after a mid-day deploy restart the first scan initializes it
        # from current positions (best available proxy). Same-day tier
        # sells do NOT move the anchor, so sequential Tier1→Tier2
        # crossings scale against the same morning base. On rebalance
        # days the runner refreshes this anchor to the achieved
        # POST-rebalance gross right after the 09:35 pipeline trades
        # (refresh_daily_book_anchor_after_rebalance) — stop_grid
        # rebalances before its tier checks, so the ~09:30 overnight
        # anchor would be the wrong base for the rest of the day.
        current_book = sum(
            float(p.get("market_value", 0) or 0) for p in positions
        )
        if state.get("daily_book_start_date") != today_iso and current_book > 0:
            state["daily_book_start"] = current_book
            state["daily_book_start_date"] = today_iso
            state_dirty = True
        daily_book_start = float(state.get("daily_book_start") or current_book)

        # ── Portfolio-level soft tiered scaler ───────────────────────
        # None disables the tiers; e.g. -4.0 configured → -8.0 effective
        # on a 2x book.
        pstop = _effective_portfolio_stop(mc)
        if pstop is not None and start_equity > 0 and current_equity > 0:
            daily_drawdown_pct = (current_equity / start_equity - 1) * 100

            if daily_drawdown_pct <= pstop * (7.0 / 3.0):
                # Tier 3 — hard liquidation, preserves original behavior.
                logger.warning(
                    f"[CUTLOSS] {mc.name}: PORTFOLIO STOP TIER 3! "
                    f"Daily drawdown: {daily_drawdown_pct:.2f}% <= "
                    f"{pstop * 7.0 / 3.0:.2f}% "
                    f"(configured {mc.cutloss_portfolio_stop}% × leverage). "
                    f"Liquidating ALL {n_positions} positions "
                    f"(equity=${current_equity:,.2f})."
                )
                state["portfolio_stop_tripped_date"] = today_iso
                state["peak_prices"] = {}
                state["last_rebalance"] = None
                # Re-entry cool-down: wiping last_rebalance alone forced a
                # full re-levered rebuy at the very next session's cron
                # (2026-06-05 → 06-08 → stopped out again 06-09). The
                # runner now waits this many trading days before
                # rebuilding the book.
                reentry_days = int(getattr(mc, "cutloss_reentry_delay_days", 2))
                state["reentry_cooldown_until"] = _add_trading_days(
                    today_iso, reentry_days
                )
                save_state(state, mc)
                _liquidate_all(mc, positions, "portfolio_stop", logger)
                logger.warning(
                    f"[CUTLOSS] {mc.name}: SUMMARY — liquidated all "
                    f"{n_positions} positions, 0/{n_positions} remaining, "
                    f"equity=${current_equity:,.2f}"
                )
                return

            elif daily_drawdown_pct <= pstop * (5.0 / 3.0):
                # Tier 2 — scale to 30% of the day-start gross book
                # (stop_grid TIER_F, NOT 30% of equity). No trip flag.
                n_scaled = _soft_scale_portfolio(
                    mc, positions, daily_book_start,
                    book_fraction=0.30,
                    tier="Tier2", dd_pct=daily_drawdown_pct, logger=logger,
                )
                # Already at/below the tier target on later ticks → DEBUG,
                # not a repeated WARN every 60s.
                log_fn = logger.warning if n_scaled else logger.debug
                log_fn(
                    f"[CUTLOSS] {mc.name}: SOFT SCALE TIER 2 — daily DD "
                    f"{daily_drawdown_pct:+.2f}% <= {pstop * 5.0 / 3.0:+.2f}%; "
                    f"scaled {n_scaled} positions pro-rata to 30% of the "
                    f"day-start book (${daily_book_start:,.2f} gross; "
                    f"stop_grid TIER_F semantics). "
                    f"Trip flag NOT set; trailing stops still active."
                )

            elif daily_drawdown_pct <= pstop:
                # Tier 1 — scale to 60% of the day-start gross book
                # (stop_grid TIER_F, NOT 60% of equity — the 2026-07-15
                # over-sale). No trip flag.
                n_scaled = _soft_scale_portfolio(
                    mc, positions, daily_book_start,
                    book_fraction=0.60,
                    tier="Tier1", dd_pct=daily_drawdown_pct, logger=logger,
                )
                log_fn = logger.warning if n_scaled else logger.debug
                log_fn(
                    f"[CUTLOSS] {mc.name}: SOFT SCALE TIER 1 — daily DD "
                    f"{daily_drawdown_pct:+.2f}% <= {pstop:+.2f}%; "
                    f"scaled {n_scaled} positions pro-rata to 60% of the "
                    f"day-start book (${daily_book_start:,.2f} gross; "
                    f"stop_grid TIER_F semantics). Trip flag NOT set."
                )

        # ── Per-position hard + trailing stops ───────────────────────
        # None (or 0) disables a stop. The 2026-06 calibration ships both
        # disabled: every trailing value tested on the 10-yr backtest cost
        # 0.6-2.3pp/mo, and hard stops were dominated by the portfolio
        # tiers (validation/STOP_GRID_REPORT.md).
        hard_stop = mc.cutloss_hard_stop or None
        trailing_stop = mc.cutloss_trailing_stop or None

        peak_prices = state.setdefault("peak_prices", {})
        held_symbols = {p["symbol"] for p in positions}
        # Prune peaks for positions no longer held.
        for sym in list(peak_prices.keys()):
            if sym not in held_symbols:
                del peak_prices[sym]
                state_dirty = True

        symbols_to_sell: list[tuple[str, float, str, float]] = []

        for p in positions:
            sym = p["symbol"]
            qty = float(p.get("qty", 0))
            avg_entry = float(p.get("avg_entry_price", 0))
            current_price = float(p.get("current_price", 0))

            if avg_entry <= 0 or current_price <= 0 or qty <= 0:
                continue

            prev_peak = peak_prices.get(sym, avg_entry)
            current_peak = max(prev_peak, current_price)
            if peak_prices.get(sym) != current_peak:
                peak_prices[sym] = current_peak
                state_dirty = True

            # Hard stop: down X% from entry
            pct_from_entry = (current_price / avg_entry - 1) * 100
            if hard_stop is not None and pct_from_entry <= hard_stop:
                logger.warning(
                    f"[CUTLOSS] {mc.name}: HARD STOP on {sym}! "
                    f"{pct_from_entry:.2f}% from entry (threshold: {hard_stop}%)"
                )
                symbols_to_sell.append((sym, qty, "hard_stop", pct_from_entry))
                continue

            # Trailing stop: down X% from peak
            pct_from_peak = (current_price / current_peak - 1) * 100
            if trailing_stop is not None and pct_from_peak <= trailing_stop:
                logger.warning(
                    f"[CUTLOSS] {mc.name}: TRAILING STOP on {sym}! "
                    f"{pct_from_peak:.2f}% from peak ${current_peak:.2f} "
                    f"(threshold: {trailing_stop}%)"
                )
                symbols_to_sell.append((sym, qty, "trailing_stop", pct_from_peak))
                continue

        # Execute the sells — risk reduction first, same rule as the
        # soft-scale and Tier-3 paths: every close order goes onto the
        # wire BEFORE any instrumentation network call. The decision-
        # price snapshot used to be fetched HERE, ahead of the sells —
        # up to ~15s of data-API latency on a risk-reduction path. It now
        # runs inside `_instrument_and_journal_sells` after placement, so
        # the "arrival"/open/prev-close reference prices are sampled a
        # few seconds AFTER the decision. That skew is acceptable for
        # slippage instrumentation (it slightly understates measured
        # slippage on a falling tape) and buys back the placement delay.
        #
        # When the fill poll can't resolve a price (rate-limited bursts
        # during exactly the sessions where stops batch), fall back to
        # the position's last market value so the redistribution budget
        # isn't silently zeroed.
        mv_estimate = {p["symbol"]: float(p.get("market_value", 0) or 0)
                       for p in positions}

        sold_symbols: list[str] = []
        sold_proceeds = 0.0
        stop_records: list[TradeRecord] = []
        for sym, qty, reason, pct in symbols_to_sell:
            logger.info(
                f"[CUTLOSS] {mc.name}: SELLING {sym} qty={qty:.2f} "
                f"reason={reason} ({pct:.2f}%)"
            )
            try:
                stop_records.append(
                    _place_position_close(mc, sym, qty, reason, logger)
                )
                sold_symbols.append(sym)
                if peak_prices.pop(sym, None) is not None:
                    state_dirty = True
            except Exception as e:
                logger.error(f"[CUTLOSS] {mc.name}: failed to sell {sym}: {e}")

        # All stop sells placed — bounded, failure-tolerant fill poll +
        # decision-price snapshot + journal, then size redistribution
        # from the backfilled fill notionals.
        if stop_records:
            _instrument_and_journal_sells(
                mc, stop_records, TradeJournal(mc.name), logger
            )
            for record in stop_records:
                notional = float(record.notional_usd or 0.0)
                if notional <= 0 and mv_estimate.get(record.symbol, 0) > 0:
                    notional = mv_estimate[record.symbol]
                    logger.warning(
                        f"[CUTLOSS] {mc.name}: {record.symbol} fill unknown — "
                        f"using market-value estimate ${notional:,.2f} for "
                        f"proceeds"
                    )
                sold_proceeds += notional

        # Ledger of everything stopped out today — redistribution must never
        # re-buy these the same day (on 2026-06-09 seven just-stopped names
        # were re-bought and re-stopped within two hours).
        if sold_symbols:
            ledger = state.get("cutloss_sold_today") or {}
            if ledger.get("date") != today_iso:
                ledger = {"date": today_iso, "symbols": []}
            ledger["symbols"] = sorted(set(ledger["symbols"]) | set(sold_symbols))
            state["cutloss_sold_today"] = ledger
            state_dirty = True

        if state_dirty:
            save_state(state, mc)

    # Summary + redistribute (outside the lock — pure logging + new API calls)
    if sold_symbols:
        remaining = n_positions - len(sold_symbols)
        logger.info(
            f"[CUTLOSS] {mc.name}: SUMMARY — sold {len(sold_symbols)} "
            f"({', '.join(sold_symbols)}), {remaining}/{n_positions} positions remaining, "
            f"equity=${current_equity:,.2f}"
        )
        try:
            _redistribute_after_cutloss(
                mc, sold_symbols, logger, top_n=top_n, proceeds=sold_proceeds
            )
        except Exception as e:
            logger.error(f"[CUTLOSS] {mc.name}: redistribution failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Redistribute freed cash into model's next-best picks (NOT into survivors).
# ═══════════════════════════════════════════════════════════════════════════

def _redistribute_after_cutloss(mc: "ModelConfig", sold_symbols: list[str],
                                logger, top_n: int = 20,
                                proceeds: float | None = None) -> None:
    """Replace sold positions with the model's next-best picks, sized from
    the actual stop-sale proceeds and capped at each name's target weight.

    Flow:
      1. Wait briefly for sells to settle.
      2. Skip-guard: if drawdown is already within 1.5pp of the (leverage-
         scaled) portfolio_stop, do nothing — the next 60s scan will likely
         liquidate anyway.
      3. Look up the model's last run from state. Direct-weights strategies
         store their ranking under "weights"; ML rankers under
         "predictions" — read whichever exists (the old code read only
         "predictions", so combo_v2 always fell through to arbitrary
         set-order picks).
      4. Pick replacements (top-ranked, not held, not stopped out today).
      5. Size each buy as its share of THIS scan's sale proceeds, capped at
         the name's own target allocation (equity × weight × leverage).
         The old sizing — (total account cash − 1% buffer) ÷ (held +
         replacements), buying only the replacements — concentrated 30% of
         equity into one stock on 2026-06-10 and grew per-buy size as the
         book shrank.
      6. Any leftover cash sits as cash until the next scheduled rebalance
         (the legacy topup-into-survivors path was removed after the
         2026-05-01 / 2026-05-07 cascades).
    """
    time.sleep(2)  # let sells settle on Alpaca's side

    try:
        positions = alpaca_request("GET", "v2/positions", mc, logger=logger)
        account = alpaca_request("GET", "v2/account", mc, logger=logger)
    except Exception as e:
        logger.error(f"[REDISTRIBUTE] {mc.name}: failed to fetch positions/account: {e}")
        return

    # Skip-guard — see docstring. Widened from 1.0pp to 1.5pp after the
    # cascades showed the brake firing too late. Uses the leverage-scaled
    # stop so the guard and the tier scaler measure the same thing, and
    # scales the buffer by the same factor (a fixed 1.5pp would be only
    # 0.75pp of underlying move at 2x — half the intended brake).
    pstop_eff = _effective_portfolio_stop(mc)
    try:
        current_eq = float(account.get("equity", 0))
        last_eq = float(account.get("last_equity", 0))
    except (TypeError, ValueError):
        current_eq = last_eq = 0
    if pstop_eff is not None and current_eq > 0 and last_eq > 0:
        drawdown_pct = (current_eq / last_eq - 1) * 100
        buffer_scale = (max(1.0, float(getattr(mc, "target_leverage", 1.0) or 1.0))
                        if getattr(mc, "cutloss_scale_by_leverage", True) else 1.0)
        buffer_pp = 1.5 * buffer_scale
        skip_threshold = pstop_eff + buffer_pp
        if drawdown_pct <= skip_threshold:
            logger.warning(
                f"[REDISTRIBUTE] {mc.name}: SKIPPED — daily drawdown {drawdown_pct:+.2f}% "
                f"is within {buffer_pp:.1f}pp of portfolio_stop ({pstop_eff:+.1f}% effective); "
                f"funnelling cash into a sinking portfolio is counterproductive."
            )
            return

    cash = float(account.get("cash", 0))
    buying_power = float(account.get("buying_power", 0))
    available = min(cash, buying_power)
    equity = float(account.get("equity", 0))
    buffer = equity * 0.01  # keep 1% of equity in cash to avoid over-allocating
    available = max(0, available - buffer)

    # Deploy only what THIS scan's stop-sales freed up, never the whole
    # cash pile (which may hold days of accumulated stop proceeds that the
    # next scheduled rebalance owns). Fail-closed: a caller that can't
    # attest proceeds gets a zero budget, not the cash pile.
    budget = max(0.0, min(available, float(proceeds or 0.0)))

    if budget < 50:
        logger.info(f"[REDISTRIBUTE] {mc.name}: only ${budget:.2f} available, skipping")
        return

    held_symbols: set[str] = set()
    for p in positions:
        sym = p["symbol"]
        if sym not in sold_symbols:
            held_symbols.add(sym)

    n_to_replace = len(sold_symbols)

    state = load_state(mc)
    history = state.get("history", [])

    # Never re-buy a name that any scan stopped out today — not just the
    # ones from this scan.
    today_iso = datetime.now().strftime("%Y-%m-%d")
    ledger = state.get("cutloss_sold_today") or {}
    stopped_today = set(ledger.get("symbols", [])) if ledger.get("date") == today_iso else set()
    excluded = held_symbols | stopped_today | set(sold_symbols)

    leverage = max(1.0, float(getattr(mc, "target_leverage", 1.0) or 1.0))
    replacements: list[tuple[str, float]] = []
    weight_map: dict[str, float] = {}

    if history:
        latest = history[-1]
        # Target sizing per name: conviction_weights are the run's actual
        # target weights (pre-leverage fractions of equity); "weights" is
        # the direct-weights ranking (same fractions); "predictions" are
        # ML scores (ranking only, no sizing meaning).
        weight_map = (latest.get("conviction_weights")
                      or latest.get("weights") or {})
        ranking_map = weight_map or latest.get("predictions") or {}
        ranked = sorted(ranking_map.items(), key=lambda x: x[1], reverse=True)

        for sym, score in ranked:
            if sym in excluded:
                continue
            replacements.append((sym, float(score)))
            if len(replacements) >= n_to_replace:
                break

    if not replacements:
        logger.info(
            f"[REDISTRIBUTE] {mc.name}: no eligible replacement candidates "
            f"(not held, not stopped today, present in the last run's ranking); "
            f"${budget:.2f} stays in cash until the next rebalance."
        )
        return

    logger.info(
        f"[REDISTRIBUTE] {mc.name}: replacing {n_to_replace} sold position(s) "
        f"with {len(replacements)} new: {', '.join(r[0] for r in replacements)} "
        f"(budget ${budget:.2f} from stop proceeds)"
    )

    journal = TradeJournal(mc.name)
    n_bought = 0
    deployed = 0.0

    # Equal-weight fallback cap when the run carries no weight info
    # (legacy ML run with target_weights=None).
    n_book = max(len(held_symbols) + n_to_replace, top_n, 1)
    fallback_w = 1.0 / n_book

    share = budget / len(replacements)
    for sym, score in replacements:
        # Cap at the strategy's own target allocation for this name —
        # redistribution must never build a bigger position than the
        # rebalance itself would.
        target_cap = equity * weight_map.get(sym, fallback_w) * leverage
        alloc = round(min(share, target_cap), 2)
        if alloc < 10:
            continue
        bought = _place_redistribute_buy(
            mc, sym, alloc, "replacement", score, journal, logger
        )
        n_bought += bought
        if bought:
            deployed += alloc

    leftover = available - deployed

    # NOTE: prior versions topped up surviving positions with leftover.
    # That created the redistribute death spiral. Leftover cash now sits
    # idle until the next scheduled rebalance.
    if leftover > 50:
        logger.info(
            f"[REDISTRIBUTE] {mc.name}: ${leftover:.2f} held as cash until next "
            f"rebalance (topup-existing disabled to prevent procyclical averaging "
            f"down on losers)."
        )

    logger.info(
        f"[REDISTRIBUTE] {mc.name}: completed — {n_bought} buys placed, "
        f"${deployed:.2f} deployed (${leftover:.2f} held cash), "
        f"{len(held_symbols) + len(replacements)}/{top_n} target positions"
    )


def _place_redistribute_buy(mc: "ModelConfig", symbol: str, alloc: float,
                            action: str, score: float,
                            journal, logger) -> int:
    """Place a single redistribution buy order. Returns 1 on success, 0 on failure."""
    order_data = {
        "symbol": _to_alpaca_symbol(symbol),
        "notional": str(alloc),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }

    try:
        result = alpaca_request("POST", "v2/orders", mc, data=order_data, logger=logger)
        order_id = result.get("id", "?")
        order_status = result.get("status", "submitted")
        label = f"NEW {symbol}" if action == "replacement" else f"TOP-UP {symbol}"
        logger.info(
            f"[REDISTRIBUTE] {mc.name}: {label} ${alloc:.2f}"
            f"{f' (score={score:.4f})' if score else ''} → {order_status}"
        )

        fill_price = None
        filled_qty = 0
        if order_id and order_id != "?" and order_status not in ("rejected", "canceled", "expired"):
            try:
                final = poll_order_status(order_id, mc, logger)
                order_status = final.get("status", order_status)
                fq = final.get("filled_qty")
                fp = final.get("filled_avg_price")
                if fq and fq != "0":
                    filled_qty = float(fq)
                if fp and fp != "0":
                    fill_price = float(fp)
            except Exception:
                pass

        record = TradeRecord(
            trade_id=f"{mc.name}_redist_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}",
            run_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            model=mc.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            side="buy",
            action=f"redistribute_{action}",
            order_type="market",
            time_in_force="day",
            notional_usd=alloc,
            order_status=order_status,
            order_id=order_id,
            shares=filled_qty,
            fill_price=fill_price,
        )
        journal.log_trade(record)
        return 1

    except Exception as e:
        logger.error(f"[REDISTRIBUTE] {mc.name}: failed to buy {symbol}: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Per-position sell + portfolio-level liquidation + soft pro-rata scaling.
# ═══════════════════════════════════════════════════════════════════════════

def _execute_cutloss_sell(mc: "ModelConfig", symbol: str, qty: float,
                          reason: str, pct: float, logger,
                          snap: dict | None = None) -> float:
    """Execute a market sell to close a position triggered by a cut-loss rule.

    LEGACY — no longer on the scan path. The hard/trailing stop path now
    places all closes first via `_place_position_close` and instruments
    them afterwards with `_instrument_and_journal_sells` (risk reduction
    before any instrumentation network call). Retained because
    pipeline.py re-exports it and it remains a correct single-symbol
    close+journal primitive for manual/operator use.

    Uses `DELETE /v2/positions/<symbol>` which closes the entire position
    atomically — handles fractional shares cleanly and never half-fills.
    Caller owns peak-price tracking; we deliberately do not touch state here.

    `snap` is the pre-fetched decision-price snapshot for this symbol
    ({"arrival", "day_open", "prev_close"} from `fetch_snapshots`); when
    provided the journal row carries decision_price / reference_open /
    prev_close and slippage_bps (sell sign convention, like core.orders).
    Instrumentation is best-effort — a missing/failed snapshot leaves the
    fields None and never affects the sell.

    Returns the filled notional in dollars (0.0 if the fill is unknown) so
    the caller can size redistribution from actual proceeds.
    """
    logger.info(
        f"[CUTLOSS] {mc.name}: SELLING {symbol} qty={qty:.2f} "
        f"reason={reason} ({pct:.2f}%)"
    )

    alpaca_sym = _to_alpaca_symbol(symbol)
    try:
        result = alpaca_request(
            "DELETE", f"v2/positions/{alpaca_sym}", mc, logger=logger,
        ) or {}
        order_id = result.get("id", "?")
        order_status = result.get("status", "accepted")

        if order_status in ("rejected", "canceled", "expired"):
            logger.error(
                f"[CUTLOSS] {mc.name}: {symbol} close order {order_status}: "
                f"{result.get('reject_reason', 'unknown')}"
            )
        else:
            logger.info(
                f"[CUTLOSS] {mc.name}: {symbol} close order placed: {order_id}, "
                f"status={order_status}"
            )

        fill_price = None
        filled_qty = qty
        if order_id and order_id != "?" and order_status not in ("rejected", "canceled", "expired"):
            try:
                final = poll_order_status(order_id, mc, logger)
                order_status = final.get("status", order_status)
                fq = final.get("filled_qty")
                fp = final.get("filled_avg_price")
                if fq and fq != "0":
                    filled_qty = float(fq)
                if fp and fp != "0":
                    fill_price = float(fp)
                logger.info(
                    f"[CUTLOSS] {mc.name}: {symbol} final status={order_status}, "
                    f"filled_qty={filled_qty}, fill_price={fill_price}"
                )
            except Exception:
                pass

        # Notional for the journal: Alpaca's `DELETE /v2/positions/<sym>`
        # doesn't return a notional, so we compute it ourselves from the
        # post-poll fill data. Falls back to 0 if either is unknown.
        notional = round(filled_qty * fill_price, 2) if (filled_qty and fill_price) else 0.0

        journal = TradeJournal(mc.name)
        record = TradeRecord(
            trade_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}",
            run_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            model=mc.name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            side="sell",
            action=reason,
            order_type="market",
            time_in_force="day",
            notional_usd=notional,
            order_status=order_status,
            order_id=order_id,
            error_message=result.get("reject_reason") if order_status in ("rejected", "canceled", "expired") else None,
            shares=filled_qty,
            fill_price=fill_price,
        )
        _stamp_slippage(record, snap)
        journal.log_trade(record)
        return notional

    except Exception as e:
        logger.error(f"[CUTLOSS] {mc.name}: order failed for {symbol}: {e}")
        raise


def _stamp_slippage(record: TradeRecord, snap: dict | None) -> None:
    """Stamp decision prices + slippage on a journal record.

    Same sign convention as core.orders: +1 buy / -1 sell, so positive
    bps = execution cost either direction (a sell filling ABOVE the
    reference is favorable → negative). Best-effort — any failure leaves
    the fields None (readers treat None as "not captured", never as
    zero slippage).
    """
    try:
        snap = snap or {}
        if record.decision_price is None:
            record.decision_price = snap.get("arrival")
        if record.reference_open is None:
            record.reference_open = snap.get("day_open")
        if record.prev_close is None:
            record.prev_close = snap.get("prev_close")
        if record.fill_price:
            sign = 1.0 if record.side == "buy" else -1.0
            if record.reference_open:
                record.slippage_bps = round(
                    sign * (record.fill_price / record.reference_open - 1.0)
                    * 1e4, 2)
            if record.decision_price:
                record.slippage_vs_arrival_bps = round(
                    sign * (record.fill_price / record.decision_price - 1.0)
                    * 1e4, 2)
    except Exception:
        pass  # instrumentation must never break the journal row


def _place_position_close(mc: "ModelConfig", symbol: str, qty: float,
                          reason: str, logger) -> TradeRecord:
    """Place `DELETE /v2/positions/<sym>` and build its journal record
    WITHOUT polling or journaling.

    Batch risk-reduction paths (Tier-3 liquidation and the per-position
    hard/trailing stop batch) must get every close order onto the wire
    before any instrumentation network call —
    `_instrument_and_journal_sells` finishes the records afterwards.
    Raises if the close request itself fails (caller logs and continues
    with the remaining positions).
    """
    alpaca_sym = _to_alpaca_symbol(symbol)
    result = alpaca_request(
        "DELETE", f"v2/positions/{alpaca_sym}", mc, logger=logger,
    ) or {}
    order_id = result.get("id", "?")
    order_status = result.get("status", "accepted")

    if order_status in ("rejected", "canceled", "expired"):
        logger.error(
            f"[CUTLOSS] {mc.name}: {symbol} close order {order_status}: "
            f"{result.get('reject_reason', 'unknown')}"
        )
    else:
        logger.info(
            f"[CUTLOSS] {mc.name}: {symbol} close order placed: {order_id}, "
            f"status={order_status}"
        )

    return TradeRecord(
        trade_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}",
        run_id=f"{mc.name}_cutloss_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        model=mc.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol=symbol,
        side="sell",
        action=reason,
        order_type="market",
        time_in_force="day",
        notional_usd=0.0,  # backfilled from the fill in the instrument pass
        order_status=order_status,
        order_id=order_id,
        error_message=result.get("reject_reason") if order_status in ("rejected", "canceled", "expired") else None,
        shares=qty,
    )


def _instrument_and_journal_sells(mc: "ModelConfig", records: list,
                                  journal, logger,
                                  total_budget: float = 25.0) -> None:
    """Best-effort fill + decision-price capture for ALREADY-PLACED stop
    sells, then journal every record.

    WHY: the 2026-07-15 soft-scale rows were journalled straight from the
    submit response (order_status=pending_new, fill_price=None) — the
    day's stop trades were blind. This pass runs strictly AFTER every
    order is placed (risk reduction first) and adds:
      1. one batched `fetch_snapshots` call → decision_price /
         reference_open / prev_close,
      2. a fill poll per order under a shared deadline (like the
         core.orders rebalance poll, but budgeted for the batch),
      3. slippage stamping via `_stamp_slippage` (sign -1 for sells).

    NOTE: the snapshot here runs AFTER the sells are placed, so the
    reference prices carry a few seconds of skew vs the true decision
    moment — acceptable for slippage instrumentation, and the price of
    never delaying a risk-reduction order (2026-07 review, finding 2).

    Latency: bounded by `total_budget` — HARD wall-clock enforcement.
    The deadline is checked before every per-order poll; each poll gets
    only the remaining budget with `final_fetch=False` (a timed-out poll
    can never add a trailing status request), and once the budget is
    exhausted the remaining records skip polling entirely. Residual
    overshoot is bounded by one in-flight HTTP request (15s socket
    timeout) started just before the deadline. This matters because the
    caller holds the global cutloss state lock for the duration —
    typically <2s in practice, since market sells fill almost instantly
    and the snapshot is one request.

    Every step is failure-tolerant: on any error the fields stay None
    and the row is journalled anyway.
    """
    if not records:
        return
    deadline = time.monotonic() + total_budget

    snapshots: dict[str, dict] = {}
    try:
        syms = sorted({r.symbol for r in records})
        snapshots = fetch_snapshots(syms, mc, logger) or {}
    except Exception as e:
        logger.warning(
            f"[CUTLOSS] {mc.name}: decision-price snapshot failed: {e} — "
            f"journaling sells without price fields"
        )
        snapshots = {}

    budget_exhausted = False
    for i, record in enumerate(records):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not budget_exhausted:
                    budget_exhausted = True
                    logger.warning(
                        f"[CUTLOSS] {mc.name}: instrumentation budget "
                        f"({total_budget:.0f}s) exhausted — journaling the "
                        f"remaining {len(records) - i} sell(s) without fill "
                        f"polling (orders were already placed)."
                    )
            elif (record.order_id and record.order_id != "?"
                    and record.order_status not in (
                        "rejected", "canceled", "expired", "failed")):
                final = poll_order_status(record.order_id, mc, logger,
                                          max_wait=min(remaining, 8.0),
                                          final_fetch=False)
                record.order_status = final.get("status", record.order_status)
                fq = final.get("filled_qty")
                fp = final.get("filled_avg_price")
                if fq and fq != "0":
                    record.shares = float(fq)
                if fp and fp != "0":
                    record.fill_price = float(fp)
                # `DELETE /v2/positions` responses carry no notional —
                # backfill it from the fill so downstream stats aren't 0.
                if (not record.notional_usd and record.shares
                        and record.fill_price):
                    record.notional_usd = round(
                        record.shares * record.fill_price, 2)
        except Exception:
            pass  # instrumentation must never block the journal row
        _stamp_slippage(record, snapshots.get(record.symbol))
        try:
            journal.log_trade(record)
        except Exception:
            pass  # journal failure shouldn't block the remaining records


def _liquidate_all(mc: "ModelConfig", positions: list, reason: str, logger) -> None:
    """Liquidate every position for a model (Tier 3 portfolio stop).

    Caller is responsible for wiping `state["peak_prices"]` and persisting
    the trip flag *before* calling this — so the flag + empty peak dict
    are committed atomically even if a sell fails midway through.

    Risk reduction comes first: every close order is placed before ANY
    instrumentation network call. Fill polling + decision-price capture
    run afterwards in one bounded, failure-tolerant pass (the old
    per-symbol place-then-poll loop could stall up to 8s between closes
    during exactly the sessions where liquidations batch).
    """
    logger.warning(
        f"[CUTLOSS] {mc.name}: LIQUIDATING ALL {len(positions)} positions ({reason})"
    )
    journal = TradeJournal(mc.name)
    records: list[TradeRecord] = []
    for p in positions:
        sym = p["symbol"]
        qty = float(p.get("qty", 0))
        if qty > 0:
            try:
                records.append(
                    _place_position_close(mc, sym, qty, reason, logger)
                )
            except Exception as e:
                logger.error(f"[CUTLOSS] {mc.name}: failed to liquidate {sym}: {e}")
    _instrument_and_journal_sells(mc, records, journal, logger)


def _soft_scale_portfolio(mc: "ModelConfig", positions: list,
                          daily_book_start: float, book_fraction: float,
                          tier: str, dd_pct: float, logger) -> int:
    """Scale gross exposure down to `book_fraction` of the DAY-START book
    by selling pro-rata across positions. Used by Tier 1 (0.60) and
    Tier 2 (0.30) of the soft portfolio stop.

    Validated semantics (validation/stop_grid.py, TIER_F={1:0.6, 2:0.3,
    3:0.0}): each tier crossing scales the book to `f` of the MORNING
    gross (`w = w_morning * f`), so sequential same-day crossings sell
    down against the same anchor — NOT `f` × current equity and NOT `f` ×
    the already-scaled book. WHY: on 2026-07-15 the old
    `target_mv = equity × f` rule over-sold — book ~$150k gross, equity
    ~$79k at the Tier-1 trigger → live sold ~$98k where the validated
    rule sells ~$60k (0.4 × the morning book).

    `daily_book_start` is the scanner's day-anchored gross book; a
    missing/zero anchor (mid-day restart before the anchor logic ran)
    falls back to the current book.

    Each position is partial-sold using `notional` market orders so the
    portfolio's relative weights stay roughly intact (no "let the winners
    run" bias against the model's intended shape).

    Risk reduction comes first: every sell order is placed before ANY
    instrumentation network call; fill polling + decision-price capture
    then run in one bounded, failure-tolerant pass (see
    `_instrument_and_journal_sells` — ~25s worst-case added tick latency).

    Returns the count of positions partially-or-fully sold.
    """
    if not positions:
        return 0
    total_mv = sum(float(p.get("market_value", 0) or 0) for p in positions)
    if total_mv <= 0:
        return 0
    book_base = (daily_book_start
                 if daily_book_start and daily_book_start > 0 else total_mv)
    target_mv = book_base * book_fraction
    excess_mv = total_mv - target_mv
    if excess_mv <= 0:
        return 0  # already at/below the tier target (idempotent re-scans)

    fraction_to_sell = min(max(excess_mv / total_mv, 0.0), 1.0)
    journal = TradeJournal(mc.name)
    reason = f"soft_scale_{tier.lower()}"
    records: list[TradeRecord] = []

    # Phase 1 — place EVERY sell order (no instrumentation network calls
    # until the whole batch is on the wire).
    for p in positions:
        sym = p["symbol"]
        mv = float(p.get("market_value", 0) or 0)
        if mv <= 0:
            continue
        sell_notional = round(mv * fraction_to_sell, 2)
        if sell_notional < 5.0:
            continue  # skip dust slices
        alpaca_sym = _to_alpaca_symbol(sym)
        try:
            resp = alpaca_request(
                "POST", "v2/orders", mc,
                data={
                    "symbol": alpaca_sym,
                    "notional": sell_notional,
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "day",
                },
                logger=logger,
            ) or {}
            status = resp.get("status", "submitted")
            logger.info(
                f"[CUTLOSS] {mc.name}: {tier} partial-sell {sym} "
                f"${sell_notional:.2f} ({fraction_to_sell*100:.1f}% of "
                f"position) reason={reason} status={status}"
            )
            ts = datetime.now(timezone.utc)
            records.append(TradeRecord(
                trade_id=f"{mc.name}_softscale_{ts.strftime('%Y%m%d_%H%M%S')}_{sym}",
                run_id=f"{mc.name}_softscale_{ts.strftime('%Y%m%d')}",
                model=mc.name,
                timestamp=ts.isoformat(),
                symbol=sym,
                side="sell",
                action=reason,
                order_type="market",
                time_in_force="day",
                notional_usd=sell_notional,
                order_id=resp.get("id"),
                order_status=status,
                current_price=float(p.get("current_price", 0) or 0),
            ))
        except Exception as e:
            logger.error(
                f"[CUTLOSS] {mc.name}: {tier} partial-sell failed for {sym}: {e}"
            )

    # Phase 2 — all sells placed: best-effort fill poll + decision-price
    # snapshot, then journal (rows are written even if every capture
    # step fails — never blocks or precedes risk reduction).
    _instrument_and_journal_sells(mc, records, journal, logger)

    n_sold = len(records)
    logger.info(
        f"[CUTLOSS] {mc.name}: {tier} scaled {n_sold} positions pro-rata "
        f"(sold ~{fraction_to_sell*100:.1f}% of each, ${excess_mv:,.2f} "
        f"excess over {book_fraction*100:.0f}% of the day-start book "
        f"${book_base:,.2f} — stop_grid TIER_F), DD was {dd_pct:+.2f}%."
    )
    return n_sold
