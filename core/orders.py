"""Order placement, rebalancing, and post-submission polling.

Three entry points:

  `fetch_inactive_assets(symbols, mc, logger, top_n)` — Alpaca asset
  status check on the top N candidates so we don't try to buy delisted
  or non-tradeable names.

  `poll_order_status(order_id, mc, logger)` — wait up to ~8s for a
  market order to reach a terminal state (filled/rejected/etc) so the
  trade journal records real fill prices/quantities.

  `rebalance_portfolio(...)` — the daily rebalance flow:
  fetches account + positions, computes per-symbol deltas, places
  sell/buy notional orders for each non-zero delta, polls for fills,
  appends to the trade journal. Driven by `target_symbols` and
  `target_weights` (conviction sizing) — equal-weight is just the
  fallback if `target_weights` is None.

Execution style (2026-07-25): `ModelConfig.exec_style` selects how
REBALANCE orders are placed. "market" (default) is the legacy path,
byte-identical to before. "marketable_limit" places LIMIT day orders at
arrival*(1 + buffer) for buys / *(1 - buffer) for sells (arrival from the
same `fetch_snapshots` decision-price capture), polls up to
`exec_fill_timeout_s`, then applies `exec_timeout_action` per unfilled
order. Cutloss/scanner sells and `liquidate_all_positions` ALWAYS stay
MARKET — risk reduction is never delayed. Every limit-path failure for a
symbol falls back to a market order for that symbol.

All Alpaca I/O goes through `core.alpaca.alpaca_request`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from core.alpaca import (
    _make_alpaca_headers, _to_alpaca_symbol, alpaca_request,
    fetch_snapshots, get_account, get_positions,
)
from core.journal import TradeRecord

if TYPE_CHECKING:
    from core.config import ModelConfig
    from core.journal import TradeJournal
    from core.run_report import RunReport


# Alpaca order states after which no further fills can arrive.
_TERMINAL_ORDER_STATES = {"filled", "canceled", "expired", "rejected",
                          "suspended", "replaced"}


@dataclass
class ExecTradeRecord(TradeRecord):
    """TradeRecord + marketable-limit execution telemetry.

    Defined HERE (not core.journal) so the default market path keeps
    journaling plain TradeRecord rows byte-identically. Only slots with
    exec_style="marketable_limit" emit this wider schema;
    `TradeJournal.log_trade` serializes subclasses via asdict() and its
    CSV schema-roll handles the wider header.
    """
    exec_style: Optional[str] = None            # "marketable_limit" | "market" (fallback)
    limit_price: Optional[float] = None         # submitted limit price (None for market)
    timed_out: Optional[bool] = None            # unfilled at exec_fill_timeout_s
    replaced_to_market: Optional[bool] = None   # remainder re-sent as a market order
    replacement_order_id: Optional[str] = None  # Alpaca id of the replacement order


def _limit_price_for(side: str, arrival: float, buffer_bps: float) -> float:
    """Marketable-limit price: arrival plus `buffer_bps` in the crossing
    direction (buys above arrival, sells below), rounded to the penny
    TOWARD marketability (buys up, sells down) so rounding never makes
    the order less likely to fill. Floors at $0.01 — Alpaca rejects
    non-positive limit prices.
    """
    frac = float(buffer_bps) / 1e4
    raw = arrival * (1.0 + frac) if side == "buy" else arrival * (1.0 - frac)
    cents = round(raw * 100.0, 6)  # shave float artifacts pre-ceil/floor
    lp = (math.ceil(cents) if side == "buy" else math.floor(cents)) / 100.0
    return max(lp, 0.01)


def _apply_order_fill(trade: TradeRecord, order_state: dict) -> None:
    """Copy status / filled qty / filled avg price from an Alpaca order
    payload onto a trade record (same semantics as the market-path poll)."""
    trade.order_status = order_state.get("status", trade.order_status)
    fq = order_state.get("filled_qty")
    fp = order_state.get("filled_avg_price")
    if fq and fq != "0":
        trade.shares = float(fq)
    if fp and fp != "0":
        trade.fill_price = float(fp)


def _compute_slippage(trade: TradeRecord) -> None:
    """Slippage vs the backtest's assumed fill (same-day open) and vs the
    arrival price. Sign convention: +1 buy / -1 sell, so positive bps =
    execution cost either direction (a sell filling ABOVE the reference
    is favorable → negative). No-op when fill_price is missing."""
    try:
        if trade.fill_price:
            sign = 1.0 if trade.side == "buy" else -1.0
            if trade.reference_open:
                trade.slippage_bps = round(
                    sign * (trade.fill_price / trade.reference_open - 1.0)
                    * 1e4, 2)
            if trade.decision_price:
                trade.slippage_vs_arrival_bps = round(
                    sign * (trade.fill_price / trade.decision_price - 1.0)
                    * 1e4, 2)
    except Exception:
        pass


def _cancel_order_and_get_final(trade: TradeRecord, mc: "ModelConfig",
                                logger) -> dict:
    """Cancel an open order and return its final state (fills applied to
    `trade`). A failed cancel (e.g. the order filled in the race) is not
    an error — the follow-up poll reports whatever actually happened."""
    try:
        alpaca_request("DELETE", f"v2/orders/{trade.order_id}", mc,
                       logger=logger)
    except Exception as e:
        logger.warning(
            f"    Cancel request failed for {trade.symbol} "
            f"({trade.order_id}): {e} — polling final state anyway")
    final = poll_order_status(trade.order_id, mc, logger, max_wait=5.0)
    _apply_order_fill(trade, final)
    return final


def _cancel_and_market_replace(trade: TradeRecord, order: dict,
                               mc: "ModelConfig", logger) -> bool:
    """Timeout action "market": cancel the unfilled limit order and re-send
    the unfilled remainder as a MARKET order. Returns True iff a
    replacement order was actually submitted (False when the limit order
    filled during the cancel race or the remainder is negligible)."""
    _cancel_order_and_get_final(trade, mc, logger)
    if trade.order_status == "filled":
        return False  # filled while we were cancelling — nothing to replace

    prev_qty = float(trade.shares or 0.0)
    prev_px = float(trade.fill_price or 0.0)
    alpaca_sym = _to_alpaca_symbol(trade.symbol)

    if order["action"] == "sell":
        # Full exit: DELETE closes whatever position remains after any
        # partial limit fills — exactly the unfilled remainder.
        resp = alpaca_request(
            "DELETE", f"v2/positions/{alpaca_sym}", mc, logger=logger) or {}
    else:
        filled_notional = prev_qty * prev_px
        remainder = round(float(order["notional"]) - filled_notional, 2)
        if remainder < 1.0:
            return False  # effectively filled; nothing worth replacing
        resp = alpaca_request("POST", "v2/orders", mc, {
            "symbol": alpaca_sym,
            "notional": remainder,
            "side": order["side"],
            "type": "market",
            "time_in_force": "day",
        }, logger=logger) or {}

    trade.replaced_to_market = True
    trade.replacement_order_id = resp.get("id")
    logger.info(
        f"    TIMEOUT->MKT {trade.symbol}: limit unfilled at timeout, "
        f"remainder replaced as market (order_id={trade.replacement_order_id})")

    if trade.replacement_order_id:
        final = poll_order_status(trade.replacement_order_id, mc, logger)
        mq = float(final.get("filled_qty") or 0.0)
        mp = float(final.get("filled_avg_price") or 0.0)
        if mq > 0:
            total = prev_qty + mq
            trade.shares = total
            if mp > 0:
                if prev_qty > 0 and prev_px > 0:
                    # Blend partial limit fill with the market remainder.
                    trade.fill_price = round(
                        (prev_qty * prev_px + mq * mp) / total, 6)
                else:
                    trade.fill_price = mp
        trade.order_status = final.get("status", trade.order_status)
    return True


def _resolve_limit_orders(limit_pending: list[tuple[TradeRecord, dict]],
                          mc: "ModelConfig", logger,
                          timeout_s: float, timeout_action: str) -> dict:
    """Poll submitted marketable-limit orders until all reach a terminal
    state or `timeout_s` elapses, then apply `timeout_action` per order
    still unfilled ("market" = cancel-and-market-replace the remainder;
    anything else = cancel and leave unfilled — reconciliation reports the
    shortfall). Failure-tolerant per order: an error resolving one order
    never blocks the others.

    Returns {"n_timeout_replaced": int, "n_unfilled_cancelled": int}.
    """
    n_timeout_replaced = 0
    n_unfilled_cancelled = 0
    deadline = time.monotonic() + max(float(timeout_s or 0), 0.0)
    pending = list(limit_pending)

    while pending:
        still_open: list[tuple[TradeRecord, dict]] = []
        for trade, order in pending:
            try:
                state = alpaca_request("GET", f"v2/orders/{trade.order_id}", mc)
                _apply_order_fill(trade, state)
                if state.get("status") not in _TERMINAL_ORDER_STATES:
                    still_open.append((trade, order))
            except Exception:
                still_open.append((trade, order))  # retry until deadline
        pending = still_open
        if not pending:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(2.0, remaining))

    if pending:
        logger.warning(
            f"  [LIMIT] {len(pending)} order(s) unfilled at "
            f"{timeout_s}s timeout — action={timeout_action}")
    for trade, order in pending:
        trade.timed_out = True
        try:
            if timeout_action == "market":
                if _cancel_and_market_replace(trade, order, mc, logger):
                    n_timeout_replaced += 1
            else:
                _cancel_order_and_get_final(trade, mc, logger)
                if trade.order_status != "filled":
                    n_unfilled_cancelled += 1
                    logger.warning(
                        f"    TIMEOUT CANCEL {trade.symbol}: limit order "
                        f"cancelled unfilled (status={trade.order_status}) — "
                        f"shortfall reported in reconciliation")
        except Exception as e:
            logger.warning(
                f"    Timeout action failed for {trade.symbol}: {e} — "
                f"order left as-is (status={trade.order_status})")

    return {"n_timeout_replaced": n_timeout_replaced,
            "n_unfilled_cancelled": n_unfilled_cancelled}


def fetch_inactive_assets(symbols: list[str], mc: "ModelConfig",
                          logger, top_n: int = 20) -> set[str]:
    """Check which of the top `2*top_n` candidate symbols are inactive on
    Alpaca. Returns the set to drop before order placement.

    Only the top candidates are checked to keep API calls bounded — a
    delisted name 50 ranks down doesn't matter, since we won't buy it.
    """
    import requests
    inactive: set[str] = set()
    headers = _make_alpaca_headers(mc)
    check_count = min(len(symbols), top_n * 2)
    checked = 0
    for sym in symbols[:check_count]:
        try:
            url = f"{mc.alpaca_base_url}/v2/assets/{_to_alpaca_symbol(sym)}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                asset = resp.json()
                if asset.get("status") != "active" or not asset.get("tradable", True):
                    inactive.add(sym)
                    logger.info(
                        f"    Filtered {sym}: status={asset.get('status')}, "
                        f"tradable={asset.get('tradable')}"
                    )
            elif resp.status_code == 404:
                inactive.add(sym)
                logger.info(f"    Filtered {sym}: not found on Alpaca")
            # else: API error — keep the symbol, it will fail at order time
            # with proper handling.
            checked += 1
        except Exception as e:
            logger.warning(f"    Asset check failed for {sym}: {e}")
            # On timeout/error, keep the symbol rather than blocking
            checked += 1
    if inactive:
        logger.info(
            f"  Inactive asset filter: removed {len(inactive)} of {checked} "
            f"checked ({', '.join(sorted(inactive))})"
        )
    else:
        logger.info(
            f"  Inactive asset filter: all {checked} checked symbols are active"
        )
    return inactive


def poll_order_status(order_id: str, mc: "ModelConfig", logger,
                      max_wait: float = 8.0, interval: float = 0.5,
                      final_fetch: bool = True) -> dict:
    """Poll Alpaca for final order status (filled/rejected/etc).

    Most market orders fill in <1s but secondary-venue routing during
    volatile minutes can take several seconds. We wait up to `max_wait`
    WALL-CLOCK seconds before giving up so the trade journal records real
    fill prices/quantities. The deadline is measured with a monotonic
    clock and includes HTTP time — the old accounting summed only the
    sleeps, so a slow API stretched an "8s" poll far past its budget
    (core.risk holds its state lock across these polls). Polling backs
    off after the first 4 attempts to limit API calls.

    `final_fetch=True` (default, the legacy behavior) makes one last
    status request after the deadline so callers get the freshest
    non-terminal state. Budget-bounded callers
    (`core.risk._instrument_and_journal_sells`) pass False so a
    timed-out poll can never add another network round-trip; they get
    the last successfully polled state instead.
    """
    terminal_states = {"filled", "canceled", "expired", "rejected",
                       "suspended", "replaced"}
    start = time.monotonic()
    attempts = 0
    last_order: dict | None = None
    while time.monotonic() - start < max_wait:
        try:
            order = alpaca_request("GET", f"v2/orders/{order_id}", mc)
            last_order = order
            status = order.get("status", "")
            if status in terminal_states:
                return order
        except Exception:
            break
        attempts += 1
        # 0.5s × 4 = 2s of fast polling, then 1s intervals — but never
        # sleep past the deadline.
        sleep_for = interval if attempts < 4 else max(interval, 1.0)
        remaining = max_wait - (time.monotonic() - start)
        if remaining <= 0:
            break
        time.sleep(min(sleep_for, remaining))
    if final_fetch:
        try:
            return alpaca_request("GET", f"v2/orders/{order_id}", mc)
        except Exception:
            pass
    if last_order is not None:
        return last_order
    return {"status": "unknown", "id": order_id}


def liquidate_all_positions(
    mc: "ModelConfig",
    journal: "TradeJournal",
    logger,
    report: "RunReport | None" = None,
    reason: str = "liquidate_all",
) -> dict:
    """Close every open position for the slot via `DELETE /v2/positions/<sym>`.

    This is the dedicated liquidation path for the V6 drawdown freeze —
    `rebalance_portfolio(target_symbols=[])` deliberately aborts on an
    empty target list, so callers that mean "go to cash" must use this
    function instead. Idempotent: with no open positions it places no
    orders and returns n_closed=0.

    Returns {"n_closed": int, "n_failed": int, "closed": [sym], "failed": [sym]}.
    """
    run_id = f"{mc.name}_{reason}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    positions = get_positions(mc, logger)
    closed: list[str] = []
    failed: list[str] = []

    if positions:
        logger.warning(
            f"  [LIQUIDATE] {mc.name}: closing all {len(positions)} positions "
            f"(reason={reason})"
        )

    for sym, pos in sorted(positions.items()):
        trade_ts = datetime.now(timezone.utc).isoformat()
        trade = TradeRecord(
            trade_id=f"{mc.name}_{trade_ts.replace(':', '').replace('-', '')}_{sym}_sell",
            run_id=run_id,
            model=mc.name,
            timestamp=trade_ts,
            symbol=sym,
            side="sell",
            action=reason,
            order_type="market",
            time_in_force="day",
            notional_usd=round(pos.get("market_value", 0) or 0, 2),
            entry_price=pos.get("avg_entry"),
            current_price=pos.get("current_price"),
            unrealized_pnl_usd=pos.get("unrealized_pl"),
            unrealized_pnl_pct=pos.get("unrealized_pl_pct"),
        )
        try:
            resp = alpaca_request(
                "DELETE", f"v2/positions/{_to_alpaca_symbol(sym)}", mc, logger=logger
            ) or {}
            order_status = resp.get("status", "accepted")
            trade.order_id = resp.get("id")
            trade.order_status = order_status
            if order_status in ("rejected", "canceled", "expired"):
                trade.error_message = f"Order {order_status}: {resp.get('reject_reason', '')}"
                logger.error(
                    f"  [LIQUIDATE] {mc.name}: {sym} close {order_status}: "
                    f"{resp.get('reject_reason', 'unknown')}"
                )
                failed.append(sym)
            else:
                if trade.order_id:
                    try:
                        final = poll_order_status(trade.order_id, mc, logger)
                        trade.order_status = final.get("status", order_status)
                        fq = final.get("filled_qty")
                        fp = final.get("filled_avg_price")
                        if fq and fq != "0":
                            trade.shares = float(fq)
                        if fp and fp != "0":
                            trade.fill_price = float(fp)
                    except Exception:
                        pass
                # The polled FINAL status decides success — an order
                # accepted then canceled at the venue (e.g. during a halt)
                # did not close the position.
                if trade.order_status in ("rejected", "canceled", "expired"):
                    trade.error_message = f"Close order {trade.order_status} after submit"
                    logger.error(
                        f"  [LIQUIDATE] {mc.name}: {sym} close order ended "
                        f"{trade.order_status}"
                    )
                    failed.append(sym)
                else:
                    closed.append(sym)
                    logger.info(f"  [LIQUIDATE] {mc.name}: closed {sym}")
        except Exception as e:
            trade.order_status = "failed"
            trade.error_message = str(e)
            logger.error(f"  [LIQUIDATE] {mc.name}: failed to close {sym}: {e}")
            failed.append(sym)
        journal.log_trade(trade)
        time.sleep(0.1)

    result = {
        "n_closed": len(closed), "n_failed": len(failed),
        "closed": closed, "failed": failed,
    }
    if failed:
        logger.error(
            f"  [LIQUIDATE] {mc.name}: {len(failed)} positions FAILED to close: "
            f"{', '.join(failed)} — will retry on the next run."
        )
    return result


def rebalance_portfolio(
    target_symbols: list[str],
    rankings: list[tuple[str, float]],
    mc: "ModelConfig",
    journal: "TradeJournal",
    logger,
    report: "RunReport",
    dry_run: bool = False,
    target_weights: dict[str, float] | None = None,
):
    """Rebalance the live portfolio to match `target_symbols` (+ optional
    conviction weights), placing Alpaca orders and logging each trade.

    If `target_weights` is given, use those allocations. Otherwise fall
    back to equal-weight across `target_symbols`.

    Side effects: writes to the run report's `rebalance` dict, places
    Alpaca orders (unless `dry_run`), appends to the trade journal.
    """
    report.start_step("rebalance")
    rb_data = report.data.setdefault("rebalance", {})
    rb_data["dry_run"] = dry_run

    # Build prediction lookup: symbol -> (predicted_return, rank)
    pred_lookup: dict[str, tuple[float, int]] = {}
    for rank_idx, (sym, pred) in enumerate(rankings):
        pred_lookup[sym] = (pred, rank_idx + 1)

    # Microsecond run_id avoids collisions when two clicks land in the
    # same second (e.g. cron + manual trigger).
    run_id = f"{mc.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

    acct = get_account(mc, logger, report)
    portfolio_value = float(acct["portfolio_value"])
    cash_available = float(acct["cash"])
    current_positions = get_positions(mc, logger)
    rb_data["positions_before"] = len(current_positions)

    if not target_symbols:
        logger.error(f"  [REBALANCE] {mc.name}: target_symbols is empty, aborting rebalance")
        report.add_error("Rebalance cancelled: no target symbols")
        report.end_step("rebalance")
        return rb_data

    target_set = set(target_symbols)
    current_set = set(current_positions.keys())

    to_sell = sorted(current_set - target_set)
    to_buy = sorted(target_set - current_set)
    to_hold = sorted(current_set & target_set)

    # Per-symbol target dollar allocations
    #
    # `target_leverage` (default 1.0) multiplies the total book size.
    # 2.0 means we target 2× equity in long positions via Reg-T margin —
    # Alpaca's buying_power must cover this or orders will reject. The
    # buying_power pre-flight below warns before submission.
    leverage = float(getattr(mc, "target_leverage", 1.0) or 1.0)
    if target_weights:
        sym_allocations = {
            sym: portfolio_value * w * leverage
            for sym, w in target_weights.items()
        }
        avg_weight = portfolio_value * leverage / len(target_symbols)
        rb_data["target_weight"] = avg_weight
        rb_data["sizing_mode"] = "conviction"
        lev_tag = f" × {leverage}x leverage" if leverage != 1.0 else ""
        logger.info(
            f"  Sizing: CONVICTION-WEIGHTED "
            f"(exposure={sum(target_weights.values()):.0%}{lev_tag})"
        )
    else:
        avg_weight = portfolio_value * leverage / len(target_symbols)
        sym_allocations = {sym: avg_weight for sym in target_symbols}
        rb_data["target_weight"] = avg_weight
        rb_data["sizing_mode"] = "equal" if leverage == 1.0 else f"equal × {leverage}x"

    rb_data["target_leverage"] = leverage

    # Pre-flight: if total target notional exceeds Alpaca buying_power,
    # auto-scale every allocation by the same factor so the rebalance fits.
    # Without this, Alpaca rejects overage orders one-by-one — the partial
    # fills leave the book in a state where the *next* rebalance sees stale
    # positions and over/under-corrects. Auto-scaling produces a clean,
    # consistent book at the available leverage.
    total_target_notional = sum(sym_allocations.values())
    bp = float(acct.get("buying_power", 0))
    if total_target_notional > bp and bp > 0:
        scale = bp / total_target_notional
        sym_allocations = {sym: alloc * scale
                           for sym, alloc in sym_allocations.items()}
        realized_leverage = (bp / portfolio_value) if portfolio_value else 0.0
        logger.warning(
            f"  [REBALANCE] AUTO-SCALED allocations: requested notional "
            f"${total_target_notional:,.2f} > buying_power ${bp:,.2f} "
            f"(requested leverage={leverage:.2f}x, equity=${portfolio_value:,.2f}). "
            f"Scaling all positions by {scale:.4f} → "
            f"realized leverage ~{realized_leverage:.2f}x. "
            f"Update target_leverage to {realized_leverage:.2f}x or lower "
            f"to silence this warning."
        )
        report.add_warning(
            f"auto-scaled to buying_power: ${total_target_notional:,.0f}→"
            f"${bp:,.0f} (realized leverage {realized_leverage:.2f}x)"
        )
        rb_data["auto_scaled_to_buying_power"] = True
        rb_data["requested_leverage"] = leverage
        # Refresh the total for downstream code that reads it
        total_target_notional = sum(sym_allocations.values())
    else:
        rb_data["auto_scaled_to_buying_power"] = False
    # Measured gross actually targeted by this rebalance, not an echo of
    # target_leverage: weights summing below 1.0 (e.g. the adaptive sleeve
    # in VIX-stress mode) make true gross < target_leverage.
    rb_data["realized_leverage"] = (
        round(total_target_notional / portfolio_value, 4) if portfolio_value else 0.0
    )
    # Total gross this rebalance targets (post auto-scale). Besides being
    # useful reporting, it is the fallback the runner uses to refresh the
    # cutloss day-start book anchor when the post-rebalance reconciliation
    # can't measure the achieved book (see
    # core.risk.refresh_daily_book_anchor_after_rebalance).
    rb_data["target_gross_usd"] = round(float(total_target_notional), 2)

    total_positions = max(len(target_set) + len(current_set), 1)
    turnover = (len(to_sell) + len(to_buy)) / total_positions

    orders: list[dict] = []
    n_rebalanced = 0
    n_held_unchanged = 0

    # -- 1. Sell positions not in target ----------------------------------
    logger.info(f"\n  SELLS - {len(to_sell)} positions to exit:")
    for sym in to_sell:
        pos = current_positions[sym]
        qty = pos["qty"]
        mv = pos["market_value"]
        pl = pos["unrealized_pl"]
        pl_pct = pos["unrealized_pl_pct"]
        entry = pos["avg_entry"]
        price = pos["current_price"]

        logger.info(
            f"    EXIT  {sym:6s}: {qty:>8.2f} shares, "
            f"entry=${entry:.2f} -> now=${price:.2f}, "
            f"val=${mv:,.2f}, P&L=${pl:+,.2f} ({pl_pct:+.1f}%)"
        )

        if not dry_run:
            orders.append({
                "action": "sell", "symbol": sym, "qty": qty,
                "notional": mv, "side": "sell",
                "trade_action": "exit_position",
                "entry_price": entry, "current_price": price,
                "position_value_before": mv,
                "unrealized_pnl_usd": pl, "unrealized_pnl_pct": pl_pct,
            })

    # -- 2. Held positions: rebalance up/down if drift > 10% --------------
    logger.info(f"\n  HOLDS - {len(to_hold)} positions to check:")
    for sym in to_hold:
        pos = current_positions[sym]
        current_value = pos["market_value"]
        sym_target = sym_allocations.get(sym, avg_weight)
        diff = sym_target - current_value
        drift_pct = abs(diff) / (sym_target + 1e-8) * 100
        pred_ret, rank = pred_lookup.get(sym, (None, None))
        pred_str = f", pred={pred_ret:+.2f}% rank=#{rank}" if pred_ret is not None else ""

        if abs(diff) > sym_target * 0.1:
            if diff > 0:
                direction = "BUY more"
                trade_action = "rebalance_up"
            else:
                direction = "TRIM"
                trade_action = "rebalance_down"

            logger.info(
                f"    REBAL {sym:6s}: ${current_value:,.0f} -> ${sym_target:,.0f} "
                f"(drift {drift_pct:.0f}%, {direction} ${abs(diff):,.0f}{pred_str})"
            )
            n_rebalanced += 1

            if not dry_run:
                side = "buy" if diff > 0 else "sell"
                orders.append({
                    "action": f"{side}_notional", "symbol": sym,
                    "notional": abs(diff), "side": side,
                    "trade_action": trade_action,
                    "entry_price": pos["avg_entry"],
                    "current_price": pos["current_price"],
                    "position_value_before": current_value,
                    "predicted_return": pred_ret, "rank": rank,
                })
        else:
            logger.info(
                f"    HOLD  {sym:6s}: ${current_value:,.0f} "
                f"(drift {drift_pct:.0f}% < 10%, no action{pred_str})"
            )
            n_held_unchanged += 1

    # -- 3. Buy new positions ---------------------------------------------
    logger.info(f"\n  BUYS - {len(to_buy)} new positions:")
    for sym in to_buy:
        sym_target = sym_allocations.get(sym, avg_weight)
        pred_ret, rank = pred_lookup.get(sym, (None, None))
        pred_str = f"pred={pred_ret:+.2f}%, rank=#{rank}" if pred_ret is not None else ""
        logger.info(f"    NEW   {sym:6s}: ${sym_target:,.0f} ({pred_str})")

        if not dry_run:
            orders.append({
                "action": "buy_notional", "symbol": sym,
                "notional": sym_target, "side": "buy",
                "trade_action": "new_position",
                "predicted_return": pred_ret, "rank": rank,
            })

    rb_data.update({
        "n_sells": len(to_sell), "n_buys": len(to_buy),
        "n_rebalanced": n_rebalanced, "n_held": n_held_unchanged,
        "sells_detail": to_sell, "buys_detail": to_buy,
        "turnover": turnover,
    })

    logger.info(
        f"\n  Summary: {len(to_sell)} exits, {len(to_buy)} new buys, "
        f"{n_rebalanced} rebalanced, {n_held_unchanged} held, "
        f"turnover: {turnover:.0%}"
    )

    if dry_run:
        logger.info("  MODE: DRY RUN - no orders sent, no trades logged")
        report.end_step("rebalance")
        return rb_data

    # Decision-price capture: one batched snapshot call per rebalance so
    # each journal row records arrival / today's open / prev close at the
    # moment the order was decided (unrecoverable after the fact). Failure
    # is non-fatal — orders proceed with the price fields left None.
    snapshots: dict[str, dict] = {}
    try:
        order_syms = sorted({o["symbol"] for o in orders})
        if order_syms:
            snapshots = fetch_snapshots(order_syms, mc, logger) or {}
    except Exception as e:
        logger.warning(f"  Decision-price snapshot capture failed: {e}")
        snapshots = {}

    # -- Execute orders + log each trade ----------------------------------
    # Marketable-limit execution (opt-in per slot, default "market" keeps
    # the legacy path byte-identical). Applies ONLY to this rebalance flow
    # — cutloss/scanner sells and liquidate_all_positions stay MARKET.
    exec_style = str(getattr(mc, "exec_style", "market") or "market")
    limit_mode = exec_style == "marketable_limit"
    limit_buffer = float(getattr(mc, "exec_limit_buffer_bps", 10.0) or 0.0)
    limit_timeout = float(getattr(mc, "exec_fill_timeout_s", 120) or 0)
    timeout_action = str(getattr(mc, "exec_timeout_action", "market") or "market")
    # Limit orders awaiting fills: journaled AFTER resolution so the row
    # carries the final outcome (fill/timeout/replacement).
    limit_pending: list[tuple[TradeRecord, dict]] = []
    if limit_mode:
        rb_data["exec_style"] = exec_style
        logger.info(
            f"  Execution: MARKETABLE-LIMIT (buffer {limit_buffer:.0f}bp, "
            f"timeout {limit_timeout:.0f}s, on-timeout={timeout_action})")

    logger.info(f"\n  Executing {len(orders)} orders...")
    executed = 0
    failed = 0
    trade_count = 0
    total_notional = 0.0
    buy_notional = 0.0
    sell_notional = 0.0
    # Final per-order outcomes for the post-rebalance reconciliation block
    # below — appended right where each record is journaled so the two can
    # never disagree.
    submitted_trades: list[TradeRecord] = []

    for order in orders:
        sym = order["symbol"]
        trade_ts = datetime.now(timezone.utc).isoformat()
        trade_id = f"{mc.name}_{trade_ts.replace(':', '').replace('-', '')}_{sym}_{order['side']}"

        # In limit mode the wider ExecTradeRecord schema is journaled (so
        # market-fallback rows still carry exec_style="market"); the
        # default path keeps emitting plain TradeRecord rows unchanged.
        record_cls = ExecTradeRecord if limit_mode else TradeRecord
        trade = record_cls(
            trade_id=trade_id,
            run_id=run_id,
            model=mc.name,
            timestamp=trade_ts,
            symbol=sym,
            side=order["side"],
            action=order["trade_action"],
            order_type="market",
            time_in_force="day",
            notional_usd=round(order["notional"], 2),
            predicted_return_pct=order.get("predicted_return"),
            rank=order.get("rank"),
            target_weight_usd=round(sym_allocations.get(sym, avg_weight), 2),
            entry_price=order.get("entry_price"),
            current_price=order.get("current_price"),
            unrealized_pnl_usd=order.get("unrealized_pnl_usd"),
            unrealized_pnl_pct=order.get("unrealized_pnl_pct"),
            position_value_before=order.get("position_value_before"),
            portfolio_value=round(portfolio_value, 2),
            cash_before=round(cash_available, 2),
            total_positions=len(target_symbols),
            rebalance_turnover_pct=round(turnover * 100, 1),
            decision_price=(snapshots.get(sym) or {}).get("arrival"),
            reference_open=(snapshots.get(sym) or {}).get("day_open"),
            prev_close=(snapshots.get(sym) or {}).get("prev_close"),
        )

        placed_limit = False
        try:
            order_ok = False
            alpaca_sym = _to_alpaca_symbol(sym)

            # Marketable-limit attempt (rebalance orders only). ANY failure
            # here — missing arrival snapshot, placement exception, venue
            # reject — falls through to the market path for this symbol.
            if limit_mode:
                trade.exec_style = "market"  # overwritten on limit success
                arrival = (snapshots.get(sym) or {}).get("arrival")
                if not arrival:
                    logger.info(
                        f"    LIMIT->MKT {sym}: no arrival snapshot — "
                        f"market order")
                else:
                    try:
                        lp = _limit_price_for(
                            order["side"], float(arrival), limit_buffer)
                        payload = {
                            "symbol": alpaca_sym,
                            "side": order["side"],
                            "type": "limit",
                            "limit_price": lp,
                            "time_in_force": "day",
                        }
                        if order["action"] == "sell":
                            # Full exit: qty-based limit sell of the whole
                            # position (the market path uses DELETE
                            # /v2/positions, which can't carry a price).
                            payload["qty"] = order["qty"]
                        else:
                            payload["notional"] = round(order["notional"], 2)
                        resp = alpaca_request(
                            "POST", "v2/orders", mc, payload, logger=logger
                        ) or {}
                        status = resp.get("status", "submitted")
                        if (status in ("rejected", "canceled", "expired")
                                or not resp.get("id")):
                            logger.warning(
                                f"    LIMIT->MKT {sym}: limit order {status} "
                                f"({resp.get('reject_reason', 'no id')}) — "
                                f"market fallback")
                        else:
                            placed_limit = True
                            order_ok = True
                            trade.order_id = resp.get("id")
                            trade.order_status = status
                            trade.order_type = "limit"
                            trade.exec_style = "marketable_limit"
                            trade.limit_price = lp
                            limit_pending.append((trade, order))
                            logger.info(
                                f"    OK  LIMIT {order['trade_action'].upper():14s} "
                                f"{sym:6s}: {order['side']} @ ${lp:.2f} "
                                f"(arrival ${float(arrival):.2f} "
                                f"{'+' if order['side'] == 'buy' else '-'}"
                                f"{limit_buffer:.0f}bp), "
                                f"order_id={trade.order_id}")
                    except Exception as e:
                        logger.warning(
                            f"    LIMIT->MKT {sym}: limit placement failed "
                            f"({e}) — market fallback")

            if placed_limit:
                pass
            elif order["action"] == "sell":
                resp = alpaca_request(
                    "DELETE", f"v2/positions/{alpaca_sym}", mc, logger=logger
                )
                order_status = resp.get("status", "accepted") if resp else "accepted"
                trade.order_id = resp.get("id") if resp else None
                trade.order_status = order_status
                trade.shares = order["qty"]
                if order_status in ("rejected", "canceled", "expired"):
                    logger.error(
                        f"    REJECTED EXIT {sym:6s}: status={order_status}, "
                        f"reason: {resp.get('reject_reason', 'unknown')}"
                    )
                    trade.error_message = f"Order {order_status}: {resp.get('reject_reason', '')}"
                    report.add_error(
                        f"Order {order_status}: EXIT {sym} - {resp.get('reject_reason', '')}"
                    )
                    failed += 1
                else:
                    order_ok = True
                    logger.info(
                        f"    OK  EXIT  {sym:6s}: closed {order['qty']:.2f} shares, "
                        f"P&L=${order.get('unrealized_pnl_usd', 0):+,.2f} "
                        f"({order.get('unrealized_pnl_pct', 0):+.1f}%)"
                    )

            elif order["action"] in ("buy_notional", "sell_notional"):
                resp = alpaca_request("POST", "v2/orders", mc, {
                    "symbol": alpaca_sym,
                    "notional": round(order["notional"], 2),
                    "side": order["side"],
                    "type": "market",
                    "time_in_force": "day",
                }, logger=logger)
                order_status = resp.get("status", "submitted")
                trade.order_id = resp.get("id")
                trade.order_status = order_status

                if order_status in ("rejected", "canceled", "expired"):
                    logger.error(
                        f"    REJECTED {order['trade_action'].upper():16s} {sym:6s}: "
                        f"{order['side']} ${order['notional']:,.2f}, "
                        f"status={order_status}, "
                        f"reason: {resp.get('reject_reason', 'unknown')}"
                    )
                    trade.error_message = f"Order {order_status}: {resp.get('reject_reason', '')}"
                    report.add_error(
                        f"Order {order_status}: {order['trade_action']} {sym} - "
                        f"{resp.get('reject_reason', '')}"
                    )
                    failed += 1
                else:
                    order_ok = True
                    filled_qty = resp.get("filled_qty")
                    filled_str = (
                        f", filled_qty={filled_qty}"
                        if filled_qty and filled_qty != "0" else ""
                    )
                    logger.info(
                        f"    OK  {order['trade_action'].upper():16s} {sym:6s}: "
                        f"{order['side']} ${order['notional']:,.2f}, "
                        f"order_id={resp.get('id', '?')}, "
                        f"status={order_status}{filled_str}"
                    )

            if order_ok:
                executed += 1
                total_notional += order["notional"]
                if order["side"] == "buy":
                    buy_notional += order["notional"]
                else:
                    sell_notional += order["notional"]

        except Exception as e:
            trade.order_status = "failed"
            trade.error_message = str(e)
            logger.error(f"    FAIL {order['trade_action'].upper():16s} {sym:6s}: {e}")
            report.add_error(f"Order failed: {order['trade_action']} {sym} - {e}")
            failed += 1

        # Poll for final order status (fills, price, qty). Limit-pending
        # orders skip this quick poll — they get the dedicated batch poll
        # (up to exec_fill_timeout_s) right after the placement loop.
        if not placed_limit and trade.order_id and trade.order_status not in (
            "failed", "rejected", "canceled", "expired"
        ):
            try:
                final = poll_order_status(trade.order_id, mc, logger)
                _apply_order_fill(trade, final)
            except Exception:
                pass

        _compute_slippage(trade)

        # Limit-pending rows are journaled after resolution so the row
        # carries the final outcome (fill / timed_out / replacement).
        if not placed_limit:
            journal.log_trade(trade)
        submitted_trades.append(trade)
        trade_count += 1
        time.sleep(0.1)

    # -- Resolve marketable-limit orders (poll → timeout action) ----------
    limit_stats = {"n_timeout_replaced": 0, "n_unfilled_cancelled": 0}
    if limit_pending:
        try:
            limit_stats = _resolve_limit_orders(
                limit_pending, mc, logger,
                timeout_s=limit_timeout, timeout_action=timeout_action)
        except Exception as e:
            logger.warning(f"  Limit-order resolution failed (non-fatal): {e}")
        for l_trade, _l_order in limit_pending:
            _compute_slippage(l_trade)
            try:
                journal.log_trade(l_trade)
            except Exception as e:
                logger.warning(
                    f"  Journal append failed for {l_trade.symbol}: {e}")

    rb_data.update({"executed": executed, "failed": failed})
    report.set("trade_log_summary", {
        "count": trade_count,
        "total_notional": round(total_notional, 2),
        "buy_notional": round(buy_notional, 2),
        "sell_notional": round(sell_notional, 2),
        "file": str(journal.jsonl_path),
    })

    logger.info(f"\n  Orders done: {executed} submitted, {failed} failed")
    logger.info(f"  Trade journal: {trade_count} trades logged -> {journal.jsonl_path}")
    logger.info(
        f"  Notional: ${total_notional:,.2f} total "
        f"(${buy_notional:,.2f} buys, ${sell_notional:,.2f} sells)"
    )

    # -- Post-rebalance reconciliation (Phase A1, additive) ----------------
    # Flow-level: final polled outcome of each order we submitted.
    # Book-level: ONE extra get_positions call comparing the achieved book
    # against the per-symbol targets. Caveat: positions are marked at
    # CURRENT prices, which have drifted since the fills — small per-symbol
    # deviations are market noise, not execution failure; large ones (or
    # missing names) point at rejects/unconfirmed orders. Everything above
    # already executed and journaled, so a failure here is never allowed to
    # touch the live path — the whole block is best-effort.
    try:
        rejected_states = ("rejected", "canceled", "expired", "failed")
        n_filled = n_rejected = n_unconfirmed = 0
        filled_notional = 0.0
        orders_notional = 0.0
        rejected_syms: list[str] = []
        unconfirmed_syms: list[str] = []
        for t in submitted_trades:
            t_notional = float(t.notional_usd or 0.0)
            orders_notional += t_notional
            if t.order_status == "filled":
                n_filled += 1
                filled_notional += t_notional
            elif t.order_status in rejected_states:
                n_rejected += 1
                rejected_syms.append(t.symbol)
            else:
                # accepted/submitted/new/pending/unknown — the order went
                # out but never confirmed filled within the poll window.
                n_unconfirmed += 1
                unconfirmed_syms.append(t.symbol)
        reconciliation: dict = {
            "flow": {
                "n_orders": len(submitted_trades),
                "n_filled": n_filled,
                "n_rejected": n_rejected,
                "n_unconfirmed": n_unconfirmed,
                "fill_rate_notional": (
                    round(filled_notional / orders_notional, 4)
                    if orders_notional > 0 else None
                ),
                "rejected_symbols": rejected_syms,
                "unconfirmed_symbols": unconfirmed_syms,
            },
            "book": None,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        # Marketable-limit outcome counters — only present in limit mode so
        # the default market-path report stays byte-identical.
        if limit_mode:
            reconciliation["flow"]["n_timeout_replaced"] = (
                limit_stats["n_timeout_replaced"])
            reconciliation["flow"]["n_unfilled_cancelled"] = (
                limit_stats["n_unfilled_cancelled"])
        try:
            positions_after = get_positions(mc, logger)
            achieved_gross = sum(
                abs(float(p.get("market_value", 0) or 0))
                for p in positions_after.values()
            )
            target_gross = float(total_target_notional)
            # Symbol-form normalization before diffing: targets use the
            # universe's dash form (BRK-B) while Alpaca reports positions
            # in dot form (BRK.B) — comparing raw keys flagged the same
            # position as both "missing" and "unexpected". Normalize both
            # sides through _to_alpaca_symbol (dash→dot; dot passes
            # through unchanged) and key achieved positions back by the
            # target's own spelling wherever a target exists.
            target_by_alpaca = {_to_alpaca_symbol(s): s
                                for s in sym_allocations}
            positions_after_norm = {
                target_by_alpaca.get(_to_alpaca_symbol(s), s): p
                for s, p in positions_after.items()
            }
            deviations = []
            for sym in set(sym_allocations) | set(positions_after_norm):
                tgt = float(sym_allocations.get(sym, 0.0))
                ach = float(
                    (positions_after_norm.get(sym) or {}).get("market_value", 0) or 0)
                deviations.append({
                    "symbol": sym,
                    "target_usd": round(tgt, 2),
                    "achieved_usd": round(ach, 2),
                    "deviation_usd": round(ach - tgt, 2),
                })
            deviations.sort(key=lambda d: -abs(d["deviation_usd"]))
            reconciliation["book"] = {
                "positions_after": len(positions_after),
                "target_gross_usd": round(target_gross, 2),
                "achieved_gross_usd": round(achieved_gross, 2),
                "gross_achieved_pct_of_target": (
                    round(achieved_gross / target_gross * 100.0, 2)
                    if target_gross > 0 else None
                ),
                # rb_data["realized_leverage"] is the gross this rebalance
                # actually targeted (post auto-scale / sub-1.0 weights), so
                # it's the honest comparison point for the achieved book.
                "target_leverage": rb_data.get("realized_leverage"),
                "achieved_leverage": (
                    round(achieved_gross / portfolio_value, 4)
                    if portfolio_value else None
                ),
                "top_deviations": deviations[:5],
                "missing_positions": sorted(
                    set(sym_allocations) - set(positions_after_norm)),
                "unexpected_positions": sorted(
                    set(positions_after_norm) - set(sym_allocations)),
            }
        except Exception as e:
            logger.warning(f"  Reconciliation book check failed: {e}")
        rb_data["reconciliation"] = reconciliation
        flow = reconciliation["flow"]
        fill_rate = flow["fill_rate_notional"]
        logger.info(
            f"  Reconciliation: {n_filled}/{flow['n_orders']} filled, "
            f"{n_rejected} rejected, {n_unconfirmed} unconfirmed"
            + (f", fill rate {fill_rate:.1%} of notional"
               if fill_rate is not None else "")
        )
        book = reconciliation.get("book")
        if book:
            logger.info(
                f"  Reconciliation book: gross "
                f"${book['achieved_gross_usd']:,.2f} achieved vs "
                f"${book['target_gross_usd']:,.2f} target, leverage "
                f"{book['achieved_leverage']} vs {book['target_leverage']} "
                f"targeted, {len(book['missing_positions'])} missing, "
                f"{len(book['unexpected_positions'])} unexpected"
            )
    except Exception as e:
        logger.warning(f"  Post-rebalance reconciliation failed (non-fatal): {e}")

    report.end_step("rebalance")
    return rb_data
