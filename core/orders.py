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

import hashlib
import json
import math
import os
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
    # Opening-auction rows (exec_style "auction_loo"): the SUBMITTED whole-
    # share quantity, so a share-based fill rate can be rebuilt from rows
    # after a partial fill (`shares` holds only the filled qty; review
    # round 4). target_weight_usd carries the anchor-priced submitted
    # notional on those rows.
    qty_submitted: Optional[float] = None


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
        except Exception as e:
            if getattr(e, "status_code", None) == 404:
                # The venue does not know this id (paper-account reset,
                # never-recorded order): terminal, never 'unknown', so a
                # finalize can stop retrying it (review round 4).
                return {"status": "not_found", "id": order_id}
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
        except Exception as e:
            if getattr(e, "status_code", None) == 404:
                return {"status": "not_found", "id": order_id}
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


def _auction_qty(order: dict, prev_close: float, buy_limit_mult: float = 1.0) -> float:
    """Whole-share quantity for an opening-auction order (opg orders cannot
    be fractional/notional). Full exits sell the whole-share part of the
    position; trims floor notional / prev_close; BUYS floor notional over
    the LIMIT price (prev_close x buy_limit_mult) so the venue's reserved
    buying power (qty x limit) never exceeds the allocation — the
    pre-flight scales buys to exactly the buying power (review round 5);
    sells never exceed the held quantity."""
    import math
    if prev_close <= 0:
        return 0.0
    if order["action"] == "sell":
        return float(math.floor(float(order["qty"])))
    px = prev_close * (buy_limit_mult if order["side"] == "buy" else 1.0)
    qty = math.floor(float(order["notional"]) / px)
    if order["side"] == "sell" and order.get("held_qty") is not None:
        qty = min(qty, math.floor(float(order["held_qty"])))
    return float(max(qty, 0))


def _auction_client_order_id(run_id: str, sym: str) -> str:
    """Deterministic idempotency key per (rebalance day, symbol): lets a
    POST whose response was lost be recovered by GET
    v2/orders:by_client_order_id instead of executing untracked at the
    open (review round 4). Kept under Alpaca's historical 48-char limit."""
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    h = hashlib.sha1(str(run_id).encode()).hexdigest()[:10]
    return f"a{day}_{_to_alpaca_symbol(sym)}_{h}"[:48]


def submit_auction_orders(orders: list[dict], mc: "ModelConfig",
                          last_close: dict, logger, rb_data: dict,
                          run_id: str | None = None,
                          stale_close: set | None = None,
                          on_submitted=None) -> dict:
    """Stage 1 of exec_style="auction": submit every planned order for the
    opening auction (time_in_force "opg", whole shares).

    - BUYS are LIMIT-on-open at last_close x (1 + exec_auction_buffer_bps):
      the backtest's assumed fill is the open, the limit only guards a
      gap-up beyond the buffer (unfilled -> cancelled -> the 09:35
      completion handles it).
    - SELLS (exits and trims) are MARKET-on-open: risk-reducing, exactly
      the backtest's fill, and never left hanging on a gap-down morning
      (a limit-protected sell leg would have filled every buy and no exit —
      transient over-leverage; review 2026-10-15).

    `last_close` is yesterday's OFFICIAL close per symbol from the bot's
    own daily panel (the signal price). Alpaca's pre-open snapshot must
    NOT be used for this: before ~09:31 its prevDailyBar is two sessions
    back (review 2026-10-15). Nothing is journalled here — fills are only
    known after 09:30; the pending records are returned for the 09:35 run
    to finalize (finalize_auction_orders)."""
    buffer = float(getattr(mc, "exec_auction_buffer_bps", 100.0) or 100.0)
    run_id = run_id or f"{mc.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    pending: list[dict] = []
    skipped: list[dict] = []
    submitted_notional = 0.0
    for order in orders:
        sym = order["symbol"]
        lc = last_close or {}
        prev_close = lc.get(sym) or lc.get(sym.replace(".", "-")) or lc.get(_to_alpaca_symbol(sym))
        if stale_close and (sym in stale_close or sym.replace(".", "-") in stale_close
                            or _to_alpaca_symbol(sym) in stale_close):
            # The panel's last bar for this name is not the prior session's
            # close: an LOO limit / fill-rate anchor off a two-session-old
            # price is wrong by the overnight gap (review round 4). The
            # 09:35 completion rebalances the name normally.
            skipped.append({"symbol": sym, "reason": "stale_close"})
            continue
        if not prev_close or float(prev_close) <= 0:
            skipped.append({"symbol": sym, "reason": "no_prev_close"})
            continue
        qty = _auction_qty(order, float(prev_close), buy_limit_mult=1.0 + buffer / 1e4)
        if qty <= 0:
            skipped.append({"symbol": sym, "reason": "sub_share",
                            "notional": round(float(order.get("notional", 0) or 0), 2)})
            continue
        coid = _auction_client_order_id(run_id, sym)
        if order["side"] == "sell":
            lp = None
            payload = {
                "symbol": _to_alpaca_symbol(sym), "side": "sell",
                "type": "market", "qty": str(int(qty)), "time_in_force": "opg",
                "client_order_id": coid,
            }
        else:
            lp = _limit_price_for("buy", float(prev_close), buffer)
            payload = {
                "symbol": _to_alpaca_symbol(sym), "side": "buy",
                "type": "limit", "limit_price": lp, "qty": str(int(qty)),
                "time_in_force": "opg", "client_order_id": coid,
            }
        recovered = False
        try:
            try:
                resp = alpaca_request("POST", "v2/orders", mc, payload, logger=logger) or {}
            except Exception as post_err:
                # The POST may have reached Alpaca although the response was
                # lost (timeout / reset): look the order up by its
                # idempotency key before declaring it skipped (round 4).
                resp = {}
                try:
                    found = alpaca_request(
                        "GET", f"v2/orders:by_client_order_id?client_order_id={coid}",
                        mc, logger=logger) or {}
                    if isinstance(found, dict) and found.get("id"):
                        resp = found
                        recovered = True
                        logger.warning(f"    AUCTION {sym}: POST failed ({post_err}) but the "
                                       f"order exists at the venue (id={found.get('id')}) — recovered")
                except Exception:
                    pass
                if not recovered:
                    raise post_err
            status = resp.get("status", "submitted")
            if status in ("rejected", "canceled", "expired") or not resp.get("id"):
                skipped.append({"symbol": sym, "reason": f"rejected:{resp.get('reject_reason', status)}"})
                continue
            est = qty * float(prev_close)
            submitted_notional += est
            pending.append({
                "symbol": sym, "side": order["side"],
                "trade_action": order.get("trade_action"),
                "order_type": payload["type"],
                "qty": qty, "limit_price": lp, "prev_close": float(prev_close),
                "notional_est": round(est, 2),
                "order_id": resp.get("id"), "order_status": status,
                "client_order_id": coid, "recovered": recovered,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "rank": order.get("rank"), "predicted_return": order.get("predicted_return"),
                "entry_price": order.get("entry_price"),
                "position_value_before": order.get("position_value_before"),
            })
            logger.info(f"    AUCTION {order['side'].upper():4s} {sym:6s} qty {int(qty)} "
                        f"{'MOO' if lp is None else 'LOO @' + str(lp)} "
                        f"(prev close {float(prev_close):.2f}) "
                        f"[{order.get('trade_action')}] id={resp.get('id')}")
            if on_submitted is not None:
                # Persist each live order as soon as the venue has it: a
                # process death mid-loop must never leave opg orders the
                # bot does not know about (review round 5).
                try:
                    on_submitted(pending[-1], run_id)
                except Exception as e:
                    logger.warning(f"    AUCTION {sym}: persist callback failed: {e}")
        except Exception as e:
            skipped.append({"symbol": sym, "reason": f"error:{e}"})
            logger.error(f"    AUCTION {sym}: submit failed: {e}")
    summary = {
        "n_planned": len(orders), "n_submitted": len(pending),
        "n_skipped": len(skipped), "skipped": skipped[:40],
        "submitted_notional_usd": round(submitted_notional, 2),
        "run_id": run_id, "pending": pending,
    }
    rb_data["exec_style"] = "auction_loo"
    rb_data["auction"] = summary
    logger.info(f"  AUCTION stage: {len(pending)} LOO orders submitted "
                f"(${submitted_notional:,.0f}), {len(skipped)} skipped")
    return summary


_AUCTION_OPEN_STATES = ("new", "accepted", "partially_filled", "pending_new",
                        "accepted_for_bidding", "held", "pending_cancel",
                        "pending_replace")
# States in which an opg order will never change again (journal it).
_AUCTION_TERMINAL_STATES = ("filled", "canceled", "expired", "rejected",
                            "suspended", "replaced", "done_for_day",
                            "not_found", "stopped", "calculated")


def recover_unpersisted_auction_orders(mc: "ModelConfig", session_iso: str,
                                       known_ids: set[str], logger) -> list[dict]:
    """Orders POSTed by today's auction stage whose record was lost (process
    death between the POST and the state save): every auction order carries
    a client_order_id starting with "a<YYYYMMDD>_", so the venue's own
    order list is the source of truth (review round 5). Returns pending
    entries for the unknown ones (prev_close unknown -> derived from the
    limit price for buys, None for sells)."""
    try:
        from datetime import date
        day = session_iso.replace("-", "")
        after = f"{session_iso}T08:00:00-04:00"
        orders = alpaca_request(
            "GET", f"v2/orders?status=all&limit=500&direction=asc&after={after}",
            mc, logger=logger) or []
    except Exception as e:
        logger.warning(f"  auction recovery: order list unavailable: {e}")
        return []
    found: list[dict] = []
    for o in orders if isinstance(orders, list) else []:
        try:
            coid = str(o.get("client_order_id") or "")
            if not coid.startswith(f"a{day}_") or str(o.get("time_in_force") or "") != "opg":
                continue
            if o.get("id") in known_ids:
                continue
            lp = float(o["limit_price"]) if o.get("limit_price") else None
            side = str(o.get("side") or "buy")
            qty = float(o.get("qty") or 0)
            prev_close = round(lp / 1.01, 4) if (lp and side == "buy") else None
            found.append({
                "symbol": str(o.get("symbol") or "").replace(".", "-"), "side": side,
                "trade_action": "recovered", "order_type": str(o.get("type") or "limit"),
                "qty": qty, "limit_price": lp, "prev_close": prev_close,
                "notional_est": round(qty * prev_close, 2) if prev_close else 0.0,
                "order_id": o.get("id"), "order_status": str(o.get("status") or "unknown"),
                "client_order_id": coid, "recovered": True,
                "submitted_at": str(o.get("submitted_at") or o.get("created_at")
                                    or f"{session_iso}T13:06:00+00:00"),
                "run_id": None,
            })
        except Exception:
            continue
    if found:
        logger.warning(f"  auction recovery: {len(found)} live opg order(s) found at the venue "
                       f"that the stage never persisted — adopted as pending")
    return found


def _existing_auction_trade_ids(journal: "TradeJournal", prefix: str) -> set[str]:
    """trade_ids of auction rows already in the slot's journal. Auction
    trade_ids are deterministic per (day, side, symbol), so a restart
    between the appends and the state save must not double-journal the
    day (review round 4). Best-effort: unreadable journal -> empty set."""
    ids: set[str] = set()
    try:
        path = getattr(journal, "jsonl_path", None)
        if not path or not os.path.exists(str(path)):
            return ids
        with open(path) as f:
            for line in f:
                if prefix not in line:
                    continue
                try:
                    rec = json.loads(line)
                    tids = [str(rec.get("trade_id") or "")]
                except Exception:
                    # A glued/truncated physical line: recover every id on it.
                    import re
                    tids = re.findall(r'"trade_id":\s*"([^"]+)"', line)
                for tid in tids:
                    if tid.startswith(prefix):
                        ids.add(tid)
    except Exception:
        pass
    return ids


def finalize_auction_orders(mc: "ModelConfig", pending: list[dict],
                            journal: "TradeJournal", logger,
                            budget_s: float = 90.0,
                            panel_open: dict | None = None) -> dict:
    """Stage 2 of exec_style="auction" (run at 09:35): read each opg
    order's final state, journal one row per order with exec_style
    "auction_loo" and slippage vs the session open, and cancel anything
    still open (defensive — the venue cancels unfilled opg at the open).
    Returns fill statistics for the report/state.

    Reference open (review round 4): `panel_open[sym]` — today's official
    open from the bot's own daily panel when its last bar is dated today —
    is preferred; otherwise the IEX snapshot's dailyBar open is used only
    when that bar is dated today (stale pre-open bars are rejected); an
    earlier session's order ("late") never gets a reference. The source
    mix is reported in stats["reference_sources"].

    Robustness rules: an unreadable order (3 polled reads) stays pending;
    a 404 is terminal ("not_found"); a cancel whose final read fails keeps
    the entry pending (a partial fill may exist); a journal append failure
    keeps the entry pending and counts nothing; rows already journaled
    (restart between appends and the state save) are counted but not
    re-appended; the whole pass is bounded by `budget_s` of wall-clock
    time after which untouched entries stay pending."""
    if not pending:
        return {"n": 0}
    syms = sorted({p["symbol"] for p in pending})
    try:
        snapshots = fetch_snapshots(syms, mc, logger) or {}
    except Exception as e:
        logger.warning(f"  auction finalize: snapshot capture failed: {e}")
        snapshots = {}
    try:
        from zoneinfo import ZoneInfo
        session_today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        session_today = datetime.now().strftime("%Y-%m-%d")
    existing_ids = _existing_auction_trade_ids(journal, f"{mc.name}_auction_")
    deadline = time.monotonic() + max(5.0, float(budget_s or 90.0))
    n_filled = n_unfilled = n_cancelled = n_not_found = n_dup = n_deferred = 0
    filled_notional = est_notional = filled_anchor = 0.0
    cost_num = cost_den = 0.0
    ref_src = {"panel": 0, "snapshot": 0, "none": 0}
    remaining: list[dict] = []
    budget_logged = False
    for p in pending:
        if time.monotonic() > deadline:
            if not budget_logged:
                logger.warning(f"  auction finalize: {budget_s:.0f}s budget exhausted — "
                               f"remaining entries kept pending for the next run")
                budget_logged = True
            remaining.append(p)
            n_deferred += 1
            continue
        try:
            ts = datetime.now(timezone.utc)
            sym = p["symbol"]
            sub_day = str(p.get("submitted_at") or ts.isoformat())[:10].replace("-", "")
            # A late finalize (an earlier session's orders) must not be
            # scored against TODAY's open (review round 3).
            late = str(p.get("submitted_at") or "")[:10] not in ("", session_today)
            ref = None
            src = "none"
            if not late:
                po = (panel_open or {}).get(sym) or (panel_open or {}).get(sym.replace(".", "-"))
                snap = snapshots.get(sym) or snapshots.get(sym.replace(".", "-")) or {}
                bar_date = snap.get("day_bar_date")
                if po and float(po) > 0:
                    ref, src = float(po), "panel"
                elif snap.get("day_open") and (bar_date is None or bar_date == session_today):
                    ref, src = float(snap["day_open"]), "snapshot"
            trade = ExecTradeRecord(
                trade_id=f"{mc.name}_auction_{sub_day}_{p['side']}_{sym}",
                run_id=p.get("run_id") or f"{mc.name}_auction_{ts.strftime('%Y%m%d')}",
                model=mc.name, timestamp=p.get("submitted_at") or ts.isoformat(),
                symbol=sym, side=p["side"],
                action=p.get("trade_action") or ("exit_position" if p["side"] == "sell" else "new_position"),
                order_type=p.get("order_type") or "limit", time_in_force="opg",
                notional_usd=float(p.get("notional_est") or 0),
                order_id=p.get("order_id"), order_status=str(p.get("order_status") or "unknown"),
                current_price=float(p.get("prev_close") or 0),
                rank=p.get("rank"), predicted_return_pct=p.get("predicted_return"),
                entry_price=p.get("entry_price"),
                position_value_before=p.get("position_value_before"),
                # Submission facts (round 4): anchor-priced submitted
                # notional and submitted whole-share qty, so a share-based
                # fill rate can be rebuilt from rows after a partial fill.
                target_weight_usd=float(p.get("notional_est") or 0) or None,
                qty_submitted=float(p.get("qty") or 0) or None,
                # No arrival price exists pre-open: decision_price stays None
                # so the vs-arrival statistic never mistakes the overnight
                # gap for an execution cost. prev_close is kept in its own
                # field.
                decision_price=None,
                reference_open=ref,
                prev_close=float(p.get("prev_close") or 0) or None,
                exec_style="auction_loo", limit_price=p.get("limit_price"),
            )
            # Polled read with retries — a failed read must never cancel a
            # possibly filled order or journal it as unfilled.
            state: dict = {}
            for attempt in range(3):
                try:
                    state = poll_order_status(p["order_id"], mc, logger, max_wait=5.0) or {}
                except Exception as e:
                    logger.warning(f"  auction finalize: poll {p['order_id']} failed: {e}")
                    state = {}
                if str(state.get("status") or "unknown") != "unknown":
                    break
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
            status = str(state.get("status") or "unknown")
            if status == "unknown":
                remaining.append(p)
                logger.warning(f"  auction finalize: {sym} ({p['order_id']}) "
                               f"unreadable — kept pending for retry")
                continue
            cancelled_here = 1 if p.get("cancel_sent") else 0
            if status in _AUCTION_OPEN_STATES:
                # Still open: cancel and re-read the FINAL state.
                trade.order_status = status
                final = _cancel_order_and_get_final(trade, mc, logger) or {}
                fstatus = str(final.get("status") or "unknown")
                if fstatus not in _AUCTION_TERMINAL_STATES:
                    # Cancel sent but no TERMINAL state read back (failed
                    # read, or still partially_filled/pending_cancel): a
                    # partial fill may exist — keep pending, never journal
                    # a non-final row (review rounds 4-5).
                    p["cancel_sent"] = True
                    remaining.append(p)
                    logger.warning(f"  auction finalize: {sym} cancel sent but state "
                                   f"{fstatus!r} is not terminal — kept pending for retry")
                    continue
                state, status = final, fstatus
                cancelled_here = 1
            elif status not in _AUCTION_TERMINAL_STATES:
                # Neither open nor terminal (a venue state we do not model):
                # retry later rather than journal a guess.
                remaining.append(p)
                logger.warning(f"  auction finalize: {sym} state {status!r} unmodelled — kept pending")
                continue
            _apply_order_fill(trade, state)
            trade.order_status = status
            if status == "filled" and not trade.fill_price and state.get("filled_avg_price"):
                trade.fill_price = float(state["filled_avg_price"])
            _compute_slippage(trade)
            fq_raw = state.get("filled_qty")
            is_filled = status == "filled" or (fq_raw not in (None, "0", 0) and bool(trade.fill_price))
            fq = fn = anchor = 0.0
            if is_filled:
                fq = float(fq_raw or p.get("qty") or 0)
                fp = float(trade.fill_price or p.get("prev_close") or 0)
                fn = fq * fp
                # Fill rate is measured at the ANCHOR price (shares
                # executed), so a gap-down morning does not read as an
                # unfilled residue.
                anchor = fq * float(p.get("prev_close") or 0)
                trade.notional_usd = round(fn, 2)
            # Journal FIRST; the statistics count an order only once its
            # row exists (or already existed) — review round 4.
            if trade.trade_id in existing_ids:
                n_dup += 1
            else:
                try:
                    journal.log_trade(trade)
                except Exception as e:
                    logger.error(f"  auction finalize: journal append failed for {sym}: "
                                 f"{e} — kept pending")
                    remaining.append(p)
                    continue
            est_notional += float(p.get("notional_est") or 0)
            n_cancelled += cancelled_here
            if status == "not_found":
                n_not_found += 1
            if is_filled:
                n_filled += 1
                filled_notional += fn
                filled_anchor += anchor
                if trade.slippage_bps is not None:
                    cost_num += trade.slippage_bps * fn
                    cost_den += fn
            else:
                n_unfilled += 1
            ref_src[src] += 1
        except Exception as e:
            # One bad order must never drop the rest: keep it pending.
            logger.error(f"  auction finalize: {p.get('symbol')} failed ({e}) — kept pending")
            remaining.append(p)
    stats = {
        "n": len(pending) - len(remaining), "n_filled": n_filled, "n_unfilled": n_unfilled,
        "n_cancelled_open": n_cancelled, "n_not_found": n_not_found,
        "n_already_journaled": n_dup, "n_budget_deferred": n_deferred,
        "n_remaining_unread": len(remaining),
        "remaining": remaining,
        "fill_rate_notional": round(filled_anchor / est_notional, 4) if est_notional else None,
        "submitted_notional_usd": round(est_notional, 2),
        "filled_notional_anchor_usd": round(filled_anchor, 2),
        "filled_notional_usd": round(filled_notional, 2),
        "vw_cost_vs_open_bps": round(cost_num / cost_den, 2) if cost_den else None,
        # The dollars the vw cost is weighted over (rows WITH a reference),
        # so a retry pass can be merged with the right weight (round 5).
        "vw_cost_weight_usd": round(cost_den, 2),
        "reference_sources": ref_src,
    }
    logger.info(f"  AUCTION fills: {n_filled}/{len(pending)} filled "
                f"(notional fill rate {stats['fill_rate_notional']}), "
                f"vw cost vs open {stats['vw_cost_vs_open_bps']} bp/side "
                f"(refs {ref_src}), {n_cancelled} still-open cancelled, "
                f"{len(remaining)} kept pending")
    return stats


def rebalance_portfolio(
    target_symbols: list[str],
    rankings: list[tuple[str, float]],
    mc: "ModelConfig",
    journal: "TradeJournal",
    logger,
    report: "RunReport",
    dry_run: bool = False,
    target_weights: dict[str, float] | None = None,
    auction: bool = False,
    last_close: dict | None = None,
    run_id: str | None = None,
    stale_close: set | None = None,
    on_submitted=None,
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
    # One run_id per rebalance DAY: an auction day's completion reuses the
    # 09:06 stage's id so execution stats group both legs as one rebalance.
    # A caller-supplied id means THIS call completes a stored auction plan;
    # a rebalance of an "auction" slot without one (stage skipped, forced
    # re-run) is an ordinary marketable-limit rebalance and is tagged so.
    completing_plan = bool(run_id)
    run_id = run_id or f"{mc.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

    acct = get_account(mc, logger, report)
    portfolio_value = float(acct["portfolio_value"])
    cash_available = float(acct["cash"])
    # The gate-state shortfall derivation (core/gate_update.py
    # _shortfall_from_result) divides the achieved gross by THIS equity;
    # it was never written into the result before (review round 5).
    rb_data["portfolio_value"] = round(portfolio_value, 2)
    current_positions = get_positions(mc, logger)
    rb_data["positions_before"] = len(current_positions)

    if not target_symbols:
        logger.error(f"  [REBALANCE] {mc.name}: target_symbols is empty, aborting rebalance")
        report.add_error("Rebalance cancelled: no target symbols")
        report.end_step("rebalance")
        return rb_data

    # Alpaca keys positions in dot form (BRK.B); the universe uses dash
    # form (BRK-B). Normalize so a held class share is recognized as held
    # instead of being sold AND bought in the same plan (review 2026-10-15).
    ours_by_alpaca = {_to_alpaca_symbol(s): s for s in target_symbols}
    current_positions = {ours_by_alpaca.get(k, k): v for k, v in current_positions.items()}
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

    # Pre-flight (rewritten 2026-10-15): compare the INCREMENTAL buying this
    # rebalance needs — new positions plus up-sizes of held names — against
    # Alpaca buying_power. The previous check compared the ENTIRE target
    # book (held positions included) to buying_power, so any rebalance that
    # started from a fully invested 2x book was auto-scaled to ~0.75x of
    # design even though only ~30-45% of the book actually trades: the
    # 2026-09-29 rebalance silently landed both gated slots at ~1.6x. Sells
    # are NOT credited: Alpaca's buying_power is the day-trading figure,
    # which intraday sales do not replenish, and the buys are submitted
    # seconds after the sells anyway. If the buys alone exceed
    # buying_power, ONLY the buy side is scaled — exits and trims always
    # execute in full.
    bp = float(acct.get("buying_power", 0))
    forced_upsizes: set = set()
    buy_need = 0.0
    for sym in to_buy:
        buy_need += float(sym_allocations.get(sym, avg_weight))
    for sym in to_hold:
        mv = float(current_positions[sym]["market_value"])
        tgt = float(sym_allocations.get(sym, avg_weight))
        diff = tgt - mv
        if diff > 0 and abs(diff) > tgt * 0.1:   # same drift rule as the loop
            buy_need += diff
    total_target_notional = sum(sym_allocations.values())
    rb_data["buy_need_usd"] = round(buy_need, 2)
    rb_data["buying_power_usd"] = round(bp, 2)
    if buy_need > bp and bp > 0:
        scale = bp / buy_need
        for sym in to_buy:
            sym_allocations[sym] = float(sym_allocations.get(sym, avg_weight)) * scale
        for sym in to_hold:
            mv = float(current_positions[sym]["market_value"])
            tgt = float(sym_allocations.get(sym, avg_weight))
            diff = tgt - mv
            if diff > 0 and abs(diff) > tgt * 0.1:
                sym_allocations[sym] = mv + diff * scale
                # The scaled increment may now sit under the 10% band; the
                # hold loop must still place it (its budget was reserved).
                forced_upsizes.add(sym)
        total_target_notional = sum(sym_allocations.values())
        realized_leverage = (total_target_notional / portfolio_value) if portfolio_value else 0.0
        logger.warning(
            f"  [REBALANCE] AUTO-SCALED BUYS: incremental buys "
            f"${buy_need:,.2f} > buying_power ${bp:,.2f} "
            f"(requested leverage={leverage:.2f}x, equity=${portfolio_value:,.2f}). "
            f"Buy side scaled by {scale:.4f}; exits/trims unchanged → "
            f"targeted gross ${total_target_notional:,.2f} (~{realized_leverage:.2f}x)."
        )
        report.add_warning(
            f"buys auto-scaled to buying_power: ${buy_need:,.0f}→${bp:,.0f} "
            f"(targeted leverage {realized_leverage:.2f}x)"
        )
        rb_data["auto_scaled_to_buying_power"] = True
        rb_data["requested_leverage"] = leverage
        rb_data["buy_scale"] = round(scale, 4)
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

        if abs(diff) > sym_target * 0.1 or (sym in forced_upsizes and diff > 0):
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
                    "held_qty": pos.get("qty"),
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
        if order_syms and not auction:   # pre-open snapshots are not used
            snapshots = fetch_snapshots(order_syms, mc, logger) or {}
    except Exception as e:
        logger.warning(f"  Decision-price snapshot capture failed: {e}")
        snapshots = {}

    # -- Auction stage (exec_style="auction", pre-open): submit LOO orders
    # and return; the 09:35 run finalizes fills and completes shortfalls.
    if auction:
        submit_auction_orders(orders, mc, last_close or {}, logger, rb_data,
                              run_id=run_id, stale_close=stale_close,
                              on_submitted=on_submitted)
        rb_data["n_orders_planned"] = len(orders)
        rb_data["run_id"] = run_id
        report.end_step("rebalance")
        return rb_data

    # -- Execute orders + log each trade ----------------------------------
    # Marketable-limit execution (opt-in per slot, default "market" keeps
    # the legacy path byte-identical). Applies ONLY to this rebalance flow
    # — cutloss/scanner sells and liquidate_all_positions stay MARKET.
    exec_style = str(getattr(mc, "exec_style", "market") or "market")
    # An "auction" slot completes its residues at marketable-limit (never
    # the legacy market path) and tags those rows so the two legs stay
    # separable in the journal (review 2026-10-15).
    limit_mode = exec_style in ("marketable_limit", "auction")
    limit_tag = ("auction_completion" if (exec_style == "auction" and completing_plan)
                 else "marketable_limit")
    limit_buffer = float(getattr(mc, "exec_limit_buffer_bps", 10.0) or 0.0)
    limit_timeout = float(getattr(mc, "exec_fill_timeout_s", 120) or 0)
    timeout_action = str(getattr(mc, "exec_timeout_action", "market") or "market")
    # Limit orders awaiting fills: journaled AFTER resolution so the row
    # carries the final outcome (fill/timeout/replacement).
    limit_pending: list[tuple[TradeRecord, dict]] = []
    if limit_mode:
        rb_data["exec_style"] = limit_tag
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
                            trade.exec_style = limit_tag
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
                "orders_notional_usd": round(orders_notional, 2),
                "filled_notional_usd": round(filled_notional, 2),
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
