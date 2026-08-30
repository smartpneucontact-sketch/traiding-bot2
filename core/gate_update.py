"""Daily gate-cadence tracking (2026-08-30 alignment fix).

The validated backtests apply the drawdown gates DAILY:
validation/book_gate.py in the research repo computes a daily multiplier
series g_t and prices `base_daily * g_t` — the simulated book re-levers
within days as its own drawdown recovers. The live bot, however, sampled
the gate only inside compute_weights() at each 21-day rebalance; between
rebalances the tier scanner could CUT exposure but nothing could ever
RAISE it. That asymmetry (de-risk daily, re-risk monthly) is a
live-vs-backtest divergence of the same class as the July tier-scale fix.
Through the Jul–Aug 2026 crash+recovery it held the gated slots at the
crash-day multiplier (~0.34) for six weeks of rally.

This module restores the validated cadence. On every pipeline run that
does NOT rebalance, it recomputes the same gate stack compute_weights()
uses — the same module-level functions, the same config fields and
getattr back-compat defaults — keyed to the HELD book (book_gate.py
computes book drawdown from the held/ffilled weights, and _book_drawdown
normalizes by gross, so the post-gate live position weights give the
identical value). If the multiplier moved by at least GATE_UPDATE_BAND,
the live book is scaled pro-rata toward it, both directions.

State contract — `applied_exposure_multiplier` is the cumulative gate
scaling the current book carries. It is listed in
core.state.SCANNER_OWNED_KEYS and every writer goes straight to disk
under state_lock:
  - the rebalance path sets it to compute_weights' exposure_multiplier,
  - this module sets it to the new multiplier after a scaling pass,
  - the tier scaler multiplies it by the fraction of the book it kept.

Execution notes: scale trades are pro-rata NOTIONAL MARKET orders (the
same convention as the validated tier layer — the backtest prices every
daily gate trade at the flat tc assumption, and the band keeps these
events occasional). Sell records get the full fill/decision-price
instrumentation pass; buy rows are journalled from the submit response
without slippage stamps (the vs-open cost measurement reads fills from
Alpaca directly, so nothing is lost for the cost studies).

Env knobs:
  GATE_DAILY_UPDATE  "1" (default) enables the daily pass; "0" disables.
  GATE_UPDATE_BAND   minimum |new − applied| multiplier move that
                     triggers trading (default 0.10).
"""

import os
import traceback
from datetime import datetime, timezone

from core.alpaca import _to_alpaca_symbol, alpaca_request, get_positions
from core.market import is_market_open
from core.combo_strategy import (
    _book_drawdown,
    _linear_dd_ramp,
    _spy_drawdown_gate,
    _to_close_panel,
)
from core.journal import TradeRecord
from core.state import load_state, save_state, state_lock

GATE_UPDATE_BAND = float(os.environ.get("GATE_UPDATE_BAND", "0.10"))
GATE_DAILY_UPDATE = os.environ.get("GATE_DAILY_UPDATE", "1") != "0"

# Skip trading when the whole book delta is below this (noise / dust).
_MIN_TOTAL_DELTA_USD = 50.0
# Per-position slices below this are skipped (matches the tier scaler).
_MIN_SLICE_USD = 5.0
# An applied multiplier this small means the book is essentially cash;
# pro-rata scaling of ~nothing cannot rebuild it — leave that to the
# next scheduled rebalance.
_MIN_APPLIED_MULT = 0.05


def config_has_gates(config) -> bool:
    """Mirror compute_weights' gate enablement exactly, including the
    getattr back-compat defaults (spy defaults True for pre-2026-06
    pickles, book defaults False)."""
    if config is None:
        return False
    return bool(getattr(config, "enable_spy_dd_gate", True)
                or getattr(config, "enable_book_dd_gate", False))


def compute_live_gate_multiplier(config, stock_data, macro_data,
                                 held_weights: dict) -> dict | None:
    """Recompute today's gate multiplier with compute_weights' own stack.

    `held_weights` is {symbol: abs market value} of the live positions —
    _book_drawdown normalizes by the sum, so any positive scaling of the
    book gives the same drawdown the validated gate sees.

    Returns {"multiplier", "spy_gate", "book_gate", "book_dd"} or None
    when nothing is computable (no data / no gated config).
    """
    if not config_has_gates(config):
        return None
    stock_px = _to_close_panel(stock_data) if stock_data else None
    macro_px = _to_close_panel(macro_data) if macro_data else None

    if getattr(config, "enable_spy_dd_gate", True):
        if macro_px is None or macro_px.empty:
            return None
        spy_gate = _spy_drawdown_gate(
            macro_px,
            lookback=getattr(config, "spy_dd_lookback", 60),
            full_dd=getattr(config, "spy_full_dd", 0.08),
            cash_dd=getattr(config, "spy_cash_dd", 0.18),
        )
    else:
        spy_gate = 1.0

    book_gate = None
    book_dd = None
    if getattr(config, "enable_book_dd_gate", False):
        if stock_px is None or stock_px.empty or not held_weights:
            return None
        book_dd = _book_drawdown(
            stock_px, held_weights,
            lookback=getattr(config, "book_dd_lookback", 60),
        )
        if book_dd is not None:
            book_gate = _linear_dd_ramp(
                book_dd,
                getattr(config, "book_full_dd", 0.15),
                getattr(config, "book_cash_dd", 0.30),
            )

    # Identical combination rule to compute_weights.
    exposure = spy_gate if book_gate is None else min(spy_gate, book_gate)
    return {
        "multiplier": round(float(exposure), 4),
        "spy_gate": round(float(spy_gate), 4),
        "book_gate": round(float(book_gate), 4) if book_gate is not None else None,
        "book_dd": round(float(book_dd), 4) if book_dd is not None else None,
    }


def set_applied_multiplier(mc, value: float | None, logger=None) -> None:
    """Write applied_exposure_multiplier straight to disk under the lock
    (scanner-owned key contract — see module docstring)."""
    try:
        with state_lock:
            disk = load_state(mc)
            if value is None:
                disk.pop("applied_exposure_multiplier", None)
            else:
                disk["applied_exposure_multiplier"] = round(float(value), 4)
            save_state(disk, mc)
    except Exception as e:
        if logger:
            logger.error(f"[GATE] {mc.name}: applied-multiplier write failed: {e}")


def _scale_book(mc, positions: list, factor: float, journal, logger,
                dry_run: bool) -> dict:
    """Pro-rata scale every position's notional by `factor` (market
    orders, both directions). Returns a summary dict."""
    total_mv = sum(float(p.get("market_value", 0) or 0) for p in positions)
    delta_total = total_mv * (factor - 1.0)
    out = {"factor": round(factor, 4), "book_before": round(total_mv, 2),
           "delta_usd": round(delta_total, 2), "orders": 0, "failed": 0,
           "dry_run": dry_run}
    if total_mv <= 0 or abs(delta_total) < _MIN_TOTAL_DELTA_USD:
        out["skipped"] = "delta_below_minimum"
        return out

    side = "buy" if factor > 1.0 else "sell"
    if side == "buy":
        # Cap the buy batch to available buying power (same protective
        # auto-scale idea as the rebalance pre-flight).
        try:
            acct = alpaca_request("GET", "v2/account", mc, logger=logger) or {}
            bp = float(acct.get("buying_power", 0) or 0)
        except Exception:
            bp = 0.0
        if 0 < bp < delta_total:
            capped = 1.0 + (bp / total_mv) * 0.98  # 2% headroom
            logger.warning(
                f"[GATE] {mc.name}: buy delta ${delta_total:,.0f} exceeds "
                f"buying power ${bp:,.0f} — capping factor "
                f"{factor:.3f} → {capped:.3f}"
            )
            factor = max(1.0, capped)
            delta_total = total_mv * (factor - 1.0)
            out["factor_capped"] = round(factor, 4)
            if delta_total < _MIN_TOTAL_DELTA_USD:
                out["skipped"] = "no_buying_power"
                return out

    from core.risk import _instrument_and_journal_sells
    ts = datetime.now(timezone.utc)
    reason = "gate_relevel" if side == "buy" else "gate_delevel"
    records: list[TradeRecord] = []
    for p in positions:
        mv = float(p.get("market_value", 0) or 0)
        if mv <= 0:
            continue
        slice_notional = round(abs(mv * (factor - 1.0)), 2)
        if slice_notional < _MIN_SLICE_USD:
            continue
        if side == "sell":
            slice_notional = min(slice_notional, round(mv, 2))
        sym = p["symbol"]
        if dry_run:
            logger.info(f"[GATE] {mc.name}: DRY RUN {side} {sym} "
                        f"${slice_notional:.2f} ({reason})")
            out["orders"] += 1
            continue
        try:
            resp = alpaca_request(
                "POST", "v2/orders", mc,
                data={
                    "symbol": _to_alpaca_symbol(sym),
                    "notional": slice_notional,
                    "side": side,
                    "type": "market",
                    "time_in_force": "day",
                },
                logger=logger,
            ) or {}
            out["orders"] += 1
            records.append(TradeRecord(
                trade_id=f"{mc.name}_{reason}_{ts.strftime('%Y%m%d_%H%M%S')}_{sym}",
                run_id=f"{mc.name}_{reason}_{ts.strftime('%Y%m%d')}",
                model=mc.name,
                timestamp=ts.isoformat(),
                symbol=sym,
                side=side,
                action=reason,
                order_type="market",
                time_in_force="day",
                notional_usd=slice_notional,
                order_id=resp.get("id"),
                order_status=resp.get("status", "submitted"),
                current_price=float(p.get("current_price", 0) or 0),
            ))
        except Exception as e:
            out["failed"] += 1
            logger.error(f"[GATE] {mc.name}: {side} failed for {sym}: {e}")

    if records:
        if side == "sell":
            # Full fill + decision-price instrumentation (shared with the
            # tier scaler).
            _instrument_and_journal_sells(mc, records, journal, logger)
        else:
            for r in records:
                try:
                    journal.log_trade(r)
                except Exception:
                    pass
    return out


def maybe_daily_gate_update(mc, model_bundle: dict, strategy_type: str,
                            stock_data: dict, macro_data: dict | None,
                            journal, logger, report,
                            min_stocks_required: int,
                            dry_run: bool = False) -> dict | None:
    """Run the daily gate-cadence check for one slot. Called from the
    runner's non-rebalance-day branch only — every earlier return
    (freeze, cool-down, stop-tripped) already exited before this point.

    Never raises: any failure logs and returns a diagnostic dict.
    """
    info: dict = {"checked": False}
    try:
        if not GATE_DAILY_UPDATE:
            info["skipped"] = "disabled_by_env"
            return info
        if strategy_type != "direct_weights":
            info["skipped"] = "not_direct_weights"
            return info
        config = model_bundle.get("combo_config")
        if not config_has_gates(config):
            info["skipped"] = "no_gates_in_config"
            return info
        if len(stock_data or {}) < min_stocks_required:
            # Not a rebalance — never trade on thin data, never count it
            # against the data-guard abort alarm either.
            info["skipped"] = f"insufficient_data_{len(stock_data or {})}"
            logger.warning(f"[GATE] {mc.name}: skipping gate check — "
                           f"only {len(stock_data or {})} stocks loaded")
            return info
        if not dry_run and not is_market_open():
            info["skipped"] = "market_closed"
            return info

        with state_lock:
            disk = load_state(mc)
        applied = disk.get("applied_exposure_multiplier")
        if applied is None:
            # Bootstrap from the last rebalance's published diagnostics
            # (present since the 2026-07-18 regime-posture publisher).
            for entry in reversed(disk.get("history", [])):
                diag = entry.get("strategy_diagnostics") or {}
                if diag.get("exposure_multiplier") is not None:
                    applied = float(diag["exposure_multiplier"])
                    set_applied_multiplier(mc, applied, logger)
                    logger.info(f"[GATE] {mc.name}: bootstrapped applied "
                                f"multiplier {applied:.4f} from history")
                    break
        if applied is None:
            info["skipped"] = "no_applied_multiplier"
            return info
        applied = float(applied)
        if applied < _MIN_APPLIED_MULT:
            info["skipped"] = "book_essentially_cash"
            return info

        positions = get_positions(mc, logger)
        pos_list = list(positions.values()) if isinstance(positions, dict) \
            else list(positions or [])
        if not pos_list:
            info["skipped"] = "no_positions"
            return info
        held_weights = {
            p["symbol"]: abs(float(p.get("market_value", 0) or 0))
            for p in pos_list
            if float(p.get("market_value", 0) or 0) != 0
        }

        gate = compute_live_gate_multiplier(config, stock_data, macro_data,
                                            held_weights)
        if gate is None:
            info["skipped"] = "gate_uncomputable"
            return info

        info.update({"checked": True, "applied": round(applied, 4), **gate})
        target = float(gate["multiplier"])
        delta = target - applied
        logger.info(
            f"[GATE] {mc.name}: applied={applied:.4f} target={target:.4f} "
            f"(book_dd={gate['book_dd']}) band={GATE_UPDATE_BAND}"
        )
        if abs(delta) < GATE_UPDATE_BAND:
            info["action"] = "hold"
            return info

        factor = target / applied
        scale = _scale_book(mc, pos_list, factor, journal, logger, dry_run)
        info["action"] = "scaled"
        info["scale"] = scale
        if not dry_run and not scale.get("skipped"):
            # Optimistic update, same convention as the tier scaler: the
            # next rebalance resets it from fresh diagnostics anyway.
            achieved_target = target
            if scale.get("factor_capped"):
                achieved_target = applied * float(scale["factor_capped"])
            set_applied_multiplier(mc, achieved_target, logger)
            # Tier anchors must track the new gross (stop_grid semantics:
            # tier fractions apply to the post-scale day book).
            from core.risk import refresh_daily_book_anchor_after_rebalance
            new_gross = scale["book_before"] + scale["delta_usd"]
            refresh_daily_book_anchor_after_rebalance(
                mc, {"target_gross_usd": max(new_gross, 0.0)}, logger,
            )
            logger.warning(
                f"[GATE] {mc.name}: book scaled ×{factor:.4f} "
                f"({applied:.4f} → {achieved_target:.4f}), "
                f"delta ${scale['delta_usd']:,.0f}, "
                f"{scale['orders']} orders ({scale['failed']} failed)"
            )
        return info
    except Exception as e:
        logger.error(f"[GATE] {mc.name}: daily gate update failed: {e}\n"
                     f"{traceback.format_exc()}")
        info["error"] = str(e)
        return info
    finally:
        try:
            if report is not None:
                report.set("gate_update", info)
            # Persist a compact trace for the dashboard (locked
            # read-modify-write; scanner-owned-key style).
            with state_lock:
                disk = load_state(mc)
                trace = disk.get("gate_update_history", [])
                trace.append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    **{k: v for k, v in info.items() if k != "scale"},
                    **({"scale": info["scale"]} if "scale" in info else {}),
                })
                disk["gate_update_history"] = trace[-30:]
                save_state(disk, mc)
        except Exception:
            pass
