"""Daily gate-cadence tracking (2026-08-30 alignment fix; reworked
2026-08-31 after the adversarial audit).

The validated backtests apply the drawdown gates DAILY:
validation/book_gate.py in the research repo computes a daily multiplier
series g_t and prices `base_daily * g_t` — the simulated book re-levers
within days as its own drawdown recovers. The live bot, however, sampled
the gate only inside compute_weights() at each 21-day rebalance; between
rebalances the tier scanner could CUT exposure but nothing could ever
RAISE it. Through the Jul–Aug 2026 crash+recovery that asymmetry held the
gated slots at the crash-day multiplier (~0.34) for six weeks of rally.

This module restores the validated cadence. On every pipeline run that
does NOT rebalance, it recomputes the same gate stack compute_weights()
uses — the same module-level functions, the same config fields and
getattr back-compat defaults — keyed to the HELD book (book_gate.py
computes book drawdown from the held weights, and _book_drawdown
normalizes by gross, so position market values give the same figure). If
the effective target moved by at least GATE_UPDATE_BAND_REL x applied, the live book is
scaled pro-rata toward it, both directions.

2026-08-31 audit corrections baked into this version:
  - get_positions() returns {symbol: {...}} with NO 'symbol' key inside
    the values, and its keys are Alpaca dot-form (BRK.B); positions get
    the symbol injected from the dict key, and held weights are mapped
    back to the universe's dash-form via the day's stock_data keys.
  - The multiplier recorded at rebalance is the REALIZED book scaling
    (gross_final / gross_pre_gate — includes the max_gross_exposure cap
    and the dust filter), not the raw gate diagnostic; the daily pass
    re-applies the gross cap via the rebalance-time gate_ref.
  - Tier cuts are STICKY until the next rebalance (stop_grid semantics):
    the scanner maintains tier_floor_multiplier and the daily target is
    gate_target × tier_floor, so a recovered gate never silently re-buys
    a tier stop's risk reduction beyond its own proportional share.
  - State writes are compositional and race-checked: the pass aborts if
    the applied multiplier changed (or a portfolio stop tripped) between
    its read and its trades, and the final write multiplies the FRESH
    disk value by the factor actually executed instead of overwriting.
  - Commits are proportional to SUBMITTED notional: zero placed orders
    commit nothing, a buying-power-capped batch commits only what it
    actually bought, and the tier day-book anchor is refreshed from the
    submitted amount — never from the pre-cap intention.
  - A raw gate at ~0 (validated g=0) liquidates fully and clears
    last_rebalance, so the next daily run rebuilds the book through the
    normal rebalance path as soon as the gate re-opens (the validated
    daily re-lever from cash).

State contract (all keys scanner-owned; every writer goes straight to
disk under state_lock):
  applied_exposure_multiplier — cumulative scaling the book carries
    (rebalance sets realized value; daily pass composes its executed
    factor; tier scaler multiplies by the fraction it kept).
  tier_floor_multiplier — cumulative tier-cut fraction this cycle
    (rebalance resets to 1.0; tier scaler multiplies down).
  gate_ref — {gross_pre_gate, max_gross_exposure} captured at rebalance.
  gate_data_skips — consecutive thin-data skips (stall alarm at >= 3).

Execution notes: scale trades are pro-rata NOTIONAL MARKET orders (the
same convention as the validated tier layer; the band keeps these events
occasional). Sell records get the full fill/decision-price
instrumentation pass; buy rows are journalled from the submit response.
Gate trades are EXCLUDED from the protocol-facing slippage calibration
(core/execution_stats.py filters non-rebalance actions).

Env knobs:
  GATE_DAILY_UPDATE      "1" (default) enables the daily pass; "0" disables.
  GATE_UPDATE_BAND_REL   minimum |target − applied| / applied move that
                         triggers trading, either direction (default
                         0.035). The absolute GATE_UPDATE_BAND knob was
                         retired 2026-10-15.
"""

import os
import traceback
from datetime import datetime, timezone

import pandas as pd

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

# Relative, SYMMETRIC band (2026-10-15 audit): trade when the target moves
# by at least this fraction of the applied multiplier, in EITHER direction.
# The validated gate has no band at all (book_gate.py applies g_t daily);
# the previous ABSOLUTE 0.10 band let the live book carry 12-33% more gross
# than validated deep in a drawdown and delayed recoveries by the same
# amount. ~3.5% of the book per step keeps dust out (the $50/$5 filters do
# the rest) without a material fidelity gap.
GATE_UPDATE_BAND_REL = float(os.environ.get("GATE_UPDATE_BAND_REL", "0.035"))
GATE_DAILY_UPDATE = os.environ.get("GATE_DAILY_UPDATE", "1") != "0"
if os.environ.get("GATE_UPDATE_BAND"):
    import logging as _logging
    _logging.getLogger("gate_update").warning(
        "GATE_UPDATE_BAND (absolute) is no longer read — the band is "
        "GATE_UPDATE_BAND_REL (relative, symmetric) since 2026-10-15")

# Skip trading when the whole book delta is below this (noise / dust).
_MIN_TOTAL_DELTA_USD = 50.0
# Per-position slices below this are skipped (matches the tier scaler).
_MIN_SLICE_USD = 5.0
# An applied multiplier this small means the book is essentially cash;
# pro-rata scaling of ~nothing cannot rebuild it — the re-entry path is
# a full rebalance (see the applied < _MIN_APPLIED_MULT branch).
_MIN_APPLIED_MULT = 0.05
# A raw gate at/below this is the validated g=0: liquidate fully.
_FULL_DELEVER_EPS = 0.02
# Consecutive thin-data skips before the stall warning fires.
_DATA_STALL_THRESHOLD = 3


def _drop_rows_dated_today(px: pd.DataFrame) -> pd.DataFrame:
    """Drop any price row dated today or later in market time (ET)."""
    try:
        from zoneinfo import ZoneInfo
        today_et = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
        idx = pd.DatetimeIndex(px.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        keep = idx.normalize() < today_et
        if keep.all():
            return px
        return px.loc[keep]
    except Exception:
        return px


def _shortfall_from_result(result: dict, gross_pre, applied_design: float,
                           leverage) -> tuple[float | None, float | None, bool]:
    """(m_real, shortfall, funding_issue) from a rebalance result dict.

    m_real = achieved gross (reconciliation, else targeted gross) over
    equity x leverage x gross_pre_gate — the TRUE level the book landed
    at. shortfall = m_real / design. funding_issue is True only on broker
    evidence that the design book could not be bought: the buying-power
    auto-scale fired, orders were rejected or cancelled unfilled, or the
    notional fill rate was below 97%. Untraded drift inside the 10% hold
    rule is NOT a funding issue (review 2026-10-15) — it moves applied,
    never the floor. (None, None, False) when inputs are missing."""
    try:
        res = result or {}
        recon = res.get("reconciliation") or {}
        book = recon.get("book") or {}
        achieved = (book.get("achieved_gross_usd") or res.get("target_gross_usd"))
        equity = res.get("portfolio_value")
        if not equity:
            # Results written before 2026-10-16 never carried the equity
            # (review round 5): recover it from the book's own leverage
            # ratio (achieved_gross / achieved_leverage) so the one-time
            # reconciliation of the Sep 29 rebalance can still run.
            try:
                ag, al = book.get("achieved_gross_usd"), book.get("achieved_leverage")
                if ag and al and float(al) > 0:
                    equity = float(ag) / float(al)
            except Exception:
                equity = None
        if not (achieved and equity and leverage and gross_pre
                and float(gross_pre) > 0 and float(equity) > 0
                and applied_design > 0):
            return None, None, False
        m_real = float(achieved) / (float(equity) * float(leverage) * float(gross_pre))
        flow = recon.get("flow") or {}
        fill_rate = flow.get("fill_rate_notional")
        funding_issue = bool(
            res.get("auto_scaled_to_buying_power")
            or int(flow.get("n_rejected", 0) or 0) > 0
            or int(flow.get("n_unfilled_cancelled", 0) or 0) > 0
            or (fill_rate is not None and float(fill_rate) < 0.97))
        return m_real, round(m_real / applied_design, 4), funding_issue
    except Exception:
        return None, None, False


def _last_funded_rebalance(history: list) -> dict:
    """The most recent history entry that actually traded (carries a
    target_gross_usd / reconciliation and is not a dry or market-closed
    run). A manual --force outside market hours appends a non-funded
    entry with fresh weights; keying off it would defeat both the
    reconciliation and the target-weight book drawdown (review 2026-10-15)."""
    for h in reversed(history or []):
        res = h.get("result") or {}
        if res.get("dry_run") or res.get("skipped_market_closed") or res.get("skipped_portfolio_stop"):
            continue
        if res.get("target_gross_usd") or res.get("reconciliation"):
            return h
    return {}


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
    """Recompute today's raw gate multiplier with compute_weights' own
    stack.

    `held_weights` is {universe-form symbol: abs market value} of the
    live positions — _book_drawdown normalizes by the sum, so any
    positive scaling of the book gives the same drawdown the validated
    gate sees.

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
        macro_px = _drop_rows_dated_today(macro_px.dropna(how="all"))
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
        # Drop trailing all-NaN rows (a pre-open partial row from the data
        # feed) BEFORE the drawdown read: on 2026-09-23 such a row made
        # _book_drawdown return None, and the combination rule below then
        # treated a missing book gate as spy-only = 1.0 — a data glitch
        # silently OPENING the gate. Daily-pass-only hygiene; the
        # validated compute_weights path is untouched.
        stock_px = stock_px.dropna(how="all")
        # Never read a row dated today (ET): the pass must see prior closes
        # only — a non-NaN partial bar would silently make the read intraday
        # (2026-10-15 review of the 09-23 incident).
        stock_px = _drop_rows_dated_today(stock_px)
        book_dd = _book_drawdown(
            stock_px, held_weights,
            lookback=getattr(config, "book_dd_lookback", 60),
        )
        if book_dd is None:
            # Book gate enabled but uncomputable: the pass must SKIP, never
            # default to an open gate (2026-09-26 live finding).
            return None
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


def set_gate_state_at_rebalance(mc, diagnostics: dict, config,
                                logger=None, result: dict | None = None,
                                leverage=None) -> None:
    """Record the gate reference state after a funded rebalance.

    applied = the REALIZED book scaling gross_final/gross_pre_gate
    (includes the max_gross_exposure cap and the dust filter — the raw
    exposure_multiplier diagnostic overstates the applied scaling
    whenever the cap binds, 2026-08-31 audit). Falls back to the raw
    exposure_multiplier for old diagnostics without the gross fields.

    2026-10-15: when the broker could not fund the design book (the
    buying-power auto-scale, or unfilled orders), the ACHIEVED gross is
    the truth: applied is set from achieved_gross / (equity x leverage x
    gross_pre_gate) and the shortfall becomes the tier floor — sticky
    until the next rebalance, exactly like a tier cut — so the daily pass
    never tries to re-lever into a book the broker already refused. (The
    2026-09-29 rebalance recorded 0.857 while the book was at ~0.67 of
    design; the pass would have mis-scaled every trade.)
    Also resets the thin-data stall counter.
    """
    try:
        exposure = diagnostics.get("exposure_multiplier")
        if exposure is None:
            return
        gross_pre = diagnostics.get("gross_pre_gate")
        gross_final = diagnostics.get("gross_final")
        applied_design = float(exposure)
        if gross_pre and gross_final and float(gross_pre) > 0:
            applied_design = float(gross_final) / float(gross_pre)
        applied, floor = applied_design, 1.0
        m_real, shortfall, funding_issue = _shortfall_from_result(
            result or {}, gross_pre, applied_design, leverage)
        if shortfall is not None and abs(shortfall - 1.0) > 0.03:
            applied = m_real                      # the truthful level
            if funding_issue and shortfall < 0.97:
                floor = min(1.0, shortfall)       # sticky: broker refused
        with state_lock:
            disk = load_state(mc)
            disk["applied_exposure_multiplier"] = round(applied, 4)
            disk["tier_floor_multiplier"] = round(floor, 4)
            disk["gate_ref"] = {
                "gross_pre_gate": float(gross_pre) if gross_pre else None,
                "max_gross_exposure": float(
                    getattr(config, "max_gross_exposure", 1.0) or 1.0),
                "rebalance_shortfall": shortfall,
            }
            disk.pop("gate_data_skips", None)
            save_state(disk, mc)
        if logger:
            logger.info(f"[GATE] {mc.name}: rebalance gate state — applied "
                        f"{applied:.4f} (design {applied_design:.4f}, raw "
                        f"gate {float(exposure):.4f}), tier floor {floor:.4f}")
            if floor < 1.0:
                logger.warning(
                    f"[GATE] {mc.name}: rebalance achieved only "
                    f"{shortfall:.1%} of the design book — recorded as a "
                    f"sticky floor until the next rebalance (no daily "
                    f"re-lever into an unfunded book)")
    except Exception as e:
        if logger:
            logger.error(f"[GATE] {mc.name}: rebalance gate-state write "
                         f"failed: {e}")


def _scale_book(mc, positions: list, factor: float, journal, logger,
                dry_run: bool) -> dict:
    """Pro-rata scale every position's notional by `factor` (market
    orders, both directions). Returns a summary dict whose
    `submitted_usd` (signed: buys +, sells −) is what actually went on
    the wire — the caller commits state from THAT, never from intent."""
    total_mv = sum(float(p.get("market_value", 0) or 0) for p in positions)
    delta_total = total_mv * (factor - 1.0)
    out = {"factor": round(factor, 4), "book_before": round(total_mv, 2),
           "delta_usd": round(delta_total, 2), "orders": 0, "failed": 0,
           "submitted_usd": 0.0, "dry_run": dry_run}
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
            # Keep the reported intent consistent with the capped batch
            # (the stale pre-cap delta inflated the tier anchor and
            # disabled the soft stop for the day — 2026-08-31 audit).
            out["factor_capped"] = round(factor, 4)
            out["factor"] = round(factor, 4)
            out["delta_usd"] = round(delta_total, 2)
            if delta_total < _MIN_TOTAL_DELTA_USD:
                out["skipped"] = "no_buying_power"
                return out

    from core.risk import _instrument_and_journal_sells
    ts = datetime.now(timezone.utc)
    reason = "gate_relevel" if side == "buy" else "gate_delevel"
    records: list[TradeRecord] = []
    submitted = 0.0
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
            submitted += slice_notional if side == "buy" else -slice_notional
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

    out["submitted_usd"] = round(submitted, 2)
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


def _bump_data_skips(mc, logger) -> int:
    """Consecutive thin-data skips — the daily cadence silently pausing
    is itself an alarm-worthy condition (2026-08-31 audit)."""
    n = 0
    try:
        with state_lock:
            disk = load_state(mc)
            n = int(disk.get("gate_data_skips", 0) or 0) + 1
            disk["gate_data_skips"] = n
            save_state(disk, mc)
    except Exception:
        pass
    if n >= _DATA_STALL_THRESHOLD and logger:
        logger.warning(
            f"[GATE] {mc.name}: {n} consecutive thin-data skips — the "
            f"daily gate cadence has been stalled for {n} sessions while "
            f"the book may be drifting off its gate target."
        )
    return n


def _clear_data_skips(mc) -> None:
    try:
        with state_lock:
            disk = load_state(mc)
            if disk.get("gate_data_skips"):
                disk["gate_data_skips"] = 0
                save_state(disk, mc)
    except Exception:
        pass


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
    today_iso = datetime.now().strftime("%Y-%m-%d")
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
            # Not a rebalance — never trade on thin data. Tracked by its
            # own consecutive-skip counter (the rebalance data-guard
            # alarm never fires on non-rebalance days).
            n = _bump_data_skips(mc, logger)
            info["skipped"] = f"insufficient_data_{len(stock_data or {})}"
            info["consecutive_data_skips"] = n
            logger.warning(f"[GATE] {mc.name}: skipping gate check — "
                           f"only {len(stock_data or {})} stocks loaded")
            return info
        if not dry_run and not is_market_open():
            info["skipped"] = "market_closed"
            return info
        _clear_data_skips(mc)

        with state_lock:
            disk = load_state(mc)
        applied = disk.get("applied_exposure_multiplier")
        if applied is None:
            # No bootstrap from history: pre-fix history multipliers
            # ignore any tier cuts made since that rebalance, and every
            # gated slot gets this key set at its next rebalance anyway
            # (2026-08-31 audit removed the bootstrap).
            info["skipped"] = "no_applied_multiplier"
            return info
        applied = float(applied)
        tier_floor = float(disk.get("tier_floor_multiplier", 1.0) or 1.0)
        gate_ref = disk.get("gate_ref") or {}

        # One-time reconciliation (2026-10-15): a gate_ref written by the
        # pre-fix rebalance code carries no 'rebalance_shortfall'. Re-derive
        # it from the last rebalance's own record (achieved gross, equity)
        # and, if the broker funded less than 97% of the design book,
        # adopt the realized multiplier + sticky floor — never trading, only
        # correcting the bookkeeping the daily pass scales against.
        if gate_ref and "rebalance_shortfall" not in gate_ref and not dry_run:
            try:
                last = _last_funded_rebalance(disk.get("history") or [])
                diag = last.get("strategy_diagnostics") or {}
                g0 = gate_ref.get("gross_pre_gate")
                gf = diag.get("gross_final")
                design = (float(gf) / float(g0)) if (g0 and gf and float(g0) > 0) else applied
                m_real, shortfall, funding_issue = _shortfall_from_result(
                    last.get("result") or {}, g0, design,
                    getattr(mc, "target_leverage", 1.0))
                with state_lock:
                    d2 = load_state(mc)
                    ref2 = d2.get("gate_ref") or {}
                    ref2["rebalance_shortfall"] = shortfall
                    ref2["reconciled_at"] = today_iso
                    ref2["funding_issue"] = funding_issue
                    if shortfall is not None and abs(shortfall - 1.0) > 0.03:
                        # COMPOSE onto the current values (review 2026-10-15):
                        # a tier cut or gate sell made since the rebalance
                        # already lives in BOTH applied and the floor;
                        # overwriting applied with the rebalance-time level
                        # would drop it from one side only and sell it twice.
                        cur = float(d2.get("applied_exposure_multiplier") or applied)
                        d2["applied_exposure_multiplier"] = round(cur * shortfall, 4)
                        if funding_issue and shortfall < 0.97:
                            d2["tier_floor_multiplier"] = round(
                                min(1.0, shortfall) * float(d2.get("tier_floor_multiplier", 1.0) or 1.0), 4)
                        logger.warning(
                            f"[GATE] {mc.name}: RECONCILED bookkeeping — the "
                            f"last rebalance funded only {shortfall:.1%} of the "
                            f"design book (realized {m_real:.4f} at rebalance); "
                            f"applied {cur:.4f} → {d2['applied_exposure_multiplier']:.4f}, "
                            f"tier floor → {d2['tier_floor_multiplier']:.4f} "
                            f"(sticky until the next rebalance)")
                        applied = float(d2["applied_exposure_multiplier"])
                        tier_floor = float(d2["tier_floor_multiplier"])
                    d2["gate_ref"] = ref2
                    save_state(d2, mc)
                    gate_ref = ref2
                info["reconciled_shortfall"] = shortfall
            except Exception as e:
                logger.warning(f"[GATE] {mc.name}: bookkeeping reconciliation skipped: {e}")

        if applied < _MIN_APPLIED_MULT:
            # Book is essentially cash — pro-rata scaling cannot rebuild
            # it. The validated gate re-levers from g~0 the moment dd
            # recovers; the live equivalent is a full rebalance, so clear
            # the clock and let tomorrow's run rebuild through the
            # normal path (compute_weights applies the current gate; a
            # still-closed gate yields an empty book and the rebalance
            # safely aborts and retries daily).
            if not dry_run and disk.get("last_rebalance") is not None:
                with state_lock:
                    d2 = load_state(mc)
                    d2["last_rebalance"] = None
                    save_state(d2, mc)
                logger.warning(
                    f"[GATE] {mc.name}: book gated ~flat (applied "
                    f"{applied:.4f}) — cleared last_rebalance so the next "
                    f"run re-enters through a full rebalance when the "
                    f"gate re-opens (validated daily re-lever from cash)."
                )
                info["action"] = "requested_reentry_rebalance"
            info["skipped"] = "book_essentially_cash"
            return info

        positions = get_positions(mc, logger)
        # get_positions returns {alpaca_symbol: {...}} with no 'symbol'
        # inside the values — inject it from the key (2026-08-31 audit:
        # the original list(values()) raised KeyError on every real run).
        if isinstance(positions, dict):
            pos_list = [{**v, "symbol": k} for k, v in positions.items()]
        else:
            pos_list = list(positions or [])
        if not pos_list:
            info["skipped"] = "no_positions"
            return info
        # Map Alpaca dot-form keys (BRK.B) back to the universe's
        # dash-form (BRK-B) so class-share positions land on the price
        # panel's columns in _book_drawdown.
        to_ours = {_to_alpaca_symbol(s): s for s in (stock_data or {})}
        held_weights = {}
        for p in pos_list:
            mv = abs(float(p.get("market_value", 0) or 0))
            if mv > 0:
                held_weights[to_ours.get(p["symbol"], p["symbol"])] = mv

        # The validated signal keys the book drawdown to the decision-date
        # TARGET weights of the held book (book_gate.py), not to drifted
        # market values: in a drawdown MV weights under-weight the deepest
        # losers and bias book_dd less negative (the gate engages later).
        # Use the last rebalance's weights when they cover the held names.
        info["book_dd_weights"] = "market_value"
        try:
            last_funded = _last_funded_rebalance(disk.get("history") or [])
            tw = last_funded.get("weights") or last_funded.get("conviction_weights")
            if isinstance(tw, dict) and tw and held_weights:
                tw_abs = {str(k): abs(float(v)) for k, v in tw.items() if float(v) != 0}
                covered = [sym for sym in held_weights if sym in tw_abs]
                if len(covered) / len(held_weights) >= 0.8:
                    held_weights = {sym: tw_abs[sym] for sym in covered}
                    info["book_dd_weights"] = "target"
        except Exception:
            pass

        gate = compute_live_gate_multiplier(config, stock_data, macro_data,
                                            held_weights)
        if gate is None:
            info["skipped"] = "gate_uncomputable"
            return info

        raw_gate = float(gate["multiplier"])
        # Re-apply the rebalance-time gross-exposure cap: the realized
        # scaling can never exceed max_gross/gross_pre_gate, so an
        # uncapped raw gate must not push effective leverage above the
        # design (2026-08-31 audit).
        g0 = gate_ref.get("gross_pre_gate")
        cap = float(gate_ref.get("max_gross_exposure", 1.0) or 1.0)
        target_gate = raw_gate
        if g0 and float(g0) > 0:
            target_gate = min(raw_gate, cap / float(g0))
        # Tier cuts persist until the next rebalance (stop_grid
        # semantics): the tier floor scales the target so a recovered
        # gate re-levers only its own gate-driven share.
        target = round(target_gate * tier_floor, 4)

        info.update({"checked": True, "applied": round(applied, 4),
                     "tier_floor": round(tier_floor, 4),
                     "target": target, **gate})
        logger.info(
            f"[GATE] {mc.name}: applied={applied:.4f} raw_gate={raw_gate:.4f} "
            f"target={target:.4f} (tier_floor={tier_floor:.4f}, "
            f"book_dd={gate['book_dd']}) band_rel={GATE_UPDATE_BAND_REL}"
        )

        full_delever = raw_gate <= _FULL_DELEVER_EPS
        band_abs = max(GATE_UPDATE_BAND_REL * applied, 0.005)
        info["band_abs"] = round(band_abs, 4)
        if not full_delever and abs(target - applied) < band_abs:
            info["action"] = "hold"
            return info

        factor = 0.0 if full_delever else target / applied

        # Pre-trade guard: abort if the scanner acted while we computed
        # (a tier cut or portfolio stop between our read and our orders
        # would make these trades fight the risk layer).
        if not dry_run:
            with state_lock:
                fresh = load_state(mc)
            if fresh.get("portfolio_stop_tripped_date") == today_iso:
                info["skipped"] = "portfolio_stop_tripped_meanwhile"
                return info
            if fresh.get("applied_exposure_multiplier") is not None and \
                    abs(float(fresh["applied_exposure_multiplier"]) - applied) > 1e-9:
                info["skipped"] = "state_changed_during_pass"
                logger.warning(f"[GATE] {mc.name}: applied multiplier moved "
                               f"during the pass — standing down this run.")
                return info

        scale = _scale_book(mc, pos_list, factor, journal, logger, dry_run)
        info["action"] = "scaled"
        info["scale"] = scale
        if dry_run or scale.get("skipped") or scale["orders"] == 0:
            if scale.get("skipped") or scale["orders"] == 0:
                # Nothing went on the wire — commit nothing (the old
                # unconditional commit desynced the multiplier from the
                # actual book on total order failure, 2026-08-31 audit).
                info["action"] = "no_orders"
            return info

        # Commit proportionally to what was actually submitted, and
        # compose onto the FRESH disk value so a concurrent tier-cut
        # write is scaled, never erased (lost-update fix).
        book_before = float(scale["book_before"]) or 1.0
        achieved_factor = max(0.0, 1.0 + float(scale["submitted_usd"]) / book_before)
        with state_lock:
            d2 = load_state(mc)
            base = d2.get("applied_exposure_multiplier")
            base = float(base) if base is not None else applied
            new_applied = round(base * achieved_factor, 4)
            d2["applied_exposure_multiplier"] = new_applied
            if full_delever:
                # Validated g=0: flat book; re-entry via full rebalance
                # as soon as the gate re-opens.
                d2["last_rebalance"] = None
            save_state(d2, mc)
        info["achieved_factor"] = round(achieved_factor, 4)
        info["new_applied"] = new_applied
        # Tier anchors must track the new gross (stop_grid semantics:
        # tier fractions apply to the post-scale day book). Anchored on
        # SUBMITTED notional — never the pre-cap intention.
        from core.risk import refresh_daily_book_anchor_after_rebalance
        new_gross = book_before + float(scale["submitted_usd"])
        refresh_daily_book_anchor_after_rebalance(
            mc, {"target_gross_usd": max(new_gross, 0.0)}, logger,
        )
        logger.warning(
            f"[GATE] {mc.name}: book scaled ×{achieved_factor:.4f} "
            f"({applied:.4f} → {new_applied:.4f}, target {target:.4f}), "
            f"submitted ${scale['submitted_usd']:,.0f}, "
            f"{scale['orders']} orders ({scale['failed']} failed)"
            + (" — FULL DELEVER, re-entry via next rebalance"
               if full_delever else "")
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
