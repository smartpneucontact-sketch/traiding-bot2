"""Daily pipeline orchestrator.

`run_pipeline()` is the top-level entry point hit by the daily cron. It:

  1. Discovers active model slots (from config file or env vars)
  2. Downloads bars + macro **once**, shared across all slots
  3. Computes macro features once
  4. Hands off to `run_single_model()` per slot, which:
     - loads the slot's pickled model
     - generates predictions
     - filters inactive Alpaca assets
     - applies V6/V8 regime + conviction-weighted sizing
     - rebalances via `core.orders.rebalance_portfolio`
     - updates persisted state + history

`load_model_bundle()` handles the custom unpickler that maps any
training-script module path (`__main__`, `scripts.train_ml_v6`, …) to
the in-tree `StackedEnsemble` / `EnsembleModel` classes so model
artifacts deserialize cleanly regardless of how they were trained.
"""

from __future__ import annotations

import logging
import os
import pickle
import traceback
from datetime import datetime, timezone


class SlotFailures(RuntimeError):
    """One or more slots raised inside run_pipeline's per-model loop. The
    other slots ran normally; the dashboard marks only `failed` as errored
    (review round 5) while the run itself still reads as errored for
    /ready."""

    def __init__(self, failed: dict):
        self.failed = dict(failed)
        super().__init__(f"{len(failed)} slot(s) failed: "
                         + "; ".join(f"{k}: {v}" for k, v in failed.items()))

from core.combo_strategy import ComboConfig, ComboStrategy
from core.config import ModelConfig, get_active_models
from core.data import download_bars, download_macro
from core.ensemble import EnsembleModel, StackedEnsemble
from core.features import compute_macro_features
from core.inference import predict_rankings
from core.journal import TradeJournal
from core.logging_setup import setup_logging
from core.market import is_market_open
from core.orders import (
    fetch_inactive_assets, liquidate_all_positions, rebalance_portfolio,
)
from core.portfolio import (
    compute_live_regime_score, conviction_weights,
    load_sector_map_for_pipeline, regime_to_exposure,
    sector_neutral_weights,
)
from core.run_report import RunReport
from core.state import (
    load_state, merge_scanner_state, save_state, should_rebalance, state_lock,
)
from core.universe import EXCLUDED_SYMBOLS, get_tradeable_symbols


# Default config. Equivalent to the legacy pipeline-level constants. Callers
# can override per-run via kwargs.
DEFAULT_TOP_N = 20
# 21 trading days (~monthly) — the cadence all three combo_v2 sleeves were
# backtested at (strategies.py rebal_freq=21 in the research repo). The
# previous 5-day setting ran the strategy 4.2x faster than anything the
# champion numbers were computed on.
DEFAULT_HORIZON = 21
# 365 days (was 300) so combo_v1's 12-month momentum lookback has full
# history with margin. v4/v6/v8/v9 only need 252 + some buffer; the extra
# 65 days adds negligible download cost.
DEFAULT_LOOKBACK_DAYS = 365

# Data-quality safeguards: refuse to rebalance with partial yfinance pulls.
# CALIBRATION FIX 2026-08-29: the normal POST-FILTER count (names with the
# required >=250d history out of the ~1,000-name union) is ~500, not ~1,000
# — the old threshold of 500 sat exactly at the normal operating point, so
# natural universe shrinkage (delistings) tipped it under in mid-August and
# silently aborted every run Aug 18-28 (counts 497-499, deterministic; all
# three slots missed their scheduled rebalances). 400 catches the guard's
# actual target — a catastrophic half-degraded download — without tripping
# on ordinary universe drift. Env-overridable for ops.
MIN_STOCKS_REQUIRED = int(os.environ.get("MIN_STOCKS_REQUIRED", "400"))
MIN_MACRO_FEATURES = 15


def load_model_bundle(model_path) -> dict:
    """Load a pickled model bundle.

    combo_v2 bundles pickle `ComboStrategy` / `ComboConfig` under whatever
    module name happened to import them at build time (`__main__` when
    run via `python scripts/build_combo_v2.py`, `core.combo_strategy` when
    re-pickled from inside the package). The custom unpickler remaps both
    so the artifact loads cleanly regardless of origin.

    The StackedEnsemble / EnsembleModel mappings remain in the class_map
    so legacy v4-v9 bundles (if anyone drops them into model/) still load.
    """
    class _PipelineUnpickler(pickle.Unpickler):
        _class_map = {
            "ComboStrategy": ComboStrategy,
            "ComboConfig": ComboConfig,
            # Legacy ML bundles — class still in tree, retained for back-compat.
            "StackedEnsemble": StackedEnsemble,
            "EnsembleModel": EnsembleModel,
        }
        _known_training_modules = {
            "__main__",
            "build_combo_v2", "scripts.build_combo_v2",
            "build_combo_v1", "scripts.build_combo_v1",
            "core.combo_strategy",
            # Legacy ML training module paths
            "scripts.train_ml_v4", "scripts.train_ml_v5",
            "scripts.train_ml_v6", "scripts.train_ml_v7",
            "scripts.train_ml_v8",
            "train_ml_v4", "train_ml_v5",
            "train_ml_v6", "train_ml_v7", "train_ml_v8",
        }

        def find_class(self, module, name):
            if name in self._class_map and module in self._known_training_modules:
                return self._class_map[name]
            return super().find_class(module, name)

    with open(model_path, "rb") as fh:
        return _PipelineUnpickler(fh).load()


def _save_state_merged(state: dict, mc: ModelConfig) -> None:
    """Persist pipeline state without clobbering scanner-owned keys.

    The pipeline holds its state dict across a multi-minute run while the
    60s cutloss scanner (same process) keeps writing the trip flag, peaks,
    cool-down and sold-today ledger. A blind save here would erase a stop
    that fired mid-run — re-opening the same-day-rebuy cascade. Re-read
    the disk copy under the shared lock and adopt the scanner's keys.
    """
    with state_lock:
        merge_scanner_state(state, mc)
        save_state(state, mc)


def run_single_model(
    mc: ModelConfig,
    stock_data: dict,
    macro_features,
    macro_data: dict | None = None,
    dry_run: bool = False,
    force: bool = False,
    top_n: int = DEFAULT_TOP_N,
    horizon: int = DEFAULT_HORIZON,
    stage: str = "main",
) -> None:
    """Execute the pipeline for one model slot.

    stage="auction" (2026-10-15, exec_style="auction" slots only): the
    pre-open pass. Runs the identical data/weights path, then submits the
    rebalance as limit-on-open orders and records them as pending — no
    last_rebalance, no protocol binding, no gate state. The regular
    stage="main" run at 09:35 finalizes those fills (journal + slippage vs
    the official open) and completes any shortfall through the normal
    rebalance, which then does all the usual bookkeeping.
    """
    logger, log_file = setup_logging(mc.name)
    report = RunReport()
    journal = TradeJournal(mc.name)

    logger.info("=" * 70)
    logger.info(f"  ML TRADING PIPELINE - Model: {mc.name.upper()}")
    logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Config: H{horizon}_LongOnly{top_n}")
    logger.info(f"  Mode: {'DRY RUN' if dry_run else 'LIVE PAPER TRADING'}")
    logger.info(f"  Force rebalance: {force}")
    logger.info(f"  Log: {log_file}")
    logger.info(f"  Trade journal: {journal.jsonl_path}")
    logger.info("=" * 70)

    # -- Step 1: Load model -------------------------------------------------
    logger.info("\n[1/5] LOADING MODEL")
    report.start_step("load_model")
    if not mc.model_path.exists():
        logger.error(f"Model not found: {mc.model_path}")
        report.add_error(f"Model not found: {mc.model_path}")
        logger.info(report.format_summary())
        return

    model_bundle = load_model_bundle(mc.model_path)
    model = model_bundle["model"]
    feature_cols = model_bundle.get("feature_cols", []) or []
    # `strategy_type` selects the prediction path. "ml_ranker" (default)
    # keeps the existing v4/v6/v8 flow (per-stock features → model.predict
    # → ranked → conviction weights). "direct_weights" means the model
    # exposes `.compute_weights(stock_data, macro_data) -> {sym: weight}`
    # and we skip the rank+conviction path entirely. Combo_v1 uses the
    # latter (it allocates to macro ETFs + multiple horizons; the rank
    # framework can't express that).
    strategy_type = model_bundle.get("strategy_type", "ml_ranker")
    logger.info(
        f"  Model: {len(feature_cols)} features, "
        f"horizon={model_bundle.get('horizon', '?')}d, "
        f"task={model_bundle.get('task', '?')}, "
        f"strategy_type={strategy_type}"
    )
    logger.info(f"  Trained: {model_bundle.get('saved_at', '?')}")
    report.set("n_features", len(feature_cols))
    report.set("strategy_type", strategy_type)
    report.end_step("load_model")

    # -- Step 2: Check state ------------------------------------------------
    with state_lock:
        state = load_state(mc)
    logger.info(
        f"\n  State: run #{state.get('run_count', 0) + 1}, "
        f"last rebalance: {state.get('last_rebalance', 'never')}"
    )

    # Cut-loss circuit breaker: if portfolio_stop tripped today, sit in cash
    # for the rest of the session. Otherwise rebalance would buy fresh
    # positions that the next 60s cutloss tick would immediately liquidate
    # again (the daily anchor is yesterday's close — see core/risk.py).
    today_iso = datetime.now().strftime("%Y-%m-%d")
    # The auction path keys every date comparison on the New York SESSION
    # date (an evening manual run after 00:00 UTC must not read today's
    # stage record as yesterday's — review round 4). today_iso (container
    # UTC date) stays for the stop/cool-down ledger the scanner stamps.
    session_iso = _session_iso()

    # Auction stage 2 — record opening-auction fills BEFORE any stop /
    # freeze / cool-down early return: the fills are facts regardless of
    # what the risk layer does next (review 2026-10-15).
    if stage == "main" and not dry_run:
        _finalize_auction_if_any(mc, state, journal, logger, report, session_iso,
                                 stock_data=stock_data)

    if state.get("portfolio_stop_tripped_date") == today_iso:
        logger.warning(
            f"  Skipping rebalance — portfolio_stop circuit breaker tripped "
            f"earlier today ({today_iso}). Staying in cash until next session."
        )
        _stamp_last_run(state, stage, dry_run)
        _save_state_merged(state, mc)
        logger.info(report.format_summary())
        return

    # V6 bad-period freeze: proactively halt new entries when SPY is in a
    # ≥12% drawdown from its 21-day peak. The reactive cutloss above fires
    # AFTER damage; the freeze tries to avoid the damage. Backtested to
    # improve Sharpe 1.06→1.19 and Max DD -65%→-56% on the 10-year window.
    # Only fires for `direct_weights` strategies that have a ComboConfig
    # with `enable_drawdown_freeze=True`. Other models are unaffected.
    #
    # `freeze_diag` is the FREEZE-ABLATION PUBLISHER (protocol_exp.json
    # `freeze_disputed`; DECISION_RULE.md adjudication): whenever a slot's
    # config carries the freeze machinery, the multiplier the freeze
    # applied to this run (0.0 = hard cut to cash, 1.0 = evaluated but not
    # in force) is published into the rebalance's strategy_diagnostics /
    # state.history AND into the persisted freeze_state dict, where
    # core/journal.py freeze_context_from_state picks it up for every
    # journal row. Stays None for slots with no freeze machinery (the
    # frozen primary) — their diagnostics/journal rows are unchanged.
    freeze_diag: dict | None = None
    if strategy_type == "direct_weights":
        config_obj = model_bundle.get("combo_config")
        if config_obj is not None and getattr(config_obj, "enable_drawdown_freeze", False):
            from core.combo_strategy import evaluate_spy_drawdown_freeze
            prior_freeze = state.get("freeze_state")
            was_frozen = bool((prior_freeze or {}).get("active"))
            is_frozen, new_freeze, reason = evaluate_spy_drawdown_freeze(
                macro_data or {}, prior_freeze, config_obj, today_iso,
            )
            freeze_diag = {
                "freeze_active": bool(is_frozen),
                "freeze_multiplier": 0.0 if is_frozen else 1.0,
            }
            new_freeze = dict(new_freeze)
            new_freeze["freeze_multiplier"] = freeze_diag["freeze_multiplier"]
            state["freeze_state"] = new_freeze
            report.set("freeze_state", new_freeze)
            report.set("freeze_reason", reason)
            if is_frozen:
                if not was_frozen:
                    logger.warning(f"  [FREEZE] {mc.name}: {reason}")
                    logger.warning("  [FREEZE] Liquidating all positions to cash.")
                else:
                    logger.info(f"  [FREEZE] {mc.name}: {reason}")
                # Liquidate on EVERY frozen run, not just the trigger run —
                # idempotent (no positions → no orders) and self-healing if
                # an earlier liquidation partially failed or the freeze
                # triggered while the market was closed. The previous code
                # called rebalance_portfolio(target_symbols=[]), which
                # aborts on an empty target list — a frozen bot silently
                # kept its full levered book.
                if not dry_run and is_market_open():
                    # Persist the freeze posture BEFORE liquidating: the
                    # journal's best-effort freeze capture
                    # (core/journal.py _load_freeze_context) reads the
                    # slot's ON-DISK state, so without this write the
                    # liquidation rows would carry the stale pre-freeze
                    # posture instead of active=True / multiplier=0.0.
                    _save_state_merged(state, mc)
                    try:
                        liquidation = liquidate_all_positions(
                            mc, journal, logger, report,
                            reason="freeze_liquidation",
                        )
                        report.set("freeze_liquidation", liquidation)
                        if liquidation["n_closed"] or liquidation["n_failed"]:
                            logger.warning(
                                f"  [FREEZE] liquidation: "
                                f"{liquidation['n_closed']} closed, "
                                f"{liquidation['n_failed']} failed"
                            )
                    except Exception as e:
                        logger.error(f"  [FREEZE] liquidation failed: {e}")
                        report.add_error(f"freeze liquidation failed: {e}")
                elif not dry_run:
                    logger.warning(
                        "  [FREEZE] Market closed — liquidation deferred to "
                        "the next run during market hours."
                    )
                _stamp_last_run(state, stage, dry_run)
                _save_state_merged(state, mc)
                logger.info(report.format_summary())
                return
            elif was_frozen:
                logger.info(f"  [FREEZE] {mc.name}: UNFROZEN — {reason}")
                # The freeze liquidated the book; without this the 21-day
                # horizon (counted from the PRE-freeze rebalance) would
                # leave the bot 100% cash for up to ~2 more weeks. The
                # freeze's own min_freeze_days already served as the
                # cool-down — re-enter on this run.
                state["last_rebalance"] = None
            else:
                logger.info(f"  [freeze check] {reason}")

    # Tier-3 re-entry cool-down: after a full portfolio-stop liquidation the
    # scanner schedules re-entry `cutloss_reentry_delay_days` trading days
    # out (core/risk.py). Re-buying the whole book at the very next session
    # was how 2026-06-05's liquidation became 2026-06-09's stop cascade.
    # Checked AFTER the freeze block so a frozen book still gets its
    # liquidation retries during the cool-down. `force=True` (manual /run)
    # overrides — the operator's deliberate escape hatch.
    cooldown_until = state.get("reentry_cooldown_until")
    if cooldown_until and today_iso <= str(cooldown_until) and not force:
        logger.warning(
            f"  Skipping rebalance — re-entry cool-down after portfolio stop "
            f"active through {cooldown_until}."
        )
        _stamp_last_run(state, stage, dry_run)
        _save_state_merged(state, mc)
        logger.info(report.format_summary())
        return
    if cooldown_until:
        state.pop("reentry_cooldown_until", None)
        # Write the removal through to disk now — the merged save at the
        # end of the run would otherwise re-adopt the stale key from the
        # disk copy. Guarded so a NEW cool-down set by a scanner re-trip
        # mid-run is never deleted.
        with state_lock:
            disk = load_state(mc)
            if disk.get("reentry_cooldown_until") == cooldown_until:
                disk.pop("reentry_cooldown_until", None)
                save_state(disk, mc)
        if force and today_iso <= str(cooldown_until):
            logger.warning(
                f"  FORCE OVERRIDE: re-entry cool-down (through {cooldown_until}) "
                f"cleared by manual force run."
            )
        else:
            logger.info(f"  Re-entry cool-down expired ({cooldown_until}); resuming.")

    if (not should_rebalance(state, horizon_days=horizon, force=force)
            and _todays_auction_plan(state, session_iso) is None):
        last = datetime.fromisoformat(state["last_rebalance"])
        days_since = (datetime.now() - last).days
        days_until = horizon - days_since
        logger.info(
            f"  Skipping - last rebalance {days_since}d ago, "
            f"next in {days_until}d"
        )
        # Daily gate-cadence tracking (2026-08-30 alignment fix): the
        # validated backtests re-evaluate the drawdown gates DAILY
        # (validation/book_gate.py prices base_daily * g_t), while live
        # the gate was sampled only at 21-day rebalances — so between
        # rebalances the book could be cut (tiers) but never re-levered.
        # This pass recomputes the same gate stack on the HELD book and
        # pro-rata scales toward it when it moved ≥ GATE_UPDATE_BAND.
        # Runs ONLY here: every stop/freeze/cool-down guard has already
        # returned above, and rebalance days set the gate themselves.
        # Never raises (all failures are contained + reported).
        from core.gate_update import maybe_daily_gate_update
        if stage == "main":
            maybe_daily_gate_update(
                mc, model_bundle, strategy_type, stock_data, macro_data,
                journal, logger, report,
                min_stocks_required=MIN_STOCKS_REQUIRED, dry_run=dry_run,
            )
            # The gate pass writes last_rebalance=None straight to disk when
            # the book is gated ~flat / fully de-levered (re-entry request).
            # last_rebalance is not scanner-owned, so the merged save below
            # would write this run's stale Step-2 value back (review round
            # 7): adopt the disk value — nothing in this branch changes it.
            try:
                with state_lock:
                    disk_lr = load_state(mc).get("last_rebalance", state.get("last_rebalance"))
                if disk_lr != state.get("last_rebalance"):
                    state["last_rebalance"] = disk_lr
            except Exception:
                pass
        _stamp_last_run(state, stage, dry_run)
        _save_state_merged(state, mc)
        logger.info(report.format_summary())
        return

    # Auction-day completion (review round 4): when TODAY's stored plan
    # exists, the main run must finish THAT plan whatever the state of the
    # 09:35 download — the plan carries its own symbols and weights, so
    # the data/macro guards and the whole prediction block are bypassed.
    plan_main = (_todays_auction_plan(state, session_iso)
                 if (stage == "main" and not dry_run) else None)

    # -- Step 3: Verify shared data is sufficient --------------------------
    logger.info("\n[3/5] COMPUTING FEATURES")
    report.start_step("compute_features")
    logger.info(
        f"  Using {len(stock_data)} stocks, "
        f"{len(macro_features.columns)} macro features"
    )
    report.set("stocks_downloaded", len(stock_data))

    if len(stock_data) < MIN_STOCKS_REQUIRED and plan_main is None:
        msg = (
            f"Insufficient stock data: {len(stock_data)} < {MIN_STOCKS_REQUIRED} required. "
            f"Download was likely rate-limited. Aborting to avoid bad trades."
        )
        logger.error(msg)
        report.add_error(msg)
        # Consecutive-abort counter: the Aug 18-28 outage ran 8+ days with
        # no operator-visible alarm because errored runs still refresh the
        # "last run" tile. /ready and /api/status surface this counter as a
        # problem at >=2 so a repeating guard abort cannot stay silent.
        try:
            # Locked, targeted write onto a fresh disk copy: the Step-2
            # `state` snapshot predates concurrent scanner/gate writes,
            # and a bare save of it here would clobber them
            # (2026-08-31 audit). Counted for the MAIN stage only — the
            # pre-open auction stage must not double the alarm.
            if stage == "main" and not dry_run:
                with state_lock:
                    disk = load_state(mc)
                    disk["data_guard_aborts"] = int(disk.get("data_guard_aborts", 0)) + 1
                    save_state(disk, mc)
                state["data_guard_aborts"] = disk["data_guard_aborts"]
        except Exception:
            pass
        report.end_step("compute_features")
        logger.info(report.format_summary())
        return
    if len(macro_features.columns) < MIN_MACRO_FEATURES and plan_main is None:
        msg = (
            f"Insufficient macro data: {len(macro_features.columns)} < "
            f"{MIN_MACRO_FEATURES} required. Download was likely rate-limited. "
            f"Aborting to avoid bad trades."
        )
        logger.error(msg)
        report.add_error(msg)
        report.end_step("compute_features")
        logger.info(report.format_summary())
        return

    # Data guards passed — reset the consecutive-abort alarm counter. Only
    # when the panel REALLY passed: a plan-day bypass on thin data must not
    # silence a running outage alarm (review round 4).
    if (state.get("data_guard_aborts") and stage == "main"
            and len(stock_data) >= MIN_STOCKS_REQUIRED):
        try:
            with state_lock:
                disk = load_state(mc)
                disk["data_guard_aborts"] = 0
                save_state(disk, mc)
            state["data_guard_aborts"] = 0
        except Exception:
            pass

    report.end_step("compute_features")

    # -- Step 4: Generate predictions ---------------------------------------
    logger.info("\n[4/5] GENERATING PREDICTIONS")
    logger.info(f"  Feature version: {mc.feature_version}")

    # Defensive defaults — these get overwritten by the strategy_type branch
    # or the v6/v8 regime block below, but every code path further down
    # reads them at least conditionally. Initialising here means a future
    # edit that drops a guard can't trip NameError.
    target_weights: dict[str, float] | None = None
    regime_exposure = 1.0

    # ── Branch 0: auction-day completion of the stored 09:06 plan ──────
    # The plan IS the book to trade: no re-plan, no re-filtering, and no
    # dependence on this run's panel or macro features (review round 4).
    if plan_main is not None:
        rankings = [(r[0], float(r[1])) for r in (plan_main.get("rankings") or [])]
        target_symbols = list(plan_main.get("target_symbols") or [s for s, _ in rankings])
        target_weights = (dict(plan_main.get("target_weights"))
                          if plan_main.get("target_weights")
                          else {s: float(w) for s, w in rankings})
        diagnostics = dict(plan_main.get("diagnostics") or {})
        try:
            model.last_diagnostics = dict(diagnostics)
        except Exception:
            pass
        if freeze_diag is not None:
            diagnostics = {**diagnostics, **freeze_diag}
        if diagnostics:
            regime_exposure = float(diagnostics.get("exposure_multiplier", 1.0))
            report.set("strategy_diagnostics", diagnostics)
        report.set("direct_weights", {s: round(w, 6) for s, w in target_weights.items()})
        report.set("target_portfolio", rankings)
        logger.info(f"  Completing this morning's auction plan: {len(target_symbols)} names "
                    f"(exposure multiplier {regime_exposure:.2f}) — no re-plan")

    # ── Branch A: direct-weights strategies (combo_v1 etc.) ───────────
    # These bypass the rank+conviction path because they need to size
    # macro ETFs and multi-horizon stock sleeves in one shot.
    elif strategy_type == "direct_weights":
        if not hasattr(model, "compute_weights"):
            msg = (
                f"strategy_type={strategy_type} but model has no "
                f"compute_weights(stock_data, macro_data) method. "
                f"Bundle is malformed; aborting slot."
            )
            logger.error(msg)
            report.add_error(msg)
            logger.info(report.format_summary())
            return

        try:
            raw_weights = model.compute_weights(stock_data, macro_data or {})
        except Exception as e:
            logger.error(f"compute_weights() failed: {e}\n{traceback.format_exc()}")
            report.add_error(f"compute_weights failed: {e}")
            logger.info(report.format_summary())
            return

        if not raw_weights:
            logger.error("compute_weights returned no positions — staying in cash")
            report.add_error("Empty target_weights from direct-weight strategy")
            logger.info(report.format_summary())
            return

        # Sort symbols by weight desc so the report logs the largest first.
        # Build a `rankings` stub so downstream code (journal, state.history)
        # has the (sym, score) tuples it expects.
        sorted_syms = sorted(raw_weights.items(), key=lambda x: -x[1])
        rankings = [(sym, float(w)) for sym, w in sorted_syms]
        target_symbols = [sym for sym, _ in sorted_syms]
        target_weights = {sym: float(w) for sym, w in raw_weights.items()}

        logger.info(
            f"  Direct weights: {len(target_weights)} positions, "
            f"gross exposure={sum(abs(w) for w in target_weights.values()):.0%}"
        )
        for sym, w in sorted_syms[:8]:
            logger.info(f"    {sym:6s}  weight={w:+.4f}")
        if len(sorted_syms) > 8:
            logger.info(f"    ... and {len(sorted_syms) - 8} more")
        report.set("direct_weights", {sym: round(w, 6) for sym, w in target_weights.items()})

        # Surface the strategy's internal regime gating instead of the old
        # hard-coded regime_exposure=1.0, which hid the actual risk posture
        # from the dashboard. ComboStrategy publishes its gate values via
        # `last_diagnostics` after compute_weights().
        diagnostics = getattr(model, "last_diagnostics", None) or {}
        if freeze_diag is not None:
            # Freeze-posture publish (ablation replay): merge the freeze
            # multiplier/state the runner-level freeze applied to THIS
            # run into the diagnostics the dashboard + journal read. The
            # frozen path (multiplier 0.0) liquidates and returns above,
            # so reaching here means the freeze was evaluated and applied
            # no cut — recorded explicitly, never inferred.
            diagnostics = {**diagnostics, **freeze_diag}
        if diagnostics:
            regime_exposure = float(diagnostics.get("exposure_multiplier", 1.0))
            report.set("strategy_diagnostics", diagnostics)
            logger.info(
                f"  Regime gates: spy={diagnostics.get('spy_gate')}, "
                f"book={diagnostics.get('book_gate')}, "
                f"applied exposure multiplier={regime_exposure:.2f}"
            )

        # Skip the prediction-quality filters and jump straight to rebalance.
        # (The rest of the function uses `rankings`, `target_symbols`,
        #  `target_weights` exactly as the rank path does.)

    else:
        rankings = predict_rankings(
            stock_data, macro_features, model, feature_cols, logger, report,
            feature_version=mc.feature_version,
        )

    # ───────────────────────────────────────────────────────────────────
    # The inactive-asset filter + EXCLUDED_SYMBOLS guard + conviction-
    # weighting block below applies only to the ML-ranker path. The
    # direct_weights branch above already filled `target_symbols`,
    # `target_weights`, `rankings`, and `regime_exposure`; nothing else
    # to do until rebalance.
    # ───────────────────────────────────────────────────────────────────
    if plan_main is not None:
        pass  # the 09:06 plan was filtered when it was built
    elif strategy_type == "direct_weights":
        # Light filter: drop any direct-weight position whose ticker is in
        # EXCLUDED_SYMBOLS (e.g. a stock you've added to the exclude list
        # for compliance reasons). ETFs are not in EXCLUDED_SYMBOLS.
        kept = {s: w for s, w in target_weights.items() if s not in EXCLUDED_SYMBOLS}
        dropped = set(target_weights.keys()) - set(kept.keys())
        if dropped:
            logger.info(f"  Excluded {len(dropped)} positions via EXCLUDED_SYMBOLS: {sorted(dropped)}")
            target_weights = kept
            target_symbols = [s for s in target_symbols if s in target_weights]
            rankings = [(s, w) for s, w in rankings if s in target_weights]
        # Inactive-asset filter (best-effort, single round; combo may hold
        # 30-50 positions including ETFs so we check all of them at once).
        if not dry_run and mc.alpaca_key and target_symbols:
            try:
                inactive = fetch_inactive_assets(target_symbols, mc, logger,
                                                 top_n=len(target_symbols))
                if inactive:
                    target_weights = {s: w for s, w in target_weights.items()
                                      if s not in inactive}
                    target_symbols = [s for s in target_symbols if s not in inactive]
                    rankings = [(s, w) for s, w in rankings if s not in inactive]
                    report.set("inactive_assets_filtered", sorted(inactive))
            except Exception as e:
                logger.warning(f"  inactive-asset filter failed: {e}")
        if not target_symbols:
            logger.error("All combo positions filtered out — staying in cash")
            report.add_error("Combo target_symbols empty after filters")
            logger.info(report.format_summary())
            return
        report.set("target_portfolio", rankings)
    else:
        # Filter inactive/untradeable assets (e.g. HOLX after delisting).
        # Walk further down the ranking until we have at least `top_n` tradeable
        # candidates so a few delistings don't abort the rebalance.
        if not dry_run and mc.alpaca_key:
            all_inactive: set[str] = set()
            check_window = top_n * 2
            max_window = min(len(rankings), top_n * 5)  # never check more than top 100
            while True:
                window_syms = [sym for sym, _ in rankings[:check_window]]
                inactive = fetch_inactive_assets(window_syms, mc, logger, top_n=top_n)
                all_inactive.update(inactive)
                tradeable_in_window = [s for s in window_syms if s not in all_inactive]
                if len(tradeable_in_window) >= top_n or check_window >= max_window:
                    break
                check_window = min(check_window + top_n, max_window)
                logger.info(
                    f"  Only {len(tradeable_in_window)}/{top_n} tradeable in top "
                    f"{check_window - top_n}; extending check to top {check_window}"
                )
            if all_inactive:
                rankings = [(sym, pred) for sym, pred in rankings if sym not in all_inactive]
                report.set("inactive_assets_filtered", sorted(all_inactive))

        if len(rankings) < top_n:
            logger.error(f"Only {len(rankings)} predictions - need at least {top_n}")
            report.add_error(f"Insufficient predictions: {len(rankings)} < {top_n}")
            logger.info(report.format_summary())
            return

        # Final guard: re-apply EXCLUDED_SYMBOLS in case a cached universe
        # pre-dates an exclusion change.
        rankings = [(sym, p) for sym, p in rankings if sym not in EXCLUDED_SYMBOLS]

        target_symbols = [sym for sym, _ in rankings[:top_n]]
        report.set("target_portfolio", rankings[:top_n])

        logger.info(f"\n  Target portfolio ({top_n} stocks):")
        for i, (sym, pred) in enumerate(rankings[:top_n]):
            logger.info(f"    {i+1:2d}. {sym:6s}  pred={pred:+6.2f}%")

        # -- V6/V8: regime score → conviction-weighted (optionally sector-neutral)
        # target_weights / regime_exposure already initialized defensively
        # above (line ~232). The v6/v8 block below may overwrite them.

    if (strategy_type != "direct_weights" and plan_main is None
            and mc.feature_version in ("v6", "v8") and macro_data is not None):
        vtag = mc.feature_version.upper()
        logger.info(f"\n  [{vtag}] Computing market regime & conviction weights...")
        try:
            regime_score = compute_live_regime_score(macro_data)
            regime_exposure = regime_to_exposure(regime_score)
            regime_label = (
                "FAVORABLE" if regime_score > 0.3
                else "HOSTILE" if regime_score < -0.3
                else "NEUTRAL"
            )
            logger.info(f"  [{vtag}] Regime score: {regime_score:+.3f} ({regime_label})")
            logger.info(f"  [{vtag}] Exposure multiplier: {regime_exposure:.2f}")

            if mc.feature_version == "v8":
                # Priority: cached sector map > embedded in model bundle > fallback.
                sector_map = load_sector_map_for_pipeline()
                if not sector_map:
                    sector_map = model_bundle.get("sector_map", {})
                if not sector_map:
                    logger.warning(
                        f"  [{vtag}] No sector map found! "
                        f"Falling back to conviction_weights (no sector constraint)."
                    )
                    target_weights = conviction_weights(
                        rankings, top_n=top_n,
                        max_weight_multiple=2.0,
                        regime_exposure=regime_exposure,
                    )
                else:
                    sc = model_bundle.get("sector_config", {})
                    max_per_sector = sc.get("max_per_sector", 3)
                    logger.info(
                        f"  [{vtag}] Sector map: {len(sector_map)} stocks mapped, "
                        f"max {max_per_sector}/sector"
                    )
                    target_weights = sector_neutral_weights(
                        rankings, sector_map,
                        top_n=top_n,
                        max_per_sector=max_per_sector,
                        max_weight_multiple=2.0,
                        regime_exposure=regime_exposure,
                    )
                # Log sector distribution
                sector_counts: dict[str, int] = {}
                for sym_s in target_weights:
                    sec = sector_map.get(sym_s, "Unknown")
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                logger.info(f"  [V8] Sector distribution: {sector_counts}")
            else:
                target_weights = conviction_weights(
                    rankings, top_n=top_n,
                    max_weight_multiple=2.0,
                    regime_exposure=regime_exposure,
                )

            total_alloc = sum(target_weights.values())
            max_w = max(target_weights.values()) if target_weights else 0
            min_w = min(target_weights.values()) if target_weights else 0
            logger.info(
                f"  [{vtag}] Conviction weights: total_alloc={total_alloc:.4f}, "
                f"max={max_w:.4f}, min={min_w:.4f}"
            )
            for sym_w, w in sorted(target_weights.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"    {sym_w:6s}  weight={w:.4f}")
            if len(target_weights) > 5:
                logger.info(f"    ... and {len(target_weights) - 5} more")

            target_symbols = list(target_weights.keys())

            report.set("v6_regime", {
                "regime_score": round(regime_score, 4),
                "regime_label": regime_label,
                "exposure_multiplier": round(regime_exposure, 4),
                "conviction_weights": {s: round(w, 6) for s, w in target_weights.items()},
            })
        except Exception as e:
            logger.warning(
                f"  [{vtag}] Regime/conviction failed, falling back to equal-weight: {e}"
            )
            report.add_error(f"{vtag} regime computation failed: {e}")
            target_weights = None
    elif mc.feature_version in ("v6", "v8"):
        logger.warning(
            f"  [{mc.feature_version.upper()}] No macro_data available "
            f"for regime - using equal-weight"
        )

    # -- Step 5: Rebalance --------------------------------------------------
    logger.info("\n[5/5] REBALANCING PORTFOLIO")

    # Auction stage 1 (pre-open): submit opening-auction orders and stop.
    if stage == "auction":
        from core.market import is_auction_window
        prev_stage = state.get("auction_stage") or {}
        # Earlier sessions' leftovers are NEVER drained here — the drain
        # (polls + backoff) would sit between the data phase and the 09:27
        # submission cutoff (review round 4). They travel into today's
        # record as orphans and the 09:35 main run finalizes them.
        carry_orphans = list(prev_stage.get("orphans") or [])
        if prev_stage.get("date") != session_iso:
            carry_orphans += list(prev_stage.get("pending") or [])
        outcome = None
        dead_today = (prev_stage.get("date") == session_iso
                      and prev_stage.get("outcome") == "submitting")
        venue_clear = False
        if dry_run:
            logger.info("  AUCTION stage DRY RUN — nothing submitted")
            outcome = "dry_run"
        elif prev_stage.get("date") == session_iso and prev_stage.get("pending"):
            # Same-day idempotency: a repeat stage-1 run must never submit a
            # second set and orphan the first (review 2026-10-15).
            logger.warning("  Auction stage already submitted today — not resubmitting")
            outcome = "skipped:already_submitted"
        elif not is_auction_window():
            logger.warning(
                "  Auction stage outside the 09:00-09:27 ET submission "
                "window — skipped; the 09:35 run will rebalance normally.")
            report.add_warning("auction stage skipped: outside submission window")
            outcome = "skipped:window"
        elif (dead_today
              and not (venue_clear := _venue_clear_for_resubmit(mc, session_iso, prev_stage, logger))):
            # An earlier stage run died mid-submission: unless the venue
            # provably holds NO order of today's book, never submit a second
            # one — the 09:35 run adopts whatever the venue holds. The
            # provisional record (plan, persisted pending) is kept on disk
            # for that adoption (review rounds 5-6).
            logger.error("  Auction stage: an earlier run today is marked 'submitting' and the venue "
                         "is not provably empty — NOT resubmitting; the 09:35 run reconciles with the venue")
            report.add_warning("auction stage skipped: prior run died mid-submission")
            outcome = "skipped:prior_submitting"
        else:
            with state_lock:
                fresh = load_state(mc)
            if fresh.get("portfolio_stop_tripped_date") in (today_iso, session_iso):
                logger.warning("  Auction stage skipped: portfolio stop tripped today")
                outcome = "skipped:portfolio_stop"
            else:
                # Yesterday's OFFICIAL close per name — never a today-dated
                # pre-market row, and never a bar older than the prior
                # session (an LOO limit / fill anchor off a two-session-old
                # close is wrong by the overnight gap — review round 4).
                last_close: dict = {}
                last_dates: dict = {}
                for _sym, _df in (stock_data or {}).items():
                    try:
                        _c = _df["close"].dropna()
                        _c = _c[[str(i)[:10] < session_iso for i in _c.index]]
                        if len(_c):
                            last_close[_sym] = float(_c.iloc[-1])
                            last_dates[_sym] = str(_c.index[-1])[:10]
                    except Exception:
                        continue
                expected_prev = _previous_session_iso(session_iso)
                stale = {s for s, d in last_dates.items() if d != expected_prev}
                panel_date = max(last_dates.values()) if last_dates else None
                if panel_date != expected_prev:
                    logger.error(f"  Auction stage skipped: panel's last bar {panel_date} != "
                                 f"prior session {expected_prev} (stale panel) — the 09:35 "
                                 f"run rebalances normally")
                    report.add_warning("auction stage skipped: stale panel")
                    outcome = "skipped:stale_panel"
                    result = None
                else:
                    stale_traded = stale & set(target_symbols)
                    if stale_traded:
                        logger.warning(f"  Auction stage: {len(stale_traded)} target names with a "
                                       f"stale last bar (!= {expected_prev}) left to the 09:35 "
                                       f"completion: {sorted(stale_traded)[:10]}")
                    plan_record = {
                        "target_symbols": list(target_symbols),
                        "rankings": [[s, float(w)] for s, w in rankings],
                        "target_weights": (dict(target_weights) if target_weights else None),
                        # The submitting run's diagnostics travel with the
                        # plan so the completion's gate-state write and
                        # history entry describe the book that was traded.
                        "diagnostics": (getattr(model, "last_diagnostics", None) or None),
                    }
                    # Provisional record BEFORE the first POST, then every
                    # accepted order is appended to disk as it is submitted:
                    # a process death mid-loop must never leave live opg
                    # orders the bot does not know about (review round 5).
                    state["auction_stage"] = {
                        "date": session_iso, "prev_session": expected_prev,
                        "outcome": "submitting", "run_id": None, "n_submitted": 0,
                        "n_skipped": 0, "plan": plan_record, "pending": [],
                        "orphans": carry_orphans,
                    }
                    _save_state_merged(state, mc)

                    def _persist_submitted(entry: dict, rid: str) -> None:
                        with state_lock:
                            d = load_state(mc)
                            rec = d.get("auction_stage") or {}
                            if rec.get("date") != session_iso:
                                return
                            rec.setdefault("pending", []).append(dict(entry))
                            if entry.get("order_id"):
                                rec.setdefault("order_ids", []).append(entry["order_id"])
                            # Running total: survives a finalize that drains
                            # pending before the venue reconciliation (round 8).
                            rec["submitted_notional_usd"] = round(
                                float(rec.get("submitted_notional_usd") or 0)
                                + float(entry.get("notional_est") or 0), 2)
                            rec["run_id"] = rid
                            rec["n_submitted"] = len(rec["pending"])
                            d["auction_stage"] = rec
                            save_state(d, mc)

                    result = rebalance_portfolio(
                        target_symbols, rankings, mc, journal, logger, report,
                        dry_run=False, target_weights=target_weights, auction=True,
                        last_close=last_close, stale_close=stale,
                        on_submitted=_persist_submitted,
                    )
                if result is not None:
                    summ = (result or {}).get("auction") or {}
                    outcome = "submitted" if int(summ.get("n_submitted", 0) or 0) > 0 else "skipped:nothing_submitted"
                    n_stale_traded = int(summ.get("n_stale_close", 0) or 0)
                    state["auction_stage"] = {
                        "date": session_iso,
                        "prev_session": expected_prev,
                        "n_stale_close": n_stale_traded,
                        "run_id": (result or {}).get("run_id") or summ.get("run_id"),
                        "outcome": outcome,
                        "n_submitted": int(summ.get("n_submitted", 0) or 0),
                        "n_skipped": int(summ.get("n_skipped", 0) or 0),
                        "skipped": summ.get("skipped", [])[:40],
                        "submitted_notional_usd": summ.get("submitted_notional_usd"),
                        "auto_scaled_to_buying_power": (result or {}).get("auto_scaled_to_buying_power"),
                        "buy_scale": (result or {}).get("buy_scale"),
                        # The plan the 09:35 completion must finish (never
                        # re-plan against a second download — review 2026-10-15).
                        "plan": plan_record,
                        "pending": summ.get("pending", []),
                        # Every order id this record ever held: the venue
                        # reconciliation must never re-adopt an order that
                        # was already finalized (review round 7).
                        "order_ids": [p.get("order_id") for p in summ.get("pending", []) if p.get("order_id")],
                        "orphans": carry_orphans,
                    }
        # Keep a dead 'submitting' record for the 09:35 venue
        # reconciliation — unless the venue was PROVEN clear, in which case
        # this run's own outcome is the truth (review round 8).
        keep_dead_record = dead_today and not venue_clear
        if (outcome and outcome not in ("submitted", "skipped:already_submitted",
                                        "skipped:prior_submitting")
                and not keep_dead_record):
            # A fresh, minimal record for today — NEVER inherit an earlier
            # day's plan / n_submitted / run_id (review round 2). A dead
            # 'submitting' record is deliberately left intact (round 6).
            state["auction_stage"] = {"date": session_iso, "outcome": outcome,
                                      "n_submitted": 0, "n_skipped": 0,
                                      "pending": [], "plan": None,
                                      "orphans": carry_orphans}
            if dead_today:
                state["auction_stage"]["replaced_dead_record"] = True
        _stamp_last_run(state, stage, dry_run)
        _save_state_merged(state, mc)
        logger.info(report.format_summary())
        return

    # Auction day completion: finish THIS MORNING'S plan, not a re-plan.
    auction_run_id = None
    auction_pending_live = False
    auction_stage_missed = False
    auction_stage_outcome = None
    auction_rerun_after_completion = False
    if stage == "main" and not dry_run:
        st_auction = state.get("auction_stage") or {}
        if _auction_pending_live(state, session_iso):
            # Pre-09:31 only: the auction orders are genuinely live. After
            # the open every opg order is terminal and the completion is
            # delta-based against live positions, so unreadable leftovers
            # (kept pending for journaling retries) never block the
            # completion (review round 3).
            auction_pending_live = True
            logger.warning("  Auction orders still live (pre-open) — completion deferred")
        elif plan_main is not None:
            auction_run_id = (st_auction.get("run_id")
                              or f"{mc.name}_auction_{session_iso.replace('-', '')}_recovered")
            logger.info(f"  Completing this morning's auction plan "
                        f"({len(target_symbols)} names, run_id {auction_run_id}) "
                        f"instead of re-planning")
        elif str(getattr(mc, "exec_style", "") or "") == "auction":
            rec_today = st_auction.get("date") == session_iso
            if rec_today and st_auction.get("completed_at"):
                # A re-run after this morning's plan was already completed:
                # an ordinary re-rebalance, not a lost stage (review round 7).
                auction_rerun_after_completion = True
                logger.info("  Auction plan already completed today — this re-run is a plain rebalance")
            else:
                # An auction slot rebalancing WITHOUT today's plan: the stage
                # never ran / died before submitting / skipped. Recorded so
                # the journal and the AUCTION1 report see a non-auction day
                # (review round 6), with the stage's own outcome if any.
                auction_stage_missed = True
                auction_stage_outcome = st_auction.get("outcome") if rec_today else None
                logger.warning("  Auction slot rebalancing via the marketable-limit path — "
                               f"no auction plan for today (stage outcome: {auction_stage_outcome})")
                report.add_warning("auction stage missed today: rebalanced via marketable-limit")

    # Avoid submitting `time_in_force=day` orders after market close — they
    # would expire unfilled. Skip the trading step on closed days; the
    # predictions and history are still saved.
    result: dict
    if auction_pending_live:
        # Checked BEFORE the market-closed test: a pre-open manual run on a
        # plan day is a deferred completion, not a closed-market skip
        # (review round 5).
        report.add_warning("Rebalance deferred: opening-auction orders still pending")
        result = {"dry_run": False, "skipped_auction_pending": True}
        # Surfaced by /ready until a later main run completes the day
        # (review round 3).
        state["auction_completion_deferred"] = session_iso
    elif not dry_run and not is_market_open():
        logger.warning(
            "  Market is closed (weekend, holiday, or outside 9:30–16:00 ET); "
            "skipping order submission. Predictions saved; rebalance will "
            "retry on the next scheduled run during market hours."
        )
        report.add_warning("Rebalance skipped: market closed")
        result = {"dry_run": False, "skipped_market_closed": True}
    else:
        # The scanner may have tripped the portfolio stop while this run
        # was computing (data download + weights can take minutes). The
        # snapshot loaded in Step 2 is stale by now — re-check the disk
        # flags before placing a full book of orders into a liquidation.
        tripped_mid_run = False
        if not dry_run:
            with state_lock:
                fresh = load_state(mc)
            fresh_cooldown = fresh.get("reentry_cooldown_until")
            if (fresh.get("portfolio_stop_tripped_date") == today_iso
                    or (fresh_cooldown and today_iso <= str(fresh_cooldown)
                        and not force)):
                tripped_mid_run = True
        if tripped_mid_run:
            logger.warning(
                "  Skipping order submission — portfolio stop tripped while "
                "this run was computing. Staying in cash."
            )
            report.add_warning("Rebalance skipped: portfolio stop tripped mid-run")
            result = {"dry_run": False, "skipped_portfolio_stop": True}
        else:
            result = rebalance_portfolio(
                target_symbols, rankings, mc, journal, logger, report,
                dry_run=dry_run, target_weights=target_weights,
                run_id=auction_run_id,
            )
            if (isinstance(result, dict) and auction_run_id
                    and not result.get("aborted_open_orders")):
                # Never merge onto a completion that traded nothing: the
                # protocol and the gate reader would count it as a funded
                # rebalance (review round 8).
                _merge_auction_into_result(result, state.get("auction_stage") or {}, session_iso)
            if isinstance(result, dict) and auction_stage_missed:
                result["auction_stage_missed"] = True
                result["auction_stage_outcome"] = auction_stage_outcome
            if isinstance(result, dict) and auction_rerun_after_completion:
                result["auction_rerun_after_completion"] = True

    # -- Save state ---------------------------------------------------------
    # Only update last_rebalance if a rebalance actually happened. Skipping
    # the order-submission step (market closed, dry_run) means no trades
    # were placed; advancing last_rebalance would lock the model out of its
    # next opportunity by making should_rebalance() think it just rebalanced.
    rebalanced = not (
        dry_run
        or (isinstance(result, dict) and (
            result.get("skipped_market_closed")
            or result.get("skipped_portfolio_stop")
            or result.get("skipped_auction_pending")
            or result.get("aborted_open_orders")))
    )
    if isinstance(result, dict) and result.get("aborted_open_orders"):
        # Orders still live at the venue: the slot stays due; /ready flags
        # it until a later run rebalances (review round 7).
        state["rebalance_aborted_open_orders"] = session_iso
    if rebalanced and isinstance(result, dict):
        if result.get("auction_stage_missed"):
            # Persisted (not just in-memory) so /ready keeps the lost
            # measurement day red across a container restart (round 8).
            state["auction_stage_missed_date"] = session_iso
            # Today's record (if any) is spent by this re-planned rebalance:
            # a venue order adopted later must never resurrect the morning
            # plan and trade the book back to it (review round 9).
            rec = state.get("auction_stage")
            if (isinstance(rec, dict) and rec.get("date") == session_iso
                    and isinstance(rec.get("plan"), dict) and not rec.get("completed_at")):
                rec["completed_at"] = datetime.now(timezone.utc).isoformat()
                rec["superseded_by_missed_rebalance"] = True
        elif auction_run_id and state.get("auction_stage_missed_date") != session_iso:
            state.pop("auction_stage_missed_date", None)
        # Names left untouched because an order was live: kept until a later
        # rebalance trades without any (review rounds 8-9).
        if result.get("blocked_symbols"):
            state["rebalance_blocked_symbols"] = {"date": session_iso,
                                                  "symbols": list(result.get("blocked_symbols") or [])}
        else:
            state.pop("rebalance_blocked_symbols", None)
    if rebalanced:
        state["last_rebalance"] = datetime.now().isoformat()
        state.pop("auction_completion_deferred", None)
        state.pop("rebalance_aborted_open_orders", None)
        if auction_run_id and isinstance(state.get("auction_stage"), dict):
            # The stored plan is spent: a forced same-day re-run re-plans
            # on fresh signals and never re-merges the auction leg.
            state["auction_stage"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        # Re-anchor the cutloss tier scaler's day-start book to what this
        # rebalance actually achieved. The 60s scanner anchored it at
        # ~09:30 on the PRE-rebalance overnight book, but the validated
        # stop_grid.py backtest rebalances BEFORE its tier checks — its
        # TIER_F targets are fractions of the POST-rebalance book. Reads
        # the achieved gross from the rebalance reconciliation
        # (result["reconciliation"]["book"]["achieved_gross_usd"]) with a
        # fallback to the summed target notionals; writes through the
        # scanner's own locked state path. Best-effort: any failure keeps
        # the morning anchor and never touches the completed rebalance.
        try:
            from core.risk import refresh_daily_book_anchor_after_rebalance
            refresh_daily_book_anchor_after_rebalance(
                mc, result if isinstance(result, dict) else None,
                logger=logger,
            )
        except Exception as e:
            logger.warning(
                f"  day-start book anchor refresh failed (non-fatal): {e}"
            )
        # Gate-cadence bookkeeping: record the REALIZED book scaling this
        # rebalance applied (gross_final/gross_pre_gate — includes the
        # gross cap and dust filter, not just the raw gate), the gate_ref
        # for the daily pass's cap re-application, and reset the tier
        # floor (core/gate_update.py, 2026-08-31 audit semantics).
        # Scanner-owned-key contract: written straight to disk under the
        # lock; the merged save below adopts it from disk.
        if strategy_type == "direct_weights":
            try:
                from core.gate_update import config_has_gates, set_gate_state_at_rebalance
                cfg_obj = model_bundle.get("combo_config")
                if config_has_gates(cfg_obj):
                    diag_now = getattr(model, "last_diagnostics", None) or {}
                    set_gate_state_at_rebalance(
                        mc, diag_now, cfg_obj, logger,
                        result=result if isinstance(result, dict) else {},
                        leverage=getattr(mc, "target_leverage", 1.0))
            except Exception as e:
                logger.warning(f"  gate-state write failed (non-fatal): {e}")
        # Forward-test clock: the first FUNDED rebalance starts the
        # pre-registered paper test, binding it to the exact protocol
        # bytes committed beforehand (any later edit → hash mismatch →
        # MODIFIED_AFTER_START). PER-SLOT (core/protocol.py
        # bind_forward_test_start): each slot binds the sha256 of ITS OWN
        # protocol file — the primary still hashes exactly protocol.json
        # (byte-identical to the old inline binding), while combo_v2_exp
        # binds protocol_exp.json and combo_v2_process* binds
        # protocol_process.json. Binding the primary's sha here for a
        # shadow slot would permanently flag it MODIFIED_AFTER_START,
        # because scoring compares against the slot's own file
        # (load_protocol_for). Set-once; never overwritten. Import is
        # local and the whole block best-effort so a missing/broken
        # protocol file can never abort a live run that already traded.
        if not state.get("forward_test_start"):
            try:
                from core.protocol import bind_forward_test_start
                if bind_forward_test_start(state, mc.name, today_iso):
                    logger.info(
                        f"  Forward test clock started: {today_iso} "
                        f"(protocol sha256 {state['protocol_sha256'][:12]}...)"
                    )
            except Exception as e:
                logger.warning(f"  forward_test_start binding failed: {e}")
        # Live-vs-backtest tracking snapshot (Phase A2) for the run report's
        # LIVE VS BACKTEST block. Read-only (one portfolio/history GET) and
        # entirely best-effort: tracking_snapshot returns None on any
        # failure, so a data hiccup can never touch a completed rebalance.
        if state.get("forward_test_start"):
            try:
                from core.tracking import tracking_snapshot
                snap = tracking_snapshot(mc, state, logger=logger)
                if snap:
                    report.set("tracking", snap)
            except Exception as e:
                logger.warning(f"  tracking snapshot failed (non-fatal): {e}")
    _stamp_last_run(state, stage, dry_run)
    state["run_count"] = state.get("run_count", 0) + 1
    history_entry: dict = {
        "date": datetime.now().isoformat(),
        "run_id": f"{mc.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}",
        "target_symbols": target_symbols,
        "strategy_type": strategy_type,
        "result": {k: v for k, v in result.items()
                   if k not in ("sells_detail", "buys_detail")},
    }
    if auction_run_id and rebalanced:
        # Lets a later finalize pass repair this entry's merged
        # reconciliation when the auction leg was unmeasured at 09:35
        # (only for a completion that actually ran — review round 8).
        history_entry["auction_run_id"] = auction_run_id
    # For ml_ranker, `rankings` is (sym, predicted_return); for
    # direct_weights it's (sym, weight). Same dict shape — different
    # semantics. Store under the right key so downstream readers
    # (performance API, dashboards) don't misinterpret weights as
    # alpha predictions.
    if strategy_type == "direct_weights":
        # Full book, not [:top_n] — for direct_weights the book IS the
        # ranking, and redistribution reads this dict to size/select
        # replacements; truncating to 20 of a 40+ name book starved it.
        history_entry["weights"] = {s: round(p, 6) for s, p in rankings}
        diagnostics = getattr(model, "last_diagnostics", None) or {}
        if freeze_diag is not None:
            # Freeze-ablation publisher, part 2: the per-rebalance freeze
            # posture lands in state.history's strategy_diagnostics — the
            # exact place core/journal.py freeze_context_from_state reads
            # the multiplier from for every journalled order row.
            diagnostics = {**diagnostics, **freeze_diag}
        if diagnostics:
            history_entry["strategy_diagnostics"] = diagnostics
    else:
        history_entry["predictions"] = {s: round(p, 4) for s, p in rankings[:top_n]}
    if target_weights is not None:
        history_entry["regime_exposure"] = round(regime_exposure, 4)
        history_entry["conviction_weights"] = {
            s: round(w, 6) for s, w in target_weights.items()
        }
    state["history"].append(history_entry)
    state["history"] = state["history"][-100:]
    _save_state_merged(state, mc)

    summary = report.format_summary()
    logger.info(summary)
    logger.info(f"Log saved to: {log_file}")


def _after_open_et() -> bool:
    """True from 09:31 ET on a trading day — the earliest moment the
    opening-auction orders are terminal (filled or cancelled)."""
    try:
        from zoneinfo import ZoneInfo
        from core.market import session_close
        now = datetime.now(ZoneInfo("America/New_York"))
        if not is_market_open(now):
            # After the session close (13:00 on early-close days — review
            # round 7) every opg order is terminal too; pre-open is False.
            return now >= session_close(now)
        return (now.hour, now.minute) >= (9, 31)
    except Exception:
        return True


def _todays_auction_plan(state: dict, today_iso: str) -> dict | None:
    """Today's stored, not-yet-completed auction plan. A record left in
    'submitting' by a run that died mid-loop counts once it holds persisted
    (or venue-recovered) orders — those orders executed at the open and the
    plan must be completed, not re-planned (review round 5)."""
    st = state.get("auction_stage") or {}
    if (st.get("date") == today_iso
            and st.get("outcome") in ("submitted", "submitting")
            and (int(st.get("n_submitted", 0) or 0) > 0 or st.get("pending"))
            and isinstance(st.get("plan"), dict)
            and not st.get("completed_at")):
        return st["plan"]
    return None


def _stamp_last_run(state: dict, stage: str, dry_run: bool = False) -> None:
    """The persisted last_run is the MAIN stage's LIVE run; the pre-open
    stage keeps its own key so the dashboard's persisted-last-run fallback
    never shows a 09:06 stage as the day's run (review round 2), and a dry
    run stamps last_dry_run only — the boot catch-up decides 'ran today'
    from last_run (review round 6)."""
    if dry_run:
        state["last_dry_run"] = datetime.now().isoformat()
        return
    key = "last_run" if stage == "main" else "auction_last_run"
    state[key] = datetime.now().isoformat()


def _merge_auction_into_result(result: dict, st: dict, today_iso: str) -> None:
    """Fold the auction leg's fill statistics into the completion's
    reconciliation so protocol fill-rate criteria and the gate-state
    funding check see the WHOLE rebalance, not just the residue.

    Idempotent (review round 7): the completion-only flow is snapshotted
    once (recon["flow_completion"]) and the merged flow is rebuilt IN PLACE
    from that snapshot plus the record's CURRENT summary on every call, so
    a later finalize pass (retry, venue-adopted order, late first
    measurement) can re-merge without double counting."""
    try:
        if st.get("date") != today_iso:
            return
        recon = result.get("reconciliation")
        if not isinstance(recon, dict):
            recon = {"flow": {}, "book": None}
            result["reconciliation"] = recon
        flow = recon.get("flow")
        if not isinstance(flow, dict):
            flow = {}
            recon["flow"] = flow
        base = recon.get("flow_completion")
        if not isinstance(base, dict):
            if "fill_rate_notional_auction" in flow:
                # Merged by pre-2026-10-16 code: the completion-only flow
                # cannot be recovered without double counting — leave it.
                return
            base = dict(flow)
            if base.pop("auction_unmeasured", None):
                # Unmeasured merge by earlier code: restore the completion's own rate.
                base["fill_rate_notional"] = base.get("fill_rate_notional_completion")
            base.pop("fill_rate_notional_completion", None)
            recon["flow_completion"] = base
        a = st.get("summary") if isinstance(st.get("summary"), dict) else {}
        a_n = int(a.get("n", 0) or 0)
        a_sub = float(a.get("submitted_notional_usd", 0) or 0)
        submitted_any = int(st.get("n_submitted", 0) or 0) > 0 or bool(st.get("pending"))
        flow.clear()
        flow.update(base)
        if a_n <= 0 or a_sub <= 0:
            recon.pop("auction", None)
            if submitted_any:
                # Orders went to the auction but none could be measured
                # (finalize failed / every entry unreadable / no anchor
                # price): the protocol's fill-rate criterion must read NA,
                # not the completion's ~100% residue rate (review round 5).
                flow["fill_rate_notional_completion"] = base.get("fill_rate_notional")
                flow["fill_rate_notional"] = None
                flow["auction_unmeasured"] = True
            return
        a_filled = int(a.get("n_filled", 0) or 0)
        # Anchor-priced fills (shares executed x prev close) so a gap
        # morning never reads as an unfilled residue.
        a_fill = float(a.get("filled_notional_anchor_usd", a.get("filled_notional_usd", 0)) or 0)
        c_sub = float(base.get("orders_notional_usd", 0) or 0)
        c_fill = float(base.get("filled_notional_usd", 0) or 0)
        # Unfilled auction notional is RE-SUBMITTED by the completion; it
        # must not be counted twice in the denominator (review round 2 —
        # the naive combined rate fired a false K3a on a fully executed book).
        overlap = min(c_sub, max(0.0, a_sub - a_fill))
        plan_notional = a_sub + c_sub - overlap
        flow["fill_rate_notional_completion"] = base.get("fill_rate_notional")
        flow["fill_rate_notional_auction"] = a.get("fill_rate_notional")
        flow["n_orders"] = int(base.get("n_orders", 0) or 0) + a_n
        flow["n_filled"] = int(base.get("n_filled", 0) or 0) + a_filled
        flow["auction_unfilled"] = max(0, a_n - a_filled)
        flow["fill_rate_notional"] = (round(min(1.0, (a_fill + c_fill) / plan_notional), 4)
                                      if plan_notional > 0 else base.get("fill_rate_notional"))
        recon["auction"] = dict(a)
    except Exception:
        pass

def _session_iso() -> str:
    """The New York session date (what every auction-path date is keyed on)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def _previous_session_iso(session_iso: str) -> str:
    """The prior trading session's date (weekends + US market holidays
    skipped) — the date yesterday's OFFICIAL close must carry."""
    from datetime import date, timedelta
    from core.market import US_MARKET_HOLIDAYS
    d = date.fromisoformat(session_iso) - timedelta(days=1)
    while d.weekday() >= 5 or d.isoformat() in US_MARKET_HOLIDAYS:
        d -= timedelta(days=1)
    return d.isoformat()


def _auction_pending_live(state: dict, session_iso: str) -> bool:
    """True only while today's auction orders are genuinely live (pre-09:31
    ET). After the open every opg order is terminal and the completion is
    delta-based, so unreadable leftovers never block it (review round 3)."""
    st = state.get("auction_stage") or {}
    return bool(st.get("date") == session_iso and st.get("pending") and not _after_open_et())


def _panel_open_today(stock_data: dict | None, session_iso: str) -> dict:
    """Today's official open per symbol from the bot's own daily panel —
    only when the panel's last bar is dated today (review round 4)."""
    out: dict = {}
    for sym, df in (stock_data or {}).items():
        try:
            col = "open" if "open" in df.columns else ("Open" if "Open" in df.columns else None)
            if col is None:
                continue
            o = df[col].dropna()
            if len(o) and str(o.index[-1])[:10] == session_iso and float(o.iloc[-1]) > 0:
                out[sym] = float(o.iloc[-1])
        except Exception:
            continue
    return out


def _accumulate_auction_summary(prev: dict | None, new: dict) -> dict:
    """A retry pass over leftovers ADDS to the day's totals instead of
    replacing them (review round 4)."""
    if not prev or int(prev.get("n", 0) or 0) <= 0:
        return dict(new)
    out = dict(prev)
    for k in ("n", "n_filled", "n_unfilled", "n_cancelled_open", "n_not_found",
              "n_already_journaled", "n_budget_deferred"):
        out[k] = int(prev.get(k, 0) or 0) + int(new.get(k, 0) or 0)
    for k in ("submitted_notional_usd", "filled_notional_anchor_usd", "filled_notional_usd"):
        out[k] = round(float(prev.get(k, 0) or 0) + float(new.get(k, 0) or 0), 2)
    est = out["submitted_notional_usd"]
    out["fill_rate_notional"] = round(out["filled_notional_anchor_usd"] / est, 4) if est else None
    # Weight each pass's vw cost by the dollars it was measured over (rows
    # WITH a reference open), not by all filled dollars (review round 5);
    # summaries written before the field existed fall back to filled $.
    pw = float(prev.get("vw_cost_weight_usd", prev.get("filled_notional_usd", 0)) or 0)
    nw = float(new.get("vw_cost_weight_usd", new.get("filled_notional_usd", 0)) or 0)
    pc, nc = prev.get("vw_cost_vs_open_bps"), new.get("vw_cost_vs_open_bps")
    num = den = 0.0
    if pc is not None and pw:
        num += float(pc) * pw
        den += pw
    if nc is not None and nw:
        num += float(nc) * nw
        den += nw
    out["vw_cost_vs_open_bps"] = round(num / den, 2) if den else (pc if pc is not None else nc)
    out["vw_cost_weight_usd"] = round(den, 2)
    out["n_remaining_unread"] = int(new.get("n_remaining_unread", 0) or 0)
    rs = dict(prev.get("reference_sources") or {})
    for k, v in (new.get("reference_sources") or {}).items():
        rs[k] = int(rs.get(k, 0) or 0) + int(v or 0)
    out["reference_sources"] = rs
    out["n_finalize_passes"] = int(prev.get("n_finalize_passes", 1) or 1) + 1
    return out


def _days_between_iso(a: str, b: str) -> int:
    from datetime import date
    try:
        return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days
    except Exception:
        return 0


ORPHAN_MAX_AGE_DAYS = 7
# Today's official opens for the auction rows are fetched from yfinance at
# finalize time (the panel ends at the prior session). Tests disable it.
FETCH_TODAY_OPENS = os.environ.get("AUCTION_FETCH_TODAY_OPENS", "1") != "0"


def _finalize_auction_if_any(mc: ModelConfig, state: dict, journal, logger,
                             report: RunReport, session_iso: str,
                             stock_data: dict | None = None) -> None:
    """Record opening-auction fills for ANY non-empty pending list (the
    orders are terminal after their session's open) and clear it. For
    today's list this waits until 09:31 ET — a pre-open run must never
    cancel the standing auction orders. Earlier sessions' unreadable
    leftovers ("orphans") are retried for ORPHAN_MAX_AGE_DAYS, then
    dropped loudly. A retry pass accumulates into the day's summary.
    Best-effort: a failure never blocks the run."""
    st = state.get("auction_stage") or {}
    pending = st.get("pending") or []
    orphans = st.get("orphans") or []
    changed = False
    rec_date = str(st.get("date") or "")
    past_open = bool(rec_date) and (rec_date != session_iso or _after_open_et())
    if (st.get("outcome") in ("submitting", "submitted") and past_open
            and not st.get("recovery_checked")):
        # Reconcile the record with the venue once per record (review
        # rounds 5-6): a run that died mid-submission ('submitting') or a
        # POST whose response AND by-id lookup were lost ('submitted' with
        # error skips) can leave live opg orders the bot never recorded.
        # Every auction order carries that session's client_order_id
        # prefix, so the venue's own list is the source of truth. A failed
        # listing keeps the record open and is retried next run.
        try:
            from core.orders import recover_unpersisted_auction_orders
            known = ({p.get("order_id") for p in pending}
                     | {o.get("order_id") for o in orphans}
                     | set(st.get("order_ids") or [])
                     | _journal_auction_order_ids(journal, mc.name, rec_date))
            known.discard(None)
            extra = recover_unpersisted_auction_orders(mc, rec_date, known, logger)
        except Exception as e:
            logger.error(f"  auction recovery failed: {e}")
            extra = None
        if extra is None:
            st["recovery_error"] = datetime.now(timezone.utc).isoformat()
            logger.error("  auction venue reconciliation unavailable — record kept open for retry")
            report.add_warning("auction venue reconciliation unavailable (retry next run)")
            changed = True
        else:
            if extra:
                _fill_recovered_anchors(extra, stock_data, rec_date, mc, logger)
                pending = list(pending) + list(extra)
                st["pending"] = pending
                st["order_ids"] = list(st.get("order_ids") or []) + [e.get("order_id") for e in extra
                                                                     if e.get("order_id")]
            was_submitting = st.get("outcome") == "submitting"
            ids_all = set(st.get("order_ids") or []) | {p.get("order_id") for p in pending}
            ids_all.discard(None)
            if was_submitting or extra:
                # Only a record that changed is re-counted. The submitted
                # notional is a running total (persist callback + adopted
                # orders), never re-summed from the CURRENT pending list —
                # a finalize may already have drained it (review round 8).
                st["n_submitted"] = max(len(ids_all), len(pending))
                st["submitted_notional_usd"] = round(
                    float(st.get("submitted_notional_usd") or 0)
                    + sum(float(p.get("notional_est") or 0) for p in extra), 2)
            if was_submitting:
                ever_held = bool(ids_all) or int((st.get("summary") or {}).get("n", 0) or 0) > 0
                st["outcome"] = "submitted" if ever_held else "skipped:nothing_submitted"
                st["recovered_from_submitting"] = True
            st["n_recovered"] = int(st.get("n_recovered", 0) or 0) + len(extra)
            st["recovery_checked"] = datetime.now(timezone.utc).isoformat()
            st.pop("recovery_error", None)
            if pending and not st.get("run_id"):
                # A record that never got its run_id (the persist callback
                # never ran) still needs one shared by every row and the
                # completion leg (review round 6).
                st["run_id"] = f"{mc.name}_auction_{rec_date.replace('-', '')}_recovered"
            for p in pending:
                if not p.get("run_id"):
                    p["run_id"] = st.get("run_id")
            if was_submitting or extra:
                logger.error(f"  auction venue reconciliation: {len(pending)} order(s) on record "
                             f"({len(extra)} recovered from the venue"
                             f"{', record was mid-submission' if was_submitting else ''})")
            changed = True
    if not pending and not orphans:
        if changed:
            state["auction_stage"] = st
            _save_state_merged(state, mc)
        return
    try:
        from core.orders import finalize_auction_orders
    except Exception as e:  # pragma: no cover
        logger.error(f"  auction finalize unavailable: {e}")
        return
    panel_open = _panel_open_today(stock_data, session_iso)
    if orphans:
        keep, dropped = [], []
        for o in orphans:
            sub = str(o.get("submitted_at") or "")[:10]
            if sub and _days_between_iso(sub, session_iso) > ORPHAN_MAX_AGE_DAYS:
                dropped.append(o)
            else:
                keep.append(o)
        if dropped:
            logger.error(f"  auction orphans DROPPED after {ORPHAN_MAX_AGE_DAYS} days unreadable: "
                         f"{[(o.get('symbol'), o.get('order_id')) for o in dropped]}")
            st["orphans_dropped"] = ((st.get("orphans_dropped") or [])
                                     + [{"symbol": o.get("symbol"), "order_id": o.get("order_id"),
                                         "submitted_at": o.get("submitted_at")} for o in dropped])[-50:]
            orphans = keep
            st["orphans"] = keep
            changed = True
        if orphans:
            try:
                o_stats = finalize_auction_orders(mc, orphans, journal, logger, panel_open=panel_open)
                st["orphans"] = list(o_stats.get("remaining") or [])
                st["orphans_summary"] = {k: v for k, v in o_stats.items() if k != "remaining"}
            except Exception as e:
                logger.error(f"  auction orphan finalize failed: {e} — orphans kept")
            changed = True
    if pending:
        if st.get("date") == session_iso and not _after_open_et():
            logger.info("  auction orders pending; finalize deferred until after the open")
        else:
            if st.get("date") == session_iso and FETCH_TODAY_OPENS:
                # The 09:35 panel ends at the prior session (yfinance `end`
                # is exclusive), so fetch today's bar for the pending names
                # only (review round 5). Best-effort; the IEX snapshot
                # remains the fallback inside finalize.
                missing = sorted({p.get("symbol") for p in pending if p.get("symbol")} - set(panel_open))
                if missing:
                    panel_open.update(_fetch_today_opens(missing, session_iso, logger))
            try:
                stats = finalize_auction_orders(mc, pending, journal, logger, panel_open=panel_open)
                new_summary = {k: v for k, v in stats.items() if k != "remaining"}
                prev_summary = st.get("summary") if isinstance(st.get("summary"), dict) else None
                now_iso = datetime.now(timezone.utc).isoformat()
                if prev_summary and int(prev_summary.get("n", 0) or 0) > 0:
                    st["summary"] = _accumulate_auction_summary(prev_summary, new_summary)
                    st["last_retry_summary"] = new_summary
                    st["last_retry_at"] = now_iso
                else:
                    st["summary"] = new_summary
                    st["finalized_at"] = now_iso
                    if st.get("date") != session_iso:
                        st["finalized_late"] = True
                        logger.warning(f"  auction fills from {st.get('date')} finalized late "
                                       f"(journal rows carry the original date)")
                st["pending"] = list(stats.get("remaining") or [])
                st.pop("summary_error", None)
                report.set("auction_fills", st["summary"])
                if st.get("completed_at") and int(stats.get("n", 0) or 0) > 0:
                    # Any measurement AFTER the completion already ran (a
                    # failed 09:35 finalize, a retry, a venue-adopted order):
                    # re-merge the completed run's reconciliation — the merge
                    # is idempotent (review round 7).
                    _repair_history_reconciliation(state, st, session_iso, logger)
            except Exception as e:
                # Keep the list AND any earlier summary intact for the next
                # attempt (round 3/4).
                logger.error(f"  auction finalize failed: {e} — pending kept for retry")
                st["summary_error"] = str(e)
            changed = True
    if changed:
        state["auction_stage"] = st
        _save_state_merged(state, mc)


def _fetch_today_opens(symbols: list, session_iso: str, logger) -> dict:
    """Today's official open for a few symbols from yfinance (daily bar of
    the current session). Best-effort: {} on any failure or when the bar
    returned is not dated today."""
    out: dict = {}
    if not symbols:
        return out
    try:
        import pandas as pd
        import yfinance as yf
        from datetime import date, timedelta
        start = session_iso
        end = (date.fromisoformat(session_iso) + timedelta(days=1)).isoformat()
        queries = {sym: str(sym).replace(".", "-") for sym in symbols}
        df = yf.download(sorted(set(queries.values())), start=start, end=end, interval="1d",
                         progress=False, threads=False, group_by="ticker", auto_adjust=False)
        if df is None or len(df) == 0:
            return out
        multi = isinstance(df.columns, pd.MultiIndex)
        for sym, q in queries.items():
            try:
                col = df[q]["Open"] if multi else df["Open"]
                col = col.dropna()
                if len(col) and str(col.index[-1])[:10] == session_iso and float(col.iloc[-1]) > 0:
                    out[sym] = float(col.iloc[-1])
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"  today's opens fetch failed (non-fatal): {e}")
    return out


def _repair_history_reconciliation(state: dict, st: dict, session_iso: str, logger) -> None:
    """Re-run the auction merge onto the history entry of today's completed
    run (matched by auction_run_id) once a real auction summary exists."""
    try:
        rid = st.get("run_id")
        for h in reversed(state.get("history") or []):
            if rid and h.get("auction_run_id") == rid and isinstance(h.get("result"), dict):
                _merge_auction_into_result(h["result"], st, st.get("date") or session_iso)
                logger.info("  repaired the completed run's reconciliation with the auction leg")
                return
    except Exception as e:
        logger.warning(f"  history reconciliation repair failed: {e}")


def _fill_recovered_anchors(entries: list, stock_data: dict | None, session_iso: str,
                            mc, logger) -> None:
    """Venue-recovered entries lack the anchor (prior-session close) the
    fill-rate statistics need: take it from the panel (prior-session bar,
    dash form), else from the Alpaca snapshot's previous daily bar (valid
    after the open). Entries that stay without an anchor keep
    notional_est 0 and are excluded from the anchor-priced rate."""
    missing = [e for e in entries if not e.get("prev_close")]
    if not missing:
        return
    prev = _previous_session_iso(session_iso)
    for e in missing:
        sym = str(e.get("symbol") or "").replace(".", "-")
        try:
            df = (stock_data or {}).get(sym)
            if df is not None:
                c = df["close"].dropna()
                c = c[[str(i)[:10] <= prev for i in c.index]]
                if len(c) and str(c.index[-1])[:10] == prev and float(c.iloc[-1]) > 0:
                    e["prev_close"] = float(c.iloc[-1])
        except Exception:
            pass
    still = [e for e in missing if not e.get("prev_close")]
    if still:
        try:
            from core.orders import fetch_snapshots
            snaps = fetch_snapshots(sorted({str(e.get("symbol")) for e in still}), mc, logger) or {}
            for e in still:
                snap = snaps.get(str(e.get("symbol"))) or snaps.get(str(e.get("symbol")).replace(".", "-")) or {}
                # prevDailyBar is the record's prior close only while the
                # snapshot's dailyBar IS the record's session (review round
                # 7): for an earlier-session record it is that session's own
                # close — leave the anchor empty instead.
                if (snap.get("day_bar_date") == session_iso
                        and snap.get("prev_close") and float(snap["prev_close"]) > 0):
                    e["prev_close"] = float(snap["prev_close"])
        except Exception as ex:
            logger.warning(f"  recovered-anchor snapshot failed (non-fatal): {ex}")
    for e in missing:
        if e.get("prev_close"):
            e["notional_est"] = round(float(e.get("qty") or 0) * float(e["prev_close"]), 2)


def _journal_auction_order_ids(journal, model_name: str, session_iso: str) -> set:
    """order_ids of the auction rows already journaled for that session
    (trade_id prefix {model}_auction_{YYYYMMDD}_). Best-effort: {}."""
    ids: set = set()
    try:
        import json as _json
        path = getattr(journal, "jsonl_path", None)
        prefix = f"{model_name}_auction_{session_iso.replace('-', '')}_"
        if not path or not os.path.exists(str(path)):
            return ids
        with open(path) as f:
            for line in f:
                if prefix not in line:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                if str(rec.get("trade_id") or "").startswith(prefix) and rec.get("order_id"):
                    ids.add(rec["order_id"])
    except Exception:
        pass
    return ids


def _venue_clear_for_resubmit(mc, session_iso: str, prev_stage: dict, logger) -> bool:
    """True only when the venue PROVABLY holds no opg order of today's
    book (empty listing) and nothing was persisted: then a stage that died
    before any POST may submit after all. A failed listing or any order
    found means 'do not resubmit' (review round 6)."""
    if prev_stage.get("pending"):
        return False
    try:
        from core.orders import recover_unpersisted_auction_orders
        found = recover_unpersisted_auction_orders(mc, session_iso, set(), logger)
    except Exception as e:
        logger.warning(f"  venue check failed: {e}")
        return False
    if found is None or found:
        return False
    logger.warning("  Auction stage: the earlier 'submitting' run left nothing at the venue — "
                   "submitting the book now")
    return True


def _run_plan_only_slots(models: list, logger, *, dry_run: bool, force: bool,
                         top_n: int, horizon: int, reason: str) -> list[str]:
    """The shared data phase failed on a MAIN run: any slot holding TODAY's
    auction plan still gets its completion — the plan carries its own
    symbols/weights and the completion prices off live snapshots, so no
    panel is needed (review round 4, the one major finding). A slot whose
    completion fails here is flagged for /ready. Returns the names run."""
    import pandas as pd
    if dry_run:
        logger.info("plan-only completion skipped: dry run has nothing to complete")
        return []
    session_iso = _session_iso()
    ran: list[str] = []
    for mc in models:
        try:
            with state_lock:
                st = load_state(mc)
            if _todays_auction_plan(st, session_iso) is None:
                continue
            logger.warning(f"{mc.name}: data phase failed ({reason}) but today's auction plan "
                           f"exists — completing the STORED plan without a panel")
            run_single_model(mc, {}, pd.DataFrame(), macro_data={},
                             dry_run=dry_run, force=force, top_n=top_n,
                             horizon=horizon, stage="main")
            ran.append(mc.name)
        except Exception as e:
            logger.error(f"{mc.name}: plan-only completion FAILED: {e}\n{traceback.format_exc()}")
            try:
                with state_lock:
                    d = load_state(mc)
                    d["auction_completion_deferred"] = session_iso
                    save_state(d, mc)
            except Exception:
                pass
    return ran


def run_pipeline(dry_run: bool = False, force: bool = False,
                 model_filter: str | None = None,
                 top_n: int = DEFAULT_TOP_N,
                 horizon: int = DEFAULT_HORIZON,
                 lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                 stage: str = "main") -> None:
    """Execute the full daily pipeline for all active models.

    Downloads data once (universe + bars + macro) and reuses it across
    every active slot. Each slot's exceptions are caught individually so
    one slot's failure doesn't bring down the others.
    """
    # Logging: the auction stage gets its own log name and NO file at all
    # on the 20 of 21 mornings when nothing is due (review round 4) — the
    # discovery/pre-checks below run on a stream-only logger first.
    logger = logging.getLogger("pipeline.pre")
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(_h)
        logger.setLevel(logging.INFO)
    # Capture the pre-check lines so they can be replayed into the run's
    # log file once it exists (a 'treated as due' decision must be
    # auditable from the downloadable log — review round 5).
    _pre_records: list = []

    class _Capture(logging.Handler):
        def emit(self, record):
            _pre_records.append((record.levelno, record.getMessage()))
    _cap = _Capture()
    logger.addHandler(_cap)
    _log_file = None

    # -- Discover models ----------------------------------------------------
    models = get_active_models()
    if model_filter:
        models = [m for m in models if m.name == model_filter]

    if not models:
        logger.error(
            "No active models found. Set MODEL_V4_ALPACA_KEY/SECRET "
            "(or ALPACA_API_KEY/SECRET) and ensure model files exist."
        )
        logger.removeHandler(_cap)
        return

    if stage == "auction":
        models = [m for m in models
                  if str(getattr(m, "exec_style", "") or "") == "auction"]
        if not models:
            logger.info("Auction stage: no slot uses exec_style='auction' — nothing to do")
            logger.removeHandler(_cap)
            return
        # 20 of every 21 sessions no auction slot is due: skip the whole
        # (yfinance) data phase instead of downloading a panel to discover
        # that — halves the daily download load and keeps the 09:35 main
        # run from ever waiting on a pointless stage (round 3 follow-up).
        if not force:
            due = []
            for m in models:
                try:
                    with state_lock:
                        st = load_state(m)
                    if should_rebalance(st, horizon_days=horizon, force=False):
                        due.append(m.name)
                    else:
                        _stamp_last_run(st, "auction", dry_run)
                        _save_state_merged(st, m)
                except Exception as e:
                    logger.warning(f"Auction stage: due-check failed for {m.name} ({e}) — treated as due")
                    due.append(m.name)
            if not due:
                logger.info(f"Auction stage: no auction slot is due to rebalance today "
                            f"({[m.name for m in models]}) — data phase skipped")
                logger.removeHandler(_cap)
                return
            models = [m for m in models if m.name in due]
        # Time-box: the data phase takes minutes; starting it after 09:20
        # ET could not finish inside the 09:27 submission window and could
        # still be running when the 09:35 main job fires.
        try:
            from zoneinfo import ZoneInfo
            from core.market import is_auction_window
            now_et = datetime.now(ZoneInfo("America/New_York"))
            if not is_auction_window(now_et) or (now_et.hour, now_et.minute) >= (9, 20):
                logger.warning("Auction stage: outside the 09:00-09:20 ET start window — skipped")
                logger.removeHandler(_cap)
                return
        except Exception:
            pass
    logging.getLogger("pipeline.pre").removeHandler(_cap)
    logger, _log_file = setup_logging("auction" if stage == "auction" else "main")
    for _lvl, _msg in _pre_records:
        logger.log(_lvl, f"[pre-check] {_msg}")
    logger.info(f"Active models ({stage}): {[m.name for m in models]}")

    # Spec-manifest invariant sweep (2026-08-30): compare the running
    # configuration against spec_manifest.json before touching the market.
    # Violations are CRITICAL-logged, persisted for /api/status, and turn
    # /ready red — but never halt the pipeline (a guard that stops
    # trading does its own damage; see the Aug 18-28 outage).
    try:
        from core.invariants import run_invariant_checks
        run_invariant_checks(logger)
    except Exception as e:
        logger.error(f"invariant sweep failed to run (non-fatal): {e}")

    # -- Download data once (shared across models) --------------------------
    report = RunReport()

    # The shared data phase is wrapped so a single upstream glitch (a
    # malformed Wikipedia table, a yfinance hiccup) logs a FULL traceback to
    # the pipeline log and aborts cleanly, instead of raising an opaque
    # one-line error to the dashboard. Historically a `float < str` crash in
    # get_tradeable_symbols killed every run here with no traceback in the log.
    try:
        logger.info("\n[1/2] DOWNLOADING MARKET DATA (shared)")
        symbols = get_tradeable_symbols(logger, report)
        if not symbols:
            logger.error("No symbols found - cannot proceed")
            if stage == "main":
                _run_plan_only_slots(models, logger, dry_run=dry_run, force=force,
                                     top_n=top_n, horizon=horizon, reason="no symbols")
            return

        stock_data = download_bars(symbols, lookback_days, logger, report)
        macro_data = download_macro(lookback_days, logger, report)

        if not stock_data:
            logger.error("No stock data downloaded - cannot proceed")
            if stage == "main":
                _run_plan_only_slots(models, logger, dry_run=dry_run, force=force,
                                     top_n=top_n, horizon=horizon, reason="no stock data")
            return

        # Universe churn monitor (Aug 18-28 outage follow-up): record the
        # daily usable-bars count + name churn so universe decay is a
        # visible trend, not a silent walk toward the data guard. A
        # partially-failed download is NOT a universe snapshot — skipping
        # it keeps the churn baseline clean (2026-08-31 audit).
        try:
            if stage == "main" and len(stock_data) >= MIN_STOCKS_REQUIRED:
                from core.universe_monitor import record_universe_snapshot
                record_universe_snapshot(sorted(stock_data.keys()), logger)
            elif stage != "main":
                logger.info("universe snapshot skipped: auction stage")
            else:
                logger.warning(
                    f"universe snapshot skipped: only {len(stock_data)} "
                    f"names downloaded (< {MIN_STOCKS_REQUIRED}) — likely a "
                    f"data problem, not real churn")
        except Exception as e:
            logger.warning(f"universe churn snapshot failed (non-fatal): {e}")

        logger.info("\n[2/2] COMPUTING MACRO FEATURES (shared)")
        macro_features = compute_macro_features(macro_data)
        logger.info(
            f"  Macro features: {len(macro_features.columns)} columns, "
            f"{len(macro_features)} days"
        )
    except Exception as e:
        logger.error(
            f"Shared data phase FAILED — no model will run this cycle: {e}\n"
            f"{traceback.format_exc()}"
        )
        report.add_error(f"Shared data phase failed: {e}")
        if stage == "main":
            _run_plan_only_slots(models, logger, dry_run=dry_run, force=force,
                                 top_n=top_n, horizon=horizon, reason=str(e)[:120])
        raise

    # -- Run each model -----------------------------------------------------
    failures: dict = {}
    for mc in models:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"  Running model: {mc.name.upper()}")
        logger.info(f"{'=' * 70}")
        try:
            run_single_model(
                mc, stock_data, macro_features,
                macro_data=macro_data,
                dry_run=dry_run, force=force,
                top_n=top_n, horizon=horizon, stage=stage,
            )
        except Exception as e:
            logger.error(f"Model {mc.name} FAILED: {e}\n{traceback.format_exc()}")
            failures[mc.name] = str(e).splitlines()[0][:200] if str(e) else type(e).__name__

    logger.info("\nAll models complete.")
    if failures:
        # Per-slot isolation keeps the other slots trading, but a failed
        # slot must be LOUD: the dashboard marks the run errored and
        # /ready turns red (review round 4 — an auction stage that failed
        # before submitting used to read as 'success').
        raise SlotFailures(failures)
