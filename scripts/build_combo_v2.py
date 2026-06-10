"""Build the combo_v2 model bundle and save to model/combo_v2/model.pkl.

The bundle is a direct-weights strategy (not an ML model). It carries:

  - model: ComboStrategy instance with embedded ComboConfig
  - strategy_type: "direct_weights" (selects runner's compute_weights branch)
  - horizon: 21 (rebalance every 21 trading days — the backtested cadence)
  - feature_cols: empty (this strategy doesn't go through predict_rankings)
  - version: "combo_v2"
  - backtest_reference: the V5 backtest figures from /Traiding 11/REPORT.md

Run from the repo root:

    cd combo-v2-bot
    python scripts/build_combo_v2.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.combo_strategy import ComboConfig, ComboStrategy

OUT = Path(__file__).resolve().parent.parent / "model" / "combo_v2" / "model.pkl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = ComboConfig(
        # Three-sleeve equal-weight blend (must sum to 1.0)
        xs_mom_weight=1.0 / 3.0, xs_mom_top_n=30,
        dual_mom_weight=1.0 / 3.0, dual_mom_top_n=30, dual_mom_vol_target=0.15,
        adaptive_weight=1.0 / 3.0, adaptive_top_n=30,
        adaptive_calm_leverage=1.5,
        adaptive_neutral_leverage=1.0,
        adaptive_stress_leverage=0.5,
        # SPY soft drawdown gate DISABLED per the 2026-06 validation: with
        # the book gate live it costs 0.40 pp/mo for 0.16 pp of extra
        # MaxDD protection (validation/BOOK_GATE_REPORT.md). The V6 hard
        # freeze below stays on — it was independently backtested.
        enable_spy_dd_gate=False,
        spy_dd_lookback=60,
        spy_full_dd=0.08,
        spy_cash_dd=0.18,
        # Book drawdown gate — keyed to the weighted DD of the book's own
        # names from their 60-day highs; catches sector crashes the
        # SPY-keyed gates are blind to (June 2026: SPY -3 %, book -20 %).
        # 12 % → 30 % backtested 2016-2026: 4.72 %/mo, Sharpe 1.16,
        # MaxDD -47.2 % (validation/BOOK_GATE_REPORT.md).
        enable_book_dd_gate=True,
        book_dd_lookback=60,
        book_full_dd=0.12,
        book_cash_dd=0.30,
        # V6 "bad period" freeze — return-max variant (combo_v2_2x_freeze_dd_v1).
        # Backtest 2016-2026: 5.22 %/mo, Sharpe 1.19, MaxDD -55.8 %, Calmar 1.12.
        # See /Traiding 11/REPORT.md V6 section for grid search + alternatives.
        enable_drawdown_freeze=True,
        dd_freeze_pct=0.12,
        dd_peak_lookback=21,
        dd_unfreeze_within_pct=0.08,
        dd_min_freeze_days=14,           # calendar days ≈ 10 trading days
        max_gross_exposure=1.0,
        min_position_weight=0.003,
    )
    strategy = ComboStrategy(config=config)

    bundle = {
        "model": strategy,
        "strategy_type": "direct_weights",
        "feature_cols": [],
        "horizon": 21,
        "version": "combo_v2.2_riskfix",
        "tag": "Combo V2.2 — 3-sleeve blend + V6 freeze + book-DD gate, "
               "21d cadence (2026-06 risk-layer recalibration)",
        "combo_config": config,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        # Updated backtest reference: combo_v2_2x_freeze_dd_v1 (V6 winner).
        # See /Traiding 11/REPORT.md (V5 + V6 sections).
        "backtest_reference": {
            "window": "2016-04-01..2026-03-27",
            "universe_size": 1040,
            "tc_bps_per_side": 5,
            "leverage_for_reference": 2.0,
            "mean_monthly_return": 0.0522,    # V6 (was 0.0496 in V5 baseline)
            "sharpe": 1.19,                    # V6 (was 1.06)
            "max_drawdown": -0.5577,           # V6 (was -0.6528)
            "calmar": 1.123,                   # V6 (was 0.87)
            "rank_among_strategies_tested": 1,
            "freeze_design": "SPY drawdown ≥ 12% from 21d peak; min 14 calendar days frozen; unfreeze within 8% of peak",
            "frozen_day_count_in_backtest": 70,
            "total_trading_days_in_backtest": 2512,
            # 2026-06 risk-layer recalibration (validation/ in the research
            # repo). Each overlay validated separately against the baseline;
            # the JOINT effect of book gate + freeze + portfolio tiers was
            # not backtested as a single configuration.
            "risk_layer_validation": {
                "stop_grid": "validation/STOP_GRID_REPORT.md — per-position "
                             "stops OFF (trailing cost 0.6-2.3pp/mo at any "
                             "threshold); portfolio tiers at -8% levered "
                             "equity: 4.77%/mo, Sharpe 1.17, MaxDD -49.6%",
                "book_gate": "validation/BOOK_GATE_REPORT.md — 12%→30% book-DD "
                             "ramp: 4.72%/mo, Sharpe 1.16, MaxDD -47.2%; SPY "
                             "soft ramp disabled (stacking cost 0.40pp/mo)",
                "as_deployed_june_2026_config": "trailing -5/hard -8/pstop -3 "
                             "unscaled: 1.80%/mo — destroyed the entire edge",
            },
            "v5_baseline_no_freeze": {
                "mean_monthly_return": 0.0496,
                "sharpe": 1.06,
                "max_drawdown": -0.6528,
                "calmar": 0.87,
            },
            # Single-sleeve attribution from V5 results/summary.csv
            "sleeve_xs_momentum_top30":      {"mean_monthly": 0.0294, "sharpe": 1.07, "calmar": 0.85},
            "sleeve_dual_momentum_voltarget": {"mean_monthly": 0.0279, "sharpe": 1.10, "calmar": 0.74},
            "sleeve_adaptive_voltarget":     {"mean_monthly": 0.0237, "sharpe": 0.92, "calmar": 0.80},
        },
    }
    with open(OUT, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved bundle → {OUT}")
    print(f"  size:          {OUT.stat().st_size / 1024:.1f} KB")
    print(f"  strategy_type: {bundle['strategy_type']}")
    print(f"  version:       {bundle['version']}")
    print(f"  saved_at:      {bundle['saved_at']}")
    print(f"  backtest:      {bundle['backtest_reference']['mean_monthly_return']*100:.2f}%/mo @"
          f" {bundle['backtest_reference']['leverage_for_reference']}x leverage")


if __name__ == "__main__":
    main()
