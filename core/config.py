"""Per-slot model configuration — what runs, where its keys live, what
risk overlays are enabled.

A "slot" is one Alpaca account paired with one model file. The dashboard
edits a JSON config (`model_config.json` on the Railway volume) with up
to N slots; this module reads that config and falls back to env vars if
no slot in the config file resolves to a usable model + key pair.

`ModelConfig` is the per-slot runtime dataclass. It carries the model
file path, Alpaca creds, and the cutloss thresholds — everything the
runner needs to operate one slot.

`MODEL_REGISTRY` lists every model name we *can* run (v4..v8) and where
to find its pickled artifact. `MODEL_DESCRIPTIONS` is the human-facing
metadata that the dashboard renders.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# DATA_DIR is Railway's persistent volume; BASE_DIR is the repo root
# (deploy/). Computed at import time so resolving paths doesn't depend on
# any other module being imported first.
_DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
_BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = _DATA_DIR / "model_config.json"


# ═══════════════════════════════════════════════════════════════════════════
# ModelConfig — runtime per-slot info
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuration for a single model instance (one Alpaca account)."""
    name: str                   # e.g. "v4", "v5"
    model_path: Path            # path to model.pkl
    feature_version: str        # "v4", "v5", or "v6"
    alpaca_key: str             # per-slot Alpaca API key
    alpaca_secret: str          # per-slot Alpaca API secret
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    state_path: Path = None     # auto-set if None
    enable_cutloss: bool = False
    # Stop thresholds, recalibrated 2026-06 from the 10-year combo_v2
    # backtest grid (validation/STOP_GRID_REPORT.md in the research repo).
    # Per-position stops are OFF (None): every trailing value tested
    # (-8..-20%) cost 0.6-2.3pp/mo — a tight trail is inside one day's
    # noise for momentum names (the v7-era -5% fired 528×/yr modeled,
    # 90×/5d live) — and hard stops were dominated by the portfolio tier.
    # None or 0 disables a per-position stop.
    cutloss_hard_stop: float | None = None
    cutloss_trailing_stop: float | None = None
    # Portfolio tier base: -4.0 configured × 2x leverage = Tier 1 at -8%
    # levered equity (scale book to 60%), Tier 2 -13.3% (→30%), Tier 3
    # -18.7% (liquidate). Backtested 4.77%/mo, Sharpe 1.17, MaxDD -49.6%
    # vs baseline 4.96%/mo / -65.3%; ~4 events/yr; the only grid config
    # whose Calmar beats baseline under all intraday path assumptions.
    cutloss_portfolio_stop: float = -4.0
    # Scale the portfolio tiers by target_leverage so the configured value
    # keeps its *underlying-move* meaning: -4.0 configured trips Tier 1 at
    # -8% levered equity on a 2x book (same -4% market move either way).
    cutloss_scale_by_leverage: bool = True
    # Trading days to wait before rebuilding the book after a Tier-3
    # portfolio-stop liquidation (next-morning re-entry turned the
    # 2026-06-05 liquidation into the 2026-06-09 stop cascade).
    cutloss_reentry_delay_days: int = 2
    # Target portfolio leverage. 1.0 = invest exactly portfolio_value (cash
    # account behavior). 2.0 = use 2× equity via Reg-T margin. Alpaca paper
    # accounts default to ~2.37x buying_power; orders will fail if the
    # requested leverage exceeds Alpaca's available buying power.
    target_leverage: float = 1.0

    def __post_init__(self):
        if self.state_path is None:
            self.state_path = _DATA_DIR / "state" / f"pipeline_state_{self.name}.json"


# ═══════════════════════════════════════════════════════════════════════════
# Model registry + dashboard descriptions
# ═══════════════════════════════════════════════════════════════════════════

# All known models and where to find their pickled artifact.
# This bot ships a single model: combo_v2. The slot framework remains
# multi-slot so additional strategies can be added without restructuring.
MODEL_REGISTRY: dict[str, dict] = {
    # combo_v2 = direct-weights 3-signal blend (xs_momentum_top30 +
    # dual_momentum_voltarget + adaptive_voltarget_momentum). The runner
    # detects `strategy_type="direct_weights"` in the bundle and calls
    # `model.compute_weights(stock_data, macro_data)` instead of the
    # ranked-prediction path. See core/combo_strategy.py.
    "combo_v2": {"feature_version": "combo", "model_dir": "combo_v2", "fallback_model": None},
}


# Human-facing model documentation shown on the dashboard's About panel.
# Keep these in sync with the actual model behavior — drift is misleading.
MODEL_DESCRIPTIONS: dict[str, dict] = {
    "combo_v2": {
        "title": "Combo V2 — 3-Signal Blend (xs_mom + dual_mom + adaptive_voltarget)",
        "summary": "V5-winning direct-weights strategy. Equal-weight blend of the "
                   "three uncorrelated single-signal strategies that performed best "
                   "across a 45-strategy 10-year backtest. Designed to clear ~3 %/mo "
                   "live at 2.0x leverage, fitting Alpaca paper buying power without "
                   "auto-scale.",
        "architecture": "Direct-weights strategy (no ML model). Three equal-weight sleeves:\n"
                        "  • 1/3 xs_momentum_top30 — long top-30 stocks by 12-1 mo return (monthly rebal)\n"
                        "  • 1/3 dual_momentum_voltarget — top-30 by 6-mo abs-mom > 0, inverse-vol, vol-target 15 %\n"
                        "  • 1/3 adaptive_voltarget_momentum — top-30 by 12-1 mom, inverse-vol, VIX-percentile leverage 0.5x/1.0x/1.5x\n"
                        "After blending, SPY-drawdown gate caps exposure to 0 when SPY is "
                        "≥18 % below its 60-day high (linear ramp from 8 % DD).",
        "features": "Computed inline from raw daily OHLCV bars + VIX. No engineered "
                    "feature set, no ML model. Per-stock: 12-1 / 6-mo cumulative returns, "
                    "60-day realised vol. Macro: SPY close (for DD gate), VIX close (for "
                    "adaptive leverage percentile).",
        "portfolio": "~30-50 stock positions (overlap across the three sleeves keeps the "
                     "count below 3 × 30). Rebalanced every 21 trading days (~monthly), "
                     "matching the backtest cadence. Weights sum to ≤100 % pre-leverage; "
                     "target_leverage (default 2.0x) scales the book in "
                     "rebalance_portfolio().",
        "risk": "Overlays inside the strategy (2026-06 recalibration, each "
                "validated on the 10-yr backtest — see validation/ in the "
                "research repo):\n"
                "  1. Book drawdown gate: weighted DD of the book's own names "
                "from their 60-day highs, ramp 12 % → 30 % toward cash — catches "
                "sector crashes SPY can't see (June 2026: SPY -3 %, book -20 %). "
                "Backtest: 4.72 %/mo, Sharpe 1.16, MaxDD -47.2 %.\n"
                "  2. V6 SPY-drawdown freeze: RETRACTED and disabled 2026-07-12 — "
                "the parameters were grid-searched in-sample and the variant is "
                "worse than no freeze under corrected execution (Traiding 11 "
                "REPORT.md Validity Notice §5). SPY soft ramp 8→18 % also "
                "disabled (redundant with the book gate, cost 0.40 pp/mo).\n"
                "  3. dual_momentum sleeve internally vol-targets to 15 %; adaptive "
                "sleeve auto-deleverages to 0.5x when VIX is in its top-tercile.\n"
                "Plus the runner's cut-loss scanner if `enable_cutloss=True`: "
                "per-position stops OFF (any trailing value cost 0.6-2.3 pp/mo in "
                "backtest); portfolio tiers at -4 % × leverage = -8 % levered "
                "equity at 2x (60 % / 30 % / liquidate at -8 / -13.3 / -18.7 %), "
                "~4 events/yr, 2-trading-day re-entry cool-down after Tier 3. "
                "Backtest: 4.77 %/mo, Sharpe 1.17, MaxDD -49.6 %.",
        "training": "No training. Pure-rule allocator. Backtest in /Traiding 11/"
                    "REPORT.md covers 10 years (2016-04 → 2026-03, 1040 stocks, "
                    "5 bp/side TC). Corrected reference (2026-07-12, engine_v2 "
                    "next_open + 6 %/yr margin): 4.80 %/mo arithmetic / 3.68 %/mo "
                    "compounded, Sharpe 1.06, MaxDD -60 % at 2.0x. CAVEAT: the "
                    "backtest universe is survivors-only, so these are upper "
                    "bounds; treat the 3 %/mo target as unproven pending the "
                    "forward paper test (REPORT.md Validity Notice).",
    },
    # ── The v4-v9 entries below are vestigial — they document the prior
    # bot's ML model family for reference, but no v4-v9 model.pkl ships
    # in this repo. MODEL_REGISTRY only includes combo_v2. Kept inline
    # so future operators understand the lineage. ────────────────────────
    "_v4_legacy_unused": {
        "title": "V4 — Single LightGBM Baseline",
        "summary": "The original production model. A single LightGBM gradient-boosted "
                   "tree trained to predict 5-day forward returns.",
        "architecture": "Single LightGBM model (gradient boosting)",
        "features": "84 features: 62 stock technical indicators (momentum, volatility, "
                    "volume, moving averages, RSI, MACD, Bollinger Bands, etc.) + "
                    "22 macro features (VIX, SPY/QQQ/IWM trends, sector ETFs, "
                    "TLT/HYG credit spreads, gold, oil, dollar index)",
        "portfolio": "Equal-weight top 20 stocks. Rebalance every 5 trading days.",
        "risk": "No intraday monitoring. Market regime not considered. "
                "Holds through volatility events.",
        "training": "Trained on ~4 years of daily data across S&P 500 + Nasdaq 100 + "
                    "Russell 1000 (~1000 stocks). Walk-forward validation.",
    },
    "v5": {
        "title": "V5 — Weighted LightGBM Ensemble",
        "summary": "Three LightGBM sub-models with different hyperparameters, combined "
                   "via weighted average. Aims for more robust predictions than a single model.",
        "architecture": "Weighted ensemble of 3 LightGBM models with optimized weights. "
                        "Final prediction = weighted average of sub-model predictions.",
        "features": "Same 84+46=130 features as V6 (62 stock technicals + 22 macro + "
                    "46 cross-sectional rank features). Uses v6 feature function (superset).",
        "portfolio": "Equal-weight top 20 stocks. Rebalance every 5 trading days.",
        "risk": "No intraday monitoring. No market regime filter.",
        "training": "Trained on ~4 years of daily data. Each sub-model uses different "
                    "regularization (num_leaves, learning rate, min_child_samples) to "
                    "promote diversity.",
    },
    "v6": {
        "title": "V6 — Stacked Ensemble with Regime Filter",
        "summary": "Five diverse base models (3x LightGBM variants, XGBoost, CatBoost) "
                   "combined via Ridge regression meta-learner. Adds market regime "
                   "awareness and conviction-weighted position sizing.",
        "architecture": "Stacked ensemble: 5 base models (LightGBM main, LightGBM "
                        "regularized, LightGBM DART, XGBoost, CatBoost) → Ridge "
                        "meta-learner. Two-level prediction pipeline.",
        "features": "130 features: 62 stock technicals + 22 macro + 46 cross-sectional "
                    "rank features (cs_*). CS features rank each stock vs all peers at "
                    "each time point — captures relative momentum, value, volume.",
        "portfolio": "Conviction-weighted top 20: allocation proportional to prediction "
                     "strength, capped at 2x equal weight. Higher-conviction picks get "
                     "more capital.",
        "risk": "Market regime filter (composite score from VIX level, SPY trend, "
                "SPY momentum, HYG credit spread) → exposure multiplier 0.4–1.0. "
                "Reduces position sizes in hostile regimes. No intraday monitoring.",
        "training": "Trained on ~4 years of daily data. Base models trained independently, "
                    "meta-learner trained on out-of-fold predictions to avoid overfitting.",
    },
    "v7": {
        "title": "V7 — V6 Ensemble + Intraday Risk Management",
        "summary": "Same stacked ensemble as V6, but adds real-time portfolio monitoring "
                   "with automatic stop-loss execution. Scans positions every minute "
                   "during market hours.",
        "architecture": "Same as V6 (5 base models + Ridge meta-learner). Adds a "
                        "real-time cut-loss scanner running every 60 seconds during "
                        "market hours (9:30 AM – 4:00 PM ET).",
        "features": "Same 130 features as V6. Identical prediction pipeline.",
        "portfolio": "Conviction-weighted top 20 (same as V6). Positions actively "
                     "monitored and can be liquidated intraday if stops are hit.",
        "risk": "Three-layer risk management:\n"
                "  1. Hard stop: auto-sell if position drops 8% from entry price\n"
                "  2. Trailing stop: auto-sell if position drops 5% from its peak since entry\n"
                "  3. Portfolio stop: tiered soft scaling — Tier 1 (-3%) → 60% exposure, "
                "Tier 2 (-5%) → 30%, Tier 3 (-7%) → full liquidate + trip flag.\n"
                "Plus V6's market regime filter for position sizing.",
        "training": "Uses same V6 model file — no retraining needed. Risk management "
                    "is purely rule-based on live price data.",
    },
    "v8": {
        "title": "V8 — Sector-Neutral Stacked Ensemble",
        "summary": "Same stacked ensemble as V6, but forces sector diversification. "
                   "Max 3 stocks from any single GICS sector in the portfolio. "
                   "Prevents concentration risk (e.g. V6's semis overweight).",
        "architecture": "Same as V6 (5 base models + Ridge meta-learner). Adds sector "
                        "classification layer and constrained portfolio construction. "
                        "Also includes 3 sector-relative features.",
        "features": "133 features: 130 V6 features + 3 sector-relative features "
                    "(sector_rel_ret_5, sector_rel_ret_20, sector_rel_vol_20). "
                    "Sector-relative features measure stock performance vs its "
                    "sector ETF (XLK, XLF, XLV, etc.).",
        "portfolio": "Sector-constrained conviction-weighted top 20: same conviction "
                     "sizing as V6, but capped at 3 stocks per GICS sector. Skips "
                     "lower-ranked stocks from overrepresented sectors and fills "
                     "with next-best from underrepresented sectors.",
        "risk": "Market regime filter (same as V6): composite score from VIX, "
                "SPY trend, SPY momentum, HYG credit → exposure multiplier 0.4-1.0. "
                "Sector diversification itself is a risk management layer — limits "
                "blow-up from any single sector drawdown.",
        "training": "Trained on same data as V6. Sector map cached from yfinance. "
                    "Walk-forward validation with sector-neutral portfolio evaluation "
                    "at each fold (HHI concentration metric tracked).",
    },
    "v9": {
        "title": "V9 — Bot 8 Quant-v6 Ensemble (Experimental)",
        "summary": "Re-trained 5-day stacked ensemble from the Bot 8 research repo. "
                   "Same base architecture as V6/V7 (5 models + Ridge meta) but with "
                   "a richer 174-column feature set (fundamentals, TLT correlation, "
                   "60d realized skew/kurtosis cross-sectional ranks). Side-by-side "
                   "experimental slot — running alongside V6/V7/V8, not replacing them.",
        "architecture": "Stacked ensemble: 5 base models (LightGBM main, LightGBM "
                        "regularized, LightGBM DART, XGBoost, CatBoost) → Ridge "
                        "meta-learner trained on the 5 base predictions "
                        "(meta_quant_only variant — production does NOT ingest the "
                        "FinBERT/Claude features that the meta_with_all variant uses).",
        "features": "174 features: 68 per-symbol (returns, vol, technicals, etc.) + "
                    "20 macro (VIX/SPY/credit/TLT) + 14 EDGAR fundamentals + 14 "
                    "fundamentals ranks + 7 FINRA short volume + 6 SEC Form 4 "
                    "insider + 11 SEC Form 8-K + 16 cross-sectional ranks + 18 "
                    "sector-relative ranks. Production currently zero-fills the "
                    "~38 alt-data columns (FINRA / Form 4 / Form 8-K / EDGAR not "
                    "yet ingested in production); per-symbol + macro + ranks come "
                    "from the existing v6 feature pipeline.",
        "portfolio": "Conviction-weighted top 20 (same as V6). Allocation proportional "
                     "to prediction strength, capped at 2× equal weight. Held 5 trading "
                     "days then rebalanced.",
        "risk": "Same soft-tiered Phase 1 risk overlay as V7:\n"
                "  1. Hard stop: auto-sell if position drops 8% from entry price.\n"
                "  2. Trailing stop: auto-sell if position drops 5% from its peak.\n"
                "  3. Portfolio stop: TIERED — Tier 1 (-3% daily DD) scales exposure "
                "to 60%, Tier 2 (-5%) to 30%, Tier 3 (-7%) full liquidate + trip flag.\n"
                "Plus V6's market regime filter for position sizing.",
        "training": "Trained 2026-05-06 on Bot 8's research data (916-symbol universe: "
                    "SP500 + SP400 + NDX100). Walk-forward validation, 6 folds. "
                    "In-sample 5d IC: meta_quant_only +0.0106, meta_with_all +0.0130. "
                    "Note the universe gap vs production (SP500 + NDX100 + R1K) — "
                    "predictions on R1K names not in training will be noisier.",
    },
    "combo_v1": {
        "title": "Combo V1 — 4-Signal Multi-Horizon Multi-Asset Momentum",
        "summary": "Direct-weights portfolio that allocates across four uncorrelated "
                   "momentum sleeves AND macro/sector ETFs (not just stocks). Skips "
                   "the rank+conviction path entirely — it emits a complete weights "
                   "dict in one shot. Designed to clear ~3%/month live at 2.5x "
                   "leverage based on 10-year backtest (Apr-2016 → Mar-2026).",
        "architecture": "Direct-weights strategy (no ML model). Four sleeves:\n"
                        "  • 30% xs_momentum_top30 — long top-30 stocks by 12-1 mo return\n"
                        "  • 25% xs_momentum_fast — long top-20 stocks by 3-1 wk return\n"
                        "  • 20% dual_momentum_vol — top-30 by 6-mo return, inverse-vol weighted\n"
                        "  • 25% ts_momentum_multiasset — long 20 macro/sector ETFs with "
                        "positive 12-mo return, inverse-vol weighted\n"
                        "After blending, SPY-drawdown gate caps exposure to 0 when SPY is "
                        "≥18% below its 60-day high (linear ramp from 8% DD).",
        "features": "Computed inline from raw daily OHLCV bars: per-stock 12-1 / 3-1 / "
                    "6-mo cumulative returns + 60-day realized vol; macro ETF "
                    "12-mo returns. No engineered feature set, no ML model.",
        "portfolio": "30-50 positions spanning ~30 stocks + ~10-15 ETFs (SPY/QQQ/IWM/"
                     "TLT/SHY/HYG/GLD/USO/UUP + 11 sector SPDRs). Rebalanced weekly "
                     "(horizon=5d). Weights sum to ≤100% pre-leverage; target_leverage "
                     "(default 2.5x) scales the book in rebalance_portfolio().",
        "risk": "Two overlays inside the strategy itself:\n"
                "  1. SPY drawdown gate: 60-day DD ≥8% → linear ramp toward cash; "
                "≥18% → full cash.\n"
                "  2. Per-sleeve vol-targeting on the dual/TS sleeves (caps each at "
                "15-20% annualized vol).\n"
                "Plus the runner's cut-loss scanner if `enable_cutloss=True` "
                "(hard -8% / trailing -5% / portfolio -3% tiered). At 2.5x leverage "
                "the cutloss tier fires quickly; pair with cutloss_portfolio_stop=-3 "
                "for predictable tier behavior.\n"
                "Reference backtest max-drawdown: -59.5% (2022 momentum/tech reversal).",
        "training": "No training. The strategy is a pure-rule allocator. The "
                    "backtest in /Traiding 11/REPORT.md covers 10 years (2016-04 → "
                    "2026-03, 1040 stocks, 5 bp/side TC) and produced 4.17%/mo mean, "
                    "2.95%/mo median, Sharpe 1.00, 10 of 11 years positive, with the "
                    "single bad year (2022) at -38.6%. Live expectation after a "
                    "15-30% friction haircut: 2.9-3.5%/mo.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard-managed config (model_config.json)
# ═══════════════════════════════════════════════════════════════════════════

# Stamp written into model_config.json once the 2026-06 stop recalibration
# has been applied to the stored slots. Bump when thresholds are re-tuned.
CUTLOSS_CALIBRATION_VERSION = "2026-06-riskfix-v1"


def _default_config() -> dict:
    """Generate default config — single slot for combo_v2."""
    return {
        "slots": [
            {
                # combo_v2 — V5-winning 3-sleeve blend at 2.0x leverage.
                # Targets ~3.5-4.2 %/mo live per the 10-year backtest
                # (4.96 %/mo mean, Sharpe 1.06, Calmar 0.87).
                # Ships with enabled=False until the operator adds an
                # Alpaca key via the dashboard Settings panel.
                "slot_id": 1,
                "model": "combo_v2",
                "enabled": False,
                "alpaca_key": "",
                "alpaca_secret": "",
                "enable_cutloss": True,
                "cutloss_hard_stop": None,
                "cutloss_trailing_stop": None,
                "cutloss_portfolio_stop": -4.0,
                "cutloss_scale_by_leverage": True,
                "cutloss_reentry_delay_days": 2,
                "target_leverage": 2.0,
            },
        ],
        "cutloss_calibration": CUTLOSS_CALIBRATION_VERSION,
        "updated_at": None,
    }


def _migrate_cutloss_calibration(config: dict) -> dict:
    """One-time migration of stored slot configs to the 2026-06 calibrated
    stop thresholds.

    model_config.json lives on the Railway volume and survives deploys, so
    changing code defaults alone never reaches a live bot. Slots still
    carrying the exact v7-era defaults (-8 / -5 / -3) — which were never
    hand-tuned, only inherited — are moved to the recalibrated values; any
    other values are treated as operator overrides and left alone. The new
    behavioral keys (leverage scaling, re-entry delay) are added to every
    slot that lacks them.
    """
    if config.get("cutloss_calibration") == CUTLOSS_CALIBRATION_VERSION:
        return config

    changed = False
    for slot in config.get("slots", []):
        if (slot.get("cutloss_hard_stop") == -8.0
                and slot.get("cutloss_trailing_stop") == -5.0
                and slot.get("cutloss_portfolio_stop") == -3.0):
            slot["cutloss_hard_stop"] = None
            slot["cutloss_trailing_stop"] = None
            slot["cutloss_portfolio_stop"] = -4.0
            print(f"[CONFIG MIGRATE] slot {slot.get('slot_id')}: stop "
                  f"thresholds moved from v7-era defaults to 2026-06 "
                  f"calibration (per-position stops OFF, portfolio -4 "
                  f"scaled by leverage = -8% levered tiers at 2x)", flush=True)
            changed = True
        if "cutloss_scale_by_leverage" not in slot:
            slot["cutloss_scale_by_leverage"] = True
            changed = True
        if "cutloss_reentry_delay_days" not in slot:
            slot["cutloss_reentry_delay_days"] = 2
            changed = True

    config["cutloss_calibration"] = CUTLOSS_CALIBRATION_VERSION
    save_model_config(config)
    if changed:
        print(f"[CONFIG MIGRATE] model_config.json stamped "
              f"{CUTLOSS_CALIBRATION_VERSION}", flush=True)
    return config


def _parse_stop_threshold(value) -> float | None:
    """Normalize a per-position stop threshold from the config file.

    None, empty string, or 0 mean "disabled" (a 0 threshold would fire on
    every tick that isn't strictly positive). Anything else is a float.
    """
    if value in (None, "", 0, 0.0):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v < 0 else None


def load_model_config() -> dict:
    """Load slot config from the persistent JSON file.

    Falls back to a fresh default config (all-empty keys) if the file is
    missing or unparseable. The dashboard never sees a hard failure.
    """
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return _default_config()


def save_model_config(config: dict) -> None:
    """Save slot config to the persistent JSON file. Sets `updated_at`.

    Atomic (temp file + os.replace): a crash mid-write would otherwise
    leave unparseable JSON, and load_model_config would then silently fall
    back to the default config — empty keys, trading disabled.
    """
    config["updated_at"] = datetime.now().isoformat()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, indent=2))
    os.replace(tmp, CONFIG_PATH)


def _resolve_model_path(model_name: str) -> Path:
    """Find model.pkl for a given model name.

    Looks first at `model/<dir>/model.pkl`, then at the registry's
    `fallback_model` filename (used by v4's legacy `ml_v4_model.pkl`).
    Returns the primary path even when missing — the caller checks
    `.exists()` to handle absence.
    """
    reg = MODEL_REGISTRY.get(model_name, {})
    model_dir = reg.get("model_dir", model_name)
    path = _BASE_DIR / "model" / model_dir / "model.pkl"
    if path.exists():
        return path
    fallback = reg.get("fallback_model")
    if fallback:
        alt = _BASE_DIR / "model" / fallback
        if alt.exists():
            return alt
    return path


def get_active_models() -> list[ModelConfig]:
    """Build active models from config file, falling back to env vars.

    Priority: dashboard-edited config file slots → env vars. Only slots
    with valid API keys AND an existing model file are activated. When no
    slot in the config resolves, env vars get a chance — that's how the
    bot bootstraps on a fresh container before the dashboard saves
    anything.
    """
    config = _migrate_cutloss_calibration(load_model_config())
    models: list[ModelConfig] = []
    activated_via_config = False

    # -- Try config file first --
    for slot in config.get("slots", []):
        if not slot.get("enabled", True):
            continue
        model_name = slot.get("model", "")
        if model_name not in MODEL_REGISTRY:
            continue

        key = slot.get("alpaca_key", "")
        secret = slot.get("alpaca_secret", "")
        model_path = _resolve_model_path(model_name)
        reg = MODEL_REGISTRY[model_name]

        print(f"[MODEL DETECT] slot {slot.get('slot_id')}: model={model_name}, "
              f"key={'YES' if key else 'NO'}, secret={'YES' if secret else 'NO'}, "
              f"model_file={model_path} exists={model_path.exists()}", flush=True)

        if key and secret and model_path.exists():
            activated_via_config = True
            models.append(ModelConfig(
                name=model_name,
                model_path=model_path,
                feature_version=reg["feature_version"],
                alpaca_key=key,
                alpaca_secret=secret,
                enable_cutloss=slot.get("enable_cutloss", False),
                cutloss_hard_stop=_parse_stop_threshold(
                    slot.get("cutloss_hard_stop")),
                cutloss_trailing_stop=_parse_stop_threshold(
                    slot.get("cutloss_trailing_stop")),
                cutloss_portfolio_stop=float(
                    slot.get("cutloss_portfolio_stop", -4.0) or -4.0),
                cutloss_scale_by_leverage=bool(
                    slot.get("cutloss_scale_by_leverage", True)),
                cutloss_reentry_delay_days=int(
                    slot.get("cutloss_reentry_delay_days", 2) or 2),
                target_leverage=float(slot.get("target_leverage", 1.0) or 1.0),
            ))

    # -- Fallback to env vars if no slot in the config actually activated.
    # Keys-but-no-pickle is treated as "config didn't work" so env vars
    # still get a chance.
    if not activated_via_config:
        print("[MODEL DETECT] No models activated from config; trying env vars", flush=True)
        for name in MODEL_REGISTRY:
            key_env = f"MODEL_{name.upper()}_ALPACA_KEY"
            secret_env = f"MODEL_{name.upper()}_ALPACA_SECRET"
            # Legacy ALPACA_API_KEY env vars are honored only for v4
            # (the original single-model setup).
            fallback_key = "ALPACA_API_KEY" if name == "v4" else None
            fallback_secret = "ALPACA_SECRET_KEY" if name == "v4" else None

            key = os.environ.get(key_env, "")
            secret = os.environ.get(secret_env, "")
            if not key and fallback_key:
                key = os.environ.get(fallback_key, "")
            if not secret and fallback_secret:
                secret = os.environ.get(fallback_secret, "")

            model_path = _resolve_model_path(name)
            reg = MODEL_REGISTRY[name]

            print(f"[MODEL DETECT] env {name}: key={'YES' if key else 'NO'}, "
                  f"secret={'YES' if secret else 'NO'}, "
                  f"model={model_path} exists={model_path.exists()}", flush=True)

            if key and secret and model_path.exists():
                models.append(ModelConfig(
                    name=name,
                    model_path=model_path,
                    feature_version=reg["feature_version"],
                    alpaca_key=key,
                    alpaca_secret=secret,
                ))

    # Debug: list pkl files actually present, for diagnosing missing-model logs.
    model_dir = _BASE_DIR / "model"
    if model_dir.exists():
        try:
            result = subprocess.run(
                ["find", str(model_dir), "-name", "*.pkl"],
                capture_output=True, text=True, timeout=5,
            )
            print(f"[MODEL DETECT] .pkl files found: {result.stdout.strip()}", flush=True)
        except Exception:
            pass

    print(f"[MODEL DETECT] Active: {[m.name for m in models]}", flush=True)
    return models


# ═══════════════════════════════════════════════════════════════════════════
# Alpaca key sanity check (dashboard "Test key" button)
# ═══════════════════════════════════════════════════════════════════════════

def test_alpaca_key(key: str, secret: str,
                    base_url: str = "https://paper-api.alpaca.markets") -> dict:
    """Test an Alpaca API key pair. Returns account info or {ok: False, error: ...}."""
    import requests
    try:
        headers = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        }
        resp = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
        if resp.status_code == 200:
            acct = resp.json()
            return {
                "ok": True,
                "status": acct.get("status", "unknown"),
                "equity": float(acct.get("equity", 0)),
                "cash": float(acct.get("cash", 0)),
                "buying_power": float(acct.get("buying_power", 0)),
                "currency": acct.get("currency", "USD"),
                "account_number": acct.get("account_number", "?"),
            }
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
