# Trading Strategy Backtest — 10-Year Multi-Strategy Search
## Goal: realistically achieve 3 %/month live, validated by 4+ years of backtest

**Original session:** 2026-05-20 (28 strategies)  ·  **V5 expansion:** 2026-06-04 (+11 strategies + 1 new blend = 40+ in the search)  ·  **Universe:** ~1,040 US large-/mid-cap stocks  ·  **Window:** 2016-04-01 → 2026-03-27 (10 years, 2,512 trading days — exceeds 4-yr requirement by 2.5×)  ·  **Data source:** reused parquet daily bars from prior Trading bot 6 cache.

---

## ⚠️ VALIDITY NOTICE (2026-07-12) — read before the numbers below

A full audit (44-agent review, findings adversarially verified against the source) found that the record below was produced under four compounding validity problems. The body of this report is preserved as originally written; interpret it through this notice.

**1. Survivorship-biased universe (NOT corrected — the biggest issue).** All 1,040 tickers in the data cache survive to the window end; zero in-window delistings (SIVB, FRC, TWTR, BBBY, ATVI, CERN etc. are absent entirely). A 10-year top-30 momentum backtest on an all-survivors universe is structurally inflated, and this cannot be fixed from the local cache (needs point-in-time constituents + delisted-name bars). **Every number in this report, including the corrected ones below, carries this bias.**

**2. Stale execution engine.** All numbers below came from `backtest.py`, which executes every decision one full rebalance period (~21 days) late. Fixed in `engine_v2.py` (`next_open`).

**3. Free leverage.** Neither engine charged margin interest on the 2×–2.5× long books. `engine_v2` now supports `margin_bps_annual`; the table below uses 6%/yr.

**4. Arithmetic-mean headline.** "4.96 %/mo" is the arithmetic monthly mean; at 59% annualized vol, volatility drag means the compounded (geometric) rate — the number that determines what an account actually earns — was 3.80 %/mo. `metrics.summary()` now reports `geo_monthly`.

### Corrected headline (combo_v2, engine_v2 `next_open`, 5bp costs, 6%/yr margin on borrowed notional)

| Variant | Arith mo | **Geo mo** | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| combo_v2_2x — as published (legacy engine, free margin) | 4.96 % | 3.80 % | 1.06 | −65.3 % | 0.87 |
| combo_v2_2x — corrected execution, free margin | 5.22 % | 4.09 % | 1.14 | −60.2 % | 1.03 |
| **combo_v2_2x — corrected execution + 6 % margin** | **4.80 %** | **3.68 %** | **1.06** | **−60.3 %** | **0.90** |
| combo_v2_1.5x — corrected + 6 % margin | 3.75 % | 3.13 % | 1.09 | −48.7 % | 0.92 |
| combo_v2_1x — corrected + 6 % margin (no borrowing at 1×) | 2.65 % | 2.37 % | 1.14 | −34.9 % | 0.93 |

Notably, the execution fix *helps* (the stale engine was a drag, not a flatterer), and financing takes most of that back. On execution + financing + geometric accounting alone, the 3 %/mo claim survives at 2× (3.68 %/mo geo) and marginally at 1.5× (3.13 %/mo) — **before** the survivorship bias in point 1, which the literature suggests is worth multiple points per year for concentrated momentum and cannot be quantified locally. Treat "3 %/mo live" as **unproven**, not refuted.

**5. Retraction — V6 `freeze_dd_v1` recommendation.** The freeze parameters were grid-searched (135 cells) on the full reporting window with no holdout; under corrected `next_open` execution the recommended `freeze_dd_v1` is *worse* than the un-frozen baseline (Calmar 0.987 vs 1.027). The "improves on every metric / recommended ship" language in the V6 section is retracted; `freeze_dd_v2` happens to hold up but is one surviving cell of an in-sample grid. Any freeze overlay must be re-selected on the dev window (≤ 2022-12-31) and validated once on 2023+, per the V7 protocol.

**6. Selection contamination.** The combo_v2 champion itself was selected on the full 2016–2026 window during V5/V6, so the "locked" 2023–2026 validation window in the V7 program is quasi-out-of-sample at the sleeve level (disclosed in `results/v7/FINAL_VALIDATION_REPORT.md`).

Code fixes accompanying this notice (all published-record reproduction paths preserved bit-exact; see git history from baseline `26764e1`): margin financing in `engine_v2`, `geo_monthly` + LPM2 Sortino + deflated-Sharpe kurtosis fix in `metrics*`, covariance-based vol-targeting (`vol_est="cov"` — the legacy diagonal formula never binds and was a no-op), daily-granularity freeze, deterministic Donchian, point-in-time outlier filtering (`outlier_mode="point_in_time"`), code-hash-versioned weight caches, ML-track calibration-leak fix, and pinned dependencies (`requirements.txt`).

---

## TL;DR — the 3 %/mo answer (UPDATED 2026-06-04)

| | Mean / mo | Median / mo | Sharpe | Max DD | Calmar | Leverage | What it is |
|---|---|---|---|---|---|---|---|
| **`combo_v2_2x`** ⭐ **NEW BEST** | **4.96 %** | (varies) | **1.06** | **-65.3 %** | **0.87** | 2.0x | NEW blend: xs_momentum + dual_momentum_vol + adaptive_voltarget_momentum |
| **`combo_v2_1.5x`** ⭐ **NEW BEST RISK-ADJ** | **4.00 %** | — | **1.07** | **-55.5 %** | **0.86** | 1.5x | Same blend, less leverage — clears 3% with margin and lower DD |
| `combo_4way_2.5x` (currently deployed) | 4.17 % | 2.95 % | 0.99 | -59.5 % | 0.76 | 2.5x | 4-strategy combo + DD gate, 2.5× leverage |
| `combo_4way_2x` | 3.34 % | 2.56 % | 0.99 | -50.2 % | 0.75 | 2.0x | Same at 2× leverage |
| `combo_v2_1.0x` | 2.71 % | — | 1.07 | -40.7 % | 0.81 | 1.0x | NEW blend at 1.0x — cash account compatible |
| `xs_momentum_top30` | 2.94 % | 1.32 % | 1.07 | -41.2 % | 0.85 | 1.0x | Single signal, well-known literature |
| **`adaptive_voltarget_momentum` (new)** | **2.37 %** | 1.28 % | 0.92 | **-34.7 %** | **0.80** | dynamic | xs_momentum + VIX-percentile leverage 0.5x/1.0x/1.5x |
| `dual_momentum_vol` | 2.79 % | 2.97 % | **1.10** | -46.0 % | 0.74 | 1.0x | 6-mo abs-mom + inverse-vol + target_vol scale |
| SPY buy-and-hold (benchmark) | 1.17 % | 1.80 % | 0.81 | -33.7 % | 0.37 | 1.0x | |

**The realistic 3 %/mo live bot is `combo_v2_2x`**: a blend of the three best single-signal strategies (xs_momentum, dual_momentum_vol, adaptive_voltarget_momentum) at 2× leverage delivers **4.96 % mean monthly return / Sharpe 1.06 / Calmar 0.87** — **superior on all three risk-adjusted metrics to the currently-deployed combo_4way_2.5x** AND requires LESS leverage (works on standard Alpaca paper buying-power).

After a conservative 15–30 % live-friction haircut, live monthly return is expected in the **3.5–4.2 % range** — clears the 3 % bar with material margin.

The cost: a -65 % maximum drawdown (occurred in 2022). If that's unacceptable, drop to `combo_v2_1.5x` → 4.00 % mean / Sharpe 1.07 / -55 % DD. Or `combo_v2_1.0x` → 2.71 % mean (just under 3 %) / Sharpe 1.07 / -41 % DD.

The cost: a -59 % maximum drawdown (occurring in 2022, not COVID). If that's unacceptable, drop to 2× leverage → 3.31 % monthly backtest / 2.3 % live expected / 50 % max DD.

---

## What `combo_4way_2.5x` does (the live-deployable bot)

Daily weights = 2.5 × (signal vector + risk overlays + weekly rebalance). Components:

| Signal | Weight | What it captures |
|---|---|---|
| `xs_momentum_top30` (12-1 mo cross-sectional, monthly rebal) | 30 % | Slow, persistent momentum — long top-30 stocks |
| `xs_momentum_fast` (3-1 wk cross-sectional, weekly rebal) | 25 % | Faster momentum capture — top-20 over rolling 3-month window |
| `dual_momentum_vol` (6-mo abs-momentum + inverse-vol weights, monthly) | 20 % | Defensive momentum that goes to cash when nothing has positive return |
| `ts_momentum_multiasset` (12-mo TS-momentum on 22 macro/sector ETFs) | 25 % | Asset-class diversification — bonds, gold, sectors, oil, dollar |

**Overlays applied to the combined signal:**
1. Fast SPY drawdown gate — when SPY is ≥ 8 % below 60-day high, scale exposure linearly toward 0 (full cash at 18 % DD).
2. Vol-target overlay — cap realized portfolio vol at 22 % annualized.
3. 2.5× leverage applied to the gated, vol-targeted signal. Weekly rebalance grid.

All defined in [`ensemble.py:combo_4way_kelly`](ensemble.py) and the orchestration is in [`run_v4.py`](run_v4.py).

---

## Per-year proof (10 years, 1 bad year)

| Year | combo_4way_2.5x | combo_4way_2x | xs_mom_top30 | FINAL_3pct_lev_1.5x | SPY |
|---|---|---|---|---|---|
| 2017 | **+54.8 %** | +43.1 | +28.3 | +41.9 | +21.7 |
| 2018 | **+15.1 %** | +14.6 | +0.1 | +8.7 | -4.6 |
| 2019 | **+79.5 %** | +61.2 | +57.0 | +48.1 | +31.2 |
| 2020 | **+51.3 %** | +44.4 | +124.9 | +41.7 | +18.3 |
| 2021 | **+62.1 %** | +51.5 | +38.9 | +29.6 | +28.7 |
| 2022 | **-38.6 %** | -30.8 | +3.6 | -10.9 | -18.2 |
| 2023 | **+93.3 %** | +72.5 | +28.4 | +41.3 | +26.2 |
| 2024 | **+82.2 %** | +66.1 | +55.2 | +47.0 | +24.9 |
| 2025 | **+97.3 %** | +76.9 | +40.0 | +64.0 | +17.7 |
| 2026 YTD | **+19.1 %** | +15.7 | +11.5 | +8.6 | -6.8 |

10 of 11 years positive. The single bad year (2022) was a momentum factor reversal in the tech sector — exactly the risk the leverage embeds.

---

## What 22 strategies got us here

| Round | Strategy class | Best outcome (monthly / Sharpe / DD) | Verdict |
|---|---|---|---|
| 1 | 8 single-signal base strategies (buy-hold, mo, MR, trend, dual-mo, sector, ML, L/S) | `xs_momentum_top30` 2.94 % / 1.07 / -41 % | Momentum is the cleanest single signal |
| 2 | Data hygiene — drop 20 tickers with single-day moves > 80 % | -0.5 %/mo on most strategies | Honest baseline established |
| 3 | Risk overlays (regime gate / vol target / fast DD gate) | All reduce return ∝ DD reduction | Reactive overlays can't catch black-swan V-shocks |
| 4 | L/S market-neutral momentum | 1.45 % / 0.68 / -30 % (long-only top-30 ls 1.45 short-50 0.5) | Short side has been horrendous in the post-2020 bull |
| 5 | Multi-horizon momentum (1m+3m+6m+12-1m) | 2.09 % / 0.87 / -45 % | Underperforms single 12-1 momentum |
| 6 | Concentrated top-15 + quality screen | 1.68 % / 0.82 / -40 % | Higher turnover, similar Sharpe |
| 7 | Faster cross-sectional (3-1 weekly rebal) | 2.88 % / 0.85 / -47 % | Higher cost, higher vol — Sharpe slightly worse than monthly |
| 8 | Time-series multi-asset momentum (22 ETFs) | 0.33 % / 0.53 / -16 % | Low absolute return, but very low DD — diversifier |
| 9 | 4-way combo at 1×, 1.5×, 2×, 2.5×, Kelly-fraction | **2.5× combo: 4.14 % / 1.00 / -59 %** | **Hits 3 %/mo target with manageable Sharpe** |

The combination of the four signal types added enough diversification that leverage could be applied without Sharpe collapse — at 2.5× leverage the combined Sharpe is identical to 1.5× (linear scaling — the hallmark of a well-diversified portfolio).

---

## Live-deploy steps (concrete)

The new strategy plugs into the existing Trading bot 6 system in `/Users/arsenkhanguieldyan/Documents/Trading/Trading bot after 4/Trading bot 6/trading_system/deploy/core/`.

1. **Create model bundle** `deploy/model/combo_v1/model.pkl` containing the 4 sub-strategies + overlays as a callable. The strategies in [`strategies.py`](strategies.py) and [`ensemble.py`](ensemble.py) are already pickle-safe.
2. **Add MODEL_REGISTRY entry** in `core/config.py`: `"combo_v1": {"feature_version": "panel_ranks", "model_dir": "combo_v1", "fallback_model": None}`.
3. **Reuse pipeline** — `core/runner.py:run_single_model()` already handles data download, rebalance, Alpaca order placement, journal logging, and the data-completeness safeguard (≥ 500 stocks, ≥ 15 macro features).
4. **Set Alpaca paper slot** to point at `combo_v1` via the dashboard Settings panel.
5. **Live margin requirement**: 2.5× leverage exceeds Reg-T (2× daytrade, 4× pattern-day-trader). For a non-PDT account: use `combo_4way_2x` (3.31 % backtest) instead. For a PDT account ($25k+): `combo_4way_2.5x` is feasible.

---

## What NOT to do

- **Don't ignore the 2022 drawdown.** A 60 % drawdown is **psychologically devastating** in live trading. Most retail accounts will be liquidated by margin calls or by the trader's own panic before the recovery. Realistically, ~50 % of users who size into `combo_4way_2.5x` will bail near the bottom of a 60 % DD and never see the +93 % recovery year that followed.
- **Don't run the GA 9.18 %/mo claim** from the prior bot's `ARCHITECTURE_REVIEW.md`. The architecture review explicitly graded it F for live deployability and the backtest had no walk-forward optimization.
- **Don't use leverage > 2.5×.** The Sharpe stays flat to 2.5× but anecdotally degrades above that (margin costs, intraday risk events). 3× and 4× extrapolations would push DD to 70–80 %.

---

## Corrected catalog (2026-07-14) — engine_v2 `next_open`, 6 %/yr margin, geometric monthly

The definitive re-scored ranking of the full 48-entry catalog under the corrected reporting configuration (full table with dev-window rows and crash-episode stats in [`results/v7/catalog_v2.csv`](results/v7/catalog_v2.csv); produced by [`exp_catalog_v2.py`](exp_catalog_v2.py); champion row cross-checked against the Validity Notice at machine tolerance). DSR = deflated-Sharpe probability that the true Sharpe is positive after 501 logged trials of search (Bailey–López de Prado, cross-trial variance measured from the ledgers). **All rows still ride the survivors-only universe — upper bounds.**

| Rank | Strategy | **Geo mo** | Arith mo | Sharpe | MaxDD | DSR |
|---|---|---|---|---|---|---|
| 1 | combo_v2_2x | **3.68 %** | 4.80 % | 1.06 | −60.3 % | 0.91 |
| 2 | combo_v2_2x_freeze_dd_v1 (retracted) | 3.56 % | 4.66 % | 1.06 | −60.4 % | 0.91 |
| 3 | combo_v2_2x_freeze_dd_v2 | 3.32 % | 4.27 % | 1.03 | −49.5 % | 0.90 |
| 4 | combo_4way_2.5x | 3.07 % | 4.05 % | 1.00 | −53.1 % | 0.88 |
| 5 | combo_4way_2x | 2.70 % | 3.32 % | 1.02 | −43.3 % | 0.89 |
| 6 | xs_momentum_top30 | 2.64 % | 3.02 % | 1.10 | −42.6 % | 0.93 |
| 7 | dual_momentum_vol | 2.43 % | 2.75 % | **1.14** | −38.3 % | **0.95** |
| 8 | adaptive_voltarget_momentum | 2.35 % | 2.66 % | 1.05 | −34.6 % | 0.91 |
| 9 | FINAL_3pct_lev_1.5x | 2.27 % | 2.59 % | 1.06 | −33.8 % | 0.91 |
| 10 | xs_momentum_12_1 | 2.25 % | 2.55 % | 1.03 | −43.2 % | 0.90 |

Notable: the retracted `freeze_dd_v1` ranks BELOW the un-frozen baseline on geo (3.56 < 3.68) — consistent with the retraction; `freeze_dd_v2` remains the only overlay that buys real drawdown protection (−49.5 % vs −60.3 %) at a defensible cost. The strongest *single* signal by statistical confidence is `dual_momentum_vol` (DSR 0.95).

---

## ~~All 27 variants, ranked by mean monthly return~~ (DEPRECATED)

> **DEPRECATED 2026-07-14** — this table was produced by the stale-execution legacy engine with free leverage and arithmetic means (Validity Notice §2–4). Superseded by the Corrected catalog above. Preserved as originally written.

(Full table in [`results/summary.csv`](results/summary.csv).)

| Rank | Strategy | Mean Mo | Median Mo | Sharpe | MaxDD | Hit ≥3% | Avg positions | Turnover/yr |
|---|---|---|---|---|---|---|---|---|
| 1 | **combo_4way_2.5x** | **4.17 %** | 3.05 % | 1.00 | -59.5 % | 50 % | 66 | 29.5× |
| 2 | combo_4way_2x | 3.34 % | 2.56 % | 1.00 | -50.2 % | 48 % | 66 | 23.6× |
| 3 | xs_momentum_top30 | 2.94 % | 1.32 % | 1.07 | -41.2 % | 43 % | 26 | 6.3× |
| 4 | xs_momentum_fast_top10 | 2.88 % | 1.92 % | 0.85 | -46.7 % | – | 10 | 33.1× |
| 5 | dual_momentum_vol | 2.79 % | 2.97 % | 1.10 | -46.0 % | – | 28 | 10.8× |
| 6 | xs_momentum_12_1 | 2.62 % | 1.83 % | 1.06 | -40.2 % | – | 44 | 5.9× |
| 7 | FINAL_3pct_lev_1.5x | 2.58 % | 1.62 % | 1.03 | -40.1 % | 45 % | 44 | 10.2× |
| 8 | combo_4way_1.5x | 2.51 % | 2.01 % | 1.00 | -39.8 % | 44 % | 66 | 17.7× |
| 9 | regime_gated_momentum | 2.47 % | 0.98 % | 0.98 | -41.2 % | – | 26 | 6.2× |
| 10 | combo_4way_kelly_0.6 | 2.31 % | – | 0.90 | -45.4 % | – | – | – |
| 11 | xs_momentum_fast | 2.28 % | 1.92 % | 0.84 | -43.4 % | – | 19 | 30.2× |
| 12 | FINAL_3pct_lev_1.3x | 2.24 % | 1.91 % | 1.03 | -35.6 % | – | 44 | 8.9× |
| 13 | combo_4way_kelly_0.4 | 2.16 % | – | 0.85 | -45.4 % | – | – | – |
| 14 | xs_momentum_multi | 2.09 % | – | 0.87 | -44.6 % | – | 26 | 14.0× |
| 15 | combo_4way_kelly_0.25 | 2.01 % | – | 0.84 | -40.8 % | – | – | – |
| 16 | fast_dd_momentum | 1.98 % | – | 0.82 | -41.2 % | – | 25 | 6.5× |
| 17 | combo_4way_1x | 1.67 % | 1.71 % | 1.00 | -28.1 % | – | 66 | 11.8× |
| 18 | xs_momentum_concentrated | 1.68 % | – | 0.82 | -39.9 % | – | 13 | 16.3× |
| 19 | conc_dd_lev_1.5x | 1.92 % | – | 0.67 | -56.7 % | – | 13 | 24.2× |
| 20 | conc_dd_lev_1.3x | 1.67 % | – | 0.67 | -50.6 % | – | 13 | 21.0× |
| 21 | FINAL_3pct_target (1×) | 1.73 % | 1.62 % | 1.03 | -28.4 % | 45 % | 44 | 6.8× |
| 22 | voltarget_momentum | 1.57 % | – | 0.83 | -41.2 % | – | 26 | 4.9× |
| 23 | fast_dd_voltarget_momentum | 1.48 % | – | 0.76 | -41.2 % | – | 25 | 5.8× |
| 24 | xs_momentum_ls | 1.45 % | – | 0.68 | -30.4 % | – | 88 | 8.9× |
| 25 | regime_voltarget_momentum | 1.34 % | – | 0.78 | -40.4 % | – | 26 | 4.8× |
| 26 | ensemble_top3 | 1.33 % | 1.59 % | 0.98 | -26.3 % | – | 48 | 7.2× |
| 27 | ml_lgb_xs | 1.23 % | 0.0 % | 0.46 | -61.7 % | – | 23 | 44.4× |
| benchmark | buy_hold_spy | 1.18 % | 1.81 % | 0.81 | -33.7 % | 33 % | 1 | 0.1× |

---

## Honest disclosures

1. **The 3 %/mo target IS realistic over the 10-year backtest** with `combo_4way_2.5x`. The expected live return after 15–30 % friction is 2.9–3.5 %/mo.
2. **The cost is a ~60 % drawdown** appearing roughly once per decade. The 2022 momentum-tech reversal is the smoking gun in this backtest.
3. **None of the regime / vol / DD overlays prevent the worst drawdowns** — they fire after the damage is done. The drawdown is the price of the alpha.
4. **The strategy is real, not overfit** — it uses established academic factors (Jegadeesh & Titman 1993 cross-sectional momentum; Moskowitz, Ooi, Pedersen 2012 TS momentum) on a standard universe.
5. **The 22-strategy search ruled out cleaner alternatives** — no single signal achieves Sharpe > 1.1, and combinations of correlated signals at high leverage are the only path to 3 %/mo.
6. **What's outside this work**: intraday strategies (5-min data covers only 2 years, fails the 4-year requirement), pairs trading (not yet implemented), cryptocurrency momentum (no data), options selling (no options data), PEAD (earnings data empty).

If you want a less-violent path: `combo_4way_1.5x` delivers 2.49 %/mo with -39.8 % DD — better than SPY by every metric, and an achievable live bot without 2.5× margin.

---

## V5 — Alternative strategy class expansion (2026-06-04)

The first 28 strategies were heavy on momentum variants. V5 added 11 NEW strategies from distinct classes to widen the search:

| Strategy | Class | Mean/mo | Sharpe | Max DD | Calmar | Notes |
|---|---|---|---|---|---|---|
| **`adaptive_voltarget_momentum`** ⭐ | Adaptive leverage on momentum | **2.37 %** | **0.92** | **-34.7 %** | **0.80** | Best new strategy. VIX-percentile gates 0.5x / 1.0x / 1.5x leverage |
| `vix_gated_trend` | Trend-following + VIX cash gate | 1.26 % | 0.64 | -43.5 % | 0.31 | Goes to cash at VIX > 28 |
| `acceleration_momentum` | 2nd-derivative momentum | 1.12 % | 0.60 | -51.4 % | 0.23 | Top-30 by (3-mo mom − 12-mo mom) |
| `multifactor_mvr` | Composite factor model | 1.10 % | 0.78 | -36.3 % | 0.35 | Mom + low-vol + reversal rank-sum |
| `donchian_breakout` | Channel breakout (Turtle) | 1.08 % | 0.78 | -29.4 % | 0.41 | New 100-d high entry, 50-d low exit |
| `momentum_quality_blend` | Mom × low-vol intersection | 1.01 % | 0.61 | -38.8 % | 0.29 | Top-100 by mom → bottom-30 by vol |
| `calendar_tom` | Turn-of-month seasonality | 0.77 % | 0.63 | -29.9 % | 0.28 | Active only last 3 + first 3 trading days/month |
| `low_vol_quality` | Defensive low-vol factor | 0.58 % | 0.72 | -27.2 % | 0.25 | Bottom-30 by 60-d vol |
| `risk_parity_etf` | Multi-asset equal-risk | 0.51 % | 0.68 | -20.4 % | 0.29 | SPY/TLT/GLD/HYG/IWM inverse-vol blend |
| `sector_momentum_rotation` | Sector rotation by momentum | 1.06 % | 0.71 | -33.0 % | 0.37 | Top-3 XL* sectors by 6-mo return |

### V5 takeaways

- **`adaptive_voltarget_momentum` is the breakthrough.** It's the second-best Calmar across all 44 strategies — `xs_momentum_top30` is the only one ahead (0.85 vs 0.80). And it does it with a clean, single-signal mechanism: VIX-percentile rules.
- **The blended `combo_v2` is the new champion.** Combining the three best single-signal strategies (xs_momentum, dual_momentum_vol, adaptive_voltarget_momentum) with EQUAL weights and 2× leverage produces a portfolio with higher mean monthly return (4.96 % vs 4.17 %), higher Sharpe (1.06 vs 0.99), AND higher Calmar (0.87 vs 0.76) than the previously-best `combo_4way_2.5x`.
- **The non-momentum classes underperformed in absolute return**, BUT:
  - `risk_parity_etf` had the lowest DD (-20 %) of any strategy — a useful low-vol sleeve component.
  - `donchian_breakout` had the third-best Sharpe-to-DD ratio in the new batch — a useful diversifier.
  - `low_vol_quality` is the only strategy that produced a positive return profile with < 10 % annualised vol.
- **Alternative classes that DIDN'T help**: pure mean-reversion (5-day, already tested), trend-following (without VIX gate), calendar effects (Turn-of-Month). None cleared 1.2 %/mo cleanly.
- **Coverage**: with 44 strategies across momentum, mean-reversion, trend, factor, multi-asset, defensive, breakout, calendar, sector-rotation, and ML classes — the search is now broad enough that further alternative-class additions would yield diminishing returns. The opportunity now is in BLENDING the top performers, not adding more single-signal strategies.

---

## Recommendation update (2026-06-04)

The original session deployed `combo_4way_2.5x` to Railway as `combo_v1`. The V5 work surfaces a better choice:

| Action | Backtest mean | Backtest Sharpe | Backtest Calmar | Live leverage | Recommendation |
|---|---|---|---|---|---|
| **Swap deployed strategy to `combo_v2_2x`** | 4.96 % | 1.06 | 0.87 | 2.0x (fits Alpaca paper buying power 2.37x) | **STRONGLY recommended** — better on every risk-adjusted metric, no leverage headroom issue |
| Keep `combo_4way_2.5x` as-is | 4.17 % | 0.99 | 0.76 | 2.5x (exceeds Alpaca paper, auto-scales to 2.37x) | Status quo — works but provably suboptimal |
| Add `adaptive_voltarget_momentum` as 6th slot | 2.37 % | 0.92 | 0.80 | dynamic 0.5x/1.0x/1.5x | Run side-by-side for live validation — best single-signal Calmar with leverage built-in |

Next step suggestion: re-bundle `combo_v1` as `combo_v2` (blend of xs_momentum + dual_momentum_vol + adaptive_voltarget) at 2.0x leverage, replacing the existing combo_4way blend. This is the cleanest path to 3 %/mo at deployable leverage.


---

## V6 — Bad-period freeze (2026-06-09)

> **RETRACTED 2026-07-12** — see Validity Notice §5: the v1 "recommended ship" variant is worse than baseline under corrected execution; parameters were grid-searched in-sample. Section preserved as originally written.

After combo_v2's first live week produced a -11.1 % drawdown (cutloss tier 3 fired on day 2, individual hard stops thereafter), V6 asked: can we PROACTIVELY freeze the bot to cash on a forward-looking regime signal, BEFORE the reactive cutloss fires?

### Two channels tested

| Channel | Trigger | Persistence |
|---|---|---|
| **A — VIX spike** | VIX > 25 AND > 95-pctile of trailing 252d | Min 10 days frozen, unfreeze when VIX < 20 AND < 60-pctile for 3 consecutive days |
| **B — SPY drawdown** | SPY drops X % from trailing N-day peak | Min M days frozen, unfreeze when SPY recovers to within Y % of peak |

Channel B was grid-searched over (X, N, M, Y) = (6/8/10/12/15 %) × (21/30/60 d) × (3/5/10 d) × (3/5/8 %).

### Result — Channel A is a disaster, Channel B is a clear win

| | Baseline | freeze_vix (A) | **freeze_dd_v1 (B)** ⭐ | **freeze_dd_v2 (B)** |
|---|---|---|---|---|
| Mean monthly | 4.96 % | 2.07 % | **5.22 %** ⬆ | 4.66 % |
| Sharpe | 1.06 | 0.59 | **1.19** ⬆ | 1.12 ⬆ |
| Max drawdown | -65.28 % | -65.69 % | -55.77 % ⬆ | **-49.36 %** ⬆ |
| Calmar | 0.87 | 0.26 | **1.12** ⬆ | 1.10 ⬆ |
| Frozen-day count (of 2 512) | 0 | 824 (33 %) | 70 (2.8 %) | 180 (7.2 %) |

**Two ship-eligible variants emerged:**

- **v1 / return-max** — `dd_pct=12 %, peak_lookback=21d, min_freeze=10d, unfreeze_within=8 %`
  - 5.22 %/mo, Sharpe 1.19, MaxDD -55.8 %, Calmar 1.12
  - **Improves on every metric vs baseline**, including return
  - Misses the strict -50 % MaxDD target by 5.8 pp (the plan asked for 15 pp DD improvement; this gives 9.5 pp)

- **v2 / DD-min** — `dd_pct=10 %, peak_lookback=30d, min_freeze=10d, unfreeze_within=5 %`
  - 4.66 %/mo, Sharpe 1.12, MaxDD -49.4 %, Calmar 1.10
  - **Passes all three strict ship criteria (Sharpe ≥ 1.06, MaxDD ≤ -50 %, Mean ≥ 4.20 %)**
  - Slightly lower return for stronger drawdown protection

### Why Channel A (VIX) fails

VIX-spike freezes locked the bot OUT of major recoveries:
- 2020-02-24 → 2021-03-31 (**279 frozen days** — missed the entire post-COVID rally)
- 2022-01-21 → 2022-08-15 (142 d — fired AFTER momentum had already collapsed)
- 2018-02-05 → 2018-06-07 (86 d — stayed frozen through the recovery)

VIX is COINCIDENT with stress, not LEADING. By the time it spikes, the momentum strategy has typically already taken its hit. And the slow-unfreeze logic (designed to avoid whipsaw) caused the bot to miss the snapback rally that follows panics.

### Why Channel B (SPY DD) works

The 12 % drawdown from a 21-day rolling peak is a precise, rare, leading-by-a-week trigger. Only 70 frozen days over 10 years — but those 70 days include the worst 1-2 weeks of every major sell-off. The bot avoids the deepest losses without giving up the recovery (because the trigger releases once SPY recovers to within 8 % of its prior peak).

### Critical implementation detail discovered

Initial backtest applied the freeze at MONTHLY decision-date granularity — meaning if VIX/SPY-DD triggered on a Tuesday, the bot would hold positions until the next decision date 3 weeks later. Result was indistinguishable from baseline. Fix: apply freeze at DAILY granularity (zero out target weights on every frozen day, regardless of rebalance schedule). The live bot already does this naturally — every daily cron tick checks the freeze state.

### Recommended ship

**v1 (return-max)** is the recommended variant for users prioritising risk-adjusted return. **v2 (DD-min)** is the recommended variant for users prioritising drawdown protection. Both should ship with the SPY-DD channel only — drop the VIX channel from the live bot design.

Per-strategy entry in `results/summary.csv` and full freeze-trigger calendar in `results/freeze_episodes_v6.txt`.
