# RESCORE — full catalog re-scored under corrected execution (engine_v2)

Generated 2026-06-11 21:54. Family ledger: `results/v7/trials/rescore.csv`; machine-readable grid: `results/v7/rescore.csv` (192 rows = 48 strategies x (3 full-window exec models + 1 dev-window next_open)). Script: `exp_rescore.py`.

Published configs reproduced exactly (tc/leverage_cap per source run script; tc=5bp everywhere except xs_momentum_ls* at 7bp). `legacy_period` reproduces backtest.py bit-for-bit; `next_open` is the corrected live model (signal at close t -> market order at next open, costs on |dw| at the open). All selection-relevant numbers below are DEV window; full-window numbers are a record correction only.

## 0. Reproduction check (legacy_period vs published summary.csv)

- 45/46 comparable strategies reproduce published mean_monthly to < 1e-9 (machine precision). The catalog frames and legacy engine are faithful.
- `donchian_breakout` differs by 2.05e-03 (0.20 pp/mo): the strategy itself is NON-DETERMINISTIC — strategies.py:690 truncates the held set via `set(list(held)[:2*n_long])`, whose order depends on per-process string-hash randomization. The diff is run-to-run strategy variance, not an engine discrepancy. (Anomaly logged; fix would be `sorted(held)`.)
- `xs_momentum_ls`: NOT comparable — published with allow_shorts=True + 80bp borrow; run_trial does not expose allow_shorts so shorts are clipped here. Legacy +2.61%/mo vs published +1.45%/mo. Rows flagged `shorts_clipped`; excluded from tax stats.
- `xs_momentum_ls_neutral`: NOT comparable — published with allow_shorts=True + 80bp borrow; run_trial does not expose allow_shorts so shorts are clipped here. Legacy +2.61%/mo vs published +0.28%/mo. Rows flagged `shorts_clipped`; excluded from tax stats.
- Verified: for strategies whose published frames were DAILY (combo_v2 freeze variants), legacy_period == next_close exactly (diff 0.0), as expected — a daily frame only ever carried 1 day of lag in the old engine.

## 1. Top-10 by FULL-window next_open Calmar (record correction — NOT for selection)

| # | strategy | mm/mo | Sharpe | MaxDD | Calmar | turn/yr | gross | pub Calmar | pub mm/mo |
|---|----------|-------|--------|-------|--------|---------|-------|------------|-----------|
| 1 | combo_v2_2x_freeze_dd_v2 | +4.66% | 1.120 | -49.4% | 1.110 | 16.7 | 1.59x | 1.105 | +4.66% |
| 2 | combo_4way_2.5x | +4.62% | 1.131 | -50.0% | 1.070 | 29.6 | 2.02x | 0.757 | +4.17% |
| 3 | combo_4way_2x | +3.69% | 1.131 | -41.3% | 1.055 | 23.6 | 1.61x | 0.747 | +3.34% |
| 4 | combo_v2_2x | +5.22% | 1.140 | -60.2% | 1.027 | 15.1 | 1.72x | 0.866 | +4.96% |
| 5 | combo_4way_1.5x | +2.76% | 1.131 | -32.5% | 1.016 | 17.7 | 1.21x | 0.727 | +2.51% |
| 6 | combo_v2_2x_freeze_dd_v1 | +5.07% | 1.151 | -60.2% | 0.987 | 16.0 | 1.67x | 1.123 | +5.22% |
| 7 | adaptive_voltarget_momentum | +2.75% | 1.085 | -34.4% | 0.973 | 9.6 | 0.92x | 0.801 | +2.37% |
| 8 | combo_4way_1x | +1.84% | 1.131 | -22.8% | 0.971 | 11.8 | 0.81x | 0.700 | +1.67% |
| 9 | FINAL_3pct_lev_1.5x | +2.70% | 1.109 | -33.5% | 0.971 | 10.2 | 1.08x | 0.758 | +2.58% |
| 10 | FINAL_3pct_lev_1.3x | +2.35% | 1.109 | -29.7% | 0.955 | 8.9 | 0.93x | 0.746 | +2.24% |

Headlines: the corrected champion record is **combo_v2_2x +5.22%/mo, Sharpe 1.140, Calmar 1.027** (published 4.96%/mo / 0.866 was understated by the stale-signal bug). `combo_v2_2x_freeze_dd_v2` is the best full-window Calmar (1.110, MaxDD -49.4%). The combo_4way family jumps the most (e.g. combo_4way_2x Calmar 0.747 -> 1.055) but its published caps (up to 3.0x) exceed the V7 2.0x discipline and its DEV mean-monthly is far below the bar.

## 2. Measured stale-signal tax (legacy_period -> next_open, full window)

Across 46 comparable strategies, the old engine's period-stale execution cost on average:

- mean_monthly: **+0.071 pp/mo mean** (+0.040 median); legacy -> next_close alone is +0.070 pp/mo mean, so nearly all of the correction is the lag fix, not the open-vs-close convention
- Sharpe: +0.048 mean (+0.064 median)
- Calmar: +0.096 mean (+0.131 median)

The tax is strongly heterogeneous: fast-rebalance and vol/dd-gated strategies paid the most (their signals decayed fastest), while a few slow gate-based strategies accidentally BENEFited from staleness (their stale gate happened to hold through whipsaws).

Largest gainers from the fix (pp/mo): `mean_reversion_5d` +0.99; `combo_4way_2.5x` +0.45; `adaptive_voltarget_momentum` +0.39; `acceleration_momentum` +0.36; `combo_4way_2x` +0.35; `fast_dd_voltarget_momentum` +0.34

Largest losers (pp/mo): `combo_v2_2x_freeze_dd_v1` -0.15; `regime_gated_momentum` -0.19; `multi_dd_lev_1.3x` -0.27; `trend_following` -0.38; `vix_gated_trend` -0.54

## 3. Biggest rank movers (full-window Calmar rank among 48; + = improved)

| strategy | legacy rank | next_open rank | change | next_open Calmar |
|----------|-------------|----------------|--------|------------------|
| xs_momentum_ls_neutral | 7 | 21 | -14 | 0.704 |
| xs_momentum_ls | 7 | 21 | -14 | 0.704 |
| xs_momentum_12_1 | 6 | 19 | -13 | 0.708 |
| fast_dd_voltarget_momentum | 30 | 18 | +12 | 0.789 |
| acceleration_momentum | 47 | 35 | +12 | 0.386 |
| mean_reversion_5d | 48 | 37 | +11 | 0.328 |
| combo_4way_1.5x | 15 | 5 | +10 | 1.016 |
| xs_momentum_top30 | 4 | 14 | -10 | 0.862 |
| combo_4way_1x | 17 | 8 | +9 | 0.971 |
| donchian_breakout | 20 | 29 | -9 | 0.527 |

Pattern: the xs_momentum family (monthly rebalance, no gate) drops the most in RELATIVE rank — it was the least hurt by staleness, so fixing the bug lifted everything else past it. combo_4way and the gated fast strategies climb.

## 4. Candidate sleeves rivaling the champion (DEV window, next_open) + G1 verdict

Champion combo_v2_2x DEV (corrected baseline): +4.55%/mo, Sharpe 1.044, MaxDD -60.2%, Calmar 0.858. G1 bar: Calmar >= 0.944, mm >= 4.0%/mo, MaxDD > -50%.

| strategy | dev mm/mo | dev Sharpe | dev MaxDD | dev Calmar | G1 metrics |
|----------|-----------|------------|-----------|------------|------------|
| combo_v2_2x_freeze_dd_v1 | +4.66% | 1.141 | -52.1% | 1.054 | fails dd |
| calendar_tom | +1.18% | 1.023 | -14.3% | 0.970 | fails mm |
| combo_v2_2x_freeze_dd_v2 | +3.75% | 0.995 | -48.3% | 0.878 | fails calmar,mm |
| combo_v2_2x | +4.55% | 1.044 | -60.2% | 0.858 | fails calmar,dd |
| FINAL_3pct_lev_1.5x | +2.24% | 0.975 | -33.5% | 0.784 | fails calmar,mm |
| combo_4way_2.5x | +3.53% | 0.943 | -49.1% | 0.778 | fails calmar,mm |
| FINAL_3pct_lev_1.3x | +1.95% | 0.975 | -29.7% | 0.775 | fails calmar,mm |
| dual_momentum_vol | +2.50% | 1.066 | -38.3% | 0.772 | fails calmar,mm |
| combo_4way_2x | +2.84% | 0.943 | -41.3% | 0.769 | fails calmar,mm |
| FINAL_3pct_target | +1.51% | 0.975 | -23.5% | 0.761 | fails calmar,mm |
| combo_4way_1.5x | +2.14% | 0.943 | -32.5% | 0.755 | fails calmar,mm |
| adaptive_voltarget_momentum | +2.13% | 0.927 | -34.4% | 0.738 | fails calmar,mm |

**G1 verdict: NO strict pass.** `combo_v2_2x_freeze_dd_v1` is the standout — dev +4.66%/mo, Sharpe 1.141, Calmar 1.054 (= 1.23x champion) — but its dev MaxDD of -52.1% misses the -50% bar by 2.1pp. `combo_v2_2x_freeze_dd_v2` passes the DD bar (-48.3%) and edges the champion on Calmar (0.878) but misses the 4.0%/mo bar (+3.75%). Both are flagged as CANDIDATE SLEEVES for the assembly phase: the freeze overlay is the only catalog mechanism that cut the champion's dev MaxDD materially while keeping mm above 3.7%.

Neighborhood stability (catalog has no grid; the two freeze variants are each other's nearest parameter neighbors): dd_v2/dd_v1 dev Calmar ratio = 83.3% — just under the 85% G1 stability bar, so the freeze parameters MUST be stability-tested on a proper grid in their own family before assembly.

Also notable (not rivals): `calendar_tom` dev Calmar 0.970 at only 0.27x gross and +1.18%/mo — a potential low-correlation ballast sleeve; the corrected combo_4way_2x (dev +2.84%/mo, Calmar 0.769) is a diversified alternative chassis but is far from the mm bar at <= 2x gross.

## 5. Full grid (from the family ledger) — mean_monthly / Calmar by exec model

| strategy | mm legacy | mm next_close | mm next_open | Calmar legacy | Calmar next_open | MaxDD next_open | dev mm | dev Calmar |
|----------|-----------|---------------|--------------|---------------|------------------|-----------------|--------|------------|
| combo_v2_2x | +4.96% | +5.21% | +5.22% | 0.866 | 1.027 | -60.2% | +4.55% | 0.858 |
| combo_v2_2x_freeze_dd_v1 | +5.22% | +5.22% | +5.07% | 1.123 | 0.987 | -60.2% | +4.66% | 1.054 |
| combo_v2_2x_freeze_dd_v2 | +4.66% | +4.66% | +4.66% | 1.105 | 1.110 | -49.4% | +3.75% | 0.878 |
| combo_4way_2.5x | +4.17% | +4.59% | +4.62% | 0.757 | 1.070 | -50.0% | +3.53% | 0.778 |
| combo_4way_2x | +3.34% | +3.67% | +3.69% | 0.747 | 1.055 | -41.3% | +2.84% | 0.769 |
| xs_momentum_top30 | +2.94% | +3.05% | +3.02% | 0.854 | 0.862 | -42.6% | +2.62% | 0.720 |
| xs_momentum_fast_top10 | +2.88% | +2.85% | +2.90% | 0.627 | 0.604 | -46.3% | +1.60% | 0.264 |
| combo_4way_1.5x | +2.51% | +2.74% | +2.76% | 0.727 | 1.016 | -32.5% | +2.14% | 0.755 |
| dual_momentum_vol | +2.79% | +2.72% | +2.75% | 0.737 | 0.870 | -38.3% | +2.50% | 0.772 |
| adaptive_voltarget_momentum | +2.37% | +2.71% | +2.75% | 0.801 | 0.973 | -34.4% | +2.13% | 0.738 |
| FINAL_3pct_lev_1.5x | +2.58% | +2.71% | +2.70% | 0.758 | 0.971 | -33.5% | +2.24% | 0.784 |
| xs_momentum_fast | +2.28% | +2.51% | +2.59% | 0.565 | 0.707 | -40.5% | +1.80% | 0.450 |
| xs_momentum_12_1 | +2.62% | +2.57% | +2.55% | 0.786 | 0.708 | -43.2% | +2.28% | 0.612 |
| xs_momentum_ls | +2.61% | +2.56% | +2.54% | 0.782 | 0.704 | -43.2% | +2.27% | 0.609 |
| xs_momentum_ls_neutral | +2.61% | +2.56% | +2.54% | 0.782 | 0.704 | -43.2% | +2.27% | 0.609 |
| combo_4way_kelly_0.6 | +2.31% | +2.42% | +2.44% | 0.569 | 0.868 | -32.2% | +1.79% | 0.606 |
| combo_4way_kelly_0.4 | +2.16% | +2.40% | +2.39% | 0.522 | 0.791 | -34.6% | +1.77% | 0.606 |
| FINAL_3pct_lev_1.3x | +2.24% | +2.35% | +2.35% | 0.746 | 0.955 | -29.7% | +1.95% | 0.775 |
| regime_gated_momentum | +2.47% | +2.30% | +2.28% | 0.707 | 0.626 | -42.6% | +1.85% | 0.502 |
| combo_4way_kelly_0.25 | +2.01% | +2.18% | +2.14% | 0.538 | 0.666 | -36.2% | +1.57% | 0.589 |
| multi_dd_lev_1.3x | +2.41% | +2.05% | +2.14% | 0.473 | 0.507 | -43.8% | +1.59% | 0.355 |
| fast_dd_momentum | +1.98% | +2.12% | +2.12% | 0.548 | 0.793 | -31.0% | +1.67% | 0.634 |
| combo_v2_2x_freeze_vix_dd_v1 | +2.07% | +2.07% | +2.05% | 0.259 | 0.263 | -63.8% | +1.24% | 0.179 |
| combo_v2_2x_freeze_vix | +2.07% | +2.07% | +2.05% | 0.259 | 0.263 | -63.8% | +1.24% | 0.179 |
| xs_momentum_multi | +2.09% | +1.99% | +2.05% | 0.532 | 0.676 | -34.7% | +1.69% | 0.536 |
| mean_reversion_5d | +1.01% | +1.94% | +1.99% | 0.136 | 0.328 | -65.0% | +1.87% | 0.283 |
| conc_dd_lev_1.5x | +1.92% | +2.03% | +1.97% | 0.337 | 0.500 | -42.5% | +1.61% | 0.392 |
| combo_4way_1x | +1.67% | +1.83% | +1.84% | 0.700 | 0.971 | -22.8% | +1.43% | 0.735 |
| fast_dd_voltarget_momentum | +1.48% | +1.83% | +1.82% | 0.412 | 0.789 | -27.3% | +1.41% | 0.594 |
| FINAL_3pct_target | +1.73% | +1.81% | +1.81% | 0.727 | 0.931 | -23.5% | +1.51% | 0.761 |
| voltarget_momentum | +1.57% | +1.80% | +1.77% | 0.441 | 0.616 | -34.3% | +1.42% | 0.479 |
| conc_dd_lev_1.3x | +1.67% | +1.77% | +1.72% | 0.341 | 0.501 | -37.9% | +1.41% | 0.396 |
| xs_momentum_concentrated | +1.68% | +1.65% | +1.60% | 0.484 | 0.466 | -38.5% | +1.43% | 0.397 |
| acceleration_momentum | +1.12% | +1.43% | +1.47% | 0.227 | 0.386 | -43.1% | +1.64% | 0.440 |
| regime_voltarget_momentum | +1.34% | +1.46% | +1.43% | 0.384 | 0.528 | -31.9% | +1.15% | 0.414 |
| ensemble_top3 | +1.33% | +1.36% | +1.36% | 0.603 | 0.825 | -19.6% | +1.13% | 0.671 |
| buy_hold_spy | +1.18% | +1.18% | +1.18% | 0.406 | 0.407 | -33.7% | +1.04% | 0.341 |
| donchian_breakout | +1.28% | +1.19% | +1.18% | 0.589 | 0.527 | -25.7% | +1.19% | 0.528 |
| momentum_quality_blend | +1.01% | +1.16% | +1.15% | 0.288 | 0.327 | -39.8% | +0.91% | 0.248 |
| multifactor_mvr | +1.10% | +0.98% | +1.00% | 0.349 | 0.270 | -42.1% | +0.88% | 0.227 |
| sector_momentum_rotation | +1.06% | +0.96% | +0.99% | 0.370 | 0.351 | -31.4% | +1.20% | 0.430 |
| sector_rotation | +1.02% | +0.88% | +0.90% | 0.368 | 0.297 | -33.6% | +0.90% | 0.293 |
| trend_following | +1.21% | +0.83% | +0.83% | 0.286 | 0.123 | -57.6% | +1.28% | 0.359 |
| calendar_tom | +0.77% | +0.77% | +0.75% | 0.281 | 0.273 | -29.3% | +1.18% | 0.970 |
| vix_gated_trend | +1.26% | +0.69% | +0.72% | 0.312 | 0.148 | -46.4% | +0.68% | 0.167 |
| low_vol_quality | +0.58% | +0.58% | +0.59% | 0.247 | 0.294 | -23.4% | +0.60% | 0.296 |
| risk_parity_etf | +0.51% | +0.54% | +0.54% | 0.290 | 0.312 | -20.2% | +0.33% | 0.178 |
| ts_momentum_multiasset | +0.33% | +0.32% | +0.32% | 0.235 | 0.308 | -12.1% | +0.22% | 0.203 |

## 6. Anomalies and caveats

- `ml_lgb_xs` SKIPPED per assignment (slow walk-forward LightGBM, documented failure; published: +1.23%/mo, Sharpe 0.46, MaxDD -61.7%, Calmar 0.17).
- `donchian_breakout` is non-deterministic across processes (set-order truncation, strategies.py:690) — its numbers carry ~0.2 pp/mo run-to-run noise.
- Published combo_v2 FREEZE-vs-baseline comparison was apples-to-oranges: the freeze variants were DAILY frames (1-day lag in the old engine) while the baseline was a sparse monthly frame (21-day stale). Corrected next_open is the first like-for-like comparison; the freeze advantage shrinks (dd_v1 full Calmar 1.123 published -> 0.987 corrected) because the corrected baseline improves and next_open execution slightly weakens the freeze (covid DD -26.8% -> -35.4%: the freeze liquidates at the next OPEN, so it still eats the overnight gap).
- `combo_v2_2x_freeze_dd_v1` full-window next_open MaxDD (-60.2%) is deeper than its dev MaxDD (-52.1%): the 2021-11 -> 2022/23 drawdown continues past DEV_END in the full window. The freeze did NOT bind through most of that slow grind.
- `combo_4way_2.5x` / `combo_4way_kelly_*` were published at leverage_cap=3.0 (above the V7 2.0x discipline); re-scored at published caps, record-only.
- Superseded names in summary.csv not re-scored: `combo_v2_2x_baseline` (identical to `combo_v2_2x`), `combo_v2_2x_freeze_dd`, `combo_v2_2x_freeze_both` (older freeze parameterizations replaced by dd_v1/dd_v2/vix_dd_v1 in run_v6.py).
- No grid cells errored; no cells added. 192/192 pre-registered runs completed.
