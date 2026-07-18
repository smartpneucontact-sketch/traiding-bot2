# WF_REPORT — walk-forward process evaluation (Track B / WS3)

Spec: `results/v7/wf/process_spec.json`, sha256 `b75aa7e7fe34eec11ebcd5b528c5a5f4d75f131fda1bd5e7cb54c773688503a0` (frozen 2026-07-18T11:53:33; embedded in every wf_* ledger note). Pool: 21 base catalog names, 0 excluded by the truncation-equivalence audit. Metric: sharpe under next_open tc5 margin600 at 1x on all history to date. Step: annual, select at Dec-31 of Y-1, trade year Y. Warmup: 2016-04..2018-12 (short - disclosed). Leg 2026 is partial (data ends 2026-03-27).

## Honesty headline

**The selection process at 2x earns +3.40%/mo (geo) vs the hindsight champion's 3.68%/mo — a gap of -0.28pp/mo.** Same-window (2019+) champion geo is +4.44%/mo (gap -1.04pp/mo). The process number — re-selecting the top-3 annually on trailing data — is the honest prior for live performance; the champion number is what hindsight paid. Survivorship inflation (WS1) applies to BOTH arms and is not corrected here.

## Truncation-equivalence audit (causality gate)

All pool members were rebuilt on a panel truncated at 2020-12-31 and compared to the full-build weights sliced to the same cutoff (atol 1e-10, coverage >= 90%). Per-name results: `results/v7/wf/equivalence.json`.

All 21 pool members passed — no exclusions.

## Per-leg selections (top-3 by Sharpe at selection date)

| leg | cutoff | pick 1 | pick 2 | pick 3 |
|---|---|---|---|---|
| 2019 | 2018-12-31 | dual_momentum_vol (1.12) | acceleration_momentum (0.85) | low_vol_quality (0.83) |
| 2020 | 2019-12-31 | low_vol_quality (1.41) | dual_momentum_vol (1.30) | risk_parity_etf (1.29) |
| 2021 | 2020-12-31 | dual_momentum_vol (1.34) | adaptive_voltarget_momentum (1.15) | donchian_breakout (1.13) |
| 2022 | 2021-12-31 | dual_momentum_vol (1.25) | donchian_breakout (1.19) | acceleration_momentum (1.09) |
| 2023 | 2022-12-31 | dual_momentum_vol (1.07) | calendar_tom (1.02) | xs_momentum_top30 (0.98) |
| 2024 | 2023-12-31 | dual_momentum_vol (1.10) | xs_momentum_top30 (1.03) | xs_momentum_12_1 (0.97) |
| 2025 | 2024-12-31 | dual_momentum_vol (1.12) | xs_momentum_top30 (1.07) | xs_momentum_12_1 (1.02) |
| 2026 | 2025-12-31 | dual_momentum_vol (1.13) | xs_momentum_top30 (1.09) | xs_momentum_12_1 (1.02) |

Full scoring table (all candidates, all legs, selection-time Sharpe and Calmar): `results/v7/wf/selections.csv`.

## Process curve vs hindsight champion vs SPY (2019-01-01 onward)

| curve | geo %/mo | sharpe | maxDD | ann vol | DSR | NW-t vs champ | NW-t vs SPY |
|---|---|---|---|---|---|---|---|
| wf_process_2x | +3.40 | 1.03 | -57.7% | 52.6% | 0.800 | -2.10 | +3.02 |
| wf_process_1.5x | +2.91 | 1.06 | -46.4% | 40.1% | 0.823 | -2.88 | +2.99 |
| wf_process_1x | +2.23 | 1.13 | -33.0% | 26.7% | 0.865 | -3.20 | +2.62 |
| champion_2019on | +4.44 | 1.15 | -60.3% | 62.6% | — | — | — |
| champion_full | +3.68 | 1.06 | -60.3% | 56.5% | — | — | — |
| spy_2019on | +1.20 | 0.83 | -33.7% | 19.6% | — | — | — |

DSR = deflated Sharpe at the program-wide deduped trial count (n=687) with cross-trial SR variance measured from the ledgers (exp_catalog_v2.ledger_sr_variance_daily, reused). Newey-West t-stats (lags=3) are on paired monthly return differences, 2019+. SPY is an adjusted-close buy-and-hold proxy (no costs) — regime context only. The champion reference reproduces the ledgered catalog_v2 `combo_v2_2x` full-window row (cross-checked at 1e-6).

## Sensitivity (fragility table — dev legs 2019-2022 ONLY, never the headline)

| variant | k | metric | dev geo %/mo | dev sharpe | dev maxDD |
|---|---|---|---|---|---|
| primary (headline) | 3 | sharpe | +2.31 | 0.85 | -48.3% |
| k2_sharpe | 2 | sharpe | +2.59 | 0.85 | -54.7% |
| k4_sharpe | 4 | sharpe | +2.43 | 0.84 | -54.5% |
| k3_calmar | 3 | calmar | +1.96 | 0.74 | -50.8% |

Sensitivity runs are family `wf_sens`, dev window only; their selections are in `selections` column of the printed run log and were never used for the full-window process curve.

## Provenance

- engine: engine_v2 next_open, tc 5bp/side, margin 600bp/yr on long gross > 1x, leverage cap 2 (scoring at cap 1)
- ledger families: wf_score (scoring), wf_process (final curves), wf_sens (dev-only variants); every note carries spec sha `b75aa7e7fe34eec1…`
- artifacts: process_spec.json, equivalence.json, scores.csv, selections.csv, wf_results.csv, process_curve.parquet
