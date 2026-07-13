# E3 — Residual (idiosyncratic) momentum sleeve

Generated: 2026-06-11T11:59:02  
Family: `E3_residual` — all selection on DEV window (2016-04..2022-12), next_open, 5 bp, leverage cap 2.0x.  
Baseline (combo_v2_2x dev): mean_monthly 4.555%/mo, Sharpe 1.044, MaxDD -60.2%, Calmar 0.858.  
G1: Calmar ≥ 0.944 AND mean_monthly ≥ 4.0% AND MaxDD > -50% AND median neighbor Calmar ≥ 85% of winner.

## Sleeve definition

`strategies_v2.residual_momentum(px, macro, sector_map, n_long=30, beta_mode, scale_mode)` — 21d cadence, warm-up 273d (same grid as xs_momentum), equal-weight 1/30. Signal = sum of daily residuals (alpha retained) over trailing 273d window excluding most recent 21d; betas OLS vs SPY (market) or [SPY, sector XL*] (market_sector; missing sector/ETF → market-only fallback). sharpe variant divides by formation-window residual std (ddof=0). Eligibility: finite close at window start/end + ≥95% finite returns (NaN→0 in regression).

## Full grid (16 pre-registered cells)

| name | mm %/mo | Sharpe | MaxDD % | Calmar | turn/yr | gross | 2018Q4 ret/dd % | covid ret/dd % | 2022 ret/dd % |
|---|---|---|---|---|---|---|---|---|---|
| e3_resid_ms_raw_replxs | 4.811 | 1.125 | -58.3 | 0.962 | 13.3 | 1.62 | -41.5 / -49.0 | -45.0 / -58.3 | -10.7 / -41.8 |
| e3_resid_m_raw_replxs | 4.605 | 1.067 | -57.9 | 0.909 | 13.1 | 1.62 | -41.3 / -48.8 | -45.1 / -57.9 | 1.1 / -45.2 |
| e3_resid_ms_raw_add25 | 4.889 | 1.091 | -61.6 | 0.908 | 12.8 | 1.63 | -42.7 / -50.7 | -48.2 / -61.6 | -4.8 / -43.0 |
| e3_resid_ms_raw_repladapt | 5.377 | 1.114 | -67.0 | 0.905 | 13.1 | 1.73 | -43.8 / -52.6 | -53.7 / -67.0 | -10.8 / -47.7 |
| e3_resid_ms_raw_alone1x | 2.928 | 1.160 | -39.3 | 0.888 | 5.5 | 0.84 | -21.5 / -28.9 | -24.2 / -39.3 | -16.6 / -27.4 |
| e3_resid_m_raw_add25 | 4.726 | 1.048 | -61.3 | 0.867 | 12.6 | 1.63 | -42.5 / -50.6 | -48.2 / -61.3 | 4.3 / -46.6 |
| e3_resid_m_raw_repladapt | 5.139 | 1.060 | -66.7 | 0.850 | 12.9 | 1.73 | -43.6 / -52.4 | -53.8 / -66.7 | 0.9 / -52.0 |
| e3_resid_m_raw_alone1x | 2.657 | 1.024 | -38.5 | 0.809 | 5.5 | 0.84 | -21.0 / -28.4 | -24.3 / -38.5 | 3.9 / -31.9 |
| e3_resid_m_shp_replxs | 3.774 | 0.978 | -53.7 | 0.790 | 14.8 | 1.62 | -40.9 / -47.8 | -41.9 / -53.7 | -2.8 / -41.2 |
| e3_resid_m_shp_add25 | 4.097 | 0.987 | -58.0 | 0.787 | 13.9 | 1.63 | -42.1 / -49.8 | -45.9 / -58.0 | 1.7 / -43.0 |
| e3_resid_ms_shp_replxs | 3.745 | 0.986 | -53.8 | 0.786 | 15.0 | 1.62 | -40.7 / -47.7 | -41.8 / -53.8 | -2.8 / -38.3 |
| e3_resid_ms_shp_add25 | 4.076 | 0.994 | -58.1 | 0.784 | 14.0 | 1.63 | -42.0 / -49.7 | -45.8 / -58.1 | 1.8 / -40.4 |
| e3_resid_m_shp_repladapt | 4.274 | 0.982 | -62.8 | 0.741 | 14.6 | 1.73 | -43.1 / -51.3 | -50.9 / -62.8 | -2.0 / -47.6 |
| e3_resid_ms_shp_repladapt | 4.241 | 0.989 | -62.9 | 0.738 | 14.8 | 1.73 | -42.9 / -51.2 | -50.8 / -62.9 | -2.0 / -44.4 |
| e3_resid_m_shp_alone1x | 1.410 | 0.729 | -32.8 | 0.482 | 8.0 | 0.84 | -20.8 / -26.7 | -18.4 / -30.8 | -5.5 / -26.1 |
| e3_resid_ms_shp_alone1x | 1.336 | 0.730 | -32.1 | 0.466 | 8.1 | 0.84 | -20.6 / -26.7 | -18.6 / -31.5 | -7.2 / -21.0 |

## G1 verdicts (blend roles; alone1x is diagnostic-only)

| config | Calmar | hard gate | median neighbor Calmar | stability (≥85%) | G1 |
|---|---|---|---|---|---|
| e3_resid_ms_raw_replxs | 0.962 | fail | 0.906 | PASS | fail |
| e3_resid_m_raw_replxs | 0.909 | fail | 0.859 | PASS | fail |
| e3_resid_ms_raw_add25 | 0.908 | fail | 0.886 | PASS | fail |
| e3_resid_ms_raw_repladapt | 0.905 | fail | 0.879 | PASS | fail |
| e3_resid_m_raw_add25 | 0.867 | fail | 0.879 | PASS | fail |
| e3_resid_m_raw_repladapt | 0.850 | fail | 0.886 | PASS | fail |
| e3_resid_m_shp_replxs | 0.790 | fail | 0.787 | PASS | fail |
| e3_resid_m_shp_add25 | 0.787 | fail | 0.787 | PASS | fail |
| e3_resid_ms_shp_replxs | 0.786 | fail | 0.787 | PASS | fail |
| e3_resid_ms_shp_add25 | 0.784 | fail | 0.787 | PASS | fail |
| e3_resid_m_shp_repladapt | 0.741 | fail | 0.789 | PASS | fail |
| e3_resid_ms_shp_repladapt | 0.738 | fail | 0.785 | PASS | fail |

Neighborhood = configs differing in exactly one grid dimension (beta_mode flip, scale_mode flip, or one of the other two blend roles); median of their dev Calmars vs 85% of the candidate's.

## Winners advanced (≤2)

- none — family fails G1.

## Live-twin parity (tol 1e-9, 3 seeded dates/variant)

| beta_mode | scale_mode | date | max abs diff | verdict |
|---|---|---|---|---|
| market | raw | 2025-08-08 | 0.0e+00 | PASS |
| market | raw | 2022-11-01 | 0.0e+00 | PASS |
| market | raw | 2023-06-05 | 0.0e+00 | PASS |
| market | sharpe | 2024-09-05 | 0.0e+00 | PASS |
| market | sharpe | 2019-05-03 | 0.0e+00 | PASS |
| market | sharpe | 2024-02-05 | 0.0e+00 | PASS |
| market_sector | raw | 2025-01-06 | 0.0e+00 | PASS |
| market_sector | raw | 2025-06-09 | 0.0e+00 | PASS |
| market_sector | raw | 2019-10-02 | 0.0e+00 | PASS |
| market_sector | sharpe | 2018-06-01 | 0.0e+00 | PASS |
| market_sector | sharpe | 2024-06-05 | 0.0e+00 | PASS |
| market_sector | sharpe | 2024-07-08 | 0.0e+00 | PASS |

## ADDED CELL (protocol: ≤3, clearly marked) — rationale

The grid has no RAW-momentum control at matched leverage (1x) and n_long (30), and no other family ledger contains one, so the core E3 claim (similar return, smaller factor crashes than raw momentum) is unverifiable from pre-registered cells alone. Added cell: `xs_momentum(px, macro, n_long=30)` standalone 1x, diagnostic only, excluded from G1. Rationale written before the run.

Result: | e3_CONTROL_xsmom_raw_alone1x | 2.617 | 0.984 | -42.6 | 0.720 | 5.9 | 0.84 | -24.8 / -31.9 | -29.9 / -42.6 | 14.3 / -29.0 | (same columns as grid table)

## Crash-episode verification (the E3 pitch) — vs raw momentum, matched 1x / n=30

| sleeve (1x, n=30, dev) | mm %/mo | Sharpe | MaxDD % | Calmar | 2018Q4 ret/dd | covid ret/dd | 2022 ret/dd |
|---|---|---|---|---|---|---|---|
| RAW xs_momentum (control) | 2.617 | 0.984 | -42.6 | 0.720 | -24.8 / -31.9 | -29.9 / -42.6 | +14.3 / -29.0 |
| resid market raw | 2.657 | 1.024 | -38.5 | 0.809 | -21.0 / -28.4 | -24.3 / -38.5 | +3.9 / -31.9 |
| resid mkt+sector raw | 2.928 | 1.160 | -39.3 | 0.888 | -21.5 / -28.9 | -24.2 / -39.3 | -16.6 / -27.4 |
| resid market sharpe | 1.410 | 0.729 | -32.8 | 0.482 | -20.8 / -26.7 | -18.4 / -30.8 | -5.5 / -26.1 |
| resid mkt+sector sharpe | 1.336 | 0.730 | -32.1 | 0.466 | -20.6 / -26.7 | -18.6 / -31.5 | -7.2 / -21.0 |

Verdict on the literature claim (dev window):
- CONFIRMED for beta-driven crashes: raw variants deliver the same-or-better
  return than raw momentum (2.66–2.93 vs 2.62 %/mo) with higher Sharpe
  (1.02–1.16 vs 0.98), smaller MaxDD (-38.5/-39.3 vs -42.6%) and shallower
  2018Q4 (-28 vs -32%) and covid (-38/-39 vs -43%) drawdowns. The sharpe-
  scaled (paper iMom) variants cut crashes hardest (MaxDD ≈ -32%, 2022 dd
  -21/-26%) but halve the return — not competitive at this program's bar.
- MIXED in 2022 (a RATE crash, not a beta crash): raw momentum earned +14.3%
  in 2022 because its energy/value tilt was the trade; residualization
  strips precisely that sector bet, so resid mkt+sector lost -16.6% (though
  with the smallest in-year dd, -27.4%). Residual momentum buys crash
  protection by giving up regime-tilt upside.

## G1 conclusion and anomalies

- **G1: FAIL for the family.** Best cell `e3_resid_ms_raw_replxs` (resid
  mkt+sector raw replacing xs_momentum, ×2.0) clears Calmar (0.962 ≥ 0.944,
  +12% over champion 0.858), mean_monthly (4.81% ≥ 4.0%) and neighborhood
  stability (median neighbor Calmar 0.906 ≥ 0.85×0.962=0.818), but FAILS
  MaxDD > -50% (-58.3%). Every 2x blend fails the same criterion; the DD is
  the covid episode at ~1.6x effective gross, which no sleeve-composition
  change can fix — it requires an exposure overlay (E1/E2 territory).
- Recommendation for assembly: `resid_ms_raw` is the best E3 artifact —
  as a replacement for xs_momentum it improves every dev headline metric of
  the champion (mm 4.81 vs 4.55, Sharpe 1.125 vs 1.044, DD -58.3 vs -60.2,
  Calmar 0.962 vs 0.858) and is the natural base under a vol-managed /
  drawdown-gate overlay. Standalone it is the highest-Sharpe 1x sleeve seen
  in the program (1.160, turnover only 5.5x/yr).
- Anomaly notes: (1) parity-check dates were drawn (seeded) from the full
  frame index, incl. post-2022 dates — a mechanical weight-equality check
  only; no validation-window performance was computed or viewed. (2) The
  repladapt cells have the highest mm (5.14–5.38%) but the worst DD (-67%):
  dropping the adaptive sleeve removes the only VIX-responsive de-risking in
  the blend — direction consistent, no anomaly. (3) No grid cell errored;
  one added cell total (raw-momentum control), within the ≤3 allowance.
