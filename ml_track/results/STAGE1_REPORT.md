# ML-1 — Stage 0 (engine fidelity) + Stage 1 (CORE 12 signal matrix)

Date: 2026-06-11. Window: **dev only** (2016-04 .. 2022-12-31), exec `next_open`, 5 bp, leverage cap 2.0x.
Code: `ml_track/` package; runner `exp_ml1_stage01.py` (== `python3 -m ml_track.run_all_stage1`);
post-hoc audit `python3 -m ml_track.verify_stage1`. Ledgers: `ml_track/results/ledger.csv`,
`results/v7/trials/ML_stage0.csv`, `results/v7/trials/ML_stage1.csv`.

## Stage 0 — GATE 0 fidelity: PASSED

`combo_v2_base_1x × 2.0` through `exp_lib.run_trial` (family `ML_stage0`) reproduces
`results/v7/baseline.json` dev **exactly** (diff < 1e-9 on all three asserted stats; in fact bit-identical):

| stat | re-run | baseline.json (dev) |
|---|---|---|
| mean_monthly | 0.045546739999310346 | 0.045546739999310346 |
| sharpe | 1.0440033129817838 | 1.0440033129817838 |
| max_drawdown | -0.6019153206391645 | -0.6019153206391645 |
| calmar | 0.8580186751301543 | 0.8580186751301543 |

Same engine, same numbers → Stage 1 portfolio comparisons are apples-to-apples.

## Setup (pre-registered)

- Candidates per decision date: union of top-80 by 12-1 momentum and top-80 by 6m return (>0 only),
  ≥273 valid closes. Pool=120 variant for S1_07/08.
- Features: F1 = 36 features, all lookbacks ≤ 273 bars (asserted; v5/v6 OBV/A-D lines windowed to 252+20
  bars to honor the cap — documented in `features.py`). Cached `ml_track/cache/features_F1*.parquet`.
- Label: r_fwd21 = close[t+22]/close[t+1] − 1 (execution-day convention). L1 = cs pct-rank, L2 = vol-scaled
  pct-rank, L3/L3v = above-median binary.
- CV: PurgedAnchoredWF, test years 2018–2022, purge label_end ≥ test_start, 21-day embargo,
  early-stop eval = last 15% of training dates. ~79k train rows (p80), ~118k (p120).
- All configs scored on the SAME OOF rows (5-day eval grid) → ICs directly comparable.
- **Momentum-prior IC benchmark (the bar): 0.01758** (p80 rows); **0.02452** on p120 rows.
- Gate 1: rank-IC ≥ 0.03 AND IC ≥ prior+0.01 AND ≥60% positive fold-years AND hit30 ≥ 0.52.

## Stage 1 matrix — full results (from `ml_track/results/ledger.csv`)

| exp_id | arch | label | pool | wgt | grid | mono | rank_ic | t | prior_ic | edge | pos_yrs | hit30 | G1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_01_Arank_L1 | A_rank | L1 | 80 | EW | 5 | – | **0.0391** | 3.46 | 0.0176 | **+0.0215** | 100% | 0.5153 | **FAIL** (hit30) |
| S1_02_Arank_L2 | A_rank | L2 | 80 | EW | 5 | – | 0.0124 | 1.40 | 0.0176 | −0.0052 | 60% | 0.5019 | FAIL |
| S1_03_Areg_L1 | A_reg | L1 | 80 | EW | 5 | – | 0.0201 | 1.96 | 0.0176 | +0.0025 | 80% | 0.5015 | FAIL |
| S1_04_Areg_L2 | A_reg | L2 | 80 | EW | 5 | – | −0.0086 | −0.80 | 0.0176 | −0.0262 | 40% | 0.4911 | FAIL |
| S1_05_Bcls_L1 | B_cls | L1 | 80 | EW | 5 | – | 0.0326 | 3.02 | 0.0176 | +0.0150 | 100% | 0.5115 | FAIL (hit30) |
| S1_06_Bcls_L2 | B_cls | L2 | 80 | EW | 5 | – | −0.0083 | −0.49 | 0.0176 | −0.0259 | 20% | 0.5000 | FAIL |
| S1_07_Areg_L1_p120 | A_reg | L1 | 120 | EW | 5 | – | 0.0334 | 3.11 | 0.0245 | +0.0089 | 100% | 0.5116 | FAIL (edge, hit30) |
| S1_08_Bcls_L1_p120 | B_cls | L1 | 120 | EW | 5 | – | 0.0191 | 1.57 | 0.0245 | −0.0054 | 60% | 0.5056 | FAIL |
| S1_09_Areg_L1_IV | A_reg | L1 | 80 | IV | 5 | – | 0.0201 | 1.96 | 0.0176 | +0.0025 | 80% | 0.5015 | FAIL |
| S1_10_Bcls_L1_IV | B_cls | L1 | 80 | IV | 5 | – | 0.0326 | 3.02 | 0.0176 | +0.0150 | 100% | 0.5115 | FAIL (hit30) |
| S1_11_Areg_L1_mono | A_reg | L1 | 80 | EW | 5 | mom↑ | 0.0204 | 2.01 | 0.0176 | +0.0028 | 80% | 0.5030 | FAIL |
| S1_12_Arank_L1_g21 | A_rank | L1 | 80 | EW | 21 | – | 0.0241 | 2.32 | 0.0176 | +0.0065 | 100% | 0.5138 | FAIL |

Per-year OOF rank-IC, best configs:

- S1_01 (A_rank/L1): 2018 +0.024, 2019 +0.026, 2020 +0.074, 2021 +0.050, 2022 +0.021 — positive every year.
- S1_05 (B_cls/L1): 2018 +0.006, 2019 +0.032, 2020 +0.038, 2021 +0.074, 2022 +0.014.
- S1_07 (A_reg/L1 p120): 2018 +0.038, 2019 +0.019, 2020 +0.030, 2021 +0.078, 2022 +0.003.

### Gate-1 verdict: **0 / 12 pass.**

The two strongest configs (S1_01, S1_05) clear three of four criteria and fail ONLY top-30
hit-rate (51.5% / 51.2% vs the 52.0% bar). Everything else fails on edge-over-prior too.

## Diagnostic I2 portfolio runs (extra cells, clearly marked)

Pre-registration allows ≤3 added cells with rationale. Rationale: even though Gate 1 failed, the two
best signals were pushed through the I2 integration ((ML + dual + adaptive)/3 × 2.0, family
`ML_stage1`) to measure whether the IC edge converts to portfolio quality — information the next
agent needs. These are diagnostics, NOT gate-passers.

| run | mean_monthly | Sharpe | MaxDD | Calmar | turn_ann | vs baseline (4.555%/mo, 1.044, −60.2%, 0.858) |
|---|---|---|---|---|---|---|
| I2_S1_01_Arank_L1 | 4.811% | 1.095 | −63.4% | 0.871 | 13.8 | +0.26pp/mo, +1.5% Calmar, **DD worse** |
| I2_S1_05_Bcls_L1 | 4.231% | 1.030 | −59.6% | 0.801 | 15.1 | worse than baseline |

Neither approaches the G1 portfolio bar (Calmar ≥ 0.944, MaxDD > −50%). The replaced xs_momentum
sleeve is essentially as good as the best ML re-ranker at the portfolio level.

## Anomalies and caveats

1. **Isotonic calibration optimism (B_cls).** Calibration is fitted on pooled OOF (per spec) and
   *raised* measured rank-IC: S1_05 raw pre-isotonic IC = **0.0246** vs 0.0326 calibrated
   (S1_06: −0.0039 raw; S1_08: +0.0186 raw). Isotonic is weakly monotone, so Spearman can only move
   through ties — and the tie structure is formed using the scored rows' own labels (mild leakage).
   The honest B_cls number is the raw one (0.0246, fails the 0.03 bar outright). Verified and
   reproducible via `verify_stage1.v5_rawscore` (ledger rows V_05/V_06/V_08). A_rank scores have no
   calibration step — S1_01's 0.0391 is clean.
2. **S1_06 NaN year-ICs**: isotonic collapsed most scores to constants on many dates (Spearman
   undefined) — the L2/L3v target is essentially unlearnable here; raw score IC is negative anyway.
3. **S1_09/S1_10 duplicate S1_03/S1_05 signal metrics by design** — weighting (EW vs IV) only acts at
   the portfolio stage, which is gated; the WF is memoized on the model key. These cells are
   informative only post-gate.
4. **Vol-scaled labels (L2) hurt everywhere** (best L2 cell 0.0124 vs 0.0391 L1 counterpart).
5. **5d training grid > 21d** (S1_01 0.0391 vs S1_12 0.0241): 4× more training rows is worth real IC.
6. **Pool 120 raises the prior too** (0.0176 → 0.0245); the model edge over the prior *shrinks*
   (S1_07 +0.0089 vs S1_03 +0.0025 is a gain, but A_reg remains far below A_rank).

## Audit (verify_stage1, fresh process)

- V1 determinism: S1_01/S1_05 re-runs match the ledger to all logged digits. OK
- V2 look-ahead: all 36 features identical when the panel is truncated at the decision date. OK
- V3 labels: 25/25 sampled r_fwd21 match hand-computed close[t+22]/close[t+1]−1; label_end_date correct. OK
- V4 Gate-1 arithmetic re-verified on all 12 rows. OK
- V5 raw-score anomaly rows reproduced exactly. OK

## Read: does ML add signal?

**At the signal level, yes — modestly.** LambdaRank on cs-rank labels (S1_01) delivers OOF rank-IC
0.039 (t = 3.5) vs the within-pool momentum prior of 0.018, positive in all five fold-years,
including 2022. That is a real, clean (no calibration step, purged/embargoed) +0.021 IC edge.

**At the selection/portfolio level, no — not yet.** The edge lives mostly in the middle of the
candidate distribution: top-30 hit-rate is 51.5% (needs 52%), and the I2 diagnostic backtest turns
the IC edge into only +1.5% Calmar with a *worse* MaxDD — nowhere near the program bar (Calmar
≥ 0.944, DD > −50%). Honest negative under the pre-registered gate: **no config advances; 0/12 pass
Gate 1; nothing is forwarded to assembly.**

For the follow-up agent (NOT run here, per budget discipline): the gradient worth probing is
A_rank/L1 with (a) head-weighted objectives (lambdarank truncation at 30, or top-k focused labels),
(b) HP_M capacity, (c) hysteresis/portfolio construction that exploits mid-ranks (e.g. rank-weighted
instead of top-N), and (d) pool-120 with the ranker (untested cell). The hit30 miss is small (0.5
percentage points); head-focused training is the most direct attack on it.
