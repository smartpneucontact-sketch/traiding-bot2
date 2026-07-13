# ML-2 — Stage-1 rescue follow-ups (Gate-1 retry) and track closure

Date: 2026-06-11. Window: **dev only** (CV test years 2018–2022; portfolio runs
2016-04..2022-12), exec `next_open`, 5 bp, leverage cap 2.0x.
Scripts: `exp_ml2_rescue.py` (4 pre-registered cells), `exp_ml2_diag.py`
(2 added-cell diagnostics, rationale below). Library: `ml_track/rescue_lib.py`.
Ledgers: `ml_track/results/ledger.csv` (rows R1–R4, *_diagI2),
`results/v7/trials/ML_stage1.csv` (rows I2_R2_Arank_L4, I2_R4_Arank_L1_p120).

## Context

Stage 1 (ML-1) ended 0/12 on Gate 1; best near-miss S1_01_Arank_L1
(A_rank/L1/p80/EW/HP_S/g5): rank-IC 0.0391 (t=3.46), edge +0.0215,
100% positive years, **hit30 51.53% vs the 52.0% bar**. Per the ML-2
assignment, at most 4 rescue follow-ups on that cell were allowed, chosen
among {HP_M, L4 momentum-residualized label, F2 extended features, pool=120}.
All four were run — exactly one cell per allowed modification, all on the
S1_01 base. No grid cell errored; no substitutions.

## Pre-registered rescue grid — results

Gate 1: rank_ic ≥ 0.03 AND edge over momentum prior ≥ 0.01 AND ≥60% positive
fold-years AND hit30 ≥ 0.52. Same purged/embargoed anchored WF, same OOF eval
rows as Stage 1 (priors: 0.0176 on p80 rows, 0.0245 on p120 rows).

| exp_id | change vs S1_01 | rank_ic | t | prior | edge | pos_yrs | hit30 | G1 |
|---|---|---|---|---|---|---|---|---|
| R1_Arank_L1_HPM | HP_M capacity | 0.0275 | 2.48 | 0.0176 | +0.0099 | 100% | 0.5090 | FAIL (ic, edge, hit30) |
| R2_Arank_L4 | L4 residualized label | **0.0423** | **3.52** | 0.0176 | **+0.0247** | 100% | 0.5179 | FAIL (hit30 only) |
| R3_Arank_L1_F2 | F2 53-feature set | 0.0311 | 2.81 | 0.0176 | +0.0135 | 80% | 0.5061 | FAIL (hit30) |
| R4_Arank_L1_p120 | pool 120 | 0.0361 | 3.07 | 0.0245 | +0.0115 | 100% | **0.5196** | FAIL (hit30 only) |

Per-year OOF rank-IC:
- R2: 2018 +0.025, 2019 +0.041, 2020 +0.092, 2021 +0.028, 2022 +0.026 (all positive).
- R4: 2018 +0.015, 2019 +0.039, 2020 +0.086, 2021 +0.016, 2022 +0.023 (all positive).
- R3 (F2): 2022 −0.002 — the extended features *reduce* robustness.

### Gate-1 verdict: **0 / 4 pass.** Cumulative ML track: 0 / 16.

hit30 is the universal failure mode: every configuration with a real IC edge
(S1_01, S1_05, R2, R3, R4) lands in 50.6–52.0%. The signal edge concentrates
in the middle of the candidate distribution and does not reach the top-30 head
that the portfolio actually buys.

## Added cells (2 of ≤3 allowed; rationale pre-written here and in exp_ml2_diag.py)

RATIONALE: R2 carries the largest signal edge of the entire track and R4 the
best hit30; Stage 1 measured portfolio conversion only for S1_01/S1_05.
Measuring conversion of the strictly stronger rescue signals quantifies the
portfolio-level ceiling of this ML design — required for an honest closure.
These are diagnostics, NOT gate candidates; nothing advances from them.

I2 integration = (ML sleeve + dual_momentum + adaptive_voltarget)/3 × 2.0,
corrected engine, next_open, 5 bp, dev window (identical to Stage-1 diagnostics).
Baseline (corrected champion, dev): 4.555%/mo, Sharpe 1.044, MaxDD −60.2%, Calmar 0.858.

| run | mean_monthly | Sharpe | MaxDD | Calmar | turn_ann | vs baseline |
|---|---|---|---|---|---|---|
| I2_R2_Arank_L4 | 4.497% | 1.027 | −62.4% | 0.801 | 13.4 | WORSE on every axis |
| I2_R4_Arank_L1_p120 | 4.837% | 1.092 | −61.4% | 0.899 | 13.8 | +0.28pp/mo, +4.8% Calmar, DD worse |

Even ignoring the Gate-1 failure, I2_R4 — the best ML portfolio of the whole
track — fails the assignment's Gate 2 on 2 of 4 criteria (mean 4.837 < 4.855;
Calmar 0.899 < 0.958; turnover 13.8 ≤ 25 OK; MaxDD −61.4% ≤ −65.2% bound OK)
and sits 4.7% below the program's G1 portfolio bar (Calmar ≥ 0.944) with
MaxDD nowhere near the −50% requirement.

## Anomalies / findings

1. **HP_M hurts the ranker** (0.0275 vs 0.0391 with HP_S): capacity is not the
   binding constraint; 600 trees × 31 leaves overfit ~80k rows under
   early-stopping on the last 15% of training dates.
2. **L4 is the best signal of the track** (IC 0.0423, t 3.52, edge +0.0247,
   positive every year) — the model's incremental information is genuinely
   orthogonal to momentum — yet its standalone-selection portfolio is *worse*
   than baseline (Calmar 0.801). Structural, not noise: the residualized score
   de-emphasizes exactly the momentum component the sleeve needs to keep pace
   with the combo. More IC of this kind does not buy portfolio quality.
3. **F2 features add mid-rank IC but subtract robustness** (2022 year-IC turns
   negative; pos-years drops to 80%). The 17 extra technical columns mostly
   re-describe short-horizon reversal/vol already captured in F1.
4. **EWM-based F2 columns** (macd_hist, trend_dev_20) use pandas `ewm` from
   series start: backward-only (look-ahead-safe), effective lookback < 273 bars
   (weight at lag 273 < 1e-9), declared at the cap — documented in
   `ml_track/rescue_lib.py`.
5. **Pool 120 is the best portfolio direction but raises the prior too**
   (0.0245): edge +0.0115 barely clears the bar, hit30 51.96% misses by
   0.04pp, and the diagnostic portfolio still has −61.4% MaxDD.

## Verdict — track closed

0/16 configurations pass the pre-registered Gate 1 (12 core + 4 rescue), and
all four portfolio-conversion diagnostics (2 in Stage 1, 2 here) land at
Calmar 0.80–0.90 against the 0.944 bar with MaxDD ≥ 60%. Stage 2 integration
and Stage 3 robustness are therefore NOT run (gated). **No incremental
candidate-level ML signal advances to assembly.** Full statement:
`ml_track/results/ML_CONCLUSION.md`.
