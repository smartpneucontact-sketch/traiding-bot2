# FINAL VALIDATION REPORT — V7 program

Rendered from `results/v7/final_validation.json` (run generated 2026-06-12T00:34:51); no recomputation.

- Protocol: V7 one-shot final validation; locked window 2023-01-01..2026-03-27 evaluated exactly once
- Full index: 2016-04-01 .. 2026-03-27 (2512 days); tc=5bp unless stated; leverage cap 2.0
- Deflated-Sharpe trial count: 450 (ledger rows pre-run: 434)
- Metric provenance: numbers in the artifact predate the 2026-07-12 metric fixes (deflated-Sharpe kurtosis term, LPM2 sortino, deduped trial counts) and are rendered as recorded; deltas are immaterial at skew=0/excess_kurt=0 inputs but the values are not regenerable byte-identically from current code.

## Selection contamination (disclosure)

The champion benchmark (`combo_v2_base_1x` × 2.0) that anchors the G3 relative gates (val Calmar ≥ champion, 2026Q1 DD ≤ champion) was itself selected on the FULL window — including 2023-2026, the very window used here as 'validation'. The candidate sleeves' parameters were tuned on dev only, but the bar they are measured against has seen the answer key. Val-window results are therefore quasi-out-of-sample at the sleeve level, not a clean out-of-sample test of the candidate-vs-champion comparison.

## Step 1 — Fidelity anchors (PASS, tol 0.001)

| name | max abs diff | tier events dev (ref) | pass |
|---|---|---|---|
| ASM_Bst_TF1 | 9.02e-17 | (15, 0, 0) ((15, 0, 0)) | PASS |
| ASM_Bst_BTF1 | 2.22e-16 | (14, 0, 0) ((14, 0, 0)) | PASS |
| ASM_Bst_T | 4.86e-17 | (15, 1, 0) ((15, 1, 0)) | PASS |
| champion_combo_v2_2x | 0.00e+00 | — | PASS |

## Step 2 — Validation window 2023-01-01..2026-03-27 (tc=5bp)

| name | mean/mo | sharpe | maxDD | calmar | worst mo | hit | turn | gross | 26Q1 ret | 26Q1 dd | tier ev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ASM_Bst_TF1 | 6.39% | 1.581 | -36.52% | 2.603 | -21.41% | 63.16% | 19.3 | 1.80 | 16.47% | -23.48% | 13/0/0 |
| ASM_Bst_BTF1 | 6.15% | 1.615 | -28.11% | 3.223 | -23.00% | 65.79% | 25.3 | 1.72 | 10.57% | -24.93% | 10/0/0 |
| ASM_Bst_T | 6.86% | 1.672 | -36.80% | 2.875 | -21.41% | 65.79% | 18.9 | 1.83 | 16.47% | -23.48% | 13/0/0 |
| champion_combo_v2_2x | 6.55% | 1.374 | -56.01% | 1.621 | -25.60% | 65.79% | 17.2 | 1.92 | 19.98% | -27.20% | — |

## Step 3 — Full window at tc 5/10/20bp

| name | tc | mean/mo | sharpe | maxDD | calmar | 2018Q4 dd | covid dd | 2022 dd | 26Q1 dd |
|---|---|---|---|---|---|---|---|---|---|
| ASM_Bst_TF1 | 5 | 5.51% | 1.411 | -46.86% | 1.517 | -44.70% | -24.09% | -29.10% | -23.48% |
| ASM_Bst_TF1 | 10 | 5.43% | 1.392 | -47.02% | 1.481 | -44.87% | -24.21% | -29.35% | -23.55% |
| ASM_Bst_TF1 | 20 | 5.28% | 1.354 | -47.34% | 1.410 | -45.21% | -24.47% | -29.85% | -23.69% |
| ASM_Bst_BTF1 | 5 | 4.93% | 1.360 | -41.74% | 1.484 | -39.38% | -20.38% | -31.87% | -24.93% |
| ASM_Bst_BTF1 | 10 | 4.84% | 1.333 | -42.05% | 1.430 | -39.71% | -20.46% | -32.35% | -25.24% |
| ASM_Bst_BTF1 | 20 | 4.65% | 1.279 | -42.67% | 1.327 | -40.34% | -20.63% | -33.29% | -25.87% |
| ASM_Bst_T | 5 | 5.71% | 1.425 | -48.59% | 1.532 | -46.50% | -41.59% | -36.57% | -23.48% |
| ASM_Bst_T | 10 | 5.64% | 1.407 | -48.70% | 1.499 | -46.62% | -41.67% | -36.75% | -23.55% |
| ASM_Bst_T | 20 | 5.50% | 1.371 | -48.93% | 1.435 | -46.86% | -41.83% | -37.09% | -23.69% |
| champion_combo_v2_2x | 5 | 5.22% | 1.140 | -60.19% | 1.027 | -50.55% | -60.19% | -43.60% | -27.20% |
| champion_combo_v2_2x | 10 | 5.15% | 1.127 | -60.22% | 1.006 | -50.66% | -60.22% | -43.63% | -27.20% |
| champion_combo_v2_2x | 20 | 5.02% | 1.100 | -60.28% | 0.965 | -50.86% | -60.28% | -43.68% | -27.20% |

## Step 4 — Tier phi-stress (full window, tc=5bp)

Champion full-window Calmar (tc=5): 1.027. Promotion risk (TF1 phi=1.0 Calmar < champion): no.

| name | phi | mean/mo | sharpe | maxDD | calmar | calmar drop vs phi0 | dd widening (pp) | tier ev |
|---|---|---|---|---|---|---|---|---|
| ASM_Bst_TF1 | 0.0 | 5.51% | 1.411 | -46.86% | 1.517 | — | — | 28/0/0 |
| ASM_Bst_TF1 | 0.5 | 4.90% | 1.342 | -48.65% | 1.253 | 17.41% | -1.80 | 48/1/0 |
| ASM_Bst_TF1 | 1.0 | 4.09% | 1.211 | -43.89% | 1.119 | 26.22% | +2.97 | 75/1/0 |
| ASM_Bst_BTF1 | 0.0 | 4.93% | 1.360 | -41.74% | 1.484 | — | — | 24/0/0 |
| ASM_Bst_BTF1 | 0.5 | 4.47% | 1.308 | -41.74% | 1.319 | 11.11% | +0.00 | 40/1/0 |
| ASM_Bst_BTF1 | 1.0 | 3.77% | 1.177 | -41.81% | 1.070 | 27.89% | -0.06 | 66/0/0 |
| ASM_Bst_T | 0.0 | 5.71% | 1.425 | -48.59% | 1.532 | — | — | 28/1/0 |
| ASM_Bst_T | 0.5 | 5.10% | 1.359 | -50.33% | 1.272 | 17.02% | -1.74 | 51/1/0 |
| ASM_Bst_T | 1.0 | 4.25% | 1.218 | -49.80% | 1.025 | 33.10% | -1.21 | 78/1/0 |

## Step 5 — Statistics

Paired monthly diffs vs champion (full window), Newey-West lag 3:

| name | mean diff /mo | NW t | n months |
|---|---|---|---|
| ASM_Bst_TF1 | +0.285% | +0.661 | 119 |
| ASM_Bst_BTF1 | -0.288% | -0.736 | 119 |
| ASM_Bst_T | +0.493% | +1.278 | 119 |

Deflated Sharpe (n_trials=450, cross-trial SR var daily=1.258e-04):

| name | SR (ann) | skew | ex.kurt | P(true SR>0) | primary |
|---|---|---|---|---|---|
| ASM_Bst_TF1 | 1.411 | -0.38 | 1.6 | 0.9966 | yes |
| ASM_Bst_BTF1 | 1.360 | -0.46 | 1.9 | 0.9945 |  |
| ASM_Bst_T | 1.425 | -0.33 | 1.7 | 0.9971 |  |

## Step 6 — Gates (no re-tuning)

G3 anchors (val Calmar, 2026Q1 DD) inherit the selection contamination disclosed above.

### ASM_Bst_TF1 — G3 FAIL / G4 FAIL

| gate | criterion | result |
|---|---|---|
| G3 | mean>=4.5%/mo | PASS |
| G3 | val_MaxDD>=-35% | FAIL |
| G3 | calmar>=champ_val(1.621) | PASS |
| G3 | 2026Q1_dd<=champ(-27.20%) | PASS |
| G4 | mean>=5.0%/mo | PASS |
| G4 | MaxDD>=-40%(-45% if Calmar>=1.5) | FAIL |
| G4 | calmar>=1.4 | PASS |
| G4 | deflated_sharpe>0 | PASS |
| G4 | deflated_sharpe>0.5 (supplementary) | PASS |

### ASM_Bst_BTF1 — G3 PASS / G4 FAIL

| gate | criterion | result |
|---|---|---|
| G3 | mean>=4.5%/mo | PASS |
| G3 | val_MaxDD>=-35% | PASS |
| G3 | calmar>=champ_val(1.621) | PASS |
| G3 | 2026Q1_dd<=champ(-27.20%) | PASS |
| G4 | mean>=5.0%/mo | FAIL |
| G4 | MaxDD>=-40%(-45% if Calmar>=1.5) | FAIL |
| G4 | calmar>=1.4 | PASS |
| G4 | deflated_sharpe>0 | PASS |
| G4 | deflated_sharpe>0.5 (supplementary) | PASS |

### ASM_Bst_T — G3 FAIL / G4 FAIL

| gate | criterion | result |
|---|---|---|
| G3 | mean>=4.5%/mo | PASS |
| G3 | val_MaxDD>=-35% | FAIL |
| G3 | calmar>=champ_val(1.621) | PASS |
| G3 | 2026Q1_dd<=champ(-27.20%) | PASS |
| G4 | mean>=5.0%/mo | PASS |
| G4 | MaxDD>=-40%(-45% if Calmar>=1.5) | FAIL |
| G4 | calmar>=1.4 | PASS |
| G4 | deflated_sharpe>0 | PASS |
| G4 | deflated_sharpe>0.5 (supplementary) | PASS |

## Per-year returns (full window, tc=5bp)

| year | ASM_Bst_TF1 | champion_combo_v2_2x |
|---|---|---|
| 2016 | 6.24% | 6.01% |
| 2017 | 74.66% | 78.44% |
| 2018 | -3.24% | 1.36% |
| 2019 | 78.43% | 91.09% |
| 2020 | 306.17% | 165.69% |
| 2021 | 98.72% | 55.68% |
| 2022 | 2.31% | 9.48% |
| 2023 | 77.64% | 71.53% |
| 2024 | 94.88% | 81.54% |
| 2025 | 91.16% | 82.51% |
| 2026 | 21.72% | 29.43% |

Ledger trial count after run: 460. Tier ledger: `results/v7/trials/FINAL_VALIDATION_tier.csv`.
