# E1 — Strategy-level vol-managed momentum (Barroso/Santa-Clara; Daniel/Moskowitz)

**Date:** 2026-06-11 · **Family:** `E1_volmanaged` · **Window:** dev only (≤ 2022-12-31)
**Engine:** engine_v2, exec=next_open, 5bp, leverage_cap=2.0 · **Script:** `exp_e1_volmanaged.py`
**Ledger:** `results/v7/trials/E1_volmanaged.csv` (56 trials = 54 pre-registered + 2 added cells)

## Verdict: G1 **FAIL** — advance 0 configs

No cell satisfies the joint hard gates (Calmar ≥ 0.944, mean ≥ 4.0%/mo, MaxDD > −50%).
The feasible region is empty: vol management on this book trades mean return for
drawdown nearly proportionally, so Calmar improves only modestly (best 0.932 vs
baseline 0.858, +8.6%) and only at low targets where the mean collapses to ~2.8%/mo.

| | mean_mo | Sharpe | MaxDD | Calmar | turnover | avg gross |
|---|---|---|---|---|---|---|
| baseline combo_v2_2x (dev) | **4.55%** | 1.044 | −60.2% | 0.858 | 14.1 | 1.62 |
| best E1: lev2 tv0.30 lb21 fl0.3 | 2.77% | 1.069 | **−35.4%** | **0.932** | 16.7 | 1.10 |
| frontier probe: lev2 tv0.60 lb21 fl0.3 (ADDED) | 4.05% | 1.023 | −51.6% | 0.877 | 15.9 | 1.53 |

## Key findings
1. **DD is cut exactly as the literature predicts.** At tv=0.30/lb=21 the dev MaxDD
   falls −60.2% → −35.4%; covid DD −60.2% → −30.4%; 2018Q4 DD −50.6% → −32.9%;
   2022 DD −43.6% → −25.0%.
2. **The cost in mean is severe and nearly proportional:** 4.55% → 2.77%/mo (−39%).
   Sharpe barely moves (1.044 → 1.069), i.e. vol-timing adds almost no alpha on this
   book — the multiplier is < 1 most of the time (avg gross 1.62 → 1.10), so the
   overlay is mostly static deleveraging, not timing.
3. **Short lookback strictly dominates:** at every target/arm, Calmar(lb21) >
   Calmar(lb63) > Calmar(lb126). The book's crashes (covid) are fast; 63/126d vol
   estimates de-risk too late and re-risk too late. (Classic Barroso 6-mo lookback
   is the *worst* setting here.)
4. **Floors rarely bind;** fl=0.3 is marginally best at lb=21 (slightly higher mean,
   same DD). At lb ≥ 63 floor cells are identical (multiplier never goes that low).
5. **Symmetric arm (BASE_LEV=1.5, cap=1.333) is uniformly dominated** by the
   descale-only 2.0x arm — releveraging in calm regimes adds little because the
   base book's own gross caps near the same place, while the lower base leverage
   costs mean everywhere.
6. **2022 trade-off:** the baseline made +8.0% in 2022; every E1 cell gives that up
   (best cell −1.0%) because book vol stayed elevated all year, keeping exposure
   low. The overlay cuts both tails.

## Episode columns — winner candidates vs baseline (dev)
| config | 2018Q4 ret/dd | covid ret/dd | 2022 ret/dd |
|---|---|---|---|
| baseline combo_v2_2x | −43.3% / −50.6% | −48.4% / −60.2% | +8.0% / −43.6% |
| E1_lev2_tv0.3_lb21_fl0.3 (best Calmar) | −27.6% / −32.9% | −24.1% / −30.4% | −1.0% / −25.0% |
| E1_lev2_tv0.5_lb21_fl0.3 (2nd ridge) | −39.2% / −46.3% | −39.8% / −45.1% | −1.1% / −37.6% |
| E1_lev2_tv0.6_lb21_fl0.3 (ADDED probe) | −42.2% / −49.5% | −43.9% / −49.7% | −0.4% / −40.3% |

## G1 gate detail
- Hard gates: **0 / 56** cells pass (Calmar ≥ 0.944 ∧ mean ≥ 4.0% ∧ MaxDD > −50%).
  - Best Calmar 0.932 (< 0.944) at mean 2.77% (< 4.0%).
  - Cells reaching mean ≥ 4.0% (only the added tv=0.60 probe, 4.05%) breach the DD
    gate (−51.6%) and sit at Calmar 0.877.
- Neighborhood stability (reported for completeness on the best cell): neighbors of
  `E1_lev2_tv0.3_lb21_fl0.3` = {tv0.4·lb21·fl0.3, tv0.3·lb63·fl0.3, tv0.3·lb21·fl0.2,
  tv0.3·lb21·flnone} → median Calmar 0.890 = 95.5% of winner (≥ 85% ⇒ stability
  itself PASSES; the family fails on the hard gates, not on stability).

## Added cells (2 of max 3 allowed, rationale)
The only competitive ridge was lb=21; the pre-registered targets stopped at 0.50
with mean 3.77%. The probes `tv=0.60, lb=21, fl=0.3` (both arms) test whether any
target reaches the 4%/mo gate before breaching −50% DD. Answer: no —
mean 4.05%/3.96% with DD −51.6% in both arms. The frontier is closed; no further
cells justified.

## Anomalies / notes
- No grid-cell errors; all 56 trials ran clean (~0.3 s each).
- Verification pass (2026-06-11, second agent): best cell reproduced exactly via
  engine_v2 direct call (mm 2.7678%, Sharpe 1.069, MaxDD −35.4%, Calmar 0.932);
  0/56 hard-gate result re-confirmed from the ledger. Neighbor-list correction:
  under the strict ordered-floor (None < 0.2 < 0.3) ±1-step convention the best
  cell has 3 neighbors {tv0.4·lb21·fl0.3, tv0.3·lb63·fl0.3, tv0.3·lb21·fl0.2},
  median Calmar 0.877 = 94.1% of winner (report's 4-neighbor variant gave 95.5%).
  Stability PASSES under both conventions; verdict unchanged.
- Duplicate metrics across floor cells at lb ≥ 63 are expected (floor never binds),
  not a logging bug.
- The base frame's gross occasionally reaches 1.167 pre-leverage, so ×2.0 rows hit
  the engine's 2.0 cap exactly as the baseline champion does; the overlay only
  descales from there (cap isn't doing extra work in the (a) arm; in the (b) arm
  1.5×1.333×1.167 ≈ 2.33 is also engine-capped, same as baseline behavior).

## Recommendation for assembly phase
Do NOT advance E1 standalone. However, `vol_managed_multiplier(unscaled_2x_dev_net,
0.30, lookback=21, cap=1.0, floor=0.3)` is the family's best DD-compressor
(−35% MaxDD, covid −30%) and is worth testing **in combination** with
return-preserving overlays (e.g. gate-style or regime overlays from other families)
if the program needs a DD reducer that keeps Sharpe ≥ baseline.

## Full grid (from family ledger)
| name | base_lev | tv | lb | floor | mean_mo | sharpe | maxDD | calmar | turn | gross | covid_dd | 2022_ret | 2022_dd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1_lev2_tv0.3_lb21_fl0.2 | 2 | 0.3 | 21 | 0.2 | 2.71% | 1.053 | -35.4% | 0.908 | 16.7 | 1.10 | -30.0% | -1.0% | -25.0% |
| E1_lev2_tv0.3_lb21_fl0.3 | 2 | 0.3 | 21 | 0.3 | 2.77% | 1.069 | -35.4% | 0.932 | 16.7 | 1.10 | -30.4% | -1.0% | -25.0% |
| E1_lev2_tv0.3_lb21_flnone | 2 | 0.3 | 21 | nan | 2.70% | 1.049 | -35.4% | 0.903 | 16.7 | 1.10 | -30.0% | -1.0% | -25.0% |
| E1_lev2_tv0.3_lb63_fl0.2 | 2 | 0.3 | 63 | 0.2 | 2.58% | 0.997 | -41.4% | 0.729 | 12.0 | 1.06 | -41.4% | 3.0% | -20.4% |
| E1_lev2_tv0.3_lb63_fl0.3 | 2 | 0.3 | 63 | 0.3 | 2.61% | 1.007 | -41.4% | 0.739 | 12.0 | 1.06 | -41.4% | 3.0% | -20.4% |
| E1_lev2_tv0.3_lb63_flnone | 2 | 0.3 | 63 | nan | 2.58% | 0.997 | -41.4% | 0.729 | 12.0 | 1.06 | -41.4% | 3.0% | -20.4% |
| E1_lev2_tv0.3_lb126_fl0.2 | 2 | 0.3 | 126 | 0.2 | 2.42% | 0.921 | -47.4% | 0.579 | 10.8 | 1.04 | -47.4% | 4.9% | -22.0% |
| E1_lev2_tv0.3_lb126_fl0.3 | 2 | 0.3 | 126 | 0.3 | 2.42% | 0.921 | -47.4% | 0.579 | 10.8 | 1.04 | -47.4% | 4.9% | -22.0% |
| E1_lev2_tv0.3_lb126_flnone | 2 | 0.3 | 126 | nan | 2.42% | 0.921 | -47.4% | 0.579 | 10.8 | 1.04 | -47.4% | 4.9% | -22.0% |
| E1_lev2_tv0.4_lb21_fl0.2 | 2 | 0.4 | 21 | 0.2 | 3.29% | 1.027 | -43.9% | 0.864 | 17.6 | 1.32 | -38.4% | -2.9% | -32.4% |
| E1_lev2_tv0.4_lb21_fl0.3 | 2 | 0.4 | 21 | 0.3 | 3.33% | 1.037 | -43.9% | 0.877 | 17.5 | 1.32 | -38.4% | -2.9% | -32.4% |
| E1_lev2_tv0.4_lb21_flnone | 2 | 0.4 | 21 | nan | 3.29% | 1.027 | -43.9% | 0.864 | 17.6 | 1.32 | -38.4% | -2.9% | -32.4% |
| E1_lev2_tv0.4_lb63_fl0.2 | 2 | 0.4 | 63 | 0.2 | 3.10% | 0.956 | -49.6% | 0.700 | 13.6 | 1.29 | -49.6% | 1.8% | -26.8% |
| E1_lev2_tv0.4_lb63_fl0.3 | 2 | 0.4 | 63 | 0.3 | 3.10% | 0.956 | -49.6% | 0.700 | 13.6 | 1.29 | -49.6% | 1.8% | -26.8% |
| E1_lev2_tv0.4_lb63_flnone | 2 | 0.4 | 63 | nan | 3.10% | 0.956 | -49.6% | 0.700 | 13.6 | 1.29 | -49.6% | 1.8% | -26.8% |
| E1_lev2_tv0.4_lb126_fl0.2 | 2 | 0.4 | 126 | 0.2 | 3.01% | 0.910 | -55.5% | 0.591 | 12.4 | 1.27 | -55.5% | 4.4% | -28.4% |
| E1_lev2_tv0.4_lb126_fl0.3 | 2 | 0.4 | 126 | 0.3 | 3.01% | 0.910 | -55.5% | 0.591 | 12.4 | 1.27 | -55.5% | 4.4% | -28.4% |
| E1_lev2_tv0.4_lb126_flnone | 2 | 0.4 | 126 | nan | 3.01% | 0.910 | -55.5% | 0.591 | 12.4 | 1.27 | -55.5% | 4.4% | -28.4% |
| E1_lev2_tv0.5_lb21_fl0.2 | 2 | 0.5 | 21 | 0.2 | 3.77% | 1.026 | -48.5% | 0.880 | 17.0 | 1.46 | -45.1% | -1.1% | -37.6% |
| E1_lev2_tv0.5_lb21_fl0.3 | 2 | 0.5 | 21 | 0.3 | 3.77% | 1.027 | -48.5% | 0.881 | 17.0 | 1.46 | -45.1% | -1.1% | -37.6% |
| E1_lev2_tv0.5_lb21_flnone | 2 | 0.5 | 21 | nan | 3.77% | 1.026 | -48.5% | 0.880 | 17.0 | 1.46 | -45.1% | -1.1% | -37.6% |
| E1_lev2_tv0.5_lb63_fl0.2 | 2 | 0.5 | 63 | 0.2 | 3.54% | 0.948 | -54.9% | 0.706 | 14.4 | 1.44 | -54.9% | 1.0% | -32.9% |
| E1_lev2_tv0.5_lb63_fl0.3 | 2 | 0.5 | 63 | 0.3 | 3.54% | 0.948 | -54.9% | 0.706 | 14.4 | 1.44 | -54.9% | 1.0% | -32.9% |
| E1_lev2_tv0.5_lb63_flnone | 2 | 0.5 | 63 | nan | 3.54% | 0.948 | -54.9% | 0.706 | 14.4 | 1.44 | -54.9% | 1.0% | -32.9% |
| E1_lev2_tv0.5_lb126_fl0.2 | 2 | 0.5 | 126 | 0.2 | 3.51% | 0.929 | -59.4% | 0.636 | 13.6 | 1.43 | -59.4% | 3.0% | -34.4% |
| E1_lev2_tv0.5_lb126_fl0.3 | 2 | 0.5 | 126 | 0.3 | 3.51% | 0.929 | -59.4% | 0.636 | 13.6 | 1.43 | -59.4% | 3.0% | -34.4% |
| E1_lev2_tv0.5_lb126_flnone | 2 | 0.5 | 126 | nan | 3.51% | 0.929 | -59.4% | 0.636 | 13.6 | 1.43 | -59.4% | 3.0% | -34.4% |
| E1_lev2_tv0.6_lb21_fl0.3 | 2 | 0.6 | 21 | 0.3 | 4.05% | 1.023 | -51.6% | 0.877 | 15.9 | 1.53 | -49.7% | -0.4% | -40.3% |
| E1_lev1.5_tv0.3_lb21_fl0.2 | 1.5 | 0.3 | 21 | 0.2 | 2.54% | 1.025 | -35.4% | 0.841 | 16.5 | 1.05 | -28.8% | -1.7% | -25.0% |
| E1_lev1.5_tv0.3_lb21_fl0.3 | 1.5 | 0.3 | 21 | 0.3 | 2.57% | 1.034 | -35.4% | 0.853 | 16.5 | 1.06 | -28.8% | -1.7% | -25.0% |
| E1_lev1.5_tv0.3_lb21_flnone | 1.5 | 0.3 | 21 | nan | 2.54% | 1.025 | -35.4% | 0.841 | 16.5 | 1.05 | -28.8% | -1.7% | -25.0% |
| E1_lev1.5_tv0.3_lb63_fl0.2 | 1.5 | 0.3 | 63 | 0.2 | 2.49% | 1.001 | -39.3% | 0.738 | 11.6 | 1.01 | -39.3% | 2.6% | -20.3% |
| E1_lev1.5_tv0.3_lb63_fl0.3 | 1.5 | 0.3 | 63 | 0.3 | 2.49% | 1.001 | -39.3% | 0.738 | 11.6 | 1.01 | -39.3% | 2.6% | -20.3% |
| E1_lev1.5_tv0.3_lb63_flnone | 1.5 | 0.3 | 63 | nan | 2.49% | 1.001 | -39.3% | 0.738 | 11.6 | 1.01 | -39.3% | 2.6% | -20.3% |
| E1_lev1.5_tv0.3_lb126_fl0.2 | 1.5 | 0.3 | 126 | 0.2 | 2.34% | 0.927 | -45.5% | 0.584 | 10.3 | 0.99 | -45.5% | 5.0% | -21.3% |
| E1_lev1.5_tv0.3_lb126_fl0.3 | 1.5 | 0.3 | 126 | 0.3 | 2.34% | 0.927 | -45.5% | 0.584 | 10.3 | 0.99 | -45.5% | 5.0% | -21.3% |
| E1_lev1.5_tv0.3_lb126_flnone | 1.5 | 0.3 | 126 | nan | 2.34% | 0.927 | -45.5% | 0.584 | 10.3 | 0.99 | -45.5% | 5.0% | -21.3% |
| E1_lev1.5_tv0.4_lb21_fl0.2 | 1.5 | 0.4 | 21 | 0.2 | 3.17% | 1.020 | -43.9% | 0.828 | 17.5 | 1.27 | -36.8% | -3.8% | -32.4% |
| E1_lev1.5_tv0.4_lb21_fl0.3 | 1.5 | 0.4 | 21 | 0.3 | 3.17% | 1.021 | -43.9% | 0.828 | 17.5 | 1.27 | -36.8% | -3.8% | -32.4% |
| E1_lev1.5_tv0.4_lb21_flnone | 1.5 | 0.4 | 21 | nan | 3.17% | 1.020 | -43.9% | 0.828 | 17.5 | 1.27 | -36.8% | -3.8% | -32.4% |
| E1_lev1.5_tv0.4_lb63_fl0.2 | 1.5 | 0.4 | 63 | 0.2 | 3.03% | 0.962 | -48.8% | 0.697 | 13.5 | 1.25 | -48.8% | 1.5% | -26.3% |
| E1_lev1.5_tv0.4_lb63_fl0.3 | 1.5 | 0.4 | 63 | 0.3 | 3.03% | 0.962 | -48.8% | 0.697 | 13.5 | 1.25 | -48.8% | 1.5% | -26.3% |
| E1_lev1.5_tv0.4_lb63_flnone | 1.5 | 0.4 | 63 | nan | 3.03% | 0.962 | -48.8% | 0.697 | 13.5 | 1.25 | -48.8% | 1.5% | -26.3% |
| E1_lev1.5_tv0.4_lb126_fl0.2 | 1.5 | 0.4 | 126 | 0.2 | 2.98% | 0.925 | -54.6% | 0.599 | 12.3 | 1.24 | -54.6% | 4.8% | -27.5% |
| E1_lev1.5_tv0.4_lb126_fl0.3 | 1.5 | 0.4 | 126 | 0.3 | 2.98% | 0.925 | -54.6% | 0.599 | 12.3 | 1.24 | -54.6% | 4.8% | -27.5% |
| E1_lev1.5_tv0.4_lb126_flnone | 1.5 | 0.4 | 126 | nan | 2.98% | 0.925 | -54.6% | 0.599 | 12.3 | 1.24 | -54.6% | 4.8% | -27.5% |
| E1_lev1.5_tv0.5_lb21_fl0.2 | 1.5 | 0.5 | 21 | 0.2 | 3.63% | 1.012 | -48.5% | 0.842 | 17.5 | 1.42 | -44.2% | -2.4% | -37.6% |
| E1_lev1.5_tv0.5_lb21_fl0.3 | 1.5 | 0.5 | 21 | 0.3 | 3.63% | 1.012 | -48.5% | 0.842 | 17.5 | 1.42 | -44.2% | -2.4% | -37.6% |
| E1_lev1.5_tv0.5_lb21_flnone | 1.5 | 0.5 | 21 | nan | 3.63% | 1.012 | -48.5% | 0.842 | 17.5 | 1.42 | -44.2% | -2.4% | -37.6% |
| E1_lev1.5_tv0.5_lb63_fl0.2 | 1.5 | 0.5 | 63 | 0.2 | 3.47% | 0.953 | -54.3% | 0.702 | 14.2 | 1.41 | -54.3% | 0.5% | -32.1% |
| E1_lev1.5_tv0.5_lb63_fl0.3 | 1.5 | 0.5 | 63 | 0.3 | 3.47% | 0.953 | -54.3% | 0.702 | 14.2 | 1.41 | -54.3% | 0.5% | -32.1% |
| E1_lev1.5_tv0.5_lb63_flnone | 1.5 | 0.5 | 63 | nan | 3.47% | 0.953 | -54.3% | 0.702 | 14.2 | 1.41 | -54.3% | 0.5% | -32.1% |
| E1_lev1.5_tv0.5_lb126_fl0.2 | 1.5 | 0.5 | 126 | 0.2 | 3.48% | 0.942 | -59.0% | 0.640 | 13.3 | 1.40 | -59.0% | 3.6% | -33.3% |
| E1_lev1.5_tv0.5_lb126_fl0.3 | 1.5 | 0.5 | 126 | 0.3 | 3.48% | 0.942 | -59.0% | 0.640 | 13.3 | 1.40 | -59.0% | 3.6% | -33.3% |
| E1_lev1.5_tv0.5_lb126_flnone | 1.5 | 0.5 | 126 | nan | 3.48% | 0.942 | -59.0% | 0.640 | 13.3 | 1.40 | -59.0% | 3.6% | -33.3% |
| E1_lev1.5_tv0.6_lb21_fl0.3 | 1.5 | 0.6 | 21 | 0.3 | 3.96% | 1.013 | -51.6% | 0.854 | 16.2 | 1.51 | -49.4% | -0.7% | -40.3% |
