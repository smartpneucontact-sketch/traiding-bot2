# E2 — Dynamic Sleeve Allocation (replace static 1/3 blend)

Date: 2026-06-11 · family=`E2_alloc` · script: `exp_e2_alloc.py` · ledger: `results/v7/trials/E2_alloc.csv` (18 rows)
Window: **dev only** (≤ 2022-12-31), exec=next_open, tc=5bp, leverage 2.0x (blend ×2.0).
Baseline to beat (combo_v2_2x dev): mean_monthly **+4.555%/mo**, Sharpe **1.044**, MaxDD **-60.2%**, Calmar **0.858**.

## Method

- Sleeve inputs: 3 cached frames (`sleeve_xs_momentum_n30`, `sleeve_dual_momentum_voltarget_n30`, `sleeve_adaptive_voltarget_n30`), each simulated ONCE at 1x via engine_v2 (next_open, 5bp) on dev-truncated prices → per-sleeve daily net returns (direct engine calls; not in the ledger).
- At each union decision date d (74 dev dates), allocation a_i(d) computed from sleeve returns **strictly before d**, restricted to the all-sleeves-active period (≥ 2017-05-02). If < lookback joint observations → equal-weight 1/3 fallback (verified to reproduce the baseline blend bit-for-bit on those dates).
- Floors via waterfall: sleeves below the floor are fixed AT the floor; the rest renormalized over the remaining mass (exact floors, sums to 1).
- Blend = Σ a_i(d)·w_i(d) on union dates/cols (same alignment as `reproduce_baseline.build_combo_v2_base`), ×2.0, scored through `exp_lib.run_trial`.

## Full grid (18 pre-registered configs, sorted by dev Calmar)

| name | method | lb | floor | mm %/mo | Sharpe | MaxDD % | Calmar | turn_ann | med nbr Calmar | stab ok | G1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2_sharpe_softmax_lb126_f10 | sharpe_softmax | 126 | 0.10 | **+4.608** | **1.066** | **-54.00** | **0.976** | 14.28 | 0.909 | yes | **no (DD)** |
| e2_sharpe_softmax_lb126_f20 | sharpe_softmax | 126 | 0.20 | +4.577 | 1.059 | -54.69 | 0.955 | 14.25 | 0.914 | yes | no (DD) |
| e2_sharpe_wtm_lb126_f20 | sharpe_wtm | 126 | 0.20 | +4.531 | 1.041 | -55.34 | 0.922 | 14.36 | 0.906 | yes | no |
| e2_sharpe_rank_lb126_f10 | sharpe_rank | 126 | 0.10 | +4.571 | 1.042 | -56.51 | 0.911 | 14.29 | 0.906 | yes | no |
| e2_sharpe_wtm_lb126_f10 | sharpe_wtm | 126 | 0.10 | +4.558 | 1.037 | -56.31 | 0.906 | 14.62 | 0.917 | yes | no |
| e2_sharpe_rank_lb126_f20 | sharpe_rank | 126 | 0.20 | +4.561 | 1.043 | -56.83 | 0.905 | 14.23 | 0.917 | yes | no |
| e2_sharpe_softmax_lb252_f10 | sharpe_softmax | 252 | 0.10 | +4.382 | 1.012 | -58.79 | 0.830 | 14.22 | 0.826 | yes | no |
| e2_sharpe_rank_lb252_f20 | sharpe_rank | 252 | 0.20 | +4.329 | 1.004 | -57.83 | 0.830 | 14.38 | 0.829 | yes | no |
| e2_sharpe_softmax_lb252_f20 | sharpe_softmax | 252 | 0.20 | +4.379 | 1.011 | -58.79 | 0.829 | 14.22 | 0.830 | yes | no |
| e2_sharpe_wtm_lb252_f20 | sharpe_wtm | 252 | 0.20 | +4.279 | 0.996 | -56.97 | 0.828 | 14.52 | 0.829 | yes | no |
| e2_invvol_lb126_f10 | invvol | 126 | 0.10 | +4.401 | 1.022 | -59.85 | 0.827 | 14.25 | 0.822 | yes | no |
| e2_invvol_lb126_f20 | invvol | 126 | 0.20 | +4.401 | 1.022 | -59.85 | 0.827 | 14.25 | 0.822 | yes | no |
| e2_sharpe_rank_lb252_f10 | sharpe_rank | 252 | 0.10 | +4.292 | 0.996 | -57.54 | 0.823 | 14.45 | 0.830 | yes | no |
| e2_erc_lb60_f20 | erc | 60 | 0.20 | +4.384 | 1.022 | -60.09 | 0.821 | 14.36 | 0.821 | yes | no |
| e2_erc_lb60_f10 | erc | 60 | 0.10 | +4.384 | 1.022 | -60.09 | 0.821 | 14.36 | 0.821 | yes | no |
| e2_invvol_lb63_f10 | invvol | 63 | 0.10 | +4.376 | 1.020 | -60.26 | 0.817 | 14.32 | 0.822 | yes | no |
| e2_invvol_lb63_f20 | invvol | 63 | 0.20 | +4.376 | 1.020 | -60.26 | 0.817 | 14.32 | 0.822 | yes | no |
| e2_sharpe_wtm_lb252_f10 | sharpe_wtm | 252 | 0.10 | +4.146 | 0.967 | -55.90 | 0.803 | 14.82 | 0.829 | yes | no |

Full numeric table: `results/v7/E2_grid_results.csv`; allocation stats: `results/v7/E2_alloc_stats.json`.

## G1 verdict: **FAIL — 0 of 18 configs pass** (MaxDD leg)

- Calmar ≥ 0.944: passed by 2 configs (softmax lb126 f10 → 0.976; f20 → 0.955).
- mean_monthly ≥ 4.0%: passed by all 18.
- **MaxDD > -50%: passed by NONE** (best -54.0%). This is the sole blocker for the top two.
- Neighborhood stability: passed by all (family surface is smooth).

Conclusion: dynamic Sharpe-tilt allocation **improves** the champion on every dev metric (Calmar +13.7%, +0.05pp/mo, Sharpe +0.022, DD -60.2 → -54.0) but cannot, by itself, pull the book DD under -50%. The covid crash (-54.0% episode DD) dominates regardless of sleeve mix because all three sleeves crash together — allocation across them is not a crash hedge. These configs are strong **inputs for assembly** with a DD-control overlay family (vol-managed / book-DD gate), not standalone G1 advances.

## Best config (record, not advanced): `e2_sharpe_softmax_lb126_f10`

`{"method": "sharpe_softmax", "lookback": 126, "floor": 0.10, "tau": 1.0}`
Dev: +4.608%/mo, Sharpe 1.066, MaxDD -54.00%, Calmar 0.976, turnover 14.28x, avg gross 1.62x.
Episodes: 2018Q4 DD -51.8%, covid DD -54.0% (= global max), 2022 ret +4.6% / DD -45.1%, worst month -33.9%.

Neighborhood (±1 step): f20 0.955, lb252 0.830, rank_lb126_f10 0.911, wtm_lb126_f10 0.906 → median 0.909 ≥ 0.85×0.976 = 0.830. **Stable.**

### Allocation character (does it deviate from 1/3?)

Yes, materially per-date, neutral on average: 62/74 dates dynamic (12 fallback during burn-in), mean abs deviation from 1/3 = 0.080 per sleeve.
- xs: avg 0.334, range [0.147, 0.529]
- dual: avg 0.346, range [0.138, 0.620]
- adapt: avg 0.321, range [0.124, 0.648]
The edge is timing (rotating toward the trailing-Sharpe leader, esp. away from crashing sleeves in 2018Q4/2022), not a static tilt. Runner-up f20 is the same with clipped extremes (floors bind at 0.20).

## Anomalies / notes

1. **Degenerate grid cells**: invvol and ERC produce identical results at floor 0.10 vs 0.20 — the three sleeves have similar vols, so allocations never go below 0.20 and the floor never binds. Effectively 14 distinct configs of 18 (all 18 logged).
2. Pure risk-based methods (invvol, ERC) are ~flat vs baseline (Calmar 0.82-0.83 vs 0.858 — slightly WORSE; they tilt toward the low-vol sleeve, which is not the high-Calmar sleeve). Performance-based (Sharpe) tilts at lb=126 are where the gains are; lb=252 is too slow and degrades.
3. Floor 0.10 ≥ 0.20 at lb126 softmax/rank (more room to de-allocate a crashing sleeve), but wtm prefers f20 — differences are second-order.
4. No grid cells errored; no cells added.
