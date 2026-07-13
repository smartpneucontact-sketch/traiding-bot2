# E4 — WEIGHTING FIXES (inverse-vol xs sleeve + a WORKING portfolio vol target)

Family: `E4_weighting` · Window: **dev only** (≤ 2022-12-31) · exec: next_open · 5 bp · lev ×2.0 (cap 2.0)
Script: `exp_e4_weighting.py` · Ledger: `results/v7/trials/E4_weighting.csv` · 10 trials (pre-registered grid, no added cells)
Diagnostics: `results/v7/e4_dual_formula_diagnostic.csv`, `results/v7/e4_verdicts.json`
New cached sleeves (reusable): `weights_store/sleeve_xs_momentum_invvol_n30.parquet`, `weights_store/sleeve_dual_momentum_truevol15_n30.parquet`

## Verdict: G1 **FAIL** — advance 0 configs

Every weighting "fix" makes dev Calmar WORSE than the unfixed baseline (0.858). The two flaws are
empirically load-bearing features: equal-weight xs keeps the high-vol winners that drive return, and the
broken (correlation-free) dual vol formula effectively disables a vol target that, when fixed, only taxes return.
A true book-level vol target at monthly decision frequency cannot dodge fast crashes (COVID DD still -44%
at the tightest target) but reliably caps the high-vol recovery rallies where the blend earns its money.

## Sanity anchor

`blend3(cached sleeves) == combo_v2_base_1x.parquet` to 0.0e+00; `e4_base_eq_2x` reproduces the baseline
row exactly (4.555%/mo, Sharpe 1.044, MaxDD -60.2%, Calmar 0.858).

## Full grid (dev window)

| name | xs | book VT | dual | mm %/mo | Sharpe | MaxDD | Calmar | ann vol | turn | covid DD | 2022 ret |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4_base_eq_2x (anchor) | eq | — | orig | **4.55** | 1.044 | -60.2% | **0.858** | 0.542 | 14.1 | -60.2% | +8.0% |
| e4_xs_invvol_2x (a) | 1/σ60 | — | orig | 4.36 | 1.020 | -59.5% | 0.822 | 0.531 | 14.8 | -59.5% | +10.7% |
| e4_vt25_eq_2x (b) | eq | 0.25 | orig | 2.16 | 0.943 | -44.1% | 0.574 | 0.282 | 8.3 | -44.1% | +15.6% |
| e4_vt25_iv_2x (b) | 1/σ60 | 0.25 | orig | 2.09 | 0.900 | -45.1% | 0.535 | 0.286 | 8.9 | -45.1% | +17.2% |
| e4_vt30_eq_2x (b) | eq | 0.30 | orig | 2.58 | 0.952 | -49.7% | 0.602 | 0.335 | 9.7 | -49.7% | +18.0% |
| e4_vt30_iv_2x (b) | 1/σ60 | 0.30 | orig | 2.51 | 0.917 | -50.2% | 0.573 | 0.339 | 10.4 | -50.2% | +20.0% |
| e4_vt35_eq_2x (b) | eq | 0.35 | orig | 2.98 | 0.967 | -53.8% | 0.639 | 0.382 | 11.0 | -53.8% | +20.1% |
| e4_vt35_iv_2x (b) | 1/σ60 | 0.35 | orig | 2.88 | 0.930 | -54.3% | 0.600 | 0.384 | 11.6 | -54.3% | +22.4% |
| e4_dualfix_eq_2x (c) | eq | — | truevol15 | 3.97 | 1.009 | -55.7% | 0.815 | 0.493 | 13.0 | -55.7% | +12.3% |
| e4_dualfix_iv_2x (c) | 1/σ60 | — | truevol15 | 3.77 | 0.982 | -55.0% | 0.776 | 0.494 | 13.6 | -55.0% | +15.1% |

G1 thresholds: Calmar ≥ 0.944, mm ≥ 4.0%, MaxDD > -50%, neighbor-median Calmar ≥ 85% of winner.
**No row clears the Calmar gate; only the vt25/vt30 rows clear the DD gate and they miss mm by ~2×.**

## Neighborhood stability (informational — gate moot)

The Calmar surface is smooth and monotone: along target {0.25→0.30→0.35→none} Calmar rises
0.574→0.602→0.639→0.858 (eq) and 0.535→0.573→0.600→0.822 (iv); the iv column is uniformly ~0.03-0.04
below eq; dualfix sits uniformly ~0.04 below orig. Every config satisfies the 85% neighbor-median test
(`results/v7/e4_verdicts.json`), i.e. these are not noise artifacts — the fixes are *systematically* worse.

## How often the vol targets bind (dev decision dates, n=74)

- **True book VT (levered 2x basis, 60d covariance):** binds 92% (vt25), 86-88% (vt30), 77-78% (vt35) of
  decision dates; average scale when binding 0.52 / 0.60 / 0.66; minimum scale 0.25 / 0.29 / 0.34.
  The levered book's ex-ante vol is typically 40-50%, so 25-35% targets are nearly always active.
- **Dual sleeve, old correlation-free formula (strategies.py:306):** median estimate 8.1% vs the 15%
  target → binds on only **2 of 74** dates (2020-05-04, 2020-06-03 — post-COVID single-name vols so extreme
  even the zero-correlation underestimate crossed 15%). So "never binds" is 97.3% literally true; the formula
  understates true book vol by ~2.6× (median true ex-ante 21.4%, p90 36.8%, max 48.7%).
- **Dual sleeve, fixed covariance formula (15% target):** binds **91.9%** of dates.

## Why the fixes fail (mechanism)

1. **Monthly decision dates make ex-ante vol targeting reactive, not protective.** COVID (2020-02-24→03-23)
   played out almost entirely between rebalances: even vt25 still took -44% in the episode. The scaling
   arrives at the *next* decision date, after vol has spiked — by then it mostly de-levers the rebound.
2. **The blend's return engine is high-vol momentum names in high-vol recoveries.** Capping book vol at
   25-35% surrenders ~50% of monthly return for ~16pp of MaxDD: Calmar falls. (2022 is the exception —
   all VT rows improve 2022 — but it cannot offset COVID-recovery give-up in the dev window.)
3. **Inverse-vol inside xs tilts away from exactly the names momentum wants.** Return falls 0.20%/mo while
   COVID DD improves only 0.6pp; turnover even rises slightly (vol-driven weight drift).
4. **The dual sleeve's broken formula was an accidental feature:** fixing it shrinks the sleeve ~30% on 92%
   of dates (gross 1.62→1.48), losing 0.59%/mo against only 4.5pp of DD.

## Anomalies / deviations

- None of the grid cells errored; no cells were added. Rationale for not using the 3 permitted extra cells:
  the Calmar surface is monotone toward "no vol target" on every axis, so any looser target or shorter
  window interpolates toward the 0.858 anchor from below and cannot reach the 0.944 gate.
- Precision note vs the brief: the old dual formula does bind twice (not literally never) — see above.

## Hand-off notes for assembly

- Do **not** fold (a), (b), or (c) into the V7 candidate book as specified here.
- If the assembly phase wants DD control, the E4 evidence says decision-date book-vol scaling is the wrong
  tool at 21-day cadence; daily overlays (E1-style vol_managed_multiplier / book_drawdown_gate) act between
  rebalances and are the better mechanism class.
- The two new sleeves remain cached in `weights_store/` for reuse:
  `xs_momentum_invvol(px, n_long=30, vol_lookback=60)` and
  `dual_momentum_truevol(px, n_long=30, target_vol=0.15, vol_lookback=60)` in `exp_e4_weighting.py`.
