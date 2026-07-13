# Book-drawdown exposure gate — backtest validation

Generated 2026-06-10 15:07. Engine: vector backtest (weights.shift(1) x close-to-close, 5bp/side on turnover incl. gate-driven exposure changes, gross cap 1.0 pre-leverage, 2.0x leverage), full sample 2016-04-01 .. 2026-03-27.

## 0. Fidelity check (mandatory)
- Reproduced baseline: mean monthly 4.956%, Sharpe 1.0632, MaxDD -65.28%
- Published (results_v6.json): 4.956%, 1.0632, -65.28%
- Deltas: 0.0000pp / 0.0000 / 0.000pp -> **PASS**

## 1. Full-period results

| variant | mean mo % | Sharpe | MaxDD % | Calmar | CAGR % | worst mo % | worst day % | avg gross | gate d/yr | cash d/yr |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_no_gate | 4.956 | 1.063 | -65.28 | 0.866 | 56.53 | -29.13 | -24.83 | 1.701 | 0.0 | 0.0 |
| spy_gate_8_18 | 4.573 | 1.085 | -55.47 | 0.949 | 52.65 | -35.32 | -16.73 | 1.634 | 26.2 | 3.2 |
| book_gate_8_20 | 3.704 | 1.101 | -49.24 | 0.866 | 42.65 | -22.78 | -16.73 | 1.362 | 132.3 | 13.6 |
| book_gate_10_25 | 4.305 | 1.132 | -46.69 | 1.064 | 49.67 | -28.48 | -16.73 | 1.516 | 92.2 | 4.6 |
| book_gate_12_25 | 4.465 | 1.124 | -47.26 | 1.079 | 50.98 | -30.68 | -16.73 | 1.560 | 63.8 | 4.6 |
| book_gate_12_30 | 4.724 | 1.157 | -47.18 | 1.164 | 54.94 | -31.96 | -16.73 | 1.601 | 63.7 | 1.6 |
| book_gate_15_30 | 4.897 | 1.151 | -50.48 | 1.119 | 56.50 | -34.26 | -16.73 | 1.640 | 36.2 | 1.6 |
| book_gate_15_35 | 4.961 | 1.146 | -51.04 | 1.120 | 57.15 | -34.32 | -16.73 | 1.658 | 36.2 | 0.7 |
| combined_min_spy_book_gate_12_30 | 4.322 | 1.102 | -47.02 | 1.056 | 49.66 | -31.96 | -16.73 | 1.560 | 72.6 | 3.4 |

## 2. Stress-episode behavior (window return / max drawdown inside window)

| variant | 2018Q4 | 2020-03 | 2020_covid_crash | 2022_full_year | 2026_jan_mar |
|---|---|---|---|---|---|
| baseline_no_gate | -45.0% / -52.3% | -27.5% / -55.1% | -45.7% / -65.3% | +2.4% / -41.7% | +18.9% / -29.4% |
| spy_gate_8_18 | -47.0% / -47.6% | -17.4% / -20.3% | -35.7% / -37.8% | +5.4% / -26.8% | +29.4% / -27.2% |
| book_gate_8_20 | -31.0% / -31.0% | -7.2% / -7.2% | -16.5% / -18.9% | -17.2% / -36.4% | +3.9% / -31.7% |
| book_gate_10_25 | -39.6% / -40.1% | -15.5% / -16.2% | -28.7% / -31.0% | -12.8% / -37.8% | +12.8% / -28.2% |
| book_gate_12_25 | -42.7% / -43.3% | -17.8% / -18.5% | -32.1% / -34.3% | -9.2% / -37.3% | +15.2% / -27.9% |
| book_gate_12_30 | -44.1% / -46.1% | -21.5% / -23.7% | -36.0% / -38.4% | -5.7% / -37.0% | +19.3% / -26.9% |
| book_gate_15_30 | -47.2% / -49.4% | -25.0% / -27.4% | -39.9% / -42.3% | -1.8% / -37.9% | +22.3% / -27.2% |
| book_gate_15_35 | -46.7% / -50.2% | -31.7% / -34.8% | -45.7% / -49.0% | +0.7% / -38.5% | +24.7% / -27.2% |
| combined_min_spy_book_gate_12_30 | -44.6% / -45.3% | -17.8% / -20.0% | -32.9% / -35.1% | -5.0% / -26.4% | +19.3% / -26.9% |

Best book-gate config by Calmar: **book_gate_12_30** (full_dd=12%, cash_dd=30%).

## 3. Notes
- Gate is computed from the un-gated book's target weights and each name's own 60-day close high; signal at close t applies to day t+1 (1-day lag), identical to the live decision cadence.
- Live ordering reproduced: blend -> gate -> gross cap 1.0 -> 2.0x leverage. Mild gate values (<~15% cut) can be absorbed by the cap on days the raw blend gross exceeds 1.0 — same as live.
- SPY soft gate 8->18 quantified on the combo for the first time here.

## 4. Recommendation
- **ENABLE the book-DD gate at full_dd=12%, cash_dd=30%** (60d trailing highs, weights = current target book renormalized, 1-day lag).
- vs baseline it costs 0.23pp/mo (4.96 -> 4.72) and buys 18.1pp of MaxDD (-65.3% -> -47.2%), Sharpe 1.06 -> 1.16, Calmar 0.87 -> 1.16, worst day -24.8% -> -16.7%. Decision rule (OFF if >0.3pp/mo for <5pp DD) clearly says ON.
- **DISABLE the SPY soft ramp 8->18 once the book gate is live.** Stacking it (min of the two gates) costs an extra 0.40pp/mo for only 0.16pp of additional MaxDD improvement — fails the same rule. The book gate sees everything the SPY gate sees (broad crashes drag the book down too) plus the sector crashes SPY is blind to. If the operator insists on keeping both, the combined config is still acceptable (4.32%/mo, Sharpe 1.10, MaxDD -47.0%) and is the best 2022 protector, but it is Calmar-dominated by book-only.
- The deployed SPY gate alone (never previously backtested): 4.57%/mo, Sharpe 1.08, MaxDD -55.5% — net positive vs baseline, but inferior to the book gate on every risk metric.
- Known costs of the book gate: 2022 flips from +2.4% to -5.7% (whipsaw in a grinding factor bear while SPY-keyed gates fared better there); ~64 engaged days/yr. Known limits: the June-2026 sector crash itself is beyond the data (cache ends 2026-03-27); in the in-sample Jan-Mar 2026 stress the gate only trims window MaxDD -29.4% -> -26.9% because fast crashes outrun a close-to-close gate.
