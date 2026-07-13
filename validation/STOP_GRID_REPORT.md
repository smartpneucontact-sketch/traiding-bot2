# Stop-Layer Grid Backtest — combo_v2_2x (2026-06-10)

Backtest-validated parameters for the live 60-second cut-loss scanner, overlaid on the
reproduced combo_v2_2x champion book (3-sleeve momentum blend, 21-trading-day decisions,
2.0x leverage, 5bp/side, 2016-04-01 → 2026-03-27, 9.32 active years).

Scripts: `reproduce_baseline.py` (fidelity check), `stop_grid.py` (overlay engine + grid).
Full results: `stop_grid_results.json` (539 runs).

## 1. Fidelity check — PASSED to machine precision

| metric | published (results_v6.json) | reproduced | diff |
|---|---|---|---|
| mean monthly | +4.9558 % | +4.9558 % | 0.000000 |
| Sharpe | 1.0632 | 1.0632 | 0.000000 |
| MaxDD | -65.279 % | -65.279 % | 0.000000 |

The overlay engine with all stops disabled also reproduces the baseline daily returns
to 5.6e-17 (machine epsilon), so every grid delta is attributable to the stop layer alone.

**Fidelity finding (separate issue):** the engine's `weights.shift(1)` operates on the
SPARSE 113-row decision-date frame, so every decision is applied one decision ROW later
— i.e. one full 21-trading-day execution lag, not the 1-day lag the docstring claims.
The published 4.96 %/mo champion number includes this lag; the live bot trades without
it. The overlay matches the published convention exactly (entries at the actual
book-switch dates), but live-vs-backtest behavior should be reconciled separately.

## 2. The as-deployed config is catastrophic in backtest

`T=-5% trailing, H=-8% hard, pstop=-3% on levered equity (tiers -3/-5/-7), delay 1`:

| | baseline (no stops) | AS-DEPLOYED |
|---|---|---|
| mean monthly | **+4.96 %** | **+1.80 %** |
| Sharpe | 1.06 | 1.12 |
| MaxDD | -65.3 % | -33.7 % |
| Calmar | 0.87 | 0.63 |
| worst day | -24.8 % | -8.0 % |
| time in market | 100 % | **28 %** |
| stop events/yr | 0 | **533** (528 trailing + 5 tier) |

It gives up 3.16 pp/mo — the strategy's entire edge over SPY — by keeping the book only
28 % invested. 528 trailing stops/yr ≈ 2.1/day on average (far more in vol spikes), which
matches the observed live behavior of 90 fires in 5 days. The trailing stop is the killer:
a -5 % trail on single momentum stocks (daily vol 2-4 %) is inside one day's noise.

## 3. Grid results by family (normal mode = open/close checkpoint approximation)

**Trailing stops — remove.** Every value tested loses return without helping risk-adjusted
performance: T=-8 → 2.68 %/mo (Calmar 0.62); T=-12 → 3.46 (0.69); T=-20 → 4.40 (0.90).
Momentum names routinely retrace >8 % off their peak and keep going; selling the dip
forfeits the recovery for up to 21 days.

**Hard stops — remove.** H=-15 → 4.38 %/mo (Calmar 1.03, 137 stops/yr); H=-25 → 4.66
(0.89). At best Calmar-neutral-to-mildly-positive but always return-negative, and
dominated by the portfolio tier at equal risk reduction.

**Portfolio tiered stop — the only layer that earns its keep.** (T=none, H=none, tiers on
levered equity vs yesterday close, Tier2/3 = x5/3, x7/3):

| pstop_eff | mo % | Sharpe | MaxDD | Calmar | worst day | tier1/2/3 per yr | TiM |
|---|---|---|---|---|---|---|---|
| none (baseline) | 4.96 | 1.06 | -65.3 % | 0.87 | -24.8 % | — | 100 % |
| -4.0 (d=2) | 4.31 | 1.38 | -35.9 % | 1.53 | -9.5 % | 10.8 / 1.8 / 0.1 | 72 % |
| **-4.5 (any d)** | **4.75** | **1.44** | **-36.3 %** | **1.70** | -10.6 % | 10.4 / 1.0 / 0 | 76 % |
| -5.0 (d=1) | 4.90 | 1.42 | -42.4 % | 1.51 | -12.1 % | 8.8 / 0.6 / 0.1 | 80 % |
| -6.0 (d=2) | 4.78 | 1.26 | -45.6 % | 1.32 | -21.3 % | 6.9 / 0.4 / 0.2 | 85 % |
| -8.0 (any d) | 4.77 | 1.17 | -49.6 % | 1.14 | -14.5 % | 3.8 / 0.3 / 0 | 91 % |

Re-entry delay only matters when Tier3 fires; it never fires for -4.5 or -8. For -6 the
d=1/2/3 spread (Calmar 1.08/1.32/1.16, MaxDD -51.6/-45.6/-50.3) comes from re-entry
timing on just 2 episodes — pure path luck, not signal.

Formal rule winner (max Calmar s.t. mean monthly ≥ baseline - 0.3pp = 4.656 %):
**pstop_eff = -4.5 %, Calmar 1.70.**

## 4. Intraday-path stress — the tight tiers don't survive it

The daily-bar model only sees tier breaches at the open and the close; an intraday dip
through the threshold that recovers by the close is invisible, yet live it WOULD fire and
lock in the dip. To bound this, tiers were re-tested against a blend of the checkpoint
bottom and the simultaneous per-ticker-lows basket (phi=0 normal, phi=0.5 heuristic
mid-case, phi=1 = every holding at its low at the same minute, extreme worst case):

| config | phi=0 mo/Calmar | phi=0.5 mo/Calmar | phi=1 mo/Calmar |
|---|---|---|---|
| P=-4.5 | 4.75 / 1.70 | 3.24 / 0.85 | 1.84 / 0.37 |
| P=-5 | 4.90 / 1.51 | 3.37 / 0.84 | 2.29 / 0.44 |
| P=-6 d=2 | 4.78 / 1.32 | 4.21 / 1.16 | 2.59 / 0.47 |
| **P=-8** | **4.77 / 1.14** | **4.40 / 1.11** | **3.96 / 0.97** |
| AS-DEPLOYED | 1.80 / 0.63 | 1.46 / 0.48 | 0.66 / 0.13 |
| baseline | 4.96 / 0.87 | — | — |

-4.5 %/-5 % levered (= -2.25 %/-2.5 % underlying at 2x) sit inside ordinary intraday
noise: their measured edge collapses below the baseline once intraday whipsaw is priced.
**P=-8 is the only family whose Calmar stays at-or-above the no-stop baseline under every
path assumption**, because -8 % levered by 4 pm is almost always a genuinely bad day, not
a wiggle.

## 5. Recommendation

**Deploy: trailing stop OFF, hard stop OFF, portfolio tiers only, with
pstop_eff = -8 % on LEVERED equity** (= -4 % underlying at 2x):

- Tier1 ≤ -8.00 % levered intraday vs yesterday close → scale book to 60 %
- Tier2 ≤ -13.33 % → scale to 30 %
- Tier3 ≤ -18.67 % → liquidate, re-enter after 1-2 days (never fired in 9.3 yrs; keep it
  as tail insurance, delay choice is untestable in-sample so keep it short)

Expected (backtest, normal mode): **+4.77 %/mo, Sharpe 1.17, MaxDD -49.6 %, Calmar 1.14,
worst day -14.5 %, ~4 tier events/yr, 91 % time in market** — vs as-deployed +1.80 %/mo
with ~533 events/yr. Sized against the 3 %/mo goal, this keeps the champion's return
while removing the worst third of its drawdown and capping single-day damage.

If a higher-Calmar/lower-DD profile is preferred and intraday whipsaw turns out milder
than the mid-case stress (verifiable after ~3 months live by counting tier touches),
tightening to -6 % (d kept short) is the next step; -4.5/-5 should NOT be deployed on
daily-bar evidence alone. "All stops off" beats the as-deployed config by 3.2 pp/mo but is
dominated by P=-8 on every risk metric at a cost of only 0.19 pp/mo.

Implementation notes for the live scanner:
1. Delete the per-position trailing (-5 %) and hard (-8 %) stops.
2. Evaluate tiers on levered account equity vs yesterday's close with thresholds
   -8/-13.33/-18.67 % (i.e. set pstop = -8 as an EFFECTIVE levered-equity threshold;
   do not let the 2x leverage halve it the way the current -3/-5/-7 config does).
3. One tier action per day max per level; scaled exposure persists until next rebalance
   (no same-day or next-day re-levering after Tier1/2).
4. Tier3 re-entry: full scheduled target after 1-2 trading days, at the close.

## 6. Honest limits of the daily-bar approximation

- **Intraday sequencing unknown.** Tier breaches checked at open and close only (plus the
  phi-stress bound); individual stop fills assume min(open, level) — conservative on gaps,
  but the order of stop vs tier actions within a day is assumed, not known.
- **Trailing peaks use daily closes**, underestimating intraday peaks: live trailing stops
  would fire even MORE often than the modeled 528/yr — which only strengthens the removal.
- **Tier close-breach fills at threshold level** assume a monotone intraday decline; the
  same-day multi-tier walk inherits that assumption.
- **Constant-weight engine**: the baseline vector engine implicitly rebalances to target
  weights daily between decisions; entries/peaks are tracked on actual price paths. The
  no-stop limit is exact, but per-position drift is approximated.
- **No market-impact**: Tier3 liquidates a 2x book at 5bp; real impact on a -19 % day
  would be worse. (Mitigated: Tier3 never fires at the recommended -8 setting.)
- **In-sample selection**: parameters chosen and evaluated on the same 2016-2026 window
  (includes COVID-2020, 2022 bear, 2024-08, 2025 episodes). The -4.5/-5 refinement points
  were added after a fine scan (mild snooping); robustness rests on the plateau: every
  pstop_eff in [-4.5, -10] beats baseline Calmar at phi=0, and -8 ± 1 stays >1.1.
- **The 21-day execution-lag quirk** of the published engine (Section 1) is faithfully
  reproduced here but does not exist live; sleeve-level live drift vs backtest is a
  separate open item.
