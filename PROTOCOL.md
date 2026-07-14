# Pre-registered forward-test protocol — combo_v2 paper test

**Protocol version:** 1.0
**Committed:** 2026-07-14, before the first funded rebalance.
**Machine-readable twin:** `protocol.json` (same criteria; the scorer reads that file).
**Binding:** at the first funded (non-dry-run) rebalance, the bot records
`forward_test_start` and the sha256 of `protocol.json` into its state file
(`core/runner.py`). Any later change to `protocol.json` produces a hash
mismatch and the test is permanently flagged **MODIFIED_AFTER_START**.

## What is being tested

The combo_v2 strategy (bundle **v2.3_nofreeze**) trading a long-only,
2.0x-levered momentum blend on an Alpaca paper account, rebalanced every
21 trading days. The paper test is the only survivorship-bias-free
performance evidence available for this strategy: the historical numbers
below come from a survivors-only universe and are therefore upper bounds.

## Reference numbers (corrected backtest)

From the corrected 10-year backtest (2016-04 → 2026-03, engine_v2
`next_open` execution, 5 bp/side transaction cost, 600 bp/yr margin cost),
research repo `Traiding 11`:

| Quantity | Value | Derivation |
|---|---|---|
| Geometric monthly return | **3.68 %/mo** | `geo_monthly` of the champion combo_v2 2x run |
| Sharpe (annualized) | 1.06 | same run |
| Max drawdown | −60.3 % | same run |
| Monthly volatility σ_m | **≈ 15.7 %/mo** | back-derived from Sharpe: σ_m = μ_m·√12 / Sharpe ≈ 0.048 × 3.464 / 1.06 ≈ 0.157 |

## Power disclosure (read before interpreting anything)

At σ_m ≈ 15.7 %/mo, the standard error of the *mean* monthly return
(SE = σ_m/√n) is:

- **≈ 9.1 pp at 3 months** (15.7/√3)
- **≈ 6.4 pp at 6 months** (15.7/√6)
- **≈ 4.5 pp at 12 months** (15.7/√12)

The test **cannot confirm 3.68 %/mo** — the signal is far smaller than the
noise at every horizon below several years. What it *can* do: reject gross
failure, calibrate real trading costs (slippage vs the 5 bp/side backtest
assumption), and verify the machine matches backtest assumptions
(vol profile, fill rates, gross exposure).

## Continuous kill criteria (checked every rebalance)

- **K1 — KILL:** live drawdown ≤ **−45 %** from the forward-test equity peak.
- **K2 — KILL-or-recost:** slippage > **25 bp/side** over **≥ 3 consecutive
  rebalances**.
- **K3 — operational pause:** fill rate < **90 %** or gross deviation
  > **15 %**, twice consecutively.

## 3-month checkpoint (operational only — no performance verdict)

- Realized volatility ∈ **[7 %, 25 %]/mo**.
- Geometric monthly return > **−10 %/mo**.
- Slippage median ≤ **15 bp/side** on **≥ 3 rebalances**.
- Reconciliation green ≥ **2/3** of rebalances.

## 6-month checkpoint (band with regime AND-condition)

Cumulative log return must exceed **expected − 1.28·σ·√6**:

> expected = 6·ln(1.0368) ≈ 0.2168; band = 0.2168 − 1.28 × 0.157 × √6
> ≈ 0.2168 − 0.4922 = **−0.2754** in log terms (≈ −24.1 % simple).

AND-ed with the regime adjustment: breaching the band KILLs **only if
also** live − 2×SPY < **−15 pp** AND live < **0** over the same window.
(A levered long-momentum book losing less than 2× the index in a bear
market is behaving as designed, not failing.)

## 12-month decision

- **SUCCESS** (→ real-money discussion): geo ≥ **1.8 %/mo** (pre-declared
  survivorship-haircut floor) AND MaxDD ≥ **−45 %** AND slippage ≤
  **15 bp/side** AND vol ≤ **25 %/mo**.
- **INCONCLUSIVE** (→ extend 6 months): geo ∈ **[0, 1.8) %/mo** with a
  documented regime headwind.
- **FAIL:** geo < **0** at 12 months, or any kill fired at any point.

## Amendment rule

This protocol may only be changed by **appending a dated amendment
section** below this line — prior sections are never edited or deleted.
Each amendment must:

1. re-issue `protocol.json` with the change, set `"amended": true`
   (the flag is **permanent** — it is never reset to false), and
2. re-record the new `protocol.json` sha256 alongside the original in the
   amendment section.

Because the state file keeps the hash bound at `forward_test_start`, any
amendment (or silent edit) after the clock starts is detected and the test
is permanently flagged **MODIFIED_AFTER_START** by the protocol scorer.

---
*(No amendments as of 2026-07-14.)*
