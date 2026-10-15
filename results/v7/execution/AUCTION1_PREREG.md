# AUCTION1 PRE-REGISTRATION — opening-auction execution measurement

Family: execution (operational; no strategy change). Frozen 2026-10-15;
sha256 of this file recorded in the committing commit. Amendments only as
dated appended sections.

Attestation: no auction order has been submitted by any slot as of this
freeze. The live code path (combo-v2-bot core/orders.py
submit_auction_orders / finalize_auction_orders; exec_style="auction")
ships DARK — no slot is set to "auction" until this document is committed.

## Why

The validated backtest prices every rebalance fill AT the official open
(engine exec_model next_open). Live, the 09:35 ET run fills at ~09:36-09:43
after limit resolution. On 2026-09-29 the notional-weighted cost of that
delay vs the open was +72 / +119 / +9 bp/side on buys (primary / Candidate
X / process) and +14 / +31 / −36 on sells, while the vs-arrival cost the
limit orders control was only +8 to +21 bp. At the champion's turnover
that drift is worth roughly 0.4 pp/month — the largest execution cost the
backtest never paid. Limit-on-open orders (Alpaca time_in_force "opg",
whole shares, submitted before 09:28 ET) execute in the opening auction,
i.e. at the price the backtest assumes.

## Design

- Slot under measurement: combo_v2_process ONLY (ungated; the simplest
  stack; its own K2-style slippage criterion applies to rebalance rows).
  The primary and Candidate X stay on marketable-limit and serve as the
  same-day control.
- Stage 1 (09:06 ET cron): the identical data/weights path, then every
  planned order submitted as a limit-on-open whole-share order at
  prev_close ± 100 bp (buys above / sells below). Nothing journalled yet.
- Stage 2 (09:35 ET, the normal run): each opg order's final state is
  read; one journal row per order with exec_style "auction_loo", fill
  price, and slippage vs the official open; any still-open opg order is
  cancelled; the normal rebalance then completes shortfalls (sub-share
  residues, unfilled orders, gaps beyond the limit) at marketable-limit.

## Pre-declared metrics (per rebalance day, from journal rows)

1. vs-open cost, notional-weighted, bp/side, by side — auction slot
   (auction_loo rows + its marketable-limit completion rows, combined and
   separately) vs the control slots (marketable_limit rows), same day.
2. Auction fill rate (filled notional / submitted notional) and the share
   of the slot's total rebalance notional executed in the auction.
3. Residual completion cost: vs-open cost of the completion rows alone.

## Pre-declared decision rule (frozen now)

Measured over the auction slot's next THREE scheduled rebalances
(expected ≈ 2026-10-28, 2026-11-25, 2026-12-23):

- ADOPT on all slots if, on at least 2 of the 3 days, the auction slot's
  combined vs-open cost is at least 15 bp/side lower than the control
  slots' notional-weighted cost AND the auction fill rate is ≥ 85% of
  submitted notional.
- REVERT the auction slot to marketable-limit if the fill rate is < 70%
  on 2 of 3 days, or if the combined cost is higher than the controls on
  2 of 3 days.
- Otherwise: extend the measurement by three more rebalances (one
  extension only), then decide by the same rule.

## Disclosures

- Paper-trading realism: Alpaca's paper venue simulates opg fills; if
  every auction order fills at exactly the official open with a 100% fill
  rate, the measurement may overstate what live money would get. That
  outcome is recorded as "paper-simulation caveat" and the ADOPT decision
  still stands for the paper program, with the caveat carried into any
  real-money discussion.
- Overnight-gap asymmetry: buys above the +100 bp limit and sells below
  the −100 bp limit do not fill at the open and complete later at
  marketable-limit — the measurement includes those completions.
- This is an operational execution change of the same class as the
  2026-07-25 marketable-limit flip (disclosed in OPS_LOG.md, not a
  protocol amendment). The process slot's PK2 slippage criterion is
  measured on rebalance rows, which include auction rows.

## Amendment 2026-10-15 (appended before any auction order; no metric or decision-rule change)

The implementation reviews that followed the freeze (three adversarial
rounds, same day, before enablement) changed two mechanics described in
the Design section. Recorded here so the frozen text and the shipped code
agree. The pre-declared metrics and the adopt/revert rule are unchanged.

1. Sell side: market-on-open, not limit-on-open at −100 bp. A sell that
   misses the open leaves the exit to the completion leg at a later,
   typically worse, price; a market-on-open order executes at the official
   open the backtest assumes. Buys stay limit-on-open at prev_close
   +100 bp (an unprotected market buy could pay any overnight gap).
   Consequence for metric 2: the auction fill rate is effectively a
   buy-side statistic; sells fill at the open in full.
2. Fill-rate price basis: "filled notional / submitted notional" is
   measured at the anchor price (shares executed × previous official
   close over shares submitted × previous official close), so a gap-down
   morning is never read as an unfilled residue. The completion leg's
   plan-based rate (both legs, re-bought residues counted once) is the
   number the slot's protocol fill criterion sees.
3. Expected measurement days, corrected for the bot's weekday-count
   cadence and market holidays: 2026-10-28, 2026-11-27 (the cadence
   falls due on Thanksgiving; the market-closed skip moves it to the
   Friday half-session), 2026-12-28 (falls due on Christmas; moves to
   the Monday). The decision report is due the session after the third
   day. Enablement is by versioned config migration on combo_v2_process
   only, recorded in OPS_LOG.md with the enabling commit.

## Amendment 2026-10-15 (second, appended after enablement, before any auction order)

Measurement mechanics clarified after the fourth implementation review;
metrics and the decision rule are unchanged.

4. Reference open for metric 1: per row, today's official open from the
   bot's own daily panel when the panel's last bar is dated today; else
   the Alpaca IEX snapshot's daily-bar open only when that bar is dated
   today (a stale pre-open bar is rejected); none for an earlier
   session's order. The source mix is recorded per day in the slot state
   (`reference_sources`). The IEX first print can differ from the
   consolidated opening-auction print; the decision report therefore
   re-prices every row against the official open from the next day's
   daily panel (by symbol and date) and reports both figures. The control
   slots' marketable-limit rows are re-priced the same way.
5. Rows carry the submitted whole-share quantity (`qty_submitted`) and
   the anchor-priced submitted notional (`target_weight_usd`), so metric 2
   is rebuilt from rows as shares executed / shares submitted at the
   previous official close even after partial fills; the slot state's
   daily summary accumulates across finalize passes.
6. Operational guards that can turn a measurement day into a non-auction
   day are recorded in the slot state and the dashboard: a stale panel at
   09:06 (`skipped:stale_panel`), a manual run holding the pipeline at
   09:06 (`auction_last_run_status = skipped`), or a stage error. Such a
   day does not count toward the three measurement days; the next
   scheduled rebalance takes its place.
