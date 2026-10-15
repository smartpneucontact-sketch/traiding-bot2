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
