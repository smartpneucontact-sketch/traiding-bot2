# N1 TRUTH-RUN PRE-REGISTRATION — frozen 2026-07-29, BEFORE data access exists

sha256 of this file recorded in the committing commit. Amendments only as
dated appended sections. This prereg is committed before the Norgate
subscription is even active — no computation it governs is possible yet,
which is the strongest pre-registration this program can offer.

## Purpose

Measure — not re-tune — the true expected return of the program's existing
frozen constructions on survivorship-free data (Norgate US Platinum:
delisted securities + point-in-time index constituents). The current honest
bracket at 2x spans [~0.75, ~3.8] %/mo geo because the local cache contains
zero delisted names and no membership history beyond the Wikipedia S&P 500
reconstruction; this run collapses that bracket to a measurement.

## Fixed settings (no grids, no searching)

- Engine: engine_v2 next_open; margin_bps_annual = 600.
- tc_bps in {5, 30, 50}: 5 = legacy reference (comparability with every
  prior ledger row), 30 = interim vs-arrival calibration (91 fills,
  2026-07-25), 50 = clean vs-open calibration (58 scheduled at-the-open
  fills, 2026-07-27, +52.4bp/side notional-weighted). All three rendered;
  the 50 column is the headline until a post-limit-execution measurement
  supersedes it (which would be its own dated amendment).
- Universes (two arms, both PIT via eligible= selection-time masks; engine
  prices = full Norgate panel so held names mark and die correctly):
  (a) PIT Russell 1000 as-of-date; (b) PIT S&P 1500 as-of-date.
- Delisting policy: positions in names reaching last_quoted_date exit at
  last quoted close (engine opt-in hook). Sensitivity arm: bankruptcy-class
  exits (terminal price < 20% of 252d-ago price) additionally haircut -30%
  on the exit fill. Both arms rendered; the no-haircut arm is primary, the
  haircut arm bounds the residual optimism (Norgate lacks CRSP delisting
  returns; series usually die visibly through OTC continuation).
- Family: norgate_truth. Every run ledgered, full+dev windows, full-window
  runs final=True under the correction-exception precedent (fixed
  pre-declared settings, measurement of existing constructions, not a
  search).

## Constructions measured (frozen as they exist at this commit)

1. Champion combo_v2 (xs30 + dual30 + adapt30 equal blend) at 1.0/1.5/2.0x.
2. The WF process rule's historical curve: spec-v2 selection rule (annual
   Dec-31 top-3-by-Sharpe from the 21-name pool, equal blend) re-scored on
   the honest panel — same rule, new data; documented as a spec-v3 data
   annotation, rule text byte-identical to spec v2.
3. Candidate X (ASM_Bst_BTF1: residual+dual+adapt static 1/3, book gate +
   tier + freeze-dd_v1-DISPUTED) via the exp_rescore600 tier machinery.
4. Crash-episode re-measurement (2018Q4, covid, 2022) for all of the above.

## Pre-committed verification gates (run before any truth number is read)

- V-harness: the tc=5 arm re-run on the OLD cache must reproduce existing
  ledger rows at machine precision (pit.csv combo_v2_baseline_2x;
  wf_results.csv 2019on; ASSEMBLY_tier ASM_Bst_BTF1) — proves the harness
  is unchanged before the data swap.
- V-data: all five N0 trust gates pass (overlap return-diff median <1bp
  with dividend verification; two-source S&P 500 membership cross-check,
  daily symmetric diff <=3 explained, 10 KNOWN_EVENTS, COUNT_BAND;
  delisting spot checks SIVB/FRC/TWTR~$54.20/BBBY~0/ATVI~$95/CERN;
  panel-integrity scan; sidecar sha256s == VM manifest).

## Pre-committed interpretation (the N2 fork — decided NOW, before numbers)

Reading the PRIMARY cells (PIT Russell 1000 arm, no-haircut, tc=50,
full window, 2x):

- Champion AND process rule both >= 2.5%/mo geo  -> VALIDATED: define
  live-capital criteria; proceed to N3 re-derivation.
- Either in [1.5, 2.5)%/mo                        -> RECALIBRATED: the
  program goal restates to the measured level; N3 optimizes from truth.
- Both < 1.5%/mo                                  -> STRATEGY-FAMILY
  STOP-LOSS: no further optimization of this family; N3 pivots to new
  research on the trusted substrate. The live paper fleet continues
  regardless (its protocols are self-contained and already pay real costs).

No other reading, averaging, or arm-shopping. The S&P 1500 arm, dev
windows, tc=5/30 columns, and haircut arm are context and robustness,
never the verdict.

## Explicitly out of scope for N1

Any parameter change, any new sleeve, any universe optimization, any gate
retuning — all of that is Phase N3, each under its own future prereg.
