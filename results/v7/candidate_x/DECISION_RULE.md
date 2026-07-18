# CANDIDATE X SELECTION RULE — frozen 2026-07-18

sha256 of this file is recorded in the committing git commit; any later edit
is an amendment and must be a dated section appended below, never an in-place
change.

**Honesty preamble.** This rule is authored AFTER the 2026-06-12 one-shot
validation; the author has seen the burned validation results. That
contamination is irreducible. Mitigations are structural: (1) every threshold
below is anchored to a commitment that predates the validation run, cited by
document and date; (2) the rule selects on DEV ledger rows only; (3) burned
validation numbers appear in this file exclusively under "disclosure, not
criterion"; (4) the pre-registered shadow forward test — not this rule — is
the arbiter. The rule's job is to pick something defensible to TEST, not to
certify an edge.

## The rule

```
Menu: ledgered DEV rows only, families ASSEMBLY / ASSEMBLY_tier / E5_overlays
/ E5_overlays_tier, constructions {A-base, B-base} x {static 1/3} x
{T, TF1, TF2, BT, BTF1, BTF2}, tc=5bp native track. sx (dynamic allocation)
constructions excluded per the ASSEMBLY interaction finding (pre-validation,
2026-06-11).

R1 Eligibility (all required):
  E1: dev mean_monthly >= 4.0%/mo        [G1 floor, in force since E1-E5]
  E2: dev MaxDD >= -45.0%                [live kill K1 = -45%, protocol.json
                                          committed 2026-07-14; deploying a
                                          candidate whose own backtest
                                          breaches the live kill bar that
                                          will judge it is incoherent]
  E3: dev turnover <= 30x/yr             [G2 leg, ASSEMBLY 2026-06-11]
R2 Primary: maximize dev Calmar among eligible rows.
R3 A-base preference: if an eligible A-base construction is within 0.05 dev
   Calmar of the top eligible B-base construction, select the A-base one
   [E3 standalone falsification, E3 report 2026-06-11; parsimony].
R4 Tie-breakers: shallower dev MaxDD, then fewer overlay components.
R5 Freeze clause: if the selection carries freeze dd_v1/v2 it deploys as a
   DISPUTED component (see adjudication below), never as a validated one.
R6 Margin re-score: before bundle build, re-score the selected construction
   dev-only at margin_bps_annual=600 (family ASM_rescore600, ledgered; the
   ASSEMBLY/E5 rows predate the margin convention). backtest_reference uses
   ONLY margin600 numbers. If margin600 dev Calmar < 1.0 or dev MaxDD < -45%,
   fall to the next-ranked eligible row and repeat R6. No other outcome of
   the re-score alters the selection (no peeking-and-switching).
R7 One selection. No re-runs of this rule against the same menu.
```

## Application (all numbers verified against ledger files 2026-07-18)

| row | source | dev mean/mo | dev Calmar | dev MaxDD | turnover | verdict |
|---|---|---|---|---|---|---|
| ASM_Ast_base | trials/ASSEMBLY.csv | 4.55% | 0.858 | −60.2% | 14.1 | E2 fail |
| ASM_Ast_F1 | trials/ASSEMBLY.csv | 4.66% | 1.054 | −52.1% | 15.2 | E2 fail |
| ASM_Bst_base | trials/ASSEMBLY.csv | 4.81% | 0.962 | −58.3% | 13.3 | E2 fail |
| ASM_Bst_F1 | trials/ASSEMBLY.csv | 4.87% | 1.152 | −50.6% | 14.4 | E2 fail |
| ASM_Ast_TF1 | trials/ASSEMBLY_tier.csv | 4.57% | 1.177 | −48.0% | 17.0 | E2 fail |
| ASM_Ast_T | trials/ASSEMBLY_tier.csv | 4.61% | 1.128 | −49.7% | 16.4 | E2 fail |
| ASM_Bst_TF1 | trials/ASSEMBLY_tier.csv | 5.01% | 1.334 | −46.9% | 15.8 | E2 fail |
| ASM_Bst_T | trials/ASSEMBLY_tier.csv | 5.10% | 1.297 | −48.6% | 15.1 | E2 fail |
| E5_B+T+F1 (A-base) | E5_grid_results.csv | 3.85% | 1.044 | −43.8% | — | E1 fail |
| E5_B+T+F2 (A-base) | E5_grid_results.csv | 3.35% | 0.904 | −42.6% | — | E1 fail |
| **ASM_Bst_BTF1** | trials/ASSEMBLY_tier.csv | **4.28%** | **1.234** | **−41.7%** | **20.8** | **ELIGIBLE — selected** |

R3 never triggers: no A-base construction is eligible. The E3 tension is
resolved by eligibility, not by outscoring. Episode DDs of the selection
(ASSEMBLY_added_cell.csv): 2018Q4 −39.4%, covid −20.4%, 2022 −31.9%;
tier events 14×tier1, 0×tier2/3.

**Honest expectation band (tier phi-stress, FINAL_VALIDATION_REPORT.md Step 4,
full window):** phi=0: 4.93%/mo, Calmar 1.484, MaxDD −41.7% · phi=0.5: 4.47%,
1.319, −41.7% · phi=1.0: 3.77%, 1.070, −41.8%. Dev tier-track numbers are
close-to-close phi=0 approximations; live is intraday — the phi=0.5/1.0 rows
are the honest expected band, and the shadow slot must not be read as
"failing" the phi=0 number.

## G4 retirement finding

G4's absolute bars (mean >= 5.0%/mo; full-window MaxDD >= −40%, −45% if
Calmar >= 1.5) were calibrated before the 2026-07-12 audit corrections. The
corrected champion itself — combo_v2_2x at next_open with 6%/yr margin:
4.80%/mo arithmetic, MaxDD −60.3% (results/v7/catalog_v2.csv) — fails both
bars. A gate the incumbent cannot pass cannot be the bar for replacing the
incumbent. G4 is hereby RETIRED as a promotion gate with this written
finding; no candidate will again be scored against it. Its replacement is
the relative shadow protocol (protocol_exp.json). This is the
anti-gate-shopping move: the old gate is retired in writing, not quietly
re-leveled until something passes.

## Burned validation — disclosure, not criterion

ASM_Bst_BTF1 one-shot validation (2026-06-12, window burned): val mean
6.15%/mo, val MaxDD −28.1%, val Calmar 3.22, G3 PASS. The G3 champion anchor
was full-window-selected (contamination disclosed in
FINAL_VALIDATION_REPORT.md). These numbers played no role in the rule above
and are never re-used as selection criteria.

## E3 tension — disclosed

A B-base construction wins although E3 falsified residual momentum as a
STANDALONE improvement (family failed G1; crashes as hard as raw momentum).
The B-base edge rests solely on the assembly interaction finding (+0.12–0.16
Calmar on every overlay stack), dev-only, small evidence. The shadow forward
test carries the burden of proof.

## Freeze adjudication

freeze_dd_v1's parameters carry tainted provenance: they originate from the
retracted V6 in-sample full-window 135-cell grid; the RESCORE_NOTES stability
flag was 83.3% (< the 85% bar); the dev-window freeze-parameter grid that
was prescribed before assembly was never run. dd_v2 shares the same
provenance and the ledgers contradict the premise that it dominates
(E5_B+T+F2 Calmar 0.904 vs F1's 1.044). Swapping to an off-menu
book+tier-only variant now would be an unledgered new construction — worse
methodology than deploying the disputed ledgered one.

RULING: Candidate X v1 ships as ASM_Bst_BTF1 exactly as ledgered, freeze
dd_v1 included, labeled DISPUTED in the bundle, this file, and
protocol_exp.json. Its adjudication is the pre-registered FREEZE ABLATION
REPLAY: the bot journals daily weights and freeze state; research replays
the identical book monthly with the freeze multiplier removed. At the
12-month review, if the freeze's live contribution is negative (costs return
without cutting realized MaxDD), freeze is disabled in any promoted
configuration. This is a binding config decision, not a slot pass/fail leg.
