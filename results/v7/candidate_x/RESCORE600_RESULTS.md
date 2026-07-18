# ASM_rescore600 — R6 margin re-score results (2026-07-18)

Executes R6 of DECISION_RULE.md (sha256 `fe482f8ee65d674d04061019d3cb227f143e385685fb90e8fff11dcc50e3e6ec`), dev-only, family
`ASM_rescore600`, ledger `results/v7/trials/ASM_rescore600.csv`.
Margin convention on the tier track: engine_v2 leg replicated post-loop,
`drag = max(gross_held-1,0) * 600bp/252` daily (see exp_rescore600.py
docstring; conservative on intraday tier-cut days, identical for all rows
on this track including the CRASH1 cells).

| row | track | margin | dev mean/mo | dev MaxDD | dev Calmar | turnover | avg_gross |
|---|---|---|---|---|---|---|---|
| ASM_Bst_BTF1 (ledgered ref) | tier_loop_cc | 0 | 4.276% | -41.7% | 1.234 | 20.8 | 1.414 |
| ASM_Bst_BTF1 (repro, this run) | tier_loop_cc | 0 | 4.276% | -41.7% | 1.234 | 20.8 | 1.414 |
| **ASM_Bst_BTF1 (selection)** | tier_loop_cc | **600** | **3.964%** | **-42.1%** | **1.101** | 20.8 | 1.414 |
| combo_v2_2x (champion context) | next_open | 600 | 4.168% | -60.3% | 0.748 | 14.1 | 1.619 |

Margin-0 reproduction asserted against `trials/ASSEMBLY_tier.csv` (tolerance
5e-6 on mean/Calmar/MaxDD; tier events identical). Champion context row
asserted equal to the `catalog_v2` dev margin600 row.

## R6 FALL-THROUGH CHECK: PASS  (Calmar 1.101 >= 1.0; MaxDD -42.1% >= -45%)

- Calmar bar (>= 1.0): 1.101 -> PASS
- MaxDD bar (>= -45%): -42.1% -> PASS

Per R6, no other outcome of this re-score alters the selection.
`backtest_reference` in the bundle must use ONLY the margin600 row above.
The margin600 ASM_Bst_BTF1 row is the frozen CRASH1 comparator
(CRASH1_PREREG.md, "Dev gates").

Episodes (margin600 selection): 2018Q4 -39.6%, covid -20.4%, 2022 -32.7% (max in-window DD).
