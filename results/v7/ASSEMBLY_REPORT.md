# ASSEMBLY — cross-product of V7 family winners (Gate G2)

**Date:** 2026-06-11 · **Family:** `ASSEMBLY` (+ `ASSEMBLY_tier` manual ledger) · **Window:** dev only (≤ 2022-12-31)
**Scripts:** `assemble_candidates.py` (12 pre-registered candidates), `assemble_added_cell.py` (1 added cell, pre-registered below)
**Ledgers:** `results/v7/trials/ASSEMBLY.csv` (16 run_trial rows), `results/v7/trials/ASSEMBLY_tier.csv` (27 manual tier rows). Grid CSVs: `results/v7/ASSEMBLY_grid_results.csv`, `ASSEMBLY_added_cell.csv`, `ASSEMBLY_G2_verdicts.csv`. Finalists: `results/v7/candidates.json`.

## Verdict: Gate G2 **FAIL for all candidates** — but the assembly works: best candidate Calmar **1.334** (1.56× champion's 0.858) at **5.01%/mo**

- **Every one of the 13 scored candidates fails exactly the MaxDD ≥ −40% leg** (the same leg that failed G1 in E1–E4). The binding episode is no longer covid (tamed to −20/−25% by tier+freeze) but the **Sep–Dec 2018 slide** (−44.7% on the best candidate; its full MaxDD −46.9% spans Sep 2018 peak → Dec 2018 trough, wider than the 2018Q4 episode window).
- The added book-gate cell proves the corner is unreachable from this component set: the only remaining DD-cutter gets DD to −41.7% but drops mean to 4.28% (< 4.5%) — the efficient frontier of this component set passes below the (mean ≥ 4.5%/mo, DD ≥ −40%) corner.
- **Finalists for the one-shot validation phase (ranked, all G2-conditional):**

| | mean_mo | Sharpe | MaxDD | Calmar | turn | gross | covid dd | failed G2 legs |
|---|---|---|---|---|---|---|---|---|
| **1. `ASM_Bst_TF1`** (E3 resid base · static ⅓ · TIER+FREEZE) | **5.013%** | **1.350** | −46.9% | **1.334** | 15.8 | 1.49 | −24.1% | DD only |
| **2. `ASM_Bst_BTF1`** (＋BOOK gate; ADDED cell) | 4.276% | 1.251 | **−41.7%** | 1.234 | 20.8 | 1.41 | −20.4% | mean, DD, Calmar |
| 3. `ASM_Bst_T` (no freeze) | 5.099% | 1.320 | −48.6% | 1.297 | 15.1 | 1.53 | −41.6% | DD, Calmar (by 0.003) |
| champion combo_v2_2x (dev) | 4.555% | 1.044 | −60.2% | 0.858 | 14.1 | 1.62 | −60.2% | — |

## Methodology

- **Candidates:** pre-registered 2×2×3 product — base ∈ {A = champion 3-sleeve (xs+dual+adapt n30), B = E3 winner `e3_resid_ms_raw_replxs` (residual mkt+sector raw n30 replaces xs)} × alloc ∈ {st = static ⅓, sx = E2 winner sharpe_softmax lb126 floor 0.10 τ1.0} × overlay ∈ {TF1 = E5 winner tier+freeze dd_v1, T = E5 runner-up tier-only, F1 = freeze dd_v1 floor}. No degenerate duplicates; all 12 completed without error. Each scored at tc ∈ {5, 10, 20} bp.
- **Rebuild conventions:** A·st = `weights_store/combo_v2_base_1x.parquet`; A·sx = `exp_e2_alloc.build_e2_blend_1x(...)` (byte-identical to the E2 G1 run); B·st = sum/3 of the 3 sleeve parquets on union dates/cols (E3 convention); B·sx = generalized E2 allocator (`softmax_blend_1x` in `assemble_candidates.py`: per-sleeve 1x net returns via engine next_open/5bp on dev-truncated prices, allocation strictly ex-ante, ⅓ fallback pre-burn-in). All ×2.0, expanded daily, freeze gate multiplicative on the decision side.
- **Tier layer:** `exp_e5_overlays.tier_loop` is module-bound to the champion base, so `assemble_candidates.TierEnv` re-instantiates its data section per base frame (held names, close-to-close returns, actual-open gaps, scheduled-rebalance reset days from the candidate's OWN un-gated base ×2 daily frame). Loop body identical (P=−0.08 → −8/−13.33/−18.67% → 60/30/0%, delay 2, phi=0, tc on all traded notional), tc parameterized.
- **Fidelity assertions (all 4 base frames):** stops-off loop reproduces engine `next_close` daily net returns to ≤ 2.8e-17.
- **Anchor reproductions (exact):** `ASM_Ast_base` = baseline.json dev (4.555/1.044/−60.2/0.858); `ASM_Asx_base` = E2 winner (4.608/1.066/−54.0/0.976); `ASM_Bst_base` = E3 winner (4.811/1.125/−58.3/0.962); `ASM_Ast_TF1/T/F1` = E5 rows (1.177/1.128/1.054). The generalized machinery is therefore a verified superset of every family's scoring path.
- F1-only candidates and the 4 un-gated base anchors are logged via `exp_lib.run_trial` (family=ASSEMBLY); tier candidates via the manual `ASSEMBLY_tier` ledger mirroring run_trial's schema (E5 precedent — the tier P&L cannot be expressed as a weight frame).

## Full grid (5 bp, native track) — 12 pre-registered + 4 anchors + 1 added cell

| candidate | mm % | Sharpe | MaxDD % | Calmar | turn | gross | 2018Q4 dd | covid dd | 2022 dd | 2022 ret | tier ev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ASM_Ast_base (=champion) | 4.555 | 1.044 | -60.2 | 0.858 | 14.1 | 1.62 | -50.6 | -60.2 | -43.6 | +8.0 |  |
| ASM_Ast_TF1 (=E5 winner) | 4.568 | 1.242 | -48.0 | 1.177 | 17.0 | 1.47 | -45.8 | -22.2 | -27.0 | +21.0 | 19/0/0 |
| ASM_Ast_T | 4.614 | 1.203 | -49.7 | 1.128 | 16.4 | 1.51 | -47.5 | -42.5 | -35.3 | +15.4 | 19/1/0 |
| ASM_Ast_F1 | 4.655 | 1.141 | -52.1 | 1.054 | 15.2 | 1.56 | -50.0 | -35.4 | -29.7 | +24.9 |  |
| ASM_Asx_base (=E2 winner) | 4.608 | 1.066 | -54.0 | 0.976 | 14.3 | 1.62 | -51.8 | -54.0 | -45.1 | +4.6 |  |
| ASM_Asx_TF1 | 4.295 | 1.160 | -50.8 | 1.009 | 17.2 | 1.47 | -48.6 | -25.6 | -33.1 | +8.3 | 17/1/0 |
| ASM_Asx_T | 4.390 | 1.139 | -52.4 | 0.991 | 16.4 | 1.51 | -50.4 | -42.7 | -41.7 | +3.2 | 19/1/0 |
| ASM_Asx_F1 | 4.615 | 1.126 | -53.4 | 1.009 | 15.3 | 1.56 | -51.3 | -33.8 | -31.1 | +22.0 |  |
| ASM_Bst_base (=E3 winner) | 4.811 | 1.125 | -58.3 | 0.962 | 13.3 | 1.62 | -49.0 | -58.3 | -41.8 | -10.7 |  |
| **ASM_Bst_TF1** | **5.013** | **1.350** | **-46.9** | **1.334** | 15.8 | 1.49 | -44.7 | -24.1 | -29.1 | +1.0 | 15/0/0 |
| ASM_Bst_T | 5.099 | 1.320 | -48.6 | 1.297 | 15.1 | 1.53 | -46.5 | -41.6 | -36.6 | -5.6 | 15/1/0 |
| ASM_Bst_F1 | 4.867 | 1.219 | -50.6 | 1.152 | 14.4 | 1.56 | -48.5 | -33.1 | -29.9 | +2.6 |  |
| ASM_Bsx_base | 4.934 | 1.158 | -51.9 | 1.125 | 13.4 | 1.61 | -50.0 | -51.9 | -41.7 | -12.7 |  |
| ASM_Bsx_TF1 | 4.933 | 1.312 | -49.7 | 1.222 | 15.8 | 1.48 | -47.7 | -23.1 | -28.3 | -2.3 | 15/0/0 |
| ASM_Bsx_T | 5.063 | 1.297 | -51.4 | 1.213 | 15.1 | 1.52 | -49.4 | -40.9 | -36.0 | -7.0 | 17/0/0 |
| ASM_Bsx_F1 | 4.860 | 1.201 | -51.8 | 1.117 | 14.4 | 1.56 | -49.7 | -31.6 | -30.3 | -0.6 |  |
| **ASM_Bst_BTF1** (ADDED) | 4.276 | 1.251 | **-41.7** | 1.234 | 20.8 | 1.41 | **-39.4** | -20.4 | -31.9 | -13.9 | 14/0/0 |

Track: tier-containing rows = tier_loop_cc; others = next_open. 2026Q1 episode columns are NaN (outside dev) — flagged for validation.

## Cost robustness (Calmar at 5/10/20 bp; turnover is tc-invariant)

| candidate | 5bp | 10bp | 20bp | 20bp drop |
|---|---|---|---|---|
| ASM_Bst_TF1 | 1.334 | 1.302 | 1.239 | 7.1% |
| ASM_Bst_T | 1.297 | 1.269 | 1.213 | 6.5% |
| ASM_Bst_BTF1 | 1.234 | 1.187 | 1.097 | 11.1% |
| ASM_Bsx_TF1 | 1.222 | 1.192 | 1.133 | 7.3% |
| ASM_Bst_F1 | 1.152 | 1.125 | 1.074 | 6.8% |
| ASM_Ast_TF1 | 1.177 | 1.145 | 1.083 | 8.0% |

All candidates pass the <25% 20bp-drop leg comfortably (range 6.5–11.1%).

## Gate G2 evaluation (5 bp, native track)

Legs: mean ≥ 4.5%/mo · MaxDD ≥ −40% · Calmar ≥ 1.3 · turnover ≤ 30×/yr · 20bp Calmar drop < 25% · covid episode DD within 5 pp of champion (−60.2%).

| candidate | mean | DD | Calmar | turn | 20bp | covid | **G2** |
|---|---|---|---|---|---|---|---|
| ASM_Bst_TF1 | ✓ 5.01 | ✗ −46.9 | ✓ 1.334 | ✓ | ✓ | ✓ | **FAIL (DD only)** |
| ASM_Bst_T | ✓ 5.10 | ✗ −48.6 | ✗ 1.297 | ✓ | ✓ | ✓ | FAIL |
| ASM_Bst_BTF1 (added) | ✗ 4.28 | ✗ −41.7 | ✗ 1.234 | ✓ | ✓ | ✓ | FAIL |
| ASM_Bsx_TF1 | ✓ 4.93 | ✗ −49.7 | ✗ 1.222 | ✓ | ✓ | ✓ | FAIL |
| ASM_Bsx_T | ✓ 5.06 | ✗ −51.4 | ✗ 1.213 | ✓ | ✓ | ✓ | FAIL |
| (remaining 8) | mixed | ✗ all | ✗ all | ✓ | ✓ | ✓ | FAIL |

Crash-episode discipline: champion dev covid DD −60.2%; worst candidate covid DD −42.7% — **all pass by wide margins**. 2026Q1 cannot be evaluated on dev (locked window); must be applied in the validation phase.

## Added cells (protocol clause 2 — rationale was written BEFORE running; 1 of max 3 used)

All 12 pre-registered candidates completed without error. Every candidate failed exactly one structural G2 leg — MaxDD ≥ −40% — with 2018Q4 binding everywhere (covid tamed to −23/−25%, 2022 to −28/−36%). The component set contains exactly one remaining fixed-param DD-cutter that E5 measured as cutting 2018Q4 DD ~4 pp at ~0.5 pp/mo mean cost: the validated book-drawdown gate (`book_drawdown_gate(W2_daily, close, 0.12, 0.30, 60)`; E5 `B+T+F1`: MaxDD −48.0 → −43.8%). The best candidate `ASM_Bst_TF1` had 0.5 pp of mean slack.

**ADDED CELL: `ASM_Bst_BTF1`** = B·st base, BOOK + TIER + FREEZE dd_v1, tc 5/10/20 (E5 composition conventions; gate computed once on the un-gated base daily frame). **Result:** 2018Q4 DD −39.4% (target hit) but MaxDD −41.7% (full Sep–Dec 2018 span) and mean 4.28% — fails mean AND DD. Conclusion: the (4.5%, −40%) corner is outside this component set's frontier; no further cells added.

## Neighborhood stability (single-component toggles)

- `ASM_Bst_TF1` (1.334): neighbors Ast_TF1 1.177, Bsx_TF1 1.222, Bst_T 1.297, Bst_F1 1.152 → median 1.200 = **0.90× (PASS ≥ 0.85)**.
- `ASM_Bst_T` (1.297): neighbors Ast_T 1.128, Bsx_T 1.213, Bst_TF1 1.334, Bst_base 0.962 → median 1.170 = **0.90× (PASS)**.
- `ASM_Bst_BTF1` (1.234): available neighbors book-off 1.334, base-A equivalent (E5 `B+T+F1`) 1.044 → median 1.189 = **0.96× (PASS on available evidence; 2 neighbors)**.

## Interaction findings

1. **The E3 residual base improves every overlay stack** (+0.12…+0.16 Calmar over the same stack on the champion base) — its 2022 sleeve behaviour triggers fewer/cleaner tier events (15 vs 19) and it enters crashes with less idiosyncratic-beta overlap.
2. **The E2 sharpe-softmax allocation is NEGATIVE in combination with the overlays**, despite being positive standalone (base Calmar 0.976/1.125 vs 0.858/0.962): every sx overlay cell is below its st counterpart (e.g. Bsx_TF1 1.222 vs Bst_TF1 1.334; Asx_TF1 1.009 vs Ast_TF1 1.177). Same redundancy pattern E5 found for vol-management: the softmax de-allocates in exactly the episodes tier/freeze already handle, and its de-allocation in recoveries costs mean (A-base mean 4.57 → 4.30). **Drop dynamic allocation from the final stack.**
3. Overlay ordering is preserved on every base: TF1 > T > F1 on Calmar; freeze's marginal is covid/2022 DD (−41.6 → −24.1% covid on Bst), tier's marginal is everything fast.
4. Tier3 never fires (0/27 runs); tier2 at most once. The tier layer's entire value is tier1 (−8% day → 60% book).

## Anomalies / caveats

- No grid-cell errors; 12/12 pre-registered candidates scored at 3 cost levels; 1 added cell (pre-registered above).
- Tier-track numbers are close-to-close phi=0 daily-bar approximations (mildly optimistic intraday; measured cc↔next_open wedge in E5 was ≤ 0.085 Calmar, far below Bst_TF1's +0.48 edge over the champion). **Re-run validation/stop_grid.py phi-stress before any promotion.**
- `ASM_Bst_TF1`'s MaxDD (−46.9%) is a Sep→Dec 2018 peak-to-trough that the 2018Q4 episode window (−44.7%) understates; neither freeze (engages late on the slow slide) nor tier (only scales to 60%) fully defuses it. Cutting it further demonstrably costs the mean leg (added cell).
- The B·sx sleeve-return inputs for the generalized allocator are engine inputs (not ledger-logged), identical in kind to E2's published procedure.
- 43 new scored trials this assignment (16 ASSEMBLY + 27 ASSEMBLY_tier); program-wide ledger total 434 trials (deflated-Sharpe input).

## Recommendation to the validation phase

Carry **`ASM_Bst_TF1`** (primary; G2-conditional on the DD leg) and **`ASM_Bst_BTF1`** (low-DD alternative) forward; `ASM_Bst_T` only as a simplicity fallback. If the program insists on MaxDD ≥ −40% at ≥ 4.5%/mo, this component set cannot deliver it — the honest options are (a) accept ~−45% dev MaxDD for 5%/mo at Calmar 1.33, or (b) accept ~4.3%/mo for −41.7%.
