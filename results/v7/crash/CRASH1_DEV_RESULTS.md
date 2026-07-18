# CRASH1 dev grid results (2026-07-18)

Prereg: `results/v7/crash/CRASH1_PREREG.md` sha256 `84f4461e84759a02a9525a6b80a5b99efe0001d7e2cb006754e8ae32c5159aaa`.
Exactly the 8 pre-registered cells; family `CRASH1`, dev-only,
tier_loop_cc track, tc=5bp, leverage_cap 2.0, margin 600bp/yr
(post-loop drag, identical convention to the comparator — see
`exp_rescore600.py`). Ledger: `results/v7/trials/CRASH1.csv`.

Base: ASM_Bst_BTF1 construction with the freeze layer REMOVED and the
crash overlay in its place via `engine_v2.apply_gate`. Construction
guard: the freeze-included frame reproduced the ledgered ASM_Bst_BTF1
margin-0 row before any cell ran.

Implementation fixations (chosen before any cell ran, not tuned —
see `exp_crash_overlay.py` docstring): 24mo = 504 trading days;
DM scale = clip(0.3/rv126(pass-1 book-gated base net),0,1),
1.0 outside bear; vol threshold = SPY rv63 > expanding 75th pctile
(min_periods=252); all gates decision-side (one-day engine lag).

Comparator (S1, `results/v7/trials/ASM_rescore600.csv`): ASM_Bst_BTF1
@margin600 — mean 3.964%/mo, MaxDD -42.1%, Calmar 1.101.

## Cells

| cell | bear def | resp | vth | bear days | mean/mo | MaxDD | Calmar | turnover | avg_gross | tier events |
|---|---|---|---|---|---|---|---|---|---|---|
| CRASH1_b24m_bin_v0 | b24m | bin | v0 | 11 | 3.876% | -42.8% | 1.028 | 21.3 | 1.436 | {"tier1": 15, "tier2": 0, "tier3": 0} |
| CRASH1_b24m_bin_v1 | b24m | bin | v1 | 11 | 3.876% | -42.8% | 1.028 | 21.3 | 1.436 | {"tier1": 15, "tier2": 0, "tier3": 0} |
| CRASH1_b24m_dm_v0 | b24m | dm | v0 | 11 | 3.878% | -42.8% | 1.029 | 21.3 | 1.436 | {"tier1": 15, "tier2": 0, "tier3": 0} |
| CRASH1_b24m_dm_v1 | b24m | dm | v1 | 11 | 3.878% | -42.8% | 1.029 | 21.3 | 1.436 | {"tier1": 15, "tier2": 0, "tier3": 0} |
| CRASH1_b200_bin_v0 **(winner)** | b200 | bin | v0 | 331 | 3.532% | -39.1% | 1.033 | 22.0 | 1.316 | {"tier1": 10, "tier2": 0, "tier3": 0} |
| CRASH1_b200_bin_v1 | b200 | bin | v1 | 323 | 3.555% | -39.5% | 1.032 | 21.7 | 1.318 | {"tier1": 10, "tier2": 0, "tier3": 0} |
| CRASH1_b200_dm_v0 | b200 | dm | v0 | 331 | 3.617% | -41.1% | 1.000 | 21.5 | 1.350 | {"tier1": 12, "tier2": 0, "tier3": 0} |
| CRASH1_b200_dm_v1 | b200 | dm | v1 | 323 | 3.624% | -40.8% | 1.010 | 21.4 | 1.351 | {"tier1": 12, "tier2": 0, "tier3": 0} |

## Frozen dev gates (evaluated on the top-Calmar cell only)

Winner: **CRASH1_b200_bin_v0** (dev Calmar 1.033, mean 3.532%/mo, MaxDD -39.1%).

| gate | requirement | value | verdict |
|---|---|---|---|
| G-cal | Calmar >= 1.05 x 1.101 = 1.156 | 1.033 | FAIL |
| G-mm | mean >= 3.964 − 0.5 = 3.464%/mo | 3.532%/mo | PASS |
| G-nbr | median neighbor Calmar >= 0.85 x 1.033 = 0.878 | 1.028 (neighbors: 1.028, 1.000, 1.032) | PASS |

## Verdict: NONE

The top-Calmar cell fails the frozen dev gates (winner-or-none;
no cell shopping). Per the prereg: fail dev -> no val shot;
family CRASH1 closed. The freeze layer remains in candidate X v1
as DISPUTED under its pre-registered ablation adjudication.

## Observation (appended post-run 2026-07-18; report-only, not a gate)

The b24m bear definition fires on only 11 dev days, so the four b24m cells
are a near-null overlay — effectively "freeze removed, nothing in its
place": dev Calmar 1.028-1.029 vs the freeze-included comparator's 1.101
(same margin600 tier track; tier1 events 15 vs 14). In-frame, the DISPUTED
freeze dd_v1 layer therefore contributed roughly +0.07 dev Calmar and
+0.09pp/mo here. This is dev-window, single-frame evidence only; it feeds
context to the pre-registered 12-month live freeze-ablation replay
(DECISION_RULE.md "Freeze adjudication"), which remains the binding
adjudicator. No cell of CRASH1 beat the freeze-included comparator's
Calmar; the family closes without a val shot.
