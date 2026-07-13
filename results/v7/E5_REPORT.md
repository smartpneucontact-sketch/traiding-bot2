# E5 — Overlay recombination on the corrected champion (combo_v2_base_1x × 2.0)

**Date:** 2026-06-11 · **Family:** `E5_overlays` (+ `E5_overlays_tier` manual ledger) · **Window:** dev only (≤ 2022-12-31)
**Engine:** engine_v2, exec=next_open, 5bp, leverage_cap=2.0 (tier combos: adapted stop_grid close-to-close loop, see Methodology) · **Script:** `exp_e5_overlays.py`
**Ledgers:** `results/v7/trials/E5_overlays.csv` (12 run_trial rows), `results/v7/trials/E5_overlays_tier.csv` (24 manual rows: 12 tiered + 12 untiered-loop comparators). Grid CSV: `results/v7/E5_grid_results.csv`.

## Verdict: G1 **PASS** — advance 2 configs

| | mean_mo | Sharpe | MaxDD | Calmar | turn | gross | track |
|---|---|---|---|---|---|---|---|
| baseline combo_v2_2x (dev) | 4.55% | 1.044 | −60.2% | 0.858 | 14.1 | 1.62 | next_open |
| **WINNER `E5_T+F1` (TIER + FREEZE dd_v1)** | **4.57%** | **1.242** | **−48.0%** | **1.177** | 17.0 | 1.47 | tier-cc |
| **RUNNER-UP `E5_T` (TIER only)** | 4.61% | 1.203 | −49.7% | 1.128 | 16.4 | 1.51 | tier-cc |
| best pure-next_open: `E5_F1` | 4.66% | 1.141 | −52.1% | 1.054 | 15.2 | 1.56 | next_open |

- `E5_T+F1`: Calmar 1.177 (= 1.37× champion, ≥ 0.944 ✓), mean 4.57% ≥ 4.0% ✓, MaxDD −48.0% > −50% ✓, neighborhood stability 0.887–0.891 ≥ 0.85 ✓. **Full G1 pass.**
- `E5_T`: Calmar 1.128 ✓, mean 4.61% ✓, MaxDD −49.7% > −50% ✓ (by only 0.3 pp), stability **convention-dependent**: 0.818 under the strict 4-neighbor convention (FAIL), 0.869 when both freeze-parameter variants count as neighbors (PASS). Advanced as runner-up with the stability flag.
- Best stack answer: **TIER + FREEZE(dd_v1), no BOOK, no VOLMGMT.**

## Methodology

- Base frame: `weights_store/combo_v2_base_1x.parquet` × 2.0, expanded daily on the dev union-price index. Overlays compose multiplicatively on the decision-side daily frame; the engine's internal shift gives every overlay the correct 1-day execution lag. Engine clip/cap (2.0×) applied as in the baseline; the cap binds only on the same days it binds for the champion (base gross max 1.167 × 2).
- **BOOK**: `book_drawdown_gate(W2_daily, close, 0.12, 0.30, 60)`, computed once on the base frame — the gate is invariant to per-day scalar multipliers (weighted-average construction), so its value is identical in every stack.
- **FREEZE**: `freeze_signal_spy_drawdown` dd_v1 (21, 0.12, 0.08, 10) and dd_v2 (30, 0.10, 0.05, 10) → daily 0/1 multiplier (58 / 151 frozen dev days). Both parameterizations run for every freeze-inclusive combo (8 v2 alternates → 24 total cells; no new grid).
- **VOLMGMT** (E1 winner, fixed): pass-1 `run_backtest_v2(base×2, dev)` → `vol_managed_multiplier(returns, 0.30, lookback=21, cap=1.0, floor=0.3)`. Multiplier built from DEV returns only.
- **TIER**: portfolio tiered stop adapted from `validation/stop_grid.py` `simulate()` (P=−0.08 ⇒ tiers −8%/−13.33%/−18.67% → 60%/30%/liquidate, re-entry delay 2 trading days, phi=0), generalized to daily-varying corrected frames: book held during day d = clip&cap(W)·shift(1); tier scale persists until the next BASE scheduled rebalance; tier3 → cash 2 days, re-enter at close; 5bp on all traded notional (rebalances, tier sells, re-entries); open-gap breaches checked against actual opens.
- **Fidelity assertions (both pass):** (1) tier loop with stops off reproduces the engine's `next_close` daily net returns to 2.8e-17; (2) anchor cells reproduce prior records exactly — `E5_none` = baseline.json dev (4.555%/1.044/−60.2%/0.858), `E5_F1` = rescore dd_v1 dev (4.66%/1.141/−52.1%/1.054), `E5_V` = E1 best cell (2.77%/1.069/−35.4%/0.932).
- **CAVEAT — tier results are daily-bar approximations**: (a) the loop is close-to-close (next_close-like), not next_open; the measured wedge on the 12 untiered comparator frames is small (untiered-cc Calmar minus next_open Calmar ∈ [−0.050, +0.085], median +0.022); (b) phi=0 checks breaches only at the open gap and the close — it misses intraday dips that recover by the close, so tier numbers are mildly OPTIMISTIC (stop_grid's phi=0.5/1.0 stress would degrade them; not re-run here, params were fixed by assignment). Tier combos must be re-validated with the stress before any live promotion.
- Tier-inclusive combos are scored in the manual ledger; their gated-but-untiered frames are exactly the corresponding no-T combos already scored via `run_trial` (e.g. the untiered comparator of `E5_B+T+F1` is `E5_B+F1`) — no duplicate runs were logged, the mapping is 1:1 by construction. Additionally each tier combo's frame was re-run through the loop with stops off (`*_loop_untiered` rows) so the TIER marginal is measured under the identical approximation.

## Full grid (24 cells, dev) — sorted by Calmar

| combo | track | mean_mo | Sharpe | MaxDD | Calmar | turn | gross | 2018Q4 dd | covid dd | 2022 ret | 2022 dd | tier ev (1/2/3) | untiered-cc Calmar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E5_T+F1 | tier-cc | 4.57% | 1.242 | -48.0% | 1.177 | 17.0 | 1.47 | -45.8% | -22.2% | +21.0% | -27.0% | 19/0/0 | 1.088 |
| E5_T | tier-cc | 4.61% | 1.203 | -49.7% | 1.128 | 16.4 | 1.51 | -47.5% | -42.5% | +15.4% | -35.3% | 19/1/0 | 0.842 |
| E5_F1 | next_open | 4.66% | 1.141 | -52.1% | 1.054 | 15.2 | 1.56 | -50.0% | -35.4% | +24.9% | -29.7% |  |  |
| E5_B+T+F1 | tier-cc | 3.85% | 1.128 | -43.8% | 1.044 | 21.6 | 1.41 | -41.4% | -19.0% | -0.5% | -26.5% | 18/0/0 | 0.976 |
| E5_T+F2 | tier-cc | 3.74% | 1.092 | -43.8% | 1.023 | 17.6 | 1.39 | -40.0% | -21.1% | -6.7% | -30.6% | 17/0/0 | 0.880 |
| E5_B+T | tier-cc | 3.79% | 1.082 | -44.4% | 0.980 | 22.1 | 1.43 | -41.9% | -34.8% | +0.2% | -31.1% | 19/0/0 | 0.902 |
| E5_V | next_open | 2.77% | 1.069 | -35.4% | 0.932 | 16.7 | 1.10 | -32.9% | -30.4% | -1.0% | -25.0% |  |  |
| E5_F1+V | next_open | 2.73% | 1.085 | -35.0% | 0.927 | 16.9 | 1.08 | -32.4% | -21.1% | +4.2% | -17.9% |  |  |
| E5_T+F1+V | tier-cc | 2.70% | 1.084 | -35.4% | 0.909 | 17.0 | 1.08 | -32.9% | -16.7% | +1.2% | -18.1% | 1/0/0 | 0.926 |
| E5_B+T+F2 | tier-cc | 3.35% | 1.026 | -42.6% | 0.904 | 21.7 | 1.34 | -38.8% | -18.6% | -13.0% | -29.3% | 16/0/0 | 0.725 |
| E5_B+F1 | next_open | 3.85% | 1.028 | -48.2% | 0.891 | 21.1 | 1.48 | -46.0% | -26.3% | -4.1% | -29.6% |  |  |
| E5_F2+V | next_open | 2.43% | 0.998 | -32.2% | 0.881 | 17.3 | 1.04 | -28.3% | -19.7% | -8.5% | -20.3% |  |  |
| E5_F2 | next_open | 3.75% | 0.995 | -48.3% | 0.878 | 16.2 | 1.47 | -43.9% | -31.4% | -10.5% | -32.1% |  |  |
| E5_B+T+F1+V | tier-cc | 2.39% | 1.009 | -32.0% | 0.869 | 19.0 | 1.04 | -29.4% | -14.1% | -8.2% | -21.8% | 1/0/0 | 0.887 |
| E5_T+V | tier-cc | 2.68% | 1.042 | -36.6% | 0.866 | 16.8 | 1.10 | -34.2% | -30.1% | -2.6% | -25.1% | 1/0/0 | 0.882 |
| E5_none | next_open | 4.55% | 1.044 | -60.2% | 0.858 | 14.1 | 1.62 | -50.6% | -60.2% | +8.0% | -43.6% |  |  |
| E5_B | next_open | 3.83% | 0.992 | -48.3% | 0.856 | 21.6 | 1.51 | -46.1% | -39.2% | -6.2% | -36.5% |  |  |
| E5_B+F1+V | next_open | 2.40% | 1.003 | -32.6% | 0.852 | 18.9 | 1.04 | -29.9% | -16.2% | -7.5% | -20.9% |  |  |
| E5_T+F2+V | tier-cc | 2.39% | 0.989 | -32.7% | 0.849 | 17.4 | 1.03 | -29.3% | -16.3% | -9.5% | -20.8% | 1/0/0 | 0.867 |
| E5_B+V | next_open | 2.40% | 0.991 | -32.7% | 0.844 | 19.2 | 1.05 | -30.0% | -21.8% | -8.1% | -23.8% |  |  |
| E5_B+T+V | tier-cc | 2.36% | 0.981 | -32.3% | 0.835 | 19.3 | 1.05 | -29.7% | -21.6% | -8.5% | -23.4% | 1/0/0 | 0.853 |
| E5_B+T+F2+V | tier-cc | 2.18% | 0.939 | -31.3% | 0.792 | 19.2 | 1.00 | -27.8% | -13.9% | -13.5% | -20.7% | 1/0/0 | 0.811 |
| E5_B+F2+V | next_open | 2.18% | 0.933 | -32.1% | 0.772 | 19.1 | 1.00 | -28.2% | -15.7% | -14.1% | -20.6% |  |  |
| E5_B+F2 | next_open | 3.24% | 0.913 | -52.6% | 0.658 | 21.5 | 1.41 | -43.3% | -24.8% | -20.8% | -33.3% |  |  |

## Marginal contribution of each overlay (dd_v1 lattice, 8 with/without pairs each)

ΔCalmar from ADDING the overlay (pairs share the same scoring track, so each Δ is internally consistent):

| overlay | mean ΔCalmar | median | range | Δmean_mo (mean) | pattern |
|---|---|---|---|---|---|
| **TIER** | **+0.074** | +0.070 | −0.066 … +0.270 | −0.04 pp | Strongly positive on every NON-V stack (+0.12 … +0.27); slightly negative (−0.01 … −0.07) on V-stacks — once vol scaling tames the book, the −8% day threshold almost never trips (19 tier events → 1) and the residual trips only cost. Mean return is essentially FREE (Δmm ≈ 0). |
| **FREEZE dd_v1** | **+0.053** | +0.039 | −0.005 … +0.196 | +0.02 pp | Positive in 7/8 pairs, near-free in mean. Biggest standalone (+0.196); ≈ 0 on top of V (−0.005): freeze and vol-scaling de-risk the same episodes. dd_v2 is uniformly worse (best v2 cell 1.023 vs 1.177). |
| **BOOK** | **−0.085** | −0.082 | −0.163 … −0.002 | −0.55 pp | Negative in ALL 8 pairs. The 12→30 book-DD gate cuts some DD but costs more mean (−0.3 … −0.8 pp/mo) and adds ~5 turns/yr. **Drop from the stack.** |
| **VOLMGMT** | **−0.119** | −0.136 | −0.268 … +0.074 | −1.66 pp | Positive ONLY standalone (+0.074); negative in every combination, and catastrophic for mean (−1.4 … −1.9 pp/mo, ≈ −40%). Its one virtue: lowest absolute DDs (−31 … −35%) for stacks that can accept ~2.2–2.7%/mo. |

## Freeze vs vol-management — the explicit answer

**No — continuous vol scaling does NOT replace the binary SPY freeze; on this book it is the freeze that makes vol scaling redundant, not vice versa.** Evidence:

1. Head-to-head: F1 alone Calmar 1.054 at 4.66%/mo vs V alone 0.932 at 2.77%/mo. The freeze achieves a comparable Calmar improvement while keeping ~100% of the champion's mean; vol scaling buys its DD cut with a ~40% mean haircut (E1's finding, confirmed in combination).
2. Redundancy is one-directional: adding F1 on top of V changes nothing (0.932 → 0.927, covid DD −30.4% → −21.1% is the only gain); adding V on top of F1 destroys the config (1.054 → 0.927, mean 4.66% → 2.73%). They de-risk the same crash episodes; V additionally de-levers all the calm periods, which is pure cost.
3. In the winning stack the answer is the same: T+F1 (1.177, 4.57%/mo) vs T+V (0.866, 2.68%/mo) vs T+F1+V (0.909, 2.70%/mo). V's presence also disarms the tier layer (19 → 1 events), absorbing its benefit at far higher mean cost.
4. The two mechanisms split the risk cleanly instead: TIER handles fast crashes (covid DD −60% → −22% in T+F1; 2022 +8% → +21%), FREEZE handles sustained index drawdowns; the remaining −48% MaxDD of the winner sits in 2018Q4 (−45.8%), a fast pre-freeze-threshold slide where neither binary layer fully engages — that, not vol, is the residual risk for the assembly phase.

## Winners — exact configs

**1. `E5_T+F1`** — params: `{"book": false, "tier": true, "freeze": "dd_v1", "volmgmt": false}` with fixed sub-params tier `{pstop_eff: -0.08, tiers: [-8%, -13.33%, -18.67%], fractions: [0.6, 0.3, 0.0], reentry_delay: 2, phi: 0}`, freeze dd_v1 `(peak_lookback=21, freeze_dd_pct=0.12, unfreeze_within_pct=0.08, min_freeze_days=10)`.
dev: mean 4.568%/mo, Sharpe 1.242, MaxDD −47.998%, Calmar 1.177, turnover 17.0×/yr, avg gross 1.47. Episodes: 2018Q4 −45.8%, covid −22.2%, 2022 +21.0% / −27.0%. Tier events: 19 tier1, 0 tier2, 0 tier3.
Rebuild (dev; for other windows re-instantiate `exp_e5_overlays`'s data section on the desired index):

```python
import sys; sys.path.insert(0, '.')
import pandas as pd
from engine_v2 import expand_daily, apply_gate
from exp_lib import load_cache, union_prices_cached
from strategies import freeze_signal_spy_drawdown

panel, macro, _ = load_cache(); pu = union_prices_cached()
idx = pu.loc[:'2022-12-31'].index                      # dev
base = pd.read_parquet('weights_store/combo_v2_base_1x.parquet').loc[:'2022-12-31']
W = expand_daily(base * 2.0, idx)
f1 = 1.0 - freeze_signal_spy_drawdown(macro, peak_lookback=21, freeze_dd_pct=0.12,
        unfreeze_within_pct=0.08, min_freeze_days=10).astype(float)
W = apply_gate(W, f1)                                  # frozen days -> 0
# tier layer (manual P&L, daily-bar approx):
from exp_e5_overlays import tier_loop                  # dev-bound module globals
net, events, traded, gross = tier_loop(W, P=-0.08, delay=2)
```

**2. `E5_T`** — params: `{"book": false, "tier": true, "freeze": null, "volmgmt": false}`, same tier sub-params.
dev: mean 4.614%/mo, Sharpe 1.203, MaxDD −49.69%, Calmar 1.128, turnover 16.4×/yr, avg gross 1.51. Episodes: 2018Q4 −47.5%, covid −42.5%, 2022 +15.4% / −35.3%. Tier events: 19/1/0.

### Neighborhood-stability evidence (±1 lattice step = toggle one overlay; freeze-param v1↔v2 also a neighbor)
- `E5_T+F1` (1.177): neighbors B+T+F1 1.044, F1 1.054, T 1.128, T+F1+V 0.909 → median 1.049, ratio **0.891 PASS**; adding T+F2 (1.023) as 5th neighbor → median 1.044, ratio **0.887 PASS**. Freeze-param ratio T+F2/T+F1 = **0.869** — above the 0.85 bar (the marginal 0.833 ratio seen at the single-overlay level improves inside the stack).
- `E5_T` (1.128): neighbors B+T 0.980, none 0.858, T+F1 1.177, T+V 0.866 → median 0.923, ratio **0.818 FAIL** (strict); with both freeze variants {+T+F2 1.023} → median 0.980, ratio **0.869 PASS**. Flagged convention-dependent.

## Anomalies / notes
- No grid-cell errors; 24/24 pre-registered cells completed; no cells added.
- The headline winner numbers are on the tier-cc track (close-to-close, phi=0 daily-bar approximation, mildly optimistic intraday). The measured cc↔next_open Calmar wedge on this family's frames is ≤ 0.085 (median +0.022), far smaller than the winner's +0.32 Calmar edge over the baseline, so track mixing does not drive the verdict — but assembly/validation must re-run the tier layer with stop_grid's phi stress before promotion.
- `E5_T`'s MaxDD passes the −50% gate by only 0.31 pp; under phi>0 stress it would likely breach. `E5_T+F1` has 2 pp of slack.
- BOOK is negative in every single pairing — the validated 12→30 gate is fully dominated on this book once tier/freeze exist; recommend retiring it from the assembly stack.
- Tier3 never fires in any dev combo (0 events); tier2 fires once (E5_T, covid). The tier layer's value is almost entirely tier1 (−8% day → 60% book) during covid and 2022.
- 36 new trials logged this family (12 `E5_overlays` + 24 `E5_overlays_tier`, of which 12 are untiered-loop comparators).
