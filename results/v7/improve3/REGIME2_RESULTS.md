# REGIME2 RESULTS — corrected relative DSR recomputation

Computed 2026-08-31 per REGIME2_PREREG.md (frozen 2026-08-30; sha in freeze commit). Pure recomputation — no engine run, no ledger row.

Inputs at evaluation time: n_days_dev=1701, n_trials=772 (deduped), sr_variance_daily=1.909317e-04 (originals: n=707/715, 2.00e-04 per RESULTS_CURRENT.md).

Anchor combo_v2_2x dev (results/v7/trials/catalog_v2.csv): sharpe 0.963057, calmar 0.748359, geo 3.1533%/mo, corrected DSR 0.754449 (original 0.740223 at n=707).

| cell | ledgered sharpe | corrected DSR | vs anchor | calmar leg (>=1.10x anchor) | geo leg (>= anchor - 0.2pp) | verdict |
|---|---|---|---|---|---|---|
| vixterm_thr1.05_m0.5 | 0.994560 | 0.779426 (orig 0.766000) | PASS | 0.8492 vs 0.8232 -> PASS | 3.0633% vs 2.9533% -> PASS | PASS |
| breadth_ad63_p10_m0.5 | 0.988219 | 0.774519 (orig 0.763061) | PASS | 0.8293 vs 0.8232 -> PASS | 3.0242% vs 2.9533% -> PASS | PASS |

## Verdict (per the frozen bar)

PASS: vixterm_thr1.05_m0.5, breadth_ad63_p10_m0.5. Per the frozen consequence structure this changes NOTHING live and enters no construction. Exactly one cell is val-eligible (higher corrected DSR): **vixterm_thr1.05_m0.5**.

The one-shot validation is NOT authorized by REGIME2 — it requires a future REGIME3 pre-registration with val gates frozen before any val computation (REGIME2_PREREG.md 'Consequences'). The determinism disclosure in the prereg anticipated this dev-side outcome; the genuine uncertainty lives entirely in that future val shot.
