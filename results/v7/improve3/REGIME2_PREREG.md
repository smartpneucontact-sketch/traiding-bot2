# REGIME2 PRE-REGISTRATION

Family: regime2 (NEW family name per the REPORT.md re-test mandate; the
failed families regime_vixterm and regime_breadth STAY FAILED — their dev
verdicts are not reopened). sha256 of this file at freeze: recorded in the
committing commit message (drafted 2026-08-30, FROZEN 2026-08-30 — the
DRAFT_ prefix was removed, this header dated, sha256 recorded at commit).
The corrected-bar choice (relative-to-incumbent, per the G4-retirement
principle) was reviewed and RATIFIED at freeze, including the determinism
disclosure that the dev-side verdict is decidable in advance — this family
knowingly freezes a reclassification plus the gated REGIME3 path, not a
dev-side test.
Amendments only as dated appended sections.

Attestation: computed nothing before freeze — the corrected-DSR
recomputation defined below has not been executed; every number cited below
is a pre-existing ledgered/published value cited by exact file + row
identifier. (Determinism disclosure below: for THIS document that
attestation is necessary but weaker than usual, because the recomputation's
inputs are all already ledgered.)

## Purpose

Re-evaluate the two Stream-2 overlay near-misses under a CORRECTED
deflated-Sharpe bar, per the recorded gate-design finding (REPORT.md,
"Improvement phase v2" Outcomes, Stream-2 bullet): "the DSR >=0.95 leg is
structurally unreachable — the ungated champion itself scores dev DSR ~0.74
at 723 deduped trials. Two near-misses (vixterm thr1.05/m0.5; breadth
ad63_p10/m0.5) pass Calmar+geo and cut MaxDD 9-11 pp, failing only that
leg. ... any re-test requires a NEW pre-registered family with a corrected
DSR bar — the failed families stay failed." This document is that new
pre-registered family.

## Cells re-evaluated (exactly these two; no new grid, no new cells)

1. vixterm_thr1.05_m0.5 — results/v7/trials/regime_vixterm.csv,
   name=vixterm_thr1.05_m0.5 window=dev (ts 2026-07-18T13:47:24; ledgered
   sharpe 0.994560, calmar 0.849208, max_drawdown -0.513780; params
   threshold=1.05, mult=0.5, margin 600). Original evaluation:
   results/v7/ws4_vixterm_results.csv (dsr_dev 0.766000 at n_trials=707;
   gate_calmar PASS, gate_geo PASS, gate_dsr FAIL).
2. breadth_ad63_p10_m0.5 — results/v7/trials/regime_breadth.csv,
   name=breadth_ad63_p10_m0.5 window=dev (ts 2026-07-18T13:48:30; ledgered
   sharpe 0.988219, calmar 0.829332, max_drawdown -0.518235; params
   feature=ad63, pct_threshold=0.1, mult=0.5, margin 600). Original
   evaluation: results/v7/ws4_breadth_results.csv (dsr_dev 0.763061 at
   n_trials=715; gate_calmar PASS, gate_geo PASS, gate_dsr FAIL).

Anti-cell-shopping note: RESULTS_CURRENT.md's Stream-2 table lists
breadth_above200_p10_m0.5 as the breadth family's best-by-Calmar cell; it is
NOT a near-miss (results/v7/ws4_breadth_results.csv row
breadth_above200_p10_m0.5: gate_geo=False) and is NOT re-evaluated here.
Only the two cells REPORT.md names — the only cells in either family passing
both non-DSR legs — are in scope.

## Verified repo facts (checked in-file 2026-08-30, before this draft)

1. metrics_v2.deflated_sharpe DOES contain a degenerate fallback: when
   sr_variance_across_trials is None/non-finite/<=0 it substitutes
   v = max(daily_SR^2, 1e-8) (metrics_v2.py, deflated_sharpe body), which
   makes the expected-max-SR benchmark ~3x the observed SR at ~500 trials
   and drives DSR toward 0 regardless of merit (the failure mode documented
   in exp_catalog_v2.ledger_sr_variance_daily's docstring).
2. HOWEVER, the vixterm/breadth evaluations did NOT hit that fallback. Both
   scripts passed the measured cross-trial variance:
   exp_ws4_vixterm.dsr_inputs() returns (trial_count(dedupe=True),
   exp_catalog_v2.ledger_sr_variance_daily()) and feeds it to
   deflated_sharpe (exp_ws4_vixterm.py; exp_ws4_breadth.py reuses
   dsr_inputs). The ledgered dsr_dev values above are therefore ALREADY the
   corrected-estimator numbers (at n=707 / 715 respectively). The premise
   "the original near-miss DSRs were computed under the degenerate fallback"
   is FALSE; the recorded defect is the absolute 0.95 BAR
   (ws4_vixterm_verdicts.json and ws4_breadth_verdicts.json, gates.dsr_min
   = 0.95; pre-committed in both scripts' docstrings), not the estimator.
3. Trial daily-return series are NOT stored for these families (the trials
   CSVs carry summary scalars only; no equity/returns artifact exists for
   any regime_* cell). NO re-render is needed anyway: the original DSR used
   only (ledgered annualized Sharpe, dev-window day count, n_trials,
   sr_variance) with skew=0 and excess_kurtosis=0 defaults — every input is
   recoverable from stored ledgers plus a day count of the union-price dev
   index. This re-evaluation is a PURE RECOMPUTATION: no engine execution,
   no run_trial call, no log=False re-render required under the
   correction-exception precedent, and NO row appended to any trials ledger.
4. Champion anchor: the ungated champion dev row is
   results/v7/trials/catalog_v2.csv name=combo_v2_2x window=dev (sharpe
   0.963057), reproduced at 0 error by the ledgered anchor cell
   results/v7/trials/regime_vixterm.csv name=vixterm_base_2x (sharpe
   0.963057, dsr_dev 0.740223 at n=707).

## Corrected DSR procedure (declared; the ONLY computation this doc governs)

For each cell above, and for the champion anchor, recompute:

    dsr = metrics_v2.deflated_sharpe(
        observed_sr_annual = the ledgered sharpe cited above (re-read from
                             the trials CSV at run time, never re-typed),
        n_obs_daily        = n_days_dev = number of union-price index rows
                             <= DEV_END (2022-12-31) — the identical
                             expression exp_ws4_vixterm.py used,
                             recorded in the output,
        n_trials           = exp_lib.trial_count(dedupe=True) at evaluation
                             time (the FULL cross-family deduped count),
        sr_variance_across_trials = exp_catalog_v2.ledger_sr_variance_daily()
                             (per-family dedupe by the trial_count key
                             columns, keep="last"),
        skew = 0.0, excess_kurtosis = 0.0  (the original defaults, kept for
                             exact comparability with the original leg's
                             semantics; return series are not stored, and
                             this prereg declares they are NOT needed))

n_trials and sr_variance at evaluation time are recorded in the output
artifact (results/v7/improve3/REGIME2_RESULTS.md) next to the originals
(707/715; 2.00e-04 daily^2 per RESULTS_CURRENT.md "Render-time DSR").

## The corrected bar (the substance of this prereg)

Original bar, cited: dev DSR >= 0.95 at the current deduped trial count
(exp_ws4_vixterm.py / exp_ws4_breadth.py docstrings, "PRE-COMMITTED GATE";
ws4_*_verdicts.json gates.dsr_min = 0.95). Retaining it is ruled out by the
verified facts: the estimator was already corrected, so recomputation can
only move DSR DOWN from 0.766/0.763 (n_trials has grown past 723 and
deflation is monotone in n_trials) — a 0.95 bar would pre-register a
foregone FAIL and re-enact the recorded miscalibration.

CORRECTED BAR (declared now, per the REPORT.md mandate and the G4-retirement
principle — "a gate the incumbent cannot pass cannot be the bar",
REPORT.md Improvement-phase-v2 decision 1):

    PASS iff corrected DSR(cell) >= corrected DSR(champion anchor),
    both computed by the procedure above at the same evaluation instant,
    with the two non-DSR legs re-affirmed unchanged from the ledgered rows
    (calmar >= 1.10 x anchor calmar; geo >= anchor geo − 0.2pp — both
    already-ledgered scalars, cited in "Cells" above; they involve no
    recomputation).

Determinism disclosure (recorded so the freeze is honest): at fixed
(n_days, n_trials, sr_variance), deflated_sharpe is monotone increasing in
the observed Sharpe, and both cells' ledgered dev Sharpes (0.994560,
0.988219) exceed the anchor's (0.963057). The dev-side verdict under the
corrected bar is therefore already decidable from cited ledger rows: both
cells will PASS. What this prereg genuinely freezes is (a) the corrected-bar
PRINCIPLE (relative to the incumbent, not an unreachable absolute), (b) the
scope (these two cells, nothing else, ever, under this family), and (c) the
consequence structure below — which is where the real uncertainty lives.

## Consequences (frozen now)

- Both cells FAIL or either non-DSR leg no longer re-affirms: finding
  closed; the near-misses stay closed permanently; family regime2 never
  ledgers a row.
- PASS: the passing cell(s) do NOT enter any construction and change nothing
  live. Exactly ONE cell becomes val-eligible (CRASH1 discipline: one winner
  per one-shot): the cell with the higher corrected DSR — decidable now by
  the monotonicity above: vixterm_thr1.05_m0.5, disclosed. Its one-shot
  validation is NOT authorized by this document: it requires a subsequent
  pre-registration (REGIME3) whose val gates are frozen BEFORE any val
  computation, and which carries the CRASH1-style program-contamination
  disclosure (the designers know the val window's regime content). The val
  run, if ever taken, is ledgered under family regime2 (new margin-schema
  family file created only then); failing it burns the family permanently.
- Multiple-testing disclosure: this is a second look at cells selected
  partly because they looked good the first time. Bounded teeth: two
  pre-existing cells only, no new search, no re-tuning, at most one future
  val shot, gates frozen in advance at every step.

## Output

results/v7/improve3/REGIME2_RESULTS.md: the recomputed corrected DSRs for
the two cells + anchor, the recorded (n_days_dev, n_trials, sr_variance)
inputs, original-vs-corrected comparison, and the verdict per the bar above.
No CSV ledger is touched.

## Explicitly out of scope

Re-rendering any cell; any new threshold/multiplier/feature; touching the
validation window; reopening regime_vixterm / regime_breadth / cov_voltarget
family verdicts; any change to metrics_v2.deflated_sharpe or to
ledger_sr_variance_daily (the estimator is used as-is, as it was originally).
