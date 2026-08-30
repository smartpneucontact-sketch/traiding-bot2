# LEV1 PRE-REGISTRATION

Family: lev1. sha256 of this file at freeze: recorded in the committing
commit message (drafted 2026-08-30, FROZEN 2026-08-30 — the DRAFT_ prefix
was removed, this header dated, and the sha256 recorded at commit; the
do-not-run guard in exp_lev_recal.py may now be removed and its `log=`
flipped on, per the freeze terms).
Amendments only as dated appended sections. No cells beyond the grid below
without a dated amendment committed BEFORE the added cells run.

Attestation: computed nothing before freeze — as of this draft no cell
governed by this document has been rendered; every number cited below is a
pre-existing ledgered value cited by exact file + row identifier.

## Purpose — a MEASUREMENT, not a decision

Locate the leverage that maximizes the champion's NET geometric return at
MEASURED execution costs. The ledgered record shows the ordering flipping as
tc rises (results/v7/trials/tc_recal.csv: at tc=50 full-window, cagr is
0.4419 at 2x / 0.3984 at 1.5x / 0.2983 at 1x — 2x still leads; whether that
survives finer leverage steps and the +52.4bp/side clean vs-open calibration
is exactly what this table measures). Precedent class: correction-exception
measurement of an existing frozen construction at fixed pre-declared settings
(exp_rescore / exp_catalog_v2 / exp_pit_survivorship / exp_tc_recal;
N1_PREREG.md "Fixed settings" style). NO pass bar, NO promotion decision:
nothing live or research-side changes on the basis of this table; any
leverage change to any live protocol requires its own future prereg.

## Fixed settings (no grids beyond the cells listed, no searching)

- Construction: champion combo_v2 base = exp_pit_survivorship.
  build_baseline_base (cached sleeves, blend /3; cache hits only under an
  unchanged exp_lib._CODE_HASH); frame = base x leverage.
- Engine: exp_lib.run_trial front door, exec_model=next_open,
  leverage_cap=2.0 (identical to every ledgered champion row — comparability
  over per-cell caps), margin_bps_annual=600.
- Margin convention: engine_v2's leg exactly as exp_rescore600.py applied it
  on the run_trial track (exp_rescore600.py champion context row:
  run_trial(..., margin_bps_annual=600)) — i.e. engine_v2.run_backtest_v2
  charges (long_gross − 1.0)_+ x 600bp/252 daily, so ONLY the levered
  portion above 1.0x NAV pays financing and the 1.0x cell pays zero
  automatically. tc=52 is the clean vs-open point estimate (+52.4 bp/side
  notional-weighted, 2026-07-27, 58 scheduled at-the-open fills —
  results/v7/RECAL_REPORT.md / exp_tc_recal.py TCS comment); tc=30 is the
  interim vs-arrival calibration.

## Cells (leverage in {1.0, 1.25, 1.5, 1.75, 2.0} x tc in {30, 52})

Windows: full + dev, both ledgered per rendered config; the 2019+ subwindow
is a DERIVED slice of the full-window equity (engine day-causal;
exp_tc_recal.py sliced_summary/sliced_turnover convention) — reported in the
results table, never a separate trial. Full-window runs go through
window="full", final=True under the correction-exception measurement
precedent above.

Already-ledgered cells are CITED, never re-run and never re-ledgered
(ledger-vs-memory discipline; also avoids duplicate-config inflation of the
deduped DSR trial count):

- lev 1.0 / 1.5 / 2.0 at tc=30: results/v7/trials/tc_recal.csv
  name=combo_v2_1x_tc30, combo_v2_1.5x_tc30, combo_v2_2x_tc30
  (window=full and window=dev — six rows).

Newly rendered under family lev1 (7 configs x 2 windows = 14 ledgered rows):

- lev 1.25 and 1.75 at tc=30 (the two interpolation points);
- lev 1.0, 1.25, 1.5, 1.75, 2.0 at tc=52 (no tc=52 row exists anywhere in
  results/v7/trials/ — verified 2026-08-30).

Cell names: combo_v2_{lev}x_tc{tc} (matching the tc_recal naming);
params_json keys: lev, tc_bps, n_long, sleeves, margin_bps_annual, source.

## Fidelity gate (hard-asserted before any lev1 row is written)

tc=5 log=False full-window re-render of the champion frame (base x 2.0) must
reproduce results/v7/trials/pit.csv name=combo_v2_baseline_2x window=full at
machine precision (|dSharpe| < 1e-6, |dCAGR| < 1e-8 — the exp_tc_recal.py
champion gate, verbatim).

## Declared output (frozen before computation)

1. Descriptive table: per (leverage, tc, window in {full, dev, 2019on}) —
   net geo_monthly, mean_monthly, Sharpe, MaxDD, Calmar, turnover_ann,
   avg_gross, with a `source` column citing the ledger file + row (or
   "derived slice of <row>") for every entry. Written to
   results/v7/improve3/lev_recal.csv.
2. ONE pre-declared summary statistic: the leverage in the grid that
   maximizes FULL-WINDOW net geo_monthly at tc=52 (ties broken toward the
   LOWER leverage). The 2019on argmax at tc=52 is printed as labeled context,
   not a second statistic. No verdict, threshold, or consequence attaches to
   either number.

Reading discipline (N1 style): no other reading, no averaging, no
window-shopping. The tc=30 column and dev rows are context and robustness,
never the headline. All rows remain survivors-only UPPER bounds
(RESULTS_CURRENT.md standing caveat).

## Ledger (family lev1 — NEW family file, schema declared)

results/v7/trials/lev1.csv, append-only, margin-family schema (margin_bps in
the header from row 1 per exp_lib.run_trial's new-family mandate for
margin-on runs):

    ts,family,name,window,exec_model,tc_bps,leverage_cap,params_json,
    mean_monthly,median_monthly,sharpe,sortino,max_drawdown,calmar,cagr,
    ann_vol,worst_month,hit_rate,turnover_ann,avg_gross,runtime_s,notes,
    margin_bps,ep_2018Q4_ret,ep_2018Q4_dd,ep_covid_ret,ep_covid_dd,
    ep_2022_ret,ep_2022_dd,ep_2026Q1_ret,ep_2026Q1_dd

(one header line in the file; wrapped here for readability). New columns
require a NEW family file. All ledgered cells count toward the deduped
deflated-Sharpe trial count.

## Explicitly out of scope

Leverage values outside the five listed; dynamic/conditional leverage; any
other construction (WF process, candidate X — each would need its own
measurement doc); any change to leverage_cap, margin rate, or execution
model; any promotion or protocol restatement. The 12-month live paper test
and its bars are untouched by this measurement.
