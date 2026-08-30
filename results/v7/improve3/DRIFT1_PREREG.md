# DRIFT1 PRE-REGISTRATION

Family: drift1. sha256 of this file at freeze: recorded in the committing
commit message (drafted 2026-08-30, FROZEN 2026-08-30 — the DRAFT_ prefix
was removed, this header dated, and the sha256 recorded at commit; the
do-not-run guard in exp_drift_band.py may now be removed and its `log=`
flipped on, per the freeze terms).
Amendments only as dated appended sections. No cells beyond the grid below
without a dated amendment committed BEFORE the added cells run.

Attestation: computed nothing before freeze — as of this draft no drift-band
transform has been executed against any engine, no cell governed by this
document has been rendered, and every number cited below is a pre-existing
ledgered value cited by exact file + row identifier.

## Hypothesis

Trading only weight drift beyond a band cuts turnover materially with a small
gross-return give-up, so NET returns at the MEASURED execution costs improve
or hold. Motivation from the ledgered record: the champion's dev-window cost
drag from tc=5 to tc=50 is ~0.5pp/mo geo at its ledgered both-sides turnover
(results/v7/trials/tc_recal.csv name=combo_v2_2x_tc50 window=dev,
turnover_ann=14.14; drag arithmetic per results/v7/RECAL_REPORT.md
"Arithmetic sanity check"). A band that removes the smallest rebalance
tickets attacks that drag directly. Costs used are the measured calibrations
(30 bp/side interim vs-arrival, 50 bp/side clean vs-open — RECAL_REPORT.md
"The measurement that forced this"), not the legacy 5 bp assumption.

## Construction (frozen as it exists at this commit)

Champion combo_v2 at 2.0x: base = exp_pit_survivorship.build_baseline_base
(cached sleeves sleeve_xs_momentum_n30 + sleeve_dual_momentum_voltarget_n30 +
sleeve_adaptive_voltarget_n30, blend /3 — cache hits only under an unchanged
exp_lib._CODE_HASH), frame = base x 2.0. Engine path identical to
exp_tc_recal.py: exp_lib.run_trial, exec_model=next_open, leverage_cap=2.0,
margin_bps_annual=600.

## The drift-band transform (declared exactly; implemented in exp_drift_band.py)

Iterative pass over the SPARSE decision-date frame (post-leverage weights):
hold a current-weights vector; on each decision date, for each name, trade
(set held := target) only if |w_target − w_held| > band; untraded names keep
their held weight. Band units: ABSOLUTE weight points of NAV on the
post-leverage frame (0.25% = 0.0025). The transformed sparse frame feeds
run_trial exactly as the baseline frame does (weights_are_daily=False; the
engine daily-expands it by ffill).

Drift-approximation disclosure (declared, not discovered later): the
transform holds weights PIECEWISE-CONSTANT between trade decisions. This is
byte-consistent with the engine's own convention — engine_v2.run_backtest_v2
ffills the decision frame and charges turnover only on decision-frame changes
(engine_v2.py, next_open branch), i.e. the ledgered record models an
implicit, costless daily re-target and NO price drift anywhere. The band is
therefore applied at decision dates on held-vs-target; the real-world
price-drift of held weights between rebalances is unmodeled in BOTH control
and treatment (zero new approximation relative to the record), and this
slightly understates the drift a live book experiences. Gross may drift off
2.0x when stale weights are held; the engine's per-day leverage cap
(_clip_and_cap, cap 2.0) handles it — the same mechanism every ledgered
champion row already rides.

Truncation note: the transform runs on the full-history frame; run_trial's
dev truncation (weights.loc[:DEV_END]) takes a prefix, and the iteration is
strictly forward — the dev slice is unaffected by post-DEV_END rows.

## Grid (dev phase — 8 cells: 4 bands x 2 tc; margin 600; window="dev")

- band in {0 (control), 0.0025, 0.005, 0.01}
- tc_bps in {30, 50}

Control/fidelity discipline (breadth-anchor precedent — exp_ws4_breadth.py
declined to re-ledger an already-proven anchor): the two band=0 cells are
FIDELITY GATES, rendered log=False and never ledgered:

1. Frame identity: drift_band_transform(W, 0.0) must equal the input frame
   exactly (bitwise; strict inequality |diff| > 0 leaves every held weight
   equal to target).
2. Metric identity: the band=0 log=False dev runs must reproduce the
   already-ledgered rows at machine precision (|dSharpe| < 1e-6,
   |dCAGR| < 1e-8 — exp_tc_recal.py gate tolerances):
   - tc=30: results/v7/trials/tc_recal.csv name=combo_v2_2x_tc30 window=dev
   - tc=50: results/v7/trials/tc_recal.csv name=combo_v2_2x_tc50 window=dev
   Those two ledger rows ARE the band=0 control values for the pass bar.

The six banded cells (3 bands x 2 tc) are ledgered, family drift1, ONLY
after this document is frozen (log=False until then; a pre-freeze render is
a protocol breach and voids the family).

## Declared metrics and pass bar (frozen before any banded cell runs)

- PRIMARY metric: dev net geo_monthly = (1+cagr)^(1/12) − 1 from the
  ledgered cagr (program headline convention, RESULTS_CURRENT.md
  "Reporting configuration").
- GUARDRAIL: dev MaxDD of a banded cell must not be worse than the band=0
  control at the same tc by more than 2pp (max_drawdown >= control − 0.02),
  at BOTH tc levels, or the band is disqualified regardless of geo.
- REPORTED (no bar): both-sides turnover_ann reduction vs control, and the
  drag decomposition per the RECAL_REPORT arithmetic.

PASS bar: some band b strictly beats the band=0 control on dev net
geo_monthly at BOTH tc=30 AND tc=50, and clears the guardrail at both tc
levels. Winner selector if multiple bands qualify (declared now): the
qualifying band with the highest dev net geo at tc=50 (the headline cost
column per N1_PREREG.md fixed settings); tie → the SMALLER band. Exactly ONE
winner exists or the family closes.

## One-shot validation (winner only; gates frozen now)

The single winning band is rendered ONCE per tc level with window="val",
final=True, family drift1 (ledgered). Comparator: band=0 control val stats
derived from a log=False full-window re-render of the ledgered
tc_recal configs (results/v7/trials/tc_recal.csv combo_v2_2x_tc30 /
combo_v2_2x_tc50 window=full; fidelity-asserted against those rows before
the val slice is read) — the control takes no new ledgered trial.

- VERDICT reads at tc=50 ONLY (tc=30 val is context, never the verdict —
  no column-shopping, N1 discipline).
- val PASS: winner val net geo_monthly >= control val net geo_monthly at
  tc=50 AND winner val MaxDD >= control val MaxDD − 2pp.
- Pass: the band becomes an execution-layer candidate for the live bundle —
  any live change still goes through the bot repo's own operational process
  (bundle version, protocol sha binding); this prereg promotes nothing live.
- Fail val: family drift1 BURNED. No second shot, no band re-tuning.
- Fail dev: no val shot; family closed.

Honesty disclosure (CRASH1 template): the family has never touched the
validation window, so it is procedurally entitled to the one-shot; but the
BASE construction's validation behavior is known program-wide and the
designers know it. Disclosed; cannot be removed. Teeth: gates above are
frozen before any banded cell renders; one winner; failure burns the family.

## Ledger (family drift1 — NEW family file, schema declared)

results/v7/trials/drift1.csv, append-only, margin-family schema (created
with margin_bps in the header from row 1 per exp_lib.run_trial's mandate
that margin-on runs use a NEW family):

    ts,family,name,window,exec_model,tc_bps,leverage_cap,params_json,
    mean_monthly,median_monthly,sharpe,sortino,max_drawdown,calmar,cagr,
    ann_vol,worst_month,hit_rate,turnover_ann,avg_gross,runtime_s,notes,
    margin_bps,ep_2018Q4_ret,ep_2018Q4_dd,ep_covid_ret,ep_covid_dd,
    ep_2022_ret,ep_2022_dd,ep_2026Q1_ret,ep_2026Q1_dd

(one header line in the file; wrapped here for readability). New columns
require a NEW family file — this schema is frozen with the document.
Cell names: drift_b{band}_tc{tc} (e.g. drift_b0.005_tc50); params_json keys:
band, lev, tc_bps, n_long, sleeves, margin_bps_annual, source.
All ledgered cells count toward the program's deduped deflated-Sharpe trial
count (exp_lib.trial_count, dedupe=True).

## Explicitly out of scope

Any band outside the four listed; any construction other than champion
combo_v2 2x; any tc level other than 30/50; any calendar/turnover-budget
rebalancing variant; leverage changes (see LEV1_PREREG.md — separate
family). Each would need its own prereg.
