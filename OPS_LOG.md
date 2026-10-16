# Operations log — dated, append-only

Live-behavior changes, incidents, and alignment fixes for the running
paper slots. This file is NOT a protocol amendment (PROTOCOL.md's
amendment rule governs the protocols themselves); it is the operational
disclosure record the periodic checkpoints cite. Newest entries at the
bottom. Entries are never edited after the fact — corrections are new
entries.

---

## 2026-08-30 — Gate-cadence alignment fix (whipsaw fix), retroactive log

**Prior entries (recorded retroactively today for completeness):**
- 2026-07-15: tier-scale base fix — live tiers sold `f × equity` where the
  validated rule (stop_grid TIER_F) sells to `f × day-start book`; fixed
  with day-anchored semantics. (Incident: ~$98k sold where ~$60k was
  validated.)
- 2026-07-20: protocol sha binding repaired — Dockerfile never COPYied the
  protocol files; per-slot binding shipped the same day with git-provable
  shas.
- 2026-07-25: exec_* slot-config passthrough fix — exec_style /
  exec_limit_buffer_bps in slot JSON were silently ignored by the config
  loader; caught before the marketable-limit flip would have no-op'd.
- 2026-08-29: Aug 18-28 silent outage closed — MIN_STOCKS_REQUIRED=500 sat
  at the normal post-filter universe count (497-499); every August
  rebalance aborted for 9 days with no operator-visible alarm. Fixed:
  threshold 400 (env-overridable), consecutive-abort counter, /ready 503
  alarm at >=2.

**Today's change — daily gate re-evaluation (core/gate_update.py):**
The validated backtests apply the drawdown gates DAILY
(validation/book_gate.py prices `base_daily * g_t`); the live bot sampled
the gate only inside compute_weights() at each 21-day rebalance. Between
rebalances the tier scanner could CUT exposure but nothing could RAISE it.
Consequence, measured from live state today: the primary carried
multiplier 0.3372 baked in on 2026-07-20 (book_dd −23.9% that day) and
Candidate X 0.4653 (2026-07-21) through six weeks that included a full
recovery — the validated gate would have re-levered within days. This is
an alignment TO the validated construction (same class as the 07-15 tier
fix), not a strategy change:

- Daily pass on non-rebalance pipeline runs: recompute the identical gate
  stack (same functions/config fields, keyed to the HELD book per
  book_gate.py semantics) and pro-rata scale the book toward the new
  multiplier when it moved >= GATE_UPDATE_BAND (default 0.10), both
  directions. Market orders, tier-layer convention; sells get the full
  instrumentation pass.
- State: `applied_exposure_multiplier` (scanner-owned) written by the
  rebalance path, the daily pass, and the tier scaler (multiplied by the
  fraction kept on tier cuts).
- Ungated configs (process slot) are structurally unaffected; the pass
  no-ops for them. Data-guard-thin days skip the pass without trading.
- Env: GATE_DAILY_UPDATE=0 disables; GATE_UPDATE_BAND tunes the band.

Also shipped in this pass:
- Universe churn monitor (core/universe_monitor.py): daily usable-bars
  count + name churn on the volume, early warning below 450 (above the
  400 hard guard), surfaced in /api/status.
- Spec-manifest invariant checker (core/invariants.py +
  spec_manifest.json): running slot configs, protocol sha bindings, tier
  fractions, and guard thresholds are compared against the committed spec
  at every pipeline start; violations log CRITICAL, persist to the
  volume, show in /api/status, and turn /ready 503. Never halts trading.
- Program calendar (program_calendar.json): checkpoint/verdict/decision
  deadlines surfaced on the dashboard; overdue decisions render red.
- Dashboard: Program Health + Program Calendar tiles; per-slot gate
  posture in /api/status; /api/invariants endpoint.

Disclosure: the daily gate pass changes live behavior of running
forward tests toward their validated reference. It will be disclosed in
the 3-month checkpoint documents alongside the Aug 18-28 outage.

## 2026-08-31 — Adversarial audit of the 08-30 pass: 1 critical + 10 majors fixed

An 81-agent adversarial audit (8 review lenses, 2-refuter verification per
finding, completeness critic) of Saturday's Tier A-D code confirmed 43
findings (many duplicates across lenses). Unique defects fixed today,
BEFORE the first live daily gate pass (Tue 09:35):

1. CRITICAL — daily gate pass dead on arrival: get_positions() returns
   {symbol: {...}} with no 'symbol' key in the values; the pass raised
   KeyError on every real book, swallowed it, and returned without
   trading — the whipsaw fix was inert. Masked by test mocks that
   injected the key. Fixed (symbol injected from dict key; mocks now
   mirror the real shape); dot-form Alpaca symbols (BRK.B) also mapped
   back to the universe's dash-form for the book-DD lookup.
2. Tier anchor inflated after a buying-power-capped re-lever (stale
   pre-cap delta) — could disable the soft stop for the day. Anchors and
   multiplier commits now derive from SUBMITTED notional only; zero
   placed orders commit nothing.
3. Lost-update race: the pass could overwrite a concurrent tier cut's
   multiplier write. Final write is now compositional (fresh disk value ×
   achieved factor) with a pre-trade abort on stop-trip/state-change.
4. Tier cuts are now STICKY until the next rebalance
   (tier_floor_multiplier; stop_grid semantics) — a recovered gate no
   longer re-buys a tier stop's risk reduction.
5. Applied multiplier now records the REALIZED rebalance scaling
   (gross_final/gross_pre_gate, includes the gross cap + dust filter) and
   the daily pass re-applies the cap via gate_ref — effective leverage
   can no longer exceed design when the cap binds.
6. Raw gate ~0 now liquidates fully and clears last_rebalance (validated
   g=0 + re-entry through the normal rebalance the day the gate
   re-opens); the history bootstrap (which ignored interim tier cuts) is
   removed.
7. Gate/tier/cutloss rows are excluded from the protocol-facing slippage
   calibration (K2 reads rebalance execution only).
8. Invariants: fts-without-bound-sha now FAILS (the July half-bound
   state was silently skipped); per-slot gate enablement checked against
   the bundle; checker errors are 'degraded' (no more transient 503s);
   protocol binding computes the sha before setting the clock.
9. Universe monitor: same-day re-runs no longer clobber the day's churn
   baseline; corrupt-file recovery; atomic writes (also state files and
   invariants status — torn-read fix for out-of-process readers).
10. data_guard_aborts writes are locked+targeted (no more clobbering
    scanner state); universe snapshots skipped on partial downloads.
11. Calendar: ET clock (was UTC), overdue covers ALL unresolved past
    events (was decisions only); invariant warnings visible on the tile
    and in /ready's 200 payload.

Tests: 249 passing (40 in tests/test_improvements_ad.py, rewritten with
production-shape mocks). Refuted by verification: 3 findings. Remaining
accepted-risk notes live in the audit result (task w6aw0s0mk).

## 2026-09-26 — September review: gate pass verified live; one glitch class closed

First month of the daily gate cadence (live since 2026-09-01, audit-fixed
build). Verified from the container's gate_update_history (19 sessions):
- Primary: 3 scaling events — re-lever 2026-09-10 (0.792→0.899), de-lever
  2026-09-15 (0.899→0.781), re-lever 2026-09-18 (0.781→0.916); 16 holds
  within the 0.10 band. Book drawdown healed −15.1% → −9.8%; gate target
  reached 1.0 on 2026-09-22.
- Candidate X: 1 re-lever (2026-09-10, 0.809→0.911); target 1.0 from
  2026-09-22, held within band since.
- Process: correctly skipped every day (ungated).
- No tier stops, no portfolio stops, no freeze, no data-guard aborts;
  universe stable at 499 names (one-day dip to 498 on 09-10); invariants
  31/31 PASS every run.
Performance 2026-08-31 → 2026-09-25 (Alpaca equity): primary +13.09%
(maxDD −6.5%), Candidate X +11.14% (−5.6%), process +11.43% (−4.7%);
SPY +0.81%. Since forward-test start: primary +13.9% vs SPY +4.2%,
Candidate X +10.6% vs +3.3%, process +9.4% vs +4.6% (all within their
1-sigma tracking bands).

FINDING + FIX — gate silently opened on a data glitch: on 2026-09-23 the
gate trace shows book_dd=null with target=1.0. Cause: a trailing all-NaN
price row (pre-open partial bar) made _book_drawdown return None, and the
combination rule treated a missing book gate as spy-only = 1.0. Harmless
that day (applied 0.916, within band), but with a deeply gated book it
would have re-levered to 1.0 on bad data. Fixed in core/gate_update.py:
trailing all-NaN rows are dropped before the drawdown read (daily-pass
only; compute_weights untouched), and an uncomputable book drawdown now
SKIPS the pass instead of defaulting to open. Two regression tests.

Also: Candidate X tracking view was blind ("bundle has no
geo_monthly_return reference") since 2026-07-21 — a display bug, fixed by
deriving geo from mean and sigma (labeled expected_geo_derived=true) in
core/tracking.py. Freeze-ablation report #2 written 5 days late (clean
null; research repo). The session-scoped daily /ready probe expired
2026-09-07 (7-day cron cap) — an external uptime monitor is still the
durable answer.

## 2026-10-15 — Oct 7-15 drawdown review: why no stop fired, and four fixes

Drawdown: primary −11.1% close-to-close / −17.7% intraday from the Oct 6
close peak to the Oct 15 09:55 ET low; Candidate X ≈ −16%; process −19.4%
intraday. SPY −1%, IWM −3.5% — a mid-cap momentum crash, not plumbing.

Why the tiers did not fire (verified by 3 adversarial reviewers against
code, state and Alpaca 5-min history): the tier stop measures ONE session
vs the previous close (anchor = last_equity) with Tier 1 at −8.0%
(−4.0 × 2.0 leverage), as the validated stop grid prescribes. Worst
sessions vs anchor: Oct 14 −5.85/−5.87/−6.58%, Oct 15 (09:55 low)
−6.95/−6.87/−7.85% (primary/X/process). The process slot had NO scanner
at all. The book-DD gate reached −12.95% at 09:35 on Oct 15; with
gross_pre_gate 1.167 under the 1.0 cap it cannot cut until −14.6% (same
in the validated construction), and the live absolute 0.10 band would
have waited until −16.4%. Proof nothing fired: applied/tier_floor
unchanged, empty October journal, stale trip flag.

Findings beyond the question:
1. Rebalance pre-flight compared the ENTIRE target book to buying_power,
   so the Sep 29 rebalance from a full book auto-scaled to ~0.75 of
   design: both gated slots have run ~1.6x (primary 0.671 of design,
   X 0.694) since Sep 29, while the gate state recorded 0.857 — the
   daily pass would have mis-scaled every trade. (Cushioned this week;
   at 2x the primary would have printed −7.9% on Oct 15.)
2. Process slot enable_cutloss=false since its 2026-07-21 activation,
   contradicting its template and protocol_process.json; manifest did not
   pin it, so invariants could not see it.
3. No scanner heartbeat / session-minimum drawdown anywhere.
4. Gate band absolute (12-33% slack deep in a drawdown); book_dd keyed to
   drifted MV weights (validated: decision-date target weights).

Fixes shipped (tests 264 passing, adversarially reviewed before deploy):
- core/orders.py: pre-flight sizes on INCREMENTAL buys (new + up-sizes);
  only the buy side is ever scaled; exits/trims execute in full.
- core/gate_update.py: rebalance records the REALIZED multiplier and a
  sticky tier floor when the broker funds <97% of design; one-time
  reconciliation of pre-fix bookkeeping from the last rebalance record
  (expected 2026-10-16 09:35: primary applied 0.857→~0.671 floor ~0.783,
  X →~0.694 floor ~0.810, both HOLD, no trades); relative symmetric band
  3.5% of applied (was absolute 0.10); today-row guard; book_dd keyed to
  the last rebalance's target weights.
- core/risk.py: heartbeat every tick, session minimum DD persisted,
  NEAR-MISS warning within 2pp of Tier 1, one anchor INFO line per
  session. dashboard: /ready 503 on a stale heartbeat in market hours;
  /api/status cutloss block; tile shows today's min DD.
- core/config.py: CUTLOSS_CALIBRATION_VERSION 2026-10-riskfix-v2 enables
  the process slot's tiers via the standard volume-config migration;
  spec_manifest pins cutloss for all three slots.

Decision recorded: design leverage (2x) returns at the next scheduled
rebalance (~2026-10-28) through the fixed pre-flight — NOT via a daily
re-lever into the drawdown (the shortfall is treated as a sticky floor,
like a tier cut). The validated path would be at 2x now; this is a
deliberate, disclosed, conservative deviation for ~9 sessions. What no
layer covers, by validated design: a multi-day drawdown from peak. The
research's P=−8 choice accepted a −14.5% worst day and −49.6% MaxDD;
whether that profile is the one the owner wants is the open question for
the 3-month checkpoints (Oct 20/21/27).

Review round 2 (adversarial, 10 agents, 7 confirmed / 0 refuted) before
deploy: the one-time reconciliation now COMPOSES onto the current applied
multiplier (overwriting it would have dropped a same-cycle tier cut from
applied but kept it in the floor and sold the cut twice); the sticky floor
is set only on broker FUNDING evidence (auto-scale fired, rejections,
unfilled, fill rate <97%) — untraded drift inside the 10% hold rule moves
applied, never the floor; scaled up-sizes keep their reserved budget;
heartbeat also on the Tier-3 trip-flag path; /ready grace covers the
session open; migration wording cites the slot template only (the process
protocol has no tier clause — tiers are an operational-safety decision,
PK1 stays pre-registered without them). From the live Sep 29 record
(achieved $140,970 on equity $88,248, auto-scale fired) the primary's
reconciliation lands at applied ≈ 0.685, floor ≈ 0.80. Tests: 270.

## 2026-10-15 (later) — Opening-auction execution built, shipped DARK

Owner asked for the "best setup" and to implement it. Decisions:
- The three live tests stay at their pre-registered 2x (a leverage change
  mid-test would flag MODIFIED_AFTER_START). The evidence-based setup for
  real money / the next cycle is 1.75x with the full validated stack
  (LEV1 table: each leverage step buys less return and more drawdown;
  Calmar falls monotonically from 0.91 at 1x to 0.80 at 2x).
- Opening-auction execution is the largest free gain (~0.4 pp/mo of drift
  the backtest never paid: 2026-09-29 vs-open cost +72/+119/+9 bp/side).

Built: exec_style="auction" (core/orders.py submit_auction_orders /
finalize_auction_orders; core/runner.py stage="auction"; dashboard cron
"auction_pipeline" 09:06 ET; core/market.is_auction_window 09:00-09:27).
Stage 1 submits the rebalance as whole-share limit-on-open orders at
prev_close ± 100 bp and records them as pending; the normal 09:35 run
journals the fills with slippage vs the official open (exec_style
"auction_loo"), cancels leftovers, and completes shortfalls at
marketable-limit. No slot is on "auction" yet — ships dark; measurement
pre-registered in the research repo (results/v7/execution/
AUCTION1_PREREG.md, sha 56cf9965…) on combo_v2_process for its next three
rebalances with a frozen adopt/revert rule. Enablement follows as a
versioned config migration once the dark deploy is verified. 278 tests.
Honest answer recorded for the owner: lower leverage and auction fills
shrink weeks like Oct 7-15 (≈ −12% at 1.75x vs −17% at 2x) — they do not
prevent momentum crashes; no validated layer does.

Auction build — adversarial review rounds (2026-10-15): round 1 confirmed 25
findings (stale pre-open snapshot anchor -> now yesterday's official close
from the bot's own panel; completion leg fell to plain market -> now
marketable-limit tagged auction_completion; split run_ids -> one per
rebalance day; fills orphaned by early returns -> finalize runs first and
for any date after 09:31 ET; sells now market-on-open; the 09:35 run
completes the STORED 09:06 plan; main run waits for a running stage; stage
bookkeeping separated from the main run's). Round 2 confirmed 16 more
(plan-based fill rate so re-bought residues never double-count — the naive
rate would have fired a false K3a; a skipped stage-1 day never inherits an
earlier day's plan; no completion while auction orders are live;
data-guard reset main-only; unreadable orders stay pending; atomic run
guard; skip flag cleared on success; auction-stage errors visible in
/ready). 283 tests. Still DARK: no slot on exec_style="auction".

Review round 3 (adversarial, 10 confirmed + 4 nits, 2 refuted), fixed before
enablement: one unreadable order no longer defers the whole completion
(three polled reads with backoff; after 09:31 ET the completion runs
regardless — every opg order is terminal by then and the completion is
delta-based against live positions; unreadable leftovers stay pending for
journaling retries only); per-order isolation in finalize (one malformed
payload never drops the rest, and an exception keeps the pending list for
the next attempt); earlier sessions' unread remainders travel as `orphans`
and are drained by every later finalize; a completed plan is stamped
`completed_at` so a forced same-day re-run re-plans on fresh signals and
never re-merges the auction leg; the data guard and the empty-target abort
are bypassed when today's stored plan exists (the completion needs no
fresh universe); the auction fill rate is measured at the anchor price
(shares executed × previous close) so a gap-down morning never reads as an
unfilled residue; a late finalize carries no reference open (excluded from
the vs-open cost); the 09:35 main run waits for a running stage until
15:30 ET instead of 900 s; /ready demotes a stage error to a warning once
a main run has succeeded and flags a deferred completion; completion rows
are tagged `auction_completion` only when a stored plan is being completed
(a rebalance of an auction slot with no plan is an ordinary
marketable-limit row); the auction stage skips the whole yfinance data
phase on the 20 of 21 sessions when no auction slot is due. 290 tests.
Still DARK — enablement (migration + manifest pin) follows once this
deploy is verified. The AUCTION1 pre-registration received a dated
appended amendment (sells market-on-open, anchor-priced fill rate,
holiday-corrected measurement days 2026-10-28 / 11-27 / 12-28).

## 2026-10-15 (evening) — Opening-auction execution ENABLED on combo_v2_process (AUCTION1)

Dark build with the round-3 fixes verified live first (deploy 7df7b2fd,
commit 831739b; the first build attempt 17d37d63 failed on Railway's
builder disk — "no space left on device" — and was rebuilt from source):
all three slots still on marketable_limit, invariants 34/34, state intact.

Enablement = versioned config migration `2026-10-auction-v1`
(core/config.py `_migrate_exec_style`): the ENABLED combo_v2_process slot
moves to exec_style "auction" with exec_auction_buffer_bps 100 (its
completion leg keeps exec_limit_buffer_bps 10); disabled slots and the
primary / Candidate X slots are untouched; one-shot — a later REVERT per
the frozen rule is never re-applied. spec_manifest.json pins exec_style
"auction" + exec_auction_buffer_bps 100.0 on the process slot (35 checks;
a silent flip back turns /ready red). program_calendar.json carries Oct 16
(first stage fire, expected no-op), Oct 28 / Nov 27 / Dec 28 (AUCTION1
days 1-3) and Dec 29 (decision report due).

What changes live from tomorrow: the 09:06 ET auction stage fires every
weekday; on the 20 of 21 sessions when no auction slot is due it stamps
auction_last_run and skips the data phase. On Oct 28 (the process slot's
next rebalance under the weekday-count cadence from Sep 29) it submits
whole-share opg orders — market-on-open sells, limit-on-open buys at the
previous official close +100 bp — before 09:27 ET and stores the plan;
the 09:35 run finalizes the fills (journal rows exec_style "auction_loo",
slippage vs the official open), completes residues at marketable-limit
under the same run_id, and merges both legs into the reconciliation
(plan-based fill rate). The primary and Candidate X stay on
marketable-limit as same-day controls. Operational execution change of
the 2026-07-25 class, not a protocol amendment: protocol_process.json is
untouched and its bound sha unchanged. Measurement and adopt/revert rule
frozen in the research repo (results/v7/execution/AUCTION1_PREREG.md,
amended 2026-10-15 before any auction order; sha b1d91cbe…). Verification
prompts scheduled for Oct 16 09:53 ET (first stage fire) and Oct 28 09:52
ET (day 1/3). 292 tests.

## 2026-10-15 (night) — Auction execution: review round 4 (19 agents, 32 confirmed / 3 refuted) fixed

Live impact before Oct 28: none beyond the 09:06 stage no longer creating
a log file on no-op mornings. Fixed, all with regression tests (322 total):
- MAJOR (critic): a shared data-phase failure at 09:35 on an auction day
  skipped the completion and would have submitted a SECOND auction plan
  the next morning. Now the main run finishes TODAY's stored plan with no
  panel at all (plan short-circuits the data/macro guards and the whole
  prediction block; run_pipeline's data-phase exits run plan-bearing
  slots on the stored plan), and a failed completion flags /ready.
- finalize: a cancel whose final read fails keeps the entry pending (a
  partial fill may exist); HTTP 404 is terminal ("not_found", typed
  AlpacaHTTPError); journal-append failure keeps the entry pending and
  counts nothing; rows already journaled (restart between appends and the
  state save) are counted, never re-appended; per-order isolation now
  really exercised (a malformed fill price keeps the entry pending and
  OUT of the submitted notional); no sleep after the last retry; 90 s
  wall-clock budget; orphans retried for 7 days then dropped loudly; a
  retry pass ACCUMULATES into the day's summary (used to replace it).
- reference open: today's official open from the bot's own panel when
  its last bar is dated today, else the IEX snapshot bar only when dated
  today (stale pre-open bars rejected), none for earlier sessions; the
  source mix is recorded per day. Rows carry qty_submitted and the
  anchor-priced submitted notional, so a share-based fill rate can be
  rebuilt from rows after a partial fill.
- submission: deterministic client_order_id per (day, symbol, run) and
  recovery by GET v2/orders:by_client_order_id when a POST's response is
  lost; names whose last bar is not the prior session's close are left to
  the completion ("stale_close"); a stale panel skips the stage.
- runner: a live stored plan is always "due" (a forced stage on a non-due
  day no longer strands the plan); the stage never drains orphans before
  the 09:27 cutoff (the main run does); the completion's regime exposure
  / diagnostics describe the executed 09:06 plan; the data-guard abort
  counter resets only when the panel really passed; the auction path is
  keyed on the New York session date (an evening manual run no longer
  mislabels same-session fills as late); per-slot failures now raise at
  the end of run_pipeline so the dashboard and /ready see them.
- dashboard: a single-slot manual run no longer clears the dropped-cron
  alarm; an auction stage that yielded to a manual/dry run leaves a
  "skipped" trace (a /ready problem when a slot was due); manual non-dry
  runs are refused inside 09:00-09:27 ET when an auction slot is due and
  not yet submitted; /ready's "today" alarms use the ET session date.
- tests: a session conftest forces DATA_DIR to a temp dir (one test used
  to run the real exec-style migration and could overwrite a live config);
  mocks now mirror the client (rejected POST raises; poll never raises).
Calendar label for Dec 28 corrected (21st weekday after Nov 27; no
holiday skip involved).

## 2026-10-16 (early) — Auction execution: review round 5 (19 agents, 27 confirmed / 8 refuted + 4 gaps) fixed

Two findings mattered beyond the auction path:
- The rebalance result never carried the account equity, so the realized
  shortfall / sticky funding-floor logic shipped on 2026-10-15 was DEAD
  in production (_shortfall_from_result returned nothing) — and the
  one-time reconciliation the gated slots were about to run at the next
  daily pass would have been a no-op. rebalance_portfolio now writes
  portfolio_value into its result. (The Sep 29 history entries still lack
  it; the reconciliation falls back to the book's achieved_leverage if
  the Oct 16 gate pass shows it did not apply — verify.)
- A container restart straddling 09:35 ET silently lost the day's main
  run (the in-memory job store computes the next fire from boot; misfire
  grace never applied). Boot-time catch-up now re-fires a missed main
  run (09:35-15:45 ET, any slot without a main run today or with an open
  auction plan) and a missed auction stage (09:06-09:19 ET, a due slot
  with no stage record); /ready flags a plan not completed by 10:00 ET.

Auction path: the stage writes a provisional 'submitting' record before
the first POST and persists every accepted order as it is submitted; a
run that died mid-loop is never resubmitted and the 09:35 run adopts
whatever the venue holds under today's client_order_id prefix; LOO buys
are sized at the limit price so reserved buying power never exceeds the
allocation; dot-form exits (BRK.B) match the stale-close and panel-open
lookups; after a cancel only a TERMINAL state is journaled (anything
else stays pending with cancel_sent); the vw cost is merged across
finalize passes by the dollars it was measured over; the journal never
glues a row onto a truncated line and the dedupe scan recovers ids from
one; an unmeasured auction leg sets the merged fill rate to NA (the
protocol's K3a must not read the completion's ~100%), and a later first
measurement repairs the completed run's history entry; the deferred-
completion branch wins over 'market closed' for a pre-open manual run;
dry runs never drive the data-guard outage counter; today's official
opens for the auction rows are fetched for the pending names (the 09:35
panel ends at the prior session); pre-check decisions are replayed into
the run's log file; per-slot failures raise a typed SlotFailures and the
dashboard marks only the failed slot. Dashboard: the auction stage waits
for a concurrent manual/dry run until its 09:20 start cutoff instead of
yielding; due-ness is recorded at skip time; only a full-slot main run
can "recover" an errored/skipped stage; manual runs — dry included — are
refused in the 09:00-09:27 window while an auction slot is due and the
stage has not run; the stale-name count reports target names only.
Tests: 346 (session conftest makes the suite runnable under bare pytest
and cleans its temp dir).

## 2026-10-16 (pre-open) — Auction execution: review round 6 (19 agents, 36 confirmed / 2 refuted + 5 gaps) fixed

Two majors were in my own round-5 catch-up logic:
- The boot catch-up would have RE-RUN a rebalance that a redeploy/OOM
  interrupted mid-submission while the dead run's DAY limit orders were
  still resting at the venue (second buy on top of a resting buy). Every
  non-auction rebalance now sweeps and cancels resting non-opg orders and
  waits for them to settle BEFORE sizing deltas (core/orders.py
  cancel_open_orders), which also covers manual re-runs.
- A same-day dry run stamped last_run, so a restart straddling 09:35 after
  a morning dry run would still have lost the day: dry runs now stamp
  last_dry_run only; the catch-up keys on live runs.
Also: catch-up one-shots carry a real misfire grace (900 s / 300 s) and a
missed/failed catch-up is a /ready problem; a boot after 09:19 on a due
day records the lost auction stage (auction_skipped_due) instead of
silently rebalancing via marketable-limit; the catch-up never collides with
a cron about to fire; NYSE early closes (13:00) are modelled
(core/market.py EARLY_CLOSE_DATES / session_close) and bound the catch-up
window and the main run's wait cutoff; a dead 'submitting' record is
preserved (never overwritten by a repeat stage) and the stage resubmits
only when the venue is provably empty; the 09:35 run reconciles EVERY
submitted/submitting record with the venue once (lost-response orders
whose by-id lookup also failed are adopted), a failed listing keeps the
record open for retry, adopted records get a synthesized run_id (so the
completion closes the plan and merges the leg), recovered orders get their
anchor from the panel/snapshot; the merge reads NA when nothing anchor-
priced was measured and keeps the completion leg's own rate across a
repair; the repair works the next day; recovered buys use the slot's
buffer; a cancel is counted only when it took effect; the equity used by
the shortfall derivation is read at the same moment as the achieved book
(portfolio_value_after); the one-time gate reconciliation retries up to 3
passes instead of stamping None forever; yfinance single-ticker frames
and dot-form symbols are handled; a SlotFailures run whose failed slot is
the auction slot never reads as a full run. Tests: 370; two tests that
depended on the wall clock (one hung the suite before 09:20 ET) are
pinned.

## 2026-10-16 (05:00 ET) — Auction execution: review round 7 (18 of 19 agents, 21 confirmed / 0 refuted) fixed; one earlier claim corrected

CORRECTION to the 2026-10-16 round-5 entry: the rebalance result DID carry
the account equity all along — core/alpaca.get_account() writes
portfolio_value / cash / buying_power into report.data["rebalance"], which
IS the result dict. The round-5 finding ("shortfall / sticky-floor logic
dead in production") came from mocks that replaced get_account without
that side effect; the live Sep 29 records carry portfolio_value (primary
$88,248.34, Candidate X $110,851.23). The round-5 line and the
achieved-leverage fallback are harmless and kept. Today's reconciliation
is expected to land at applied 0.685 / floor 0.799 (primary) and 0.712 /
0.831 (Candidate X), bookkeeping only.

Round-7 fixes (one cross-cutting critic agent did not run — spend limit):
- Pre-existing, on today's branch: a gate pass that writes
  last_rebalance=None (book gated ~flat / full de-lever re-entry request)
  was overwritten by the runner's merged save with the stale value; the
  not-due branch now adopts the disk value after the gate pass.
- Live since the round-6 deploy: the boot catch-up listener recorded EVERY
  scheduler job (the 60 s scanner grew bot_status each minute, and a
  single scanner misfire would have turned /ready red until restart). It
  now records only the catch-up one-shots and is registered only when one
  was scheduled.
- The resting-order sweep now ABORTS the rebalance (slot stays due, /ready
  flags it) when a resting limit order will not cancel or an in-flight
  market order (the scanner's) does not settle within 20 s, instead of
  sizing on top of a live order; a failed settle listing is retried; a
  cancel is counted only on a confirmed terminal state; a failed initial
  listing (3 tries) proceeds as before the sweep existed.
- The post-trade equity read no longer overwrites the result's pre-trade
  cash / buying power.
- _after_open_et uses the session close (13:00 on early closes): opg
  orders are terminal after a half-day close.
- The auction stage checks the submission window before touching the
  venue, and a dead 'submitting' record is never overwritten by a skip.
- Venue reconciliation keys "known" orders on every id the record ever
  held (persistent order_ids + journal rows), so a retry after a failed
  listing never re-adopts already-finalized orders; only a changed record
  is re-counted.
- Recovered anchors use the snapshot's previous close only when the
  snapshot's daily bar is the record's own session.
- The auction merge is idempotent (completion-only flow snapshotted once,
  merged flow rebuilt in place) and re-runs on every measuring pass after
  the completion, so a late or venue-adopted order reaches the completed
  run's reconciliation exactly once.
- A forced re-run after a completed plan is marked
  auction_rerun_after_completion, not auction_stage_missed; a missed stage
  carries the stage's own outcome.
- Lost-stage detection at boot now covers the whole window in which a
  non-auction rebalance of the due slot will still happen (it stopped at
  09:50).
Tests: 388.

## 2026-10-16 (05:45 ET) — Compact review round 8 (7 agents, 8 confirmed / 2 refuted) fixed

The walk-through of today's runs (09:06 stage with no slot due; 09:35 gate
passes with the one-time reconciliation; process slot not due) found
nothing that affects today. No slot is due before 2026-10-28, so the
resting-order sweep does not run today.
- MAJOR (my round-7 design): one unsettled order (e.g. the scanner's market
  sell on a halted name) aborted the WHOLE book with no same-day retry; on
  an auction day it would have left the book half-rebalanced. The guard is
  now per-symbol: names with a live order at the venue are left untouched
  this run and the rest of the book is rebalanced (rebalance_blocked_symbols
  -> /ready warning). Only when every planned order is blocked is the run
  aborted (slot stays due, /ready problem). Any non-limit open order type
  (market, stop, trailing_stop, unknown) is waited on and blocks its name;
  the bot never cancels a non-limit order.
- An aborted completion no longer receives the auction merge (it would have
  entered history as a funded rebalance with an auction-only fill rate);
  the protocol scorer, /api/reconciliation and the gate's last-funded
  reader skip aborted and deferred entries; auction_run_id is stamped on
  history only for a completion that ran.
- A dead 'submitting' record keeps a running submitted notional (persist
  callback) and its outcome is decided from every order it ever held, so a
  retry after a failed listing no longer turns an executed morning into
  'nothing submitted / $0'.
- A dead record is replaced by the stage's own outcome when the venue was
  proven clear (replaced_dead_record), instead of masquerading as "died
  before submitting".
- The lost-measurement-day alarm is persisted (auction_stage_missed_date)
  so it survives a container restart; a failed venue reconciliation today
  is a /ready warning.
Tests: 398.

## 2026-10-16 (06:15 ET) — Compact review round 9 (7 agents, 9 confirmed / 1 refuted; regression critic: none) fixed

- Names with a live order at the venue now come out of the plan BEFORE the
  buying-power pre-flight (a blocked buy's reservation is already out of
  buying power; counting it again auto-scaled the other buys and set a
  false sticky funding floor on gated slots). A held blocked name is pinned
  at its market value in the book target; an unheld one is dropped.
- An auction completion whose whole residue is blocked still completes the
  plan (the auction leg already traded the book); only an ordinary
  rebalance with every name blocked aborts.
- A cancel that fails transiently is re-sent inside the 20 s settle loop
  (at most 3 tries).
- A missed-stage rebalance (re-planned at marketable-limit) spends today's
  stage record, so a venue order adopted later can never resurrect the
  morning plan and trade the book back to it; a same-day completion never
  clears today's missed-stage alarm.
- The blocked-names warning stays on /ready until a later rebalance trades
  without any (it used to expire at midnight and never clear on a clean
  re-run). Accepted limitation: blocked names are NOT retried automatically
  — the operator re-runs the slot once the order settles, otherwise the
  next scheduled rebalance corrects them; this needs a live order on a
  planned name at 09:35 on a rebalance day (rare).
Tests: 404.
