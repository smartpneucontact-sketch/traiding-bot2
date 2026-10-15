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
