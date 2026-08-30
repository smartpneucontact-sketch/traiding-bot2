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
