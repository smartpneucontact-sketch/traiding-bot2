# Freeze-ablation report #2 — Candidate X DISPUTED freeze component

Period: 2026-08-31 → 2026-09-25 (written 2026-09-26; due 2026-09-21 per
program calendar — written 5 days late, disclosed). Per DECISION_RULE.md's
adjudication plan for the DISPUTED freeze (protocol_exp.json
`freeze_disputed`).

## Evidence (pulled from the live container 2026-09-26)

- `freeze_state` (pipeline_state_combo_v2_exp.json): `active: false`,
  `freeze_multiplier: 1.0`, `since: null` — never in force this period.
- Rebalance diagnostics (2026-08-31 rebalance): `freeze_active: false`.
- Trigger distance: freeze fires at SPY ≤ −12% from its 21-day peak. The
  worst such drawdown in the period was −2.58% (2026-09-10) — 22% of the
  trigger at the closest approach. SPY finished the window +0.81%
  (2026-08-31 → 2026-09-25); the book, by contrast, rose ~+12%.

## Ablation verdict for the period

With-freeze path ≡ without-freeze path. Ablation delta: $0.00, 0.00pp.
DISPUTED status UNCHANGED (two consecutive null periods; the component
has still never been tested by a qualifying market). Next report due
2026-10-21 (aligned with the 3-month checkpoint).

## Note on what WAS active this period

The daily gate-cadence pass (core/gate_update.py, live since 2026-09-01)
re-levered the Candidate X book once (2026-09-10, 0.809 → 0.911) and held
within band otherwise; no tier stops fired. Those are book-DD-gate
events, not freeze events, and are disclosed in OPS_LOG.md — recorded
here only so the two mechanisms are never conflated in adjudication.
