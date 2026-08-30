# Freeze-ablation report #1 — Candidate X DISPUTED freeze component

Period: 2026-07-21 (forward-test start) → 2026-08-30. Written 2026-08-30
per DECISION_RULE.md's adjudication plan for the DISPUTED freeze
(protocol_exp.json `freeze_disputed`; journaling published by the runner's
freeze-ablation publisher since first funded rebalance).

## Evidence (pulled from the live container 2026-08-30)

- `freeze_state` (pipeline_state_combo_v2_exp.json): `active: false`,
  `freeze_multiplier: 1.0`, `since: null` — the freeze has NEVER been in
  force since the forward test began.
- Rebalance history diagnostics: every entry carries
  `freeze_active: false`, `freeze_multiplier: 1.0`.
- Journal: 49 rows, none carrying an in-force freeze context.
- Trigger distance: the freeze fires at SPY ≤ −12% from its 21-day peak.
  The worst such drawdown in the window was −3.38% (2026-07-29) — the
  market used 28% of the trigger at its closest approach. (Note the July
  book drawdown that gated the slots was a MID-CAP book event; SPY's own
  21-day drawdown stayed shallow, which is exactly the SPY-keyed-gate
  blindness the book-DD gate exists for.)

## Ablation verdict for the period

With-freeze path ≡ without-freeze path: the component was evaluated on
every run and never altered a single weight or order. Ablation delta:
$0.00, 0.00pp. The DISPUTED status is UNCHANGED — a component that never
fires is neither vindicated nor convicted by a window that never tested
it. Adjudication continues; next report due 2026-09-21 (program calendar).

## Standing context

The freeze's research-side verdict remains RETRACTED for the primary
(in-sample selection; worse than no freeze under corrected execution —
REPORT.md Validity Notice §5). Candidate X carries it as a deliberately
DISPUTED live component precisely so this journaling can adjudicate it on
out-of-sample evidence. Nothing in this period moves that needle.
