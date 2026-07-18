# CRASH1 PRE-REGISTRATION — frozen 2026-07-18, BEFORE any grid cell runs

Family: CRASH1. sha256 of this file recorded in the committing commit.
Amendments only as dated appended sections.

## Hypothesis

A Daniel–Moskowitz-style momentum-crash overlay (de-risk when the market is
in a bear state with elevated volatility) can replace the DISPUTED freeze
layer in the candidate-X stack (see results/v7/candidate_x/DECISION_RULE.md)
with cleaner provenance: the bear/vol conditioning is a published academic
mechanism, not an in-sample grid artifact.

## Honesty disclosure

The family has never touched the validation window (2023-01-01..2026-03-27),
so it is procedurally entitled to a one-shot validation. But: the BASE it
overlays has known validation behavior, and the designers know the val window
contains no 2018Q4-type slide. That program-level contamination is disclosed
here and cannot be removed. Teeth: (1) the one-shot validation gates below
are frozen NOW, before the dev grid runs; (2) exactly ONE winner may take the
one-shot; (3) failing the one-shot burns the family permanently.

## Grid (8 cells, dev-only, family CRASH1, every cell ledgered)

Base frame: the candidate-X construction (ASM_Bst_BTF1's blend + book gate +
tier) with the FREEZE LAYER REMOVED and the crash overlay in its place.
Engine: next_open, tc=5bp, margin_bps_annual=600, leverage_cap 2.0.

- Bear definition (2): SPY trailing 24-month total return < 0; SPY close <
  its 200-day moving average.
- Response (2): binary de-lever to 0.5x while bear; continuous DM-style
  scale = clip(bear_indicator x (target_vol / trailing 126d realized vol)).
- Vol threshold (2): trailing 63d SPY realized vol > rolling 75th percentile
  required (on/off).

(2 x 2 x 2 = 8 cells. No other cells, no extensions without a dated
amendment BEFORE running them.)

## Dev gates (all required, evaluated against the margin600 comparator)

Comparator: the candidate-X construction re-scored at margin 600 (family
ASM_rescore600) — same cost basis.

- dev Calmar >= 1.05 x comparator dev Calmar
- dev mean_monthly >= comparator dev mean − 0.5pp/mo
- neighborhood stability: median Calmar of the winning cell's adjacent cells
  >= 0.85 x winner Calmar

## One-shot validation gates (frozen now)

- val Calmar >= 1.62 (the champion val anchor, carried WITH its
  selection-contamination disclosure verbatim)
- val MaxDD >= −35%
- 2026Q1 DD <= −27.2% (champion's episode)

## Consequences

- Pass dev AND val: candidate X v2 = same construction with crash overlay
  replacing freeze. Earliest swap at v1's 3-month operational checkpoint; a
  swap resets the shadow 12-month clock (new bundle version, new
  protocol_exp sha binding, old record archived).
- Pass dev, fail val: family BURNED. No second shot, no re-tuning.
- Fail dev: no val shot; family closed.
