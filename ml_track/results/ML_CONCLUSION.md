# ML track — CONCLUSION: no incremental candidate-level signal (track closed)

Date: 2026-06-11. Scope: ML-1 (Stage 0 fidelity + CORE 12 matrix) and ML-2
(4 rescue follow-ups + 2 diagnostic portfolio conversions). All selection on
the dev window only (CV test years 2018–2022; portfolio 2016-04..2022-12),
corrected engine, next_open, 5 bp, ≤2.0x gross.

## Bottom line

**The LightGBM re-ranker adds a real, clean, statistically significant rank
signal over within-pool momentum — and it does not convert into portfolio
improvement under any tested design. 0 of 16 configurations pass the
pre-registered Gate 1; the best portfolio conversion reaches Calmar 0.899 vs
the 0.944 bar with a worse drawdown than the champion. Nothing advances to
assembly.**

## Evidence trail (all in ml_track/results/ledger.csv and results/v7/trials/)

1. Stage 0: engine fidelity exact (baseline reproduced to <1e-9).
2. Stage 1 CORE 12: best S1_01 A_rank/L1 rank-IC 0.0391 (t 3.46), edge +0.0215,
   positive all 5 fold-years — fails only hit30 (51.53% vs 52%). 0/12 pass.
3. ML-2 rescue (every allowed modification of S1_01, pre-registered, 4 cells):
   - HP_M capacity: hurts (IC 0.0275). FAIL.
   - L4 momentum-residualized label: best signal of the track
     (IC 0.0423, t 3.52, edge +0.0247, all years positive). FAIL hit30 (51.79%).
   - F2 extended features (53 cols): IC 0.0311 but 2022 turns negative. FAIL.
   - pool 120: IC 0.0361, best hit30 51.96% — 0.04pp short. FAIL.
   0/4 pass; cumulative 0/16.
4. Portfolio conversion (4 diagnostics across both stages, I2 blend
   (ML+dual+adaptive)/3 × 2.0 vs baseline 4.555%/mo / Sharpe 1.044 /
   MaxDD −60.2% / Calmar 0.858):
   - S1_01: 4.811%/mo, Calmar 0.871, DD −63.4%
   - S1_05: 4.231%/mo, Calmar 0.801, DD −59.6%
   - R2/L4: 4.497%/mo, Calmar 0.801, DD −62.4% (best IC → worst conversion)
   - R4/p120: 4.837%/mo, Calmar 0.899, DD −61.4% (best of track)
   None approaches the program bar (Calmar ≥ 0.944, MaxDD > −50%); the best
   would also fail the ML-2 Gate 2 (mean ≥ +0.3pp/mo AND Calmar ≥ +0.10).

## Why it fails (mechanism, not bad luck)

- The IC edge lives in the **middle** of the candidate distribution: every
  config with real edge has top-30 hit-rate 50.6–52.0%. The portfolio only
  buys the head, where the ML ranking is statistically indistinguishable from
  the momentum prior it must beat.
- Making the signal *more* incremental (L4 residualization) makes the
  portfolio *worse*: orthogonal-to-momentum information is by construction
  the part the momentum sleeve cannot monetize as a selection rule.
- The binding portfolio constraint of the whole program is drawdown, and the
  ML re-ranker has no mechanism to address it: every ML variant carries
  −59% to −63% MaxDD, like the champion it would replace.

## Disposition

- Stage 2 integration (I1/I2/I3) and Stage 3 robustness: NOT run — gated off
  by Gate 1; running them would be selection on the validation of a failed
  signal. `gate2_passed.json` is intentionally NOT written.
- Honest negative, publishable: the candidate-level ML re-ranker track is
  closed for V7 assembly. DD reduction must come from portfolio/overlay
  mechanics (vol management, gating, allocation), not candidate re-ranking.
- If ML is ever revisited, the only directions not falsified here are
  head-focused objectives outside the allowed grid (e.g. lambdarank truncation
  at k=30 with head-weighted gains) and regime/overlay-level ML — both new
  pre-registrations, not extensions of this track.

Reports: ml_track/results/STAGE1_REPORT.md (ML-1),
results/v7/ML2_REPORT.md (ML-2). Reproduce: `python3 exp_ml2_rescue.py`,
`python3 exp_ml2_diag.py` (deterministic, seed 7).
