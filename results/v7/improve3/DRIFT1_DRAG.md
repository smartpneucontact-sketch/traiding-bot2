# DRIFT1 drag decomposition (prereg 'REPORTED' item)

Computed 2026-08-31 from results/v7/improve3/drift1_dev.csv
(RECAL_REPORT arithmetic: monthly cost drag ~= turnover_ann x tc/1e4 / 12).

| band | tc | turnover_ann | est. cost drag %/mo | net geo %/mo | turnover cut vs control |
|---|---|---|---|---|---|
| 0 | 30 | 14.14 | 0.354 | +2.850 | 0.0% |
| 0.0025 | 30 | 14.07 | 0.352 | +2.851 | 0.5% |
| 0.005 | 30 | 13.98 | 0.350 | +2.848 | 1.1% |
| 0.01 | 30 | 13.79 | 0.345 | +2.846 | 2.5% |
| 0 | 50 | 14.14 | 0.589 | +2.608 | 0.0% |
| 0.0025 | 50 | 14.07 | 0.586 | +2.610 | 0.5% |
| 0.005 | 50 | 13.98 | 0.583 | +2.608 | 1.1% |
| 0.01 | 50 | 13.79 | 0.575 | +2.609 | 2.5% |

Reading: even the widest band (1.0% weight points) removes only ~2% of turnover — the champion's turnover is monthly name churn, which drift bands cannot suppress. The declared decomposition confirms the dev verdict's practical conclusion: cost relief must come from execution quality, not trade suppression.
