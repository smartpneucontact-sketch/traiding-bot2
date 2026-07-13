"""ASSEMBLY added cell (1 of max 3, rationale pre-registered in
results/v7/ASSEMBLY_REPORT.md): ASM_Bst_BTF1 = E3 residual base, static 1/3,
BOOK drawdown gate + TIER + FREEZE dd_v1, tc in {5,10,20}bp.

Composition identical to exp_e5_overlays combo B+T+F1, on the Bst base:
W = expand_daily(blend*2) x g_book x f1 -> generalized tier loop.
g_book computed once on the UN-GATED base daily frame (scalar-gate invariant).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine_v2 import book_drawdown_gate, apply_gate  # noqa: E402
from assemble_candidates import (TierEnv, build_blend, tier_row, cand_params,  # noqa: E402
                                 F1_MULT, P_TIER, DELAY, TCS, panel, cutoff, idx)

blend = build_blend("B", "st")
env = TierEnv(blend)

g_book = book_drawdown_gate(env.W2_daily, panel["close"].loc[:cutoff],
                            full_dd=0.12, cash_dd=0.30, lookback=60)
g_book = g_book.reindex(idx).fillna(1.0).clip(0.0, 1.0)

W = apply_gate(apply_gate(env.W2_daily, g_book), F1_MULT)

rows = []
for tc in TCS:
    name = "ASM_Bst_BTF1" + ("" if tc == 5.0 else f"_tc{int(tc)}")
    params = cand_params("B", "st", "TF1", tc)
    params["overlay"] = "book(0.12,0.30,60) + tier(P-0.08,2d,phi0) + freeze dd_v1"
    params["added_cell"] = True
    net, ev, traded, gross = env.tier_loop(W, P_TIER, DELAY, tc)
    row = tier_row(name, net, ev, traded, gross, params, tc,
                   "ADDED CELL (pre-registered rationale in ASSEMBLY_REPORT.md);")
    rows.append({"cand": name, "tc": tc, **{k: row[k] for k in
                 ("mean_monthly", "sharpe", "max_drawdown", "calmar",
                  "turnover_ann", "avg_gross", "ep_2018Q4_dd", "ep_covid_dd",
                  "ep_2022_dd", "ep_2022_ret")},
                 "tier_events": json.dumps(ev)})
    print(f"{name:22s} mm={row['mean_monthly']*100:6.3f}% "
          f"sharpe={row['sharpe']:.3f} dd={row['max_drawdown']*100:6.1f}% "
          f"calmar={row['calmar']:.3f} 2018Q4={row['ep_2018Q4_dd']*100:6.1f}% "
          f"covid={row['ep_covid_dd']*100:6.1f}% events={ev}")

pd.DataFrame(rows).to_csv(ROOT / "results" / "v7" /
                          "ASSEMBLY_added_cell.csv", index=False)
