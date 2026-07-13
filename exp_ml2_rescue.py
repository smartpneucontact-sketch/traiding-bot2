"""ML-2 — Stage 1 rescue follow-ups (pre-registered, max 4 cells).

Gate 1 failed 0/12 in Stage 1. Best near-miss: S1_01_Arank_L1
(A_rank / L1 / pool 80 / EW / HP_S / 5d grid) — rank-IC 0.0391 (t=3.46),
edge over prior +0.0215, pos-years 100%, hit30 51.53% vs the 52.0% bar.

Rescue grid — exactly one cell per allowed modification, all on the S1_01
base (per the ML-2 assignment: choose among HP_M / L4 / F2 / pool=120):
  R1_Arank_L1_HPM   : HP_M capacity (num_leaves 31, depth 6, 600 trees, lr .03)
  R2_Arank_L4       : momentum-residualized label (per-date OLS resid of
                      pctrank(fwd) on pctrank(mom_12_1), re-pct-ranked)
  R3_Arank_L1_F2    : F2 extended feature set (53 cols, see rescue_lib)
  R4_Arank_L1_p120  : candidate pool 120 with the ranker (untested cell)

Gate 1 (unchanged): rank_ic >= 0.03 AND ic >= prior+0.01 AND >=60% positive
fold-years AND hit30 >= 0.52. Dev window only (CV test years 2018-2022).

Run:  /opt/anaconda3/bin/python3 exp_ml2_rescue.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pandas as pd

from ml_track.experiments import gate1
from ml_track.rescue_lib import log_signal_row, run_walkforward_x

RESCUE = {
    "R1_Arank_L1_HPM": dict(arch="A_rank", label="L1", pool=80,
                            weighting="EW", hp="HP_M", grid=5,
                            monotone=False, fset="F1"),
    "R2_Arank_L4": dict(arch="A_rank", label="L4", pool=80,
                        weighting="EW", hp="HP_S", grid=5,
                        monotone=False, fset="F1"),
    "R3_Arank_L1_F2": dict(arch="A_rank", label="L1", pool=80,
                           weighting="EW", hp="HP_S", grid=5,
                           monotone=False, fset="F2"),
    "R4_Arank_L1_p120": dict(arch="A_rank", label="L1", pool=120,
                             weighting="EW", hp="HP_S", grid=5,
                             monotone=False, fset="F1"),
}


def main():
    rows = []
    for exp_id, spec in RESCUE.items():
        print(f"\n=== {exp_id}: {spec} ===")
        wf = run_walkforward_x(spec["arch"], spec["label"], spec["pool"],
                               spec["grid"], spec["monotone"], spec["hp"],
                               fset=spec["fset"])
        m = wf["metrics"]
        g1 = gate1(m)
        print(f"  rank_ic={m['rank_ic']:+.4f} t={m['rank_ic_t']:.2f} "
              f"prior={m['prior_ic']:+.4f} edge={m['ic_minus_prior']:+.4f} "
              f"pos_yrs={m['pct_pos_years']:.0%} hit30={m['hit30']:.4f} "
              f"-> G1 {'PASS' if g1 else 'FAIL'}")
        print(f"  ic_by_year={m['ic_by_year']}")
        rows.append(log_signal_row(exp_id, spec, m, g1,
                                   notes="ML-2 rescue follow-up on S1_01"))

    df = pd.DataFrame(rows)
    print("\n── RESCUE MATRIX ──")
    print(df[["exp_id", "features", "hp", "pool", "label", "rank_ic",
              "prior_ic", "ic_vs_prior", "pct_pos_years", "hit30",
              "g1_pass"]].to_string(index=False))


if __name__ == "__main__":
    main()
