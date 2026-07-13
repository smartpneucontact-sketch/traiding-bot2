"""ML-1 Stage 0 (fidelity) + Stage 1 (CORE 12 signal matrix).

Run from the workspace root:
    /opt/anaconda3/bin/python3 -m ml_track.run_all_stage1
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import json
import pandas as pd

from ml_track import config as C
from ml_track.backtest_adapter import corrected_baseline
from ml_track.experiments import RESULTS, run_experiment


def main():
    print("── STAGE 0: engine fidelity (GATE 0) ──")
    base = corrected_baseline()
    print(f"GATE 0 PASSED: dev mean_monthly={base['mean_monthly']:.6%} "
          f"sharpe={base['sharpe']:.6f} calmar={base['calmar']:.6f}")
    (RESULTS / "stage0_fidelity.json").write_text(json.dumps(base, indent=2))

    print("\n── STAGE 1: CORE 12 ──")
    rows = []
    for exp_id, spec in C.STAGE1_EXPERIMENTS.items():
        try:
            rows.append(run_experiment(exp_id, spec))
        except Exception as e:
            print(f"  !! {exp_id} ERRORED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            rows.append({"exp_id": exp_id, "notes": f"ERROR {e}"})

    df = pd.DataFrame(rows)
    print("\n── STAGE 1 MATRIX ──")
    cols = [c for c in ("exp_id", "rank_ic", "prior_ic", "ic_vs_prior",
                        "pct_pos_years", "hit30", "g1_pass",
                        "port_mean_monthly", "port_calmar") if c in df.columns]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
