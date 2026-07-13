"""ML-2 — diagnostic I2 portfolio backtests (ADDED CELLS, clearly marked).

Gate-1 status after the 4 pre-registered rescue cells: 0/4 pass (all fail
hit30). These two runs are NOT gate candidates and are NOT advanced; they are
diagnostics under the program protocol's added-cell allowance (<=3, rationale
required):

RATIONALE: R2 (A_rank/L4) carries the largest signal edge of the whole ML
track (OOF rank-IC 0.0423, +0.0247 over the momentum prior) and R4
(A_rank/L1/p120) the best top-30 hit-rate (51.96%). Stage 1 measured the
portfolio conversion of S1_01 (IC 0.0391 -> Calmar 0.871 vs baseline 0.858).
Measuring conversion of the strictly-stronger rescue signals quantifies the
ceiling of this ML design at the portfolio level and is required for an
honest ML_CONCLUSION.md ("does more IC move portfolio Calmar at all?").

I2 = (ML sleeve + dual_momentum + adaptive_voltarget)/3 x 2.0, corrected
engine, next_open, 5bp, dev window — identical to Stage-1 diagnostics.

Run:  /opt/anaconda3/bin/python3 exp_ml2_diag.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import exp_lib
from ml_track.backtest_adapter import run_ml_sleeve
from ml_track.experiments import gate1
from ml_track.portfolio import build_decision_frame
from ml_track.rescue_lib import log_signal_row, run_walkforward_x

DIAG = {
    "R2_Arank_L4": dict(arch="A_rank", label="L4", pool=80, weighting="EW",
                        hp="HP_S", grid=5, monotone=False, fset="F1"),
    "R4_Arank_L1_p120": dict(arch="A_rank", label="L1", pool=120,
                             weighting="EW", hp="HP_S", grid=5,
                             monotone=False, fset="F1"),
}


def main():
    panel, _, _ = exp_lib.load_cache()
    for exp_id, spec in DIAG.items():
        print(f"\n=== DIAG I2 {exp_id} ===")
        wf = run_walkforward_x(spec["arch"], spec["label"], spec["pool"],
                               spec["grid"], spec["monotone"], spec["hp"],
                               fset=spec["fset"])
        m = wf["metrics"]
        oof = wf["oof"]
        psc = oof[oof["_is_port"]][["score", "sigma60"]]
        frame = build_decision_frame(psc, panel["close"].columns,
                                     weighting=spec["weighting"])
        res = run_ml_sleeve(
            frame, name=f"I2_{exp_id}",
            params={**spec, "exp_id": exp_id, "added_cell": True},
            notes="ML-2 ADDED CELL: G1-fail diagnostic portfolio run")
        s = res["summary"]
        print(f"  mm={s['mean_monthly']:.4%} sharpe={s['sharpe']:.3f} "
              f"dd={s['max_drawdown']:.1%} calmar={s['calmar']:.3f} "
              f"turn={s['turnover_annualized']:.1f}")
        log_signal_row(f"{exp_id}_diagI2", spec, m, gate1(m),
                       notes="ML-2 added cell: diagnostic I2 portfolio run",
                       port=s)


if __name__ == "__main__":
    main()
