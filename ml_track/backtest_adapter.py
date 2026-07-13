"""Bridge to the corrected V7 engine via exp_lib.run_trial.

GATE 0 (fidelity): corrected_baseline() re-runs the corrected champion
(combo_v2_base_1x x 2.0, next_open, 5bp) through run_trial and asserts the
dev-window mean_monthly / sharpe / calmar match results/v7/baseline.json to
1e-9. Same engine, same numbers — else STOP.

I2 integration: ML sleeve replaces the xs_momentum sleeve in the combo blend:
(ML_sleeve + dual_momentum + adaptive_voltarget) / 3 x LEV.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import exp_lib
from ml_track.config import COST_BPS, LEV

ROOT = Path(exp_lib.ROOT)


def corrected_baseline() -> dict:
    base = pd.read_parquet(ROOT / "weights_store" / "combo_v2_base_1x.parquet")
    res = exp_lib.run_trial(
        base * LEV, name="ML_stage0_fidelity_combo_v2_2x",
        family="ML_stage0", params={"source": "combo_v2_base_1x x 2.0"},
        window="dev", exec_model="next_open", tc_bps=COST_BPS,
        leverage_cap=LEV, notes="GATE 0 fidelity re-run of corrected champion")
    ref = json.loads((ROOT / "results" / "v7" / "baseline.json").read_text())["dev"]
    got = res["summary"]
    for k in ("mean_monthly", "sharpe", "calmar"):
        if abs(got[k] - ref[k]) > 1e-9:
            raise AssertionError(
                f"GATE 0 FAILED: {k} got {got[k]!r} vs baseline {ref[k]!r}")
    return {k: got[k] for k in ("mean_monthly", "sharpe", "max_drawdown",
                                "calmar")}


def _sleeves() -> tuple[pd.DataFrame, pd.DataFrame]:
    dual = pd.read_parquet(
        ROOT / "weights_store" / "sleeve_dual_momentum_voltarget_n30.parquet")
    adp = pd.read_parquet(
        ROOT / "weights_store" / "sleeve_adaptive_voltarget_n30.parquet")
    return dual, adp


def blend_i2(ml_frame: pd.DataFrame) -> pd.DataFrame:
    """(ML + dual + adaptive)/3 x LEV on the union decision grid.
    Each sparse frame is ffilled onto the union grid (a sleeve's last decision
    persists between its own rebalances); missing history = 0 exposure —
    identical mechanics to the combo_v2 base blend."""
    dual, adp = _sleeves()
    dates = dual.index.union(adp.index).union(ml_frame.index)
    cols = dual.columns
    parts = []
    for f in (ml_frame.reindex(columns=cols).fillna(0.0), dual, adp):
        parts.append(f.reindex(dates).ffill().fillna(0.0))
    return (parts[0] + parts[1] + parts[2]) / 3.0 * LEV


def run_ml_sleeve(decision_frame: pd.DataFrame, name: str,
                  params: dict | None = None, notes: str = "") -> dict:
    """I2 integration backtest of an ML sleeve (sparse 21d frame, sum<=1)."""
    w = blend_i2(decision_frame)
    return exp_lib.run_trial(
        w, name=name, family="ML_stage1", params=params or {},
        window="dev", exec_model="next_open", tc_bps=COST_BPS,
        leverage_cap=LEV, notes=notes or "I2 = (ML+dual+adaptive)/3 x 2.0")
