"""E1 — strategy-level VOL-MANAGED momentum (Barroso/Santa-Clara; Daniel/Moskowitz).

Scale the champion book daily by m_t = clip(target_vol / realized_vol(strategy),
floor, cap); realized vol = trailing std of the UNSCALED strategy's daily net
returns (lagged 1d inside engine_v2.vol_managed_multiplier).

Pre-registered grid (54 cells):
  (a) BASE_LEV=2.0, descale-only: target_vol in {0.30, 0.40, 0.50} (LEVERED
      basis = 15/20/25% on the 1x book), lookback in {21, 63, 126}d, cap=1.0,
      floor in {0.2, 0.3, None}                                       -> 27
  (b) BASE_LEV=1.5, symmetric: same targets/lookbacks, cap=1.333 (max gross
      2.0), floor in {0.2, 0.3, None}                                 -> 27

Protocol: dev window ONLY (run_trial enforces). Family ledger:
results/v7/trials/E1_volmanaged.csv
"""
from __future__ import annotations

import sys
sys.path.insert(0, '.')

import itertools
import json

import numpy as np
import pandas as pd

from engine_v2 import (BTConfigV2, run_backtest_v2, expand_daily,
                       vol_managed_multiplier)
from exp_lib import load_cache, union_prices_cached, run_trial
from metrics_v2 import DEV_END

FAMILY = "E1_volmanaged"

TARGETS = [0.30, 0.40, 0.50]
LOOKBACKS = [21, 63, 126]
FLOORS = [0.2, 0.3, None]
ARMS = [
    {"base_lev": 2.0, "cap": 1.0},     # (a) descale-only
    {"base_lev": 1.5, "cap": 1.333},   # (b) symmetric, max gross 2.0
]


def main():
    panel, macro, _ = load_cache()
    pu = union_prices_cached()
    opn = panel["open"]
    cutoff = pd.Timestamp(DEV_END)
    pu_d, opn_d = pu.loc[:cutoff], opn.loc[:cutoff]

    base = pd.read_parquet("weights_store/combo_v2_base_1x.parquet")
    base_d = base.loc[:cutoff]
    cfg = BTConfigV2(tc_bps=5.0, leverage_cap=2.0, exec_model="next_open")

    rows = []
    for arm in ARMS:
        bl, cap = arm["base_lev"], arm["cap"]
        # PASS 1: unscaled dev-window daily net returns for this BASE_LEV.
        # Engine called directly (not via run_trial) to avoid polluting the
        # ledger; multiplier is built from dev-window returns only.
        bt0 = run_backtest_v2(base_d * bl, pu_d, cfg,
                              name=f"E1_unscaled_lev{bl}",
                              open_prices=opn_d)
        unscaled = bt0["returns"]
        print(f"[pass1] lev={bl}: mean_monthly={bt0['summary']['mean_monthly']:.4%} "
              f"sharpe={bt0['summary']['sharpe']:.3f} "
              f"maxdd={bt0['summary']['max_drawdown']:.1%} "
              f"calmar={bt0['summary']['calmar']:.3f}")

        for tv, lb, fl in itertools.product(TARGETS, LOOKBACKS, FLOORS):
            m = vol_managed_multiplier(unscaled, tv, lookback=lb,
                                       cap=cap, floor=fl)
            W = expand_daily(base_d * bl, pu_d.index).mul(m, axis=0)
            fl_tag = "none" if fl is None else f"{fl:g}"
            name = f"E1_lev{bl:g}_tv{tv:g}_lb{lb}_fl{fl_tag}"
            params = {"base_lev": bl, "target_vol": tv, "lookback": lb,
                      "cap": cap, "floor": fl}
            res = run_trial(W, name=name, family=FAMILY, params=params,
                            window="dev", weights_are_daily=True,
                            notes="vol-managed overlay on combo_v2 base")
            r = res["row"]
            print(f"  {name}: mm={r['mean_monthly']:.4%} sh={r['sharpe']:.3f} "
                  f"dd={r['max_drawdown']:.1%} calmar={r['calmar']:.3f} "
                  f"to={r['turnover_ann']:.1f} gross={r['avg_gross']:.2f}")
            rows.append({**params, **r})

    df = pd.DataFrame(rows)
    df.to_csv("results/v7/trials/E1_grid_flat.csv", index=False)
    analyze(df)


def floor_key(fl):
    return -1.0 if fl is None or (isinstance(fl, float) and np.isnan(fl)) else fl


def neighbors(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """±1-grid-step neighbors along each axis (same arm/base_lev)."""
    fls_sorted = sorted(FLOORS, key=floor_key)  # None < 0.2 < 0.3
    def steps(vals, v):
        i = vals.index(v)
        out = []
        if i > 0:
            out.append(vals[i - 1])
        if i < len(vals) - 1:
            out.append(vals[i + 1])
        return out

    cur_fl = None if pd.isna(row["floor"]) else row["floor"]
    masks = []
    arm = df["base_lev"] == row["base_lev"]
    fl_match = df["floor"].apply(lambda x: floor_key(None if pd.isna(x) else x))
    for tv in steps(TARGETS, row["target_vol"]):
        masks.append(arm & (df["target_vol"] == tv)
                     & (df["lookback"] == row["lookback"])
                     & (fl_match == floor_key(cur_fl)))
    for lb in steps(LOOKBACKS, int(row["lookback"])):
        masks.append(arm & (df["target_vol"] == row["target_vol"])
                     & (df["lookback"] == lb)
                     & (fl_match == floor_key(cur_fl)))
    for fl in steps(fls_sorted, cur_fl):
        masks.append(arm & (df["target_vol"] == row["target_vol"])
                     & (df["lookback"] == row["lookback"])
                     & (fl_match == floor_key(fl)))
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    return df[mask]


def analyze(df: pd.DataFrame):
    CALMAR_GATE, MM_GATE, DD_GATE = 0.944, 0.04, -0.50
    df = df.sort_values("calmar", ascending=False).reset_index(drop=True)
    print("\n===== TOP 10 by dev Calmar =====")
    cols = ["name", "mean_monthly", "sharpe", "max_drawdown", "calmar",
            "turnover_ann", "avg_gross", "ep_covid_dd", "ep_2022_ret",
            "ep_2022_dd"]
    print(df[cols].head(10).to_string())

    print("\n===== G1 evaluation =====")
    hard = df[(df["calmar"] >= CALMAR_GATE) & (df["mean_monthly"] >= MM_GATE)
              & (df["max_drawdown"] > DD_GATE)]
    print(f"configs passing hard gates: {len(hard)}")
    winners = []
    for _, row in hard.iterrows():
        nb = neighbors(df, row)
        med_nb = nb["calmar"].median()
        ok = med_nb >= 0.85 * row["calmar"]
        print(f"  {row['name']}: calmar={row['calmar']:.3f} "
              f"nb_median={med_nb:.3f} ({len(nb)} neighbors) "
              f"stability {'PASS' if ok else 'FAIL'}")
        if ok and len(winners) < 2:
            winners.append((row, nb, med_nb))
    if not winners:
        print("NO configs pass G1 (incl. stability).")
    return winners


if __name__ == "__main__":
    main()
