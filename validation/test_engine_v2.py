"""engine_v2 correctness tests.

1. legacy_period reproduces the published combo_v2_2x_baseline (results_v6.json)
   to machine precision — proves v2 is a superset of backtest.py.
2. Synthetic hand-computed cases pin next_close and next_open math exactly.
3. Corrected modes switch books exactly one trading day after each decision.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python3 validation/test_engine_v2.py
(plain script, no pytest dependency needed; exits non-zero on failure)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest import BTConfig, run_backtest  # noqa: E402
from engine_v2 import BTConfigV2, run_backtest_v2  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = ""):
    flag = "OK  " if cond else "FAIL"
    print(f"  {flag} {label} {detail}")
    if not cond:
        FAILURES.append(label)


# ───── 1. Synthetic hand-computed test ───────────────────────────────────
def test_synthetic():
    print("\n[1] synthetic 2-ticker hand-computed case")
    idx = pd.bdate_range("2024-01-01", periods=8)
    close = pd.DataFrame({
        "AAA": [100, 102, 101, 103, 104, 102, 105, 106],
        "BBB": [50, 50, 51, 52, 51, 53, 54, 53],
    }, index=idx, dtype=float)
    opn = pd.DataFrame({
        "AAA": [99, 101, 102, 102, 103, 103, 104, 105],
        "BBB": [50, 49, 50, 51, 52, 52, 53, 54],
    }, index=idx, dtype=float)
    # Decisions: day0 → 100% AAA; day3 → 50/50.
    w = pd.DataFrame(
        [[1.0, 0.0], [0.5, 0.5]],
        index=[idx[0], idx[3]], columns=["AAA", "BBB"],
    )
    cfg0 = dict(tc_bps=0.0, leverage_cap=2.0)

    # next_close: book in force on day t is the decision from t-1.
    bt = run_backtest_v2(w, close, BTConfigV2(exec_model="next_close", **cfg0))
    r = bt["returns"]
    # day1: effective = decision(day0) = 100% AAA → 102/100−1 = 0.02
    check("next_close day1", np.isclose(r.iloc[1], 0.02), f"got {r.iloc[1]:.6f}")
    # day3: effective = decision(day2 → still day0 book) = AAA → 103/101−1
    check("next_close day3", np.isclose(r.iloc[3], 103 / 101 - 1), f"got {r.iloc[3]:.6f}")
    # day4: decision(day3)=50/50 → 0.5·(104/103−1) + 0.5·(51/52−1)
    exp4 = 0.5 * (104 / 103 - 1) + 0.5 * (51 / 52 - 1)
    check("next_close day4 (switch)", np.isclose(r.iloc[4], exp4), f"got {r.iloc[4]:.6f}")

    # next_open: switch day4 — old book (AAA) carries the full c2c move plus
    # delta carries intraday: r = w_prev·r_cc + Δw·r_intra
    bt = run_backtest_v2(w, close, BTConfigV2(exec_model="next_open", **cfg0),
                         open_prices=opn)
    r = bt["returns"]
    # day1 (entry): prev_book=0; Δw=AAA 1.0 → intraday only: 102/101−1
    check("next_open day1 (entry intraday)", np.isclose(r.iloc[1], 102 / 101 - 1),
          f"got {r.iloc[1]:.6f}")
    # day2 (no switch): full c2c on AAA = 101/102−1
    check("next_open day2", np.isclose(r.iloc[2], 101 / 102 - 1), f"got {r.iloc[2]:.6f}")
    # day4 (switch to 50/50): w_prev=AAA·1; r_cc_AAA=104/103−1;
    # Δw = (−0.5 AAA, +0.5 BBB); r_intra_AAA=104/103−1, r_intra_BBB=51/52−1
    exp4 = (104 / 103 - 1) + (-0.5) * (104 / 103 - 1) + 0.5 * (51 / 52 - 1)
    check("next_open day4 (switch)", np.isclose(r.iloc[4], exp4), f"got {r.iloc[4]:.6f}")

    # costs: tc=10bp, turnover day1 = 1.0, day4 = |Δw|=1.0
    bt = run_backtest_v2(w, close, BTConfigV2(exec_model="next_close",
                                              tc_bps=10.0, leverage_cap=2.0))
    r_tc = bt["returns"]
    check("tc charged on switch", np.isclose(r_tc.iloc[4], exp_next_close_day4(close) - 0.001),
          f"got {r_tc.iloc[4]:.6f}")


def exp_next_close_day4(close):
    return 0.5 * (104 / 103 - 1) + 0.5 * (51 / 52 - 1)


# ───── 2. Switch-day timing ──────────────────────────────────────────────
def test_switch_timing():
    print("\n[2] corrected modes switch exactly decision+1")
    idx = pd.bdate_range("2024-01-01", periods=30)
    close = pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=idx)
    close["AAA"] = 100 * (1.01 ** np.arange(30))
    w = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]],
                     index=[idx[5], idx[20]], columns=["AAA", "BBB"])
    bt = run_backtest_v2(w, close, BTConfigV2(exec_model="next_close", tc_bps=0))
    eff = bt["weights"]
    check("flat before first exec", (eff.loc[idx[5]] == 0).all())
    check("book live at decision+1", eff.loc[idx[6], "AAA"] == 1.0)
    check("switch at second decision+1",
          eff.loc[idx[20], "AAA"] == 1.0 and eff.loc[idx[21], "BBB"] == 1.0)

    # Degenerate equivalence: daily decisions → legacy == next_close
    w_daily = pd.DataFrame(np.random.RandomState(0).rand(30, 2),
                           index=idx, columns=["AAA", "BBB"])
    a = run_backtest_v2(w_daily, close, BTConfigV2(exec_model="legacy_period", tc_bps=5))
    b = run_backtest_v2(w_daily, close, BTConfigV2(exec_model="next_close", tc_bps=5))
    check("daily-decision degenerate equivalence",
          np.allclose(a["returns"].values, b["returns"].values, atol=1e-12))


# ───── 3. Legacy regression vs published record ──────────────────────────
def test_legacy_regression():
    print("\n[3] legacy_period reproduces published combo_v2_2x_baseline")
    sys.path.insert(0, str(ROOT / "validation"))
    from reproduce_baseline import build_combo_v2_base  # noqa: E402
    from strategies import union_prices  # noqa: E402

    with open(ROOT / "data_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    px, macro = cache["panel"]["close"], cache["macro"]
    pu = union_prices(px, macro)
    base = build_combo_v2_base(px, macro)

    v1 = run_backtest(base * 2.0, pu, BTConfig(tc_bps=5.0, leverage_cap=2.0))
    v2 = run_backtest_v2(base * 2.0, pu,
                         BTConfigV2(exec_model="legacy_period", tc_bps=5.0,
                                    leverage_cap=2.0))
    check("legacy == v1 daily returns (machine precision)",
          np.allclose(v1["returns"].values, v2["returns"].values, atol=1e-14))

    with open(ROOT / "results" / "results_v6.json") as f:
        pub = json.load(f)["combo_v2_2x_baseline"]
    s = v2["summary"]
    for k in ("mean_monthly", "sharpe", "max_drawdown"):
        check(f"legacy matches published {k}",
              np.isclose(s[k], pub[k], atol=1e-9),
              f"v2={s[k]:.9f} pub={pub[k]:.9f}")

    # Corrected modes on the same weights — report the lag tax for the record.
    for mode in ("next_close", "next_open"):
        op = cache["panel"]["open"] if mode == "next_open" else None
        bt = run_backtest_v2(base * 2.0, pu,
                             BTConfigV2(exec_model=mode, tc_bps=5.0,
                                        leverage_cap=2.0),
                             open_prices=op)
        sm = bt["summary"]
        print(f"      [{mode:10s}] mo={sm['mean_monthly']*100:+.3f}% "
              f"sharpe={sm['sharpe']:.3f} maxDD={sm['max_drawdown']*100:.1f}% "
              f"calmar={sm['calmar']:.3f}")


if __name__ == "__main__":
    test_synthetic()
    test_switch_timing()
    test_legacy_regression()
    print(f"\n{'ALL TESTS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    sys.exit(0 if not FAILURES else 1)
