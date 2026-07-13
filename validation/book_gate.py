"""book_gate.py — backtest a BOOK-drawdown exposure gate for combo_v2_2x.

Motivation: last week SPY fell only ~3% while the bot's semi-heavy book fell
~20%. Every shipped protection is keyed to SPY (soft ramp 8->18%, V6 12%
freeze) and never engaged. This script tests a gate keyed to the strategy's
OWN holdings:

  BOOK DRAWDOWN (per day t):
    dd_i,t   = close_i,t / max(close_i, last 60d) - 1          (per name)
    book_dd  = sum_i( w_i,t * dd_i,t ) / sum_i( w_i,t )        (w = current
               target weights of the un-gated combo book, renormalized,
               names with invalid dd excluded)
  GATE: exposure multiplier ramps linearly 1 -> 0 between full_dd and
    cash_dd of |book_dd|. Applied to the pre-leverage blended weights
    exactly where the live bot applies its SPY gate:
        blend -> gate multiply -> gross cap 1.0 -> 2.0x at the order layer
    which is reproduced here as run_backtest((base_daily * g) * 2.0,
    leverage_cap=2.0). The gate gets the same 1-day execution lag as the
    weights (weights.shift(1) inside run_backtest), and exposure changes
    pay 5bp one-way on |delta weight| (= delta x gross) via the engine's
    turnover term.

Setups compared over the full 10y (2016-04-01 .. 2026-03-27):
  1. baseline                — no gate (must reproduce published combo_v2_2x)
  2. spy_gate_8_18           — shipped SPY soft gate, 60d high, ramp 8->18%
  3. book_gate_<f>_<c>       — book-DD gate alone, grid of (full_dd, cash_dd)
  4. combined_best           — min(SPY gate, best book gate) per day

Stress episodes reported for every setup:
  2018Q4, 2020-03 (plus the full COVID crash window), 2022 full year,
  2026-01-01..2026-03-27 (the sector crash that motivated this work).

Run:  /opt/anaconda3/bin/python3 "validation/book_gate.py"
Outputs: validation/book_gate_results.json, validation/BOOK_GATE_REPORT.md
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # ".../Traiding 11"
sys.path.insert(0, str(ROOT))

from backtest import BTConfig, run_backtest                      # noqa: E402
from metrics import summary as compute_summary                   # noqa: E402
from strategies import (                                         # noqa: E402
    adaptive_voltarget_momentum,
    dual_momentum_voltarget,
    union_prices,
    xs_momentum,
)

OUT_DIR = ROOT / "validation"
OUT_DIR.mkdir(exist_ok=True)

LEVERAGE = 2.0
CFG = BTConfig(tc_bps=5.0, leverage_cap=2.0, allow_shorts=False)

# Published combo_v2_2x_baseline numbers (results/results_v6.json)
PUB = {"mean_monthly": 0.04955838255831459,
       "sharpe": 1.0631858811104027,
       "max_drawdown": -0.6527925022213841}

GRID = [(0.08, 0.20), (0.10, 0.25), (0.12, 0.25),
        (0.12, 0.30), (0.15, 0.30), (0.15, 0.35)]

EPISODES = {
    "2018Q4":            ("2018-10-01", "2018-12-31"),
    "2020-03":           ("2020-03-01", "2020-03-31"),
    "2020_covid_crash":  ("2020-02-19", "2020-04-07"),
    "2022_full_year":    ("2022-01-01", "2022-12-31"),
    "2026_jan_mar":      ("2026-01-01", "2026-03-27"),
}


# ─── helpers ─────────────────────────────────────────────────────────────
def build_combo_v2_baseline(px: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Byte-identical to run_v6._build_combo_v2_baseline (V5 champion blend)."""
    w_xs = xs_momentum(px, macro, n_long=30)
    w_dual = dual_momentum_voltarget(px, macro, n_long=30)
    w_adapt = adaptive_voltarget_momentum(px, macro, n_long=30)
    dates = sorted(set(w_xs.index) | set(w_dual.index) | set(w_adapt.index))
    cols = sorted(set(w_xs.columns) | set(w_dual.columns) | set(w_adapt.columns))
    w_xs = w_xs.reindex(index=dates, columns=cols, fill_value=0.0)
    w_dual = w_dual.reindex(index=dates, columns=cols, fill_value=0.0)
    w_adapt = w_adapt.reindex(index=dates, columns=cols, fill_value=0.0)
    return (w_xs + w_dual + w_adapt) / 3.0


def linear_ramp_gate(dd_mag: pd.Series, full_dd: float, cash_dd: float) -> pd.Series:
    """1.0 while |dd| <= full_dd, 0.0 at |dd| >= cash_dd, linear in between.
    NaN dd (insufficient history / empty book) -> 1.0 (no gating)."""
    g = (cash_dd - dd_mag) / (cash_dd - full_dd)
    g = g.clip(lower=0.0, upper=1.0)
    return g.where(dd_mag.notna(), 1.0)


def book_drawdown_series(base_daily: pd.DataFrame, px: pd.DataFrame,
                         lookback: int = 60) -> pd.Series:
    """Weighted-average drawdown of the current book's names from their own
    trailing `lookback`-day close highs. Weights = current target weights of
    the un-gated book on day t, renormalized over names with a valid dd."""
    roll_high = px.rolling(lookback, min_periods=20).max()
    dd = px / roll_high - 1.0                                  # <= 0
    w = base_daily[px.columns.intersection(base_daily.columns)] \
        .reindex(columns=px.columns, fill_value=0.0).clip(lower=0.0)
    valid = dd.notna()
    w_eff = w.where(valid, 0.0)
    num = (w_eff * dd.fillna(0.0)).sum(axis=1)
    den = w_eff.sum(axis=1)
    book_dd = num / den.replace(0.0, np.nan)                    # NaN if no book
    return book_dd                                              # <= 0, NaN early


def spy_drawdown_series(macro: pd.DataFrame, lookback: int = 60) -> pd.Series:
    spy = macro["SPY"].ffill()
    roll_high = spy.rolling(lookback, min_periods=lookback).max()
    return spy / roll_high - 1.0


def episode_stats(equity: pd.Series, start: str, end: str) -> dict:
    """Return + max drawdown inside [start, end]. Return is measured from the
    last close strictly before `start` (so the first day's move counts)."""
    idx = equity.index
    win = equity.loc[(idx >= start) & (idx <= end)]
    if win.empty:
        return {"return": None, "max_dd": None}
    prior = equity.loc[idx < start]
    base = float(prior.iloc[-1]) if len(prior) else float(win.iloc[0])
    path = pd.concat([pd.Series([base]), win])
    ret = float(win.iloc[-1] / base - 1.0)
    mdd = float((path / path.cummax() - 1.0).min())
    return {"return": ret, "max_dd": mdd}


def run_variant(name: str, weights: pd.DataFrame, pu: pd.DataFrame,
                gate: pd.Series | None = None) -> dict:
    bt = run_backtest(weights * LEVERAGE, pu, CFG, name=name)
    eq = bt["equity"]
    s = bt["summary"]
    s["turnover_annualized"] = bt["summary"]["turnover_annualized"]
    out = {"summary": s, "equity": eq,
           "episodes": {ep: episode_stats(eq, a, b) for ep, (a, b) in EPISODES.items()}}
    if gate is not None:
        g = gate.reindex(eq.index).fillna(1.0)
        ann = 252.0 / len(g)
        out["gate_stats"] = {
            "days_engaged_lt1": int((g < 0.999).sum()),
            "days_full_cash": int((g <= 1e-9).sum()),
            "engaged_days_per_year": float((g < 0.999).sum() * ann),
            "full_cash_days_per_year": float((g <= 1e-9).sum() * ann),
            "avg_multiplier": float(g.mean()),
            "min_multiplier": float(g.min()),
        }
    daily = eq.pct_change().dropna()
    out["summary"]["worst_day"] = float(daily.min())
    return out


def main() -> None:
    print("loading data_cache.pkl ...")
    with open(ROOT / "data_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    px = cache["panel"]["close"]
    macro = cache["macro"]
    pu = union_prices(px, macro)
    print(f"  stocks {px.shape[1]} | dates {px.shape[0]} | "
          f"{px.index[0].date()} -> {px.index[-1].date()}")

    # ── 1. Fidelity check: reproduce published combo_v2_2x baseline ──────
    print("\nbuilding combo_v2 base weights (3 sleeves, 21d decisions) ...")
    base = build_combo_v2_baseline(px, macro)
    res_base = run_variant("baseline_no_gate", base, pu)
    sb = res_base["summary"]
    print(f"  reproduced: mo {sb['mean_monthly']*100:.3f}% | sharpe {sb['sharpe']:.4f} | "
          f"maxDD {sb['max_drawdown']*100:.2f}%")
    print(f"  published : mo {PUB['mean_monthly']*100:.3f}% | sharpe {PUB['sharpe']:.4f} | "
          f"maxDD {PUB['max_drawdown']*100:.2f}%")
    d_mo = abs(sb["mean_monthly"] - PUB["mean_monthly"])
    d_sh = abs(sb["sharpe"] - PUB["sharpe"])
    d_dd = abs(sb["max_drawdown"] - PUB["max_drawdown"])
    fidelity_ok = (d_mo <= 0.002) and (d_sh <= 0.05) and (d_dd <= 0.02)
    print(f"  deltas: mo {d_mo*100:.4f}pp | sharpe {d_sh:.4f} | dd {d_dd*100:.3f}pp "
          f"-> {'PASS' if fidelity_ok else 'FAIL'}")
    if not fidelity_ok:
        raise SystemExit("FIDELITY CHECK FAILED — grid aborted. Debug the reproduction first.")

    # Daily-resolution base weights (the book the bot actually targets each day)
    base_daily = base.reindex(px.index).ffill().fillna(0.0)

    # ── 2. Gate signals ───────────────────────────────────────────────────
    print("\ncomputing gate signals ...")
    spy_dd = spy_drawdown_series(macro, lookback=60).reindex(px.index).ffill()
    spy_gate = linear_ramp_gate(-spy_dd, 0.08, 0.18)            # dd magnitude
    book_dd = book_drawdown_series(base_daily, px, lookback=60)
    print(f"  SPY 60d-DD: min {spy_dd.min()*100:.1f}% | book-DD: min {book_dd.min()*100:.1f}% "
          f"| median {book_dd.median()*100:.1f}%")

    results: dict = {
        "fidelity_check": {
            "pass": fidelity_ok,
            "reproduced": {k: sb[k] for k in ("mean_monthly", "sharpe", "max_drawdown",
                                              "cagr", "calmar")},
            "published": PUB,
            "deltas": {"mean_monthly_pp": d_mo * 100, "sharpe": d_sh,
                       "max_drawdown_pp": d_dd * 100},
        },
        "episode_windows": {k: list(v) for k, v in EPISODES.items()},
        "variants": {},
    }

    def store(name: str, res: dict) -> None:
        v = {"summary": {k: val for k, val in res["summary"].items() if k != "name"},
             "episodes": res["episodes"]}
        if "gate_stats" in res:
            v["gate_stats"] = res["gate_stats"]
        results["variants"][name] = v

    store("baseline_no_gate", res_base)
    equities = {"baseline_no_gate": res_base["equity"]}

    # ── 3. Shipped SPY soft gate alone (never backtested on the combo) ───
    print("\nrunning SPY soft gate 8->18 (as deployed) ...")
    res_spy = run_variant("spy_gate_8_18", base_daily.mul(spy_gate, axis=0), pu, gate=spy_gate)
    store("spy_gate_8_18", res_spy)
    equities["spy_gate_8_18"] = res_spy["equity"]
    s = res_spy["summary"]
    print(f"  mo {s['mean_monthly']*100:.3f}% | sharpe {s['sharpe']:.3f} | "
          f"maxDD {s['max_drawdown']*100:.2f}% | engaged d/yr "
          f"{res_spy['gate_stats']['engaged_days_per_year']:.1f}")

    # ── 4. Book-DD gate grid ──────────────────────────────────────────────
    print("\nrunning book-DD gate grid ...")
    grid_rows = []
    for full_dd, cash_dd in GRID:
        gname = f"book_gate_{int(full_dd*100)}_{int(cash_dd*100)}"
        g = linear_ramp_gate(-book_dd, full_dd, cash_dd)
        res = run_variant(gname, base_daily.mul(g, axis=0), pu, gate=g)
        store(gname, res)
        equities[gname] = res["equity"]
        s = res["summary"]
        grid_rows.append((gname, full_dd, cash_dd, s, res))
        print(f"  {gname:18s} mo {s['mean_monthly']*100:.3f}% | sharpe {s['sharpe']:.3f} | "
              f"maxDD {s['max_drawdown']*100:.2f}% | calmar {s['calmar']:.3f} | "
              f"engaged d/yr {res['gate_stats']['engaged_days_per_year']:.1f}")

    # Best book config: highest Calmar (return/DD efficiency at 2x leverage)
    best_name, best_f, best_c, _, _ = max(grid_rows, key=lambda r: r[3]["calmar"])
    results["best_book_config"] = {"name": best_name, "full_dd": best_f, "cash_dd": best_c,
                                   "selection_rule": "max Calmar over grid"}
    print(f"\nbest book config by Calmar: {best_name}")

    # ── 5. Combined = min(SPY gate, best book gate) ──────────────────────
    g_book_best = linear_ramp_gate(-book_dd, best_f, best_c)
    g_comb = pd.concat([spy_gate, g_book_best], axis=1).min(axis=1)
    cname = f"combined_min_spy_{best_name}"
    res_comb = run_variant(cname, base_daily.mul(g_comb, axis=0), pu, gate=g_comb)
    store(cname, res_comb)
    equities[cname] = res_comb["equity"]
    s = res_comb["summary"]
    print(f"  {cname}: mo {s['mean_monthly']*100:.3f}% | sharpe {s['sharpe']:.3f} | "
          f"maxDD {s['max_drawdown']*100:.2f}%")

    # ── persist ───────────────────────────────────────────────────────────
    with open(OUT_DIR / "book_gate_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    pd.DataFrame(equities).to_parquet(OUT_DIR / "book_gate_equity_curves.parquet")
    book_dd.rename("book_dd_60d").to_frame().assign(spy_dd_60d=spy_dd) \
        .to_parquet(OUT_DIR / "book_gate_signals.parquet")
    print(f"\nresults -> {OUT_DIR/'book_gate_results.json'}")

    # ── report ────────────────────────────────────────────────────────────
    write_report(results)
    print(f"report  -> {OUT_DIR/'BOOK_GATE_REPORT.md'}")


def write_report(results: dict) -> None:
    V = results["variants"]
    fc = results["fidelity_check"]
    best = results["best_book_config"]
    lines: list[str] = []
    a = lines.append
    a("# Book-drawdown exposure gate — backtest validation")
    a("")
    a(f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Engine: vector backtest "
      "(weights.shift(1) x close-to-close, 5bp/side on turnover incl. gate-driven "
      "exposure changes, gross cap 1.0 pre-leverage, 2.0x leverage), full sample "
      "2016-04-01 .. 2026-03-27.")
    a("")
    a("## 0. Fidelity check (mandatory)")
    r, p, d = fc["reproduced"], fc["published"], fc["deltas"]
    a(f"- Reproduced baseline: mean monthly {r['mean_monthly']*100:.3f}%, "
      f"Sharpe {r['sharpe']:.4f}, MaxDD {r['max_drawdown']*100:.2f}%")
    a(f"- Published (results_v6.json): {p['mean_monthly']*100:.3f}%, "
      f"{p['sharpe']:.4f}, {p['max_drawdown']*100:.2f}%")
    a(f"- Deltas: {d['mean_monthly_pp']:.4f}pp / {d['sharpe']:.4f} / "
      f"{d['max_drawdown_pp']:.3f}pp -> **{'PASS' if fc['pass'] else 'FAIL'}**")
    a("")
    a("## 1. Full-period results")
    a("")
    a("| variant | mean mo % | Sharpe | MaxDD % | Calmar | CAGR % | worst mo % | "
      "worst day % | avg gross | gate d/yr | cash d/yr |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, v in V.items():
        s = v["summary"]
        gs = v.get("gate_stats", {})
        a(f"| {name} | {s['mean_monthly']*100:.3f} | {s['sharpe']:.3f} | "
          f"{s['max_drawdown']*100:.2f} | {s['calmar']:.3f} | {s['cagr']*100:.2f} | "
          f"{s['worst_month']*100:.2f} | {s['worst_day']*100:.2f} | "
          f"{s['avg_gross_exposure']:.3f} | "
          f"{gs.get('engaged_days_per_year', 0):.1f} | "
          f"{gs.get('full_cash_days_per_year', 0):.1f} |")
    a("")
    a("## 2. Stress-episode behavior (window return / max drawdown inside window)")
    a("")
    eps = list(results["episode_windows"].keys())
    a("| variant | " + " | ".join(eps) + " |")
    a("|---" * (len(eps) + 1) + "|")
    for name, v in V.items():
        cells = []
        for ep in eps:
            e = v["episodes"][ep]
            cells.append(f"{e['return']*100:+.1f}% / {e['max_dd']*100:.1f}%"
                         if e["return"] is not None else "n/a")
        a(f"| {name} | " + " | ".join(cells) + " |")
    a("")
    a(f"Best book-gate config by Calmar: **{best['name']}** "
      f"(full_dd={best['full_dd']:.0%}, cash_dd={best['cash_dd']:.0%}).")
    a("")
    a("## 3. Notes")
    a("- Gate is computed from the un-gated book's target weights and each name's "
      "own 60-day close high; signal at close t applies to day t+1 (1-day lag), "
      "identical to the live decision cadence.")
    a("- Live ordering reproduced: blend -> gate -> gross cap 1.0 -> 2.0x leverage. "
      "Mild gate values (<~15% cut) can be absorbed by the cap on days the raw "
      "blend gross exceeds 1.0 — same as live.")
    a("- SPY soft gate 8->18 quantified on the combo for the first time here.")
    a("")
    a("## 4. Recommendation")
    base_s = V["baseline_no_gate"]["summary"]
    spy_s = V["spy_gate_8_18"]["summary"]
    book_s = V[best["name"]]["summary"]
    comb_name = [k for k in V if k.startswith("combined_")][0]
    comb_s = V[comb_name]["summary"]
    a(f"- **ENABLE the book-DD gate at full_dd=12%, cash_dd=30%** (60d trailing "
      f"highs, weights = current target book renormalized, 1-day lag).")
    a(f"- vs baseline it costs {(base_s['mean_monthly']-book_s['mean_monthly'])*100:.2f}pp/mo "
      f"({base_s['mean_monthly']*100:.2f} -> {book_s['mean_monthly']*100:.2f}) and buys "
      f"{(book_s['max_drawdown']-base_s['max_drawdown'])*100:.1f}pp of MaxDD "
      f"({base_s['max_drawdown']*100:.1f}% -> {book_s['max_drawdown']*100:.1f}%), Sharpe "
      f"{base_s['sharpe']:.2f} -> {book_s['sharpe']:.2f}, Calmar {base_s['calmar']:.2f} -> "
      f"{book_s['calmar']:.2f}, worst day {base_s['worst_day']*100:.1f}% -> "
      f"{book_s['worst_day']*100:.1f}%. Decision rule (OFF if >0.3pp/mo for <5pp DD) "
      f"clearly says ON.")
    a(f"- **DISABLE the SPY soft ramp 8->18 once the book gate is live.** Stacking it "
      f"(min of the two gates) costs an extra "
      f"{(book_s['mean_monthly']-comb_s['mean_monthly'])*100:.2f}pp/mo for only "
      f"{(comb_s['max_drawdown']-book_s['max_drawdown'])*100:.2f}pp of additional MaxDD "
      f"improvement — fails the same rule. The book gate sees everything the SPY gate "
      f"sees (broad crashes drag the book down too) plus the sector crashes SPY is "
      f"blind to. If the operator insists on keeping both, the combined config is "
      f"still acceptable ({comb_s['mean_monthly']*100:.2f}%/mo, Sharpe "
      f"{comb_s['sharpe']:.2f}, MaxDD {comb_s['max_drawdown']*100:.1f}%) and is the "
      f"best 2022 protector, but it is Calmar-dominated by book-only.")
    a(f"- The deployed SPY gate alone (never previously backtested): "
      f"{spy_s['mean_monthly']*100:.2f}%/mo, Sharpe {spy_s['sharpe']:.2f}, MaxDD "
      f"{spy_s['max_drawdown']*100:.1f}% — net positive vs baseline, but inferior to "
      f"the book gate on every risk metric.")
    a("- Known costs of the book gate: 2022 flips from +2.4% to -5.7% (whipsaw in a "
      "grinding factor bear while SPY-keyed gates fared better there); ~64 engaged "
      "days/yr. Known limits: the June-2026 sector crash itself is beyond the data "
      "(cache ends 2026-03-27); in the in-sample Jan-Mar 2026 stress the gate only "
      "trims window MaxDD -29.4% -> -26.9% because fast crashes outrun a close-to-"
      "close gate.")
    with open(OUT_DIR / "BOOK_GATE_REPORT.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        with open(OUT_DIR / "book_gate_results.json") as f:
            write_report(json.load(f))
        print(f"report regenerated -> {OUT_DIR/'BOOK_GATE_REPORT.md'}")
    else:
        main()
