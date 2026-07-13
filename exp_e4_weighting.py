"""E4 — WEIGHTING FIXES (V7 research program, family=E4_weighting).

Two known flaws in the combo_v2 blend:
  (1) the xs_momentum sleeve is EQUAL-weight while its two siblings are
      inverse-vol — fix: weight the same top-30 by 1/sigma60 (renormalized).
  (2) dual_momentum_voltarget's 15% vol target uses a correlation-free
      formula (strategies.py:306: sqrt(sum((w*sigma)^2))) that mathematically
      never binds for a 30-name book — fix: ex-ante covariance vol
      (engine_v2.ex_ante_book_vol, 60d window ending at the decision date).

Pre-registered grid (10 scored configs, all x2.0 lev, next_open, 5bp, dev):
  base   : baseline reproduction (diagnostic anchor)                      [1]
  (a)    : xs inverse-vol sleeve replaces xs in the blend                 [1]
  (b)    : TRUE book vol target on the FULL levered blended book,
           target in {0.25, 0.30, 0.35} x (xs eq | xs inv-vol)            [6]
  (c)    : dual sleeve with the FIXED internal formula (target 15%),
           x (xs eq | xs inv-vol)                                          [2]

Diagnostics: binding frequency of the true book vol target per target level
and of the old vs fixed dual-sleeve formula (old binds never, by math).

Run:  cd "<workspace>" && /opt/anaconda3/bin/python3 exp_e4_weighting.py
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from engine_v2 import ex_ante_book_vol
from exp_lib import cached_weights, load_cache, run_trial, union_prices_cached
from metrics_v2 import DEV_END

FAMILY = "E4_weighting"
LEV = 2.0
DEV_CUT = pd.Timestamp(DEV_END)


# ─────────────────────────────────────────────────────────────────────────
# Sleeve builders (fixes)
# ─────────────────────────────────────────────────────────────────────────

def xs_momentum_invvol(
    prices: pd.DataFrame,
    lookback_long: int = 252,
    lookback_skip: int = 21,
    n_long: int = 30,
    rebal_freq: int = 21,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    """xs_momentum top-N (identical 12-1 selection, same date grid) but
    weighted 1/sigma60 renormalized to sum=1 (the inverse-vol pattern of
    adaptive_voltarget_momentum / dual_momentum_voltarget)."""
    px = prices
    mom = px.shift(lookback_skip) / px.shift(lookback_long) - 1.0
    valid = mom.notna()
    daily = px.pct_change(fill_method=None)
    vol = daily.rolling(vol_lookback).std() * np.sqrt(252)

    w_list = []
    for i, dt in enumerate(px.index):
        if i % rebal_freq != 0 or i < lookback_long + lookback_skip:
            continue
        row = mom.loc[dt][valid.loc[dt]]
        if len(row) < n_long:
            continue
        top = row.nlargest(n_long).index
        v = vol.loc[dt][top].replace(0.0, np.nan).dropna()
        if v.empty:
            continue
        inv = 1.0 / v
        wnorm = inv / inv.sum()
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[wnorm.index] = wnorm.values
        w_list.append(wrow.rename(dt))
    return pd.concat(w_list, axis=1).T


def dual_momentum_truevol(
    prices: pd.DataFrame,
    lookback: int = 126,
    n_long: int = 30,
    target_vol: float = 0.15,
    vol_lookback: int = 60,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """dual_momentum_voltarget with the scale computed from the ACTUAL
    ex-ante covariance vol (engine_v2.ex_ante_book_vol, 60d daily-return
    window ending at the decision date) instead of the correlation-free
    strategies.py:306 formula. Selection identical to the original."""
    px = prices
    ret = px.pct_change(lookback, fill_method=None)
    daily = px.pct_change(fill_method=None)
    vol = daily.rolling(vol_lookback).std() * np.sqrt(252)

    w_list = []
    for i, dt in enumerate(px.index):
        if i % rebal_freq != 0 or i < max(lookback, vol_lookback) + 5:
            continue
        row = ret.loc[dt].dropna()
        row = row[row > 0]
        if len(row) == 0:
            continue
        top = row.nlargest(min(n_long, len(row))).index
        v = vol.loc[dt][top].replace(0.0, np.nan).dropna()
        if v.empty:
            continue
        raw_w = 1.0 / v
        raw_w = raw_w / raw_w.sum()
        window = daily.loc[:dt].tail(vol_lookback)
        ev = ex_ante_book_vol(raw_w, window)          # <- THE FIX
        scale = min(1.0, target_vol / max(ev, 1e-6))
        wrow = pd.Series(0.0, index=px.columns)
        wrow.loc[raw_w.index] = raw_w.values * scale
        w_list.append(wrow.rename(dt))
    return pd.concat(w_list, axis=1).T


def dual_formula_diagnostic(
    prices: pd.DataFrame,
    lookback: int = 126,
    n_long: int = 30,
    target_vol: float = 0.15,
    vol_lookback: int = 60,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Per decision date: old correlation-free vol estimate vs fixed ex-ante
    covariance vol for the dual sleeve's raw inverse-vol book."""
    px = prices
    ret = px.pct_change(lookback, fill_method=None)
    daily = px.pct_change(fill_method=None)
    vol = daily.rolling(vol_lookback).std() * np.sqrt(252)
    rows = []
    for i, dt in enumerate(px.index):
        if i % rebal_freq != 0 or i < max(lookback, vol_lookback) + 5:
            continue
        row = ret.loc[dt].dropna()
        row = row[row > 0]
        if len(row) == 0:
            continue
        top = row.nlargest(min(n_long, len(row))).index
        v = vol.loc[dt][top].replace(0.0, np.nan).dropna()
        if v.empty:
            continue
        raw_w = 1.0 / v
        raw_w = raw_w / raw_w.sum()
        old_est = float(np.sqrt(np.sum((raw_w * vol.loc[dt][raw_w.index]) ** 2)))
        new_est = ex_ante_book_vol(raw_w, daily.loc[:dt].tail(vol_lookback))
        rows.append({"date": dt, "old_vol_est": old_est, "new_vol_est": new_est,
                     "old_binds": old_est > target_vol,
                     "new_binds": new_est > target_vol})
    return pd.DataFrame(rows).set_index("date")


# ─────────────────────────────────────────────────────────────────────────
# Blend + true book vol target overlay
# ─────────────────────────────────────────────────────────────────────────

def blend3(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(set(a.index) | set(b.index) | set(c.index))
    cols = sorted(set(a.columns) | set(b.columns) | set(c.columns))
    return (a.reindex(index=dates, columns=cols, fill_value=0.0)
            + b.reindex(index=dates, columns=cols, fill_value=0.0)
            + c.reindex(index=dates, columns=cols, fill_value=0.0)) / 3.0


def true_book_voltarget(
    blend_1x: pd.DataFrame,
    rets: pd.DataFrame,
    target: float,
    lev: float = LEV,
    window: int = 60,
) -> tuple[pd.DataFrame, pd.Series]:
    """At each decision date, ex-ante vol of the FULL levered book
    (blend_row * lev) from the actual 60d covariance ending at the decision
    date; scale the whole book by min(1, target/exante). Returns the
    POST-leverage frame and the per-date scale series."""
    scales = {}
    for dt, row in blend_1x.iterrows():
        win = rets.loc[:dt].tail(window)
        if len(win) < 40:
            scales[dt] = 1.0
            continue
        ev = ex_ante_book_vol(row * lev, win)
        scales[dt] = 1.0 if ev <= 1e-9 else min(1.0, target / ev)
    s = pd.Series(scales)
    return blend_1x.mul(s, axis=0) * lev, s


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    panel, macro, _ = load_cache()
    px = panel["close"]
    pu = union_prices_cached()

    w_xs = pd.read_parquet("weights_store/sleeve_xs_momentum_n30.parquet")
    w_dual = pd.read_parquet("weights_store/sleeve_dual_momentum_voltarget_n30.parquet")
    w_adapt = pd.read_parquet("weights_store/sleeve_adaptive_voltarget_n30.parquet")
    base_ref = pd.read_parquet("weights_store/combo_v2_base_1x.parquet")

    b_eq = blend3(w_xs, w_dual, w_adapt)
    # Sanity: blend of cached sleeves must reproduce the cached base frame.
    chk = (b_eq.reindex_like(base_ref).fillna(0.0) - base_ref).abs().max().max()
    print(f"sanity: |blend3(sleeves) - combo_v2_base_1x|_max = {chk:.2e}")
    assert chk < 1e-9, "cached sleeves do not reproduce combo_v2_base_1x"

    print("building xs inverse-vol sleeve...")
    w_xs_iv = cached_weights(
        "sleeve_xs_momentum_invvol_n30",
        lambda: xs_momentum_invvol(px, n_long=30, vol_lookback=60),
    )
    print("building dual true-vol(15%) sleeve...")
    w_dual_fix = cached_weights(
        "sleeve_dual_momentum_truevol15_n30",
        lambda: dual_momentum_truevol(px, n_long=30, target_vol=0.15,
                                      vol_lookback=60),
    )

    b_iv = blend3(w_xs_iv, w_dual, w_adapt)
    b_eq_dualfix = blend3(w_xs, w_dual_fix, w_adapt)
    b_iv_dualfix = blend3(w_xs_iv, w_dual_fix, w_adapt)

    rets_pu = pu.pct_change(fill_method=None)
    cols_b = [c for c in b_eq.columns if c in rets_pu.columns]
    rets_b = rets_pu[cols_b]

    results = {}

    def trial(name: str, w_post_lev: pd.DataFrame, params: dict, notes: str = ""):
        r = run_trial(w_post_lev, name=name, family=FAMILY, params=params,
                      window="dev", notes=notes)
        s = r["row"]
        print(f"  {name:<28s} mm={s['mean_monthly']*100:6.3f}%  "
              f"sharpe={s['sharpe']:.3f}  dd={s['max_drawdown']*100:6.1f}%  "
              f"calmar={s['calmar']:.3f}  to={s['turnover_ann']:.1f}")
        results[name] = r
        return r

    print("\n[1/3] baseline + (a) xs inverse-vol")
    trial("e4_base_eq_2x", b_eq * LEV,
          {"xs": "eq", "voltarget": None, "dual": "orig", "lev": LEV},
          notes="baseline reproduction (diagnostic anchor)")
    trial("e4_xs_invvol_2x", b_iv * LEV,
          {"xs": "invvol60", "voltarget": None, "dual": "orig", "lev": LEV},
          notes="(a) xs sleeve 1/sigma60 replaces equal-weight")

    print("\n[2/3] (b) TRUE book vol target (levered basis), 3 targets x 2 xs modes")
    bind_stats = {}
    for tgt in (0.25, 0.30, 0.35):
        for tag, bl in (("eq", b_eq), ("iv", b_iv)):
            wf, sc = true_book_voltarget(bl, rets_b, tgt, lev=LEV, window=60)
            sc_dev = sc.loc[:DEV_CUT]
            frac = float((sc_dev < 0.9999).mean())
            avg_when = float(sc_dev[sc_dev < 0.9999].mean()) if frac > 0 else 1.0
            bind_stats[f"vt{int(tgt*100)}_{tag}"] = {
                "frac_binding_dev": frac, "avg_scale_when_binding": avg_when,
                "min_scale_dev": float(sc_dev.min()),
            }
            trial(f"e4_vt{int(tgt*100)}_{tag}_2x", wf,
                  {"xs": "eq" if tag == "eq" else "invvol60",
                   "voltarget": tgt, "vol_window": 60, "basis": "levered_2x",
                   "dual": "orig", "lev": LEV},
                  notes=f"(b) true book voltarget {tgt}; binds {frac:.0%} of dev "
                        f"decision dates (avg scale {avg_when:.2f} when binding)")

    print("\n[3/3] (c) dual sleeve with FIXED internal 15% formula")
    trial("e4_dualfix_eq_2x", b_eq_dualfix * LEV,
          {"xs": "eq", "voltarget": None, "dual": "truevol15_w60", "lev": LEV},
          notes="(c) dual sleeve ex-ante covariance vol target 15%")
    trial("e4_dualfix_iv_2x", b_iv_dualfix * LEV,
          {"xs": "invvol60", "voltarget": None, "dual": "truevol15_w60",
           "lev": LEV},
          notes="(c)+(a) combined")

    # ── Diagnostics: old vs fixed dual formula binding ────────────────────
    print("\ndiagnostic: dual sleeve old vs fixed vol formula (dev window)")
    diag = dual_formula_diagnostic(px, target_vol=0.15)
    diag_dev = diag.loc[:DEV_CUT]
    diag.to_csv("results/v7/e4_dual_formula_diagnostic.csv")
    old_frac = float(diag_dev["old_binds"].mean())
    new_frac = float(diag_dev["new_binds"].mean())
    print(f"  old formula binds: {old_frac:.1%} of {len(diag_dev)} dev decision dates "
          f"(median est {diag_dev['old_vol_est'].median():.3f})")
    print(f"  fixed formula binds: {new_frac:.1%} "
          f"(median est {diag_dev['new_vol_est'].median():.3f})")

    # ── G1 evaluation ─────────────────────────────────────────────────────
    rows = {n: r["row"] for n, r in results.items()}
    BASE_CALMAR = 0.858019  # baseline.json dev
    G1_CALMAR = 1.10 * BASE_CALMAR

    neighbors = {
        "e4_base_eq_2x":     ["e4_xs_invvol_2x", "e4_vt35_eq_2x", "e4_dualfix_eq_2x"],
        "e4_xs_invvol_2x":   ["e4_base_eq_2x", "e4_vt35_iv_2x", "e4_dualfix_iv_2x"],
        "e4_vt25_eq_2x":     ["e4_vt30_eq_2x", "e4_vt25_iv_2x"],
        "e4_vt30_eq_2x":     ["e4_vt25_eq_2x", "e4_vt35_eq_2x", "e4_vt30_iv_2x"],
        "e4_vt35_eq_2x":     ["e4_vt30_eq_2x", "e4_base_eq_2x", "e4_vt35_iv_2x"],
        "e4_vt25_iv_2x":     ["e4_vt30_iv_2x", "e4_vt25_eq_2x"],
        "e4_vt30_iv_2x":     ["e4_vt25_iv_2x", "e4_vt35_iv_2x", "e4_vt30_eq_2x"],
        "e4_vt35_iv_2x":     ["e4_vt30_iv_2x", "e4_xs_invvol_2x", "e4_vt35_eq_2x"],
        "e4_dualfix_eq_2x":  ["e4_base_eq_2x", "e4_dualfix_iv_2x"],
        "e4_dualfix_iv_2x":  ["e4_xs_invvol_2x", "e4_dualfix_eq_2x"],
    }

    verdicts = {}
    for n, s in rows.items():
        nb = [rows[m]["calmar"] for m in neighbors[n] if m in rows]
        nb_med = float(np.median(nb)) if nb else float("nan")
        stab = nb_med >= 0.85 * s["calmar"]
        g1 = (s["calmar"] >= G1_CALMAR and s["mean_monthly"] >= 0.04
              and s["max_drawdown"] > -0.50 and stab)
        verdicts[n] = {"calmar": s["calmar"], "mm": s["mean_monthly"],
                       "dd": s["max_drawdown"], "neighbor_median_calmar": nb_med,
                       "stability_ok": bool(stab), "g1": bool(g1)}
        print(f"  G1 {n:<26s} calmar={s['calmar']:.3f} (>= {G1_CALMAR:.3f}?) "
              f"mm={s['mean_monthly']*100:.2f}% dd={s['max_drawdown']*100:.1f}% "
              f"nb_med={nb_med:.3f} stab={stab} -> {'PASS' if g1 else 'fail'}")

    with open("results/v7/e4_verdicts.json", "w") as f:
        json.dump({"verdicts": verdicts, "bind_stats": bind_stats,
                   "dual_formula": {"old_binds_frac_dev": old_frac,
                                    "new_binds_frac_dev": new_frac}},
                  f, indent=2, default=str)
    print("\nwrote results/v7/e4_verdicts.json and "
          "results/v7/e4_dual_formula_diagnostic.csv")


if __name__ == "__main__":
    main()
