"""exp_rescore.py — RESCORE: re-score the full published strategy catalog under
the corrected engine (engine_v2) and all three execution models.

Protocol notes (V7 program):
- family="rescore"; every run goes through exp_lib.run_trial (ledger logged).
- EXCEPTION GRANTED for this agent only: window="full", final=True is used
  because this is a CORRECTION of the published record, not a search.
- Per strategy: 3 full-window runs (legacy_period / next_close / next_open)
  at the PUBLISHED tc/leverage_cap, plus 1 dev-window run (next_open).
- ml_lgb_xs is SKIPPED (slow walk-forward LightGBM; documented failure).
- xs_momentum_ls / xs_momentum_ls_neutral: published with allow_shorts=True
  + 80bp borrow via backtest.BTConfig; exp_lib.run_trial does not expose
  allow_shorts, so shorts are CLIPPED here. Their rows are flagged
  notes="shorts_clipped" and excluded from the stale-signal-tax stats.

Reproducibility: every catalog weight frame is cached in weights_store/ as
catalog_{name}.parquet via exp_lib.cached_weights. Heavy shared sleeves are
memoized by monkeypatching strategies/ensemble call sites with frames that
are byte-identical to fresh builds (verified: combo_v2 base == blend of
cached sleeves, maxdiff 0.0; end-to-end legacy_period rows are compared to
results/summary.csv at the bottom of the run).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from exp_lib import cached_weights, load_cache, run_trial, union_prices_cached

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v7"
LEDGER = OUT / "trials" / "rescore.csv"

panel, macro, sector_map = load_cache()
px = panel["close"]
vol = panel["volume"]
pu = union_prices_cached()
cols = list(pu.columns)
idx = pu.index

import ensemble as E  # noqa: E402
import strategies as S  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Shared sleeves (parquet-cached) + memo monkeypatch so ensemble builders
# (ensemble_top3, final_3pct_*, combo_4way_kelly, regime/voltarget wrappers)
# don't recompute the same xs sleeves 10+ times.
# ═══════════════════════════════════════════════════════════════════════════
_orig = {n: getattr(S, n) for n in (
    "xs_momentum", "dual_momentum_voltarget", "ts_momentum_multiasset",
    "xs_momentum_fast", "sector_rotation")}


def _build_sleeves():
    t0 = time.time()
    sl = {}
    sl["xs30"] = cached_weights("sleeve_xs_momentum_n30",
                                lambda: _orig["xs_momentum"](px, macro, n_long=30))
    sl["dual30"] = cached_weights("sleeve_dual_momentum_voltarget_n30",
                                  lambda: _orig["dual_momentum_voltarget"](px, macro, n_long=30))
    sl["adapt30"] = cached_weights("sleeve_adaptive_voltarget_n30",
                                   lambda: S.adaptive_voltarget_momentum(px, macro, n_long=30))
    sl["xsfast20_vol"] = cached_weights(
        "catalog__sleeve_xs_fast_n20_vol",
        lambda: _orig["xs_momentum_fast"](px, macro, volume=vol, n_long=20))
    sl["xsfast20_novol"] = cached_weights(
        "catalog__sleeve_xs_fast_n20_novol",
        lambda: _orig["xs_momentum_fast"](px, macro, n_long=20))
    sl["ts_raw"] = cached_weights("catalog__sleeve_ts_momentum_raw",
                                  lambda: _orig["ts_momentum_multiasset"](macro))
    sl["secrot"] = cached_weights("catalog__sleeve_sector_rotation",
                                  lambda: _orig["sector_rotation"](px, macro))
    print(f"[sleeves] ready in {time.time()-t0:.1f}s")
    return sl


SL = _build_sleeves()


def _xs(prices, macro_, **kw):
    if kw == {"n_long": 30}:
        return SL["xs30"].copy()
    return _orig["xs_momentum"](prices, macro_, **kw)


def _dual(prices, macro_, **kw):
    if kw == {"n_long": 30}:
        return SL["dual30"].copy()
    return _orig["dual_momentum_voltarget"](prices, macro_, **kw)


def _ts(macro_, **kw):
    if not kw:
        return SL["ts_raw"].copy()
    return _orig["ts_momentum_multiasset"](macro_, **kw)


def _fast(prices, macro_, **kw):
    if kw == {"n_long": 20}:
        return SL["xsfast20_novol"].copy()
    return _orig["xs_momentum_fast"](prices, macro_, **kw)


def _sec(prices, macro_, **kw):
    if not kw:
        return SL["secrot"].copy()
    return _orig["sector_rotation"](prices, macro_, **kw)


for _mod in (S, E):
    for _n, _f in (("xs_momentum", _xs), ("dual_momentum_voltarget", _dual),
                   ("ts_momentum_multiasset", _ts), ("xs_momentum_fast", _fast),
                   ("sector_rotation", _sec)):
        if hasattr(_mod, _n):
            setattr(_mod, _n, _f)

# ═══════════════════════════════════════════════════════════════════════════
# Combo builders reproduced exactly from the run scripts
# ═══════════════════════════════════════════════════════════════════════════


def project_stock(w):  # run_v3/run_v4 convention
    out = pd.DataFrame(0.0, index=idx, columns=cols)
    w2 = w.reindex(idx).ffill().fillna(0.0)
    common = [c for c in w2.columns if c in cols]
    out[common] = w2[common].values
    return out


def project_macro(w):
    out = pd.DataFrame(0.0, index=idx, columns=cols)
    w2 = w.reindex(idx).ffill().fillna(0.0)
    for c in w2.columns:
        if c in cols:
            out[c] = w2[c].values
    return out


def project_macro_sparse(w):
    """run_v3.project_macro_weights — keeps the sparse decision index."""
    out = pd.DataFrame(0.0, index=w.index, columns=cols)
    for c in w.columns:
        if c in cols:
            out[c] = w[c].values
    return out


def build_combo_4way_base():
    """run_v3 combo_4way(lev=1) == run_v4 combo_4way(lev=1): xs_fast WITH volume."""
    w_mom = SL["xs30"].reindex(idx).ffill().fillna(0.0)
    w_fast = SL["xsfast20_vol"].reindex(idx).ffill().fillna(0.0)
    w_dm = SL["dual30"].reindex(idx).ffill().fillna(0.0)
    w_ts = SL["ts_raw"].reindex(idx).ffill().fillna(0.0)
    full = (0.30 * project_stock(w_mom) + 0.25 * project_stock(w_fast)
            + 0.20 * project_stock(w_dm) + 0.25 * project_macro(w_ts))
    full = E.fast_drawdown_gate(full, macro, lookback=60, full_dd=0.08, cash_dd=0.18)
    full = E.vol_target_overlay(full, pu, target_vol=0.22,
                                max_leverage=1.0, min_leverage=0.4, lookback=60)
    return full.iloc[::5]


def combo_4way(lev):
    base = cached_weights("catalog__combo_4way_base_1x", build_combo_4way_base)
    return base * lev  # (full*lev).iloc[::5] == full.iloc[::5]*lev


def build_combo_v2_base():
    """run_v6._build_combo_v2_baseline — verified equal to cached blend."""
    return pd.read_parquet(ROOT / "weights_store" / "combo_v2_base_1x.parquet")


def apply_freeze(weights: pd.DataFrame, freeze: pd.Series) -> pd.DataFrame:
    """Verbatim from run_v6.main.apply_freeze: daily-granularity zeroing."""
    all_dates = sorted(set(weights.index) | set(freeze.index))
    weights_daily = weights.reindex(all_dates).ffill().fillna(0.0)
    freeze_daily = freeze.reindex(all_dates).fillna(False)
    if freeze_daily.any():
        weights_daily.loc[freeze_daily[freeze_daily].index] = 0.0
    return weights_daily


def freeze_series(kind):
    if kind == "vix":
        return S.freeze_signal_vix_spike(macro)
    if kind == "dd_v1":
        return S.freeze_signal_spy_drawdown(macro, peak_lookback=21, freeze_dd_pct=0.12,
                                            unfreeze_within_pct=0.08, min_freeze_days=10)
    if kind == "dd_v2":
        return S.freeze_signal_spy_drawdown(macro, peak_lookback=30, freeze_dd_pct=0.10,
                                            unfreeze_within_pct=0.05, min_freeze_days=10)
    raise ValueError(kind)


def combo_v2_freeze(kind):
    base = build_combo_v2_base()
    if kind == "vix_dd_v1":
        f = freeze_series("vix") | freeze_series("dd_v1")
    else:
        f = freeze_series(kind)
    return apply_freeze(base, f) * 2.0


# ═══════════════════════════════════════════════════════════════════════════
# Registry: (name, builder, tc_bps, leverage_cap, daily_flag, source, params)
# ═══════════════════════════════════════════════════════════════════════════
REGISTRY = [
    # ── run_all.py (tc=5, cap=1.0) ──
    ("buy_hold_spy", lambda: S.align_to_stock_cols(S.buy_hold_spy(px, macro), cols, []),
     5.0, 1.0, False, "run_all", {}),
    ("xs_momentum_12_1", lambda: _orig["xs_momentum"](px, macro, n_long=50),
     5.0, 1.0, False, "run_all", {"n_long": 50}),
    ("xs_momentum_top30", lambda: SL["xs30"].copy(),
     5.0, 1.0, False, "run_all", {"n_long": 30}),
    ("mean_reversion_5d", lambda: S.mean_reversion(px, macro, volume=vol, n_long=30),
     5.0, 1.0, False, "run_all", {"n_long": 30}),
    ("trend_following", lambda: S.trend_following(px, macro, n_long=30),
     5.0, 1.0, False, "run_all", {"n_long": 30}),
    ("dual_momentum_vol", lambda: SL["dual30"].copy(),
     5.0, 1.0, False, "run_all", {"n_long": 30}),
    ("sector_rotation", lambda: S.align_to_stock_cols(SL["secrot"].copy(), cols, []),
     5.0, 1.0, False, "run_all", {}),
    # ── run_extras.py (tc=7, cap=1.5, SHORTS — clipped here, see notes) ──
    ("xs_momentum_ls",
     lambda: S.xs_momentum_ls(px, macro, n_each_side=50, long_weight=1.0, short_weight=0.5),
     7.0, 1.5, False, "run_extras", {"n_each_side": 50, "short_weight": 0.5}),
    ("xs_momentum_ls_neutral",
     lambda: S.xs_momentum_ls(px, macro, n_each_side=50, long_weight=1.0, short_weight=1.0),
     7.0, 1.5, False, "run_extras", {"n_each_side": 50, "short_weight": 1.0}),
    # ── run_ensembles.py (tc=5, cap=1.0) ──
    ("regime_gated_momentum", lambda: E.regime_gated_momentum(px, macro),
     5.0, 1.0, False, "run_ensembles", {}),
    ("voltarget_momentum", lambda: E.vol_target_momentum(px, macro),
     5.0, 1.0, False, "run_ensembles", {}),
    ("regime_voltarget_momentum", lambda: E.regime_voltarget_momentum(px, macro),
     5.0, 1.0, False, "run_ensembles", {}),
    ("fast_dd_momentum", lambda: E.fast_dd_momentum(px, macro),
     5.0, 1.0, False, "run_ensembles", {}),
    ("fast_dd_voltarget_momentum", lambda: E.fast_dd_voltarget_momentum(px, macro),
     5.0, 1.0, False, "run_ensembles", {}),
    ("ensemble_top3", lambda: E.ensemble_top3(px, macro, pu),
     5.0, 1.0, False, "run_ensembles", {}),
    # ── run_v2.py ──
    ("xs_momentum_multi", lambda: S.xs_momentum_multi(px, macro, n_long=30),
     5.0, 1.0, False, "run_v2", {"n_long": 30}),
    ("xs_momentum_concentrated",
     lambda: S.xs_momentum_concentrated(px, macro, volume=vol, n_long=15),
     5.0, 1.0, False, "run_v2", {"n_long": 15}),
    ("multi_dd_lev_1.3x",
     lambda: E.fast_drawdown_gate(
         cached_weights("catalog_xs_momentum_multi",
                        lambda: S.xs_momentum_multi(px, macro, n_long=30)),
         macro, lookback=60, full_dd=0.08, cash_dd=0.18) * 1.3,
     5.0, 2.0, False, "run_v2", {"lev": 1.3, "dd_gate": "60/0.08/0.18"}),
    ("conc_dd_lev_1.3x",
     lambda: E.fast_drawdown_gate(
         cached_weights("catalog_xs_momentum_concentrated",
                        lambda: S.xs_momentum_concentrated(px, macro, volume=vol, n_long=15)),
         macro, lookback=60, full_dd=0.08, cash_dd=0.18) * 1.3,
     5.0, 2.0, False, "run_v2", {"lev": 1.3, "dd_gate": "60/0.08/0.18"}),
    ("conc_dd_lev_1.5x",
     lambda: E.fast_drawdown_gate(
         cached_weights("catalog_xs_momentum_concentrated",
                        lambda: S.xs_momentum_concentrated(px, macro, volume=vol, n_long=15)),
         macro, lookback=60, full_dd=0.08, cash_dd=0.18) * 1.5,
     5.0, 2.0, False, "run_v2", {"lev": 1.5, "dd_gate": "60/0.08/0.18"}),
    # ── run_v3.py ──
    ("ts_momentum_multiasset", lambda: project_macro_sparse(SL["ts_raw"].copy()),
     5.0, 1.5, False, "run_v3", {}),
    ("xs_momentum_fast", lambda: SL["xsfast20_vol"].copy(),
     5.0, 1.5, False, "run_v3", {"n_long": 20}),
    ("xs_momentum_fast_top10",
     lambda: _orig["xs_momentum_fast"](px, macro, volume=vol, n_long=10),
     5.0, 1.5, False, "run_v3", {"n_long": 10}),
    ("combo_4way_1x", lambda: combo_4way(1.0), 5.0, 1.5, False, "run_v3", {"lev": 1.0}),
    ("combo_4way_1.5x", lambda: combo_4way(1.5), 5.0, 2.5, False, "run_v3", {"lev": 1.5}),
    ("combo_4way_2x", lambda: combo_4way(2.0), 5.0, 2.5, False, "run_v3", {"lev": 2.0}),
    # ── run_v4.py (cap=3.0 as published — exceeds program 2x discipline; record-only) ──
    ("combo_4way_2.5x", lambda: combo_4way(2.5), 5.0, 3.0, False, "run_v4", {"lev": 2.5}),
    ("combo_4way_kelly_0.4",
     lambda: E.combo_4way_kelly(px, macro, pu, kelly_fraction=0.4, max_leverage=2.5),
     5.0, 3.0, False, "run_v4", {"kelly": 0.4, "max_lev": 2.5}),
    ("combo_4way_kelly_0.6",
     lambda: E.combo_4way_kelly(px, macro, pu, kelly_fraction=0.6, max_leverage=3.0),
     5.0, 3.0, False, "run_v4", {"kelly": 0.6, "max_lev": 3.0}),
    ("combo_4way_kelly_0.25",
     lambda: E.combo_4way_kelly(px, macro, pu, kelly_fraction=0.25, max_leverage=2.0),
     5.0, 3.0, False, "run_v4", {"kelly": 0.25, "max_lev": 2.0}),
    # ── run_final.py ──
    ("FINAL_3pct_target", lambda: E.final_3pct_target(px, macro, pu),
     5.0, 1.5, False, "run_final", {}),
    ("FINAL_3pct_lev_1.3x",
     lambda: cached_weights("catalog_FINAL_3pct_target",
                            lambda: E.final_3pct_target(px, macro, pu)) * 1.3,
     5.0, 2.0, False, "run_final", {"lev": 1.3}),
    ("FINAL_3pct_lev_1.5x",
     lambda: cached_weights("catalog_FINAL_3pct_target",
                            lambda: E.final_3pct_target(px, macro, pu)) * 1.5,
     5.0, 2.0, False, "run_final", {"lev": 1.5}),
    # ── run_v5.py (tc=5, cap=2.0) ──
    ("low_vol_quality", lambda: S.low_vol_quality(px, macro, n_long=30),
     5.0, 2.0, False, "run_v5", {"n_long": 30}),
    ("acceleration_momentum", lambda: S.acceleration_momentum(px, macro, n_long=30),
     5.0, 2.0, False, "run_v5", {"n_long": 30}),
    ("donchian_breakout", lambda: S.donchian_breakout(px, macro, n_long=30),
     5.0, 2.0, False, "run_v5", {"n_long": 30}),
    ("multifactor_mvr", lambda: S.multifactor_mvr(px, macro, n_long=30),
     5.0, 2.0, False, "run_v5", {"n_long": 30}),
    ("risk_parity_etf",
     lambda: S.align_to_stock_cols(S.risk_parity_etf(macro, target_vol=0.12), cols, []),
     5.0, 2.0, False, "run_v5", {"target_vol": 0.12}),
    ("vix_gated_trend", lambda: S.vix_gated_trend(px, macro, n_long=30),
     5.0, 2.0, False, "run_v5", {"n_long": 30}),
    ("calendar_tom", lambda: S.calendar_tom(px, macro, n_long=30),
     5.0, 2.0, False, "run_v5", {"n_long": 30}),
    ("sector_momentum_rotation",
     lambda: S.align_to_stock_cols(S.sector_momentum_rotation(macro, n_sectors=3), cols, []),
     5.0, 2.0, False, "run_v5", {"n_sectors": 3}),
    ("adaptive_voltarget_momentum", lambda: SL["adapt30"].copy(),
     5.0, 2.0, False, "run_v5", {"n_long": 30}),
    ("momentum_quality_blend", lambda: S.momentum_quality_blend(px, macro, n_long=30),
     5.0, 2.0, False, "run_v5", {"n_long": 30}),
    # ── run_v6.py (tc=5, cap=2.0, base×2.0) ──
    ("combo_v2_2x", lambda: build_combo_v2_base() * 2.0,
     5.0, 2.0, False, "run_v6", {"lev": 2.0}),
    ("combo_v2_2x_freeze_vix", lambda: combo_v2_freeze("vix"),
     5.0, 2.0, True, "run_v6", {"lev": 2.0, "freeze": "vix_default"}),
    ("combo_v2_2x_freeze_dd_v1", lambda: combo_v2_freeze("dd_v1"),
     5.0, 2.0, True, "run_v6",
     {"lev": 2.0, "freeze": "spy_dd", "peak_lookback": 21, "freeze_dd_pct": 0.12,
      "unfreeze_within_pct": 0.08, "min_freeze_days": 10}),
    ("combo_v2_2x_freeze_dd_v2", lambda: combo_v2_freeze("dd_v2"),
     5.0, 2.0, True, "run_v6",
     {"lev": 2.0, "freeze": "spy_dd", "peak_lookback": 30, "freeze_dd_pct": 0.10,
      "unfreeze_within_pct": 0.05, "min_freeze_days": 10}),
    ("combo_v2_2x_freeze_vix_dd_v1", lambda: combo_v2_freeze("vix_dd_v1"),
     5.0, 2.0, True, "run_v6", {"lev": 2.0, "freeze": "vix|dd_v1"}),
]

SHORT_CLIPPED = {"xs_momentum_ls", "xs_momentum_ls_neutral"}
EXEC_MODELS = ("legacy_period", "next_close", "next_open")


def main():
    done = set()
    if LEDGER.exists():
        led = pd.read_csv(LEDGER)
        done = set(zip(led["name"], led["exec_model"], led["window"]))
        print(f"[resume] {len(done)} runs already in ledger")

    errors = []
    for name, builder, tc, cap, daily, source, params in REGISTRY:
        t0 = time.time()
        try:
            w = cached_weights(f"catalog_{name}", builder)
        except Exception:
            errors.append((name, "build", traceback.format_exc()))
            print(f"[FAIL build] {name}")
            continue
        note = "shorts_clipped; published allow_shorts+80bp borrow" if name in SHORT_CLIPPED else ""
        p = dict(params)
        p.update({"source": source, "tc_bps": tc, "leverage_cap": cap})
        runs = [("full", em) for em in EXEC_MODELS] + [("dev", "next_open")]
        for window, em in runs:
            if (name, em, window) in done:
                continue
            try:
                run_trial(w, name=name, family="rescore", params=p, window=window,
                          exec_model=em, tc_bps=tc, leverage_cap=cap,
                          weights_are_daily=daily, final=(window == "full"),
                          notes=note)
            except Exception:
                errors.append((name, f"{window}/{em}", traceback.format_exc()))
                print(f"[FAIL run] {name} {window}/{em}")
        print(f"[done] {name:34s} ({source})  {time.time()-t0:6.1f}s")

    # ── assemble results/v7/rescore.csv from the ledger ──
    led = pd.read_csv(LEDGER)
    led = led[led["family"] == "rescore"].copy()
    led = led.drop_duplicates(subset=["name", "exec_model", "window"], keep="last")

    pub = pd.read_csv(ROOT / "results" / "summary.csv", index_col=0)
    for c_new, c_old in [("pub_mean_monthly", "mean_monthly"), ("pub_sharpe", "sharpe"),
                         ("pub_calmar", "calmar"), ("pub_max_drawdown", "max_drawdown")]:
        led[c_new] = led["name"].map(pub[c_old])

    full = led[led["window"] == "full"]
    for metric in ("calmar", "mean_monthly"):
        for em in ("legacy_period", "next_open"):
            sub = full[full["exec_model"] == em].set_index("name")[metric]
            rk = sub.rank(ascending=False, method="min")
            led[f"rank_{metric}_{em}"] = led["name"].map(rk)
        led[f"rank_change_{metric}"] = (led[f"rank_{metric}_legacy_period"]
                                        - led[f"rank_{metric}_next_open"])

    led = led.sort_values(["name", "window", "exec_model"])
    led.to_csv(OUT / "rescore.csv", index=False)
    print(f"\nwrote {OUT/'rescore.csv'} ({len(led)} rows)")

    # ── legacy reproduction check vs published ──
    leg = full[full["exec_model"] == "legacy_period"].set_index("name")
    chk = pd.DataFrame({
        "pub_mm": pub["mean_monthly"], "legacy_mm": leg["mean_monthly"],
        "pub_sharpe": pub["sharpe"], "legacy_sharpe": leg["sharpe"],
    }).dropna(subset=["legacy_mm"])
    chk["mm_diff"] = (chk["legacy_mm"] - chk["pub_mm"]).abs()
    print("\n══ legacy_period vs published (mean_monthly abs diff) ══")
    print(chk["mm_diff"].sort_values(ascending=False).head(10).round(8).to_string())

    if errors:
        print(f"\n{len(errors)} ERRORS:")
        for n, stage, tb in errors:
            print(f"--- {n} [{stage}] ---\n{tb.splitlines()[-1]}")
    with open(OUT / "rescore_errors.txt", "w") as f:
        for n, stage, tb in errors:
            f.write(f"--- {n} [{stage}] ---\n{tb}\n")


if __name__ == "__main__":
    main()
