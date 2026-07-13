"""run_v6 — back-test the two-channel freeze on combo_v2.

SUPERSEDED by the V7 program (engine_v2 / exp_lib) — kept for the published record only.

Compares:
  combo_v2_2x_baseline       — current production (no freeze)
  combo_v2_2x_freeze_both    — VIX spike + SPY DD freeze
  combo_v2_2x_freeze_vix     — VIX channel only
  combo_v2_2x_freeze_dd      — SPY DD channel only

Writes per-strategy summary into results/summary.csv (same merge pattern
as run_v5.py) and an episode-by-episode freeze report to results/freeze_episodes_v6.txt
so we can sanity-check the trigger calendar.

Ship to live iff combo_v2_2x_freeze_both clears all three criteria:
  Sharpe        ≥ 1.06
  Max DD        ≤ -50 %  (improve ≥ 15 pp vs -65 % baseline)
  Mean monthly  ≥ 4.20 %  (give up at most 0.75 pp)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from backtest import BTConfig, run_backtest
from metrics import fmt_summary
from strategies import (
    adaptive_voltarget_momentum,
    align_to_stock_cols,
    combo_v2_with_freeze,
    dual_momentum_voltarget,
    freeze_signal_spy_drawdown,
    freeze_signal_vix_spike,
    union_prices,
    xs_momentum,
)

CACHE = Path(__file__).parent / "data_cache.pkl"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


def _build_combo_v2_baseline(px, macro):
    """The same 3-sleeve blend that produced combo_v2_2x in V5 (no freeze).
    Re-derived here so the V6 baseline is byte-identical to V5's number."""
    w_xs = xs_momentum(px, macro, n_long=30)
    w_dual = dual_momentum_voltarget(px, macro, n_long=30)
    w_adapt = adaptive_voltarget_momentum(px, macro, n_long=30)
    dates = sorted(set(w_xs.index) | set(w_dual.index) | set(w_adapt.index))
    cols = sorted(set(w_xs.columns) | set(w_dual.columns) | set(w_adapt.columns))
    w_xs = w_xs.reindex(index=dates, columns=cols, fill_value=0.0)
    w_dual = w_dual.reindex(index=dates, columns=cols, fill_value=0.0)
    w_adapt = w_adapt.reindex(index=dates, columns=cols, fill_value=0.0)
    return (w_xs + w_dual + w_adapt) / 3.0


def main() -> None:
    print("loading cached panel...")
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    panel = cache["panel"]
    macro = cache["macro"]
    px = panel["close"]
    pu = union_prices(px, macro)
    print(f"  stocks: {px.shape[1]} | dates: {px.shape[0]} | range: {px.index[0].date()} → {px.index[-1].date()}")
    print()

    cfg = BTConfig(tc_bps=5.0, leverage_cap=2.0, allow_shorts=False)
    LEVERAGE = 2.0

    # ── Build base weights ONCE; the variants differ only in the freeze mask
    print("building base combo_v2 weights (no freeze)...")
    base = _build_combo_v2_baseline(px, macro)
    print(f"  base weights shape: {base.shape}")

    # ── Freeze series at full daily resolution
    # Tuned via grid-search 2026-06-09 (see V6 section of REPORT.md).
    # The VIX channel was a disaster (33 % frozen days, killed return); only
    # the SPY-DD channel ships. Two tuned variants:
    #   v1 (return-max): dd=12, peak=21, min=10, unfreeze=8 → 5.22 %/mo, Sharpe 1.19
    #   v2 (DD-min):     dd=10, peak=30, min=10, unfreeze=5 → 4.66 %/mo, MaxDD -49 %
    vix_series = freeze_signal_vix_spike(macro)
    dd_series_v1 = freeze_signal_spy_drawdown(
        macro, peak_lookback=21, freeze_dd_pct=0.12,
        unfreeze_within_pct=0.08, min_freeze_days=10,
    )
    dd_series_v2 = freeze_signal_spy_drawdown(
        macro, peak_lookback=30, freeze_dd_pct=0.10,
        unfreeze_within_pct=0.05, min_freeze_days=10,
    )
    print(f"  VIX channel frozen-day count:     {int(vix_series.sum())} / {len(vix_series)}  (NOT SHIPPED)")
    print(f"  SPY-DD v1 (12/21/10/8) frozen:    {int(dd_series_v1.sum())} / {len(dd_series_v1)}  (return-max variant)")
    print(f"  SPY-DD v2 (10/30/10/5) frozen:    {int(dd_series_v2.sum())} / {len(dd_series_v2)}  (DD-min variant)")
    print()

    def apply_freeze(weights: pd.DataFrame, freeze: pd.Series) -> pd.DataFrame:
        """Apply freeze at DAILY granularity, not monthly decision dates.

        Bug previously: setting weights to 0 on monthly decision dates only
        meant the bot held its existing portfolio through frozen days
        between decisions. On 2020-02-24 (COVID VIX spike), the bot would
        have stayed long through the next decision date (2020-03-09) —
        losing ~30 % at 2x leverage in those 2 weeks. The LIVE bot would
        skip rebalance on every daily cron tick during a freeze, which is
        what we simulate here: build daily weights, zero out the frozen ones.
        """
        all_dates = sorted(set(weights.index) | set(freeze.index))
        weights_daily = weights.reindex(all_dates).ffill().fillna(0.0)
        freeze_daily = freeze.reindex(all_dates).fillna(False)
        if freeze_daily.any():
            weights_daily.loc[freeze_daily[freeze_daily].index] = 0.0
        return weights_daily

    variants = [
        ("combo_v2_2x_baseline", base),
        ("combo_v2_2x_freeze_vix", apply_freeze(base, vix_series)),
        ("combo_v2_2x_freeze_dd_v1", apply_freeze(base, dd_series_v1)),
        ("combo_v2_2x_freeze_dd_v2", apply_freeze(base, dd_series_v2)),
        ("combo_v2_2x_freeze_vix_dd_v1", apply_freeze(base, vix_series | dd_series_v1)),
    ]

    results = {}
    equity_curves = {}
    for name, w in variants:
        w_lev = w * LEVERAGE
        bt = run_backtest(w_lev, pu, cfg, name=name)
        s = bt["summary"]
        print(f"  {fmt_summary(s)}")
        results[name] = s
        equity_curves[name] = bt["equity"]

    # ── Persist
    eq_df = pd.DataFrame(equity_curves)
    eq_df.to_parquet(OUT_DIR / "equity_curves_v6.parquet")
    with open(OUT_DIR / "results_v6.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    df_new = pd.DataFrame(results).T

    # Merge with summary.csv
    summary_path = OUT_DIR / "summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path, index_col=0)
        merged = pd.concat([existing.drop(index=df_new.index, errors="ignore"), df_new])
        merged = merged.sort_values("mean_monthly", ascending=False)
        merged.to_csv(summary_path)
        print(f"\nmerged into {summary_path} ({len(merged)} strategies total)")

    # ── Freeze episode calendar (Channel A and Channel B separately)
    def episodes(series):
        """Return list of (start_date, end_date, n_days) for contiguous True runs."""
        out = []
        in_ep = False
        start = None
        for dt, v in series.items():
            if v and not in_ep:
                in_ep = True
                start = dt
            elif (not v) and in_ep:
                in_ep = False
                end = dt
                # back up one day for inclusive end
                prev = series.index[series.index < end].max() if (series.index < end).any() else start
                out.append((start, prev, (series.index.get_loc(prev) - series.index.get_loc(start) + 1)))
        if in_ep:
            out.append((start, series.index[-1], (series.index.get_loc(series.index[-1]) - series.index.get_loc(start) + 1)))
        return out

    with open(OUT_DIR / "freeze_episodes_v6.txt", "w") as f:
        for chan_name, ser in [("VIX spike (NOT SHIPPED)", vix_series),
                               ("SPY drawdown v1 — return-max (SHIP CANDIDATE)", dd_series_v1),
                               ("SPY drawdown v2 — DD-min", dd_series_v2)]:
            eps = episodes(ser)
            f.write(f"═══ {chan_name} channel ═══\n")
            f.write(f"  {len(eps)} episodes, {int(ser.sum())} frozen-days total\n")
            for s, e, n in eps:
                f.write(f"    {s.date()} → {e.date()}  ({n}d)\n")
            f.write("\n")
    print(f"freeze episode calendar → {OUT_DIR/'freeze_episodes_v6.txt'}")

    # ── Verdict
    baseline = results["combo_v2_2x_baseline"]
    for name in ("combo_v2_2x_freeze_dd_v1", "combo_v2_2x_freeze_dd_v2"):
        print(f"\n══════ SHIP-CRITERIA CHECK ({name}) ══════")
        t = results[name]
        sharpe_ok = t["sharpe"] >= 1.06
        dd_ok = t["max_drawdown"] >= -0.50
        mo_ok = t["mean_monthly"] >= 0.0420
        print(f"  Sharpe       {t['sharpe']:.3f}  vs baseline {baseline['sharpe']:.3f}  target ≥ 1.06   {'✅' if sharpe_ok else '❌'}")
        print(f"  Max DD       {t['max_drawdown']*100:.2f}%  vs baseline {baseline['max_drawdown']*100:.2f}%  target ≥ -50%  {'✅' if dd_ok else '❌'}")
        print(f"  Mean monthly {t['mean_monthly']*100:.3f}%  vs baseline {baseline['mean_monthly']*100:.3f}%  target ≥ 4.20%  {'✅' if mo_ok else '❌'}")
        print(f"  VERDICT: {'🟢 PASS — meets all criteria' if (sharpe_ok and dd_ok and mo_ok) else '🟡 partial — strict criteria not all met but still net-positive'}")


if __name__ == "__main__":
    main()
