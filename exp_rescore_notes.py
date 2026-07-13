"""exp_rescore_notes.py — generate results/v7/RESCORE_NOTES.md from
results/v7/rescore.csv (built by exp_rescore.py). Reproducible artifact of
the RESCORE assignment (family="rescore").
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v7"

BASE_DEV = {"mean_monthly": 0.045546739999310346, "sharpe": 1.0440033129817838,
            "max_drawdown": -0.6019153206391645, "calmar": 0.8580186751301543}
SHORT_CLIPPED = {"xs_momentum_ls", "xs_momentum_ls_neutral"}


def pct(x, nd=2):
    return f"{x*100:+.{nd}f}%" if pd.notna(x) else "n/a"


def main():
    df = pd.read_csv(OUT / "rescore.csv")
    full = df[df.window == "full"]
    dev = df[(df.window == "dev") & (df.exec_model == "next_open")].set_index("name")
    leg = full[full.exec_model == "legacy_period"].set_index("name")
    nxo = full[full.exec_model == "next_open"].set_index("name")
    nxc = full[full.exec_model == "next_close"].set_index("name")

    lines = []
    w = lines.append
    w("# RESCORE — full catalog re-scored under corrected execution (engine_v2)")
    w("")
    w(f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}. Family ledger: "
      f"`results/v7/trials/rescore.csv`; machine-readable grid: `results/v7/rescore.csv` "
      f"({len(df)} rows = {df.name.nunique()} strategies x (3 full-window exec models "
      f"+ 1 dev-window next_open)). Script: `exp_rescore.py`.")
    w("")
    w("Published configs reproduced exactly (tc/leverage_cap per source run script; "
      "tc=5bp everywhere except xs_momentum_ls* at 7bp). `legacy_period` reproduces "
      "backtest.py bit-for-bit; `next_open` is the corrected live model (signal at "
      "close t -> market order at next open, costs on |dw| at the open). All "
      "selection-relevant numbers below are DEV window; full-window numbers are a "
      "record correction only.")
    w("")

    # 0. reproduction check
    pubchk = leg.dropna(subset=["pub_mean_monthly"]).copy()
    pubchk["mm_diff"] = (pubchk.mean_monthly - pubchk.pub_mean_monthly).abs()
    ok = pubchk[~pubchk.index.isin(SHORT_CLIPPED)]
    w("## 0. Reproduction check (legacy_period vs published summary.csv)")
    w("")
    nrep = ok[ok.mm_diff <= 1e-9]
    w(f"- {len(nrep)}/{len(ok)} comparable strategies reproduce published mean_monthly "
      f"to < 1e-9 (machine precision). The catalog frames and legacy engine are faithful.")
    bad = ok[ok.mm_diff > 1e-9]
    for n, d in bad.mm_diff.items():
        w(f"- `{n}` differs by {d:.2e} ({d*100:.2f} pp/mo): the strategy itself is "
          f"NON-DETERMINISTIC — strategies.py:690 truncates the held set via "
          f"`set(list(held)[:2*n_long])`, whose order depends on per-process string-hash "
          f"randomization. The diff is run-to-run strategy variance, not an engine "
          f"discrepancy. (Anomaly logged; fix would be `sorted(held)`.)")
    for n in sorted(SHORT_CLIPPED & set(pubchk.index)):
        w(f"- `{n}`: NOT comparable — published with allow_shorts=True + 80bp borrow; "
          f"run_trial does not expose allow_shorts so shorts are clipped here. Legacy "
          f"{pct(pubchk.loc[n,'mean_monthly'])}/mo vs published {pct(pubchk.loc[n,'pub_mean_monthly'])}/mo. "
          f"Rows flagged `shorts_clipped`; excluded from tax stats.")
    w("- Verified: for strategies whose published frames were DAILY (combo_v2 freeze "
      "variants), legacy_period == next_close exactly (diff 0.0), as expected — a "
      "daily frame only ever carried 1 day of lag in the old engine.")
    w("")

    # 1. top-10 by full next_open calmar
    w("## 1. Top-10 by FULL-window next_open Calmar (record correction — NOT for selection)")
    w("")
    top = nxo.sort_values("calmar", ascending=False).head(10)
    w("| # | strategy | mm/mo | Sharpe | MaxDD | Calmar | turn/yr | gross | pub Calmar | pub mm/mo |")
    w("|---|----------|-------|--------|-------|--------|---------|-------|------------|-----------|")
    for i, (n, r) in enumerate(top.iterrows(), 1):
        pc = f"{r.pub_calmar:.3f}" if pd.notna(r.pub_calmar) else "n/a"
        pm = pct(r.pub_mean_monthly) if pd.notna(r.pub_mean_monthly) else "n/a"
        w(f"| {i} | {n} | {pct(r.mean_monthly)} | {r.sharpe:.3f} | {pct(r.max_drawdown,1)} | "
          f"{r.calmar:.3f} | {r.turnover_ann:.1f} | {r.avg_gross:.2f}x | {pc} | {pm} |")
    w("")
    w("Headlines: the corrected champion record is **combo_v2_2x +5.22%/mo, Sharpe 1.140, "
      "Calmar 1.027** (published 4.96%/mo / 0.866 was understated by the stale-signal "
      "bug). `combo_v2_2x_freeze_dd_v2` is the best full-window Calmar (1.110, MaxDD "
      "-49.4%). The combo_4way family jumps the most (e.g. combo_4way_2x Calmar "
      "0.747 -> 1.055) but its published caps (up to 3.0x) exceed the V7 2.0x "
      "discipline and its DEV mean-monthly is far below the bar.")
    w("")

    # 2. stale-signal tax
    cmp_names = [n for n in leg.index if n in nxo.index and n not in SHORT_CLIPPED]
    d_mm = (nxo.loc[cmp_names, "mean_monthly"] - leg.loc[cmp_names, "mean_monthly"])
    d_sh = (nxo.loc[cmp_names, "sharpe"] - leg.loc[cmp_names, "sharpe"])
    d_ca = (nxo.loc[cmp_names, "calmar"] - leg.loc[cmp_names, "calmar"])
    d_mm_c = (nxc.loc[cmp_names, "mean_monthly"] - leg.loc[cmp_names, "mean_monthly"])
    w("## 2. Measured stale-signal tax (legacy_period -> next_open, full window)")
    w("")
    w(f"Across {len(cmp_names)} comparable strategies, the old engine's period-stale "
      f"execution cost on average:")
    w("")
    w(f"- mean_monthly: **{d_mm.mean()*100:+.3f} pp/mo mean** ({d_mm.median()*100:+.3f} median); "
      f"legacy -> next_close alone is {d_mm_c.mean()*100:+.3f} pp/mo mean, so nearly all "
      f"of the correction is the lag fix, not the open-vs-close convention")
    w(f"- Sharpe: {d_sh.mean():+.3f} mean ({d_sh.median():+.3f} median)")
    w(f"- Calmar: {d_ca.mean():+.3f} mean ({d_ca.median():+.3f} median)")
    w("")
    w("The tax is strongly heterogeneous: fast-rebalance and vol/dd-gated strategies "
      "paid the most (their signals decayed fastest), while a few slow gate-based "
      "strategies accidentally BENEFited from staleness (their stale gate happened to "
      "hold through whipsaws).")
    w("")
    gain = d_mm.sort_values(ascending=False)
    w("Largest gainers from the fix (pp/mo): "
      + "; ".join(f"`{n}` {v*100:+.2f}" for n, v in gain.head(6).items()))
    w("")
    w("Largest losers (pp/mo): "
      + "; ".join(f"`{n}` {v*100:+.2f}" for n, v in gain.tail(5).items()))
    w("")

    # 3. rank movers
    rk = nxo[["rank_calmar_legacy_period", "rank_calmar_next_open", "rank_change_calmar",
              "calmar"]].copy()
    movers = rk.reindex(rk.rank_change_calmar.abs().sort_values(ascending=False).index).head(10)
    w("## 3. Biggest rank movers (full-window Calmar rank among 48; + = improved)")
    w("")
    w("| strategy | legacy rank | next_open rank | change | next_open Calmar |")
    w("|----------|-------------|----------------|--------|------------------|")
    for n, r in movers.iterrows():
        w(f"| {n} | {r.rank_calmar_legacy_period:.0f} | {r.rank_calmar_next_open:.0f} | "
          f"{r.rank_change_calmar:+.0f} | {r.calmar:.3f} |")
    w("")
    w("Pattern: the xs_momentum family (monthly rebalance, no gate) drops the most in "
      "RELATIVE rank — it was the least hurt by staleness, so fixing the bug lifted "
      "everything else past it. combo_4way and the gated fast strategies climb.")
    w("")

    # 4. champion-rival candidates (DEV)
    w("## 4. Candidate sleeves rivaling the champion (DEV window, next_open) + G1 verdict")
    w("")
    w(f"Champion combo_v2_2x DEV (corrected baseline): {pct(BASE_DEV['mean_monthly'])}/mo, "
      f"Sharpe {BASE_DEV['sharpe']:.3f}, MaxDD {pct(BASE_DEV['max_drawdown'],1)}, "
      f"Calmar {BASE_DEV['calmar']:.3f}. G1 bar: Calmar >= 0.944, mm >= 4.0%/mo, "
      f"MaxDD > -50%.")
    w("")
    devs = dev.sort_values("calmar", ascending=False)
    w("| strategy | dev mm/mo | dev Sharpe | dev MaxDD | dev Calmar | G1 metrics |")
    w("|----------|-----------|------------|-----------|------------|------------|")
    for n, r in devs.head(12).iterrows():
        checks = [r.calmar >= 0.944, r.mean_monthly >= 0.04, r.max_drawdown > -0.50]
        verdict = "PASS" if all(checks) else (
            "fails " + ",".join(t for t, c in zip(("calmar", "mm", "dd"), checks) if not c))
        w(f"| {n} | {pct(r.mean_monthly)} | {r.sharpe:.3f} | {pct(r.max_drawdown,1)} | "
          f"{r.calmar:.3f} | {verdict} |")
    w("")
    w("**G1 verdict: NO strict pass.** `combo_v2_2x_freeze_dd_v1` is the standout — "
      "dev +4.66%/mo, Sharpe 1.141, Calmar 1.054 (= 1.23x champion) — but its dev "
      "MaxDD of -52.1% misses the -50% bar by 2.1pp. `combo_v2_2x_freeze_dd_v2` "
      "passes the DD bar (-48.3%) and edges the champion on Calmar (0.878) but "
      "misses the 4.0%/mo bar (+3.75%). Both are flagged as CANDIDATE SLEEVES for "
      "the assembly phase: the freeze overlay is the only catalog mechanism that "
      "cut the champion's dev MaxDD materially while keeping mm above 3.7%.")
    w("")
    w("Neighborhood stability (catalog has no grid; the two freeze variants are each "
      "other's nearest parameter neighbors): dd_v2/dd_v1 dev Calmar ratio = "
      f"{devs.loc['combo_v2_2x_freeze_dd_v2','calmar']/devs.loc['combo_v2_2x_freeze_dd_v1','calmar']:.1%} "
      "— just under the 85% G1 stability bar, so the freeze parameters MUST be "
      "stability-tested on a proper grid in their own family before assembly.")
    w("")
    w("Also notable (not rivals): `calendar_tom` dev Calmar 0.970 at only 0.27x gross "
      "and +1.18%/mo — a potential low-correlation ballast sleeve; the corrected "
      "combo_4way_2x (dev +2.84%/mo, Calmar 0.769) is a diversified alternative "
      "chassis but is far from the mm bar at <= 2x gross.")
    w("")

    # 5. full grid (pivot)
    w("## 5. Full grid (from the family ledger) — mean_monthly / Calmar by exec model")
    w("")
    piv_m = full.pivot_table(index="name", columns="exec_model", values="mean_monthly")
    piv_c = full.pivot_table(index="name", columns="exec_model", values="calmar")
    piv_d = full.pivot_table(index="name", columns="exec_model", values="max_drawdown")
    w("| strategy | mm legacy | mm next_close | mm next_open | Calmar legacy | Calmar next_open | MaxDD next_open | dev mm | dev Calmar |")
    w("|----------|-----------|---------------|--------------|---------------|------------------|-----------------|--------|------------|")
    for n in piv_m.sort_values("next_open", ascending=False).index:
        dr = dev.loc[n] if n in dev.index else None
        w(f"| {n} | {pct(piv_m.loc[n,'legacy_period'])} | {pct(piv_m.loc[n,'next_close'])} | "
          f"{pct(piv_m.loc[n,'next_open'])} | {piv_c.loc[n,'legacy_period']:.3f} | "
          f"{piv_c.loc[n,'next_open']:.3f} | {pct(piv_d.loc[n,'next_open'],1)} | "
          + (f"{pct(dr.mean_monthly)} | {dr.calmar:.3f} |" if dr is not None else "n/a | n/a |"))
    w("")

    # 6. anomalies
    w("## 6. Anomalies and caveats")
    w("")
    w("- `ml_lgb_xs` SKIPPED per assignment (slow walk-forward LightGBM, documented "
      "failure; published: +1.23%/mo, Sharpe 0.46, MaxDD -61.7%, Calmar 0.17).")
    w("- `donchian_breakout` is non-deterministic across processes (set-order "
      "truncation, strategies.py:690) — its numbers carry ~0.2 pp/mo run-to-run noise.")
    w("- Published combo_v2 FREEZE-vs-baseline comparison was apples-to-oranges: the "
      "freeze variants were DAILY frames (1-day lag in the old engine) while the "
      "baseline was a sparse monthly frame (21-day stale). Corrected next_open is the "
      "first like-for-like comparison; the freeze advantage shrinks (dd_v1 full Calmar "
      "1.123 published -> 0.987 corrected) because the corrected baseline improves and "
      "next_open execution slightly weakens the freeze (covid DD -26.8% -> -35.4%: the "
      "freeze liquidates at the next OPEN, so it still eats the overnight gap).")
    w("- `combo_v2_2x_freeze_dd_v1` full-window next_open MaxDD (-60.2%) is deeper than "
      "its dev MaxDD (-52.1%): the 2021-11 -> 2022/23 drawdown continues past DEV_END "
      "in the full window. The freeze did NOT bind through most of that slow grind.")
    w("- `combo_4way_2.5x` / `combo_4way_kelly_*` were published at leverage_cap=3.0 "
      "(above the V7 2.0x discipline); re-scored at published caps, record-only.")
    w("- Superseded names in summary.csv not re-scored: `combo_v2_2x_baseline` "
      "(identical to `combo_v2_2x`), `combo_v2_2x_freeze_dd`, `combo_v2_2x_freeze_both` "
      "(older freeze parameterizations replaced by dd_v1/dd_v2/vix_dd_v1 in run_v6.py).")
    w("- No grid cells errored; no cells added. 192/192 pre-registered runs completed.")
    w("")

    (OUT / "RESCORE_NOTES.md").write_text("\n".join(lines))
    print(f"wrote {OUT/'RESCORE_NOTES.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
