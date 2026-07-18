"""WS3 runner — walk-forward process evaluation (plan: what-can-we-do
roadmap, Track B / WS3). Library: wf_process.py; tests:
validation/test_wf_process.py.

Order of operations (binding):
  1. mandatory truncation-equivalence audit for every pool member at
     cutoff 2020-12-31 (results -> results/v7/wf/equivalence.json; failures
     are excluded from the pool with reasons);
  2. spec freeze: write results/v7/wf/process_spec.json ONCE (exclusions
     documented inside; sha256 sealed). If the file already exists it is
     loaded and verified — NEVER overwritten (frozen means frozen);
  3. dev legs first (2019-2022): score + select, printable table;
  4. sensitivity variants (k in {2,4}, metric=calmar) on DEV LEGS ONLY,
     family "wf_sens" — fragility table, never the headline;
  5. remaining legs (2023-2026): scoring runs whose slice crosses DEV_END go
     through exp_lib with window="full", final=True (spec dev_lock_note);
  6. process curve: ONE final=True window="full" engine run per leverage
     (family "wf_process"), stats reported on the equity >= 2019-01-01;
  7. artifacts: process_curve.parquet, selections.csv, wf_results.csv,
     WF_REPORT.md — process vs hindsight champion (catalog_v2 combo_v2_2x)
     vs SPY, DSR + Newey-West t-stats, and the honesty headline (the
     process-vs-hindsight gap).

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python exp_wf_process.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import wf_process as W
from exp_lib import load_cache, trial_count, union_prices_cached
from metrics_v2 import DEV_END, newey_west_tstat

ROOT = Path(__file__).resolve().parent
OUT = W.WF_DIR
SCORES_CSV = OUT / "scores.csv"
EQUIV_JSON = OUT / "equivalence.json"

# Champion cross-check: the direct reference run must reproduce the ledgered
# catalog_v2 combo_v2_2x full-window row (same code path -> tight tolerance).
CATALOG_CSV = ROOT / "results" / "v7" / "catalog_v2.csv"
CHAMPION = "combo_v2_2x"
CHAMPION_PUB_GEO = 0.0368  # REPORT.md corrected headline (3.68%/mo)

SENS_VARIANTS = [  # (tag, k, metric column) — dev legs only, lev 2.0 only
    ("k2_sharpe", 2, "sharpe_at_selection"),
    ("k4_sharpe", 4, "sharpe_at_selection"),
    ("k3_calmar", 3, "calmar_at_selection"),
]


def run_equivalence() -> tuple[dict, list[dict]]:
    """Mandatory causality audit for every pool member (wf_process docstring).
    Truncated builds are parquet-cached (wf_trunc_* keys) so re-runs are
    cache hits."""
    panel, macro, _ = load_cache()
    pu = union_prices_cached()
    px, vol, cols = panel["close"], panel["volume"], list(pu.columns)
    full = W.build_pool_weights()
    tag = pd.Timestamp(W.EQUIV_CUTOFF).strftime("%Y%m%d")

    results, exclusions = {}, []
    print(f"\n══ truncation-equivalence audit @ {W.EQUIV_CUTOFF} ══")
    for name in W.POOL:
        t0 = time.time()

        def make_builder(nm):
            # adapt the zero-arg registry builder to (px, macro, vol, cols)
            return lambda p, m, v, c: W.pool_builders(p, m, v, c)[nm]()

        res = W.truncation_equivalence_test(
            make_builder(name), W.EQUIV_CUTOFF,
            px=px, macro=macro, vol=vol, cols=cols,
            full_weights=full[name],
            trunc_cache_key=f"wf_trunc_{tag}_{name}")
        results[name] = res
        flag = "OK  " if res["ok"] else "EXCL"
        print(f"  {flag} {name:30s} maxdiff {res['max_abs_diff']:.2e} "
              f"cover {res['coverage']:.0%} ({time.time()-t0:5.1f}s)"
              + (f"  <- {res['reason']}" if not res["ok"] else ""))
        if not res["ok"]:
            exclusions.append({"name": name, "reason": res["reason"]})
    EQUIV_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"  wrote {EQUIV_JSON} — {len(exclusions)} exclusion(s)")
    return results, exclusions


def get_spec(exclusions: list[dict]) -> dict:
    if W.SPEC_PATH.exists():
        spec = W.load_spec()
        print(f"\n[spec] FROZEN spec loaded (sha {spec['sha256'][:16]}…) — "
              f"refusing to rewrite")
        frozen_excl = {e["name"] for e in spec.get("exclusions", [])}
        now_excl = {e["name"] for e in exclusions}
        if frozen_excl != now_excl:
            raise SystemExit(
                f"equivalence audit no longer matches the frozen spec "
                f"(frozen {sorted(frozen_excl)} vs now {sorted(now_excl)}) — "
                f"strategy code changed after pre-registration; STOP.")
        return spec
    spec = W.write_spec(exclusions=exclusions)
    print(f"\n[spec] frozen NEW spec -> {W.SPEC_PATH} (sha {spec['sha256'][:16]}…)")
    return spec


def score_legs(years, pool_names, weights_by_name, spec_sha) -> pd.DataFrame:
    """Score the given legs, resuming from scores.csv."""
    existing = pd.read_csv(SCORES_CSV) if SCORES_CSV.exists() else None
    frames = [] if existing is None else [existing]
    wsub = {n: weights_by_name[n] for n in pool_names}
    for y in years:
        have = 0 if existing is None else int((existing["leg_year"] == y).sum())
        if have >= len(pool_names):
            continue
        t0 = time.time()
        df = W.score_pool(y, wsub, spec_sha, existing=existing)
        frames = [f[f["leg_year"] != y] for f in frames] + [df]
        allsc = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["leg_year", "name"], keep="last")
        allsc.to_csv(SCORES_CSV, index=False)
        existing = allsc
        frames = [allsc]
        print(f"  [leg {y}] scored {len(pool_names)-have} candidates "
              f"({time.time()-t0:.0f}s)")
    allsc = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["leg_year", "name"], keep="last") if frames else pd.DataFrame()
    return allsc


def selection_table(scores: pd.DataFrame, years, k, metric_col) -> dict[int, list[str]]:
    sel = {}
    for y in years:
        sel[y] = W.select_topk(scores[scores["leg_year"] == y], k=k,
                               metric_col=metric_col)
    return sel


def print_selections(scores: pd.DataFrame, selections: dict[int, list[str]]):
    for y in sorted(selections):
        sc = scores[scores["leg_year"] == y].set_index("name")
        picks = ", ".join(
            f"{n} (SR {sc.loc[n, 'sharpe_at_selection']:.2f})"
            for n in selections[y])
        print(f"  {y} (cutoff {W.leg_cutoff(y).date()}): {picks}")


def main() -> None:
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading cache + pool weight frames (warm from catalog_v2)…")
    weights_by_name = W.build_pool_weights()
    pu = union_prices_cached()

    # 1. mandatory causality audit, then 2. spec freeze (exclusions documented)
    _, exclusions = run_equivalence()
    spec = get_spec(exclusions)
    sha = spec["sha256"]
    pool = [n for n in spec["pool"]
            if n not in {e["name"] for e in spec["exclusions"]}]
    print(f"[pool] {len(pool)} of {len(spec['pool'])} names in play")

    # 3. dev legs first
    dev_years = list(W.DEV_LEGS)
    print(f"\n══ scoring dev legs {dev_years[0]}-{dev_years[-1]} ══")
    scores = score_legs(dev_years, pool, weights_by_name, sha)
    dev_sel = selection_table(scores, dev_years, W.K, "sharpe_at_selection")
    print("\n══ dev-leg selections (top-3 by Sharpe at selection date) ══")
    print_selections(scores, dev_sel)

    # 4. sensitivity variants — DEV LEGS ONLY, family wf_sens, lev 2.0
    print("\n══ sensitivity variants (dev legs only, family wf_sens) ══")
    sens_rows = []
    sens_selections = {}
    for tag, k, mcol in SENS_VARIANTS:
        sel = selection_table(scores, dev_years, k, mcol)
        sens_selections[tag] = sel
        needed = sorted(set().union(*sel.values()))
        daily = {n: W.expand_daily_aligned(weights_by_name[n], pu.index,
                                           list(pu.columns)) for n in needed}
        frame = W.assemble_process_frame(sel, daily, pu.index,
                                         dev_years[0], dev_years[-1])
        res = W.run_process_curve(
            frame, 2.0, sha, family="wf_sens", window="dev", final=False,
            tag=f"wf_sens_{tag}_2x", k=k,
            first_leg=dev_years[0], last_leg=dev_years[-1])
        _, s = W.sliced_stats(res["equity"], f"{dev_years[0]}-01-01",
                              f"wf_sens_{tag}_2x")
        sens_rows.append({"variant": tag, "k": k,
                          "metric": mcol.replace("_at_selection", ""),
                          "geo_monthly": s["geo_monthly"], "sharpe": s["sharpe"],
                          "max_drawdown": s["max_drawdown"],
                          "selections": json.dumps({str(y): v for y, v in sel.items()})})
        print(f"  {tag:10s} dev 2019-2022: geo {s['geo_monthly']*100:+.2f}%/mo "
              f"| SR {s['sharpe']:.2f} | maxDD {s['max_drawdown']*100:.1f}%")

    # 5. remaining legs (2023 scores inside dev; 2024+ cross DEV_END -> final)
    later_years = [y for y in range(W.FIRST_LEG, W.LAST_LEG + 1)
                   if y not in dev_years]
    print(f"\n══ scoring legs {later_years[0]}-{later_years[-1]} ══")
    scores = score_legs(later_years, pool, weights_by_name, sha)
    all_years = list(range(W.FIRST_LEG, W.LAST_LEG + 1))
    selections = selection_table(scores, all_years, W.K, "sharpe_at_selection")
    print("\n══ all leg selections ══")
    print_selections(scores, selections)

    # selections.csv: every score row + rank/selected flags
    sel_df = scores.copy()
    sel_df["rank_at_selection"] = sel_df.groupby("leg_year")[
        "sharpe_at_selection"].rank(ascending=False, method="first")
    sel_df["selected"] = [
        n in selections[y] for y, n in zip(sel_df["leg_year"], sel_df["name"])]
    sel_df = sel_df.sort_values(["leg_year", "rank_at_selection"])
    sel_df.to_csv(OUT / "selections.csv", index=False)

    # 6. the process curve — ONE final run per leverage
    print("\n══ process curve (final=True, family wf_process) ══")
    needed = sorted(set().union(*selections.values()))
    daily = {n: W.expand_daily_aligned(weights_by_name[n], pu.index,
                                       list(pu.columns)) for n in needed}
    frame = W.assemble_process_frame(selections, daily, pu.index,
                                     W.FIRST_LEG, W.LAST_LEG)
    proc_eq, proc_stats = {}, {}
    for lev in W.LEVERAGES:
        res = W.run_process_curve(frame, lev, sha)
        eq, s = W.sliced_stats(res["equity"], f"{W.FIRST_LEG}-01-01",
                               f"wf_process_{lev:g}x")
        proc_eq[lev], proc_stats[lev] = eq, s
        print(f"  {lev:g}x: geo {s['geo_monthly']*100:+.2f}%/mo | "
              f"SR {s['sharpe']:.2f} | maxDD {s['max_drawdown']*100:.1f}%")

    # 7. references + report
    ch_full_eq, ch_full_sum = W.champion_reference()
    cat = pd.read_csv(CATALOG_CSV)
    cat_row = cat[(cat["name"] == CHAMPION) & (cat["window"] == "full")].iloc[0]
    d_sr = abs(ch_full_sum["sharpe"] - float(cat_row["sharpe"]))
    if d_sr > 1e-6:
        raise SystemExit(
            f"champion reference does not reproduce the catalog_v2 ledgered "
            f"row (sharpe diff {d_sr:.2e}) — comparisons would be invalid.")
    print(f"\n[champion] reference reproduces catalog_v2 row "
          f"(sharpe diff {d_sr:.1e})")
    ch_eq, ch_sum = W.sliced_stats(ch_full_eq, f"{W.FIRST_LEG}-01-01",
                                   "champion_2019on")
    spy_eq, spy_sum = W.sliced_stats(W.spy_reference(), f"{W.FIRST_LEG}-01-01",
                                     "spy_2019on")

    write_outputs(spec, scores, selections, sens_rows, proc_eq, proc_stats,
                  ch_eq, ch_sum, ch_full_sum, spy_eq, spy_sum)
    print(f"\ndone in {(time.time()-t_start)/60:.1f} min")


def write_outputs(spec, scores, selections, sens_rows, proc_eq, proc_stats,
                  ch_eq, ch_sum, ch_full_sum, spy_eq, spy_sum) -> None:
    sha = spec["sha256"]
    n_trials = trial_count(dedupe=True)

    # process_curve.parquet — the daily equity curves, common 2019+ index
    curves = pd.DataFrame({
        **{f"process_{lev:g}x": proc_eq[lev] for lev in W.LEVERAGES},
        "champion_combo_v2_2x": ch_eq,
        "spy": spy_eq,
    })
    curves.to_parquet(OUT / "process_curve.parquet")

    # wf_results.csv + report rows
    ch_m = W.monthly_rets(ch_eq)
    spy_m = W.monthly_rets(spy_eq)
    rows = []
    for lev in W.LEVERAGES:
        s = proc_stats[lev]
        pm = W.monthly_rets(proc_eq[lev])
        rows.append({
            "name": f"wf_process_{lev:g}x", "window": "2019on",
            "geo_monthly": s["geo_monthly"], "mean_monthly": s["mean_monthly"],
            "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"],
            "cagr": s["cagr"], "ann_vol": s["ann_vol"], "calmar": s["calmar"],
            "worst_month": s["worst_month"], "n_days": s["n_days"],
            "dsr": W.process_dsr(s["sharpe"], s["n_days"]),
            "nw_t_vs_champion": newey_west_tstat((pm - ch_m).dropna()),
            "nw_t_vs_spy": newey_west_tstat((pm - spy_m).dropna()),
            "gap_geo_vs_champion_full_pub": s["geo_monthly"] - CHAMPION_PUB_GEO,
            "gap_geo_vs_champion_2019on": s["geo_monthly"] - ch_sum["geo_monthly"],
        })
    for nm, s in (("champion_2019on", ch_sum), ("champion_full", ch_full_sum),
                  ("spy_2019on", spy_sum)):
        rows.append({"name": nm,
                     "window": "full" if nm.endswith("full") else "2019on",
                     "geo_monthly": s["geo_monthly"],
                     "mean_monthly": s["mean_monthly"], "sharpe": s["sharpe"],
                     "max_drawdown": s["max_drawdown"], "cagr": s["cagr"],
                     "ann_vol": s["ann_vol"], "calmar": s["calmar"],
                     "worst_month": s["worst_month"], "n_days": s["n_days"]})
    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT / "wf_results.csv", index=False)
    print(f"wrote {OUT/'wf_results.csv'}, process_curve.parquet, selections.csv")

    # WF_REPORT.md
    p2 = next(r for r in rows if r["name"] == "wf_process_2x")
    lines = [
        "# WF_REPORT — walk-forward process evaluation (Track B / WS3)",
        "",
        f"Spec: `results/v7/wf/process_spec.json`, sha256 `{sha}` "
        f"(frozen {spec['frozen_at']}; embedded in every wf_* ledger note). "
        f"Pool: {len(spec['pool'])} base catalog names, "
        f"{len(spec['exclusions'])} excluded by the truncation-equivalence "
        f"audit. Metric: {spec['metric']}. Step: {spec['step']}. "
        f"Warmup: {spec['warmup']}. Leg 2026 is partial (data ends "
        f"{ch_eq.index[-1].date()}).",
        "",
        "## Honesty headline",
        "",
        f"**The selection process at 2x earns {p2['geo_monthly']*100:+.2f}%/mo "
        f"(geo) vs the hindsight champion's {CHAMPION_PUB_GEO*100:.2f}%/mo — "
        f"a gap of {p2['gap_geo_vs_champion_full_pub']*100:+.2f}pp/mo.** "
        f"Same-window (2019+) champion geo is "
        f"{ch_sum['geo_monthly']*100:+.2f}%/mo (gap "
        f"{p2['gap_geo_vs_champion_2019on']*100:+.2f}pp/mo). The process "
        f"number — re-selecting the top-3 annually on trailing data — is the "
        f"honest prior for live performance; the champion number is what "
        f"hindsight paid. Survivorship inflation (WS1) applies to BOTH arms "
        f"and is not corrected here.",
        "",
        "## Truncation-equivalence audit (causality gate)",
        "",
        f"All pool members were rebuilt on a panel truncated at "
        f"{W.EQUIV_CUTOFF} and compared to the full-build weights sliced to "
        f"the same cutoff (atol {W.EQUIV_ATOL:g}, coverage >= "
        f"{W.EQUIV_MIN_COVERAGE:.0%}). Per-name results: "
        f"`results/v7/wf/equivalence.json`.",
        "",
    ]
    if spec["exclusions"]:
        lines += ["Excluded (non-causal):", ""]
        lines += [f"- `{e['name']}` — {e['reason']}" for e in spec["exclusions"]]
    else:
        lines += ["All 21 pool members passed — no exclusions."]
    lines += ["", "## Per-leg selections (top-3 by Sharpe at selection date)", "",
              "| leg | cutoff | pick 1 | pick 2 | pick 3 |",
              "|---|---|---|---|---|"]
    for y in sorted(selections):
        sc = scores[scores["leg_year"] == y].set_index("name")
        cells = [f"{n} ({sc.loc[n, 'sharpe_at_selection']:.2f})"
                 for n in selections[y]]
        lines.append(f"| {y} | {W.leg_cutoff(y).date()} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Full scoring table (all candidates, all legs, selection-time Sharpe "
        "and Calmar): `results/v7/wf/selections.csv`.",
        "",
        "## Process curve vs hindsight champion vs SPY (2019-01-01 onward)",
        "",
        "| curve | geo %/mo | sharpe | maxDD | ann vol | DSR | NW-t vs champ | NW-t vs SPY |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        dsr = f"{r['dsr']:.3f}" if "dsr" in r and np.isfinite(r.get("dsr", np.nan)) else "—"
        t1 = f"{r['nw_t_vs_champion']:+.2f}" if "nw_t_vs_champion" in r else "—"
        t2 = f"{r['nw_t_vs_spy']:+.2f}" if "nw_t_vs_spy" in r else "—"
        lines.append(
            f"| {r['name']} | {r['geo_monthly']*100:+.2f} | "
            f"{r['sharpe']:.2f} | {r['max_drawdown']*100:.1f}% | "
            f"{r['ann_vol']*100:.1f}% | {dsr} | {t1} | {t2} |")
    lines += [
        "",
        f"DSR = deflated Sharpe at the program-wide deduped trial count "
        f"(n={n_trials}) with cross-trial SR variance measured from the "
        f"ledgers (exp_catalog_v2.ledger_sr_variance_daily, reused). "
        f"Newey-West t-stats (lags=3) are on paired monthly return "
        f"differences, 2019+. SPY is an adjusted-close buy-and-hold proxy "
        f"(no costs) — regime context only. The champion reference "
        f"reproduces the ledgered catalog_v2 `combo_v2_2x` full-window row "
        f"(cross-checked at 1e-6).",
        "",
        "## Sensitivity (fragility table — dev legs 2019-2022 ONLY, never the headline)",
        "",
        "| variant | k | metric | dev geo %/mo | dev sharpe | dev maxDD |",
        "|---|---|---|---|---|---|",
    ]
    # primary process on the same dev window for comparison
    eq2 = proc_eq[2.0]
    dev_eq = eq2.loc[:pd.Timestamp(DEV_END)]
    dev_eq = dev_eq / dev_eq.iloc[0] * 100_000.0
    from metrics import summary as _sum
    ds = _sum(dev_eq, name="primary_dev")
    lines.append(f"| primary (headline) | {W.K} | sharpe | "
                 f"{ds['geo_monthly']*100:+.2f} | {ds['sharpe']:.2f} | "
                 f"{ds['max_drawdown']*100:.1f}% |")
    for r in sens_rows:
        lines.append(f"| {r['variant']} | {r['k']} | {r['metric']} | "
                     f"{r['geo_monthly']*100:+.2f} | {r['sharpe']:.2f} | "
                     f"{r['max_drawdown']*100:.1f}% |")
    lines += [
        "",
        "Sensitivity runs are family `wf_sens`, dev window only; their "
        "selections are in `selections` column of the printed run log and "
        "were never used for the full-window process curve.",
        "",
        "## Provenance",
        "",
        f"- engine: engine_v2 next_open, tc {W.TC_BPS:g}bp/side, margin "
        f"{W.MARGIN_BPS:g}bp/yr on long gross > 1x, leverage cap "
        f"{W.PROC_LEV_CAP:g} (scoring at cap {W.SCORE_LEV_CAP:g})",
        f"- ledger families: wf_score (scoring), wf_process (final curves), "
        f"wf_sens (dev-only variants); every note carries spec sha `{sha[:16]}…`",
        f"- artifacts: process_spec.json, equivalence.json, scores.csv, "
        f"selections.csv, wf_results.csv, process_curve.parquet",
    ]
    (OUT / "WF_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'WF_REPORT.md'}")

    print("\n══ key numbers ══")
    for r in rows:
        if r["name"].startswith("wf_process"):
            print(f"  {r['name']}: geo {r['geo_monthly']*100:+.2f}%/mo | "
                  f"SR {r['sharpe']:.2f} | maxDD {r['max_drawdown']*100:.1f}% | "
                  f"DSR {r['dsr']:.3f} | gap vs 3.68%: "
                  f"{r['gap_geo_vs_champion_full_pub']*100:+.2f}pp/mo")
    print(f"  champion 2019+: geo {ch_sum['geo_monthly']*100:+.2f}%/mo | "
          f"SPY 2019+: geo {spy_sum['geo_monthly']*100:+.2f}%/mo")


if __name__ == "__main__":
    main()
