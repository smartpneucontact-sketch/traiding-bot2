"""exp_lev_recal.py — LEV1: leverage optimum at measured costs
(governed by results/v7/improve3/LEV1_PREREG.md).

STATUS: FROZEN 2026-08-30 (prereg sha256
484d9a0efb08a9cb4231b17ec4bd489b428dc4ddbf6c31806617215146cac2a9, recorded
in the freeze commit). The do-not-run guard was removed and PREREG_FROZEN
set True in this commit, per the freeze terms — new cells now ledger to
family lev1 on first render.

MEASUREMENT ONLY (prereg: no pass bar, no promotion decision) — descriptive
table + ONE pre-declared summary statistic: the grid leverage maximizing
FULL-WINDOW net geo_monthly at tc=52 (ties -> lower leverage).

Grid: leverage in {1.0, 1.25, 1.5, 1.75, 2.0} x tc in {30, 52} on champion
combo_v2 (exp_pit_survivorship.build_baseline_base x lev), next_open,
leverage_cap=2.0, margin_bps_annual=600 applied exactly as exp_rescore600.py
did on the run_trial track (run_trial(..., margin_bps_annual=600) ->
engine_v2 charges (long_gross − 1)_+ x 600bp/252 daily — only the levered
portion above 1.0x NAV pays financing; the 1.0x cell pays zero). Windows:
full (final=True, correction-exception measurement precedent) + dev,
ledgered; 2019on = derived slice of the full equity (exp_tc_recal
convention), never a separate trial.

Cited, never re-run (prereg "Cells"): lev {1.0, 1.5, 2.0} at tc=30 —
results/v7/trials/tc_recal.csv combo_v2_{1,1.5,2}x_tc30, full + dev rows.
Newly rendered under family lev1: lev {1.25, 1.75} at tc=30 and all five
leverages at tc=52 (no tc=52 row exists in results/v7/trials/ — verified
2026-08-30).

Fidelity gate (hard-asserted first): tc=5 log=False full re-render of
base x 2.0 == results/v7/trials/pit.csv combo_v2_baseline_2x full row
(|dSR| < 1e-6, |dCAGR| < 1e-8).

Ledger: results/v7/trials/lev1.csv (NEW margin family; schema in the
prereg). Output table: results/v7/improve3/lev_recal.csv.

Run (AFTER freeze only):
  cd "Traiding 11" && /opt/anaconda3/bin/python exp_lev_recal.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))

from exp_lib import TRIALS_DIR, load_cache, run_trial
from exp_tc_recal import (ledger_row_to_summary, sliced_summary,
                          sliced_turnover)

# ── prereg freeze switch ────────────────────────────────────────────────────
# results/v7/improve3/LEV1_PREREG.md frozen 2026-08-30 (sha in the freeze
# commit) — ledgering enabled per the freeze terms.
PREREG_FROZEN = True

FAMILY = "lev1"
LEDGER = TRIALS_DIR / f"{FAMILY}.csv"
OUT_CSV = ROOT / "results" / "v7" / "improve3" / "lev_recal.csv"

MARGIN_BPS = 600.0
LEVS = (1.0, 1.25, 1.5, 1.75, 2.0)
TCS = (30.0, 52.0)
TC_HEADLINE = 52.0
# lev/tc cells already ledgered elsewhere -> cited, never re-run (prereg).
CITED_FROM_TC_RECAL = {(1.0, 30.0), (1.5, 30.0), (2.0, 30.0)}
GATE_TOL_SR = 1e-6
GATE_TOL_CAGR = 1e-8

NOTE = ("LEV1 prereg (results/v7/improve3/LEV1_PREREG.md): leverage "
        "measurement at measured tc (52=clean vs-open 2026-07-27, "
        "30=interim vs-arrival); MEASUREMENT ONLY, no pass bar; margin per "
        "exp_rescore600 run_trial track")


def build_base() -> pd.DataFrame:
    from exp_pit_survivorship import build_baseline_base
    panel, macro, _ = load_cache()
    return build_baseline_base(panel["close"], macro)


def fidelity_gate(base: pd.DataFrame) -> None:
    """tc=5 log=False full re-render vs the ledgered pit.csv champion row."""
    pit = pd.read_csv(TRIALS_DIR / "pit.csv")
    ref = pit[(pit["name"] == "combo_v2_baseline_2x")
              & (pit["window"] == "full")].iloc[-1]
    r5 = run_trial(base * 2.0, name="combo_v2_2x_tc5_gate", family=FAMILY,
                   params={"gate": True}, window="full",
                   exec_model="next_open", tc_bps=5.0, leverage_cap=2.0,
                   margin_bps_annual=MARGIN_BPS, final=True, log=False,
                   notes="fidelity gate re-render, NOT ledgered")
    d_sr = abs(r5["summary"]["sharpe"] - float(ref["sharpe"]))
    d_cg = abs(r5["summary"]["cagr"] - float(ref["cagr"]))
    print(f"[gate] |dSR|={d_sr:.2e} |dCAGR|={d_cg:.2e} "
          f"vs trials/pit.csv combo_v2_baseline_2x full")
    assert d_sr < GATE_TOL_SR and d_cg < GATE_TOL_CAGR, \
        "fidelity gate FAILED vs results/v7/trials/pit.csv"


def result_row(lev: float, tc: float, window: str, s: dict, turnover: float,
               source: str, avg_gross: float | None = None) -> dict:
    # avg_gross is a prereg-declared output column (2026-08-31 audit: it
    # was missing from the first render of lev_recal.csv).
    return {"name": f"combo_v2_{lev:g}x", "leverage": lev, "tc_bps": tc,
            "window": window, "geo_monthly": s["geo_monthly"],
            "mean_monthly": s["mean_monthly"], "sharpe": s["sharpe"],
            "max_drawdown": s["max_drawdown"], "calmar": s["calmar"],
            "turnover_ann": turnover,
            "avg_gross": (avg_gross if avg_gross is not None
                          else s.get("avg_gross_exposure")),
            "source": source}


def _sliced_avg_gross(weights: pd.DataFrame, start: str) -> float | None:
    """2019on avg gross from the daily-ffilled decision frame (pre-cap
    approximation, labeled as derived in the source column)."""
    try:
        daily = weights.ffill().abs().sum(axis=1)
        return float(daily.loc[start:].mean())
    except Exception:
        return None


def cited_rows(base: pd.DataFrame, rows: list[dict]) -> None:
    led = pd.read_csv(TRIALS_DIR / "tc_recal.csv")
    for lev, tc in sorted(CITED_FROM_TC_RECAL):
        for window in ("full", "dev"):
            r = led[(led["name"] == f"combo_v2_{lev:g}x_tc{tc:g}")
                    & (led["window"] == window)]
            assert len(r) >= 1, \
                f"missing cited row combo_v2_{lev:g}x_tc{tc:g} {window}"
            r = r.iloc[-1]
            rows.append(result_row(
                lev, tc, window, ledger_row_to_summary(r),
                float(r["turnover_ann"]),
                f"results/v7/trials/tc_recal.csv "
                f"name=combo_v2_{lev:g}x_tc{tc:g} window={window}",
                avg_gross=float(r["avg_gross"]) if "avg_gross" in r else None))
        # 2019on for the cited cells (prereg-declared table completeness,
        # 2026-08-31 audit): log=False full re-render asserted against the
        # cited full row, then sliced — no new ledger rows.
        name = f"combo_v2_{lev:g}x_tc{tc:g}"
        fref = led[(led["name"] == name) & (led["window"] == "full")].iloc[-1]
        w = base * lev
        res = run_trial(w, name=f"{name}_2019on_gate", family=FAMILY,
                        params={"derived": True}, window="full",
                        exec_model="next_open", tc_bps=tc, leverage_cap=2.0,
                        margin_bps_annual=MARGIN_BPS, final=True, log=False,
                        notes="cited-cell 2019on derivation re-render, NOT ledgered")
        d_sr = abs(res["summary"]["sharpe"] - float(fref["sharpe"]))
        assert d_sr < GATE_TOL_SR, \
            f"2019on derivation re-render mismatch vs tc_recal.csv {name}"
        rows.append(result_row(
            lev, tc, "2019on",
            sliced_summary(res["equity"], "2019-01-01", None,
                           f"{name}_2019on"),
            sliced_turnover(res["weights"], "2019-01-01", None),
            f"derived: log=False re-render asserted == "
            f"results/v7/trials/tc_recal.csv name={name} window=full, "
            f"equity sliced >=2019-01-01",
            avg_gross=_sliced_avg_gross(res["weights"], "2019-01-01")))


def render_cells(base: pd.DataFrame, rows: list[dict]) -> None:
    done: set[tuple[str, str]] = set()
    if LEDGER.exists():
        led = pd.read_csv(LEDGER)
        done = set(zip(led["name"], led["window"]))
        print(f"[resume] {len(done)} runs already in ledger")

    for lev in LEVS:
        w = base * lev
        for tc in TCS:
            if (lev, tc) in CITED_FROM_TC_RECAL:
                continue
            name = f"combo_v2_{lev:g}x_tc{tc:g}"
            params = {"lev": lev, "tc_bps": tc, "n_long": 30,
                      "sleeves": "xs30+dual30+adapt30",
                      "margin_bps_annual": MARGIN_BPS,
                      "source": "exp_lev_recal"}
            for window in ("full", "dev"):
                log = PREREG_FROZEN and (name, window) not in done
                t0 = time.time()
                res = run_trial(w, name=name, family=FAMILY, params=params,
                                window=window, exec_model="next_open",
                                tc_bps=tc, leverage_cap=2.0,
                                margin_bps_annual=MARGIN_BPS,
                                final=(window == "full"), log=log, notes=NOTE)
                s = res["summary"]
                # Provenance must reflect the ACTUAL reason a row is not
                # freshly ledgered: on resume the cell already sits in the
                # family ledger (2026-08-31 audit — the old string claimed
                # 'prereg not frozen' for resumed cells).
                if log:
                    src = (f"results/v7/trials/lev1.csv name={name} "
                           f"window={window}")
                elif (name, window) in done:
                    src = (f"results/v7/trials/lev1.csv name={name} "
                           f"window={window} (resume: already ledgered)")
                else:
                    src = (f"UNLEDGERED (log=False, prereg not frozen) "
                           f"name={name} window={window}")
                rows.append(result_row(lev, tc, window, s,
                                       float(s["turnover_annualized"]), src))
                print(f"[lev1] {name:22s} {window:4s} "
                      f"geo={s['geo_monthly']*100:+.3f}%/mo "
                      f"({time.time()-t0:.0f}s)")
                if window == "full":
                    rows.append(result_row(
                        lev, tc, "2019on",
                        sliced_summary(res["equity"], "2019-01-01", None,
                                       f"{name}_2019on"),
                        sliced_turnover(res["weights"], "2019-01-01", None),
                        f"derived: {src} equity sliced >=2019-01-01",
                        avg_gross=_sliced_avg_gross(res["weights"],
                                                    "2019-01-01")))


def summarize(df: pd.DataFrame) -> None:
    print("\n== LEV1 table (geo %/mo; margin 600; survivors-only upper "
          "bounds) ==")
    piv = df[df["window"] == "full"].pivot_table(
        index="leverage", columns="tc_bps", values="geo_monthly",
        aggfunc="last")
    print((piv * 100).round(3).to_string())

    # THE one pre-declared summary statistic (prereg "Declared output" #2).
    full52 = df[(df["window"] == "full") & (df["tc_bps"] == TC_HEADLINE)]
    best = full52.sort_values(["geo_monthly", "leverage"],
                              ascending=[False, True]).iloc[0]
    print(f"\nPRE-DECLARED STATISTIC: full-window net geo at tc={TC_HEADLINE:g} "
          f"is maximized at leverage {best['leverage']:g}x "
          f"({best['geo_monthly']*100:+.3f}%/mo) [{best['source']}]")
    ctx = df[(df["window"] == "2019on") & (df["tc_bps"] == TC_HEADLINE)]
    if len(ctx):
        b19 = ctx.sort_values(["geo_monthly", "leverage"],
                              ascending=[False, True]).iloc[0]
        print(f"context (not a declared statistic): 2019on argmax at "
              f"tc={TC_HEADLINE:g} = {b19['leverage']:g}x "
              f"({b19['geo_monthly']*100:+.3f}%/mo)")
    print("\nMEASUREMENT ONLY — no promotion decision reads from this table "
          "(prereg 'Purpose'). Any live leverage change needs its own prereg.")


def main() -> None:
    t0 = time.time()
    base = build_base()
    fidelity_gate(base)
    rows: list[dict] = []
    cited_rows(base, rows)
    render_cells(base, rows)
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["leverage", "tc_bps", "window"], keep="last")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    summarize(df)
    print(f"total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
