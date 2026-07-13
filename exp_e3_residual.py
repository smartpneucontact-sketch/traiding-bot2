"""E3 — Residual (idiosyncratic) momentum sleeve (Blitz/Huij/Martens 2011).

Pre-registered grid (16 cells, family="E3_residual", window="dev" only):
  beta_mode  ∈ {market, market_sector}
  scale_mode ∈ {raw, sharpe}
  role       ∈ {alone1x (diagnostic, ungated), add25 (4th sleeve at 25%),
                replxs (replace xs_momentum), repladapt (replace adaptive)}
Blend roles run at 2.0x leverage (blend × 2.0), next_open, 5 bp, cap 2.0x.

Also: live-twin parity assertion (3 seeded random decision dates per
variant, tol 1e-9) and the E3 report at results/v7/E3_REPORT.md.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python3 exp_e3_residual.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from exp_lib import cached_weights, load_cache, run_trial, trial_count
from strategies_v2 import residual_momentum, _residual_momentum_weights_live

FAMILY = "E3_residual"
N_LONG = 30
ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "results" / "v7" / "E3_REPORT.md"

BETA_MODES = ["market", "market_sector"]
SCALE_MODES = ["raw", "sharpe"]
ROLES = ["alone1x", "add25", "replxs", "repladapt"]
BLEND_ROLES = ["add25", "replxs", "repladapt"]

# G1 thresholds (dev): 1.10 × champion dev Calmar 0.858 = 0.944
G1_CALMAR = 0.944
G1_MM = 0.04
G1_MAXDD = -0.50
G1_STAB_FRAC = 0.85


def short(beta_mode: str, scale_mode: str) -> str:
    return f"{'m' if beta_mode == 'market' else 'ms'}_{'raw' if scale_mode == 'raw' else 'shp'}"


def build_sleeves(px, macro, sector_map) -> dict[str, pd.DataFrame]:
    sleeves = {}
    for bm, sm_ in product(BETA_MODES, SCALE_MODES):
        key = f"sleeve_residual_mom_{bm}_{sm_}_n{N_LONG}"
        sleeves[(bm, sm_)] = cached_weights(
            key,
            lambda bm=bm, sm_=sm_: residual_momentum(
                px, macro, sector_map, n_long=N_LONG,
                beta_mode=bm, scale_mode=sm_,
            ),
        )
        print(f"built/loaded {key}: {sleeves[(bm, sm_)].shape}")
    return sleeves


def parity_check(sleeves, px, macro, sector_map) -> list[str]:
    """Live twin == backtest frame rows at 3 seeded random decision dates."""
    rng = np.random.default_rng(7)
    lines = []
    for (bm, sm_), frame in sleeves.items():
        dates = frame.index[frame.abs().sum(axis=1) > 0]
        picks = rng.choice(len(dates), size=3, replace=False)
        for k in picks:
            d = dates[k]
            live = _residual_momentum_weights_live(
                px.loc[:d], macro.loc[:d], sector_map,
                n_long=N_LONG, beta_mode=bm, scale_mode=sm_,
            )
            live_row = pd.Series(0.0, index=px.columns)
            for s, w in live.items():
                live_row.loc[s] = w
            diff = float((live_row - frame.loc[d]).abs().max())
            assert diff <= 1e-9, (
                f"PARITY FAIL {bm}/{sm_} @ {d.date()}: max|diff|={diff:.2e}"
            )
            lines.append(f"| {bm} | {sm_} | {d.date()} | {diff:.1e} | PASS |")
    print(f"parity: {len(lines)} checks passed (tol 1e-9)")
    return lines


def blend(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """combo_v2 convention: union dates/cols, fill 0, simple average."""
    dates = sorted(set().union(*[f.index for f in frames]))
    cols = sorted(set().union(*[f.columns for f in frames]))
    out = sum(f.reindex(index=dates, columns=cols, fill_value=0.0) for f in frames)
    return out / len(frames)


def main():
    panel, macro, sector_map = load_cache()
    px = panel["close"]

    sleeves = build_sleeves(px, macro, sector_map)
    parity_lines = parity_check(sleeves, px, macro, sector_map)

    w_xs = pd.read_parquet(ROOT / "weights_store" / "sleeve_xs_momentum_n30.parquet")
    w_du = pd.read_parquet(ROOT / "weights_store" / "sleeve_dual_momentum_voltarget_n30.parquet")
    w_ad = pd.read_parquet(ROOT / "weights_store" / "sleeve_adaptive_voltarget_n30.parquet")

    # Sanity: blend convention reproduces the cached combo base exactly.
    base = pd.read_parquet(ROOT / "weights_store" / "combo_v2_base_1x.parquet")
    b2 = blend([w_xs, w_du, w_ad])
    assert np.allclose(
        b2.reindex(index=base.index, columns=base.columns).fillna(0.0).values,
        base.values, atol=1e-12,
    ), "blend convention drifted from combo_v2_base_1x"

    rows = {}
    for bm, sm_, role in product(BETA_MODES, SCALE_MODES, ROLES):
        w_res = sleeves[(bm, sm_)]
        if role == "alone1x":
            w, lev = w_res * 1.0, 1.0
        elif role == "add25":
            w, lev = blend([w_xs, w_du, w_ad, w_res]) * 2.0, 2.0
        elif role == "replxs":
            w, lev = blend([w_res, w_du, w_ad]) * 2.0, 2.0
        else:  # repladapt
            w, lev = blend([w_xs, w_du, w_res]) * 2.0, 2.0
        name = f"e3_resid_{short(bm, sm_)}_{role}"
        params = {"beta_mode": bm, "scale_mode": sm_, "role": role,
                  "n_long": N_LONG, "leverage": lev,
                  "lookback_long": 252, "lookback_skip": 21, "rebal_freq": 21}
        res = run_trial(w, name=name, family=FAMILY, params=params,
                        window="dev", notes="E3 pre-registered grid")
        rows[(bm, sm_, role)] = res["row"]
        r = res["row"]
        print(f"{name:34s} mm={r['mean_monthly']*100:6.3f}% sh={r['sharpe']:.3f} "
              f"dd={r['max_drawdown']*100:6.1f}% cal={r['calmar']:.3f} "
              f"2022ret={r['ep_2022_ret']*100:6.1f}% 2022dd={r['ep_2022_dd']*100:6.1f}%")

    # ── G1 gate on blend roles ────────────────────────────────────────────
    def neighbors(cfg):
        bm, sm_, role = cfg
        out = [(b, sm_, role) for b in BETA_MODES if b != bm]
        out += [(bm, s, role) for s in SCALE_MODES if s != sm_]
        out += [(bm, sm_, r) for r in BLEND_ROLES if r != role]
        return out

    gated = {c: r for c, r in rows.items() if c[2] in BLEND_ROLES}
    verdicts = {}
    for cfg, r in sorted(gated.items(), key=lambda kv: -(kv[1]["calmar"] or -9)):
        hard = (r["calmar"] >= G1_CALMAR and r["mean_monthly"] >= G1_MM
                and r["max_drawdown"] > G1_MAXDD)
        nb = [rows[n]["calmar"] for n in neighbors(cfg) if n in rows]
        med_nb = float(np.median(nb)) if nb else float("nan")
        stab = np.isfinite(med_nb) and med_nb >= G1_STAB_FRAC * r["calmar"]
        verdicts[cfg] = {"hard": hard, "stab": bool(stab), "med_nb": med_nb,
                         "pass": hard and bool(stab)}

    winners = [c for c, v in verdicts.items() if v["pass"]]
    winners = sorted(winners, key=lambda c: -gated[c]["calmar"])[:2]

    write_report(rows, verdicts, winners, parity_lines)
    print(f"\nG1 winners: {[rows[c]['name'] for c in winners] or 'NONE'}")
    print(f"total trials logged (all families): {trial_count()}")
    return rows, verdicts, winners


def write_report(rows, verdicts, winners, parity_lines):
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    def fmt(r):
        return (f"| {r['name']} | {r['mean_monthly']*100:.3f} | {r['sharpe']:.3f} | "
                f"{r['max_drawdown']*100:.1f} | {r['calmar']:.3f} | "
                f"{r['turnover_ann']:.1f} | {r['avg_gross']:.2f} | "
                f"{r['ep_2018Q4_ret']*100:.1f} / {r['ep_2018Q4_dd']*100:.1f} | "
                f"{r['ep_covid_ret']*100:.1f} / {r['ep_covid_dd']*100:.1f} | "
                f"{r['ep_2022_ret']*100:.1f} / {r['ep_2022_dd']*100:.1f} |")

    lines = [
        "# E3 — Residual (idiosyncratic) momentum sleeve",
        "",
        f"Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}  ",
        "Family: `E3_residual` — all selection on DEV window (2016-04..2022-12), "
        "next_open, 5 bp, leverage cap 2.0x.  ",
        "Baseline (combo_v2_2x dev): mean_monthly 4.555%/mo, Sharpe 1.044, "
        "MaxDD -60.2%, Calmar 0.858.  ",
        "G1: Calmar ≥ 0.944 AND mean_monthly ≥ 4.0% AND MaxDD > -50% AND "
        "median neighbor Calmar ≥ 85% of winner.",
        "",
        "## Sleeve definition",
        "",
        "`strategies_v2.residual_momentum(px, macro, sector_map, n_long=30, "
        "beta_mode, scale_mode)` — 21d cadence, warm-up 273d (same grid as "
        "xs_momentum), equal-weight 1/30. Signal = sum of daily residuals "
        "(alpha retained) over trailing 273d window excluding most recent 21d; "
        "betas OLS vs SPY (market) or [SPY, sector XL*] (market_sector; "
        "missing sector/ETF → market-only fallback). sharpe variant divides "
        "by formation-window residual std (ddof=0). Eligibility: finite close "
        "at window start/end + ≥95% finite returns (NaN→0 in regression).",
        "",
        "## Full grid (16 pre-registered cells)",
        "",
        "| name | mm %/mo | Sharpe | MaxDD % | Calmar | turn/yr | gross | "
        "2018Q4 ret/dd % | covid ret/dd % | 2022 ret/dd % |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg in sorted(rows, key=lambda c: -(rows[c]["calmar"] or -9)):
        lines.append(fmt(rows[cfg]))

    lines += ["", "## G1 verdicts (blend roles; alone1x is diagnostic-only)", "",
              "| config | Calmar | hard gate | median neighbor Calmar | "
              "stability (≥85%) | G1 |", "|---|---|---|---|---|---|"]
    for cfg, v in sorted(verdicts.items(), key=lambda kv: -(rows[kv[0]]["calmar"] or -9)):
        r = rows[cfg]
        lines.append(
            f"| {r['name']} | {r['calmar']:.3f} | "
            f"{'PASS' if v['hard'] else 'fail'} | {v['med_nb']:.3f} | "
            f"{'PASS' if v['stab'] else 'fail'} | "
            f"{'PASS' if v['pass'] else 'fail'} |")
    lines += [
        "",
        "Neighborhood = configs differing in exactly one grid dimension "
        "(beta_mode flip, scale_mode flip, or one of the other two blend "
        "roles); median of their dev Calmars vs 85% of the candidate's.",
        "",
        "## Winners advanced (≤2)",
        "",
    ]
    if winners:
        for c in winners:
            r = rows[c]
            lines.append(f"- **{r['name']}** — params `{r['params_json']}` — "
                         f"mm {r['mean_monthly']*100:.3f}%/mo, Sharpe "
                         f"{r['sharpe']:.3f}, MaxDD {r['max_drawdown']*100:.1f}%, "
                         f"Calmar {r['calmar']:.3f}")
    else:
        lines.append("- none — family fails G1.")

    lines += ["", "## Live-twin parity (tol 1e-9, 3 seeded dates/variant)", "",
              "| beta_mode | scale_mode | date | max abs diff | verdict |",
              "|---|---|---|---|---|"] + parity_lines + [""]
    REPORT.write_text("\n".join(lines))
    print(f"report written: {REPORT}")


def addendum():
    """ADDED CELL (1 of ≤3 allowed, diagnostic, ungated).

    Rationale (written to the report BEFORE running, per protocol): the E3
    pitch is 'similar return, smaller factor crashes than RAW momentum'.
    The pre-registered grid contains no raw-momentum control at the same
    1x leverage / n_long=30, and no other family ledger has one, so the
    claim cannot be verified without this cell. It is a comparison
    diagnostic only and takes no part in G1 selection.
    """
    rationale = (
        "\n## ADDED CELL (protocol: ≤3, clearly marked) — rationale\n\n"
        "The grid has no RAW-momentum control at matched leverage (1x) and "
        "n_long (30), and no other family ledger contains one, so the core "
        "E3 claim (similar return, smaller factor crashes than raw momentum) "
        "is unverifiable from pre-registered cells alone. Added cell: "
        "`xs_momentum(px, macro, n_long=30)` standalone 1x, diagnostic only, "
        "excluded from G1. Rationale written before the run.\n"
    )
    with open(REPORT, "a") as f:
        f.write(rationale)

    panel, macro, _ = load_cache()
    w_xs = pd.read_parquet(ROOT / "weights_store" / "sleeve_xs_momentum_n30.parquet")
    res = run_trial(w_xs * 1.0, name="e3_CONTROL_xsmom_raw_alone1x",
                    family=FAMILY,
                    params={"strategy": "xs_momentum", "n_long": N_LONG,
                            "leverage": 1.0, "role": "alone1x_control"},
                    window="dev",
                    notes="ADDED CELL: raw-momentum control for crash comparison")
    r = res["row"]
    line = (f"\nResult: | {r['name']} | {r['mean_monthly']*100:.3f} | "
            f"{r['sharpe']:.3f} | {r['max_drawdown']*100:.1f} | "
            f"{r['calmar']:.3f} | {r['turnover_ann']:.1f} | {r['avg_gross']:.2f} | "
            f"{r['ep_2018Q4_ret']*100:.1f} / {r['ep_2018Q4_dd']*100:.1f} | "
            f"{r['ep_covid_ret']*100:.1f} / {r['ep_covid_dd']*100:.1f} | "
            f"{r['ep_2022_ret']*100:.1f} / {r['ep_2022_dd']*100:.1f} | "
            f"(same columns as grid table)\n")
    with open(REPORT, "a") as f:
        f.write(line)
    print(line)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--addendum":
        addendum()
    else:
        main()
