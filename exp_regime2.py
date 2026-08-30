"""exp_regime2.py — REGIME2: corrected relative DSR bar for the two
Stream-2 overlay near-misses (governed by
results/v7/improve3/REGIME2_PREREG.md, FROZEN 2026-08-30, sha256
9a6e6a2aca803ff9aec7f7c85d7e07538067bc617ffac216f310354439a9b93e recorded
in the freeze commit).

PURE RECOMPUTATION — no engine run, no run_trial call, no ledger append
(the prereg's "Verified repo facts" #3). The only computation is
metrics_v2.deflated_sharpe on already-ledgered Sharpes with evaluation-time
(n_trials, sr_variance), plus the re-affirmation of the two non-DSR legs
from ledgered scalars. Output: results/v7/improve3/REGIME2_RESULTS.md.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))

from exp_lib import TRIALS_DIR, trial_count
from exp_catalog_v2 import ledger_sr_variance_daily
from metrics_v2 import deflated_sharpe

OUT = ROOT / "results" / "v7" / "improve3" / "REGIME2_RESULTS.md"

CELLS = {
    # name -> (ledger file, row name); every scalar is RE-READ at run time.
    "vixterm_thr1.05_m0.5": ("regime_vixterm.csv", "vixterm_thr1.05_m0.5"),
    "breadth_ad63_p10_m0.5": ("regime_breadth.csv", "breadth_ad63_p10_m0.5"),
}
ANCHOR = ("catalog_v2.csv", "combo_v2_2x")
ORIGINALS = {  # cited context from ws4_*_results.csv (prereg "Cells")
    "vixterm_thr1.05_m0.5": (0.766000, 707),
    "breadth_ad63_p10_m0.5": (0.763061, 715),
    "anchor": (0.740223, 707),
}


def dev_row(fname: str, name: str) -> pd.Series:
    led = pd.read_csv(TRIALS_DIR / fname)
    r = led[(led["name"] == name) & (led["window"] == "dev")]
    assert len(r) >= 1, f"missing {name} dev in trials/{fname}"
    return r.iloc[-1]


def geo_from_cagr(cagr: float) -> float:
    return (1.0 + cagr) ** (1.0 / 12.0) - 1.0


def main() -> None:
    # n_days_dev: the IDENTICAL expression exp_ws4_vixterm.py used
    # (prereg "Corrected DSR procedure").
    from exp_ws4_vixterm import build_base_daily
    w_daily = build_base_daily()
    n_days_dev = int(w_daily.loc[:"2022-12-31"].shape[0])
    n_trials = trial_count(dedupe=True)
    sr_var = ledger_sr_variance_daily()

    anchor = dev_row(*ANCHOR)
    anchor_sh = float(anchor["sharpe"])
    anchor_calmar = float(anchor["calmar"])
    anchor_geo = geo_from_cagr(float(anchor["cagr"]))
    dsr_anchor = deflated_sharpe(anchor_sh, n_days_dev, n_trials,
                                 sr_variance_across_trials=sr_var)

    lines = [
        "# REGIME2 RESULTS — corrected relative DSR recomputation",
        "",
        f"Computed {date.today().isoformat()} per REGIME2_PREREG.md (frozen "
        "2026-08-30; sha in freeze commit). Pure recomputation — no engine "
        "run, no ledger row.",
        "",
        f"Inputs at evaluation time: n_days_dev={n_days_dev}, "
        f"n_trials={n_trials} (deduped), sr_variance_daily={sr_var:.6e} "
        f"(originals: n=707/715, 2.00e-04 per RESULTS_CURRENT.md).",
        "",
        f"Anchor combo_v2_2x dev (results/v7/trials/catalog_v2.csv): sharpe "
        f"{anchor_sh:.6f}, calmar {anchor_calmar:.6f}, geo "
        f"{anchor_geo:.4%}/mo, corrected DSR {dsr_anchor:.6f} (original "
        f"{ORIGINALS['anchor'][0]:.6f} at n={ORIGINALS['anchor'][1]}).",
        "",
        "| cell | ledgered sharpe | corrected DSR | vs anchor | calmar leg "
        "(>=1.10x anchor) | geo leg (>= anchor - 0.2pp) | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    verdicts = {}
    for cell, (fname, name) in CELLS.items():
        r = dev_row(fname, name)
        sh = float(r["sharpe"])
        calmar = float(r["calmar"])
        geo = geo_from_cagr(float(r["cagr"]))
        dsr = deflated_sharpe(sh, n_days_dev, n_trials,
                              sr_variance_across_trials=sr_var)
        leg_dsr = dsr >= dsr_anchor
        leg_calmar = calmar >= 1.10 * anchor_calmar
        leg_geo = geo >= anchor_geo - 0.002
        ok = leg_dsr and leg_calmar and leg_geo
        verdicts[cell] = (ok, dsr)
        lines.append(
            f"| {cell} | {sh:.6f} | {dsr:.6f} (orig "
            f"{ORIGINALS[cell][0]:.6f}) | "
            f"{'PASS' if leg_dsr else 'FAIL'} | "
            f"{calmar:.4f} vs {1.10 * anchor_calmar:.4f} -> "
            f"{'PASS' if leg_calmar else 'FAIL'} | "
            f"{geo:.4%} vs {anchor_geo - 0.002:.4%} -> "
            f"{'PASS' if leg_geo else 'FAIL'} | "
            f"{'PASS' if ok else 'FAIL'} |")

    passing = [c for c, (ok, _) in verdicts.items() if ok]
    lines += ["", "## Verdict (per the frozen bar)", ""]
    if not passing:
        lines.append("Both cells FAIL — finding closed permanently; family "
                     "regime2 never ledgers a row.")
    else:
        val_cell = max(passing, key=lambda c: verdicts[c][1])
        lines += [
            f"PASS: {', '.join(passing)}. Per the frozen consequence "
            f"structure this changes NOTHING live and enters no "
            f"construction. Exactly one cell is val-eligible (higher "
            f"corrected DSR): **{val_cell}**.",
            "",
            "The one-shot validation is NOT authorized by REGIME2 — it "
            "requires a future REGIME3 pre-registration with val gates "
            "frozen before any val computation (REGIME2_PREREG.md "
            "'Consequences'). The determinism disclosure in the prereg "
            "anticipated this dev-side outcome; the genuine uncertainty "
            "lives entirely in that future val shot.",
        ]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
