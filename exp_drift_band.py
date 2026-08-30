"""exp_drift_band.py — DRIFT1: drift-band rebalancing on the champion at
measured costs (governed by results/v7/improve3/DRIFT1_PREREG.md).

STATUS: FROZEN 2026-08-30 (prereg sha256
ee52fe6c3be1f6961a8b471e6ea52dd66a96d6928ddba141020e2d963ca13811, recorded
in the freeze commit). The do-not-run guard was removed and PREREG_FROZEN
set True in this commit, per the freeze terms — dev cells now ledger to
family drift1 on first render.

Design (verbatim from the prereg — the prereg is authoritative):
  - Construction: champion combo_v2 2x via exp_pit_survivorship.
    build_baseline_base (cached sleeves, blend /3) x 2.0 — the exact frame
    exp_tc_recal.py's champion track scored.
  - Transform: iterative pass over the SPARSE decision-date frame; hold
    current weights; on each decision date trade a name only if
    |target − held| > band (absolute weight points, post-leverage frame);
    held weights are piecewise-constant between decisions — byte-consistent
    with engine_v2's own no-price-drift convention (disclosed in the prereg).
  - Grid: band in {0 (control), 0.0025, 0.005, 0.01} x tc in {30, 50},
    window="dev", margin_bps_annual=600, next_open, leverage_cap=2.0.
  - Fidelity gates (hard-asserted before any banded cell):
      (1) band=0 transform returns the input frame bit-identically;
      (2) band=0 log=False dev runs reproduce results/v7/trials/tc_recal.csv
          combo_v2_2x_tc30 / combo_v2_2x_tc50 dev rows
          (|dSharpe| < 1e-6, |dCAGR| < 1e-8).
  - Pass bar + winner selector + one-shot val gates: see the prereg. The dev
    phase here computes the verdict inputs only; validate() is a separate
    entry point that must never run before a dev PASS is recorded.

Ledger: results/v7/trials/drift1.csv (NEW margin family; schema in the
prereg). Every ledgered cell counts toward the deduped DSR trial count.

Run (AFTER freeze only):
  cd "Traiding 11" && /opt/anaconda3/bin/python exp_drift_band.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))

from exp_lib import TRIALS_DIR, load_cache, run_trial

# ── prereg freeze switch ────────────────────────────────────────────────────
# results/v7/improve3/DRIFT1_PREREG.md frozen 2026-08-30 (sha in the freeze
# commit) — ledgering enabled per the freeze terms.
PREREG_FROZEN = True

FAMILY = "drift1"
LEDGER = TRIALS_DIR / f"{FAMILY}.csv"
OUT_CSV = ROOT / "results" / "v7" / "improve3" / "drift1_dev.csv"

MARGIN_BPS = 600.0
LEV = 2.0
TCS = (30.0, 50.0)
BANDS = (0.0025, 0.005, 0.01)   # band=0 is the log=False control, not a cell
GATE_TOL_SR = 1e-6
GATE_TOL_CAGR = 1e-8

NOTE = ("DRIFT1 prereg (results/v7/improve3/DRIFT1_PREREG.md): drift-band "
        "rebalancing on champion combo_v2 2x at measured tc; piecewise-"
        "constant held weights (engine-consistent, disclosed)")


# ────────────────────────────────────────────────────────────────────────────
# transform

def drift_band_transform(w: pd.DataFrame, band: float) -> pd.DataFrame:
    """Sparse decision frame -> drift-banded sparse frame.

    Row 0 establishes the full book. On each later decision date, a name
    trades (held := target) only if |target − held| > band; otherwise the
    held weight carries forward unchanged (piecewise-constant between
    decisions — the engine's own convention; see prereg disclosure).

    band=0.0 reproduces the input EXACTLY: strict inequality means a name
    with target == held simply keeps that identical value, and every name
    with any difference trades to target. Asserted by the caller.
    """
    a = w.to_numpy(dtype=float)
    out = np.empty_like(a)
    held = a[0].copy()
    out[0] = held
    for i in range(1, a.shape[0]):
        target = a[i]
        mask = np.abs(target - held) > band
        held = held.copy()
        held[mask] = target[mask]
        out[i] = held
    return pd.DataFrame(out, index=w.index, columns=w.columns)


# ────────────────────────────────────────────────────────────────────────────
# helpers

def geo_from_cagr(cagr: float) -> float:
    return (1.0 + cagr) ** (1.0 / 12.0) - 1.0


def build_champion_frame() -> pd.DataFrame:
    from exp_pit_survivorship import build_baseline_base
    panel, macro, _ = load_cache()
    return build_baseline_base(panel["close"], macro) * LEV


def control_refs() -> dict[float, pd.Series]:
    """The ledgered band=0 control rows (prereg: tc_recal.csv dev rows)."""
    led = pd.read_csv(TRIALS_DIR / "tc_recal.csv")
    out = {}
    for tc in TCS:
        r = led[(led["name"] == f"combo_v2_2x_tc{tc:g}")
                & (led["window"] == "dev")]
        assert len(r) >= 1, \
            f"missing control row combo_v2_2x_tc{tc:g} dev in trials/tc_recal.csv"
        out[tc] = r.iloc[-1]
    return out


def run_cell(w: pd.DataFrame, *, name: str, tc: float, band: float,
             window: str = "dev", log: bool = False, final: bool = False,
             notes: str = NOTE) -> dict:
    """Single engine path for every cell — identical to exp_tc_recal.py's
    champion track (run_trial front door)."""
    params = {"band": band, "lev": LEV, "tc_bps": tc, "n_long": 30,
              "sleeves": "xs30+dual30+adapt30",
              "margin_bps_annual": MARGIN_BPS, "source": "exp_drift_band"}
    return run_trial(w, name=name, family=FAMILY, params=params,
                     window=window, exec_model="next_open", tc_bps=tc,
                     leverage_cap=2.0, margin_bps_annual=MARGIN_BPS,
                     final=final, log=log, notes=notes)


# ────────────────────────────────────────────────────────────────────────────
# dev phase

def fidelity_gates(w_frame: pd.DataFrame, refs: dict[float, pd.Series]) -> None:
    """Prereg gates 1+2 — hard-asserted before any banded cell renders."""
    w0 = drift_band_transform(w_frame, 0.0)
    assert np.array_equal(w0.to_numpy(), w_frame.to_numpy()) \
        and w0.index.equals(w_frame.index) and w0.columns.equals(w_frame.columns), \
        "gate 1 FAILED: band=0 transform is not bit-identical to the input"
    print("[gate 1] band=0 transform == input frame (bit-identical)")

    for tc, ref in refs.items():
        r = run_cell(w0, name=f"drift_b0_tc{tc:g}_gate", tc=tc, band=0.0,
                     log=False, notes="fidelity gate re-render, NOT ledgered")
        d_sr = abs(r["summary"]["sharpe"] - float(ref["sharpe"]))
        d_cg = abs(r["summary"]["cagr"] - float(ref["cagr"]))
        print(f"[gate 2] tc={tc:g}: |dSR|={d_sr:.2e} |dCAGR|={d_cg:.2e} "
              f"vs trials/tc_recal.csv combo_v2_2x_tc{tc:g} dev")
        assert d_sr < GATE_TOL_SR and d_cg < GATE_TOL_CAGR, \
            f"gate 2 FAILED at tc={tc:g} vs results/v7/trials/tc_recal.csv"


def dev_phase() -> pd.DataFrame:
    refs = control_refs()
    w_frame = build_champion_frame()
    fidelity_gates(w_frame, refs)

    done: set[tuple[str, str]] = set()
    if LEDGER.exists():
        led = pd.read_csv(LEDGER)
        done = set(zip(led["name"], led["window"]))
        print(f"[resume] {len(done)} cells already ledgered")

    rows: list[dict] = []
    for tc in TCS:
        ref = refs[tc]
        rows.append({"band": 0.0, "tc_bps": tc,
                     "geo_monthly": geo_from_cagr(float(ref["cagr"])),
                     "max_drawdown": float(ref["max_drawdown"]),
                     "sharpe": float(ref["sharpe"]),
                     "turnover_ann": float(ref["turnover_ann"]),
                     "source": f"results/v7/trials/tc_recal.csv "
                               f"name=combo_v2_2x_tc{tc:g} window=dev"})
    for band in BANDS:
        wb = drift_band_transform(w_frame, band)
        for tc in TCS:
            name = f"drift_b{band:g}_tc{tc:g}"
            t0 = time.time()
            log = PREREG_FROZEN and (name, "dev") not in done
            res = run_cell(wb, name=name, tc=tc, band=band, log=log)
            s = res["summary"]
            rows.append({"band": band, "tc_bps": tc,
                         "geo_monthly": s["geo_monthly"],
                         "max_drawdown": s["max_drawdown"],
                         "sharpe": s["sharpe"],
                         "turnover_ann": s["turnover_annualized"],
                         "source": (f"results/v7/trials/drift1.csv name={name} "
                                    f"window=dev" if log else
                                    f"UNLEDGERED (log=False, prereg not "
                                    f"frozen) name={name}")})
            print(f"[dev] {name:20s} geo={s['geo_monthly']*100:+.3f}%/mo "
                  f"dd={s['max_drawdown']*100:5.1f}% "
                  f"turn={s['turnover_annualized']:.1f} "
                  f"({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def verdict(df: pd.DataFrame) -> None:
    """Prereg pass bar: band b beats band=0 on dev net geo at BOTH tc levels
    AND MaxDD guardrail (>= control − 2pp) at both. Winner: highest geo at
    tc=50; tie -> smaller band."""
    ctrl = df[df["band"] == 0.0].set_index("tc_bps")
    qual = []
    for band in BANDS:
        sub = df[df["band"] == band].set_index("tc_bps")
        beats = all(sub.loc[tc, "geo_monthly"] > ctrl.loc[tc, "geo_monthly"]
                    for tc in TCS)
        guard = all(sub.loc[tc, "max_drawdown"] >= ctrl.loc[tc, "max_drawdown"] - 0.02
                    for tc in TCS)
        print(f"  band {band:g}: beats-control(both tc)={beats} "
              f"dd-guardrail={guard}")
        if beats and guard:
            qual.append((band, float(sub.loc[50.0, "geo_monthly"])))
    if not qual:
        print("\nDEV VERDICT: FAIL — no band beats the band=0 control at both "
              "tc levels with the guardrail intact. No val shot; family "
              "closes per prereg.")
        return
    qual.sort(key=lambda t: (-t[1], t[0]))  # max geo@tc50, tie -> smaller band
    winner = qual[0][0]
    print(f"\nDEV VERDICT: PASS — winner band {winner:g} "
          f"(highest dev geo at tc=50 among {len(qual)} qualifier(s)). "
          f"ONE-SHOT validation per prereg: validate({winner:g}) — a "
          f"separate, deliberate invocation AFTER this verdict is committed.")


def main() -> None:
    df = dev_phase()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(df)} rows)")
    print("\n== drift1 dev table (geo %/mo | maxDD | turnover) ==")
    for _, r in df.sort_values(["tc_bps", "band"]).iterrows():
        print(f"  tc={r['tc_bps']:g} band={r['band']:g}: "
              f"{r['geo_monthly']*100:+.3f}% | {r['max_drawdown']*100:5.1f}% "
              f"| {r['turnover_ann']:5.1f}  [{r['source']}]")
    print("\n== pass-bar evaluation (prereg section 'Declared metrics') ==")
    verdict(df)


# ────────────────────────────────────────────────────────────────────────────
# one-shot validation (winner only — NEVER called by main; prereg gates)

def validate(winner_band: float) -> None:
    """One-shot val of the single dev winner (window='val', final=True, both
    tc levels; verdict reads at tc=50 only). Control comparator derived from
    a log=False full re-render of the ledgered tc_recal configs, fidelity-
    asserted before the val slice is read. See prereg 'One-shot validation'.
    Refuses to run before the freeze."""
    if not PREREG_FROZEN:
        raise SystemExit("validate(): prereg not frozen — no val shot.")
    assert winner_band in BANDS, "winner must be one of the pre-registered bands"
    w_frame = build_champion_frame()
    refs = control_refs()
    fidelity_gates(w_frame, refs)
    wb = drift_band_transform(w_frame, winner_band)

    led_full = pd.read_csv(TRIALS_DIR / "tc_recal.csv")
    for tc in TCS:
        # control: full-window log=False re-render of the ledgered config,
        # asserted vs tc_recal full row, then sliced to the val window.
        fref = led_full[(led_full["name"] == f"combo_v2_2x_tc{tc:g}")
                        & (led_full["window"] == "full")].iloc[-1]
        rc = run_cell(w_frame, name=f"ctrl_b0_tc{tc:g}_val", tc=tc, band=0.0,
                      window="full", final=True, log=False,
                      notes="control comparator re-render, NOT ledgered")
        d_sr = abs(rc["summary"]["sharpe"] - float(fref["sharpe"]))
        assert d_sr < GATE_TOL_SR, \
            f"control full re-render mismatch vs tc_recal.csv at tc={tc:g}"
        from metrics import summary as msum
        eqc = rc["equity"].loc["2023-01-01":]
        sc = msum(eqc / eqc.iloc[0] * 100_000.0, name=f"ctrl_val_tc{tc:g}")

        rv = run_cell(wb, name=f"drift_b{winner_band:g}_tc{tc:g}", tc=tc,
                      band=winner_band, window="val", final=True,
                      log=PREREG_FROZEN)
        sv = rv["summary"]
        print(f"[val tc={tc:g}] winner geo={sv['geo_monthly']*100:+.3f}%/mo "
              f"dd={sv['max_drawdown']*100:.1f}% | control "
              f"geo={sc['geo_monthly']*100:+.3f}%/mo "
              f"dd={sc['max_drawdown']*100:.1f}%")
        if tc == 50.0:
            ok = (sv["geo_monthly"] >= sc["geo_monthly"]
                  and sv["max_drawdown"] >= sc["max_drawdown"] - 0.02)
            print(f"\nVAL VERDICT (tc=50 only, per prereg): "
                  f"{'PASS' if ok else 'FAIL — family drift1 BURNED'}")


if __name__ == "__main__":
    main()
