"""WS4(a) — risk-free-adjusted Sharpe reporting (^IRX), Stream 2 of the
Model Improvement Roadmap v2.

REPORTING CHANGE, NO GATE. This script selects nothing and gates nothing: it
quantifies how much of the published Sharpe ratios is just the T-bill rate.
(The gate template for the sibling WS4 experiments — dev DSR >= 0.95 at the
current deduped trial count AND dev Calmar >= baseline+10% AND geo give-up
<= 0.2pp/mo, one-shot val LATER — does not apply to a reporting correction;
recorded here for family-level consistency of the docstring convention.)

What it does
------------
1. Fetches ^IRX (13-week T-bill discount rate) through the provenance-frozen
   data_free layer. ^IRX is quoted in PERCENT (5.2 = 5.2%); divided by 100
   here to the annualized-decimal convention metrics._excess_daily expects.
2. Verifies the additive rf-Series extension of metrics.sharpe/sortino
   (scalar path byte-identical; constant Series == scalar) — hard asserts.
3. EXACT raw-vs-excess dev-window Sharpe for the champion (combo_v2_2x) and
   the top-3 single-strategy rows of the catalog_v2 full-window geo ranking
   (xs_momentum_top30, dual_momentum_vol, adaptive_voltarget_momentum):
   their cached catalog weight frames (cache-hit ONLY — a miss aborts, so we
   provably score the exact frames catalog_v2 scored) are re-rendered through
   run_trial with log=False. Identical re-runs of already-ledgered configs
   are not new hypotheses (exp_lib convention) — NOTHING is appended to any
   ledger; the catalog_v2 family already carries these configs.
4. APPROXIMATE full-window excess Sharpe for the catalog_v2 top-10 (by full
   geo_monthly): equity curves are not stored per catalog row and re-running
   window="full" would require final=True, which this task is barred from
   invoking (no final=True/val runs). Instead the render-time identity
   sharpe_excess ~= sharpe_raw - mean(rf)/ann_vol (exact when rf is constant;
   the error term is cov(r, rf)/var-scale, negligible at daily granularity)
   is applied to the STORED catalog scalars. The approximation is validated
   against the exact dev-window pairs from step 3 and the max abs error is
   printed and stored in the output.

Window note: exact pairs are dev-only (<= 2022-12-31). Dev spans the
near-zero-rate 2016-2021 era, so the dev delta UNDERSTATES the full-window
effect — the 2023-2026 ~5% T-bill era is where raw Sharpe flatters most.
The approximate full-window column carries that part of the story.

Output: results/v7/rf_sharpe_table.csv (consumed by report_current.py).

Engine for exact pairs: next_open, tc/cap per the original catalog registry
entry, margin_bps_annual=600 (the corrected catalog_v2 cost basis).

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python exp_ws4_rf.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_free import fetch_yf_series
from exp_lib import _CODE_HASH, WEIGHTS_STORE, run_trial
from metrics import sharpe
from metrics_v2 import DEV_END

ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "results" / "v7" / "rf_sharpe_table.csv"
CATALOG = ROOT / "results" / "v7" / "catalog_v2.csv"

MARGIN_BPS = 600.0

# (name, tc_bps, leverage_cap) per exp_rescore.REGISTRY — the exact catalog
# scoring settings. All four are sparse decision-date frames.
EXACT_SET = [
    ("combo_v2_2x", 5.0, 2.0),                  # champion
    ("xs_momentum_top30", 5.0, 1.0),            # top single #1 (full geo)
    ("dual_momentum_vol", 5.0, 1.0),            # top single #2
    ("adaptive_voltarget_momentum", 5.0, 2.0),  # top single #3
]

TOP_N_APPROX = 10


def load_irx() -> pd.Series:
    """^IRX as an annualized DECIMAL rate series (percent / 100)."""
    s = fetch_yf_series("^IRX")
    return s / 100.0


def cached_frame_or_die(key: str) -> pd.DataFrame:
    """Load a weights_store frame for the CURRENT code hash, never rebuild.

    Rebuilding here could silently score different frames than catalog_v2
    did; a cache miss means the strategy sources changed since the catalog
    run and the whole comparison is void — abort instead.
    """
    p = WEIGHTS_STORE / f"{key}__{_CODE_HASH}.parquet"
    if not p.exists():
        raise SystemExit(
            f"cache miss: {p.name} — strategy sources changed since the "
            f"catalog_v2 run (code hash {_CODE_HASH}); rf table would not "
            f"be comparing the ledgered frames. Aborting."
        )
    return pd.read_parquet(p)


def verify_metrics_extension(irx: pd.Series) -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-02", periods=400)
    eq = pd.Series(100e3 * np.cumprod(1 + rng.normal(5e-4, 0.01, 400)), index=idx)
    assert sharpe(eq) == sharpe(eq, rf=0.0), "scalar default changed"
    const = pd.Series(0.05, index=idx)
    assert abs(sharpe(eq, rf=const) - sharpe(eq, rf=0.05)) < 1e-12, \
        "constant-series != scalar"
    assert np.isfinite(sharpe(eq, rf=irx)), "real ^IRX series not accepted"
    print("metrics rf-extension checks: OK")


def main() -> None:
    t0 = time.time()
    irx = load_irx()
    print(f"^IRX: {len(irx)} rows, {irx.index[0].date()} -> "
          f"{irx.index[-1].date()}; last = {irx.iloc[-1]:.4f}")
    verify_metrics_extension(irx)

    cat = pd.read_csv(CATALOG)
    cat_dev = cat[cat["window"] == "dev"].drop_duplicates("name", keep="last")
    cat_full = (cat[cat["window"] == "full"]
                .drop_duplicates("name", keep="last")
                .sort_values("geo_monthly", ascending=False))

    rows = []

    # ── exact dev-window pairs (log=False re-render; nothing ledgered) ──
    print("\nexact dev-window raw vs excess Sharpe (re-render, log=False):")
    approx_errs = []
    for name, tc, cap in EXACT_SET:
        w = cached_frame_or_die(f"catalog_{name}")
        r = run_trial(w, name=name, family="catalog_v2",
                      params={"rf_rerender": True}, window="dev",
                      exec_model="next_open", tc_bps=tc, leverage_cap=cap,
                      margin_bps_annual=MARGIN_BPS, final=False, log=False)
        eq = r["equity"]
        sh_raw = sharpe(eq)
        sh_exc = sharpe(eq, rf=irx)
        # fidelity: must reproduce the ledgered catalog_v2 dev sharpe
        led = cat_dev.loc[cat_dev["name"] == name, "sharpe"]
        d_led = abs(sh_raw - float(led.iloc[0])) if len(led) else float("nan")
        assert d_led < 1e-6, (
            f"{name}: re-rendered dev sharpe {sh_raw:.6f} != ledgered "
            f"{float(led.iloc[0]):.6f} — fidelity broken, table void")
        rf_mean = float(irx.reindex(eq.index).ffill().fillna(0.0).mean())
        ann_vol = float(eq.pct_change().dropna().std() * np.sqrt(252.0))
        sh_approx = sh_raw - rf_mean / ann_vol
        approx_errs.append(abs(sh_approx - sh_exc))
        rows.append({"name": name, "window": "dev", "method": "exact",
                     "sharpe_raw": sh_raw, "sharpe_excess": sh_exc,
                     "sharpe_delta": sh_exc - sh_raw,
                     "rf_mean_ann": rf_mean})
        print(f"  {name:30s} raw {sh_raw:6.3f} -> excess {sh_exc:6.3f} "
              f"(delta {sh_exc-sh_raw:+.3f}; rf_mean {rf_mean:.2%}; "
              f"approx err {abs(sh_approx-sh_exc):.4f})")
    max_err = max(approx_errs)
    print(f"approximation max abs error on exact pairs: {max_err:.4f}")

    # ── approximate full-window pairs for the top-10 ──
    # Full window = the panel span every catalog row was scored on
    # (2016-04 .. 2026-03); rf averaged over that span.
    from exp_lib import union_prices_cached
    pu_idx = union_prices_cached().index
    rf_full = float(irx.reindex(pu_idx).ffill().fillna(0.0).mean())
    rf_dev = float(irx.reindex(pu_idx[pu_idx <= pd.Timestamp(DEV_END)])
                   .ffill().fillna(0.0).mean())
    print(f"\nrf mean: dev {rf_dev:.2%} | full {rf_full:.2%}")
    print(f"approx full-window excess Sharpe, catalog_v2 top-{TOP_N_APPROX}:")
    for _, r in cat_full.head(TOP_N_APPROX).iterrows():
        sh_raw = float(r["sharpe"])
        av = float(r["ann_vol"])
        sh_exc = sh_raw - rf_full / av
        rows.append({"name": r["name"], "window": "full",
                     "method": "approx_from_scalars",
                     "sharpe_raw": sh_raw, "sharpe_excess": sh_exc,
                     "sharpe_delta": sh_exc - sh_raw,
                     "rf_mean_ann": rf_full})
        print(f"  {r['name']:30s} raw {sh_raw:6.3f} -> excess {sh_exc:6.3f} "
              f"(delta {sh_exc-sh_raw:+.3f})")

    out = pd.DataFrame(rows)
    out["approx_max_abs_err_on_exact_pairs"] = max_err
    out["code_hash"] = _CODE_HASH
    out["generated"] = pd.Timestamp.now().isoformat(timespec="seconds")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(out)} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
