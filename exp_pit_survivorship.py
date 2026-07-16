"""WS1c — universe+survivorship joint bound, pit_cache_only arm (plan: what-can-we-do
roadmap, Track B / WS1).

The local bar cache is survivors-only, so every published number is an upper
bound. This experiment measures HOW MUCH of that inflation comes from the
strategy being allowed to SELECT from the survivor universe, by re-running the
combo_v2 blend with point-in-time index membership (data_pit.membership_frame)
applied at selection time:

    baseline        : the three combo_v2 sleeves with eligible=None —
                      byte-identical to run_v6._build_combo_v2_baseline /
                      validation.reproduce_baseline.build_combo_v2_base, and
                      must reproduce catalog_v2's combo_v2_2x full-window row
                      (hard gate below).
    pit_cache_only  : the SAME three sleeves (n_long=30 each, blended /3)
                      with eligible=membership — at each decision date the
                      sleeve may only pick names that were actually in the
                      S&P 500 on that date. Engine prices stay UNMASKED so a
                      held name still marks correctly after an index exit.

Design notes (binding):
- Selection-time filtering, not panel masking: signals read shifted/rolling
  rows, so masking the panel corrupts lookbacks and blinds new entrants for
  252d (see strategies._restrict_to_eligible).
- delta = baseline - pit is a JOINT universe+survivorship effect, NOT a pure
  survivorship bound: measurement revealed a mean 79% of the baseline's gross
  weight sits on names that were NOT S&P members as-of the decision date
  (60% on never-members) — the cache universe is far broader than the index,
  so restricting to PIT S&P membership removes both the survivor-picking
  freedom AND the mid-cap universe where the published edge concentrates.
  The two are not separable with free data (no PIT history for the broader
  universe). The pit arm itself also REMAINS survivorship-inflated: dead
  members' bars are absent (24.8% of members at 2016 -> 0.6% at 2026), so
  even the pit numbers are upper bounds for an S&P-only variant. The
  pit_plus_stooq arm (blocked on the manual Stooq bulk download) tightens
  that residual.
- Both arms share identical data quality (same cache, same engine, same
  costs), which is what makes the delta clean.

EXCEPTION GRANTED (same precedent as exp_rescore.py / exp_catalog_v2.py):
window="full", final=True — this is a CORRECTION-STYLE measurement under
fixed, pre-declared settings (next_open, tc=5bp, margin 600bp/yr, cap 2.0),
not a search. Every run is ledgered (family="pit") and counts toward the
deflated-Sharpe trial count.

Grid: 2 universes x 3 leverages {1.0, 1.5, 2.0} x 2 windows {full, dev}.
Resume-safe: completed (name, window) pairs are skipped on re-run; the
results table is assembled from the ledger, not from in-memory state.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python exp_pit_survivorship.py
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data_pit
import strategies as S
from exp_lib import _CODE_HASH, TRIALS_DIR, cached_weights, load_cache, run_trial

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v7" / "pit"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = TRIALS_DIR / "pit.csv"

FAMILY = "pit"
MARGIN_BPS = 600.0
TC_BPS = 5.0
LEV_CAP = 2.0
LEVERAGES = (1.0, 1.5, 2.0)
NOTE = ("pit-correction exception (exp_rescore precedent); "
        "survivorship bound measurement")

# Baseline fidelity gate = catalog_v2's combo_v2_2x full-window row (which
# itself reproduced the REPORT.md corrected headline). If the baseline arm
# here doesn't match, the delta table is meaningless — hard-stop the report.
BASELINE_GATE_NAME = "combo_v2_baseline_2x"
BASELINE_TOL = {"geo_monthly": (0.0368, 0.002), "sharpe": (1.06, 0.03),
                "max_drawdown": (-0.603, 0.01)}


# ---------------------------------------------------------------------------
# weight builders (reproduce_baseline.build_combo_v2_base pattern)

def _blend3(w_xs: pd.DataFrame, w_dual: pd.DataFrame,
            w_adapt: pd.DataFrame) -> pd.DataFrame:
    """Byte-identical to run_v6._build_combo_v2_baseline's blend step."""
    dates = sorted(set(w_xs.index) | set(w_dual.index) | set(w_adapt.index))
    cols = sorted(set(w_xs.columns) | set(w_dual.columns) | set(w_adapt.columns))
    w_xs = w_xs.reindex(index=dates, columns=cols, fill_value=0.0)
    w_dual = w_dual.reindex(index=dates, columns=cols, fill_value=0.0)
    w_adapt = w_adapt.reindex(index=dates, columns=cols, fill_value=0.0)
    return (w_xs + w_dual + w_adapt) / 3.0


def build_baseline_base(px, macro) -> pd.DataFrame:
    """combo_v2 base, eligible=None — reuses the EXISTING sleeve cache keys
    (exp_rescore conventions), so under an unchanged _CODE_HASH these are
    pure cache hits and provably the same frames catalog_v2 scored."""
    w_xs = cached_weights("sleeve_xs_momentum_n30",
                          lambda: S.xs_momentum(px, macro, n_long=30))
    w_dual = cached_weights("sleeve_dual_momentum_voltarget_n30",
                            lambda: S.dual_momentum_voltarget(px, macro, n_long=30))
    w_adapt = cached_weights("sleeve_adaptive_voltarget_n30",
                             lambda: S.adaptive_voltarget_momentum(px, macro, n_long=30))
    return _blend3(w_xs, w_dual, w_adapt)


def build_pit_base(px, macro, membership: pd.DataFrame) -> pd.DataFrame:
    """combo_v2 base with selection-time PIT eligibility. NEW cache keys
    ("pit__" prefix) — they must never collide with the legacy sleeve keys,
    whose frames were built without the eligible filter.

    The membership frame is itself an input to the weights, so its identity
    must be part of the cache key (exp_lib's _CODE_HASH only covers strategy
    source): a re-scrape or RENAMES fix would otherwise silently serve
    sleeves built from the OLD membership.
    """
    mem_tag = hashlib.sha256(
        pd.util.hash_pandas_object(membership, index=True).values.tobytes()
    ).hexdigest()[:10]
    w_xs = cached_weights(
        f"pit__{mem_tag}__sleeve_xs_momentum_n30",
        lambda: S.xs_momentum(px, macro, n_long=30, eligible=membership))
    w_dual = cached_weights(
        f"pit__{mem_tag}__sleeve_dual_momentum_voltarget_n30",
        lambda: S.dual_momentum_voltarget(px, macro, n_long=30,
                                          eligible=membership))
    w_adapt = cached_weights(
        f"pit__{mem_tag}__sleeve_adaptive_voltarget_n30",
        lambda: S.adaptive_voltarget_momentum(px, macro, n_long=30,
                                              eligible=membership))
    return _blend3(w_xs, w_dual, w_adapt)


# ---------------------------------------------------------------------------
# run + assemble

def run_grid(bases: dict[str, pd.DataFrame]) -> None:
    done: set[tuple[str, str]] = set()
    if LEDGER.exists():
        led = pd.read_csv(LEDGER)
        done = set(zip(led["name"], led["window"]))
        print(f"[resume] {len(done)} runs already in ledger")

    for universe, base in bases.items():
        for lev in LEVERAGES:
            name = f"combo_v2_{universe}_{lev:g}x"
            params = {"universe": universe, "lev": lev, "n_long": 30,
                      "sleeves": "xs30+dual30+adapt30",
                      "source": "exp_pit_survivorship"}
            w = base * lev
            for window in ("full", "dev"):
                if (name, window) in done:
                    continue
                t0 = time.time()
                run_trial(w, name=name, family=FAMILY, params=params,
                          window=window, exec_model="next_open",
                          tc_bps=TC_BPS, leverage_cap=LEV_CAP,
                          margin_bps_annual=MARGIN_BPS,
                          final=(window == "full"), notes=NOTE)
                print(f"[done] {name:32s} {window:4s}  {time.time()-t0:5.1f}s")


def assemble() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(results, delta) from the ledger. results: one row per
    (universe, leverage, window); delta: full-window baseline minus pit."""
    led = pd.read_csv(LEDGER)
    led = led[led["family"] == FAMILY].copy()
    led = led.drop_duplicates(subset=["name", "window"], keep="last")
    meta = led["params_json"].map(json.loads)
    led["universe"] = [m["universe"] for m in meta]
    led["leverage"] = [m["lev"] for m in meta]
    led["geo_monthly"] = (1.0 + led["cagr"]) ** (1.0 / 12.0) - 1.0

    cols = ["universe", "leverage", "window", "geo_monthly", "mean_monthly",
            "sharpe", "max_drawdown", "calmar"]
    results = (led[cols]
               .rename(columns={"mean_monthly": "arith_monthly"})
               .sort_values(["window", "universe", "leverage"])
               .reset_index(drop=True))

    full = results[results["window"] == "full"]
    base = full[full["universe"] == "baseline"].set_index("leverage")
    pit = full[full["universe"] == "pit_cache_only"].set_index("leverage")
    metric_cols = ["geo_monthly", "arith_monthly", "sharpe",
                   "max_drawdown", "calmar"]
    delta = (base[metric_cols] - pit[metric_cols]).reset_index()
    delta.insert(0, "universe", "delta_baseline_minus_pit")
    delta["window"] = "full"
    return results, delta


def check_baseline_gate(results: pd.DataFrame) -> None:
    row = results[(results["universe"] == "baseline")
                  & (results["leverage"] == 2.0)
                  & (results["window"] == "full")]
    if row.empty:
        raise SystemExit("BASELINE GATE: no full-window baseline 2x row")
    row = row.iloc[0]
    ok = True
    print("\n== baseline fidelity gate (vs catalog_v2 combo_v2_2x, full) ==")
    for k, (target, tol) in BASELINE_TOL.items():
        d = abs(float(row[k]) - target)
        flag = "OK " if d <= tol else "FAIL"
        ok &= d <= tol
        print(f"  {flag} {k:14s} got={float(row[k]):+.4f} "
              f"target={target:+.4f} tol={tol}")
    if not ok:
        raise SystemExit(
            "BASELINE GATE FAILED — pit deltas would be measured against a "
            "non-reproducing baseline; PIT_REPORT.md not written.")


# ---------------------------------------------------------------------------
# report

def _baseline_offuniverse_stats(base: pd.DataFrame, membership: pd.DataFrame,
                                px: pd.DataFrame) -> dict:
    """Composition diagnostic: share of the baseline's gross weight sitting on
    names NOT PIT-eligible at each decision date, and the subset that was
    NEVER an index member in-window. Quantifies how much of the delta is
    universe narrowing (cache universe -> S&P) vs. dead-cohort selection
    freedom — essential for reading the delta honestly."""
    elig = S._eligibility_asof(membership, px.index)
    ever = set(membership.columns)
    inelig, never = [], []
    for dt, r in base.iterrows():
        held = r[r.abs() > 1e-12]
        if held.empty:
            continue
        erow = elig.loc[dt]
        ok = set(erow.index[erow.eq(True)])
        gross = held.abs().sum()
        inelig.append(held[~held.index.isin(ok)].abs().sum() / gross)
        never.append(held[~held.index.isin(ever)].abs().sum() / gross)
    inelig, never = pd.Series(inelig), pd.Series(never)
    return {"inelig_mean": float(inelig.mean()),
            "inelig_median": float(inelig.median()),
            "inelig_max": float(inelig.max()),
            "never_mean": float(never.mean())}


def _missing_rate_on(px: pd.DataFrame, membership: pd.DataFrame,
                     date: pd.Timestamp) -> tuple[float, int]:
    """(missing_rate, n_members) on an exact panel date — the sampled grid in
    missing_names.csv (every 21st trading day) does not land on the panel's
    final date, so the end-of-window headline is computed directly."""
    members = set(membership.columns[membership.loc[date]])
    have = set(px.columns[px.loc[date].notna()])
    n_missing = len(members - have)
    return (n_missing / len(members) if members else float("nan"), len(members))


def write_report(results: pd.DataFrame, delta: pd.DataFrame,
                 missing: pd.DataFrame, membership: pd.DataFrame,
                 px: pd.DataFrame, base_baseline: pd.DataFrame) -> Path:
    def tbl(df: pd.DataFrame) -> str:
        d = df.copy()
        for c in ("geo_monthly", "arith_monthly", "max_drawdown"):
            d[c] = (d[c] * 100).map(lambda v: f"{v:+.2f}%")
        for c in ("sharpe", "calmar"):
            d[c] = d[c].map(lambda v: f"{v:.2f}")
        d["leverage"] = d["leverage"].map(lambda v: f"{v:g}x")
        cols = ["leverage", "geo_monthly", "arith_monthly", "sharpe",
                "max_drawdown", "calmar"]
        d = d[cols]
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        body = "\n".join("| " + " | ".join(str(v) for v in r) + " |"
                         for r in d.itertuples(index=False))
        return f"{head}\n{sep}\n{body}"

    full = results[results["window"] == "full"]
    dev = results[results["window"] == "dev"]
    m_first, m_last = missing.iloc[0], missing.iloc[-1]
    end_rate, end_members = _missing_rate_on(px, membership, px.index[-1])
    counts = membership.sum(axis=1)
    off = _baseline_offuniverse_stats(base_baseline, membership, px)

    md = f"""# PIT universe+survivorship joint bound — pit_cache_only arm (WS1c)

Generated by `exp_pit_survivorship.py` (family `pit`); do not edit by hand.

## What was measured

The bar cache is **survivors-only**: names that delisted, blew up, or were
acquired inside the window (ATVI, CELG, TWTR, SIVB, FRC, ...) have no bars, so
every published backtest is an upper bound. This experiment bounds that
inflation by re-running the combo_v2 blend with **point-in-time S&P 500
membership** (Wikipedia change log, frozen + sha256'd; see `data_pit.py`)
applied **at selection time**:

- **baseline** — the three sleeves (`xs_momentum`, `dual_momentum_voltarget`,
  `adaptive_voltarget_momentum`, `n_long=30` each, blended /3) with
  `eligible=None`, i.e. exactly `run_v6._build_combo_v2_baseline`. Reproduces
  catalog_v2's `combo_v2_2x` full-window row (fidelity gate passed below).
- **pit_cache_only** — the SAME sleeves with `eligible=membership`: at each
  decision date a sleeve may only pick names that were actually in the index
  on that date. Signals are still computed on full price history (panel
  masking corrupts shifted/rolling lookbacks and blinds new entrants for
  252d), and **engine prices stay unmasked** so a held name marks correctly
  after an index exit.

Engine fixed both arms: `next_open`, tc = {TC_BPS:g} bp, margin =
{MARGIN_BPS:g} bp/yr on long gross > 1.0x NAV, leverage_cap = {LEV_CAP:g}.
Membership reconstruction guarded: daily count stayed in
[{int(counts.min())}, {int(counts.max())}] (band {data_pit.COUNT_BAND}), all
{len(data_pit.KNOWN_EVENTS)} known-event spot checks passed.

## Why the delta is a LOWER bound

The delisted cohort's bars are **still absent** from the cache, so the pit arm
cannot actually HOLD dead-cohort names — it only loses the freedom to select
survivors that were not yet index members. The strategy can't buy the names
that would have hurt it, because it can't see them. With those bars present
(the `pit_plus_stooq` arm, blocked on a manual Stooq bulk download) the
measured drag would be larger. Missing-name rate at the start of the window:
**{m_first['missing_rate']:.1%}** of {int(m_first['n_members'])} members on
{pd.Timestamp(m_first['date']).date()} had no bars, decaying to
**{end_rate:.1%}** of {end_members} members by {px.index[-1].date()}
(last sampled grid row: {m_last['missing_rate']:.1%} on
{pd.Timestamp(m_last['date']).date()}; fresh `missing_names.csv` alongside
this report) — the bound is weakest early in the window, exactly where the
missing cohort is largest.

## Results (full window, 2016-04 → 2026-03)

### baseline (survivors-only selection)

{tbl(full[full['universe'] == 'baseline'])}

### pit_cache_only (selection restricted to PIT members)

{tbl(full[full['universe'] == 'pit_cache_only'])}

### DELTA (baseline − pit) = joint universe+survivorship effect, full window (NOT a pure survivorship bound — see composition caveat)

{tbl(delta)}

## Dev window (≤ 2022-12-31), for reference

### baseline

{tbl(dev[dev['universe'] == 'baseline'])}

### pit_cache_only

{tbl(dev[dev['universe'] == 'pit_cache_only'])}

## Composition caveat — the delta is NOT purely dead-cohort selection freedom

The bar cache is broader than the S&P 500: the panel holds
{px.shape[1]} names but only {membership.shape[1]} were EVER index members
in-window. Measured on the published baseline blend's decision dates, on
average **{off['inelig_mean']:.0%}** of gross weight (median
{off['inelig_median']:.0%}, max {off['inelig_max']:.0%}) sat on names that
were NOT S&P members as-of the decision date — and **{off['never_mean']:.0%}**
on names that were never members at any point in the window. The PIT
restriction therefore changes two things at once: (1) it removes not-yet /
no-longer members (the survivorship mechanism proper), and (2) it shrinks the
selection universe from the whole cache (~{px.shape[1]} survivors-of-today
names) to the ~{int(counts.median())}-name index. The cache's non-S&P names
are ALSO survivors-of-today (they exist in the cache because they exist now),
so the published number is survivorship-inflated across its whole universe —
but the delta below should be read as "published strategy vs. its
PIT-S&P-restricted twin", i.e. the honest bound on the PUBLISHED
configuration, not a per-name S&P survivorship decomposition.

## Sign discussion (why the delta can be small or even negative)

The survivorship *bound* is an aggregate-expectation statement, not a
name-by-name one. Individual exits cut both ways: acquired names typically
leave the index **at a premium** (CELG, TWTR, ATVI closed near deal prices),
while failures (SIVB, FRC) leave near zero. A momentum selector restricted to
PIT members can also end up holding *better* names in some periods —
non-members it would otherwise have picked were often recent IPOs/new
entrants with hot trailing returns that mean-reverted. The honest reading:
the full-window delta measures the **selection-freedom component** of
survivorship inflation under this cache; the **holding-dead-names component**
is unmeasured here and is the part with the mechanically negative expectation
(you cannot lose less than 0 on names you were never allowed to hold, but the
real index holder rode SIVB to ~0). Hence: lower bound.

## Caveats

- `missing_names.csv`: per-rebalance-date member counts, bar coverage and the
  missing tickers themselves (headline: {m_first['missing_rate']:.1%} missing
  in 2016 → {end_rate:.1%} in 2026).
- Wikipedia's change log is titled "Selected changes"; within this window the
  reconstructed count band held, but the log is the single source.
- All 12 runs ledgered in `results/v7/trials/pit.csv` and count toward the
  deflated-Sharpe trial total.

Provenance: strategy code hash `{_CODE_HASH}`; membership from the frozen
events parquet (`data_pit/sp500_events.parquet`, sha256 in its
`.meta.json`); rows generated {pd.Timestamp.now().isoformat(timespec='seconds')}.
"""
    path = OUT / "PIT_REPORT.md"
    path.write_text(md)
    return path


# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    panel, macro, _ = load_cache()
    px = panel["close"]

    print("building PIT membership over the panel's daily index "
          "(guards run inside)...")
    membership = data_pit.membership_frame(px.index)
    print(f"  membership: {membership.shape[0]} days x "
          f"{membership.shape[1]} ever-members")

    print("refreshing missing-name report...")
    missing = data_pit.write_missing_name_report(px, membership)
    print(f"  {len(missing)} rebalance dates; missing_rate "
          f"first={missing['missing_rate'].iloc[0]:.1%} "
          f"last={missing['missing_rate'].iloc[-1]:.1%}")

    print("building baseline sleeves (existing cache keys — should be hits)...")
    base_baseline = build_baseline_base(px, macro)
    print("building pit sleeves (pit__ keys — fresh builds on first run, "
          "several minutes each)...")
    base_pit = build_pit_base(px, macro, membership)

    run_grid({"baseline": base_baseline, "pit_cache_only": base_pit})

    results, delta = assemble()
    check_baseline_gate(results)

    out_csv = OUT / "pit_results.csv"
    pd.concat([results, delta], ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    report_path = write_report(results, delta, missing, membership, px,
                               base_baseline)
    print(f"wrote {report_path}")

    print("\n== DELTA (baseline - pit_cache_only), full window ==")
    for _, r in delta.iterrows():
        print(f"  {r['leverage']:g}x: geo {r['geo_monthly']*100:+.2f}pp/mo | "
              f"sharpe {r['sharpe']:+.2f} | "
              f"maxDD {r['max_drawdown']*100:+.2f}pp")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
