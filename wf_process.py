"""WS3 — walk-forward process evaluation library (plan: what-can-we-do
roadmap, Track B / WS3).

The honest live-performance estimate is the performance of the SELECTION
PROCESS — re-selecting the book periodically on trailing data — not the
performance of the hindsight champion. This module implements the
pre-registered process:

    pool   : the 21 base single-strategy catalog names (POOL below), taken
             from exp_rescore.REGISTRY, excluding leverage/freeze/combo
             variants, shorts (xs_momentum_ls*), ml, and the buy_hold_spy
             benchmark.
    metric : Sharpe under next_open / tc=5bp / margin=600bp/yr at 1x on all
             history up to the selection date.
    step   : annual — select at Dec-31 of Y-1, trade year Y, legs 2019..2026.
    blend  : equal-weight of the top-3, run at leverage {2.0, 1.5, 1.0}.

Spec discipline (binding):
- write_spec() writes results/v7/wf/process_spec.json exactly ONCE and
  REFUSES to overwrite an existing file. The sha256 of the canonical json is
  recorded inside the file and embedded in every ledger note this module
  logs. The spec is frozen BEFORE any leg runs.
- Scoring at cutoffs 2023-12-31 .. 2025-12-31 necessarily reads data past
  metrics_v2.DEV_END. That is inherent to walk-forward and allowed here ONLY
  because the spec is frozen first; those scoring runs go through exp_lib's
  front door with window="full", final=True (the lock is honored, not
  bypassed), notes carrying the spec sha.
- truncation_equivalence_test is MANDATORY per pool member: a sleeve whose
  weights change when the input panel is truncated is non-causal and would
  leak the future into every leg — such sleeves are excluded from the pool
  (documented in the spec's exclusions list).

Scoring implementation note: each candidate is scored with ONE engine run
per leg, weights sliced to <= the cutoff. engine_v2 is day-causal (row-wise
cumprod; no forward information), so the Sharpe computed on the returned
equity sliced to <= cutoff is IDENTICAL to what a fully truncated engine run
would produce — verified property, and the sliced-Sharpe convention is
declared in the spec and in every wf_score ledger note.

DSR reuses exp_catalog_v2.ledger_sr_variance_daily (credit: exp_catalog_v2)
so the deflation uses the measured cross-trial SR variance, not the
degenerate fallback.

Library only — exp_wf_process.py is the runner. Tests:
validation/test_wf_process.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import strategies as S
from engine_v2 import BTConfigV2, run_backtest_v2
from exp_catalog_v2 import ledger_sr_variance_daily  # REUSED (see docstring)
from exp_lib import TRIALS_DIR, cached_weights, load_cache, run_trial, \
    trial_count, union_prices_cached
from metrics import summary as eq_summary
from metrics_v2 import DEV_END, deflated_sharpe, newey_west_tstat  # noqa: F401

ROOT = Path(__file__).resolve().parent
WF_DIR = ROOT / "results" / "v7" / "wf"
SPEC_PATH = WF_DIR / "process_spec.json"

# ── pre-registered process parameters (mirrored into the spec file) ────────
TC_BPS = 5.0
MARGIN_BPS = 600.0
SCORE_LEV_CAP = 1.0    # scoring is "at 1x"
PROC_LEV_CAP = 2.0     # process runs at blend x leverage, capped at 2.0
K = 3
LEVERAGES = (2.0, 1.5, 1.0)
FIRST_LEG = 2019
LAST_LEG = 2026
DEV_LEGS = (2019, 2020, 2021, 2022)
EQUIV_CUTOFF = "2020-12-31"
EQUIV_ATOL = 1e-10
EQUIV_MIN_COVERAGE = 0.9

# The 21 base single-strategy catalog names — exp_rescore.REGISTRY minus
# leverage variants (multi_dd_lev*, conc_dd_lev*, FINAL_3pct_lev*, combo_*
# at lev != base, kelly), freeze variants (combo_v2_2x_freeze_*), combos /
# ensemble overlays (combo_4way*, combo_v2*, FINAL_3pct_target,
# regime_*/voltarget_*/fast_dd_*/ensemble_top3), shorts (xs_momentum_ls,
# xs_momentum_ls_neutral), ml (not in REGISTRY), and the buy_hold_spy
# benchmark. Listed explicitly (spec requirement):
POOL = [
    # run_all.py
    "xs_momentum_12_1",
    "xs_momentum_top30",
    "mean_reversion_5d",
    "trend_following",
    "dual_momentum_vol",
    "sector_rotation",
    # run_v2.py
    "xs_momentum_multi",
    "xs_momentum_concentrated",
    # run_v3.py
    "ts_momentum_multiasset",
    "xs_momentum_fast",
    "xs_momentum_fast_top10",
    # run_v5.py
    "low_vol_quality",
    "acceleration_momentum",
    "donchian_breakout",
    "multifactor_mvr",
    "risk_parity_etf",
    "vix_gated_trend",
    "calendar_tom",
    "sector_momentum_rotation",
    "adaptive_voltarget_momentum",
    "momentum_quality_blend",
]
assert len(POOL) == 21, "pool must be exactly the 21 base catalog names"


# ═══════════════════════════════════════════════════════════════════════════
# Spec (write-once, sha256-sealed)
# ═══════════════════════════════════════════════════════════════════════════

def spec_sha256(spec: dict) -> str:
    """sha256 of the canonical json of the spec WITHOUT its own sha field."""
    d = {k: v for k, v in spec.items() if k != "sha256"}
    canon = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def write_spec(exclusions: list[dict] | None = None,
               path: Path | str = SPEC_PATH) -> dict:
    """Write the frozen process spec ONCE. Refuses to overwrite (frozen
    means frozen). `exclusions` = truncation-equivalence failures, each
    {"name": ..., "reason": ...} — the exclusion RULE is pre-registered; the
    list is filled from the mandatory causality audit run before the spec is
    sealed and before any leg runs."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists — the walk-forward spec is FROZEN; "
            f"refusing to overwrite. Delete it manually only if you intend "
            f"to restart the pre-registration (which voids all wf_* runs)."
        )
    spec = {
        "program": "wf_process (Track B / WS3 walk-forward process evaluation)",
        "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "pool": list(POOL),
        "pool_source": (
            "exp_rescore.REGISTRY base single-strategy entries; EXCLUDED by "
            "construction: leverage/freeze/combo variants, shorts "
            "(xs_momentum_ls, xs_momentum_ls_neutral), ml (not in REGISTRY), "
            "benchmark buy_hold_spy"),
        "exclusion_rule": (
            f"mandatory truncation_equivalence_test at cutoff {EQUIV_CUTOFF} "
            f"(build-on-full-then-slice vs build-on-truncated-panel must "
            f"match on common decision dates, atol {EQUIV_ATOL}, coverage >= "
            f"{EQUIV_MIN_COVERAGE:.0%}); failures are non-causal and excluded"),
        "exclusions": exclusions or [],
        "metric": "sharpe under next_open tc5 margin600 at 1x on all history to date",
        "scoring_note": (
            "ONE engine run per candidate per leg, weights sliced to <= "
            "Dec-31 of Y-1; the selection Sharpe is computed on the returned "
            "equity sliced to <= cutoff, which is identical to a truncated "
            "run because engine_v2 is day-causal"),
        "k": K,
        "blend": "equal",
        "leverages": list(LEVERAGES),
        "leverage_cap_process": PROC_LEV_CAP,
        "step": "annual, select at Dec-31 of Y-1, trade year Y",
        "first_leg": FIRST_LEG,
        "last_leg": LAST_LEG,
        "warmup": "2016-04..2018-12 (short - disclosed)",
        "dev_end": DEV_END,
        "dev_lock_note": (
            "scoring at Dec-31 2023+ uses data past DEV_END — inherent to "
            "walk-forward and allowed here ONLY because this spec is frozen "
            "before any leg runs; those scoring runs use window='full', "
            "final=True through exp_lib's front door (the lock is honored, "
            "not silently bypassed), with this spec's sha256 in notes"),
        "process_runs": (
            "one final=True window='full' engine run per leverage, family "
            "'wf_process', next_open tc5 margin600 cap2.0, daily blend frame; "
            "reported stats are computed on the equity sliced to >= "
            f"{FIRST_LEG}-01-01 (pre-2019 is warmup, held flat at zero weight)"),
        "sensitivity": (
            "k in {2,4} (metric sharpe) and metric=calmar (k=3), DEV LEGS "
            "ONLY (2019-2022), family 'wf_sens', published as a fragility "
            "table — NEVER used for the headline"),
    }
    spec["sha256"] = spec_sha256(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def load_spec(path: Path | str = SPEC_PATH) -> dict:
    """Load the frozen spec and verify its recorded sha256 (tamper check)."""
    spec = json.loads(Path(path).read_text())
    got = spec_sha256(spec)
    if got != spec.get("sha256"):
        raise ValueError(
            f"process_spec.json sha mismatch: recorded {spec.get('sha256')!r} "
            f"!= recomputed {got!r} — the spec was modified after freezing.")
    return spec


# ═══════════════════════════════════════════════════════════════════════════
# Pool builders — verbatim replicas of the exp_rescore.REGISTRY entries,
# parameterized by (px, macro, vol, cols) so the SAME code can build on the
# full panel and on a truncated panel (truncation-equivalence requirement).
# exp_rescore itself is deliberately NOT imported: its module-level memo
# monkeypatch would serve full-history cached frames to truncated builds.
# ═══════════════════════════════════════════════════════════════════════════

def _project_macro_sparse(w: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Verbatim exp_rescore.project_macro_sparse (sparse decision index)."""
    out = pd.DataFrame(0.0, index=w.index, columns=cols)
    for c in w.columns:
        if c in cols:
            out[c] = w[c].values
    return out


def pool_builders(px, macro, vol, cols) -> dict:
    """name -> zero-arg builder, closed over the given panel slices."""
    return {
        "xs_momentum_12_1": lambda: S.xs_momentum(px, macro, n_long=50),
        "xs_momentum_top30": lambda: S.xs_momentum(px, macro, n_long=30),
        "mean_reversion_5d": lambda: S.mean_reversion(px, macro, volume=vol, n_long=30),
        "trend_following": lambda: S.trend_following(px, macro, n_long=30),
        "dual_momentum_vol": lambda: S.dual_momentum_voltarget(px, macro, n_long=30),
        "sector_rotation": lambda: S.align_to_stock_cols(
            S.sector_rotation(px, macro), cols, []),
        "xs_momentum_multi": lambda: S.xs_momentum_multi(px, macro, n_long=30),
        "xs_momentum_concentrated": lambda: S.xs_momentum_concentrated(
            px, macro, volume=vol, n_long=15),
        "ts_momentum_multiasset": lambda: _project_macro_sparse(
            S.ts_momentum_multiasset(macro), cols),
        "xs_momentum_fast": lambda: S.xs_momentum_fast(px, macro, volume=vol, n_long=20),
        "xs_momentum_fast_top10": lambda: S.xs_momentum_fast(px, macro, volume=vol, n_long=10),
        "low_vol_quality": lambda: S.low_vol_quality(px, macro, n_long=30),
        "acceleration_momentum": lambda: S.acceleration_momentum(px, macro, n_long=30),
        "donchian_breakout": lambda: S.donchian_breakout(px, macro, n_long=30),
        "multifactor_mvr": lambda: S.multifactor_mvr(px, macro, n_long=30),
        "risk_parity_etf": lambda: S.align_to_stock_cols(
            S.risk_parity_etf(macro, target_vol=0.12), cols, []),
        "vix_gated_trend": lambda: S.vix_gated_trend(px, macro, n_long=30),
        "calendar_tom": lambda: S.calendar_tom(px, macro, n_long=30),
        "sector_momentum_rotation": lambda: S.align_to_stock_cols(
            S.sector_momentum_rotation(macro, n_sectors=3), cols, []),
        "adaptive_voltarget_momentum": lambda: S.adaptive_voltarget_momentum(
            px, macro, n_long=30),
        "momentum_quality_blend": lambda: S.momentum_quality_blend(px, macro, n_long=30),
    }


def build_pool_weights() -> dict[str, pd.DataFrame]:
    """Full-history 1x weight frames for the pool, via the SAME cache keys
    exp_catalog_v2 used (catalog_{name}) — under an unchanged _CODE_HASH
    these are pure cache hits and provably the frames catalog_v2 scored."""
    panel, macro, _ = load_cache()
    pu = union_prices_cached()
    builders = pool_builders(panel["close"], macro, panel["volume"], list(pu.columns))
    return {n: cached_weights(f"catalog_{n}", builders[n]) for n in POOL}


# ═══════════════════════════════════════════════════════════════════════════
# Mandatory causality audit
# ═══════════════════════════════════════════════════════════════════════════

def truncation_equivalence_test(
    sleeve_builder,
    cutoff,
    *,
    px: pd.DataFrame,
    macro: pd.DataFrame,
    vol: pd.DataFrame | None,
    cols: list[str],
    full_weights: pd.DataFrame | None = None,
    atol: float = EQUIV_ATOL,
    min_coverage: float = EQUIV_MIN_COVERAGE,
    trunc_cache_key: str | None = None,
) -> dict:
    """Causality audit for one sleeve builder.

    `sleeve_builder(px, macro, vol, cols)` -> sparse decision-date weight
    frame. Build on the FULL panel (or reuse `full_weights`), slice to
    <= cutoff; rebuild on the panel truncated at cutoff; the two frames must
    match on common decision dates/columns (atol) AND the common dates must
    cover >= min_coverage of the sliced-full dates (so a builder emitting a
    disjoint decision grid cannot vacuously pass). Any use of post-cutoff
    data (full-sample normalization, end-anchored grids, ...) fails here —
    a non-causal sleeve would leak the future into EVERY walk-forward leg.
    """
    cutoff = pd.Timestamp(cutoff)
    if full_weights is None:
        full_weights = sleeve_builder(px, macro, vol, cols)
    w_full = full_weights.loc[:cutoff]

    def _build_trunc():
        return sleeve_builder(
            px.loc[:cutoff], macro.loc[:cutoff],
            None if vol is None else vol.loc[:cutoff], cols)

    w_trunc = (cached_weights(trunc_cache_key, _build_trunc)
               if trunc_cache_key else _build_trunc())

    common_d = w_full.index.intersection(w_trunc.index)
    common_c = w_full.columns.intersection(w_trunc.columns)
    res = {
        "cutoff": str(cutoff.date()),
        "n_full_dates": int(len(w_full)),
        "n_trunc_dates": int(len(w_trunc)),
        "n_common_dates": int(len(common_d)),
    }
    if len(w_full) == 0 or len(common_d) == 0:
        res.update(ok=False, max_abs_diff=float("nan"), coverage=0.0,
                   reason="no common decision dates <= cutoff")
        return res
    a = w_full.loc[common_d, common_c].astype(float).fillna(0.0).to_numpy()
    b = w_trunc.loc[common_d, common_c].astype(float).fillna(0.0).to_numpy()
    max_diff = float(np.abs(a - b).max()) if a.size else 0.0
    coverage = len(common_d) / len(w_full)
    ok = (max_diff <= atol) and (coverage >= min_coverage)
    if ok:
        reason = ""
    elif max_diff > atol:
        reason = f"non-causal: max |w_full_sliced - w_trunc| = {max_diff:.3g} > atol {atol:g}"
    else:
        reason = f"decision-date coverage {coverage:.1%} < {min_coverage:.0%}"
    res.update(ok=bool(ok), max_abs_diff=max_diff,
               coverage=round(float(coverage), 4), reason=reason)
    return res


# ═══════════════════════════════════════════════════════════════════════════
# Selection
# ═══════════════════════════════════════════════════════════════════════════

def leg_cutoff(year: int) -> pd.Timestamp:
    """Selection date for leg `year`: Dec-31 of Y-1."""
    return pd.Timestamp(f"{year - 1}-12-31")


def score_window(year: int) -> tuple[str, bool]:
    """(window, final) for the scoring runs of leg `year`. Cutoffs <= DEV_END
    stay in the dev window; later cutoffs need window='full', final=True
    (spec dev_lock_note)."""
    if leg_cutoff(year) <= pd.Timestamp(DEV_END):
        return "dev", False
    return "full", True


def score_pool(
    year: int,
    weights_by_name: dict[str, pd.DataFrame],
    spec_sha: str,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score every pool candidate for leg `year` (ONE engine run each,
    weights sliced to <= Dec-31 Y-1), logging family 'wf_score'. Rows already
    present in `existing` (leg_year, name) are reused, not re-run."""
    cutoff = leg_cutoff(year)
    window, final = score_window(year)
    have = set()
    rows: list[dict] = []
    if existing is not None and len(existing):
        prior = existing[existing["leg_year"] == year]
        have = set(prior["name"])
        rows = prior.to_dict("records")
    for name in weights_by_name:
        if name in have:
            continue
        res = run_trial(
            weights_by_name[name].loc[:cutoff],
            name=name, family="wf_score",
            params={"leg_year": year, "score_cutoff": str(cutoff.date()),
                    "spec_sha": spec_sha[:16], "k": K, "metric": "sharpe"},
            window=window, exec_model="next_open", tc_bps=TC_BPS,
            leverage_cap=SCORE_LEV_CAP, margin_bps_annual=MARGIN_BPS,
            final=final,
            notes=(f"wf_score leg={year} cutoff={cutoff.date()} "
                   f"spec_sha={spec_sha}; selection metric computed on "
                   f"equity<=cutoff (engine day-causal; this row's stats "
                   f"cover the {window} window)"),
        )
        eq = res["equity"].loc[:cutoff]
        s = eq_summary(eq, name=name)
        rows.append({
            "leg_year": year, "name": name,
            "score_cutoff": str(cutoff.date()),
            "sharpe_at_selection": s["sharpe"],
            "calmar_at_selection": s["calmar"],
            "geo_at_selection": s["geo_monthly"],
            "maxdd_at_selection": s["max_drawdown"],
        })
    df = pd.DataFrame(rows).sort_values(
        ["sharpe_at_selection"], ascending=False).reset_index(drop=True)
    return df


def select_topk(scores: pd.DataFrame, k: int = K,
                metric_col: str = "sharpe_at_selection") -> list[str]:
    """Top-k names by the metric, descending; deterministic name tie-break."""
    s = scores.sort_values([metric_col, "name"], ascending=[False, True])
    return list(s["name"].head(k))


# ═══════════════════════════════════════════════════════════════════════════
# Process-curve assembly
# ═══════════════════════════════════════════════════════════════════════════

def expand_daily_aligned(w_sparse: pd.DataFrame, index: pd.Index,
                         cols: list[str]) -> pd.DataFrame:
    """Sparse decision frame -> daily frame on `index`, aligned to `cols`
    (ffill decisions; zero before the first decision / for absent names)."""
    d = w_sparse.reindex(index).ffill().fillna(0.0)
    return d.reindex(columns=cols).fillna(0.0)


def assemble_process_frame(
    selections: dict[int, list[str]],
    daily_by_name: dict[str, pd.DataFrame],
    index: pd.Index,
    first_leg: int,
    last_leg: int,
) -> pd.DataFrame:
    """Concatenate per-leg equal blends into ONE daily decision frame at 1x.

    Leg Y owns exactly the trading days of calendar year Y; the concatenated
    index must equal the calendar slice [first_leg, last_leg] with no
    overlaps and no gaps (hard-checked). Causality: the daily rows inside
    year Y are ffilled from decisions each selected sleeve made on data
    available before those days, and the selection itself used only data
    <= Dec-31 Y-1. The engine applies its own 1-day execution shift."""
    years = index.year
    parts = []
    for y in range(first_leg, last_leg + 1):
        days = index[years == y]
        if len(days) == 0:
            continue  # partial final leg: calendar simply ends
        sel = selections[y]
        if not sel:
            raise ValueError(f"leg {y}: empty selection")
        blk = daily_by_name[sel[0]].loc[days].copy()
        for n in sel[1:]:
            blk = blk + daily_by_name[n].loc[days]
        parts.append(blk / float(len(sel)))
    out = pd.concat(parts)
    target = index[(years >= first_leg) & (years <= last_leg)]
    if out.index.has_duplicates:
        raise ValueError("leg concatenation produced overlapping dates")
    if not out.index.equals(target):
        raise ValueError(
            f"leg concatenation gap: got {len(out)} rows, calendar slice has "
            f"{len(target)}")
    return out


def run_process_curve(frame_1x: pd.DataFrame, leverage: float, spec_sha: str,
                      *, family: str = "wf_process", window: str = "full",
                      final: bool = True, tag: str = "",
                      k: int | None = None,
                      first_leg: int | None = None,
                      last_leg: int | None = None) -> dict:
    """ONE engine run of the concatenated process frame at `leverage`.

    k/first_leg/last_leg default to the headline spec constants but MUST be
    overridden by sensitivity variants so their ledger params describe the
    variant actually run (k=2/4, dev-leg range), not the headline config.
    Re-runs of an already-ledgered (name, window) re-execute the engine for
    the caller but skip the ledger append — identical re-renders are not new
    hypotheses and would only inflate raw row counts.
    """
    name = tag or f"wf_process_{leverage:g}x"
    ledger = TRIALS_DIR / f"{family}.csv"
    already = False
    if ledger.exists():
        led = pd.read_csv(ledger)
        already = bool(((led["name"] == name) & (led["window"] == window)).any())
    return run_trial(
        frame_1x * leverage,
        name=name, family=family, log=not already,
        params={"leverage": leverage,
                "k": k if k is not None else K, "blend": "equal",
                "first_leg": first_leg if first_leg is not None else FIRST_LEG,
                "last_leg": last_leg if last_leg is not None else LAST_LEG,
                "spec_sha": spec_sha[:16]},
        window=window, exec_model="next_open", tc_bps=TC_BPS,
        leverage_cap=PROC_LEV_CAP, margin_bps_annual=MARGIN_BPS,
        weights_are_daily=True, final=final,
        notes=(f"wf process curve lev={leverage:g} spec_sha={spec_sha}; "
               f"reported stats computed on equity >= {FIRST_LEG}-01-01 "
               f"(pre-{FIRST_LEG} is flat warmup at zero weight; this row's "
               f"summary covers the {window} window incl. that flat span)"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def sliced_stats(equity: pd.Series, start: str, name: str) -> tuple[pd.Series, dict]:
    """Slice equity to >= start, renormalize to 100k, summarize."""
    eq = equity.loc[pd.Timestamp(start):].dropna()
    eq = eq / eq.iloc[0] * 100_000.0
    return eq, eq_summary(eq, name=name)


def monthly_rets(equity: pd.Series) -> pd.Series:
    return equity.resample("ME").last().pct_change().dropna()


def champion_reference() -> tuple[pd.Series, dict]:
    """Reproduce the catalog_v2 hindsight champion (combo_v2_2x, next_open,
    tc5, cap 2.0, margin 600) with a DIRECT engine call. Not ledgered: it
    byte-reproduces the already-ledgered catalog_v2 combo_v2_2x full-window
    trial (the runner cross-checks against that row) — logging it again
    would only inflate the deduped trial count with a non-hypothesis."""
    panel, macro, _ = load_cache()
    pu = union_prices_cached()
    w = cached_weights(
        "catalog_combo_v2_2x",
        lambda: pd.read_parquet(ROOT / "weights_store" / "combo_v2_base_1x.parquet") * 2.0)
    cfg = BTConfigV2(tc_bps=TC_BPS, leverage_cap=PROC_LEV_CAP,
                     exec_model="next_open", margin_bps_annual=MARGIN_BPS)
    bt = run_backtest_v2(w, pu, cfg, name="ref_combo_v2_2x",
                         open_prices=panel["open"])
    return bt["equity"], bt["summary"]


def spy_reference() -> pd.Series:
    """SPY buy-and-hold equity proxy from the adjusted close (no costs) —
    regime context only, disclosed as such in the report."""
    pu = union_prices_cached()
    spy = pu["SPY"].dropna()
    return spy / spy.iloc[0] * 100_000.0


def process_dsr(sharpe_annual: float, n_days: int) -> float:
    """Render-time DSR at the program-wide deduped trial count, with the
    measured cross-trial SR variance (exp_catalog_v2 helper, reused)."""
    return deflated_sharpe(sharpe_annual, n_days, trial_count(dedupe=True),
                           sr_variance_across_trials=ledger_sr_variance_daily())
