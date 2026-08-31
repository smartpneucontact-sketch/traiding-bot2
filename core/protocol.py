"""Pre-registered forward-test protocol — loader, integrity hash, scorer.

The paper test is only credible evidence if the success/kill criteria were
committed BEFORE the first funded rebalance and provably never edited
afterwards. `protocol.json` (repo root) is the machine-readable contract;
its sha256 is recorded into the model's state at the first funded
rebalance (core/runner.py). Any later mismatch means the protocol was
modified mid-test and the run is permanently flagged MODIFIED_AFTER_START
by `score_protocol` (Phase A2).

Scoring conventions (see score_protocol):
  - Criterion polarity follows the action. Kill/pause/fail/inconclusive
    actions state the FIRING condition (op true → fires); review/success
    actions state the PASS requirement (op false → fails).
  - Status vocabulary is PASS | WARN | KILL | NA. NA means the input
    doesn't exist yet (pre-start, checkpoint not reached, or the metric
    is unmeasured) — never silently PASS.
  - "Reconciliation green" (M3-RECON) reuses the two already-registered
    K3 thresholds: fill rate >= 90% by notional AND, when the book block
    exists, |achieved-vs-target gross - 100%| <= 15%.
  - The INCONCLUSIVE band (M12-INCONCLUSIVE) is half-open [0, 1.8): its
    upper edge IS the registered SUCCESS floor, and protocol.json
    declares it exclusive ("geo >= 1.8 belongs to M12-GEO"), so a geo of
    exactly 1.8%/mo scores SUCCESS, never INCONCLUSIVE.
  - M6-BAND's registered threshold (-0.2754) is the drawdown band
    evaluated at exactly t=6 months. The scorer re-evaluates the band at
    the ACTUAL elapsed t — expected(t) − 1.28·σ·√t — so the criterion
    stays meaningful past month 6 (the 6mo review reads it at t=6, where
    the two coincide). The registered threshold is still reported; the
    value actually compared is `effective_threshold`.

Shadow slots (Improvement Roadmap v2, Streams 1+3): each model slot binds
its OWN protocol file — combo_v2 keeps the frozen protocol.json (its
recorded sha semantics are untouched); combo_v2_exp binds
protocol_exp.json (Candidate X relative shadow protocol); combo_v2_process
/ combo_v2_proc bind protocol_process.json. Resolution is
`protocol_path_for(model_name)` / `load_protocol_for(model_name)`.
Shadow protocols add two schema extensions the primary never uses:
  - Relative criteria: metric names starting with "rel_" are computed
    versus a baseline slot (criterion field `relative_to`, default
    "combo_v2") on the COMMON WINDOW — max of the two slots'
    forward_test_start dates — from both slots' tracking. The values are
    supplied to score_protocol via the `rel_metrics` keyword (the caller
    computes them with core/tracking.py; see compute_relative_metrics).
  - action "REPORT_ONLY": scored INFO, never PASS/WARN/KILL, never
    enters the overall verdict, and its threshold may be null.

Deliberately dependency-free (json + hashlib + math only): this module
must be importable from the live order path, the dashboard, and offline
tooling without dragging in pandas/yfinance.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

_BASE_DIR = Path(__file__).resolve().parent.parent
PROTOCOL_JSON_PATH = _BASE_DIR / "protocol.json"
PROTOCOL_MD_PATH = _BASE_DIR / "PROTOCOL.md"
PROTOCOL_EXP_JSON_PATH = _BASE_DIR / "protocol_exp.json"
PROTOCOL_PROCESS_JSON_PATH = _BASE_DIR / "protocol_process.json"

# The slot every rel_* criterion defaults to comparing against.
PRIMARY_MODEL = "combo_v2"

# Per-slot protocol files. Exact model-name match first; versioned process
# bundles (combo_v2_process.<year>.<sha8>) resolve by prefix below. Any
# unknown name falls back to the primary protocol.json — the pre-shadow
# behavior, so existing callers are unchanged.
SLOT_PROTOCOL_FILES: dict[str, Path] = {
    "combo_v2": PROTOCOL_JSON_PATH,            # FROZEN primary — never edit
    "combo_v2_exp": PROTOCOL_EXP_JSON_PATH,
    "combo_v2_process": PROTOCOL_PROCESS_JSON_PATH,
    "combo_v2_proc": PROTOCOL_PROCESS_JSON_PATH,  # registry short alias
}


def protocol_path_for(model_name: str | None) -> Path:
    """The protocol file governing a model slot.

    Exact registry-name match first, then longest-prefix match so
    versioned process bundle names (combo_v2_process.2026.ab12cd34) and
    exp variants (combo_v2_exp_v2) resolve to their family's protocol.
    Unknown names get the primary protocol.json (back-compat: that is
    what every caller received before shadow slots existed).
    """
    name = model_name or ""
    if name in SLOT_PROTOCOL_FILES:
        return SLOT_PROTOCOL_FILES[name]
    for prefix in sorted(SLOT_PROTOCOL_FILES, key=len, reverse=True):
        if prefix != PRIMARY_MODEL and name.startswith(prefix):
            return SLOT_PROTOCOL_FILES[prefix]
    return PROTOCOL_JSON_PATH


def load_protocol_for(model_name: str | None) -> dict:
    """load_protocol() against the slot's own protocol file."""
    return load_protocol(protocol_path_for(model_name))


def bind_forward_test_start(state: dict, model_name: str,
                            today_iso: str) -> bool:
    """Set-once forward-test clock + PER-SLOT protocol sha binding.

    Generalizes what core/runner.py does inline for the primary (bind
    protocol.json's sha at the first funded rebalance): each slot binds
    the sha256 of ITS OWN protocol file. For combo_v2 this hashes exactly
    PROTOCOL_JSON_PATH — byte-identical to the primary's existing
    recorded-sha semantics. Returns True when the binding was written,
    False when the clock was already running (never overwrites).
    """
    if state.get("forward_test_start"):
        return False
    # Sha FIRST: if hashing throws (missing/unreadable protocol file),
    # neither key is set and the binding retries next run — setting the
    # clock first left fts bound with no sha, the exact half-bound state
    # the invariant sweep flags (2026-08-31 audit).
    sha = file_sha256(protocol_path_for(model_name))
    state["forward_test_start"] = today_iso
    state["protocol_sha256"] = sha
    return True

# Every criterion must carry exactly these keys (extras like
# `and_conditions` are allowed — the 6mo regime AND-condition needs one).
REQUIRED_CRITERION_FIELDS = (
    "id", "description", "metric", "op", "threshold", "checkpoint", "action",
)
ALLOWED_OPS = ("<", "<=", ">", ">=", "==", "between")
ALLOWED_CHECKPOINTS = ("continuous", "3mo", "6mo", "12mo")


def file_sha256(path: str | Path) -> str:
    """Hex sha256 of a file's bytes. Chunked so it stays O(1) memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_criterion(crit: dict) -> None:
    missing = [f for f in REQUIRED_CRITERION_FIELDS if f not in crit]
    if missing:
        raise ValueError(
            f"protocol criterion {crit.get('id', '<no id>')!r} missing "
            f"required fields: {missing}"
        )
    if crit["op"] not in ALLOWED_OPS:
        raise ValueError(
            f"protocol criterion {crit['id']!r}: unknown op {crit['op']!r} "
            f"(allowed: {ALLOWED_OPS})"
        )
    if crit["checkpoint"] not in ALLOWED_CHECKPOINTS:
        raise ValueError(
            f"protocol criterion {crit['id']!r}: unknown checkpoint "
            f"{crit['checkpoint']!r} (allowed: {ALLOWED_CHECKPOINTS})"
        )
    thr = crit["threshold"]
    if thr is None and crit["action"] == "REPORT_ONLY":
        # Report-only entries display a value and never compare it, so a
        # null threshold is legal (and protocol_process.json's
        # P-vs-CANDIDATE-X deliberately has none).
        return
    if crit["op"] == "between":
        if (not isinstance(thr, (list, tuple)) or len(thr) != 2
                or not all(isinstance(x, (int, float)) for x in thr)
                or thr[0] > thr[1]):
            raise ValueError(
                f"protocol criterion {crit['id']!r}: 'between' needs a "
                f"[lo, hi] numeric pair, got {thr!r}"
            )
    elif not isinstance(thr, (int, float)):
        raise ValueError(
            f"protocol criterion {crit['id']!r}: threshold must be numeric "
            f"for op {crit['op']!r}, got {thr!r}"
        )


def load_protocol(path: str | Path | None = None) -> dict:
    """Load + validate protocol.json.

    Returns the parsed dict with a computed `_sha256` key added (leading
    underscore = derived, not part of the committed file) so callers that
    score or display the protocol always carry the hash of the exact bytes
    they read. Raises ValueError on any schema violation — a malformed
    protocol must fail loudly, never score silently.
    """
    p = Path(path) if path is not None else PROTOCOL_JSON_PATH
    with open(p, encoding="utf-8") as fh:
        proto = json.load(fh)

    criteria = proto.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError(f"{p}: 'criteria' must be a non-empty list")
    seen_ids: set[str] = set()
    for crit in criteria:
        if not isinstance(crit, dict):
            raise ValueError(f"{p}: criterion is not an object: {crit!r}")
        _validate_criterion(crit)
        if crit["id"] in seen_ids:
            raise ValueError(f"{p}: duplicate criterion id {crit['id']!r}")
        seen_ids.add(crit["id"])

    proto["_sha256"] = file_sha256(p)
    return proto


# ═══════════════════════════════════════════════════════════════════════════
# Protocol scorer (Phase A2)
# ═══════════════════════════════════════════════════════════════════════════

# Trading-day months per checkpoint (a "month" = 21 trading days, matching
# core/tracking.py). A checkpoint's criteria stay NA until it is reached —
# their live values are still reported so the scoreboard shows trajectory.
CHECKPOINT_MONTHS = {"continuous": 0.0, "3mo": 3.0, "6mo": 6.0, "12mo": 12.0}

# Actions whose criterion states the FIRING condition (op true → the named
# outcome fires). The remaining actions (OPERATIONAL_REVIEW_IF_FAIL,
# REQUIRED_FOR_SUCCESS, KILL_IF_AND_CONDITIONS_ALSO_FAIL) state the PASS
# requirement — op false → the criterion fails.
_FIRES_WHEN_TRUE = {
    "KILL", "KILL_OR_RECOST", "OPERATIONAL_PAUSE", "FAIL",
    "INCONCLUSIVE_EXTEND_6MO",
}

# Status when a criterion fires, by action. REQUIRED_FOR_SUCCESS failing is
# WARN (it denies the 12mo SUCCESS verdict; it does not kill the test).
_FIRED_STATUS = {
    "KILL": "KILL",
    "KILL_OR_RECOST": "KILL",
    "FAIL": "KILL",
    "OPERATIONAL_PAUSE": "WARN",
    "OPERATIONAL_REVIEW_IF_FAIL": "WARN",
    "INCONCLUSIVE_EXTEND_6MO": "WARN",
    "REQUIRED_FOR_SUCCESS": "WARN",
    "KILL_IF_AND_CONDITIONS_ALSO_FAIL": "WARN",  # upgraded when ANDs fail too
}


def _op_holds(value: float, op: str, threshold, *,
              half_open: bool = False) -> bool:
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return value == threshold
    if op == "between":
        lo, hi = threshold
        # half_open ([lo, hi)) is used for the INCONCLUSIVE band, whose
        # upper edge is the registered SUCCESS floor (protocol.json:
        # "Upper bound 1.8 is exclusive — >= 1.8 belongs to M12-GEO").
        # A geo of exactly 1.8%/mo must score SUCCESS, not INCONCLUSIVE.
        return lo <= value < hi if half_open else lo <= value <= hi
    raise ValueError(f"unknown op {op!r}")  # load_protocol already rejects


def _band_threshold_at(reference: dict, t_months: float):
    """M6-BAND lower band evaluated at the ACTUAL elapsed time t (months):

        band(t) = t·ln(1 + geo_ref) − 1.28·σ_ref·√t

    with geo_ref / σ_ref from protocol.json's `reference` block (3.68%/mo,
    15.7%/mo). At t=6 this reproduces the registered fixed threshold
    (−0.2754) exactly, so the 6mo checkpoint review reads the committed
    number; past month 6 the band keeps scaling with elapsed time instead
    of holding a longer window to a 6-month band. Returns None when the
    reference inputs are missing (caller falls back to the registered
    fixed threshold)."""
    geo = (reference or {}).get("geo_monthly_pct")
    sigma = (reference or {}).get("sigma_monthly_pct")
    if geo is None or sigma is None or t_months <= 0:
        return None
    return (t_months * math.log(1.0 + float(geo) / 100.0)
            - 1.28 * (float(sigma) / 100.0) * math.sqrt(t_months))


# ── Relative (two-slot) metrics — pure math, computed on the COMMON
# window (max of the two slots' forward_test_start dates). The caller
# produces both tracking dicts with core/tracking.py's compute_tracking
# using that shared start date and hands the results here.

def common_window_start(start_a: Optional[str],
                        start_b: Optional[str]) -> Optional[str]:
    """Common forward-test window start: the LATER of two ISO dates.

    Relative criteria are only meaningful over days both slots traded, so
    the shared window begins when the second slot's clock started. None
    when either clock hasn't started (no common window exists yet).
    """
    if not start_a or not start_b:
        return None
    return max(start_a[:10], start_b[:10])


def live_calmar_from_tracking(tracking: Optional[dict]) -> Optional[float]:
    """Live Calmar = annualized geometric return / |MaxDD| from a
    compute_tracking() dict: ((1 + geo_m)^12 - 1) / |max_drawdown|.

    None when tracking isn't "ok", inputs are missing, or MaxDD is 0 (a
    zero-drawdown window makes Calmar unbounded — report unmeasured, and
    let the criterion stay NA rather than fabricate a huge number).
    """
    tr = tracking or {}
    if tr.get("status") != "ok":
        return None
    geo_pct = tr.get("live_geo_monthly_pct")
    dd_pct = tr.get("max_drawdown_pct")
    if geo_pct is None or dd_pct is None or dd_pct == 0:
        return None
    ann_return = (1.0 + float(geo_pct) / 100.0) ** 12 - 1.0
    return ann_return / (abs(float(dd_pct)) / 100.0)


def compute_relative_metrics(own_tracking: Optional[dict],
                             baseline_tracking: Optional[dict]) -> dict:
    """rel_* metric values for one slot versus a baseline slot.

    Both dicts must be compute_tracking() outputs over the SAME common
    window (see common_window_start). Every metric is own − baseline, so
    the registered ops read naturally (e.g. rel_max_drawdown_diff_pp >=
    -3 ⇔ own drawdown at most 3pp deeper). Any missing input → that
    metric is None (NA downstream), never a guess.

      rel_max_drawdown_diff_pp: own MaxDD% − baseline MaxDD%
      rel_geo_monthly_diff_pp:  own geo%/mo − baseline geo%/mo
      rel_live_calmar_diff:     own live Calmar − baseline live Calmar
    """
    own = own_tracking if (own_tracking or {}).get("status") == "ok" else {}
    base = (baseline_tracking
            if (baseline_tracking or {}).get("status") == "ok" else {})

    def diff(a, b, ndigits=4):
        if a is None or b is None:
            return None
        return round(float(a) - float(b), ndigits)

    own_calmar = live_calmar_from_tracking(own or None)
    base_calmar = live_calmar_from_tracking(base or None)
    return {
        "rel_max_drawdown_diff_pp": diff(
            own.get("max_drawdown_pct"), base.get("max_drawdown_pct")),
        "rel_geo_monthly_diff_pp": diff(
            own.get("live_geo_monthly_pct"), base.get("live_geo_monthly_pct")),
        "rel_live_calmar_diff": diff(own_calmar, base_calmar),
        # Components for display / audit — not criteria inputs.
        "own_live_calmar": (round(own_calmar, 4)
                            if own_calmar is not None else None),
        "baseline_live_calmar": (round(base_calmar, 4)
                                 if base_calmar is not None else None),
        "own_months_elapsed": own.get("months_elapsed"),
        "baseline_months_elapsed": base.get("months_elapsed"),
    }


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _recon_fill_rate(recon: dict) -> Optional[float]:
    flow = (recon or {}).get("flow") or {}
    fr = flow.get("fill_rate_notional")
    return float(fr) if fr is not None else None


def _recon_gross_dev_pct(recon: dict) -> Optional[float]:
    book = (recon or {}).get("book") or {}
    pct = book.get("gross_achieved_pct_of_target")
    return abs(100.0 - float(pct)) if pct is not None else None


def _recon_is_green(recon: dict) -> Optional[bool]:
    """Green = fill rate >= 90% AND gross deviation <= 15% (when the book
    block exists). None when the fill rate itself is unmeasured."""
    fr = _recon_fill_rate(recon)
    if fr is None:
        return None
    dev = _recon_gross_dev_pct(recon)
    return fr >= 0.90 and (dev is None or dev <= 15.0)


def _metric_values(tracking: dict, slippage_stats: dict,
                   reconciliation_history: list[dict]) -> dict:
    """Map every protocol metric name to its current value (None = NA).

    Consecutive-rebalance metrics collapse the window so the registered op
    still reads naturally: K2 reports the MIN of the last 3 rebalances'
    slippage (min > 25 ⇔ all three > 25); K3a the MAX of the last 2 fill
    rates (max < 90 ⇔ both < 90); K3b the MIN of the last 2 gross
    deviations (min > 15 ⇔ both > 15). Any unmeasured entry in the window
    → None, never a guess.
    """
    tr = tracking if (tracking or {}).get("status") == "ok" else {}
    slip = slippage_stats or {}
    recons = reconciliation_history or []

    vals: dict = {
        # Drawdown from the forward-test equity peak. The kill uses the
        # WORST observed drawdown — once −45% is ever touched, K1 latches.
        "live_drawdown_pct": tr.get("max_drawdown_pct"),
        "geo_monthly_pct": tr.get("live_geo_monthly_pct"),
        "realized_vol_monthly_pct": tr.get("realized_vol_monthly_pct"),
        # Whole-window cumulative values, scored once the 6mo checkpoint
        # is reached. M6-BAND's threshold is re-evaluated at the actual
        # elapsed t (_band_threshold_at), so comparing the whole window
        # against it stays meaningful past month 6.
        "cum_log_return_6mo": tr.get("cum_log_return"),
        "cum_return_6mo_pct": tr.get("cum_return_pct"),
        "live_minus_2x_spy_cum_return_pp": (
            (tr.get("spy") or {}).get("live_minus_2x_spy_pp")),
        "max_drawdown_pct": tr.get("max_drawdown_pct"),
        "slippage_bps_per_side": slip.get("calibrated_cost_bps_per_side"),
    }

    run_means = [r.get("notional_weighted_mean_bps")
                 for r in (slip.get("by_run") or [])]
    measured = [m for m in run_means if m is not None]
    last3 = run_means[-3:]
    vals["slippage_bps_per_side_3_consecutive_rebalances"] = (
        min(last3) if len(last3) == 3 and all(m is not None for m in last3)
        else None)
    # Median per-side slippage across measured rebalances, >= 3 required.
    vals["slippage_bps_per_side_median"] = (
        round(_median(measured), 2) if len(measured) >= 3 else None)

    fills = [_recon_fill_rate(r) for r in recons[-2:]]
    vals["fill_rate_pct_2_consecutive_rebalances"] = (
        round(max(fills) * 100.0, 2)
        if len(fills) == 2 and all(f is not None for f in fills) else None)
    devs = [_recon_gross_dev_pct(r) for r in recons[-2:]]
    vals["gross_deviation_pct_2_consecutive_rebalances"] = (
        round(min(devs), 2)
        if len(devs) == 2 and all(d is not None for d in devs) else None)
    greens = [_recon_is_green(r) for r in recons[-3:]]
    vals["reconciliation_green_count_of_last_3"] = (
        sum(1 for g in greens if g)
        if len(greens) == 3 and all(g is not None for g in greens) else None)
    return vals


def score_protocol(tracking: Optional[dict], slippage_stats: Optional[dict],
                   reconciliation_history: Optional[list[dict]],
                   state: Optional[dict], proto: dict, *,
                   rel_metrics: Optional[dict] = None) -> dict:
    """Score live metrics against the pre-registered protocol.

    Args:
      tracking: core.tracking.compute_tracking output (or None/pre-start).
      slippage_stats: core.execution_stats.compute_slippage_stats output.
      reconciliation_history: chronological reconciliation dicts (the
        `result.reconciliation` blocks from state.history).
      state: on-disk pipeline state (forward_test_start, protocol_sha256).
      proto: load_protocol() output (must carry `_sha256`).
      rel_metrics: OPTIONAL relative-metric values for shadow protocols,
        keyed by baseline model name:
          {"combo_v2": {"rel_max_drawdown_diff_pp": ..., ...},
           "_common_window_start": {"combo_v2": "YYYY-MM-DD", ...}}
        (compute_relative_metrics output per baseline, both slots' tracking
        computed over the common window). Criteria whose metric starts
        with "rel_" read from here via their `relative_to` field (default
        PRIMARY_MODEL); a missing entry scores NA, never a guess. The
        primary protocol has no rel_ criteria, so passing None keeps its
        scoring byte-identical to the pre-shadow behavior.

    Returns {overall, binding, forward_test_start, months_elapsed,
    checkpoint, criteria: [{id, metric, value, op, threshold, checkpoint,
    action, status, ...}]}. Before forward_test_start everything is NA and
    overall is NOT_STARTED — only the binding integrity is scored.
    REPORT_ONLY criteria score INFO (value displayed) and never enter the
    overall verdict.
    """
    state = state or {}
    fts = state.get("forward_test_start")
    bound_sha = state.get("protocol_sha256")
    if bound_sha is None:
        binding = "not_bound_yet"
    elif bound_sha == proto.get("_sha256"):
        binding = "intact"
    else:
        binding = "MODIFIED_AFTER_START"

    started = bool(fts)
    months = 0.0
    if started and (tracking or {}).get("status") == "ok":
        months = float(tracking.get("months_elapsed") or 0.0)

    vals = _metric_values(tracking or {}, slippage_stats or {},
                          reconciliation_history or [])

    criteria_out: list[dict] = []
    for crit in proto["criteria"]:
        is_relative = str(crit["metric"]).startswith("rel_")
        if is_relative:
            baseline = crit.get("relative_to") or PRIMARY_MODEL
            value = ((rel_metrics or {}).get(baseline) or {}).get(
                crit["metric"])
        else:
            value = vals.get(crit["metric"])
        cp_needed = CHECKPOINT_MONTHS[crit["checkpoint"]]
        out = {
            "id": crit["id"],
            "metric": crit["metric"],
            "description": crit["description"],
            "op": crit["op"],
            "threshold": crit["threshold"],
            "checkpoint": crit["checkpoint"],
            "action": crit["action"],
            "value": value,
            "status": "NA",
        }
        if is_relative:
            out["relative_to"] = baseline
            cw = ((rel_metrics or {}).get("_common_window_start") or {})
            if cw.get(baseline):
                out["common_window_start"] = cw[baseline]
        if not started:
            out["na_reason"] = "forward test not started"
        elif months < cp_needed:
            out["na_reason"] = (
                f"checkpoint {crit['checkpoint']} not reached "
                f"({months:.2f} months elapsed)")
        elif value is None:
            out["na_reason"] = (
                "relative metric not measurable yet (baseline slot "
                "tracking missing or no common window)"
                if is_relative else "metric not measurable yet")
        elif crit["action"] == "REPORT_ONLY":
            # Displayed, never judged: INFO regardless of any threshold,
            # excluded from the overall verdict below.
            out["status"] = "INFO"
        else:
            threshold = crit["threshold"]
            if crit["id"] == "M6-BAND":
                # Evaluate the band at the actual elapsed t so the
                # criterion stays meaningful past the 6mo checkpoint (at
                # exactly t=6 this equals the registered −0.2754).
                band = _band_threshold_at(proto.get("reference"), months)
                if band is not None:
                    threshold = band
                    out["effective_threshold"] = round(band, 4)
                    out["evaluated_at_months"] = round(months, 4)
            holds = _op_holds(
                value, crit["op"], threshold,
                # INCONCLUSIVE band is half-open [lo, hi): its upper edge
                # is the SUCCESS floor (exactly 1.8 → SUCCESS).
                half_open=(crit["action"] == "INCONCLUSIVE_EXTEND_6MO"))
            fired = holds if crit["action"] in _FIRES_WHEN_TRUE else not holds
            if not fired:
                out["status"] = "PASS"
            else:
                status = _FIRED_STATUS.get(crit["action"], "WARN")
                if crit["action"] == "KILL_IF_AND_CONDITIONS_ALSO_FAIL":
                    ands = crit.get("and_conditions") or []
                    and_results = []
                    for cond in ands:
                        v = vals.get(cond["metric"])
                        and_results.append(
                            None if v is None
                            else _op_holds(v, cond["op"], cond["threshold"]))
                    out["and_conditions"] = [
                        {**cond, "value": vals.get(cond["metric"]),
                         "holds": res}
                        for cond, res in zip(ands, and_results)
                    ]
                    # KILL only when every AND-condition demonstrably holds
                    # (an unmeasured condition can never confirm a kill).
                    if and_results and all(r is True for r in and_results):
                        status = "KILL"
                out["status"] = status
        criteria_out.append(out)

    # REPORT_ONLY rows are context, never verdict inputs.
    decisive = [c for c in criteria_out if c["action"] != "REPORT_ONLY"]
    if not started:
        overall = "NOT_STARTED"
    elif any(c["status"] == "KILL" for c in decisive):
        overall = "KILL"
    elif any(c["status"] == "WARN" for c in decisive):
        overall = "WARN"
    elif all(c["status"] == "NA" for c in decisive):
        overall = "NO_DATA"
    else:
        overall = "PASS"

    if not started:
        checkpoint = "pre-start"
    else:
        checkpoint = "continuous"
        for label in ("3mo", "6mo", "12mo"):
            if months >= CHECKPOINT_MONTHS[label]:
                checkpoint = label

    # 12mo-leg bookkeeping for the shadow decision rules (protocol_exp:
    # FAIL when >=2 REL legs fail; SUCCESS = all REQUIRED legs pass). The
    # counts are reported — the written decision_rule stays the authority.
    req = [c for c in decisive if c["action"] == "REQUIRED_FOR_SUCCESS"]
    rel_req = [c for c in req if str(c["metric"]).startswith("rel_")]
    required_summary = {
        "total": len(req),
        "passed": sum(1 for c in req if c["status"] == "PASS"),
        "failed": sum(1 for c in req if c["status"] == "WARN"),
        "na": sum(1 for c in req if c["status"] == "NA"),
        "rel_legs_total": len(rel_req),
        "rel_legs_failed": sum(1 for c in rel_req if c["status"] == "WARN"),
    }

    return {
        "overall": overall,
        "binding": binding,
        "required_for_success": required_summary,
        "forward_test_start": fts,
        "protocol_sha256_bound": bound_sha,
        "protocol_sha256_current": proto.get("_sha256"),
        "protocol_version": proto.get("protocol_version"),
        "amended": proto.get("amended"),
        "months_elapsed": round(months, 4),
        "checkpoint": checkpoint,
        "criteria": criteria_out,
    }
