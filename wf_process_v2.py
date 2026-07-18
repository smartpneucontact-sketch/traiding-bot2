"""Stream 3 (Model Improvement Roadmap v2) — walk-forward process spec v2.

The v1 spec (results/v7/wf/process_spec.json, sha
b75aa7e7fe34eec11ebcd5b528c5a5f4d75f131fda1bd5e7cb54c773688503a0) is the
WS3 *evaluation* pre-registration and stays byte-immutable. This module
writes/loads the *live-process* spec v2 (results/v7/wf/process_spec_v2.json)
which SUPERSEDES v1 for the shadow slot #3 process:

    cadence  : annual re-selection at Dec-31 of Y-1 (WS3 evidence — the
               annual selections were stable 8/8 legs; quarterly cadence has
               zero evidence and would be fresh selection pressure).
    pool     : the SAME 21 base catalog names, copied verbatim from the v1
               spec (combos would change the measured object).
    rule     : k=3 equal blend by Sharpe (v1 metric string unchanged),
               leverage 2.0 for comparability with the primary.
    bootstrap: the already-ledgered 2026 leg — dual_momentum_vol +
               xs_momentum_top30 + xs_momentum_12_1; first LIVE re-selection
               2026-12-31.
    amendment: pool additions only via dated append-only amendment (with
               orchestrator approval) + a fresh truncation-equivalence
               audit, effective at the NEXT selection date.

Spec discipline is inherited from wf_process (write-once file; sha256 of the
canonical json sealed inside; load verifies). Machinery is REUSED via import
— spec_sha256/load_spec, leg_cutoff/score_window, select_topk,
build_pool_weights, truncation_equivalence_test all come from wf_process;
nothing is re-implemented here.

Ledger family for v2 scoring runs: 'wf_score_v2' (v1's 'wf_score' rows may
be REUSED where they already exist — re-running an identical, already-
ledgered computation is not a new hypothesis; the reuse is recorded in the
selection file's provenance block).

CLI runner for selections: wf_select.py. Tests: validation/test_wf_v2.py.

Running this module as a script writes the v2 spec once (or verifies the
already-frozen one) and prints its sha.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import wf_process as W
from exp_lib import run_trial
from metrics import summary as eq_summary

# REUSED wf_process machinery (do not duplicate):
from wf_process import (  # noqa: F401
    K,
    MARGIN_BPS,
    SCORE_LEV_CAP,
    TC_BPS,
    build_pool_weights,
    leg_cutoff,
    load_spec,
    score_window,
    select_topk,
    spec_sha256,
    truncation_equivalence_test,
)

ROOT = Path(__file__).resolve().parent
WF_DIR = W.WF_DIR
SPEC_V1_PATH = W.SPEC_PATH
SPEC_V2_PATH = WF_DIR / "process_spec_v2.json"
SELECTIONS_DIR = WF_DIR / "selections"
SCORES_V2_CSV = WF_DIR / "scores_v2.csv"

# Frozen v1 spec sha — v2's `supersedes` field must cite exactly this.
V1_SPEC_SHA = "b75aa7e7fe34eec11ebcd5b528c5a5f4d75f131fda1bd5e7cb54c773688503a0"

SCORE_FAMILY_V2 = "wf_score_v2"
PROC_LEVERAGE = 2.0

# Bootstrap = the already-ledgered 2026 leg of the v1 evaluation
# (results/v7/wf/scores.csv leg_year=2026, family wf_score, in rank order).
BOOTSTRAP_YEAR = 2026
BOOTSTRAP_CUTOFF = "2025-12-31"
BOOTSTRAP_PICKS = ["dual_momentum_vol", "xs_momentum_top30", "xs_momentum_12_1"]

FIRST_LIVE_RESELECTION = "2026-12-31"


# ═══════════════════════════════════════════════════════════════════════════
# Spec v2 (write-once, sha256-sealed — same sealing machinery as v1)
# ═══════════════════════════════════════════════════════════════════════════

def write_spec_v2(path: Path | str = SPEC_V2_PATH,
                  v1_path: Path | str = SPEC_V1_PATH) -> dict:
    """Write the frozen process spec v2 ONCE. Refuses to overwrite (frozen
    means frozen — the same discipline as wf_process.write_spec). The pool,
    exclusions, metric and scoring convention are COPIED from the verified
    v1 spec, never re-typed."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists — process spec v2 is FROZEN; refusing "
            f"to overwrite. Changes require a dated append-only amendment "
            f"(orchestrator approval) or a v3 spec with its own supersedes "
            f"chain."
        )
    v1 = load_spec(v1_path)  # verifies v1's sha seal (tamper check)
    if v1["sha256"] != V1_SPEC_SHA:
        raise ValueError(
            f"v1 spec sha {v1['sha256']!r} != expected {V1_SPEC_SHA!r} — "
            f"refusing to write a supersedes chain onto an unrecognized v1.")
    spec = {
        "program": ("wf_process_v2 (Stream 3 process automation — live "
                    "annual re-selection for shadow slot #3)"),
        "version": 2,
        "supersedes": {
            "spec": "results/v7/wf/process_spec.json",
            "sha256": V1_SPEC_SHA,
            "note": ("v1 is the WS3 walk-forward EVALUATION pre-registration "
                     "and remains byte-immutable; v2 governs the LIVE "
                     "process slot from the 2026 bootstrap leg onward"),
        },
        "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        # pool copied verbatim from the verified v1 spec (same 21 names —
        # combos would change the measured object):
        "pool": list(v1["pool"]),
        "pool_source": v1["pool_source"],
        "exclusion_rule": v1["exclusion_rule"],
        "exclusions": list(v1["exclusions"]),
        "cadence": ("annual re-selection at Dec-31 of Y-1, trade year Y "
                    "(WS3 evidence: selections stable 8/8 legs 2019-2026; "
                    "quarterly cadence has zero evidence and would be fresh "
                    "selection pressure)"),
        "metric": v1["metric"],
        "scoring_note": v1["scoring_note"],
        "k": W.K,
        "blend": "equal",
        "leverage": PROC_LEVERAGE,
        "leverage_note": ("2.0x for comparability with the primary slot; the "
                          "historical process MaxDD -57.7% vs the -45% kill "
                          "bar is disclosed as ACCEPTED RISK (adjudicated in "
                          "protocol_process.json, bot repo)"),
        "score_family": SCORE_FAMILY_V2,
        "score_family_note": (
            "v2 scoring runs are ledgered under family 'wf_score_v2' with "
            "this spec's sha256 in notes, window='full', final=True through "
            "exp_lib's front door (live cutoffs are past DEV_END — the v1 "
            "dev_lock_note rationale applies: allowed ONLY because this spec "
            "is frozen before the runs). Rows already ledgered by the v1 "
            "evaluation (family wf_score) MAY be reused instead of re-run — "
            "an identical re-execution is not a new hypothesis; any reuse is "
            "recorded in the selection file's provenance block."),
        "dev_end": v1["dev_end"],
        "selection_files": (
            "results/v7/wf/selections/selection_<year>.json — written ONCE "
            "by wf_select.py, which hard-refuses a duplicate cutoff; each "
            "file carries picks, selection-time Sharpes for the full pool, "
            "this spec's sha256, data-revision info, and its own sha256 "
            "(canonical-json seal, same convention as this spec)"),
        "bootstrap": {
            "selection_year": BOOTSTRAP_YEAR,
            "cutoff": BOOTSTRAP_CUTOFF,
            "picks": list(BOOTSTRAP_PICKS),
            "source": ("already-ledgered 2026 leg of the v1 evaluation "
                       "(results/v7/wf/scores.csv leg_year=2026, family "
                       "wf_score)"),
        },
        "first_live_reselection": FIRST_LIVE_RESELECTION,
        "deployment_note": ("the bot never self-modifies: selection_<year>."
                           "json -> scripts/build_variant.py --variant "
                           "process -> versioned bundle "
                           "combo_v2_process.<year>.<sha8> -> commit + "
                           "deploy; an unported sleeve fails the build "
                           "loudly (porting + parity is a deploy "
                           "prerequisite)"),
        "amendment_rules": (
            "pool additions ONLY via dated append-only amendment with "
            "orchestrator approval PLUS a fresh truncation-equivalence "
            "audit of the added names at the v1 settings (cutoff "
            f"{W.EQUIV_CUTOFF}, atol {W.EQUIV_ATOL:g}, coverage >= "
            f"{W.EQUIV_MIN_COVERAGE:.0%}); amendments take effect at the "
            "NEXT selection date, never retroactively; removals only for "
            "data unavailability, documented the same way; the core rule "
            "(k=3 equal blend by Sharpe, annual Dec-31 cadence, leverage "
            "2.0) is frozen — changing it requires a v3 spec with its own "
            "supersedes chain"),
    }
    spec["sha256"] = spec_sha256(spec)  # REUSED v1 sealing (same canon)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def load_spec_v2(path: Path | str = SPEC_V2_PATH) -> dict:
    """Load spec v2, verify its sha seal (via wf_process.load_spec — the
    seal convention is identical) AND its version/supersedes chain."""
    spec = load_spec(path)  # REUSED: recomputes + verifies sha256
    if spec.get("version") != 2:
        raise ValueError(f"{path}: expected version 2, got "
                         f"{spec.get('version')!r}")
    sup = spec.get("supersedes") or {}
    if sup.get("sha256") != V1_SPEC_SHA:
        raise ValueError(
            f"{path}: supersedes.sha256 {sup.get('sha256')!r} != frozen v1 "
            f"sha {V1_SPEC_SHA!r} — broken supersedes chain.")
    return spec


# ═══════════════════════════════════════════════════════════════════════════
# Scoring under family wf_score_v2
# ═══════════════════════════════════════════════════════════════════════════

def score_pool_v2(
    year: int,
    weights_by_name: dict[str, pd.DataFrame],
    spec_sha: str,
    existing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Family-'wf_score_v2' twin of wf_process.score_pool: identical row
    schema and engine settings (ONE run per candidate, weights sliced to
    <= Dec-31 Y-1, sliced-Sharpe convention), differing ONLY in the ledger
    family and the v2 spec sha carried in params/notes. score_pool hardcodes
    family='wf_score', so the loop is re-stated here; every ingredient
    (leg_cutoff, score_window, run_trial, eq_summary, TC/margin constants)
    is imported from the v1 machinery, not duplicated."""
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
            name=name, family=SCORE_FAMILY_V2,
            params={"leg_year": year, "score_cutoff": str(cutoff.date()),
                    "spec_sha": spec_sha[:16], "spec_version": 2,
                    "k": W.K, "metric": "sharpe"},
            window=window, exec_model="next_open", tc_bps=TC_BPS,
            leverage_cap=SCORE_LEV_CAP, margin_bps_annual=MARGIN_BPS,
            final=final,
            notes=(f"wf_score_v2 leg={year} cutoff={cutoff.date()} "
                   f"spec_v2_sha={spec_sha}; selection metric computed on "
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
    return pd.DataFrame(rows).sort_values(
        ["sharpe_at_selection"], ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    if SPEC_V2_PATH.exists():
        spec = load_spec_v2()
        print(f"[spec v2] FROZEN spec verified (sha {spec['sha256'][:16]}…) "
              f"— refusing to rewrite")
    else:
        spec = write_spec_v2()
        print(f"[spec v2] frozen NEW spec -> {SPEC_V2_PATH} "
              f"(sha {spec['sha256'][:16]}…)")
    print(f"  supersedes v1 sha {spec['supersedes']['sha256'][:16]}…")
    print(f"  bootstrap {spec['bootstrap']['selection_year']}: "
          f"{', '.join(spec['bootstrap']['picks'])}")
    print(f"  first live re-selection: {spec['first_live_reselection']}")
