"""Stream 3 — annual selection CLI for the wf process (spec v2).

    /opt/anaconda3/bin/python wf_select.py \
        --spec results/v7/wf/process_spec_v2.json --cutoff 2025-12-31

Writes results/v7/wf/selections/selection_<year>.json (year = cutoff year
+ 1: select at Dec-31 of Y-1, trade year Y) containing picks, selection-time
Sharpes for the FULL pool, the spec's sha256, data-revision info, and its
own sha256 (canonical-json seal, wf_process.spec_sha256 convention).

Discipline (binding):
- DUPLICATE-CUTOFF REFUSAL: if selection_<year>.json already exists this
  script hard-errors BEFORE any scoring runs. A selection is made once;
  re-running it is either a no-op re-render (forbidden — it would tempt
  silent revision) or a data-revision dispute, which must go through a dated
  amendment, not an overwrite.
- Scoring reuse-or-recompute: rows already ledgered by the v1 evaluation
  (results/v7/wf/scores.csv, family wf_score) are REUSED when the leg is
  complete there — identical re-execution is not a new hypothesis. Otherwise
  candidates are scored fresh under ledger family 'wf_score_v2'
  (wf_process_v2.score_pool_v2) with the v2 spec sha in notes, resuming from
  results/v7/wf/scores_v2.csv.
- The bootstrap selection (2026, cutoff 2025-12-31) must reproduce the
  spec's ledgered bootstrap picks exactly — any divergence is a hard error.

The bot never self-modifies: this file's output feeds
scripts/build_variant.py --variant process (bot repo) as a build input.

Tests: validation/test_wf_v2.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

import exp_lib
import wf_process as W
import wf_process_v2 as W2
from wf_process_v2 import SCORES_V2_CSV, SELECTIONS_DIR, SPEC_V2_PATH

ROOT = Path(__file__).resolve().parent
V1_SCORES_CSV = W.WF_DIR / "scores.csv"
SCORE_COLS = ["leg_year", "name", "score_cutoff", "sharpe_at_selection",
              "calmar_at_selection", "geo_at_selection", "maxdd_at_selection"]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection_path(year: int, out_dir: Path | str = SELECTIONS_DIR) -> Path:
    return Path(out_dir) / f"selection_{year}.json"


def cutoff_to_year(cutoff: str) -> int:
    """Selection year for a cutoff: Dec-31 of Y-1 selects for year Y.
    Non-Dec-31 cutoffs are rejected (annual cadence is the spec)."""
    ts = pd.Timestamp(cutoff)
    year = ts.year + 1
    if W.leg_cutoff(year) != ts:  # REUSED: the spec's cutoff convention
        raise ValueError(
            f"cutoff {cutoff!r} is not a Dec-31 selection date — the v2 "
            f"cadence is annual (select at Dec-31 of Y-1, trade year Y).")
    return year


def data_revision_info() -> dict:
    """Pin WHAT data/code this selection was computed from: sha256 of the
    frozen input panel (data_cache.pkl — the panel exp_lib scoring reads)
    and the strategy-code hash that keys every cached weight frame."""
    cache = ROOT / "data_cache.pkl"
    return {
        "data_cache_pkl": {"path": "data_cache.pkl",
                           "sha256": _sha256_file(cache)},
        "strategies_code_hash": exp_lib._CODE_HASH,
        "note": ("data_cache.pkl is the frozen input panel for all wf "
                 "scoring; strategies_code_hash (sha1 of strategies.py/"
                 "strategies_v2.py/ensemble.py) keys the weight cache — a "
                 "different hash at the next selection means strategy code "
                 "changed and the equivalence audit must be re-checked"),
    }


def gather_scores(year: int, spec: dict) -> tuple[pd.DataFrame, dict]:
    """(scores df for the leg, provenance dict). Reuses the complete v1 leg
    from scores.csv when available; otherwise recomputes under family
    wf_score_v2 (resuming from scores_v2.csv)."""
    pool = list(spec["pool"])
    if V1_SCORES_CSV.exists():
        v1 = pd.read_csv(V1_SCORES_CSV)
        leg = v1[v1["leg_year"] == year]
        if set(pool) <= set(leg["name"]):
            source = {
                "mode": "reused",
                "file": "results/v7/wf/scores.csv",
                "file_sha256": _sha256_file(V1_SCORES_CSV),
                "ledger_family": "wf_score",
                "note": ("complete leg already ledgered by the v1 "
                         "evaluation; identical re-execution is not a new "
                         "hypothesis (spec v2 score_family_note)"),
            }
            df = leg[leg["name"].isin(pool)][SCORE_COLS].reset_index(drop=True)
            return df, source
    # recompute path — ledger family wf_score_v2, resume from scores_v2.csv
    existing = pd.read_csv(SCORES_V2_CSV) if SCORES_V2_CSV.exists() else None
    weights = W2.build_pool_weights()  # REUSED (cache hits under same code)
    df = W2.score_pool_v2(year, {n: weights[n] for n in pool},
                          spec["sha256"], existing=existing)
    merged = df if existing is None else pd.concat(
        [existing[existing["leg_year"] != year], df], ignore_index=True)
    SCORES_V2_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(SCORES_V2_CSV, index=False)
    source = {"mode": "computed", "ledger_family": W2.SCORE_FAMILY_V2,
              "file": "results/v7/wf/scores_v2.csv",
              "file_sha256": _sha256_file(SCORES_V2_CSV)}
    return df[SCORE_COLS].reset_index(drop=True), source


def build_selection(scores: pd.DataFrame, spec: dict, cutoff: str,
                    data_revision: dict, scores_source: dict) -> dict:
    """Assemble the sealed selection record (pure — no I/O, testable on
    synthetic score rows). Picks via wf_process.select_topk (REUSED:
    metric descending, deterministic name tie-break)."""
    year = cutoff_to_year(cutoff)
    k = int(spec["k"])
    picks = W.select_topk(scores, k=k)
    if len(picks) < k:
        raise ValueError(f"only {len(picks)} scored names for leg {year} — "
                         f"cannot select top-{k}")
    ranked = scores.sort_values(
        ["sharpe_at_selection", "name"], ascending=[False, True]
    ).reset_index(drop=True)
    sharpe_by_name = dict(zip(ranked["name"], ranked["sharpe_at_selection"]))
    boot = spec.get("bootstrap") or {}
    if year == boot.get("selection_year") and picks != list(boot.get("picks", [])):
        raise ValueError(
            f"bootstrap divergence: computed picks {picks} != spec bootstrap "
            f"{boot.get('picks')} — the ledgered 2026 leg is the binding "
            f"bootstrap; STOP and audit the score source.")
    sel = {
        "kind": "wf_process_v2_selection",
        "spec_version": 2,
        "spec_sha256": spec["sha256"],
        "selection_year": year,
        "cutoff": cutoff,
        "k": k,
        "blend": spec["blend"],
        "metric": spec["metric"],
        "leverage": spec["leverage"],
        "picks": picks,
        "picks_sharpe_at_selection": {n: float(sharpe_by_name[n]) for n in picks},
        "pool_scores": [
            {"rank": i + 1,
             "name": r["name"],
             "sharpe_at_selection": float(r["sharpe_at_selection"]),
             "calmar_at_selection": float(r["calmar_at_selection"]),
             "geo_at_selection": float(r["geo_at_selection"]),
             "maxdd_at_selection": float(r["maxdd_at_selection"])}
            for i, r in ranked.iterrows()],
        "scores_source": scores_source,
        "data_revision": data_revision,
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    sel["sha256"] = W.spec_sha256(sel)  # REUSED canonical-json seal
    return sel


def load_selection(path: Path | str) -> dict:
    """Load + verify a selection file's self-sha (tamper check)."""
    sel = json.loads(Path(path).read_text())
    got = W.spec_sha256(sel)
    if got != sel.get("sha256"):
        raise ValueError(
            f"{path}: sha mismatch (recorded {sel.get('sha256')!r} != "
            f"recomputed {got!r}) — selection file modified after sealing.")
    return sel


def write_selection(sel: dict, out_dir: Path | str = SELECTIONS_DIR) -> Path:
    path = selection_path(sel["selection_year"], out_dir)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists — a selection is made ONCE per cutoff; "
            f"refusing to overwrite. Disputes go through a dated amendment.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sel, indent=2) + "\n")
    return path


def run_selection(cutoff: str, spec_path: Path | str = SPEC_V2_PATH,
                  out_dir: Path | str = SELECTIONS_DIR,
                  scores: pd.DataFrame | None = None,
                  scores_source: dict | None = None,
                  data_revision: dict | None = None) -> Path:
    """Full pipeline. `scores`/`scores_source`/`data_revision` are injectable
    for tests only; production callers pass just cutoff/spec_path."""
    year = cutoff_to_year(cutoff)
    target = selection_path(year, out_dir)
    if target.exists():  # HARD refusal BEFORE any scoring runs
        raise FileExistsError(
            f"{target} already exists — refusing to run the {cutoff} "
            f"selection twice (duplicate-cutoff refusal; see module "
            f"docstring).")
    spec = W2.load_spec_v2(spec_path)
    if scores is None:
        scores, scores_source = gather_scores(year, spec)
    if data_revision is None:
        data_revision = data_revision_info()
    sel = build_selection(scores, spec, cutoff, data_revision,
                          scores_source or {"mode": "injected"})
    return write_selection(sel, out_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default=str(SPEC_V2_PATH),
                    help="path to the frozen process_spec_v2.json")
    ap.add_argument("--cutoff", required=True,
                    help="selection date YYYY-MM-DD (must be a Dec-31)")
    args = ap.parse_args(argv)
    path = run_selection(args.cutoff, spec_path=args.spec)
    sel = load_selection(path)
    print(f"[wf_select] wrote {path} (sha {sel['sha256'][:16]}…)")
    print(f"  year {sel['selection_year']}  cutoff {sel['cutoff']}  "
          f"spec sha {sel['spec_sha256'][:16]}…")
    for n in sel["picks"]:
        print(f"  pick: {n:28s} sharpe {sel['picks_sharpe_at_selection'][n]:.4f}")
    print(f"  scores: {sel['scores_source'].get('mode')} "
          f"({sel['scores_source'].get('ledger_family')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
