"""Stream 3 tests: process spec v2 freeze semantics + wf_select discipline —
all on temp paths / synthetic score rows (no engine runs, no ledger writes,
no touching the real results/v7/wf artifacts).

1. spec v2 freeze — write_spec_v2 to a temp path succeeds, seals a
   verifiable sha256 (v1 canonical convention), carries the supersedes
   field citing exactly the frozen v1 sha, copies the 21-name pool from
   v1, and a SECOND write REFUSES (frozen means frozen); load_spec_v2
   detects tampering, wrong version, and a broken supersedes chain; a
   v1 spec whose sha is not the frozen b75aa7e7… is rejected as a
   supersedes base.
2. wf_select duplicate-cutoff refusal — run_selection writes once, then
   HARD-errors on the same cutoff (both at the pre-scoring guard and at
   write_selection); non-Dec-31 cutoffs are rejected.
3. selection file schema + sha self-consistency — required keys present,
   picks = top-k by Sharpe with deterministic tie-break, sha256 verifies
   via load_selection, tampering is detected; bootstrap divergence
   (computed picks != spec bootstrap) is a hard error.

check() asserts (not just records) so failures also surface under pytest.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python validation/test_wf_v2.py
(plain script, no pytest dependency needed; exits non-zero on failure)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wf_process as W  # noqa: E402
import wf_process_v2 as W2  # noqa: E402
import wf_select as SEL  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = ""):
    flag = "OK  " if cond else "FAIL"
    print(f"  {flag} {label} {detail}")
    if not cond:
        FAILURES.append(label)
    assert cond, f"{label} {detail}".strip()


def _temp_v1_spec(td: Path) -> Path:
    """A real v1 spec written by wf_process.write_spec into the temp dir.
    Its sha differs from the frozen production sha, which write_spec_v2
    must REJECT — used for the supersedes-base guard test."""
    p = td / "process_spec_v1_toy.json"
    W.write_spec(exclusions=[], path=p)
    return p


# ───── 1. spec v2 freeze semantics ────────────────────────────────────────
def test_spec_v2_freeze():
    print("\n[1] spec v2 freeze (write-once + sha seal + supersedes)")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "process_spec_v2.json"
        # production v1 spec is on disk and frozen — use it as the base
        spec = W2.write_spec_v2(path=p, v1_path=W2.SPEC_V1_PATH)
        check("spec v2 written with sha256", "sha256" in spec and p.exists())
        check("sha verifies (v1 canonical convention)",
              W.spec_sha256(spec) == spec["sha256"])
        check("version field == 2", spec.get("version") == 2)
        sup = spec.get("supersedes") or {}
        check("supersedes cites frozen v1 sha",
              sup.get("sha256") == W2.V1_SPEC_SHA
              and sup.get("sha256") == ("b75aa7e7fe34eec11ebcd5b528c5a5f4d7"
                                        "5f131fda1bd5e7cb54c773688503a0"))
        v1 = W.load_spec(W2.SPEC_V1_PATH)
        check("pool copied verbatim from v1 (21 names)",
              spec["pool"] == v1["pool"] and len(spec["pool"]) == 21)
        check("bootstrap = ledgered 2026 leg",
              spec["bootstrap"]["picks"] == ["dual_momentum_vol",
                                             "xs_momentum_top30",
                                             "xs_momentum_12_1"]
              and spec["bootstrap"]["cutoff"] == "2025-12-31")
        check("first live re-selection 2026-12-31",
              spec["first_live_reselection"] == "2026-12-31")
        check("k=3 equal blend, leverage 2.0",
              spec["k"] == 3 and spec["blend"] == "equal"
              and spec["leverage"] == 2.0)
        check("amendment rules present",
              "truncation-equivalence" in spec["amendment_rules"]
              and "NEXT selection date" in spec["amendment_rules"])
        try:
            W2.write_spec_v2(path=p, v1_path=W2.SPEC_V1_PATH)
            check("second write_spec_v2 refuses", False)
        except FileExistsError:
            check("second write_spec_v2 refuses", True)
        loaded = W2.load_spec_v2(p)
        check("load_spec_v2 round-trips", loaded["sha256"] == spec["sha256"])
        # tamper detection (sha seal)
        p.write_text(p.read_text().replace('"k": 3', '"k": 5'))
        try:
            W2.load_spec_v2(p)
            check("load_spec_v2 detects tampering", False)
        except ValueError:
            check("load_spec_v2 detects tampering", True)
        # wrong version / broken supersedes chain
        for mut, label in ((lambda s: s.update(version=1),
                            "load_spec_v2 rejects version != 2"),
                           (lambda s: s["supersedes"].update(sha256="00" * 32),
                            "load_spec_v2 rejects broken supersedes chain")):
            bad = json.loads(json.dumps(spec))
            mut(bad)
            bad["sha256"] = W.spec_sha256(bad)  # re-seal so ONLY the chain fails
            q = td / "bad.json"
            q.write_text(json.dumps(bad))
            try:
                W2.load_spec_v2(q)
                check(label, False)
            except ValueError:
                check(label, True)
        # a v1 base whose sha is not the frozen production sha is rejected
        toy_v1 = _temp_v1_spec(td)
        try:
            W2.write_spec_v2(path=td / "v2_from_toy.json", v1_path=toy_v1)
            check("write_spec_v2 rejects unrecognized v1 base", False)
        except ValueError:
            check("write_spec_v2 rejects unrecognized v1 base", True)


# ───── synthetic scores (hand-computable) ─────────────────────────────────
def _toy_scores(year: int, pool: list[str]) -> pd.DataFrame:
    """Descending Sharpe by pool order, with a deliberate tie in ranks 2-3
    so the deterministic name tie-break is exercised."""
    rows = []
    for i, n in enumerate(pool):
        sharpe = 2.0 - 0.05 * i
        if i in (1, 2):
            sharpe = 1.9  # tie -> alphabetical name order decides
        rows.append({"leg_year": year, "name": n,
                     "score_cutoff": str(W.leg_cutoff(year).date()),
                     "sharpe_at_selection": sharpe,
                     "calmar_at_selection": 1.0, "geo_at_selection": 0.02,
                     "maxdd_at_selection": -0.30})
    return pd.DataFrame(rows)


def _spec_in(td: Path) -> dict:
    return W2.write_spec_v2(path=td / "spec_v2.json",
                            v1_path=W2.SPEC_V1_PATH)


# ───── 2. duplicate-cutoff refusal ────────────────────────────────────────
def test_duplicate_cutoff_refusal():
    print("\n[2] wf_select duplicate-cutoff refusal")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        spec = _spec_in(td)
        out = td / "selections"
        pool = spec["pool"]
        # future leg (2027, cutoff 2026-12-31) — avoids the bootstrap guard
        scores = _toy_scores(2027, pool)
        src = {"mode": "injected", "ledger_family": "test"}
        rev = {"note": "synthetic (test)"}
        path = SEL.run_selection("2026-12-31", spec_path=td / "spec_v2.json",
                                 out_dir=out, scores=scores,
                                 scores_source=src, data_revision=rev)
        check("first run writes selection_2027.json",
              path.name == "selection_2027.json" and path.exists())
        try:
            SEL.run_selection("2026-12-31", spec_path=td / "spec_v2.json",
                              out_dir=out, scores=scores,
                              scores_source=src, data_revision=rev)
            check("second run for same cutoff REFUSES", False)
        except FileExistsError:
            check("second run for same cutoff REFUSES", True)
        # write-level guard too (belt and braces)
        sel = SEL.load_selection(path)
        try:
            SEL.write_selection(sel, out_dir=out)
            check("write_selection refuses existing file", False)
        except FileExistsError:
            check("write_selection refuses existing file", True)
        # cadence guard
        try:
            SEL.cutoff_to_year("2026-06-30")
            check("non-Dec-31 cutoff rejected", False)
        except ValueError:
            check("non-Dec-31 cutoff rejected", True)
        check("cutoff_to_year maps Dec-31 Y-1 -> Y",
              SEL.cutoff_to_year("2026-12-31") == 2027)


# ───── 3. selection schema + sha self-consistency ─────────────────────────
def test_selection_schema_and_sha():
    print("\n[3] selection file schema + sha self-consistency")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        spec = _spec_in(td)
        pool = spec["pool"]
        scores = _toy_scores(2027, pool)
        sel = SEL.build_selection(scores, spec, "2026-12-31",
                                  {"note": "synthetic"}, {"mode": "injected"})
        required = ["kind", "spec_version", "spec_sha256", "selection_year",
                    "cutoff", "k", "blend", "metric", "leverage", "picks",
                    "picks_sharpe_at_selection", "pool_scores",
                    "scores_source", "data_revision", "created_at", "sha256"]
        check("all required keys present",
              all(k in sel for k in required),
              str([k for k in required if k not in sel]))
        check("spec sha carried", sel["spec_sha256"] == spec["sha256"])
        check("k picks", len(sel["picks"]) == spec["k"])
        # rank 1 = pool[0] (sharpe 2.0); ranks 2-3 tie at 1.9 -> name order
        tied = sorted([pool[1], pool[2]])
        check("picks = top-k with deterministic tie-break",
              sel["picks"] == [pool[0]] + tied, str(sel["picks"]))
        check("pool_scores covers the full pool, rank 1..n",
              len(sel["pool_scores"]) == len(pool)
              and [r["rank"] for r in sel["pool_scores"]]
              == list(range(1, len(pool) + 1)))
        check("picks_sharpe matches score rows",
              sel["picks_sharpe_at_selection"][pool[0]] == 2.0
              and all(sel["picks_sharpe_at_selection"][n] == 1.9 for n in tied))
        # sha seal round-trip through disk
        out = td / "selections"
        path = SEL.write_selection(sel, out_dir=out)
        loaded = SEL.load_selection(path)
        check("selection sha verifies on load",
              loaded["sha256"] == sel["sha256"]
              and W.spec_sha256(loaded) == loaded["sha256"])
        path.write_text(path.read_text().replace('"k": 3', '"k": 4'))
        try:
            SEL.load_selection(path)
            check("load_selection detects tampering", False)
        except ValueError:
            check("load_selection detects tampering", True)
        # bootstrap divergence guard: toy scores for the bootstrap year would
        # pick names != the spec's ledgered bootstrap -> hard error
        boot_scores = _toy_scores(spec["bootstrap"]["selection_year"], pool)
        try:
            SEL.build_selection(boot_scores, spec,
                                spec["bootstrap"]["cutoff"],
                                {"note": "synthetic"}, {"mode": "injected"})
            check("bootstrap divergence is a hard error", False)
        except ValueError:
            check("bootstrap divergence is a hard error", True)


# ───── 4. production artifacts (read-only sanity, skipped if absent) ──────
def test_production_artifacts():
    print("\n[4] production artifacts (read-only)")
    if not W2.SPEC_V2_PATH.exists():
        print("  SKIP production spec v2 not written yet")
        return
    spec = W2.load_spec_v2()
    check("production spec v2 loads + verifies",
          spec["version"] == 2
          and spec["supersedes"]["sha256"] == W2.V1_SPEC_SHA)
    boot = SEL.selection_path(spec["bootstrap"]["selection_year"])
    if not boot.exists():
        print("  SKIP bootstrap selection not written yet")
        return
    sel = SEL.load_selection(boot)
    check("bootstrap selection sha verifies + picks match spec",
          sel["picks"] == spec["bootstrap"]["picks"]
          and sel["spec_sha256"] == spec["sha256"])


if __name__ == "__main__":
    test_spec_v2_freeze()
    test_duplicate_cutoff_refusal()
    test_selection_schema_and_sha()
    test_production_artifacts()
    print("\n" + ("ALL OK" if not FAILURES else f"FAILURES: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)
