"""Shadow-protocol tests (Improvement Roadmap v2, Streams 1+3).

Covers:
  - protocol_exp.json / protocol_process.json parse + validate with the
    exact pre-registered thresholds;
  - per-slot protocol resolution (combo_v2 → frozen protocol.json,
    combo_v2_exp → protocol_exp.json, combo_v2_process[.<ver>] →
    protocol_process.json) and per-slot sha binding;
  - the PRIMARY protocol resolution UNCHANGED — its bytes are pinned to
    the sha recorded before the shadow work started;
  - common-window logic (max of the two forward_test_start dates);
  - relative-metric scoring on synthetic two-slot tracking data with
    hand-computed expectations;
  - REPORT_ONLY entries never entering the overall verdict;
  - journal freeze-state capture for the freeze ablation replay.
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.protocol import (  # noqa: E402
    PROTOCOL_EXP_JSON_PATH,
    PROTOCOL_JSON_PATH,
    PROTOCOL_PROCESS_JSON_PATH,
    bind_forward_test_start,
    common_window_start,
    compute_relative_metrics,
    file_sha256,
    live_calmar_from_tracking,
    load_protocol,
    load_protocol_for,
    protocol_path_for,
    score_protocol,
)
from core.tracking import compute_tracking  # noqa: E402

# sha256 of the FROZEN primary protocol.json, recorded 2026-07-18 before
# any shadow-slot work. If this ever fails, the primary's pre-registered
# contract was edited — that is exactly the MODIFIED_AFTER_START event
# the whole apparatus exists to catch. Never update this constant without
# a dated amendment approved by the orchestrator.
PRIMARY_PROTOCOL_SHA256 = (
    "8aca8c1334b26bdd3ce29c4b7257d44595ba0241f6c20f71abe78126375b9339")

EXP = load_protocol(PROTOCOL_EXP_JSON_PATH)
PROC = load_protocol(PROTOCOL_PROCESS_JSON_PATH)
PRIMARY = load_protocol(PROTOCOL_JSON_PATH)


# ───── protocol files parse + validate with the registered thresholds ────

def test_protocol_exp_parses_with_registered_thresholds():
    by_id = {c["id"]: c for c in EXP["criteria"]}
    assert set(by_id) == {
        "XK1", "XK2", "XK3a", "XK3b", "X3-VOL", "X3-SLIP", "X3-RECON",
        "X12-REL-DD", "X12-REL-CALMAR", "X12-REL-GEO", "X12-ABS-DD"}

    # Continuous kill tighter than the primary: DD control is the pitch.
    assert by_id["XK1"]["op"] == "<=" and by_id["XK1"]["threshold"] == -40
    assert by_id["XK1"]["action"] == "KILL"
    # XK2/XK3 mirror the primary's K2/K3a/K3b exactly.
    prim = {c["id"]: c for c in PRIMARY["criteria"]}
    for mine, theirs in (("XK2", "K2"), ("XK3a", "K3a"), ("XK3b", "K3b")):
        for field in ("metric", "op", "threshold", "checkpoint", "action"):
            assert by_id[mine][field] == prim[theirs][field], (mine, field)
    # 3mo is operational-only: vol band [6, 22], slip <= 15, recon green.
    assert by_id["X3-VOL"]["threshold"] == [6, 22]
    assert by_id["X3-SLIP"]["threshold"] == 15
    assert by_id["X3-RECON"]["threshold"] == 2
    for cid in ("X3-VOL", "X3-SLIP", "X3-RECON"):
        assert by_id[cid]["action"] == "OPERATIONAL_REVIEW_IF_FAIL"
        assert by_id[cid]["checkpoint"] == "3mo"
    assert "no performance decision" in by_id["X3-VOL"]["description"].lower()
    assert "OPERATIONAL-ONLY" in EXP["three_month_note"]

    # 12mo relative-first legs.
    assert by_id["X12-REL-DD"]["threshold"] == -3
    assert by_id["X12-REL-CALMAR"]["threshold"] == 0
    assert by_id["X12-REL-GEO"]["threshold"] == -1.0
    assert by_id["X12-ABS-DD"]["threshold"] == -35
    for cid in ("X12-REL-DD", "X12-REL-CALMAR", "X12-REL-GEO", "X12-ABS-DD"):
        assert by_id[cid]["action"] == "REQUIRED_FOR_SUCCESS"
        assert by_id[cid]["checkpoint"] == "12mo"
        assert by_id[cid]["op"] == ">="
    for cid in ("X12-REL-DD", "X12-REL-CALMAR", "X12-REL-GEO"):
        assert by_id[cid]["relative_to"] == "combo_v2"
        assert by_id[cid]["metric"].startswith("rel_")

    # Power disclosure + decision rule + freeze DISPUTED clause.
    assert EXP["power_disclosure"]["paired_diff_se_pp_per_month"] == 1.2
    assert "carry the verdict" in EXP["power_disclosure"]["note"]
    assert "All four" in EXP["decision_rule"]["SUCCESS"]
    assert "XK1" in EXP["decision_rule"]["FAIL"]
    assert "2 of the three X12-REL" in EXP["decision_rule"]["FAIL"]
    assert "MIXED" in EXP["decision_rule"]
    fz = EXP["freeze_disputed"]
    assert fz["status"] == "DISPUTED"
    assert "12 months" in fz["adjudication_scope"]
    assert "NOT a slot pass/fail leg" in fz["adjudication_scope"]
    assert "without the freeze multiplier" in fz["ablation_replay"].lower() \
        or "freeze multiplier removed" in fz["ablation_replay"].lower()


def test_protocol_process_parses_with_registered_thresholds():
    by_id = {c["id"]: c for c in PROC["criteria"]}
    assert set(by_id) == {
        "PK1", "PK2", "PK3a", "PK3b",
        "P12-GEO", "P12-REL-PRIMARY", "P12-DD", "P-vs-CANDIDATE-X"}

    assert by_id["PK1"]["op"] == "<=" and by_id["PK1"]["threshold"] == -45
    prim = {c["id"]: c for c in PRIMARY["criteria"]}
    for mine, theirs in (("PK2", "K2"), ("PK3a", "K3a"), ("PK3b", "K3b")):
        for field in ("metric", "op", "threshold", "checkpoint", "action"):
            assert by_id[mine][field] == prim[theirs][field], (mine, field)

    assert by_id["P12-GEO"]["threshold"] == 1.65
    assert by_id["P12-REL-PRIMARY"]["threshold"] == -1.5
    assert by_id["P12-REL-PRIMARY"]["relative_to"] == "combo_v2"
    assert by_id["P12-DD"]["threshold"] == -45
    for cid in ("P12-GEO", "P12-REL-PRIMARY", "P12-DD"):
        assert by_id[cid]["action"] == "REQUIRED_FOR_SUCCESS"
        assert by_id[cid]["checkpoint"] == "12mo"

    # vs-candidate-X is REPORT-ONLY with no threshold at all.
    rep = by_id["P-vs-CANDIDATE-X"]
    assert rep["action"] == "REPORT_ONLY"
    assert rep["threshold"] is None
    assert rep["relative_to"] == "combo_v2_exp"

    # P-SWAP pre-authorized annual re-selection clause.
    swap = PROC["P_SWAP"]
    assert "PRE-AUTHORIZED" in swap["clause"]
    assert "selection" in swap["clause"].lower()
    assert "sha" in swap["clause"].lower()
    assert "10 trading days" in swap["miss_rule"]
    assert "OPERATIONAL FAILURE" in swap["miss_rule"]
    assert "NOT an amendment" in PROC["amendment_rule"]


def test_report_only_threshold_null_only_for_report_only(tmp_path):
    """A null threshold is rejected unless the action is REPORT_ONLY."""
    bad = {"criteria": [{
        "id": "B1", "description": "x", "metric": "geo_monthly_pct",
        "op": ">=", "threshold": None, "checkpoint": "12mo",
        "action": "REQUIRED_FOR_SUCCESS"}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="threshold"):
        load_protocol(p)


# ───── per-slot resolution + sha binding ─────────────────────────────────

def test_per_slot_protocol_resolution():
    assert protocol_path_for("combo_v2") == PROTOCOL_JSON_PATH
    assert protocol_path_for("combo_v2_exp") == PROTOCOL_EXP_JSON_PATH
    assert protocol_path_for("combo_v2_process") == PROTOCOL_PROCESS_JSON_PATH
    assert protocol_path_for("combo_v2_proc") == PROTOCOL_PROCESS_JSON_PATH
    # Versioned process bundles resolve by prefix.
    assert (protocol_path_for("combo_v2_process.2026.ab12cd34")
            == PROTOCOL_PROCESS_JSON_PATH)
    # Unknown / legacy names keep the pre-shadow behavior (primary file).
    assert protocol_path_for("v6") == PROTOCOL_JSON_PATH
    assert protocol_path_for(None) == PROTOCOL_JSON_PATH


def test_primary_protocol_resolution_and_bytes_unchanged():
    """The frozen primary contract: same file, same bytes, same sha as
    recorded before any shadow-slot work. This is the tamper tripwire."""
    proto = load_protocol_for("combo_v2")
    assert proto["_sha256"] == PRIMARY_PROTOCOL_SHA256
    assert file_sha256(PROTOCOL_JSON_PATH) == PRIMARY_PROTOCOL_SHA256
    # And the shadow files are genuinely different contracts.
    assert EXP["_sha256"] != PRIMARY_PROTOCOL_SHA256
    assert PROC["_sha256"] != PRIMARY_PROTOCOL_SHA256
    assert EXP["_sha256"] != PROC["_sha256"]


def test_bind_forward_test_start_per_slot_set_once():
    # Each slot binds ITS OWN protocol file's sha.
    st_exp: dict = {}
    assert bind_forward_test_start(st_exp, "combo_v2_exp", "2026-08-24")
    assert st_exp["forward_test_start"] == "2026-08-24"
    assert st_exp["protocol_sha256"] == file_sha256(PROTOCOL_EXP_JSON_PATH)

    st_proc: dict = {}
    assert bind_forward_test_start(
        st_proc, "combo_v2_process.2026.ab12cd34", "2026-09-01")
    assert st_proc["protocol_sha256"] == file_sha256(
        PROTOCOL_PROCESS_JSON_PATH)

    # Primary semantics untouched: binds exactly protocol.json's sha —
    # identical to what core/runner.py records inline today.
    st_prim: dict = {}
    assert bind_forward_test_start(st_prim, "combo_v2", "2026-07-20")
    assert st_prim["protocol_sha256"] == PRIMARY_PROTOCOL_SHA256

    # Set-once: a second call never overwrites a running clock.
    assert not bind_forward_test_start(st_prim, "combo_v2", "2026-07-21")
    assert st_prim["forward_test_start"] == "2026-07-20"


# ───── common-window logic ───────────────────────────────────────────────

def test_common_window_start_is_max_of_the_two_dates():
    assert common_window_start("2026-07-20", "2026-08-24") == "2026-08-24"
    assert common_window_start("2026-08-24", "2026-07-20") == "2026-08-24"
    assert common_window_start("2026-07-20T09:30:00", "2026-07-20") \
        == "2026-07-20"
    assert common_window_start(None, "2026-08-24") is None
    assert common_window_start("2026-07-20", None) is None


# ───── relative metrics: hand-computed on synthetic tracking dicts ───────

def _tr(geo_pct, dd_pct, months=12.0):
    return {"status": "ok", "months_elapsed": months,
            "live_geo_monthly_pct": geo_pct, "max_drawdown_pct": dd_pct,
            "cum_log_return": 0.0, "cum_return_pct": 0.0,
            "realized_vol_monthly_pct": 14.0,
            "spy": {"live_minus_2x_spy_pp": 0.0}}


def test_live_calmar_hand_computed():
    # geo 2.5%/mo, MaxDD -20%: Calmar = (1.025^12 - 1) / 0.20
    expect = ((1.025 ** 12) - 1.0) / 0.20
    assert live_calmar_from_tracking(_tr(2.5, -20.0)) \
        == pytest.approx(expect, rel=1e-12)
    # Unmeasurable cases → None, never a fabricated number.
    assert live_calmar_from_tracking(_tr(2.5, 0.0)) is None
    assert live_calmar_from_tracking({"status": "insufficient_data"}) is None
    assert live_calmar_from_tracking(None) is None


def test_compute_relative_metrics_hand_computed():
    own = _tr(2.5, -20.0)     # candidate
    base = _tr(3.0, -30.0)    # primary
    rel = compute_relative_metrics(own, base)
    assert rel["rel_geo_monthly_diff_pp"] == pytest.approx(-0.5)
    assert rel["rel_max_drawdown_diff_pp"] == pytest.approx(10.0)
    own_calmar = ((1.025 ** 12) - 1.0) / 0.20     # 1.72446...
    base_calmar = ((1.030 ** 12) - 1.0) / 0.30    # 1.41921...
    assert rel["rel_live_calmar_diff"] == pytest.approx(
        own_calmar - base_calmar, abs=1e-4)
    # Missing baseline → all diffs None (NA downstream).
    rel_na = compute_relative_metrics(own, None)
    assert rel_na["rel_geo_monthly_diff_pp"] is None
    assert rel_na["rel_max_drawdown_diff_pp"] is None
    assert rel_na["rel_live_calmar_diff"] is None


# ───── end-to-end: synthetic two-slot equity curves through
#       core/tracking.py on the common window, then the exp scorer ───────

def _daily_points(start_iso: str, values: list[float]):
    """[(ts_ms, equity)] on consecutive calendar days from start_iso."""
    from datetime import datetime, timedelta, timezone
    t0 = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
    return [(int((t0 + timedelta(days=i)).timestamp() * 1000), v)
            for i, v in enumerate(values)]


def _piecewise(knots: list[tuple[int, float]]) -> list[float]:
    """Linear interpolation through (index, value) knots, inclusive."""
    out = []
    for (i0, v0), (i1, v1) in zip(knots, knots[1:]):
        for i in range(i0, i1):
            out.append(v0 + (v1 - v0) * (i - i0) / (i1 - i0))
    out.append(knots[-1][1])
    return out


COMMON = "2026-08-24"        # candidate started later → common start
PRIMARY_START = "2026-07-20"


def _two_slot_common_window_tracking():
    # Primary: 20 pre-common days at 95 (must be EXCLUDED by the common
    # window), then 100 → 150 → 105 → 140 over 252 steps (12.0 months).
    prim_vals = [95.0] * 20 + _piecewise(
        [(0, 100.0), (126, 150.0), (200, 105.0), (252, 140.0)])
    prim_pts = _daily_points("2026-08-04", prim_vals)
    # Candidate: starts AT the common date, 100 → 130 → 104 → 135.
    cand_vals = _piecewise(
        [(0, 100.0), (126, 130.0), (200, 104.0), (252, 135.0)])
    cand_pts = _daily_points(COMMON, cand_vals)

    common = common_window_start(COMMON, PRIMARY_START)
    assert common == COMMON
    ref = {"geo_monthly_return": 0.03, "mean_monthly_return": 0.04,
           "sharpe": 1.0}
    prim_tr = compute_tracking(prim_pts, ref, common)
    cand_tr = compute_tracking(cand_pts, ref, common)
    return cand_tr, prim_tr


def test_common_window_slices_both_curves_and_matches_hand_math():
    cand_tr, prim_tr = _two_slot_common_window_tracking()
    # The primary's 20 pre-common points are excluded: exactly 252 steps.
    assert prim_tr["n_days"] == 252
    assert cand_tr["n_days"] == 252
    assert prim_tr["months_elapsed"] == pytest.approx(12.0)
    # Hand math: geo = (end/100)^(1/12) − 1; MaxDD from peak→trough.
    assert prim_tr["live_geo_monthly_pct"] == pytest.approx(
        (1.40 ** (1 / 12.0) - 1.0) * 100.0, abs=1e-3)     # 2.8436
    assert prim_tr["max_drawdown_pct"] == pytest.approx(-30.0, abs=1e-6)
    assert cand_tr["live_geo_monthly_pct"] == pytest.approx(
        (1.35 ** (1 / 12.0) - 1.0) * 100.0, abs=1e-3)     # 2.5324
    assert cand_tr["max_drawdown_pct"] == pytest.approx(-20.0, abs=1e-6)


def test_exp_relative_scoring_end_to_end_all_four_legs_pass():
    cand_tr, prim_tr = _two_slot_common_window_tracking()
    rel = compute_relative_metrics(cand_tr, prim_tr)
    # Hand-computed expectations on the common window.
    assert rel["rel_max_drawdown_diff_pp"] == pytest.approx(10.0, abs=1e-3)
    assert rel["rel_geo_monthly_diff_pp"] == pytest.approx(
        (1.35 ** (1 / 12.0) - 1.40 ** (1 / 12.0)) * 100.0,  # −0.3112
        abs=2e-3)
    cand_calmar = 0.35 / 0.20                              # 1.75
    prim_calmar = 0.40 / 0.30                              # 1.3333
    assert rel["rel_live_calmar_diff"] == pytest.approx(
        cand_calmar - prim_calmar, abs=1e-3)               # +0.4167

    state = {"forward_test_start": COMMON,
             "protocol_sha256": EXP["_sha256"]}
    score = score_protocol(
        cand_tr, None, None, state, EXP,
        rel_metrics={"combo_v2": rel,
                     "_common_window_start": {"combo_v2": COMMON}})
    by_id = {c["id"]: c for c in score["criteria"]}
    assert score["binding"] == "intact"
    assert score["checkpoint"] == "12mo"
    # All four 12mo legs pass: 10 >= -3, +0.42 >= 0, -0.31 >= -1.0,
    # -20 >= -35.
    for cid in ("X12-REL-DD", "X12-REL-CALMAR", "X12-REL-GEO",
                "X12-ABS-DD"):
        assert by_id[cid]["status"] == "PASS", cid
    for cid in ("X12-REL-DD", "X12-REL-CALMAR", "X12-REL-GEO"):
        assert by_id[cid]["relative_to"] == "combo_v2"
        assert by_id[cid]["common_window_start"] == COMMON
    assert by_id["XK1"]["status"] == "PASS"   # -20% > -40%
    rs = score["required_for_success"]
    assert rs["total"] == 4 and rs["passed"] == 4 and rs["failed"] == 0
    assert rs["rel_legs_failed"] == 0


def test_exp_kill_tighter_than_primary_at_minus_41():
    """DD −41%: kills the exp slot (XK1 ≤ −40) but NOT the primary
    (K1 ≤ −45) — the tighter bar is the candidate's own pitch."""
    tr = _tr(1.0, -41.0, months=2.0)
    tr["live_drawdown_pct"] = -41.0
    exp_state = {"forward_test_start": "2026-08-24",
                 "protocol_sha256": EXP["_sha256"]}
    score = score_protocol(tr, None, None, exp_state, EXP)
    assert {c["id"]: c for c in score["criteria"]}["XK1"]["status"] == "KILL"
    assert score["overall"] == "KILL"

    prim_state = {"forward_test_start": "2026-07-20",
                  "protocol_sha256": PRIMARY["_sha256"]}
    score_p = score_protocol(tr, None, None, prim_state, PRIMARY)
    assert {c["id"]: c for c in score_p["criteria"]}["K1"]["status"] == "PASS"
    assert score_p["overall"] != "KILL"


def test_exp_rel_legs_failed_count_feeds_the_fail_rule():
    """Decision rule: FAIL when >= 2 REL legs fail. The scorer reports
    the count (the written rule stays the authority)."""
    tr = _tr(2.0, -30.0, months=12.5)
    rel = {"rel_max_drawdown_diff_pp": -5.0,   # fail (< -3)
           "rel_live_calmar_diff": -0.2,       # fail (< 0)
           "rel_geo_monthly_diff_pp": -0.5}    # pass (>= -1.0)
    state = {"forward_test_start": "2026-08-24",
             "protocol_sha256": EXP["_sha256"]}
    score = score_protocol(tr, None, None, state, EXP,
                           rel_metrics={"combo_v2": rel})
    by_id = {c["id"]: c for c in score["criteria"]}
    assert by_id["X12-REL-DD"]["status"] == "WARN"
    assert by_id["X12-REL-CALMAR"]["status"] == "WARN"
    assert by_id["X12-REL-GEO"]["status"] == "PASS"
    assert score["required_for_success"]["rel_legs_failed"] == 2
    assert score["overall"] == "WARN"  # REQUIRED failures deny SUCCESS


def test_rel_criteria_na_without_baseline_data():
    """No rel_metrics (baseline slot dark / no common window) → REL legs
    NA with an explanatory reason, never a guess."""
    tr = _tr(2.0, -20.0, months=12.5)
    state = {"forward_test_start": "2026-08-24",
             "protocol_sha256": EXP["_sha256"]}
    score = score_protocol(tr, None, None, state, EXP)
    by_id = {c["id"]: c for c in score["criteria"]}
    for cid in ("X12-REL-DD", "X12-REL-CALMAR", "X12-REL-GEO"):
        assert by_id[cid]["status"] == "NA"
        assert "relative metric" in by_id[cid]["na_reason"]
    # The absolute backstop still scores.
    assert by_id["X12-ABS-DD"]["status"] == "PASS"


def test_rel_criteria_na_before_12mo_checkpoint():
    tr = _tr(2.0, -20.0, months=6.0)
    rel = {"rel_max_drawdown_diff_pp": 10.0,
           "rel_live_calmar_diff": 0.4,
           "rel_geo_monthly_diff_pp": -0.3}
    state = {"forward_test_start": "2026-08-24",
             "protocol_sha256": EXP["_sha256"]}
    score = score_protocol(tr, None, None, state, EXP,
                           rel_metrics={"combo_v2": rel})
    by_id = {c["id"]: c for c in score["criteria"]}
    for cid in ("X12-REL-DD", "X12-REL-CALMAR", "X12-REL-GEO"):
        assert by_id[cid]["status"] == "NA"
        assert "12mo not reached" in by_id[cid]["na_reason"] \
            or "checkpoint 12mo" in by_id[cid]["na_reason"]


# ───── process protocol scoring + REPORT_ONLY ────────────────────────────

def test_process_scoring_and_report_only_never_enters_verdict():
    tr = _tr(2.0, -30.0, months=12.5)
    rel_metrics = {
        "combo_v2": {"rel_geo_monthly_diff_pp": -1.0},       # pass >= -1.5
        # Catastrophic vs candidate X — must stay INFO, never WARN/KILL.
        "combo_v2_exp": {"rel_geo_monthly_diff_pp": -50.0},
        "_common_window_start": {"combo_v2": "2026-09-01",
                                 "combo_v2_exp": "2026-09-01"},
    }
    state = {"forward_test_start": "2026-09-01",
             "protocol_sha256": PROC["_sha256"]}
    score = score_protocol(tr, None, None, state, PROC,
                           rel_metrics=rel_metrics)
    by_id = {c["id"]: c for c in score["criteria"]}
    assert by_id["PK1"]["status"] == "PASS"       # -30 > -45
    assert by_id["P12-GEO"]["status"] == "PASS"   # 2.0 >= 1.65
    assert by_id["P12-REL-PRIMARY"]["status"] == "PASS"
    assert by_id["P12-REL-PRIMARY"]["relative_to"] == "combo_v2"
    assert by_id["P12-DD"]["status"] == "PASS"    # -30 >= -45
    rep = by_id["P-vs-CANDIDATE-X"]
    assert rep["status"] == "INFO"
    assert rep["value"] == pytest.approx(-50.0)
    assert rep["relative_to"] == "combo_v2_exp"
    # REPORT_ONLY excluded from the verdict: overall PASS despite -50.
    assert score["overall"] == "PASS"
    # And it is not counted as a required leg.
    assert score["required_for_success"]["total"] == 3


def test_process_geo_below_floor_denies_success():
    tr = _tr(1.5, -30.0, months=12.5)   # below the 1.65 floor
    state = {"forward_test_start": "2026-09-01",
             "protocol_sha256": PROC["_sha256"]}
    score = score_protocol(tr, None, None, state, PROC)
    by_id = {c["id"]: c for c in score["criteria"]}
    assert by_id["P12-GEO"]["status"] == "WARN"
    assert score["overall"] == "WARN"


def test_primary_scoring_unaffected_by_rel_machinery():
    """The primary protocol has no rel_ criteria: passing rel_metrics or
    not yields identical scores (byte-identical pre-shadow behavior)."""
    tr = _tr(3.0, -20.0, months=1.0)
    state = {"forward_test_start": "2026-07-20",
             "protocol_sha256": PRIMARY["_sha256"]}
    a = score_protocol(tr, None, None, state, PRIMARY)
    b = score_protocol(tr, None, None, state, PRIMARY,
                       rel_metrics={"combo_v2": {
                           "rel_geo_monthly_diff_pp": -99.0}})
    assert a == b
    assert a["binding"] == "intact"
    assert all("relative_to" not in c for c in a["criteria"])


# ───── journal freeze-state capture (ablation replay) ────────────────────

def test_freeze_context_from_state_extracts_multiplier_and_state():
    from core.journal import freeze_context_from_state
    state = {
        "freeze_state": {"active": True, "frozen_since": "2026-09-10",
                         "peak_at_freeze": 512.3},
        "history": [
            {"date": "2026-08-24", "strategy_diagnostics":
                {"freeze_multiplier": 1.0, "freeze_active": False}},
            {"date": "2026-09-12", "strategy_diagnostics":
                {"freeze_multiplier": 0.4, "freeze_active": True,
                 "book_gate": 0.9}},
        ],
    }
    ctx = freeze_context_from_state(state)
    assert ctx["freeze_active"] is True
    assert ctx["freeze_multiplier"] == pytest.approx(0.4)  # NEWEST entry
    assert json.loads(ctx["freeze_state"])["frozen_since"] == "2026-09-10"

    # A slot with no freeze machinery (the frozen primary) → all None.
    ctx0 = freeze_context_from_state(
        {"history": [{"strategy_diagnostics": {"book_gate": 1.0}}]})
    assert ctx0 == {"freeze_active": None, "freeze_multiplier": None,
                    "freeze_state": None}
    assert freeze_context_from_state(None)["freeze_active"] is None


def test_log_trade_autofills_freeze_fields_from_slot_state(
        tmp_path, monkeypatch):
    import core.journal as journal_mod
    from core.journal import TradeJournal, TradeRecord
    monkeypatch.setattr(journal_mod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(journal_mod, "_TRADE_DIR", tmp_path / "trades")

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "pipeline_state_combo_v2_exp.json").write_text(json.dumps({
        "freeze_state": {"active": True, "frozen_since": "2026-09-10"},
        "history": [{"strategy_diagnostics": {"freeze_multiplier": 0.0,
                                              "freeze_active": True}}],
    }))

    def rec(model):
        return TradeRecord(
            trade_id=f"{model}_t1", run_id=f"{model}_r1", model=model,
            timestamp="2026-09-12T14:30:00Z", symbol="NVDA", side="sell",
            action="rebalance_down", order_type="market",
            time_in_force="day", notional_usd=1000.0)

    TradeJournal("combo_v2_exp").log_trade(rec("combo_v2_exp"))
    rows = TradeJournal("combo_v2_exp").get_trades()
    assert len(rows) == 1
    assert rows[0]["freeze_active"] is True
    assert rows[0]["freeze_multiplier"] == 0.0   # frozen → 0, not None
    assert json.loads(rows[0]["freeze_state"])["active"] is True

    # Primary slot (no freeze keys in state, and here no state at all):
    # fields journal as None — rows unchanged, never blocked.
    TradeJournal("combo_v2").log_trade(rec("combo_v2"))
    prow = TradeJournal("combo_v2").get_trades()[0]
    assert prow["freeze_active"] is None
    assert prow["freeze_multiplier"] is None
    assert prow["freeze_state"] is None


# ───── dashboard: /api/compare + /v2 Compare panel ───────────────────────

@pytest.fixture(scope="module")
def client():
    import dashboard
    dashboard.app.config["TESTING"] = True
    with dashboard.app.test_client() as c:
        yield c


def test_api_compare_lists_all_registered_slots(client):
    import pipeline
    r = client.get("/api/compare")
    assert r.status_code == 200
    data = r.get_json()
    slots = {s["model"]: s for s in data["slots"]}
    assert set(slots) == set(pipeline.MODEL_REGISTRY)
    prim = slots["combo_v2"]
    assert prim["protocol_file"] == "protocol.json"
    # Pre-start: no tracking numbers, protocol NOT_STARTED — but the
    # payload shape is complete for the panel.
    for field in ("forward_test_start", "geo_monthly_pct",
                  "max_drawdown_pct", "live_calmar", "protocol_overall"):
        assert field in prim
    if "combo_v2_exp" in slots:
        assert slots["combo_v2_exp"]["protocol_file"] == "protocol_exp.json"
    if "combo_v2_process" in slots:
        assert (slots["combo_v2_process"]["protocol_file"]
                == "protocol_process.json")


def test_api_protocol_resolves_per_slot_files(client):
    import pipeline
    r = client.get("/api/protocol/combo_v2")
    assert r.status_code == 200
    assert r.get_json()["protocol_sha256_current"] == PRIMARY_PROTOCOL_SHA256
    if "combo_v2_exp" in pipeline.MODEL_REGISTRY:
        r2 = client.get("/api/protocol/combo_v2_exp")
        assert r2.status_code == 200
        d2 = r2.get_json()
        assert d2["protocol_sha256_current"] == EXP["_sha256"]
        assert d2["protocol_version"] == "exp-1.0"
        assert {c["id"] for c in d2["criteria"]} >= {"XK1", "X12-REL-DD"}


def test_v2_page_has_compare_panel_via_auth_helper(client):
    html = client.get("/v2").get_data(as_text=True)
    assert 'id="compare-panel"' in html
    assert "apiGet(`/api/compare`)" in html
    assert "renderCompare" in html


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
