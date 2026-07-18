"""Review findings R3 — regression tests.

  F1 (MAJOR): core/runner.py bound state["protocol_sha256"] to the PRIMARY
      protocol.json for EVERY slot at the first funded rebalance, while
      scoring compares against the slot's OWN protocol file
      (core/protocol.py load_protocol_for). A shadow slot would bind the
      primary's sha and be permanently flagged MODIFIED_AFTER_START. The
      runner now binds via the per-slot helper bind_forward_test_start;
      the primary's semantics stay byte-identical (same file, same sha).

  F2 (MAJOR): scripts/build_variant.py _build_candidate_x never set
      residual_sector_map, so the built bundle carried None and the
      residual sleeve silently degraded beta_mode="market_sector" to
      market-only betas — an unledgered construction. The builder now
      embeds the research sector map (data_cache.pkl 'sector_map', via
      --sector-map-from) and REFUSES to build with a missing/empty map.

  F3 (minor): core/journal.py's freeze-state capture had no publisher —
      nothing wrote freeze_multiplier / freeze_active into diagnostics.
      The runner's freeze block now publishes the applied multiplier into
      the rebalance strategy_diagnostics (report + state.history) and
      into the persisted freeze_state dict, and persists state BEFORE the
      freeze liquidation so journal rows capture the active posture.

Run:  /opt/anaconda3/bin/python -m pytest tests/test_review3_findings.py -q
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_data_"))
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pytest

from core.protocol import (
    PROTOCOL_EXP_JSON_PATH,
    PROTOCOL_JSON_PATH,
    PROTOCOL_PROCESS_JSON_PATH,
    bind_forward_test_start,
    file_sha256,
    load_protocol_for,
    score_protocol,
)

RUNNER_SRC = (REPO / "core" / "runner.py").read_text()
BUILD = REPO / "scripts" / "build_variant.py"

FULL_REFERENCE = {
    "family": "ASM_rescore600",
    "source_ledger": "results/v7/trials/ASM_rescore600.csv",
    "window": "dev",
    "tc_bps_per_side": 5,
    "margin_bps_annual": 600,
    "leverage_for_reference": 2.0,
    "engine": "tier_loop_cc",
    "mean_monthly_return": 0.0396,
    "sharpe": 1.16,
    "max_drawdown": -0.421,
    "calmar": 1.10,
}


def _run_build(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILD), *args],
        capture_output=True, text=True, cwd=str(REPO), timeout=120)


# ═══════════════════════════════════════════════════════════════════════════
# F1 — per-slot protocol sha binding at the first funded rebalance
# ═══════════════════════════════════════════════════════════════════════════

def test_runner_uses_per_slot_binding_helper():
    """The runner must bind through bind_forward_test_start (per-slot
    file), never through an unconditional hash of the primary
    protocol.json."""
    assert "bind_forward_test_start(state, mc.name, today_iso)" in RUNNER_SRC
    assert "file_sha256(PROTOCOL_JSON_PATH)" not in RUNNER_SRC, (
        "runner still binds the PRIMARY protocol.json sha unconditionally "
        "— a shadow slot would be permanently MODIFIED_AFTER_START"
    )


def test_primary_binding_semantics_byte_identical():
    """The primary's recorded-sha semantics are untouched: the helper
    hashes exactly PROTOCOL_JSON_PATH — the same file, same sha the old
    inline runner code recorded — and scores 'intact' against the
    primary's own protocol."""
    st: dict = {}
    assert bind_forward_test_start(st, "combo_v2", "2026-07-18")
    assert st["forward_test_start"] == "2026-07-18"
    assert st["protocol_sha256"] == file_sha256(PROTOCOL_JSON_PATH)

    proto = load_protocol_for("combo_v2")
    assert st["protocol_sha256"] == proto["_sha256"]
    scored = score_protocol(None, None, None, st, proto)
    assert scored["binding"] == "intact"


def test_shadow_slot_binds_its_own_protocol_sha():
    """A shadow slot binds ITS OWN file's sha (what scoring compares
    against), not the primary's. Under the pre-fix behavior (primary sha
    bound for every slot) the exp slot would score MODIFIED_AFTER_START
    forever."""
    st: dict = {}
    assert bind_forward_test_start(st, "combo_v2_exp", "2026-07-18")
    assert st["protocol_sha256"] == file_sha256(PROTOCOL_EXP_JSON_PATH)
    assert st["protocol_sha256"] != file_sha256(PROTOCOL_JSON_PATH)

    proto_exp = load_protocol_for("combo_v2_exp")
    assert score_protocol(None, None, None, st, proto_exp)["binding"] \
        == "intact"

    # The exact failure mode the fix removes: primary sha vs exp protocol.
    bad = {"forward_test_start": "2026-07-18",
           "protocol_sha256": file_sha256(PROTOCOL_JSON_PATH)}
    assert score_protocol(None, None, None, bad, proto_exp)["binding"] \
        == "MODIFIED_AFTER_START"

    # Versioned process bundle names resolve to the process protocol too.
    st_proc: dict = {}
    assert bind_forward_test_start(
        st_proc, "combo_v2_process.2026.ab12cd34", "2026-07-18")
    assert st_proc["protocol_sha256"] == file_sha256(
        PROTOCOL_PROCESS_JSON_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# F2 — candidate_x must embed the research sector map (or refuse to build)
# ═══════════════════════════════════════════════════════════════════════════

def test_candidate_x_refuses_empty_sector_map(tmp_path):
    empty = tmp_path / "cache_empty.pkl"
    empty.write_bytes(pickle.dumps({"sector_map": {}}))
    r = _run_build("--variant", "candidate_x", "--dry",
                   "--sector-map-from", str(empty))
    assert r.returncode == 2, r.stdout
    assert "sector_map" in r.stderr
    assert "REFUSING" in r.stderr or "refus" in r.stderr.lower()

    # Same refusal when the key is missing entirely.
    nokey = tmp_path / "cache_nokey.pkl"
    nokey.write_bytes(pickle.dumps({"panel": {}}))
    r = _run_build("--variant", "candidate_x", "--dry",
                   "--sector-map-from", str(nokey))
    assert r.returncode == 2
    assert "sector_map" in r.stderr


def test_candidate_x_refuses_missing_or_wrong_schema_map(tmp_path):
    r = _run_build("--variant", "candidate_x", "--dry",
                   "--sector-map-from", str(tmp_path / "nope.pkl"))
    assert r.returncode == 2
    assert "not found" in r.stderr

    # A map whose sector names cannot resolve via SECTOR_TO_ETF is a
    # wrong-schema map: market_sector mode would be inert.
    bad = tmp_path / "cache_bad.pkl"
    bad.write_bytes(pickle.dumps(
        {"sector_map": {"AAPL": "TECH", "JPM": "FIN"}}))
    r = _run_build("--variant", "candidate_x", "--dry",
                   "--sector-map-from", str(bad))
    assert r.returncode == 2
    assert "SECTOR_TO_ETF" in r.stderr


def test_candidate_x_builds_and_embeds_injected_map(tmp_path):
    """Non-dry build to a tmp path with a small valid map: the bundle's
    ComboConfig must carry the injected map verbatim."""
    cache = tmp_path / "cache.pkl"
    sector_map = {"AAPL": "Technology", "JPM": "Financial Services",
                  "XOM": "Energy"}
    cache.write_bytes(pickle.dumps({"sector_map": sector_map}))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(FULL_REFERENCE))
    out = tmp_path / "bundle" / "model.pkl"
    r = _run_build("--variant", "candidate_x",
                   "--reference-json", str(ref),
                   "--sector-map-from", str(cache),
                   "--out", str(out))
    assert r.returncode == 0, r.stderr
    bundle = pickle.loads(out.read_bytes())
    cfg = bundle["combo_config"]
    assert cfg.residual_sector_map == sector_map
    assert cfg.residual_beta_mode == "market_sector"
    assert cfg.residual_scale_mode == "raw"
    assert abs(cfg.residual_weight - 1.0 / 3.0) < 1e-12


def test_sector_map_applies_only_to_candidate_x(tmp_path):
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({
        "year": 2026,
        "picks": ["dual_momentum_vol", "xs_momentum_top30",
                  "adaptive_voltarget_momentum"],
        "spec_sha256": "deadbeef"}))
    cache = tmp_path / "cache.pkl"
    cache.write_bytes(pickle.dumps({"sector_map": {"AAPL": "Technology"}}))
    r = _run_build("--variant", "process", "--selection-json", str(sel),
                   "--sector-map-from", str(cache), "--dry")
    assert r.returncode == 2
    assert "candidate_x" in r.stderr


def test_deployed_exp_bundle_carries_resolvable_sector_map():
    """The DEPLOYED combo_v2_exp bundle (rebuilt after this fix) must ship
    a non-empty ticker→sector map whose sectors resolve to the SPDR ETFs
    the bot's macro panel carries — otherwise the residual sleeve trades
    market-only betas, which no ledger row describes."""
    from core.combo_strategy import SECTOR_TO_ETF
    from core.runner import load_model_bundle

    bundle = load_model_bundle(REPO / "model" / "combo_v2_exp" / "model.pkl")
    cfg = bundle["combo_config"]
    sm = cfg.residual_sector_map
    assert isinstance(sm, dict) and len(sm) > 0, (
        "combo_v2_exp bundle has no residual_sector_map — market_sector "
        "betas silently degrade to market-only (unledgered construction)"
    )
    # Spot-check: a known ticker maps to a sector that resolves to an ETF.
    assert SECTOR_TO_ETF.get(sm.get("AAPL")) == "XLK"
    resolvable = sum(1 for s in sm.values() if s in SECTOR_TO_ETF)
    assert resolvable / len(sm) > 0.9, (
        f"only {resolvable}/{len(sm)} tickers resolve via SECTOR_TO_ETF — "
        "wrong sector-name schema in the bundled map"
    )
    # The strategy object inside the bundle sees the same map.
    assert bundle["model"].config.residual_sector_map == sm


# ═══════════════════════════════════════════════════════════════════════════
# F3 — freeze posture publisher (ablation replay)
# ═══════════════════════════════════════════════════════════════════════════

def test_runner_publishes_freeze_diag_into_diagnostics_and_state():
    """Source contract: the freeze block builds freeze_diag
    (freeze_active + applied multiplier), embeds the multiplier into the
    persisted freeze_state dict, and merges freeze_diag into the
    diagnostics at BOTH publish sites (report + state.history)."""
    assert '"freeze_active": bool(is_frozen)' in RUNNER_SRC
    assert '"freeze_multiplier": 0.0 if is_frozen else 1.0' in RUNNER_SRC
    assert 'new_freeze["freeze_multiplier"]' in RUNNER_SRC
    # Two merge sites: report/strategy_diagnostics and history entry.
    assert RUNNER_SRC.count("{**diagnostics, **freeze_diag}") == 2


def test_runner_persists_freeze_state_before_liquidation():
    """The journal's freeze capture reads the slot's ON-DISK state, so the
    frozen path must write state before liquidating — otherwise the
    liquidation rows carry the stale pre-freeze posture."""
    start = RUNNER_SRC.index("freeze_diag = {")
    end = RUNNER_SRC.index('reason="freeze_liquidation"')
    assert start < end
    assert "_save_state_merged(state, mc)" in RUNNER_SRC[start:end], (
        "no state persist between the freeze evaluation and the freeze "
        "liquidation — journal rows would miss the active freeze"
    )


def test_freeze_context_prefers_fresh_freeze_state_multiplier():
    """freeze_state (updated on EVERY run, including frozen early-return
    runs that write no history) outranks the last rebalance's history
    diagnostics; history remains the fallback for older states."""
    from core.journal import freeze_context_from_state

    state = {
        "freeze_state": {"active": True, "since_iso": "2026-09-10",
                         "freeze_multiplier": 0.0},
        "history": [{"date": "2026-08-24", "strategy_diagnostics":
                     {"freeze_multiplier": 1.0, "freeze_active": False}}],
    }
    ctx = freeze_context_from_state(state)
    assert ctx["freeze_active"] is True
    assert ctx["freeze_multiplier"] == 0.0          # freeze_state wins
    assert json.loads(ctx["freeze_state"])["freeze_multiplier"] == 0.0

    # Fallback: pre-fix states without the multiplier in freeze_state
    # still read the newest history diagnostics.
    legacy = {
        "freeze_state": {"active": False, "since_iso": None},
        "history": [{"strategy_diagnostics": {"freeze_multiplier": 1.0,
                                              "freeze_active": False}}],
    }
    assert freeze_context_from_state(legacy)["freeze_multiplier"] == 1.0


def test_published_freeze_posture_reaches_journal_rows(tmp_path, monkeypatch):
    """End-to-end through the journal: a state shaped exactly like the
    runner now writes it (freeze_state with embedded multiplier; history
    diagnostics with the merged freeze keys) auto-fills every journalled
    order row for the exp slot; the frozen primary stays all-None."""
    import core.journal as journal_mod
    from core.journal import TradeJournal, TradeRecord
    monkeypatch.setattr(journal_mod, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(journal_mod, "_TRADE_DIR", tmp_path / "trades")

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    # Frozen run: runner wrote freeze_state (multiplier 0.0) and returned
    # early — NO fresh history entry; the last one predates the freeze.
    (state_dir / "pipeline_state_combo_v2_exp.json").write_text(json.dumps({
        "freeze_state": {"active": True, "since_iso": "2026-09-10",
                         "freeze_multiplier": 0.0},
        "history": [{"date": "2026-08-24", "strategy_diagnostics":
                     {"freeze_multiplier": 1.0, "freeze_active": False,
                      "book_gate": 0.95}}],
    }))
    # Primary slot: no freeze machinery, state has neither key.
    (state_dir / "pipeline_state_combo_v2.json").write_text(json.dumps({
        "history": [{"strategy_diagnostics": {"book_gate": 1.0}}],
    }))

    def rec(model):
        return TradeRecord(
            trade_id=f"{model}_t1", run_id=f"{model}_r1", model=model,
            timestamp="2026-09-12T14:30:00Z", symbol="NVDA", side="sell",
            action="exit_position", order_type="market",
            time_in_force="day", notional_usd=1000.0)

    TradeJournal("combo_v2_exp").log_trade(rec("combo_v2_exp"))
    row = TradeJournal("combo_v2_exp").get_trades()[0]
    assert row["freeze_active"] is True
    assert row["freeze_multiplier"] == 0.0
    assert json.loads(row["freeze_state"])["freeze_multiplier"] == 0.0

    TradeJournal("combo_v2").log_trade(rec("combo_v2"))
    prow = TradeJournal("combo_v2").get_trades()[0]
    assert prow["freeze_active"] is None
    assert prow["freeze_multiplier"] is None
    assert prow["freeze_state"] is None
