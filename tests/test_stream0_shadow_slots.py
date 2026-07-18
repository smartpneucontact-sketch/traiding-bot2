"""Stream 0 tests — shadow-slot registry/migration (core/config.py) and
the parameterized bundle builder (scripts/build_variant.py).

Run:  /opt/anaconda3/bin/python -m pytest tests/test_stream0_shadow_slots.py -q
"""
from __future__ import annotations

import dataclasses
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

import core.config as cfg

BUILD = REPO / "scripts" / "build_variant.py"

FULL_REFERENCE = {
    "family": "ASM_rescore600",
    "source_ledger": "results/v7/trials/ASM_rescore600.csv",
    "window": "2016-04-01..2022-12-31",
    "tc_bps_per_side": 5,
    "margin_bps_annual": 600,
    "leverage_for_reference": 2.0,
    "engine": "engine_v2 next_open",
    "mean_monthly_return": 0.041,
    "sharpe": 1.1,
    "max_drawdown": -0.43,
    "calmar": 1.1,
}


def _residual_ported() -> bool:
    import core.combo_strategy as cs
    return "residual_weight" in {f.name for f in dataclasses.fields(cs.ComboConfig)}


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILD), *args],
        capture_output=True, text=True, cwd=str(REPO), timeout=120)


# ═══════════════════════════════════════════════════════════════════════════
# core/config.py — registry, descriptions, migration
# ═══════════════════════════════════════════════════════════════════════════

def test_registry_has_shadow_models():
    for name, model_dir in (("combo_v2_exp", "combo_v2_exp"),
                            ("combo_v2_process", "combo_v2_process")):
        assert name in cfg.MODEL_REGISTRY
        assert cfg.MODEL_REGISTRY[name]["model_dir"] == model_dir
        assert cfg.MODEL_REGISTRY[name]["feature_version"] == "combo"
    # frozen primary untouched
    assert cfg.MODEL_REGISTRY["combo_v2"] == {
        "feature_version": "combo", "model_dir": "combo_v2",
        "fallback_model": None}


def test_descriptions_are_honest():
    exp = cfg.MODEL_DESCRIPTIONS["combo_v2_exp"]
    text = json.dumps(exp)
    assert "DISPUTED" in text                  # freeze rides as disputed
    assert "disclosure" in text.lower()        # burned val = disclosure only
    assert "ASM_Bst_BTF1" in text
    proc = cfg.MODEL_DESCRIPTIONS["combo_v2_process"]
    ptext = json.dumps(proc)
    assert "never self-modifies" in ptext.lower() or "NEVER self-modifies" in ptext
    assert "wf_select" in ptext or "walk-forward" in ptext.lower()


def test_default_config_includes_disabled_shadow_slots():
    conf = cfg._default_config()
    models = {s["model"]: s for s in conf["slots"]}
    assert conf["shadow_slots"] == cfg.SHADOW_SLOTS_VERSION
    for name in ("combo_v2_exp", "combo_v2_process"):
        slot = models[name]
        assert slot["enabled"] is False
        assert slot["alpaca_key"] == "" and slot["alpaca_secret"] == ""
        assert slot["target_leverage"] == 2.0


def test_migration_appends_shadow_slots(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "model_config.json")
    legacy = {
        "slots": [{
            "slot_id": 1, "model": "combo_v2", "enabled": True,
            "alpaca_key": "K", "alpaca_secret": "S",
            "target_leverage": 2.0,
        }],
        "cutloss_calibration": cfg.CUTLOSS_CALIBRATION_VERSION,
    }
    out = cfg._ensure_shadow_slots(json.loads(json.dumps(legacy)))
    assert out["shadow_slots"] == cfg.SHADOW_SLOTS_VERSION
    models = [s["model"] for s in out["slots"]]
    assert models == ["combo_v2", "combo_v2_exp", "combo_v2_process"]
    # slot 1 untouched, appended slots disabled with empty keys
    assert out["slots"][0] == legacy["slots"][0]
    for s in out["slots"][1:]:
        assert s["enabled"] is False and s["alpaca_key"] == ""
    # persisted to disk
    on_disk = json.loads((tmp_path / "model_config.json").read_text())
    assert [s["model"] for s in on_disk["slots"]] == models


def test_migration_is_idempotent_and_respects_operator_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "model_config.json")
    # operator already added combo_v2_exp on slot 2 with keys + a slot_id-3
    # slot of a different model
    conf = {
        "slots": [
            {"slot_id": 1, "model": "combo_v2", "enabled": True,
             "alpaca_key": "K", "alpaca_secret": "S"},
            {"slot_id": 2, "model": "combo_v2_exp", "enabled": True,
             "alpaca_key": "K2", "alpaca_secret": "S2"},
            {"slot_id": 3, "model": "combo_v1", "enabled": False,
             "alpaca_key": "", "alpaca_secret": ""},
        ],
    }
    out = cfg._ensure_shadow_slots(conf)
    models = [s["model"] for s in out["slots"]]
    # exp NOT duplicated; process appended with a non-colliding slot_id
    assert models.count("combo_v2_exp") == 1
    assert models.count("combo_v2_process") == 1
    proc = next(s for s in out["slots"] if s["model"] == "combo_v2_process")
    assert proc["slot_id"] == 4
    # operator's exp keys untouched
    exp = next(s for s in out["slots"] if s["model"] == "combo_v2_exp")
    assert exp["alpaca_key"] == "K2" and exp["enabled"] is True
    # second run: stamped → returned unchanged
    before = json.dumps(out, sort_keys=True)
    again = cfg._ensure_shadow_slots(out)
    assert json.dumps(again, sort_keys=True) == before


def test_migration_is_failure_tolerant(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "model_config.json")

    def boom(_):
        raise OSError("disk full")

    monkeypatch.setattr(cfg, "save_model_config", boom)
    conf = {"slots": [{"slot_id": 1, "model": "combo_v2"}]}
    out = cfg._ensure_shadow_slots(conf)   # must not raise
    assert out is conf


# ═══════════════════════════════════════════════════════════════════════════
# scripts/build_variant.py — CLI behavior
# ═══════════════════════════════════════════════════════════════════════════

def test_candidate_x_dry():
    """--dry either builds the manifest (residual port landed) or exits
    loudly with the porting-prerequisite error (port not landed)."""
    r = _run("--variant", "candidate_x", "--dry")
    if _residual_ported():
        assert r.returncode == 0, r.stderr
        assert "combo_v2_exp.1_asm_bst_btf1" in r.stdout
        assert "DRY RUN" in r.stdout
        assert "enable_drawdown_freeze = True" in r.stdout
        assert "DISPUTED" in r.stdout
    else:
        assert r.returncode == 2
        assert "PORTING PREREQUISITE NOT MET" in r.stderr


def test_candidate_x_nondry_requires_reference():
    r = _run("--variant", "candidate_x")
    assert r.returncode == 2
    assert "--reference-json" in r.stderr


def test_reference_schema_is_validated(tmp_path):
    bad = tmp_path / "ref.json"
    bad.write_text(json.dumps({"family": "ASM_rescore600"}))
    r = _run("--variant", "candidate_x", "--dry",
             "--reference-json", str(bad))
    assert r.returncode == 2
    assert "missing required keys" in r.stderr
    assert "margin_bps_annual" in r.stderr


def test_refuses_primary_dir():
    r = _run("--variant", "candidate_x", "--dry",
             "--out", "model/combo_v2/model.pkl")
    assert r.returncode == 2
    assert "FROZEN primary" in r.stderr
    assert not any("combo_v2/model.pkl.tmp" in str(p)
                   for p in (REPO / "model" / "combo_v2").glob("*"))


def test_process_fails_loudly_on_unported_sleeve(tmp_path):
    sel = tmp_path / "selection_2026.json"
    sel.write_text(json.dumps({
        "year": 2026,
        "picks": ["dual_momentum_vol", "xs_momentum_top30",
                  "xs_momentum_12_1"],
        "spec_sha256": "deadbeef"}))
    r = _run("--variant", "process", "--selection-json", str(sel), "--dry")
    assert r.returncode == 2
    assert "PORTING PREREQUISITE NOT MET" in r.stderr
    assert "xs_momentum_12_1" in r.stderr


def test_process_requires_selection_and_schema(tmp_path):
    r = _run("--variant", "process", "--dry")
    assert r.returncode == 2 and "--selection-json" in r.stderr
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"year": 2026}))
    r = _run("--variant", "process", "--selection-json", str(sel), "--dry")
    assert r.returncode == 2 and "missing required keys" in r.stderr
    # wrong k
    sel.write_text(json.dumps({"year": 2026, "picks": ["xs_momentum_top30"],
                               "spec_sha256": "x"}))
    r = _run("--variant", "process", "--selection-json", str(sel), "--dry")
    assert r.returncode == 2 and "k=3" in r.stderr


def test_process_build_roundtrip(tmp_path):
    """Full non-dry process build to a tmp path: version format, injected
    reference, provenance schema, plain equal blend with overlays OFF."""
    sel = tmp_path / "selection_2026.json"
    sel.write_text(json.dumps({
        "year": 2026,
        "picks": ["dual_momentum_vol", "xs_momentum_top30",
                  "adaptive_voltarget_momentum"],
        "spec_sha256": "deadbeef", "sharpes": {"dual_momentum_vol": 1.2}}))
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(FULL_REFERENCE))
    out = tmp_path / "bundle" / "model.pkl"
    r = _run("--variant", "process", "--selection-json", str(sel),
             "--reference-json", str(ref), "--out", str(out))
    assert r.returncode == 0, r.stderr
    bundle = pickle.loads(out.read_bytes())

    import hashlib
    sha8 = hashlib.sha256(sel.read_bytes()).hexdigest()[:8]
    assert bundle["version"] == f"combo_v2_process.2026.{sha8}"
    assert bundle["strategy_type"] == "direct_weights"
    assert bundle["horizon"] == 21
    assert bundle["backtest_reference"]["family"] == "ASM_rescore600"
    assert bundle["backtest_reference"]["margin_bps_annual"] == 600
    prov = bundle["provenance"]
    for key in ("disputed_components", "burned_val_disclosure",
                "phi_stress_band", "survivorship"):
        assert key in prov          # common provenance schema, explicit
    assert prov["survivorship"]     # caveat always present
    assert prov["selection"]["picks"][0] == "dual_momentum_vol"
    c = bundle["combo_config"]
    assert abs(c.xs_mom_weight - 1 / 3) < 1e-12
    assert abs(c.dual_mom_weight - 1 / 3) < 1e-12
    assert abs(c.adaptive_weight - 1 / 3) < 1e-12
    assert c.enable_book_dd_gate is False
    assert c.enable_spy_dd_gate is False
    assert c.enable_drawdown_freeze is False


def test_build_combo_v2_untouched():
    """The primary builder must remain byte-identical to its frozen form —
    this script never edits it. Guard: it still contains no reference to
    build_variant and still targets model/combo_v2."""
    src = (REPO / "scripts" / "build_combo_v2.py").read_text()
    assert "build_variant" not in src
    assert '"combo_v2"' in src or "combo_v2" in src
    bv = (REPO / "scripts" / "build_variant.py").read_text()
    assert "import build_combo_v2" not in bv
    assert "from build_combo_v2" not in bv
