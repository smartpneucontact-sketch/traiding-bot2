"""Unit tests for the Phase A0 pre-registration pieces.

Covers:
  - protocol.json parses, criteria fields validate, key thresholds match
    the committed protocol (guards silent transcription drift)
  - file_sha256: stable across reads, matches hashlib directly, and a
    doctored copy of protocol.json yields a different hash (the tamper
    check the MODIFIED_AFTER_START flag depends on)
  - load_protocol rejects malformed criteria loudly
  - sleeve diagnostics in ComboStrategy.compute_weights: sleeves keys
    present, per-sleeve contributions sum to the pre-gate blend, and the
    combined output is identical to the pre-change math (expected weights
    computed inline from the untouched sleeve/gate helpers)

Run:  /opt/anaconda3/bin/python3 -m pytest tests/test_protocol.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

# Must be set before any core.* import — modules resolve DATA_DIR at import time.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="combo_test_protocol_"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.protocol import (
    ALLOWED_CHECKPOINTS, ALLOWED_OPS, PROTOCOL_JSON_PATH,
    REQUIRED_CRITERION_FIELDS, file_sha256, load_protocol,
)


# ───── protocol.json parsing + validation ────────────────────────────────

def test_protocol_json_parses_and_validates():
    proto = load_protocol()
    assert proto["protocol_version"] == "1.0"
    assert proto["amended"] is False
    assert proto["reference"]["geo_monthly_pct"] == 3.68
    assert proto["reference"]["sigma_monthly_pct"] == 15.7
    assert proto["power_disclosure"]["se_mean_monthly_pp"] == {
        "3mo": 9.1, "6mo": 6.4, "12mo": 4.5,
    }
    # Computed hash of the exact bytes read, exposed for the A2 scorer.
    assert proto["_sha256"] == file_sha256(PROTOCOL_JSON_PATH)


def test_criteria_fields_validate():
    proto = load_protocol()
    criteria = proto["criteria"]
    ids = [c["id"] for c in criteria]
    assert len(ids) == len(set(ids)), "criterion ids must be unique"
    for crit in criteria:
        for field in REQUIRED_CRITERION_FIELDS:
            assert field in crit, f"{crit['id']}: missing {field}"
        assert crit["op"] in ALLOWED_OPS
        assert crit["checkpoint"] in ALLOWED_CHECKPOINTS
        if crit["op"] == "between":
            lo, hi = crit["threshold"]
            assert lo <= hi
        else:
            assert isinstance(crit["threshold"], (int, float))

    # Spot-check the committed numbers (guards transcription drift).
    by_id = {c["id"]: c for c in criteria}
    assert by_id["K1"]["threshold"] == -45 and by_id["K1"]["action"] == "KILL"
    assert by_id["K2"]["threshold"] == 25
    assert by_id["K3a"]["threshold"] == 90 and by_id["K3b"]["threshold"] == 15
    assert by_id["M3-VOL"]["threshold"] == [7, 25]
    assert by_id["M6-BAND"]["threshold"] == pytest.approx(-0.2754)
    assert len(by_id["M6-BAND"]["and_conditions"]) == 2
    assert by_id["M12-GEO"]["threshold"] == 1.8
    assert by_id["M12-DD"]["threshold"] == -45
    assert by_id["M12-SLIP"]["threshold"] == 15
    assert by_id["M12-VOL"]["threshold"] == 25
    assert by_id["M12-FAIL"]["action"] == "FAIL"


def test_load_protocol_rejects_malformed_criteria(tmp_path):
    proto = json.loads(PROTOCOL_JSON_PATH.read_text())

    # Missing required field
    bad = json.loads(json.dumps(proto))
    del bad["criteria"][0]["threshold"]
    p = tmp_path / "missing_field.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing required fields"):
        load_protocol(p)

    # Unknown op
    bad = json.loads(json.dumps(proto))
    bad["criteria"][0]["op"] = "!="
    p = tmp_path / "bad_op.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="unknown op"):
        load_protocol(p)

    # Empty criteria list
    bad = json.loads(json.dumps(proto))
    bad["criteria"] = []
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="non-empty"):
        load_protocol(p)


# ───── file_sha256: stability + tamper detection ─────────────────────────

def test_file_sha256_stable_and_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    payload = b"combo_v2 forward test\n" * 1000
    p.write_bytes(payload)
    assert file_sha256(p) == file_sha256(p)
    assert file_sha256(p) == hashlib.sha256(payload).hexdigest()


def test_doctored_protocol_copy_yields_different_hash(tmp_path):
    """The tamper test: loosening one threshold must change the hash the
    runner bound at forward_test_start."""
    original_hash = file_sha256(PROTOCOL_JSON_PATH)
    doctored = json.loads(PROTOCOL_JSON_PATH.read_text())
    for crit in doctored["criteria"]:
        if crit["id"] == "M12-GEO":
            crit["threshold"] = 0.5  # quietly lower the success bar
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(doctored, indent=2))
    assert file_sha256(p) != original_hash
    # Still schema-valid — only the HASH catches this kind of tampering.
    load_protocol(p)


# ───── sleeve diagnostics in compute_weights ─────────────────────────────

def _synthetic_market(n_stocks: int = 35, with_vix: bool = True):
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    stock_data = {}
    for i in range(n_stocks):
        drift = 0.0005 + 0.002 * (i / n_stocks)
        px = 100 * np.exp(np.cumsum(rng.normal(drift, 0.02, len(idx))))
        stock_data[f"S{i:02d}"] = pd.DataFrame({"close": px}, index=idx)
    spy = pd.DataFrame(
        {"close": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))},
        index=idx)
    macro = {"SPY": spy}
    if with_vix:
        macro["VIX"] = pd.DataFrame(
            {"close": rng.uniform(12, 35, len(idx))}, index=idx)
    return stock_data, macro


def _expected_combined_pre_change(stock_data, macro, c):
    """The exact pre-change compute_weights math, rebuilt inline from the
    (untouched) sleeve and gate helpers. Any drift in the blend/gate/cap
    pipeline shows up as a mismatch against this reference."""
    from core.combo_strategy import (
        _adaptive_voltarget_momentum_weights, _book_drawdown,
        _dual_momentum_voltarget_weights, _linear_dd_ramp,
        _spy_drawdown_gate, _to_close_panel, _xs_momentum_weights,
    )
    stock_px = _to_close_panel(stock_data)
    macro_px = _to_close_panel(macro)
    w_xs = _xs_momentum_weights(stock_px, n_long=c.xs_mom_top_n)
    w_dual = _dual_momentum_voltarget_weights(
        stock_px, n_long=c.dual_mom_top_n, target_vol=c.dual_mom_vol_target)
    w_adapt = _adaptive_voltarget_momentum_weights(
        stock_px, macro_px, n_long=c.adaptive_top_n,
        calm_leverage=c.adaptive_calm_leverage,
        neutral_leverage=c.adaptive_neutral_leverage,
        stress_leverage=c.adaptive_stress_leverage,
    )
    combined: dict[str, float] = {}
    for sleeve, sw in ((w_xs, c.xs_mom_weight),
                       (w_dual, c.dual_mom_weight),
                       (w_adapt, c.adaptive_weight)):
        for sym, w in sleeve.items():
            combined[sym] = combined.get(sym, 0.0) + sw * w
    pre_gate = dict(combined)

    spy_gate = (_spy_drawdown_gate(macro_px, lookback=c.spy_dd_lookback,
                                   full_dd=c.spy_full_dd, cash_dd=c.spy_cash_dd)
                if c.enable_spy_dd_gate else 1.0)
    book_gate = None
    if c.enable_book_dd_gate:
        bdd = _book_drawdown(stock_px, combined, lookback=c.book_dd_lookback)
        if bdd is not None:
            book_gate = _linear_dd_ramp(bdd, c.book_full_dd, c.book_cash_dd)
    exposure = spy_gate if book_gate is None else min(spy_gate, book_gate)
    combined = {s: w * exposure for s, w in combined.items()}
    gross = sum(abs(w) for w in combined.values())
    if gross > c.max_gross_exposure:
        scale = c.max_gross_exposure / gross
        combined = {s: w * scale for s, w in combined.items()}
    final = {s: w for s, w in combined.items()
             if abs(w) >= c.min_position_weight}
    return final, pre_gate


def test_sleeve_diagnostics_sum_to_blend_and_weights_unchanged():
    from core.combo_strategy import ComboConfig, ComboStrategy

    stock_data, macro = _synthetic_market()
    cfg = ComboConfig(xs_mom_top_n=10, dual_mom_top_n=10, adaptive_top_n=10)
    strat = ComboStrategy(cfg)
    weights = strat.compute_weights(stock_data, macro)
    assert weights, "synthetic panel must produce a book"

    # 1) Combined output identical to the pre-change math.
    expected, pre_gate = _expected_combined_pre_change(stock_data, macro, cfg)
    assert set(weights) == set(expected)
    for sym, w in weights.items():
        assert w == pytest.approx(expected[sym], abs=1e-12), sym

    diag = strat.last_diagnostics
    sleeves = diag["sleeves"]
    assert set(sleeves) == {"xs_momentum", "dual_momentum", "adaptive"}
    assert all(sleeves[name] for name in sleeves), "each sleeve holds names"
    assert all(v != 0.0 for contrib in sleeves.values()
               for v in contrib.values())

    # 2) Per-symbol contributions sum to the (pre-gate) blend. Tolerance
    #    covers the 6dp rounding of up to 3 sleeve entries per symbol.
    summed: dict[str, float] = {}
    for contrib in sleeves.values():
        for sym, v in contrib.items():
            summed[sym] = summed.get(sym, 0.0) + v
    assert set(summed) == set(pre_gate)
    for sym, v in summed.items():
        assert v == pytest.approx(pre_gate[sym], abs=2e-6), sym
    assert diag["gross_pre_gate"] == pytest.approx(
        sum(abs(v) for v in summed.values()), abs=1e-3)

    # 3) sleeve_gross = sum of abs contributions per sleeve.
    for name, contrib in sleeves.items():
        assert diag["sleeve_gross"][name] == pytest.approx(
            sum(abs(v) for v in contrib.values()), abs=1e-9)

    # 4) Adaptive VIX leverage is one of the three configured regimes.
    assert diag["adaptive_vix_leverage"] in (
        cfg.adaptive_calm_leverage,
        cfg.adaptive_neutral_leverage,
        cfg.adaptive_stress_leverage,
    )


def test_adaptive_vix_leverage_neutral_without_vix_data():
    """No VIX column → the sleeve trades at neutral leverage and reports it."""
    from core.combo_strategy import ComboConfig, ComboStrategy

    stock_data, macro = _synthetic_market(with_vix=False)
    cfg = ComboConfig(xs_mom_top_n=10, dual_mom_top_n=10, adaptive_top_n=10)
    strat = ComboStrategy(cfg)
    weights = strat.compute_weights(stock_data, macro)
    assert weights
    assert strat.last_diagnostics["adaptive_vix_leverage"] == pytest.approx(
        cfg.adaptive_neutral_leverage)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
