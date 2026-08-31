"""Config-vs-spec invariant checker (2026-08-30).

Five times now a setting LOOKED set but was never read, or was read with
a different value than the operator intended:
  1. tier scale base (f × equity instead of f × day-start book),
  2. slot target_leverage=1 while the spec said 2 (exp ran 4 days at
     half leverage),
  3. exec_* slot-JSON fields silently ignored by config loading (the
     limit-order flip would have no-op'd),
  4. protocol files missing from the Docker image (sha binding broke),
  5. MIN_STOCKS_REQUIRED sitting exactly at the normal operating point
     (the Aug 18-28 silent outage).

This module compares the RUNNING system — the loaded ModelConfigs from
the volume's model_config.json, the module constants actually imported
by the trading code, the protocol files on disk, the bound shas in slot
state — against `spec_manifest.json`, the committed statement of what
the operator intends to be running. Any mismatch is a loud failure:
logged CRITICAL, written to the volume (invariants_status.json), shown
in /api/status, and turning /ready into a 503.

Failures never halt trading: the Aug outage showed that guards which
stop the pipeline do their own damage. The checker's job is to make
divergence IMPOSSIBLE TO MISS, not to trade on the operator's behalf.

Run points: pipeline start (core.runner.run_pipeline), the dashboard's
/api/invariants endpoint, and `python -m core.invariants` in CI/tests.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

_DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
STATUS_PATH = _DATA_DIR / "invariants_status.json"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "spec_manifest.json"

# ModelConfig fields the manifest may pin per slot. A field absent from
# the manifest entry is simply not checked.
_SLOT_FIELDS = (
    "target_leverage",
    "exec_style",
    "exec_limit_buffer_bps",
    "exec_fill_timeout_s",
    "exec_timeout_action",
    "enable_cutloss",
    "cutloss_portfolio_stop",
    "cutloss_scale_by_leverage",
)


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def run_invariant_checks(logger=None) -> dict:
    """Run every check. Returns the status dict (also written to the
    volume). Never raises."""
    failures: list[str] = []
    warnings: list[str] = []
    checker_errors: list[str] = []
    n_checks = 0

    def fail(msg: str) -> None:
        failures.append(msg)

    def warn(msg: str) -> None:
        warnings.append(msg)

    def broke(msg: str) -> None:
        # The CHECKER could not run a check — that is a degraded sweep,
        # not a proven spec divergence. Kept out of `failures` so a
        # transient introspection error can never stick /ready at 503
        # (2026-08-31 audit).
        checker_errors.append(msg)

    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except Exception as e:
        result = {
            "ts": datetime.now().isoformat(),
            "pass": False,
            "failures": [f"spec_manifest.json unreadable: {e}"],
            "warnings": [],
            "n_checks": 1,
        }
        _persist(result, logger)
        return result

    # ── Slot expectations vs the loaded ModelConfigs ────────────────────
    try:
        from core.config import get_active_models
        active = {mc.name: mc for mc in get_active_models()}
    except Exception as e:
        active = {}
        broke(f"get_active_models() failed: {e}")

    for slot_name, expect in (manifest.get("slots") or {}).items():
        n_checks += 1
        mc = active.get(slot_name)
        if mc is None:
            if expect.get("enabled", True):
                fail(f"slot '{slot_name}': expected ACTIVE but is not "
                     f"(disabled, missing from model_config.json, or its "
                     f"API keys are unset)")
            continue
        if not expect.get("enabled", True):
            fail(f"slot '{slot_name}': expected DISABLED but is active")
            continue
        for field in _SLOT_FIELDS:
            if field not in expect:
                continue
            n_checks += 1
            actual = getattr(mc, field, None)
            wanted = expect[field]
            if isinstance(wanted, float):
                ok = actual is not None and abs(float(actual) - wanted) < 1e-9
            else:
                ok = actual == wanted
            if not ok:
                fail(f"slot '{slot_name}'.{field}: running with "
                     f"{actual!r}, spec says {wanted!r}")
        n_checks += 1
        if not Path(mc.model_path).exists():
            fail(f"slot '{slot_name}': model bundle missing at "
                 f"{mc.model_path}")
        # Gate enablement — the one config the whole gate-cadence layer
        # keys on, previously unchecked (2026-08-31 audit). Lives in the
        # bundle's combo_config, so this loads the pickle (seconds).
        if "gated" in expect and Path(mc.model_path).exists():
            n_checks += 1
            try:
                from core.gate_update import config_has_gates
                from core.runner import load_model_bundle
                bundle = load_model_bundle(mc.model_path)
                actually_gated = config_has_gates(bundle.get("combo_config"))
                if actually_gated != bool(expect["gated"]):
                    fail(f"slot '{slot_name}': bundle gates are "
                         f"{'ON' if actually_gated else 'OFF'}, spec says "
                         f"{'ON' if expect['gated'] else 'OFF'}")
            except Exception as e:
                broke(f"slot '{slot_name}': gate enablement check "
                      f"errored: {e}")
        # Protocol binding: once the forward test started, the protocol
        # file's bytes must still hash to the sha bound at first funded
        # rebalance (edits after start void the pre-registration).
        try:
            from core.protocol import protocol_path_for
            from core.state import load_state, state_lock
            with state_lock:
                st = load_state(mc)
            bound = st.get("protocol_sha256")
            if st.get("forward_test_start") and not bound:
                # The exact July failure mode this check exists for: the
                # clock started but no sha ever bound. Silently skipping
                # it here defeated the check (2026-08-31 audit).
                n_checks += 1
                fail(f"slot '{slot_name}': forward test started "
                     f"{st['forward_test_start']} but NO protocol sha256 "
                     f"is bound")
            elif st.get("forward_test_start") and bound:
                n_checks += 1
                ppath = protocol_path_for(slot_name)
                actual_sha = _sha256_file(ppath)
                if actual_sha is None:
                    fail(f"slot '{slot_name}': protocol file unreadable "
                         f"at {ppath}")
                elif actual_sha != bound:
                    fail(f"slot '{slot_name}': protocol file sha "
                         f"{actual_sha[:12]} != bound {bound[:12]} "
                         f"(MODIFIED_AFTER_START)")
        except Exception as e:
            broke(f"slot '{slot_name}': protocol binding check errored: {e}")

    extra = set(active) - set(manifest.get("slots") or {})
    for name in sorted(extra):
        warn(f"slot '{name}' is active but not in spec_manifest.json")

    # ── Runtime constants actually imported by the trading code ─────────
    runtime = manifest.get("runtime") or {}

    if "min_stocks_required" in runtime:
        n_checks += 1
        try:
            from core.runner import MIN_STOCKS_REQUIRED
            wanted = int(runtime["min_stocks_required"])
            if MIN_STOCKS_REQUIRED != wanted:
                if os.environ.get("MIN_STOCKS_REQUIRED"):
                    warn(f"MIN_STOCKS_REQUIRED={MIN_STOCKS_REQUIRED} via env "
                         f"override (spec default {wanted})")
                else:
                    fail(f"MIN_STOCKS_REQUIRED is {MIN_STOCKS_REQUIRED}, "
                         f"spec says {wanted}")
        except Exception as e:
            broke(f"MIN_STOCKS_REQUIRED check errored: {e}")

    n_checks += 1
    try:
        from core.runner import MIN_STOCKS_REQUIRED as _min_req
        from core.universe_monitor import WARN_THRESHOLD
        if WARN_THRESHOLD <= _min_req:
            fail(f"UNIVERSE_WARN_THRESHOLD ({WARN_THRESHOLD}) must sit "
                 f"ABOVE MIN_STOCKS_REQUIRED ({_min_req}) to warn before "
                 f"the hard abort")
    except Exception as e:
        warn(f"universe threshold ordering check errored: {e}")

    if "tier_fractions" in runtime:
        n_checks += 1
        try:
            from core.risk import TIER_FRACTIONS
            wanted = {int(k): float(v)
                      for k, v in runtime["tier_fractions"].items()}
            if {int(k): float(v) for k, v in TIER_FRACTIONS.items()} != wanted:
                fail(f"TIER_FRACTIONS is {TIER_FRACTIONS}, spec says {wanted}")
        except ImportError:
            fail("core.risk.TIER_FRACTIONS constant is missing")
        except Exception as e:
            broke(f"tier fraction check errored: {e}")

    n_checks += 1
    try:
        from core.gate_update import GATE_DAILY_UPDATE, GATE_UPDATE_BAND
        band_max = float(runtime.get("gate_update_band_max", 0.5))
        if not (0.0 < GATE_UPDATE_BAND <= band_max):
            fail(f"GATE_UPDATE_BAND={GATE_UPDATE_BAND} outside "
                 f"(0, {band_max}]")
        if runtime.get("gate_daily_update", True) and not GATE_DAILY_UPDATE:
            warn("daily gate updates DISABLED by env (spec expects on)")
    except Exception as e:
        warn(f"gate knob check errored: {e}")

    result = {
        "ts": datetime.now().isoformat(),
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
        # Checks the checker itself could not run — a degraded sweep, not
        # a proven divergence; /ready stays 200 on these.
        "checker_errors": checker_errors,
        "degraded": bool(checker_errors),
        "n_checks": n_checks,
    }
    _persist(result, logger)
    return result


def _persist(result: dict, logger=None) -> None:
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_PATH.with_name(STATUS_PATH.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=1))
        os.replace(tmp, STATUS_PATH)
    except Exception:
        pass
    if logger is not None:
        if result["failures"]:
            for f in result["failures"]:
                logger.critical(f"[INVARIANT VIOLATION] {f}")
        for e in result.get("checker_errors", []):
            logger.error(f"[invariant DEGRADED] {e}")
        for w in result["warnings"]:
            logger.warning(f"[invariant] {w}")
        if result["pass"]:
            logger.info(f"[invariant] all {result['n_checks']} checks passed"
                        + (f" ({len(result['warnings'])} warnings)"
                           if result["warnings"] else "")
                        + (f" ({len(result['checker_errors'])} checks "
                           f"DEGRADED)" if result.get("checker_errors")
                           else ""))


def load_last_status() -> dict | None:
    """Latest persisted result, for /api/status and /ready. Never raises."""
    try:
        if STATUS_PATH.exists():
            return json.loads(STATUS_PATH.read_text())
    except Exception:
        pass
    return None


if __name__ == "__main__":
    import sys
    res = run_invariant_checks()
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["pass"] else 1)
