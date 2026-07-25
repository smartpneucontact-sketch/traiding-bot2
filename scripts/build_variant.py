"""Parameterized shadow-slot bundle builder (Roadmap v2, Streams 1 & 3).

Builds the NON-PRIMARY model bundles:

    --variant candidate_x   →  model/combo_v2_exp/model.pkl
                               (ASM_Bst_BTF1 — frozen construction, see
                               results/v7/candidate_x/DECISION_RULE.md)
    --variant process       →  model/combo_v2_process/model.pkl
                               (walk-forward annual re-selection, needs
                               --selection-json from research wf_select.py)

HARD RULES
  • scripts/build_combo_v2.py and model/combo_v2/ are the FROZEN primary.
    This script never imports the former and refuses to write into the
    latter.
  • backtest_reference numbers are INJECTED via --reference-json (the
    margin-600 re-score output, family ASM_rescore600) — never hardcoded
    here. A non-dry build without --reference-json fails loudly.
  • Per-variant builders are frozen constructions: changing a knob means a
    new version string and a dated amendment in the governing document,
    not an edit-in-place.
  • Unported sleeves fail the build loudly: porting + 1e-9 live-twin
    parity on the bot's data path is a deploy PREREQUISITE, never a
    build-time shortcut.

Usage (from the repo root, /opt/anaconda3/bin/python):

    python scripts/build_variant.py --variant candidate_x --dry
    python scripts/build_variant.py --variant candidate_x \
        --reference-json results/rescore600_asm_bst_btf1.json
    python scripts/build_variant.py --variant candidate_x \
        --reference-json ref.json --sector-map-from data_cache.pkl
    python scripts/build_variant.py --variant process \
        --selection-json selection_2026.json --reference-json ref.json
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

VARIANT_MODEL_DIR = {
    "candidate_x": "combo_v2_exp",
    "process": "combo_v2_process",
}

# Keys the injected --reference-json must carry (schema of the margin-600
# re-score output; the Integrate phase wires the actual file). Extra keys
# pass through into the bundle untouched.
REQUIRED_REFERENCE_KEYS = (
    "family",                # e.g. "ASM_rescore600"
    "source_ledger",         # ledger file the numbers come from
    "window",
    "tc_bps_per_side",
    "margin_bps_annual",
    "leverage_for_reference",
    "engine",
    "mean_monthly_return",
    "sharpe",
    "max_drawdown",
    "calmar",
)

# Research pool name → ComboConfig weight field, for sleeves that have a
# ported live twin in core.combo_strategy. Anything not listed here is
# UNPORTED and fails the process build loudly.
PORTED_SLEEVES = {
    "xs_momentum_top30": "xs_mom_weight",
    "dual_momentum_vol": "dual_mom_weight",          # research pool name
    "dual_momentum_voltarget": "dual_mom_weight",    # bot-side alias
    "adaptive_voltarget_momentum": "adaptive_weight",
    # xs_momentum_12_1 = strategies.xs_momentum(n_long=50) — the SAME 12-1
    # signal as xs_momentum_top30, top-50. Ported 2026-07-25 as the second
    # xs instance (ComboConfig.xs2_*) so a selection can blend BOTH xs
    # parameterizations at once (2026 bootstrap leg). Parity:
    # tests/test_xs2_parity.py (1e-9 vs research strategies.xs_momentum).
    "xs_momentum_12_1": "xs2_weight",
}

# Non-weight ComboConfig knobs a pick pins down (research exp_rescore.py
# REGISTRY params → live-twin fields). Set EXPLICITLY at build time even
# where they match the ComboConfig class default, so the bundle does not
# silently drift if a default ever changes.
PORTED_SLEEVE_PARAMS = {
    "xs_momentum_top30": {"xs_mom_top_n": 30},   # REGISTRY {"n_long": 30}
    "xs_momentum_12_1": {"xs2_top_n": 50},       # REGISTRY {"n_long": 50}
}

SURVIVORSHIP_CAVEAT = (
    "Backtest universe is survivors-only (0 in-window delistings); every "
    "number is inflated by an unquantified amount — treat all references "
    "as upper bounds (REPORT.md Validity Notice; PIT floor 1.08%/mo)."
)

# The research data cache whose `sector_map` key (ticker → Yahoo-style
# sector name) the ledgered ASM_Bst_BTF1 construction consumed
# (strategies_v2.py / assemble_candidates.py pass it into
# residual_momentum's market_sector betas). Overridable via
# --sector-map-from; candidate_x REFUSES to build without a non-empty map
# — _residual_momentum_scores silently degrades beta_mode="market_sector"
# to market-only betas when the map is empty, which would be an
# UNLEDGERED construction.
DEFAULT_SECTOR_MAP_PICKLE = (
    "/Users/arsenkhanguieldyan/Documents/Trading/Traiding 11/data_cache.pkl"
)


def _die(msg: str, code: int = 2) -> None:
    print(f"\nBUILD_VARIANT FATAL: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(code)


def _load_reference(path: str | None, dry: bool) -> dict:
    """Load and schema-check the injected backtest reference."""
    if path is None:
        if dry:
            return {
                "status": "PENDING_INJECTION",
                "note": ("backtest_reference must be injected via "
                         "--reference-json with the margin-600 re-score "
                         "output (family ASM_rescore600) before a real "
                         "build — numbers are never hardcoded here."),
            }
        _die("--reference-json is required for a non-dry build: "
             "backtest_reference numbers are injected from the margin-600 "
             "re-score (family ASM_rescore600), never hardcoded. "
             "Run with --dry to preview without a reference.")
    p = Path(path)
    if not p.exists():
        _die(f"--reference-json not found: {p}")
    try:
        ref = json.loads(p.read_text())
    except Exception as e:
        _die(f"--reference-json is not valid JSON ({e}): {p}")
    missing = [k for k in REQUIRED_REFERENCE_KEYS if k not in ref]
    if missing:
        _die(f"--reference-json {p} is missing required keys: {missing}. "
             f"Required schema: {list(REQUIRED_REFERENCE_KEYS)}")
    ref.setdefault("survivorship_bias", SURVIVORSHIP_CAVEAT)
    return ref


def _shared_manifest(*, model, combo_config, version: str, tag: str,
                     reference: dict, provenance: dict) -> dict:
    """Manifest core shared by every variant bundle.

    Same runner-facing keys as the primary bundle (strategy_type,
    horizon, combo_config, ...) plus a mandatory `provenance` block whose
    schema is common across variants: disputed_components,
    burned_val_disclosure, phi_stress_band, survivorship — absent items
    are explicit (None / empty), never silently missing.
    """
    for k in ("disputed_components", "burned_val_disclosure",
              "phi_stress_band", "survivorship"):
        provenance.setdefault(k, None)
    provenance["survivorship"] = provenance["survivorship"] or SURVIVORSHIP_CAVEAT
    return {
        "model": model,
        "strategy_type": "direct_weights",
        "feature_cols": [],
        "horizon": 21,
        "version": version,
        "tag": tag,
        "combo_config": combo_config,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "built_by": "scripts/build_variant.py",
        "backtest_reference": reference,
        "provenance": provenance,
    }


def _load_sector_map(path: str) -> dict:
    """Load the research sector map (ticker → Yahoo-style sector name).

    ASM_Bst_BTF1 is ledgered with residual_beta_mode="market_sector" and
    the research sector map; a bundle built without the map ships
    residual_sector_map=None and _residual_momentum_scores degrades every
    name to market-only betas — a construction that exists in no ledger.
    Fails loudly on a missing file, a missing/empty 'sector_map' key, or
    a map whose sector names cannot resolve through the bot's
    SECTOR_TO_ETF table (wrong schema).
    """
    p = Path(path)
    if not p.exists():
        _die(f"--sector-map-from not found: {p}\n"
             "Candidate X requires the research sector map (data_cache.pkl "
             "key 'sector_map', ticker → Yahoo-style sector) — the ledgered "
             "ASM_Bst_BTF1 uses market_sector residual betas; without the "
             "map the sleeve silently degrades to market-only betas (an "
             "unledgered construction). Point --sector-map-from at the "
             "research data cache.")
    try:
        with open(p, "rb") as fh:
            cache = pickle.load(fh)
    except Exception as e:
        _die(f"--sector-map-from {p} failed to unpickle ({e})")
    sector_map = cache.get("sector_map") if isinstance(cache, dict) else None
    if not isinstance(sector_map, dict) or not sector_map:
        _die(f"--sector-map-from {p}: no non-empty 'sector_map' dict "
             f"(got {type(sector_map).__name__}). REFUSING to build "
             "candidate_x with an empty sector map — market_sector residual "
             "betas would silently degrade to market-only (unledgered "
             "construction).")
    import core.combo_strategy as cs
    resolvable = {s for s in sector_map.values() if s in cs.SECTOR_TO_ETF}
    if not resolvable:
        _die(f"--sector-map-from {p}: none of the map's sector names "
             f"(sample: {sorted(set(map(str, sector_map.values())))[:5]}) "
             "resolve via core.combo_strategy.SECTOR_TO_ETF — wrong map "
             "schema; expected Yahoo-style names like 'Technology'.")
    return dict(sector_map)


# ═══════════════════════════════════════════════════════════════════════════
# Variant: candidate_x  (frozen construction — ASM_Bst_BTF1)
# ═══════════════════════════════════════════════════════════════════════════

def _require_residual_port(cs_module) -> None:
    """Loud porting-prerequisite guard for the residual momentum sleeve.

    The B-base construction needs strategies_v2.residual_momentum
    (market_sector) ported into ComboStrategy as a 4th sleeve with 1e-9
    live-twin parity (plan step S2, tests/test_residual_parity.py). Until
    that lands, this build must fail — there is no fallback construction.
    """
    fields = {f.name for f in dataclasses.fields(cs_module.ComboConfig)}
    has_field = "residual_weight" in fields
    has_fn = any(
        callable(getattr(cs_module, name, None)) and "residual" in name.lower()
        for name in dir(cs_module)
    )
    if has_field and has_fn:
        return
    _die(
        "PORTING PREREQUISITE NOT MET — residual momentum sleeve is not in "
        "core/combo_strategy.py.\n"
        f"  ComboConfig has 'residual_weight' field : {has_field}\n"
        f"  module has a residual sleeve function   : {has_fn}\n"
        "Candidate X (ASM_Bst_BTF1) is a B-base construction: it REQUIRES "
        "the strategies_v2.residual_momentum (market_sector) port as a 4th "
        "sleeve, with 1e-9 live-twin parity on E3's seeded dates on the "
        "bot's data path (tests/test_residual_parity.py) and the frozen-"
        "primary golden test green (tests/test_primary_bundle_frozen.py). "
        "Land the port (plan step S2), then re-run this build. Do NOT "
        "substitute another sleeve — that would be an unledgered "
        "construction, off the frozen DECISION_RULE.md menu.")


def _build_candidate_x(reference: dict, sector_map: dict) -> dict:
    """Frozen builder for Candidate X v1 = ASM_Bst_BTF1, exactly as
    ledgered (trials/ASSEMBLY_tier.csv), per the frozen selection rule
    results/v7/candidate_x/DECISION_RULE.md (2026-07-18).

    Construction: B-base static-1/3 blend (residual + dual + adaptive) +
    book gate 0.12/0.30/60 + portfolio tier (slot cutloss config, see
    provenance) + freeze dd_v1 ENABLED as a DISPUTED component. The
    residual sleeve runs market_sector betas against the INJECTED
    `sector_map` (the research data-cache map the ledgered construction
    used — see _load_sector_map); an empty map is rejected upstream.
    Any change here = new version string + dated amendment upstream.
    """
    import core.combo_strategy as cs
    _require_residual_port(cs)
    if not sector_map:
        _die("candidate_x requires a non-empty sector map (ticker → "
             "sector); see --sector-map-from.")

    kwargs = dict(
        # B-base: residual momentum replaces xs_momentum_top30; static 1/3.
        # Residual = strategies_v2 market_sector/raw with the research
        # sector map EMBEDDED in the bundle (ComboConfig.residual_sector_map)
        # — leaving it None degrades to market-only betas (unledgered).
        xs_mom_weight=0.0,
        residual_weight=1.0 / 3.0,
        residual_top_n=30,
        residual_beta_mode="market_sector",
        residual_scale_mode="raw",
        residual_lookback_long=252,
        residual_lookback_skip=21,
        residual_sector_map=dict(sector_map),
        dual_mom_weight=1.0 / 3.0,
        adaptive_weight=1.0 / 3.0,
        dual_mom_top_n=30, dual_mom_vol_target=0.15,
        adaptive_top_n=30,
        adaptive_calm_leverage=1.5,
        adaptive_neutral_leverage=1.0,
        adaptive_stress_leverage=0.5,
        # "B" overlay: book drawdown gate, as ledgered.
        enable_spy_dd_gate=False,
        enable_book_dd_gate=True,
        book_dd_lookback=60, book_full_dd=0.12, book_cash_dd=0.30,
        # "F1" overlay: freeze dd_v1 ON — DISPUTED (see provenance block).
        enable_drawdown_freeze=True,
        dd_freeze_pct=0.12, dd_peak_lookback=21,
        dd_unfreeze_within_pct=0.08, dd_min_freeze_days=14,
        max_gross_exposure=1.0,
        min_position_weight=0.003,
    )
    try:
        config = cs.ComboConfig(**kwargs)
    except TypeError as e:
        _die("PORTING PREREQUISITE NOT MET — ComboConfig rejected the "
             f"candidate-X kwargs ({e}). The residual sleeve port must add "
             "back-compat 'residual_weight' (default 0.0) to ComboConfig; "
             "see plan step S2.")

    provenance = {
        "selection": {
            "rule": "results/v7/candidate_x/DECISION_RULE.md (frozen 2026-07-18, "
                    "research repo 'Traiding 11'); amendments append-only",
            "row": "ASM_Bst_BTF1",
            "ledger": "results/v7/trials/ASSEMBLY_tier.csv",
            "dev_mean_monthly": 0.0428,
            "dev_calmar": 1.234,
            "dev_max_drawdown": -0.417,
            "dev_turnover_x_per_yr": 20.8,
            "note": "DEV-ledger numbers at the pre-margin cost basis; the "
                    "deployed backtest_reference uses ONLY the injected "
                    "margin-600 re-score (rule R6).",
        },
        "disputed_components": {
            "freeze_dd_v1": {
                "status": "DISPUTED — enabled but unvalidated provenance",
                "taint": "parameters from the retracted V6 in-sample "
                         "full-window 135-cell grid; RESCORE_NOTES stability "
                         "83.3% < 85% bar; the prescribed dev-window freeze "
                         "grid was never run",
                "why_it_ships": "swapping to an off-menu variant now would be "
                                "an unledgered new construction — worse "
                                "methodology than deploying the disputed "
                                "ledgered one (DECISION_RULE.md ruling)",
                "adjudication": "pre-registered monthly FREEZE ABLATION "
                                "REPLAY: bot journals daily weights + freeze "
                                "state; research replays the identical book "
                                "without the freeze multiplier. Negative live "
                                "contribution at the 12-month review → freeze "
                                "disabled in any promoted config (binding "
                                "config decision, not a slot pass/fail leg).",
            },
        },
        "burned_val_disclosure": {
            "status": "disclosure, not criterion",
            "date": "2026-06-12",
            "val_mean_monthly": 0.0615,
            "val_max_drawdown": -0.281,
            "val_calmar": 3.22,
            "note": "one-shot validation window burned BEFORE the selection "
                    "rule was authored; these numbers played no role in "
                    "selection and are never re-used as criteria. The shadow "
                    "forward test (protocol_exp.json) is the arbiter.",
        },
        "phi_stress_band": {
            "source": "FINAL_VALIDATION_REPORT.md Step 4, full window",
            "phi_0.0": {"mean_monthly": 0.0493, "calmar": 1.484, "max_drawdown": -0.417},
            "phi_0.5": {"mean_monthly": 0.0447, "calmar": 1.319, "max_drawdown": -0.417},
            "phi_1.0": {"mean_monthly": 0.0377, "calmar": 1.070, "max_drawdown": -0.418},
            "note": "dev tier-track numbers are close-to-close phi=0 "
                    "approximations; live is intraday — phi=0.5..1.0 is the "
                    "honest expected band. The shadow slot must NOT be read "
                    "as failing the phi=0 number.",
        },
        "e3_tension": "B-base wins on the assembly-interaction finding only "
                      "(+0.12–0.16 Calmar, dev-only); E3 falsified residual "
                      "momentum standalone. The shadow test carries the "
                      "burden of proof.",
        "tier_layer": {
            "implemented_via": "slot cutloss config, NOT this bundle",
            "required_slot_settings": {
                "enable_cutloss": True,
                "cutloss_hard_stop": None,
                "cutloss_trailing_stop": None,
                "cutloss_portfolio_stop": -4.0,
                "cutloss_scale_by_leverage": True,
                "cutloss_reentry_delay_days": 2,
            },
            "note": "the 'T' in ASM_Bst_BTF1 — leverage-scaled portfolio "
                    "tiers (-8% levered equity Tier 1 at 2x).",
        },
        "target_leverage_required": 2.0,
        "protocol": "protocol_exp.json (per-slot; XK1 continuous kill "
                    "DD<=-40%; relative 12-month legs vs primary)",
    }

    strategy = cs.ComboStrategy(config=config)
    return _shared_manifest(
        model=strategy,
        combo_config=config,
        version="combo_v2_exp.1_asm_bst_btf1",
        tag="Candidate X v1 — ASM_Bst_BTF1: B-base 1/3 blend "
            "(residual+dual+adaptive) + book gate 12/30 + tier + freeze "
            "dd_v1 (DISPUTED), 21d cadence, shadow slot #2",
        reference=reference,
        provenance=provenance,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Variant: process  (walk-forward annual re-selection, spec v2)
# ═══════════════════════════════════════════════════════════════════════════

# wf_select.py (research) emits "selection_year"; accept both spellings —
# the loud-failure contract is about MISSING data, not key naming drift.
REQUIRED_SELECTION_KEYS = ("picks", "spec_sha256")


def _build_process(selection_json_path: str, reference: dict) -> dict:
    """Frozen builder for the process slot: equal blend of the year's
    selected sleeves, from research-side wf_select.py output.

    The bot NEVER self-modifies: selection_<year>.json is produced in the
    research repo, this builder turns it into a bundle versioned
    combo_v2_process.<year>.<sha8-of-selection-file>, and the swap is
    pre-authorized by protocol_process.json. Unported sleeves fail loudly.
    """
    import core.combo_strategy as cs

    if not selection_json_path:
        _die("--variant process requires --selection-json "
             "(research wf_select.py output, e.g. selection_2026.json)")
    p = Path(selection_json_path)
    if not p.exists():
        _die(f"--selection-json not found: {p}")
    raw = p.read_bytes()
    sha8 = hashlib.sha256(raw).hexdigest()[:8]
    try:
        sel = json.loads(raw)
    except Exception as e:
        _die(f"--selection-json is not valid JSON ({e}): {p}")

    missing = [k for k in REQUIRED_SELECTION_KEYS if k not in sel]
    if "year" not in sel and "selection_year" not in sel:
        missing.append("year|selection_year")
    if missing:
        _die(f"selection file {p} missing required keys: {missing}. "
             f"Required: {list(REQUIRED_SELECTION_KEYS)} + year|selection_year "
             f"(wf_select.py output schema).")

    year = int(sel.get("year", sel.get("selection_year")))
    picks = list(sel["picks"])
    if len(picks) != 3:
        _die(f"selection has {len(picks)} picks {picks}; process spec v2 "
             f"is k=3 equal blend — a different k requires a dated spec "
             f"amendment, not a build-time override.")

    unported = [name for name in picks if name not in PORTED_SLEEVES]
    if unported:
        _die(
            "PORTING PREREQUISITE NOT MET — selection includes sleeves with "
            f"no ported live twin in core/combo_strategy.py: {unported}.\n"
            f"Ported sleeves: {sorted(set(PORTED_SLEEVES))}.\n"
            "Porting + parity on the bot's data path is a DEPLOY "
            "PREREQUISITE (plan Stream 3.2). Port the sleeve, add it to "
            "PORTED_SLEEVES with its ComboConfig weight field, land its "
            "parity test, then rebuild. Never substitute or drop a pick.")

    fields = [PORTED_SLEEVES[name] for name in picks]
    if len(set(fields)) != len(fields):
        _die(f"selection picks {picks} map to duplicate ComboConfig weight "
             f"fields {fields} — cannot express this blend; check the "
             f"selection file / PORTED_SLEEVES mapping.")

    # Equal blend, no in-strategy overlays: the measured walk-forward
    # object (process_spec_v2) is the PLAIN equal-weight blend at 2.0x —
    # NO book gate, NO SPY gate, NO freeze (the WS3 process curve carried
    # none of them; the tier stop lives in the slot cutloss config, not
    # this bundle). All three overlay flags are therefore forced False
    # below and echoed in provenance["overlays_config"].
    kwargs = {f.name: 0.0 for f in dataclasses.fields(cs.ComboConfig)
              if f.name.endswith("_weight") and f.name != "min_position_weight"}
    for field in fields:
        kwargs[field] = 1.0 / 3.0
    for name in picks:
        kwargs.update(PORTED_SLEEVE_PARAMS.get(name, {}))
    kwargs.update(
        enable_spy_dd_gate=False,
        enable_book_dd_gate=False,
        enable_drawdown_freeze=False,
        max_gross_exposure=1.0,
        min_position_weight=0.003,
    )
    try:
        config = cs.ComboConfig(**kwargs)
    except (TypeError, ValueError) as e:
        _die(f"ComboConfig rejected the process blend ({e}); "
             f"kwargs={kwargs}")

    provenance = {
        "process_spec": {
            "version": "v2 (annual Dec-31 re-selection, k=3 equal blend by "
                       "Sharpe, 21-name pool, 2.0x)",
            "supersedes_v1_sha256": "b75aa7e7fe34eec11ebcd5b528c5a5f4d75f"
                                    "131fda1bd5e7cb54c773688503a0",
            "spec_sha256": sel["spec_sha256"],
        },
        "selection": {
            "file": str(p),
            "file_sha256_8": sha8,
            "year": year,
            "picks": picks,
            "sharpes": sel.get("sharpes"),
            "data_revision_shas": sel.get("data_revision_shas"),
        },
        "disputed_components": None,
        "burned_val_disclosure": None,
        "phi_stress_band": None,
        "accepted_risk": "historical process path MaxDD -57.7% at 2.0x is "
                         "DEEPER than the PK1 kill bar (-45%); a repeat "
                         "kills the slot — disclosed and accepted "
                         "(protocol_process.json).",
        "hindsight_tax": "measured process-vs-champion gap about "
                         "-1.0pp/mo (WF process 3.40%/mo vs hindsight "
                         "ceiling 4.44%/mo).",
        "governance": "bot never self-modifies: research wf_select.py -> "
                      "selection_<year>.json -> this builder -> commit + "
                      "deploy. Annual swap pre-authorized by "
                      "protocol_process.json; missing the Dec-31 window by "
                      ">10 trading days = logged operational failure.",
        "overlays": "NONE in-strategy (plain equal blend = the measured "
                    "walk-forward object). Risk control: slot tier config + "
                    "PK1 kill DD<=-45%.",
        # Verified against process_spec_v2 (sha 162bdb25...): the measured
        # WS3 process curve had NO overlays, so every in-strategy gate is
        # OFF. Echoed from the BUILT config, not hand-written, so a drift
        # in the builder shows up here.
        "overlays_config": {
            "enable_spy_dd_gate": bool(config.enable_spy_dd_gate),
            "enable_book_dd_gate": bool(config.enable_book_dd_gate),
            "enable_drawdown_freeze": bool(config.enable_drawdown_freeze),
            "tier_stop": "slot cutloss config, NOT this bundle (operator "
                         "runs the process slot with enable_cutloss=false "
                         "to match the measured object)",
        },
        "sleeve_params": {
            name: dict(PORTED_SLEEVE_PARAMS.get(name, {})) for name in picks
        },
        "target_leverage_required": 2.0,
        "protocol": "protocol_process.json",
    }

    strategy = cs.ComboStrategy(config=config)
    return _shared_manifest(
        model=strategy,
        combo_config=config,
        version=f"combo_v2_process.{year}.{sha8}",
        tag=f"Process slot {year} — walk-forward k=3 equal blend: "
            f"{' + '.join(picks)} (spec v2), shadow slot #3",
        reference=reference,
        provenance=provenance,
    )


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_out(variant: str, out_arg: str | None) -> Path:
    out = (Path(out_arg) if out_arg
           else _BASE_DIR / "model" / VARIANT_MODEL_DIR[variant] / "model.pkl")
    out = out.resolve()
    frozen_primary_dir = (_BASE_DIR / "model" / "combo_v2").resolve()
    if frozen_primary_dir == out.parent or frozen_primary_dir in out.parents:
        _die(f"REFUSED: output {out} is inside the FROZEN primary bundle "
             f"directory {frozen_primary_dir}. The primary is built only by "
             f"scripts/build_combo_v2.py and never overwritten here.")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANT_MODEL_DIR))
    ap.add_argument("--reference-json", default=None,
                    help="margin-600 re-score output (ASM_rescore600) — "
                         "REQUIRED for a non-dry build")
    ap.add_argument("--selection-json", default=None,
                    help="wf_select.py selection_<year>.json (process only)")
    ap.add_argument("--sector-map-from", default=None,
                    help="pickle whose 'sector_map' key is the research "
                         "ticker→sector dict embedded in the candidate_x "
                         "bundle (market_sector residual betas); default: "
                         f"{DEFAULT_SECTOR_MAP_PICKLE}")
    ap.add_argument("--dry", action="store_true",
                    help="construct config + manifest, print summary, write "
                         "nothing")
    ap.add_argument("--out", default=None,
                    help="override output path (never the primary dir)")
    args = ap.parse_args(argv)

    out = _resolve_out(args.variant, args.out)
    reference = _load_reference(args.reference_json, args.dry)

    if args.variant == "candidate_x":
        if args.selection_json:
            _die("--selection-json applies only to --variant process")
        sector_map = _load_sector_map(
            args.sector_map_from or DEFAULT_SECTOR_MAP_PICKLE)
        bundle = _build_candidate_x(reference, sector_map)
    else:
        if args.sector_map_from:
            _die("--sector-map-from applies only to --variant candidate_x")
        bundle = _build_process(args.selection_json, reference)

    cfg = bundle["combo_config"]
    print(f"variant:   {args.variant}")
    print(f"version:   {bundle['version']}")
    print(f"tag:       {bundle['tag']}")
    print(f"reference: {reference.get('family', reference.get('status'))}")
    print("combo_config:")
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, dict) and len(v) > 10:
            # e.g. residual_sector_map (~1000 tickers) — summarize, don't spew.
            v = f"<dict: {len(v)} entries>"
        print(f"    {f.name} = {v}")
    print("provenance blocks: "
          f"{[k for k, v in bundle['provenance'].items() if v is not None]}")

    if args.dry:
        print(f"\nDRY RUN — nothing written (target would be {out})")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nsaved bundle → {out}")
    print(f"  size: {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
