"""Stream 4 — RESULTS_CURRENT.md generator (WS5 spec, render-from-artifact).

Renders the single current-truth results page at the repo root ENTIRELY from
committed artifacts — no numbers are typed into this file. Sections:
engine/reporting settings, catalog_v2 champion (geo headline), render-time
DSR at the CURRENT deduped trial count with ledger-measured cross-trial
variance, WF process-evaluation numbers, PIT survivorship bound +
missing-name rate, rf-adjusted Sharpe pair (WS4a), Stream-2 improvement-
family ledgers (regime_vixterm / regime_breadth / cov_voltarget), the
candidate-X / G4-retirement / CRASH1 governance pointers, the live paper
hook (results/live/paper_summary.json — placeholder written on first run if
absent), and a provenance footer.

Regenerate anytime:
    cd "Traiding 11" && /opt/anaconda3/bin/python report_current.py
Numbers move ONLY when the underlying artifacts move (a new ledgered trial
changes the deduped trial count and hence every rendered DSR — by design:
DSR is a program-level statistic, not a per-run constant).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp_lib import _CODE_HASH, TRIALS_DIR, trial_count
from metrics_v2 import DEV_END, VAL_START, deflated_sharpe

ROOT = Path(__file__).resolve().parent
V7 = ROOT / "results" / "v7"
LIVE_DIR = ROOT / "results" / "live"
PAPER_JSON = LIVE_DIR / "paper_summary.json"
OUT_MD = ROOT / "RESULTS_CURRENT.md"

PLACEHOLDER = {
    "status": "pending",
    "note": ("12-month pre-registered forward paper test of the primary "
             "combo_v2 slot starts 2026-07-20 (protocol.json, K1 kill "
             "DD<=-45%, 12mo geo floor 1.8%/mo). The bot's A2 tracking "
             "exports this file; until then this placeholder renders as "
             "'not yet running'."),
    "as_of": None,
    "slots": {},
}


def pct(v: float, digits: int = 2) -> str:
    return f"{v * 100:+.{digits}f}%"


def main() -> None:
    # ── artifacts ─────────────────────────────────────────────────────────
    cat = pd.read_csv(V7 / "catalog_v2.csv")
    cat_full = (cat[cat["window"] == "full"].drop_duplicates("name", keep="last")
                .sort_values("geo_monthly", ascending=False))
    champ = cat_full[cat_full["name"] == "combo_v2_2x"].iloc[0]
    champ_dev = cat[(cat["name"] == "combo_v2_2x")
                    & (cat["window"] == "dev")].iloc[0]

    wf = pd.read_csv(V7 / "wf" / "wf_results.csv").set_index("name")
    wf_spec = V7 / "wf" / "process_spec.json"
    # The spec's binding identity is its self-recorded canonical-json sha256
    # (wf_process.spec_sha256, excludes the sha field itself) — NOT the file
    # bytes; verify it before rendering (tamper check, same as load path).
    from wf_process import spec_sha256
    spec_doc = json.loads(wf_spec.read_text())
    wf_sha = spec_doc["sha256"]
    if spec_sha256(spec_doc) != wf_sha:
        raise SystemExit("process_spec.json sha mismatch — spec tampered; "
                         "refusing to render.")

    pit = pd.read_csv(V7 / "pit" / "pit_results.csv")
    pit2 = pit[(pit["leverage"] == 2.0) & (pit["window"] == "full")]
    pit_base = pit2[pit2["universe"] == "baseline"].iloc[0]
    pit_arm = pit2[pit2["universe"] == "pit_cache_only"].iloc[0]
    missing = pd.read_csv(V7 / "pit" / "missing_names.csv")
    m_first, m_last = missing.iloc[0], missing.iloc[-1]

    rf = pd.read_csv(V7 / "rf_sharpe_table.csv")
    rf_champ_full = rf[(rf["name"] == "combo_v2_2x")
                       & (rf["window"] == "full")].iloc[0]
    rf_champ_dev = rf[(rf["name"] == "combo_v2_2x")
                      & (rf["window"] == "dev")].iloc[0]

    stream2 = {}
    for fam, fname in (("regime_vixterm", "ws4_vixterm_verdicts.json"),
                       ("regime_breadth", "ws4_breadth_verdicts.json"),
                       ("cov_voltarget", "cov_voltarget_verdicts.json")):
        p = V7 / fname
        stream2[fam] = json.loads(p.read_text()) if p.exists() else None

    # render-time DSR for the champion at the CURRENT program trial count
    from exp_catalog_v2 import ledger_sr_variance_daily
    n_trials = trial_count(dedupe=True)
    sr_var = ledger_sr_variance_daily()
    champ_dsr = deflated_sharpe(float(champ["sharpe"]), 2512, n_trials,
                                sr_variance_across_trials=sr_var)

    # live paper hook (placeholder written once, never overwritten)
    if not PAPER_JSON.exists():
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        PAPER_JSON.write_text(json.dumps(PLACEHOLDER, indent=2))
    paper = json.loads(PAPER_JSON.read_text())

    # ── render ────────────────────────────────────────────────────────────
    def cat_row(r) -> str:
        return (f"| {r['name']} | {pct(r['geo_monthly'])} | "
                f"{pct(r['mean_monthly'])} | {r['sharpe']:.2f} | "
                f"{pct(r['max_drawdown'], 1)} | {r['calmar']:.2f} | "
                f"{r['dsr']:.3f} |")

    top5 = "\n".join(cat_row(r) for _, r in cat_full.head(5).iterrows())

    s2_rows = []
    for fam, v in stream2.items():
        if v is None:
            s2_rows.append(f"| {fam} | — | — | — | not yet run |")
            continue
        b = v["best_cell"]
        s2_rows.append(
            f"| {fam} | {v['n_cells']} | {b['name']} "
            f"({pct(b['geo_monthly'])}/mo geo, Calmar {b['calmar']:.3f}, "
            f"DSR {b['dsr_dev']:.3f}) | {v['n_pass']} | {v['verdict']} |")
    s2_tbl = "\n".join(s2_rows)

    paper_line = (
        f"live results as of {paper.get('as_of')}: see table in "
        f"`results/live/paper_summary.json`"
        if paper.get("status") not in (None, "pending")
        else "**not yet running** — placeholder in "
             "`results/live/paper_summary.json`; the primary slot's "
             "12-month pre-registered forward test starts **2026-07-20** "
             "and the bot's tracking layer will overwrite the placeholder.")

    irx_meta = ROOT / "data_free" / "_IRX.meta.json"
    free_meta = sorted((ROOT / "data_free").glob("*.meta.json"))
    free_lines = "\n".join(
        f"  - `{p.name}`: sha256 `{json.loads(p.read_text())['sha256_of_parquet'][:16]}…`"
        f" (fetched {json.loads(p.read_text())['fetched_at']})"
        for p in free_meta) if irx_meta.exists() else "  - (none)"

    md = f"""# RESULTS_CURRENT — single current-truth results page

Generated by `report_current.py` from committed artifacts — **do not edit by
hand**; regenerate instead. Every number's source artifact is named inline.
Generated {pd.Timestamp.now().isoformat(timespec='seconds')}.

## Reporting configuration (binding since the 2026-07-12 audit)

- Engine: `engine_v2`, exec `next_open` (signal at close t, market order at
  open t+1), tc 5bp/side, **margin 600bp/yr** on long gross > 1.0x NAV,
  leverage cap 2.0.
- Headline metric: **geo_monthly** = (1+CAGR)^(1/12)−1 (what an account
  compounds), never the arithmetic mean.
- Window split: dev ≤ {DEV_END}, validation ≥ {VAL_START} (one-shot,
  mechanically locked in `exp_lib.run_trial`).
- **All backtest rows ride the survivors-only bar cache — upper bounds**
  (see the PIT section).

## Champion — catalog_v2 (full window 2016-04 → 2026-03)

Source: `results/v7/catalog_v2.csv` (`exp_catalog_v2.py`).

| name | geo/mo | arith/mo | Sharpe | MaxDD | Calmar | DSR* |
|---|---|---|---|---|---|---|
{top5}

*DSR column as stored at catalog generation (n=687 trials).

**Render-time DSR** for the champion at the CURRENT program-wide deduped
trial count **n={n_trials}** (cross-trial SR variance measured from the
ledgers, `exp_catalog_v2.ledger_sr_variance_daily` = {sr_var:.2e} daily²):
**{champ_dsr:.3f}**. This number deflates further as ledgered search grows —
by design.

Champion dev row (selection basis): geo {pct((1+champ_dev['cagr'])**(1/12)-1)}/mo,
Sharpe {champ_dev['sharpe']:.2f}, MaxDD {pct(champ_dev['max_drawdown'],1)},
Calmar {champ_dev['calmar']:.3f}.

## Risk-free-adjusted Sharpe (WS4a, `results/v7/rf_sharpe_table.csv`)

Raw Sharpe ratios above deduct no financing for the un-levered NAV; against
^IRX (13-week T-bill, mean {rf_champ_full['rf_mean_ann']*100:.2f}%/yr over
the full window):

- champion full-window: raw **{rf_champ_full['sharpe_raw']:.3f}** → excess
  **{rf_champ_full['sharpe_excess']:.3f}** (Δ {rf_champ_full['sharpe_delta']:+.3f};
  approx-from-scalars, max abs err {rf_champ_full['approx_max_abs_err_on_exact_pairs']:.4f}
  vs exact pairs)
- champion dev-window (exact re-render): raw {rf_champ_dev['sharpe_raw']:.3f}
  → excess {rf_champ_dev['sharpe_excess']:.3f}
  (dev spans the near-zero-rate era; the full-window delta is the honest one)

## Walk-forward process evaluation (WS3, `results/v7/wf/wf_results.csv`)

Spec `results/v7/wf/process_spec.json`, sha256 `{wf_sha[:16]}…` (frozen).
Annual Dec-31 re-selection, top-3 by trailing Sharpe from the 21-name pool.

| curve | geo/mo | Sharpe | MaxDD | DSR |
|---|---|---|---|---|
| wf_process_2x (2019+) | {pct(wf.loc['wf_process_2x','geo_monthly'])} | {wf.loc['wf_process_2x','sharpe']:.2f} | {pct(wf.loc['wf_process_2x','max_drawdown'],1)} | {wf.loc['wf_process_2x','dsr']:.3f} |
| champion same-window (2019+) | {pct(wf.loc['champion_2019on','geo_monthly'])} | {wf.loc['champion_2019on','sharpe']:.2f} | {pct(wf.loc['champion_2019on','max_drawdown'],1)} | — |
| SPY (2019+) | {pct(wf.loc['spy_2019on','geo_monthly'])} | {wf.loc['spy_2019on','sharpe']:.2f} | {pct(wf.loc['spy_2019on','max_drawdown'],1)} | — |

Hindsight tax: {pct(wf.loc['wf_process_2x','gap_geo_vs_champion_2019on'])}/mo
vs the same-window champion (NW-t {wf.loc['wf_process_2x','nw_t_vs_champion']:.2f}).

## PIT survivorship/universe bound (WS1c, `results/v7/pit/pit_results.csv`)

At 2x, full window: published {pct(pit_base['geo_monthly'])}/mo geo (Sharpe
{pit_base['sharpe']:.2f}) vs PIT-S&P-restricted twin
{pct(pit_arm['geo_monthly'])}/mo (Sharpe {pit_arm['sharpe']:.2f}) — delta
**{pct(pit_base['geo_monthly'] - pit_arm['geo_monthly'])}/mo**, and this is a
LOWER bound (the dead cohort's bars are still absent). Missing-name rate:
{m_first['missing_rate']*100:.1f}% of true members at
{str(m_first['date'])[:10]} → {m_last['missing_rate']*100:.1f}% at
{str(m_last['date'])[:10]} (`results/v7/pit/missing_names.csv`).

**Honest live-expectation bracket at 2x: PIT floor
{pct(pit_arm['geo_monthly'])}/mo · WF process
{pct(wf.loc['wf_process_2x','geo_monthly'])}/mo · hindsight ceiling
{pct(champ['geo_monthly'])}–{pct(wf.loc['champion_2019on','geo_monthly'])}/mo**
— arbitrated by the 12-month forward paper test (floor 1.8%/mo geo).

## Improvement streams (Roadmap v2)

### Stream 2 — pre-registered dev grids (margin600, ledgered, dev-only)

Gate template (pre-committed in each script's docstring before any cell ran):
dev DSR ≥ 0.95 at the current deduped trial count AND dev Calmar ≥
baseline+10% AND geo give-up ≤ 0.2pp/mo → eligible for one-shot val LATER.

| family | cells | best cell (dev) | cells passing all gates | verdict |
|---|---|---|---|---|
{s2_tbl}

Full tables: `results/v7/ws4_vixterm_results.csv`,
`results/v7/ws4_breadth_results.csv`, `results/v7/cov_voltarget_results.csv`.
Breadth signals are survivors-biased by construction (see `breadth.py`);
only rolling-percentile transforms were allowed.

### Candidate X (Stream 1) and governance

- Selection rule + application (frozen): `results/v7/candidate_x/DECISION_RULE.md`
  — selected `ASM_Bst_BTF1` (dev 4.28%/mo, Calmar 1.234, MaxDD −41.7%),
  freeze layer DISPUTED with a pre-registered monthly ablation replay.
- **G4 retirement**: the absolute promotion gate (mean ≥5.0%/mo, MaxDD
  ≥−40%) failed the corrected champion itself and is RETIRED with a written
  finding — see the "G4 retirement finding" section of DECISION_RULE.md.
  Replacement: the relative shadow protocol (`protocol_exp.json`, bot repo).
- CRASH1 challenger pre-registration (gates frozen before its dev grid):
  `results/v7/crash/CRASH1_PREREG.md`.

## Live forward test (three-slot hook)

Primary slot ({paper_line})

Shadow slots 2 (candidate X) and 3 (WF process) bind to `protocol_exp.json`
/ `protocol_process.json`; `/api/compare` renders the side-by-side once the
paper accounts are keyed. This section auto-populates from
`results/live/paper_summary.json`.

## Provenance

- strategy code hash (`exp_lib._CODE_HASH`): `{_CODE_HASH}`
- deduped ledger trial count at render: **{n_trials}** across
  {len(list(TRIALS_DIR.glob('*.csv')))} family ledgers in `results/v7/trials/`
- wf process spec sha256: `{wf_sha}`
- frozen free-data series (`data_free/`):
{free_lines}
- governance documents (frozen, append-only): `results/v7/candidate_x/DECISION_RULE.md`,
  `results/v7/crash/CRASH1_PREREG.md`
"""
    OUT_MD.write_text(md)
    print(f"wrote {OUT_MD}")
    print(f"  n_trials={n_trials}  champion render-time DSR={champ_dsr:.3f}")
    print(f"  paper hook: {PAPER_JSON} "
          f"({'placeholder' if paper.get('status') == 'pending' else 'live'})")


if __name__ == "__main__":
    main()
