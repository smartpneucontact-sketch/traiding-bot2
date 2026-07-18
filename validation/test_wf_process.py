"""Track B / WS3 tests: wf_process spec freeze, truncation-equivalence
detection power, top-k selection, and leg concatenation — all on synthetic
data (no data_cache, no ledger writes, no network).

1. spec freeze — write_spec to a temp path succeeds, seals a verifiable
   sha256, and a SECOND write_spec call REFUSES (frozen means frozen);
   load_spec detects tampering.
2. truncation-equivalence — a causal toy builder (trailing-return top-k)
   passes; a deliberately NON-CAUSAL builder (weights scaled by the panel's
   FINAL price — future data) is detected and fails; a builder emitting a
   disjoint decision grid fails the coverage guard.
3. select_topk — hand-computable top-k on synthetic score rows, including
   the deterministic name tie-break.
4. assemble_process_frame — leg concatenation on a synthetic calendar has no
   overlaps/gaps, blends switch exactly at year boundaries, and a gap is
   detected.

check() asserts (not just records) so failures also surface under pytest.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python validation/test_wf_process.py
(plain script, no pytest dependency needed; exits non-zero on failure)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import wf_process as W  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = ""):
    flag = "OK  " if cond else "FAIL"
    print(f"  {flag} {label} {detail}")
    if not cond:
        FAILURES.append(label)
    assert cond, f"{label} {detail}".strip()


# ───── 1. spec freeze ─────────────────────────────────────────────────────
def test_spec_freeze():
    print("\n[1] spec freeze (write-once + sha seal)")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "process_spec.json"
        spec = W.write_spec(exclusions=[{"name": "toy", "reason": "test"}],
                            path=p)
        check("spec written with sha256", "sha256" in spec and p.exists())
        check("sha verifies", W.spec_sha256(spec) == spec["sha256"])
        loaded = W.load_spec(p)
        check("load_spec round-trips", loaded["sha256"] == spec["sha256"])
        check("pool has 21 names", len(loaded["pool"]) == 21,
              f"got {len(loaded['pool'])}")
        try:
            W.write_spec(exclusions=[], path=p)
            check("second write_spec refuses", False)
        except FileExistsError:
            check("second write_spec refuses", True)
        # tamper detection
        tampered = p.read_text().replace('"k": 3', '"k": 5')
        p.write_text(tampered)
        try:
            W.load_spec(p)
            check("load_spec detects tampering", False)
        except ValueError:
            check("load_spec detects tampering", True)


# ───── 2. truncation-equivalence detection power ──────────────────────────
def _toy_panel(n_days=600, n_cols=6, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n_days)
    cols = [f"S{i}" for i in range(n_cols)]
    px = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (n_days, n_cols)),
                                 axis=0)),
        index=idx, columns=cols)
    macro = pd.DataFrame({"SPY": px.mean(axis=1)}, index=idx)
    return px, macro, cols


def _causal_builder(px, macro, vol, cols):
    """Top-2 by trailing 60d return, every 21st day — strictly causal."""
    mom = px.pct_change(60, fill_method=None)
    rows = []
    for i, dt in enumerate(px.index):
        if i % 21 != 0 or i < 60:
            continue
        top = mom.loc[dt].nlargest(2).index
        w = pd.Series(0.0, index=px.columns, name=dt)
        w[top] = 0.5
        rows.append(w)
    return pd.DataFrame(rows)


def _noncausal_builder(px, macro, vol, cols):
    """Same picks, but weights scaled by the panel's FINAL prices — the
    truncated rebuild sees different 'final' prices, so this leaks."""
    w = _causal_builder(px, macro, vol, cols)
    scale = (px.iloc[-1] / px.iloc[-1].max()).clip(lower=0.1)  # future data!
    return w * scale


def _disjoint_grid_builder(px, macro, vol, cols):
    """Decision grid anchored at the END of the sample (non-causal grid):
    truncation shifts every decision date -> near-zero common coverage."""
    mom = px.pct_change(60, fill_method=None)
    n = len(px.index)
    rows = []
    for i, dt in enumerate(px.index):
        if (n - 1 - i) % 21 != 0 or i < 60:
            continue
        top = mom.loc[dt].nlargest(2).index
        w = pd.Series(0.0, index=px.columns, name=dt)
        w[top] = 0.5
        rows.append(w)
    return pd.DataFrame(rows)


def test_truncation_equivalence():
    print("\n[2] truncation_equivalence_test detection power")
    px, macro, cols = _toy_panel()
    cutoff = "2020-06-30"
    kw = dict(px=px, macro=macro, vol=None, cols=cols)
    res = W.truncation_equivalence_test(_causal_builder, cutoff, **kw)
    check("causal builder passes", res["ok"],
          f"maxdiff {res['max_abs_diff']:.2e} cover {res['coverage']:.0%}")
    res = W.truncation_equivalence_test(_noncausal_builder, cutoff, **kw)
    check("non-causal (future-price scaled) builder detected", not res["ok"],
          f"maxdiff {res['max_abs_diff']:.2e}")
    check("  ... via value mismatch", "non-causal" in res["reason"])
    # cutoff chosen so (n_days - n_trunc) is NOT a multiple of the 21-day
    # grid — otherwise the end-anchored grids would coincidentally align.
    cutoff2 = px.index[400]  # 600-401=199, 199 % 21 != 0
    res = W.truncation_equivalence_test(_disjoint_grid_builder, cutoff2, **kw)
    check("end-anchored decision grid detected", not res["ok"],
          f"coverage {res['coverage']:.0%}")


# ───── 3. select_topk on hand-computable scores ───────────────────────────
def test_select_topk():
    print("\n[3] select_topk (synthetic ledger scores)")
    scores = pd.DataFrame({
        "leg_year": [2020] * 5,
        "name": ["alpha", "bravo", "charlie", "delta", "echo"],
        "sharpe_at_selection": [0.4, 1.2, 0.9, 1.2, -0.3],
        "calmar_at_selection": [2.0, 0.1, 0.5, 0.3, 1.5],
    })
    top3 = W.select_topk(scores, k=3, metric_col="sharpe_at_selection")
    # hand-computed: bravo/delta tie at 1.2 -> name ascending; then charlie
    check("top-3 by sharpe with name tie-break",
          top3 == ["bravo", "delta", "charlie"], f"got {top3}")
    top2 = W.select_topk(scores, k=2, metric_col="calmar_at_selection")
    check("top-2 by calmar", top2 == ["alpha", "echo"], f"got {top2}")


# ───── 4. leg concatenation on a synthetic calendar ───────────────────────
def test_assemble_process_frame():
    print("\n[4] assemble_process_frame (no overlaps, no gaps)")
    idx = pd.bdate_range("2018-06-01", "2022-06-30")
    cols = ["A", "B"]
    daily = {
        "s1": pd.DataFrame(
            {"A": np.full(len(idx), 1.0), "B": np.zeros(len(idx))}, index=idx),
        "s2": pd.DataFrame(
            {"A": np.zeros(len(idx)), "B": np.full(len(idx), 1.0)}, index=idx),
    }
    selections = {2019: ["s1"], 2020: ["s1", "s2"], 2021: ["s2"],
                  2022: ["s1", "s2"]}
    out = W.assemble_process_frame(selections, daily, idx, 2019, 2022)
    target = idx[(idx.year >= 2019) & (idx.year <= 2022)]
    check("index equals calendar slice exactly (no gaps/overlaps)",
          out.index.equals(target) and not out.index.has_duplicates)
    check("2019 leg is pure s1",
          bool((out.loc["2019", "A"] == 1.0).all()
               and (out.loc["2019", "B"] == 0.0).all()))
    check("2020 leg is the equal blend",
          bool((out.loc["2020", "A"] == 0.5).all()
               and (out.loc["2020", "B"] == 0.5).all()))
    check("switch happens exactly at the year boundary",
          float(out.loc[out.loc["2021"].index[0], "A"]) == 0.0
          and float(out.loc[out.loc["2020"].index[-1], "A"]) == 0.5)
    check("partial final leg ends with the calendar",
          out.index[-1] == idx[-1])
    # a hole in the calendar must be detected
    holed = idx[(idx < "2020-03-01") | (idx > "2020-09-01")]
    daily_h = {k: v.loc[holed] for k, v in daily.items()}
    try:
        bad = W.assemble_process_frame(selections, daily_h, idx, 2019, 2022)
        check("calendar gap detected", False, f"got {len(bad)} rows")
    except (ValueError, KeyError):
        check("calendar gap detected", True)


def main():
    test_spec_freeze()
    test_truncation_equivalence()
    test_select_topk()
    test_assemble_process_frame()
    print(f"\n{'ALL OK' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
