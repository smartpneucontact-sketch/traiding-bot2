"""Point-in-time eligibility (`eligible` kwarg) tests for the three combo_v2
sleeves in strategies.py (Track B, Step B0).

1. Bit-equality: eligible=None output vs a pickle of the PRISTINE (pre-edit)
   functions run on the same synthetic panel. The pickle is a one-time
   session artifact; when absent this check is skipped with a notice.
2. Permanent invariant (needs no pickle): eligible=None output equals
   eligible=all-True-frame output exactly (DataFrame.equals).
3. Filtering: flipping one ticker ineligible mid-window changes selections
   after the flip date ONLY; the flipped name carries zero weight from the
   flip onward while pre-flip rows stay bit-identical.
4. Pre-history fallback: decision dates before eligible.index[0] are
   unrestricted (match the eligible=None run) — disclosed design choice.
5. Names absent from eligible.columns count as NOT eligible.

check() asserts (not just records) so failures also surface under pytest.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python3 validation/test_pit_eligible.py
(plain script, no pytest dependency needed; exits non-zero on failure)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategies import (  # noqa: E402
    adaptive_voltarget_momentum,
    dual_momentum_voltarget,
    xs_momentum,
)

# One-time pre-edit artifact (see module docstring, test 1).
PRISTINE_PKL = Path(
    "/private/tmp/claude-501/-Users-arsenkhanguieldyan-Documents-Trading-Traiding-11/"
    "f21dd2aa-2d62-4792-ae84-fe8549e95efe/scratchpad/pristine_sleeves_b0.pkl"
)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = ""):
    flag = "OK  " if cond else "FAIL"
    print(f"  {flag} {label} {detail}")
    if not cond:
        FAILURES.append(label)
    assert cond, f"{label} {detail}".strip()


N_LONG = 3       # small top-N so a 10-ticker panel exercises real selection
FLIP_I = 350     # eligibility flip index — past every sleeve's warm-up
HOT = "T09"      # runaway winner (see build_synthetic_panel)


def build_synthetic_panel(seed: int = 7, n_days: int = 420, n_tickers: int = 10):
    """Geometric random walks with one deterministic runaway winner (T09,
    drift ~13x the field) so all three momentum sleeves always select it —
    removing it via `eligible` must visibly change every post-flip decision.
    Deterministic under a fixed numpy version (seeded default_rng)."""
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    drift = np.full(n_tickers, 0.0003)
    drift[-1] = 0.004
    rets = rng.normal(0.0, 0.01, size=(n_days, n_tickers)) + drift
    px = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)),
                      index=idx, columns=tickers)
    macro = pd.DataFrame({
        "SPY": 300.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n_days))),
        "VIX": rng.uniform(12.0, 30.0, n_days),
    }, index=idx)
    return px, macro


# Same n_long across pristine pickle + all runs so frames are comparable.
SLEEVES = {
    "xs": lambda px, macro, **kw: xs_momentum(px, macro, n_long=N_LONG, **kw),
    "dual": lambda px, macro, **kw: dual_momentum_voltarget(px, macro, n_long=N_LONG, **kw),
    "adapt": lambda px, macro, **kw: adaptive_voltarget_momentum(px, macro, n_long=N_LONG, **kw),
}

# pytest compatibility: the test functions take (px, macro, base) positionally
# for the plain-script driver below; these module-scoped fixtures supply the
# same objects when collected by pytest. Guarded import keeps the script path
# pytest-free as documented in the module docstring.
try:
    import pytest
except ImportError:  # plain-script invocation needs no pytest
    pytest = None

if pytest is not None:
    @pytest.fixture(scope="module", name="_panel")
    def _panel_fixture():
        px, macro = build_synthetic_panel()
        base = {k: f(px, macro) for k, f in SLEEVES.items()}
        return px, macro, base

    @pytest.fixture(name="px")
    def _px_fixture(_panel):
        return _panel[0]

    @pytest.fixture(name="macro")
    def _macro_fixture(_panel):
        return _panel[1]

    @pytest.fixture(name="base")
    def _base_fixture(_panel):
        return _panel[2]


def test_bit_equality(px, macro, base):
    print("\n[1] eligible=None bit-equality (pristine pickle + all-True invariant)")
    if PRISTINE_PKL.exists():
        with open(PRISTINE_PKL, "rb") as f:
            pristine = pickle.load(f)
        for k in SLEEVES:
            check(f"{k}: eligible=None bit-equal to pristine pre-edit output",
                  base[k].equals(pristine[k]))
    else:
        print("  SKIP pristine pickle absent (one-time pre-edit artifact)")
    all_true = pd.DataFrame(True, index=px.index, columns=px.columns)
    for k, f in SLEEVES.items():
        check(f"{k}: eligible=None equals eligible=all-True",
              base[k].equals(f(px, macro, eligible=all_true)))


def test_flip(px, macro, base):
    print("\n[2] mid-window eligibility flip changes selections post-flip only")
    flip_dt = px.index[FLIP_I]
    elig = pd.DataFrame(True, index=px.index, columns=px.columns)
    elig.loc[flip_dt:, HOT] = False
    for k, f in SLEEVES.items():
        w = f(px, macro, eligible=elig)
        b = base[k]
        check(f"{k}: same decision dates as baseline", w.index.equals(b.index))
        pre = w.index < flip_dt
        post = ~pre
        check(f"{k}: baseline holds {HOT} on every post-flip decision (precondition)",
              bool((b.loc[post, HOT] > 0).all()))
        check(f"{k}: pre-flip rows bit-identical", w.loc[pre].equals(b.loc[pre]))
        check(f"{k}: {HOT} weight zero on every post-flip decision",
              bool((w.loc[post, HOT] == 0.0).all()))
        check(f"{k}: post-flip rows differ from baseline",
              not w.loc[post].equals(b.loc[post]))


def test_prehistory_fallback(px, macro, base):
    print("\n[3] dates before eligible.index[0] fall back to unrestricted")
    flip_dt = px.index[FLIP_I]
    elig_late = pd.DataFrame(True, index=px.index[FLIP_I:], columns=px.columns)
    elig_late[HOT] = False
    for k, f in SLEEVES.items():
        w = f(px, macro, eligible=elig_late)
        b = base[k]
        check(f"{k}: same decision dates as baseline", w.index.equals(b.index))
        pre = w.index < flip_dt
        check(f"{k}: pre-history decisions match eligible=None run",
              w.loc[pre].equals(b.loc[pre]))
        check(f"{k}: {HOT} zero once history starts",
              bool((w.loc[~pre, HOT] == 0.0).all()))


def test_missing_column(px, macro, base):
    print("\n[4] names absent from eligible.columns are NOT eligible")
    cols = [c for c in px.columns if c != HOT]
    elig_nc = pd.DataFrame(True, index=px.index, columns=cols)
    for k, f in SLEEVES.items():
        w = f(px, macro, eligible=elig_nc)
        check(f"{k}: {HOT} never selected", bool((w[HOT] == 0.0).all()))
        check(f"{k}: output differs from baseline (which holds {HOT})",
              not w.equals(base[k]))


if __name__ == "__main__":
    px, macro = build_synthetic_panel()
    base = {k: f(px, macro) for k, f in SLEEVES.items()}
    test_bit_equality(px, macro, base)
    test_flip(px, macro, base)
    test_prehistory_fallback(px, macro, base)
    test_missing_column(px, macro, base)
    print(f"\n{'ALL TESTS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    sys.exit(0 if not FAILURES else 1)
