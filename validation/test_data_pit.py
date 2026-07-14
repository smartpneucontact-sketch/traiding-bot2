"""data_pit correctness tests — NO network.

1. Backward reconstruction pinned on a synthetic 5-event log with
   hand-computed expected memberships (including the effective-ON-date
   convention and add-only / remove-only rows).
2. Same-day add+remove of one ticker (reuse, e.g. FOX 2019-03-19) is a
   membership no-op.
3. normalize_ticker: '.'->'-' share classes, RENAMES mapping, passthrough.
4. Guard trips: doctored event log pushes the daily count out of the
   [495, 510] band -> ValueError; doctored spot-check expectation ->
   AssertionError; spot checks outside the index are skipped, not failed.
5. If the frozen scrape exists on disk (data_pit/sp500_events.parquet),
   the full KNOWN_EVENTS set is re-asserted against a fresh reconstruction
   — reads parquet only, never the network.

check() asserts (not just records) so failures also surface under pytest.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python validation/test_data_pit.py
(plain script, no pytest dependency needed; exits non-zero on failure)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pit import (  # noqa: E402
    COUNT_BAND,
    EVENTS_PARQUET,
    CURRENT_PARQUET,
    KNOWN_EVENTS,
    normalize_ticker,
    reconstruct_membership,
)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = ""):
    flag = "OK  " if cond else "FAIL"
    print(f"  {flag} {label} {detail}")
    if not cond:
        FAILURES.append(label)
    assert cond, f"{label} {detail}".strip()


def _events(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [(pd.Timestamp(d), t, a) for d, t, a in rows],
        columns=["date", "ticker", "action"],
    )


# ───── 1. synthetic 5-event hand-computed reconstruction ─────────────────
def test_reconstruction_synthetic():
    print("\n[1] synthetic 5-event backward reconstruction")
    # Current members {A, B, C}; walk back through:
    #   2023-01-11: add B, remove D   (paired row)
    #   2023-01-09: remove F          (remove-only row)
    #   2023-01-05: add C, remove E   (paired row)
    # Hand-computed membership over bdays 2023-01-02..2023-01-13:
    #   01-02..01-04 : {A, D, E, F}
    #   01-05..01-06 : {A, C, D, F}   (C in ON its add date; E out ON it)
    #   01-09..01-10 : {A, C, D}      (F out ON 01-09)
    #   01-11..01-13 : {A, B, C}      (B in, D out ON 01-11)
    ev = _events([
        ("2023-01-11", "B", "add"), ("2023-01-11", "D", "remove"),
        ("2023-01-09", "F", "remove"),
        ("2023-01-05", "C", "add"), ("2023-01-05", "E", "remove"),
    ])
    idx = pd.bdate_range("2023-01-02", "2023-01-13")
    mem = reconstruct_membership(ev, frozenset({"A", "B", "C"}), idx)

    expected = {
        "2023-01-02": {"A", "D", "E", "F"},
        "2023-01-03": {"A", "D", "E", "F"},
        "2023-01-04": {"A", "D", "E", "F"},
        "2023-01-05": {"A", "C", "D", "F"},
        "2023-01-06": {"A", "C", "D", "F"},
        "2023-01-09": {"A", "C", "D"},
        "2023-01-10": {"A", "C", "D"},
        "2023-01-11": {"A", "B", "C"},
        "2023-01-12": {"A", "B", "C"},
        "2023-01-13": {"A", "B", "C"},
    }
    check("columns are the ever-member set",
          set(mem.columns) == {"A", "B", "C", "D", "E", "F"},
          f"got {sorted(mem.columns)}")
    for d, exp in expected.items():
        row = mem.loc[pd.Timestamp(d)]
        got = set(row.index[row])
        check(f"membership {d}", got == exp, f"expected {sorted(exp)} got {sorted(got)}")
    check("dtype is bool", (mem.dtypes == bool).all())


# ───── 2. same-day ticker reuse is a no-op ───────────────────────────────
def test_same_day_reuse_noop():
    print("\n[2] same-day add+remove of one ticker (reuse) is a no-op")
    # X 'removed' and 're-added' 2023-01-06 (two entities, one ticker):
    # X must be a member on BOTH sides of the date.
    ev = _events([
        ("2023-01-06", "X", "add"), ("2023-01-06", "X", "remove"),
    ])
    idx = pd.bdate_range("2023-01-02", "2023-01-10")
    mem = reconstruct_membership(ev, frozenset({"X", "Y"}), idx)
    check("member before reuse date", bool(mem.at[pd.Timestamp("2023-01-03"), "X"]))
    check("member on reuse date", bool(mem.at[pd.Timestamp("2023-01-06"), "X"]))
    check("member after reuse date", bool(mem.at[pd.Timestamp("2023-01-09"), "X"]))
    check("bystander untouched", mem["Y"].all())


# ───── 3. ticker normalization ───────────────────────────────────────────
def test_normalization():
    print("\n[3] normalize_ticker")
    cases = [
        ("BRK.B", "BRK-B"),   # share-class dot -> dash (yfinance style)
        ("BF.B", "BF-B"),
        ("FB", "META"),       # RENAMES
        ("ANTM", "ELV"),
        ("WLTW", "WTW"),
        ("FLT", "CPAY"),
        ("UTX", "RTX"),
        ("ECHO", "SATS"),     # reverse-direction rename (cache is canonical)
        (" aapl ", "AAPL"),   # whitespace + case
        ("TSLA", "TSLA"),     # passthrough
    ]
    for raw, exp in cases:
        got = normalize_ticker(raw)
        check(f"{raw!r} -> {exp}", got == exp, f"got {got!r}")


# ───── 4. guard trips ────────────────────────────────────────────────────
def test_guard_trips():
    print("\n[4] guards raise on doctored inputs")
    idx = pd.bdate_range("2023-01-02", "2023-01-13")
    cur = frozenset(f"T{i:03d}" for i in range(500))

    # (a) clean log inside the band passes
    ev_ok = _events([("2023-01-09", "T000", "add")])  # 499 -> 500, in band
    mem = reconstruct_membership(ev_ok, cur, idx, count_band=COUNT_BAND)
    check("clean log passes count band", mem.shape[1] == 500)

    # (b) doctored log: 10 adds on one date -> only 490 members before it,
    # below the 495 floor -> ValueError
    ev_bad = _events([("2023-01-09", f"T{i:03d}", "add") for i in range(10)])
    try:
        reconstruct_membership(ev_bad, cur, idx, count_band=COUNT_BAND)
        check("count-band guard trips", False, "no exception raised")
    except ValueError as e:
        check("count-band guard trips", "495" in str(e), str(e)[:80])

    # (c) count_band=None skips the guard (synthetic universes)
    mem = reconstruct_membership(ev_bad, cur, idx, count_band=None)
    check("count_band=None skips guard",
          int(mem.sum(axis=1).min()) == 490)

    # (d) doctored spot check -> AssertionError
    ev = _events([("2023-01-09", "Z", "add")])
    bad_checks = [("Z", "2023-01-04", True)]  # Z was NOT a member then
    try:
        reconstruct_membership(ev, frozenset({"Z"}), idx,
                               spot_checks=bad_checks)
        check("spot-check guard trips", False, "no exception raised")
    except AssertionError as e:
        check("spot-check guard trips", "Z@2023-01-04" in str(e), str(e)[:80])

    # (e) spot checks outside the index are skipped, not failed
    out_of_window = [("Z", "1999-01-04", True)]
    reconstruct_membership(ev, frozenset({"Z"}), idx,
                           spot_checks=out_of_window)
    check("out-of-window spot check skipped", True)


# ───── 5. frozen-scrape regression (parquet only, no network) ────────────
def test_frozen_scrape_known_events():
    print("\n[5] KNOWN_EVENTS vs frozen scrape (skipped if not fetched)")
    if not (EVENTS_PARQUET.exists() and CURRENT_PARQUET.exists()):
        print("  SKIP frozen parquets absent — run data_pit.py once first")
        return
    events = pd.read_parquet(EVENTS_PARQUET)
    current = pd.read_parquet(CURRENT_PARQUET)
    cur_set = frozenset(normalize_ticker(t) for t in current["ticker"])
    idx = pd.bdate_range("2016-04-01", "2026-03-27")
    # KNOWN_EVENTS passed as the guard: raises inside on any mismatch
    mem = reconstruct_membership(events, cur_set, idx,
                                 count_band=COUNT_BAND,
                                 spot_checks=KNOWN_EVENTS)
    counts = mem.sum(axis=1)
    check("daily count within guard band",
          COUNT_BAND[0] <= counts.min() and counts.max() <= COUNT_BAND[1],
          f"[{int(counts.min())}, {int(counts.max())}]")
    check(f"all {len(KNOWN_EVENTS)} known-event spot checks pass", True)


if __name__ == "__main__":
    test_reconstruction_synthetic()
    test_same_day_reuse_noop()
    test_normalization()
    test_guard_trips()
    test_frozen_scrape_known_events()
    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
