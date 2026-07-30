"""data_norgate correctness tests — synthetic fixtures, NO network, NO
subscription required.

Builds a tiny fake VM export tree in a temp dir (3 alive symbols incl. the
'.' share class BRK.B + 2 delisted symbols incl. the combined trap
FOO.A-202006, with a manifest of correct sha256s) and tests:

1. manifest sha verification: clean ingest passes; a tampered file raises;
   an unlisted extra file raises; frozen-once-ingested (re-ingest raises
   without refresh=True).
2. symbol normalization: '.' -> '-' share classes, '-YYYYMM' delisting
   suffix PRESERVED, combined case, symbol_map current_ticker mapping for
   the alive subset.
3. house schema vs data.py conventions: lowercase open/high/low/close/
   volume (+ unadj_close, dividend), DatetimeIndex; load_norgate_panel
   load_panel-compatibility; dead names kept (no min_obs survivor filter).
4. membership_frame shape/dtype vs data_pit's membership frame conventions
   (bool dtype, DatetimeIndex, normalized-symbol columns, as-of ffill).
5. gate machinery on constructed PASS and FAIL cases: an all-clean fixture
   passes all five gates and writes the marker; a name with a broken
   dividend adjustment fails gate 2; a membership discrepancy of 5 names
   fails gate 3; a TRUNCATED membership frame (research-window KNOWN_EVENTS
   unverifiable) fails gate 3 — never a silent skip; a shifted-returns
   cache fails gate 1; a wrong delisting expectation fails gate 4; any FAIL
   removes a stale marker and require_trust_gates_pass raises.
6. refresh hygiene: ingest(refresh=True) after a symbol rename removes the
   stale artifact, keeps the record clean (expected-set-derived, no rglob
   absorption), and gate 5 fails if the stale file is re-injected.
7. network-check accounting: a network=False PASS marker records
   gate2_spot_check_ran=false and is REFUSED by require_trust_gates_pass
   unless the explicit allow_skipped_network_check=True override is passed;
   a (stubbed) network=True marker satisfies the guard.

The fixture builds TOTALRETURN back-adjusted closes from unadjusted closes
+ dividends by exact back-chaining, so the gate-2 ex-date identity holds at
machine precision on the clean fixture and is broken by construction
(adjusted == unadjusted, dividends ignored) on the broken name.

Run (both work; the suite is green under either):
  /opt/anaconda3/bin/python -m pytest validation/test_data_norgate.py -q
  /opt/anaconda3/bin/python validation/test_data_norgate.py
(script mode needs no pytest; exits non-zero on failure)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_norgate as dn  # noqa: E402
from data_norgate import (  # noqa: E402
    BARS_FIELDS,
    MARKER_NAME,
    REPORT_NAME,
    base_ticker,
    delisting_events,
    ingest,
    load_norgate_panel,
    membership_frame,
    normalize_symbol,
    require_trust_gates_pass,
    split_delist_suffix,
    trust_gates,
    verify_manifest,
)

# pytest compatibility: the test functions take a `tmp: Path` argument; under
# pytest that resolves via this fixture, under script mode __main__ passes a
# mkdtemp Path directly. Script mode must keep working without pytest.
try:
    import pytest
except ImportError:  # pragma: no cover - script mode without pytest
    pytest = None

if pytest is not None:
    @pytest.fixture
    def tmp(tmp_path: Path) -> Path:
        return tmp_path

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = ""):
    flag = "OK  " if cond else "FAIL"
    print(f"  {flag} {label} {detail}")
    if not cond:
        FAILURES.append(label)
    assert cond, f"{label} {detail}".strip()


# ---------------------------------------------------------------------------
# fixture: fake VM export mimicking the documented Norgate format

ALIVE = ["AAA", "BRK.B", "CCC"]            # incl. a '.' share class
DEAD = ["SIVB-202303", "FOO.A-202006"]     # incl. the combined '.'+suffix trap
ALL_RAW = ALIVE + DEAD
FIX_START, FIX_END = "2019-01-02", "2023-06-30"
DEAD_END = {"SIVB-202303": "2023-03-10", "FOO.A-202006": "2020-06-30"}

# synthetic gate-3/4 inputs sized to the 5-name universe
SYN_COUNT_BAND = (2, 6)
SYN_KNOWN_EVENTS = [
    ("SIVB", "2023-03-10", True), ("SIVB", "2023-03-15", False),
    ("FOO-A", "2020-06-30", True), ("FOO-A", "2020-07-01", False),
    ("AAA", "2020-01-06", True),
]


def _one_symbol_bars(rng: np.random.Generator, dates: pd.DatetimeIndex,
                     start_px: float, broken_dividend: bool = False,
                     ) -> pd.DataFrame:
    """Norgate-format raw bar frame with an EXACT total-return adjustment.

    unadj follows a small random walk; quarterly dividends of ~0.5% yield.
    adjusted close is back-chained from factor_t = (unadj_t + div_t) /
    unadj_{t-1} with adj[last] = unadj[last] — exactly the TOTALRETURN
    proportional back-adjustment convention. broken_dividend=True instead
    sets adjusted == unadjusted (the adjustment chain 'forgot' dividends):
    the gate-2 ex-date identity then misses by ~ the dividend yield.
    """
    n = len(dates)
    rets = rng.normal(0.0004, 0.01, n)
    rets[0] = 0.0
    unadj = start_px * np.exp(np.cumsum(rets))
    div = np.zeros(n)
    for i in range(63, n, 63):                       # ~quarterly ex-dates
        div[i] = round(unadj[i] * 0.005, 4)
    if broken_dividend:
        adj = unadj.copy()
    else:
        factor = np.ones(n)
        factor[1:] = (unadj[1:] + div[1:]) / unadj[:-1]
        adj = np.empty(n)
        adj[-1] = unadj[-1]
        for i in range(n - 2, -1, -1):               # exact back-chain
            adj[i] = adj[i + 1] / factor[i + 1]
    opn = adj * (1 + rng.normal(0, 0.002, n))
    hi = np.maximum(opn, adj) * 1.004
    lo = np.minimum(opn, adj) * 0.996
    vol = rng.integers(1_000, 50_000, n).astype(float)
    return pd.DataFrame(
        {"Open": opn, "High": hi, "Low": lo, "Close": adj,
         "Volume": vol, "Unadjusted Close": unadj, "Dividend": div},
        index=dates,
    )


def write_manifest(raw: Path) -> None:
    """(Re)build manifest.json over the current raw tree, in the
    norgate_export.py format (files = list of entries)."""
    entries = [
        {"path": p.relative_to(raw).as_posix(),
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
         "bytes": p.stat().st_size}
        for p in sorted(raw.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    ]
    (raw / "manifest.json").write_text(json.dumps(
        {"export_timestamp_utc": "2026-07-29T00:00:00Z",
         "norgatedata_version": "fixture", "n_files": len(entries),
         "files": entries},
        indent=2))


def build_fixture(tmp: Path, broken_dividend_symbol: str | None = None,
                  mem_end: str | None = None,
                  ) -> tuple[Path, dict[str, pd.DataFrame]]:
    """Write the fake export tree + manifest; return (raw_dir, raw bars).

    mem_end truncates the membership frame (bars untouched) — the
    truncated-export case gate 3 must FAIL on when research-window
    KNOWN_EVENTS become unverifiable."""
    raw = tmp / "export"
    (raw / "bars").mkdir(parents=True)
    (raw / "membership").mkdir()
    rng = np.random.default_rng(7)
    full_idx = pd.bdate_range(FIX_START, FIX_END)

    bars: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(ALL_RAW):
        idx = full_idx if sym not in DEAD else \
            full_idx[full_idx <= pd.Timestamp(DEAD_END[sym])]
        df = _one_symbol_bars(rng, idx, start_px=50.0 + 25 * i,
                              broken_dividend=(sym == broken_dividend_symbol))
        df.to_parquet(raw / "bars" / f"{sym}.parquet")
        bars[sym] = df

    # membership: everyone in while alive (wide 0/1 frame, raw symbols)
    mem_idx = full_idx if mem_end is None else \
        full_idx[full_idx <= pd.Timestamp(mem_end)]
    mem = pd.DataFrame(0, index=mem_idx, columns=ALL_RAW, dtype=int)
    for sym in ALIVE:
        mem[sym] = 1
    for sym in DEAD:
        mem.loc[mem.index <= pd.Timestamp(DEAD_END[sym]), sym] = 1
    mem.to_parquet(raw / "membership" / "sp500.parquet")

    # full norgate_export.py delistings schema (loader needs only
    # symbol + last_quoted_date; the rest passes through)
    dl = pd.DataFrame({
        "symbol": DEAD,
        "base_symbol": [s.rsplit("-", 1)[0] for s in DEAD],
        "delist_yyyymm": [s.rsplit("-", 1)[1] for s in DEAD],
        "last_quoted_date": [DEAD_END[s] for s in DEAD],
        "final_adjusted_close": [float(bars[s]["Close"].iloc[-1]) for s in DEAD],
        "final_unadjusted_close": [float(bars[s]["Unadjusted Close"].iloc[-1])
                                   for s in DEAD],
        "security_name": ["SVB Financial (fixture)", "Foo Class A (fixture)"],
    })
    dl.to_parquet(raw / "delistings.parquet")

    write_manifest(raw)
    return raw, bars


def synthetic_wiki_membership(dest: Path) -> pd.DataFrame:
    """Independent-source stand-in: the Norgate membership collapsed to base
    tickers (what data_pit.membership_frame returns for the real run)."""
    mem = pd.read_parquet(dest / "membership" / "sp500.parquet")
    mem.index = pd.to_datetime(mem.index)
    return dn._collapse_to_base(mem)


def synthetic_expectations(dest: Path) -> list[dict]:
    dl = delisting_events(dest)
    t_sivb = float(dl.loc[dl["base_ticker"] == "SIVB",
                          "terminal_unadj_close"].iloc[0])
    return [
        {"base": "SIVB", "months": ["202303"],
         "terminal_band": (t_sivb * 0.99, t_sivb * 1.01),
         "note": "fixture terminal-price band"},
        {"base": "FOO-A", "months": ["202006"], "terminal_band": None,
         "note": "fixture existence+month"},
    ]


def cache_from_dest(dest: Path) -> dict:
    """A survivors-only cache panel that agrees with Norgate exactly."""
    panel = load_norgate_panel(("close",), dest=dest)
    close = panel["close"]
    alive_syms = [normalize_symbol(s) for s in ALIVE]
    return {"close": close[alive_syms]}


def gate_args(dest: Path) -> dict:
    return dict(
        dest=dest,
        wiki_membership=synthetic_wiki_membership(dest),
        known_events=SYN_KNOWN_EVENTS,
        count_band=SYN_COUNT_BAND,
        delisting_expectations=synthetic_expectations(dest),
        network=False,
        min_overlap_days=60,
        min_names=2,
    )


# ───── 1. manifest verification + frozen-once-ingested ───────────────────
def test_manifest_and_freeze(tmp: Path):
    print("\n[1] manifest sha verification + frozen-once-ingested")
    raw, _ = build_fixture(tmp / "t1")
    dest = tmp / "t1" / "house"

    rec = ingest(raw, dest)
    check("clean ingest passes", (dest / "_ingest_record.json").exists(),
          f"{rec['n_bars_files']} bar files")
    check("all 5 bar files ingested", rec["n_bars_files"] == 5)

    # frozen: second ingest without refresh raises
    try:
        ingest(raw, dest)
        check("frozen-once-ingested raises", False, "no exception")
    except RuntimeError as e:
        check("frozen-once-ingested raises", "FROZEN" in str(e))
    ingest(raw, dest, refresh=True)
    check("refresh=True re-ingests", True)

    # tamper: append a byte to one bar file -> verify + ingest both raise
    victim = raw / "bars" / "AAA.parquet"
    victim.write_bytes(victim.read_bytes() + b"\x00")
    try:
        verify_manifest(raw)
        check("tampered file fails verify_manifest", False, "no exception")
    except ValueError as e:
        check("tampered file fails verify_manifest", "mismatch" in str(e))
    try:
        ingest(raw, tmp / "t1" / "house2")
        check("tampered file fails ingest", False, "no exception")
    except ValueError:
        check("tampered file fails ingest", True)

    # unlisted extra file -> raise (bidirectional losslessness)
    raw2, _ = build_fixture(tmp / "t1b")
    (raw2 / "bars" / "SNEAKY.parquet").write_bytes(b"not in manifest")
    try:
        verify_manifest(raw2)
        check("unlisted extra file fails", False, "no exception")
    except ValueError as e:
        check("unlisted extra file fails", "not in manifest" in str(e))

    # legacy plain-mapping manifest format is also accepted
    raw3, _ = build_fixture(tmp / "t1c")
    man = json.loads((raw3 / "manifest.json").read_text())
    mapping = {e["path"]: e["sha256"] for e in man["files"]}
    (raw3 / "manifest.json").write_text(json.dumps(
        {"created_at": "2026-07-29T00:00:00Z", "files": mapping}))
    verify_manifest(raw3)
    ingest(raw3, tmp / "t1c" / "house")
    check("dict-format manifest also accepted", True)


# ───── 2. symbol normalization ────────────────────────────────────────────
def test_normalization(tmp: Path):
    print("\n[2] symbol normalization + symbol_map")
    cases = [
        ("BRK.B", "BRK-B"),                    # '.' class -> '-' (yfinance)
        ("BF.B", "BF-B"),
        ("AAA", "AAA"),                        # passthrough
        ("SIVB-202303", "SIVB-202303"),        # delist suffix PRESERVED
        ("FOO.A-202006", "FOO-A-202006"),      # combined trap
        (" aapl ", "AAPL"),                    # whitespace + case
    ]
    for raw_s, exp in cases:
        got = normalize_symbol(raw_s)
        check(f"normalize {raw_s!r} -> {exp}", got == exp, f"got {got!r}")
    check("split suffix SIVB-202303",
          split_delist_suffix("SIVB-202303") == ("SIVB", "202303"))
    check("split suffix BRK.B", split_delist_suffix("BRK.B") == ("BRK.B", None))
    check("base_ticker FOO.A-202006", base_ticker("FOO.A-202006") == "FOO-A")

    # delist-suffix parsing is the EXPORTER'S rule (norgate_export.
    # parse_norgate_symbol), not a diverging re-implementation: month must
    # be 01-12, so 'ABC-201399' / 'ABC-201300' are NOT delisting suffixes.
    from norgate_export import INDEX_NAMES, parse_norgate_symbol
    check("invalid month 99 not a delist suffix",
          split_delist_suffix("ABC-201399") == ("ABC-201399", None))
    check("invalid month 00 not a delist suffix",
          split_delist_suffix("ABC-201300") == ("ABC-201300", None))
    check("normalize keeps invalid-month name whole",
          normalize_symbol("ABC-201399") == "ABC-201399")
    for s in ["SIVB-202303", "BRK.B", "FOO.A-202006", "ABC-201399", "AAPL"]:
        p = parse_norgate_symbol(s)
        check(f"loader parse == exporter parse for {s!r}",
              split_delist_suffix(s) == (p.base, p.delist_yyyymm))

    # membership tag contract == the exporter's filenames, exactly
    check("MEMBERSHIP_TAGS == exporter INDEX_NAMES keys",
          dn.MEMBERSHIP_TAGS == tuple(INDEX_NAMES)
          and dn.MEMBERSHIP_TAGS == ("sp500", "sp400", "sp600",
                                     "sp1500", "r1000", "r3000"))

    raw, _ = build_fixture(tmp / "t2")
    dest = tmp / "t2" / "house"
    ingest(raw, dest)
    sm = pd.read_parquet(dest / "symbol_map.parquet").set_index("symbol")
    check("symbol_map has all 5 symbols", len(sm) == 5)
    check("BRK-B alive with current_ticker",
          bool(sm.at["BRK-B", "alive"]) and sm.at["BRK-B", "current_ticker"] == "BRK-B")
    check("SIVB-202303 dead, current_ticker None",
          not bool(sm.at["SIVB-202303", "alive"])
          and sm.at["SIVB-202303", "current_ticker"] is None)
    check("FOO-A-202006 base ticker FOO-A",
          sm.at["FOO-A-202006", "base_ticker"] == "FOO-A"
          and sm.at["FOO-A-202006", "delist_yyyymm"] == "202006")
    check("bars filenames normalized",
          (dest / "bars" / "BRK-B.parquet").exists()
          and (dest / "bars" / "FOO-A-202006.parquet").exists())
    check("every bars parquet has a sidecar",
          all(p.with_suffix(p.suffix + ".meta.json").exists()
              for p in (dest / "bars").glob("*.parquet")))


# ───── 3. schema + loader vs data.py conventions ──────────────────────────
def test_schema_and_loader(tmp: Path):
    print("\n[3] house schema + load_norgate_panel compatibility")
    raw, raw_bars = build_fixture(tmp / "t3")
    dest = tmp / "t3" / "house"
    ingest(raw, dest)

    b = pd.read_parquet(dest / "bars" / "AAA.parquet")
    check("bars columns == house schema",
          list(b.columns) == list(BARS_FIELDS), f"got {list(b.columns)}")
    check("DatetimeIndex", isinstance(b.index, pd.DatetimeIndex))
    # data.py load_panel default fields are a subset of ours (lowercase)
    check("data.py field names covered",
          set(("close", "open", "high", "low", "volume")) <= set(b.columns))
    check("adjusted close values preserved",
          np.allclose(b["close"].values, raw_bars["AAA"]["Close"].values))
    check("unadj + dividend carried",
          np.allclose(b["unadj_close"].values,
                      raw_bars["AAA"]["Unadjusted Close"].values)
          and float(b["dividend"].sum()) > 0)

    panel = load_norgate_panel(dest=dest)
    check("panel dict has default fields",
          set(panel) == {"close", "open", "high", "low", "volume"})
    close = panel["close"]
    check("columns are normalized symbols",
          set(close.columns) == {"AAA", "BRK-B", "CCC",
                                 "SIVB-202303", "FOO-A-202006"})
    check("dead name kept despite short life (no survivor filter)",
          "FOO-A-202006" in close.columns)
    check("dead name NaN after last_quoted_date",
          close["SIVB-202303"].loc["2023-03-11":].isna().all()
          and np.isfinite(close.at[pd.Timestamp("2023-03-10"), "SIVB-202303"]))
    win = load_norgate_panel(("close",), start="2020-01-01",
                             end="2020-12-31", dest=dest)["close"]
    check("start/end window respected",
          win.index[0] >= pd.Timestamp("2020-01-01")
          and win.index[-1] <= pd.Timestamp("2020-12-31"))
    extra = load_norgate_panel(("close", "unadj_close", "dividend"),
                               dest=dest)
    check("unadj_close/dividend loadable as fields",
          set(extra) == {"close", "unadj_close", "dividend"})


# ───── 4. membership frame vs data_pit conventions ────────────────────────
def test_membership_frame(tmp: Path):
    print("\n[4] membership_frame shape/dtype vs data_pit conventions")
    raw, _ = build_fixture(tmp / "t4")
    dest = tmp / "t4" / "house"
    ingest(raw, dest)

    idx = pd.bdate_range("2019-06-03", "2023-06-30")
    mem = membership_frame("sp500", idx, dest=dest)

    # reference shape: data_pit.reconstruct_membership output conventions
    from data_pit import reconstruct_membership
    ref = reconstruct_membership(
        pd.DataFrame([(pd.Timestamp("2020-01-06"), "Z", "add")],
                     columns=["date", "ticker", "action"]),
        frozenset({"Z"}), idx)
    check("index equals requested daily_index", mem.index.equals(idx))
    check("index type matches data_pit frame",
          type(mem.index) is type(ref.index))
    check("all-bool dtypes like data_pit frame",
          (mem.dtypes == bool).all() and (ref.dtypes == bool).all())
    check("columns are normalized symbols",
          set(mem.columns) == {"AAA", "BRK-B", "CCC",
                               "SIVB-202303", "FOO-A-202006"})
    check("alive members True throughout", mem["AAA"].all())
    check("dead member True while listed, False after",
          bool(mem.at[pd.Timestamp("2023-03-10"), "SIVB-202303"])
          and not mem["SIVB-202303"].loc["2023-03-11":].any())
    counts = mem.sum(axis=1)
    check("daily counts inside synthetic band",
          SYN_COUNT_BAND[0] <= counts.min() and counts.max() <= SYN_COUNT_BAND[1],
          f"[{int(counts.min())}, {int(counts.max())}]")

    dl = delisting_events(dest)
    check("delisting_events has both dead names",
          set(dl["symbol"]) == {"SIVB-202303", "FOO-A-202006"})
    check("terminal prices joined",
          dl["terminal_unadj_close"].notna().all())


# ───── 5a. gate machinery: constructed all-PASS case ──────────────────────
def test_gates_pass(tmp: Path):
    print("\n[5a] trust gates: clean fixture -> all five PASS + marker")
    raw, _ = build_fixture(tmp / "t5")
    dest = tmp / "t5" / "house"
    ingest(raw, dest)
    out = tmp / "t5" / "results"

    res = trust_gates(cache_from_dest(dest), out_dir=out, **gate_args(dest))
    for name, ok in res["gates"].items():
        check(f"{name} PASS", ok,
              "" if ok else str(res["gate_details"][name][:3]))
    check("overall verdict PASS", res["verdict"] == "PASS")
    check("report written", (out / REPORT_NAME).exists())
    check("PASS marker written", (out / MARKER_NAME).exists())
    marker = json.loads((out / MARKER_NAME).read_text())
    check("marker records ingest sha",
          marker["ingest_record_sha256"]
          == hashlib.sha256((dest / "_ingest_record.json").read_bytes()).hexdigest())
    check("gate2 yf spot check recorded as SKIPPED",
          "SKIPPED" in marker["gate_metrics"]
          ["gate2_dividend_verification"]["yf_spot_check"])
    check("marker records gate2_spot_check_ran = False",
          marker["gate2_spot_check_ran"] is False)

    # a network=False marker must NOT satisfy the N1 guard silently
    try:
        require_trust_gates_pass(dest=dest, out_dir=out)
        check("guard refuses skipped-network marker", False, "no exception")
    except RuntimeError as e:
        check("guard refuses skipped-network marker",
              "gate2_spot_check_ran" in str(e))
    got = require_trust_gates_pass(dest=dest, out_dir=out,
                                   allow_skipped_network_check=True)
    check("explicit override accepts current marker",
          got["verdict"] == "PASS")

    # a legacy marker WITHOUT the field is fail-closed too
    legacy = {k: v for k, v in marker.items() if k != "gate2_spot_check_ran"}
    (out / MARKER_NAME).write_text(json.dumps(legacy, default=str))
    try:
        require_trust_gates_pass(dest=dest, out_dir=out)
        check("marker missing gate2_spot_check_ran refused", False,
              "no exception")
    except RuntimeError as e:
        check("marker missing gate2_spot_check_ran refused",
              "gate2_spot_check_ran" in str(e))
    (out / MARKER_NAME).write_text(json.dumps(marker, default=str))

    # staleness: re-ingest -> the marker must be rejected until gates re-run
    # (even with the network-check override — staleness is checked first)
    ingest(raw, dest, refresh=True)
    try:
        require_trust_gates_pass(dest=dest, out_dir=out,
                                 allow_skipped_network_check=True)
        check("stale marker rejected after re-ingest", False, "no exception")
    except RuntimeError as e:
        check("stale marker rejected after re-ingest", "STALE" in str(e))


# ───── 5b. gate 2 FAIL: broken dividend adjustment ────────────────────────
def test_gate2_fail_broken_dividend(tmp: Path):
    print("\n[5b] gate 2 FAIL: one name with a broken dividend adjustment")
    raw, _ = build_fixture(tmp / "t6", broken_dividend_symbol="AAA")
    dest = tmp / "t6" / "house"
    ingest(raw, dest)
    out = tmp / "t6" / "results"
    # pre-seed a stale marker: a FAIL run must remove it
    out.mkdir(parents=True)
    (out / MARKER_NAME).write_text('{"verdict": "PASS"}')

    res = trust_gates(cache_from_dest(dest), out_dir=out, **gate_args(dest))
    check("gate2 FAILS on broken adjustment",
          not res["gates"]["gate2_dividend_verification"])
    check("broken name identified in details",
          any("AAA" in d for d in
              res["gate_details"]["gate2_dividend_verification"]))
    check("other gates unaffected",
          res["gates"]["gate3_membership_crosscheck"]
          and res["gates"]["gate4_delisting_spot_checks"]
          and res["gates"]["gate5_panel_integrity"])
    check("overall verdict FAIL", res["verdict"] == "FAIL")
    check("stale marker removed on FAIL", not (out / MARKER_NAME).exists())
    try:
        require_trust_gates_pass(dest=dest, out_dir=out)
        check("require_trust_gates_pass raises on FAIL", False, "no exception")
    except RuntimeError:
        check("require_trust_gates_pass raises on FAIL", True)


# ───── 5c. gate 3 FAIL: 5-name membership discrepancy ─────────────────────
def test_gate3_fail_membership(tmp: Path):
    print("\n[5c] gate 3 FAIL: membership discrepancy of 5 names")
    raw, _ = build_fixture(tmp / "t7")
    dest = tmp / "t7" / "house"
    ingest(raw, dest)
    out = tmp / "t7" / "results"

    wiki_bad = synthetic_wiki_membership(dest)
    flip_days = wiki_bad.index[(wiki_bad.index >= "2021-03-01")
                               & (wiki_bad.index <= "2021-03-12")]
    wiki_bad.loc[flip_days, :] = ~wiki_bad.loc[flip_days, :]  # 5-name diff
    args = gate_args(dest)
    args["wiki_membership"] = wiki_bad

    res = trust_gates(cache_from_dest(dest), out_dir=out, **args)
    check("gate3 FAILS on 5-name symmetric diff",
          not res["gates"]["gate3_membership_crosscheck"])
    check("diff magnitude reported",
          any("diff max 5" in d for d in
              res["gate_details"]["gate3_membership_crosscheck"]))
    check("overall verdict FAIL", res["verdict"] == "FAIL")
    check("no marker written", not (out / MARKER_NAME).exists())

    # explained symbols shrink the diff below the threshold again
    args["explained"] = frozenset({"AAA", "BRK-B", "CCC"})  # 5 - 3 = 2 <= 3
    res2 = trust_gates(cache_from_dest(dest), out_dir=out, **args)
    check("explained exclusions bring gate3 back under threshold",
          res2["gates"]["gate3_membership_crosscheck"])


# ───── 5d. gate 1 and gate 4 FAIL cases ───────────────────────────────────
def test_gate1_gate4_fail(tmp: Path):
    print("\n[5d] gate 1 FAIL (return mismatch) and gate 4 FAIL (bad expectation)")
    raw, _ = build_fixture(tmp / "t8")
    dest = tmp / "t8" / "house"
    ingest(raw, dest)
    out = tmp / "t8" / "results"

    # cache whose returns disagree by ~20bp/day on every name
    bad_cache = cache_from_dest(dest)
    rng = np.random.default_rng(3)
    noisy = bad_cache["close"] * np.exp(
        rng.normal(0, 0.002, bad_cache["close"].shape).cumsum(axis=0))
    res = trust_gates({"close": noisy}, out_dir=out, **gate_args(dest))
    check("gate1 FAILS on shifted returns",
          not res["gates"]["gate1_overlap_return_diff"])
    check("overall verdict FAIL", res["verdict"] == "FAIL")

    # wrong month expectation -> gate 4 FAIL
    args = gate_args(dest)
    args["delisting_expectations"] = [
        {"base": "SIVB", "months": ["209901"], "terminal_band": None,
         "note": "wrong month on purpose"}]
    res2 = trust_gates(cache_from_dest(dest), out_dir=out, **args)
    check("gate4 FAILS on wrong expectation",
          not res2["gates"]["gate4_delisting_spot_checks"])
    check("gate4 failure names the symbol",
          any("SIVB" in d for d in
              res2["gate_details"]["gate4_delisting_spot_checks"]))


# ───── 5e. gate 3 FAIL: truncated membership frame (no silent skip) ───────
def test_gate3_truncated_membership(tmp: Path):
    print("\n[5e] gate 3 FAIL: truncated membership frame -> events "
          "unverifiable, never silently skipped")
    # membership export stops 2022-12-30: the SIVB events (2023-03) lie
    # inside the research window but OUTSIDE the frame -> gate 3 must FAIL.
    raw, _ = build_fixture(tmp / "t9", mem_end="2022-12-30")
    dest = tmp / "t9" / "house"
    ingest(raw, dest)
    out = tmp / "t9" / "results"

    res = trust_gates(cache_from_dest(dest), out_dir=out, **gate_args(dest))
    check("gate3 FAILS on truncated membership window",
          not res["gates"]["gate3_membership_crosscheck"])
    d3 = res["gate_details"]["gate3_membership_crosscheck"]
    check("failure names the truncated window / unverifiable events",
          any("membership export window too short" in d for d in d3),
          str(d3))
    check("unverifiable events are itemized with the frame window",
          any("UNVERIFIABLE" in d and "2022-12-30" in d for d in d3))
    check("overall verdict FAIL", res["verdict"] == "FAIL")
    check("no marker written", not (out / MARKER_NAME).exists())
    m3 = res["gate_metrics"]["gate3_membership_crosscheck"]
    check("checked < required recorded in metrics",
          m3["n_known_events_checked"] < m3["n_known_events_required"])

    # ... but a research window that itself excludes the missing events is
    # the legitimate skip: gate 3 passes again.
    args = gate_args(dest)
    args["research_window"] = ("2019-01-01", "2022-12-30")
    res2 = trust_gates(cache_from_dest(dest), out_dir=out, **args)
    check("narrowed research window exempts the excluded events",
          res2["gates"]["gate3_membership_crosscheck"],
          str(res2["gate_details"]["gate3_membership_crosscheck"]))


# ───── 6. refresh hygiene: stale artifacts removed, record stays clean ────
def test_refresh_removes_stale(tmp: Path):
    print("\n[6] ingest(refresh=True): stale artifacts removed, record from "
          "the expected set, gate 5 catches re-injection")
    raw, _ = build_fixture(tmp / "t10")
    dest = tmp / "t10" / "house"
    ingest(raw, dest)
    check("pre-refresh alive CCC ingested",
          (dest / "bars" / "CCC.parquet").exists())

    # VM re-export: the company delisted, CCC -> CCC-202307
    (raw / "bars" / "CCC.parquet").rename(raw / "bars" / "CCC-202307.parquet")
    write_manifest(raw)
    rec = ingest(raw, dest, refresh=True)

    check("stale CCC.parquet removed on refresh",
          not (dest / "bars" / "CCC.parquet").exists())
    check("stale CCC sidecar removed on refresh",
          not (dest / "bars" / "CCC.parquet.meta.json").exists())
    check("renamed CCC-202307.parquet present",
          (dest / "bars" / "CCC-202307.parquet").exists())
    check("record artifacts exclude the stale file",
          "bars/CCC.parquet" not in rec["artifacts"])
    check("record artifacts include the renamed file",
          "bars/CCC-202307.parquet" in rec["artifacts"])
    frozen = json.loads((dest / "_ingest_record.json").read_text())
    check("frozen record on disk matches (no rglob absorption)",
          "bars/CCC.parquet" not in frozen["artifacts"]
          and "bars/CCC-202307.parquet" in frozen["artifacts"])
    g5 = dn._gate5_integrity(dest)
    check("gate5 PASSES on the clean refreshed tree", g5["passed"],
          str(g5["details"][:5]))

    # manually re-inject the stale file -> gate 5 must FAIL (record mismatch)
    shutil.copy(dest / "bars" / "CCC-202307.parquet",
                dest / "bars" / "CCC.parquet")
    g5b = dn._gate5_integrity(dest)
    check("gate5 FAILS on re-injected stale file", not g5b["passed"])
    check("stale file flagged against sidecar/record",
          any("CCC.parquet" in d and
              ("not in ingest record" in d or "sidecar missing" in d)
              for d in g5b["details"]))


# ───── 7. network=True spot check satisfies the N1 guard ──────────────────
def test_network_marker_satisfies_guard(tmp: Path):
    print("\n[7] (stubbed) network=True marker records spot-check ran and "
          "satisfies require_trust_gates_pass")
    raw, _ = build_fixture(tmp / "t11")
    dest = tmp / "t11" / "house"
    ingest(raw, dest)
    out = tmp / "t11" / "results"

    orig = dn._gate2_yfinance_spot_check
    dn._gate2_yfinance_spot_check = \
        lambda d, names, n: ("PASS (20/20 < 5bp)", ["  yf stub"], True)
    try:
        args = gate_args(dest)
        args["network"] = True
        res = trust_gates(cache_from_dest(dest), out_dir=out, **args)
    finally:
        dn._gate2_yfinance_spot_check = orig

    check("verdict PASS with spot check run", res["verdict"] == "PASS")
    marker = json.loads((out / MARKER_NAME).read_text())
    check("marker records gate2_spot_check_ran = True",
          marker["gate2_spot_check_ran"] is True)
    got = require_trust_gates_pass(dest=dest, out_dir=out)
    check("guard accepts network marker WITHOUT override",
          got["verdict"] == "PASS")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="norgate_fixture_"))
    try:
        test_manifest_and_freeze(tmp)
        test_normalization(tmp)
        test_schema_and_loader(tmp)
        test_membership_frame(tmp)
        test_gates_pass(tmp)
        test_gate2_fail_broken_dividend(tmp)
        test_gate3_fail_membership(tmp)
        test_gate1_gate4_fail(tmp)
        test_gate3_truncated_membership(tmp)
        test_refresh_removes_stale(tmp)
        test_network_marker_satisfies_guard(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
