"""Point-in-time S&P 500 membership from Wikipedia's constituent page.

Why this exists
---------------
The local bar cache is survivors-only (see data.py docstring): every
backtest built on load_panel is an upper bound because names that delisted,
blew up, or were acquired mid-window are absent. This module reconstructs
WHO WAS IN THE INDEX ON EACH DAY from Wikipedia's
"List of S&P 500 companies" page — the current-constituents table plus the
"Selected changes" log (402 change rows, 1976->present as of the frozen
scrape). That membership frame is the `eligible` mask for the WS1c
survivorship-bound experiments and the input to the missing-name report
that quantifies how much of the true index the cache cannot see.

Method: backward reconstruction. Start from the CURRENT constituents and
walk the change log newest -> oldest, undoing each event: a name ADDED on
date e was NOT a member before e; a name REMOVED on date e WAS a member
before e. Events take effect ON their listed date (verified against the
TSLA 2020-12-21 add and nine other well-documented events — see
KNOWN_EVENTS).

Provenance discipline
---------------------
The raw HTML is persisted (data_pit/raw/) and every derived parquet gets a
.meta.json sidecar carrying the source URL, scrape timestamp and sha256.
Once fetched, artifacts are FROZEN: fetch_sp500_events() reads the parquet
and never touches the network unless refresh=True. Wikipedia is editable —
the sha256'd raw HTML is the audit trail for what was actually scraped.

Known limitations
-----------------
- Wikipedia's log is titled "Selected changes": pre-2016 coverage is
  sparse (it cannot reconstruct deep history), but within the research
  window (2016-04-01..2026-03-27) the reconstructed daily count stays in
  [503, 507] — consistent with the real index (503 share classes, briefly
  more around multi-class adds), so the log is effectively complete there.
- Ticker renames are NOT membership events. Where the log records an event
  under an old symbol (FB add, FLT add, ...) the RENAMES table maps it to
  the canonical symbol so the company's membership is continuous.
- Same-day add+remove of the SAME ticker means ticker reuse across two
  corporate entities (FOX/FOXA 2019-03-19, GAS 2011-12-12); for index
  membership that is a no-op and both rows are dropped before the walk.
"""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PIT_DIR = ROOT / "data_pit"
RAW_DIR = PIT_DIR / "raw"
RESULTS_PIT_DIR = ROOT / "results" / "v7" / "pit"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia 403s the default urllib UA; a browser-like string is required.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

EVENTS_PARQUET = PIT_DIR / "sp500_events.parquet"
CURRENT_PARQUET = PIT_DIR / "sp500_current.parquet"

# Reconstructed daily count must stay inside this band or membership_frame
# raises: the real index holds 500 companies / ~503 share classes, and any
# excursion means the change log has a gap or a rename was mis-handled.
COUNT_BAND = (495, 510)

# Renames-not-events: old ticker -> canonical ticker. Canonical follows the
# local bar cache / in-window trading symbol (yfinance style, '-' share
# classes). Every entry below was verified against the frozen scrape:
# the change log carries the event under the OLD symbol while the
# current-constituents table lists the NEW one (or, for ECHO/FI, the wiki
# page renamed a symbol AFTER the bar-cache snapshot).
#   FB   -> META  add logged as FB 2013-12-23; current table has META
#   ANTM -> ELV   no event either side (rename 2022-06); ELV in current
#   WLTW -> WTW   add logged as WLTW 2016-01-05; current table has WTW
#   FLT  -> CPAY  add logged as FLT 2018-06-20; current has CPAY, same date
#   CDAY -> DAY   add logged as CDAY 2021-09-20; current table has DAY
#   HRS  -> LHX   add logged as HRS 2008-09-16; current table has LHX
#   UTX  -> RTX   RTN removed 2020-04-06 but UTX->RTX rename has no event
#   ECHO -> SATS  add logged as SATS 2026-03-23; wiki current shows ECHO —
#                 the cache and the in-window symbol are SATS, so SATS is
#                 canonical here (reverse of the usual direction)
#   FI   -> FISV  wiki current + cache both say FISV; FI mapped for safety
#                 in case a future scrape flips the symbol
# MAINTENANCE: after any refresh=True re-scrape, re-run the __main__ block;
# a new dangling add (ticker added, never removed, absent from the current
# table) or a removed-but-current ticker signals a rename missing here.
RENAMES: dict[str, str] = {
    "FB": "META",
    "ANTM": "ELV",
    "WLTW": "WTW",
    "FLT": "CPAY",
    "CDAY": "DAY",
    "HRS": "LHX",
    "UTX": "RTX",
    "ECHO": "SATS",
    "FI": "FISV",
}

# Ten well-documented membership events, each CONFIRMED in the frozen
# scrape (change-log row cited in the comment). Asserted after every
# reconstruction as (ticker, date, expected_membership) — the day before an
# event and the event day itself, so both the direction and the
# effective-ON-date convention are pinned.
KNOWN_EVENTS: list[tuple[str, str, bool]] = [
    # TSLA added 2020-12-21 (log row: add TSLA / rem AIV)
    ("TSLA", "2020-12-18", False), ("TSLA", "2020-12-21", True),
    # SIVB removed 2023-03-15 (FDIC receivership; add PODD)
    ("SIVB", "2023-03-14", True), ("SIVB", "2023-03-15", False),
    # FRC removed 2023-05-04 (FDIC receivership; add AXON)
    ("FRC", "2023-05-03", True), ("FRC", "2023-05-04", False),
    # TWTR removed 2022-11-01 (Musk acquisition; add ACGL)
    ("TWTR", "2022-10-31", True), ("TWTR", "2022-11-01", False),
    # CELG removed 2019-11-21 (BMY acquisition; add NOW)
    ("CELG", "2019-11-20", True), ("CELG", "2019-11-21", False),
    # ETSY added 2020-09-21 (rem HRB); later removed 2024-09-23
    ("ETSY", "2020-09-18", False), ("ETSY", "2020-09-21", True),
    # MRNA added 2021-07-21 (rem ALXN)
    ("MRNA", "2021-07-20", False), ("MRNA", "2021-07-21", True),
    # CEG added 2022-02-02 (Exelon spin-off; add-only row)
    ("CEG", "2022-02-01", False), ("CEG", "2022-02-02", True),
    # ATVI removed 2023-10-18 (Microsoft acquisition; add LULU)
    ("ATVI", "2023-10-17", True), ("ATVI", "2023-10-18", False),
    # Fiserv: no in-window event; member throughout via the current table
    # (wiki current lists FISV, date_added 2001-04-02)
    ("FISV", "2020-06-01", True),
    ("FISV", "2016-04-01", True),
]


# ---------------------------------------------------------------------------
# provenance helpers

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_meta(artifact: Path, extra: dict) -> None:
    """Sidecar {artifact}.meta.json: source, timestamps, sha256 of artifact."""
    meta = {
        "artifact": artifact.name,
        "sha256": _sha256_file(artifact),
        "source": WIKI_URL,
        **extra,
    }
    with open(artifact.with_suffix(artifact.suffix + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# normalization

def normalize_ticker(t: str) -> str:
    """Wikipedia symbol -> canonical bar-cache symbol.

    '.' share-class separators become '-' (BRK.B -> BRK-B, yfinance style),
    then RENAMES maps stale symbols to the canonical one. Events keep RAW
    tickers on disk; normalization happens at reconstruction time so a
    RENAMES update never requires a re-scrape.
    """
    t = str(t).strip().upper().replace(".", "-")
    return RENAMES.get(t, t)


# ---------------------------------------------------------------------------
# fetch + freeze

def _fetch_html() -> str:
    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def fetch_sp500_events(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events, current) frames, scraping Wikipedia at most once.

    Frozen semantics: if the parquets already exist and refresh is False,
    they are read back verbatim — no network. refresh=True re-scrapes,
    overwrites the parquets and drops a new dated raw-HTML snapshot.

    events schema : date, ticker (RAW wiki symbol), action ('add'|'remove'),
                    security_name, reason, source, scraped_at
    current schema: ticker (RAW wiki symbol), security_name, gics_sector,
                    gics_sub_industry, date_added, cik, source, scraped_at
    """
    if EVENTS_PARQUET.exists() and CURRENT_PARQUET.exists() and not refresh:
        return pd.read_parquet(EVENTS_PARQUET), pd.read_parquet(CURRENT_PARQUET)

    PIT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    scraped_at = datetime.now(timezone.utc).isoformat()
    html = _fetch_html()
    raw_path = RAW_DIR / f"sp500_page_{datetime.now():%Y%m%d}.html"
    raw_path.write_text(html)
    _write_meta(raw_path, {"scraped_at": scraped_at})

    tables = pd.read_html(io.StringIO(html))
    # Table 0 = current constituents, table 1 = "Selected changes" log;
    # identified by shape, asserted by column content below.
    cur_raw, chg_raw = tables[0].copy(), tables[1].copy()
    if "Symbol" not in cur_raw.columns:
        raise ValueError("Wikipedia layout changed: table 0 lacks 'Symbol'")
    if chg_raw.shape[1] != 6:
        raise ValueError("Wikipedia layout changed: change table not 6 cols")

    chg_raw.columns = ["date", "add_ticker", "add_name",
                       "rem_ticker", "rem_name", "reason"]
    chg_raw["date"] = pd.to_datetime(chg_raw["date"], format="%B %d, %Y")

    rows = []
    for _, r in chg_raw.iterrows():
        if pd.notna(r["add_ticker"]):
            rows.append((r["date"], str(r["add_ticker"]).strip(), "add",
                         r["add_name"], r["reason"]))
        if pd.notna(r["rem_ticker"]):
            rows.append((r["date"], str(r["rem_ticker"]).strip(), "remove",
                         r["rem_name"], r["reason"]))
    events = pd.DataFrame(
        rows, columns=["date", "ticker", "action", "security_name", "reason"]
    ).sort_values(["date", "ticker"]).reset_index(drop=True)
    events["source"] = WIKI_URL
    events["scraped_at"] = scraped_at

    current = pd.DataFrame({
        "ticker": cur_raw["Symbol"].astype(str).str.strip(),
        "security_name": cur_raw["Security"],
        "gics_sector": cur_raw["GICS Sector"],
        "gics_sub_industry": cur_raw["GICS Sub-Industry"],
        "date_added": pd.to_datetime(cur_raw["Date added"], errors="coerce"),
        "cik": cur_raw["CIK"],
    })
    current["source"] = WIKI_URL
    current["scraped_at"] = scraped_at

    events.to_parquet(EVENTS_PARQUET)
    _write_meta(EVENTS_PARQUET, {"scraped_at": scraped_at,
                                 "raw_html": raw_path.name,
                                 "raw_html_sha256": _sha256_file(raw_path),
                                 "n_change_rows": int(len(chg_raw)),
                                 "n_events": int(len(events))})
    current.to_parquet(CURRENT_PARQUET)
    _write_meta(CURRENT_PARQUET, {"scraped_at": scraped_at,
                                  "raw_html": raw_path.name,
                                  "n_constituents": int(len(current))})
    return events, current


# ---------------------------------------------------------------------------
# reconstruction (pure — unit-testable without network)

def reconstruct_membership(
    events: pd.DataFrame,
    current_members: frozenset[str],
    daily_index: pd.DatetimeIndex,
    count_band: tuple[int, int] | None = None,
    spot_checks: list[tuple[str, str, bool]] | None = None,
) -> pd.DataFrame:
    """Backward reconstruction: bool DataFrame (daily_index x ever-members).

    Walk events newest -> oldest from `current_members`, undoing each one:
    undo of an add = the ticker was NOT a member before that date; undo of
    a remove = it WAS. Events are effective ON their date. Same-day
    add+remove pairs of one (normalized) ticker are ticker reuse, a
    membership no-op, and are dropped before the walk.

    Guards (both skippable with None so synthetic tests can use tiny
    universes): count_band raises if any daily member count leaves the
    band; spot_checks raises on any (ticker, date, expected) mismatch —
    checks whose date falls outside daily_index are skipped, not failed.
    """
    ev = events[["date", "ticker", "action"]].copy()
    ev["ticker"] = ev["ticker"].map(normalize_ticker)
    ev = ev.drop_duplicates()
    # ticker reuse: same normalized ticker added AND removed the same day
    nuniq = ev.groupby(["date", "ticker"])["action"].transform("nunique")
    ev = ev[nuniq == 1]

    daily_index = pd.DatetimeIndex(daily_index).sort_values()
    ev_desc = ev.sort_values("date", ascending=False)
    ev_list = list(ev_desc.itertuples(index=False))

    members = set(current_members)
    ever: set[str] = set(members) | set(ev["ticker"])
    snapshots: dict[pd.Timestamp, frozenset[str]] = {}
    ei = 0
    for d in daily_index[::-1]:
        while ei < len(ev_list) and ev_list[ei].date > d:
            e = ev_list[ei]
            if e.action == "add":
                members.discard(e.ticker)
            elif e.action == "remove":
                members.add(e.ticker)
            else:
                raise ValueError(f"unknown action {e.action!r}")
            ei += 1
        snapshots[d] = frozenset(members)

    cols = sorted(t for t in ever
                  if any(t in s for s in snapshots.values()))
    arr = np.zeros((len(daily_index), len(cols)), dtype=bool)
    col_ix = {t: j for j, t in enumerate(cols)}
    for i, d in enumerate(daily_index):
        for t in snapshots[d]:
            arr[i, col_ix[t]] = True
    mem = pd.DataFrame(arr, index=daily_index, columns=cols)

    if count_band is not None:
        _guard_count_band(mem, count_band)
    if spot_checks is not None:
        _guard_spot_checks(mem, spot_checks)
    return mem


def _guard_count_band(mem: pd.DataFrame, band: tuple[int, int]) -> None:
    counts = mem.sum(axis=1)
    lo, hi = band
    bad = counts[(counts < lo) | (counts > hi)]
    if len(bad):
        raise ValueError(
            f"membership count left [{lo}, {hi}] on {len(bad)} days; "
            f"first: {bad.index[0].date()} -> {int(bad.iloc[0])} members. "
            f"Change log gap or unmapped rename — see RENAMES MAINTENANCE."
        )


def _guard_spot_checks(mem: pd.DataFrame,
                       checks: list[tuple[str, str, bool]]) -> None:
    failures = []
    for ticker, date, expected in checks:
        d = pd.Timestamp(date)
        if d < mem.index[0] or d > mem.index[-1]:
            continue  # check outside the reconstructed window
        # spot dates are business days; exact match required
        if d not in mem.index:
            failures.append(f"{ticker}@{date}: date not in index")
            continue
        got = bool(mem.at[d, ticker]) if ticker in mem.columns else False
        if got != expected:
            failures.append(f"{ticker}@{date}: expected {expected}, got {got}")
    if failures:
        raise AssertionError(
            "known-event spot checks failed: " + "; ".join(failures)
        )


# ---------------------------------------------------------------------------
# public frame builders

def membership_frame(
    daily_index: pd.DatetimeIndex,
    tickers: list[str] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Bool DataFrame (daily_index x normalized ever-member tickers).

    Cached to data_pit/membership_{hash}.parquet where the hash covers the
    frozen events parquet's sha256 AND the index fingerprint, so a
    re-scrape or a different window can never serve a stale frame. Guards
    (count band + known-event spot checks) run on every call, cached or
    not — they are cheap and they are the point.
    """
    events, current = fetch_sp500_events(refresh=refresh)

    idx = pd.DatetimeIndex(daily_index).sort_values()
    # RENAMES is part of the reconstruction logic: a mapping fix changes the
    # output without touching the events file, so it must be in the cache key.
    renames_tag = json.dumps(sorted(RENAMES.items()))
    fingerprint = hashlib.sha256(
        (_sha256_file(EVENTS_PARQUET)
         + f"|{idx[0].isoformat()}|{idx[-1].isoformat()}|{len(idx)}"
         + f"|{renames_tag}"
         ).encode()
    ).hexdigest()[:12]
    cache_path = PIT_DIR / f"membership_{fingerprint}.parquet"

    if cache_path.exists():
        mem = pd.read_parquet(cache_path)
        mem.index = pd.to_datetime(mem.index)
        _guard_count_band(mem, COUNT_BAND)
        _guard_spot_checks(mem, KNOWN_EVENTS)
    else:
        cur_set = frozenset(normalize_ticker(t) for t in current["ticker"])
        mem = reconstruct_membership(events, cur_set, idx,
                                     count_band=COUNT_BAND,
                                     spot_checks=KNOWN_EVENTS)
        mem.to_parquet(cache_path)
        _write_meta(cache_path, {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "events_sha256": _sha256_file(EVENTS_PARQUET),
            "index_start": idx[0], "index_end": idx[-1],
            "n_days": int(len(idx)), "n_ever_members": int(mem.shape[1]),
        })

    if tickers is not None:
        mem = mem.reindex(columns=list(tickers), fill_value=False)
    return mem


def universe_pit(date: str | pd.Timestamp,
                 membership: pd.DataFrame | None = None) -> frozenset[str]:
    """Members on `date` (as-of: last membership row <= date)."""
    if membership is None:
        d = pd.Timestamp(date)
        membership = membership_frame(pd.bdate_range(d - pd.Timedelta(days=7), d))
    row = membership.loc[:pd.Timestamp(date)]
    if row.empty:
        raise ValueError(f"{date} precedes the membership index")
    last = row.iloc[-1]
    return frozenset(last.index[last])


# ---------------------------------------------------------------------------
# missing-name report

def missing_name_report(
    close_panel: pd.DataFrame,
    membership: pd.DataFrame,
    rebal_freq: int = 21,
) -> pd.DataFrame:
    """Per rebalance date: how much of the true index the cache can see.

    Samples every `rebal_freq`-th TRADING day (membership dates intersected
    with the close panel's index — bdate_range holidays would otherwise show
    as spurious missing_rate=1.0 rows, since no name has bars on a holiday).
    A member "has bars" if the close panel holds a non-NaN close for it ON
    that date. Columns: date, n_members, n_with_bars, missing_rate,
    missing_names (';'-joined, sorted).
    """
    rows = []
    trading_days = membership.index.intersection(close_panel.index)
    for d in trading_days[::rebal_freq]:
        members = set(membership.columns[membership.loc[d]])
        have = set(close_panel.columns[close_panel.loc[d].notna()])
        missing = sorted(members - have)
        rows.append({
            "date": d,
            "n_members": len(members),
            "n_with_bars": len(members) - len(missing),
            "missing_rate": len(missing) / len(members) if members else np.nan,
            "missing_names": ";".join(missing),
        })
    return pd.DataFrame(rows)


def write_missing_name_report(
    close_panel: pd.DataFrame,
    membership: pd.DataFrame,
    rebal_freq: int = 21,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Build the report and persist it to results/v7/pit/missing_names.csv."""
    report = missing_name_report(close_panel, membership, rebal_freq)
    path = out_path or (RESULTS_PIT_DIR / "missing_names.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return report


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    START, END = "2016-04-01", "2026-03-27"

    events, current = fetch_sp500_events()
    print(f"events: {len(events)} rows "
          f"({events['date'].min().date()} -> {events['date'].max().date()}), "
          f"current constituents: {len(current)}")

    idx = pd.bdate_range(START, END)
    mem = membership_frame(idx)
    counts = mem.sum(axis=1)
    print(f"membership: {mem.shape[0]} days x {mem.shape[1]} ever-members, "
          f"daily count in [{int(counts.min())}, {int(counts.max())}] "
          f"(guard band {COUNT_BAND}) — all {len(KNOWN_EVENTS)} spot checks passed")

    from exp_lib import load_cache  # cheap path to the real close panel
    panel, _, _ = load_cache()
    report = write_missing_name_report(panel["close"], mem)
    print(f"missing-name report: {len(report)} rebalance dates "
          f"-> {RESULTS_PIT_DIR / 'missing_names.csv'}")
    for probe in ["2016-04-01", "2020-01-01", "2024-01-01"]:
        r = report[report["date"] >= probe].iloc[0]
        print(f"  {r['date'].date()}: {r['n_members']} members, "
              f"{r['n_with_bars']} with bars, "
              f"missing {r['missing_rate']:.1%}")
    print(f"  overall missing_rate: first={report['missing_rate'].iloc[0]:.1%} "
          f"last={report['missing_rate'].iloc[-1]:.1%}")
