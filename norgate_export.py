"""norgate_export.py — one-time Norgate Data export, run INSIDE the Windows VM.

THIS SCRIPT RUNS IN THE WINDOWS VM WITH NORGATE DATA UPDATER (NDU) RUNNING.
It cannot run on macOS: the `norgatedata` package talks to NDU on the same
machine. On macOS only the pure helpers at the top of this file are
importable (they are unit-tested by validation/test_norgate_export_helpers.py
without any Norgate dependency).

What it produces (plan Phase N0, prereg results/v7/norgate/N1_PREREG.md):

    <out>/bars/<SYMBOL>.parquet        one file per symbol; TOTALRETURN-
                                       adjusted OHLCV + Unadjusted Close +
                                       Dividend, padding NONE, from 2010-01-01
    <out>/membership/<TAG>.parquet     boolean date x symbol frame per index
                                       (sp500/sp400/sp600/sp1500/r1000/r3000)
    <out>/delistings.parquet           delisted (-YYYYMM) symbols: last quoted
                                       date, final adjusted/unadjusted close,
                                       security name
    <out>/manifest.json                export timestamp, norgatedata version,
                                       settings, per-file sha256 + row counts

Universe = union of the two survivorship-free watchlists:
    "Russell 3000 Current & Past" + "S&P Composite 1500 Current & Past"

Usage (inside the VM, NDU running, after `pip install norgatedata pandas pyarrow`):

    python norgate_export.py --out D:\\shared\\data_norgate_raw
    python norgate_export.py --out ... --limit 25        # smoke run
    python norgate_export.py --out ... --refresh         # re-pull everything

Resume-safe: existing per-symbol bar files are skipped unless --refresh.
Membership frames, delistings.parquet and manifest.json are always rebuilt
at the end of a run (they are single files, cheap relative to the bar pull).

Self-test of the pure helpers (works anywhere, no norgatedata needed):

    python norgate_export.py --selftest

LICENSING: the exported data is licensed to this machine's subscriber only.
It must never enter a shared/public repo. data_norgate_raw/ and
data_norgate/ are in the research repo's .gitignore.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants (settings frozen by the plan / N1 prereg — do not tune)

START_DATE = "2010-01-01"

WATCHLISTS = [
    "Russell 3000 Current & Past",
    "S&P Composite 1500 Current & Past",
]

# CANDIDATE index name strings. Norgate's exact naming MUST be verified in
# the Norgate Data Viewer at run time (Databases -> US Indices). If any name
# below returns ZERO constituents across the whole universe, the export
# aborts loudly (see _check_membership_nonzero) and you must correct the
# string here to match the Viewer before re-running.
INDEX_NAMES = {
    "sp500":  "S&P 500",
    "sp400":  "S&P MidCap 400",
    "sp600":  "S&P SmallCap 600",
    "sp1500": "S&P Composite 1500",
    "r1000":  "Russell 1000",
    "r3000":  "Russell 3000",
}

# Bar columns kept (order preserved where present). Norgate pandas frames
# carry a DatetimeIndex named 'Date'; extra columns (e.g. Turnover) dropped.
BAR_COLUMNS = ["Open", "High", "Low", "Close", "Volume",
               "Unadjusted Close", "Dividend"]

PROGRESS_EVERY = 500

# ---------------------------------------------------------------------------
# Pure helper layer — importable and testable WITHOUT norgatedata.
# ---------------------------------------------------------------------------

_DELIST_RE = re.compile(r"^(?P<base>.+)-(?P<ym>\d{6})$")
_UNSAFE_FN = re.compile(r'[\\/:*?"<>|]')


class ParsedSymbol(NamedTuple):
    symbol: str            # full Norgate symbol, unchanged (unique key)
    base: str              # symbol without any -YYYYMM delisting suffix
    delist_yyyymm: str | None  # 'YYYYMM' string, or None if listed
    is_delisted: bool


def parse_norgate_symbol(symbol: str) -> ParsedSymbol:
    """Split a Norgate symbol into (base, delisting suffix).

    Delisted symbols carry a -YYYYMM suffix (e.g. 'SIVB-202303').
    Share classes use '.' in Norgate ('BRK.B'), so a trailing '-' + 6 digits
    is unambiguous. Month must be 01-12 to count as a delisting suffix.
    """
    s = str(symbol).strip()
    m = _DELIST_RE.match(s)
    if m:
        ym = m.group("ym")
        month = int(ym[4:6])
        if 1 <= month <= 12:
            return ParsedSymbol(s, m.group("base"), ym, True)
    return ParsedSymbol(s, s, None, False)


def symbol_to_filename(symbol: str) -> str:
    """Norgate symbol -> safe parquet stem (Windows + macOS).

    '.' is kept (BRK.B -> 'BRK.B.parquet' is legal on both platforms);
    genuinely unsafe filename characters become '_'. Deterministic, so
    resume checks and the macOS loader agree on names.
    """
    return _UNSAFE_FN.sub("_", str(symbol).strip())


def build_membership_frame(series_by_symbol: dict[str, pd.Series]) -> pd.DataFrame:
    """dict {symbol: 0/1 Series indexed by date} -> boolean date x symbol frame.

    Union date index, sorted; missing dates for a symbol = False (not a
    member). Columns sorted for deterministic file bytes. Empty input ->
    empty frame.
    """
    if not series_by_symbol:
        return pd.DataFrame(dtype=bool)
    frame = pd.DataFrame({sym: s.astype(float) for sym, s in
                          sorted(series_by_symbol.items())})
    frame = frame.sort_index()
    return frame.fillna(0.0).astype(bool)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_row_count(path: Path) -> int | None:
    """Row count from parquet metadata (cheap, no full read); None if not parquet."""
    if path.suffix != ".parquet":
        return None
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def file_entry(path: Path, base_dir: Path) -> dict:
    """Manifest entry for one artifact: relative path, sha256, bytes, rows."""
    entry = {
        "path": path.relative_to(base_dir).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    rows = parquet_row_count(path)
    if rows is not None:
        entry["rows"] = rows
    return entry


def build_manifest(out_dir: Path, settings: dict, counts: dict,
                   norgatedata_version: str) -> dict:
    """Scan <out_dir> for data artifacts and assemble the manifest dict.

    Pure with respect to Norgate: works on any directory tree of files
    (unit-tested against synthetic fixtures). manifest.json itself is
    excluded from the file list.
    """
    files = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files.append(file_entry(p, out_dir))
    return {
        "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "norgatedata_version": norgatedata_version,
        "settings": settings,
        "counts": counts,
        "n_files": len(files),
        "files": files,
    }


# ---------------------------------------------------------------------------
# VM-side export (requires norgatedata + NDU running)
# ---------------------------------------------------------------------------

def _import_norgatedata():
    try:
        import norgatedata  # noqa: F401
        return norgatedata
    except ImportError as e:
        sys.exit(
            "ERROR: the 'norgatedata' package is not importable.\n"
            "This script runs INSIDE the Windows VM with the Norgate Data\n"
            "Updater (NDU) running on the same machine. It cannot run on\n"
            "macOS. Inside the VM: pip install norgatedata pandas pyarrow,\n"
            "start NDU, log in, wait for the database download, then re-run.\n"
            f"(import error: {e})"
        )


def _get_universe(nd, limit: int | None) -> list[str]:
    symbols: set[str] = set()
    for wl in WATCHLISTS:
        wl_syms = nd.watchlist_symbols(wl)
        if not wl_syms:
            sys.exit(
                f"ERROR: watchlist '{wl}' returned no symbols. Verify the\n"
                "exact watchlist name in the Norgate Data Viewer and check\n"
                "that your subscription level includes 'Current & Past'\n"
                "watchlists (US Platinum required)."
            )
        print(f"  watchlist '{wl}': {len(wl_syms)} symbols")
        symbols.update(wl_syms)
    universe = sorted(symbols)
    print(f"  union universe: {len(universe)} symbols")
    if limit is not None:
        universe = universe[:limit]
        print(f"  --limit {limit}: truncated to {len(universe)} symbols")
    return universe


def _fetch_bars(nd, symbol: str) -> pd.DataFrame | None:
    df = nd.price_timeseries(
        symbol,
        stock_price_adjustment_setting=nd.StockPriceAdjustmentType.TOTALRETURN,
        padding_setting=nd.PaddingType.NONE,
        start_date=START_DATE,
        timeseriesformat="pandas-dataframe",
    )
    if df is None or len(df) == 0:
        return None
    keep = [c for c in BAR_COLUMNS if c in df.columns]
    if "Close" not in keep:
        print(f"  WARNING: {symbol}: no 'Close' column "
              f"(columns={list(df.columns)}); skipped")
        return None
    return df[keep]


def _export_bars(nd, universe: list[str], bars_dir: Path,
                 refresh: bool) -> tuple[dict[str, Path], list[str]]:
    """Pull per-symbol bars. Returns ({symbol: written path}, [empty symbols])."""
    bars_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    empty: list[str] = []
    skipped = 0
    for i, sym in enumerate(universe, 1):
        path = bars_dir / f"{symbol_to_filename(sym)}.parquet"
        if path.exists() and not refresh:
            written[sym] = path
            skipped += 1
        else:
            df = _fetch_bars(nd, sym)
            if df is None:
                empty.append(sym)
            else:
                df.to_parquet(path)
                written[sym] = path
        if i % PROGRESS_EVERY == 0 or i == len(universe):
            print(f"  bars {i}/{len(universe)} "
                  f"(written {len(written) - skipped}, resumed-skip {skipped}, "
                  f"empty {len(empty)})", flush=True)
    return written, empty


def _export_delistings(nd, universe: list[str], bars: dict[str, Path],
                       out_dir: Path) -> pd.DataFrame:
    rows = []
    for sym in universe:
        parsed = parse_norgate_symbol(sym)
        if not parsed.is_delisted:
            continue
        try:
            lqd = nd.last_quoted_date(sym)
        except Exception as e:  # keep the export alive; record the gap
            print(f"  WARNING: last_quoted_date({sym}) failed: {e}")
            lqd = None
        try:
            name = nd.security_name(sym)
        except Exception:
            name = None
        final_adj = final_unadj = None
        path = bars.get(sym)
        if path is not None and path.exists():
            df = pd.read_parquet(path)
            if len(df):
                last = df.iloc[-1]
                final_adj = float(last["Close"]) if "Close" in df.columns else None
                if "Unadjusted Close" in df.columns:
                    final_unadj = float(last["Unadjusted Close"])
        rows.append({
            "symbol": sym,
            "base_symbol": parsed.base,
            "delist_yyyymm": parsed.delist_yyyymm,
            "last_quoted_date": str(lqd) if lqd is not None else None,
            "final_adjusted_close": final_adj,
            "final_unadjusted_close": final_unadj,
            "security_name": name,
        })
    delist = pd.DataFrame(
        rows, columns=["symbol", "base_symbol", "delist_yyyymm",
                       "last_quoted_date", "final_adjusted_close",
                       "final_unadjusted_close", "security_name"])
    delist.to_parquet(out_dir / "delistings.parquet")
    print(f"  delistings.parquet: {len(delist)} delisted symbols")
    return delist


def _export_membership(nd, universe: list[str], mem_dir: Path) -> dict[str, int]:
    """index_constituent_timeseries per symbol per index -> boolean frames."""
    mem_dir.mkdir(parents=True, exist_ok=True)
    collected: dict[str, dict[str, pd.Series]] = {t: {} for t in INDEX_NAMES}
    for i, sym in enumerate(universe, 1):
        for tag, index_name in INDEX_NAMES.items():
            try:
                ts = nd.index_constituent_timeseries(
                    sym, index_name,
                    padding_setting=nd.PaddingType.NONE,
                    timeseriesformat="pandas-dataframe",
                )
            except Exception as e:
                print(f"  WARNING: index_constituent_timeseries({sym}, "
                      f"'{index_name}') failed: {e}")
                continue
            if ts is None or len(ts) == 0:
                continue
            col = "Index Constituent" if "Index Constituent" in ts.columns \
                else ts.columns[0]
            s = ts[col]
            if (s != 0).any():
                collected[tag][sym] = s
        if i % PROGRESS_EVERY == 0 or i == len(universe):
            sizes = {t: len(d) for t, d in collected.items()}
            print(f"  membership {i}/{len(universe)} members-so-far {sizes}",
                  flush=True)

    counts = {}
    for tag, series_by_symbol in collected.items():
        frame = build_membership_frame(series_by_symbol)
        frame.to_parquet(mem_dir / f"{tag}.parquet")
        counts[tag] = frame.shape[1]
        print(f"  membership/{tag}.parquet: {frame.shape[1]} symbols x "
              f"{frame.shape[0]} dates  (index name '{INDEX_NAMES[tag]}')")
    _check_membership_nonzero(counts)
    return counts


def _check_membership_nonzero(counts: dict[str, int]) -> None:
    """LOUD runtime check: every candidate index name must have matched
    at least one constituent, otherwise the name string is wrong."""
    dead = [t for t, n in counts.items() if n == 0]
    if dead:
        sys.exit(
            "ERROR: these indices returned ZERO constituents across the "
            f"whole universe: {dead}.\n"
            f"Candidate names used: "
            f"{ {t: INDEX_NAMES[t] for t in dead} }\n"
            "The index name strings in INDEX_NAMES do not match Norgate's.\n"
            "Open the Norgate Data Viewer -> US Indices, find the exact\n"
            "index names, correct INDEX_NAMES in this script, and re-run\n"
            "(bar files are resume-safe; membership rebuilds)."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Norgate one-time export (run inside the Windows VM "
                    "with NDU running). See module docstring.")
    ap.add_argument("--out", default="data_norgate_raw",
                    help="output directory (e.g. the UTM shared folder)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke run: only the first N symbols")
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull symbols even if their parquet exists")
    ap.add_argument("--selftest", action="store_true",
                    help="run pure-helper unit tests (no norgatedata needed)")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    nd = _import_norgatedata()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print(f"Norgate export -> {out_dir.resolve()}  "
          f"(norgatedata {nd.__version__})")

    print("[1/5] universe")
    universe = _get_universe(nd, args.limit)

    print("[2/5] per-symbol bars (TOTALRETURN, padding NONE, "
          f"start {START_DATE})")
    bars, empty = _export_bars(nd, universe, out_dir / "bars", args.refresh)

    print("[3/5] delistings")
    delist = _export_delistings(nd, universe, bars, out_dir)

    print("[4/5] index membership")
    mem_counts = _export_membership(nd, universe, out_dir / "membership")

    print("[5/5] manifest.json")
    settings = {
        "start_date": START_DATE,
        "watchlists": WATCHLISTS,
        "index_names": INDEX_NAMES,
        "adjustment": "StockPriceAdjustmentType.TOTALRETURN",
        "padding": "PaddingType.NONE",
        "bar_columns": BAR_COLUMNS,
        "limit": args.limit,
        "refresh": bool(args.refresh),
        "started_utc": started,
    }
    counts = {
        "universe_symbols": len(universe),
        "bar_files": len(bars),
        "empty_symbols": len(empty),
        "delisted_symbols": int(len(delist)),
        "membership_symbols": mem_counts,
    }
    manifest = build_manifest(out_dir, settings, counts, nd.__version__)
    manifest["empty_symbols"] = empty
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"DONE: {manifest['n_files']} files, "
          f"{counts['bar_files']} bar files, "
          f"{counts['delisted_symbols']} delistings. Manifest written.")


# ---------------------------------------------------------------------------
# --selftest: pure-helper tests (mirrors validation/test_norgate_export_helpers.py
# so the VM copy can self-verify without the repo)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    import tempfile

    # parse_norgate_symbol
    p = parse_norgate_symbol("SIVB-202303")
    assert p == ("SIVB-202303", "SIVB", "202303", True), p
    p = parse_norgate_symbol("AAPL")
    assert p == ("AAPL", "AAPL", None, False), p
    p = parse_norgate_symbol("BRK.B")
    assert p == ("BRK.B", "BRK.B", None, False), p
    p = parse_norgate_symbol("ABC-201399")   # month 99: not a delist suffix
    assert not p.is_delisted and p.base == "ABC-201399", p
    p = parse_norgate_symbol("TWTR-202210")
    assert p.is_delisted and p.base == "TWTR" and p.delist_yyyymm == "202210"

    # symbol_to_filename
    assert symbol_to_filename("BRK.B") == "BRK.B"
    assert symbol_to_filename("SIVB-202303") == "SIVB-202303"
    assert symbol_to_filename("A/B:C") == "A_B_C"

    # build_membership_frame
    idx1 = pd.to_datetime(["2020-01-02", "2020-01-03"])
    idx2 = pd.to_datetime(["2020-01-03", "2020-01-06"])
    frame = build_membership_frame({
        "BBB": pd.Series([1, 1], index=idx2),
        "AAA": pd.Series([1, 0], index=idx1),
    })
    assert list(frame.columns) == ["AAA", "BBB"]
    assert frame.shape == (3, 2) and frame.dtypes.eq(bool).all()
    assert bool(frame.loc["2020-01-02", "AAA"]) is True
    assert bool(frame.loc["2020-01-03", "AAA"]) is False
    assert bool(frame.loc["2020-01-06", "AAA"]) is False   # missing -> False
    assert bool(frame.loc["2020-01-02", "BBB"]) is False
    assert build_membership_frame({}).empty

    # manifest from a synthetic export tree
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "bars").mkdir()
        pd.DataFrame({"Close": [1.0, 2.0, 3.0]}).to_parquet(
            root / "bars" / "AAPL.parquet")
        (root / "manifest.json").write_text("{}")   # must be excluded
        m = build_manifest(root, {"start_date": START_DATE},
                           {"bar_files": 1}, "1.0.99-test")
        assert m["n_files"] == 1 and m["norgatedata_version"] == "1.0.99-test"
        entry = m["files"][0]
        assert entry["path"] == "bars/AAPL.parquet"
        assert entry["rows"] == 3
        assert entry["sha256"] == sha256_file(root / "bars" / "AAPL.parquet")
        assert len(entry["sha256"]) == 64

    print("norgate_export --selftest: ALL PASS")


if __name__ == "__main__":
    main()
