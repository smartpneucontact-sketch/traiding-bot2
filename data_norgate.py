"""Norgate US Platinum — macOS-side loader + trust gates (Phase N0).

Why this exists
---------------
The plan (what-can-we-do-squishy-iverson.md, Phase N0) mandates a one-time
VM-side export refreshed monthly — never a live connection. The Windows VM
runs `norgate_export.py` (NDU + `norgatedata`) and writes a raw parquet tree
plus a manifest of sha256s; this module is the ONLY thing macOS runs against
that tree. It (a) proves the transfer was lossless, (b) normalizes symbols
and schema into the house format used by data.py / data_pit.py / exp_lib,
and (c) implements the five N0 trust gates that must ALL pass before any
truth-run number is believed (see results/v7/norgate/N1_PREREG.md, V-data).

THE SUBSCRIPTION DOES NOT EXIST YET: everything here is buildable and
testable against synthetic fixtures that mimic the documented export format
(validation/test_data_norgate.py). The raw-tree contract below is therefore
the specification the VM exporter must satisfy.

Raw export contract (VM side writes, macOS side reads)
------------------------------------------------------
raw_dir/
  manifest.json           REQUIRED. "files" is either a list of entries
                          [{"path": "<relpath>", "sha256": "<hex>", ...}]
                          (what norgate_export.py writes) or a plain
                          {"<relpath>": "<sha256 hex>"} mapping. Every data
                          file under raw_dir except manifest.json itself
                          must be listed; verification is bidirectional
                          (listed-but-missing AND present-but-unlisted both
                          fail) — that is the lossless-transfer proof.
  bars/<SYMBOL>.parquet   One file per Norgate symbol (raw Norgate naming:
                          '.' share classes e.g. BRK.B; delisted names carry
                          the '-YYYYMM' suffix e.g. SIVB-202303). Columns
                          (Norgate export names, case-insensitive): Open,
                          High, Low, Close, Volume, Unadjusted Close,
                          Dividend. Close/Open/High/Low are TOTALRETURN
                          back-adjusted (same convention as the yfinance
                          auto_adjust=True cache). DatetimeIndex.
  membership/<tag>.parquet  Wide 0/1 (or bool) frame, date x Norgate symbol,
                          from index_constituent_timeseries. Tags are
                          EXACTLY the exporter's filenames (the keys of
                          norgate_export.INDEX_NAMES): sp500, sp400, sp600,
                          sp1500, r1000, r3000 (see MEMBERSHIP_TAGS).
  delistings.parquet      Columns: symbol (Norgate naming), last_quoted_date
                          (+ optional security_name, reason).

House tree written by ingest() (frozen once written)
----------------------------------------------------
dest/
  bars/<sym>.parquet      Normalized symbol, lowercase schema: open, high,
                          low, close, volume, unadj_close, dividend.
  membership/<tag>.parquet  Bool frame, date x normalized symbol.
  delistings.parquet      symbol, norgate_symbol, base_ticker,
                          delist_yyyymm, last_quoted_date, terminal_close,
                          terminal_unadj_close (+ passthrough columns).
  symbol_map.parquet      norgate_symbol, symbol, base_ticker,
                          delist_yyyymm, alive, current_ticker (base ticker
                          for the alive subset, None for dead names).
  _ingest_record.json     Provenance: manifest sha256, every artifact's
                          sha256, timestamps. Its presence FREEZES the tree:
                          re-ingesting requires refresh=True.
Every parquet gets a .meta.json sidecar (data_pit._write_meta pattern).

Symbol normalization
--------------------
Norgate '.' share classes -> '-' (yfinance/cache style: BRK.B -> BRK-B).
The '-YYYYMM' delisting suffix is PRESERVED — it is the unique key that
protects against ticker reuse across corporate entities. A combined case
like FOO.A-202006 normalizes to FOO-A-202006. symbol_map carries the
symbol -> current_ticker mapping for the alive subset (what gate 1 and the
strategies' cache-overlap logic key on).

Licensing note: exported Norgate data stays out of any shared repo.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# The exporter's pure helper layer is importable WITHOUT norgatedata (see
# norgate_export.py's module docstring); reusing its symbol parser keeps the
# VM-side and macOS-side delist-suffix rules identical by construction
# (month must be 01-12 to count as a '-YYYYMM' delisting suffix).
from norgate_export import INDEX_NAMES, parse_norgate_symbol

ROOT = Path(__file__).resolve().parent
RAW_DIR_DEFAULT = ROOT / "data_norgate_raw"
DEST_DEFAULT = ROOT / "data_norgate"
RESULTS_NORGATE_DIR = ROOT / "results" / "v7" / "norgate"

MANIFEST_NAME = "manifest.json"
INGEST_RECORD_NAME = "_ingest_record.json"
REPORT_NAME = "TRUST_GATES_REPORT.md"
MARKER_NAME = "trust_gates_PASS.json"

# Norgate export column names -> house lowercase schema (data.py convention:
# lowercase open/high/low/close/volume; plus the two verification columns).
RAW_COLUMN_MAP = {
    "open": "open", "high": "high", "low": "low", "close": "close",
    "volume": "volume",
    "unadjusted close": "unadj_close", "unadjusted_close": "unadj_close",
    "unadj_close": "unadj_close",
    "dividend": "dividend", "dividends": "dividend",
}
BARS_FIELDS = ("open", "high", "low", "close", "volume",
               "unadj_close", "dividend")

# Membership tags: EXACTLY the exporter's filenames (norgate_export writes
# membership/<tag>.parquet for each key of INDEX_NAMES).
MEMBERSHIP_TAGS: tuple[str, ...] = tuple(INDEX_NAMES)

# ---------------------------------------------------------------------------
# Trust-gate thresholds — verbatim from the frozen plan (Phase N0) and
# N1_PREREG.md V-data. DO NOT loosen without a dated prereg amendment.
GATE1_MEDIAN_BP = 1.0      # median per-name mean |return diff| must be <1bp
GATE1_REVIEW_BP = 5.0      # names above this go on the hand-review list
GATE1_MIN_OVERLAP_DAYS = 60
GATE1_MIN_NAMES = 200      # real overlap is ~1,040 names; tests override
GATE2_EVENT_TOL = 1e-3     # 10bp per ex-date identity event (rounding room)
GATE2_MAX_BROKEN_FRAC = 0.005   # <=0.5% of all ex-date events may exceed tol
GATE2_NAME_BROKEN_FRAC = 0.5    # any single name >50% broken events -> FAIL
GATE3_MAX_DAILY_DIFF = 3   # daily symmetric diff (after explained) <= 3
GATE5_REL_TOL = 1e-9       # OHLC coherence tolerance

# N1 research window (plan / N1_PREREG.md). Gate 3's KNOWN_EVENTS mandate is
# "all 10 KNOWN_EVENTS match": every event dated inside this window MUST be
# verifiable in the membership frame — a frame whose window cannot reach an
# in-window event is a gate-3 FAILURE (membership export window too short),
# never a silent skip. All of data_pit.KNOWN_EVENTS lie inside this window.
RESEARCH_WINDOW: tuple[str, str] = ("2016-04-01", "2026-03-27")

# Known-delisting spot-check expectations (gate 4) — from the plan verbatim:
# "SIVB, FRC, TWTR ~= $54.20 terminal, BBBY -> ~0, ATVI ~= $95, CERN".
# months = acceptable '-YYYYMM' suffixes (Norgate stamps the last-quoted
# month; OTC continuations can shift it, hence the small windows).
# terminal_band = (lo, hi) on the UNADJUSTED terminal close, None = only
# existence + month are checked.
KNOWN_DELISTINGS: list[dict] = [
    {"base": "SIVB", "months": ["202303"], "terminal_band": None,
     "note": "SVB Financial, FDIC receivership 2023-03"},
    {"base": "FRC", "months": ["202305"], "terminal_band": None,
     "note": "First Republic, FDIC receivership 2023-05"},
    {"base": "TWTR", "months": ["202210", "202211"],
     "terminal_band": (53.5, 54.5),
     "note": "Musk take-private at $54.20, last trade 2022-10-27"},
    {"base": "BBBY",
     "months": [f"2023{m:02d}" for m in range(4, 11)],
     "terminal_band": (0.0, 0.5),
     "note": "Bed Bath & Beyond bankruptcy -> ~0 through OTC"},
    {"base": "ATVI", "months": ["202310"], "terminal_band": (94.0, 96.0),
     "note": "Microsoft acquisition at $95.00, 2023-10"},
    {"base": "CERN", "months": ["202205", "202206"],
     "terminal_band": (94.0, 96.0),
     "note": "Oracle acquisition at $95.00, 2022-06"},
]


# ---------------------------------------------------------------------------
# provenance helpers (data_pit._write_meta pattern)

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _write_meta(artifact: Path, extra: dict, source: str) -> None:
    """Sidecar {artifact}.meta.json: source, sha256 of artifact, extras."""
    meta = {
        "artifact": artifact.name,
        "sha256": _sha256_file(artifact),
        "source": source,
        **extra,
    }
    with open(artifact.with_suffix(artifact.suffix + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# symbol normalization

def split_delist_suffix(symbol: str) -> tuple[str, str | None]:
    """'SIVB-202303' -> ('SIVB', '202303'); 'BRK.B' -> ('BRK.B', None).

    Thin wrapper over norgate_export.parse_norgate_symbol — the ONE
    implementation of the '-YYYYMM' rule (month must be 01-12; e.g.
    'ABC-201399' is NOT a delisting suffix), shared with the VM exporter.
    """
    p = parse_norgate_symbol(symbol)
    return p.base, p.delist_yyyymm


def normalize_symbol(symbol: str) -> str:
    """Norgate symbol -> house symbol.

    '.' share-class separators -> '-' (BRK.B -> BRK-B, yfinance/cache style).
    The '-YYYYMM' delisting suffix is PRESERVED as part of the unique key
    (protection against ticker reuse); the class dot inside the base still
    normalizes (FOO.A-202006 -> FOO-A-202006).
    """
    base, yyyymm = split_delist_suffix(str(symbol).strip().upper())
    base = base.replace(".", "-")
    return f"{base}-{yyyymm}" if yyyymm else base


def base_ticker(symbol: str) -> str:
    """Normalized symbol -> base ticker without any delisting suffix."""
    b, _ = split_delist_suffix(normalize_symbol(symbol))
    return b


# ---------------------------------------------------------------------------
# manifest verification (lossless-transfer proof)

def _manifest_files(manifest: dict) -> dict[str, str]:
    """Normalize the manifest 'files' section to {relpath: sha256}.

    Accepts both the norgate_export.py list-of-entries format
    ([{"path": ..., "sha256": ..., ...}]) and a plain mapping.
    """
    raw = manifest.get("files")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        return {e["path"]: e["sha256"] for e in raw}
    raise ValueError("manifest 'files' section missing or malformed")


def verify_manifest(raw_dir: Path) -> dict:
    """Verify every raw file's sha256 against manifest.json, bidirectionally.

    Raises ValueError listing every problem: manifest missing, listed file
    missing, sha mismatch, or data file present on disk but not listed.
    Returns the parsed manifest on success.
    """
    raw_dir = Path(raw_dir)
    man_path = raw_dir / MANIFEST_NAME
    if not man_path.exists():
        raise ValueError(f"manifest missing: {man_path}")
    manifest = json.loads(man_path.read_text())
    files = _manifest_files(manifest)
    if not files:
        raise ValueError(f"manifest has no 'files' section: {man_path}")

    problems: list[str] = []
    for rel, expected in sorted(files.items()):
        p = raw_dir / rel
        if not p.exists():
            problems.append(f"listed but missing: {rel}")
            continue
        got = _sha256_file(p)
        if got != expected:
            problems.append(f"sha256 mismatch: {rel} "
                            f"(manifest {expected[:12]}.., disk {got[:12]}..)")
    on_disk = {
        p.relative_to(raw_dir).as_posix()
        for p in raw_dir.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME
        and not p.name.endswith(".meta.json")
    }
    for rel in sorted(on_disk - set(files)):
        problems.append(f"present but not in manifest: {rel}")
    if problems:
        raise ValueError(
            f"manifest verification FAILED ({len(problems)} problems) — "
            "transfer is not lossless:\n  " + "\n  ".join(problems)
        )
    return manifest


# ---------------------------------------------------------------------------
# ingest

def _normalize_bars(df: pd.DataFrame, src: str) -> pd.DataFrame:
    """Raw Norgate bar frame -> house lowercase schema, DatetimeIndex."""
    cols = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in RAW_COLUMN_MAP:
            cols[c] = RAW_COLUMN_MAP[key]
    df = df.rename(columns=cols)
    missing = [f for f in BARS_FIELDS if f not in df.columns]
    if missing:
        raise ValueError(f"{src}: missing required columns {missing} "
                         f"(export contract: {list(BARS_FIELDS)})")
    df = df[list(BARS_FIELDS)].copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["dividend"] = df["dividend"].fillna(0.0)
    return df


def ingest(raw_dir: str | Path = RAW_DIR_DEFAULT,
           dest: str | Path = DEST_DEFAULT,
           refresh: bool = False) -> dict:
    """Verify + normalize the VM export into the frozen house tree.

    Steps: (1) bidirectional sha256 manifest verification (raises on any
    mismatch — lossless transfer or nothing); (2) the EXPECTED artifact set
    is derived from the raw manifest BEFORE anything is written; on
    refresh=True every dest file NOT in that set is removed (logged) so no
    stale artifact from a prior ingest — e.g. an alive 'CCC' surviving after
    the company became 'CCC-202307' — can leak into the refreshed tree;
    (3) symbol normalization with collision detection; (4) house-schema
    bars, membership, delistings, symbol_map parquets, each with a
    .meta.json sidecar; (5) an ingest record whose artifacts section is
    derived from the expected set (NEVER a post-hoc rglob of dest, which
    would absorb strays) and that freezes the tree — a second ingest()
    raises unless refresh=True (the frozen-once-ingested rule).
    """
    raw_dir, dest = Path(raw_dir), Path(dest)
    if not raw_dir.is_absolute():
        raw_dir = ROOT / raw_dir
    if not dest.is_absolute():
        dest = ROOT / dest

    record_path = dest / INGEST_RECORD_NAME
    if record_path.exists() and not refresh:
        raise RuntimeError(
            f"{dest} is already ingested and FROZEN ({record_path.name} "
            "exists). Pass refresh=True to explicitly re-ingest."
        )

    manifest = verify_manifest(raw_dir)
    manifest_files = _manifest_files(manifest)
    manifest_sha = _sha256_file(raw_dir / MANIFEST_NAME)
    ingested_at = datetime.now(timezone.utc).isoformat()

    # ---- expected artifact set — from the RAW MANIFEST, before writing ----
    # (verify_manifest proved disk == manifest bidirectionally, so the
    # manifest is the authoritative listing of what this export contains.)
    raw_bar_rels = sorted(r for r in manifest_files
                          if r.startswith("bars/") and r.endswith(".parquet"))
    if not raw_bar_rels:
        raise ValueError(f"no bar files under {raw_dir / 'bars'}")
    raw_mem_rels = sorted(r for r in manifest_files
                          if r.startswith("membership/")
                          and r.endswith(".parquet"))
    has_delistings = "delistings.parquet" in manifest_files

    bar_syms: dict[str, str] = {}       # raw relpath -> normalized symbol
    seen: dict[str, str] = {}
    for rel in raw_bar_rels:
        norgate_sym = Path(rel).stem
        sym = normalize_symbol(norgate_sym)
        if sym in seen:
            raise ValueError(f"symbol collision after normalization: "
                             f"{norgate_sym!r} and {seen[sym]!r} both -> {sym!r}")
        seen[sym] = norgate_sym
        bar_syms[rel] = sym

    expected_parquets = (
        {f"bars/{sym}.parquet" for sym in bar_syms.values()}
        | {f"membership/{Path(rel).stem}.parquet" for rel in raw_mem_rels}
        | {"symbol_map.parquet"}
        | ({"delistings.parquet"} if has_delistings else set())
    )
    expected_files = (expected_parquets
                      | {f"{p}.meta.json" for p in expected_parquets}
                      | {INGEST_RECORD_NAME})

    # ---- refresh: purge stale artifacts from any prior ingest ----
    if refresh and dest.exists():
        for p in sorted(dest.rglob("*")):
            if p.is_file() \
                    and p.relative_to(dest).as_posix() not in expected_files:
                print(f"refresh: removing stale artifact "
                      f"{p.relative_to(dest).as_posix()} "
                      "(not in this export's expected set)")
                p.unlink()

    bars_out = dest / "bars"
    mem_out = dest / "membership"
    bars_out.mkdir(parents=True, exist_ok=True)
    mem_out.mkdir(parents=True, exist_ok=True)

    # ---- bars ----
    sym_rows = []
    for rel in raw_bar_rels:
        p = raw_dir / rel
        norgate_sym = Path(rel).stem
        sym = bar_syms[rel]
        df = _normalize_bars(pd.read_parquet(p), src=str(p))
        out = bars_out / f"{sym}.parquet"
        df.to_parquet(out)
        _write_meta(out, {
            "norgate_symbol": norgate_sym,
            "raw_file": str(p.relative_to(raw_dir)),
            "raw_sha256": manifest_files[p.relative_to(raw_dir).as_posix()],
            "ingested_at": ingested_at,
            "n_rows": int(len(df)),
            "first_date": df.index[0], "last_date": df.index[-1],
        }, source=f"norgate_export:{p.relative_to(raw_dir)}")
        b, yyyymm = split_delist_suffix(sym)
        sym_rows.append({
            "norgate_symbol": norgate_sym,
            "symbol": sym,
            "base_ticker": b,
            "delist_yyyymm": yyyymm,
            "alive": yyyymm is None,
            "current_ticker": b if yyyymm is None else None,
        })
    symbol_map = pd.DataFrame(sym_rows).sort_values("symbol").reset_index(drop=True)
    sm_path = dest / "symbol_map.parquet"
    symbol_map.to_parquet(sm_path)
    _write_meta(sm_path, {"ingested_at": ingested_at,
                          "n_symbols": int(len(symbol_map)),
                          "n_alive": int(symbol_map["alive"].sum())},
                source="derived:ingest(symbol normalization)")

    # ---- membership ----
    mem_tags = []
    for rel in raw_mem_rels:
        p = raw_dir / rel
        tag = Path(rel).stem
        raw = pd.read_parquet(p)
        raw.index = pd.to_datetime(raw.index)
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()
        norm_cols = {c: normalize_symbol(c) for c in raw.columns}
        mem = raw.rename(columns=norm_cols)
        if mem.columns.duplicated().any():  # collisions -> OR
            mem = mem.T.groupby(level=0).any().T
        mem = mem.astype(bool)
        out = mem_out / f"{tag}.parquet"
        mem.to_parquet(out)
        _write_meta(out, {
            "raw_file": str(p.relative_to(raw_dir)),
            "raw_sha256": manifest_files[p.relative_to(raw_dir).as_posix()],
            "ingested_at": ingested_at,
            "n_days": int(len(mem)), "n_symbols": int(mem.shape[1]),
        }, source=f"norgate_export:{p.relative_to(raw_dir)}")
        mem_tags.append(tag)

    # ---- delistings ----
    dl_raw_path = raw_dir / "delistings.parquet"
    if has_delistings:
        dl = pd.read_parquet(dl_raw_path).copy()
        if "symbol" not in dl.columns or "last_quoted_date" not in dl.columns:
            raise ValueError("delistings.parquet must have columns "
                             "['symbol', 'last_quoted_date']")
        dl["norgate_symbol"] = dl["symbol"].astype(str)
        dl["symbol"] = dl["norgate_symbol"].map(normalize_symbol)
        dl["base_ticker"] = dl["symbol"].map(base_ticker)
        dl["delist_yyyymm"] = dl["symbol"].map(
            lambda s: split_delist_suffix(s)[1])
        dl["last_quoted_date"] = pd.to_datetime(dl["last_quoted_date"])
        term_c, term_u = [], []
        for s in dl["symbol"]:
            bp = bars_out / f"{s}.parquet"
            if bp.exists():
                b = pd.read_parquet(bp)
                term_c.append(float(b["close"].dropna().iloc[-1]))
                term_u.append(float(b["unadj_close"].dropna().iloc[-1]))
            else:
                term_c.append(np.nan)
                term_u.append(np.nan)
        dl["terminal_close"] = term_c
        dl["terminal_unadj_close"] = term_u
        dl_path = dest / "delistings.parquet"
        dl.to_parquet(dl_path)
        _write_meta(dl_path, {
            "raw_file": "delistings.parquet",
            "raw_sha256": manifest_files["delistings.parquet"],
            "ingested_at": ingested_at, "n_delistings": int(len(dl)),
        }, source="norgate_export:delistings.parquet")
    else:
        dl = pd.DataFrame()

    # ---- ingest record (freezes the tree) ----
    # artifacts derive from the manifest-declared EXPECTED set, never from a
    # post-hoc rglob of dest (which would absorb strays into the record).
    artifacts = {
        rel: _sha256_file(dest / rel) for rel in sorted(expected_parquets)
    }
    record = {
        "ingested_at": ingested_at,
        "raw_dir": str(raw_dir),
        "manifest_sha256": manifest_sha,
        "manifest_created_at": (manifest.get("created_at")
                                or manifest.get("export_timestamp_utc")),
        "raw_manifest_verified": True,
        "n_bars_files": len(raw_bar_rels),
        "membership_tags": mem_tags,
        "n_delistings": int(len(dl)),
        "artifacts": artifacts,
    }
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return record


# ---------------------------------------------------------------------------
# accessors

def _require_ingested(dest: Path) -> dict:
    record_path = Path(dest) / INGEST_RECORD_NAME
    if not record_path.exists():
        raise FileNotFoundError(
            f"{dest} is not an ingested Norgate tree ({INGEST_RECORD_NAME} "
            "missing) — run ingest() first."
        )
    return json.loads(record_path.read_text())


def load_norgate_panel(
    fields: tuple[str, ...] = ("close", "open", "high", "low", "volume"),
    start: str | None = None,
    end: str | None = None,
    dest: str | Path = DEST_DEFAULT,
    min_obs: int = 1,
) -> dict[str, pd.DataFrame]:
    """load_panel-compatible dict[field] -> DataFrame (date x symbol).

    Differences from data.py.load_panel, all deliberate:
    - min_obs defaults to 1, NOT 500: delisted names with short series are
      the entire point of this dataset and must never be filtered out.
    - no outlier filter: Norgate series are corporate-action clean; real
      crashes must stay in (gate 5 scans integrity instead).
    - fields may also include 'unadj_close' and 'dividend'.
    Delisted symbols simply stop: their columns are NaN after
    last_quoted_date (exit-at-last-close is the engine's job, not the
    loader's).
    """
    dest = Path(dest)
    _require_ingested(dest)
    bad = [f for f in fields if f not in BARS_FIELDS]
    if bad:
        raise ValueError(f"unknown fields {bad}; available: {list(BARS_FIELDS)}")
    panels: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}
    for p in sorted((dest / "bars").glob("*.parquet")):
        df = pd.read_parquet(p)
        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]
        if len(df) < min_obs:
            continue
        for f in fields:
            panels[f][p.stem] = df[f]
    out = {f: pd.DataFrame(d).sort_index() for f, d in panels.items()}
    common = sorted(set.intersection(*[set(df.columns) for df in out.values()])) \
        if out else []
    return {f: df[common] for f, df in out.items()}


def membership_frame(index_tag: str,
                     daily_index: pd.DatetimeIndex,
                     dest: str | Path = DEST_DEFAULT) -> pd.DataFrame:
    """Bool DataFrame (daily_index x normalized symbol) for one index tag.

    Exactly the data_pit.membership_frame shape (bool dtype, DatetimeIndex
    rows, normalized-symbol columns) so it drops straight into the
    strategies' eligible= kwarg. As-of semantics: membership is forward-
    filled onto daily_index; dates before the first membership row are
    False (unknown = not eligible — conservative, unlike exp_lib's
    unrestricted pre-window convention, because here the history is
    supposed to cover the window; a large False head means a data problem).
    """
    dest = Path(dest)
    _require_ingested(dest)
    path = dest / "membership" / f"{index_tag}.parquet"
    if not path.exists():
        have = sorted(p.stem for p in (dest / "membership").glob("*.parquet"))
        raise FileNotFoundError(f"no membership tag {index_tag!r}; have {have}")
    mem = pd.read_parquet(path)
    mem.index = pd.to_datetime(mem.index)
    idx = pd.DatetimeIndex(daily_index).sort_values()
    out = mem.reindex(idx, method="ffill")
    return out.fillna(False).astype(bool)


def delisting_events(dest: str | Path = DEST_DEFAULT) -> pd.DataFrame:
    """The normalized delistings table (see module docstring for columns)."""
    dest = Path(dest)
    _require_ingested(dest)
    path = dest / "delistings.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — export had no delistings")
    df = pd.read_parquet(path)
    df["last_quoted_date"] = pd.to_datetime(df["last_quoted_date"])
    return df


def _collapse_to_base(mem: pd.DataFrame) -> pd.DataFrame:
    """Membership over symbols -> membership over base tickers (OR across
    '-YYYYMM' incarnations of one ticker)."""
    based = mem.copy()
    based.columns = [base_ticker(c) for c in based.columns]
    if based.columns.duplicated().any():
        based = based.T.groupby(level=0).any().T
    return based.astype(bool)


# ---------------------------------------------------------------------------
# trust gates (Phase N0 — all five must PASS before N1 may run)

def _gate1_overlap_returns(cache_panel: dict, dest: Path,
                           min_overlap_days: int,
                           min_names: int) -> dict:
    """Gate 1: return-diff vs the cache overlap.

    Threshold (plan verbatim): median per-name mean |daily return diff|
    < 1bp; names > 5bp are listed for hand review (Norgate wins any
    disagreement — the review is about understanding, not about editing
    Norgate)."""
    cache_close = cache_panel["close"]
    sm = pd.read_parquet(Path(dest) / "symbol_map.parquet")
    alive = sm[sm["alive"]]
    overlap = sorted(set(alive["current_ticker"]) & set(cache_close.columns))
    details: list[str] = []
    per_name: dict[str, float] = {}
    skipped = []
    for t in overlap:
        sym = alive.loc[alive["current_ticker"] == t, "symbol"].iloc[0]
        bars = pd.read_parquet(Path(dest) / "bars" / f"{sym}.parquet")
        rn = bars["close"].pct_change()
        rc = cache_close[t].pct_change()
        both = pd.concat([rn, rc], axis=1, join="inner").dropna()
        if len(both) < min_overlap_days:
            skipped.append(t)
            continue
        per_name[t] = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().mean() * 1e4)
    if len(per_name) < min_names:
        return {"name": "gate1_overlap_return_diff", "passed": False,
                "details": [f"only {len(per_name)} overlap names with >= "
                            f"{min_overlap_days} common days (need >= "
                            f"{min_names}); skipped {len(skipped)}"]}
    s = pd.Series(per_name)
    med = float(s.median())
    review = s[s > GATE1_REVIEW_BP].sort_values(ascending=False)
    passed = med < GATE1_MEDIAN_BP
    details.append(f"{len(s)} overlap names; median per-name mean |diff| = "
                   f"{med:.4f}bp (threshold < {GATE1_MEDIAN_BP}bp)")
    details.append(f"names > {GATE1_REVIEW_BP}bp (HAND-REVIEW; Norgate wins "
                   f"disagreements): {len(review)}")
    for t, v in review.head(25).items():
        details.append(f"  review: {t} mean|diff|={v:.2f}bp")
    if skipped:
        details.append(f"skipped (insufficient overlap days): {len(skipped)}")
    return {"name": "gate1_overlap_return_diff", "passed": passed,
            "details": details,
            "metrics": {"n_names": len(s), "median_bp": med,
                        "n_review": int(len(review))}}


def _gate2_dividends(dest: Path, network: bool,
                     yf_names: int = 20) -> dict:
    """Gate 2: dividend verification from the unadj + dividend columns.

    Ex-date identity for TOTALRETURN back-adjustment: on every ex-date t,
      adjusted_return(t) == (unadj_t + dividend_t) / unadj_{t-1} - 1
    within GATE2_EVENT_TOL (rounding room). FAIL if > 0.5% of all events
    break the identity, or any single name has > 50% broken events (a
    broken adjustment chain, not rounding).

    yfinance 20-name spot check: only with network=True (skippable NOW,
    must be run once with network=True before the real N1 — the skip is
    recorded in the report and marker)."""
    details: list[str] = []
    n_events = 0
    n_broken = 0
    broken_names: list[tuple[str, int, int, float]] = []
    div_names: list[str] = []
    for p in sorted((Path(dest) / "bars").glob("*.parquet")):
        b = pd.read_parquet(p)
        ex = b.index[(b["dividend"] > 0)]
        if len(ex) == 0:
            continue
        div_names.append(p.stem)
        prev_unadj = b["unadj_close"].shift(1)
        implied = (b["unadj_close"] + b["dividend"]) / prev_unadj - 1.0
        actual = b["close"].pct_change()
        err = (implied - actual).abs().loc[ex].dropna()
        n_events += len(err)
        bad = int((err > GATE2_EVENT_TOL).sum())
        n_broken += bad
        if len(err) and bad / len(err) > GATE2_NAME_BROKEN_FRAC:
            broken_names.append((p.stem, bad, len(err), float(err.max())))
    if n_events == 0:
        return {"name": "gate2_dividend_verification", "passed": False,
                "details": ["no dividend events found anywhere — export is "
                            "missing the Dividend column content"]}
    frac = n_broken / n_events
    passed = frac <= GATE2_MAX_BROKEN_FRAC and not broken_names
    details.append(f"{n_events} ex-date events across {len(div_names)} names; "
                   f"{n_broken} exceed {GATE2_EVENT_TOL*1e4:.0f}bp identity "
                   f"tolerance ({frac:.3%}; threshold <= "
                   f"{GATE2_MAX_BROKEN_FRAC:.1%})")
    for sym, bad, tot, mx in broken_names[:25]:
        details.append(f"  BROKEN ADJUSTMENT: {sym} {bad}/{tot} events, "
                       f"max err {mx*1e4:.1f}bp")
    yf_status = "SKIPPED (network=False)"
    if network:
        yf_status, yf_details, yf_ok = _gate2_yfinance_spot_check(
            dest, div_names, yf_names)
        details.extend(yf_details)
        passed = passed and yf_ok
    # ran = the spot check actually fetched names and reached a verdict
    # (SKIPPED and ERROR statuses both mean the mandated check did NOT run).
    spot_check_ran = yf_status.startswith(("PASS", "FAIL"))
    details.append(f"yfinance {yf_names}-name spot check: {yf_status}")
    return {"name": "gate2_dividend_verification", "passed": passed,
            "details": details,
            "metrics": {"n_events": n_events, "broken_frac": frac,
                        "n_broken_names": len(broken_names),
                        "yf_spot_check": yf_status,
                        "yf_spot_check_ran": spot_check_ran}}


def _gate2_yfinance_spot_check(dest: Path, div_names: list[str],
                               n: int) -> tuple[str, list[str], bool]:
    """Network sub-check: n alive dividend payers, Norgate adjusted returns
    vs yfinance auto_adjust=True. PASS if >= 90% of fetched names show mean
    |return diff| < 5bp. Returns (status, detail lines, ok)."""
    try:
        import yfinance as yf  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        return (f"ERROR (yfinance unavailable: {e})",
                ["yfinance import failed — rerun with the package installed"],
                False)
    sm = pd.read_parquet(Path(dest) / "symbol_map.parquet")
    alive_div = [s for s in div_names
                 if s in set(sm.loc[sm["alive"], "symbol"])][:n]
    details, oks = [], []
    for sym in alive_div:
        bars = pd.read_parquet(Path(dest) / "bars" / f"{sym}.parquet")
        t = sm.loc[sm["symbol"] == sym, "current_ticker"].iloc[0]
        try:
            yfd = yf.download(t, start=str(bars.index[0].date()),
                              end=str(bars.index[-1].date()),
                              auto_adjust=True, progress=False)
            ry = yfd["Close"].squeeze().pct_change()
            rn = bars["close"].pct_change()
            both = pd.concat([rn, ry], axis=1, join="inner").dropna()
            d = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().mean() * 1e4)
            ok = d < 5.0
            oks.append(ok)
            details.append(f"  yf {t}: mean|diff|={d:.2f}bp "
                           f"{'ok' if ok else 'MISMATCH'}")
        except Exception as e:  # pragma: no cover - network dependent
            details.append(f"  yf {t}: fetch failed ({e})")
    if not oks:
        return "ERROR (no names fetched)", details, False
    frac = sum(oks) / len(oks)
    ok = frac >= 0.9
    return (f"{'PASS' if ok else 'FAIL'} ({sum(oks)}/{len(oks)} < 5bp)",
            details, ok)


def _gate3_membership(dest: Path, index_tag: str,
                      wiki_membership: pd.DataFrame | None,
                      known_events: list[tuple[str, str, bool]] | None,
                      count_band: tuple[int, int] | None,
                      explained: frozenset[str],
                      max_daily_diff: int = GATE3_MAX_DAILY_DIFF,
                      research_window: tuple[str, str] = RESEARCH_WINDOW,
                      ) -> dict:
    """Gate 3: Norgate S&P 500 membership vs the independent Wikipedia
    reconstruction (data_pit). Plan verbatim: daily symmetric diff <= 3
    explained, ALL 10 KNOWN_EVENTS match, COUNT_BAND holds — this validates
    BOTH sources at once.

    KNOWN_EVENTS coverage rule: every event dated inside research_window
    MUST be verifiable in the Norgate membership frame. An event the
    frame's window cannot reach is a FAILURE (membership export window too
    short / events unverifiable), NOT a skip — a truncated export must
    never earn a PASS by silently checking fewer events than mandated.
    Only events the requested research window itself excludes are exempt."""
    import data_pit  # lazy: only gate 3 needs it

    mem_path = Path(dest) / "membership" / f"{index_tag}.parquet"
    if not mem_path.exists():
        return {"name": "gate3_membership_crosscheck", "passed": False,
                "details": [f"membership tag {index_tag!r} missing from tree"]}
    nor = pd.read_parquet(mem_path)
    nor.index = pd.to_datetime(nor.index)
    nor_base = _collapse_to_base(nor)

    if known_events is None:
        known_events = data_pit.KNOWN_EVENTS
    if count_band is None:
        count_band = data_pit.COUNT_BAND
    if wiki_membership is None:
        wiki_membership = data_pit.membership_frame(nor_base.index)

    details: list[str] = []
    passed = True

    # (a) COUNT_BAND on the Norgate side (wiki side is guarded by data_pit)
    counts = nor_base.sum(axis=1)
    bad_days = counts[(counts < count_band[0]) | (counts > count_band[1])]
    if len(bad_days):
        passed = False
        details.append(f"COUNT_BAND {count_band} violated on {len(bad_days)} "
                       f"days; first {bad_days.index[0].date()} -> "
                       f"{int(bad_days.iloc[0])}")
    else:
        details.append(f"Norgate daily count in "
                       f"[{int(counts.min())}, {int(counts.max())}] — "
                       f"COUNT_BAND {count_band} holds")

    # (b) daily symmetric diff <= 3 after explained exclusions
    common = nor_base.index.intersection(wiki_membership.index)
    if len(common) == 0:
        return {"name": "gate3_membership_crosscheck", "passed": False,
                "details": ["no common dates between Norgate and wiki frames"]}
    cols = sorted((set(nor_base.columns) | set(wiki_membership.columns))
                  - set(explained))
    a = nor_base.reindex(index=common, columns=cols, fill_value=False)
    b = wiki_membership.reindex(index=common, columns=cols, fill_value=False)
    diff = (a.astype(bool) ^ b.astype(bool)).sum(axis=1)
    worst = int(diff.max())
    if worst > max_daily_diff:
        passed = False
        d0 = diff.idxmax()
        names = [c for c in cols
                 if bool(a.at[d0, c]) != bool(b.at[d0, c])][:10]
        details.append(f"daily symmetric diff max {worst} > {max_daily_diff} "
                       f"(first worst day {d0.date()}: {names})")
    else:
        details.append(f"daily symmetric diff max {worst} <= {max_daily_diff} "
                       f"over {len(common)} common days "
                       f"({len(explained)} explained symbols excluded)")

    # (c) KNOWN_EVENTS spot checks against the NORGATE frame. The mandate
    # is ALL events inside the research window; an event outside the frame's
    # window is a FAILURE (truncated export), never a silent skip.
    rw_lo, rw_hi = pd.Timestamp(research_window[0]), pd.Timestamp(research_window[1])
    frame_lo, frame_hi = nor_base.index[0], nor_base.index[-1]
    ev_fail = []
    n_checked = 0
    n_required = 0
    n_outside_research = 0
    for ticker, date, expected in known_events:
        d = pd.Timestamp(date)
        if d < rw_lo or d > rw_hi:
            # excluded by the REQUESTED research window itself — the only
            # legitimate reason not to verify an event.
            n_outside_research += 1
            continue
        n_required += 1
        if d < frame_lo or d > frame_hi:
            ev_fail.append(
                f"{ticker}@{date}: UNVERIFIABLE — membership frame window "
                f"[{frame_lo.date()}, {frame_hi.date()}] does not reach the "
                f"event (membership export window too short)")
            continue
        if d not in nor_base.index:
            ev_fail.append(f"{ticker}@{date}: date not in Norgate index")
            continue
        n_checked += 1
        got = bool(nor_base.at[d, ticker]) if ticker in nor_base.columns else False
        if got != expected:
            ev_fail.append(f"{ticker}@{date}: expected {expected}, got {got}")
    if n_checked < n_required:
        passed = False
        details.append(
            f"KNOWN_EVENTS: only {n_checked}/{n_required} research-window "
            f"events verifiable in the membership frame "
            f"[{frame_lo.date()}, {frame_hi.date()}] — membership export "
            "window too short / events unverifiable")
    if ev_fail:
        passed = False
        details.append("KNOWN_EVENTS failures: " + "; ".join(ev_fail))
    elif n_checked == n_required:
        details.append(
            f"all {n_checked}/{n_required} research-window KNOWN_EVENTS "
            "match"
            + (f" ({n_outside_research} outside research window "
               f"{research_window} exempt)" if n_outside_research else ""))

    return {"name": "gate3_membership_crosscheck", "passed": passed,
            "details": details,
            "metrics": {"max_daily_diff": worst, "n_common_days": len(common),
                        "n_known_events_checked": n_checked,
                        "n_known_events_required": n_required}}


def _gate4_delistings(dest: Path, expectations: list[dict]) -> dict:
    """Gate 4: known-delisting spot checks (existence, month, terminal
    unadjusted price band) against the delistings table."""
    try:
        dl = delisting_events(dest)
    except FileNotFoundError as e:
        return {"name": "gate4_delisting_spot_checks", "passed": False,
                "details": [str(e)]}
    details: list[str] = []
    passed = True
    for exp in expectations:
        base, months = exp["base"], set(exp["months"])
        band = exp.get("terminal_band")
        cand = dl[(dl["base_ticker"] == base)
                  & (dl["delist_yyyymm"].isin(months))]
        if cand.empty:
            passed = False
            got = dl.loc[dl["base_ticker"] == base, "delist_yyyymm"].tolist()
            details.append(f"FAIL {base}: no delisting with suffix in "
                           f"{sorted(months)} (found {got or 'nothing'}) — "
                           f"{exp.get('note', '')}")
            continue
        row = cand.iloc[0]
        term = float(row["terminal_unadj_close"])
        if band is not None and not (band[0] <= term <= band[1]):
            passed = False
            details.append(f"FAIL {base}: terminal unadj close {term:.2f} "
                           f"outside {band} — {exp.get('note', '')}")
        else:
            details.append(f"ok {row['symbol']}: last quoted "
                           f"{row['last_quoted_date'].date()}, terminal "
                           f"unadj {term:.2f}"
                           + (f" in {band}" if band else ""))
    return {"name": "gate4_delisting_spot_checks", "passed": passed,
            "details": details, "metrics": {"n_checks": len(expectations)}}


def _gate5_integrity(dest: Path) -> dict:
    """Gate 5: panel-integrity scan + provenance re-verification.

    Bars: monotonic unique DatetimeIndex; close/unadj_close > 0; OHLC
    coherence (high >= max(open, close), low <= min(open, close)); volume
    and dividend >= 0; no interior NaN closes. Provenance: every parquet
    has a .meta.json sidecar whose sha256 matches the file, the artifact
    shas match the frozen ingest record, and the ingest record attests the
    raw VM manifest verified (sidecar sha256s == VM manifest, transitively).
    """
    dest = Path(dest)
    details: list[str] = []
    problems: list[str] = []

    record = _require_ingested(dest)
    if not record.get("raw_manifest_verified"):
        problems.append("ingest record does not attest raw manifest verification")

    n_files = 0
    for p in sorted(dest.rglob("*.parquet")):
        n_files += 1
        rel = p.relative_to(dest).as_posix()
        side = p.with_suffix(p.suffix + ".meta.json")
        if not side.exists():
            problems.append(f"{rel}: sidecar missing")
        else:
            meta = json.loads(side.read_text())
            if meta.get("sha256") != _sha256_file(p):
                problems.append(f"{rel}: sidecar sha256 != file")
        rec_sha = record.get("artifacts", {}).get(rel)
        if rec_sha is None:
            problems.append(f"{rel}: not in ingest record")
        elif rec_sha != _sha256_file(p):
            problems.append(f"{rel}: sha256 != frozen ingest record")

    tol = GATE5_REL_TOL
    for p in sorted((dest / "bars").glob("*.parquet")):
        sym = p.stem
        b = pd.read_parquet(p)
        if not isinstance(b.index, pd.DatetimeIndex):
            problems.append(f"{sym}: index not DatetimeIndex")
            continue
        if not b.index.is_monotonic_increasing:
            problems.append(f"{sym}: index not sorted")
        if b.index.duplicated().any():
            problems.append(f"{sym}: duplicate dates")
        v = b.dropna(subset=["close"])
        if (v["close"] <= 0).any() or (v["unadj_close"] <= 0).any():
            problems.append(f"{sym}: non-positive close/unadj_close")
        hi_ok = v["high"] >= v[["open", "close"]].max(axis=1) * (1 - tol)
        lo_ok = v["low"] <= v[["open", "close"]].min(axis=1) * (1 + tol)
        if (~hi_ok).any() or (~lo_ok).any():
            problems.append(f"{sym}: OHLC incoherent on "
                            f"{int((~hi_ok).sum() + (~lo_ok).sum())} rows")
        if (v["volume"] < 0).any() or (v["dividend"] < 0).any():
            problems.append(f"{sym}: negative volume/dividend")
        c = b["close"]
        live = c.loc[c.first_valid_index():c.last_valid_index()]
        if live.isna().any():
            problems.append(f"{sym}: {int(live.isna().sum())} interior NaN closes")

    passed = not problems
    details.append(f"{n_files} parquet artifacts checked (sidecars + frozen "
                   "ingest-record shas + raw-manifest attestation)")
    details.append("bars scan: index/price/OHLC/volume/dividend/NaN checks")
    details.extend(f"FAIL {q}" for q in problems[:50])
    if len(problems) > 50:
        details.append(f"... and {len(problems) - 50} more")
    return {"name": "gate5_panel_integrity", "passed": passed,
            "details": details, "metrics": {"n_problems": len(problems)}}


def trust_gates(
    cache_panel: dict,
    dest: str | Path = DEST_DEFAULT,
    index_tag: str = "sp500",
    wiki_membership: pd.DataFrame | None = None,
    known_events: list[tuple[str, str, bool]] | None = None,
    count_band: tuple[int, int] | None = None,
    explained: frozenset[str] = frozenset(),
    delisting_expectations: list[dict] | None = None,
    network: bool = False,
    out_dir: str | Path = RESULTS_NORGATE_DIR,
    min_overlap_days: int = GATE1_MIN_OVERLAP_DAYS,
    min_names: int = GATE1_MIN_NAMES,
    research_window: tuple[str, str] = RESEARCH_WINDOW,
) -> dict:
    """Run all five N0 trust gates; write TRUST_GATES_REPORT.md and, on an
    all-PASS verdict, the trust_gates_PASS.json marker N1 requires.

    Defaults target the real run: cache_panel = exp_lib.load_cache()[0],
    wiki/known_events/count_band from data_pit, delisting expectations =
    KNOWN_DELISTINGS. Tests inject synthetic versions of all of these.
    A FAIL verdict DELETES any existing marker so N1 can never run on a
    stale pass. network=True additionally runs the yfinance 20-name spot
    check inside gate 2; the marker records gate2_spot_check_ran, and
    require_trust_gates_pass() REFUSES a marker where it is false unless
    explicitly overridden — a network=False marker can never satisfy the
    N1 guard silently.
    """
    dest, out_dir = Path(dest), Path(out_dir)
    record = _require_ingested(dest)
    if delisting_expectations is None:
        delisting_expectations = KNOWN_DELISTINGS

    gates = [
        _gate1_overlap_returns(cache_panel, dest, min_overlap_days, min_names),
        _gate2_dividends(dest, network=network),
        _gate3_membership(dest, index_tag, wiki_membership, known_events,
                          count_band, explained,
                          research_window=research_window),
        _gate4_delistings(dest, delisting_expectations),
        _gate5_integrity(dest),
    ]
    all_pass = all(g["passed"] for g in gates)
    gate2_spot_check_ran = bool(
        gates[1].get("metrics", {}).get("yf_spot_check_ran", False))
    ran_at = datetime.now(timezone.utc).isoformat()

    # ---- report ----
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NORGATE TRUST GATES REPORT (Phase N0)",
        "",
        f"- ran_at: {ran_at}",
        f"- dest tree: {dest}",
        f"- ingest record manifest_sha256: {record['manifest_sha256']}",
        f"- network (yfinance spot check): {network}",
        f"- gate2_spot_check_ran: {gate2_spot_check_ran}"
        + ("" if gate2_spot_check_ran else
           "  (marker will NOT satisfy require_trust_gates_pass without an "
           "explicit override)"),
        "",
        f"## OVERALL VERDICT: {'PASS' if all_pass else 'FAIL'}",
        "",
        "N1 may run ONLY on an all-PASS verdict (see N1_PREREG.md, V-data)."
        " Thresholds are frozen in data_norgate.py; loosening any of them"
        " requires a dated prereg amendment.",
        "",
    ]
    for g in gates:
        lines.append(f"## {g['name']}: {'PASS' if g['passed'] else 'FAIL'}")
        lines.extend(f"- {d}" for d in g["details"])
        lines.append("")
    report_path = out_dir / REPORT_NAME
    report_path.write_text("\n".join(lines))

    # ---- marker (all-PASS only; FAIL removes any stale marker) ----
    marker_path = out_dir / MARKER_NAME
    result = {
        "verdict": "PASS" if all_pass else "FAIL",
        "ran_at": ran_at,
        "dest": str(dest),
        "ingest_record_sha256": _sha256_file(dest / INGEST_RECORD_NAME),
        "manifest_sha256": record["manifest_sha256"],
        "gate2_spot_check_ran": gate2_spot_check_ran,
        "gates": {g["name"]: bool(g["passed"]) for g in gates},
        "gate_metrics": {g["name"]: g.get("metrics", {}) for g in gates},
        "report": str(report_path),
        "report_sha256": _sha256_file(report_path),
    }
    if all_pass:
        with open(marker_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    elif marker_path.exists():
        marker_path.unlink()
    result["gate_details"] = {g["name"]: g["details"] for g in gates}
    return result


def require_trust_gates_pass(dest: str | Path = DEST_DEFAULT,
                             out_dir: str | Path = RESULTS_NORGATE_DIR,
                             allow_skipped_network_check: bool = False) -> dict:
    """N1-runner guard: raise unless a CURRENT all-PASS marker exists.

    'Current' = the marker's ingest_record_sha256 matches the tree on disk
    right now, so a re-ingest (even with identical raw data) invalidates the
    marker until trust_gates() is re-run. Returns the marker dict.

    The marker must also attest gate2_spot_check_ran (gate 2's mandated
    20-name yfinance spot check actually ran): a marker earned with
    network=False is REFUSED. allow_skipped_network_check=True is the only
    escape hatch — it must be an explicit, deliberate exception and is
    loudly logged here; N1's runner MUST NOT pass it silently (its default
    is False, and the real N1 requires one network=True gates pass).
    """
    dest, out_dir = Path(dest), Path(out_dir)
    marker_path = out_dir / MARKER_NAME
    if not marker_path.exists():
        raise RuntimeError(
            f"trust gates have not passed: {marker_path} missing — run "
            "data_norgate.trust_gates() and get an all-PASS verdict first."
        )
    marker = json.loads(marker_path.read_text())
    if marker.get("verdict") != "PASS":
        raise RuntimeError(f"trust gates marker verdict is "
                           f"{marker.get('verdict')!r}, not PASS")
    current = _sha256_file(dest / INGEST_RECORD_NAME)
    if marker.get("ingest_record_sha256") != current:
        raise RuntimeError(
            "trust gates marker is STALE: the ingested tree changed since "
            "the gates ran — re-run trust_gates()."
        )
    if not marker.get("gate2_spot_check_ran", False):
        if not allow_skipped_network_check:
            raise RuntimeError(
                "trust gates marker was earned WITHOUT gate 2's mandated "
                "yfinance 20-name spot check (gate2_spot_check_ran is not "
                "true) — re-run trust_gates(network=True). Passing "
                "allow_skipped_network_check=True is the only override and "
                "must be an explicit, logged exception, never N1's default."
            )
        print("WARNING: require_trust_gates_pass OVERRIDE — accepting a "
              "trust-gates marker whose gate-2 yfinance spot check was "
              "SKIPPED (allow_skipped_network_check=True). The real N1 "
              "still requires one network=True trust_gates() pass.")
    return marker


if __name__ == "__main__":
    # Real-data entry point (works only once the subscription exists and the
    # VM export has been copied to data_norgate_raw/). Default is
    # network=True: the real run MUST include gate 2's yfinance spot check —
    # a --no-network marker cannot satisfy the N1 guard.
    import argparse

    ap = argparse.ArgumentParser(
        description="Ingest the VM Norgate export and run the N0 trust gates.")
    ap.add_argument("--no-network", action="store_true",
                    help="skip gate 2's mandated yfinance 20-name spot check "
                         "(the resulting PASS marker will NOT satisfy "
                         "require_trust_gates_pass, the N1 guard)")
    ap.add_argument("--refresh", action="store_true",
                    help="explicitly re-ingest an already-frozen tree "
                         "(stale artifacts from the prior ingest are removed)")
    args = ap.parse_args()
    network = not args.no_network
    if not network:
        print("=" * 72)
        print("WARNING: --no-network — gate 2's MANDATED 20-name yfinance "
              "spot check\nwill be SKIPPED. Any PASS marker written will "
              "record gate2_spot_check_ran\n= false and will NOT satisfy "
              "require_trust_gates_pass() (the N1 guard)\nwithout an "
              "explicit, logged allow_skipped_network_check=True override.\n"
              "Re-run WITHOUT --no-network before the real N1.")
        print("=" * 72)

    rec = ingest(refresh=args.refresh)
    print(f"ingested {rec['n_bars_files']} bar files, membership tags "
          f"{rec['membership_tags']}, {rec['n_delistings']} delistings")
    from exp_lib import load_cache
    panel, _, _ = load_cache()
    res = trust_gates(panel, network=network)
    print(f"TRUST GATES: {res['verdict']} -> {res['report']}")
    for name, ok in res["gates"].items():
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
