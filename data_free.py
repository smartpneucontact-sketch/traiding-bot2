"""Provenance-disciplined free-data fetch layer (yfinance).

Every series fetched here is cached as data_free/{safe_ticker}.parquet with a
{safe_ticker}.meta.json sidecar recording source, fetch time, yfinance
version, row count, date span and the sha256 of the parquet bytes. Once
fetched, a series is FROZEN: subsequent calls serve the local file after
verifying the sha256, and never touch the network. This makes results built
on these series reproducible and auditable (validation/check_provenance.py
can walk the tree), and prevents the silent-revision failure mode of Yahoo
data — values for past dates DO change between fetches (splits/backfills/
corrections), so "same ticker, different day" is not the same dataset.
Re-fetching is an explicit, deliberate act via refresh=True, which rewrites
both files (new sha256 = new dataset revision).

Known limitations / provenance
------------------------------
- yfinance scrapes an unofficial Yahoo endpoint: no SLA, occasional gaps,
  and historical values can be revised server-side at any time. The meta
  sidecar pins WHICH revision a result was computed from; it cannot make
  Yahoo itself stable.
- Bars are fetched with auto_adjust=False, so "Close" is the RAW close.
  Irrelevant for the intended consumers (index/rate series like ^IRX,
  ^VIX9D, ^VIX3M, which pay no dividends and never split), but equities
  fetched through this layer will NOT match the dividend/split-adjusted
  closes in data_cache.pkl — do not mix the two for equity return math.
- ^IRX is quoted as an annualized discount-rate PERCENT (e.g. 5.2 = 5.2%),
  not a price; callers converting to a daily risk-free rate own that
  transformation.
- No survivorship handling here: this layer fetches whatever Yahoo serves
  for a live symbol today. Delisted names generally return nothing.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FREE_DIR = ROOT / "data_free"


def _safe_ticker(ticker: str) -> str:
    """Filesystem-safe cache stem: ^IRX -> _IRX, BRK/B -> BRK_B."""
    return ticker.replace("^", "_").replace("/", "_")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_yf_series(
    ticker: str,
    field: str = "Close",
    dest: Path = DATA_FREE_DIR,
    refresh: bool = False,
) -> pd.Series:
    """Return one daily field for `ticker`, network-fetching at most once.

    Cache hit: verifies the parquet sha256 against the meta sidecar (frozen
    data — a mismatch means tampering/corruption and raises rather than
    silently serving a different dataset), checks the cached field matches,
    and returns without importing yfinance at all. Cache miss or
    refresh=True: fetches period="max" via yfinance and rewrites BOTH files.

    yfinance is imported lazily inside the fetch branch: cached reads must
    work (and validation must pass) on environments without it installed.
    """
    dest = Path(dest)
    stem = _safe_ticker(ticker)
    pq_path = dest / f"{stem}.parquet"
    meta_path = dest / f"{stem}.meta.json"

    if not refresh and pq_path.exists() and not meta_path.exists():
        # A parquet without its provenance sidecar is frozen data of unknown
        # origin — refusing beats silently re-fetching over it (the sidecar
        # is what makes the freeze auditable). Restore the sidecar or pass
        # refresh=True to deliberately replace both files.
        raise RuntimeError(
            f"{pq_path.name} exists but {meta_path.name} is missing — "
            f"restore the meta sidecar or pass refresh=True to re-fetch."
        )
    if not refresh and pq_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        digest = _sha256_file(pq_path)
        if digest != meta.get("sha256_of_parquet"):
            raise RuntimeError(
                f"{pq_path.name}: sha256 mismatch vs meta sidecar — frozen "
                f"data was modified outside fetch_yf_series(refresh=True)."
            )
        if meta.get("field") != field:
            raise ValueError(
                f"{pq_path.name} caches field {meta.get('field')!r}, not "
                f"{field!r} — pass refresh=True (replaces the cached field) "
                f"or a different dest."
            )
        df = pd.read_parquet(pq_path)
        s = df[field]
        s.name = ticker
        return s

    import yfinance as yf  # lazy: not needed for cached reads

    hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker!r}")
    if field not in hist.columns:
        raise KeyError(
            f"{ticker!r}: field {field!r} not in yfinance columns "
            f"{list(hist.columns)}"
        )
    s = hist[field].dropna()
    if s.empty:
        raise RuntimeError(f"{ticker!r}: field {field!r} is all-NaN")
    # tz-naive daily index, matching the repo's bar-cache convention.
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()

    dest.mkdir(parents=True, exist_ok=True)
    s.to_frame(field).to_parquet(pq_path)
    meta = {
        "source": "yfinance",
        "ticker": ticker,
        "field": field,
        "auto_adjust": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "rows": int(len(s)),
        "first": s.index[0].date().isoformat(),
        "last": s.index[-1].date().isoformat(),
        "sha256_of_parquet": _sha256_file(pq_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    out = s.copy()
    out.name = ticker
    return out


if __name__ == "__main__":
    s = fetch_yf_series("^IRX")
    print("^IRX:", len(s), "rows,", s.index[0].date(), "->", s.index[-1].date())
    print(json.dumps(json.load(open(DATA_FREE_DIR / "_IRX.meta.json")), indent=2))
