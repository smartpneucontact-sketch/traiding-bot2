"""norgate_export pure-helper tests — NO norgatedata, NO network.

The VM-side export script cannot run here (the Norgate subscription and the
Windows VM do not exist yet), but its pure helper layer is importable on any
machine and is pinned here against synthetic fixtures that mimic the
documented export format:

1. parse_norgate_symbol: -YYYYMM delisting suffixes vs. share classes,
   month-validity guard, passthrough for listed names.
2. symbol_to_filename: deterministic, Windows+macOS-safe, idempotent.
3. build_membership_frame: union date index, missing dates -> False,
   sorted deterministic columns, boolean dtype, empty input.
4. Manifest construction from a synthetic export tree in tmp: relative
   posix paths, sha256 correctness (recomputed independently), parquet row
   counts, manifest.json self-exclusion, settings/counts passthrough.
5. sha256_file streaming correctness vs hashlib one-shot.

check() asserts so failures also surface under pytest.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python validation/test_norgate_export_helpers.py
(plain script, no pytest dependency needed; exits non-zero on failure)
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from norgate_export import (  # noqa: E402
    START_DATE,
    build_manifest,
    build_membership_frame,
    file_entry,
    parse_norgate_symbol,
    sha256_file,
    symbol_to_filename,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)
    assert cond, f"{name}: {detail}"


# ---------------------------------------------------------------------------
print("1. parse_norgate_symbol")
p = parse_norgate_symbol("SIVB-202303")
check("delisted suffix parsed", p.is_delisted and p.base == "SIVB"
      and p.delist_yyyymm == "202303" and p.symbol == "SIVB-202303", str(p))
p = parse_norgate_symbol("AAPL")
check("listed passthrough", p == ("AAPL", "AAPL", None, False), str(p))
p = parse_norgate_symbol("BRK.B")
check("share class not a delisting", not p.is_delisted and p.base == "BRK.B",
      str(p))
p = parse_norgate_symbol("ABC-201399")
check("month 99 rejected as suffix", not p.is_delisted
      and p.base == "ABC-201399", str(p))
p = parse_norgate_symbol("ABC-201300")
check("month 00 rejected as suffix", not p.is_delisted, str(p))
p = parse_norgate_symbol("X-Y-202210")
check("suffix taken from rightmost dash", p.is_delisted and p.base == "X-Y"
      and p.delist_yyyymm == "202210", str(p))
p = parse_norgate_symbol("  TWTR-202210 ")
check("whitespace stripped", p.is_delisted and p.base == "TWTR", str(p))

# ---------------------------------------------------------------------------
print("2. symbol_to_filename")
check("dot kept", symbol_to_filename("BRK.B") == "BRK.B")
check("delist suffix kept", symbol_to_filename("SIVB-202303") == "SIVB-202303")
check("unsafe chars replaced", symbol_to_filename('A/B:C*D?E"F<G>H|I') ==
      "A_B_C_D_E_F_G_H_I", symbol_to_filename('A/B:C*D?E"F<G>H|I'))
check("idempotent", symbol_to_filename(symbol_to_filename("A/B")) ==
      symbol_to_filename("A/B"))

# ---------------------------------------------------------------------------
print("3. build_membership_frame")
idx1 = pd.to_datetime(["2020-01-02", "2020-01-03"])
idx2 = pd.to_datetime(["2020-01-03", "2020-01-06"])
frame = build_membership_frame({
    "BBB": pd.Series([1, 1], index=idx2),
    "AAA": pd.Series([1, 0], index=idx1),
})
check("columns sorted", list(frame.columns) == ["AAA", "BBB"])
check("union index sorted", list(frame.index) ==
      sorted(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])))
check("boolean dtype", bool(frame.dtypes.eq(bool).all()))
check("member day True", bool(frame.loc[pd.Timestamp("2020-01-02"), "AAA"]))
check("explicit 0 is False",
      not bool(frame.loc[pd.Timestamp("2020-01-03"), "AAA"]))
check("missing date is False",
      not bool(frame.loc[pd.Timestamp("2020-01-06"), "AAA"]))
check("other symbol missing date False",
      not bool(frame.loc[pd.Timestamp("2020-01-02"), "BBB"]))
check("empty input -> empty frame", build_membership_frame({}).empty)

# ---------------------------------------------------------------------------
print("4. manifest from synthetic export tree")
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / "bars").mkdir()
    (root / "membership").mkdir()
    pd.DataFrame(
        {"Open": [1.0, 2.0], "Close": [1.5, 2.5],
         "Unadjusted Close": [1.5, 2.5], "Dividend": [0.0, 0.1]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    ).to_parquet(root / "bars" / "AAPL.parquet")
    pd.DataFrame(
        {"Close": [10.0, 9.0, 0.4]},
        index=pd.to_datetime(["2023-03-08", "2023-03-09", "2023-03-10"]),
    ).to_parquet(root / "bars" / "SIVB-202303.parquet")
    build_membership_frame(
        {"AAPL": pd.Series([1], index=pd.to_datetime(["2020-01-02"]))}
    ).to_parquet(root / "membership" / "sp500.parquet")
    (root / "manifest.json").write_text('{"stale": true}')

    settings = {"start_date": START_DATE, "watchlists": ["fake"]}
    counts = {"bar_files": 2, "delisted_symbols": 1}
    m = build_manifest(root, settings, counts, "9.9.9-test")

    check("manifest.json excluded", m["n_files"] == 3
          and all(e["path"] != "manifest.json" for e in m["files"]),
          str([e["path"] for e in m["files"]]))
    check("relative posix paths sorted",
          [e["path"] for e in m["files"]] ==
          ["bars/AAPL.parquet", "bars/SIVB-202303.parquet",
           "membership/sp500.parquet"], str([e["path"] for e in m["files"]]))
    aapl = m["files"][0]
    check("row count from parquet metadata", aapl["rows"] == 2, str(aapl))
    sivb = m["files"][1]
    check("delisted bar rows", sivb["rows"] == 3, str(sivb))
    expected_sha = hashlib.sha256(
        (root / "bars" / "AAPL.parquet").read_bytes()).hexdigest()
    check("sha256 matches independent hashlib", aapl["sha256"] == expected_sha)
    check("byte sizes recorded", all(e["bytes"] > 0 for e in m["files"]))
    check("settings passthrough", m["settings"] == settings)
    check("counts passthrough", m["counts"] == counts)
    check("version recorded", m["norgatedata_version"] == "9.9.9-test")
    check("UTC timestamp present", "export_timestamp_utc" in m
          and m["export_timestamp_utc"].endswith("+00:00"),
          m.get("export_timestamp_utc", "<missing>"))

    # file_entry on a non-parquet file: no rows key
    txt = root / "notes.txt"
    txt.write_text("hello")
    e = file_entry(txt, root)
    check("non-parquet entry has no rows", "rows" not in e
          and e["path"] == "notes.txt", str(e))

# ---------------------------------------------------------------------------
print("5. sha256_file streaming vs one-shot")
with tempfile.TemporaryDirectory() as td:
    big = Path(td) / "big.bin"
    big.write_bytes(b"\x01\x02" * (1 << 20))  # 2 MiB, crosses chunk boundary
    check("streaming sha256 == hashlib one-shot",
          sha256_file(big) == hashlib.sha256(big.read_bytes()).hexdigest())

# ---------------------------------------------------------------------------
if FAILURES:
    print(f"\nFAILED: {len(FAILURES)} checks: {FAILURES}")
    sys.exit(1)
print("\nALL PASS: norgate_export pure helpers verified against synthetic "
      "fixtures.")
