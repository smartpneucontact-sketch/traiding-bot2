"""Track B, Step B0 foundation tests: exp_lib margin plumbing + data_free.

1. run_trial margin passthrough — WITHOUT touching the real data_cache or
   ledgers: exp_lib.load_cache / union_prices_cached / run_backtest_v2 are
   monkeypatched with tiny fakes, the captured BTConfigV2 is inspected, and
   trial rows go to throwaway families whose CSVs are deleted afterwards.
   Also pins the byte-identical-default contract (no new params_json key or
   ledger column at margin 0.0) and the new-family-only schema guard.
2. data_free.fetch_yf_series — fetches a tiny REAL series (^IRX) into a temp
   dir (needs network on this step only), verifies parquet+meta+sha256, then
   proves the second call serves the frozen cache without any network by
   stubbing sys.modules["yfinance"] to raise. refresh=True is NOT exercised
   (it would be a second network hit; same code path as the first fetch).

check() asserts (not just records) so failures also surface under pytest.

Run:  cd "Traiding 11" && /opt/anaconda3/bin/python validation/test_b0_foundation.py
(plain script, no pytest dependency needed; exits non-zero on failure)
"""
from __future__ import annotations

import inspect
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import exp_lib  # noqa: E402
from data_free import _safe_ticker, _sha256_file, fetch_yf_series  # noqa: E402
from engine_v2 import BTConfigV2  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = ""):
    flag = "OK  " if cond else "FAIL"
    print(f"  {flag} {label} {detail}")
    if not cond:
        FAILURES.append(label)
    assert cond, f"{label} {detail}".strip()


# ───── 1. run_trial margin passthrough (fully faked I/O) ─────────────────
def test_margin_passthrough():
    print("\n[1] exp_lib.run_trial margin_bps_annual plumbing")

    sig = inspect.signature(exp_lib.run_trial)
    check("signature has margin_bps_annual",
          "margin_bps_annual" in sig.parameters)
    check("default is 0.0",
          sig.parameters["margin_bps_annual"].default == 0.0)

    idx = pd.bdate_range("2021-01-04", periods=10)
    px = pd.DataFrame({"AAA": np.linspace(100, 109, 10),
                       "BBB": np.linspace(50, 59, 10)}, index=idx)
    panel = {"open": px * 0.99, "close": px}
    weights = pd.DataFrame([[1.0, 0.0]], index=[idx[0]],
                           columns=["AAA", "BBB"])
    eq = pd.Series(np.linspace(100_000, 101_000, 10), index=idx)
    fake_summary = {"mean_monthly": 0.01, "sharpe": 1.0, "max_drawdown": -0.1}

    captured: dict = {}

    def fake_run_backtest_v2(w, pu, cfg, name="", open_prices=None,
                             weights_are_daily=False):
        captured["cfg"] = cfg
        return {"equity": eq, "returns": eq.pct_change().fillna(0.0),
                "weights": px * 0.0, "summary": dict(fake_summary)}

    saved = (exp_lib.load_cache, exp_lib.union_prices_cached,
             exp_lib.run_backtest_v2)
    exp_lib.load_cache = lambda: (panel, pd.DataFrame(), {})
    exp_lib.union_prices_cached = lambda: px
    exp_lib.run_backtest_v2 = fake_run_backtest_v2

    fam_margin, fam_plain = "_b0_tmp_margin", "_b0_tmp_plain"
    led_margin = exp_lib.TRIALS_DIR / f"{fam_margin}.csv"
    led_plain = exp_lib.TRIALS_DIR / f"{fam_plain}.csv"
    try:
        # margin=600 run into a NEW family
        params_in = {"x": 1}
        res = exp_lib.run_trial(weights, name="b0_margin", family=fam_margin,
                                params=params_in, margin_bps_annual=600.0)
        cfg = captured["cfg"]
        check("cfg is BTConfigV2", isinstance(cfg, BTConfigV2))
        check("cfg.margin_bps_annual == 600", cfg.margin_bps_annual == 600.0)
        check("row margin_bps == 600", res["row"].get("margin_bps") == 600.0)
        pj = json.loads(res["row"]["params_json"])
        check("params_json carries margin",
              pj.get("margin_bps_annual") == 600.0 and pj.get("x") == 1)
        check("caller params dict not mutated", params_in == {"x": 1})
        df = pd.read_csv(led_margin)
        check("ledger CSV has margin_bps column",
              "margin_bps" in df.columns and float(df["margin_bps"].iloc[0]) == 600.0)

        # default run: byte-identical legacy schema (no key, no column)
        res0 = exp_lib.run_trial(weights, name="b0_plain", family=fam_plain,
                                 params={"x": 1})
        check("default cfg.margin_bps_annual == 0",
              captured["cfg"].margin_bps_annual == 0.0)
        check("default row has no margin_bps", "margin_bps" not in res0["row"])
        check("default params_json unchanged",
              json.loads(res0["row"]["params_json"]) == {"x": 1})
        with open(led_plain) as f:
            header = f.readline().strip().split(",")
        check("default ledger header has no margin_bps",
              "margin_bps" not in header)

        # schema guard: margin run into a pre-margin family must refuse
        raised = False
        try:
            exp_lib.run_trial(weights, name="b0_guard", family=fam_plain,
                              params={"x": 2}, margin_bps_annual=600.0)
        except ValueError:
            raised = True
        check("margin into pre-margin family raises ValueError", raised)
        check("guard did not append to ledger",
              len(pd.read_csv(led_plain)) == 1)
    finally:
        (exp_lib.load_cache, exp_lib.union_prices_cached,
         exp_lib.run_backtest_v2) = saved
        led_margin.unlink(missing_ok=True)
        led_plain.unlink(missing_ok=True)


# ───── 2. data_free fetch + frozen cache ─────────────────────────────────
def test_data_free():
    print("\n[2] data_free.fetch_yf_series (^IRX, temp dest; network on 1st call)")
    check("safe_ticker ^IRX", _safe_ticker("^IRX") == "_IRX")
    check("safe_ticker BRK/B", _safe_ticker("BRK/B") == "BRK_B")

    dest = Path(tempfile.mkdtemp(prefix="b0_data_free_"))
    saved_mod = sys.modules.get("yfinance")
    try:
        s1 = fetch_yf_series("^IRX", dest=dest)
        pq = dest / "_IRX.parquet"
        mp = dest / "_IRX.meta.json"
        check("parquet written", pq.exists())
        check("meta sidecar written", mp.exists())
        with open(mp) as f:
            meta = json.load(f)
        check("meta.source", meta.get("source") == "yfinance")
        check("meta.ticker/field",
              meta.get("ticker") == "^IRX" and meta.get("field") == "Close")
        check("meta.rows matches series", meta.get("rows") == len(s1))
        check("meta first/last match index",
              meta.get("first") == s1.index[0].date().isoformat()
              and meta.get("last") == s1.index[-1].date().isoformat())
        check("sha256 matches parquet bytes",
              meta.get("sha256_of_parquet") == _sha256_file(pq))
        check("series sane (^IRX history reaches pre-2000)",
              len(s1) > 5000 and s1.index.is_monotonic_increasing)

        # freeze: stub yfinance so ANY import/use after this is a hard fail
        stub = types.ModuleType("yfinance")

        def _no_network(*a, **k):
            raise AssertionError("network hit: cached read touched yfinance")
        stub.Ticker = _no_network
        stub.download = _no_network
        sys.modules["yfinance"] = stub

        s2 = fetch_yf_series("^IRX", dest=dest)
        pd.testing.assert_series_equal(s1, s2, check_freq=False)
        check("second call served from frozen cache (no network)", True)

        raised = False
        try:
            fetch_yf_series("^IRX", field="Open", dest=dest)
        except ValueError:
            raised = True
        check("cached-field mismatch raises ValueError", raised)

        # tamper: frozen data modified outside refresh must be rejected
        with open(pq, "ab") as f:
            f.write(b"\x00")
        raised = False
        try:
            fetch_yf_series("^IRX", dest=dest)
        except RuntimeError:
            raised = True
        check("tampered parquet raises RuntimeError", raised)
    finally:
        if saved_mod is not None:
            sys.modules["yfinance"] = saved_mod
        else:
            sys.modules.pop("yfinance", None)
        shutil.rmtree(dest, ignore_errors=True)


if __name__ == "__main__":
    test_margin_passthrough()
    test_data_free()
    print(f"\n{'ALL OK' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
