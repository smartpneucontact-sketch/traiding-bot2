"""Universe churn monitor (2026-08-30, Aug 18–28 outage follow-up).

The Aug 18–28 silent outage happened because the tradeable universe had
decayed to 497–499 names — just under a data guard that sat exactly at
the normal operating point — and nothing tracked the decay. The guard
now aborts loudly (data_guard_aborts alarm), but the DRIFT itself was
invisible: names delist or get renamed, the union universe only ever
shrinks, and nobody sees the trend until a threshold breaks.

This module records one snapshot per pipeline day: how many names had
usable bars, which names appeared/disappeared vs the previous snapshot,
and an early warning when the count crosses UNIVERSE_WARN_THRESHOLD
(default 450 — deliberately ABOVE the hard MIN_STOCKS_REQUIRED=400 abort
so the operator hears about decay while rebalances still run).

The history lives on the Railway volume (universe_churn.json) and is
surfaced by the dashboard's /api/status.
"""

import json
import os
from datetime import datetime
from pathlib import Path

_DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
CHURN_PATH = _DATA_DIR / "universe_churn.json"
WARN_THRESHOLD = int(os.environ.get("UNIVERSE_WARN_THRESHOLD", "450"))
_MAX_SNAPSHOTS = 120


def record_universe_snapshot(symbols_with_bars: list[str], logger) -> dict:
    """Append today's universe snapshot (idempotent per calendar day —
    a re-run replaces the same day's entry). Never raises."""
    today = datetime.now().strftime("%Y-%m-%d")
    snap = {"date": today, "count": len(symbols_with_bars)}
    try:
        history: list[dict] = []
        prev_names: list[str] | None = None
        if CHURN_PATH.exists():
            payload = json.loads(CHURN_PATH.read_text())
            history = payload.get("history", [])
            prev_names = payload.get("last_names")
        # Same-day re-run: drop today's earlier entry, diff vs yesterday's
        # names is preserved because last_names is only rewritten below.
        history = [h for h in history if h.get("date") != today]

        if prev_names is not None:
            prev_set, cur_set = set(prev_names), set(symbols_with_bars)
            snap["added"] = sorted(cur_set - prev_set)[:25]
            snap["removed"] = sorted(prev_set - cur_set)[:25]
            snap["n_added"] = len(cur_set - prev_set)
            snap["n_removed"] = len(prev_set - cur_set)
        snap["warn"] = snap["count"] < WARN_THRESHOLD
        history.append(snap)
        history = history[-_MAX_SNAPSHOTS:]

        CHURN_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHURN_PATH.write_text(json.dumps({
            "history": history,
            "last_names": sorted(symbols_with_bars),
            "warn_threshold": WARN_THRESHOLD,
            "updated": datetime.now().isoformat(),
        }, indent=1))

        if snap["warn"]:
            logger.warning(
                f"[UNIVERSE] count {snap['count']} is below the early-warning "
                f"threshold {WARN_THRESHOLD} (hard abort at "
                f"MIN_STOCKS_REQUIRED) — the universe is decaying; "
                f"see universe_churn.json for the names lost."
            )
        elif snap.get("n_removed"):
            logger.info(
                f"[UNIVERSE] {snap['count']} names "
                f"(+{snap.get('n_added', 0)}/-{snap['n_removed']} vs prev): "
                f"lost {snap['removed'][:8]}"
            )
        else:
            logger.info(f"[UNIVERSE] {snap['count']} names with usable bars")
    except Exception as e:
        logger.error(f"[UNIVERSE] churn snapshot failed (non-fatal): {e}")
    return snap


def load_universe_status() -> dict | None:
    """Compact view for the dashboard: current count, threshold, trend of
    the last 30 snapshots, most recent losses. Never raises."""
    try:
        if not CHURN_PATH.exists():
            return None
        payload = json.loads(CHURN_PATH.read_text())
        history = payload.get("history", [])
        if not history:
            return None
        latest = history[-1]
        return {
            "count": latest.get("count"),
            "date": latest.get("date"),
            "warn": bool(latest.get("warn")),
            "warn_threshold": payload.get("warn_threshold", WARN_THRESHOLD),
            "trend": [{"date": h.get("date"), "count": h.get("count")}
                      for h in history[-30:]],
            "recent_removed": latest.get("removed", []),
        }
    except Exception:
        return None
