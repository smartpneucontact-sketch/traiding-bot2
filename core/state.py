"""Per-model run state — JSON-on-disk, one file per model.

State persists between runs (cron-driven daily pipeline). Tracks the last
rebalance time, the run counter, and the rolling history of past run
summaries. The cutloss scanner also stores its peak-prices and
portfolio-stop trip flag here under separate keys.

Schema is intentionally untyped (`dict`) — both this module and the
cutloss scanner add keys to it freely, and a strict schema would just
ratchet up coupling for no real safety win.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid circular import at runtime
    from pipeline import ModelConfig


# Serialises read-modify-write windows on the state file between the 60s
# cutloss scanner and the daily pipeline (both run in the dashboard
# process). Lives here so both core.risk and core.runner can share it
# without an import cycle.
state_lock = threading.Lock()

# Keys owned by the cutloss scanner. The pipeline holds its loaded state
# dict across a multi-minute rebalance; before saving it must re-read the
# disk copy and adopt these keys, or a stop that fired mid-rebalance gets
# erased (trip flag, cool-down, sold-today ledger, peaks, daily anchors).
#
# daily_book_start / daily_book_start_date have TWO writers — the
# scanner's ~09:30 first-tick anchor and the runner's post-rebalance
# refresh (core.risk.refresh_daily_book_anchor_after_rebalance) — but
# both write straight to disk under state_lock, so the disk copy is
# always the authoritative one to adopt here. Before these keys were
# listed, a pipeline run whose Step-2 snapshot predated the scanner's
# anchor would silently drop it at the merged save.
SCANNER_OWNED_KEYS = (
    "peak_prices",
    "daily_portfolio_start",
    "daily_portfolio_start_date",
    "daily_book_start",
    "daily_book_start_date",
    "portfolio_stop_tripped_date",
    "reentry_cooldown_until",
    "cutloss_sold_today",
    # Gate-cadence tracking (core/gate_update.py): the applied multiplier
    # is written straight to disk under the lock by THREE writers — the
    # rebalance path, the daily gate pass, and the tier scaler — so the
    # pipeline's merged save must always adopt the disk copy.
    # tier_floor_multiplier: cumulative tier-cut fraction for the current
    # rebalance cycle (stop_grid cuts persist to the next rebalance —
    # 2026-08-31 audit); gate_ref: rebalance-time {gross_pre_gate,
    # max_gross_exposure} so the daily pass can re-apply the gross cap;
    # gate_data_skips: consecutive thin-data gate-pass skips (stall alarm).
    "applied_exposure_multiplier",
    "tier_floor_multiplier",
    "gate_ref",
    "gate_data_skips",
    "gate_update_history",
)


def load_state(mc: "ModelConfig") -> dict:
    """Load pipeline state from JSON for a specific model.

    Returns a fresh empty-history dict if the state file doesn't exist
    (first run for this slot).
    """
    mc.state_path.parent.mkdir(parents=True, exist_ok=True)
    if mc.state_path.exists():
        return json.loads(mc.state_path.read_text())
    return {"last_rebalance": None, "last_run": None, "run_count": 0, "history": []}


def save_state(state: dict, mc: "ModelConfig") -> None:
    """Save pipeline state to JSON for a specific model.

    Atomic (write-to-temp + os.replace, 2026-08-31 audit): the dashboard
    and other processes read these files without the in-process lock, so
    a plain write_text could hand them a torn file mid-write."""
    mc.state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = mc.state_path.with_name(mc.state_path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, mc.state_path)


def merge_scanner_state(state: dict, mc: "ModelConfig") -> dict:
    """Adopt the scanner-owned keys from the on-disk state into `state`.

    Call (under `state_lock`) right before the pipeline saves a state dict
    it has held across a long-running rebalance. If a Tier-3 portfolio
    stop tripped while the pipeline was working, also honor the scanner's
    last_rebalance=None so the post-cool-down re-entry still happens.
    """
    disk = load_state(mc)
    for key in SCANNER_OWNED_KEYS:
        if key in disk:
            state[key] = disk[key]
        else:
            state.pop(key, None)
    today_iso = datetime.now().strftime("%Y-%m-%d")
    if disk.get("portfolio_stop_tripped_date") == today_iso:
        state["last_rebalance"] = disk.get("last_rebalance")
    return state


def trading_days_between(start: datetime, end: datetime) -> int:
    """Count Mon–Fri days strictly between `start` and `end`.

    Holidays aren't subtracted — overshooting by a day on the rebalance
    cadence is harmless and avoids importing the holiday calendar here.
    Same-day returns 0.
    """
    if end <= start:
        return 0
    days = 0
    cur = start.date() + timedelta(days=1)
    end_date = end.date()
    while cur <= end_date:
        if cur.weekday() < 5:
            days += 1
        cur = cur + timedelta(days=1)
    return days


def should_rebalance(state: dict, horizon_days: int, force: bool = False) -> bool:
    """Return True if `horizon_days` trading days have passed since the
    last rebalance (or if `force` is set, or if it's never run).

    `horizon_days` is passed in (instead of imported from pipeline.HORIZON)
    so this module stays decoupled from the pipeline-level config.
    """
    if force:
        return True
    last = state.get("last_rebalance")
    if last is None:
        return True
    last_date = datetime.fromisoformat(last)
    return trading_days_between(last_date, datetime.now()) >= horizon_days
