"""US equity market hours and holiday calendar.

Half-days (early closes — the day after Thanksgiving and Christmas Eve)
close at 13:00 ET (EARLY_CLOSE_DATES / session_close); is_market_open and
the dashboard's catch-up / wait cutoffs honour that close (review round 6).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo


US_MARKET_HOLIDAYS: frozenset[str] = frozenset({
    # 2025
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17",
    "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
    "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
    "2027-11-25", "2027-12-24",
})


# NYSE early closes (13:00 ET): the day after Thanksgiving and Christmas
# Eve when it falls on a weekday (plus July 3 when it is a weekday and
# July 4 is not observed on it). Review round 6: the boot catch-up and the
# main run's wait cutoff must not submit into a closed half-day session.
EARLY_CLOSE_DATES: frozenset[str] = frozenset({
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
    "2027-11-26",
})

_NY = ZoneInfo("America/New_York")


def session_close(now: Optional[datetime] = None) -> datetime:
    """The regular-session close for `now`'s date in ET: 13:00 on early-close
    dates, 16:00 otherwise (the caller decides whether the day is open)."""
    if now is None:
        now = datetime.now(_NY)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_NY)
    hour = 13 if now.strftime("%Y-%m-%d") in EARLY_CLOSE_DATES else 16
    return now.replace(hour=hour, minute=0, second=0, microsecond=0)


def is_auction_window(now: Optional[datetime] = None) -> bool:
    """True on a trading day between 09:00 and 09:27 ET — the window in
    which opening-auction orders (time_in_force "opg") can still be
    submitted to Alpaca (its cutoff is 09:28 ET). Same calendar as
    is_market_open."""
    if now is None:
        now = datetime.now(_NY)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_NY)
    if now.weekday() >= 5:
        return False
    if now.strftime("%Y-%m-%d") in US_MARKET_HOLIDAYS:
        return False
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    cutoff = now.replace(hour=9, minute=27, second=0, microsecond=0)
    return start <= now < cutoff


def is_market_open(now: Optional[datetime] = None) -> bool:
    """Return True if US equities are open at `now` (defaults to now in ET).

    Closed on weekends, US market holidays, and outside 09:30–16:00 ET.
    """
    if now is None:
        now = datetime.now(_NY)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_NY)
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    if now.strftime("%Y-%m-%d") in US_MARKET_HOLIDAYS:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = session_close(now)
    return open_t <= now <= close_t
