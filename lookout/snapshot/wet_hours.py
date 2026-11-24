"""
wet_hours.py: The Overview's wet hours, over the last 7 and 30 days (#657).

For each window, its wet hours of its countable hours, and the hours and
days it is missing; and the 30-day strip, each station-local day's wet and
countable hours. Every hour is judged by the one wet-hour helper
(`lookout.core.wet_hours`); a missing hour lowers the countable hours,
never the wet ones, and is never dry.

The windows end with the last whole hour before the build, in the station
zone: an hour is judged once it is over, so the hour in progress is in
neither window. The 7-day window is the 168 hours before that, the 30-day
window the 720. The strip is today and the 29 days before it, today with
its whole hours so far.

Functions:
- wet_hours(stitched, clock): The `WetHours` field.
"""

from datetime import timedelta
from typing import Tuple

import pandas as pd

from lookout.core.wet_hours import daily_wet_hours, hour_start, judge_hours
from lookout.snapshot.clock import StationClock
from lookout.snapshot.model import WetHours, WetHoursDay, WetHoursWindow
from lookout.utils.dateutc import HOUR_MS, day_span, from_local_date

WINDOW_DAYS = (7, 30)
"""The windows' lengths in days; the longest sets the hours judged."""

STRIP_DAYS = 30
"""The strip's days: today and the 29 before it."""


def wet_hours(stitched: pd.DataFrame, clock: StationClock) -> WetHours:
    """
    The wet hours over the last 7 and 30 days, and the 30-day strip.

    :param stitched: The archive with the live reading, any order.
    :param clock: The station clock.
    :return: The `WetHours`.
    """
    end_ms = hour_start(clock.now_ms, clock.tz)
    first_ms = min(
        end_ms - max(WINDOW_DAYS) * 24 * HOUR_MS,
        from_local_date(clock.today - timedelta(days=STRIP_DAYS - 1), clock.tz),
    )
    hours = judge_hours(stitched, clock.tz, first_ms, end_ms)
    week, month = WINDOW_DAYS
    return WetHours(
        through_ms=end_ms,
        last_7_days=_window(hours, week, end_ms),
        last_30_days=_window(hours, month, end_ms),
        days=_strip(hours, clock),
    )


def _window(hours: pd.DataFrame, days: int, end_ms: int) -> WetHoursWindow:
    """
    One window's tally: wet of countable hours, and what it is missing.

    :param hours: The judged hours, at least the window's.
    :param days: The window's length in days.
    :param end_ms: The window's end, exclusive: the current hour's start.
    :return: The window.
    """
    span = days * 24
    inside = hours[hours["hour_ms"] >= end_ms - span * HOUR_MS]
    countable = int(inside["countable"].sum())
    per_day = daily_wet_hours(inside)
    return WetHoursWindow(
        days=days,
        hours=span,
        wet_hours=int(inside["wet"].sum()),
        countable_hours=countable,
        missing_hours=span - countable,
        missing_days=int((per_day["countable_hours"] == 0).sum()),
    )


def _strip(hours: pd.DataFrame, clock: StationClock) -> Tuple[WetHoursDay, ...]:
    """
    Each of the strip's days: its wet and countable hours.

    :param hours: The judged hours, from at least the strip's first midnight.
    :param clock: The station clock.
    :return: `STRIP_DAYS` `WetHoursDay`s, oldest first; null for a day with
        no countable hour.
    """
    first = clock.today - timedelta(days=STRIP_DAYS - 1)
    first_ms = from_local_date(first, clock.tz)
    per_day = daily_wet_hours(hours[hours["hour_ms"] >= first_ms]).set_index("day")
    strip = []
    for day in day_span(first, clock.today):
        countable = int(per_day["countable_hours"].get(day, 0))
        wet = int(per_day["wet_hours"].get(day, 0))
        strip.append(
            WetHoursDay(
                day=day.isoformat(),
                wet_hours=wet if countable else None,
                countable_hours=countable or None,
            )
        )
    return tuple(strip)
