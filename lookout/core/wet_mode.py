"""
wet_mode.py: Wet mode or dry mode for each day, a pure replay (#656).

Wet mode starts when the last 7 days' wet share reaches 15% and ends when
the last 14 days' share falls below 2% (GLOSSARY: wet share, wet mode).
When fewer than half the last 14 days' hours are countable, the calendar
decides instead: Nov 1 to Apr 30 is wet. The mode is never stored: it is
replayed over the archive's daily series of wet and countable hours from a
fixed start state, so a rebuild or a first deploy gives the same mode
(spec #647).

Functions:
- wet_mode(days): The mode for each day, and what decided it.
"""

import pandas as pd

ON_DAYS = 7
"""Wet mode starts on the wet share of this many days before the day."""

ON_SHARE = 0.15
"""Wet mode starts when the `ON_DAYS` wet share reaches this."""

OFF_DAYS = 14
"""Wet mode ends on the wet share of this many days before the day."""

OFF_SHARE = 0.02
"""Wet mode ends when the `OFF_DAYS` wet share falls below this."""

MIN_COUNTED = OFF_DAYS * 24 / 2
"""
Fewer countable hours than this in the `OFF_DAYS` before a day, and the
calendar decides its mode: half the window's hours.
"""

WET_MONTHS = (11, 12, 1, 2, 3, 4)
"""The calendar's wet months, Nov 1 to Apr 30."""

WET = "wet"
DRY = "dry"
WEATHER = "weather"
CALENDAR = "calendar"


def _before(hours: pd.Series, day: int, span: int) -> float:
    """The sum of `hours` over the `span` days before day number `day`."""
    return float(hours.iloc[max(0, day - span) : day].sum())  # noqa: E203


def _share(wet: pd.Series, countable: pd.Series, day: int, span: int) -> float:
    """The wet share of the `span` days before day `day`; NaN if none counted."""
    counted = _before(countable, day, span)
    if counted == 0:
        return float("nan")
    return _before(wet, day, span) / counted


def wet_mode(days: pd.DataFrame) -> pd.DataFrame:
    """
    The mode for each day, replayed from the first day of `days`.

    A day's mode is decided by the whole days before it, never by the day
    itself, so the mode holds through a day. A day absent from `days`, and
    any day before the first, has no countable hour. So the replay starts
    on the calendar, which decides until half the last 14 days can be
    counted, and the weather goes on from the calendar's mode. A window
    with no countable hour holds the mode.

    :param days: One row per station-local day, oldest first, indexed by
        the day (a DatetimeIndex), with columns `wet_hours` and
        `countable_hours`.
    :return: Indexed like `days`, with columns `mode` ("wet" or "dry") and
        `decided_by` ("weather" or "calendar").
    """
    every_day = pd.date_range(days.index.min(), days.index.max(), freq="D")
    full = days.reindex(every_day, fill_value=0)
    wet = full["wet_hours"].astype(float)
    countable = full["countable_hours"].astype(float)
    mode = DRY
    rows = []
    for i, day in enumerate(every_day):
        if _before(countable, i, OFF_DAYS) < MIN_COUNTED:
            mode = WET if day.month in WET_MONTHS else DRY
            rows.append((mode, CALENDAR))
            continue
        if mode == DRY and _share(wet, countable, i, ON_DAYS) >= ON_SHARE:
            mode = WET
        elif mode == WET and _share(wet, countable, i, OFF_DAYS) < OFF_SHARE:
            mode = DRY
        rows.append((mode, WEATHER))
    result = pd.DataFrame(rows, index=every_day, columns=["mode", "decided_by"])
    return result.reindex(days.index)
