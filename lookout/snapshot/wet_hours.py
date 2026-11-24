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

The Rain tab's wet hours since Oct 1 (#663) end at the same hour, and
are set against usual wet hours to date (GLOSSARY): each earlier water
year that covers this date (at least half its hours from Oct 1 to this
date countable) gives its wet share, times this year's countable hours.

Functions:
- wet_hours(stitched, clock): The `WetHours` field.
- wet_hours_to_date(stitched, clock): The Rain tab's `WetHoursToDate`.
"""

from datetime import timedelta
from typing import List, Optional, Tuple

import pandas as pd

from lookout.core.rainfall_analysis import water_year, water_year_start
from lookout.core.wet_hours import daily_wet_hours, hour_start, judge_hours
from lookout.snapshot.clock import StationClock
from lookout.snapshot.model import (
    EarlierWetYear,
    UsualWetHours,
    WetHours,
    WetHoursDay,
    WetHoursToDate,
    WetHoursWindow,
)
from lookout.utils.dateutc import HOUR_MS, day_span, from_local_date, local_time

WINDOW_DAYS = (7, 30)
"""The windows' lengths in days; the longest sets the hours judged."""

STRIP_DAYS = 30
"""The strip's days: today and the 29 before it."""

MIN_COVERED = 0.5
"""An earlier year covers the date with at least this share of its hours
from Oct 1 to the date countable."""

BAND_YEARS = 3
"""From this many earlier years the usual is a band, not plain values."""

PERCENTILE_YEARS = 4
"""From this many earlier years the band is called a percentile band."""

BAND = (0.25, 0.75)
"""The band's percentiles of the earlier years' wet shares."""


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


def wet_hours_to_date(stitched: pd.DataFrame, clock: StationClock) -> WetHoursToDate:
    """
    This water year's wet hours since Oct 1, against usual wet hours to date.

    :param stitched: The archive with the live reading, any order.
    :param clock: The station clock.
    :return: The `WetHoursToDate`.
    """
    end_ms = hour_start(clock.now_ms, clock.tz)
    end = local_time(end_ms, clock.tz)
    year = water_year(end.date())
    first = water_year_start(year)
    hours = judge_hours(stitched, clock.tz, from_local_date(first, clock.tz), end_ms)
    countable = int(hours["countable"].sum())
    earlier = _earlier_years(stitched, clock.tz, year, end, countable)
    return WetHoursToDate(
        water_year=year,
        first_day=first.isoformat(),
        through_ms=end_ms,
        hours=len(hours),
        wet_hours=int(hours["wet"].sum()),
        countable_hours=countable,
        missing_hours=len(hours) - countable,
        usual=_usual(earlier, countable),
    )


def _same_time(end: pd.Timestamp, year: int, tz: str) -> int:
    """
    This date and hour in an earlier water year, as epoch ms.

    :param end: This year's end, station-local.
    :param year: The earlier water year.
    :param tz: The station-local zone.
    :return: The same month, day and hour in that year; Feb 29 is Feb 28.
    """
    calendar_year = year - 1 if end.month >= water_year_start(year).month else year
    day = end.day
    if (end.month, day) == (2, 29):
        day = 28
    when = pd.Timestamp(calendar_year, end.month, day, end.hour).tz_localize(
        tz, ambiguous=True, nonexistent="shift_forward"
    )
    return int(when.value // 10**6)


def _earlier_years(
    stitched: pd.DataFrame, tz: str, year: int, end: pd.Timestamp, countable: int
) -> List[EarlierWetYear]:
    """
    Each earlier water year that covers this date, oldest first.

    :param stitched: The archive.
    :param tz: The station-local zone.
    :param year: This water year.
    :param end: This year's end, station-local.
    :param countable: This year's countable hours, which scale each share.
    :return: The years whose span from Oct 1 is at least half countable.
    """
    if stitched.empty:
        return []
    oldest = local_time(int(stitched["dateutc"].min()), tz).date()
    years = []
    for earlier in range(water_year(oldest), year):
        start_ms = from_local_date(water_year_start(earlier), tz)
        hours = judge_hours(stitched, tz, start_ms, _same_time(end, earlier, tz))
        counted = int(hours["countable"].sum())
        if counted == 0 or counted < MIN_COVERED * len(hours):
            continue
        wet = int(hours["wet"].sum())
        years.append(
            EarlierWetYear(
                water_year=earlier,
                wet_hours=wet,
                countable_hours=counted,
                usual_hours=_as_hours(wet / counted, countable),
            )
        )
    return years


def _as_hours(share: float, countable: int) -> int:
    """
    A wet share as this year's wet hours: the one scaling of usual wet hours.

    :param share: An earlier year's wet share, or a percentile of them.
    :param countable: This year's countable hours.
    :return: Whole hours.
    """
    return round(share * countable)


def _usual(earlier: List[EarlierWetYear], countable: int) -> Optional[UsualWetHours]:
    """
    Usual wet hours to date from the earlier years that cover the date.

    :param earlier: Those years, oldest first.
    :param countable: This year's countable hours.
    :return: The band from `BAND_YEARS` years, the plain values below;
        None with no year.
    """
    if not earlier:
        return None
    low = high = band = None
    if len(earlier) >= BAND_YEARS:
        shares = pd.Series([y.wet_hours / y.countable_hours for y in earlier])
        low, high = (_as_hours(float(shares.quantile(q)), countable) for q in BAND)
        band = "percentile" if len(earlier) >= PERCENTILE_YEARS else "range"
    return UsualWetHours(
        years=len(earlier),
        earlier=tuple(earlier),
        low_hours=low,
        high_hours=high,
        band=band,
    )
