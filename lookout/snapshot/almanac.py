"""
almanac.py: The Overview's almanac grid and rolling-window rain table (#82).

- The grid: one row per water year (Oct through Sep, GLOSSARY.md), newest
  first, each month's rain under its own gap status, and the season's total.
- The rolling table: rain over the last 7, 30, 90 and 365 days and the
  season so far, each against the same window ending on the same date in
  every earlier year the archive reaches, newest first.

Every total comes from the build's `CountedDays` (`rain` and `rain_total`),
the Rain tab's own, so a window has one rain total and one status everywhere
(the One Number Rule): an archive sum of each day's highest `dailyrainin`,
with today's live total stitched in, judged and prorated by the gap rules.
Ambient's year, month and week counters are never read.

Functions:
- almanac(days): the `Almanac` section.
"""

from datetime import date, timedelta
from typing import List, Optional, Tuple

from lookout.core.rainfall_analysis import water_year, water_year_start
from lookout.snapshot.days import CountedDays
from lookout.snapshot.gaps import STANDARD_WINDOWS, ArchiveCoverage
from lookout.snapshot.model import (
    Almanac,
    AlmanacMonth,
    AlmanacYear,
    RollingWindow,
    RollingYear,
)

ROLLING_DAYS: Tuple[Tuple[str, int], ...] = tuple(
    (key, first_back + 1)
    for key, first_back, last_back in STANDARD_WINDOWS
    if last_back == 0 and first_back > 0
)
"""The standard windows that end today and span more than a day (7, 30,
90 and 365 days), as `Gaps.windows` keys them; the season ("water_year")
follows them."""

SEASON = "water_year"


def almanac(days: CountedDays) -> Almanac:
    """
    The almanac grid and the rolling-window rain table.

    :param days: The build's counted days and daily rain series; windows
        are judged by its rain day kind.
    :return: The section.
    """
    return Almanac(years=_years(days), rolling=_rolling(days))


def _years(days: CountedDays) -> Tuple[AlmanacYear, ...]:
    """
    Every water year from the archive's first through the current one.

    :param days: The counted days.
    :return: The rows, newest first; none for an empty archive.
    """
    coverage = days.rain
    first_day = coverage.first_day
    if first_day is None:
        return ()
    today = coverage.today
    rows = []
    for year in range(water_year(today), water_year(first_day) - 1, -1):
        start = water_year_start(year)
        months: List[Optional[AlmanacMonth]] = []
        for offset in range(12):
            first = _add_months(start, offset)
            months.append(None if first > today else _month(first, days))
        season = coverage.window(
            start,
            water_year_start(year + 1) - timedelta(days=1),
            key=f"{SEASON}_{year}",
        )
        rows.append(
            AlmanacYear(
                water_year=year,
                months=tuple(months),
                season=season,
                season_in=days.rain_total(season),
            )
        )
    return tuple(rows)


def _month(first: date, days: CountedDays) -> AlmanacMonth:
    """
    One month's cell; the current month also carries its rain so far.

    :param first: The month's first day.
    :param days: The counted days.
    :return: The cell.
    """
    coverage = days.rain
    status = coverage.month(first.year, first.month)
    to_date = None
    if _add_months(first, 1) > coverage.today:
        to_date = coverage.window(first, coverage.today)
    return AlmanacMonth(
        status=status,
        rain_in=days.rain_total(status),
        to_date=to_date,
        to_date_in=(None if to_date is None else days.rain_total(to_date)),
    )


def _rolling(days: CountedDays) -> Tuple[RollingWindow, ...]:
    """
    Each rolling window, this year's then each earlier year's.

    This year's windows are the snapshot's standard windows, keys and all,
    so they match `Gaps.windows` and the Rain tab's totals.

    :param days: The counted days.
    :return: The windows, shortest first, the season last.
    """
    coverage = days.rain
    ends = _same_dates(coverage)
    rows = []
    for key, length in (*ROLLING_DAYS, (SEASON, None)):
        years = []
        for back, end in enumerate(ends):
            first = (
                water_year_start(water_year(end))
                if length is None
                else end - timedelta(days=length - 1)
            )
            status = coverage.window(first, end, key if back == 0 else None)
            years.append(
                RollingYear(
                    year=end.year,
                    water_year=water_year(end),
                    status=status,
                    rain_in=days.rain_total(status),
                )
            )
        rows.append(RollingWindow(key=key, years=tuple(years)))
    return tuple(rows)


def _same_dates(coverage: ArchiveCoverage) -> List[date]:
    """
    Today, then today's date in each earlier year the archive reaches.

    Feb 29 falls back to Feb 28 in a year without one.

    :param coverage: The rain coverage.
    :return: The end days, newest first.
    """
    today = coverage.today
    ends = [today]
    first_day = coverage.first_day
    if first_day is None:
        return ends
    day = 28 if (today.month, today.day) == (2, 29) else today.day
    year = today.year - 1
    while True:
        end = date(year, today.month, day)
        if end < first_day:
            return ends
        ends.append(end)
        year -= 1


def _add_months(day: date, months: int) -> date:
    """
    The first of a month counted from another.

    :param day: Any day of the month to count from.
    :param months: How many months later.
    :return: The first day of that month.
    """
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)
