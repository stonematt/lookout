"""
almanac.py: The Overview's almanac grid and rolling-window rain table (#82).

- The grid: one row per water year (Oct through Sep, GLOSSARY.md), newest
  first, each month's rain under its own gap status, and the season's total.
- The rolling table: rain over the last 7, 30, 90 and 365 days and the
  season so far, each against the same window ending on the same date in
  every earlier year the archive reaches, newest first.

Every total comes from the Rain tab's `RainCoverage`, `daily_totals` and
`window_total`, so a window has one rain total and one status everywhere
(the One Number Rule): an archive sum of each day's highest `dailyrainin`,
with today's live total stitched in, judged and prorated by the gap rules.
Ambient's year, month and week counters are never read.

Functions:
- almanac(archive, current, now): the `Almanac` section.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from lookout.core.rainfall_analysis import water_year, water_year_start
from lookout.snapshot.gaps import STANDARD_WINDOWS
from lookout.snapshot.model import (
    Almanac,
    AlmanacMonth,
    AlmanacYear,
    RollingWindow,
    RollingYear,
    SnapshotConditions,
)
from lookout.snapshot.rain_tab import RainCoverage, daily_totals, window_total
from lookout.utils.dateutc import DEFAULT_TZ

ROLLING_DAYS: Tuple[Tuple[str, int], ...] = tuple(
    (key, first_back + 1)
    for key, first_back, last_back in STANDARD_WINDOWS
    if last_back == 0 and first_back > 0
)
"""The standard windows that end today and span more than a day (7, 30,
90 and 365 days), as `Gaps.windows` keys them; the season ("water_year")
follows them."""

SEASON = "water_year"


def almanac(
    archive: pd.DataFrame,
    current: SnapshotConditions,
    now: datetime,
    tz: str = DEFAULT_TZ,
) -> Almanac:
    """
    The almanac grid and the rolling-window rain table.

    :param archive: The station's archive as stored, for the gap rules and
        the daily totals.
    :param current: Current conditions, live stitched: today's rain.
    :param now: The build time, timezone-aware.
    :param tz: The station-local zone.
    :return: The section.
    """
    coverage = RainCoverage(archive, current, now, tz)
    totals = daily_totals(coverage)
    return Almanac(
        years=_years(totals, coverage),
        rolling=_rolling(totals, coverage),
    )


def _years(
    totals: Dict[date, float], coverage: RainCoverage
) -> Tuple[AlmanacYear, ...]:
    """
    Every water year from the archive's first through the current one.

    :param totals: `daily_totals`.
    :param coverage: The archive's rain coverage.
    :return: The rows, newest first; none for an empty archive.
    """
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
            months.append(None if first > today else _month(first, totals, coverage))
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
                season_in=window_total(season, totals, coverage),
            )
        )
    return tuple(rows)


def _month(
    first: date, totals: Dict[date, float], coverage: RainCoverage
) -> AlmanacMonth:
    """
    One month's cell; the current month also carries its rain so far.

    :param first: The month's first day.
    :param totals: `daily_totals`.
    :param coverage: The archive's rain coverage.
    :return: The cell.
    """
    status = coverage.month(first.year, first.month)
    to_date = None
    if _add_months(first, 1) > coverage.today:
        to_date = coverage.window(first, coverage.today)
    return AlmanacMonth(
        status=status,
        rain_in=window_total(status, totals, coverage),
        to_date=to_date,
        to_date_in=(
            None if to_date is None else window_total(to_date, totals, coverage)
        ),
    )


def _rolling(
    totals: Dict[date, float], coverage: RainCoverage
) -> Tuple[RollingWindow, ...]:
    """
    Each rolling window, this year's then each earlier year's.

    This year's windows are the snapshot's standard windows, keys and all,
    so they match `Gaps.windows` and the Rain tab's totals.

    :param totals: `daily_totals`.
    :param coverage: The archive's rain coverage.
    :return: The windows, shortest first, the season last.
    """
    ends = _same_dates(coverage)
    rows = []
    for key, days in (*ROLLING_DAYS, (SEASON, None)):
        years = []
        for back, end in enumerate(ends):
            first = (
                water_year_start(water_year(end))
                if days is None
                else end - timedelta(days=days - 1)
            )
            status = coverage.window(first, end, key if back == 0 else None)
            years.append(
                RollingYear(
                    year=end.year,
                    status=status,
                    rain_in=window_total(status, totals, coverage),
                )
            )
        rows.append(RollingWindow(key=key, years=tuple(years)))
    return tuple(rows)


def _same_dates(coverage: RainCoverage) -> List[date]:
    """
    Today, then today's date in each earlier year the archive reaches.

    Feb 29 falls back to Feb 28 in a year without one.

    :param coverage: The archive's rain coverage.
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
