"""
days.py: the build's counted days and daily rain series, made once (#196).

`build_snapshot` counts the archive's days once and extracts the daily rain
series once, then hands both to every section, so no two sections can judge
the same day two ways:

- The counted days (`ArchiveCoverage`): each station-local day's verdict
  from the archive's readings (GLOSSARY.md, "counted day"). Temperature,
  solar and the gaps section judge their windows by it.
- The daily rain series: each day's highest `dailyrainin`, NaN for a day
  with rows but no counter reading; today's is the live reading's daily
  counter when that reading is today's (`SnapshotConditions.rain_today_in`).
- The rain day kind: a view over the two in which a counted day with no
  counter value is a gap. The Rain tab, the almanac, rain events and the
  calm lead judge rain windows by it, so one outage reads the same in each,
  and a dead rain gauge never hatches a temperature or solar window.

A day the archive cannot count, a counted day with no counter value and an
archive with no rain counter column are all unknown for rain, never dry.

Classes and functions:
- CountedDays: the counted days, the daily rain series and the rain view.
- count_days(archive, current, clock): the one `CountedDays` of a build.
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from lookout.core.rainfall_analysis import extract_daily_rainfall
from lookout.snapshot.clock import StationClock
from lookout.snapshot.gaps import ArchiveCoverage, comparable, prorate
from lookout.snapshot.model import GapStatus, SnapshotConditions
from lookout.snapshot.values import iso_day, number
from lookout.utils.dateutc import day_span

RAIN_COUNTER = "dailyrainin"
"""The archive column holding the daily rain counter."""


@dataclass(frozen=True)
class CountedDays:
    """
    The build's counted days, its daily rain series and their rain view.

    :param coverage: The counted days: each day's plain kind.
    :param rain_series: The daily rain series: day -> inches, NaN for a day
        whose counter has no reading; today live.
    :param rain: The rain day kind: `coverage` with every counted day that
        has no counter value judged a gap.
    :param rain_in: Each rain-counted day's total: the series over the days
        `rain` counts.
    """

    coverage: ArchiveCoverage
    rain_series: Mapping[date, float]
    rain: ArchiveCoverage
    rain_in: Mapping[date, float]

    @property
    def clock(self) -> StationClock:
        """The station clock the days were counted on."""
        return self.coverage.clock

    def rain_total(self, status: GapStatus) -> Optional[float]:
        """
        One rain window's total: the sum over its counted days, prorated.

        :param status: The window's gap status, judged by `rain`.
        :return: Inches; None when the window is invalid.
        """

        def total() -> Optional[float]:
            """The sum over the counted days, prorated."""
            days = self.rain.counted_days(
                iso_day(status.first_day), iso_day(status.last_day)
            )
            return number(prorate(status, sum(self.rain_in.get(d, 0.0) for d in days)))

        return comparable(status, total)

    def rain_days(self, first_day: date, last_day: date) -> pd.Series:
        """
        Each day's rain from `first_day` to `last_day`, for dry spells.

        :param first_day: The first day.
        :param last_day: The last day, inclusive.
        :return: Inches by day, oldest first; NaN for a day not rain-counted,
            so a gap breaks a dry spell rather than lengthening it.
        """
        index = day_span(first_day, last_day)
        return pd.Series(
            [self.rain_in.get(d, np.nan) for d in index], index=index, dtype=float
        )


def count_days(
    archive: pd.DataFrame, current: SnapshotConditions, clock: StationClock
) -> CountedDays:
    """
    Count the archive's days and extract its daily rain series, once.

    :param archive: The station's archive as stored.
    :param current: Current conditions, live stitched: today's rain.
    :param clock: The station clock.
    :return: The counted days, shared by every section.
    """
    coverage = ArchiveCoverage(archive, clock)
    series = _daily_rain(archive, current, clock)
    unread = {
        d: "gap" for d in coverage.held_days() if not np.isfinite(series.get(d, np.nan))
    }
    rain = coverage.excluding(unread)
    rain_in = {d: v for d, v in series.items() if rain.day_kind(d) == "ok"}
    return CountedDays(
        coverage=coverage, rain_series=series, rain=rain, rain_in=rain_in
    )


def _daily_rain(
    archive: pd.DataFrame, current: SnapshotConditions, clock: StationClock
) -> Dict[date, float]:
    """
    The daily rain series: each day's highest counter; today's live.

    :param archive: The station's archive as stored.
    :param current: Current conditions: today's live rain, or None.
    :param clock: The station clock.
    :return: Day -> inches; NaN for a day with rows but no counter reading.
        Empty for an archive with no rain counter column.
    """
    series: Dict[date, float] = {}
    if not archive.empty and RAIN_COUNTER in archive:
        frame = extract_daily_rainfall(archive, clock.tz)
        series = {d: float(r) for d, r in zip(frame["date"], frame["rainfall"])}
    if current.rain_today_in is not None:
        series[clock.today] = float(current.rain_today_in)
    return series
