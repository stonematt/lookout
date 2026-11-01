"""
temperature.py: The Overview's temperature context (#81).

Three views, each fact computed once here (the One Number Rule):

- **Periods**: the lowest and highest reading, and the middle half of
  readings, over today, yesterday, 7, 30, 90 and 365 days. Each window's
  values come from its counted days and only when its gap status allows.
- **Ten days**: each of the last ten local days' low and high, against the
  usual low and high. The usual (design fork, see `_usual_days`) is the
  median daily low and median daily high over the days within `SPREAD_DAYS`
  either side of that date in earlier years.
- **Strips**: how unusual this moment is for the date. A temperature dot is
  a past day's mean at this local hour; a rain dot is that day's rain by
  this time of day, so "so far" compares like with like. Past days are the
  same `SPREAD_DAYS` window around today's date in earlier years.

An earlier year is read only when its 15-day window is comparable
(`lookout.snapshot.gaps.comparable`), and only over that window's counted
days, so a gap never reads as a calm day. Gap statuses are the build's
counted days, the archive as stored; values read the live-stitched history,
so they agree with the first screen. The old history min/max and polar preparers in
`lookout.core.data_processing` read the wall clock and ignore gaps, so they
are not composed here.

Functions:
- temperature_context(stitched, current, observed_at_ms, coverage): The
  `TemperatureContext` section.
"""

from datetime import date, timedelta
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lookout.snapshot.gaps import ArchiveCoverage, comparable, standard_window_keys
from lookout.snapshot.model import (
    NormalStrip,
    SnapshotConditions,
    TemperatureContext,
    TemperatureDay,
    TemperaturePeriod,
)
from lookout.snapshot.values import number
from lookout.utils.dateutc import local_time, to_local

SPREAD_DAYS = 7
"""Past days within this many days either side of a date are its normal."""

TEN_DAYS = 10
"""The list covers this many local days, today included."""


def _same_date(year: int, day: date) -> date:
    """`day` moved to `year`; Feb 29 becomes Feb 28 in a common year."""
    try:
        return day.replace(year=year)
    except ValueError:
        return day.replace(year=year, day=28)


class _Readings:
    """
    The stitched history by station-local day, hour and time of day.

    :param stitched: `stitch_live` output.
    :param tz: The station-local zone.
    """

    def __init__(self, stitched: pd.DataFrame, tz: str) -> None:
        frame = pd.DataFrame({"dateutc": stitched["dateutc"].to_numpy()})
        for column in ("tempf", "dailyrainin"):
            frame[column] = (
                pd.to_numeric(stitched[column], errors="coerce").to_numpy()
                if column in stitched
                else np.nan
            )
        local = to_local(frame["dateutc"], tz)
        frame["day"] = local.dt.date
        frame["hour"] = local.dt.hour
        frame["minute"] = local.dt.hour * 60 + local.dt.minute
        self.frame = frame
        self._days = frame.groupby("day")

    def temperatures(self, days: Sequence[date]) -> np.ndarray:
        """Every temperature reading on `days`."""
        wanted = set(days)
        rows = self.frame[self.frame["day"].isin(wanted)]
        return rows["tempf"].dropna().to_numpy(dtype=float)

    def daily_extremes(self, days: Sequence[date]) -> Tuple[List[float], List[float]]:
        """The daily lows and highs of those of `days` that have readings."""
        lows: List[float] = []
        highs: List[float] = []
        for day in days:
            if day not in self._days.groups:
                continue
            values = self._days.get_group(day)["tempf"].dropna()
            if not values.empty:
                lows.append(float(values.min()))
                highs.append(float(values.max()))
        return lows, highs

    def hour_mean(self, day: date, hour: int) -> Optional[float]:
        """The mean temperature on `day` during local `hour`, or None."""
        if day not in self._days.groups:
            return None
        rows = self._days.get_group(day)
        values = rows.loc[rows["hour"] == hour, "tempf"].dropna()
        return None if values.empty else float(values.mean())

    def rain_by(self, day: date, minute: int) -> Optional[float]:
        """The most rain `day` had counted up to local `minute`, or None."""
        if day not in self._days.groups:
            return None
        rows = self._days.get_group(day)
        values = rows.loc[rows["minute"] <= minute, "dailyrainin"].dropna()
        return None if values.empty else float(values.max())


FINDING_ORDER = ("30d", "7d", "90d", "365d", "yesterday")
"""The windows the headline tries, longest useful first."""


def _position(
    now: Optional[float], q1: Optional[float], q3: Optional[float]
) -> Optional[str]:
    """Where `now` sits against the middle half, or None when unknown."""
    if now is None or q1 is None or q3 is None:
        return None
    return "above" if now > q3 else "below" if now < q1 else "inside"


def _finding(periods: Sequence[TemperaturePeriod]) -> Optional[str]:
    """The key of the window the headline uses, or None."""
    by_key = {p.status.key: p for p in periods}
    for key in FINDING_ORDER:
        if by_key[key].position is not None:
            return key
    return None


def _periods(
    coverage: ArchiveCoverage, readings: _Readings, now: Optional[float]
) -> Tuple[TemperaturePeriod, ...]:
    """The range bar of each standard window, with where `now` sits."""
    windows = {w.key: w for w in coverage.standard_windows()}
    periods: List[TemperaturePeriod] = []
    for key in standard_window_keys():
        status = windows[key]
        counted = coverage.counted_days(
            date.fromisoformat(status.first_day), date.fromisoformat(status.last_day)
        )
        values = comparable(status, lambda c=counted: readings.temperatures(c))
        if values is None or len(values) == 0:
            periods.append(TemperaturePeriod(status, None, None, None, None, 0, None))
            continue
        q1, q3 = np.quantile(values, [0.25, 0.75])
        periods.append(
            TemperaturePeriod(
                status=status,
                low=number(values.min(), 1),
                q1=number(q1, 1),
                q3=number(q3, 1),
                high=number(values.max(), 1),
                readings=int(len(values)),
                position=_position(now, number(q1, 1), number(q3, 1)),
            )
        )
    return tuple(periods)


def _normal_years(
    coverage: ArchiveCoverage, around: date
) -> List[Tuple[int, Tuple[date, ...]]]:
    """
    The earlier years whose window around `around` is comparable.

    :param coverage: The archive's coverage.
    :param around: The date whose normal is wanted.
    :return: Each usable year, newest first, with its window's counted days.
    """
    first = coverage.first_day
    if first is None:
        return []
    usable: List[Tuple[int, Tuple[date, ...]]] = []
    for year in range(around.year - 1, first.year - 1, -1):
        centre = _same_date(year, around)
        first_day = centre - timedelta(days=SPREAD_DAYS)
        last_day = centre + timedelta(days=SPREAD_DAYS)
        status = coverage.window(first_day, last_day, f"{year}")
        days = comparable(status, lambda: coverage.counted_days(first_day, last_day))
        if days:
            usable.append((year, days))
    return usable


def _median(values: Sequence[float]) -> Optional[float]:
    """The median, or None when empty."""
    return number(float(np.median(values)), 1) if len(values) else None


def _ten_days(
    coverage: ArchiveCoverage, readings: _Readings, today: date
) -> Tuple[Tuple[TemperatureDay, ...], Tuple[int, ...]]:
    """
    The last ten days against their usual.

    Design fork: the usual is the median daily low and high rather than a
    quartile band, so the list's usual bar reads as "a typical day's low to
    high" and is the same grammar as the day's own bar.
    """
    rows: List[TemperatureDay] = []
    all_years: set = set()
    for back in range(TEN_DAYS - 1, -1, -1):
        day = today - timedelta(days=back)
        status = coverage.window(day, day, day.isoformat())
        extremes = comparable(status, lambda d=day: readings.daily_extremes([d]))
        low = high = None
        if extremes is not None and extremes[0]:
            low, high = number(extremes[0][0], 1), number(extremes[1][0], 1)
        lows: List[float] = []
        highs: List[float] = []
        years: List[int] = []
        for year, days in _normal_years(coverage, day):
            year_lows, year_highs = readings.daily_extremes(days)
            if year_lows:
                lows += year_lows
                highs += year_highs
                years.append(year)
        all_years.update(years)
        rows.append(
            TemperatureDay(
                day=day.isoformat(),
                status=status,
                low=low,
                high=high,
                usual_low=_median(lows),
                usual_high=_median(highs),
                usual_years=tuple(years),
                usual_days=len(lows),
            )
        )
    return tuple(rows), tuple(sorted(all_years, reverse=True))


def _share(values: Sequence[float], test: Callable[[float], bool]) -> int:
    """The whole percent of `values` for which `test` holds."""
    return int(round(100 * sum(1 for v in values if test(v)) / len(values)))


def _strip(
    kind: str,
    unit: str,
    now: Optional[float],
    now_ms: Optional[int],
    past: Callable[[date], Optional[float]],
    coverage: ArchiveCoverage,
    today: Optional[date],
) -> NormalStrip:
    """One strip: `past(day)` is the day's value to set `now` among."""
    values: List[float] = []
    years: List[int] = []
    if today is not None and now is not None:
        for year, days in _normal_years(coverage, today):
            found = [v for v in (past(d) for d in days) if v is not None]
            if found:
                values += found
                years.append(year)
    missing = "no_reading" if now is None else None
    if missing is None and not values:
        missing = "no_comparison"
    values.sort()
    ready = missing is None
    return NormalStrip(
        kind=kind,  # type: ignore[arg-type]
        unit=unit,  # type: ignore[arg-type]
        now=now,
        now_ms=now_ms,
        values=tuple(round(v, 3) for v in values) if ready else (),
        below_pct=_share(values, lambda v: v < now) if ready else None,
        equal_pct=_share(values, lambda v: v == now) if ready else None,
        above_pct=_share(values, lambda v: v > now) if ready else None,
        years=tuple(years) if ready else (),
        missing=missing,  # type: ignore[arg-type]
    )


def temperature_context(
    stitched: pd.DataFrame,
    current: SnapshotConditions,
    observed_at_ms: Optional[int],
    coverage: ArchiveCoverage,
) -> TemperatureContext:
    """
    The Overview's temperature across periods, ten-day list and strips.

    :param stitched: `stitch_live` output, newest first.
    :param current: Current conditions, live stitched.
    :param observed_at_ms: The newest observation, or None.
    :param coverage: The build's counted days (`CountedDays.coverage`), for
        gap statuses; its clock says which day is today.
    :return: The `TemperatureContext` section.
    """
    readings = _Readings(stitched, coverage.tz) if not stitched.empty else None
    if readings is None:
        by_key = {w.key: w for w in coverage.standard_windows()}
        empty = tuple(
            TemperaturePeriod(by_key[key], None, None, None, None, 0, None)
            for key in standard_window_keys()
        )
        return TemperatureContext(None, None, None, empty, (), (), ())

    periods = _periods(coverage, readings, current.temperature_f)
    ten_days, ten_day_years = _ten_days(coverage, readings, coverage.today)

    # The strips are set at the newest observation's date and time, so a
    # stale reading is compared with its own day.
    observed = (
        None if observed_at_ms is None else local_time(observed_at_ms, coverage.tz)
    )
    observed_day = None if observed is None else observed.date()
    hour = 0 if observed is None else observed.hour
    minute = 0 if observed is None else observed.hour * 60 + observed.minute
    strips = (
        _strip(
            "temperature",
            "f",
            current.temperature_f,
            observed_at_ms,
            lambda d: readings.hour_mean(d, hour),
            coverage,
            observed_day,
        ),
        _strip(
            "rain",
            "in",
            current.rain_today_in,
            observed_at_ms,
            lambda d: readings.rain_by(d, minute),
            coverage,
            observed_day,
        ),
    )
    return TemperatureContext(
        now=current.temperature_f,
        now_ms=observed_at_ms,
        finding=_finding(periods),
        periods=periods,
        ten_days=ten_days,
        ten_day_years=ten_day_years,
        strips=strips,
    )
