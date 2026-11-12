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
  The list also carries its highs against usual (#540): the mean high
  minus the mean usual high over the days that have both, in whole degrees.
- **Strips**: how unusual this moment is for the date. A temperature dot is
  a past day's mean at this local hour; a rain dot is that day's rain by
  this time of day, so "so far" compares like with like. Past days are the
  same `SPREAD_DAYS` window around today's date in earlier years.
- **Full history** (#463): the same range bar over the archive's first
  day through today, under the same gap rules, beside the periods.
- **Usual band** (#462): for each station-local hour of today, the median
  and middle half of that hour's readings over the usual window
  (`usual_window`): the same earlier-year windows when one is comparable,
  else the station's own recent days, the median only while the station
  has under `MIN_READING_DAYS` counted days. The band also carries now
  against the usual for this hour (#518): the current temperature minus
  the band's median at the reading's station-local hour, in whole degrees.

An earlier year is read only when its 15-day window is comparable
(`lookout.snapshot.gaps.comparable`), and only over that window's counted
days, so a gap never reads as a calm day. Gap statuses are the build's
counted days, the archive as stored; values read the live-stitched history,
so they agree with the first screen. The old history min/max and polar
preparers in `lookout.core.data_processing` read the wall clock and ignore
gaps, so they are not composed here.

Functions:
- temperature_context(stitched, current, observed_at_ms, coverage): The
  `TemperatureContext` section.
- usual_band(stitched, coverage, now_f, observed_at_ms): Today's
  `UsualBand`, all 24 hours, with now against the usual for this hour.
- same_date(year, day): A date moved to another year, Feb 29 to Feb 28.
- usual_window(coverage, day): The `UsualWindow`, the days the usual for
  `day` is over, and its basis.
- usual_readings(stitched, tz, days, column): Each station-local hour's
  readings of any column on those days, for a usual band or a percentile
  at an hour.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lookout.snapshot.gaps import ArchiveCoverage, comparable, standard_window_keys
from lookout.snapshot.model import (
    GapStatus,
    NormalStrip,
    SnapshotConditions,
    TemperatureContext,
    TemperatureDay,
    TemperaturePeriod,
    TenDayHighs,
    UsualBand,
    UsualBasis,
    UsualHour,
    UsualNow,
)
from lookout.snapshot.values import number
from lookout.utils.dateutc import from_local_date, is_fresh, local_time, to_local

SPREAD_DAYS = 7
"""Past days within this many days either side of a date are its normal."""

TEN_DAYS = 10
"""The list covers this many local days, today included."""

RECENT_DAYS = 14
"""Without an earlier year, the usual is over at most this many recent days."""

MIN_READING_DAYS = 20
"""Under this many counted days, a recent-days usual is the median only."""

HOURS = tuple(range(24))
"""The station-local wall-clock hours of a day."""

FULL_HISTORY = "full_history"
"""The Full history window's key: the archive's first day through today."""


def same_date(year: int, day: date) -> date:
    """
    `day` moved to `year`; Feb 29 becomes Feb 28 in a common year.

    :param year: The year.
    :param day: The date.
    :return: The same month and day in `year`.
    """
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


def _period(
    status: GapStatus,
    coverage: ArchiveCoverage,
    readings: Optional[_Readings],
    now: Optional[float],
) -> TemperaturePeriod:
    """
    One window's range bar, with where `now` sits.

    :param status: The window's gap status; values only when it allows.
    :param coverage: The build's counted days.
    :param readings: The stitched readings, or None when there are none.
    :param now: The current temperature.
    :return: The period; null values when invalid or without readings.
    """
    if readings is None:
        return TemperaturePeriod(status, None, None, None, None, 0, None)
    counted = coverage.counted_days(
        date.fromisoformat(status.first_day), date.fromisoformat(status.last_day)
    )
    values = comparable(status, lambda: readings.temperatures(counted))
    if values is None or len(values) == 0:
        return TemperaturePeriod(status, None, None, None, None, 0, None)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return TemperaturePeriod(
        status=status,
        low=number(values.min(), 1),
        q1=number(q1, 1),
        q3=number(q3, 1),
        high=number(values.max(), 1),
        readings=int(len(values)),
        position=_position(now, number(q1, 1), number(q3, 1)),
    )


def _periods(
    coverage: ArchiveCoverage, readings: Optional[_Readings], now: Optional[float]
) -> Tuple[TemperaturePeriod, ...]:
    """
    The range bar of each standard window, with where `now` sits.

    :param coverage: The build's counted days.
    :param readings: The stitched readings, or None when there are none.
    :param now: The current temperature.
    :return: One period per standard window, in order.
    """
    windows = {w.key: w for w in coverage.standard_windows()}
    return tuple(
        _period(windows[key], coverage, readings, now) for key in standard_window_keys()
    )


def _full_history(
    coverage: ArchiveCoverage, readings: Optional[_Readings], now: Optional[float]
) -> Optional[TemperaturePeriod]:
    """
    The Full history window: the archive's first day through today (#463).

    It follows the One Gap Status Rule like every window, so a record with
    more than a tenth of its days missing has no range here.

    :param coverage: The build's counted days.
    :param readings: The stitched readings, or None when there are none.
    :param now: The current temperature.
    :return: Its period; None for an empty archive.
    """
    first = coverage.first_day
    if first is None:
        return None
    status = coverage.window(first, coverage.today, FULL_HISTORY)
    return _period(status, coverage, readings, now)


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
        centre = same_date(year, around)
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


def _ten_day_highs(rows: Sequence[TemperatureDay]) -> Optional[TenDayHighs]:
    """
    The ten-day list's highs against their usual highs (#540).

    Only days with both a high and a usual high count, so a gap or a day
    without earlier years never pulls either mean. The mean of the
    differences is the difference of the means over the same days; it is
    taken in decimal, as `_whole_degrees` does, so a one-decimal half rounds
    half away from zero and never off by float error. `int` of a decimal
    zero is 0, so a small negative mean never reads -0.

    :param rows: The ten-day list.
    :return: The whole-degree delta and its day count, or None when no day
        has both.
    """
    differences = [
        Decimal(str(row.high)) - Decimal(str(row.usual_high))
        for row in rows
        if row.high is not None and row.usual_high is not None
    ]
    if not differences:
        return None
    mean = sum(differences) / len(differences)
    return TenDayHighs(
        delta_f=int(mean.quantize(Decimal("1"), rounding=ROUND_HALF_UP)),
        days=len(differences),
    )


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


@dataclass(frozen=True)
class UsualWindow:
    """
    The days a date's usual is over, and the basis that picked them.

    :param basis: "earlier_years", "recent_days" or "median_only" (see
        `lookout.snapshot.model.UsualBand`).
    :param years: The earlier years used, newest first; empty otherwise.
    :param days: The counted days, oldest first.
    """

    basis: UsualBasis
    years: Tuple[int, ...]
    days: Tuple[date, ...]


def usual_window(coverage: ArchiveCoverage, day: date) -> UsualWindow:
    """
    The days the usual for `day` is over (the glossary's usual band).

    An earlier year covers `day` when its `SPREAD_DAYS` window around the
    same date is comparable (`_normal_years`, the ten-day list's and the
    strips' rule); then every such year's counted days are used and recent
    days are not. Else the counted days among the `RECENT_DAYS` before
    `day`, as "recent_days" when the archive holds `MIN_READING_DAYS` or
    more counted days before `day`, "median_only" when it holds fewer.
    `day` itself is never in its own usual. Earlier years win even when the
    station holds under `MIN_READING_DAYS` counted days: a comparable
    15-day window is a fuller basis than the recent days.

    :param coverage: The build's counted days.
    :param day: The station-local date the usual is for.
    :return: The window.
    """
    years = _normal_years(coverage, day)
    if years:
        return UsualWindow(
            basis="earlier_years",
            years=tuple(year for year, _ in years),
            days=tuple(sorted(d for _, days in years for d in days)),
        )
    yesterday = day - timedelta(days=1)
    first = coverage.first_day
    reading_days = (
        coverage.counted_days(first, yesterday)
        if first is not None and first <= yesterday
        else ()
    )
    recent = coverage.counted_days(day - timedelta(days=RECENT_DAYS), yesterday)
    basis: UsualBasis = (
        "recent_days" if len(reading_days) >= MIN_READING_DAYS else "median_only"
    )
    return UsualWindow(basis=basis, years=(), days=recent)


def usual_readings(
    stitched: pd.DataFrame, tz: str, days: Sequence[date], column: str
) -> Dict[int, np.ndarray]:
    """
    Each station-local hour's readings of `column` on `days`.

    Hours are wall-clock: on a 25-hour day both 1 am hours are hour 1, and
    a 23-hour day has no hour 2.

    :param stitched: `stitch_live` output (any row order).
    :param tz: The station-local zone.
    :param days: The station-local days, typically a `UsualWindow`'s.
    :param column: The archive column, e.g. "tempf".
    :return: Hour 0 to 23 -> its non-null readings; empty where none.
    """
    by_hour: Dict[int, np.ndarray] = {hour: np.empty(0) for hour in HOURS}
    if stitched.empty or not days or column not in stitched:
        return by_hour
    stamps = stitched["dateutc"]
    start = from_local_date(min(days), tz)
    end = from_local_date(max(days) + timedelta(days=1), tz)
    rows = stitched[(stamps >= start) & (stamps < end)]
    local = to_local(rows["dateutc"], tz)
    frame = pd.DataFrame(
        {
            "day": local.dt.date.to_numpy(),
            "hour": local.dt.hour.to_numpy(),
            "value": pd.to_numeric(rows[column], errors="coerce").to_numpy(),
        }
    )
    frame = frame[frame["day"].isin(set(days))].dropna(subset=["value"])
    for hour, group in frame.groupby("hour"):
        by_hour[int(hour)] = group["value"].to_numpy(dtype=float)
    return by_hour


def _usual_hour(hour: int, values: np.ndarray, basis: UsualBasis) -> UsualHour:
    """
    One hour of the band.

    :param hour: The station-local hour.
    :param values: Its readings over the band's days.
    :param basis: The band's basis; "median_only" leaves the middle half null.
    :return: The hour's median, middle half and count.
    """
    if len(values) == 0:
        return UsualHour(hour=hour, median=None, q1=None, q3=None, readings=0)
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    middle_half = basis != "median_only"
    return UsualHour(
        hour=hour,
        median=number(median, 1),
        q1=number(q1, 1) if middle_half else None,
        q3=number(q3, 1) if middle_half else None,
        readings=int(len(values)),
    )


def _whole_degrees(now_f: float, median: float) -> int:
    """
    `now_f - median` in whole degrees, half away from zero (+2.5 is +3).

    Both are one-decimal readings, so the difference is taken in decimal:
    72.0 - 60.1 is 11.9, not 11.900000000000006.

    :param now_f: The current temperature.
    :param median: The usual median for its hour.
    :return: The whole-degree difference.
    """
    difference = Decimal(str(now_f)) - Decimal(str(median))
    return int(difference.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _usual_now(
    hours: Sequence[UsualHour],
    basis: UsualBasis,
    now_f: Optional[float],
    observed_at_ms: Optional[int],
    coverage: ArchiveCoverage,
) -> Optional[UsualNow]:
    """
    Now against the usual for this hour (#518).

    The hour is the reading's station-local hour (ADR-0002), read against
    the band the snapshot carries, so the line and the drawn band agree.
    A reading is current by the Feature Rule's test, `is_fresh` against the
    build's `now` (the clock's), never the wall clock.

    :param hours: The band's 24 hours.
    :param basis: The band's basis.
    :param now_f: The current temperature, or None.
    :param observed_at_ms: The newest observation, or None.
    :param coverage: The build's counted days; its clock is the build's.
    :return: The hour, delta and basis; None with no current temperature,
        a stale reading, or no median for that hour.
    """
    if now_f is None or observed_at_ms is None:
        return None
    if not is_fresh(observed_at_ms, coverage.now_ms):
        return None
    hour = local_time(observed_at_ms, coverage.tz).hour
    median = hours[hour].median
    if median is None:
        return None
    return UsualNow(hour=hour, delta_f=_whole_degrees(now_f, median), basis=basis)


def usual_band(
    stitched: pd.DataFrame,
    coverage: ArchiveCoverage,
    now_f: Optional[float],
    observed_at_ms: Optional[int],
) -> UsualBand:
    """
    Today's usual temperature band by station-local hour.

    :param stitched: `stitch_live` output.
    :param coverage: The build's counted days; its clock says which day is
        today and what time it is.
    :param now_f: The current temperature, live stitched, or None.
    :param observed_at_ms: The newest observation, or None.
    :return: The `UsualBand`, all 24 hours, with now against the usual.
    """
    window = usual_window(coverage, coverage.today)
    by_hour = usual_readings(stitched, coverage.tz, window.days, "tempf")
    hours = tuple(_usual_hour(h, by_hour[h], window.basis) for h in HOURS)
    return UsualBand(
        basis=window.basis,
        unit="f",
        years=window.years,
        days=len(window.days),
        hours=hours,
        now_delta=_usual_now(hours, window.basis, now_f, observed_at_ms, coverage),
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
        return TemperatureContext(
            now=None,
            now_ms=None,
            finding=None,
            periods=_periods(coverage, None, None),
            ten_days=(),
            ten_day_years=(),
            ten_day_highs=None,
            strips=(),
            usual_band=usual_band(
                stitched, coverage, current.temperature_f, observed_at_ms
            ),
            full_history=_full_history(coverage, None, None),
        )

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
        ten_day_highs=_ten_day_highs(ten_days),
        strips=strips,
        usual_band=usual_band(
            stitched, coverage, current.temperature_f, observed_at_ms
        ),
        full_history=_full_history(coverage, readings, current.temperature_f),
    )
