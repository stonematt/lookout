"""
rain_tab.py: The Rain tab's section of the snapshot (#83).

The Streamlit Rain tab's data prep, moved behind `build_snapshot` and set to
the new site's rules:

- Every rain total is an archive sum with today's live total stitched in
  (spec #66, decided 2026-10-30): a day's total is what its readings add
  to `dailyrainin`, a counter drop being the reset (#654), and today's is
  the live reading's (`SnapshotConditions.rain_today_in`), so the Rain tab,
  the Overview and `/current` show one number for today.
- Every value over a window goes through the gap rules
  (`lookout.snapshot.gaps`): a total is summed over the window's counted days
  and prorated; an invalid window's value is None and never computed.
- Rain windows are judged by the build's rain day kind
  (`lookout.snapshot.days.CountedDays.rain`): the counted days, with a day
  whose rain counter has no reading counted as a gap. The archive holds
  that day's rows, so `Gaps.windows` (every quantity) still counts it, but
  its rain is unknown, as it is for dry spells; each `RainWindow` carries
  its own status.
- Year-over-year accumulation runs over the water year from Oct 1
  (GLOSSARY.md), where Streamlit used the calendar year.
- This year's rain to date is set against the usual (#539): the median of
  the earlier water years' rain by this date, each read at the same place
  the year-over-year finding compares (`WaterYearCurve.to_date_in`), so a
  year counts exactly when the finding can name it.
- Wet hours since Oct 1 against usual wet hours to date come from the one
  wet-hour helper (`lookout.snapshot.wet_hours.wet_hours_to_date`, #663),
  so they agree with the Overview's meter; the prorated
  `RecentRain.wet_hours` stays in the snapshot until R2 (#667).
- An earlier water year counts toward the usual rain to date only if it
  also covers this date, the one rule usual wet hours apply
  (`WetHoursToDate.usual.earlier`): no usual and no verdict before then.
  The verdict (#640) reads the published percent: "about the same" within
  `ABOUT_THE_SAME` of 100, else "wetter" or "drier".
- The window strip (#640) sets each window's rain against the same window
  a whole number of years back, each earlier window judged by the gap
  rules, as a range: from 3 years the band usual wet hours use
  (`lookout.snapshot.wet_hours.usual_band`), below that the plain values.
  A window outside it is marked "above usual range" or "below usual
  range"; one still running is never below.

Functions:
- rain_tab(history, days): the `RainTab` section.

Integration: any other section that totals rain over a window (the
Overview's rolling-window tables) reads the same `CountedDays` and calls
`CountedDays.rain_total`, so each window has one rain total and one status
everywhere (the One Number Rule).
"""

from datetime import date, timedelta
from statistics import median
from typing import FrozenSet, List, Optional, Sequence, Tuple

import pandas as pd

from lookout.core.rainfall_analysis import (
    WET_DAY_IN,
    dry_spells,
    hourly_rain,
    water_year,
    water_year_day_index,
    water_year_start,
)
from lookout.snapshot.days import CountedDays
from lookout.snapshot.gaps import PRORATED, ArchiveCoverage, comparable, prorate
from lookout.snapshot.model import (
    DrySpell,
    DrySpells,
    EarlierWindowRain,
    GapReason,
    GapStatus,
    RainDay,
    RainHour,
    RainTab,
    RainWindow,
    RecentRain,
    StripMark,
    StripWindow,
    UsualToDate,
    Verdict,
    WaterYearCurve,
    WetHoursToDate,
)
from lookout.snapshot.values import iso_day, number
from lookout.snapshot.wet_hours import usual_band, wet_hours_to_date
from lookout.utils.dateutc import day_span, from_local_date

RECENT_WINDOW = "90d"
"""The standard window the daily bars and the heatmap cover."""

HOURS = 24
"""Local hours in a heatmap day."""

_DAY_MS = 24 * 60 * 60 * 1000

PLACES = 365
"""Places in a water-year curve: Oct 1 to Sep 30, Feb 29 folded in."""

ABOUT_THE_SAME = 15
"""The verdict is "about the same" while this water year's rain to date is
within this many percent of the usual rain to date (#640)."""


def rain_tab(history: pd.DataFrame, days: CountedDays) -> RainTab:
    """
    The Rain tab's section.

    :param history: The archive with the live reading stitched in
        (`stitch_live`), for today's hours.
    :param days: The build's counted days and daily rain series; rain
        windows are judged by its rain day kind.
    :return: The section.
    """
    windows = days.rain.standard_windows()
    recent_status = next(s for s in windows if s.key == RECENT_WINDOW)
    water_year_status = next(s for s in windows if s.key == "water_year")
    water_years = _water_years(days)
    wet_to_date = wet_hours_to_date(history, days.clock)
    to_date = _to_date_years(water_years, _covered(wet_to_date))
    rain_windows = tuple(
        RainWindow(key=s.key, status=s, total_in=days.rain_total(s)) for s in windows
    )
    return RainTab(
        windows=rain_windows,
        recent=_recent(recent_status, history, days),
        dry_spells=_dry_spells(water_year_status, days),
        day_index=water_year_day_index(days.clock.today),
        water_years=water_years,
        usual_to_date=_usual_to_date(
            water_years[-1].to_date_in if water_years else None, to_date
        ),
        wet_hours_to_date=wet_to_date,
        window_strip=_window_strip(rain_windows, to_date, days),
    )


def _recent(status: GapStatus, history: pd.DataFrame, days: CountedDays) -> RecentRain:
    """
    Daily and hourly rain over the days of the recent window.

    :param status: The recent window's gap status.
    :param history: The stitched history, for the hours (today's included).
    :param days: The counted days; the clock's build hour says which hours
        today are still to come.
    :return: The recent days, with wet hours prorated like any window value.
    """
    coverage, totals = days.rain, days.rain_in
    first, last = iso_day(status.first_day), iso_day(status.last_day)
    span_days = day_span(first, last)
    span = history
    if not history.empty:
        # A day of readings before the window: its first hour's rain is the
        # rise from them; the first reading of a span adds nothing (#654).
        lead_in = from_local_date(first, coverage.tz) - _DAY_MS
        span = history[history["dateutc"] >= lead_in]
    hours = hourly_rain(span, coverage.tz)
    by_hour = {(d, h): r for d, h, r in hours.itertuples(index=False, name=None)}
    # The one "still to come" decision: the hours after the build hour on
    # the window's last day, today. The browser reads it from the snapshot.
    first_to_come = coverage.clock.local_now.hour + 1

    rain_in: List[Optional[float]] = []
    reasons = []
    hourly: List[Tuple[Optional[float], ...]] = []
    for d in span_days:
        # A counted day with nothing archived yet (just after midnight) is
        # 0, as `rain_total` sums it. Its hours are not: an hour with no
        # counter reading is unknown (None), never dry (#147).
        reason = _day_reason(coverage, d)
        if reason is not None:
            rain_in.append(None)
            reasons.append(reason)
            hourly.append((None,) * HOURS)
            continue
        rain_in.append(number(totals.get(d, 0.0)))
        reasons.append(None)
        to_come = first_to_come if d == last else HOURS
        hourly.append(
            tuple(
                number(by_hour.get((d, h))) if h < to_come else None
                for h in range(HOURS)
            )
        )

    cells = [
        (d, h, r)
        for d, row in zip(span_days, hourly)
        for h, r in enumerate(row)
        if r is not None
    ]

    def wet_hours() -> Optional[float]:
        """
        Hours with rain among the known ones, scaled over the window's
        missing days; an unknown hour is neither wet nor dry.
        """
        return number(prorate(status, float(sum(1 for _, _, r in cells if r > 0))))

    def wettest() -> Optional[RainHour]:
        """The hour with the most rain; a maximum is not prorated."""
        wettest_cell = max(cells, key=lambda c: c[2], default=None)
        if wettest_cell is None or wettest_cell[2] <= 0:
            return None
        d, h, r = wettest_cell
        return RainHour(day=d.isoformat(), hour=h, rain_in=r)

    return RecentRain(
        window=status.key,
        first_day=status.first_day,
        rain_in=tuple(rain_in),
        reason=tuple(reasons),
        hourly_in=tuple(hourly),
        first_hour_to_come=first_to_come,
        wet_hours=comparable(status, wet_hours),
        wettest_hour=comparable(status, wettest),
    )


def _dry_spells(water_year_status: GapStatus, days: CountedDays) -> DrySpells:
    """
    The longest dry spell in the record, and this water year's rain days.

    :param water_year_status: The water year's gap status.
    :param days: The counted days and the daily rain series.
    :return: The dry-spell stats.
    """
    coverage, totals = days.rain, days.rain_in
    today = coverage.today

    spells = pd.DataFrame(columns=["first_day", "last_day", "days"])
    if totals:
        spells = dry_spells(days.rain_days(min(totals), today))

    longest = None
    if not spells.empty:
        row = spells.loc[spells["days"].idxmax()]
        longest = DrySpell(
            first_day=row["first_day"].isoformat(),
            last_day=row["last_day"].isoformat(),
            days=int(row["days"]),
        )

    first_wy = iso_day(water_year_status.first_day)
    wy_days = coverage.counted_days(first_wy, today)

    def rain_days() -> Optional[float]:
        """Rain days, scaled over the water year's missing days."""
        wet = sum(1 for d in wy_days if totals.get(d, 0.0) >= WET_DAY_IN)
        return number(prorate(water_year_status, float(wet)))

    def wettest() -> Optional[RainDay]:
        """The day with the most rain; a maximum is not prorated."""
        best = max(wy_days, key=lambda d: totals.get(d, 0.0), default=None)
        if best is None or totals.get(best, 0.0) < WET_DAY_IN:
            return None
        return RainDay(day=best.isoformat(), rain_in=number(totals[best]))

    return DrySpells(
        longest=longest,
        water_year_rain_days=comparable(water_year_status, rain_days),
        water_year_wettest=comparable(water_year_status, wettest),
    )


def _water_years(days: CountedDays) -> Tuple[WaterYearCurve, ...]:
    """
    Every water year from the archive's first through the current one.

    Each place is the window from Oct 1 to that day, judged and prorated by
    the gap rules, so a curve reads exactly as the window totals do.

    :param days: The counted days and the daily rain series.
    :return: The curves, oldest first.
    """
    coverage, totals = days.rain, days.rain_in
    first_day = coverage.first_day
    if first_day is None:
        return ()
    today = coverage.today
    today_place = water_year_day_index(today)
    curves = []
    for year in range(water_year(first_day), water_year(today) + 1):
        start = water_year_start(year)
        end = min(water_year_start(year + 1) - timedelta(days=1), today)
        cumulative: List[Optional[float]] = [None] * PLACES
        prorated = [False] * PLACES
        reasons: List[Optional[GapReason]] = [None] * PLACES
        last_place = 0
        running = 0.0
        status = coverage.window(start, start)
        day = start
        while day <= end:
            # The counted days of Oct 1..day are its "ok" days: `totals`.
            running += totals.get(day, 0.0)
            place = water_year_day_index(day)
            status = coverage.window(start, day, key=f"water_year_{year}")
            cumulative[place] = comparable(
                status, lambda: number(prorate(status, running))
            )
            prorated[place] = status.status == PRORATED
            # Feb 29 shares Feb 28's place: missing when either day is.
            reasons[place] = reasons[place] or _day_reason(coverage, day)
            last_place = place
            day += timedelta(days=1)
        reached = last_place + 1
        curves.append(
            WaterYearCurve(
                water_year=year,
                first_day=start.isoformat(),
                cumulative_in=tuple(cumulative[:reached]),
                prorated=tuple(prorated[:reached]),
                status=status,
                total_in=cumulative[last_place],
                to_date_in=(cumulative[today_place] if today_place < reached else None),
                reason=tuple(reasons[:reached]),
            )
        )
    return tuple(curves)


def _day_reason(coverage: ArchiveCoverage, day: date) -> Optional[GapReason]:
    """
    Why a day is not in the record for rain, as the rain day kind judges it.

    :param coverage: The rain day kind (`CountedDays.rain`).
    :param day: The station-local day.
    :return: Its gap reason ("gap", "before_archive", ...); None for a
        counted day.
    """
    if coverage.day_kind(day) == "ok":
        return None
    return coverage.window(day, day).reason or "gap"


def _covered(wet_to_date: WetHoursToDate) -> FrozenSet[int]:
    """
    The earlier water years that cover this date: usual wet hours' years,
    so the rule lives in one place (`lookout.snapshot.wet_hours`).

    :param wet_to_date: This water year's wet hours since Oct 1.
    :return: Those water years.
    """
    usual = wet_to_date.usual
    return frozenset(y.water_year for y in usual.earlier) if usual else frozenset()


def _to_date_years(
    curves: Tuple[WaterYearCurve, ...], covered: FrozenSet[int]
) -> Tuple[WaterYearCurve, ...]:
    """
    The earlier water years the usual rain to date rests on.

    :param curves: The water-year curves, oldest first; the last is this one.
    :param covered: The earlier water years that cover this date.
    :return: Those with a value by this date (`to_date_in`) that also cover
        it, oldest first.
    """
    return tuple(
        c for c in curves[:-1] if c.to_date_in is not None and c.water_year in covered
    )


def _usual_to_date(
    current: Optional[float], earlier: Tuple[WaterYearCurve, ...]
) -> Optional[UsualToDate]:
    """
    This water year's rain to date against the usual rain by this date.

    :param current: This water year's rain to date; None when unknown.
    :param earlier: The earlier years the usual rests on (`_to_date_years`).
    :return: The usual, this year's percent of it and the verdict; None
        when no earlier year counts.
    """
    if not earlier:
        return None
    # The percent is of the published values, so it reads from them exactly;
    # Python's round (half to even), as the feature's percentile is.
    usual_in = round(median(c.to_date_in for c in earlier), 3)  # as `number`
    percent = None
    if current is not None and usual_in > 0:
        percent = round(100 * current / usual_in)
    return UsualToDate(
        usual_in=usual_in,
        years=len(earlier),
        percent=percent,
        verdict=_verdict(current, percent),
        water_years=tuple(c.water_year for c in earlier),
    )


def _verdict(current: Optional[float], percent: Optional[int]) -> Optional[Verdict]:
    """
    Wetter, drier or about the same as the usual rain to date (#640).

    :param current: This water year's rain to date; None when unknown.
    :param percent: `current` as a whole percent of the usual; None when
        the usual is 0 (or `current` is None).
    :return: The verdict; None when this year's rain is unknown.
    """
    if current is None:
        return None
    if percent is None:  # a usual of 0
        return "wetter" if current > 0 else "about the same"
    if percent > 100 + ABOUT_THE_SAME:
        return "wetter"
    if percent < 100 - ABOUT_THE_SAME:
        return "drier"
    return "about the same"


def _years_back(day: date, years: int) -> date:
    """
    The same date a whole number of years back; Feb 29 as Feb 28.

    :param day: The date.
    :param years: How many years back.
    :return: That date.
    """
    if (day.month, day.day) == (2, 29):
        day = day.replace(day=28)
    return day.replace(year=day.year - years)


def _window_strip(
    windows: Tuple[RainWindow, ...],
    to_date: Tuple[WaterYearCurve, ...],
    days: CountedDays,
) -> Tuple[StripWindow, ...]:
    """
    Each window's rain against its usual range from earlier years.

    :param windows: The Rain tab's windows.
    :param to_date: The earlier water years the usual rain to date rests
        on; the water year's window rests on the same.
    :param days: The counted days and the daily rain series.
    :return: One `StripWindow` per window, in order.
    """
    today = days.clock.today
    strip = []
    for window in windows:
        if window.key == "water_year":
            earlier = [
                EarlierWindowRain(
                    last_day=_years_back(
                        today, water_year(today) - c.water_year
                    ).isoformat(),
                    rain_in=c.to_date_in,
                )
                for c in to_date
            ]
        else:
            earlier = _earlier_windows(window.status, days)
        running = iso_day(window.status.last_day) == today
        strip.append(_strip_window(window, earlier, running))
    return tuple(strip)


def _earlier_windows(status: GapStatus, days: CountedDays) -> List[EarlierWindowRain]:
    """
    The same window in each earlier year the record reaches, where its rain
    can be read (valid or prorated).

    :param status: This window's rain gap status.
    :param days: The counted days and the daily rain series.
    :return: Each readable earlier window's rain, oldest first.
    """
    earliest = days.rain.first_day
    first, last = iso_day(status.first_day), iso_day(status.last_day)
    length = last - first
    earlier = []
    back = 1
    while earliest is not None:
        end = _years_back(last, back)
        if end - length < earliest:
            break
        total = days.rain_total(days.rain.window(end - length, end))
        if total is not None:
            earlier.append(EarlierWindowRain(last_day=end.isoformat(), rain_in=total))
        back += 1
    return earlier[::-1]


def _strip_window(
    window: RainWindow, earlier: Sequence[EarlierWindowRain], running: bool
) -> StripWindow:
    """
    One window against its usual range.

    :param window: The window and its rain.
    :param earlier: The earlier years' rain over it, oldest first.
    :param running: Whether the window ends today, so its rain can grow.
    :return: The `StripWindow`.
    """
    values = [e.rain_in for e in earlier]
    low = high = band = None
    usual = usual_band(values)
    if usual is not None:
        low, high, band = number(usual[0]), number(usual[1]), usual[2]
    elif values:
        low, high = min(values), max(values)
    mark: Optional[StripMark] = None
    rain = window.total_in
    if rain is not None and low is not None and high is not None:
        if rain > high:
            mark = "above usual range"
        elif rain < low and not running:
            mark = "below usual range"
    return StripWindow(
        key=window.key,
        rain_in=rain,
        years=len(earlier),
        earlier=tuple(earlier),
        low_in=low,
        high_in=high,
        band=band,
        mark=mark,
    )
