"""
rain_tab.py: The Rain tab's section of the snapshot (#83).

The Streamlit Rain tab's data prep, moved behind `build_snapshot` and set to
the new site's rules:

- Every rain total is an archive sum with today's live total stitched in
  (spec #66, decided 2026-10-30): a day's total is its highest `dailyrainin`,
  and today's is the live reading's (`SnapshotConditions.rain_today_in`), so
  the Rain tab, the Overview and `/current` show one number for today.
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

Functions:
- rain_tab(history, days): the `RainTab` section.

Integration: any other section that totals rain over a window (the
Overview's rolling-window tables) reads the same `CountedDays` and calls
`CountedDays.rain_total`, so each window has one rain total and one status
everywhere (the One Number Rule).
"""

from datetime import timedelta
from typing import List, Optional, Tuple

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
from lookout.snapshot.gaps import PRORATED, comparable, prorate
from lookout.snapshot.model import (
    DrySpell,
    DrySpells,
    GapStatus,
    RainDay,
    RainHour,
    RainTab,
    RainWindow,
    RecentRain,
    WaterYearCurve,
)
from lookout.snapshot.values import iso_day, number
from lookout.utils.dateutc import day_span, from_local_date

RECENT_WINDOW = "90d"
"""The standard window the daily bars and the heatmap cover."""

HOURS = 24
"""Local hours in a heatmap day."""

PLACES = 365
"""Places in a water-year curve: Oct 1 to Sep 30, Feb 29 folded in."""


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
    return RainTab(
        windows=tuple(
            RainWindow(key=s.key, status=s, total_in=days.rain_total(s))
            for s in windows
        ),
        recent=_recent(recent_status, history, days),
        dry_spells=_dry_spells(water_year_status, days),
        day_index=water_year_day_index(days.clock.today),
        water_years=_water_years(days),
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
        span = history[history["dateutc"] >= from_local_date(first, coverage.tz)]
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
        if coverage.day_kind(d) != "ok":
            reason = coverage.window(d, d).reason
            rain_in.append(None)
            reasons.append(reason or "gap")
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
            )
        )
    return tuple(curves)
