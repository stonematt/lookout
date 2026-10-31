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
- Year-over-year accumulation runs over the water year from Oct 1
  (GLOSSARY.md), where Streamlit used the calendar year.

Functions:
- rain_tab(archive, history, current, now): the `RainTab` section.
- daily_totals(archive, current, coverage): each day's total, today live.
- window_total(status, totals, coverage): one window's rain, gap rules held.

Integration: any other section that totals rain over a window (the
Overview's rolling-window tables) calls `daily_totals` and `window_total`, so
each window has one rain total everywhere (the One Number Rule).
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lookout.core.rainfall_analysis import (
    WET_DAY_IN,
    dry_spells,
    extract_daily_rainfall,
    hourly_rain,
    water_year,
    water_year_day_index,
    water_year_start,
)
from lookout.snapshot.gaps import PRORATED, ArchiveCoverage, comparable, prorate
from lookout.snapshot.model import (
    DrySpell,
    DrySpells,
    GapStatus,
    RainDay,
    RainHour,
    RainTab,
    RainWindow,
    RecentRain,
    SnapshotConditions,
    WaterYearCurve,
)
from lookout.utils.dateutc import DEFAULT_TZ

RECENT_WINDOW = "90d"
"""The standard window the daily bars and the heatmap cover."""

USUAL_SPELL_DAYS = 30
"""A spell is "usual for this date" when it ended this many days either side
of it, in any year."""

HOURS = 24
"""Local hours in a heatmap day."""

PLACES = 365
"""Places in a water-year curve: Oct 1 to Sep 30, Feb 29 folded in."""

_DIGITS = 3
"""Inches are published to a thousandth."""


def rain_tab(
    archive: pd.DataFrame,
    history: pd.DataFrame,
    current: SnapshotConditions,
    now: datetime,
    tz: str = DEFAULT_TZ,
) -> RainTab:
    """
    The Rain tab's section.

    :param archive: The station's archive as stored, for the gap rules and
        the daily totals.
    :param history: The archive with the live reading stitched in
        (`stitch_live`), for today's hours.
    :param current: Current conditions, live stitched: today's rain and the
        last rain time.
    :param now: The build time, timezone-aware.
    :param tz: The station-local zone.
    :return: The section.
    """
    coverage = ArchiveCoverage(archive, now, tz)
    totals = daily_totals(archive, current, coverage)
    windows = coverage.standard_windows()
    recent_status = next(s for s in windows if s.key == RECENT_WINDOW)
    water_year_status = next(s for s in windows if s.key == "water_year")
    return RainTab(
        windows=tuple(
            RainWindow(key=s.key, total_in=window_total(s, totals, coverage))
            for s in windows
        ),
        recent=_recent(recent_status, totals, history, coverage, now, tz),
        dry_spells=_dry_spells(totals, current, water_year_status, coverage, tz),
        day_index=water_year_day_index(coverage.today),
        water_years=_water_years(totals, coverage),
    )


def daily_totals(
    archive: pd.DataFrame,
    current: SnapshotConditions,
    coverage: ArchiveCoverage,
) -> Dict[date, float]:
    """
    Each counted day's rain: its highest daily counter; today's live total.

    Only days the gap rules count ("ok" days) are present, so a day missing
    from the record is never read as a dry one.

    :param archive: The station's archive as stored.
    :param current: Current conditions, live stitched.
    :param coverage: The archive's coverage as of the build time.
    :return: Day -> inches, for every counted day.
    """
    totals: Dict[date, float] = {}
    if not archive.empty:
        frame = extract_daily_rainfall(archive)
        totals = {d: float(r) for d, r in zip(frame["date"], frame["rainfall"])}
    if current.rain_today_in is not None:
        totals[coverage.today] = float(current.rain_today_in)
    return {
        d: total
        for d, total in totals.items()
        if coverage.day_kind(d) == "ok" and not np.isnan(total)
    }


def window_total(
    status: GapStatus, totals: Dict[date, float], coverage: ArchiveCoverage
) -> Optional[float]:
    """
    One window's rain: the sum over its counted days, prorated.

    :param status: The window's gap status.
    :param totals: `daily_totals`.
    :param coverage: The coverage `status` came from.
    :return: Inches; None when the window is invalid.
    """

    def total() -> Optional[float]:
        """The sum over the counted days, prorated."""
        days = coverage.counted_days(_day(status.first_day), _day(status.last_day))
        return _round(prorate(status, sum(totals.get(d, 0.0) for d in days)))

    return comparable(status, total)


def _recent(
    status: GapStatus,
    totals: Dict[date, float],
    history: pd.DataFrame,
    coverage: ArchiveCoverage,
    now: datetime,
    tz: str,
) -> RecentRain:
    """
    Daily and hourly rain over the days of the recent window.

    :param status: The recent window's gap status.
    :param totals: `daily_totals`.
    :param history: The stitched history, for the hours (today's included).
    :param coverage: The archive's coverage.
    :param now: The build time: hours after its hour today are still to come.
    :param tz: The station-local zone.
    :return: The recent days, with wet hours prorated like any window value.
    """
    first, last = _day(status.first_day), _day(status.last_day)
    days = [first + timedelta(days=i) for i in range(status.days)]
    span = history
    if not history.empty:
        start_ms = pd.Timestamp(first).tz_localize(tz).value // 10**6
        span = history[history["dateutc"] >= start_ms]
    hours = hourly_rain(span, tz)
    by_hour = {(d, h): r for d, h, r in hours.itertuples(index=False, name=None)}
    now_local = pd.Timestamp(now).tz_convert(tz)

    rain_in: List[Optional[float]] = []
    reasons = []
    hourly: List[Tuple[Optional[float], ...]] = []
    for d in days:
        if d not in totals:
            reason = coverage.window(d, d).reason
            rain_in.append(None)
            reasons.append(reason or "gap")
            hourly.append((None,) * HOURS)
            continue
        rain_in.append(_round(totals[d]))
        reasons.append(None)
        to_come = now_local.hour + 1 if d == last else HOURS
        hourly.append(
            tuple(
                _round(by_hour.get((d, h), 0.0)) if h < to_come else None
                for h in range(HOURS)
            )
        )

    cells = [
        (d, h, r)
        for d, row in zip(days, hourly)
        for h, r in enumerate(row)
        if r is not None
    ]

    def wet_hours() -> Optional[float]:
        """Hours with rain, scaled over the window's missing days."""
        return _round(prorate(status, float(sum(1 for _, _, r in cells if r > 0))))

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
        wet_hours=comparable(status, wet_hours),
        wettest_hour=comparable(status, wettest),
    )


def _dry_spells(
    totals: Dict[date, float],
    current: SnapshotConditions,
    water_year_status: GapStatus,
    coverage: ArchiveCoverage,
    tz: str,
) -> DrySpells:
    """
    The current dry spell against the completed ones in the record.

    :param totals: `daily_totals`: the counted days.
    :param current: Current conditions, for the live last rain time.
    :param water_year_status: The water year's gap status.
    :param coverage: The archive's coverage.
    :param tz: The station-local zone.
    :return: The dry-spell stats.
    """
    today = coverage.today
    current_days = None
    if current.last_rain_ms is not None:
        last_rain = pd.Timestamp(current.last_rain_ms, unit="ms", tz="UTC")
        current_days = max(0, (today - last_rain.tz_convert(tz).date()).days)

    spells = pd.DataFrame(columns=["first_day", "last_day", "days"])
    if totals:
        first = min(totals)
        index = [first + timedelta(days=i) for i in range((today - first).days + 1)]
        series = pd.Series([totals.get(d, np.nan) for d in index], index=index)
        spells = dry_spells(series)

    near = [
        int(row.days)
        for row in spells.itertuples(index=False)
        if _days_apart(row.last_day, today) <= USUAL_SPELL_DAYS
    ]
    longest = None
    if not spells.empty:
        row = spells.loc[spells["days"].idxmax()]
        longest = DrySpell(
            first_day=row["first_day"].isoformat(),
            last_day=row["last_day"].isoformat(),
            days=int(row["days"]),
        )

    first_wy = _day(water_year_status.first_day)
    wy_days = coverage.counted_days(first_wy, today)

    def rain_days() -> Optional[float]:
        """Rain days, scaled over the water year's missing days."""
        wet = sum(1 for d in wy_days if totals.get(d, 0.0) >= WET_DAY_IN)
        return _round(prorate(water_year_status, float(wet)))

    def wettest() -> Optional[RainDay]:
        """The day with the most rain; a maximum is not prorated."""
        best = max(wy_days, key=lambda d: totals.get(d, 0.0), default=None)
        if best is None or totals.get(best, 0.0) < WET_DAY_IN:
            return None
        return RainDay(day=best.isoformat(), rain_in=_round(totals[best]))

    return DrySpells(
        current_days=current_days,
        usual_days=float(np.median(near)) if near else None,
        usual_sample=len(near),
        longest=longest,
        water_year_rain_days=comparable(water_year_status, rain_days),
        water_year_wettest=comparable(water_year_status, wettest),
    )


def _water_years(
    totals: Dict[date, float], coverage: ArchiveCoverage
) -> Tuple[WaterYearCurve, ...]:
    """
    Every water year from the archive's first through the current one.

    Each place is the window from Oct 1 to that day, judged and prorated by
    the gap rules, so a curve reads exactly as the window totals do.

    :param totals: `daily_totals`: the counted days.
    :param coverage: The archive's coverage.
    :return: The curves, oldest first.
    """
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
            cumulative[place] = _round(prorate(status, running))
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


def _days_apart(a: date, b: date) -> int:
    """Days between two dates' places in the year, either way round."""
    gap = abs(water_year_day_index(a) - water_year_day_index(b))
    return min(gap, PLACES - gap)


def _day(iso: str) -> date:
    """A date from its ISO form."""
    return date.fromisoformat(iso)


def _round(value: Optional[float]) -> Optional[float]:
    """Inches to a thousandth, None kept."""
    return None if value is None else round(float(value), _DIGITS)
