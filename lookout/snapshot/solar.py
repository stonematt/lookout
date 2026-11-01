"""
solar.py: The Solar tab's section of the snapshot (#85).

The Streamlit Solar tab's data prep (`lookout.ui.solar`, its tiles in
`lookout.core.solar_tiles` and its heatmap pivots in
`lookout.core.solar_data_transformer`), moved behind `build_snapshot` and set
to the new site's rules:

- Energy comes from the energy catalog: 15-minute periods in kWh per m².
  A day's energy is the sum of its periods; today's is the energy so far.
- Every value over a day, a month or a window goes through the gap rules
  (`lookout.snapshot.gaps`): a total is summed over the counted days and
  prorated; an invalid one is None and never computed. Streamlit summed
  whatever periods there were, so a gap read as a dark day.
- A day counts for solar only when the archive counts it *and* the catalog
  holds it (#123): the catalog only extends forward from its newest period,
  so a gap later backfilled into the archive is still missing from it. Such
  a day is missing by the same rule: no periods is a gap, under 80% of the
  archive's 15-minute slots is low coverage, before the catalog's first day
  is before the record, after its last is not archived yet. Each window and
  month carries this status, since it can differ from the archive's.
- The 24-hour view is the solar trace from the live-stitched history, so its
  newest point is the live reading (`SnapshotConditions.solar_wm2`), over the
  usual range for each time of day across the 30 days before today.
- The snapshot carries chart-ready series. The site draws the daily bars,
  the three heatmaps (year by month, month by day, day by 15 minutes) and a
  picked day's 15-minute bars from them; nothing is a Plotly figure.

Functions:
- solar_tab(energy_catalog, archive, history, observed_at_ms, clock): the
  `SolarTab` section, None without a catalog.
"""

from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from lookout.snapshot.clock import StationClock
from lookout.snapshot.gaps import (
    LOW_COVERAGE,
    ArchiveCoverage,
    comparable,
    prorate,
    standard_window_keys,
)
from lookout.snapshot.model import (
    GapStatus,
    SolarDays,
    SolarMonth,
    SolarRecent,
    SolarTab,
    SolarTrace,
    SolarWindow,
)
from lookout.snapshot.values import iso_day, number
from lookout.utils.dateutc import INTERVAL_MS, day_span, from_local_date, to_local

RECENT_WINDOW = "90d"
"""The standard window whose days carry their 15-minute energy."""

SLOT_MINUTES = 15
"""The energy catalog's period length."""

SLOTS = 24 * 60 // SLOT_MINUTES
"""Slots in a local day, by clock."""

SLOT_MS = SLOT_MINUTES * 60 * 1000

USUAL_DAYS = 30
"""The days before today the 24-hour view's usual range is drawn from."""

TRACE_SLOTS = 24 * 60 * 60 * 1000 // INTERVAL_MS
"""Steps in 24 hours; the trace holds one more, the newest observation's."""

_WH_DIGITS = 1


def solar_tab(
    energy_catalog: Optional[pd.DataFrame],
    archive: pd.DataFrame,
    history: pd.DataFrame,
    observed_at_ms: Optional[int],
    clock: StationClock,
) -> Optional[SolarTab]:
    """
    The Solar tab's section.

    :param energy_catalog: The energy catalog (`period_start`, `energy_kwh`),
        or None when there is none.
    :param archive: The station's archive as stored, for the gap rules and
        the usual range.
    :param history: The archive with the live reading stitched in
        (`stitch_live`), for the 24-hour trace.
    :param observed_at_ms: The newest observation, live or archived.
    :param clock: The station clock.
    :return: The section; None without an energy catalog.
    """
    if energy_catalog is None:
        return None
    archived = ArchiveCoverage(archive, clock)
    periods = _periods(energy_catalog, clock.tz)
    coverage = archived.excluding(_uncatalogued(periods, archive, archived))
    totals = _daily_totals(periods, coverage)
    keys = standard_window_keys()
    windows = [s for s in coverage.standard_windows() if s.key in keys]
    recent_status = next(s for s in windows if s.key == RECENT_WINDOW)
    return SolarTab(
        windows=tuple(
            SolarWindow(
                key=s.key,
                energy_kwh_m2=_window_total(s, totals, coverage),
                status=s,
            )
            for s in windows
        ),
        days=_days(totals, coverage),
        months=tuple(
            SolarMonth(
                key=s.key,
                energy_kwh_m2=_window_total(s, totals, coverage),
                status=s,
            )
            for s in coverage.months()
        ),
        recent=_recent(recent_status, periods, totals),
        last_24h=_last_24h(
            archive,
            history,
            periods,
            observed_at_ms,
            archive_coverage=archived,
            solar_coverage=coverage,
        ),
    )


def _periods(catalog: pd.DataFrame, tz: str) -> pd.DataFrame:
    """
    The catalog's periods with their local day and clock slot.

    :param catalog: The energy catalog.
    :param tz: The station-local zone.
    :return: Columns `start_ms`, `day`, `slot` (0..95 by local clock) and
        `kwh`, oldest first.
    """
    if catalog.empty:
        return pd.DataFrame(columns=["start_ms", "day", "slot", "kwh"])
    start = pd.to_datetime(catalog["period_start"], utc=True)
    local = start.dt.tz_convert(tz)
    frame = pd.DataFrame(
        {
            "start_ms": start.astype("int64") // 10**6,
            "day": local.dt.date,
            "slot": local.dt.hour * (60 // SLOT_MINUTES)
            + local.dt.minute // SLOT_MINUTES,
            "kwh": catalog["energy_kwh"].astype(float),
        }
    )
    return frame.dropna(subset=["kwh"]).sort_values("start_ms")


def _uncatalogued(
    periods: pd.DataFrame, archive: pd.DataFrame, coverage: ArchiveCoverage
) -> Dict[date, str]:
    """
    The days the archive counts but the catalog does not hold, and why.

    A day is held when the catalog has a period for at least `LOW_COVERAGE`
    of the 15-minute slots the archive has rows in that day.

    :param periods: `_periods`.
    :param archive: The archive as stored.
    :param coverage: The archive's coverage.
    :return: Day -> "gap" (no periods), "low" (too few), "before_archive"
        (before the catalog's first day) or "not_archived" (after its last).
    """
    first = coverage.first_day
    if first is None:
        return {}
    days = day_span(first, coverage.today)
    if periods.empty:
        return {d: "not_archived" for d in days}
    held = periods.groupby("day")["start_ms"].nunique().to_dict()
    stamps = pd.Series(archive["dateutc"], dtype="int64")
    stamps = stamps[stamps <= coverage.now_ms]
    slots = pd.Series((stamps // SLOT_MS).unique() * SLOT_MS)
    local = to_local(slots, coverage.tz)
    due = local.dt.date.value_counts().to_dict()
    first_held, last_held = min(held), max(held)

    missing: Dict[date, str] = {}
    for d in days:
        # A day with no archived slots yet (today, just begun) needs none.
        if held.get(d, 0) >= LOW_COVERAGE * due.get(d, 0):
            continue
        if d < first_held:
            missing[d] = "before_archive"
        elif d > last_held:
            missing[d] = "not_archived"
        else:
            missing[d] = "low" if d in held else "gap"
    return missing


def _daily_totals(
    periods: pd.DataFrame, coverage: ArchiveCoverage
) -> Dict[date, float]:
    """
    Each counted day's energy: the sum of its periods.

    Every day the gap rules count ("ok" days) is present, and only those.
    `coverage` already judges a day the catalog lacks as missing, so a day
    missing from the record is never read as a dark one; a counted day with
    no periods has nothing due yet (today, just begun) and holds 0.

    :param periods: `_periods`.
    :param coverage: The coverage for solar (`_uncatalogued` excluded).
    :return: Day -> kWh per m².
    """
    if coverage.first_day is None:
        return {}
    sums = dict(periods.groupby("day")["kwh"].sum()) if len(periods) else {}
    counted = coverage.counted_days(coverage.first_day, coverage.today)
    return {d: float(sums.get(d, 0.0)) for d in counted}


def _window_total(
    status: GapStatus, totals: Dict[date, float], coverage: ArchiveCoverage
) -> Optional[float]:
    """
    One window's or month's energy: the sum over its counted days, prorated.

    :param status: Its gap status.
    :param totals: `_daily_totals`.
    :param coverage: The coverage `status` came from.
    :return: kWh per m²; None when it is invalid.
    """

    def total() -> Optional[float]:
        """The sum over the counted days, prorated."""
        days = coverage.counted_days(
            iso_day(status.first_day), iso_day(status.last_day)
        )
        return number(prorate(status, sum(totals[d] for d in days)))

    return comparable(status, total)


def _days(totals: Dict[date, float], coverage: ArchiveCoverage) -> SolarDays:
    """
    Every day from the archive's first through today.

    :param totals: `_daily_totals`.
    :param coverage: The archive's coverage.
    :return: The days; empty for an empty archive.
    """
    first = coverage.first_day
    if first is None:
        return SolarDays(
            first_day=coverage.today.isoformat(), energy_kwh_m2=(), reason=()
        )
    energy: List[Optional[float]] = []
    reasons = []
    for d in day_span(first, coverage.today):
        if d in totals:
            energy.append(number(totals[d]))
            reasons.append(None)
        else:
            energy.append(None)
            reasons.append(coverage.window(d, d).reason)
    return SolarDays(
        first_day=first.isoformat(), energy_kwh_m2=tuple(energy), reason=tuple(reasons)
    )


def _recent(
    status: GapStatus, periods: pd.DataFrame, totals: Dict[date, float]
) -> SolarRecent:
    """
    The recent window's days, 15 minutes at a time.

    :param status: The recent window's gap status.
    :param periods: `_periods`.
    :param totals: `_daily_totals`: only these days get slots.
    :return: The recent days.
    """
    first, last = iso_day(status.first_day), iso_day(status.last_day)
    days = day_span(first, last)
    inside = periods[(periods["day"] >= first) & (periods["day"] <= last)]
    # A repeated hour on the day daylight saving ends is summed into its slot.
    by_slot = inside.groupby(["day", "slot"])["kwh"].sum().to_dict()
    rows = []
    for d in days:
        if d not in totals:
            rows.append((None,) * SLOTS)
            continue
        rows.append(
            tuple(
                _wh(by_slot[(d, s)]) if (d, s) in by_slot else None
                for s in range(SLOTS)
            )
        )
    return SolarRecent(
        window=status.key,
        first_day=status.first_day,
        slot_minutes=SLOT_MINUTES,
        energy_wh_m2=tuple(rows),
    )


def _last_24h(
    archive: pd.DataFrame,
    history: pd.DataFrame,
    periods: pd.DataFrame,
    observed_at_ms: Optional[int],
    archive_coverage: ArchiveCoverage,
    solar_coverage: ArchiveCoverage,
) -> Optional[SolarTrace]:
    """
    The 24-hour view, ending at the newest observation.

    :param archive: The archive as stored, for the usual range.
    :param history: The live-stitched history, for the trace.
    :param periods: `_periods`, for the energy.
    :param observed_at_ms: The newest observation.
    :param archive_coverage: The archive's coverage, for the usual range: it
        is drawn from the archive's readings, not the catalog.
    :param solar_coverage: The coverage for solar (the catalog's missing
        days excluded), for the energy.
    :return: The trace; None without an observation.
    """
    if observed_at_ms is None or history.empty:
        return None
    end = observed_at_ms // INTERVAL_MS * INTERVAL_MS
    start = end - TRACE_SLOTS * INTERVAL_MS
    solar = _trace(history, start, end)

    today = solar_coverage.today
    usual = archive_coverage.window(
        today - timedelta(days=USUAL_DAYS),
        today - timedelta(days=1),
        key="solar_usual",
    )
    band = comparable(
        usual, lambda: _usual_band(archive, usual, archive_coverage, start)
    )

    day_ms = TRACE_SLOTS * INTERVAL_MS
    energy = solar_coverage.window(today - timedelta(days=1), today, key="last_24h")
    before = solar_coverage.window(
        today - timedelta(days=2), today - timedelta(days=1), key="previous_24h"
    )
    return SolarTrace(
        start_ms=int(start),
        step_minutes=INTERVAL_MS // 60_000,
        solar_wm2=solar,
        usual=usual,
        usual_low_wm2=band[0] if band else None,
        usual_median_wm2=band[1] if band else None,
        usual_high_wm2=band[2] if band else None,
        energy=energy,
        energy_kwh_m2=comparable(
            energy,
            _energy_between(periods, observed_at_ms - day_ms, observed_at_ms),
        ),
        previous_kwh_m2=comparable(
            before,
            _energy_between(
                periods, observed_at_ms - 2 * day_ms, observed_at_ms - day_ms
            ),
        ),
    )


def _trace(history: pd.DataFrame, start: int, end: int) -> Tuple[Optional[float], ...]:
    """
    Solar radiation in five-minute slots from `start` through `end`.

    :param history: The live-stitched history, newest first.
    :param start: The first slot.
    :param end: The last slot, the newest observation's.
    :return: One value per slot, None where nothing was read.
    """
    count = (end - start) // INTERVAL_MS + 1
    values: List[Optional[float]] = [None] * count
    if "solarradiation" not in history:
        return tuple(values)
    stamps = history["dateutc"].astype("int64")
    rows = history[(stamps >= start) & (stamps < end + INTERVAL_MS)]
    slots = (rows["dateutc"].astype("int64") - start) // INTERVAL_MS
    # Newest first: a slot read twice keeps its newest reading, the live one.
    for slot, value in zip(slots[::-1], rows["solarradiation"][::-1]):
        if not pd.isna(value):
            values[int(slot)] = float(value)
    return tuple(values)


def _usual_band(
    archive: pd.DataFrame, usual: GapStatus, coverage: ArchiveCoverage, start: int
) -> Tuple[Tuple[Optional[float], ...], ...]:
    """
    The usual range at each trace slot's time of day: its quartiles and
    median over the comparison days' readings at that time.

    :param archive: The archive as stored.
    :param usual: The comparison days' gap status.
    :param coverage: The archive's coverage.
    :param start: The trace's first slot.
    :return: (low, median, high), each parallel to the trace.
    """
    counted = set(
        coverage.counted_days(iso_day(usual.first_day), iso_day(usual.last_day))
    )
    first_ms = from_local_date(iso_day(usual.first_day), coverage.tz)
    last_ms = from_local_date(iso_day(usual.last_day) + timedelta(days=1), coverage.tz)
    stamps = archive["dateutc"].astype("int64")
    rows = archive[(stamps >= first_ms) & (stamps < last_ms)]
    local = to_local(rows["dateutc"], coverage.tz)
    frame = pd.DataFrame(
        {"tod": _time_of_day(local), "solar": rows["solarradiation"].astype(float)}
    )[local.dt.date.isin(counted).to_numpy()]
    quartiles = frame.dropna().groupby("tod")["solar"].quantile([0.25, 0.5, 0.75])
    table = quartiles.unstack().to_dict("index")

    slots = pd.Series([start + i * INTERVAL_MS for i in range(TRACE_SLOTS + 1)])
    keys = _time_of_day(to_local(slots, coverage.tz))

    def column(q: float) -> Tuple[Optional[float], ...]:
        """One quantile at every trace slot."""
        return tuple(
            round(float(table[k][q]), _WH_DIGITS) if k in table else None for k in keys
        )

    return column(0.25), column(0.5), column(0.75)


def _energy_between(
    periods: pd.DataFrame, first_ms: int, end_ms: int
) -> Callable[[], Optional[float]]:
    """
    The energy of the periods starting in [first_ms, end_ms), deferred so
    `comparable` can skip it.

    :param periods: `_periods`.
    :param first_ms: The span's start.
    :param end_ms: Its end, exclusive.
    :return: A function computing kWh per m².
    """

    def total() -> Optional[float]:
        """The periods' sum."""
        inside = (periods["start_ms"] >= first_ms) & (periods["start_ms"] < end_ms)
        return number(float(periods.loc[inside, "kwh"].sum()))

    return total


def _time_of_day(local: pd.Series) -> List[int]:
    """Each local time's five-minute slot of the day, 0..287."""
    minutes = local.dt.hour * 60 + local.dt.minute
    return list((minutes // (INTERVAL_MS // 60_000)).astype(int))


def _wh(kwh: float) -> float:
    """kWh as Wh, to a tenth."""
    return round(float(kwh) * 1000, _WH_DIGITS)
