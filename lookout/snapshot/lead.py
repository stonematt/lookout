"""
lead.py: The Overview's hero lead, chosen by the Adaptive Lead Rule.

The rules are checked in order (spec #66 "Lead choice", DESIGN.md "The
Adaptive Lead Rule"), and the first that holds leads:

1. **rain**: an event is live, or the newest event ended at most
   `RAIN_HOURS` ago. Its value is the hours since that event ended (0 while
   one is live). The Event Catalog's `end_time` is the event's last
   positive event counter reading, so "ended" means the station closed it.
2. **temperature**: the current temperature is at least `TEMPERATURE_F`
   either side of the month median. The month median is the median of every
   reading at this local hour in the same calendar month of earlier years:
   a whole-day median would let the daily swing alone lead every afternoon.
   Each earlier month is used only when its gap status is comparable; with
   none, the rule cannot hold ("no_comparison").
3. **solar**: the current radiation is at least `SOLAR_WM2`.
4. **calm**, otherwise: the days since it last rained (`lastRain`) against
   the usual dry spell, the median completed dry spell in the last 365 days.
   A dry spell is a run of dry days between two rain days (at least
   `RAIN_DAY_IN`); a day the archive cannot count breaks the run, so a gap
   never reads as dry. With the 365-day window invalid, calm still leads,
   without a usual dry spell ("no_comparison").

Values are measured at the newest observation, read from the live-stitched
history, so they agree with the first screen. Gap statuses read the archive
as stored (`lookout.snapshot.gaps`).

Every rule's value is computed, met or not, for "Why this leads". The hero
chart's data ("bold now over a soft usual range", DESIGN.md) is computed for
the lead only.

Functions:
- hero_lead(archive, stitched, event_catalog, active, current, observed_at_ms,
  now): The `HeroLead` section.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lookout.snapshot.current import to_ms
from lookout.snapshot.gaps import ArchiveCoverage, comparable
from lookout.snapshot.model import (
    ActiveRainEvent,
    HeroLead,
    LeadKind,
    LeadRule,
    LeadTrace,
    SnapshotConditions,
)
from lookout.utils.dateutc import DEFAULT_TZ

RAIN_HOURS = 12.0
"""Rain leads while an event is live or ended at most this many hours ago."""

TEMPERATURE_F = 10.0
"""Temperature leads at least this far, either side, from the month median."""

SOLAR_WM2 = 400.0
"""Solar leads at or above this radiation."""

RAIN_DAY_IN = 0.01
"""A day with at least this much rain (one bucket tip) ends a dry spell."""

DRY_SPELL_WINDOW = "365d"
"""The window the usual dry spell is measured over."""

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
RAIN_SLOT_MINUTES = 15
_RAIN_SLOT_MS = RAIN_SLOT_MINUTES * 60_000

_Usual = Tuple[Optional[float], Optional[float], Optional[float]]
_NO_USUAL: _Usual = (None, None, None)


def _round(value: Optional[float], digits: int) -> Optional[float]:
    """Round a value, keeping None."""
    return None if value is None else round(float(value), digits)


def _quartiles(values: Sequence[float], digits: int) -> _Usual:
    """The lower quartile, median and upper quartile; Nones when empty."""
    if len(values) == 0:
        return _NO_USUAL
    q = np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])
    return (
        round(float(q[0]), digits),
        round(float(q[1]), digits),
        round(float(q[2]), digits),
    )


def _local(ms: pd.Series, tz: str) -> pd.Series:
    """Epoch ms as station-local timestamps."""
    return pd.to_datetime(ms, unit="ms", utc=True).dt.tz_convert(tz)


# ---------- rain ----------


def _newest_event(
    event_catalog: Optional[pd.DataFrame], observed_at_ms: Optional[int]
) -> Optional[pd.Series]:
    """The catalog event that ended last, up to the newest observation."""
    if event_catalog is None or event_catalog.empty or observed_at_ms is None:
        return None
    ends = event_catalog["end_time"].map(to_ms)
    ended = ends[ends.notna() & (ends <= observed_at_ms)]
    if ended.empty:
        return None
    return event_catalog.loc[ended.astype("int64").idxmax()]


def _rain_rule(
    newest: Optional[pd.Series],
    active: Optional[ActiveRainEvent],
    observed_at_ms: Optional[int],
) -> LeadRule:
    """The rain rule: live, or the hours since the newest event ended."""
    value: Optional[float] = None
    if active is not None:
        value = 0.0
    elif newest is not None and observed_at_ms is not None:
        value = round((observed_at_ms - to_ms(newest["end_time"])) / HOUR_MS, 2)
    return LeadRule(
        kind="rain",
        met=value is not None and value <= RAIN_HOURS,
        value=value,
        unit="hours",
        threshold=RAIN_HOURS,
        reference=None,
        compared=(),
        missing="no_event" if value is None else None,
    )


def _rain_chart(
    stitched: pd.DataFrame,
    event_catalog: Optional[pd.DataFrame],
    newest: Optional[pd.Series],
    active: Optional[ActiveRainEvent],
    current: SnapshotConditions,
    observed_at_ms: Optional[int],
) -> Tuple[Optional[float], _Usual, int, Optional[LeadTrace]]:
    """
    The rain chart: the event so far over completed events' totals.

    :return: (now, usual, sample, trace).
    """
    if active is not None:
        event_id, start_ms, end_ms = active.event_id, active.start_ms, observed_at_ms
        now: Optional[float] = active.total_in
    elif newest is not None:
        event_id = str(newest["event_id"])
        start_ms, end_ms = to_ms(newest["start_time"]), to_ms(newest["end_time"])
        now = _round(newest.get("total_rainfall"), 3)
    else:
        return current.rain_event_in, _NO_USUAL, 0, None

    totals: List[float] = []
    if event_catalog is not None and not event_catalog.empty:
        others = event_catalog[event_catalog["event_id"].astype(str) != event_id]
        totals = [float(t) for t in others["total_rainfall"].dropna()]

    trace = None
    if start_ms is not None and end_ms is not None and "eventrainin" in stitched:
        first = start_ms - start_ms % _RAIN_SLOT_MS
        slots = (end_ms - first) // _RAIN_SLOT_MS + 1
        rows = stitched[
            (stitched["dateutc"] >= first) & (stitched["dateutc"] <= end_ms)
        ]
        index = (rows["dateutc"] - first) // _RAIN_SLOT_MS
        per_slot = rows["eventrainin"].groupby(index).max()
        values = tuple(
            _round(per_slot.get(i), 3) if pd.notna(per_slot.get(i)) else None
            for i in range(int(slots))
        )
        trace = LeadTrace(
            start_ms=int(first),
            slot_minutes=RAIN_SLOT_MINUTES,
            values=values,
            usual_low=(),
            usual_median=(),
            usual_high=(),
        )
    return now, _quartiles(totals, 3), len(totals), trace


# ---------- temperature and solar: today against the month's hours ----------


class _MonthHours:
    """
    The same calendar month in earlier years, by local hour of day.

    Only months whose gap status is comparable are read.

    :param archive: The archive as stored.
    :param coverage: Its coverage as of `now`.
    """

    def __init__(self, archive: pd.DataFrame, coverage: ArchiveCoverage) -> None:
        self.tz = coverage.tz
        today = coverage.today
        first = coverage.first_day
        keys: List[str] = []
        frames: List[pd.DataFrame] = []
        local = _local(archive["dateutc"], self.tz) if not archive.empty else None
        if first is not None and local is not None:
            for year in range(today.year - 1, first.year - 1, -1):
                status = coverage.month(year, today.month)
                rows = comparable(
                    status,
                    lambda y=year: archive[
                        (local.dt.year == y) & (local.dt.month == today.month)
                    ],
                )
                if rows is not None:
                    keys.append(status.key)
                    frames.append(rows)
        self.keys: Tuple[str, ...] = tuple(keys)
        self.rows = pd.concat(frames) if frames else pd.DataFrame()

    def days(self) -> int:
        """How many days the comparison set holds."""
        if self.rows.empty:
            return 0
        return int(_local(self.rows["dateutc"], self.tz).dt.date.nunique())

    def by_hour(self, column: str, digits: int) -> Dict[int, _Usual]:
        """The quartiles of `column` at each local hour of day."""
        if self.rows.empty or column not in self.rows:
            return {}
        values = pd.to_numeric(self.rows[column], errors="coerce")
        hours = _local(self.rows["dateutc"], self.tz).dt.hour
        return {
            int(hour): _quartiles(group.dropna().to_numpy(), digits)
            for hour, group in values.groupby(hours)
        }


def _temperature_rule(
    current: SnapshotConditions,
    usual_now: _Usual,
    month_hours: _MonthHours,
) -> LeadRule:
    """The temperature rule: degrees from this hour's month median."""
    temperature, median = current.temperature_f, usual_now[1]
    value = None
    if temperature is not None and median is not None:
        value = round(temperature - median, 1)
    missing = None
    if median is None:
        missing = "no_comparison"
    elif temperature is None:
        missing = "no_reading"
    return LeadRule(
        kind="temperature",
        met=value is not None and abs(value) >= TEMPERATURE_F,
        value=value,
        unit="degrees_f",
        threshold=TEMPERATURE_F,
        reference=median,
        compared=month_hours.keys if median is not None else (),
        missing=missing,
    )


def _solar_rule(current: SnapshotConditions) -> LeadRule:
    """The solar rule: the current radiation."""
    value = current.solar_wm2
    return LeadRule(
        kind="solar",
        met=value is not None and value >= SOLAR_WM2,
        value=value,
        unit="wm2",
        threshold=SOLAR_WM2,
        reference=None,
        compared=(),
        missing="no_reading" if value is None else None,
    )


def _day_trace(
    stitched: pd.DataFrame,
    column: str,
    by_hour: Dict[int, _Usual],
    today: date,
    tz: str,
    digits: int,
) -> LeadTrace:
    """Today's hourly means from local midnight, with each hour's band."""
    midnight = pd.Timestamp(today).tz_localize(tz)
    next_midnight = pd.Timestamp(today + timedelta(days=1)).tz_localize(tz)
    start_ms = int(midnight.value // 10**6)
    slots = int((next_midnight - midnight) / pd.Timedelta(hours=1))
    hours = [
        pd.Timestamp(start_ms + i * HOUR_MS, unit="ms", tz="UTC").tz_convert(tz).hour
        for i in range(slots)
    ]

    values: List[Optional[float]] = [None] * slots
    if column in stitched:
        rows = stitched[stitched["dateutc"] >= start_ms]
        index = (rows["dateutc"] - start_ms) // HOUR_MS
        means = pd.to_numeric(rows[column], errors="coerce").groupby(index).mean()
        for i, mean in means.items():
            if 0 <= i < slots and pd.notna(mean):
                values[int(i)] = round(float(mean), digits)

    bands = [by_hour.get(hour, _NO_USUAL) for hour in hours]
    return LeadTrace(
        start_ms=start_ms,
        slot_minutes=60,
        values=tuple(values),
        usual_low=tuple(band[0] for band in bands),
        usual_median=tuple(band[1] for band in bands),
        usual_high=tuple(band[2] for band in bands),
    )


# ---------- calm: days since rain against the usual dry spell ----------


def _dry_spells(
    stitched: pd.DataFrame, coverage: ArchiveCoverage
) -> Tuple[Optional[List[int]], Tuple[str, ...]]:
    """
    Every completed dry spell in the 365-day window, in days.

    :return: (spell lengths, or None when the window is invalid; its key when
        comparable).
    """
    status = coverage.rolling(365, DRY_SPELL_WINDOW)

    def spells() -> List[int]:
        first, last = date.fromisoformat(status.first_day), coverage.today
        counted = set(coverage.counted_days(first, last))
        daily: Dict[date, float] = {}
        if "dailyrainin" in stitched and not stitched.empty:
            days = _local(stitched["dateutc"], coverage.tz).dt.date
            rain = pd.to_numeric(stitched["dailyrainin"], errors="coerce")
            daily = rain.groupby(days).max().to_dict()
        lengths: List[int] = []
        run: Optional[int] = None  # dry days since a rain day; None: unbounded
        day = first
        while day <= last:
            if day not in counted:
                run = None
            elif daily.get(day, 0.0) >= RAIN_DAY_IN:
                if run:
                    lengths.append(run)
                run = 0
            elif run is not None:
                run += 1
            day += timedelta(days=1)
        return lengths

    lengths = comparable(status, spells)
    return lengths, ((status.key,) if lengths is not None else ())


def _calm_rule(
    current: SnapshotConditions,
    observed_at_ms: Optional[int],
    usual: _Usual,
    compared: Tuple[str, ...],
    met: bool,
) -> LeadRule:
    """The calm rule: days since it last rained, against the usual spell."""
    value = None
    if current.last_rain_ms is not None and observed_at_ms is not None:
        value = round((observed_at_ms - current.last_rain_ms) / DAY_MS, 2)
    missing = None
    if value is None:
        missing = "no_reading"
    elif usual[1] is None:
        missing = "no_comparison"
    return LeadRule(
        kind="calm",
        met=met,
        value=value,
        unit="days",
        threshold=None,
        reference=usual[1],
        compared=compared if usual[1] is not None else (),
        missing=missing,
    )


# ---------- the section ----------


def hero_lead(
    archive: pd.DataFrame,
    stitched: pd.DataFrame,
    event_catalog: Optional[pd.DataFrame],
    active: Optional[ActiveRainEvent],
    current: SnapshotConditions,
    observed_at_ms: Optional[int],
    now: datetime,
    tz: str = DEFAULT_TZ,
) -> HeroLead:
    """
    The hero lead: every lead rule's value, the lead, and its chart data.

    :param archive: The archive as stored, for gap statuses.
    :param stitched: `stitch_live` output, newest first.
    :param event_catalog: The Event Catalog, or None.
    :param active: The active rain event, or None.
    :param current: Current conditions, live stitched.
    :param observed_at_ms: The newest observation, or None.
    :param now: The build time, timezone-aware.
    :param tz: The station-local zone.
    :return: The `HeroLead` section.
    """
    coverage = ArchiveCoverage(archive, now, tz)
    month_hours = _MonthHours(archive, coverage)
    temperature_by_hour = month_hours.by_hour("tempf", 1)
    observed_hour = (
        None
        if observed_at_ms is None
        else pd.Timestamp(observed_at_ms, unit="ms", tz="UTC").tz_convert(tz).hour
    )
    temperature_now = temperature_by_hour.get(observed_hour, _NO_USUAL)

    newest = _newest_event(event_catalog, observed_at_ms)
    spells, spell_window = _dry_spells(stitched, coverage)
    spell_usual = _quartiles(spells or [], 2)

    rules = [
        _rain_rule(newest, active, observed_at_ms),
        _temperature_rule(current, temperature_now, month_hours),
        _solar_rule(current),
    ]
    calm_met = not any(r.met for r in rules)
    rules.append(
        _calm_rule(current, observed_at_ms, spell_usual, spell_window, calm_met)
    )
    kind: LeadKind = next(r.kind for r in rules if r.met)

    if kind == "rain":
        now_value, usual, sample, trace = _rain_chart(
            stitched, event_catalog, newest, active, current, observed_at_ms
        )
        unit = "in"
    elif kind in ("temperature", "solar"):
        column, digits = (
            ("tempf", 1) if kind == "temperature" else ("solarradiation", 0)
        )
        by_hour = (
            temperature_by_hour
            if kind == "temperature"
            else (month_hours.by_hour(column, digits))
        )
        now_value = (
            current.temperature_f if kind == "temperature" else current.solar_wm2
        )
        usual = by_hour.get(observed_hour, _NO_USUAL)
        sample = month_hours.days()
        trace = _day_trace(stitched, column, by_hour, coverage.today, tz, digits)
        unit = "f" if kind == "temperature" else "wm2"
    else:
        now_value, usual, sample, trace = (
            rules[3].value,
            spell_usual,
            len(spells or []),
            None,
        )
        unit = "days"

    return HeroLead(
        kind=kind,
        rules=tuple(rules),
        unit=unit,
        now=now_value,
        now_ms=observed_at_ms,
        usual_low=usual[0],
        usual_median=usual[1],
        usual_high=usual[2],
        sample=sample,
        trace=trace,
    )
