"""
wet_hours.py: The one wet-hour helper: each clock hour countable, and wet.

GLOSSARY: a **countable hour** is a station-local clock hour that has
readings and no gap over 30 minutes between them; the gap from the reading
before the hour to its first reading counts as part of the hour. A **wet
hour** is a countable hour with at least one tip of the gauge (0.01 in). An
hour that is not countable is neither wet nor dry: a reader draws it as
missing, never as dry (#647).

Each reading's rain is what it adds to the daily counter
(`lookout.core.rainfall_analysis.reading_rain`, on
`lookout.core.rain_counter.rain_increments`), so a counter reset late after
midnight, or at 21:00, never makes a false wet hour (#654). Only readings
that carry the counter count: an hour whose rows lack it cannot say whether
it rained. Hours are cut in the station zone (ADR-0002, ADR-0005), so a
day can hold 23 or 25 of them.

Every reader of wet hours takes them from here: the Overview's meter and
strip (`lookout.snapshot.wet_hours`), the wet-mode replay and the Rain tab.

Functions:
- judge_hours(df, tz, start_ms, end_ms): Each clock hour in a span, with
  whether it is countable and whether it is wet.
- daily_wet_hours(hours): Each station-local day's wet and countable hours.
- hour_start(ms, tz): The start of the station-local clock hour holding a
  time.
"""

import pandas as pd

from lookout.core.rain_counter import LEARN_MS
from lookout.core.rainfall_analysis import reading_rain
from lookout.utils.dateutc import DAY_MS, HOUR_MS, to_local

TIP_IN = 0.01
"""One tip of the gauge: an hour with this much rain is wet."""

MAX_GAP_MS = 30 * 60 * 1000
"""A countable hour has no gap between readings longer than this."""

LEAD_MS = LEARN_MS + DAY_MS
"""
Readings this long before a span are read too: the reading before its
first hour dates that hour's first gap, and the counter helper learns the
station's usual reset time from the 30 days before each gap.
"""

HOURS_COLUMNS = ["hour_ms", "day", "countable", "wet"]


def _local_ms(dateutc: pd.Series, tz: str) -> pd.Series:
    """Each time as station-local wall-clock epoch ms."""
    local = to_local(dateutc, tz).dt.tz_localize(None)
    return pd.Series(local.to_numpy(dtype="int64") // 10**6, index=dateutc.index)


def hour_start(ms: int, tz: str) -> int:
    """
    The start of the station-local clock hour that holds a time.

    :param ms: The time, epoch ms.
    :param tz: The station-local zone.
    :return: The hour's start, epoch ms.
    """
    local = int(_local_ms(pd.Series([ms]), tz).iloc[0])
    return ms - local % HOUR_MS


def judge_hours(df: pd.DataFrame, tz: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Each station-local clock hour in a span: countable, and wet.

    :param df: Readings with `dateutc` (epoch ms) and `dailyrainin`, any
        order; readings outside the span are ignored, but give the
        `LEAD_MS` before it so its first hour is judged like any other.
    :param tz: The station-local zone.
    :param start_ms: The span's start; the hour holding it is the first.
    :param end_ms: The span's end, exclusive: give an hour's start, so the
        last hour is a whole one.
    :return: `hour_ms` (the hour's start, epoch ms), `day` (its
        station-local date), `countable` and `wet`, one row per hour,
        oldest first. Empty when the span holds no hour.
    """
    first = hour_start(start_ms, tz)
    hour_ms = pd.Series(range(first, end_ms, HOUR_MS), dtype="int64")
    if hour_ms.empty:
        return pd.DataFrame(columns=HOURS_COLUMNS)
    judged = _judged_readings(df, tz, first, end_ms)
    hours = pd.DataFrame({"hour_ms": hour_ms}).merge(judged, on="hour_ms", how="left")
    countable = hours["countable"].astype("boolean").fillna(False).astype(bool)
    wet = countable & hours["wet"].astype("boolean").fillna(False).astype(bool)
    return pd.DataFrame(
        {
            "hour_ms": hour_ms,
            "day": to_local(hour_ms, tz).dt.date,
            "countable": countable,
            "wet": wet,
        }
    )


def _judged_readings(
    df: pd.DataFrame, tz: str, first_ms: int, end_ms: int
) -> pd.DataFrame:
    """
    The hours that hold a reading with the counter, judged.

    :param df: Readings with `dateutc` and `dailyrainin`.
    :param tz: The station-local zone.
    :param first_ms: The first hour's start.
    :param end_ms: The span's end, exclusive.
    :return: `hour_ms`, `countable` and `wet`, one row per hour with a
        reading; none for an hour without one.
    """
    empty = pd.DataFrame({"hour_ms": [], "countable": [], "wet": []})
    if df.empty or "dailyrainin" not in df:
        return empty
    times = df["dateutc"].astype("int64")
    readings = reading_rain(df[(times >= first_ms - LEAD_MS) & (times < end_ms)], tz)
    if readings.empty:
        return empty
    dateutc = readings["dateutc"].astype("int64")
    frame = pd.DataFrame(
        {
            "hour_ms": dateutc - _local_ms(dateutc, tz) % HOUR_MS,
            # Nothing read says when the reading before the first one was
            # (an archive's first, or the end of an outage longer than the
            # lead): its hour is never countable.
            "gap_ms": dateutc.diff().fillna(float("inf")),
            "rain_in": readings["rain_in"],
        }
    )
    frame = frame[frame["hour_ms"] >= first_ms]
    per_hour = frame.groupby("hour_ms", as_index=False).agg(
        gap_ms=("gap_ms", "max"), rain_in=("rain_in", "sum")
    )
    countable = per_hour["gap_ms"] <= MAX_GAP_MS
    return pd.DataFrame(
        {
            "hour_ms": per_hour["hour_ms"].astype("int64"),
            "countable": countable,
            "wet": countable & (per_hour["rain_in"].round(3) >= TIP_IN),
        }
    )


def daily_wet_hours(hours: pd.DataFrame) -> pd.DataFrame:
    """
    Each station-local day's wet hours and countable hours.

    :param hours: Judged hours, from `judge_hours`.
    :return: `day`, `wet_hours` and `countable_hours` (integers), one row
        per day the hours touch, oldest first. A day with no countable hour
        has 0 of each: the caller decides how to show it.
    """
    if hours.empty:
        return pd.DataFrame(columns=["day", "wet_hours", "countable_hours"])
    days = hours.groupby("day", as_index=False, sort=True).agg(
        wet_hours=("wet", "sum"), countable_hours=("countable", "sum")
    )
    return days.astype({"wet_hours": "int64", "countable_hours": "int64"})
