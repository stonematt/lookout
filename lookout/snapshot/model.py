"""
model.py: The dashboard snapshot, the contract between Python and TypeScript.

`build_snapshot` returns a `Snapshot`; the refresh run publishes its two JSON
documents on Storj and the Vercel site reads them:

- `Snapshot.dashboard()` -> `snapshot/<MAC>.dashboard.json`, served by
  `/dashboard`: every field of the snapshot.
- `Snapshot.context()` -> `snapshot/<MAC>.context.json`, which `/current`
  serves next to the live `lastData`: the fields named in `CONTEXT_FIELDS`.

JSON shape rules, so both sides can rely on them:

- Every document carries `schema_version` (`SCHEMA_VERSION`) and `kind`
  ("dashboard", "context", or "event_curve" for a curve file).
- Times are integers, epoch milliseconds UTC, in fields ending `_ms`.
- Station-local calendar days (ADR-0002) are ISO dates, "YYYY-MM-DD", in
  fields ending `_day`.
- Units are imperial and named in the field (`_f`, `_in`, `_mph`, `_inhg`).
- A value that is not known is `null`, never NaN and never left out.
- Trends are words: "rising", "falling" or "steady".
- A value computed over a month or a window is null when that month's or
  window's `GapStatus` is "invalid": compute it through
  `lookout.snapshot.gaps.comparable`, which never runs an invalid one.

Adding a field is not a breaking change: add it to the dataclass and both
sides' goldens, keep `SCHEMA_VERSION`. Renaming, removing or changing the
meaning of a field is breaking: bump `SCHEMA_VERSION` in the same change.

Golden documents for the TypeScript tests live in `tests/golden/snapshots/`
(see `tests/test_snapshot.py` for how to regenerate them).
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Tuple

SCHEMA_VERSION = 1

# The dashboard fields the context block carries, in order.
CONTEXT_FIELDS: Tuple[str, ...] = (
    "station",
    "built_at_ms",
    "observed_at_ms",
    "current",
    "active_rain_event",
)


@dataclass(frozen=True)
class Station:
    """
    The station a snapshot describes.

    :param mac: Its MAC address, the key of every file it has on Storj.
    :param name: Its human-readable name in Ambient, or None when unnamed.
    """

    mac: str
    name: Optional[str]

    @classmethod
    def from_device(cls, device: Dict[str, Any]) -> "Station":
        """
        Read the station from an Ambient `/devices` entry.

        :param device: One entry, with `macAddress` and `info.name`.
        :return: The station.
        """
        return cls(
            mac=device["macAddress"],
            name=(device.get("info") or {}).get("name"),
        )


@dataclass(frozen=True)
class SnapshotConditions:
    """
    The newest reading, live where the live reading reports it.

    Each value is read once, from the newest row of the stitched history
    (`lookout.snapshot.current.stitch_live`), so every view shows the same
    number. None when the station did not report that quantity. Ambient
    names in brackets.

    :param temperature_f: Outdoor temperature (`tempf`).
    :param feels_like_f: Feels-like temperature (`feelsLike`).
    :param dew_point_f: Dew point (`dewPoint`).
    :param humidity_pct: Outdoor relative humidity (`humidity`).
    :param barometer_inhg: Relative barometric pressure (`baromrelin`).
    :param wind_speed_mph: Wind speed (`windspeedmph`).
    :param wind_gust_mph: Wind gust (`windgustmph`).
    :param wind_dir_deg: Wind direction in degrees (`winddir`).
    :param rain_rate_in_hr: Rain over the last hour (`hourlyrainin`).
    :param rain_today_in: Rain since local midnight (`dailyrainin`).
    :param rain_event_in: Rain in the current event (`eventrainin`).
    :param solar_wm2: Solar radiation (`solarradiation`).
    :param uv_index: UV index (`uv`).
    :param last_rain_ms: When it last rained (`lastRain`).
    :param temp_trend: Against 24 h before: "rising", "falling", "steady".
    :param barometer_trend: Against 3 h before: "rising", "falling", "steady".
    """

    temperature_f: Optional[float]
    feels_like_f: Optional[float]
    dew_point_f: Optional[float]
    humidity_pct: Optional[float]
    barometer_inhg: Optional[float]
    wind_speed_mph: Optional[float]
    wind_gust_mph: Optional[float]
    wind_dir_deg: Optional[float]
    rain_rate_in_hr: Optional[float]
    rain_today_in: Optional[float]
    rain_event_in: Optional[float]
    solar_wm2: Optional[float]
    uv_index: Optional[float]
    last_rain_ms: Optional[int]
    temp_trend: Optional[str]
    barometer_trend: Optional[str]


@dataclass(frozen=True)
class ActiveRainEvent:
    """
    The rain event still under way: ongoing in the Event Catalog and still
    reporting event rain live.

    :param event_id: Its Event Catalog id.
    :param start_ms: When it started.
    :param duration_minutes: From its start to the newest observation.
    :param total_in: Its rain so far: the live `eventrainin`.
    :param avg_rate_in_hr: `total_in` over its duration, None at zero length.
    """

    event_id: str
    start_ms: int
    duration_minutes: float
    total_in: float
    avg_rate_in_hr: Optional[float]


GapState = Literal["valid", "prorated", "invalid"]
GapReason = Literal[
    "gap", "low_coverage", "not_archived", "before_archive", "in_progress"
]


@dataclass(frozen=True)
class GapStatus:
    """
    One month's or one window's gap status: the One Gap Status Rule.

    Computed by `lookout.snapshot.gaps`, which holds the rules.

    :param key: Which one: "YYYY-MM" for a month, else the window's name
        ("today", "yesterday", "7d", "30d", "90d", "365d", "water_year").
    :param first_day: Its first station-local day.
    :param last_day: Its last station-local day, inclusive.
    :param days: How many days it spans.
    :param missing_days: Its days that cannot be counted: no archived rows,
        low coverage, after the archive ends, before it began, or not yet come.
    :param status: "valid"; "prorated" (shown with ≈: up to 10% of its days
        missing, a total scaled to the whole window); or "invalid" (no value
        is computed).
    :param reason: Why it is not valid: "gap", "low_coverage",
        "not_archived", "before_archive" or "in_progress"; None when valid.
    """

    key: str
    first_day: str
    last_day: str
    days: int
    missing_days: int
    status: GapState
    reason: Optional[GapReason]


@dataclass(frozen=True)
class Gaps:
    """
    The gap status of every month in the archive and every standard window.

    :param months: Each station-local month, oldest first, from the archive's
        first month through the current one.
    :param windows: The standard windows ending today, in the order `key`
        lists them. Views with windows of their own call
        `lookout.snapshot.gaps.window_status`.
    """

    months: Tuple[GapStatus, ...]
    windows: Tuple[GapStatus, ...]


@dataclass(frozen=True)
class ConditionRibbon:
    """
    The last 48 hours in 15-minute slots, for the Overview's first screen.

    Slot `i` starts at `start_ms + i * slot_minutes` minutes; the last slot
    holds the newest observation. The per-slot tuples are parallel; None
    means the slot has no reading for that value. A slot without a rain
    value is not known from the archive (a gap, or the span only the live
    reading covers) and is hatched. See `lookout.snapshot.ribbon`.

    :param start_ms: When the first slot starts.
    :param slot_minutes: Each slot's length (15).
    :param temperature_f: The newest outdoor temperature in each slot.
    :param rain_in: The rain that fell in each slot, from the daily counter.
    :param daylight: Whether the sun was up (any solar radiation) in each slot.
    :param temperature_low_f: The lowest reading over the 48 hours, live
        included.
    :param temperature_high_f: The highest reading over the 48 hours.
    :param rain_hours: Hours of slots with rain.
    :param known_hours: Hours of slots with a rain value.
    """

    start_ms: int
    slot_minutes: int
    temperature_f: Tuple[Optional[float], ...]
    rain_in: Tuple[Optional[float], ...]
    daylight: Tuple[Optional[bool], ...]
    temperature_low_f: Optional[float]
    temperature_high_f: Optional[float]
    rain_hours: float
    known_hours: float


@dataclass(frozen=True)
class RainEventRow:
    """
    One Event Catalog event, as the Rain Events tab lists it.

    An event is keyed by `start_ms`, not its catalog `event_id`: the catalog
    mints a new id for an ongoing event on every run, but its start (the
    first reading with event rain) never moves. The curve file and the
    page's `?event=` link both use the start.

    :param start_ms: When it started; its key.
    :param end_ms: Its last reading with event rain; the newest observation
        while it is live.
    :param duration_minutes: From start to end.
    :param total_in: Its rain: the live `eventrainin` while it is live.
    :param peak_rate_in_hr: Its highest rate between 5-minute readings.
    :param quality: The catalog's rating: "excellent", "good", "fair",
        "poor", or "invalid" when the archive has no data in its range.
    :param completeness_pct: Readings present against expected, 0-100.
    :param max_gap_minutes: Its longest gap over 10 minutes; 0 for none.
    :param water_year: The water year it started in (named by its end).
    :param rank: Its place by total among every listed event, 1 the wettest.
    :param ongoing: Still open in the Event Catalog.
    :param live: Ongoing and still reporting event rain live.
    :param has_curve: The archive holds readings in its range, so it has a
        curve; False shows totals only ("n/a: gap in record").
    """

    start_ms: int
    end_ms: int
    duration_minutes: float
    total_in: float
    peak_rate_in_hr: Optional[float]
    quality: str
    completeness_pct: Optional[float]
    max_gap_minutes: Optional[float]
    water_year: int
    rank: int
    ongoing: bool
    live: bool
    has_curve: bool


@dataclass(frozen=True)
class EventStats:
    """
    The summary metrics over a set of events (`summarize_events`).

    :param count: How many.
    :param total_in: Their rain added up.
    :param median_in: Their median total; None for no events.
    :param mean_duration_minutes: Their mean duration; None for no events.
    """

    count: int
    total_in: float
    median_in: Optional[float]
    mean_duration_minutes: Optional[float]


@dataclass(frozen=True)
class EventSeason:
    """
    The events that started in one water year, summarised.

    :param water_year: The water year, named by the year it ends in.
    :param stats: Its events' summary metrics.
    """

    water_year: int
    stats: EventStats


@dataclass(frozen=True)
class EventCurve:
    """
    One event's time series, for its accumulation and rate charts.

    Parallel arrays, one entry per archive reading in the event, oldest
    first. A reading that follows a gap of more than 10 minutes has its gap
    length in `gap_minutes` and no rate: the rate across a gap is not known.

    A completed event's curve is published once, as its own document
    (`EventCurve.document`, key from `lookout.snapshot.curves.curve_key`);
    the ongoing event's curve rides in the snapshot.

    :param start_ms: The event's start; its key.
    :param end_ms: Its end, or the newest reading while it is ongoing.
    :param t_ms: Each reading's time.
    :param event_in: The event rain so far at each reading (`eventrainin`).
    :param rate_in_hr: The rain rate since the previous reading, None after
        a gap.
    :param gap_minutes: The gap before the reading when over 10 minutes,
        else None.
    """

    start_ms: int
    end_ms: int
    t_ms: Tuple[int, ...]
    event_in: Tuple[Optional[float], ...]
    rate_in_hr: Tuple[Optional[float], ...]
    gap_minutes: Tuple[Optional[float], ...]

    def document(self, station: "Station") -> Dict[str, Any]:
        """
        The curve file: `snapshot/<MAC>.events/<start_ms>.json`.

        :param station: The station the event belongs to.
        :return: A JSON-ready dict, headed like the snapshot documents.
        """
        return _document("event_curve", {"station": asdict(station), **asdict(self)})


@dataclass(frozen=True)
class RainEvents:
    """
    The Rain Events tab: the Event Catalog's events and their summaries.

    :param events: Every event, newest first.
    :param all_time: The summary over every event.
    :param seasons: The summary per water year, newest first.
    :param ongoing_curve: The ongoing event's curve, live stitched; None
        when no event is ongoing.
    """

    events: Tuple[RainEventRow, ...]
    all_time: EventStats
    seasons: Tuple[EventSeason, ...]
    ongoing_curve: Optional[EventCurve]


@dataclass(frozen=True)
class Snapshot:
    """
    Everything one refresh run publishes for one station.

    :param station: The station.
    :param built_at_ms: When the snapshot was built (the injected `now`).
    :param observed_at_ms: The newest observation: the live reading, or the
        archive's newest row when that is newer or there is no live reading.
    :param archive_newest_ms: The archive's newest row, so the site can name
        the lag and mark the span only the live reading covers.
    :param current: Current conditions.
    :param active_rain_event: The active rain event, or None.
    :param gaps: The gap status of each month and standard window.
    :param condition_ribbon: The 48-hour condition ribbon, or None without
        an observation.
    :param rain_events: The Rain Events tab's list and summaries.
    """

    station: Station
    built_at_ms: int
    observed_at_ms: Optional[int]
    archive_newest_ms: Optional[int]
    current: SnapshotConditions
    active_rain_event: Optional[ActiveRainEvent]
    gaps: Gaps
    condition_ribbon: Optional[ConditionRibbon]
    rain_events: RainEvents

    def dashboard(self) -> Dict[str, Any]:
        """
        The dashboard document: every field.

        :return: A JSON-ready dict.
        """
        return _document("dashboard", asdict(self))

    def context(self) -> Dict[str, Any]:
        """
        The context block: the fields `/current` serves with `lastData`.

        :return: A JSON-ready dict.
        """
        fields = asdict(self)
        return _document("context", {name: fields[name] for name in CONTEXT_FIELDS})


def _document(kind: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Head a snapshot document with its version and kind, JSON-ready.

    :param kind: "dashboard" or "context".
    :param fields: The snapshot fields it carries.
    :return: The document.
    """
    return _plain({"schema_version": SCHEMA_VERSION, "kind": kind, **fields})


def _plain(value: Any) -> Any:
    """
    Make a value JSON-ready: NaN to None, numpy scalars to Python ones.

    :param value: A dict, list or scalar.
    :return: The same shape with only JSON types.
    """
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
