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
Correcting a wrong value is not: a value that was wrongly a number and is
now null as unknown (an hour with no readings, #147), or an optional value
now null in a case the reader already handles (`RainEvents.open_curve`
while no event is live), keeps `SCHEMA_VERSION`. A bump also makes the site
refuse every curve file already written, and those are never rewritten.

Golden documents for the TypeScript tests live in `tests/golden/snapshots/`
(see `tests/test_snapshot.py` for how to regenerate them).
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Tuple

SCHEMA_VERSION = 2

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
    The rain event still under way: open in the Event Catalog and still
    reporting event rain live. This is the one "raining now" decision: the
    Rain Events tab marks the same event `live`.

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


LeadKind = Literal["rain", "temperature", "solar", "calm"]
LeadRuleUnit = Literal["hours", "degrees_f", "wm2", "days"]
LeadChartUnit = Literal["in", "f", "wm2", "days"]
LeadMissing = Literal["no_event", "no_reading", "no_comparison"]


@dataclass(frozen=True)
class LeadRule:
    """
    One rule of the Adaptive Lead Rule, with its live value.

    Computed by `lookout.snapshot.lead`, which holds the rules.

    :param kind: Which rule: "rain", "temperature", "solar" or "calm".
    :param met: Whether the rule holds. Calm holds only when no rule before
        it does: it is the "otherwise".
    :param value: Its live value, in `unit`: rain, the hours since the
        newest event ended (0 while one is live); temperature, the current
        temperature minus the month median (signed); solar, the current
        radiation; calm, the days since it last rained. None when unknown.
    :param unit: The unit of `value` and `threshold`.
    :param threshold: What the rule needs: rain, at most 12 hours; temperature,
        at least 10° either side; solar, at least 400 W/m²; calm, None.
    :param reference: What `value` was measured against: temperature, the
        month median at this hour (°F); calm, the usual dry spell (days);
        else None. None also when its comparison is invalid.
    :param compared: The `GapStatus` keys (in `Snapshot.gaps`) the reference
        was computed over, newest first: the earlier years' same month for
        temperature, the "365d" window for calm. Only comparable ones.
    :param missing: Why `value` or `reference` is None: "no_event",
        "no_reading" or "no_comparison"; None when nothing is missing.
    """

    kind: LeadKind
    met: bool
    value: Optional[float]
    unit: LeadRuleUnit
    threshold: Optional[float]
    reference: Optional[float]
    compared: Tuple[str, ...]
    missing: Optional[LeadMissing]


@dataclass(frozen=True)
class LeadTrace:
    """
    The hero chart's bold "now" line, in slots, with its usual band per slot.

    Slot `i` starts at `start_ms + i * slot_minutes` minutes. Temperature and
    solar: today's hourly means from local midnight (hours still to come are
    None), with the band of the same local hour in the compared months.
    Rain: the event's rain so far in 15-minute slots from its start, with no
    per-slot band (empty tuples): its usual range is the scalar one on
    `HeroLead`. A slot with no reading is None: an archive gap, or the span
    only the live reading covers.

    :param start_ms: When the first slot starts.
    :param slot_minutes: Each slot's length.
    :param values: The bold line's value in each slot.
    :param usual_low: The band's lower quartile in each slot, or ().
    :param usual_median: The band's median in each slot, or ().
    :param usual_high: The band's upper quartile in each slot, or ().
    """

    start_ms: int
    slot_minutes: int
    values: Tuple[Optional[float], ...]
    usual_low: Tuple[Optional[float], ...]
    usual_median: Tuple[Optional[float], ...]
    usual_high: Tuple[Optional[float], ...]


@dataclass(frozen=True)
class HeroLead:
    """
    The Overview's hero lead: the most notable thing right now, and why.

    :param kind: The rule that leads: the first in `rules` that is met.
    :param rules: Every rule in order (rain, temperature, solar, calm), for
        the "Why this leads" disclosure.
    :param unit: The chart's unit: "in" (rain), "f", "wm2" or "days" (calm).
    :param now: The bold "now" value: the event's rain so far, the current
        temperature or radiation, or the days since it last rained.
    :param now_ms: When `now` was observed.
    :param usual_low: The usual range's lower quartile: at this hour for
        temperature and solar, over completed events' totals for rain, over
        completed dry spells for calm. None without a comparison set.
    :param usual_median: Its median.
    :param usual_high: Its upper quartile.
    :param sample: How many the usual range is over: days (temperature,
        solar), events (rain) or dry spells (calm).
    :param trace: The bold line over time; None for calm.
    """

    kind: LeadKind
    rules: Tuple[LeadRule, ...]
    unit: LeadChartUnit
    now: Optional[float]
    now_ms: Optional[int]
    usual_low: Optional[float]
    usual_median: Optional[float]
    usual_high: Optional[float]
    sample: int
    trace: Optional[LeadTrace]


@dataclass(frozen=True)
class TemperaturePeriod:
    """
    The temperature across one window: the Overview's range bars.

    Computed from the live-stitched readings over the window's counted days,
    only when its gap status allows (the five values are null for an invalid
    window; a prorated one is shown with a tilde, its readings those of the
    counted days).

    :param status: The window's gap status; `status.key` names it ("today",
        "yesterday", "7d", "30d", "90d", "365d").
    :param low: The lowest reading.
    :param q1: The 25th percentile of readings.
    :param q3: The 75th percentile of readings.
    :param high: The highest reading.
    :param readings: How many readings the values are over.
    :param position: Where the current temperature sits against the middle
        half: "above", "below" or "inside"; null when the window or the
        reading is unknown.
    """

    status: GapStatus
    low: Optional[float]
    q1: Optional[float]
    q3: Optional[float]
    high: Optional[float]
    readings: int
    position: Optional[Literal["above", "below", "inside"]]


@dataclass(frozen=True)
class TemperatureDay:
    """
    One day of the ten-day high-low list, against normal.

    Normal for a day is the median daily low and median daily high over the
    days within 7 either side of this date in earlier years, each year used
    only when that 15-day window's gap status allows.

    :param day: The station-local day, "YYYY-MM-DD".
    :param status: The day's own gap status; its low and high are null when
        invalid.
    :param low: The day's lowest reading (so far, for today).
    :param high: The day's highest reading.
    :param usual_low: The median daily low in earlier years; null with none.
    :param usual_high: The median daily high.
    :param usual_years: The earlier years it is over, newest first.
    :param usual_days: How many days it is over.
    """

    day: str
    status: GapStatus
    low: Optional[float]
    high: Optional[float]
    usual_low: Optional[float]
    usual_high: Optional[float]
    usual_years: Tuple[int, ...]
    usual_days: int


@dataclass(frozen=True)
class NormalStrip:
    """
    One "now against normal" strip: the current value among past days.

    The past days are those within 7 either side of today's date in earlier
    years, each year used only when its 15-day window's gap status allows.
    A temperature dot is that day's mean at this local hour; a rain dot is
    that day's rain by this time of day, so "so far" compares like with like.

    :param kind: "temperature" or "rain".
    :param unit: "f" or "in".
    :param now: The current value (temperature, rain so far today).
    :param now_ms: When it was observed.
    :param values: One value per past day, ascending.
    :param below_pct: The whole percent of `values` strictly below `now`.
    :param equal_pct: The whole percent equal to `now` (rain ties at zero).
    :param above_pct: The whole percent strictly above `now`.
    :param years: The earlier years used, newest first.
    :param missing: "no_reading" or "no_comparison" when the strip cannot
        be drawn; else null.
    """

    kind: Literal["temperature", "rain"]
    unit: Literal["f", "in"]
    now: Optional[float]
    now_ms: Optional[int]
    values: Tuple[float, ...]
    below_pct: Optional[int]
    equal_pct: Optional[int]
    above_pct: Optional[int]
    years: Tuple[int, ...]
    missing: Optional[Literal["no_reading", "no_comparison"]]


@dataclass(frozen=True)
class TemperatureContext:
    """
    The Overview's temperature context (#81).

    :param now: The current temperature (the range bars' now rule).
    :param now_ms: When it was observed.
    :param finding: The key of the window the headline places `now` in: the
        first of 30d, 7d, 90d, 365d, yesterday that can be compared; null
        when none can.
    :param periods: Today, yesterday, 7, 30, 90 and 365 days, in that order.
    :param ten_days: The last ten local days, oldest first, ending today.
    :param ten_day_years: Every earlier year the list's normal is over,
        newest first.
    :param strips: The temperature strip, then the rain strip.
    """

    now: Optional[float]
    now_ms: Optional[int]
    finding: Optional[str]
    periods: Tuple[TemperaturePeriod, ...]
    ten_days: Tuple[TemperatureDay, ...]
    ten_day_years: Tuple[int, ...]
    strips: Tuple[NormalStrip, ...]


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
    :param live: It is the snapshot's `active_rain_event`: the one "it is
        raining now" decision, which the hero reads too (open in the Event
        Catalog and the live reading still reporting event rain).
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
    live: bool
    has_curve: bool


@dataclass(frozen=True)
class EventStats:
    """
    The summary metrics over a set of events (`summarize_events`).

    :param count: How many.
    :param total_in: Their totals added up: rain in events only, not the
        rain over a window. Rain outside any cataloged event (a shower under
        the catalog's minimum) is not in it, and an event counts whole in
        the water year it started in, so it can differ from
        `RainTab.water_years`, the water year's rain. It is not gap-judged:
        it sums the events listed, and a gap's missed events are missing
        from the list too.
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

    `stats` is an events-only sum (see `EventStats.total_in`): the listed
    events added up, never a window total, so it is not gap-judged and not
    made null. `status` beside it says whether the record over that water
    year is whole, so the sum is never read as the year's rain without it
    (#147).

    :param water_year: The water year, named by the year it ends in.
    :param stats: Its events' summary metrics.
    :param status: The water year's rain gap status, the same as its
        `RainTab.water_years` curve's (Oct 1 to Sep 30, or to today); None
        when the archive does not reach that water year.
    """

    water_year: int
    stats: EventStats
    status: Optional[GapStatus]


@dataclass(frozen=True)
class EventCurve:
    """
    One event's time series, for its accumulation and rate charts.

    Parallel arrays, one entry per archive reading in the event, oldest
    first. A reading that follows a gap of more than 10 minutes has its gap
    length in `gap_minutes` and no rate: the rate across a gap is not known.

    A completed event's curve is published once, as its own document
    (`EventCurve.document`, key from `lookout.snapshot.curves.curve_key`);
    the curve of the live event (`Snapshot.active_rain_event`) rides in the
    snapshot (`RainEvents.open_curve`).

    :param start_ms: The event's start; its key.
    :param end_ms: Its end, or the newest reading while it is open.
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
    :param open_curve: The curve of the live event, the snapshot's
        `active_rain_event` (open in the Event Catalog and still reporting
        event rain), live stitched; its curve file is written only once the
        catalog closes it. None when no event is live, even while the catalog
        still holds one open (#147): that event's file follows its close.
    """

    events: Tuple[RainEventRow, ...]
    all_time: EventStats
    seasons: Tuple[EventSeason, ...]
    open_curve: Optional[EventCurve]


@dataclass(frozen=True)
class RainWindow:
    """
    The rain over one standard window: an archive sum with today's live
    total in it (`lookout.snapshot.rain_tab`).

    :param key: The window, as `Gaps.windows` names it ("today", "7d",
        "water_year", ...).
    :param status: Its rain gap status (`rain_tab.RainCoverage`): the
        `Gaps.windows` status with the same key, except that a day whose rain
        counter has no reading is missing here, so its rain is never read as
        0. Show the total by this status, not the archive's.
    :param total_in: Its rain; prorated when its status is; None when invalid.
    """

    key: str
    status: GapStatus
    total_in: Optional[float]


@dataclass(frozen=True)
class RainHour:
    """
    One station-local hour's rain.

    :param day: The day.
    :param hour: The hour it starts, 0 to 23 local.
    :param rain_in: The rain that fell in it.
    """

    day: str
    hour: int
    rain_in: float


@dataclass(frozen=True)
class RainDay:
    """
    One station-local day's rain.

    :param day: The day.
    :param rain_in: Its total.
    """

    day: str
    rain_in: float


@dataclass(frozen=True)
class RecentRain:
    """
    Daily and hourly rain over the days of one standard window, for the
    Rain tab's daily bars and its accumulation heatmap.

    Per-day tuples are parallel, oldest first, from `first_day`. A day that
    cannot be counted (its one-day gap status is invalid) has a None total,
    its reason, and 24 None hours. Today's total is the live one.

    :param window: The standard window the days are ("90d"); its gap status
        and its total are the ones with that key.
    :param first_day: The first day.
    :param rain_in: Each day's total.
    :param reason: Why a day cannot be counted, None when it can.
    :param hourly_in: Each day's 24 local hours: the rain in each, None when
        it is not known: an hour still to come today, or one with no rain
        counter reading in the archive (#147; never 0). A day's known hours
        add up to its total, to rounding; rain counted during a missing hour
        lands in the next known one.
    :param wet_hours: Known hours with any rain over the window; prorated
        when the window is; None when it is invalid.
    :param wettest_hour: The hour with the most rain; None when the window is
        invalid or dry.
    """

    window: str
    first_day: str
    rain_in: Tuple[Optional[float], ...]
    reason: Tuple[Optional[GapReason], ...]
    hourly_in: Tuple[Tuple[Optional[float], ...], ...]
    wet_hours: Optional[float]
    wettest_hour: Optional[RainHour]


@dataclass(frozen=True)
class DrySpell:
    """
    A run of dry days with a rain day on each side.

    :param first_day: Its first dry day.
    :param last_day: Its last dry day.
    :param days: How many days it ran.
    """

    first_day: str
    last_day: str
    days: int


@dataclass(frozen=True)
class DrySpells:
    """
    Dry spells and rain days in the record, for the Rain tab.

    A rain day has at least 0.01 in. Spells with a day not in the record are
    left out, so a missing day is never read as dry. The days since it last
    rained and the usual dry spell are the calm lead rule's
    (`HeroLead.rules`), so the Overview and the Rain tab show one number.

    :param longest: The longest completed dry spell in the record.
    :param water_year_rain_days: Rain days this water year; prorated when
        its status is; None when invalid.
    :param water_year_wettest: The wettest day this water year; None when
        the water year is invalid or dry.
    """

    longest: Optional[DrySpell]
    water_year_rain_days: Optional[float]
    water_year_wettest: Optional[RainDay]


@dataclass(frozen=True)
class WaterYearCurve:
    """
    One water year's cumulative rain, day by day from Oct 1.

    Place `i` is the day `i` days after Oct 1, with Feb 29 folded into
    Feb 28 (`lookout.core.rainfall_analysis.water_year_day_index`), so every
    year has the same 365 places. Each value is the window from Oct 1 to
    that day, through the gap rules: prorated when that window is, None when
    it is invalid.

    :param water_year: Named by the year it ends in.
    :param first_day: Its Oct 1.
    :param cumulative_in: The rain from Oct 1 through each day so far.
    :param prorated: Whether each value is prorated (≈).
    :param status: The gap status of the whole year so far.
    :param total_in: The rain over the whole year so far: the last value.
    :param to_date_in: The value at `RainTab.day_index`, the place of today,
        so years compare to the same date; None when it is invalid or the
        year has not reached it.
    """

    water_year: int
    first_day: str
    cumulative_in: Tuple[Optional[float], ...]
    prorated: Tuple[bool, ...]
    status: GapStatus
    total_in: Optional[float]
    to_date_in: Optional[float]


@dataclass(frozen=True)
class RainTab:
    """
    Everything the Rain tab shows (`lookout.snapshot.rain_tab`). Every rain
    total is an archive sum with today's live total stitched in.

    :param windows: The rain over each standard window, in `Gaps.windows`
        order.
    :param recent: Daily and hourly rain over the last 90 days.
    :param dry_spells: The current dry spell against the record.
    :param day_index: Today's place in the water-year curves.
    :param water_years: Every water year the archive reaches, oldest first.
    """

    windows: Tuple[RainWindow, ...]
    recent: RecentRain
    dry_spells: DrySpells
    day_index: int
    water_years: Tuple[WaterYearCurve, ...]


@dataclass(frozen=True)
class AlmanacMonth:
    """
    One calendar month's rain in the almanac grid (`lookout.snapshot.almanac`).

    :param status: The month's gap status, keyed "YYYY-MM"; the current month
        is invalid, "in_progress", until its last day has ended.
    :param rain_in: Its rain; prorated when its status is; None when invalid.
    :param to_date: For the current month only, the window from its first
        day to today; None for every other month.
    :param to_date_in: The rain over `to_date`, live today included; None
        when that window is invalid or there is none.
    """

    status: GapStatus
    rain_in: Optional[float]
    to_date: Optional[GapStatus]
    to_date_in: Optional[float]


@dataclass(frozen=True)
class AlmanacYear:
    """
    One water year's row of the almanac grid: Oct through Sep.

    :param water_year: Named by the year it ends in (GLOSSARY.md).
    :param months: Twelve months, Oct first; None for a month not begun.
    :param season: The whole water year, Oct 1 to Sep 30, keyed
        "water_year_<year>"; invalid, "in_progress", while it runs.
    :param season_in: Its rain; None when invalid. The current season's
        rain so far is `Almanac.rolling`'s "water_year" window.
    """

    water_year: int
    months: Tuple[Optional[AlmanacMonth], ...]
    season: GapStatus
    season_in: Optional[float]


@dataclass(frozen=True)
class RollingYear:
    """
    One year's rain over a rolling window ending on today's date that year.

    :param year: The calendar year the window ends in: the table's column.
    :param status: The window's rain gap status. The current year's is the
        `RainTab.windows` status with the same key.
    :param rain_in: Its rain; prorated when its status is; None when invalid.
    :param water_year: The water year the window ends in; the season row
        ("water_year", since Oct 1) is named by it, not by `year`, which is
        one behind from Oct through Dec (#147).
    """

    year: int
    status: GapStatus
    rain_in: Optional[float]
    water_year: int


@dataclass(frozen=True)
class RollingWindow:
    """
    One row of the rolling-window rain table.

    :param key: "7d", "30d", "90d", "365d" or "water_year" (since Oct 1).
    :param years: The window ending today, then the same window ending on
        the same date in each earlier year the archive reaches, newest first.
    """

    key: str
    years: Tuple[RollingYear, ...]


@dataclass(frozen=True)
class Almanac:
    """
    The Overview's almanac grid and rolling-window rain table (#82). Every
    total is an archive sum with today's live total stitched in.

    :param years: Water-year rows, newest first.
    :param rolling: The rolling windows, shortest first, the season last.
    """

    years: Tuple[AlmanacYear, ...]
    rolling: Tuple[RollingWindow, ...]


@dataclass(frozen=True)
class SolarWindow:
    """
    The solar energy over one standard window.

    :param key: The window's key in `Gaps.windows`.
    :param energy_kwh_m2: kWh per m² from the energy catalog, prorated when
        the window is; None when it is invalid.
    :param status: Its gap status for solar: the archive's, with any day the
        energy catalog lacks judged missing too (#123).
    """

    key: str
    energy_kwh_m2: Optional[float]
    status: GapStatus


@dataclass(frozen=True)
class SolarDays:
    """
    Each day's solar energy, from the archive's first day through today.

    Tuples are parallel, oldest first, from `first_day`. A day that cannot be
    counted has a None total and its reason; today's is the energy so far.

    :param first_day: The archive's first day.
    :param energy_kwh_m2: Each day's kWh per m².
    :param reason: Why a day cannot be counted, None when it can.
    """

    first_day: str
    energy_kwh_m2: Tuple[Optional[float], ...]
    reason: Tuple[Optional[GapReason], ...]


@dataclass(frozen=True)
class SolarMonth:
    """
    One month's solar energy.

    :param key: "YYYY-MM", its key in `Gaps.months`.
    :param energy_kwh_m2: kWh per m², prorated when the month is; None when
        it is invalid (the month under way is).
    :param status: Its gap status for solar, as `SolarWindow.status`.
    """

    key: str
    energy_kwh_m2: Optional[float]
    status: GapStatus


@dataclass(frozen=True)
class SolarRecent:
    """
    Each recent day's energy, 15 minutes at a time, for picking a day.

    :param window: The standard window the days are ("90d").
    :param first_day: The first day.
    :param slot_minutes: The slot length, 15.
    :param energy_wh_m2: Each day's slots by local clock, midnight first
        (96: on a daylight-saving day the missing hour is None and the
        repeated one is summed). A day that cannot be counted, a slot the
        catalog has no period for, and a slot still to come are None.
    """

    window: str
    first_day: str
    slot_minutes: int
    energy_wh_m2: Tuple[Tuple[Optional[float], ...], ...]


@dataclass(frozen=True)
class SolarTrace:
    """
    The 24-hour view: the solar trace up to the newest observation, over
    the usual range for each time of day.

    Slot `i` starts `i * step_minutes` after `start_ms`; the last slot is
    the newest observation's, so it holds the live reading. A slot with no
    reading (a gap, or the span only the live reading covers) is None. The
    usual tuples are parallel to `solar_wm2`.

    :param start_ms: The first slot.
    :param step_minutes: The slot length, 5 (a duration, so not an `_ms`
        field: those are epoch times).
    :param solar_wm2: Each slot's solar radiation.
    :param usual: The gap status of the comparison days: the 30 before today.
    :param usual_low_wm2: The 25th percentile at each slot's time of day over
        the comparison days; None when `usual` is invalid.
    :param usual_median_wm2: The median, likewise.
    :param usual_high_wm2: The 75th percentile, likewise.
    :param energy: The gap status of yesterday and today, the days the 24
        hours reach.
    :param energy_kwh_m2: The energy over the 24 hours, from the catalog;
        None when `energy` is invalid.
    :param previous_kwh_m2: The energy over the 24 hours before; None when
        the days they reach cannot all be counted.
    """

    start_ms: int
    step_minutes: int
    solar_wm2: Tuple[Optional[float], ...]
    usual: GapStatus
    usual_low_wm2: Optional[Tuple[Optional[float], ...]]
    usual_median_wm2: Optional[Tuple[Optional[float], ...]]
    usual_high_wm2: Optional[Tuple[Optional[float], ...]]
    energy: GapStatus
    energy_kwh_m2: Optional[float]
    previous_kwh_m2: Optional[float]


@dataclass(frozen=True)
class SolarTab:
    """
    Everything the Solar tab shows (`lookout.snapshot.solar`), from the
    energy catalog and, for the 24-hour view, the live-stitched history.

    :param windows: The energy over each standard window but the water year.
    :param days: Every day's energy, for the daily bars and day heatmaps.
    :param months: Every month's energy, for the year-by-month heatmap.
    :param recent: The last 90 days, 15 minutes at a time.
    :param last_24h: The 24-hour view, or None without an observation.
    """

    windows: Tuple[SolarWindow, ...]
    days: SolarDays
    months: Tuple[SolarMonth, ...]
    recent: SolarRecent
    last_24h: Optional[SolarTrace]


CatalogName = Literal["events", "energy"]


@dataclass(frozen=True)
class StaleCatalog:
    """
    A catalog this run could not bring up to date (#118).

    The run caught the catalog's update error, logged it and built the
    snapshot from the last good catalog in storage; the site shows that
    catalog's figures as old.

    :param catalog: "events" (the Event Catalog, the Rain Events tab) or
        "energy" (the energy catalog, the Solar tab).
    :param as_of_ms: The newest time the last good catalog covers: its
        latest event end or energy period end. None when none is stored.
    """

    catalog: CatalogName
    as_of_ms: Optional[int]


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
    :param hero_lead: The hero lead and every lead rule's live value.
    :param rain_tab: The Rain tab's data.
    :param temperature_context: The Overview's temperature across periods,
        ten-day list and "now against normal" strips.
    :param almanac: The Overview's almanac grid and rolling-window tables.
    :param solar_tab: The Solar tab's data, or None without an energy
        catalog.
    :param stale_catalogs: The catalogs this run could not update and built
        from as last stored; empty on a normal run. `build_snapshot` never
        sets it: the refresh run does, after a caught catalog failure.
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
    hero_lead: HeroLead
    rain_tab: RainTab
    temperature_context: TemperatureContext
    almanac: Almanac
    solar_tab: Optional[SolarTab]
    stale_catalogs: Tuple[StaleCatalog, ...] = ()

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
