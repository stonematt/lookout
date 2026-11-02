"""
current.py: Current conditions for the snapshot, live stitched over archive.

The live `lastData` is usually ahead of the archive. `stitch_live` puts it on
top of the archive as the one newest row, once; current conditions, trends and
the active rain event all read from that row, so every view shows the same
number for the same quantity.

Functions:
- stitch_live(archive, live): The archive with the live reading as newest row.
- current_conditions(stitched, clock): The newest row's values and the trends.
- active_rain_event(event_catalog, current, observed_at_ms): The live event.
"""

from typing import Any, Callable, Dict, Optional

import pandas as pd

from lookout.core.weather_events import find_active_rain_event
from lookout.snapshot.clock import StationClock
from lookout.snapshot.model import ActiveRainEvent, SnapshotConditions
from lookout.utils.dateutc import local_time
from lookout.utils.trend_utils import (
    calculate_barometer_trend,
    calculate_temperature_trend,
)

# Snapshot field -> Ambient `lastData` / archive column.
LIVE_FIELDS: Dict[str, str] = {
    "temperature_f": "tempf",
    "feels_like_f": "feelsLike",
    "dew_point_f": "dewPoint",
    "humidity_pct": "humidity",
    "barometer_inhg": "baromrelin",
    "wind_speed_mph": "windspeedmph",
    "wind_gust_mph": "windgustmph",
    "wind_dir_deg": "winddir",
    "rain_rate_in_hr": "hourlyrainin",
    "rain_today_in": "dailyrainin",
    "rain_event_in": "eventrainin",
    "solar_wm2": "solarradiation",
    "uv_index": "uv",
}

# The trend utilities' arrows, as the snapshot's words.
TREND_WORDS = {"↑": "rising", "↓": "falling", "→": "steady"}

# Live keys that hold times, not readings; stitched as tz-aware timestamps.
_TIME_KEYS = ("date", "lastRain")


def to_ms(value: Any) -> Optional[int]:
    """
    Read a time as epoch milliseconds UTC.

    :param value: Epoch ms, an ISO-8601 string or a Timestamp; naive is UTC.
    :return: Epoch ms, or None when the value is missing.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.value // 10**6)


def _live_row(live: Dict[str, Any], archive: pd.DataFrame) -> Dict[str, Any]:
    """
    Shape a live reading as one archive row.

    Readings keep their Ambient names; times become timestamps in the
    archive's timezone; `dateutc` is epoch ms. Keys the archive does not have
    are dropped when there is an archive.

    :param live: The live `lastData`.
    :param archive: The archive it is stitched onto.
    :return: The row as a dict.
    """
    row: Dict[str, Any] = {"dateutc": to_ms(live["dateutc"])}
    for key, value in live.items():
        if key == "dateutc" or value is None:
            continue
        if not archive.empty and key not in archive.columns:
            continue
        if key in _TIME_KEYS:
            ms = to_ms(value)
            if ms is None:
                continue
            ts = pd.Timestamp(ms, unit="ms", tz="UTC")
            column = archive[key] if key in archive.columns else None
            tz = getattr(getattr(column, "dt", None), "tz", None)
            row[key] = ts.tz_convert(tz) if tz is not None else ts
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            row[key] = value
    return row


def stitch_live(archive: pd.DataFrame, live: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    The archive, newest first, with the live reading as its newest row.

    The live reading is added only when it is newer than every archived row,
    so a reading the archive already holds is never counted twice.

    :param archive: The archive, any order and index.
    :param live: The live `lastData`, or None.
    :return: A new frame, newest first, index 0..n-1.
    """
    ordered = archive
    if not archive.empty:
        ordered = archive.sort_values("dateutc", ascending=False)
    ordered = ordered.reset_index(drop=True)

    if not live or live.get("dateutc") is None:
        return ordered
    live_ms = to_ms(live["dateutc"])
    if not ordered.empty and live_ms <= int(ordered["dateutc"].iloc[0]):
        return ordered

    top = pd.DataFrame([_live_row(live, ordered)])
    if ordered.empty:
        return top
    return pd.concat([top, ordered], ignore_index=True)


def _value(row: pd.Series, column: str) -> Optional[float]:
    """
    One reading from a row, None when absent or NaN.

    :param row: The stitched newest row.
    :param column: The Ambient name.
    :return: The value as a float, or None.
    """
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _trend(
    calculate: Callable[[pd.DataFrame, float], str],
    stitched: pd.DataFrame,
    current: Optional[float],
) -> Optional[str]:
    """
    One trend as a word, None when there is no current value to compare.

    :param calculate: The trend utility (temperature or barometer).
    :param stitched: The stitched history, newest first.
    :param current: The value the trend is about.
    :return: "rising", "falling", "steady" or None.
    """
    if current is None:
        return None
    return TREND_WORDS[calculate(stitched, current)]


def current_conditions(
    stitched: pd.DataFrame, clock: StationClock
) -> SnapshotConditions:
    """
    Current conditions from the stitched history's newest row.

    Trends compare that row with the history before it: temperature with
    24 hours earlier, the barometer with 3 hours earlier.

    Rain today is the newest row's daily counter only when that row is on
    today's local date: just after midnight, with no live reading newer than
    yesterday's last row, that counter is yesterday's total, so rain today
    is None until a reading from today arrives.

    :param stitched: `stitch_live` output; may be empty.
    :param clock: The station clock: it says which day is today.
    :return: The current conditions; every value None on an empty history.
    """
    newest = stitched.iloc[0] if not stitched.empty else pd.Series(dtype=object)
    values = {field: _value(newest, column) for field, column in LIVE_FIELDS.items()}
    if not _is_today(newest.get("dateutc"), clock):
        values["rain_today_in"] = None

    return SnapshotConditions(
        **values,
        last_rain_ms=to_ms(newest.get("lastRain")),
        temp_trend=_trend(
            calculate_temperature_trend, stitched, values["temperature_f"]
        ),
        barometer_trend=_trend(
            calculate_barometer_trend, stitched, values["barometer_inhg"]
        ),
    )


def _is_today(observed: Any, clock: StationClock) -> bool:
    """
    Whether a reading was taken on the build time's local date.

    :param observed: The reading's `dateutc` (ms), or None.
    :param clock: The station clock.
    :return: True when both fall on the same station-local day.
    """
    if observed is None or pd.isna(observed):
        return False
    return local_time(observed, clock.tz).date() == clock.today


def active_rain_event(
    event_catalog: Optional[pd.DataFrame],
    current: SnapshotConditions,
    observed_at_ms: Optional[int],
) -> Optional[ActiveRainEvent]:
    """
    The rain event still under way, with its live total.

    The Event Catalog says which event is open; the live event rain gates
    it and is its total, since the live reading replaces the archived one.
    This is the one "raining now" decision: the hero and the Rain Events
    tab's `live` row both read it.

    :param event_catalog: The Event Catalog, passed in.
    :param current: Current conditions, live stitched.
    :param observed_at_ms: The newest observation, the event's "so far".
    :return: The active event, or None.
    """
    event = find_active_rain_event(event_catalog, current.rain_event_in)
    if event is None or observed_at_ms is None:
        return None

    start_ms = to_ms(event["start_time"])
    duration_minutes = (observed_at_ms - start_ms) / 60_000
    total_in = current.rain_event_in
    rate = total_in / (duration_minutes / 60) if duration_minutes > 0 else None
    return ActiveRainEvent(
        event_id=str(event["event_id"]),
        start_ms=start_ms,
        duration_minutes=duration_minutes,
        total_in=total_in,
        avg_rate_in_hr=rate,
    )
