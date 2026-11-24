"""
rain_now.py: Last rain and the now story (ADR-0011).

**Last rain** is the newest reading the daily rain counter rose on: the
gauge's last tip. The rain each reading adds comes from the one counter
helper (`lookout.core.rain_counter.rain_increments`, through
`lookout.core.rainfall_analysis.reading_rain`), over the stitched history:
the archive's readings, rain corrected before the build, with the live
reading on top when it is newer, as every section reads them. It is never
Ambient's `lastRain`.

The now story is told while last rain is within `STORY_HOURS` of the build
time (`StationClock.now_ms`), the same window in which the Feature Rule
picks rain (`lookout.snapshot.feature`):

- **raining**: last rain within `RAINING_MINUTES`;
- **the rate**: the rain the readings in the `RATE_MINUTES` up to the build
  time add, so inches per hour;
- **the event**: its start and total, by Ambient's event bounds: the live
  event (`ActiveRainEvent`) when one is live, else the newest Event Catalog
  event that ended at or after last rain and started at or before it;
- **the chart** (#661): the `STORY_HOURS` up to the build time in
  `SLOT_MINUTES` slots of rain, from the same readings' rain, with the
  event's running total (`RainSlots`).

The thresholds are Lookout Post's (ADR-0011): unchecked on Salmon Run.

Functions:
- rain_now(stitched, event_catalog, active, clock): The `RainNow` for one
  snapshot.
- rain_leads(rain, clock): Whether last rain is within `STORY_HOURS`: the
  Feature Rule's rain clause.
"""

from typing import List, Optional, Tuple

import pandas as pd

from lookout.core.rainfall_analysis import reading_rain
from lookout.snapshot.clock import StationClock
from lookout.snapshot.current import to_ms
from lookout.snapshot.events import newest_event
from lookout.snapshot.model import ActiveRainEvent, RainNow, RainSlots
from lookout.snapshot.values import number

RAINING_MINUTES = 30
"""It is raining while last rain is at most this many minutes old."""

STORY_HOURS = 3
"""The now story is told, and rain leads, while last rain is at most this
many hours old."""

RATE_MINUTES = 60
"""The rate is the rain in this many minutes up to the build time."""

SLOT_MINUTES = 5
"""The story's chart counts rain in slots this many minutes long (#661)."""

_MINUTE_MS = 60_000


def _within(last_rain_ms: int, minutes: int, clock: StationClock) -> bool:
    """
    Whether last rain is at most `minutes` before the build time.

    :param last_rain_ms: Last rain.
    :param minutes: The window.
    :param clock: The build's clock.
    :return: True at exactly `minutes`, and for a tip after the build time.
    """
    return clock.now_ms - last_rain_ms <= minutes * _MINUTE_MS


def _story_open(last_rain_ms: Optional[int], clock: StationClock) -> bool:
    """
    Whether last rain is within `STORY_HOURS` of the build time.

    :param last_rain_ms: Last rain, or None.
    :param clock: The build's clock.
    :return: False without a last rain.
    """
    if last_rain_ms is None:
        return False
    return _within(last_rain_ms, STORY_HOURS * 60, clock)


def _event(
    event_catalog: Optional[pd.DataFrame],
    active: Optional[ActiveRainEvent],
    last_rain_ms: int,
    clock: StationClock,
) -> Tuple[Optional[int], Optional[float]]:
    """
    The current rain event's start and total, by Ambient's event bounds.

    :param event_catalog: The Event Catalog, or None.
    :param active: The live event, or None.
    :param last_rain_ms: Last rain.
    :param clock: The build's clock: an event ended by its time counts.
    :return: (start, total): the live event's, else those of the newest
        ended event that holds last rain; (None, None) when neither is.
    """
    if active is not None:
        return active.start_ms, active.total_in
    newest = newest_event(event_catalog, clock.now_ms)
    if newest is None:
        return None, None
    start_ms = to_ms(newest["start_time"])
    if not start_ms <= last_rain_ms <= to_ms(newest["end_time"]):
        return None, None
    return start_ms, number(newest["total_rainfall"])


def rain_leads(rain: RainNow, clock: StationClock) -> bool:
    """
    Whether last rain is within `STORY_HOURS` of the build time.

    The Feature Rule's rain clause (ADR-0011), and the window of the now
    story.

    :param rain: The snapshot's `RainNow`.
    :param clock: The build's clock.
    :return: False without a last rain.
    """
    return _story_open(rain.last_rain_ms, clock)


def rain_now(
    stitched: pd.DataFrame,
    event_catalog: Optional[pd.DataFrame],
    active: Optional[ActiveRainEvent],
    clock: StationClock,
) -> RainNow:
    """
    Last rain and, while it is within `STORY_HOURS`, the now story.

    :param stitched: `stitch_live` output.
    :param event_catalog: The Event Catalog, or None.
    :param active: The build's active rain event, or None.
    :param clock: The build's clock: the windows count to its `now`, and its
        zone tells the counter's resets.
    :return: The snapshot's `RainNow`.
    """
    readings = pd.DataFrame(columns=["dateutc", "rain_in"])
    if not stitched.empty and "dailyrainin" in stitched.columns:
        readings = reading_rain(stitched, clock.tz)
    tipped = readings.loc[readings["rain_in"] > 0, "dateutc"]
    last_rain_ms = int(tipped.max()) if not tipped.empty else None
    if last_rain_ms is None or not _story_open(last_rain_ms, clock):
        return RainNow(
            last_rain_ms=last_rain_ms,
            raining=False,
            rate_in_hr=None,
            event_start_ms=None,
            event_total_in=None,
        )
    since_ms = clock.now_ms - RATE_MINUTES * _MINUTE_MS
    window = (readings["dateutc"] > since_ms) & (readings["dateutc"] <= clock.now_ms)
    rate = number(readings.loc[window, "rain_in"].sum())
    start_ms, total_in = _event(event_catalog, active, last_rain_ms, clock)
    return RainNow(
        last_rain_ms=last_rain_ms,
        raining=_within(last_rain_ms, RAINING_MINUTES, clock),
        rate_in_hr=rate,
        event_start_ms=start_ms,
        event_total_in=total_in,
        last_3_hours=_slots(readings, start_ms, total_in, clock),
    )


def _slots(
    readings: pd.DataFrame,
    event_start_ms: Optional[int],
    event_total_in: Optional[float],
    clock: StationClock,
) -> RainSlots:
    """
    The story's chart: `STORY_HOURS` of rain in `SLOT_MINUTES` slots up to
    the build time, with the event's running total.

    The total runs back from the event's total, less the rain of the slots
    after each one, so the line ends on the number the story tells; a
    counter that read more than Ambient's total clamps at 0.

    :param readings: `reading_rain` output: each reading's rain.
    :param event_start_ms: The event's start, or None.
    :param event_total_in: The event's total, or None.
    :param clock: The build's clock.
    :return: The slots, oldest first.
    """
    slot_ms = SLOT_MINUTES * _MINUTE_MS
    count = STORY_HOURS * 60 // SLOT_MINUTES
    start_ms = clock.now_ms - count * slot_ms
    inside = (readings["dateutc"] > start_ms) & (readings["dateutc"] <= clock.now_ms)
    rows = readings.loc[inside]
    # A reading on a slot's end belongs to that slot: (start, end].
    index = ((rows["dateutc"] - start_ms - 1) // slot_ms).astype(int)
    sums = rows.groupby(index)["rain_in"].sum()
    rain: List[Optional[float]] = [
        number(sums[i]) if i in sums.index else None for i in range(count)
    ]
    totals: List[Optional[float]] = [None] * count
    if event_start_ms is not None and event_total_in is not None:
        after = 0.0
        for i in reversed(range(count)):
            if start_ms + (i + 1) * slot_ms < event_start_ms:
                break
            totals[i] = number(max(0.0, event_total_in - after))
            after += rain[i] or 0.0
    return RainSlots(
        start_ms=start_ms,
        slot_minutes=SLOT_MINUTES,
        rain_in=tuple(rain),
        total_in=tuple(totals),
    )
