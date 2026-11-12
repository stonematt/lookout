"""
trace.py: A reading's last 24 hours as hourly means, for a tile's mini chart.

One builder serves the indoor sensors' traces (#466) and the outdoor
pressure and humidity series (#554), so every 24-hour line in the snapshot
sits on the same slots: `TRACE_HOURS` hourly slots, the last the hour that
holds the history's newest reading. The window ends at that reading, not the
build clock, so the line ends where the tile's number was read.

Each slot is the mean of the readings in it; a slot with no reading is None,
so a gap stays a gap (story 31). Values are cast on read: the live archives
store some columns as int64 and some as double (#466 column read).

Functions:
- trace_start(newest_ms): The first slot's start.
- numbers(frame, column): One column's readings as floats, by `dateutc`.
- hourly_means(readings, start_ms, digits): The slots' rounded means.
- outdoor_trace(stitched): The outdoor pressure and humidity series.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from lookout.snapshot.model import OutdoorTrace
from lookout.snapshot.values import number
from lookout.utils.dateutc import HOUR_MS

TRACE_HOURS = 24
"""Slots in a trace: one per hour."""

SLOT_MINUTES = HOUR_MS // 60_000

PRESSURE_COLUMN = "baromrelin"
HUMIDITY_COLUMN = "humidity"
PRESSURE_DIGITS = 2
HUMIDITY_DIGITS = 0


def trace_start(newest_ms: int) -> int:
    """
    The first slot's start: `TRACE_HOURS - 1` hours before the newest's hour.

    :param newest_ms: The history's newest reading.
    :return: The start, on the hour.
    """
    return (newest_ms // HOUR_MS - (TRACE_HOURS - 1)) * HOUR_MS


def numbers(frame: pd.DataFrame, column: str) -> pd.Series:
    """
    One column's readings as floats, unreadable and null dropped.

    :param frame: Rows indexed by `dateutc`, in any order.
    :param column: The Ambient column.
    :return: The readings, in the frame's order.
    """
    values = pd.to_numeric(frame[column].astype(object), errors="coerce")
    return values.astype("float64").dropna()


def hourly_means(
    readings: pd.Series, start_ms: int, digits: int
) -> Tuple[Optional[float], ...]:
    """
    The mean reading in each of `TRACE_HOURS` hourly slots from `start_ms`.

    :param readings: The readings, indexed by `dateutc`; none past the last
        slot.
    :param start_ms: The first slot's start, on the hour.
    :param digits: Decimal places to keep.
    :return: One value per slot; None for a slot with no reading.
    """
    inside = readings[readings.index >= start_ms]
    slot = (np.asarray(inside.index, dtype="int64") - start_ms) // HOUR_MS
    means = inside.groupby(slot).mean()
    return tuple(number(means.get(i), digits) for i in range(TRACE_HOURS))


def outdoor_trace(stitched: pd.DataFrame) -> Optional[OutdoorTrace]:
    """
    The outdoor pressure and humidity over the 24 hours to the newest reading.

    :param stitched: The live-stitched history (`dateutc` in ms), any order.
    :return: Both series on one set of slots; a series whose column the
        station lacks is all None. None when the history is empty.
    """
    if stitched.empty or "dateutc" not in stitched:
        return None
    rows = stitched.set_index(stitched["dateutc"].astype("int64"), drop=False)
    start_ms = trace_start(int(rows.index.max()))

    def series(column: str, digits: int) -> Tuple[Optional[float], ...]:
        if column not in rows:
            return (None,) * TRACE_HOURS
        return hourly_means(numbers(rows, column), start_ms, digits)

    return OutdoorTrace(
        start_ms=start_ms,
        slot_minutes=SLOT_MINUTES,
        pressure_inhg=series(PRESSURE_COLUMN, PRESSURE_DIGITS),
        humidity_pct=series(HUMIDITY_COLUMN, HUMIDITY_DIGITS),
    )
