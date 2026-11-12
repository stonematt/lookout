"""
indoor.py: The station's indoor sensors and battery flags (#466).

A sensor counts only while it is still reporting: an indoor sensor while its
temperature column, and a battery flag while its column, has a non-null
reading in the last `CURRENT_HOURS` of the live-stitched history, measured
back from the history's newest reading, not the wall clock. A sensor no
longer reporting is left out, so a battery whose last stored value is 0 from
a sensor long gone is absent, not low.

The live archives store these columns as several types (#466 column read):
`battrain` is a text category ("1"), `batt_co2` and some `battout` and
`humidityin` int64, the rest double. Every value is cast on read here; the
archive is never changed.

Battery flags: the value is the flag's newest non-null reading in the
window, cast to a number. 1 is OK and 0 is low; any other number, or a value
that does not parse as one, leaves the flag out and logs a warning: a value
is never guessed to be low. Printable names come from `BATTERY_NAMES`, one
fixed map keyed by column; a `batt*` column not in it is left out and logged.
Flags are not tied to indoor sensors: `battout`, `battrain` and `batt_co2`
belong to outdoor sensors, and a console may report no `battin`.

Indoor sensors are the console (`tempinf`, `humidityin`) and the numbered
sensors (`temp<n>f`, `humidity<n>`, 1 to `SENSOR_COUNT`), named as their
batteries are in `BATTERY_NAMES`. Each carries its newest temperature and
humidity in the window and its hourly mean temperature over the 24 hours
ending in the newest reading's hour.

Functions:
- indoor_sensors(stitched): The `IndoorSensor` entries.
- battery_flags(stitched): The `BatteryFlag` entries.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from lookout.snapshot.model import BatteryFlag, IndoorSensor, IndoorTrace
from lookout.snapshot.values import number
from lookout.utils.dateutc import HOUR_MS
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

CURRENT_HOURS = 24
"""A sensor is current with a reading this many hours back from the newest."""

SENSOR_COUNT = 10
"""Ambient's numbered sensors: 1 to 10."""

TEMPERATURE_DIGITS = 1

BATTERY_NAMES: Dict[str, str] = {
    "battout": "Outdoor sensor",
    "battin": "Console",
    **{f"batt{n}": f"Sensor {n}" for n in range(1, SENSOR_COUNT + 1)},
    "battrain": "Rain gauge",
    "batt_co2": "CO2 sensor",
    "batt_25": "PM2.5 sensor",
    "batt_lightning": "Lightning sensor",
    **{f"battsm{n}": f"Soil sensor {n}" for n in range(1, 5)},
}
"""Each battery flag's column and the sensor name the site prints, in the
order the flags are listed (#466)."""

BATTERY_OK = 1.0
BATTERY_LOW = 0.0


class IndoorColumns(NamedTuple):
    """One indoor sensor's key, printable name and Ambient columns."""

    key: str
    name: str
    temperature: str
    humidity: str


INDOOR_SENSORS: Tuple[IndoorColumns, ...] = (
    IndoorColumns("in", BATTERY_NAMES["battin"], "tempinf", "humidityin"),
    *(
        IndoorColumns(str(n), BATTERY_NAMES[f"batt{n}"], f"temp{n}f", f"humidity{n}")
        for n in range(1, SENSOR_COUNT + 1)
    ),
)
"""Each indoor sensor, the console first, named as its battery is."""


def indoor_sensors(stitched: pd.DataFrame) -> Tuple[IndoorSensor, ...]:
    """
    Each indoor sensor still reporting, the console first, then by number.

    :param stitched: The live-stitched history (`dateutc` in ms), any order.
    :return: The sensors; empty when the station has none.
    """
    recent, newest_ms = _recent(stitched)
    if newest_ms is None:
        return ()
    start_ms = (newest_ms // HOUR_MS - (CURRENT_HOURS - 1)) * HOUR_MS
    sensors: List[IndoorSensor] = []
    for key, name, temperature, humidity in INDOOR_SENSORS:
        if temperature not in recent:
            continue
        readings = _numbers(recent, temperature)
        if readings.empty:
            continue
        newest = readings.index[0]
        humid = _numbers(recent, humidity) if humidity in recent else readings[:0]
        sensors.append(
            IndoorSensor(
                key=key,
                name=name,
                temperature_f=float(number(readings.iloc[0], TEMPERATURE_DIGITS)),
                humidity_pct=None if humid.empty else number(humid.iloc[0], 1),
                observed_at_ms=int(newest),
                trace=_trace(readings, start_ms),
            )
        )
    return tuple(sensors)


def battery_flags(stitched: pd.DataFrame) -> Tuple[BatteryFlag, ...]:
    """
    Each battery still reporting, OK or low, in `BATTERY_NAMES` order.

    :param stitched: The live-stitched history (`dateutc` in ms), any order.
    :return: The flags; a stale, unmapped or unreadable one left out.
    """
    for column in stitched.columns:
        if str(column).startswith("batt") and column not in BATTERY_NAMES:
            logger.warning(f"Battery flag {column} has no sensor name: left out")
    recent, newest_ms = _recent(stitched)
    if newest_ms is None:
        return ()
    flags: List[BatteryFlag] = []
    for column, name in BATTERY_NAMES.items():
        if column not in recent:
            continue
        held = recent[column].dropna()
        if held.empty:
            continue
        low = _low(column, held.iloc[0])
        if low is not None:
            flags.append(BatteryFlag(key=column, name=name, low=low))
    return tuple(flags)


def _recent(stitched: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    The history's last `CURRENT_HOURS`, newest first, back from its newest row.

    :param stitched: The live-stitched history, any order.
    :return: The rows after the newest reading less `CURRENT_HOURS`, through
        it, newest first and indexed by `dateutc`; and the newest reading's
        time, None when the history is empty.
    """
    if stitched.empty or "dateutc" not in stitched:
        return stitched.iloc[0:0], None
    dateutc = stitched["dateutc"].astype("int64")
    newest_ms = int(dateutc.max())
    recent = stitched[dateutc > newest_ms - CURRENT_HOURS * HOUR_MS]
    recent = recent.set_index(recent["dateutc"].astype("int64"), drop=False)
    return recent.sort_index(ascending=False), newest_ms


def _numbers(recent: pd.DataFrame, column: str) -> pd.Series:
    """
    One column's readings as floats, newest first, unreadable and null dropped.

    :param recent: The window's rows, newest first, indexed by `dateutc`.
    :param column: The Ambient column.
    :return: The readings.
    """
    values = pd.to_numeric(recent[column].astype(object), errors="coerce")
    return values.astype("float64").dropna()


def _trace(readings: pd.Series, start_ms: int) -> IndoorTrace:
    """
    Hourly mean temperatures from `start_ms` over `CURRENT_HOURS` slots.

    :param readings: The sensor's temperatures, indexed by `dateutc`.
    :param start_ms: The first slot's start, on the hour.
    :return: The trace; a slot with no reading is None.
    """
    inside = readings[readings.index >= start_ms]
    slot = (np.asarray(inside.index, dtype="int64") - start_ms) // HOUR_MS
    means = inside.groupby(slot).mean()
    return IndoorTrace(
        start_ms=start_ms,
        slot_minutes=HOUR_MS // 60_000,
        temperature_f=tuple(
            number(means.get(i), TEMPERATURE_DIGITS) for i in range(CURRENT_HOURS)
        ),
    )


def _low(column: str, value: Any) -> Optional[bool]:
    """
    Read a battery flag's newest value: 1 is OK, 0 is low.

    :param column: The flag's column, for the log line.
    :param value: Its newest non-null stored value, of any stored type.
    :return: True for low, False for OK; None, logged, for any other value
        or one that does not parse as a number.
    """
    try:
        reading = float(value)
    except (TypeError, ValueError):
        logger.warning(f"Battery flag {column} reads {value!r}, not a number: left out")
        return None
    if reading == BATTERY_OK:
        return False
    if reading == BATTERY_LOW:
        return True
    logger.warning(f"Battery flag {column} reads {value!r}, not 1 or 0: left out")
    return None
