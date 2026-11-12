"""
wind.py: The wind rose, where the wind comes from (#465).

For each window in `WINDOWS` (24 h, 7 days and 30 days back from the build
time) the rose gives:

- 16 direction bins of 22.5°, centred on N (N is 348.75° up to 11.25°), each
  with its windy readings' average wind and highest gust;
- the share of calm readings, which have no direction and are left out of
  the bins;
- the share of windy readings from each of 8 compass points of 45°, centred
  on N (N is 337.5° up to 22.5°).

A reading's direction (`winddir`, where the wind blows from) is honest only
while the wind moves the vane: Ambient keeps reporting the vane's last
resting direction at 0 mph. So a reading is **calm** when its wind speed
(`windspeedmph`) is under `CALM_BELOW_MPH`, Beaufort force 0 ("calm", under
1 mph), or when it has no direction. Every other reading with a speed is
**windy**, and only windy readings fill the bins and the points. A reading
with no wind speed is not counted at all.

Bin and point edges are half-open: a direction on an edge belongs to the bin
clockwise of it (11.25° is NNE). The bins' order, labels and edges are
decided here, so the site never computes them (ADR-0003).

A window holds the readings after `now - hours`, through `now`, from the
live-stitched history, so the live reading counts once. Its gap status is
that of the station-local days it touches (`ArchiveCoverage.window`), so it
carries the record's depth as `shown` (#463) and is judged by the One Gap
Status Rule: an invalid window has no values. The direction bin is derived
on read here and never stored in the archive.

The windows are rolling hours, not local days, so their gap statuses are
keyed `STATUS_PREFIX` + the window's key (`wind_7d`): 168 h back touches 8
local days, and the standard `7d` window spans 7 (#508). A window's
`status.shown` says whether the record covers it; `record_depth` lists only
the standard windows, so it names no unlock date for a wind window.

Functions:
- wind_rose(stitched, coverage): The `WindRose` section.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from lookout.snapshot.gaps import ArchiveCoverage, comparable
from lookout.snapshot.model import (
    GapStatus,
    WindBin,
    WindPoint,
    WindRose,
    WindRoseWindow,
)
from lookout.snapshot.values import number
from lookout.utils.dateutc import HOUR_MS, local_time

WINDOWS: Tuple[Tuple[str, int], ...] = (("24h", 24), ("7d", 7 * 24), ("30d", 30 * 24))
"""Each window as (key, hours back from the build time), shortest first."""

STATUS_PREFIX = "wind_"
"""Prefixed to a window's key to key its gap status apart from the standard
windows, which span whole local days."""

CALM_BELOW_MPH = 1.0
"""A reading under this wind speed is calm: Beaufort force 0."""

DIRECTIONS: Tuple[str, ...] = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)
"""The 16 direction bins, clockwise from N."""

POINTS: Tuple[str, ...] = DIRECTIONS[::2]
"""The 8 compass points, clockwise from N."""

BIN_DEG = 360 / len(DIRECTIONS)
POINT_DEG = 360 / len(POINTS)

SPEED_DIGITS = 1
"""Decimals kept on a bin's average wind and highest gust (mph)."""


def wind_rose(stitched: pd.DataFrame, coverage: ArchiveCoverage) -> WindRose:
    """
    The wind rose over each of `WINDOWS`.

    :param stitched: The live-stitched history (`dateutc` in ms, Ambient
        wind columns), any order.
    :param coverage: The build's archive coverage: its clock and day kinds.
    :return: The section.
    """
    readings = _readings(stitched)
    return WindRose(
        calm_below_mph=CALM_BELOW_MPH,
        windows=tuple(
            _window(readings, coverage, key, hours) for key, hours in WINDOWS
        ),
    )


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """
    One wind column as floats; all NaN when the archive lacks it.

    :param frame: The history.
    :param name: The Ambient column.
    :return: The values, NaN where unknown.
    """
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce").astype("float64")


def _readings(stitched: pd.DataFrame) -> pd.DataFrame:
    """
    The readings with a wind speed, each judged calm or put in its bin.

    :param stitched: The live-stitched history.
    :return: Columns `dateutc`, `speed`, `gust`, `calm`, `bin` and `point`
        (the last two -1 for a calm reading).
    """
    if stitched.empty or "dateutc" not in stitched:
        return pd.DataFrame(
            {
                name: pd.Series(dtype=dtype)
                for name, dtype in (
                    ("dateutc", "int64"),
                    ("speed", "float64"),
                    ("gust", "float64"),
                    ("calm", "bool"),
                    ("bin", "int64"),
                    ("point", "int64"),
                )
            }
        )
    speed = _column(stitched, "windspeedmph")
    known = speed.notna()
    speed = speed[known]
    direction = _column(stitched, "winddir")[known]
    calm = direction.isna() | (speed < CALM_BELOW_MPH)
    bearing = direction.fillna(0.0) % 360
    return pd.DataFrame(
        {
            "dateutc": stitched.loc[known, "dateutc"].astype("int64"),
            "speed": speed,
            "gust": _column(stitched, "windgustmph")[known],
            "calm": calm,
            "bin": np.where(calm, -1, _sector(bearing, BIN_DEG)),
            "point": np.where(calm, -1, _sector(bearing, POINT_DEG)),
        }
    )


def _sector(direction: pd.Series, width: float) -> np.ndarray:
    """
    The sector each direction falls in, sectors of `width` centred on N.

    :param direction: Degrees in [0, 360).
    :param width: The sector width in degrees, a divisor of 360.
    :return: The sector index, 0 for the one centred on N, clockwise.
    """
    count = int(round(360 / width))
    return (np.floor((direction + width / 2) / width).astype("int64") % count).values


def _window(
    readings: pd.DataFrame, coverage: ArchiveCoverage, key: str, hours: int
) -> WindRoseWindow:
    """
    The rose over one window back from the build time.

    :param readings: The judged readings (`_readings`).
    :param coverage: The build's archive coverage.
    :param key: The window's key.
    :param hours: Its span in hours.
    :return: The window.
    """
    now_ms = coverage.now_ms
    start_ms = now_ms - hours * HOUR_MS
    first_day = local_time(start_ms, coverage.tz).date()
    status = coverage.window(first_day, coverage.today, STATUS_PREFIX + key)
    inside = readings[
        (readings["dateutc"] > start_ms) & (readings["dateutc"] <= now_ms)
    ]
    held = comparable(status, lambda: inside)
    return _summarise(key, hours, status, inside.iloc[:0] if held is None else held)


def _summarise(
    key: str, hours: int, status: GapStatus, held: pd.DataFrame
) -> WindRoseWindow:
    """
    Count and summarise one window's readings.

    :param key: The window's key.
    :param hours: Its span in hours.
    :param status: Its gap status.
    :param held: Its readings; none when it is invalid.
    :return: The window.
    """
    windy = held[~held["calm"]]
    total = len(held)
    calm = total - len(windy)
    return WindRoseWindow(
        key=key,
        hours=hours,
        status=status,
        readings=total,
        calm_readings=calm,
        calm_pct=_percent(calm, total),
        bins=_bins(windy),
        points=_points(windy),
    )


def _bins(windy: pd.DataFrame) -> Tuple[WindBin, ...]:
    """
    The 16 direction bins of a window's windy readings.

    :param windy: The window's windy readings.
    :return: The bins, clockwise from N.
    """
    bins = []
    for index, direction in enumerate(DIRECTIONS):
        from_bin = windy[windy["bin"] == index]
        centre = index * BIN_DEG
        bins.append(
            WindBin(
                direction=direction,
                centre_deg=centre,
                from_deg=(centre - BIN_DEG / 2) % 360,
                to_deg=centre + BIN_DEG / 2,
                readings=len(from_bin),
                average_mph=number(from_bin["speed"].mean(), SPEED_DIGITS),
                gust_mph=number(from_bin["gust"].max(), SPEED_DIGITS),
            )
        )
    return tuple(bins)


def _points(windy: pd.DataFrame) -> Tuple[WindPoint, ...]:
    """
    The 8 compass points' shares of a window's windy readings.

    :param windy: The window's windy readings.
    :return: The points, clockwise from N.
    """
    points = []
    for index, direction in enumerate(POINTS):
        count = int((windy["point"] == index).sum())
        points.append(
            WindPoint(
                direction=direction,
                centre_deg=index * POINT_DEG,
                readings=count,
                share_pct=_percent(count, len(windy)),
            )
        )
    return tuple(points)


def _percent(part: int, whole: int) -> Optional[int]:
    """
    `part` of `whole` as a whole percent, None when `whole` is 0.

    :param part: The count.
    :param whole: What it is a share of.
    :return: The percent.
    """
    return None if whole == 0 else int(round(100 * part / whole))
