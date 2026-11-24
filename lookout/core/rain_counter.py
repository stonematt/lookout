"""
rain_counter.py: The rain each reading adds to the daily rain counter.

Ambient's `dailyrainin` counts up through a day and drops back when it
resets. The reset does not fall at the station's local midnight: Lookout
Post's counter resets a few minutes late, so its 00:00 reading still holds
yesterday's total, and Salmon Run's resets at 21:00 Pacific, midnight
Eastern (#328, #134). So a reset is told by the counter, never the clock:
a drop is the reset, whenever it falls, and the reading after it holds only
rain since the reset.

A reset can hide in a gap: when more rain fell after the reset than the
counter held before it, the counter after the gap is higher, not lower.
Given the readings' times, a gap that spans the station's usual reset
time (learned from its drops) holds a reset too.

Every reader of the counter's rain takes it from here: hourly rain, daily
totals, rain so far today, the condition ribbon, event curves, event
quality and carryover detection (#654).

A drop that is a glitch, not a reset, is read as a reset too (spec #647).

Functions:
- rain_increments(counter, dateutc, tz): The rain each reading adds.
- usual_reset(counter, dateutc, tz): The station's usual reset time of day.
"""

from typing import List, Optional

import pandas as pd

from lookout.utils.dateutc import to_local

STEADY_STEP_MS = 10 * 60 * 1000
"""A step at most this long is a steady one: its drop dates a reset."""

HIDDEN_RESET_MARGIN = pd.Timedelta(minutes=30)
"""
A gap holds a hidden reset when the usual reset time falls inside it with
this much to spare on both sides, so a reset a few minutes off its usual
time is never misread.
"""

_ROUND = "5min"


def _last_known(counter: pd.Series) -> pd.Series:
    """Each reading's previous counter value, NaN before the first."""
    return counter.ffill().shift()


def usual_reset(
    counter: pd.Series, dateutc: pd.Series, tz: str
) -> Optional[pd.Timedelta]:
    """
    The station-local time of day its counter usually resets.

    The most common time of day, to five minutes, of drops after a steady
    step.

    :param counter: The daily counter, oldest first.
    :param dateutc: The readings' times, epoch ms, aligned with `counter`.
    :param tz: The station-local zone.
    :return: The time after local midnight; None when no drop is seen.
    """
    daily = pd.to_numeric(counter, errors="coerce").astype(float)
    known_at = dateutc.where(daily.notna()).ffill().shift()
    steady = (dateutc - known_at) <= STEADY_STEP_MS
    drops = (daily < _last_known(daily)) & steady
    if not drops.any():
        return None
    local = to_local(dateutc[drops], tz).dt.tz_localize(None)
    times = (local - local.dt.normalize()).dt.floor(_ROUND)
    return pd.Timedelta(times.mode().iloc[0])


def _hidden_resets(
    daily: pd.Series, dateutc: pd.Series, tz: str, reset: pd.Timedelta
) -> pd.Series:
    """
    Readings after a gap that spans the usual reset time.

    :param daily: The daily counter, oldest first.
    :param dateutc: The readings' times, epoch ms.
    :param tz: The station-local zone.
    :param reset: The usual reset time after local midnight.
    :return: True for each such reading, aligned with `daily`.
    """
    known_at = dateutc.where(daily.notna()).ffill().shift()
    gap = daily.notna() & ((dateutc - known_at) > STEADY_STEP_MS)
    hidden = pd.Series(False, index=daily.index)
    if not gap.any():
        return hidden
    after = to_local(dateutc[gap], tz).dt.tz_localize(None)
    before = to_local(known_at[gap].astype("int64"), tz).dt.tz_localize(None)
    flags: List[bool] = []
    for start, end in zip(before + HIDDEN_RESET_MARGIN, after - HIDDEN_RESET_MARGIN):
        at = start.normalize() + reset
        if at < start:
            at += pd.Timedelta(days=1)
        flags.append(bool(at <= end))
    hidden[gap] = flags
    return hidden


def rain_increments(
    counter: pd.Series,
    dateutc: Optional[pd.Series] = None,
    tz: Optional[str] = None,
) -> pd.Series:
    """
    The rain each reading adds to the daily counter.

    Each reading is compared with the last one before it that reported the
    counter, so one missing reading does not void the next. A rise is the
    rain since that reading; a drop is the reset, and the new value is all
    new rain. Given the readings' times and zone, a reading after a gap
    that spans the station's usual reset time (`usual_reset`, local
    midnight when no drop is seen) holds a reset too: its value is all new
    rain even when it is higher than the one before the gap.

    :param counter: The daily counter (`dailyrainin`), oldest first; NaN or
        missing values for readings without it.
    :param dateutc: The readings' times, epoch ms, aligned with `counter`;
        None to tell resets by drops alone.
    :param tz: The station-local zone; needed with `dateutc`.
    :return: One increment per reading, aligned with `counter`, never
        negative. NaN for a reading without the counter, and for the first
        one with it: nothing before it says how much is new. The caller
        decides what that first reading means.
    """
    daily = pd.to_numeric(counter, errors="coerce").astype(float)
    step = daily - _last_known(daily)
    increment = step.where(step.isna() | (step >= 0), daily)
    if dateutc is not None and tz is not None:
        times = pd.Series(dateutc.to_numpy(dtype="int64"), index=daily.index)
        reset = usual_reset(daily, times, tz) or pd.Timedelta(0)
        hidden = _hidden_resets(daily, times, tz, reset) & step.notna()
        increment = increment.mask(hidden, daily)
    return increment.clip(lower=0)
