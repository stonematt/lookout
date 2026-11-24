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
Given the readings' times, a gap that spans the station's usual reset time
holds a reset too. That time is learned, for each gap, from the drops in
the 30 days before it, so a reset that moves (a console's time zone set,
#651) is followed within a month.

Every reader of the counter's rain takes it from here: hourly rain, daily
totals, rain so far today, the condition ribbon, event curves, event
quality and carryover detection (#654). Only hourly rain, daily totals and
rain so far pass the readings' times: the ribbon, event curves and event
quality give no rate across a gap, and carryover detection asks whether
the counter itself reset across one.

A drop that is a glitch, not a reset, is read as a reset too (spec #647).

Functions:
- rain_increments(counter, dateutc, tz): The rain each reading adds.
- usual_reset(counter, dateutc, tz, before_ms): The station's usual reset
  time of day.
"""

from typing import Optional

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

LEARN_MS = 30 * 24 * 60 * 60 * 1000
"""The usual reset time is learned from the drops in this span before a gap."""

MIN_DROPS = 3
"""Fewer drops than this in the span teach nothing: local midnight stands."""

_ROUND = "5min"


def _last_known(counter: pd.Series) -> pd.Series:
    """Each reading's previous counter value, NaN before the first."""
    return counter.ffill().shift()


def _known_at(counter: pd.Series, dateutc: pd.Series) -> pd.Series:
    """Each reading's previous counter reading's time, NaN before the first."""
    return dateutc.where(counter.notna()).ffill().shift()


def _drop_times(counter: pd.Series, dateutc: pd.Series) -> pd.Series:
    """The times of the drops after a steady step, epoch ms, oldest first."""
    steady = (dateutc - _known_at(counter, dateutc)) <= STEADY_STEP_MS
    drops = (counter < _last_known(counter)) & steady
    return dateutc[drops]


def usual_reset(
    counter: pd.Series,
    dateutc: pd.Series,
    tz: str,
    before_ms: Optional[int] = None,
) -> Optional[pd.Timedelta]:
    """
    The station-local time of day its counter usually resets.

    The most common time of day, to five minutes, of the drops after a
    steady step in the `LEARN_MS` before `before_ms`.

    :param counter: The daily counter, oldest first.
    :param dateutc: The readings' times, epoch ms, aligned with `counter`.
    :param tz: The station-local zone.
    :param before_ms: The end of the span learned from; the newest reading
        when None.
    :return: The time after local midnight; None with fewer than
        `MIN_DROPS` drops in the span.
    """
    daily = pd.to_numeric(counter, errors="coerce").astype(float)
    return _mode(_drop_times(daily, dateutc), tz, before_ms)


def _mode(
    drops: pd.Series, tz: str, before_ms: Optional[int]
) -> Optional[pd.Timedelta]:
    """The most common time of day of `drops` in the span before `before_ms`."""
    end = drops.max() if before_ms is None else before_ms
    recent = drops[(drops <= end) & (drops > end - LEARN_MS)]
    if len(recent) < MIN_DROPS:
        return None
    local = to_local(recent, tz).dt.tz_localize(None)
    times = (local - local.dt.normalize()).dt.floor(_ROUND)
    return pd.Timedelta(times.mode().iloc[0])


def _hidden_resets(daily: pd.Series, dateutc: pd.Series, tz: str) -> pd.Series:
    """
    Readings after a gap that spans the usual reset time.

    :param daily: The daily counter, oldest first.
    :param dateutc: The readings' times, epoch ms.
    :param tz: The station-local zone.
    :return: True for each such reading, aligned with `daily`.
    """
    known_at = _known_at(daily, dateutc)
    gap = daily.notna() & ((dateutc - known_at) > STEADY_STEP_MS)
    hidden = pd.Series(False, index=daily.index)
    if not gap.any():
        return hidden
    drops = _drop_times(daily, dateutc)
    for i in gap[gap].index:
        start_ms = int(known_at[i])
        reset = _mode(drops, tz, start_ms) or pd.Timedelta(0)
        start, end = (
            to_local(pd.Series([start_ms, int(dateutc[i])]), tz)
            .dt.tz_localize(None)
            .tolist()
        )
        start, end = start + HIDDEN_RESET_MARGIN, end - HIDDEN_RESET_MARGIN
        at = start.normalize() + reset
        if at < start:
            at += pd.Timedelta(days=1)
        hidden[i] = at <= end
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
    that spans the station's usual reset time (`usual_reset` over the 30
    days before the gap; local midnight when it has too few drops) holds a
    reset too: its value is all new rain even when it is higher than the
    one before the gap.

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
        hidden = _hidden_resets(daily, times, tz) & step.notna()
        increment = increment.mask(hidden, daily)
    return increment.clip(lower=0)
