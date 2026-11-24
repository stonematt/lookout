"""
rain_counter.py: The rain each reading adds to the daily rain counter.

Ambient's `dailyrainin` counts up through a day and drops back when it
resets. The reset does not fall at the station's local midnight: Lookout
Post's counter resets a few minutes late, so its 00:00 reading still holds
yesterday's total, and Salmon Run's resets at 21:00 Pacific, midnight
Eastern (#328, #134). So a reset is told by the counter, never the clock:
a drop is the reset, whenever it falls, and the reading after it holds only
rain since the reset.

Every reader of the counter's rain takes it from here: hourly rain, daily
totals, the condition ribbon, event curves, event quality and carryover
detection (#654).

A drop that is a glitch, not a reset, is read as a reset too (spec #647).
A reset hidden inside a gap, where the counter after the gap is above the
one before it, cannot be told from the counter; the rise is the increment.

Functions:
- rain_increments(counter): The rain each reading adds.
"""

import pandas as pd


def rain_increments(counter: pd.Series) -> pd.Series:
    """
    The rain each reading adds to the daily counter.

    Each reading is compared with the last one before it that reported the
    counter, so one missing reading does not void the next. A rise is the
    rain since that reading; a drop is the reset, and the new value is all
    new rain.

    :param counter: The daily counter (`dailyrainin`), oldest first; NaN or
        missing values for readings without it.
    :return: One increment per reading, aligned with `counter`, never
        negative. NaN for a reading without the counter, and for the first
        one with it: nothing before it says how much is new. The caller
        decides what that first reading means.
    """
    daily = pd.to_numeric(counter, errors="coerce").astype(float)
    step = daily - daily.ffill().shift()
    return step.where(step.isna() | (step >= 0), daily)
