"""
ribbon.py: The 48-hour condition ribbon on the Overview's first screen.

The ribbon is the last 48 hours in 15-minute slots, ending with the slot that
holds the newest observation, read from the stitched history so the live
reading is its newest point. Each slot carries:

- the newest temperature in it (so the last slot is the current temperature),
- the rain that fell in it, from the daily counter's increments,
- whether the sun was up (any solar radiation in it).

Rain is told from the counter only between readings at most one slot apart.
Across a longer silence (an archive gap, or the span between the archive's
newest row and the live reading) the counter's rise cannot be placed in any
one slot, so those slots carry no rain value and the site hatches them: only
the live total is known there.

Functions:
- condition_ribbon(stitched, observed_at_ms): The ribbon, or None.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from lookout.snapshot.model import ConditionRibbon

SLOT_MINUTES = 15
SLOTS = 48 * 60 // SLOT_MINUTES
SLOT_MS = SLOT_MINUTES * 60 * 1000

# Readings further apart than this cannot place a rain increment in a slot.
_MAX_STEP_MS = SLOT_MS


def _rain_increments(ordered: pd.DataFrame) -> pd.Series:
    """
    The rain each reading adds to the daily counter, NaN where it is unknown.

    The counter resets at local midnight, so a fall means the new value is
    all new rain. An increment after a silence longer than one slot is
    unknown: it cannot be placed in one slot.

    :param ordered: The stitched history, oldest first.
    :return: One increment per row, aligned with `ordered`.
    """
    if "dailyrainin" not in ordered.columns:
        return pd.Series(np.nan, index=ordered.index)
    daily = pd.to_numeric(ordered["dailyrainin"], errors="coerce")
    step = daily.diff()
    # The oldest reading has nothing before it to rise from: unknown.
    increment = step.where(step.isna() | (step >= 0), daily)
    after_silence = ordered["dateutc"].diff() > _MAX_STEP_MS
    return increment.mask(after_silence)


def _maybe(value: float) -> Optional[float]:
    """A float, or None for NaN."""
    return None if pd.isna(value) else float(value)


def condition_ribbon(
    stitched: pd.DataFrame, observed_at_ms: Optional[int]
) -> Optional[ConditionRibbon]:
    """
    The 48-hour condition ribbon.

    :param stitched: `stitch_live` output, newest first.
    :param observed_at_ms: The newest observation; None when there is none.
    :return: The ribbon, or None without an observation.
    """
    if observed_at_ms is None or stitched.empty:
        return None

    last_start = observed_at_ms - observed_at_ms % SLOT_MS
    start_ms = last_start - (SLOTS - 1) * SLOT_MS

    ordered = stitched.sort_values("dateutc").reset_index(drop=True)
    rows = ordered.assign(rain=_rain_increments(ordered))
    rows = rows[rows["dateutc"] >= start_ms]
    index = ((rows["dateutc"] - start_ms) // SLOT_MS).astype(int)
    grouped = rows.groupby(index)

    temperature: List[Optional[float]] = [None] * SLOTS
    rain: List[Optional[float]] = [None] * SLOTS
    daylight: List[Optional[bool]] = [None] * SLOTS

    if "tempf" in rows.columns:
        # Oldest first, so `last` is the newest reading in each slot.
        for i, value in grouped["tempf"].last().items():
            temperature[i] = _maybe(value)
    for i, increments in grouped["rain"]:
        if increments.notna().any():
            rain[i] = round(float(increments.sum()), 3)
    if "solarradiation" in rows.columns:
        for i, solar in grouped["solarradiation"]:
            if solar.notna().any():
                daylight[i] = bool((solar > 0).any())

    temps = rows["tempf"].dropna() if "tempf" in rows.columns else pd.Series()
    known = [value for value in rain if value is not None]
    return ConditionRibbon(
        start_ms=int(start_ms),
        slot_minutes=SLOT_MINUTES,
        temperature_f=tuple(temperature),
        rain_in=tuple(rain),
        daylight=tuple(daylight),
        temperature_low_f=_maybe(temps.min()) if not temps.empty else None,
        temperature_high_f=_maybe(temps.max()) if not temps.empty else None,
        rain_hours=sum(1 for value in known if value > 0) * SLOT_MINUTES / 60,
        known_hours=len(known) * SLOT_MINUTES / 60,
    )
