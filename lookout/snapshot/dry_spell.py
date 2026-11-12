"""
dry_spell.py: The dry spell now: days since it last rained, and the usual.

The usual dry spell is the median completed dry spell in the last 365 days.
A dry spell is a run of dry days between two rain days
(`lookout.core.rainfall_analysis.dry_spells`), over the build's daily rain
series, the Rain tab's own; a day not counted for rain breaks the run, so a
gap never reads as dry. The 365-day window is judged by the rain day kind
(`lookout.snapshot.days`); with it not comparable there is no usual dry
spell.

Computed once per build (`build.py`); the hero lead's calm rule, the feature
and the widget view read the result.

Functions:
- completed_dry_spells(days): Every completed dry spell in the 365-day window.
- dry_spell(current, observed_at_ms, spells): The `DrySpellNow` field.
"""

from datetime import date
from typing import List, Optional, Tuple

import numpy as np

from lookout.core.rainfall_analysis import dry_spells
from lookout.snapshot.days import CountedDays
from lookout.snapshot.gaps import comparable
from lookout.snapshot.model import DrySpellNow, SnapshotConditions
from lookout.utils.dateutc import DAY_MS

DRY_SPELL_WINDOW = "365d"
"""The window the usual dry spell is measured over."""

DIGITS = 2
"""Decimals kept on the days since rain and the usual dry spell."""


def completed_dry_spells(
    days: CountedDays,
) -> Tuple[Optional[List[int]], Tuple[str, ...]]:
    """
    Every completed dry spell in the 365-day rain window, in days.

    :param days: The counted days and the daily rain series.
    :return: (spell lengths, or None when the window is invalid; its key when
        comparable).
    """
    status = days.rain.rolling(365, DRY_SPELL_WINDOW)

    def spells() -> List[int]:
        series = days.rain_days(date.fromisoformat(status.first_day), days.clock.today)
        return [int(length) for length in dry_spells(series)["days"]]

    lengths = comparable(status, spells)
    return lengths, ((status.key,) if lengths is not None else ())


def dry_spell(
    current: SnapshotConditions,
    observed_at_ms: Optional[int],
    spells: Optional[List[int]],
) -> DrySpellNow:
    """
    The dry spell field: days since it last rained, and the usual dry spell.

    :param current: Current conditions, live stitched.
    :param observed_at_ms: The newest observation, or None.
    :param spells: Completed dry spell lengths from `completed_dry_spells`,
        or None when the 365-day window is not comparable.
    :return: The `DrySpellNow`; `usual_days` is None without a comparison set.
    """
    days_since_rain = None
    if current.last_rain_ms is not None and observed_at_ms is not None:
        days_since_rain = round(
            (observed_at_ms - current.last_rain_ms) / DAY_MS, DIGITS
        )
    usual_days = None
    if spells:
        usual_days = round(float(np.median(np.asarray(spells, dtype=float))), DIGITS)
    return DrySpellNow(days_since_rain=days_since_rain, usual_days=usual_days)
