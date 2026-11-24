"""
dry_spell.py: The dry spell now: whole dry days since it last rained, and
the usual.

The days since it last rained count whole dry days (GLOSSARY dry spell,
#668): completed station-local days under one tip of rain, back from
yesterday to the last rain day. Today is never a whole day yet, and rain
today makes 0: no dry spell yet. A day not counted for rain breaks the run,
so a gap never reads as dry: the count stops there, and with no known dry
day since the gap the days since rain are unknown. Ambient's `lastRain`
plays no part.

The usual dry spell is the median completed dry spell in the last 365 days.
A dry spell is a run of dry days between two rain days
(`lookout.core.rainfall_analysis.dry_spells`), over the build's daily rain
series, the Rain tab's own; a day not counted for rain breaks the run, so a
gap never reads as dry. The 365-day window is judged by the rain day kind
(`lookout.snapshot.days`); with it not comparable there is no usual dry
spell.

Computed once per build (`build.py`); the feature and the widget view read
the result, so every reader of `days_since_rain` sees the one count.

Functions:
- completed_dry_spells(days): Every completed dry spell in the 365-day window.
- whole_dry_days(days): The whole dry days since the last rain day.
- dry_spell(days, spells): The `DrySpellNow` field.
"""

from datetime import date, timedelta
from typing import List, Optional

import numpy as np

from lookout.core.rainfall_analysis import WET_DAY_IN, dry_spells
from lookout.snapshot.days import CountedDays
from lookout.snapshot.gaps import comparable
from lookout.snapshot.model import DrySpellNow

DRY_SPELL_WINDOW = "365d"
"""The window the usual dry spell is measured over."""

DIGITS = 2
"""Decimals kept on the usual dry spell."""


def completed_dry_spells(
    days: CountedDays,
) -> Optional[List[int]]:
    """
    Every completed dry spell in the 365-day rain window, in days.

    :param days: The counted days and the daily rain series.
    :return: The spell lengths, or None when the window is invalid.
    """
    status = days.rain.rolling(365, DRY_SPELL_WINDOW)

    def spells() -> List[int]:
        series = days.rain_days(date.fromisoformat(status.first_day), days.clock.today)
        return [int(length) for length in dry_spells(series)["days"]]

    return comparable(status, spells)


def whole_dry_days(days: CountedDays) -> Optional[int]:
    """
    The whole dry days since the last rain day, back from yesterday.

    :param days: The counted days and the daily rain series; today's live.
    :return: 0 with rain today or yesterday; the count of known dry days
        before a rain day or a day not counted for rain; None when not one
        known dry day lies between yesterday and such a day not counted.
    """
    totals = days.rain_in
    today = days.clock.today
    # Today's rain, live, whatever its day kind: rain today is never dry.
    if days.rain_series.get(today, 0.0) >= WET_DAY_IN:
        return 0
    count = 0
    day = today - timedelta(days=1)
    while True:
        rain = totals.get(day)
        if rain is None or np.isnan(rain):
            return count or None
        if rain >= WET_DAY_IN:
            return count
        count += 1
        day -= timedelta(days=1)


def dry_spell(days: CountedDays, spells: Optional[List[int]]) -> DrySpellNow:
    """
    The dry spell field: whole dry days since it last rained, and the usual.

    :param days: The counted days and the daily rain series.
    :param spells: Completed dry spell lengths from `completed_dry_spells`,
        or None when the 365-day window is not comparable.
    :return: The `DrySpellNow`; `usual_days` is None without a comparison set.
    """
    usual_days = None
    if spells:
        usual_days = round(float(np.median(np.asarray(spells, dtype=float))), DIGITS)
    return DrySpellNow(days_since_rain=whole_dry_days(days), usual_days=usual_days)
