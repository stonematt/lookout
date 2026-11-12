"""
history.py: How deep the station's record is, and what that shows (#463).

Principle 4: a comparison appears only once the record supports it, and one
line names the next unlock; a gap inside a shown period stays visible.

A comparison is shown when the record covers its first day: the archive's
first station-local day is on or before it (`ArchiveCoverage.covers`, which
every gap status carries as `shown`). That is the record's depth, not its
gaps: a shown window with a gap inside it still reports the gap by its
status, and a window the record does not reach is hidden, not hatched.

The comparisons, in order:

- the standard windows ("today" ... "365d", then "water_year"), as
  `Gaps.windows` judges them. A window ending today and starting `n` days
  back unlocks `n` days after the record's first day; the water year since
  Oct 1 unlocks on the first Oct 1 inside the record.
- "earlier_year": today's date in an earlier year, ±`SPREAD_DAYS`, which
  the usual band, the strips, the ten-day usual and the almanac's earlier
  columns compare with. It unlocks once last year's window around today
  starts inside the record.
- "full_history": shown whenever the record holds a reading.

The next unlock is the comparison not yet shown with the earliest unlock
date; on a tie, the earlier in that order. Each tile shows when the one
comparison it needs does: the temperature tile and the last ten days need an
earlier day ("yesterday", or the usual band and the list have nothing to set
today against); the tiles of today's readings need "today"; daylight needs
no record.

Functions:
- history_depth(coverage): The `HistoryDepth` section.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from lookout.core.rainfall_analysis import water_year, water_year_start
from lookout.snapshot.gaps import STANDARD_WINDOWS, ArchiveCoverage
from lookout.snapshot.model import (
    HistoryComparison,
    HistoryDepth,
    HistoryTile,
    NextUnlock,
)
from lookout.snapshot.temperature import FULL_HISTORY, SPREAD_DAYS, same_date

WATER_YEAR = "water_year"
EARLIER_YEAR = "earlier_year"

TILES: Tuple[Tuple[str, Optional[str]], ...] = (
    ("temperature", "yesterday"),
    ("rain", "today"),
    ("last_10_days", "yesterday"),
    ("dry_spell", "today"),
    ("wind", "today"),
    ("pressure", "today"),
    ("humidity", "today"),
    ("sun", "today"),
    ("daylight", None),
)
"""Each Overview tile in the mosaic's order and the comparison it needs.
Inside, the tenth, is decided with the indoor readings (#466)."""


def history_depth(coverage: ArchiveCoverage) -> HistoryDepth:
    """
    The record's depth: each comparison, each tile and the next unlock.

    :param coverage: The build's counted days; its clock says which day is
        today.
    :return: The `HistoryDepth` section.
    """
    first = coverage.first_day
    windows = {status.key: status for status in coverage.standard_windows()}
    comparisons: List[HistoryComparison] = []
    for key, first_back, _ in STANDARD_WINDOWS:
        unlock = None if first is None else first + timedelta(days=first_back)
        comparisons.append(_comparison(key, windows[key].shown, unlock))
    comparisons.append(
        _comparison(
            WATER_YEAR,
            windows[WATER_YEAR].shown,
            None if first is None else _water_year_unlock(first),
        )
    )
    comparisons.append(
        _comparison(
            EARLIER_YEAR,
            coverage.covers(_earlier_year_start(coverage.today)),
            None if first is None else _earlier_year_unlock(first),
        )
    )
    comparisons.append(_comparison(FULL_HISTORY, first is not None, None))

    shown: Dict[str, bool] = {c.key: c.shown for c in comparisons}
    tiles = tuple(
        HistoryTile(key, needs, True if needs is None else shown[needs])
        for key, needs in TILES
    )
    return HistoryDepth(
        first_day=None if first is None else first.isoformat(),
        days=0 if first is None else (coverage.today - first).days + 1,
        comparisons=tuple(comparisons),
        tiles=tiles,
        next_unlock=_next_unlock(comparisons),
    )


def _comparison(key: str, shown: bool, unlock: Optional[date]) -> HistoryComparison:
    """
    One comparison, its unlock date named only while it is not shown.

    :param key: The comparison.
    :param shown: Whether the record covers it.
    :param unlock: The day it is first shown; None when the archive is empty.
    :return: The comparison.
    """
    if shown or unlock is None:
        return HistoryComparison(key, shown, None)
    return HistoryComparison(key, shown, unlock.isoformat())


def _water_year_unlock(first: date) -> date:
    """
    The first day whose water year since Oct 1 lies inside the record.

    :param first: The record's first day.
    :return: `first` when it is an Oct 1, else the next Oct 1.
    """
    start = water_year_start(water_year(first))
    return first if start == first else water_year_start(water_year(first) + 1)


def _earlier_year_start(today: date) -> date:
    """
    The first day of last year's window around today's date.

    :param today: The station-local date.
    :return: Today's date a year back, less `SPREAD_DAYS`.
    """
    return same_date(today.year - 1, today) - timedelta(days=SPREAD_DAYS)


def _earlier_year_unlock(first: date) -> date:
    """
    The first day whose earlier-year window starts inside the record.

    :param first: The record's first day.
    :return: About a year and `SPREAD_DAYS` after `first`; Feb 29 moves the
        answer a day at most, so the candidate steps forward until it holds.
    """
    later = first + timedelta(days=SPREAD_DAYS)
    day = same_date(later.year + 1, later)
    while _earlier_year_start(day) < first:
        day += timedelta(days=1)
    return day


def _next_unlock(comparisons: List[HistoryComparison]) -> Optional[NextUnlock]:
    """
    The comparison not yet shown that unlocks first.

    :param comparisons: Every comparison, in order.
    :return: The earliest unlock, the first in order on a tie; None when
        none is pending.
    """
    pending = [c for c in comparisons if c.unlocks_on is not None]
    if not pending:
        return None
    soonest = min(pending, key=lambda c: c.unlocks_on or "")
    return NextUnlock(key=soonest.key, date=soonest.unlocks_on or "")
