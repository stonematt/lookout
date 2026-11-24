"""
wet_mode.py: Today's wet mode or dry mode, for the snapshot (#658).

The mode is replayed over the archive alone: every station-local day from
the archive's first reading to today, each judged by the one wet-hour
helper (`lookout.core.wet_hours`) and fed to the pure replay
(`lookout.core.wet_mode`). Nothing is stored between runs, so a rebuild or
a first deploy gives the same mode. A day's mode is decided by the days
before it, so today's rain never flips today's mode and the mode holds
through the day. Every station uses Lookout Post's thresholds, the
replay's constants (spec #647).

Functions:
- wet_mode(stitched, clock): The `WetMode` field.
"""

import pandas as pd

from lookout.core.wet_hours import daily_wet_hours, hour_start, judge_hours
from lookout.core.wet_mode import wet_mode as replay
from lookout.snapshot.clock import StationClock
from lookout.snapshot.model import WetMode
from lookout.utils.dateutc import POST_2020_MS


def wet_mode(stitched: pd.DataFrame, clock: StationClock) -> WetMode:
    """
    Today's mode, replayed over every day of the archive.

    :param stitched: The archive with the live reading, any order.
    :param clock: The station clock.
    :return: The mode and what decided it.
    """
    end_ms = hour_start(clock.now_ms, clock.tz)
    first_ms = int(stitched["dateutc"].min()) if not stitched.empty else end_ms
    # A row before the floor is invalid (ADR-0001): it never stretches the span.
    start_ms = max(first_ms, POST_2020_MS)
    hours = judge_hours(stitched, clock.tz, min(start_ms, end_ms), end_ms)
    days = daily_wet_hours(hours)
    index = pd.DatetimeIndex(pd.to_datetime(days["day"]), name="day")
    series = days[["wet_hours", "countable_hours"]].set_axis(index)
    today = pd.Timestamp(clock.today)
    # Today may hold no judged hour yet (the build's first hour, or an
    # archive that stopped): a day with none counts nothing.
    if today not in series.index:
        series.loc[today] = (0, 0)
    modes = replay(series.sort_index())
    mode, decided_by = modes.loc[today, ["mode", "decided_by"]]
    return WetMode(mode=mode, decided_by=decided_by)
