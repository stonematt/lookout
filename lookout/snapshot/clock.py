"""
clock.py: the station clock, the build time and the station's zone together.

`build_snapshot` makes one `StationClock` and hands it to every section, so
no section takes its own `now` or zone. The time zone math itself (local
midnight, local time, day spans) lives in `lookout.utils.dateutc`.

Classes:
- StationClock(now, tz): the build time (`now`, `now_ms`, `local_now`) and
  its station-local date (`today`) in the station's zone (`tz`).

Example:
    clock = StationClock(pd.Timestamp("2026-01-02T05:30Z"), "America/Los_Angeles")
    clock.today  # date(2026, 1, 1): still New Year's Day at the station
"""

from dataclasses import dataclass
from datetime import date, datetime
import pandas as pd

from lookout.utils.dateutc import DEFAULT_TZ


@dataclass(frozen=True)
class StationClock:
    """
    The build time and the station's zone.

    :param now: The build time, timezone-aware; a `datetime` is held as a
        `pd.Timestamp`.
    :param tz: The station's IANA zone.
    :raises ValueError: If `now` has no timezone.
    """

    now: datetime
    tz: str = DEFAULT_TZ

    def __post_init__(self) -> None:
        now = pd.Timestamp(self.now)
        if now.tzinfo is None:
            raise ValueError("StationClock needs a timezone-aware `now`")
        object.__setattr__(self, "now", now)

    @property
    def now_ms(self) -> int:
        """The build time in epoch ms."""
        return int(self.now.value // 10**6)

    @property
    def local_now(self) -> pd.Timestamp:
        """The build time in the station's zone."""
        return self.now.tz_convert(self.tz)

    @property
    def today(self) -> date:
        """The build time's station-local date."""
        return self.local_now.date()
