"""
The station clock: the build time and the station's zone, handed together
to every snapshot section.
"""

from datetime import date, datetime

import pandas as pd
import pytest

from lookout.snapshot.clock import StationClock
from lookout.utils.dateutc import DEFAULT_TZ

# 2026-01-02T05:30Z: still Jan 1 on the Pacific coast, Jan 2 in Tokyo.
BUILD = "2026-01-02T05:30:00Z"
BUILD_MS = 1767331800000


def test_today_is_the_build_times_date_in_the_station_zone():
    assert StationClock(pd.Timestamp(BUILD), "America/Los_Angeles").today == date(
        2026, 1, 1
    )
    assert StationClock(pd.Timestamp(BUILD), "Asia/Tokyo").today == date(2026, 1, 2)


def test_it_holds_the_build_time_in_ms_and_in_the_station_zone():
    clock = StationClock(pd.Timestamp(BUILD), "America/Los_Angeles")
    assert clock.now_ms == BUILD_MS
    assert clock.local_now == pd.Timestamp("2026-01-01 21:30", tz="America/Los_Angeles")


def test_it_takes_a_datetime_and_defaults_to_the_station_zone():
    clock = StationClock(datetime.fromisoformat("2026-01-02T05:30:00+00:00"))
    assert clock.tz == DEFAULT_TZ
    assert clock.now_ms == BUILD_MS


def test_a_timezone_unaware_build_time_is_refused():
    with pytest.raises(ValueError):
        StationClock(datetime(2026, 1, 2, 5, 30))
