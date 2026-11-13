"""
The usual band by hour (#462), through `build_snapshot`.

For each station-local hour of today, the median and middle half of that
hour's temperature readings over the band's days, and which basis those days
are:

- "earlier_years": once one earlier year's ±7-day window around today is
  comparable, every such year's counted days; recent days are not used.
- "recent_days": else, with 20 or more counted days before today, the
  counted days among the 14 before today.
- "median_only": else the same recent days, the median only.

The stations are synthetic, built to end Tue 2026-01-06 noon local (or a
DST week, for the 23- and 25-hour days), each starting on a local midnight
so its first day counts.
"""

from dataclasses import replace
from datetime import date, timedelta
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import pytest

from lookout.snapshot.model import UsualBand
from tests.fixtures.archive import Gap, build_archive
from tests.snapshot_helpers import build, seasoned_station

TZ = "America/Los_Angeles"
NOW = "2026-01-06T20:02:00Z"  # Tue noon local
LAST_ROW = "2026-01-06T20:00:00Z"
TODAY = date(2026, 1, 6)


def station(start: str, end: str = LAST_ROW, gaps: Sequence[Gap] = ()) -> pd.DataFrame:
    """A station from `start` to `end`, its temperature varied day by day."""
    archive = build_archive(start, end, gaps=gaps)
    local = pd.to_datetime(archive["dateutc"], unit="ms", utc=True).dt.tz_convert(TZ)
    swing = 3.0 * np.sin(local.dt.dayofyear.to_numpy() * 0.9)
    archive["tempf"] = (archive["tempf"] + swing).round(1)
    return archive


def band(archive: pd.DataFrame, now: str = NOW) -> UsualBand:
    """The usual band `build_snapshot` writes for `archive`, no live reading."""
    return build(archive, None, now).temperature_context.usual_band


def hour_readings(archive: pd.DataFrame, days: Iterable[date], hour: int) -> np.ndarray:
    """Every temperature on `days` during local `hour`."""
    local = pd.to_datetime(archive["dateutc"], unit="ms", utc=True).dt.tz_convert(TZ)
    rows = archive[local.dt.date.isin(set(days)) & (local.dt.hour == hour)]
    return rows["tempf"].dropna().to_numpy(dtype=float)


def span(first: date, last: date) -> List[date]:
    """The days `first` to `last`, inclusive."""
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


class TestEarlierYears:
    @pytest.fixture(scope="class")
    def first_year(self) -> pd.DataFrame:
        # Today's window in 2025 is 2024-12-30..2025-01-13: covered from its
        # first day, local midnight.
        return station("2024-12-30T08:00:00Z")

    def test_the_first_earlier_year_is_the_basis(self, first_year):
        usual = band(first_year)

        assert usual.basis == "earlier_years"
        assert usual.years == (2025,)
        assert usual.days == 15
        assert usual.unit == "f"

    def test_each_hour_is_the_median_and_middle_half_of_its_window(self, first_year):
        usual = band(first_year)
        window = span(date(2024, 12, 30), date(2025, 1, 13))

        for hour in (0, 7, 15, 23):
            values = hour_readings(first_year, window, hour)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            row = usual.hours[hour]
            assert row.hour == hour
            assert row.readings == len(values) == 15 * 12
            assert row.median == round(float(median), 1)
            assert row.q1 == round(float(q1), 1)
            assert row.q3 == round(float(q3), 1)

    def test_one_day_short_of_the_window_falls_back_to_recent_days(self):
        # The archive begins a day into 2025's window: it is not comparable.
        usual = band(station("2024-12-31T08:00:00Z"))

        assert usual.basis == "recent_days"
        assert usual.years == ()
        assert usual.days == 14

    def test_an_earlier_year_wins_over_a_thin_record(self):
        # 2025's window, then nothing until Jan 4: 17 counted days in all.
        gone = Gap("2025-01-14T08:00:00Z", "2026-01-04T08:00:00Z")
        usual = band(station("2024-12-30T08:00:00Z", gaps=(gone,)))

        assert usual.basis == "earlier_years"
        assert usual.years == (2025,)
        assert usual.days == 15

    def test_every_comparable_earlier_year_newest_first(self):
        usual = seasoned_station().temperature_context.usual_band

        assert usual.basis == "earlier_years"
        assert usual.years == (2025, 2024)
        assert usual.days == 30
        # 15 local hours = 3 pm PDT, the seasoned live reading's hour.
        assert usual.hours[15].readings == 360
        assert usual.hours[15].median == 60.1


class TestRecentDays:
    def test_twenty_days_of_readings_is_recent_days_over_the_last_14(self):
        # Local Dec 17..Jan 5: exactly 20 counted days before today.
        archive = station("2025-12-17T08:00:00Z")
        usual = band(archive)
        recent = span(TODAY - timedelta(days=14), TODAY - timedelta(days=1))

        assert usual.basis == "recent_days"
        assert usual.days == 14
        values = hour_readings(archive, recent, 9)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        assert usual.hours[9].readings == len(values) == 14 * 12
        assert usual.hours[9].median == round(float(median), 1)
        assert (usual.hours[9].q1, usual.hours[9].q3) == (
            round(float(q1), 1),
            round(float(q3), 1),
        )

    def test_today_is_never_its_own_usual(self):
        archive = station("2025-12-17T08:00:00Z")
        before = band(archive)
        warmer = archive.copy()
        today_rows = warmer["dateutc"] >= int(
            pd.Timestamp("2026-01-06T08:00Z").value // 10**6
        )
        warmer.loc[today_rows, "tempf"] = 99.0

        # Only now against the usual follows today's (newest) reading.
        assert replace(band(warmer), now_delta=None) == replace(before, now_delta=None)

    def test_a_gap_day_is_not_counted_so_n_says_so(self):
        gone = Gap("2026-01-02T08:00:00Z", "2026-01-03T08:00:00Z")  # local Jan 2
        usual = band(station("2025-12-10T08:00:00Z", gaps=(gone,)))

        assert usual.basis == "recent_days"
        assert usual.days == 13

    def test_a_partial_first_day_is_not_a_day_of_readings(self):
        # From noon local Dec 17: 20 calendar days with readings, but Dec 17
        # is a low-coverage day, so 19 count.
        usual = band(station("2025-12-17T20:00:00Z"))

        assert usual.basis == "median_only"


class TestMedianOnly:
    def test_nineteen_days_of_readings_is_the_median_only(self):
        archive = station("2025-12-18T08:00:00Z")
        usual = band(archive)
        recent = span(TODAY - timedelta(days=14), TODAY - timedelta(days=1))

        assert usual.basis == "median_only"
        assert usual.years == ()
        assert usual.days == 14
        values = hour_readings(archive, recent, 9)
        assert usual.hours[9].median == round(float(np.median(values)), 1)
        assert all(h.q1 is None and h.q3 is None for h in usual.hours)

    def test_a_station_days_old_has_a_median_over_what_it_has(self):
        usual = band(station("2026-01-03T08:00:00Z"))

        assert usual.basis == "median_only"
        assert usual.days == 3
        assert usual.hours[0].readings == 3 * 12

    def test_a_first_day_station_has_no_usual_yet(self):
        usual = band(station("2026-01-06T08:00:00Z"))

        assert usual.basis == "median_only"
        assert usual.days == 0
        assert len(usual.hours) == 24
        assert all(h.median is None and h.readings == 0 for h in usual.hours)

    def test_an_empty_archive_still_has_24_hours(self):
        usual = band(station("2025-12-17T08:00:00Z").iloc[0:0])

        assert usual.basis == "median_only"
        assert [h.hour for h in usual.hours] == list(range(24))


class TestStationLocalHours:
    def test_24_hours_in_order(self):
        usual = band(station("2025-12-17T08:00:00Z"))

        assert [h.hour for h in usual.hours] == list(range(24))

    def test_the_spring_forward_day_has_no_2_am(self):
        # 2026-03-08 is 23 hours long; recent days are Feb 26..Mar 11.
        archive = station("2026-02-10T08:00:00Z", "2026-03-12T19:00:00Z")
        usual = band(archive, "2026-03-12T19:02:00Z")

        assert usual.basis == "recent_days"
        assert usual.days == 14
        assert usual.hours[2].readings == 13 * 12
        assert usual.hours[3].readings == 14 * 12

    def test_the_fall_back_day_counts_1_am_twice(self):
        # 2025-11-02 is 25 hours long; recent days are Oct 27..Nov 9.
        archive = station("2025-10-10T07:00:00Z", "2025-11-10T20:00:00Z")
        usual = band(archive, "2025-11-10T20:02:00Z")

        assert usual.basis == "recent_days"
        assert usual.days == 14
        assert usual.hours[1].readings == 15 * 12
        assert usual.hours[2].readings == 14 * 12


class TestContract:
    def test_the_band_is_in_the_temperature_context_document(self):
        doc = build(station("2025-12-17T08:00:00Z"), None, NOW).dashboard()
        usual = doc["temperature_context"]["usual_band"]

        assert set(usual) == {"basis", "unit", "years", "days", "hours", "now_delta"}
        assert set(usual["hours"][0]) == {"hour", "median", "q1", "q3", "readings"}
        assert doc["schema_version"] == 4
