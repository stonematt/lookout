"""
The Rain tab's water-year verdict and window strip (#640, spec #647).

The verdict sets this water year's rain to date against the usual rain to
date: "wetter", "drier" or "about the same" (within 15%), and none before
an earlier water year covers the date. The window strip sets each window's
rain against its usual range from the same window in earlier years, marked
"above usual range" or "below usual range" when outside it. Both go through
`build_snapshot` on a simulated archive.
"""

from datetime import date
from typing import Dict, Optional

import pandas as pd
import pytest

from lookout.snapshot import Station, build_snapshot
from lookout.snapshot.model import Snapshot, StripWindow
from tests.fixtures.archive import STATION_TZ, Gap, RainSpell, build_archive
from tests.snapshot_helpers import raining_station, seasoned_station

STATION = Station(mac="00:11:22:33:44:55", name="Fixture Station", tz=STATION_TZ)


def build(archive: pd.DataFrame, now: str) -> Snapshot:
    """`build_snapshot` for the fixture station, no live reading, no catalogs."""
    return build_snapshot(
        archive=archive,
        live=None,
        event_catalog=None,
        energy_catalog=None,
        station=STATION,
        now=pd.Timestamp(now),
    )


def strip(snapshot: Snapshot) -> Dict[str, StripWindow]:
    """The window strip by key."""
    return {w.key: w for w in snapshot.rain_tab.window_strip}


def approx(value: float):
    """Rain totals to a thousandth of an inch."""
    return pytest.approx(value, abs=1e-3)


class TestVerdict:
    """
    One earlier October in the record (WY2025, 1.00 in by Oct 20) and this
    one: the archive holds Sep 25 - Oct 31 2024 and Sep 25 2025 on, so the
    earlier year covers Oct 1 - Oct 20 and the usual is 1.00 in.
    """

    START = "2024-09-25T07:00:00Z"
    END = "2025-10-21T07:00:00Z"
    NOW = "2025-10-21T06:59:00Z"  # Mon Oct 20, 23:59 PDT
    BETWEEN = Gap("2024-11-01T07:00:00Z", "2025-09-25T07:00:00Z")
    EARLIER = RainSpell("2024-10-10T18:00:00Z", 1, 1.0)

    def build(self, inches: float, start: Optional[str] = None) -> Snapshot:
        """This station with `inches` of rain on Oct 10 2025."""
        archive = build_archive(
            start or self.START,
            self.END,
            rain=[self.EARLIER, RainSpell("2025-10-10T18:00:00Z", 1, inches)],
            gaps=[self.BETWEEN],
        )
        return build(archive, self.NOW)

    @pytest.mark.parametrize(
        "inches, percent, verdict",
        [
            (1.16, 116, "wetter"),
            (1.15, 115, "about the same"),
            (1.0, 100, "about the same"),
            (0.85, 85, "about the same"),
            (0.84, 84, "drier"),
        ],
    )
    def test_within_15_percent_of_usual_is_about_the_same(
        self, inches, percent, verdict
    ):
        usual = self.build(inches).rain_tab.usual_to_date

        assert usual.years == 1
        assert usual.usual_in == approx(1.0)
        assert usual.percent == percent
        assert usual.verdict == verdict

    def test_no_verdict_before_an_earlier_year_covers_the_date(self):
        # The record begins Oct 10 2024: WY2025 does not reach back to Oct 1.
        snapshot = self.build(1.5, start="2024-10-10T07:00:00Z")

        assert snapshot.rain_tab.usual_to_date is None

    def test_a_young_station_has_no_verdict(self):
        assert raining_station().rain_tab.usual_to_date is None

    def test_the_seasoned_station_is_wetter_than_its_earlier_year(self):
        # WY2026 has 145% of WY2025's 0.304 in by this date.
        usual = seasoned_station().rain_tab.usual_to_date

        assert usual.percent == 145
        assert usual.verdict == "wetter"


class TestWindowStrip:
    """
    Five Octobers, WY2022 to this one, with Nov - Aug missing each year, so
    the windows within October have four earlier years and the long ones
    none.
    """

    NOW = "2025-10-21T06:59:00Z"  # Mon Oct 20, 23:59 PDT
    YEARS = (2021, 2022, 2023, 2024)

    @pytest.fixture(scope="class")
    def snapshot(self) -> Snapshot:
        gaps = [
            Gap(f"{y}-11-01T07:00:00Z", f"{y + 1}-09-01T07:00:00Z") for y in self.YEARS
        ]
        rain = []
        for i, y in enumerate(self.YEARS):
            # Oct 15: 0.2 .. 0.8 in; yesterday, Oct 19: 0.1 .. 0.4 in;
            # today, Oct 20: 0.1 in. So 7 days: 0.4, 0.7, 1.0, 1.3 in.
            rain.append(RainSpell(f"{y}-10-15T18:00:00Z", 1, 0.2 * (i + 1)))
            rain.append(RainSpell(f"{y}-10-19T18:00:00Z", 1, 0.1 * (i + 1)))
            rain.append(RainSpell(f"{y}-10-20T18:00:00Z", 1, 0.1))
        rain.append(RainSpell("2025-10-15T18:00:00Z", 1, 1.5))
        archive = build_archive(
            "2021-09-01T07:00:00Z", "2025-10-21T07:00:00Z", rain=rain, gaps=gaps
        )
        return build(archive, self.NOW)

    def test_the_strip_follows_the_windows(self, snapshot):
        keys = [w.key for w in snapshot.rain_tab.windows]

        assert [w.key for w in snapshot.rain_tab.window_strip] == keys

    def test_a_window_above_its_percentile_band_is_marked_above(self, snapshot):
        week = strip(snapshot)["7d"]

        assert week.rain_in == approx(1.5)
        assert week.years == 4
        assert [e.rain_in for e in week.earlier] == [
            approx(0.4),
            approx(0.7),
            approx(1.0),
            approx(1.3),
        ]
        assert week.earlier[0].last_day == "2021-10-20"
        assert week.band == "percentile"
        # Linear percentiles: 0.4 + 0.75 * 0.3 and 1.0 + 0.25 * 0.3.
        assert (week.low_in, week.high_in) == (approx(0.625), approx(1.075))
        assert week.mark == "above usual range"

    def test_a_complete_window_below_its_range_is_marked_below(self, snapshot):
        yesterday = strip(snapshot)["yesterday"]

        assert yesterday.rain_in == 0.0
        assert yesterday.mark == "below usual range"

    def test_a_window_still_running_is_never_marked_below(self, snapshot):
        # Today has had no rain yet; earlier Oct 20s had 0.1 in by midnight.
        today = strip(snapshot)["today"]

        assert today.rain_in == 0.0
        assert (today.low_in, today.high_in) == (approx(0.1), approx(0.1))
        assert today.mark is None

    def test_a_window_no_earlier_year_can_read_has_no_range(self, snapshot):
        year = strip(snapshot)["365d"]

        assert year.years == 0
        assert year.earlier == ()
        assert (year.low_in, year.high_in, year.band, year.mark) == (None,) * 4

    def test_the_water_year_rests_on_the_years_the_verdict_does(self, snapshot):
        water_year = strip(snapshot)["water_year"]
        usual = snapshot.rain_tab.usual_to_date

        assert water_year.years == usual.years == 4
        assert usual.water_years == (2022, 2023, 2024, 2025)
        assert water_year.earlier[-1].last_day == "2024-10-20"

    def test_a_young_station_carries_its_values_and_no_range(self):
        snapshot = raining_station()
        windows = {w.key: w.total_in for w in snapshot.rain_tab.windows}

        for window in snapshot.rain_tab.window_strip:
            assert window.rain_in == windows[window.key]
            assert window.years == 0
            assert window.low_in is None and window.mark is None

    def test_one_earlier_year_is_its_plain_value(self):
        snapshot = TestVerdict().build(0.5)
        water_year = strip(snapshot)["water_year"]

        assert water_year.years == 1
        assert water_year.band is None
        assert (water_year.low_in, water_year.high_in) == (approx(1.0), approx(1.0))
        assert water_year.mark is None  # still running: never "below"
        assert water_year.earlier[0].last_day == date(2024, 10, 20).isoformat()
