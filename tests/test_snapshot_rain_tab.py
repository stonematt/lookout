"""
The Rain tab's section of the snapshot (#83; spec #66 seam 1).

`build_snapshot(...).rain_tab`: window totals, daily and hourly rain, dry
spells and the water-year curves. Every rain total is an archive sum with
today's live total in it; every value over a window follows the gap rules;
years run from Oct 1. Archives are synthetic five-minute stations from
`tests.fixtures.archive`, and every `now` is fixed.
"""

from datetime import date
from typing import Dict, Optional

import pandas as pd
import pytest

from lookout.snapshot import Station, build_snapshot
from lookout.snapshot.model import Snapshot
from tests.fixtures.archive import Gap, RainSpell, build_archive
from tests.snapshot_helpers import raining_station

STATION = Station(mac="00:11:22:33:44:55", name="Fixture Station")


def ms(when: str) -> int:
    """Epoch ms for an ISO-8601 UTC time."""
    return int(pd.Timestamp(when).value // 10**6)


def build(archive: pd.DataFrame, now: str, live: Optional[Dict] = None) -> Snapshot:
    """`build_snapshot` for the fixture station, no catalogs."""
    return build_snapshot(
        archive=archive,
        live=live,
        event_catalog=None,
        energy_catalog=None,
        station=STATION,
        now=pd.Timestamp(now),
    )


def windows(snapshot: Snapshot) -> Dict[str, Optional[float]]:
    """The Rain tab's window totals by key."""
    return {w.key: w.total_in for w in snapshot.rain_tab.windows}


def curve(snapshot: Snapshot, year: int):
    """One water year's curve."""
    return next(c for c in snapshot.rain_tab.water_years if c.water_year == year)


def approx(value: float):
    """Rain totals to a thousandth of an inch."""
    return pytest.approx(value, abs=1e-3)


# Jan 2026, local midnight is 08:00Z.
JAN_RAIN = [
    RainSpell("2026-01-02T18:00:00Z", 2, 0.1),  # Jan 2: 0.2 in
    RainSpell("2026-01-05T18:00:00Z", 3, 0.1),  # Jan 5: 0.3 in
    RainSpell("2026-01-09T18:00:00Z", 1, 0.25),  # Jan 9: 0.25 in
]


class TestOneNumberForToday:
    def test_today_is_the_live_total_everywhere_on_the_tab(self):
        snapshot = raining_station()
        tab = snapshot.rain_tab

        today = snapshot.current.rain_today_in
        assert today == 0.3  # the live reading, ahead of the archive's 0.18
        assert windows(snapshot)["today"] == today
        assert tab.recent.rain_in[-1] == today
        assert sum(h for h in tab.recent.hourly_in[-1] if h) == approx(today)

    def test_the_windows_follow_the_gap_status_order(self):
        snapshot = raining_station()

        assert [w.key for w in snapshot.rain_tab.windows] == [
            s.key for s in snapshot.gaps.windows
        ]


class TestWindowTotals:
    def test_a_window_total_is_the_sum_of_its_daily_totals(self):
        archive = build_archive(
            "2025-12-01T08:00:00Z", "2026-01-10T08:00:00Z", rain=JAN_RAIN
        )

        totals = windows(build(archive, "2026-01-10T07:59:00Z"))

        assert totals["today"] == approx(0.25)
        assert totals["yesterday"] == 0.0
        assert totals["7d"] == approx(0.55)
        assert totals["30d"] == approx(0.75)

    def test_a_window_that_cannot_be_compared_has_no_total(self):
        snapshot = raining_station()

        assert snapshot.gaps.windows[2].status == "invalid"  # 7d, before archive
        assert windows(snapshot)["7d"] is None
        assert windows(snapshot)["water_year"] is None

    def test_a_window_missing_a_day_is_prorated_over_the_rest(self):
        # 30 days, Jan 5 missing: the 0.5 in counted over 29 days, scaled.
        archive = build_archive(
            "2025-12-01T08:00:00Z",
            "2026-01-10T08:00:00Z",
            rain=JAN_RAIN,
            gaps=[Gap("2026-01-05T08:00:00Z", "2026-01-06T08:00:00Z")],
        )

        snapshot = build(archive, "2026-01-10T07:59:00Z")

        assert snapshot.gaps.windows[3].status == "prorated"
        assert windows(snapshot)["30d"] == approx(0.45 * 30 / 29)
        assert windows(snapshot)["7d"] is None  # 1 day of 7 missing: invalid


class TestRecentDays:
    def test_a_day_not_in_the_record_has_no_total_and_says_why(self):
        archive = build_archive(
            "2025-10-01T07:00:00Z",
            "2026-01-10T08:00:00Z",
            rain=JAN_RAIN,
            gaps=[Gap("2026-01-05T08:00:00Z", "2026-01-06T08:00:00Z")],
        )

        recent = build(archive, "2026-01-10T07:59:00Z").rain_tab.recent

        assert recent.first_day == "2025-10-12"
        assert len(recent.rain_in) == len(recent.hourly_in) == 90
        jan_5 = (date(2026, 1, 5) - date(2025, 10, 12)).days
        assert recent.rain_in[jan_5] is None
        assert recent.reason[jan_5] == "gap"
        assert recent.hourly_in[jan_5] == (None,) * 24
        assert recent.rain_in[jan_5 - 3] == approx(0.2)
        assert recent.reason[jan_5 - 3] is None

    def test_hours_still_to_come_today_are_not_known(self):
        archive = build_archive(
            "2025-12-01T08:00:00Z", "2026-01-09T19:30:00Z", rain=JAN_RAIN
        )

        recent = build(archive, "2026-01-09T19:30:00Z").rain_tab.recent

        # 11:30 local: hours 0-11 are in, from noon on they are still to come.
        assert recent.hourly_in[-1][10] == approx(0.25)
        assert recent.hourly_in[-1][11] == 0.0
        assert recent.hourly_in[-1][12:] == (None,) * 12

    def test_an_hour_with_no_readings_is_not_known(self):
        # Jan 7, 2-4 am local, no rows: the day still counts (over 80% of
        # it is in), but those two hours are unknown, not dry (#147).
        archive = build_archive(
            "2025-10-01T07:00:00Z",
            "2026-01-10T08:00:00Z",
            rain=JAN_RAIN,
            gaps=[Gap("2026-01-07T10:00:00Z", "2026-01-07T12:00:00Z")],
        )

        recent = build(archive, "2026-01-10T07:59:00Z").rain_tab.recent
        jan_7 = (date(2026, 1, 7) - date.fromisoformat(recent.first_day)).days

        assert recent.reason[jan_7] is None
        assert recent.rain_in[jan_7] == 0.0
        assert recent.hourly_in[jan_7][1] == 0.0
        assert recent.hourly_in[jan_7][2:4] == (None, None)
        assert recent.hourly_in[jan_7][4] == 0.0

    def test_an_hour_whose_counter_has_no_reading_is_not_known(self):
        # Jan 7, 2 am local: the rows are there, the rain counter is not.
        archive = build_archive(
            "2025-10-01T07:00:00Z", "2026-01-10T08:00:00Z", rain=JAN_RAIN
        )
        hour = (archive["dateutc"] >= ms("2026-01-07T10:00:00Z")) & (
            archive["dateutc"] < ms("2026-01-07T11:00:00Z")
        )
        archive.loc[hour, "dailyrainin"] = float("nan")

        recent = build(archive, "2026-01-10T07:59:00Z").rain_tab.recent
        jan_7 = (date(2026, 1, 7) - date.fromisoformat(recent.first_day)).days

        assert recent.hourly_in[jan_7][2] is None
        assert recent.hourly_in[jan_7][3] == 0.0

    def test_wet_hours_count_known_hours_only(self):
        # Jan 5 rains 10 am to 1 pm local; the 11 am hour has no readings.
        # Its rain shows in the noon hour, the first reading after the gap.
        archive = build_archive(
            "2025-10-01T07:00:00Z",
            "2026-01-10T08:00:00Z",
            rain=JAN_RAIN,
            gaps=[Gap("2026-01-05T19:00:00Z", "2026-01-05T20:00:00Z")],
        )

        recent = build(archive, "2026-01-10T07:59:00Z").rain_tab.recent
        jan_5 = (date(2026, 1, 5) - date.fromisoformat(recent.first_day)).days

        assert recent.hourly_in[jan_5][11] is None
        assert recent.hourly_in[jan_5][12] == approx(0.2)
        assert recent.rain_in[jan_5] == approx(0.3)
        assert recent.wet_hours == 5  # 2 + 2 known + 1

    def test_the_wettest_hour_and_wet_hours_cover_the_window(self):
        archive = build_archive(
            "2025-10-01T07:00:00Z", "2026-01-10T08:00:00Z", rain=JAN_RAIN
        )

        recent = build(archive, "2026-01-10T07:59:00Z").rain_tab.recent

        assert recent.wet_hours == 6  # 2 + 3 + 1
        assert (recent.wettest_hour.day, recent.wettest_hour.hour) == (
            "2026-01-09",
            10,
        )
        assert recent.wettest_hour.rain_in == approx(0.25)

    def test_an_invalid_window_has_no_wet_hours(self):
        recent = raining_station().rain_tab.recent

        assert recent.wet_hours is None
        assert recent.wettest_hour is None


class TestDrySpells:
    def test_the_longest_spell_on_record(self):
        # Spells: Jan 3-4 (2 d) and Jan 6-8 (3 d), Jan 10-14 (5 d).
        rain = [
            *JAN_RAIN,
            RainSpell("2026-01-15T18:00:00Z", 1, 0.1),
        ]
        archive = build_archive(
            "2025-12-01T08:00:00Z", "2026-01-17T08:00:00Z", rain=rain
        )

        spells = build(archive, "2026-01-17T07:59:00Z").rain_tab.dry_spells

        assert (spells.longest.first_day, spells.longest.days) == ("2026-01-10", 5)

    def test_a_spell_across_a_gap_is_not_counted(self):
        archive = build_archive(
            "2025-12-01T08:00:00Z",
            "2026-01-10T08:00:00Z",
            rain=JAN_RAIN,
            gaps=[Gap("2026-01-07T08:00:00Z", "2026-01-08T08:00:00Z")],
        )

        spells = build(archive, "2026-01-10T07:59:00Z").rain_tab.dry_spells

        # Jan 6-8 holds the missing Jan 7: only Jan 3-4 is left.
        assert (spells.longest.first_day, spells.longest.days) == ("2026-01-03", 2)

    def test_water_year_rain_days_and_wettest_day(self):
        archive = build_archive(
            "2025-10-01T07:00:00Z", "2026-01-10T08:00:00Z", rain=JAN_RAIN
        )

        spells = build(archive, "2026-01-10T07:59:00Z").rain_tab.dry_spells

        assert spells.water_year_rain_days == 3
        assert spells.water_year_wettest.day == "2026-01-05"
        assert spells.water_year_wettest.rain_in == approx(0.3)


class TestWaterYears:
    """Year-over-year accumulation starts each year on Oct 1."""

    RAIN = [
        RainSpell("2024-12-10T18:00:00Z", 2, 0.5),  # WY2025: 1.0 in, Dec 10
        RainSpell("2025-09-30T18:00:00Z", 1, 0.4),  # WY2025's last day
        RainSpell("2025-10-01T18:00:00Z", 1, 0.2),  # WY2026's first day
        RainSpell("2025-12-31T18:00:00Z", 1, 0.3),  # across New Year:
        RainSpell("2026-01-01T18:00:00Z", 1, 0.1),  # the same year
    ]

    @pytest.fixture(scope="class")
    def snapshot(self) -> Snapshot:
        archive = build_archive(
            "2024-09-20T07:00:00Z", "2026-01-10T08:00:00Z", rain=self.RAIN
        )
        return build(archive, "2026-01-10T07:59:00Z")

    def test_each_curve_starts_on_oct_1(self, snapshot):
        years = [(c.water_year, c.first_day) for c in snapshot.rain_tab.water_years]

        assert years == [
            (2024, "2023-10-01"),
            (2025, "2024-10-01"),
            (2026, "2025-10-01"),
        ]
        assert curve(snapshot, 2026).cumulative_in[0] == approx(0.2)

    def test_sep_30_closes_one_year_and_oct_1_opens_the_next(self, snapshot):
        wy2025 = curve(snapshot, 2025)

        assert len(wy2025.cumulative_in) == 365
        assert wy2025.cumulative_in[-2] == approx(1.0)
        assert wy2025.total_in == approx(1.4)
        assert wy2025.status.status == "valid"

    def test_the_new_year_does_not_reset_the_curve(self, snapshot):
        wy2026 = curve(snapshot, 2026)
        dec_31 = (date(2025, 12, 31) - date(2025, 10, 1)).days

        assert wy2026.cumulative_in[dec_31] == approx(0.5)
        assert wy2026.cumulative_in[dec_31 + 1] == approx(0.6)
        assert wy2026.total_in == approx(0.6)

    def test_years_compare_at_the_same_place(self, snapshot):
        tab = snapshot.rain_tab

        assert tab.day_index == (date(2026, 1, 9) - date(2025, 10, 1)).days
        assert len(curve(snapshot, 2026).cumulative_in) == tab.day_index + 1
        assert curve(snapshot, 2026).to_date_in == approx(0.6)
        assert curve(snapshot, 2025).to_date_in == approx(1.0)

    def test_the_water_year_total_is_the_water_year_window(self, snapshot):
        assert windows(snapshot)["water_year"] == curve(snapshot, 2026).total_in

    def test_a_year_the_archive_began_inside_is_not_compared(self, snapshot):
        wy2024 = curve(snapshot, 2024)

        assert wy2024.status.reason == "before_archive"
        assert set(wy2024.cumulative_in) == {None}
        assert wy2024.total_in is None
        assert wy2024.to_date_in is None

    def test_a_missing_day_makes_the_curve_invalid_then_prorated(self):
        archive = build_archive(
            "2025-09-20T07:00:00Z",
            "2026-01-10T08:00:00Z",
            rain=self.RAIN[2:],
            gaps=[Gap("2025-10-03T07:00:00Z", "2025-10-04T07:00:00Z")],
        )

        wy2026 = curve(build(archive, "2026-01-10T07:59:00Z"), 2026)

        # Oct 3 missing: Oct 1-3 to Oct 1-9 miss over 10% of their days.
        assert wy2026.cumulative_in[1] == approx(0.2)
        assert wy2026.cumulative_in[2:9] == (None,) * 7
        assert wy2026.cumulative_in[9] == approx(0.2 * 10 / 9)
        assert wy2026.prorated[9] and not wy2026.prorated[1]
        assert wy2026.status.status == "prorated"
