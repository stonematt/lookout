"""
Wet hours and countable hours over the last 7 and 30 days (#657, spec #647).

GLOSSARY: a **countable hour** is a station-local clock hour that has
readings and no gap over 30 minutes between them, the gap from the reading
before the hour to its first reading included; a **wet hour** is a
countable hour with at least one tip (0.01 in). Hours are cut in the
station zone (ADR-0002, ADR-0005), and each reading's rain comes from the
counter helper, so a late or 21:00 counter reset never makes a false wet
hour (#654).

Through `build_snapshot`; archives are synthetic five-minute stations from
`tests.fixtures.archive`, local days Pacific. The build is at 12:10 local
on Sat Jan 10, 2026, so the windows end with 11:00-12:00, the last whole
hour.
"""

from typing import Sequence

import pandas as pd

from lookout.snapshot import Snapshot
from lookout.snapshot.model import WetHoursDay
from tests.fixtures.archive import Gap, RainSpell, build_archive
from tests.snapshot_helpers import build, live_row

NOW = "2026-01-10T20:10:00Z"  # 12:10 local
START = "2025-11-20T08:00:00Z"
FIVE_MINUTES = 5 / 60


def tip(when: str) -> RainSpell:
    """One tip of the gauge, 0.01 in, at the reading at `when` (UTC)."""
    return RainSpell(when, FIVE_MINUTES, 0.12)


def snapshot_of(
    rain: Sequence[RainSpell] = (),
    gaps: Sequence[Gap] = (),
    daily_reset: pd.Timedelta = pd.Timedelta(0),
    now: str = NOW,
) -> Snapshot:
    """`build_snapshot` at `now`; the live reading is the archive's newest row."""
    archive = build_archive(START, now, rain=rain, gaps=gaps, daily_reset=daily_reset)
    newest = pd.Timestamp(int(archive["dateutc"].max()), unit="ms", tz="UTC")
    return build(archive, live_row(archive, newest.isoformat()), now)


def strip_day(snapshot: Snapshot, iso: str) -> WetHoursDay:
    """The strip's entry for one station-local day."""
    return next(d for d in snapshot.wet_hours.days if d.day == iso)


class TestWetHours:
    def test_a_dry_week_has_every_hour_countable_and_none_wet(self):
        week = snapshot_of().wet_hours.last_7_days

        assert (week.days, week.hours) == (7, 168)
        assert week.wet_hours == 0
        assert week.countable_hours == 168
        assert week.missing_hours == 0
        assert week.missing_days == 0

    def test_a_drizzle_with_a_tip_every_40_minutes_is_wet_each_hour(self):
        # Jan 8, 10:50 to 13:30 local: tips 40 minutes apart.
        tips = [
            tip(t)
            for t in (
                "2026-01-08T18:50:00Z",
                "2026-01-08T19:30:00Z",
                "2026-01-08T20:10:00Z",
                "2026-01-08T20:50:00Z",
                "2026-01-08T21:30:00Z",
            )
        ]

        snapshot = snapshot_of(rain=tips)

        assert snapshot.wet_hours.last_7_days.wet_hours == 4
        assert strip_day(snapshot, "2026-01-08").wet_hours == 4

    def test_a_single_tip_makes_exactly_one_wet_hour(self):
        snapshot = snapshot_of(rain=[tip("2026-01-08T18:50:00Z")])

        assert snapshot.wet_hours.last_7_days.wet_hours == 1
        assert snapshot.wet_hours.last_30_days.wet_hours == 1

    def test_steady_rain_is_wet_for_each_hour_it_falls(self):
        snapshot = snapshot_of(rain=[RainSpell("2026-01-08T18:00:00Z", 3, 0.05)])

        assert snapshot.wet_hours.last_7_days.wet_hours == 3

    def test_rain_outside_the_window_is_not_counted(self):
        # Jan 2, eight days back: in the 30 days, not the 7.
        snapshot = snapshot_of(rain=[tip("2026-01-02T18:50:00Z")])

        assert snapshot.wet_hours.last_7_days.wet_hours == 0
        assert snapshot.wet_hours.last_30_days.wet_hours == 1


class TestCountableHours:
    def test_a_gap_over_30_minutes_makes_its_hour_not_countable(self):
        # Jan 8, 14:00 local: readings at 14:00 and 14:40, 40 minutes apart.
        gap = Gap("2026-01-08T22:05:00Z", "2026-01-08T22:40:00Z")

        week = snapshot_of(gaps=[gap]).wet_hours.last_7_days

        assert week.countable_hours == 167
        assert week.missing_hours == 1
        assert week.countable_hours + week.missing_hours == week.hours

    def test_a_gap_of_30_minutes_keeps_its_hour(self):
        # Readings at 14:00 and 14:30: a 30-minute gap is not over 30.
        gap = Gap("2026-01-08T22:05:00Z", "2026-01-08T22:30:00Z")

        assert snapshot_of(gaps=[gap]).wet_hours.last_7_days.countable_hours == 168

    def test_the_gap_before_an_hours_first_reading_counts_in_that_hour(self):
        # Readings at 14:30 and 15:10: 14:00 keeps its hour, 15:00 loses it.
        gap = Gap("2026-01-08T22:35:00Z", "2026-01-08T23:10:00Z")

        snapshot = snapshot_of(gaps=[gap])

        assert snapshot.wet_hours.last_7_days.missing_hours == 1
        assert strip_day(snapshot, "2026-01-08").countable_hours == 23

    def test_a_missing_hour_lowers_the_countable_hours_never_the_wet(self):
        # A tip at 10:50 local, then no readings from 11:00 to 12:00: 11:00
        # has none, and 12:00's first follows a 65-minute gap.
        rain = [tip("2026-01-08T18:50:00Z")]
        gap = Gap("2026-01-08T19:00:00Z", "2026-01-08T20:00:00Z")

        week = snapshot_of(rain=rain, gaps=[gap]).wet_hours.last_7_days

        assert week.wet_hours == 1
        assert week.countable_hours == 166

    def test_rain_counted_across_a_long_gap_is_never_a_wet_hour(self):
        # Offline 11:00-13:00 local while it rains: the counter is higher
        # after the gap, but the hour after it is not countable either.
        rain = [RainSpell("2026-01-08T19:00:00Z", 2, 0.06)]
        gap = Gap("2026-01-08T19:00:00Z", "2026-01-08T21:00:00Z")

        week = snapshot_of(rain=rain, gaps=[gap]).wet_hours.last_7_days

        assert week.wet_hours == 0
        assert week.missing_hours == 3

    def test_a_missing_day_is_null_in_the_strip_and_named_in_each_window(self):
        # All of Jan 5 local is missing, and an hour on Jan 7.
        gaps = [
            Gap("2026-01-05T08:00:00Z", "2026-01-06T08:00:00Z"),
            Gap("2026-01-07T19:00:00Z", "2026-01-07T20:00:00Z"),
        ]

        snapshot = snapshot_of(gaps=gaps)

        day = strip_day(snapshot, "2026-01-05")
        assert (day.wet_hours, day.countable_hours) == (None, None)
        # 24 h of Jan 5 and Jan 6 00:00 after it; 11:00 and 12:00 on Jan 7.
        assert snapshot.wet_hours.last_7_days.missing_hours == 27
        assert snapshot.wet_hours.last_7_days.missing_days == 1
        assert snapshot.wet_hours.last_30_days.missing_days == 1

    def test_the_first_hour_back_from_a_long_outage_is_not_countable(self):
        # Offline 48 days, back at 10:00 local on Jan 8: nothing read says
        # how long the gap before 10:00's first reading was.
        gap = Gap(START, "2026-01-08T18:00:00Z")

        snapshot = snapshot_of(gaps=[gap])

        assert strip_day(snapshot, "2026-01-08").countable_hours == 13

    def test_hours_since_the_newest_reading_are_missing(self):
        # The station stops at 08:00 local; the build is at 12:10.
        gap = Gap("2026-01-10T16:00:00Z", NOW)

        week = snapshot_of(gaps=[gap]).wet_hours.last_7_days

        assert week.missing_hours == 4


class TestTheStrip:
    def test_it_holds_the_last_30_station_local_days_oldest_first(self):
        days = snapshot_of().wet_hours.days

        assert len(days) == 30
        assert days[0].day == "2025-12-12"
        assert days[-1].day == "2026-01-10"
        assert days[1].countable_hours == 24

    def test_today_counts_its_whole_hours_so_far(self):
        today = strip_day(snapshot_of(), "2026-01-10")

        assert today.countable_hours == 12

    def test_hours_are_cut_in_the_station_zone(self):
        # 23:30 local on Jan 8 is 07:30 UTC on Jan 9.
        snapshot = snapshot_of(rain=[tip("2026-01-09T07:30:00Z")])

        assert strip_day(snapshot, "2026-01-08").wet_hours == 1
        assert strip_day(snapshot, "2026-01-09").wet_hours == 0


class TestTheMidnightFix:
    """A counter reset never makes a false wet hour (#654)."""

    def test_a_reset_five_minutes_after_midnight(self):
        # Lookout Post: 0.3 in on Jan 8, 10:00-13:00 local; Jan 9 is dry.
        snapshot = snapshot_of(
            rain=[RainSpell("2026-01-08T18:00:00Z", 3, 0.1)],
            daily_reset=pd.Timedelta(minutes=5),
        )

        assert strip_day(snapshot, "2026-01-08").wet_hours == 3
        assert strip_day(snapshot, "2026-01-09").wet_hours == 0
        assert snapshot.wet_hours.last_7_days.wet_hours == 3

    def test_a_reset_at_nine_pm(self):
        # Salmon Run: the counter resets at 21:00 local; 0.2 in on Jan 8,
        # 10:00-12:00 local, so the counter holds it at the reset.
        snapshot = snapshot_of(
            rain=[RainSpell("2026-01-08T18:00:00Z", 2, 0.1)],
            daily_reset=pd.Timedelta(hours=-3),
        )

        assert strip_day(snapshot, "2026-01-08").wet_hours == 2
        assert snapshot.wet_hours.last_7_days.wet_hours == 2


class TestTheDocument:
    def test_the_dashboard_carries_the_wet_hours(self):
        document = snapshot_of(rain=[tip("2026-01-08T18:50:00Z")]).dashboard()

        wet = document["wet_hours"]
        assert wet["last_7_days"] == {
            "days": 7,
            "hours": 168,
            "wet_hours": 1,
            "countable_hours": 168,
            "missing_hours": 0,
            "missing_days": 0,
        }
        assert wet["days"][-1] == {
            "day": "2026-01-10",
            "wet_hours": 0,
            "countable_hours": 12,
        }

    def test_the_context_document_leaves_them_out(self):
        assert "wet_hours" not in snapshot_of().context()
