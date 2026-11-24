"""
Wet mode in the snapshot, and the wet-or-dry tile's title (#658, spec #647).

GLOSSARY: wet mode starts when the last 7 days' wet share reaches 15% and
ends when the last 14 days' share falls below 2%; when fewer than half the
last 14 days' hours are countable, the calendar decides (Nov 1 to Apr 30 is
wet). The snapshot carries the mode, what decided it, and the tile title
that follows it ("Wet hours" or "Dry spell"). The mode is replayed over the
archive alone, so a rebuild gives the same mode.

Through `build_snapshot`; archives are synthetic five-minute stations from
`tests.fixtures.archive`, local days Pacific.
"""

from typing import Sequence

import pandas as pd

from lookout.snapshot import Snapshot
from tests.fixtures.archive import Gap, RainSpell, build_archive
from tests.snapshot_helpers import build, live_row


def snapshot_of(
    start: str,
    now: str,
    rain: Sequence[RainSpell] = (),
    gaps: Sequence[Gap] = (),
) -> Snapshot:
    """`build_snapshot` at `now`; the live reading is the archive's newest row."""
    archive = build_archive(start, now, rain=rain, gaps=gaps)
    newest = pd.Timestamp(int(archive["dateutc"].max()), unit="ms", tz="UTC")
    return build(archive, live_row(archive, newest.isoformat()), now)


def daily_rain(first: str, days: int, hours: float) -> Sequence[RainSpell]:
    """Steady rain for `hours` from 10:00 local, each of `days` days from `first`."""
    return [
        RainSpell(f"{day.date()}T18:00:00Z", hours, 0.05)
        for day in pd.date_range(first, periods=days, freq="D")
    ]


# Summer: the calendar says dry. Rain 4 h a day for a week is 16.7% wet.
SUMMER_START = "2026-06-01T07:00:00Z"
SUMMER_NOW = "2026-07-01T19:10:00Z"  # 12:10 local, Wed Jul 1
WET_WEEK = daily_rain("2026-06-23", 7, 4)


class TestTheWeatherDecides:
    def test_a_wet_week_in_summer_is_wet_mode(self):
        snapshot = snapshot_of(SUMMER_START, SUMMER_NOW, rain=WET_WEEK)

        assert snapshot.wet_mode.mode == "wet"
        assert snapshot.wet_mode.decided_by == "weather"
        assert snapshot.tile_titles.wet_or_dry == "Wet hours"

    def test_a_dry_summer_month_is_dry_mode(self):
        snapshot = snapshot_of(SUMMER_START, SUMMER_NOW)

        assert snapshot.wet_mode.mode == "dry"
        assert snapshot.wet_mode.decided_by == "weather"
        assert snapshot.tile_titles.wet_or_dry == "Dry spell"

    def test_a_dry_midwinter_month_is_dry_mode_against_the_calendar(self):
        # January is wet by the calendar; a fully counted dry month is not.
        snapshot = snapshot_of("2025-12-15T08:00:00Z", "2026-01-20T20:10:00Z")

        assert snapshot.wet_mode.mode == "dry"
        assert snapshot.wet_mode.decided_by == "weather"
        assert snapshot.tile_titles.wet_or_dry == "Dry spell"

    def test_one_dry_week_after_the_wet_week_holds_wet_mode(self):
        # The last 14 days hold the wet week: 28 of 336 hours, over 2%.
        snapshot = snapshot_of(SUMMER_START, "2026-07-08T19:10:00Z", rain=WET_WEEK)

        assert snapshot.wet_mode.mode == "wet"

    def test_two_dry_weeks_after_the_wet_week_end_wet_mode(self):
        snapshot = snapshot_of(SUMMER_START, "2026-07-16T19:10:00Z", rain=WET_WEEK)

        assert snapshot.wet_mode.mode == "dry"
        assert snapshot.wet_mode.decided_by == "weather"

    def test_rain_today_does_not_decide_todays_mode(self):
        # A day's mode is decided by the whole days before it.
        today = daily_rain("2026-07-01", 1, 2)
        snapshot = snapshot_of(SUMMER_START, SUMMER_NOW, rain=today)

        assert snapshot.wet_mode.mode == "dry"


class TestTheCalendarDecides:
    def test_patchy_data_in_december_is_wet_by_the_calendar(self):
        # Ten days of the last 14 missing: fewer than half its hours count.
        gap = Gap("2026-12-06T08:00:00Z", "2026-12-16T08:00:00Z")
        snapshot = snapshot_of(
            "2026-11-01T07:00:00Z", "2026-12-20T20:10:00Z", gaps=[gap]
        )

        assert snapshot.wet_mode.mode == "wet"
        assert snapshot.wet_mode.decided_by == "calendar"
        assert snapshot.tile_titles.wet_or_dry == "Wet hours"

    def test_patchy_data_in_summer_is_dry_by_the_calendar(self):
        gap = Gap("2026-06-17T07:00:00Z", "2026-06-27T07:00:00Z")
        snapshot = snapshot_of(SUMMER_START, SUMMER_NOW, rain=WET_WEEK, gaps=[gap])

        assert snapshot.wet_mode.mode == "dry"
        assert snapshot.wet_mode.decided_by == "calendar"

    def test_a_new_station_starts_on_the_calendar(self):
        # Three days of record: Lookout Post's thresholds, nothing to count.
        snapshot = snapshot_of("2026-11-28T08:00:00Z", "2026-12-01T20:10:00Z")

        assert snapshot.wet_mode.mode == "wet"
        assert snapshot.wet_mode.decided_by == "calendar"

    def test_an_archive_that_stopped_weeks_ago_falls_to_the_calendar(self):
        archive = build_archive(SUMMER_START, "2026-06-20T07:00:00Z", rain=WET_WEEK)
        snapshot = build(archive, None, "2026-07-20T19:10:00Z")

        assert snapshot.wet_mode.decided_by == "calendar"
        assert snapshot.wet_mode.mode == "dry"


class TestARebuild:
    def test_a_rebuild_gives_the_same_mode(self):
        archive = build_archive(SUMMER_START, SUMMER_NOW, rain=WET_WEEK)
        first = build(archive, None, SUMMER_NOW)
        again = build(archive.copy(), None, SUMMER_NOW)

        assert first.wet_mode == again.wet_mode
        assert first.tile_titles == again.tile_titles

    def test_the_mode_holds_through_the_day(self):
        # Morning and late evening of the same day, the evening with more rain.
        rain = list(WET_WEEK) + list(daily_rain("2026-07-01", 1, 6))
        morning = snapshot_of(SUMMER_START, "2026-07-01T14:10:00Z", rain=rain)
        evening = snapshot_of(SUMMER_START, "2026-07-02T06:50:00Z", rain=rain)

        assert morning.wet_mode == evening.wet_mode


def test_the_dashboard_document_carries_the_mode_and_the_title():
    document = snapshot_of(SUMMER_START, SUMMER_NOW, rain=WET_WEEK).dashboard()

    assert document["wet_mode"] == {"mode": "wet", "decided_by": "weather"}
    assert document["tile_titles"] == {"wet_or_dry": "Wet hours"}
    assert document["schema_version"] == 4
