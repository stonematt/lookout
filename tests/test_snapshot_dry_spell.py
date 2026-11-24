"""
The dry spell field, through `build_snapshot`: the days since it last rained
in whole dry days (GLOSSARY dry spell, #668), and the usual dry spell,
computed once for every reader.

A whole dry day is a completed station-local day with under 0.01 in of rain.
The count runs back from yesterday to the last rain day; today is never a
whole day yet, and rain today makes 0. A day the archive cannot count
breaks the run, so a gap never reads as dry. Ambient's `lastRain` plays no
part. Archives are synthetic five-minute stations, local days Pacific.
"""

from typing import Dict, Optional, Sequence

import pandas as pd
import pytest

from lookout.snapshot import Snapshot
from tests.fixtures.archive import Gap, RainSpell, build_archive
from tests.snapshot_helpers import (
    GOLDEN_SCENARIOS,
    build,
    dry_station,
    live_row,
    rain_every_fifth_day,
    raining_station,
)

START = "2025-12-20T08:00:00Z"  # local midnight, Sat Dec 20
# One hour of rain from 10:00 local on Sat Jan 3: Jan 3 is the last rain day.
LAST_RAIN = (RainSpell("2026-01-03T18:00:00Z", 1, 0.12),)


def snapshot_at(
    now: str,
    rain: Sequence[RainSpell] = LAST_RAIN,
    gaps: Sequence[Gap] = (),
    live: Optional[Dict] = None,
) -> Snapshot:
    """`build_snapshot` at `now`; the live reading is the archive's newest row."""
    archive = build_archive(START, now, rain=rain, gaps=gaps)
    newest = pd.Timestamp(int(archive["dateutc"].max()), unit="ms", tz="UTC")
    reading = live_row(archive, newest.isoformat())
    reading.update(live or {})
    return build(archive, reading, now)


def whole_dry_days(now: str, **kwargs) -> Optional[int]:
    """The snapshot's `days_since_rain` at `now`."""
    return snapshot_at(now, **kwargs).dry_spell.days_since_rain


@pytest.mark.parametrize("name", sorted(GOLDEN_SCENARIOS))
def test_the_field_is_in_the_dashboard_and_not_the_context(name):
    snapshot = GOLDEN_SCENARIOS[name]()

    assert snapshot.dashboard()["dry_spell"] == {
        "days_since_rain": snapshot.dry_spell.days_since_rain,
        "usual_days": snapshot.dry_spell.usual_days,
    }
    assert "dry_spell" not in snapshot.context()


class TestWholeDryDays:
    def test_rain_today_is_no_dry_spell_yet(self):
        assert whole_dry_days("2026-01-03T20:10:00Z") == 0  # 12:10 Jan 3

    def test_rain_yesterday_and_none_today_is_still_none(self):
        assert whole_dry_days("2026-01-04T20:10:00Z") == 0  # 12:10 Jan 4

    def test_a_day_and_a_half_dry_before_midnight_is_still_none(self):
        # 23:50 Jan 4: about 37 hours since the last tip, no whole dry day.
        assert whole_dry_days("2026-01-05T07:50:00Z") == 0

    def test_the_first_whole_dry_day_counts_at_local_midnight(self):
        assert whole_dry_days("2026-01-05T08:10:00Z") == 1  # 00:10 Jan 5

    def test_a_dry_run_counts_each_whole_day(self):
        # Jan 4 to Jan 8 are whole dry days; Jan 9 is today.
        assert whole_dry_days("2026-01-09T20:10:00Z") == 5

    def test_the_count_is_a_whole_number_in_the_document(self):
        document = snapshot_at("2026-01-09T20:10:00Z").dashboard()

        days = document["dry_spell"]["days_since_rain"]
        assert days == 5 and isinstance(days, int)

    def test_ambient_last_rain_plays_no_part(self):
        # Ambient says it rained an hour ago; the archive has no tip since Jan 3.
        live = {"lastRain": "2026-01-09T19:10:00.000Z"}

        assert whole_dry_days("2026-01-09T20:10:00Z", live=live) == 5

    def test_a_tip_under_a_rain_day_keeps_the_day_dry(self):
        # 0.005 in on Jan 7 is under one tip: Jan 7 is still a dry day.
        rain = LAST_RAIN + (RainSpell("2026-01-07T18:00:00Z", 1, 0.005),)

        assert whole_dry_days("2026-01-09T20:10:00Z", rain=rain) == 5


class TestAGapBreaksTheRun:
    def test_only_the_dry_days_after_a_gap_day_count(self):
        # Jan 6 has no readings: only Jan 7 and Jan 8 are known dry days.
        gap = Gap("2026-01-06T08:00:00Z", "2026-01-07T08:00:00Z")

        assert whole_dry_days("2026-01-09T20:10:00Z", gaps=[gap]) == 2

    def test_a_gap_yesterday_leaves_the_dry_spell_unknown(self):
        # No known dry day since the gap: not "no dry spell yet".
        gap = Gap("2026-01-08T08:00:00Z", "2026-01-09T08:00:00Z")

        assert whole_dry_days("2026-01-09T20:10:00Z", gaps=[gap]) is None


def test_a_young_station_has_days_since_rain_and_no_usual_dry_spell():
    snapshot = snapshot_at("2026-01-09T20:10:00Z")

    assert snapshot.dry_spell.days_since_rain == 5
    assert snapshot.dry_spell.usual_days is None


def test_a_day_past_the_archive_end_leaves_the_dry_golden_unknown():
    # The real slice ends 20:55 local on its last day, three hours before the
    # build: that day is not archived, so it cannot be counted as dry.
    assert dry_station().dry_spell.days_since_rain is None


def test_a_year_of_rain_gives_both():
    # Rain every fifth day to Sat Jan 3; built 22:35 local Fri Jan 9.
    end = "2026-01-10T06:35:00Z"
    snapshot = build(rain_every_fifth_day(end), None, end)

    assert snapshot.dry_spell.usual_days == 4.0
    assert snapshot.dry_spell.days_since_rain == 5


def test_a_live_event_is_no_dry_spell_yet():
    assert raining_station().dry_spell.days_since_rain == 0
