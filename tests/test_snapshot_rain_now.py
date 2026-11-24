"""
Last rain and the now story (ADR-0011, #659), through `build_snapshot`.

**Last rain** is the newest reading the daily counter rose on, from the
archive's readings (rain corrected) with the live reading stitched on; never
Ambient's `lastRain`. It is raining while last rain is within 30 minutes of
the build time. While it is within 3 hours the now story is told (the rain
in the last 60 minutes, and the event's start and total by Ambient's bounds)
and the Feature Rule picks rain; past 3 hours the story is empty and rain
does not lead.

The SCENARIO station's event 3 rains 0.12 in/h from 22:00Z on 8 Jan to
02:00Z on 9 Jan: 0.01 in at every reading from 22:05Z to 01:55Z, its last
tip, and 0.48 in in all.
"""

import pandas as pd

from lookout.core.rain_corrections import correct_counters, detect_carryovers
from lookout.snapshot.model import Snapshot
from tests.fixtures.archive import RainSpell, build_archive
from tests.snapshot_helpers import (
    build,
    event_catalog,
    live_row,
    ms,
    raining_station,
    rain_every_fifth_day,
    scenario_until,
)

LAST_TIP = ms("2026-01-09T01:55:00Z")
"""Event 3's last tip."""

EVENT_3 = ms("2026-01-08T22:00:00Z")
"""Event 3's start, the key of its `rain_events` row."""

MINUTE_MS = 60_000


def at(milliseconds: int) -> str:
    """An ISO time from epoch ms."""
    return pd.Timestamp(milliseconds, unit="ms", tz="UTC").isoformat()


def after_tip(minutes: float, extra_ms: int = 0) -> Snapshot:
    """
    The SCENARIO station built `minutes` (plus `extra_ms`) after event 3's last
    tip, its archive up to the build time and no live reading.
    """
    built = LAST_TIP + int(minutes * MINUTE_MS) + extra_ms
    archive = scenario_until(at(built))
    return build(archive, None, at(built), event_catalog(archive))


class TestLastRain:
    def test_it_is_the_last_reading_the_counter_rose_on(self):
        snapshot = after_tip(90)

        assert snapshot.rain_now.last_rain_ms == LAST_TIP

    def test_it_is_not_ambients_last_rain(self):
        # Ambient says it rained 10 minutes ago; the counter has not moved.
        archive = scenario_until("2026-01-09T04:00:00Z")
        live = live_row(scenario_until("2026-01-09T04:05:00Z"), "2026-01-09T04:00:00Z")
        live["lastRain"] = "2026-01-09T03:50:00.000Z"

        snapshot = build(archive, live, "2026-01-09T04:01:00Z", event_catalog(archive))

        assert snapshot.current.last_rain_ms == ms("2026-01-09T03:50:00Z")
        assert snapshot.rain_now.last_rain_ms == LAST_TIP
        assert snapshot.rain_now.raining is False

    def test_the_live_reading_counts_when_newer_than_the_archive(self):
        snapshot = raining_station()

        # The archive ends at 23:55Z; the live reading at 00:25Z has more rain.
        assert snapshot.rain_now.last_rain_ms == ms("2026-01-09T00:25:00Z")
        assert snapshot.rain_now.raining is True

    def test_it_is_read_after_carryover_corrections(self):
        # The counter misses its reset inside the 07:00Z-09:00Z gap, so the
        # 09:00Z reading still holds 0.48 in: a tip only until corrected.
        built = "2026-01-09T10:00:00Z"
        archive = scenario_until(built)
        corrected = correct_counters(archive, detect_carryovers(archive))

        raw = build(archive, None, built, event_catalog(archive))
        snapshot = build(corrected, None, built, event_catalog(corrected))

        assert raw.rain_now.last_rain_ms == ms("2026-01-09T09:00:00Z")
        assert snapshot.rain_now.last_rain_ms == LAST_TIP

    def test_it_stays_set_long_after_the_story_ends(self):
        snapshot = after_tip(5 * 60)

        assert snapshot.rain_now.last_rain_ms == LAST_TIP
        assert snapshot.rain_now.raining is False
        assert snapshot.rain_now.rate_in_hr is None
        assert snapshot.rain_now.event_start_ms is None
        assert snapshot.rain_now.event_total_in is None

    def test_no_tip_on_record_is_none(self):
        archive = build_archive("2026-01-01T08:00:00Z", "2026-01-03T08:00:00Z")

        snapshot = build(archive, None, "2026-01-03T08:00:00Z")

        assert snapshot.rain_now.last_rain_ms is None
        assert snapshot.rain_now.raining is False
        assert snapshot.feature.kind != "rain"


class TestRaining:
    def test_raining_at_exactly_30_minutes(self):
        assert after_tip(30).rain_now.raining is True

    def test_not_raining_a_moment_past_30_minutes(self):
        snapshot = after_tip(30, extra_ms=1)

        assert snapshot.rain_now.raining is False
        assert snapshot.feature.kind == "rain"


class TestRate:
    def test_it_is_the_rain_in_the_last_60_minutes(self):
        # 01:00Z to 02:00Z: the readings from 01:05Z to 01:55Z, 0.01 in each.
        snapshot = after_tip(5)

        assert snapshot.rain_now.rate_in_hr == 0.11

    def test_a_reading_exactly_60_minutes_old_is_out(self):
        # Built at 02:25Z: 01:25Z is out, 01:30Z to 01:55Z in.
        snapshot = after_tip(30)

        assert snapshot.rain_now.rate_in_hr == 0.06

    def test_zero_after_an_hour_without_a_tip(self):
        assert after_tip(120).rain_now.rate_in_hr == 0.0


class TestEvent:
    def test_the_live_event_gives_the_start_and_total(self):
        snapshot = after_tip(30)

        assert snapshot.active_rain_event is not None
        assert snapshot.rain_now.event_start_ms == EVENT_3
        assert snapshot.rain_now.event_total_in == 0.48

    def test_an_ended_event_that_holds_last_rain_gives_them(self):
        # Ambient ends this station's events after one dry hour.
        spell = RainSpell("2026-01-08T22:00:00Z", 1, 0.12)
        archive = build_archive(
            "2026-01-07T08:00:00Z",
            "2026-01-09T01:00:00Z",
            rain=(spell,),
            event_reset_hours=1.0,
        )

        snapshot = build(archive, None, "2026-01-09T01:00:00Z", event_catalog(archive))

        assert snapshot.active_rain_event is None
        assert snapshot.rain_now.last_rain_ms == ms("2026-01-08T22:55:00Z")
        assert snapshot.rain_now.event_start_ms == EVENT_3
        assert snapshot.rain_now.event_total_in == 0.12
        assert snapshot.feature.event_start_ms == EVENT_3

    def test_without_an_event_catalog_the_event_is_unknown(self):
        archive = scenario_until("2026-01-09T02:25:00Z")

        snapshot = build(archive, None, "2026-01-09T02:25:00Z")

        assert snapshot.rain_now.raining is True
        assert snapshot.rain_now.event_start_ms is None
        assert snapshot.rain_now.event_total_in is None
        assert snapshot.feature.kind == "rain"


class TestFeatureRule:
    def test_rain_leads_at_exactly_3_hours(self):
        snapshot = after_tip(180)

        assert snapshot.feature.kind == "rain"
        assert snapshot.feature.event_start_ms == snapshot.rain_now.event_start_ms
        assert snapshot.widget_view == "rain_event"
        assert snapshot.rain_now.rate_in_hr == 0.0

    def test_rain_does_not_lead_a_moment_past_3_hours(self):
        snapshot = after_tip(180, extra_ms=1)

        assert snapshot.feature.kind != "rain"
        assert snapshot.feature.event_start_ms is None
        assert snapshot.widget_view != "rain_event"
        assert snapshot.rain_now.rate_in_hr is None

    def test_a_pause_between_showers_still_leads(self):
        # 40 minutes between tips: not raining, still rain.
        snapshot = after_tip(40)

        assert snapshot.rain_now.raining is False
        assert snapshot.feature.kind == "rain"

    def test_a_live_event_past_3_hours_of_no_tips_does_not_lead(self):
        # Ambient's event runs on for 6 dry hours; rain stopped leading at 3.
        snapshot = after_tip(4 * 60)

        assert snapshot.active_rain_event is not None
        assert snapshot.feature.kind != "rain"

    def test_the_dry_spell_clause_reads_the_dry_spell_not_last_rain(self):
        # 9 Jan, in the wet months: a dry spell longer than usual still leads.
        end = "2026-01-09T06:35:00Z"
        snapshot = build(rain_every_fifth_day(end), None, end)

        assert snapshot.rain_now.last_rain_ms is not None
        assert snapshot.dry_spell.days_since_rain > snapshot.dry_spell.usual_days
        assert snapshot.feature.kind == "dry_spell"


class TestPublished:
    def test_the_dashboard_carries_rain_now(self):
        dashboard = after_tip(30).dashboard()

        assert dashboard["rain_now"] == {
            "last_rain_ms": LAST_TIP,
            "raining": True,
            "rate_in_hr": 0.06,
            "event_start_ms": EVENT_3,
            "event_total_in": 0.48,
        }

    def test_the_context_block_does_not(self):
        assert "rain_now" not in after_tip(30).context()
