"""
The widget view, through `build_snapshot`: which view a widget shows now.

It is the feature (ADR-0009) mapped to a widget's three views:

1. rain_event, when the feature is rain (an event is live, of any size, or
   ended in the last 12 h);
2. dry_spell, when the feature is dry_spell (the current dry spell is longer
   than the usual dry spell, the snapshot's `dry_spell` field, in any month,
   and no reading is unusual);
3. otherwise temperature_ranges (an unusual reading, or quiet).

The view is in the dashboard and in the context block, so a widget reading
`/current` gets it with the live reading.
"""

import pandas as pd

from tests.fixtures.archive import RainSpell, build_archive
from tests.snapshot_helpers import (
    build,
    event_catalog,
    live_row,
    ms,
    rain_every_fifth_day,
    raining_station,
    scenario_until,
)

NOW = "2026-01-10T20:00:00Z"


def spell(snapshot):
    """The dry spell field: days since it last rained, the usual dry spell."""
    return snapshot.dry_spell


def test_a_live_event_shows_the_rain_event():
    snapshot = raining_station()

    assert snapshot.active_rain_event is not None
    assert snapshot.feature.kind == "rain"
    assert snapshot.widget_view == "rain_event"


def test_a_live_event_of_four_hundredths_or_less_shows_it_too():
    # The old 0.04 in threshold went with the widget's own rules (ADR-0009).
    full = scenario_until("2026-01-09T00:30:00Z")
    archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
    live = live_row(full, "2026-01-09T00:25:00Z")
    live["eventrainin"] = 0.04

    snapshot = build(archive, live, "2026-01-09T00:27:00Z", event_catalog(archive))

    assert snapshot.active_rain_event is not None
    assert snapshot.active_rain_event.total_in == 0.04
    assert snapshot.widget_view == "rain_event"


def test_an_event_ended_within_12_hours_shows_the_rain_event():
    # Event 3 ended Fri 2026-01-09 about 02:00Z; built eight hours later.
    archive = scenario_until("2026-01-09T10:00:00Z")

    snapshot = build(archive, None, "2026-01-09T10:00:00Z", event_catalog(archive))

    assert snapshot.active_rain_event is None
    assert snapshot.feature.kind == "rain"
    assert snapshot.widget_view == "rain_event"


def test_a_dry_spell_longer_than_usual_shows_the_dry_spell_in_any_month():
    # Thu Jan 8, 10:30 pm local: five and a half days dry against a usual
    # four. Mid-hour, so no reading is at its hour's edge and none is unusual
    # against 2025's same dates.
    end = "2026-01-09T06:35:00Z"
    snapshot = build(rain_every_fifth_day(end), None, end)

    assert spell(snapshot).days_since_rain > spell(snapshot).usual_days == 4.0
    assert snapshot.feature.kind == "dry_spell"
    assert snapshot.widget_view == "dry_spell"


def test_an_unusual_reading_in_a_long_dry_spell_shows_temperature_ranges():
    # The same dry spell, the live temperature far above 2025's same hour.
    end = "2026-01-09T06:35:00Z"
    full = rain_every_fifth_day(end)
    live = live_row(full, "2026-01-09T06:30:00Z")
    live["tempf"] = 99.0
    archive = full[full["dateutc"] < ms("2026-01-09T06:30:00Z")]

    snapshot = build(archive, live, end)

    assert spell(snapshot).days_since_rain > spell(snapshot).usual_days
    assert snapshot.feature.kind == "unusual"
    assert snapshot.widget_view == "temperature_ranges"


def test_a_dry_spell_shorter_than_usual_shows_temperature_ranges():
    # Mon Jan 5, 11 pm local: two and a half days dry against a usual four.
    end = "2026-01-06T07:00:00Z"
    snapshot = build(rain_every_fifth_day(end), None, end)

    assert spell(snapshot).days_since_rain < spell(snapshot).usual_days == 4.0
    assert snapshot.widget_view == "temperature_ranges"


def test_without_a_usual_dry_spell_a_long_dry_run_shows_temperature_ranges():
    # Ten days of record: the 365-day window is invalid, so there is no usual.
    archive = build_archive(
        "2026-01-01T08:00:00Z",
        "2026-01-10T20:00:00Z",
        rain=(RainSpell("2026-01-02T18:00:00Z", 1, 0.12),),
    )

    snapshot = build(archive, None, NOW)

    assert spell(snapshot).days_since_rain > 7
    assert spell(snapshot).usual_days is None
    assert snapshot.widget_view == "temperature_ranges"


def test_a_snapshot_with_no_history_shows_temperature_ranges():
    snapshot = build(pd.DataFrame({"dateutc": []}), None, NOW)

    assert snapshot.widget_view == "temperature_ranges"


def test_the_context_block_carries_the_view():
    snapshot = raining_station()

    assert snapshot.context()["widget_view"] == "rain_event"
    assert snapshot.dashboard()["widget_view"] == "rain_event"
