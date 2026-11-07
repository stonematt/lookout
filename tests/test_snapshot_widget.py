"""
The widget view, through `build_snapshot`: which view a widget shows now.

The rules are checked in order and the first that holds is the view:

1. rain_event, if an event is live with more than 0.04 in in it;
2. dry_spell, if it is June to September at the station and it has been dry
   for at least 3 days;
3. otherwise temperature_ranges.

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
    raining_station,
    scenario_until,
)

SUMMER_NOW = "2026-07-10T20:00:00Z"  # Fri Jul 10, 1 pm local


def dry_station(month: str, rain_day: str):
    """
    A station from the 1st to the 10th of `month` ("YYYY-MM"), 1 pm local,
    that last rained for an hour on `rain_day` (the day of the month).
    """
    now = f"{month}-10T20:00:00Z"
    full = build_archive(
        f"{month}-01T08:00:00Z",
        f"{month}-10T20:05:00Z",
        rain=(RainSpell(f"{month}-{rain_day}T18:00:00Z", 1, 0.12),),
    )
    archive = full[full["dateutc"] < ms(now)].reset_index(drop=True)
    live = live_row(full, now)
    return build(archive, live, f"{month}-10T20:02:00Z", event_catalog(archive))


def days_dry(snapshot) -> float:
    """The days since it last rained: the calm rule's value."""
    return next(r.value for r in snapshot.hero_lead.rules if r.kind == "calm")


def test_a_live_event_over_four_hundredths_shows_the_rain_event():
    snapshot = raining_station()

    assert snapshot.active_rain_event is not None
    assert snapshot.active_rain_event.total_in > 0.04
    assert snapshot.widget_view == "rain_event"


def test_a_live_event_of_exactly_four_hundredths_does_not():
    full = scenario_until("2026-01-09T00:30:00Z")
    archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
    live = live_row(full, "2026-01-09T00:25:00Z")
    live["eventrainin"] = 0.04

    snapshot = build(archive, live, "2026-01-09T00:27:00Z", event_catalog(archive))

    assert snapshot.active_rain_event is not None
    assert snapshot.active_rain_event.total_in == 0.04
    assert snapshot.widget_view == "temperature_ranges"


def test_a_summer_dry_spell_of_three_days_or_more_shows_the_dry_spell():
    snapshot = dry_station("2026-07", "05")

    assert days_dry(snapshot) >= 3
    assert snapshot.widget_view == "dry_spell"


def test_summer_rain_yesterday_shows_temperature_ranges():
    snapshot = dry_station("2026-07", "09")

    assert days_dry(snapshot) < 3
    assert snapshot.widget_view == "temperature_ranges"


def test_the_same_dry_spell_in_winter_shows_temperature_ranges():
    snapshot = dry_station("2026-01", "05")

    assert snapshot.active_rain_event is None
    assert days_dry(snapshot) >= 3
    assert snapshot.widget_view == "temperature_ranges"


def test_a_snapshot_with_no_history_shows_temperature_ranges():
    snapshot = build(pd.DataFrame({"dateutc": []}), None, SUMMER_NOW)

    assert snapshot.widget_view == "temperature_ranges"


def test_the_context_block_carries_the_view():
    snapshot = raining_station()

    assert snapshot.context()["widget_view"] == "rain_event"
    assert snapshot.dashboard()["widget_view"] == "rain_event"
