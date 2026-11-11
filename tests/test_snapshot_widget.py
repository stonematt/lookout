"""
The widget view, through `build_snapshot`: which view a widget shows now.

The rules are checked in order and the first that holds is the view:

1. rain_event, if an event is live with more than 0.04 in in it;
2. dry_spell, if the current dry spell is longer than the usual dry spell
   (the calm lead's comparison), in any month;
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

NOW = "2026-01-10T20:00:00Z"


def rain_every_fifth_day(end: str) -> pd.DataFrame:
    """
    A year of rain for an hour every fifth day, the last on Sat 2026-01-03,
    running dry to `end`: every completed dry spell is four days long.
    """
    days = pd.date_range(end="2026-01-03", periods=73, freq="5D")
    spells = tuple(RainSpell(f"{day.date()}T18:00:00Z", 1, 0.12) for day in days)
    return build_archive("2025-01-01T08:00:00Z", end, rain=spells)


def calm(snapshot):
    """The calm rule: days since it last rained, against the usual dry spell."""
    return next(r for r in snapshot.hero_lead.rules if r.kind == "calm")


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


def test_a_dry_spell_longer_than_usual_shows_the_dry_spell_in_any_month():
    # Thu Jan 8, 11 pm local: five and a half days dry against a usual four.
    end = "2026-01-09T07:00:00Z"
    snapshot = build(rain_every_fifth_day(end), None, end)

    assert calm(snapshot).value > calm(snapshot).reference == 4.0
    assert snapshot.widget_view == "dry_spell"


def test_a_dry_spell_shorter_than_usual_shows_temperature_ranges():
    # Mon Jan 5, 11 pm local: two and a half days dry against a usual four.
    end = "2026-01-06T07:00:00Z"
    snapshot = build(rain_every_fifth_day(end), None, end)

    assert calm(snapshot).value < calm(snapshot).reference == 4.0
    assert snapshot.widget_view == "temperature_ranges"


def test_without_a_usual_dry_spell_a_long_dry_run_shows_temperature_ranges():
    # Ten days of record: the 365-day window is invalid, so there is no usual.
    archive = build_archive(
        "2026-01-01T08:00:00Z",
        "2026-01-10T20:00:00Z",
        rain=(RainSpell("2026-01-02T18:00:00Z", 1, 0.12),),
    )

    snapshot = build(archive, None, NOW)

    assert calm(snapshot).value > 7
    assert calm(snapshot).reference is None
    assert snapshot.widget_view == "temperature_ranges"


def test_a_snapshot_with_no_history_shows_temperature_ranges():
    snapshot = build(pd.DataFrame({"dateutc": []}), None, NOW)

    assert snapshot.widget_view == "temperature_ranges"


def test_the_context_block_carries_the_view():
    snapshot = raining_station()

    assert snapshot.context()["widget_view"] == "rain_event"
    assert snapshot.dashboard()["widget_view"] == "rain_event"
