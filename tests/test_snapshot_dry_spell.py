"""
The dry spell field, through `build_snapshot`: days since it last rained and
the usual dry spell, computed once. The calm rule in `hero_lead` restates the
same two numbers, so the two never disagree while both are written.
"""

import pytest

from tests.snapshot_helpers import (
    GOLDEN_SCENARIOS,
    build,
    dry_station,
    raining_station,
)
from tests.test_snapshot_widget import rain_every_fifth_day


def calm(snapshot):
    """The calm lead rule."""
    return next(r for r in snapshot.hero_lead.rules if r.kind == "calm")


@pytest.mark.parametrize("name", sorted(GOLDEN_SCENARIOS))
def test_the_field_equals_the_calm_rule_on_every_golden_scenario(name):
    snapshot = GOLDEN_SCENARIOS[name]()

    assert snapshot.dry_spell.days_since_rain == calm(snapshot).value
    assert snapshot.dry_spell.usual_days == calm(snapshot).reference


@pytest.mark.parametrize("name", sorted(GOLDEN_SCENARIOS))
def test_the_field_is_in_the_dashboard_and_not_the_context(name):
    snapshot = GOLDEN_SCENARIOS[name]()

    assert snapshot.dashboard()["dry_spell"] == {
        "days_since_rain": snapshot.dry_spell.days_since_rain,
        "usual_days": snapshot.dry_spell.usual_days,
    }
    assert "dry_spell" not in snapshot.context()


def test_a_young_station_has_days_since_rain_and_no_usual_dry_spell():
    snapshot = dry_station()

    assert snapshot.dry_spell.days_since_rain is not None
    assert snapshot.dry_spell.usual_days is None


def test_a_year_of_rain_gives_both():
    # Rain every fifth day for a year: every completed dry spell is four days.
    end = "2026-01-09T07:00:00Z"
    snapshot = build(rain_every_fifth_day(end), None, end)

    assert snapshot.dry_spell.usual_days == 4.0
    assert snapshot.dry_spell.days_since_rain > 4.0
    assert snapshot.dry_spell.days_since_rain == calm(snapshot).value


def test_a_live_event_keeps_days_since_rain():
    snapshot = raining_station()

    assert snapshot.dry_spell.days_since_rain == calm(snapshot).value
