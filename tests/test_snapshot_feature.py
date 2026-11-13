"""
The feature (ADR-0009, #464), through `build_snapshot`.

The Feature Rule, checked in order, first that holds:

1. rain: an event is live, or the newest event ended at most 12 h before
   the build time; it names that event by `event_start_ms` (#523);
2. unusual: a reading (temperature, humidity, pressure, solar; never wind
   gust, #533) is below the 10th or above the 90th percentile of its
   readings at this hour over the usual window's earlier years; skipped
   without an earlier year; the one furthest from the 50th wins, ties to
   the first in that list;
3. dry_spell: the current dry spell is longer than the usual dry spell;
4. quiet: otherwise.

The hold (#533): a subject (the kind, and for unusual the reading) chosen
under 3 hours ago is kept when the rule now picks a different one, unless
the rule picks rain; a held feature's values are recomputed every build.

The widget view follows: rain -> rain_event, dry_spell -> dry_spell, else
temperature_ranges.

The seasoned station (#461) has two earlier years around 20 May, so the
unusual rule applies to it; its live reading is at 15:00 PDT. Its humidity,
pressure and gust are the same at every 15:00 hour, so they sit at the 50th
until a test moves them; temperature and solar are set to their usual
median for the rules after unusual.
"""

from dataclasses import replace
from datetime import date, timedelta
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from lookout.snapshot.build import build_snapshot
from lookout.snapshot.feature import previous_feature
from lookout.snapshot.model import (
    Feature,
    FeatureDirection,
    FeatureKind,
    FeatureReading,
    Snapshot,
)
from lookout.utils.dateutc import DAY_MS, HOUR_MS, STATION_FRESH_MINUTES
from tests.fixtures.archive import (
    SEASONED_NOW,
    SEASONED_PUSHED_TEMPF,
    SEASONED_RAIN,
    RainSpell,
    build_archive,
    seasoned_archive,
)
from tests.snapshot_helpers import (
    SEASONED_ARCHIVE_NEWEST,
    SEASONED_BUILT,
    SEASONED_STATION,
    build,
    event_catalog,
    live_row,
    ms,
    rain_every_fifth_day,
    raining_station,
    scenario_until,
    seasoned_station,
)

TZ = "America/Los_Angeles"
HOUR = 15
"""The seasoned live reading's station-local hour."""

USUAL_DAYS = [
    date(year, 5, 13) + timedelta(days=n) for year in (2024, 2025) for n in range(15)
]
"""The seasoned usual window: 13-27 May in 2024 and 2025."""

FRESH_MS = STATION_FRESH_MINUTES * 60_000
"""The oldest observation the unusual rule still judges, exclusive."""


def hour_readings(
    archive: pd.DataFrame, days: Sequence[date], hour: int, column: str
) -> np.ndarray:
    """Every `column` reading on `days` during local `hour`."""
    local = pd.to_datetime(archive["dateutc"], unit="ms", utc=True).dt.tz_convert(TZ)
    rows = archive[local.dt.date.isin(set(days)) & (local.dt.hour == hour)]
    return rows[column].dropna().to_numpy(dtype=float)


def seasoned(
    changes: Optional[Dict[str, float]] = None,
    rain: Sequence[RainSpell] = SEASONED_RAIN,
    ordinary: bool = True,
    built: str = SEASONED_BUILT,
    previous: Optional[Feature] = None,
) -> Snapshot:
    """
    The seasoned station, its live reading changed.

    :param changes: Live fields to set, Ambient names.
    :param rain: The station's rain spells.
    :param ordinary: Set the live temperature and solar to their usual median
        first, so neither is unusual unless `changes` says so.
    :param built: The build time; the live reading stays at `SEASONED_NOW`.
    :param previous: The last published feature, or None.
    :return: The snapshot.
    """
    full = seasoned_archive(rain=rain)
    archive = full[full["dateutc"] <= ms(SEASONED_ARCHIVE_NEWEST)].reset_index(
        drop=True
    )
    live = live_row(full, SEASONED_NOW)
    if ordinary:
        for column in ("tempf", "solarradiation"):
            live[column] = float(
                np.median(hour_readings(full, USUAL_DAYS, HOUR, column))
            )
    live.update(changes or {})
    return build_snapshot(
        archive=archive,
        live=live,
        event_catalog=event_catalog(archive),
        energy_catalog=None,
        station=SEASONED_STATION,
        now=pd.Timestamp(built),
        previous_feature=previous,
    )


def after_live(milliseconds: int) -> str:
    """The build time `milliseconds` after the seasoned live reading."""
    return pd.Timestamp(
        ms(SEASONED_NOW) + milliseconds, unit="ms", tz="UTC"
    ).isoformat()


def usual_tempf(share: float) -> float:
    """The seasoned usual temperature at 15:00 at quantile `share`."""
    values = hour_readings(seasoned_archive(), USUAL_DAYS, HOUR, "tempf")
    return float(np.quantile(values, share))


# Thu Jan 8, 10:30 pm local: mid-hour, so no reading sits at its hour's edge.
DRY_END = "2026-01-09T06:35:00Z"
DRY_LAST = "2026-01-09T06:30:00Z"


class TestUnusual:
    def test_the_seasoned_golden_features_its_pushed_temperature(self):
        snapshot = seasoned_station()

        assert snapshot.temperature_context.usual_band.basis == "earlier_years"
        feature = snapshot.feature
        assert feature.kind == "unusual"
        assert feature.reading == "temperature"
        assert feature.value == SEASONED_PUSHED_TEMPF
        assert feature.unit == "f"
        # Above every one of the 360 usual readings.
        assert feature.percentile == 100
        assert snapshot.widget_view == "temperature_ranges"

    def test_a_reading_below_the_10th_percentile_is_unusual(self):
        snapshot = seasoned({"tempf": usual_tempf(0.0) - 5})

        assert snapshot.feature.kind == "unusual"
        assert snapshot.feature.reading == "temperature"
        assert snapshot.feature.percentile == 0
        assert snapshot.widget_view == "temperature_ranges"

    def test_a_reading_inside_the_10th_to_90th_is_not(self):
        snapshot = seasoned({"tempf": usual_tempf(0.85)})

        assert snapshot.feature.kind == "quiet"

    def test_the_percentile_is_the_share_below_with_ties_counted_half(self):
        # Humidity reads 65 at every 15:00 reading; 40 is below them all.
        snapshot = seasoned({"humidity": 40.0})

        assert snapshot.feature.kind == "unusual"
        assert snapshot.feature.reading == "humidity"
        assert snapshot.feature.value == 40.0
        assert snapshot.feature.unit == "pct"
        assert snapshot.feature.percentile == 0

    def test_a_reading_equal_to_every_usual_reading_sits_at_the_50th(self):
        # Pressure, gust and humidity read the same at every 15:00 reading.
        snapshot = seasoned()

        assert snapshot.current.barometer_inhg == 30.0
        assert snapshot.feature.kind == "quiet"

    def test_the_reading_furthest_out_wins(self):
        # Temperature at the 95th is out; humidity past every reading is further.
        snapshot = seasoned({"tempf": usual_tempf(0.95), "humidity": 99.0})

        assert snapshot.feature.reading == "humidity"
        assert snapshot.feature.unit == "pct"
        assert snapshot.feature.percentile == 100

    def test_an_unusual_wind_gust_never_becomes_the_feature(self):
        # Gust is the same at every 15:00 reading; 30 mph is past them all.
        snapshot = seasoned({"windgustmph": 30.0})

        assert snapshot.current.wind_gust_mph == 30.0
        assert snapshot.feature.kind == "quiet"

    def test_an_unusual_gust_does_not_outrank_a_reading_that_is_out(self):
        snapshot = seasoned({"tempf": usual_tempf(0.95), "windgustmph": 30.0})

        assert snapshot.feature.reading == "temperature"

    def test_each_reading_maps_to_its_column(self):
        cases = {
            "baromrelin": ("pressure", "inhg", 29.0),
            "solarradiation": ("solar", "wm2", 1000.0),
            "humidity": ("humidity", "pct", 99.0),
            "tempf": ("temperature", "f", 99.0),
        }
        for column, (reading, unit, value) in cases.items():
            feature = seasoned({column: value}).feature

            assert (feature.kind, feature.reading, feature.unit) == (
                "unusual",
                reading,
                unit,
            ), column
            assert feature.value == value

    def test_a_tie_goes_to_the_first_in_rule_order(self):
        # Temperature at the 100th and humidity at the 0th: both 50 out.
        snapshot = seasoned({"tempf": SEASONED_PUSHED_TEMPF, "humidity": 40.0})

        assert snapshot.feature.reading == "temperature"
        assert snapshot.feature.percentile == 100

    def test_the_same_tie_the_other_way_round_still_goes_to_temperature(self):
        snapshot = seasoned({"tempf": usual_tempf(0.0) - 5, "humidity": 99.0})

        assert snapshot.feature.reading == "temperature"
        assert snapshot.feature.percentile == 0

    def test_a_station_with_no_earlier_year_skips_the_rule(self):
        # Thirty days of record: the band is recent days, too thin to judge.
        archive = build_archive("2025-12-08T08:00:00Z", "2026-01-06T20:00:00Z")
        live = live_row(archive, "2026-01-06T19:55:00Z")
        live["tempf"] = 120.0

        snapshot = build(archive, live, "2026-01-06T20:00:00Z")

        assert snapshot.temperature_context.usual_band.basis == "recent_days"
        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.reading is None
        assert snapshot.feature.percentile is None
        assert snapshot.widget_view == "temperature_ranges"

    def test_a_missing_reading_is_not_judged(self):
        snapshot = seasoned({"tempf": None, "humidity": 40.0})

        assert snapshot.feature.reading == "humidity"


class TestStale:
    """
    The unusual rule judges only a fresh observation: one under
    `STATION_FRESH_MINUTES` (15) old at the build time, the age at which the
    site calls the live reading stale and the station heartbeat stops.
    """

    def test_a_station_silent_for_days_is_not_unusual(self):
        # Ambient keeps handing back the last reading; the build runs on.
        snapshot = seasoned(ordinary=False, built=after_live(10 * DAY_MS))

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.reading is None
        assert snapshot.widget_view == "temperature_ranges"

    def test_just_inside_the_limit_it_is_still_unusual(self):
        snapshot = seasoned(ordinary=False, built=after_live(FRESH_MS - 1))

        assert snapshot.feature.kind == "unusual"
        assert snapshot.feature.reading == "temperature"
        assert snapshot.feature.value == SEASONED_PUSHED_TEMPF
        assert snapshot.feature.percentile == 100

    def test_at_the_limit_it_falls_through(self):
        snapshot = seasoned(ordinary=False, built=after_live(FRESH_MS))

        assert snapshot.feature.kind == "quiet"

    def test_a_stale_unusual_reading_falls_through_to_a_dry_spell(self):
        full = rain_every_fifth_day(DRY_END)
        archive = full[full["dateutc"] < ms(DRY_LAST)]
        live = live_row(full, DRY_LAST)
        live["tempf"] = 99.0
        built = pd.Timestamp(ms(DRY_LAST) + FRESH_MS, unit="ms", tz="UTC")

        snapshot = build(archive, live, built.isoformat())

        assert snapshot.dry_spell.days_since_rain > snapshot.dry_spell.usual_days
        assert snapshot.feature.kind == "dry_spell"
        assert snapshot.widget_view == "dry_spell"


class TestRain:
    def test_a_live_event_is_the_feature(self):
        snapshot = raining_station()

        assert snapshot.active_rain_event is not None
        assert snapshot.feature.kind == "rain"
        assert snapshot.feature.reading is None
        assert snapshot.widget_view == "rain_event"

    def test_rain_beats_an_unusual_reading(self):
        # Rain from 11:00 to 12:00 PDT today, three hours before the build.
        rain = SEASONED_RAIN + (RainSpell("2026-05-20T18:00:00Z", 1, 0.1),)
        snapshot = seasoned({"tempf": SEASONED_PUSHED_TEMPF}, rain=rain)

        assert snapshot.feature.kind == "rain"
        assert snapshot.widget_view == "rain_event"

    def test_an_event_that_ended_12_hours_before_the_build_still_holds(self):
        archive = scenario_until("2026-01-09T12:00:00Z")
        catalog = event_catalog(archive)
        end = int(pd.Timestamp(catalog["end_time"].max()).value // 10**6)
        at = pd.Timestamp(end + 12 * 3_600_000, unit="ms", tz="UTC").isoformat()

        snapshot = build(archive, None, at, catalog)

        assert snapshot.active_rain_event is None
        assert snapshot.feature.kind == "rain"
        assert snapshot.widget_view == "rain_event"

    def test_one_millisecond_past_12_hours_it_does_not(self):
        # Measured from the build time: the archive's last row is hours older.
        archive = scenario_until("2026-01-09T12:00:00Z")
        catalog = event_catalog(archive)
        end = int(pd.Timestamp(catalog["end_time"].max()).value // 10**6)
        at = pd.Timestamp(end + 12 * 3_600_000 + 1, unit="ms", tz="UTC").isoformat()

        snapshot = build(archive, None, at, catalog)

        assert snapshot.feature.kind == "quiet"
        assert snapshot.widget_view == "temperature_ranges"


class TestRainNamesItsEvent:
    """A rain feature carries its event's `start_ms`, the Event Catalog key."""

    def test_a_live_event_is_named(self):
        snapshot = raining_station()

        live = [row for row in snapshot.rain_events.events if row.live]
        assert len(live) == 1
        assert snapshot.feature.event_start_ms == live[0].start_ms
        assert snapshot.feature.event_start_ms == ms("2026-01-08T22:00:00Z")  # event 3

    def test_an_ended_event_is_named_by_its_start(self):
        # Event 3 rained 22:00Z-02:00Z; six hours after, nothing is live.
        archive = scenario_until("2026-01-09T12:00:00Z")

        snapshot = build(archive, None, "2026-01-09T08:00:00Z", event_catalog(archive))

        assert snapshot.active_rain_event is None
        assert snapshot.feature.kind == "rain"
        assert snapshot.feature.event_start_ms == ms("2026-01-08T22:00:00Z")
        named = [
            row
            for row in snapshot.rain_events.events
            if row.start_ms == snapshot.feature.event_start_ms
        ]
        assert len(named) == 1
        assert not named[0].live

    def test_the_newest_ended_event_is_named_not_an_earlier_one(self):
        # Event 2 (07 Jan 04:00Z-14:00Z) ended after event 1; event 3 is to come.
        archive = scenario_until("2026-01-07T20:00:00Z")

        snapshot = build(archive, None, "2026-01-07T20:00:00Z", event_catalog(archive))

        assert snapshot.feature.kind == "rain"
        assert snapshot.feature.event_start_ms == ms("2026-01-07T04:00:00Z")

    def test_a_feature_past_the_rain_hours_names_no_event(self):
        archive = scenario_until("2026-01-09T12:00:00Z")
        catalog = event_catalog(archive)
        end = int(pd.Timestamp(catalog["end_time"].max()).value // 10**6)
        at = pd.Timestamp(end + 12 * 3_600_000 + 1, unit="ms", tz="UTC").isoformat()

        snapshot = build(archive, None, at, catalog)

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.event_start_ms is None

    def test_a_dry_spell_names_no_event(self):
        snapshot = build(rain_every_fifth_day(DRY_END), None, DRY_END)

        assert snapshot.feature.kind == "dry_spell"
        assert snapshot.feature.event_start_ms is None

    def test_the_dashboard_publishes_the_key(self):
        dashboard = raining_station().dashboard()

        assert dashboard["feature"]["kind"] == "rain"
        assert dashboard["feature"]["event_start_ms"] == ms("2026-01-08T22:00:00Z")


class TestDrySpell:
    def test_an_unusual_reading_beats_a_long_dry_spell(self):
        full = rain_every_fifth_day(DRY_END)
        archive = full[full["dateutc"] < ms(DRY_LAST)]
        live = live_row(full, DRY_LAST)
        live["tempf"] = 99.0

        snapshot = build(archive, live, DRY_END)

        assert snapshot.dry_spell.days_since_rain > snapshot.dry_spell.usual_days
        assert snapshot.feature.kind == "unusual"
        assert snapshot.widget_view == "temperature_ranges"

    def test_a_dry_spell_longer_than_usual_beats_quiet(self):
        snapshot = build(rain_every_fifth_day(DRY_END), None, DRY_END)

        assert snapshot.temperature_context.usual_band.basis == "earlier_years"
        assert snapshot.dry_spell.days_since_rain > snapshot.dry_spell.usual_days
        assert snapshot.feature.kind == "dry_spell"
        assert snapshot.feature.reading is None
        assert snapshot.widget_view == "dry_spell"

    def test_without_a_usual_dry_spell_it_is_quiet(self):
        snapshot = seasoned()

        assert snapshot.dry_spell.usual_days is None
        assert snapshot.feature.kind == "quiet"
        assert snapshot.widget_view == "temperature_ranges"


class TestPublished:
    def test_the_dashboard_carries_the_feature(self):
        dashboard = seasoned_station().dashboard()

        assert dashboard["feature"] == {
            "kind": "unusual",
            "reading": "temperature",
            "value": SEASONED_PUSHED_TEMPF,
            "unit": "f",
            "percentile": 100,
            "event_start_ms": None,
            "since_ms": ms(SEASONED_BUILT),
            "held": False,
            "outside_usual": True,
            "direction": "above",
        }

    def test_a_quiet_feature_carries_nulls(self):
        dashboard = seasoned().dashboard()

        assert dashboard["feature"] == {
            "kind": "quiet",
            "reading": None,
            "value": None,
            "unit": None,
            "percentile": None,
            "event_start_ms": None,
            "since_ms": ms(SEASONED_BUILT),
            "held": False,
            "outside_usual": None,
            "direction": None,
        }

    def test_a_snapshot_with_no_history_is_quiet(self):
        snapshot = build(pd.DataFrame({"dateutc": []}), None, "2026-01-10T20:00:00Z")

        assert snapshot.feature.kind == "quiet"
        assert snapshot.widget_view == "temperature_ranges"


BUILT_MS = ms(SEASONED_BUILT)
"""The seasoned build time: the hold counts back from it."""


def chosen(
    kind: FeatureKind,
    reading: Optional[FeatureReading] = None,
    ago_ms: int = HOUR_MS,
    direction: Optional[FeatureDirection] = None,
) -> Feature:
    """
    A last published feature, its subject chosen `ago_ms` before the build.

    Its values are deliberately stale, so a test sees them recomputed.
    `direction` is the side it was chosen on (#584).
    """
    return Feature(
        kind=kind,
        reading=reading,
        value=-1.0 if kind == "unusual" else None,
        unit=None,
        percentile=99 if kind == "unusual" else None,
        event_start_ms=None,
        since_ms=BUILT_MS - ago_ms,
        held=False,
        outside_usual=True if kind == "unusual" else None,
        direction=direction,
    )


class TestHold:
    """
    #533: a subject chosen under 3 hours ago is kept when the rule now picks
    a different one; rain takes over at once.
    """

    def test_with_no_previous_feature_the_subject_is_chosen_now(self):
        snapshot = seasoned()

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.since_ms == BUILT_MS
        assert snapshot.feature.held is False

    def test_a_subject_chosen_an_hour_ago_is_kept(self):
        # The rule picks quiet; the dry spell chosen an hour ago holds.
        snapshot = seasoned(previous=chosen("dry_spell"))

        assert snapshot.feature.kind == "dry_spell"
        assert snapshot.feature.held is True
        assert snapshot.feature.since_ms == BUILT_MS - HOUR_MS

    def test_the_widget_view_follows_the_held_feature(self):
        snapshot = seasoned(previous=chosen("dry_spell"))

        assert snapshot.widget_view == "dry_spell"

    def test_a_held_quiet_day_keeps_out_an_unusual_reading(self):
        snapshot = seasoned({"tempf": SEASONED_PUSHED_TEMPF}, previous=chosen("quiet"))

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.reading is None
        assert snapshot.feature.held is True

    def test_a_different_unusual_reading_is_a_different_subject(self):
        # Humidity is the furthest out now; temperature was chosen an hour ago.
        snapshot = seasoned(
            {"tempf": usual_tempf(0.95), "humidity": 99.0},
            previous=chosen("unusual", "temperature"),
        )

        assert snapshot.feature.reading == "temperature"
        assert snapshot.feature.held is True
        assert snapshot.feature.unit == "f"
        assert snapshot.feature.value == usual_tempf(0.95)
        # Recomputed this build (the stale 99 is gone), still above the 90th.
        assert 90 < snapshot.feature.percentile < 99
        assert snapshot.feature.outside_usual is True

    def test_just_under_3_hours_it_still_holds(self):
        previous = chosen("dry_spell", ago_ms=3 * HOUR_MS - 1)

        snapshot = seasoned(previous=previous)

        assert snapshot.feature.kind == "dry_spell"
        assert snapshot.feature.held is True

    def test_at_3_hours_the_rule_picks_again(self):
        snapshot = seasoned(previous=chosen("dry_spell", ago_ms=3 * HOUR_MS))

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == BUILT_MS

    def test_a_choice_time_after_the_build_holds_nothing(self):
        # A clock gone back must not hold a subject for ever.
        snapshot = seasoned(previous=chosen("dry_spell", ago_ms=-HOUR_MS))

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == BUILT_MS

    def test_the_same_subject_keeps_its_first_choice_time(self):
        previous = chosen("unusual", "temperature", ago_ms=5 * HOUR_MS)

        snapshot = seasoned({"tempf": SEASONED_PUSHED_TEMPF}, previous=previous)

        assert snapshot.feature.reading == "temperature"
        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == BUILT_MS - 5 * HOUR_MS
        # Picked by the rule this build, so its values are this build's.
        assert snapshot.feature.value == SEASONED_PUSHED_TEMPF
        assert snapshot.feature.percentile == 100

    def test_a_held_subject_chosen_again_is_no_longer_held(self):
        previous = chosen("unusual", "temperature")
        previous = replace(previous, held=True)

        snapshot = seasoned({"tempf": SEASONED_PUSHED_TEMPF}, previous=previous)

        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == BUILT_MS - HOUR_MS


class TestHeldUnusual:
    """
    A held unusual feature carries its reading as judged this build. Back
    inside the 10th to 90th, it says so: held, the percentile, and
    `outside_usual` False.
    """

    def test_a_reading_back_inside_its_band_is_not_outside_usual(self):
        # Humidity was chosen an hour ago; it is back at its usual 65 now.
        snapshot = seasoned(previous=chosen("unusual", "humidity"))

        feature = snapshot.feature
        assert (feature.kind, feature.reading, feature.held) == (
            "unusual",
            "humidity",
            True,
        )
        assert feature.value == snapshot.current.humidity_pct
        assert feature.unit == "pct"
        assert feature.percentile == 50
        assert feature.outside_usual is False
        assert snapshot.widget_view == "temperature_ranges"

    def test_a_held_reading_still_out_keeps_its_published_rounding(self):
        # Humidity is at the 0th; temperature ties it and wins the rule.
        snapshot = seasoned(
            {"tempf": SEASONED_PUSHED_TEMPF, "humidity": 40.0},
            previous=chosen("unusual", "humidity"),
        )

        assert snapshot.feature.reading == "humidity"
        assert snapshot.feature.percentile == 0
        assert snapshot.feature.outside_usual is True

    def test_a_held_reading_with_no_value_now_is_not_held(self):
        snapshot = seasoned({"tempf": None}, previous=chosen("unusual", "temperature"))

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == BUILT_MS

    def test_a_held_reading_gone_stale_is_not_held(self):
        built = after_live(FRESH_MS)
        previous = chosen("unusual", "temperature", ago_ms=0)
        previous = replace(previous, since_ms=ms(built) - HOUR_MS)

        snapshot = seasoned(ordinary=False, built=built, previous=previous)

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == ms(built)

    def test_a_held_reading_without_an_earlier_year_is_not_held(self):
        archive = build_archive("2025-12-08T08:00:00Z", "2026-01-06T20:00:00Z")
        live = live_row(archive, "2026-01-06T19:55:00Z")
        now = "2026-01-06T20:00:00Z"
        previous = chosen("unusual", "temperature")
        previous = replace(previous, since_ms=ms(now) - HOUR_MS)

        snapshot = build_snapshot(
            archive=archive,
            live=live,
            event_catalog=None,
            energy_catalog=None,
            station=SEASONED_STATION,
            now=pd.Timestamp(now),
            previous_feature=previous,
        )

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.held is False


class TestDirection:
    """
    #584: an unusual feature carries `direction`, the side of the usual band
    its subject was chosen on, kept unchanged through the hold.
    """

    def test_a_reading_chosen_above_its_band_says_above(self):
        snapshot = seasoned({"tempf": SEASONED_PUSHED_TEMPF})

        assert snapshot.feature.direction == "above"

    def test_a_reading_chosen_below_its_band_says_below(self):
        snapshot = seasoned({"tempf": usual_tempf(0.02)})

        assert snapshot.feature.reading == "temperature"
        assert snapshot.feature.outside_usual is True
        assert snapshot.feature.direction == "below"

    @pytest.mark.parametrize("kind", ["quiet", "dry_spell"])
    def test_a_feature_with_no_reading_has_no_direction(self, kind):
        assert seasoned(previous=chosen(kind)).feature.direction is None

    def test_a_held_reading_keeps_its_side_after_crossing_the_50th(self):
        # Chosen below its band an hour ago; temperature is now at its 80th,
        # above the median and inside the band. The rule picks quiet, so the
        # temperature is held.
        previous = chosen("unusual", "temperature", direction="below")

        snapshot = seasoned({"tempf": usual_tempf(0.8)}, previous=previous)

        feature = snapshot.feature
        assert feature.held is True
        assert feature.outside_usual is False
        assert feature.percentile is not None and feature.percentile > 50
        assert feature.direction == "below"

    def test_a_held_reading_keeps_its_side_when_another_wins_the_rule(self):
        # Humidity was chosen above its band; it is back at its usual 65, and
        # temperature is the furthest out below.
        previous = chosen("unusual", "humidity", direction="above")

        snapshot = seasoned({"tempf": usual_tempf(0.02)}, previous=previous)

        assert snapshot.feature.reading == "humidity"
        assert snapshot.feature.held is True
        assert snapshot.feature.direction == "above"

    def test_a_held_reading_out_on_the_other_side_keeps_its_first_side(self):
        # Chosen below; it is now above its band, but held, so unchanged.
        previous = chosen("unusual", "temperature", direction="below")

        snapshot = seasoned(
            {"tempf": usual_tempf(0.95), "humidity": 99.0}, previous=previous
        )

        assert snapshot.feature.reading == "temperature"
        assert snapshot.feature.held is True
        assert snapshot.feature.direction == "below"

    def test_a_held_reading_from_before_the_field_has_no_side(self):
        # Published before #584: its side was not recorded, and is not guessed.
        previous = chosen("unusual", "humidity", direction=None)

        snapshot = seasoned(previous=previous)

        assert snapshot.feature.held is True
        assert snapshot.feature.direction is None

    def test_the_same_reading_picked_again_says_its_side_now(self):
        previous = chosen("unusual", "temperature", ago_ms=5 * HOUR_MS)
        previous = replace(previous, direction="below")

        snapshot = seasoned({"tempf": SEASONED_PUSHED_TEMPF}, previous=previous)

        assert snapshot.feature.held is False
        assert snapshot.feature.direction == "above"

    def test_the_last_published_feature_is_read_with_its_direction(self):
        dashboard = {
            "feature": {
                "kind": "unusual",
                "reading": "humidity",
                "value": 40.0,
                "unit": "pct",
                "percentile": 4,
                "event_start_ms": None,
                "since_ms": BUILT_MS - HOUR_MS,
                "held": False,
                "outside_usual": True,
                "direction": "below",
            }
        }

        assert previous_feature(dashboard).direction == "below"
        del dashboard["feature"]["direction"]
        assert previous_feature(dashboard).direction is None
        dashboard["feature"]["direction"] = "sideways"
        assert previous_feature(dashboard).direction is None


class TestRainTakesOver:
    def test_rain_takes_over_inside_a_hold(self):
        rain = SEASONED_RAIN + (RainSpell("2026-05-20T18:00:00Z", 1, 0.1),)

        snapshot = seasoned(rain=rain, previous=chosen("unusual", "humidity"))

        assert snapshot.feature.kind == "rain"
        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == BUILT_MS
        assert snapshot.widget_view == "rain_event"

    def test_rain_is_not_held_once_its_rule_stops_holding(self):
        # The rain rule's own 12 hours after an event is its hold.
        snapshot = seasoned(previous=chosen("rain"))

        assert snapshot.feature.kind == "quiet"
        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == BUILT_MS
