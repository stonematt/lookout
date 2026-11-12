"""
The feature (ADR-0009, #464), through `build_snapshot`.

The Feature Rule, checked in order, first that holds:

1. rain: an event is live, or the newest event ended at most 12 h before
   the build time;
2. unusual: a reading (temperature, humidity, wind gust, pressure, solar)
   is below the 10th or above the 90th percentile of its readings at this
   hour over the usual window's earlier years; skipped without an earlier
   year; the one furthest from the 50th wins, ties to the first in that
   list;
3. dry_spell: the current dry spell is longer than the usual dry spell;
4. quiet: otherwise.

The widget view follows: rain -> rain_event, dry_spell -> dry_spell, else
temperature_ranges.

The seasoned station (#461) has two earlier years around 20 May, so the
unusual rule applies to it; its live reading is at 15:00 PDT. Its humidity,
pressure and gust are the same at every 15:00 hour, so they sit at the 50th
until a test moves them; temperature and solar are set to their usual
median for the rules after unusual.
"""

from datetime import date, timedelta
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from lookout.snapshot.build import build_snapshot
from lookout.snapshot.model import Snapshot
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
) -> Snapshot:
    """
    The seasoned station, its live reading changed.

    :param changes: Live fields to set, Ambient names.
    :param rain: The station's rain spells.
    :param ordinary: Set the live temperature and solar to their usual median
        first, so neither is unusual unless `changes` says so.
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
        now=pd.Timestamp(SEASONED_BUILT),
    )


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
        # Temperature at the 95th is out; gust past every reading is further.
        snapshot = seasoned({"tempf": usual_tempf(0.95), "windgustmph": 30.0})

        assert snapshot.feature.reading == "wind_gust"
        assert snapshot.feature.unit == "mph"
        assert snapshot.feature.percentile == 100

    def test_each_reading_maps_to_its_column(self):
        cases = {
            "baromrelin": ("pressure", "inhg", 29.0),
            "solarradiation": ("solar", "wm2", 1000.0),
            "windgustmph": ("wind_gust", "mph", 30.0),
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
        }

    def test_a_quiet_feature_carries_nulls(self):
        dashboard = seasoned().dashboard()

        assert dashboard["feature"] == {
            "kind": "quiet",
            "reading": None,
            "value": None,
            "unit": None,
            "percentile": None,
        }

    def test_a_snapshot_with_no_history_is_quiet(self):
        snapshot = build(pd.DataFrame({"dateutc": []}), None, "2026-01-10T20:00:00Z")

        assert snapshot.feature.kind == "quiet"
        assert snapshot.widget_view == "temperature_ranges"
