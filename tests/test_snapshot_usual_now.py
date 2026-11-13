"""
Now against the usual for this hour (#518), through `build_snapshot`.

The usual band carries `now_delta`: the current reading's station-local
hour, the current temperature minus the band's median for that hour rounded
to whole degrees F, and the band's basis. It is null when there is no
current temperature, when the reading is stale by the Feature Rule's test
(`is_fresh` against the build's `now`), or when the band has no median for
that hour. Python rounds half away from zero, so +2.5 is +3 and -2.5 is -3.

The young stations end Tue 2026-01-06 noon local; the live reading is at
noon and the build two minutes later, unless a test moves it.
"""

from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from lookout.snapshot import Snapshot
from lookout.snapshot.model import UsualNow
from tests.fixtures.archive import build_archive
from tests.snapshot_helpers import (
    build,
    dry_station,
    live_row,
    ms,
    raining_station,
    seasoned_station,
)

TZ = "America/Los_Angeles"
LIVE = "2026-01-06T20:00:00Z"  # Tue noon local, the live reading
BUILT = "2026-01-06T20:02:00Z"
TODAY = date(2026, 1, 6)


def station(start: str) -> pd.DataFrame:
    """A station from `start` through the live reading, varied day by day."""
    archive = build_archive(start, "2026-01-06T20:05:00Z")  # end exclusive
    local = pd.to_datetime(archive["dateutc"], unit="ms", utc=True).dt.tz_convert(TZ)
    swing = 3.0 * np.sin(local.dt.dayofyear.to_numpy() * 0.9)
    archive["tempf"] = (archive["tempf"] + swing).round(1)
    return archive


def snapshot(
    archive: pd.DataFrame,
    built: str = BUILT,
    tempf: Optional[float] = None,
    no_temperature: bool = False,
) -> Snapshot:
    """
    `build_snapshot` with the archive's last row (at `LIVE`) as the live one.

    :param archive: The station, through `LIVE`.
    :param built: The build time.
    :param tempf: The live temperature, when a test sets it.
    :param no_temperature: The live reading carries no temperature.
    :return: The snapshot.
    """
    live: Dict = live_row(archive, LIVE)
    if tempf is not None:
        live["tempf"] = tempf
    if no_temperature:
        live["tempf"] = None
    stored = archive[archive["dateutc"] < ms(LIVE)].reset_index(drop=True)
    return build(stored, live, built)


def now_delta(snap: Snapshot) -> Optional[UsualNow]:
    """The snapshot's now against the usual for this hour."""
    return snap.temperature_context.usual_band.now_delta


@pytest.fixture(scope="module")
def recent() -> pd.DataFrame:
    """20 counted days before today: the band's basis is recent days."""
    return station("2025-12-17T08:00:00Z")


class TestSeasoned:
    def test_the_delta_is_from_the_earlier_years_band(self):
        snap = seasoned_station()
        usual = snap.temperature_context.usual_band

        assert usual.basis == "earlier_years"
        # The live reading is 72.0 at 3 pm PDT; hour 15's median is 60.1.
        assert usual.hours[15].median == 60.1
        assert snap.current.temperature_f == 72.0
        assert now_delta(snap) == UsualNow(hour=15, delta_f=12, basis="earlier_years")


class TestYoungStation:
    def test_the_delta_is_from_the_recent_days_band(self, recent):
        snap = snapshot(recent)
        usual = snap.temperature_context.usual_band
        days = {TODAY - timedelta(days=n) for n in range(1, 15)}
        local = pd.to_datetime(recent["dateutc"], unit="ms", utc=True)
        local = local.dt.tz_convert(TZ)
        at_noon = recent[local.dt.date.isin(days) & (local.dt.hour == 12)]
        median = round(float(np.median(at_noon["tempf"])), 1)

        assert usual.basis == "recent_days"
        # 54.8 now against a 59.2 noon median: -4.4, so -4.
        assert usual.hours[12].median == median == 59.2
        assert snap.current.temperature_f == 54.8
        assert now_delta(snap) == UsualNow(hour=12, delta_f=-4, basis="recent_days")

    def test_a_median_only_band_still_gives_the_delta(self):
        snap = snapshot(station("2026-01-03T08:00:00Z"))
        usual = snap.temperature_context.usual_band

        assert usual.basis == "median_only"
        # 54.8 now against a 56.5 noon median: -1.7, so -2.
        assert usual.hours[12].median == 56.5
        assert snap.current.temperature_f == 54.8
        assert now_delta(snap) == UsualNow(hour=12, delta_f=-2, basis="median_only")


class TestNull:
    def test_a_stale_reading_has_no_delta(self, recent):
        # 15 minutes old at the build: no longer fresh (STATION_FRESH_MINUTES).
        assert now_delta(snapshot(recent, built="2026-01-06T20:15:00Z")) is None

    def test_a_reading_just_under_15_minutes_old_still_has_one(self, recent):
        assert now_delta(snapshot(recent, built="2026-01-06T20:14:59Z")) is not None

    def test_no_current_temperature_has_no_delta(self, recent):
        snap = snapshot(recent, no_temperature=True)

        assert snap.current.temperature_f is None
        assert now_delta(snap) is None

    def test_no_median_for_the_hour_has_no_delta(self):
        # A first-day station: the band is over no days, so every median is null.
        snap = snapshot(station("2026-01-06T08:00:00Z"))

        assert snap.temperature_context.usual_band.hours[12].median is None
        assert snap.current.temperature_f is not None
        assert now_delta(snap) is None

    def test_an_empty_archive_without_a_live_reading_has_no_delta(self):
        empty = station("2025-12-17T08:00:00Z").iloc[0:0]

        assert now_delta(build(empty, None, BUILT)) is None


class TestRounding:
    @pytest.mark.parametrize(
        "offset, delta",
        [(2.5, 3), (-2.5, -3), (2.4, 2), (-2.4, -2), (0.4, 0), (-0.4, 0)],
    )
    def test_whole_degrees_half_away_from_zero(self, recent, offset, delta):
        median = snapshot(recent).temperature_context.usual_band.hours[12].median
        tempf = float(Decimal(str(median)) + Decimal(str(offset)))

        assert now_delta(snapshot(recent, tempf=tempf)).delta_f == delta


class TestGoldens:
    def test_the_dry_golden_carries_the_median_only_delta(self):
        # 39.4 at 11:55 pm against the band's 40.4: -1.0. Built at local
        # midnight, the reading's hour is read against today's band, the one
        # the snapshot carries.
        snap = dry_station()

        assert snap.current.temperature_f == 39.4
        assert snap.temperature_context.usual_band.hours[23].median == 40.4
        assert now_delta(snap) == UsualNow(hour=23, delta_f=-1, basis="median_only")

    def test_the_raining_golden_carries_the_median_only_delta(self):
        # 59.3 at 4:25 pm against 59.2: +0.1, so 0.
        snap = raining_station()

        assert snap.current.temperature_f == 59.3
        assert snap.temperature_context.usual_band.hours[16].median == 59.2
        assert now_delta(snap) == UsualNow(hour=16, delta_f=0, basis="median_only")

    def test_the_field_is_in_the_dashboard_document(self):
        doc = seasoned_station().dashboard()
        found = doc["temperature_context"]["usual_band"]["now_delta"]

        assert found == {"hour": 15, "delta_f": 12, "basis": "earlier_years"}
        assert doc["schema_version"] == 4

    def test_a_null_delta_is_json_null(self, recent):
        doc = snapshot(recent, built="2026-01-06T20:15:00Z").dashboard()

        assert doc["temperature_context"]["usual_band"]["now_delta"] is None
