"""
Sun and moon in the snapshot (#344): `build_snapshot(...).sun_moon`.

The station's coordinates come from the Ambient device, read with its zone on
every build (ADR-0005), and the block is computed for the station-local day.
A station Ambient sends no coordinates for builds with no block. Expected
sun times are published sunrise-table values for Salem, Oregon and Tokyo,
to two minutes; moon dates are the published new, quarter and full moons.
"""

from datetime import date
from typing import Optional

import pandas as pd
import pytest

from lookout.snapshot import Snapshot, Station, build_snapshot
from lookout.snapshot.model import StationLocation
from lookout.snapshot.sun_moon import moon_on
from tests.fixtures.ambient import load_devices
from tests.fixtures.archive import FAKE_MAC, build_archive

PACIFIC = "America/Los_Angeles"
TOKYO = "Asia/Tokyo"
SALEM = StationLocation(lat=44.94, lon=-123.03)
TOKYO_CITY = StationLocation(lat=35.68, lon=139.69)


def build_at(
    now: str, tz: str = PACIFIC, location: Optional[StationLocation] = SALEM
) -> Snapshot:
    """
    Build a station's snapshot at `now`, with a day of dry archive.

    :param now: The build time, UTC.
    :param tz: The station's zone.
    :param location: Its coordinates, or None when Ambient sent none.
    :return: The snapshot.
    """
    start = (pd.Timestamp(now) - pd.Timedelta(days=1)).isoformat()
    return build_snapshot(
        archive=build_archive(start, now, tz=tz),
        live=None,
        event_catalog=None,
        energy_catalog=None,
        station=Station(mac=FAKE_MAC, name="Fixture Station", tz=tz, location=location),
        now=pd.Timestamp(now),
    )


def local(ms: int, tz: str) -> pd.Timestamp:
    """An epoch-ms time in the station's zone."""
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert(tz)


def clock_minutes(ms: int) -> int:
    """Minutes past local midnight on the Pacific station's clock."""
    when = local(ms, PACIFIC)
    return when.hour * 60 + when.minute


def assert_near(ms: int, tz: str, expected: str) -> None:
    """`ms` is within two minutes of the local time `expected`."""
    gap = abs(local(ms, tz) - pd.Timestamp(expected, tz=tz))
    assert gap <= pd.Timedelta(minutes=2), f"{local(ms, tz)} vs {expected}"


# ---------- coordinates from the Ambient device ----------


def test_the_station_reads_its_coordinates_from_the_ambient_device():
    device = load_devices()[0]

    assert Station.from_device(device).location == StationLocation(45.0, -123.0)


def test_the_coordinates_follow_the_device_on_every_read():
    device = load_devices()[0]
    device["info"]["coords"]["coords"] = {"lat": 35.68, "lon": 139.69}
    device["lastData"]["tz"] = TOKYO

    assert Station.from_device(device).location == TOKYO_CITY


@pytest.mark.parametrize(
    "coords",
    [None, {}, {"coords": {}}, {"coords": {"lat": "x", "lon": 1}}],
)
def test_a_device_without_usable_coordinates_has_no_location(coords):
    device = load_devices()[0]
    if coords is None:
        del device["info"]["coords"]
    else:
        device["info"]["coords"] = coords

    assert Station.from_device(device).location is None


def test_the_published_station_does_not_carry_its_coordinates():
    snapshot = build_at("2026-01-07T20:00:00Z")

    assert "location" not in snapshot.dashboard()["station"]
    assert "location" not in snapshot.context()["station"]


# ---------- the block ----------


def test_a_station_with_coordinates_gets_sunrise_and_sunset_on_its_local_day():
    # 2026-01-07 12:00 local; USNO Salem: rise 07:50, set 16:46.
    sun_moon = build_at("2026-01-07T20:00:00Z").sun_moon

    assert sun_moon is not None
    assert_near(sun_moon.sunrise_ms, PACIFIC, "2026-01-07 07:50")
    assert_near(sun_moon.sunset_ms, PACIFIC, "2026-01-07 16:46")
    assert sun_moon.daylight_minutes == pytest.approx(536, abs=2)


def test_the_local_day_is_the_stations_not_utc():
    # 2026-01-07 03:00Z is still 6 January in Salem, already 7 January in Tokyo.
    pacific = build_at("2026-01-07T03:00:00Z").sun_moon
    tokyo = build_at("2026-01-07T03:00:00Z", TOKYO, TOKYO_CITY).sun_moon

    assert local(pacific.sunrise_ms, PACIFIC).date() == date(2026, 1, 6)
    assert local(pacific.sunset_ms, PACIFIC).date() == date(2026, 1, 6)
    assert local(tokyo.sunrise_ms, TOKYO).date() == date(2026, 1, 7)
    assert_near(tokyo.sunrise_ms, TOKYO, "2026-01-07 06:51")


@pytest.mark.parametrize(
    "now, day, jump",
    [
        # Spring forward: 8 March 2026, 02:00 PST -> 03:00 PDT, at 01:30 PST.
        ("2026-03-08T09:30:00Z", date(2026, 3, 8), 1),
        # Fall back: 1 November 2026, 02:00 PDT -> 01:00 PST, at 01:30 PST.
        ("2026-11-01T09:30:00Z", date(2026, 11, 1), -1),
    ],
)
def test_across_a_dst_change_the_sun_is_on_the_right_local_day(now, day, jump):
    sun_moon = build_at(now).sun_moon
    before = build_at((pd.Timestamp(now) - pd.Timedelta(days=1)).isoformat()).sun_moon

    assert local(sun_moon.sunrise_ms, PACIFIC).date() == day
    assert local(sun_moon.sunset_ms, PACIFIC).date() == day
    # The clock moved an hour; the sun about two minutes.
    moved = clock_minutes(sun_moon.sunrise_ms) - clock_minutes(before.sunrise_ms)
    assert moved == pytest.approx(60 * jump, abs=3)


def test_a_station_without_coordinates_builds_with_no_block():
    snapshot = build_at("2026-01-07T20:00:00Z", location=None)

    assert snapshot.sun_moon is None
    assert snapshot.dashboard()["sun_moon"] is None


@pytest.mark.parametrize(
    "now, sign",
    [
        ("2026-03-20T20:00:00Z", 1),  # spring: days lengthen
        ("2026-09-22T20:00:00Z", -1),  # autumn: days shorten
    ],
)
def test_the_daylight_change_is_signed_by_the_season(now, sign):
    sun_moon = build_at(now).sun_moon

    # Near the equinox at 45 N the day changes by about three minutes.
    assert sun_moon.daylight_change_minutes * sign >= 2


def test_the_daylight_change_is_today_against_yesterday():
    today = build_at("2026-03-20T20:00:00Z").sun_moon
    yesterday = build_at("2026-03-19T20:00:00Z").sun_moon

    change = today.daylight_minutes - yesterday.daylight_minutes
    assert today.daylight_change_minutes == pytest.approx(change, abs=1)


# ---------- the moon, against known dates ----------


def test_the_moon_on_a_known_new_moon():
    # New moon 8 April 2024, 18:21 UTC (the total solar eclipse).
    moon = moon_on(date(2024, 4, 8))

    assert moon.phase == "new"
    assert moon.illumination_pct <= 2
    assert moon.days_to_full == 15


def test_the_moon_on_a_known_full_moon():
    # Full moon 23 April 2024, 23:49 UTC.
    moon = moon_on(date(2024, 4, 23))

    assert moon.phase == "full"
    assert moon.illumination_pct >= 98
    assert moon.days_to_full == 0


def test_the_moon_at_its_quarters():
    # First quarter 15 April 2024 19:13 UTC; last quarter 1 May 2024 11:27 UTC.
    first = moon_on(date(2024, 4, 15))
    last = moon_on(date(2024, 5, 1))

    assert first.phase == "first quarter"
    assert 40 <= first.illumination_pct <= 60
    assert first.days_to_full == 8
    # The day after a principal phase takes the in-between name.
    assert moon_on(date(2024, 4, 16)).phase == "waxing gibbous"
    assert moon_on(date(2024, 4, 24)).phase == "waning gibbous"
    assert last.phase == "last quarter"
    assert 40 <= last.illumination_pct <= 60


def test_the_block_carries_the_moon_for_the_stations_local_day():
    # Full moon 23 April 2024 23:49 UTC: 16:49 on the 23rd in Salem, but
    # 08:49 on the 24th in Tokyo.
    pacific = build_at("2024-04-23T19:00:00Z").sun_moon
    tokyo = build_at("2024-04-23T03:00:00Z", TOKYO, TOKYO_CITY).sun_moon

    assert pacific.moon_phase == "full"
    assert pacific.days_to_full_moon == 0
    assert tokyo.moon_phase == "waxing gibbous"
    assert tokyo.days_to_full_moon == 1
