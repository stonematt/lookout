"""
A station is picked by its MAC address, never by Ambient's device order (#63).

The station inventory (ADR-0006) names the stations Lookout archives; its
first is the default station (#271). The lookup
(`ambient_client.find_station` / `get_station_device`) returns the named
device from the account's device list and raises `StationNotFoundError`
when it is not there, so a second device on the account can never silently
take over. A device the inventory does not list is ignored by every job,
the refresh run among them.
"""

from unittest.mock import patch

import pytest

from lookout.api import ambient_client
from lookout.api.ambient_client import StationNotFoundError
from lookout.cli import refresh
from lookout.storage.station_inventory import INVENTORY_KEY, resolve_station
from lookout.storage.storj import StorageReadError
from lookout.utils.archive_lock import hold_archive_lock
from tests.fixtures.ambient import devices_body, load_devices
from tests.fixtures.archive import FAKE_MAC
from tests.fixtures.storage import BUCKET, bucket_keys, seed_inventory
from tests.test_catchup import (  # noqa: F401 (catchup_station is a fixture)
    CatchupStation,
    catchup_station,
)

OTHER_MAC = "AA:BB:CC:DD:EE:FF"


def _two_stations() -> list:
    """The fixture station listed second, behind a newer one."""
    (fixture,) = load_devices()
    (other,) = devices_body(fixture["lastData"])
    other["macAddress"] = OTHER_MAC
    other["info"] = {"name": "New Station"}
    return [other, fixture]


@pytest.fixture
def inventory(fake_s3) -> str:
    """Store a station inventory listing only the fixture station."""
    seed_inventory(FAKE_MAC)
    return FAKE_MAC


def test_find_station_picks_the_named_mac_not_the_first():
    device = ambient_client.find_station(_two_stations(), FAKE_MAC)
    assert device["macAddress"] == FAKE_MAC


def test_find_station_fails_loudly_when_the_station_is_not_listed():
    with pytest.raises(StationNotFoundError, match="12:34:56:78:9A:BC"):
        ambient_client.find_station(_two_stations(), "12:34:56:78:9A:BC")


def test_find_station_fails_loudly_on_an_empty_device_list():
    # get_devices() answers [] on an API error; that must not pass for "none".
    with pytest.raises(StationNotFoundError, match="no devices"):
        ambient_client.find_station([], FAKE_MAC)


def test_find_station_matches_the_mac_case_insensitively():
    found = ambient_client.find_station(_two_stations(), f" {FAKE_MAC.lower()} ")
    assert found["macAddress"] == FAKE_MAC


def test_get_station_device_returns_the_named_station(ambient_replay):
    ambient_replay.devices = _two_stations()
    device = ambient_client.get_station_device(OTHER_MAC)
    assert device["macAddress"] == OTHER_MAC


def test_resolve_station_defaults_to_the_inventory_s_first(inventory):
    assert resolve_station(BUCKET) == FAKE_MAC


def test_with_no_inventory_there_is_no_default_and_ambient_is_not_called(
    fake_s3, ambient_replay
):
    from lookout.cli import pull_archive

    with pytest.raises(StorageReadError, match="station_inventory.json is missing"):
        pull_archive.get_device_info(BUCKET)
    assert ambient_replay.calls == []


# ---------- catchup ----------


def _history_macs(replay) -> set:
    return {path.removeprefix("/devices/") for path, _ in replay.calls} - {"/devices"}


def test_catchup_updates_the_default_station_not_the_first(inventory, ambient_replay):
    from lookout.cli import catchup

    ambient_replay.devices = _two_stations()

    catchup.main("lookout", dry_run=True, pages=1, repair_archive=False)

    assert _history_macs(ambient_replay) == {FAKE_MAC}


def test_catchup_stops_before_touching_storage_when_the_default_is_missing(
    ambient_replay, fake_s3
):
    from lookout.cli import catchup

    seed_inventory("12:34:56:78:9A:BC")
    ambient_replay.devices = _two_stations()

    with pytest.raises(StationNotFoundError):
        catchup.main("lookout", dry_run=False, pages=1, repair_archive=False)

    assert _history_macs(ambient_replay) == set()
    assert bucket_keys(fake_s3) == [INVENTORY_KEY]


def test_catchup_command_exits_nonzero_when_the_default_is_missing(
    monkeypatch, ambient_replay, fake_s3
):
    import runpy

    seed_inventory("12:34:56:78:9A:BC")
    monkeypatch.setattr("sys.argv", ["catchup.py", "--dry-run"])

    with pytest.raises(SystemExit) as exited:
        runpy.run_module("lookout.cli.catchup", run_name="__main__")

    assert exited.value.code == 1


# ---------- the other CLIs ----------


def _cli_station_mac(cli: str) -> str:
    """Run one CLI's station lookup and return the MAC it picked."""
    if cli == "manage_day_periods":
        from lookout.cli import manage_day_periods

        return manage_day_periods.get_station()[0]
    if cli == "pull_archive":
        from lookout.cli import pull_archive

        return pull_archive.get_device_info(BUCKET)[0]
    from lookout.cli import rain_corrections_cli

    return rain_corrections_cli.get_station()[1]


@pytest.mark.parametrize(
    "cli", ["manage_day_periods", "pull_archive", "rain_corrections_cli"]
)
def test_clis_pick_the_default_station_not_the_first(cli, inventory, ambient_replay):
    ambient_replay.devices = _two_stations()
    assert _cli_station_mac(cli) == FAKE_MAC


def test_manage_day_periods_exits_when_the_default_is_missing(ambient_replay, fake_s3):
    from lookout.cli import manage_day_periods

    seed_inventory("12:34:56:78:9A:BC")
    with pytest.raises(SystemExit) as exited:
        manage_day_periods.get_station()
    assert exited.value.code == 1


def test_pull_archive_device_mac_overrides_the_default(fake_s3, ambient_replay):
    from lookout.cli import pull_archive

    seed_inventory(FAKE_MAC, OTHER_MAC)
    ambient_replay.devices = _two_stations()
    assert pull_archive.get_device_info(BUCKET, OTHER_MAC)[0] == OTHER_MAC


def test_pull_archive_refuses_a_device_mac_outside_the_inventory(
    inventory, ambient_replay, capsys
):
    from lookout.cli import pull_archive

    ambient_replay.devices = _two_stations()
    with pytest.raises(SystemExit) as exited:
        with patch("sys.argv", ["pull_archive", "--device-mac", OTHER_MAC]):
            pull_archive.main()
    assert exited.value.code == 1
    assert "not in the station inventory" in capsys.readouterr().out
    assert ambient_replay.calls == []


# ---------- the refresh run ----------


def test_the_refresh_run_ignores_a_device_not_in_the_inventory(
    catchup_station,  # noqa: F811 (the fixture)
):
    # The account lists a second device, first; the inventory lists only the
    # fixture station, so the run never asks for the other's history nor
    # writes anything under its MAC.
    station: CatchupStation = catchup_station(devices=_two_stations())

    with hold_archive_lock():  # as `refresh.main` holds it
        runs = refresh.run(bucket=BUCKET, pages=20, clock=lambda: station.now)

    assert [run.mac for run in runs] == [FAKE_MAC]
    assert runs[0].ok
    assert _history_macs(station.replay) == {FAKE_MAC}
    assert not [key for key in station.keys() if OTHER_MAC in key]
