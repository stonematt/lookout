"""
The station is pinned by its MAC address, not by Ambient's device order (#63).

`STATION_MAC` names the one station Lookout archives. The lookup
(`ambient_client.find_station` / `get_station_device`) returns that device
from the account's device list and raises `StationNotFoundError` when it is
not there, so a second station on the account can never silently take over.
"""

import pytest

from lookout.api import ambient_client
from lookout.api.ambient_client import StationNotFoundError
from tests.fixtures.ambient import devices_body, load_devices
from tests.fixtures.archive import FAKE_MAC

OTHER_MAC = "AA:BB:CC:DD:EE:FF"


def _two_stations() -> list:
    """The pinned fake station listed second, behind a newer one."""
    (pinned,) = load_devices()
    (other,) = devices_body(pinned["lastData"])
    other["macAddress"] = OTHER_MAC
    other["info"] = {"name": "New Station"}
    return [other, pinned]


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin the fake station."""
    monkeypatch.setenv("STATION_MAC", FAKE_MAC)
    return FAKE_MAC


def test_find_station_picks_the_pinned_mac_not_the_first(pinned):
    device = ambient_client.find_station(_two_stations())
    assert device["macAddress"] == FAKE_MAC


def test_find_station_fails_loudly_when_the_pin_is_not_listed(monkeypatch):
    monkeypatch.setenv("STATION_MAC", "12:34:56:78:9A:BC")
    with pytest.raises(StationNotFoundError, match="12:34:56:78:9A:BC"):
        ambient_client.find_station(_two_stations())


def test_find_station_fails_loudly_on_an_empty_device_list(pinned):
    # get_devices() answers [] on an API error; that must not pass for "none".
    with pytest.raises(StationNotFoundError, match="no devices"):
        ambient_client.find_station([])


def test_find_station_matches_the_mac_case_insensitively(monkeypatch):
    monkeypatch.setenv("STATION_MAC", f" {FAKE_MAC.lower()} ")
    assert ambient_client.find_station(_two_stations())["macAddress"] == FAKE_MAC


def test_an_unset_pin_means_the_station_the_archive_is_keyed_under(monkeypatch):
    monkeypatch.delenv("STATION_MAC", raising=False)
    devices = _two_stations()
    devices[0]["macAddress"] = "98:CD:AC:22:0D:E5"
    assert ambient_client.find_station(devices) is devices[0]


def test_get_station_device_reads_the_account_device_list(pinned, ambient_replay):
    ambient_replay.devices = _two_stations()
    device = ambient_client.get_station_device()
    assert device["macAddress"] == FAKE_MAC
    assert [path for path, _ in ambient_replay.calls] == ["/devices"]


# ---------- catchup ----------


def _history_macs(replay) -> set:
    return {path.removeprefix("/devices/") for path, _ in replay.calls} - {"/devices"}


def test_catchup_updates_the_pinned_station_not_the_first(pinned, ambient_replay):
    from lookout.cli import catchup

    ambient_replay.devices = _two_stations()

    catchup.main("lookout", dry_run=True, pages=1, repair_archive=False)

    assert _history_macs(ambient_replay) == {FAKE_MAC}


def test_catchup_stops_before_touching_storage_when_the_pin_is_missing(
    monkeypatch, ambient_replay, fake_s3
):
    from lookout.cli import catchup

    monkeypatch.setenv("STATION_MAC", "12:34:56:78:9A:BC")
    ambient_replay.devices = _two_stations()

    with pytest.raises(StationNotFoundError):
        catchup.main("lookout", dry_run=False, pages=1, repair_archive=False)

    assert _history_macs(ambient_replay) == set()
    assert fake_s3.list_objects_v2(Bucket="lookout").get("KeyCount") == 0


def test_catchup_command_exits_nonzero_when_the_pin_is_missing(
    monkeypatch, ambient_replay
):
    import runpy

    monkeypatch.setenv("STATION_MAC", "12:34:56:78:9A:BC")
    monkeypatch.setattr("sys.argv", ["catchup.py", "--dry-run"])

    with pytest.raises(SystemExit) as exited:
        runpy.run_module("lookout.cli.catchup", run_name="__main__")

    assert exited.value.code == 1
