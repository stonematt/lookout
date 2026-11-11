"""
The stations CLI (#270): `lookout.cli.stations`, the station inventory's writer.

The Ambient account lists two invented stations: the fixture station
(`FAKE_MAC`), whose archive is stored in the moto bucket, and a second one
(`NEW_MAC`) with no archive, whose history Ambient serves from recorded
pages built by `build_archive`. Nothing here touches a real secret, the
live Storj or the live Ambient API.
"""

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytest

from lookout.api import ambient_client
from lookout.cli import stations
from lookout.storage import station_inventory, storj
from lookout.utils.archive_lock import hold_archive_lock
from tests.fixtures.ambient import (
    AmbientReplay,
    FakeResponse,
    devices_body,
    to_ambient_records,
)
from tests.fixtures.archive import FAKE_MAC, Gap, build_archive
from tests.fixtures.storage import BUCKET, bucket_keys, flaky_storj

NEW_MAC = "00:11:22:33:44:66"
OFFLINE_MAC = "00:11:22:33:44:77"
ARCHIVE_KEY = f"{FAKE_MAC}.parquet"
NEW_ARCHIVE_KEY = f"{NEW_MAC}.parquet"
INVENTORY_KEY = station_inventory.INVENTORY_KEY
BACKUP_PREFIX = station_inventory.BACKUP_PREFIX


@dataclass
class Account:
    """
    The fake Ambient account and bucket a test runs the CLI against.

    :param s3: Client on the moto gateway.
    :param replay: The fake Ambient API; `.calls` lists each request.
    :param archive: The fixture station's stored archive, newest first.
    :param history: What Ambient holds for `NEW_MAC`, newest first.
    """

    s3: object
    replay: AmbientReplay
    archive: pd.DataFrame
    history: pd.DataFrame

    def keys(self) -> List[str]:
        """
        Every key in the bucket.

        :return: The keys, sorted.
        """
        return bucket_keys(self.s3)

    def raw(self, key: str) -> bytes:
        """
        A stored object's bytes.

        :param key: Its key.
        :return: The bytes.
        """
        return self.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()

    def inventory(self) -> Optional[List[str]]:
        """
        The stored inventory's MACs.

        :return: The list, or None when none is stored.
        """
        if INVENTORY_KEY not in self.keys():
            return None
        return json.loads(self.raw(INVENTORY_KEY))["stations"]

    def store_inventory(self, macs: List[str]) -> None:
        """
        Store an inventory, as an earlier `add` would have.

        :param macs: The MACs, in priority order.
        """
        storj.save_json_to_s3({"stations": macs}, BUCKET, INVENTORY_KEY)

    def history_calls(self, mac: str = NEW_MAC) -> List[Dict]:
        """
        The params of each history request for one station.

        :param mac: The station.
        :return: One params dict per `/devices/{mac}` call, in order.
        """
        return [p for path, p in self.replay.calls if path == f"/devices/{mac}"]


def two_devices(archive: pd.DataFrame, history: pd.DataFrame) -> List[Dict]:
    """
    The `/devices` body for an account with both invented stations.

    :param archive: The fixture station's archive; its newest row is `lastData`.
    :param history: The second station's history; likewise.
    :return: The fixture station first, then the second one.
    """
    first = devices_body(to_ambient_records(archive.head(1))[0])[0]
    second = devices_body(to_ambient_records(history.head(1))[0])[0]
    second = {**second, "macAddress": NEW_MAC, "info": {"name": "Second Station"}}
    return [first, second]


@pytest.fixture
def account(fake_s3, ambient_replay, frozen_today) -> Callable[..., Account]:
    """
    Seed the account: the fixture archive stored, the second station's history.

    :return: `seed(history=None)`; `history` is the second station's
        readings (three days by default).
    """

    def seed(history: Optional[pd.DataFrame] = None) -> Account:
        archive = build_archive("2026-01-01", "2026-01-02")
        if history is None:
            history = build_archive("2026-01-05", "2026-01-08")
        storj.save_df_to_s3(archive, BUCKET, ARCHIVE_KEY)
        ambient_replay.devices = two_devices(archive, history)
        ambient_replay.histories[NEW_MAC] = to_ambient_records(history)
        return Account(fake_s3, ambient_replay, archive, history)

    return seed


def run(argv: List[str], capsys: pytest.CaptureFixture) -> Tuple[int, str]:
    """
    Run the CLI as the shell does.

    :param argv: Arguments after the program name; `--bucket` is added.
    :param capsys: pytest's capture.
    :return: The exit status and stdout.
    """
    capsys.readouterr()
    status = stations.main([argv[0], "--bucket", BUCKET, *argv[1:]])
    return status, capsys.readouterr().out


def inventory_backups(account: Account) -> List[str]:
    """
    The inventory's backup copies.

    :param account: The account after a run.
    :return: Their keys, sorted.
    """
    return [k for k in account.keys() if k.startswith(BACKUP_PREFIX)]


# ---------- scan ----------


class TestScan:
    def test_lists_every_device_with_no_inventory_and_writes_nothing(
        self, account, capsys
    ):
        acct = account()
        keys = acct.keys()

        status, out = run(["scan"], capsys)

        assert status == 0
        assert "Devices on the Ambient account: 2" in out
        assert FAKE_MAC in out and "Fixture Station" in out
        assert NEW_MAC in out and "Second Station" in out
        assert out.count("not in the inventory") == 2
        assert "No station inventory is stored yet" in out
        assert acct.keys() == keys

    def test_marks_inventory_membership_and_the_default_station(self, account, capsys):
        acct = account()
        acct.store_inventory([FAKE_MAC, OFFLINE_MAC])
        keys = acct.keys()

        status, out = run(["scan"], capsys)

        assert status == 0
        lines = out.splitlines()
        fixture = next(line for line in lines if FAKE_MAC in line)
        second = next(line for line in lines if NEW_MAC in line)
        offline = next(line for line in lines if OFFLINE_MAC in line)
        assert "in the inventory, #1 (default station)" in fixture
        assert "not in the inventory" in second
        assert "(not on the account)" in offline
        assert acct.keys() == keys

    def test_a_failed_inventory_read_fails_the_scan(self, account, capsys, monkeypatch):
        acct = account()
        acct.store_inventory([FAKE_MAC])
        keys = acct.keys()
        flaky_storj(monkeypatch, acct.s3, read=INVENTORY_KEY)

        status, out = run(["scan"], capsys)

        assert status == 1
        assert "scan failed" in out
        assert acct.keys() == keys

    def test_an_account_with_no_devices_is_refused(self, account, capsys):
        acct = account()
        acct.replay.devices = []

        status, out = run(["scan"], capsys)

        assert status == 1
        assert "Ambient returned no devices" in out


# ---------- add: adopt ----------


class TestAddAdopts:
    def test_the_first_add_seeds_the_inventory_and_adopts_the_archive_unchanged(
        self, account, capsys
    ):
        acct = account()
        before = acct.raw(ARCHIVE_KEY)

        status, out = run(["add", FAKE_MAC], capsys)

        assert status == 0
        assert "Adopted the stored archive, unchanged" in out
        assert "Seeded the station inventory" in out
        assert acct.inventory() == [FAKE_MAC]
        assert acct.raw(ARCHIVE_KEY) == before
        assert acct.history_calls(FAKE_MAC) == []
        assert inventory_backups(acct) == []  # nothing stored to back up

    def test_the_seeded_station_is_the_default_station(self, account, capsys):
        account()

        run(["add", FAKE_MAC.lower()], capsys)
        run(["add", NEW_MAC], capsys)

        inventory = station_inventory.load_inventory(BUCKET)
        assert inventory.macs == (FAKE_MAC, NEW_MAC)
        assert inventory.default == FAKE_MAC

    def test_a_corrupt_stored_archive_is_not_adopted(self, account, capsys):
        acct = account()
        acct.s3.put_object(Bucket=BUCKET, Key=ARCHIVE_KEY, Body=b"not parquet")

        status, out = run(["add", FAKE_MAC], capsys)

        assert status == 1
        assert "add failed" in out
        assert acct.inventory() is None
        assert acct.history_calls(FAKE_MAC) == []

    def test_a_stored_archive_with_no_valid_reading_is_not_adopted(
        self, account, capsys
    ):
        acct = account()
        storj.save_df_to_s3(acct.archive.assign(dateutc=0), BUCKET, ARCHIVE_KEY)
        before = acct.raw(ARCHIVE_KEY)

        status, out = run(["add", FAKE_MAC], capsys)

        assert status == 1
        assert "add failed" in out
        assert acct.raw(ARCHIVE_KEY) == before
        assert acct.inventory() is None
        assert acct.history_calls(FAKE_MAC) == []


# ---------- add: build the first archive ----------


class TestAddBuilds:
    def test_builds_the_first_archive_back_to_the_oldest_reading(self, account, capsys):
        acct = account()
        acct.store_inventory([FAKE_MAC])

        status, out = run(["add", NEW_MAC], capsys)

        assert status == 0
        assert "Built and saved the first archive" in out
        stored = storj.load_archive(BUCKET, NEW_MAC)
        assert stored["dateutc"].tolist() == sorted(acct.history["dateutc"].tolist())
        assert (
            stored["tempf"].tolist()
            == acct.history.sort_values("dateutc")["tempf"].tolist()
        )
        assert acct.inventory() == [FAKE_MAC, NEW_MAC]

    def test_a_first_build_takes_no_archive_backup(self, account, capsys):
        acct = account()
        acct.store_inventory([FAKE_MAC])

        status, _ = run(["add", NEW_MAC], capsys)

        assert status == 0
        assert [k for k in acct.keys() if NEW_MAC in k] == [NEW_ARCHIVE_KEY]

    def test_saves_the_archive_newest_first(self, account, capsys):
        account()

        run(["add", NEW_MAC], capsys)

        raw = storj.read_df_from_s3(BUCKET, NEW_ARCHIVE_KEY, file_type="parquet")
        assert raw["dateutc"].is_monotonic_decreasing

    def test_pages_back_from_now_until_ambient_has_nothing_older(self, account, capsys):
        acct = account()

        run(["add", NEW_MAC], capsys)

        calls = acct.history_calls()
        # Three days of 5-minute readings: three full pages, then an empty one.
        assert len(calls) == 4
        assert "endDate" not in calls[0]
        newest_first = sorted(acct.history["dateutc"], reverse=True)
        for page, call in enumerate(calls[1:], start=1):
            assert call["endDate"] == newest_first[page * 288 - 1]

    def test_crosses_a_gap_in_the_history(self, account, capsys):
        history = build_archive(
            "2026-01-01",
            "2026-01-08",
            gaps=[Gap("2026-01-02", "2026-01-06")],
        )
        account(history=history)

        status, _ = run(["add", NEW_MAC], capsys)

        assert status == 0
        stored = storj.load_archive(BUCKET, NEW_MAC)
        assert len(stored) == len(history)
        assert stored["dateutc"].min() == history["dateutc"].min()

    def test_an_inclusive_end_date_still_ends_the_walk(
        self, account, capsys, monkeypatch
    ):
        """Were Ambient's `endDate` inclusive, each page would repeat its end."""
        acct = account()
        replay = acct.replay

        def inclusive(url: str, params: Optional[Dict] = None, **kw: object) -> object:
            params = dict(params or {})
            if url.endswith(NEW_MAC) and params.get("endDate") is not None:
                params["endDate"] = int(params["endDate"]) + 1
            return replay(url, params, **kw)

        monkeypatch.setattr(ambient_client, "requests", SimpleNamespace(get=inclusive))

        status, _ = run(["add", NEW_MAC], capsys)

        assert status == 0
        stored = storj.load_archive(BUCKET, NEW_MAC)
        assert stored["dateutc"].tolist() == sorted(acct.history["dateutc"].tolist())

    def test_an_unreadable_history_page_fails_the_add(
        self, account, capsys, monkeypatch
    ):
        acct = account()
        acct.store_inventory([FAKE_MAC])
        replay = acct.replay

        def garbled(url: str, params: Optional[Dict] = None, **kw: object) -> object:
            if url.endswith(NEW_MAC) and (params or {}).get("endDate") is not None:
                return FakeResponse([{"no": "dateutc"}])
            return replay(url, params, **kw)

        monkeypatch.setattr(ambient_client, "requests", SimpleNamespace(get=garbled))

        status, out = run(["add", NEW_MAC], capsys)

        assert status == 1
        assert "add failed" in out
        assert NEW_ARCHIVE_KEY not in acct.keys()
        assert acct.inventory() == [FAKE_MAC]

    def test_ambient_paging_waits_a_second_before_every_request(
        self, account, capsys, monkeypatch
    ):
        acct = account()
        timeline: List[Tuple[str, float]] = []
        replay = acct.replay

        def get(url: str, params: Optional[Dict] = None, **kw: object) -> object:
            timeline.append(("get", 0.0))
            return replay(url, params, **kw)

        monkeypatch.setattr(ambient_client, "requests", SimpleNamespace(get=get))
        monkeypatch.setattr(
            ambient_client,
            "time",
            SimpleNamespace(sleep=lambda s: timeline.append(("sleep", s))),
        )

        status, _ = run(["add", NEW_MAC], capsys)

        assert status == 0
        gets = [i for i, (kind, _) in enumerate(timeline) if kind == "get"]
        assert len(gets) == 5  # the device list, then four history pages
        starts = [0] + [index + 1 for index in gets[:-1]]
        for start, index in zip(starts, gets):
            between = timeline[start:index]
            assert sum(s for kind, s in between if kind == "sleep") >= 1

    def test_a_station_ambient_holds_no_readings_for_is_refused(self, account, capsys):
        acct = account()
        acct.replay.histories[NEW_MAC] = []
        acct.store_inventory([FAKE_MAC])

        status, out = run(["add", NEW_MAC], capsys)

        assert status == 1
        assert "Ambient holds no readings" in out
        assert NEW_ARCHIVE_KEY not in acct.keys()
        assert acct.inventory() == [FAKE_MAC]


# ---------- add: refusals and failures ----------


class TestAddRefuses:
    def test_a_mac_not_on_the_account_is_refused(self, account, capsys):
        acct = account()
        keys = acct.keys()

        status, out = run(["add", OFFLINE_MAC], capsys)

        assert status == 1
        assert f"{OFFLINE_MAC} is not on the Ambient account" in out
        assert acct.keys() == keys

    def test_a_mac_already_in_the_inventory_is_refused(self, account, capsys):
        acct = account()
        acct.store_inventory([FAKE_MAC, NEW_MAC])
        keys = acct.keys()

        status, out = run(["add", NEW_MAC.lower()], capsys)

        assert status == 1
        assert "already in the station inventory, #2" in out
        assert acct.keys() == keys
        assert acct.history_calls() == []

    def test_a_failed_archive_read_is_not_taken_for_no_archive(
        self, account, capsys, monkeypatch
    ):
        acct = account()
        acct.store_inventory([FAKE_MAC])
        storj.save_df_to_s3(acct.history, BUCKET, NEW_ARCHIVE_KEY)
        before = acct.raw(NEW_ARCHIVE_KEY)
        flaky_storj(monkeypatch, acct.s3, read=NEW_ARCHIVE_KEY)

        status, out = run(["add", NEW_MAC], capsys)

        assert status == 1
        assert "add failed" in out
        assert acct.history_calls() == []
        assert acct.raw(NEW_ARCHIVE_KEY) == before
        assert acct.inventory() == [FAKE_MAC]

    def test_a_corrupt_inventory_fails_the_add(self, account, capsys):
        acct = account()
        acct.s3.put_object(Bucket=BUCKET, Key=INVENTORY_KEY, Body=b'{"stations": []}')

        status, out = run(["add", NEW_MAC], capsys)

        assert status == 1
        assert "add failed" in out
        assert NEW_ARCHIVE_KEY not in acct.keys()


class TestInventoryBackup:
    def test_the_inventory_is_backed_up_before_it_is_written(self, account, capsys):
        acct = account()
        acct.store_inventory([FAKE_MAC])
        before = acct.raw(INVENTORY_KEY)

        status, _ = run(["add", NEW_MAC], capsys)

        assert status == 0
        backups = inventory_backups(acct)
        assert len(backups) == 1
        assert acct.raw(backups[0]) == before
        assert acct.inventory() == [FAKE_MAC, NEW_MAC]

    def test_a_failed_backup_writes_nothing(self, account, capsys, monkeypatch):
        acct = account()
        acct.store_inventory([FAKE_MAC])
        keys = acct.keys()
        flaky_storj(monkeypatch, acct.s3, copy=True)

        status, out = run(["add", NEW_MAC], capsys)

        assert status == 1
        assert "add failed" in out
        assert acct.keys() == keys
        assert acct.inventory() == [FAKE_MAC]
        assert acct.history_calls() == []


class TestFailedBackfill:
    def test_a_failed_history_page_leaves_the_inventory_unchanged(
        self, account, capsys, monkeypatch
    ):
        acct = account()
        acct.store_inventory([FAKE_MAC])
        replay = acct.replay

        def failing(url: str, params: Optional[Dict] = None, **kw: object) -> object:
            if url.endswith(NEW_MAC) and (params or {}).get("endDate") is not None:
                replay.calls.append(("/devices/" + NEW_MAC, dict(params)))
                response = FakeResponse({"error": "server error"})
                response.status_code = 500
                return response
            return replay(url, params, **kw)

        monkeypatch.setattr(ambient_client, "requests", SimpleNamespace(get=failing))

        status, out = run(["add", NEW_MAC], capsys)

        assert status == 1
        assert "add failed" in out
        assert NEW_ARCHIVE_KEY not in acct.keys()
        assert acct.inventory() == [FAKE_MAC]

    def test_a_failed_inventory_write_is_recovered_by_adding_again(
        self, account, capsys, monkeypatch
    ):
        acct = account()
        acct.store_inventory([FAKE_MAC])
        flaky_storj(monkeypatch, acct.s3, write=INVENTORY_KEY)

        status, _ = run(["add", NEW_MAC], capsys)

        assert status == 1
        assert acct.inventory() == [FAKE_MAC]
        built = acct.raw(NEW_ARCHIVE_KEY)

        monkeypatch.setattr(storj, "get_s3_client", lambda: acct.s3)
        status, out = run(["add", NEW_MAC], capsys)

        assert status == 0
        assert "Adopted the stored archive, unchanged" in out
        assert acct.raw(NEW_ARCHIVE_KEY) == built
        assert acct.inventory() == [FAKE_MAC, NEW_MAC]


# ---------- the archive-writer lock (#244) ----------


class TestArchiveWriterLock:
    def test_a_held_lock_refuses_add_naming_the_lock_file(
        self, account, capsys, archive_lock_file
    ):
        acct = account()
        keys = acct.keys()

        with hold_archive_lock():
            status, out = run(["add", NEW_MAC], capsys)

        assert status == 1
        assert str(archive_lock_file) in out
        assert acct.replay.calls == []
        assert acct.keys() == keys

    def test_scan_runs_while_the_lock_is_held(self, account, capsys):
        acct = account()

        with hold_archive_lock():
            status, _ = run(["scan"], capsys)

        assert status == 0
        assert acct.replay.calls != []


# ---------- the inventory reader ----------


class TestInventoryReader:
    def test_missing_reads_as_none_and_load_refuses_it(self, fake_s3):
        assert station_inventory.read_inventory(BUCKET) is None
        with pytest.raises(storj.StorageReadError, match="missing"):
            station_inventory.load_inventory(BUCKET)

    @pytest.mark.parametrize(
        "body",
        [b"not json", b"[]", b'{"stations": []}', b'{"stations": ["A", "a"]}'],
        ids=["unparseable", "not-an-object", "empty", "duplicate"],
    )
    def test_a_corrupt_inventory_is_a_failed_read(self, fake_s3, body):
        fake_s3.put_object(Bucket=BUCKET, Key=INVENTORY_KEY, Body=body)

        with pytest.raises(storj.StorageReadError):
            station_inventory.read_inventory(BUCKET)

    def test_a_failed_read_raises(self, fake_s3, monkeypatch):
        flaky_storj(monkeypatch, fake_s3, read=INVENTORY_KEY)

        with pytest.raises(storj.StorageReadError):
            station_inventory.read_inventory(BUCKET)

    def test_the_inventory_lives_at_its_documented_key(self):
        # Outside `snapshot/`, which holds what the site reads.
        assert INVENTORY_KEY == "station_inventory.json"
