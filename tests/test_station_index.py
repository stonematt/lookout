"""
The station index (#272): `snapshot/stations.json`, the list of stations the
site can show, which the refresh run publishes last, after every station.

Tested through the refresh run (`refresh.main`) against the moto bucket and
recorded Ambient pages, with the two-station setup of `tests/test_refresh.py`
(the fixture station and Salmon Run). Plus the golden station indexes the
lane server serves at `/stations` and the site's contract test reads, under
`tests/golden/stations/<scenario>.stations.json`. Regenerate them with:

    LOOKOUT_REGEN_GOLDEN=1 python -m pytest tests/test_station_index.py \
        -p no:cacheprovider
"""

import copy
import json
from pathlib import Path
from typing import Callable, Dict, List

import pytest

from lookout.cli import refresh
from lookout.snapshot.station_index import (
    STATION_INDEX_KEY,
    IndexEntry,
    StationIndex,
    read_station_index,
)
from lookout.storage import storj
from tests.fixtures.ambient import to_ambient_records
from tests.fixtures.archive import FAKE_MAC, SCENARIO, SEASONED_MAC, build_archive
from tests.fixtures.storage import BUCKET, flaky_storj, seed_inventory
from tests.snapshot_helpers import (
    REGENERATE,
    dry_station,
    raining_station,
    seasoned_station,
)
from tests.test_catchup import CatchupStation, ms, reporting
from tests.test_refresh import (  # noqa: F401 (seed, catchup_station, fixed_code)
    EVENT_CATALOG_KEY,
    SECOND_ARCHIVE_KEY,
    SECOND_MAC,
    SECOND_NAME,
    STORM_REPORT_TO,
    catchup_station,
    dashboard_key,
    fixed_code,
    second_device,
    seed,
    stored_json,
)

GOLDEN_DIR = Path(__file__).parent / "golden" / "stations"
ARCHIVE_TO = "2026-01-09T00:00:00Z"
TZ = "America/Los_Angeles"


@pytest.fixture
def two_stations(seed) -> Callable[..., CatchupStation]:  # noqa: F811
    """
    Both stations in the storm week, as `TestEveryStation` seeds them: the
    account lists Salmon Run first, and both archives start at the
    scenario's start.

    :return: `make(inventory=(FAKE_MAC, SECOND_MAC), second_archive=True)`.
    """

    def make(
        inventory: tuple = (FAKE_MAC, SECOND_MAC), second_archive: bool = True
    ) -> CatchupStation:
        full = build_archive(
            SCENARIO.start, STORM_REPORT_TO, rain=SCENARIO.rain, gaps=SCENARIO.gaps
        )
        archive = full[full["dateutc"] < ms(ARCHIVE_TO)]
        (first,) = reporting(full)
        station = seed(
            archive=archive,
            history=to_ambient_records(full),
            devices=[second_device(first), first],
            now=STORM_REPORT_TO,
        )
        records = station.replay.history
        station.replay.histories = {
            FAKE_MAC: records,
            SECOND_MAC: copy.deepcopy(records),
        }
        if second_archive:
            storj.save_df_to_s3(archive, BUCKET, SECOND_ARCHIVE_KEY)
        seed_inventory(*inventory)
        return station

    return make


def run_main(station: CatchupStation) -> int:
    """One refresh run, as the cron runs it, at the station's frozen now."""
    return refresh.main(["--bucket", BUCKET], clock=lambda: station.now)


def entry(station: CatchupStation, mac: str, name: str) -> Dict:
    """
    The index entry a station that published this run should have.

    :param station: The seeded station.
    :param mac: The station.
    :param name: Its Ambient name.
    :return: Its newest reading from its dashboard document; its archive
        start from the archive the test seeded.
    """
    return {
        "mac": mac,
        "name": name,
        "tz": TZ,
        "newest_reading_ms": stored_json(station, dashboard_key(mac))["observed_at_ms"],
        "archive_start_ms": int(station.archive["dateutc"].min()),
    }


def store_index(stations: List[Dict]) -> None:
    """Store a station index, as an earlier run left it."""
    storj.save_json_to_s3(
        {
            "schema_version": 1,
            "kind": "station_index",
            "built_at_ms": ms("2026-01-08T00:00:00Z"),
            "stations": stations,
        },
        BUCKET,
        STATION_INDEX_KEY,
    )


OLD_FIXTURE_ENTRY = {
    "mac": FAKE_MAC,
    "name": "Fixture Station (last good run)",
    "tz": TZ,
    "newest_reading_ms": ms("2026-01-08T00:00:00Z"),
    "archive_start_ms": ms("2025-12-01T00:00:00Z"),
}


class TestTheRefreshRunPublishesTheIndex:
    @pytest.mark.parametrize(
        "inventory", [(FAKE_MAC, SECOND_MAC), (SECOND_MAC, FAKE_MAC)]
    )
    def test_it_lists_every_station_in_inventory_order(self, two_stations, inventory):
        station = two_stations(inventory=inventory)

        assert run_main(station) == 0

        index = stored_json(station, STATION_INDEX_KEY)
        names = {FAKE_MAC: "Fixture Station", SECOND_MAC: SECOND_NAME}
        assert index == {
            "schema_version": 1,
            "kind": "station_index",
            "built_at_ms": ms(STORM_REPORT_TO),
            "stations": [entry(station, mac, names[mac]) for mac in inventory],
        }
        assert index["stations"][0]["archive_start_ms"] == ms(SCENARIO.start)

    def test_it_is_published_after_every_station(self, two_stations, caplog):
        station = two_stations()

        with caplog.at_level("INFO"):
            assert run_main(station) == 0

        published = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith(f"Published s3://{BUCKET}/snapshot/")
            and not r.getMessage().endswith(".status.json (ok)")
        ]
        assert published[-1] == f"Published s3://{BUCKET}/{STATION_INDEX_KEY}"
        assert published.count(published[-1]) == 1
        for mac in (FAKE_MAC, SECOND_MAC):
            assert f"Published s3://{BUCKET}/{dashboard_key(mac)}" in published

    def test_a_failed_station_keeps_its_last_good_entry(
        self, two_stations, monkeypatch
    ):
        station = two_stations()
        store_index([OLD_FIXTURE_ENTRY])
        # The fixture station's Event Catalog read fails: it publishes nothing.
        flaky_storj(monkeypatch, station.s3, read=EVENT_CATALOG_KEY)

        assert run_main(station) == 1

        assert dashboard_key(FAKE_MAC) not in station.keys()
        stations = stored_json(station, STATION_INDEX_KEY)["stations"]
        assert stations == [
            OLD_FIXTURE_ENTRY,
            entry(station, SECOND_MAC, SECOND_NAME),
        ]

    def test_a_failed_station_with_no_last_good_entry_is_left_out(self, two_stations):
        # Salmon Run has no archive, so it fails before it publishes.
        station = two_stations(second_archive=False)

        assert run_main(station) == 1

        stations = stored_json(station, STATION_INDEX_KEY)["stations"]
        assert [s["mac"] for s in stations] == [FAKE_MAC]

    def test_a_station_the_inventory_dropped_leaves_the_index(self, two_stations):
        station = two_stations(inventory=(SECOND_MAC,))
        store_index([OLD_FIXTURE_ENTRY])

        assert run_main(station) == 0

        stations = stored_json(station, STATION_INDEX_KEY)["stations"]
        assert [s["mac"] for s in stations] == [SECOND_MAC]

    def test_a_run_where_no_station_has_an_entry_writes_no_index(self, two_stations):
        station = two_stations(inventory=(SECOND_MAC,), second_archive=False)

        assert run_main(station) == 1

        assert STATION_INDEX_KEY not in station.keys()

    @pytest.mark.parametrize("stored", ["unreadable", "corrupt"])
    def test_a_stored_index_that_cannot_be_read_is_left_as_it_is(
        self, two_stations, monkeypatch, fake_healthchecks, caplog, stored
    ):
        station = two_stations(second_archive=False)
        if stored == "corrupt":
            station.s3.put_object(Bucket=BUCKET, Key=STATION_INDEX_KEY, Body=b"{")
        else:
            store_index([OLD_FIXTURE_ENTRY])
            flaky_storj(monkeypatch, station.s3, read=STATION_INDEX_KEY)
        before = station.s3.get_object(Bucket=BUCKET, Key=STATION_INDEX_KEY)[
            "Body"
        ].read()

        assert run_main(station) == 1

        # Salmon Run failed, and its last good entry cannot be known: the
        # index is not rewritten without it. The fixture station published.
        after = station.s3.get_object(Bucket=BUCKET, Key=STATION_INDEX_KEY)[
            "Body"
        ].read()
        assert after == before
        assert dashboard_key(FAKE_MAC) in station.keys()
        assert fake_healthchecks.pings[-1] == "lookout-refresh/1"
        assert "Station index not published" in caplog.text

    def test_a_stored_index_is_not_read_when_every_station_published(
        self, two_stations
    ):
        station = two_stations()
        station.s3.put_object(Bucket=BUCKET, Key=STATION_INDEX_KEY, Body=b"{")

        assert run_main(station) == 0

        stations = stored_json(station, STATION_INDEX_KEY)["stations"]
        assert [s["mac"] for s in stations] == [FAKE_MAC, SECOND_MAC]

    def test_an_index_write_that_fails_fails_the_run_but_not_the_stations(
        self, two_stations, monkeypatch, fake_healthchecks
    ):
        station = two_stations()
        flaky_storj(monkeypatch, station.s3, write=STATION_INDEX_KEY)

        assert run_main(station) == 1

        assert STATION_INDEX_KEY not in station.keys()
        assert fake_healthchecks.pings == [
            "lookout-refresh/start",
            "station-fixture-station",
            "station-salmon-run",
            "lookout-refresh/1",
        ]
        for mac in (FAKE_MAC, SECOND_MAC):
            status = stored_json(station, f"snapshot/{mac}.status.json")
            assert status["outcome"] == "ok"


# ---------- the golden indexes ----------


GOLDEN_ARCHIVE_START = ms("2022-05-04T17:00:00Z")
GOLDEN_SECOND = {
    "mac": SECOND_MAC,
    "name": SECOND_NAME,
    "tz": TZ,
    "newest_reading_ms": None,
    "archive_start_ms": None,
}
# Each golden index's first station: the scenario's own.
GOLDEN_FIRST = {
    "dry": {"mac": FAKE_MAC, "name": "Fixture Station"},
    "raining": {"mac": FAKE_MAC, "name": "Fixture Station"},
    "seasoned": {"mac": SEASONED_MAC, "name": "Seasoned Station"},
}
SCENARIOS = {
    "dry": dry_station,
    "raining": raining_station,
    "seasoned": seasoned_station,
}


class TestTheReaderRefusesWhatTheSiteRefuses:
    """`read_station_index` agrees with the site's `parseStationIndex`."""

    @staticmethod
    def stored(fake_s3, **changes) -> None:
        doc = {
            "schema_version": 1,
            "kind": "station_index",
            "built_at_ms": ms("2026-01-08T00:00:00Z"),
            "stations": [OLD_FIXTURE_ENTRY],
        }
        doc.update(changes)
        storj.save_json_to_s3(doc, BUCKET, STATION_INDEX_KEY)

    def test_a_good_index_is_read(self, fake_s3):
        self.stored(fake_s3)

        index = read_station_index(BUCKET)

        assert [e.mac for e in index.stations] == [FAKE_MAC]

    @pytest.mark.parametrize(
        "changes", [{"built_at_ms": True}, {"stations": []}], ids=["bool", "empty"]
    )
    def test_a_bool_time_or_no_stations_is_a_failed_read(self, fake_s3, changes):
        self.stored(fake_s3, **changes)

        with pytest.raises(storj.StorageReadError):
            read_station_index(BUCKET)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_station_indexes_match_the_golden_files(scenario):
    # The scenario's station first, then Salmon Run, which has no snapshot:
    # on the lane server, `?station=` for it answers 404.
    snapshot = SCENARIOS[scenario]()
    index = StationIndex(
        built_at_ms=snapshot.built_at_ms,
        stations=(
            IndexEntry(
                mac=snapshot.station.mac,
                name=snapshot.station.name,
                tz=snapshot.station.tz,
                newest_reading_ms=snapshot.observed_at_ms,
                archive_start_ms=GOLDEN_ARCHIVE_START,
            ),
            IndexEntry(**GOLDEN_SECOND),
        ),
    )

    path = GOLDEN_DIR / f"{scenario}.stations.json"
    text = json.dumps(index.document(), indent=2, allow_nan=False) + "\n"
    if REGENERATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert json.loads(path.read_text()) == {
        "schema_version": 1,
        "kind": "station_index",
        "built_at_ms": snapshot.built_at_ms,
        "stations": [
            {
                **GOLDEN_FIRST[scenario],
                "tz": TZ,
                "newest_reading_ms": snapshot.observed_at_ms,
                "archive_start_ms": GOLDEN_ARCHIVE_START,
            },
            GOLDEN_SECOND,
        ],
    }
