"""
One refresh run end to end (spec #66, seam 3): `lookout.cli.refresh`.

Extends the catchup end-to-end test: the station is seeded with
`tests.test_catchup.catchup_station` (archive in the moto bucket, recorded
Ambient pages, a fixed clock), then `refresh.run` runs instead of catchup.
Later links add to this module: catalog updates and the Event Catalog
auto-save (#73), the Healthchecks heartbeats (#74).
"""

import copy
import importlib.abc
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

import pandas as pd
import pytest

from lookout.api import awn_controller as awn
from lookout.cli import refresh
from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.rain_events import RainEventCatalog, is_event_ongoing
from lookout.snapshot import Snapshot, curves
from lookout.snapshot import status as run_status
from lookout.snapshot.status import CodeVersion
from lookout.snapshot.station_index import STATION_INDEX_KEY
from tests.snapshot_helpers import FIXED_CODE
from lookout.storage import storj
from lookout.storage.station_inventory import INVENTORY_KEY
from lookout.utils.archive_lock import LOCK_FILE_ENV, hold_archive_lock
from lookout.utils.log_util import LOG_FORMAT
from tests.fixtures.ambient import load_devices, to_ambient_records
from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    STATION_TZ,
    RainSpell,
    build_archive,
    text_as_category,
)
from tests.fixtures.real import AMBIENT_END, load_real_slice
from tests.fixtures.storage import BUCKET, FlakyStorj, flaky_storj, seed_inventory
from tests.test_catchup import (  # noqa: F401 (catchup_station is a fixture)
    ARCHIVE_KEY,
    BACKUP_KEY,
    RECORDED_ROWS,
    CatchupStation,
    catchup_station,
    ms,
    reporting,
)

DASHBOARD_KEY = f"snapshot/{FAKE_MAC}.dashboard.json"
CONTEXT_KEY = f"snapshot/{FAKE_MAC}.context.json"
CURVE_PREFIX = f"snapshot/{FAKE_MAC}.events/"
STATUS_KEY = f"snapshot/{FAKE_MAC}.status.json"
REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_IS_DEV_MODE = awn.is_dev_mode


def run_refresh(station: CatchupStation) -> Snapshot:
    """
    One refresh run, as the cron does it, at the station's frozen now.

    The inventory lists only the fixture station, so the run is that
    station's; its failure is raised here as the run caught it. The run
    holds the archive-writer lock, as `main` does, since the archive store
    refuses to save without it.

    :param station: The seeded station.
    :return: The published snapshot.
    """
    with hold_archive_lock():
        (run,) = refresh.run(bucket=BUCKET, pages=20, clock=lambda: station.now)
    if run.error is not None:
        raise run.error
    return run.snapshot


def stored_json(station: CatchupStation, key: str) -> Dict:
    """
    Read a JSON document back from the bucket.

    :param station: The seeded station.
    :param key: Its key.
    :return: The parsed document.
    """
    body = station.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return json.loads(body)


def snapshot_keys(station: CatchupStation) -> List[str]:
    """
    The snapshot documents' keys: under `snapshot/`, not curve files, not
    the run's status document (#268), which every run writes, and not the
    station index (#272), which every run that publishes writes.
    """
    return [
        key
        for key in station.keys()
        if key.startswith("snapshot/")
        and not key.startswith(CURVE_PREFIX)
        and not key.endswith(".status.json")
        and key != STATION_INDEX_KEY
    ]


@pytest.fixture(autouse=True)
def fixed_code(monkeypatch: pytest.MonkeyPatch) -> CodeVersion:
    """
    The code a run reports in its status document, fixed, so no test here
    asks git about the real checkout.

    :return: The fixed version and SHA.
    """
    monkeypatch.setattr(refresh, "code_version", lambda: FIXED_CODE)
    return FIXED_CODE


def curve_keys(station: CatchupStation) -> List[str]:
    """The event curve files' keys, `snapshot/<MAC>.events/<start_ms>.json`."""
    return [key for key in station.keys() if key.startswith(CURVE_PREFIX)]


@pytest.fixture
def seed(
    catchup_station: Callable[..., CatchupStation],  # noqa: F811 (the fixture)
) -> Callable[..., CatchupStation]:
    """
    `catchup_station` under a local name: seed the station for a refresh run.

    :return: `seed(archive=None, history=None, devices=None, now=None)`.
    """
    return catchup_station


# ---------- the run against recorded pages ----------


class TestRefreshRun:
    def test_both_snapshot_files_land_under_snapshot(self, seed):
        station = seed()

        snapshot = run_refresh(station)

        assert snapshot_keys(station) == [CONTEXT_KEY, DASHBOARD_KEY]
        assert stored_json(station, DASHBOARD_KEY) == snapshot.dashboard()
        assert stored_json(station, CONTEXT_KEY) == snapshot.context()

    def test_each_step_logs_one_peak_memory_line_in_run_order(self, seed, caplog):
        station = seed()

        with caplog.at_level("INFO"):
            run_refresh(station)

        steps = [
            m.group(1)
            for m in (
                re.fullmatch(r"Peak memory after (\w+): \d+ MB", r.getMessage())
                for r in caplog.records
            )
            if m
        ]
        assert steps == [
            "load_stations",
            "ambient_fetch",
            "archive_load",
            "archive_merge",
            "archive_save",
            "correct_rain",
            "load_catalogs",
            "update_catalogs",
            "read_previous",
            "build",
            "publish",
            "publish_curves",
        ]

    def test_the_archive_is_appended_first_with_the_schema_unchanged(self, seed):
        station = seed()

        run_refresh(station)

        stored = station.stored()
        assert len(stored) == len(station.archive) + RECORDED_ROWS
        assert stored["dateutc"].is_unique
        # The save strips the columns Lookout derived (#191); the rest stay.
        archive = station.archive.drop(columns=["hour", "direction_bin", "value_bin"])
        assert list(stored.columns) == list(archive.columns)
        # The text column is stored dictionary-encoded (#285); `passkey` and
        # `loc` hold nothing on the slice, so they stay as they are.
        assert stored.dtypes.astype(str).to_dict() == text_as_category(
            archive.dtypes, ["battrain"]
        )

    def test_the_snapshot_is_built_from_the_caught_up_archive(self, seed):
        station = seed()

        run_refresh(station)

        dashboard = stored_json(station, DASHBOARD_KEY)

        # The recorded pages reach the live reading (07:55Z), so they meet.
        assert dashboard["archive_newest_ms"] == ms("2025-11-19T07:55:00Z")
        assert dashboard["observed_at_ms"] == ms("2025-11-19T07:55:00Z")
        assert dashboard["built_at_ms"] == ms(AMBIENT_END)
        assert dashboard["station"] == {
            "mac": FAKE_MAC,
            "name": "Fixture Station",
            "tz": STATION_TZ,
        }
        assert dashboard["current"]["temperature_f"] == 39.4

    def test_a_second_run_republishes_over_the_first(self, seed):
        station = seed()
        run_refresh(station)

        run_refresh(station)

        assert snapshot_keys(station) == [CONTEXT_KEY, DASHBOARD_KEY]


class TestActiveEventFromTheStoredCatalog:
    @pytest.fixture
    def raining(self, seed) -> CatchupStation:
        """
        Mid event 3: the archive to 00:00Z with its Event Catalog stored,
        Ambient reporting to 00:25Z.
        """
        full = build_archive(
            SCENARIO.start,
            "2026-01-09T00:30:00Z",
            rain=SCENARIO.rain,
            gaps=SCENARIO.gaps,
        )
        archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
        catalog = RainEventCatalog(FAKE_MAC).detect_and_catalog_events(
            archive, auto_save=False
        )
        station = seed(
            archive=archive,
            history=to_ambient_records(full),
            devices=reporting(full),
            now="2026-01-09T00:30:00Z",
        )
        RainEventCatalog(FAKE_MAC).save_catalog(catalog)
        return station

    def test_the_ongoing_event_is_active_with_the_live_total(self, raining):
        event = run_refresh(raining).active_rain_event

        # The run re-detects the ongoing event, which mints a new id (held
        # finding); it is the stored catalog's event 3 by its start.
        assert event.start_ms == ms("2026-01-08T22:00:00Z")
        assert event.total_in == 0.3  # live at 00:25Z; the catalog says 0.24


# ---------- the shell entry point ----------


class TestMain:
    def test_a_published_run_exits_zero(self, seed):
        station = seed()

        assert refresh.main(["--bucket", BUCKET]) == 0
        assert snapshot_keys(station) == [CONTEXT_KEY, DASHBOARD_KEY]

    def test_a_failed_run_exits_one_and_publishes_nothing(self, seed, monkeypatch):
        station = seed()
        seed_inventory("AA:BB:CC:DD:EE:FF")  # a station not on the account

        assert refresh.main(["--bucket", BUCKET]) == 1
        assert snapshot_keys(station) == []


# ---------- no Streamlit on the refresh path ----------


class _BlockStreamlit(importlib.abc.MetaPathFinder):
    """Refuse every import of Streamlit, and remember the attempts."""

    def __init__(self) -> None:
        self.attempts: List[str] = []

    def find_spec(
        self, name: str, path: Optional[object] = None, target: Optional[object] = None
    ) -> None:
        if name == "streamlit" or name.startswith("streamlit."):
            self.attempts.append(name)
            raise ImportError(f"the refresh run imported {name}")
        return None


def test_a_run_never_reaches_streamlit(seed, monkeypatch):
    """
    With Streamlit unloaded and unimportable, a full run still publishes, and
    nothing on its path tried to import it (even where an error is caught).
    """
    station = seed()
    # Dev mode as production reads it: no DEV_MODE, no Streamlit secrets.
    monkeypatch.setattr(awn, "is_dev_mode", REAL_IS_DEV_MODE)
    monkeypatch.delenv("DEV_MODE", raising=False)
    for name in [m for m in sys.modules if m.split(".")[0] == "streamlit"]:
        monkeypatch.delitem(sys.modules, name)
    blocker = _BlockStreamlit()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    run_refresh(station)

    assert blocker.attempts == []
    assert snapshot_keys(station) == [CONTEXT_KEY, DASHBOARD_KEY]


def test_importing_the_refresh_command_loads_no_streamlit():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    code = (
        "import sys, lookout.cli.refresh\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'streamlit'))\n"
    )

    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert out.stdout.strip() == "[]"


# ---------- Healthchecks heartbeats (#74) ----------


class TestHeartbeats:
    def run_main(self, station):
        return refresh.main(["--bucket", BUCKET], clock=lambda: station.now)

    def test_a_published_run_sends_start_then_station_then_zero(
        self, seed, fake_healthchecks
    ):
        station = seed()

        assert self.run_main(station) == 0
        assert fake_healthchecks.pings[0] == "lookout-refresh/start"
        assert fake_healthchecks.pings[-1] == "lookout-refresh/0"
        assert len(fake_healthchecks.pings) == 3
        assert fake_healthchecks.pings[1].startswith("station-")

    def test_a_failed_run_sends_a_non_zero_status_and_no_station_ping(
        self, seed, fake_healthchecks, monkeypatch
    ):
        station = seed()
        seed_inventory("AA:BB:CC:DD:EE:FF")  # a station not on the account

        assert self.run_main(station) == 1
        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/1"]

    def test_a_stale_live_reading_skips_the_station_ping(self, seed, fake_healthchecks):
        station = seed()

        assert (
            refresh.main(
                ["--bucket", BUCKET], clock=lambda: station.now + timedelta(hours=2)
            )
            == 0
        )
        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/0"]

    def test_a_curve_write_failure_fails_a_published_run(
        self, seed, fake_healthchecks, monkeypatch, caplog
    ):
        station = TestCatalogUpdatesAndSaves().storm(seed, "2026-01-10T00:00:00Z")
        real_put = station.s3.put_object

        class FailingCurveClient:
            def __getattr__(self, name):
                return getattr(station.s3, name)

            def put_object(self, **kwargs):
                if kwargs["Key"].startswith(CURVE_PREFIX):
                    raise OSError("curve write refused")
                return real_put(**kwargs)

        monkeypatch.setattr(curves, "get_s3_client", lambda: FailingCurveClient())

        assert self.run_main(station) == 1

        assert sorted(snapshot_keys(station)) == sorted([DASHBOARD_KEY, CONTEXT_KEY])
        assert curve_keys(station) == []
        assert "curve write refused" in caplog.text
        assert fake_healthchecks.pings[0] == "lookout-refresh/start"
        assert fake_healthchecks.pings[1].startswith("station-")
        assert fake_healthchecks.pings[-1] == "lookout-refresh/1"

    def test_a_curve_listing_failure_fails_a_published_run(
        self, seed, fake_healthchecks, monkeypatch, caplog
    ):
        station = TestCatalogUpdatesAndSaves().storm(seed, "2026-01-10T00:00:00Z")
        failing = {"on": True}

        class ListingClient:
            def __getattr__(self, name):
                return getattr(station.s3, name)

            def get_paginator(self, name):
                if failing["on"]:
                    raise OSError("curve listing refused")
                return station.s3.get_paginator(name)

        monkeypatch.setattr(curves, "get_s3_client", lambda: ListingClient())

        assert self.run_main(station) == 1

        assert sorted(snapshot_keys(station)) == sorted([DASHBOARD_KEY, CONTEXT_KEY])
        assert curve_keys(station) == []
        assert "curve listing refused" in caplog.text
        assert fake_healthchecks.pings[-1] == "lookout-refresh/1"

        failing["on"] = False
        fake_healthchecks.pings.clear()

        assert self.run_main(station) == 0

        assert curve_keys(station) != []
        assert fake_healthchecks.pings[-1] == "lookout-refresh/0"

    def test_a_curve_failure_still_follows_a_stale_reading(
        self, seed, fake_healthchecks, monkeypatch
    ):
        station = TestCatalogUpdatesAndSaves().storm(seed, "2026-01-10T00:00:00Z")

        def refuse(*args, **kwargs):
            raise OSError("curve write refused")

        monkeypatch.setattr(refresh, "publish_event_curves", refuse)
        late = station.now + timedelta(hours=2)

        assert refresh.main(["--bucket", BUCKET], clock=lambda: late) == 1
        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/1"]

    def test_a_healthchecks_outage_does_not_fail_the_run(self, seed, fake_healthchecks):
        station = seed()
        fake_healthchecks.fail = True

        assert self.run_main(station) == 0
        assert len(snapshot_keys(station)) == 2

    @pytest.mark.parametrize(
        "fail",
        [{"read": ARCHIVE_KEY}, {"write": ARCHIVE_KEY}, {"copy": True}],
        ids=["archive-read", "archive-save", "archive-backup"],
    )
    def test_a_failed_archive_read_save_or_backup_reports_a_failed_run(
        self, seed, fake_healthchecks, monkeypatch, fail
    ):
        """#160: the run exits 1, `/1` goes out, nothing publishes."""
        station = seed()
        before = stored_bytes(station, ARCHIVE_KEY)
        flaky_storj(monkeypatch, station.s3, **fail)

        assert self.run_main(station) == 1

        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/1"]
        assert snapshot_keys(station) == []
        assert stored_bytes(station, ARCHIVE_KEY) == before

    def test_an_empty_archive_reports_a_failed_run_and_is_left_as_stored(
        self, seed, fake_healthchecks
    ):
        station = seed(archive=load_real_slice().head(0))
        before = stored_bytes(station, ARCHIVE_KEY)

        assert self.run_main(station) == 1

        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/1"]
        assert snapshot_keys(station) == []
        assert stored_bytes(station, ARCHIVE_KEY) == before


# ---------- the station's zone, from Ambient on every run (#197) ----------


def devices_with_zone(tz: object) -> List[Dict]:
    """
    The recorded `/devices` body, its `lastData.tz` set to `tz`.

    :param tz: The zone to report; `None` drops the field.
    :return: The body.
    """
    devices = load_devices()
    devices[0]["lastData"].pop("tz", None)
    if tz is not None:
        devices[0]["lastData"]["tz"] = tz
    return devices


class TestStationZone:
    def run_main(self, station: CatchupStation) -> int:
        return refresh.main(["--bucket", BUCKET], clock=lambda: station.now)

    def test_both_documents_name_the_zone_ambient_sends(self, seed):
        station = seed(devices=devices_with_zone("America/Denver"))

        run_refresh(station)

        assert stored_json(station, DASHBOARD_KEY)["station"]["tz"] == "America/Denver"
        assert stored_json(station, CONTEXT_KEY)["station"]["tz"] == "America/Denver"

    @pytest.mark.parametrize("tz", [None, "", "Mars/Olympus_Mons"])
    def test_no_usable_zone_exits_1_and_writes_nothing(
        self, seed, fake_healthchecks, tz
    ):
        """ADR-0005: no zone is guessed; the last good snapshot stays up."""
        station = seed(devices=devices_with_zone(tz))
        before = stored_bytes(station, ARCHIVE_KEY)

        assert self.run_main(station) == 1

        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/1"]
        assert snapshot_keys(station) == []
        assert stored_bytes(station, ARCHIVE_KEY) == before


# ---------- the catalogs: update on every run, save by rule (#73) ----------

EVENT_CATALOG_KEY = f"{FAKE_MAC}.event_catalog.parquet"
ENERGY_CATALOG_KEY = f"{FAKE_MAC}.energy_catalog.parquet"


def stored_bytes(station: CatchupStation, key: str) -> Optional[bytes]:
    """What a key holds, or None if it was never written."""
    try:
        return station.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except station.s3.exceptions.NoSuchKey:
        return None


class TestCatalogUpdatesAndSaves:
    def storm(self, seed, report_to: str) -> CatchupStation:
        """
        Event 3 ongoing at 00:00Z with its Event Catalog stored (as in
        `TestActiveEventFromTheStoredCatalog`), Ambient reporting to
        `report_to`.
        """
        full = build_archive(
            SCENARIO.start,
            report_to,
            rain=SCENARIO.rain,
            gaps=SCENARIO.gaps,
        )
        archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
        catalog = RainEventCatalog(FAKE_MAC).detect_and_catalog_events(
            archive, auto_save=False
        )
        station = seed(
            archive=archive,
            history=to_ambient_records(full),
            devices=reporting(full),
            now=report_to,
        )
        RainEventCatalog(FAKE_MAC).save_catalog(catalog)
        assert is_event_ongoing(catalog.sort_values("start_time").iloc[-1])
        return station

    def test_a_completing_event_saves_the_event_catalog(self, seed):
        station = self.storm(seed, "2026-01-10T00:00:00Z")
        before = stored_bytes(station, EVENT_CATALOG_KEY)

        run_refresh(station)

        stored = RainEventCatalog(FAKE_MAC).load_catalog()
        last = stored.sort_values("start_time").iloc[-1]
        assert not is_event_ongoing(last)
        assert last["total_rainfall"] == pytest.approx(0.48, abs=0.01)
        assert stored_bytes(station, EVENT_CATALOG_KEY) != before

    def test_a_still_ongoing_event_does_not_save_it(self, seed):
        station = self.storm(seed, "2026-01-09T00:30:00Z")
        before = stored_bytes(station, EVENT_CATALOG_KEY)
        stored_before = RainEventCatalog(FAKE_MAC).load_catalog()

        run_refresh(station)

        assert stored_bytes(station, EVENT_CATALOG_KEY) == before
        stored = RainEventCatalog(FAKE_MAC).load_catalog()
        assert stored.equals(stored_before)

    def test_a_completed_stored_event_is_not_saved_again(self, seed):
        station = self.storm(seed, "2026-01-10T00:00:00Z")
        run_refresh(station)  # completes the event and saves
        after_first = stored_bytes(station, EVENT_CATALOG_KEY)

        run_refresh(station)

        assert stored_bytes(station, EVENT_CATALOG_KEY) == after_first

    def test_the_snapshot_uses_the_updated_catalog_without_saving(self, seed):
        station = self.storm(seed, "2026-01-09T00:30:00Z")

        snapshot = run_refresh(station)

        # The stored catalog says 0.24 at 00:00Z; the run re-detected to 00:25Z.
        assert snapshot.active_rain_event.total_in == 0.3

    def test_no_stored_event_catalog_is_seeded_from_its_completed_events(self, seed):
        station = seed()  # the real slice holds completed events

        run_refresh(station)

        assert stored_bytes(station, EVENT_CATALOG_KEY) is not None
        assert stored_ongoing() and not any(stored_ongoing())

    def test_after_seeding_a_run_with_nothing_new_saves_nothing(self, seed):
        station = seed()

        run_refresh(station)  # seeds the Event Catalog
        seeded = stored_bytes(station, EVENT_CATALOG_KEY)
        run_refresh(station)

        assert stored_bytes(station, EVENT_CATALOG_KEY) == seeded
        assert stored_bytes(station, ENERGY_CATALOG_KEY) is None


# Two events after SCENARIO's three: event 4 starts and completes between two
# runs; event 5 is still wet at one run and completed by the next.
EVENT_4 = RainSpell("2026-01-10T02:00:00Z", hours=2, inches_per_hour=0.12)
EVENT_5 = RainSpell("2026-01-11T00:00:00Z", hours=2, inches_per_hour=0.12)
STEADY_STATE_RAIN = SCENARIO.rain + (EVENT_4, EVENT_5)
STEADY_STATE_END = "2026-01-11T12:00:00Z"


def storj_archive() -> pd.DataFrame:
    """The archive as the bucket holds it now."""
    return storj.get_df_from_s3(BUCKET, ARCHIVE_KEY, file_type="parquet")


def stored_starts() -> List[int]:
    """The stored Event Catalog's event starts, epoch ms, in order."""
    stored = RainEventCatalog(FAKE_MAC).load_catalog()
    starts = pd.to_datetime(stored["start_time"], utc=True).sort_values()
    return [int(ts.value // 1_000_000) for ts in starts]


def stored_ongoing() -> List[bool]:
    """Whether each stored event is ongoing, in start order."""
    stored = RainEventCatalog(FAKE_MAC).load_catalog().sort_values("start_time")
    return [is_event_ongoing(row) for _, row in stored.iterrows()]


def event_backups(station: CatchupStation) -> List[str]:
    """The Event Catalog's backup keys."""
    prefix = f"backups/{FAKE_MAC}.event_catalog"
    return [key for key in station.keys() if key.startswith(prefix)]


class CatalogCopyRefused(FlakyStorj):
    """`FlakyStorj` whose copies fail only from the Event Catalog's key."""

    def copy_object(self, **kwargs: object) -> Dict:
        if kwargs["CopySource"]["Key"] == EVENT_CATALOG_KEY:
            raise OSError("storj copy refused")
        return self.s3.copy_object(**kwargs)


class TestEventCatalogReadOutcome:
    """
    #180, ADR-0004: the run acts on the Event Catalog's read outcome. Failed
    aborts the run before any write; missing is detected and saved; a
    failed backup stops the save.
    """

    def run_main(self, station: CatchupStation) -> int:
        return refresh.main(["--bucket", BUCKET], clock=lambda: station.now)

    @pytest.mark.parametrize("corrupt", [False, True], ids=["read-fails", "corrupt"])
    def test_a_failed_read_exits_1_and_writes_no_catalog_or_snapshot(
        self, seed, fake_healthchecks, monkeypatch, corrupt
    ):
        station = TestCatalogUpdatesAndSaves().storm(seed, "2026-01-10T00:00:00Z")
        if corrupt:
            station.s3.put_object(
                Bucket=BUCKET, Key=EVENT_CATALOG_KEY, Body=b"not parquet"
            )
        before = stored_bytes(station, EVENT_CATALOG_KEY)
        if not corrupt:
            flaky_storj(monkeypatch, station.s3, read=EVENT_CATALOG_KEY)

        assert self.run_main(station) == 1

        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/1"]
        assert stored_bytes(station, EVENT_CATALOG_KEY) == before
        assert event_backups(station) == []
        assert stored_bytes(station, ENERGY_CATALOG_KEY) is None
        assert snapshot_keys(station) == []
        assert curve_keys(station) == []

    def test_a_missing_catalog_is_detected_and_saved(self, seed, fake_healthchecks):
        station = seed()  # no Event Catalog stored; the slice has completed events
        assert stored_bytes(station, EVENT_CATALOG_KEY) is None

        assert self.run_main(station) == 0

        assert stored_bytes(station, EVENT_CATALOG_KEY) is not None
        assert event_backups(station) == []  # nothing stored, nothing to back up
        assert len(snapshot_keys(station)) == 2

    def test_a_failed_backup_stops_the_save(self, seed, fake_healthchecks, monkeypatch):
        station = TestCatalogUpdatesAndSaves().storm(seed, "2026-01-10T00:00:00Z")
        before = stored_bytes(station, EVENT_CATALOG_KEY)
        flaky = CatalogCopyRefused(station.s3)
        monkeypatch.setattr(storj, "get_s3_client", lambda: flaky)

        assert self.run_main(station) == 0  # the save waits for the next run

        assert stored_bytes(station, EVENT_CATALOG_KEY) == before
        assert event_backups(station) == []
        assert len(snapshot_keys(station)) == 2


class TestEventCatalogSavesRunAfterRun:
    """
    The refresh run is the Event Catalog's only writer (#76 removes the save
    button), so every completed event must reach Storj, run after run.
    """

    @pytest.fixture
    def weather(self) -> pd.DataFrame:
        """SCENARIO's week and events 4 and 5, as Ambient would report them."""
        return build_archive(
            SCENARIO.start,
            STEADY_STATE_END,
            rain=STEADY_STATE_RAIN,
            gaps=SCENARIO.gaps,
        )

    def run_at(
        self,
        seed,
        weather: pd.DataFrame,
        now: str,
        archive: Optional[pd.DataFrame] = None,
    ) -> CatchupStation:
        """
        One refresh run at `now`, with Ambient reporting up to then.

        :param archive: The archive to start from; what the bucket holds
            when None, so a later run carries on from the one before.
        """
        if archive is None:
            archive = storj_archive()
        reported = weather[weather["dateutc"] < ms(now)]
        station = seed(
            archive=archive,
            history=to_ambient_records(reported),
            devices=reporting(reported),
            now=now,
        )
        run_refresh(station)
        return station

    def first_run(self, seed, weather: pd.DataFrame) -> CatchupStation:
        """
        Run 1: event 3 ongoing in the stored catalog at 00:00Z on the 9th;
        by 00:00Z on the 10th it has completed and nothing else has started.
        """
        archive = weather[weather["dateutc"] < ms("2026-01-09T00:00:00Z")]
        catalog = RainEventCatalog(FAKE_MAC).detect_and_catalog_events(
            archive, auto_save=False
        )
        assert is_event_ongoing(catalog.sort_values("start_time").iloc[-1])
        seed(archive=archive)
        RainEventCatalog(FAKE_MAC).save_catalog(catalog)
        return self.run_at(seed, weather, "2026-01-10T00:00:00Z", archive)

    def test_run_one_saves_when_the_stored_ongoing_event_completes(self, seed, weather):
        station = self.first_run(seed, weather)

        assert stored_starts() == [
            ms("2026-01-04T18:00:00Z"),
            ms("2026-01-07T04:00:00Z"),
            ms("2026-01-08T22:00:00Z"),
        ]
        assert stored_ongoing() == [False, False, False]
        assert event_backups(station)  # the stored one was backed up first

    def test_a_new_event_that_starts_and_completes_between_runs_is_saved(
        self, seed, weather
    ):
        station = self.first_run(seed, weather)
        after_run_1 = stored_bytes(station, EVENT_CATALOG_KEY)

        station = self.run_at(seed, weather, "2026-01-10T12:00:00Z")

        assert stored_bytes(station, EVENT_CATALOG_KEY) != after_run_1
        assert stored_starts()[-2:] == [
            ms("2026-01-08T22:00:00Z"),
            ms("2026-01-10T02:00:00Z"),
        ]
        assert stored_ongoing() == [False] * 4
        stored = RainEventCatalog(FAKE_MAC).load_catalog().sort_values("start_time")
        assert stored.iloc[-1]["total_rainfall"] == pytest.approx(0.24, abs=0.01)

    def test_a_run_with_nothing_new_writes_nothing(self, seed, weather):
        self.first_run(seed, weather)
        station = self.run_at(seed, weather, "2026-01-10T12:00:00Z")
        after_run_2 = stored_bytes(station, EVENT_CATALOG_KEY)
        backups = event_backups(station)

        station = self.run_at(seed, weather, "2026-01-10T18:00:00Z")

        assert stored_bytes(station, EVENT_CATALOG_KEY) == after_run_2
        assert event_backups(station) == backups

    def test_an_event_still_ongoing_is_not_saved_until_it_completes(
        self, seed, weather
    ):
        self.first_run(seed, weather)
        station = self.run_at(seed, weather, "2026-01-10T12:00:00Z")
        after_run_2 = stored_bytes(station, EVENT_CATALOG_KEY)

        station = self.run_at(seed, weather, "2026-01-11T01:00:00Z")  # 5 is wet

        assert stored_bytes(station, EVENT_CATALOG_KEY) == after_run_2
        assert len(stored_starts()) == 4

        self.run_at(seed, weather, STEADY_STATE_END)  # event 5 has completed

        assert stored_starts()[-1] == ms("2026-01-11T00:00:00Z")
        assert stored_ongoing() == [False] * 5

    def test_with_no_stored_catalog_the_first_completed_event_seeds_one(
        self, seed, weather
    ):
        dry = weather[weather["dateutc"] < ms("2026-01-04T12:00:00Z")]

        station = self.run_at(seed, weather, "2026-01-04T19:00:00Z", dry)

        assert stored_bytes(station, EVENT_CATALOG_KEY) is None  # event 1 wet

        self.run_at(seed, weather, "2026-01-05T08:00:00Z")  # event 1 done

        assert stored_starts() == [ms("2026-01-04T18:00:00Z")]
        assert stored_ongoing() == [False]


class TestEnergyCatalog:
    @pytest.fixture
    def stored_energy(self, seed) -> CatchupStation:
        """A station with an energy catalog stored from half the archive."""
        station = seed()
        archive = station.archive
        cut = (
            archive["dateutc"].min()
            + (archive["dateutc"].max() - archive["dateutc"].min()) // 2
        )
        catalog = EnergyCatalog(FAKE_MAC).detect_and_calculate_periods(
            archive[archive["dateutc"] < cut], auto_save=False
        )
        assert not catalog.empty
        EnergyCatalog(FAKE_MAC).save_catalog(catalog)
        return station

    def test_the_catalog_updates_in_memory_when_young(self, stored_energy):
        before = stored_bytes(stored_energy, ENERGY_CATALOG_KEY)

        run_refresh(stored_energy)

        assert stored_bytes(stored_energy, ENERGY_CATALOG_KEY) == before

    def test_a_catalog_over_two_days_old_is_saved_with_new_periods(
        self, stored_energy, monkeypatch
    ):
        stored_len = len(EnergyCatalog(FAKE_MAC).load_catalog())
        monkeypatch.setattr(
            EnergyCatalog, "get_catalog_age", lambda self: pd.Timedelta(days=3)
        )

        run_refresh(stored_energy)

        saved = EnergyCatalog(FAKE_MAC).load_catalog()
        assert len(saved) > stored_len
        assert saved["period_start"].is_unique


def energy_backups(station: CatchupStation) -> List[str]:
    """The energy catalog's backup keys, on either backup route."""
    return [
        key
        for key in station.keys()
        if f"{FAKE_MAC}.energy_catalog" in key and key != ENERGY_CATALOG_KEY
    ]


class TestEnergyCatalogReadOutcome:
    """
    #183, ADR-0004: the run acts on the energy catalog's read outcome, as on
    the Event Catalog's. Failed aborts the run before any write; missing is
    recomputed from the archive.
    """

    def run_main(self, station: CatchupStation) -> int:
        return refresh.main(["--bucket", BUCKET], clock=lambda: station.now)

    @pytest.mark.parametrize("corrupt", [False, True], ids=["read-fails", "corrupt"])
    def test_a_failed_read_exits_1_and_writes_no_catalog_or_snapshot(
        self, seed, fake_healthchecks, monkeypatch, corrupt
    ):
        station = seed()  # no Event Catalog stored: a run would seed one
        if corrupt:
            station.s3.put_object(
                Bucket=BUCKET, Key=ENERGY_CATALOG_KEY, Body=b"not parquet"
            )
        else:
            EnergyCatalog(FAKE_MAC).save_catalog(
                EnergyCatalog(FAKE_MAC).detect_and_calculate_periods(
                    station.archive, auto_save=False
                )
            )
        before = stored_bytes(station, ENERGY_CATALOG_KEY)
        monkeypatch.setattr(  # old enough that a run would save over it
            EnergyCatalog, "get_catalog_age", lambda self: pd.Timedelta(days=3)
        )
        if not corrupt:
            flaky_storj(monkeypatch, station.s3, read=ENERGY_CATALOG_KEY)

        assert self.run_main(station) == 1

        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/1"]
        assert stored_bytes(station, ENERGY_CATALOG_KEY) == before
        assert energy_backups(station) == []
        assert stored_bytes(station, EVENT_CATALOG_KEY) is None
        assert snapshot_keys(station) == []
        assert curve_keys(station) == []

    def test_a_missing_catalog_is_recomputed_from_the_archive(
        self, seed, fake_healthchecks, monkeypatch
    ):
        station = seed()
        assert stored_bytes(station, ENERGY_CATALOG_KEY) is None
        built_from: Dict = {}
        real = refresh.build_snapshot

        def spy(**kwargs):
            built_from.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(refresh, "build_snapshot", spy)

        assert self.run_main(station) == 0

        energy = built_from["energy_catalog"]
        assert energy is not None and not energy.empty
        assert len(snapshot_keys(station)) == 2


# ---------- event curve files, written once per completed event (#84) ----------

EVENT_1 = ms("2026-01-04T18:00:00Z")
EVENT_2 = ms("2026-01-07T04:00:00Z")
EVENT_3 = ms("2026-01-08T22:00:00Z")


def curve_key(start_ms: int) -> str:
    """A curve file's key."""
    return f"{CURVE_PREFIX}{start_ms}.json"


class TestEventCurveFiles:
    storm = TestCatalogUpdatesAndSaves.storm

    def test_each_completed_event_gets_its_curve_file(self, seed):
        station = self.storm(seed, "2026-01-10T00:00:00Z")

        run_refresh(station)

        assert curve_keys(station) == sorted(
            curve_key(start) for start in (EVENT_1, EVENT_2, EVENT_3)
        )
        document = stored_json(station, curve_key(EVENT_3))
        assert document["kind"] == "event_curve"
        assert document["start_ms"] == EVENT_3
        assert document["event_in"][-1] == pytest.approx(0.48)

    def test_the_ongoing_event_gets_no_curve_file(self, seed):
        station = self.storm(seed, "2026-01-09T00:30:00Z")

        snapshot = run_refresh(station)

        assert curve_keys(station) == sorted(
            curve_key(start) for start in (EVENT_1, EVENT_2)
        )
        # Its curve rides in the snapshot instead, under the same key.
        assert snapshot.rain_events.open_curve.start_ms == EVENT_3

    def test_a_curve_file_is_written_once_and_never_rewritten(self, seed):
        station = self.storm(seed, "2026-01-10T00:00:00Z")
        run_refresh(station)
        sentinel = b'{"written": "by the first run"}'
        station.s3.put_object(Bucket=BUCKET, Key=curve_key(EVENT_2), Body=sentinel)

        run_refresh(station)
        run_refresh(station)

        assert stored_bytes(station, curve_key(EVENT_2)) == sentinel

    def test_a_later_run_writes_nothing_when_every_curve_is_stored(
        self, seed, monkeypatch
    ):
        station = self.storm(seed, "2026-01-10T00:00:00Z")
        run_refresh(station)
        writes: List[str] = []
        real_put = station.s3.put_object

        def counting_put(**kwargs):
            writes.append(kwargs["Key"])
            return real_put(**kwargs)

        monkeypatch.setattr(station.s3, "put_object", counting_put)
        monkeypatch.setattr(curves, "get_s3_client", lambda: station.s3)

        run_refresh(station)

        assert [key for key in writes if key.startswith(CURVE_PREFIX)] == []

    def test_the_backfill_is_bounded_per_run_newest_first(self, seed, monkeypatch):
        station = self.storm(seed, "2026-01-10T00:00:00Z")
        monkeypatch.setattr(curves, "CURVE_BACKFILL_LIMIT", 2)

        run_refresh(station)
        first = curve_keys(station)
        run_refresh(station)

        assert first == sorted(curve_key(start) for start in (EVENT_2, EVENT_3))
        assert curve_keys(station) == sorted(
            curve_key(start) for start in (EVENT_1, EVENT_2, EVENT_3)
        )

    def test_the_event_list_and_the_curve_files_share_the_key(self, seed):
        station = self.storm(seed, "2026-01-10T00:00:00Z")

        run_refresh(station)

        listed = stored_json(station, DASHBOARD_KEY)["rain_events"]["events"]
        assert sorted(curve_key(row["start_ms"]) for row in listed) == (
            curve_keys(station)
        )

    def test_one_event_failing_to_write_does_not_stop_the_others(
        self, seed, monkeypatch, caplog
    ):
        """#160: the middle event's write fails; the older one still lands."""
        station = self.storm(seed, "2026-01-10T00:00:00Z")
        real_put = station.s3.put_object

        class OneBadKey:
            def __getattr__(self, name):
                return getattr(station.s3, name)

            def put_object(self, **kwargs):
                if kwargs["Key"] == curve_key(EVENT_2):
                    raise OSError("curve write refused")
                return real_put(**kwargs)

        monkeypatch.setattr(curves, "get_s3_client", lambda: OneBadKey())

        with pytest.raises(refresh.CurveStepFailed):
            run_refresh(station)

        assert curve_keys(station) == sorted(
            curve_key(start) for start in (EVENT_1, EVENT_3)
        )
        assert "curve write refused" in caplog.text
        assert str(EVENT_2) in caplog.text

    def test_one_event_failing_to_build_does_not_stop_the_others(
        self, seed, monkeypatch, caplog
    ):
        from lookout.snapshot import events

        station = self.storm(seed, "2026-01-10T00:00:00Z")
        real_curve = events._curve

        def curve(readings, start_ms, end_ms):
            if start_ms == EVENT_2:
                raise ValueError("readings out of order")
            return real_curve(readings, start_ms, end_ms)

        monkeypatch.setattr(events, "_curve", curve)

        with pytest.raises(refresh.CurveStepFailed):
            run_refresh(station)

        assert curve_keys(station) == sorted(
            curve_key(start) for start in (EVENT_1, EVENT_3)
        )
        assert "readings out of order" in caplog.text

    def test_a_curve_failing_to_build_fails_the_run_after_publish(
        self, seed, monkeypatch, fake_healthchecks
    ):
        """#233: not green; exit 1, `/1`, snapshot out, other curves landed."""
        from lookout.snapshot import events

        station = self.storm(seed, "2026-01-10T00:00:00Z")
        real_curve = events._curve

        def curve(readings, start_ms, end_ms):
            if start_ms == EVENT_2:
                raise ValueError("readings out of order")
            return real_curve(readings, start_ms, end_ms)

        monkeypatch.setattr(events, "_curve", curve)

        status = refresh.main(["--bucket", BUCKET], clock=lambda: station.now)

        assert status == 1
        assert fake_healthchecks.pings[-1] == "lookout-refresh/1"
        assert DASHBOARD_KEY in snapshot_keys(station)
        assert curve_keys(station) == sorted(
            curve_key(start) for start in (EVENT_1, EVENT_3)
        )


# ---------- rain corrections, applied once on read (#95) ----------


class TestRainCorrections:
    """
    The run saves the archive raw and publishes it corrected, once.

    The weather is SCENARIO's week with more rain on the carryover day
    (Friday 2026-01-09, local), so a second correction would show: Friday's
    raw daily counter peaks at 0.48" carried over plus 0.24" of its own rain.
    Corrected once it is 0.24"; corrected twice, 0.
    """

    CARRY = 0.48
    FRIDAY = "2026-01-09"
    FRIDAY_NOON = "2026-01-09T20:00:00Z"
    NOW = "2026-01-11T08:00:00Z"

    @pytest.fixture
    def station(self, seed) -> CatchupStation:
        """Archived to Friday's start; Ambient reports the rest of the week."""
        from lookout.core import rain_corrections

        weather = build_archive(
            SCENARIO.start,
            SCENARIO.end,
            rain=(*SCENARIO.rain, RainSpell("2026-01-09T18:00:00Z", 2, 0.12)),
            gaps=SCENARIO.gaps,
        )
        rain_corrections.save_catalog(
            {"corrections": rain_corrections.detect_carryovers(weather)}, FAKE_MAC
        )
        station = seed(
            archive=weather[weather["dateutc"] < ms("2026-01-09T08:00:00Z")],
            history=to_ambient_records(weather),
            devices=reporting(weather),
            now=self.NOW,
        )
        station.weather = weather
        return station

    def published_friday(self, station: CatchupStation) -> float:
        """Friday's daily rain as the published Rain tab has it."""
        recent = stored_json(station, DASHBOARD_KEY)["rain_tab"]["recent"]
        index = (pd.Timestamp(self.FRIDAY) - pd.Timestamp(recent["first_day"])).days
        return recent["rain_in"][index]

    def test_two_runs_publish_raw_minus_the_carryover_once(self, station):
        raw = station.weather.set_index("dateutc")
        raw_friday = raw.loc[ms(self.FRIDAY_NOON) + 6 * 3600 * 1000, "dailyrainin"]
        assert raw_friday == pytest.approx(self.CARRY + 0.24)

        run_refresh(station)
        first = self.published_friday(station)
        run_refresh(station)
        second = self.published_friday(station)

        assert first == second == pytest.approx(raw_friday - self.CARRY)

    def test_two_runs_leave_the_stored_counters_raw(self, station):
        run_refresh(station)
        run_refresh(station)

        merged = station.stored().merge(
            station.weather, on="dateutc", suffixes=("", "_raw")
        )
        assert len(merged) == len(station.weather)
        for col in ("dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin"):
            assert (merged[col] - merged[col + "_raw"]).abs().max() < 1e-9

    def test_an_unreadable_catalog_fails_the_run_and_publishes_nothing(
        self, station, fake_healthchecks, monkeypatch
    ):
        """
        #160: a catalog read error is not "no corrections". The run fails
        (`/1`) rather than publish Friday's rain uncorrected; the last good
        snapshot stays up and the next run retries.
        """
        flaky_storj(monkeypatch, station.s3, read=RAIN_CATALOG_KEY)

        assert refresh.main(["--bucket", BUCKET], clock=lambda: station.now) == 1

        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/1"]
        assert snapshot_keys(station) == []

    def test_a_corrupt_catalog_fails_the_run(self, station, fake_healthchecks):
        station.s3.put_object(Bucket=BUCKET, Key=RAIN_CATALOG_KEY, Body=b"{not json")

        assert refresh.main(["--bucket", BUCKET], clock=lambda: station.now) == 1

        assert snapshot_keys(station) == []


RAIN_CATALOG_KEY = f"{FAKE_MAC}.rain_corrections.json"


# ---------- the live reading, corrected like the archive (#118) ----------

CARRY_RAIN = (*SCENARIO.rain, RainSpell("2026-01-09T18:00:00Z", 2, 0.12))


def carryover_week(end: str = SCENARIO.end) -> pd.DataFrame:
    """
    SCENARIO's week with more rain on the carryover day, Friday 2026-01-09
    local (as `TestRainCorrections`), raw: newest first.
    """
    return build_archive(SCENARIO.start, end, rain=CARRY_RAIN, gaps=SCENARIO.gaps)


def store_corrections(weather: pd.DataFrame) -> List[Dict]:
    """Detect the week's carryovers and store them as the station's catalog."""
    from lookout.core import rain_corrections

    found = rain_corrections.detect_carryovers(weather)
    rain_corrections.save_catalog({"corrections": found}, FAKE_MAC)
    return found


class TestLiveRainCorrection:
    """
    The live reading carries the stored corrections the archive does.

    Ambient's history stops one reading short of the live one, so the live
    reading is stitched in as the newest row (as on the VPS between runs).
    """

    CARRY = 0.48
    FRIDAY = "2026-01-09"
    LIVE = "2026-01-09T23:55:00Z"  # Friday 15:55 local
    NOW = "2026-01-10T00:00:00Z"

    @pytest.fixture
    def station(self, seed) -> CatchupStation:
        """Archived to Friday's start, Ambient's history to just before live."""
        weather = carryover_week(self.NOW)
        assert int(weather["dateutc"].iloc[0]) == ms(self.LIVE)
        assert [c["affected_date"] for c in store_corrections(weather)] == [self.FRIDAY]
        station = seed(
            archive=weather[weather["dateutc"] < ms("2026-01-09T08:00:00Z")],
            history=to_ambient_records(weather.iloc[1:]),
            devices=reporting(weather),
            now=self.NOW,
        )
        station.weather = weather
        return station

    def test_rain_today_is_the_live_counter_minus_the_carryover(self, station):
        raw_live = float(station.weather["dailyrainin"].iloc[0])
        assert raw_live == pytest.approx(self.CARRY + 0.24)

        snapshot = run_refresh(station)

        assert snapshot.observed_at_ms == ms(self.LIVE)  # the live row is used
        expected = max(raw_live - self.CARRY, 0.0)
        assert snapshot.current.rain_today_in == pytest.approx(expected)
        published = stored_json(station, DASHBOARD_KEY)
        assert published["current"]["rain_today_in"] == pytest.approx(expected)

    def test_the_rain_tabs_today_agrees(self, station):
        run_refresh(station)

        published = stored_json(station, DASHBOARD_KEY)
        recent = published["rain_tab"]["recent"]
        index = (pd.Timestamp(self.FRIDAY) - pd.Timestamp(recent["first_day"])).days
        assert index == len(recent["rain_in"]) - 1  # Friday is today
        assert recent["rain_in"][index] == pytest.approx(
            published["current"]["rain_today_in"]
        )

    def test_the_stored_archive_stays_raw(self, station):
        run_refresh(station)

        merged = station.stored().merge(
            station.weather, on="dateutc", suffixes=("", "_raw")
        )
        assert len(merged) == len(station.weather) - 1  # all but the live row
        for col in refresh.RAIN_COUNTERS:
            assert (merged[col] - merged[col + "_raw"]).abs().max() < 1e-9


class TestCorrectRain:
    """`correct_rain` on the live reading: the archive's rules, its days."""

    @pytest.fixture
    def weather(self, fake_s3) -> pd.DataFrame:
        weather = carryover_week("2026-01-11T12:00:00Z")  # into Sunday
        store_corrections(weather)
        return weather

    STATION = refresh.Station(mac=FAKE_MAC, name=None, tz=STATION_TZ)

    @pytest.mark.parametrize(
        "reading",
        [
            "2026-01-08T20:00:00Z",  # Thursday: no correction covers it
            "2026-01-09T20:00:00Z",  # Friday, the affected day
            "2026-01-11T00:00:00Z",  # Saturday: the week, month and year
            "2026-01-11T07:55:00Z",  # Saturday 23:55 local, UTC Sunday
            "2026-01-11T10:00:00Z",  # Sunday: the week has reset
        ],
    )
    def test_each_counter_gets_the_archive_rows_correction(self, weather, reading):
        row = weather[weather["dateutc"] == ms(reading)]
        live = to_ambient_records(row)[0]

        archive, corrected = refresh.correct_rain(
            self.STATION, BUCKET, weather, dict(live)
        )

        fixed = archive[archive["dateutc"] == ms(reading)].iloc[0]
        for col in refresh.RAIN_COUNTERS:
            assert corrected[col] == pytest.approx(float(fixed[col]))
        assert {
            k: v for k, v in corrected.items() if k not in refresh.RAIN_COUNTERS
        } == {k: v for k, v in live.items() if k not in refresh.RAIN_COUNTERS}

    def test_a_day_no_correction_covers_is_unchanged(self, weather):
        live = to_ambient_records(
            weather[weather["dateutc"] == ms("2026-01-08T20:00:00Z")]
        )[0]

        _, corrected = refresh.correct_rain(self.STATION, BUCKET, weather, dict(live))

        assert corrected == live

    def test_the_saturday_live_reading_keeps_its_daily_counter(self, weather):
        reading = ms("2026-01-11T00:00:00Z")
        live = to_ambient_records(weather[weather["dateutc"] == reading])[0]

        _, corrected = refresh.correct_rain(self.STATION, BUCKET, weather, dict(live))

        assert corrected["dailyrainin"] == live["dailyrainin"]
        assert corrected["weeklyrainin"] == pytest.approx(
            max(live["weeklyrainin"] - TestLiveRainCorrection.CARRY, 0)
        )

    def test_after_the_week_resets_only_month_and_year_are_corrected(self, weather):
        reading = ms("2026-01-11T10:00:00Z")
        live = to_ambient_records(weather[weather["dateutc"] == reading])[0]

        _, corrected = refresh.correct_rain(self.STATION, BUCKET, weather, dict(live))

        carry = TestLiveRainCorrection.CARRY
        assert corrected["dailyrainin"] == live["dailyrainin"]
        assert corrected["weeklyrainin"] == live["weeklyrainin"]
        for col in ("monthlyrainin", "yearlyrainin"):
            assert corrected[col] == pytest.approx(max(live[col] - carry, 0))

    def test_a_counter_that_is_not_a_number_is_left_alone(self, weather):
        live = to_ambient_records(
            weather[weather["dateutc"] == ms("2026-01-09T20:00:00Z")]
        )[0]
        live["dailyrainin"] = "n/a"
        live["weeklyrainin"] = float("nan")

        _, corrected = refresh.correct_rain(self.STATION, BUCKET, weather, dict(live))

        assert corrected["dailyrainin"] == "n/a"
        assert pd.isna(corrected["weeklyrainin"])
        assert corrected["monthlyrainin"] < live["monthlyrainin"]

    def test_a_station_with_no_corrections_is_unchanged(self, fake_s3):
        weather = carryover_week()
        live = to_ambient_records(weather.head(1))[0]

        archive, corrected = refresh.correct_rain(self.STATION, BUCKET, weather, live)

        assert archive is weather
        assert corrected is live

    def test_a_catalog_read_error_raises_rather_than_reading_as_none(
        self, weather, fake_s3, monkeypatch
    ):
        flaky_storj(monkeypatch, fake_s3, read=RAIN_CATALOG_KEY)
        live = to_ambient_records(weather.head(1))[0]

        with pytest.raises(storj.StorageReadError):
            refresh.correct_rain(self.STATION, BUCKET, weather, live)

    def test_no_live_reading_stays_none(self, weather):
        _, corrected = refresh.correct_rain(self.STATION, BUCKET, weather, None)

        assert corrected is None


# ---------- one catalog's failure never blocks the snapshot (#118) ----------


class CatalogBroken(RuntimeError):
    """The error a broken catalog update raises in these tests."""


class TestCatalogFailureIsolation:
    """
    A catalog whose update raises is used as last stored and marked stale;
    the snapshot still publishes and the run exits 0.
    """

    @pytest.fixture
    def stored(self, seed) -> CatchupStation:
        """Both catalogs stored from the first half of the archive."""
        station = seed()
        archive = station.archive
        cut = (
            archive["dateutc"].min()
            + (archive["dateutc"].max() - archive["dateutc"].min()) // 2
        )
        half = archive[archive["dateutc"] < cut]
        EnergyCatalog(FAKE_MAC).save_catalog(
            EnergyCatalog(FAKE_MAC).detect_and_calculate_periods(half, auto_save=False)
        )
        RainEventCatalog(FAKE_MAC).save_catalog(
            RainEventCatalog(FAKE_MAC).detect_and_catalog_events(half, auto_save=False)
        )
        return station

    @pytest.fixture
    def built_from(self, monkeypatch) -> Dict:
        """The catalogs `build_snapshot` is handed, captured."""
        seen: Dict = {}
        real = refresh.build_snapshot

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(refresh, "build_snapshot", spy)
        return seen

    @staticmethod
    def breaks(monkeypatch, catalog: type) -> None:
        def refuse(self, *args, **kwargs):
            raise CatalogBroken(f"{catalog.__name__} update refused")

        monkeypatch.setattr(catalog, "update_catalog_with_new_data", refuse)

    def errors(self, caplog) -> List[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelname == "ERROR" and r.exc_info and r.exc_info[0] is CatalogBroken
        ]

    def test_an_energy_failure_publishes_with_the_stored_energy_catalog(
        self, stored, built_from, monkeypatch, caplog, fake_healthchecks
    ):
        last_good = EnergyCatalog(FAKE_MAC).load_catalog()
        before = stored_bytes(stored, ENERGY_CATALOG_KEY)
        self.breaks(monkeypatch, EnergyCatalog)

        assert refresh.main(["--bucket", BUCKET], clock=lambda: stored.now) == 0

        assert sorted(snapshot_keys(stored)) == sorted([DASHBOARD_KEY, CONTEXT_KEY])
        pd.testing.assert_frame_equal(built_from["energy_catalog"], last_good)
        assert built_from["event_catalog"] is not None  # the other still updated
        assert stored_bytes(stored, ENERGY_CATALOG_KEY) == before
        assert len(self.errors(caplog)) == 1
        assert "Energy catalog" in self.errors(caplog)[0]
        published = stored_json(stored, DASHBOARD_KEY)
        assert published["stale_catalogs"] == [
            {
                "catalog": "energy",
                "as_of_ms": refresh.to_ms(last_good["period_end"].max()),
            }
        ]
        assert published["solar_tab"] is not None
        assert fake_healthchecks.pings[-1] == "lookout-refresh/0"

    def test_an_event_failure_publishes_with_the_stored_event_catalog(
        self, stored, built_from, monkeypatch, caplog, fake_healthchecks
    ):
        last_good = RainEventCatalog(FAKE_MAC).load_catalog()
        before = stored_bytes(stored, EVENT_CATALOG_KEY)
        self.breaks(monkeypatch, RainEventCatalog)

        assert refresh.main(["--bucket", BUCKET], clock=lambda: stored.now) == 0

        assert sorted(snapshot_keys(stored)) == sorted([DASHBOARD_KEY, CONTEXT_KEY])
        pd.testing.assert_frame_equal(built_from["event_catalog"], last_good)
        assert built_from["energy_catalog"] is not None
        assert stored_bytes(stored, EVENT_CATALOG_KEY) == before
        assert len(self.errors(caplog)) == 1
        assert "Event Catalog" in self.errors(caplog)[0]
        published = stored_json(stored, DASHBOARD_KEY)
        assert published["stale_catalogs"] == [
            {
                "catalog": "events",
                "as_of_ms": refresh.to_ms(last_good["end_time"].max()),
            }
        ]
        assert len(published["rain_events"]["events"]) == len(last_good)
        assert fake_healthchecks.pings[-1] == "lookout-refresh/0"

    def test_a_failure_with_none_stored_marks_it_stale_without_a_date(
        self, seed, monkeypatch, caplog
    ):
        station = seed()
        self.breaks(monkeypatch, EnergyCatalog)

        snapshot = run_refresh(station)

        assert [asdict(s) for s in snapshot.stale_catalogs] == [
            {"catalog": "energy", "as_of_ms": None}
        ]
        assert snapshot.solar_tab is None
        assert len(self.errors(caplog)) == 1

    def test_a_failed_save_keeps_the_updated_catalog_fresh(
        self, seed, built_from, monkeypatch, caplog
    ):
        station = TestCatalogUpdatesAndSaves().storm(seed, "2026-01-10T00:00:00Z")

        def refuse(self, *args, **kwargs):
            raise CatalogBroken("save refused")

        monkeypatch.setattr(RainEventCatalog, "save_if_events_completed", refuse)

        snapshot = run_refresh(station)

        assert snapshot.stale_catalogs == ()
        last = built_from["event_catalog"].sort_values("start_time").iloc[-1]
        assert not is_event_ongoing(last)  # the update, completed, not the stored
        assert "Event Catalog not saved" in self.errors(caplog)[0]

    def test_without_a_failure_no_catalog_is_stale(self, stored, caplog):
        run_refresh(stored)

        assert stored_json(stored, DASHBOARD_KEY)["stale_catalogs"] == []
        assert not [
            r
            for r in caplog.records
            if r.levelname == "ERROR" and r.name == refresh.logger.name
        ]


# ---------- a run that changes nothing leaves the archive alone (#242) ----------


class TestNoopArchiveSave:
    def archive_writes(self, station, monkeypatch) -> List[str]:
        """Record every key written; the archive goes up by upload_fileobj."""
        writes: List[str] = []
        real_put = station.s3.put_object
        real_upload = station.s3.upload_fileobj

        def counting_put(**kwargs):
            writes.append(kwargs["Key"])
            return real_put(**kwargs)

        def counting_upload(**kwargs):
            writes.append(kwargs["Key"])
            return real_upload(**kwargs)

        monkeypatch.setattr(station.s3, "put_object", counting_put)
        monkeypatch.setattr(station.s3, "upload_fileobj", counting_upload)
        monkeypatch.setattr(storj, "get_s3_client", lambda: station.s3)
        return writes

    def test_a_merge_that_changes_nothing_saves_no_archive_and_takes_no_backup(
        self, seed, monkeypatch, caplog
    ):
        station = seed(history=[])
        writes = self.archive_writes(station, monkeypatch)
        before = stored_bytes(station, ARCHIVE_KEY)

        with caplog.at_level("INFO"):
            run_refresh(station)

        assert ARCHIVE_KEY not in writes
        assert stored_bytes(station, ARCHIVE_KEY) == before
        assert BACKUP_KEY not in station.keys()
        assert "skipped" in caplog.text.lower()

    def test_a_skipped_save_still_publishes_the_snapshot(self, seed):
        station = seed(history=[])

        snapshot = run_refresh(station)

        assert snapshot_keys(station) == [CONTEXT_KEY, DASHBOARD_KEY]
        assert stored_json(station, DASHBOARD_KEY) == snapshot.dashboard()

    def test_a_run_that_adds_a_reading_saves_the_archive(self, seed, monkeypatch):
        station = seed()
        writes = self.archive_writes(station, monkeypatch)

        run_refresh(station)

        assert ARCHIVE_KEY in writes
        assert len(station.stored()) == len(station.archive) + RECORDED_ROWS


# ---------- the archive-writer lock (#244) ----------


class TestArchiveWriterLock:
    def run_main(self, station: CatchupStation) -> int:
        return refresh.main(["--bucket", BUCKET], clock=lambda: station.now)

    def test_a_held_lock_skips_the_run_with_no_pings_and_exits_0(
        self, seed, fake_healthchecks, caplog
    ):
        station = seed()
        keys = station.keys()

        with hold_archive_lock():
            assert self.run_main(station) == 0

        assert fake_healthchecks.pings == []
        skips = [r for r in caplog.records if "holds the lock" in r.getMessage()]
        assert len(skips) == 1
        assert station.replay.calls == []
        assert station.keys() == keys

    def test_a_finished_run_leaves_the_lock_free(self, seed):
        station = seed()

        assert self.run_main(station) == 0

        with hold_archive_lock():
            pass

    def test_a_failed_run_leaves_the_lock_free(self, seed, monkeypatch):
        station = seed()
        seed_inventory("AA:BB:CC:DD:EE:FF")  # a station not on the account

        assert self.run_main(station) == 1

        with hold_archive_lock():
            pass

    def test_a_lock_that_cannot_be_opened_fails_the_run(
        self, seed, tmp_path, monkeypatch, fake_healthchecks, caplog
    ):
        # The lock's directory cannot be made: a file stands where it goes.
        station = seed()
        keys = station.keys()
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("")
        monkeypatch.setenv(LOCK_FILE_ENV, str(blocker / "archive-writer.lock"))

        assert self.run_main(station) == 1

        assert fake_healthchecks.pings[-1] == "lookout-refresh/1"
        assert station.replay.calls == []
        status = stored_json(station, STATUS_KEY)
        assert status["outcome"] == "failed"
        assert status["error"] in ("FileExistsError", "NotADirectoryError")
        assert set(station.keys()) - set(keys) == {STATUS_KEY}
        assert "Refresh failed" in caplog.text


# ---------- the run's status document (#268) ----------


class TestStatusDocument:
    """`main` writes `snapshot/<MAC>.status.json` at the end of every run."""

    def run_main(self, station: CatchupStation) -> int:
        return refresh.main(["--bucket", BUCKET], clock=lambda: station.now)

    def test_a_published_run_writes_an_ok_status_with_its_code(self, seed):
        station = seed()

        assert self.run_main(station) == 0

        assert stored_json(station, STATUS_KEY) == {
            "kind": "status",
            "station_mac": FAKE_MAC,
            "version": "2026.11.1",
            "sha": FIXED_CODE.sha,
            "started_at_ms": ms(AMBIENT_END),
            "ended_at_ms": ms(AMBIENT_END),
            "outcome": "ok",
            "error": None,
            "built_at_ms": stored_json(station, DASHBOARD_KEY)["built_at_ms"],
        }

    def test_the_run_is_timed_from_start_to_end(self, seed):
        station = seed()
        calls = []

        def clock():
            # The run's start, then 42 s later for every later call.
            calls.append(None)
            return station.now + timedelta(seconds=0 if len(calls) == 1 else 42)

        refresh.main(["--bucket", BUCKET], clock=clock)

        status = stored_json(station, STATUS_KEY)
        assert status["ended_at_ms"] - status["started_at_ms"] == 42_000

    def test_the_status_names_the_code_the_run_started_on(self, seed, monkeypatch):
        # A `git pull` from the next cron tick moves HEAD while this run works;
        # the status still names the commit the run started on.
        station = seed()
        started_on, pulled = "a" * 40, "b" * 40
        head = {"sha": started_on}
        real_load_stations = refresh.load_stations

        def load_stations_then_pull(bucket):
            runs = real_load_stations(bucket)
            head["sha"] = pulled
            return runs

        monkeypatch.setattr(refresh, "load_stations", load_stations_then_pull)

        def code() -> CodeVersion:
            return CodeVersion(version="2026.11.1", sha=head["sha"])

        assert (
            refresh.main(["--bucket", BUCKET], clock=lambda: station.now, code=code)
            == 0
        )

        status = stored_json(station, STATUS_KEY)
        assert (status["version"], status["sha"]) == ("2026.11.1", started_on)

    def test_a_failed_run_writes_a_failed_status_and_still_exits_1(
        self, seed, monkeypatch, fake_healthchecks
    ):
        station = seed()
        seed_inventory("AA:BB:CC:DD:EE:FF")  # a station not on the account

        assert self.run_main(station) == 1

        status = stored_json(station, "snapshot/AA:BB:CC:DD:EE:FF.status.json")
        assert (status["outcome"], status["error"]) == (
            "failed",
            "StationNotFoundError",
        )
        assert status["built_at_ms"] is None
        assert snapshot_keys(station) == []
        assert fake_healthchecks.pings[-1] == "lookout-refresh/1"

    def test_a_failed_archive_read_still_writes_the_status(self, seed, monkeypatch):
        station = seed()
        flaky_storj(monkeypatch, station.s3, read=ARCHIVE_KEY)

        assert self.run_main(station) == 1

        status = stored_json(station, STATUS_KEY)
        assert (status["outcome"], status["error"]) == ("failed", "StorageReadError")

    def test_a_curve_failure_after_publish_is_failed_with_the_snapshot_built(
        self, seed, monkeypatch
    ):
        station = TestCatalogUpdatesAndSaves().storm(seed, "2026-01-10T00:00:00Z")

        def refuse(*args, **kwargs):
            raise OSError("curve write refused at s3://secret-ish/path")

        monkeypatch.setattr(refresh, "publish_event_curves", refuse)

        assert self.run_main(station) == 1

        status = stored_json(station, STATUS_KEY)
        assert (status["outcome"], status["error"]) == ("failed", "CurveStepFailed")
        assert (
            status["built_at_ms"] == stored_json(station, DASHBOARD_KEY)["built_at_ms"]
        )
        assert "secret-ish" not in json.dumps(status)

    def test_a_failed_status_write_leaves_a_good_run_green(
        self, seed, monkeypatch, fake_healthchecks, caplog
    ):
        station = seed()
        real_put = station.s3.put_object

        class StatusRefused:
            def __getattr__(self, name):
                return getattr(station.s3, name)

            def put_object(self, **kwargs):
                if kwargs["Key"] == STATUS_KEY:
                    raise OSError("status write refused")
                return real_put(**kwargs)

        monkeypatch.setattr(run_status, "get_s3_client", lambda: StatusRefused())

        assert self.run_main(station) == 0

        assert STATUS_KEY not in station.keys()
        assert snapshot_keys(station) == [CONTEXT_KEY, DASHBOARD_KEY]
        assert fake_healthchecks.pings[-1] == "lookout-refresh/0"
        assert "Status document not written" in caplog.text

    def test_a_failed_status_write_does_not_mask_the_run_failure(
        self, seed, monkeypatch, fake_healthchecks, caplog
    ):
        station = seed()
        seed_inventory("AA:BB:CC:DD:EE:FF")  # a station not on the account

        def refuse():
            raise OSError("status write refused")

        monkeypatch.setattr(run_status, "get_s3_client", refuse)

        assert self.run_main(station) == 1

        assert fake_healthchecks.pings[-1] == "lookout-refresh/1"
        assert "Station AA:BB:CC:DD:EE:FF failed" in caplog.text
        assert "Status document not written" in caplog.text

    def test_an_interrupted_run_writes_a_failed_status(self, seed, monkeypatch):
        station = seed()

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(refresh, "load_stations", interrupt)

        with pytest.raises(KeyboardInterrupt):
            self.run_main(station)

        status = stored_json(station, STATUS_KEY)
        assert (status["outcome"], status["error"]) == ("failed", "KeyboardInterrupt")

    def test_a_skipped_run_writes_no_status(self, seed):
        station = seed()

        with hold_archive_lock():
            assert self.run_main(station) == 0

        assert STATUS_KEY not in station.keys()

    def test_run_alone_writes_no_status(self, seed):
        station = seed()

        run_refresh(station)

        assert STATUS_KEY not in station.keys()


# ---------- every station in the inventory (#271) ----------

SECOND_MAC = "00:11:22:33:44:66"
SECOND_NAME = "Salmon Run"
SECOND_ARCHIVE_KEY = f"{SECOND_MAC}.parquet"
FIRST_TAG = f"[Fixture Station {FAKE_MAC}] "
SECOND_TAG = f"[{SECOND_NAME} {SECOND_MAC}] "
STORM_REPORT_TO = "2026-01-10T00:00:00Z"
# Lookout Post's sensors a WS-2902 (Salmon Run) lacks: soil moisture, PM2.5
# and the WH31E.
WS2902_MISSING = (
    "soilhum1",
    "battsm1",
    "battsm2",
    "battsm3",
    "pm25",
    "pm25_24h",
    "aqi_pm25",
    "aqi_pm25_24h",
    "batt_25",
    "temp1f",
    "humidity1",
    "batt1",
    "feelsLike1",
    "dewPoint1",
)


def second_device(first: Dict, drop: tuple = ()) -> Dict:
    """
    The second station's `/devices` entry: the first's, renamed and re-keyed.

    :param first: The fixture station's entry.
    :param drop: `lastData` fields the second station does not report.
    :return: A new entry under `SECOND_MAC`, named `SECOND_NAME`.
    """
    device = copy.deepcopy(first)
    device["macAddress"] = SECOND_MAC
    device["info"] = {"name": SECOND_NAME}
    device["lastData"] = {k: v for k, v in device["lastData"].items() if k not in drop}
    return device


def without(records: List[Dict], fields: tuple) -> List[Dict]:
    """History records with some fields left out, as a station lacking them."""
    return [{k: v for k, v in r.items() if k not in fields} for r in records]


def dashboard_key(mac: str) -> str:
    """A station's dashboard document."""
    return f"snapshot/{mac}.dashboard.json"


def status_key(mac: str) -> str:
    """A station's status document."""
    return f"snapshot/{mac}.status.json"


class TestEveryStation:
    """
    One run refreshes every station in the station inventory, in its order:
    the fixture station and a second one, Salmon Run, on the same account.
    """

    @pytest.fixture
    def two_stations(self, seed) -> Callable[..., CatchupStation]:
        """
        Both stations in the storm week, the account listing Salmon Run
        first; neither has an Event Catalog stored, so each run detects and
        seeds one and writes the completed events' curve files.

        :return: `make(inventory=(FAKE_MAC, SECOND_MAC), second_archive=True)`.
        """

        def make(
            inventory: tuple = (FAKE_MAC, SECOND_MAC), second_archive: bool = True
        ) -> CatchupStation:
            full = build_archive(
                SCENARIO.start, STORM_REPORT_TO, rain=SCENARIO.rain, gaps=SCENARIO.gaps
            )
            archive = full[full["dateutc"] < ms("2026-01-09T00:00:00Z")]
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

    def run_main(self, station: CatchupStation) -> int:
        return refresh.main(["--bucket", BUCKET], clock=lambda: station.now)

    @pytest.mark.parametrize(
        "inventory", [(FAKE_MAC, SECOND_MAC), (SECOND_MAC, FAKE_MAC)]
    )
    def test_stations_run_in_inventory_order_and_each_publishes(
        self, two_stations, inventory, caplog
    ):
        station = two_stations(inventory=inventory)

        with caplog.at_level("INFO"):
            assert self.run_main(station) == 0

        published = [
            mac
            for r in caplog.records
            for mac in (FAKE_MAC, SECOND_MAC)
            if r.getMessage() == f"Published s3://{BUCKET}/{dashboard_key(mac)}"
        ]
        assert published == list(inventory)
        for mac, name in ((FAKE_MAC, "Fixture Station"), (SECOND_MAC, SECOND_NAME)):
            dashboard = stored_json(station, dashboard_key(mac))
            assert dashboard["station"]["mac"] == mac
            assert dashboard["station"]["name"] == name
            assert f"snapshot/{mac}.context.json" in station.keys()
            curves_written = [
                k for k in station.keys() if k.startswith(f"snapshot/{mac}.events/")
            ]
            assert curves_written, mac

    def test_all_ambient_calls_come_first_and_within_the_throttle(
        self, two_stations, monkeypatch
    ):
        from lookout.api import ambient_client

        station = two_stations()
        replay = station.replay
        timeline: List[str] = []

        def get(url, params=None, **kwargs):
            timeline.append("get")
            return replay(url, params, **kwargs)

        monkeypatch.setattr(ambient_client, "requests", SimpleNamespace(get=get))
        monkeypatch.setattr(
            ambient_client,
            "time",
            SimpleNamespace(sleep=lambda _s: timeline.append("sleep")),
        )
        real_backup = storj.backup_archive

        def backup(**kwargs):
            timeline.append(f"backup {kwargs['mac']}")
            return real_backup(**kwargs)

        monkeypatch.setattr(storj, "backup_archive", backup)

        assert self.run_main(station) == 0

        # One device list, then each station's history, then the stations.
        paths = [path for path, _ in replay.calls]
        assert paths[0] == "/devices" and paths.count("/devices") == 1
        assert {p for p in paths[1:]} == {
            f"/devices/{FAKE_MAC}",
            f"/devices/{SECOND_MAC}",
        }
        first_backup = timeline.index(f"backup {FAKE_MAC}")
        assert "get" not in timeline[first_backup:]
        later = timeline[first_backup:]
        assert later.count(f"backup {SECOND_MAC}") == 1
        # Every request waited out Ambient's rate limit first.
        for i, entry in enumerate(timeline):
            if entry == "get":
                assert timeline[i - 1] == "sleep"

    def test_a_failed_station_still_publishes_the_others_and_exits_1(
        self, two_stations, monkeypatch, fake_healthchecks, caplog
    ):
        station = two_stations()
        # The fixture station's Event Catalog read fails, in the station phase.
        flaky_storj(monkeypatch, station.s3, read=EVENT_CATALOG_KEY)

        assert self.run_main(station) == 1

        assert dashboard_key(SECOND_MAC) in station.keys()
        assert dashboard_key(FAKE_MAC) not in station.keys()
        failed = [
            r for r in caplog.records if f"Station {FAKE_MAC} failed" in r.getMessage()
        ]
        assert failed and failed[0].station == FIRST_TAG
        assert fake_healthchecks.pings == [
            "lookout-refresh/start",
            "station-salmon-run",
            "lookout-refresh/1",
        ]
        first = stored_json(station, status_key(FAKE_MAC))
        second = stored_json(station, status_key(SECOND_MAC))
        assert (first["outcome"], first["error"]) == ("failed", "StorageReadError")
        assert first["built_at_ms"] is None
        assert (second["outcome"], second["error"]) == ("ok", None)
        assert (
            second["built_at_ms"]
            == stored_json(station, dashboard_key(SECOND_MAC))["built_at_ms"]
        )

    def test_a_station_with_no_archive_fails_and_is_never_rebuilt(
        self, two_stations, caplog
    ):
        station = two_stations(second_archive=False)

        assert self.run_main(station) == 1

        assert dashboard_key(FAKE_MAC) in station.keys()
        assert not [k for k in station.keys() if SECOND_MAC in k and "status" not in k]
        assert f"/devices/{SECOND_MAC}" not in [p for p, _ in station.replay.calls]
        status = stored_json(station, status_key(SECOND_MAC))
        assert (status["outcome"], status["error"]) == ("failed", "StorageReadError")
        assert f"s3://{BUCKET}/{SECOND_ARCHIVE_KEY} is missing" in caplog.text

    def test_a_missing_inventory_aborts_the_run(
        self, two_stations, fake_healthchecks, caplog
    ):
        station = two_stations()
        station.s3.delete_object(Bucket=BUCKET, Key=INVENTORY_KEY)
        keys = station.keys()

        assert self.run_main(station) == 1

        assert station.replay.calls == []
        assert station.keys() == keys
        assert fake_healthchecks.pings[-1] == "lookout-refresh/1"
        assert "Refresh failed" in caplog.text
        assert "station_inventory.json is missing" in caplog.text

    def test_each_station_s_log_lines_carry_its_name_and_mac(
        self, two_stations, caplog
    ):
        station = two_stations()
        caplog.clear()  # only the run's own lines

        with caplog.at_level("INFO"):
            assert self.run_main(station) == 0

        formatter = logging.Formatter(LOG_FORMAT)
        lines = [formatter.format(r) for r in caplog.records if hasattr(r, "station")]
        second_reads = [line for line in lines if f"/{SECOND_ARCHIVE_KEY}" in line]
        assert second_reads and all(SECOND_TAG in line for line in second_reads)
        fetches = [line for line in lines if f"Fetching history: {SECOND_MAC}" in line]
        assert fetches and all(SECOND_TAG in line for line in fetches)
        peaks = [line for line in lines if "Peak memory after publish:" in line]
        assert [FIRST_TAG in peaks[0], SECOND_TAG in peaks[1]] == [True, True]
        # Nothing tagged as Salmon Run is about the fixture station's objects.
        assert not [
            line for line in lines if SECOND_TAG in line and f"{FAKE_MAC}." in line
        ]

    def test_the_run_ends_with_one_line_per_station(self, two_stations, caplog):
        station = two_stations(second_archive=False)

        with caplog.at_level("INFO"):
            self.run_main(station)

        last_two = [r.getMessage() for r in caplog.records[-2:]]
        assert re.fullmatch(
            rf"{FAKE_MAC} Fixture Station: published, newest reading \d+ min old",
            last_two[0],
        )
        assert re.fullmatch(
            rf"{SECOND_MAC} {SECOND_NAME}: failed \(StorageReadError\), "
            r"newest reading \d+ min old",
            last_two[1],
        )

    def test_a_station_without_soil_pm25_or_wh31e_builds_its_snapshot(self, seed):
        # The fixture station is the real slice, which has all three sensors;
        # Salmon Run, a WS-2902, has none of them in its archive or readings.
        station = seed()
        (first,) = load_devices()
        real = load_real_slice()
        lacking = real.drop(columns=[c for c in WS2902_MISSING if c in real.columns])
        storj.save_df_to_s3(lacking, BUCKET, SECOND_ARCHIVE_KEY)
        station.replay.devices = [first, second_device(first, drop=WS2902_MISSING)]
        station.replay.histories[SECOND_MAC] = without(
            station.replay.history, WS2902_MISSING
        )
        seed_inventory(FAKE_MAC, SECOND_MAC)

        assert self.run_main(station) == 0

        dashboard = stored_json(station, dashboard_key(SECOND_MAC))
        assert dashboard["station"]["mac"] == SECOND_MAC
        assert dashboard["observed_at_ms"] == ms("2025-11-19T07:55:00Z")
        stored = station.stored(SECOND_ARCHIVE_KEY)
        assert not set(WS2902_MISSING) & set(stored.columns)


# ---------- the feature hold reads the last published snapshot (#533) ----------


class TestTheHoldReadsTheLastSnapshot:
    """
    #533: the refresh run reads the last published dashboard snapshot and
    passes its feature to `build_snapshot`, so a subject chosen under three
    hours ago is held. Missing means no hold; a failed read fails the
    station (ADR-0004) and the last published snapshot stays up.
    """

    def run_main(self, station: CatchupStation) -> int:
        return refresh.main(["--bucket", BUCKET], clock=lambda: station.now)

    def publish_feature(self, station: CatchupStation, feature: Dict) -> None:
        """Rewrite the stored dashboard's feature, as an earlier run left it."""
        dashboard = stored_json(station, DASHBOARD_KEY)
        dashboard["feature"] = feature
        station.s3.put_object(
            Bucket=BUCKET, Key=DASHBOARD_KEY, Body=json.dumps(dashboard).encode()
        )

    def other_subject(self, snapshot: Snapshot) -> str:
        """A bare kind the rule did not pick: the one a hold would keep."""
        assert snapshot.feature.kind in ("quiet", "dry_spell", "unusual")
        return "dry_spell" if snapshot.feature.kind != "dry_spell" else "quiet"

    def bare(self, kind: str, since_ms: Optional[int]) -> Dict:
        """A published bare feature; without `since_ms`, one from before #533."""
        feature = {
            "kind": kind,
            "reading": None,
            "value": None,
            "unit": None,
            "percentile": None,
            "event_start_ms": None,
        }
        if since_ms is not None:
            feature.update(since_ms=since_ms, held=False, outside_usual=None)
        return feature

    def test_with_no_last_snapshot_the_subject_is_chosen_now(self, seed):
        station = seed()

        snapshot = run_refresh(station)

        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == snapshot.built_at_ms
        published = stored_json(station, DASHBOARD_KEY)["feature"]
        assert published["since_ms"] == snapshot.built_at_ms

    def test_a_subject_chosen_an_hour_ago_is_held(self, seed):
        station = seed()
        first = run_refresh(station)
        kept = self.other_subject(first)
        since = first.built_at_ms - 3_600_000
        self.publish_feature(station, self.bare(kept, since))

        snapshot = run_refresh(station)

        assert snapshot.feature.kind == kept
        assert snapshot.feature.held is True
        assert snapshot.feature.since_ms == since
        assert stored_json(station, DASHBOARD_KEY)["feature"]["held"] is True

    def test_a_snapshot_from_before_the_hold_holds_nothing(self, seed):
        # The first run after the deploy: the stored feature has no since_ms.
        station = seed()
        first = run_refresh(station)
        self.publish_feature(station, self.bare(self.other_subject(first), None))

        snapshot = run_refresh(station)

        assert snapshot.feature.kind == first.feature.kind
        assert snapshot.feature.held is False
        assert snapshot.feature.since_ms == snapshot.built_at_ms

    def test_a_held_wind_gust_from_before_the_change_holds_nothing(self, seed):
        # Wind gust can no longer win, so a stored one is not kept either.
        station = seed()
        first = run_refresh(station)
        gust = {
            **self.bare("unusual", first.built_at_ms - 3_600_000),
            "reading": "wind_gust",
            "value": 30.0,
            "unit": "mph",
            "percentile": 100,
            "outside_usual": True,
        }
        self.publish_feature(station, gust)

        snapshot = run_refresh(station)

        assert snapshot.feature.kind == first.feature.kind
        assert snapshot.feature.reading != "wind_gust"
        assert snapshot.feature.held is False

    def test_a_failed_read_publishes_without_a_hold(
        self, seed, fake_healthchecks, monkeypatch, caplog
    ):
        station = seed()
        assert self.run_main(station) == 0
        flaky_storj(monkeypatch, station.s3, read=DASHBOARD_KEY)

        with caplog.at_level("WARNING"):
            assert self.run_main(station) == 0

        assert stored_json(station, DASHBOARD_KEY)["feature"]["held"] is False
        assert fake_healthchecks.pings[-1] == "lookout-refresh/0"
        assert any(DASHBOARD_KEY in r.getMessage() for r in caplog.records)

    def test_a_corrupt_last_snapshot_publishes_without_a_hold(
        self, seed, fake_healthchecks, caplog
    ):
        # ADR-0009's narrow exception to ADR-0004: a corrupt document must
        # not fail the station on every run, or it is never replaced.
        station = seed()
        assert self.run_main(station) == 0
        station.s3.put_object(Bucket=BUCKET, Key=DASHBOARD_KEY, Body=b"not json")

        with caplog.at_level("WARNING"):
            assert self.run_main(station) == 0

        feature = stored_json(station, DASHBOARD_KEY)["feature"]
        assert feature["held"] is False
        warnings = [
            r
            for r in caplog.records
            if r.levelname == "WARNING" and DASHBOARD_KEY in r.getMessage()
        ]
        assert warnings
        assert fake_healthchecks.pings[-1] == "lookout-refresh/0"

    def test_a_last_snapshot_that_is_not_an_object_publishes_without_a_hold(
        self, seed, fake_healthchecks
    ):
        # Valid JSON that is not an object is a found document the rule
        # cannot read: only the hold is lost, the station never fails on it.
        station = seed()
        assert self.run_main(station) == 0
        station.s3.put_object(Bucket=BUCKET, Key=DASHBOARD_KEY, Body=b"[1]")

        assert self.run_main(station) == 0

        assert stored_json(station, DASHBOARD_KEY)["feature"]["held"] is False
        assert fake_healthchecks.pings[-1] == "lookout-refresh/0"
