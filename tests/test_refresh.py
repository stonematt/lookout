"""
One refresh run end to end (spec #66, seam 3): `lookout.cli.refresh`.

Extends the catchup end-to-end test: the station is seeded with
`tests.test_catchup.catchup_station` (archive in the moto bucket, recorded
Ambient pages, a fixed clock), then `refresh.run` runs instead of catchup.
Later links add to this module: catalog updates and the Event Catalog
auto-save (#73), the Healthchecks heartbeats (#74).
"""

import importlib.abc
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import pytest

from lookout.api import awn_controller as awn
from lookout.cli import refresh
from lookout.core.energy_catalog import EnergyCatalog
from lookout.core.rain_events import RainEventCatalog, is_event_ongoing
from lookout.snapshot import Snapshot, curves
from lookout.storage import storj
from tests.fixtures.ambient import to_ambient_records
from tests.fixtures.archive import FAKE_MAC, SCENARIO, RainSpell, build_archive
from tests.fixtures.real import AMBIENT_END
from tests.fixtures.storage import BUCKET
from tests.test_catchup import (  # noqa: F401 (catchup_station is a fixture)
    ARCHIVE_KEY,
    RECORDED_ROWS,
    CatchupStation,
    catchup_station,
    ms,
    reporting,
)

DASHBOARD_KEY = f"snapshot/{FAKE_MAC}.dashboard.json"
CONTEXT_KEY = f"snapshot/{FAKE_MAC}.context.json"
CURVE_PREFIX = f"snapshot/{FAKE_MAC}.events/"
REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_IS_DEV_MODE = awn.is_dev_mode


def run_refresh(station: CatchupStation) -> Snapshot:
    """
    One refresh run, as the cron does it, at the station's frozen now.

    :param station: The seeded station.
    :return: The published snapshot.
    """
    return refresh.run(bucket=BUCKET, pages=20, clock=lambda: station.now)


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
    """The snapshot documents' keys: under `snapshot/`, not curve files."""
    return [
        key
        for key in station.keys()
        if key.startswith("snapshot/") and not key.startswith(CURVE_PREFIX)
    ]


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

    def test_the_archive_is_appended_first_with_the_schema_unchanged(self, seed):
        station = seed()

        run_refresh(station)

        stored = station.stored()
        assert len(stored) == len(station.archive) + RECORDED_ROWS
        assert stored["dateutc"].is_unique
        assert list(stored.columns) == list(station.archive.columns)
        assert stored.dtypes.astype(str).to_dict() == (
            station.archive.dtypes.astype(str).to_dict()
        )

    def test_the_snapshot_is_built_from_the_caught_up_archive(self, seed):
        station = seed()

        run_refresh(station)

        dashboard = stored_json(station, DASHBOARD_KEY)

        # The recorded pages reach the live reading (07:55Z), so they meet.
        assert dashboard["archive_newest_ms"] == ms("2025-11-19T07:55:00Z")
        assert dashboard["observed_at_ms"] == ms("2025-11-19T07:55:00Z")
        assert dashboard["built_at_ms"] == ms(AMBIENT_END)
        assert dashboard["station"] == {"mac": FAKE_MAC, "name": "Fixture Station"}
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
        monkeypatch.setenv("STATION_MAC", "AA:BB:CC:DD:EE:FF")  # not on the account

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
        monkeypatch.setenv("STATION_MAC", "AA:BB:CC:DD:EE:FF")  # not on the account

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

    def test_a_curve_write_failure_does_not_fail_a_published_run(
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

        assert self.run_main(station) == 0

        assert sorted(snapshot_keys(station)) == sorted([DASHBOARD_KEY, CONTEXT_KEY])
        assert curve_keys(station) == []
        assert "curve write refused" in caplog.text
        assert fake_healthchecks.pings[0] == "lookout-refresh/start"
        assert fake_healthchecks.pings[1].startswith("station-")
        assert fake_healthchecks.pings[-1] == "lookout-refresh/0"

    def test_a_curve_write_failure_still_follows_a_stale_reading(
        self, seed, fake_healthchecks, monkeypatch
    ):
        station = TestCatalogUpdatesAndSaves().storm(seed, "2026-01-10T00:00:00Z")

        def refuse(*args, **kwargs):
            raise OSError("curve write refused")

        monkeypatch.setattr(refresh, "publish_event_curves", refuse)
        late = station.now + timedelta(hours=2)

        assert refresh.main(["--bucket", BUCKET], clock=lambda: late) == 0
        assert fake_healthchecks.pings == ["lookout-refresh/start", "lookout-refresh/0"]

    def test_a_healthchecks_outage_does_not_fail_the_run(self, seed, fake_healthchecks):
        station = seed()
        fake_healthchecks.fail = True

        assert self.run_main(station) == 0
        assert len(snapshot_keys(station)) == 2


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

    STATION = refresh.Station(mac=FAKE_MAC, name=None)

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
