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
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

from lookout.api import awn_controller as awn
from lookout.cli import refresh
from lookout.core.rain_events import RainEventCatalog
from lookout.snapshot import Snapshot
from tests.fixtures.ambient import to_ambient_records
from tests.fixtures.archive import FAKE_MAC, SCENARIO, build_archive
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
    """Keys under the `snapshot/` prefix."""
    return [key for key in station.keys() if key.startswith("snapshot/")]


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
        stored = RainEventCatalog(FAKE_MAC).load_catalog()
        ongoing = stored.sort_values("start_time").iloc[-1]

        event = run_refresh(raining).active_rain_event

        assert event.event_id == ongoing["event_id"]
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
