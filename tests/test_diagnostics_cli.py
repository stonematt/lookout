"""
The diagnostics CLI (#104): `lookout.cli.diagnostics`.

The station is the shared SCENARIO week (a carryover gap and a plain gap)
seeded in the moto bucket by `tests.test_catchup.catchup_station`, with the
fake Ambient API serving `/devices` and history. Nothing here touches a real
secret or the live Storj.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from lookout.api import awn_controller as awn
from lookout.cli import diagnostics
from lookout.core import rain_corrections
from lookout.core.archive_merge import MergeReport
from lookout.core.rain_events import RainEventCatalog
from lookout.storage import storj
from tests.fixtures.ambient import devices_body, to_ambient_records
from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    build_archive,
    scenario_archive,
)
from tests.fixtures.storage import BUCKET, FROZEN_DAY
from tests.fixtures.storage import flaky_storj
from tests.test_catchup import (  # noqa: F401 (catchup_station is a fixture)
    ARCHIVE_KEY,
    CatchupStation,
    catchup_station,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
NOW = "2026-01-11T09:00:00Z"
BACKUP_KEY = f".backups/{FAKE_MAC}/{FROZEN_DAY}/{ARCHIVE_KEY}"
CATALOG_KEY = f"{FAKE_MAC}.energy_catalog.parquet"
CATALOG_BACKUP_KEY = f".backups/{FAKE_MAC}.energy_catalog/{FROZEN_DAY}/{CATALOG_KEY}"
EVENT_KEY = f"{FAKE_MAC}.event_catalog.parquet"
CORRUPT_CATALOG = b"not a parquet file"
CARRY_GAP_LINE = "2026-01-09 06:55  2026-01-09 09:00"
PLAIN_GAP_LINE = "2026-01-10 17:55  2026-01-10 22:00"
RAIN_COUNTERS = ["dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin"]


@pytest.fixture
def seed(catchup_station) -> Callable[..., CatchupStation]:  # noqa: F811
    """
    Seed the SCENARIO week: archive in the bucket, `lastData` at its newest row.

    :return: `seed(history=())`: `history` is what Ambient serves for fills.
    """

    def _seed(history: Optional[pd.DataFrame] = None) -> CatchupStation:
        archive = scenario_archive()
        records = [] if history is None else to_ambient_records(history)
        last = to_ambient_records(archive.head(1))[0]
        return catchup_station(
            archive=archive, history=records, devices=devices_body(last), now=NOW
        )

    return _seed


def run(argv: List[str], capsys: pytest.CaptureFixture) -> Tuple[int, str]:
    """
    Run the CLI as the shell does.

    :param argv: Arguments after the program name; `--bucket` is added.
    :param capsys: pytest's capture.
    :return: The exit status and stdout.
    """
    capsys.readouterr()
    status = diagnostics.main([argv[0], "--bucket", BUCKET, *argv[1:]])
    return status, capsys.readouterr().out


class TestGaps:
    def test_lists_both_scenario_gaps_oldest_first(self, seed, capsys):
        seed()

        status, out = run(["gaps"], capsys)

        assert status == 0
        assert "Gaps over 30 minutes: 2" in out
        assert out.index(CARRY_GAP_LINE) < out.index(PLAIN_GAP_LINE)
        assert "125" in out  # carryover gap: 06:55 to 09:00
        assert "245" in out  # plain gap: 17:55 to 22:00

    def test_threshold_drops_the_shorter_gap(self, seed, capsys):
        seed()

        status, out = run(["gaps", "--threshold", "200"], capsys)

        assert status == 0
        assert "Gaps over 200 minutes: 1" in out
        assert PLAIN_GAP_LINE in out
        assert CARRY_GAP_LINE not in out

    def test_a_date_window_keeps_only_gaps_inside_it(self, seed, capsys):
        seed()

        status, out = run(["gaps", "--since", "2026-01-10"], capsys)

        assert status == 0
        assert "Gaps over 30 minutes: 1" in out
        assert PLAIN_GAP_LINE in out


class TestFillGap:
    def test_dry_run_fetches_and_writes_nothing(self, seed, capsys):
        station = seed()
        keys = station.keys()

        status, out = run(["fill-gap", "--index", "1", "--dry-run"], capsys)

        assert status == 0
        assert "Dry run" in out
        assert "2026-01-10 17:55" in out
        assert station.keys() == keys
        # Only the device list was asked for, never history.
        assert [path for path, _ in station.replay.calls] == ["/devices"]
        assert len(station.stored()) == len(station.archive)

    def test_fills_the_gap_after_backing_up_the_archive(self, seed, capsys):
        truth = build_archive(
            SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=SCENARIO.gaps[:1]
        )
        station = seed(history=truth)
        before = len(station.archive)

        status, out = run(["fill-gap", "--index", "1"], capsys)

        assert status == 0
        assert f"Merged {len(truth) - before} added, 0 filled, 0 updated" in out
        assert len(station.stored()) == len(truth)
        assert BACKUP_KEY in station.keys()
        backed_up = storj.get_df_from_s3(BUCKET, BACKUP_KEY, file_type="parquet")
        assert len(backed_up) == before

    def test_a_fill_saves_the_rain_counters_raw(self, seed, capsys):
        """
        With a corrections catalog stored, a fill and save leaves every
        stored rain counter raw (#95), archived rows and fetched rows alike.
        """
        truth = build_archive(
            SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=SCENARIO.gaps[:1]
        )
        station = seed(history=truth)
        rain_corrections.save_catalog(
            {"corrections": rain_corrections.detect_carryovers(truth)}, FAKE_MAC
        )

        status, _ = run(["fill-gap", "--index", "1"], capsys)

        assert status == 0
        merged = station.stored().merge(truth, on="dateutc", suffixes=("", "_raw"))
        assert len(merged) == len(truth)
        for col in RAIN_COUNTERS:
            assert (merged[col] - merged[col + "_raw"]).abs().max() < 1e-9

    @pytest.mark.parametrize(
        "fail", [{"write": ARCHIVE_KEY}, {"copy": True}], ids=["save", "backup"]
    )
    def test_a_failed_save_or_backup_fails_and_claims_nothing(
        self, seed, capsys, monkeypatch, fail
    ):
        """#160: no "saved" line and a non-zero exit when storage refuses."""
        truth = build_archive(
            SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=SCENARIO.gaps[:1]
        )
        station = seed(history=truth)
        flaky_storj(monkeypatch, station.s3, **fail)

        status, out = run(["fill-gap", "--index", "1"], capsys)

        assert status == 1
        assert "Merged" not in out
        assert "fill-gap failed" in out
        assert len(station.stored()) == len(station.archive)

    def test_a_fill_that_would_drop_a_stored_reading_is_refused(
        self, seed, capsys, monkeypatch
    ):
        """#190: the fill-gap save is guarded by the archive it loaded."""
        station = seed()
        before = station.stored()

        def lossy(device, archive, start, end):
            return awn.GapFill(
                archive.iloc[1:], start, end, "filled", MergeReport(updated=1)
            )

        monkeypatch.setattr(diagnostics.awn, "fill_archive_gap", lossy)

        status, out = run(["fill-gap", "--index", "1"], capsys)

        assert status == 1
        assert "Merged" not in out
        assert "fill-gap failed" in out
        assert_frame_equal(station.stored(), before)

    def test_a_failed_archive_read_fails_the_fill_and_saves_nothing(
        self, seed, capsys, monkeypatch
    ):
        station = seed()
        keys = station.keys()
        flaky_storj(monkeypatch, station.s3, read=ARCHIVE_KEY)

        status, out = run(["fill-gap", "--index", "1"], capsys)

        assert status == 1
        assert "fill-gap failed" in out
        assert station.keys() == keys

    def test_a_gap_ambient_cannot_fill_fails_and_saves_nothing(self, seed, capsys):
        station = seed()

        status, out = run(
            [
                "fill-gap",
                "--start",
                "2026-01-10T17:55:00Z",
                "--end",
                "2026-01-10T22:00:00Z",
            ],
            capsys,
        )

        assert status == 1
        assert "no new data" in out
        assert len(station.stored()) == len(station.archive)
        assert BACKUP_KEY not in station.keys()

    def test_a_fill_that_only_updates_values_saves_and_prints_the_counts(
        self, seed, capsys
    ):
        changed = scenario_archive().copy()
        changed["tempf"] = changed["tempf"] + 1.0
        station = seed(history=changed)
        before = len(station.archive)

        status, out = run(["fill-gap", "--index", "1"], capsys)

        assert status == 0
        assert "Merged 0 added, 0 filled," in out
        assert " 0 updated" not in out
        assert BACKUP_KEY in station.keys()
        assert len(station.stored()) == before

    def test_an_index_outside_the_window_is_refused(self, seed, capsys):
        seed()

        status, out = run(["fill-gap", "--index", "9", "--dry-run"], capsys)

        assert status == 1
        assert "No gap 9" in out

    def test_no_gap_named_is_refused(self, seed, capsys):
        seed()

        status, out = run(["fill-gap", "--dry-run"], capsys)

        assert status == 1
        assert "--index" in out


class TestEnergyCatalog:
    def test_reports_without_saving_by_default(self, seed, capsys):
        station = seed()

        status, out = run(["energy-catalog"], capsys)

        assert status == 0
        assert "Regenerated:" in out
        assert "Not saved" in out
        assert CATALOG_KEY not in station.keys()

    def test_save_dry_run_writes_nothing(self, seed, capsys):
        station = seed()

        status, out = run(["energy-catalog", "--save", "--dry-run"], capsys)

        assert status == 0
        assert "Dry run" in out
        assert CATALOG_KEY not in station.keys()

    def test_dry_run_without_save_is_refused(self, seed, capsys):
        station = seed()

        status, out = run(["energy-catalog", "--dry-run"], capsys)

        assert status == 1
        assert "--dry-run only applies with --save" in out
        assert "Regenerated:" not in out
        assert CATALOG_KEY not in station.keys()

    def test_save_backs_up_the_stored_catalog_then_replaces_it(self, seed, capsys):
        station = seed()
        stored = pd.DataFrame(
            {
                "period_start": pd.to_datetime(["2025-12-01 10:00"]),
                "period_end": pd.to_datetime(["2025-12-01 10:15"]),
                "energy_kwh": [0.01],
            }
        )
        storj.save_df_to_s3(stored, BUCKET, CATALOG_KEY)

        status, out = run(["energy-catalog", "--save"], capsys)

        assert status == 0
        assert "backed up and replaced" in out
        backup = storj.get_df_from_s3(BUCKET, CATALOG_BACKUP_KEY, file_type="parquet")
        assert len(backup) == 1
        replaced = station.stored(CATALOG_KEY)
        assert len(replaced) > 1
        assert replaced["energy_kwh"].sum() > 0.01

    @pytest.mark.parametrize("corrupt", [False, True], ids=["read-fails", "corrupt"])
    def test_a_failed_read_reports_an_error_and_saves_nothing(
        self, seed, capsys, monkeypatch, corrupt
    ):
        station = seed()
        body = b"not parquet" if corrupt else b"stored"
        station.s3.put_object(Bucket=BUCKET, Key=CATALOG_KEY, Body=body)
        if not corrupt:
            flaky_storj(monkeypatch, station.s3, read=CATALOG_KEY)

        status, out = run(["energy-catalog", "--save"], capsys)

        assert status == 1
        assert "energy-catalog failed" in out
        assert "backed up and replaced" not in out
        stored = station.s3.get_object(Bucket=BUCKET, Key=CATALOG_KEY)["Body"].read()
        assert stored == body
        assert [k for k in station.keys() if "energy_catalog" in k] == [CATALOG_KEY]

    def test_with_none_stored_save_writes_the_regenerated_catalog(self, seed, capsys):
        station = seed()

        status, out = run(["energy-catalog", "--save"], capsys)

        assert status == 0
        assert "Stored catalog: none" in out
        assert len(station.stored(CATALOG_KEY)) > 1


class TestRainCatalog:
    def backups(self, station) -> List[str]:
        return [k for k in station.keys() if k.startswith("backups/")]

    def store_catalog(self, station) -> pd.DataFrame:
        """Store a one-event catalog that matches no detected event."""
        stored = pd.DataFrame(
            {
                "event_id": ["old"],
                "start_time": pd.to_datetime(["2025-12-01 10:00"], utc=True),
                "end_time": pd.to_datetime(["2025-12-01 11:00"], utc=True),
            }
        )
        storj.save_df_to_s3(stored, BUCKET, EVENT_KEY)
        return stored

    def test_dry_run_reports_a_found_catalog_and_writes_nothing(self, seed, capsys):
        station = seed()
        self.store_catalog(station)
        keys = station.keys()

        status, out = run(["rain-catalog"], capsys)

        assert status == 0
        assert "Stored catalog: found, 1 events" in out
        assert "Detected: 3 events" in out
        assert "Added: 3" in out
        assert "Dropped: 1" in out
        assert "Not saved" in out
        assert station.keys() == keys

    def test_dry_run_reports_a_missing_catalog(self, seed, capsys):
        station = seed()

        status, out = run(["rain-catalog"], capsys)

        assert status == 0
        assert "Stored catalog: missing" in out
        assert EVENT_KEY not in station.keys()

    def test_dry_run_reports_a_corrupt_catalog_and_writes_nothing(self, seed, capsys):
        station = seed()
        storj.get_s3_client().put_object(
            Bucket=BUCKET, Key=EVENT_KEY, Body=CORRUPT_CATALOG
        )
        keys = station.keys()

        status, out = run(["rain-catalog"], capsys)

        assert status == 0
        assert "Stored catalog: failed" in out
        assert station.keys() == keys

    def test_nothing_detected_exits_1_and_writes_nothing(
        self, seed, capsys, monkeypatch
    ):
        station = seed()
        self.store_catalog(station)
        keys = station.keys()
        monkeypatch.setattr(
            RainEventCatalog,
            "detect_and_catalog_events",
            lambda self, archive, auto_save=False: pd.DataFrame(),
        )

        status, out = run(["rain-catalog", "--save"], capsys)

        assert status == 1
        assert "No rain events could be detected" in out
        assert "Detected:" not in out
        assert station.keys() == keys

    def test_save_backs_up_a_found_catalog_then_writes(self, seed, capsys):
        station = seed()
        self.store_catalog(station)

        status, out = run(["rain-catalog", "--save"], capsys)

        assert status == 0
        assert "backed up to backups/" in out
        (backup,) = self.backups(station)
        assert len(storj.get_df_from_s3(BUCKET, backup, file_type="parquet")) == 1
        assert len(station.stored(EVENT_KEY)) == 3

    def test_save_with_no_catalog_writes_with_no_backup(self, seed, capsys):
        station = seed()

        status, out = run(["rain-catalog", "--save"], capsys)

        assert status == 0
        assert "no backup" in out
        assert self.backups(station) == []
        assert len(station.stored(EVENT_KEY)) == 3

    def test_save_backs_up_a_corrupt_catalog_raw_then_writes(self, seed, capsys):
        station = seed()
        storj.get_s3_client().put_object(
            Bucket=BUCKET, Key=EVENT_KEY, Body=CORRUPT_CATALOG
        )

        status, out = run(["rain-catalog", "--save"], capsys)

        assert status == 0
        (backup,) = self.backups(station)
        raw = storj.get_s3_client().get_object(Bucket=BUCKET, Key=backup)
        assert raw["Body"].read() == CORRUPT_CATALOG
        assert len(station.stored(EVENT_KEY)) == 3

    def test_save_with_a_failing_backup_writes_nothing(self, seed, capsys, monkeypatch):
        station = seed()
        self.store_catalog(station)
        stored = storj.read_object(BUCKET, EVENT_KEY)

        def refuse(*args, **kwargs):
            raise storj.BackupError("copy refused")

        monkeypatch.setattr(storj, "backup_object", refuse)

        status, out = run(["rain-catalog", "--save"], capsys)

        assert status == 1
        assert "unchanged" in out
        assert storj.read_object(BUCKET, EVENT_KEY) == stored


class TestReadOnlyCommands:
    def test_status_reports_the_lag_to_the_device(self, seed, capsys):
        station = seed()
        newest = int(station.archive["dateutc"].max())
        late = to_ambient_records(station.archive.head(1))[0]
        late["dateutc"] = newest + 3 * 3600 * 1000
        station.replay.devices = devices_body(late)

        status, out = run(["status"], capsys)

        assert status == 0
        assert "Archive lag: 3.0 hours" in out
        assert "2026-01-11 07:55" in out  # archive newest, UTC

    def test_last_record_prints_the_live_reading(self, seed, capsys):
        seed()

        status, out = run(["last-record"], capsys)

        assert status == 0
        assert FAKE_MAC in out
        assert "tempf:" in out
        assert "reading time: 2026-01-11 07:55 UTC" in out

    def test_coverage_counts_samples_per_hour(self, seed, capsys):
        seed()

        status, out = run(["coverage", "--days", "7"], capsys)

        assert status == 0
        lines = out.splitlines()
        # Sunday 2026-01-04 local: 24 hours of 12 samples, 288 in all.
        sunday = next(line for line in lines if line.startswith("2026-01-04"))
        assert sunday.split()[1:25] == ["12"] * 24
        assert sunday.split()[-1] == "288"
        # Friday's gap leaves local hours 0 and 23 short, and the day total too.
        friday = next(line for line in lines if line.startswith("2026-01-09"))
        assert int(friday.split()[-1]) < 288


def test_importing_the_diagnostics_cli_loads_no_streamlit():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    code = (
        "import sys, lookout.cli.diagnostics\n"
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
