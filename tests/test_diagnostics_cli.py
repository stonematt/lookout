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

from lookout.cli import diagnostics
from lookout.core import rain_corrections
from lookout.storage import storj
from tests.fixtures.ambient import devices_body, to_ambient_records
from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    build_archive,
    scenario_archive,
)
from tests.fixtures.storage import BUCKET, FROZEN_DAY
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
        assert f"Added {len(truth) - before} rows" in out
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
