"""
The memory harness (#281): `lookout.cli.memory_harness`.

One smoke test on the frozen real slice, through the command's public entry
points: every refresh step gets a peak RSS and a tracemalloc peak, and the
run reaches neither Storj nor Ambient. No memory figure is asserted (#234):
the numbers differ by machine, so the test checks only that each is there.
Beside it, the out-of-range `--new` refusal and the MB unit the memory utils
report in.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lookout.cli import memory_harness
from lookout.utils.memory_utils import get_df_memory_usage
from tests.fixtures.archive import STATION_TZ
from tests.fixtures.real import load_real_slice

REFRESH_STEPS = [
    "load_station",
    "update_archive",
    "correct_rain",
    "load_catalogs",
    "update_catalogs",
    "build",
    "publish",
    "publish_curves",
]


def fixture_archive(tmp_path: Path) -> Path:
    """
    Write the real slice as a local archive parquet.

    :param tmp_path: The test's temp dir.
    :return: The parquet's path.
    """
    path = tmp_path / "archive.parquet"
    load_real_slice().to_parquet(path, engine="pyarrow")
    return path


def test_each_refresh_step_reports_its_peaks_and_nothing_leaves(tmp_path):
    archive = fixture_archive(tmp_path)
    before = archive.read_bytes()

    report = memory_harness.measure(archive, tz=STATION_TZ, new=12)

    assert [step.step for step in report.steps] == REFRESH_STEPS
    assert report.baseline_rss_mb > 0
    for step in report.steps:
        assert step.rss_peak_mb > 0, step
        assert step.tracemalloc_peak_mb > 0, step
    # The run's writes all landed in the fake bucket, and nothing tried
    # the network: Storj and Ambient were never reached.
    assert report.connections_attempted == 0
    assert f"{memory_harness.HARNESS_MAC}.parquet" in report.fake_writes
    assert f"snapshot/{memory_harness.HARNESS_MAC}.dashboard.json" in (
        report.fake_writes
    )
    assert archive.read_bytes() == before


def test_the_fetch_catches_up_in_one_page_however_old_the_archive(tmp_path):
    # The real slice is long past; the fetch's now is the harness's, not the
    # wall clock's, so one page reaches it, as a cron run's does (#292).
    archive = fixture_archive(tmp_path)

    report = memory_harness.measure(archive, tz=STATION_TZ, new=12, trace=False)

    assert report.history_requests == 1


def test_an_archive_cannot_hold_back_all_its_readings(tmp_path):
    archive = fixture_archive(tmp_path)

    with pytest.raises(ValueError, match="--new"):
        memory_harness.measure(archive, tz=STATION_TZ, new=len(load_real_slice()))


def test_dataframe_sizes_are_in_megabytes():
    # 131,072 float64 values are exactly 1 MB; the index adds a few bytes.
    frame = pd.DataFrame({"value": np.zeros(131_072)})

    assert get_df_memory_usage(frame) == pytest.approx(1.0, abs=0.001)


def test_the_command_prints_a_line_per_step_and_writes_json(tmp_path, capsys):
    archive = fixture_archive(tmp_path)
    out = tmp_path / "report.json"

    status = memory_harness.main([str(archive), "--tz", STATION_TZ, "--json", str(out)])

    assert status == 0
    printed = capsys.readouterr().out
    for name in REFRESH_STEPS:
        assert name in printed
    steps = json.loads(out.read_text())["steps"]
    assert [step["step"] for step in steps] == REFRESH_STEPS
