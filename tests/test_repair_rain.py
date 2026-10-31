"""
The rain repair (#95): `lookout.cli.repair_rain`.

The stored archive holds counters damaged by repeated corrections inside a
window; the fake Ambient API serves the raw truth. The repair drops the
window's rows, refills them through gap fill, cross-checks them against the
protected raw backup, and only with `--write` backs up and saves. Ambient and
storage are faked (`AmbientReplay`, moto); nothing touches the live Storj.
"""

from typing import Callable, List, Optional, Tuple

import pandas as pd
import pytest

from lookout.api import awn_controller as awn
from lookout.cli import repair_rain
from lookout.storage import storj
from lookout.utils.dateutc import from_local_date
from tests.fixtures.ambient import devices_body, to_ambient_records
from tests.fixtures.archive import FAKE_MAC, SCENARIO, Gap, build_archive
from tests.fixtures.storage import BUCKET, FROZEN_DAY
from tests.test_catchup import (  # noqa: F401 (catchup_station is a fixture)
    ARCHIVE_KEY,
    CatchupStation,
    catchup_station,
)

BACKUP_KEY = f".backups/{FAKE_MAC}/{FROZEN_DAY}/{ARCHIVE_KEY}"
REFERENCE_KEY = f"backups/{FAKE_MAC}.raw-pre-95_20251222.parquet"
COUNTERS = ["dailyrainin", "weeklyrainin", "monthlyrainin", "yearlyrainin"]
NOW = "2026-01-11T09:00:00Z"
# The window under repair: two local days in the middle of the week.
FIRST, LAST = "2026-01-08", "2026-01-09"
START_MS = from_local_date(FIRST)
END_MS = from_local_date("2026-01-10")
DAMAGE = 0.5


def damaged(truth: pd.DataFrame) -> pd.DataFrame:
    """
    The archive as storage holds it: every yearly counter from the window's
    first day on is `DAMAGE` low, inside the window and after it.
    """
    out = truth.copy()
    out.loc[out["dateutc"] >= START_MS, "yearlyrainin"] -= DAMAGE
    return out


@pytest.fixture
def truth() -> pd.DataFrame:
    """SCENARIO's week as the station reported it, raw."""
    return build_archive(
        SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=SCENARIO.gaps
    )


@pytest.fixture
def seed(catchup_station, monkeypatch) -> Callable[..., CatchupStation]:  # noqa: F811
    """
    Seed the damaged archive, the raw reference, and what Ambient serves.

    :return: `seed(truth, served=None, reference=None)`.
    """
    monkeypatch.setattr(awn.time, "sleep", lambda _s: None)

    def _seed(
        truth: pd.DataFrame,
        served: Optional[pd.DataFrame] = None,
        reference: Optional[pd.DataFrame] = None,
    ) -> CatchupStation:
        served = truth if served is None else served
        station = catchup_station(
            archive=damaged(truth),
            history=to_ambient_records(served),
            devices=devices_body(to_ambient_records(truth.head(1))[0]),
            now=NOW,
        )
        storj.save_df_to_s3(
            truth if reference is None else reference, BUCKET, REFERENCE_KEY
        )
        return station

    return _seed


def run(argv: List[str], capsys: pytest.CaptureFixture) -> Tuple[int, str]:
    """
    Run the repair as the shell does, on the test window.

    :param argv: Extra arguments.
    :return: The exit status and stdout.
    """
    capsys.readouterr()
    status = repair_rain.main(["--bucket", BUCKET, "--window", FIRST, LAST, *argv])
    return status, capsys.readouterr().out


def in_window(df: pd.DataFrame) -> pd.Series:
    return (df["dateutc"] >= START_MS) & (df["dateutc"] < END_MS)


def test_the_default_windows_are_the_damaged_ones():
    args = repair_rain.parse_args([])

    assert args.window == [
        ("2024-12-16", "2024-12-31"),
        ("2025-10-14", "2025-12-31"),
    ]
    assert args.write is False


def test_dry_run_prints_the_per_day_diff_and_writes_nothing(seed, truth, capsys):
    station = seed(truth)
    keys = station.keys()

    status, out = run([], capsys)

    assert status == 0
    assert "Dry run" in out
    for day in (FIRST, LAST):
        line = next(line for line in out.splitlines() if line.startswith(day))
        that_day = truth[truth["date"].dt.strftime("%Y-%m-%d") == day]
        assert f"rows {len(that_day)} -> {len(that_day)}" in line
        year_end = that_day["yearlyrainin"].iloc[0]  # its last reading
        assert f"yearly {year_end - DAMAGE:.3f} -> {year_end:.3f}" in line
    assert "Cross-check" in out and "0 over tolerance" in out
    assert station.keys() == keys
    pd.testing.assert_frame_equal(
        station.stored().reset_index(drop=True),
        station.archive.reset_index(drop=True),
        check_dtype=False,
    )


def test_write_backs_up_then_replaces_only_the_window(seed, truth, capsys):
    station = seed(truth)

    status, out = run(["--write"], capsys)

    assert status == 0
    assert BACKUP_KEY in station.keys()
    backed_up = storj.get_df_from_s3(BUCKET, BACKUP_KEY, file_type="parquet")
    assert len(backed_up) == len(station.archive)
    stored = station.stored()
    assert len(stored) == len(truth)
    assert stored["dateutc"].is_unique
    raw = truth.set_index("dateutc")
    old = station.archive.set_index("dateutc")
    got = stored.set_index("dateutc")
    inside = got[in_window(stored).to_numpy()]
    outside = got[~in_window(stored).to_numpy()]
    for col in COUNTERS:
        # Inside the window: raw again. Outside: as stored, damage and all.
        assert (inside[col] - raw.loc[inside.index, col]).abs().max() < 1e-9
        assert (outside[col] - old.loc[outside.index, col]).abs().max() < 1e-9


def test_a_day_that_refills_short_aborts_the_write(seed, truth, capsys):
    short = Gap("2026-01-08T18:00:00Z", "2026-01-08T21:00:00Z")
    served = build_archive(
        SCENARIO.start, SCENARIO.end, rain=SCENARIO.rain, gaps=(*SCENARIO.gaps, short)
    )
    station = seed(truth, served=served)
    keys = station.keys()

    status, out = run(["--write"], capsys)

    assert status == 1
    assert "2026-01-08" in out and "short" in out
    assert "Nothing written" in out
    assert station.keys() == keys
    assert len(station.stored()) == len(station.archive)


def test_a_cross_check_mismatch_aborts_the_write(seed, truth, capsys):
    reference = truth.copy()
    reference.loc[in_window(reference), "weeklyrainin"] += 0.1
    station = seed(truth, reference=reference)
    keys = station.keys()

    status, out = run(["--write"], capsys)

    assert status == 1
    assert "over tolerance" in out
    assert "Nothing written" in out
    assert station.keys() == keys


def test_a_reference_that_covers_no_refilled_row_aborts_the_write(seed, truth, capsys):
    """The cross-check must check something; zero overlap is not a pass."""
    reference = truth[~in_window(truth)]
    station = seed(truth, reference=reference)
    keys = station.keys()

    status, out = run(["--write"], capsys)

    assert status == 1
    assert "could be checked" in out
    assert "Nothing written" in out
    assert station.keys() == keys
