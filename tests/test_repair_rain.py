"""
The rain repair (#95): `lookout.cli.repair_rain`.

The stored archive holds counters damaged by repeated corrections inside a
window. The protected raw backup holds the raw truth, at 3 decimals, up to
its end; the fake Ambient API serves it at 2 decimals, as Ambient does now.
The repair restores a window's rain counters from the backup where the backup
covers the row, refills the rest through gap fill, cross-checks the refilled
rows, and only with `--write` backs up and saves. Ambient and storage are
faked (`AmbientReplay`, moto); nothing touches the live Storj.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from lookout.api import awn_controller as awn
from lookout.cli import repair_rain
from lookout.storage import storj
from lookout.utils.archive_lock import hold_archive_lock
from lookout.utils.dateutc import from_local_date, normalize
from tests.fixtures.ambient import devices_body, to_ambient_records
from tests.fixtures.archive import (
    FAKE_MAC,
    SCENARIO,
    RainSpell,
    build_archive,
)
from tests.fixtures.storage import BUCKET, FROZEN_DAY, flaky_storj
from tests.test_diagnostics_cli import with_second_station
from tests.test_refresh import SECOND_MAC
from tests.test_catchup import (  # noqa: F401 (catchup_station is a fixture)
    ARCHIVE_KEY,
    CatchupStation,
    catchup_station,
)

BACKUP_KEY = f".backups/{FAKE_MAC}/{FROZEN_DAY}/{ARCHIVE_KEY}"
REFERENCE_KEY = f"backups/{FAKE_MAC}.raw-pre-95_20251222.parquet"
COUNTERS = list(repair_rain.RAIN_COLUMNS)
NOW = "2026-01-11T09:00:00Z"
# The window under repair: two local days in the middle of the week.
FIRST, LAST = "2026-01-08", "2026-01-09"
START_MS = from_local_date(FIRST)
END_MS = from_local_date("2026-01-10")
# Where a shortened backup ends: Friday 12:00 local, inside the window.
CUT_MS = from_local_date("2026-01-09") + 12 * 3_600_000
FIVE_MIN = 300_000
DAMAGE = 0.5


def damaged(truth: pd.DataFrame) -> pd.DataFrame:
    """
    The archive as storage holds it: every yearly counter from the window's
    first day on is `DAMAGE` low, inside the window and after it.
    """
    out = truth.copy()
    out.loc[out["dateutc"] >= START_MS, "yearlyrainin"] -= DAMAGE
    return out


def two_decimals(df: pd.DataFrame) -> pd.DataFrame:
    """The rows as Ambient serves them now: rain counters to 2 decimals."""
    out = df.copy()
    out[COUNTERS] = repair_rain.round_half_up(out[COUNTERS])
    return out


@pytest.fixture
def truth() -> pd.DataFrame:
    """
    SCENARIO's week as the station reported it, raw. A drizzle on Thursday
    morning (0.005" a reading) gives the window's counters a third decimal,
    so the backup's values and Ambient's 2-decimal ones differ.
    """
    drizzle = RainSpell("2026-01-08T16:00:00Z", hours=2, inches_per_hour=0.06)
    return build_archive(
        SCENARIO.start,
        SCENARIO.end,
        rain=(*SCENARIO.rain, drizzle),
        gaps=SCENARIO.gaps,
    )


@pytest.fixture
def seed(catchup_station, monkeypatch) -> Callable[..., CatchupStation]:  # noqa: F811
    """
    Seed the archive, the raw reference, and what Ambient serves.

    :return: `seed(truth, served=None, reference=None, archive=None)`.
        Ambient serves `truth` at 2 decimals by default; the reference is
        `truth`; the archive is `damaged(truth)`.
    """
    monkeypatch.setattr(awn.time, "sleep", lambda _s: None)

    def _seed(
        truth: pd.DataFrame,
        served: Optional[pd.DataFrame] = None,
        reference: Optional[pd.DataFrame] = None,
        archive: Optional[pd.DataFrame] = None,
    ) -> CatchupStation:
        served = two_decimals(truth) if served is None else served
        station = catchup_station(
            archive=damaged(truth) if archive is None else archive,
            history=to_ambient_records(served),
            devices=devices_body(to_ambient_records(truth.head(1))[0]),
            now=NOW,
        )
        storj.save_df_to_s3(
            truth if reference is None else reference, BUCKET, REFERENCE_KEY
        )
        return station

    return _seed


def readback_lines(caplog: pytest.LogCaptureFixture) -> List[str]:
    """The read-back check lines the archive store logged."""
    return [m for m in caplog.messages if m.startswith("Read-back check")]


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


def by_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("dateutc").sort_index()


def assert_counters_equal(got: pd.DataFrame, want: pd.DataFrame) -> None:
    """Rain counters equal, row for row on `dateutc`, exactly."""
    want = want.loc[got.index]
    for col in COUNTERS:
        np.testing.assert_array_equal(got[col].to_numpy(), want[col].to_numpy())


def test_the_default_windows_are_the_damaged_ones():
    args = repair_rain.parse_args([])

    assert args.window == [
        ("2024-12-16", "2024-12-31"),
        ("2025-10-14", "2025-12-31"),
    ]
    assert args.write is False


def test_rain_columns_cover_every_counter_the_corrections_touch():
    assert set(repair_rain.RAIN_COLUMNS) >= {
        "dailyrainin",
        "weeklyrainin",
        "monthlyrainin",
        "yearlyrainin",
        "eventrainin",
        "hourlyrainin",
    }


def test_dry_run_restores_from_the_backup_and_writes_nothing(seed, truth, capsys):
    station = seed(truth)
    keys = station.keys()

    status, out = run([], capsys)

    assert status == 0
    assert "Dry run" in out
    rows = int(in_window(truth).sum())
    assert f"restored {rows} from the backup, refilled 0 from Ambient" in out
    for day in (FIRST, LAST):
        line = next(line for line in out.splitlines() if line.startswith(day))
        that_day = truth[truth["date"].dt.strftime("%Y-%m-%d") == day]
        assert f"rows {len(that_day)} -> {len(that_day)}" in line
        year_end = that_day["yearlyrainin"].iloc[0]  # its last reading
        assert f"yearly {year_end - DAMAGE:.3f} -> {year_end:.3f}" in line
    assert f"{rows} restored rows checked, 0 differ from the backup" in out
    assert f"Archive replaced {rows} readings with {rows}" in out
    # Nothing to refill: no history page is fetched, only the device list.
    assert [path for path, _ in station.replay.calls] == ["/devices"]
    assert station.keys() == keys
    pd.testing.assert_frame_equal(
        station.stored().reset_index(drop=True),
        station.archive.reset_index(drop=True),
        check_dtype=False,
    )


def test_write_restores_only_the_rain_columns_of_window_rows(
    seed, truth, capsys, frozen_today
):
    """Backup values, 3 decimals; every other column and row as stored."""
    reference = truth.copy()
    reference["tempf"] = 99.0  # the backup's other columns are not used
    station = seed(truth, reference=reference)

    status, out = run(["--write"], capsys)

    assert status == 0, out
    assert REPAIR_KEY in station.keys()
    backed_up = storj.get_df_from_s3(BUCKET, REPAIR_KEY, file_type="parquet")
    assert len(backed_up) == len(station.archive)
    stored = station.stored()
    assert len(stored) == len(station.archive)
    assert stored["dateutc"].is_unique
    got, old = by_time(stored), by_time(station.archive)
    inside = got[in_window(stored.sort_values("dateutc")).to_numpy()]
    outside = got.drop(inside.index)
    assert_counters_equal(inside, by_time(truth))
    assert_counters_equal(outside, old)  # damage and all
    assert (got["tempf"] == old.loc[got.index, "tempf"]).all()


def test_rows_past_the_backup_end_are_refilled_from_ambient(seed, truth, capsys):
    reference = truth[truth["dateutc"] < CUT_MS]
    station = seed(truth, reference=reference)

    status, out = run(["--write"], capsys)

    assert status == 0, out
    covered = in_window(truth) & (truth["dateutc"] < CUT_MS)
    past = in_window(truth) & (truth["dateutc"] >= CUT_MS)
    assert (
        f"restored {int(covered.sum())} from the backup, "
        f"refilled {int(past.sum())} from Ambient"
    ) in out
    assert f"0 compared, 0 mismatched (max diff 0.000), {int(past.sum())}" in out
    got = by_time(station.stored())
    assert_counters_equal(got.loc[truth.loc[covered, "dateutc"]], by_time(truth))
    assert_counters_equal(
        got.loc[truth.loc[past, "dateutc"]], by_time(two_decimals(truth))
    )
    after = truth[truth["dateutc"] >= END_MS]["dateutc"]
    assert_counters_equal(got.loc[after], by_time(station.archive))


def test_a_covered_row_missing_from_the_backup_is_reported_and_refilled(
    seed, truth, capsys
):
    lost = START_MS + 6 * 3_600_000
    station = seed(truth, reference=truth[truth["dateutc"] != lost])

    status, out = run(["--write"], capsys)

    assert status == 0, out
    assert "1 row in the backup's range but not in it" in out
    assert pd.to_datetime(lost, unit="ms", utc=True).isoformat() in out
    got = by_time(station.stored())
    assert_counters_equal(got.loc[[lost]], by_time(two_decimals(truth)))
    others = truth.loc[in_window(truth) & (truth["dateutc"] != lost), "dateutc"]
    assert_counters_equal(got.loc[others], by_time(truth))


def _hole_and_two_strays(truth: pd.DataFrame, served_bump: float):
    """
    An archive missing row X, and a backup missing rows Y1 and Y2 on either
    side of it. The repair refills Y1..Y2 from Ambient, which brings X back
    too, and X is the one refilled row the backup can check.
    """
    y1 = START_MS + 6 * 3_600_000
    x, y2 = y1 + FIVE_MIN, y1 + 2 * FIVE_MIN
    archive = damaged(truth)
    archive = archive[archive["dateutc"] != x]
    reference = truth[~truth["dateutc"].isin([y1, y2])]
    served = two_decimals(truth)
    served.loc[served["dateutc"] == x, "weeklyrainin"] += served_bump
    return archive, reference, served


def test_a_refilled_row_the_backup_holds_is_checked_at_two_decimals(
    seed, truth, capsys
):
    archive, reference, served = _hole_and_two_strays(truth, 0.0)
    seed(truth, served=served, reference=reference, archive=archive)

    status, out = run([], capsys)

    assert status == 0, out
    assert "refilled 3 from Ambient" in out
    assert "1 compared, 0 mismatched" in out


def test_a_cross_check_mismatch_aborts_the_write(seed, truth, capsys):
    archive, reference, served = _hole_and_two_strays(truth, 0.1)
    station = seed(truth, served=served, reference=reference, archive=archive)
    keys = station.keys()

    status, out = run(["--write"], capsys)

    assert status == 1
    assert "1 compared, 1 mismatched" in out
    assert "Nothing written" in out
    assert station.keys() == keys


def test_a_day_that_refills_short_aborts_the_write(seed, truth, capsys):
    served = truth[
        (truth["dateutc"] < from_local_date("2026-01-09") + 14 * 3_600_000)
        | (truth["dateutc"] >= from_local_date("2026-01-09") + 17 * 3_600_000)
    ]
    station = seed(
        truth,
        served=two_decimals(served),
        reference=truth[truth["dateutc"] < CUT_MS],
    )
    keys = station.keys()

    status, out = run(["--write"], capsys)

    assert status == 1
    assert "2026-01-09" in out and "short" in out
    assert "Nothing written" in out
    assert station.keys() == keys
    assert len(station.stored()) == len(station.archive)


def test_the_rain_repair_may_drop_and_blank_as_an_explicit_repair(
    seed, truth, capsys, monkeypatch, frozen_today, caplog
):
    """#190: the rain repair passes the repair allowance at its save."""
    station = seed(truth)
    before = station.stored()
    repair_window = repair_rain.repair_window

    def lossy(*args, **kwargs):
        repaired, repair = repair_window(*args, **kwargs)
        # Inside the window only: the read-back flags a loss outside it.
        inside = in_window(repaired)
        lost = repaired[inside].iloc[1:].assign(tempf=float("nan"))
        return pd.concat([repaired[~inside], lost], ignore_index=True), repair

    monkeypatch.setattr(repair_rain, "repair_window", lossy)
    monkeypatch.setattr(repair_rain, "report_window", lambda *_a, **_k: [])

    with caplog.at_level("INFO"):
        status, _out = run(["--write"], capsys)

    # Saved; the read-back then rightly flags the dropped restored row, and
    # nothing else.
    assert status == 1
    failed = [line for line in readback_lines(caplog) if "FAILED" in line]
    assert len(failed) == 1
    assert "FAILED: restored rows equal the raw reference" in failed[0]
    stored = station.stored()
    assert len(stored) == len(before) - 1
    assert stored["tempf"].isna().sum() == in_window(stored).sum()


def test_a_failed_archive_read_stops_the_repair_before_any_save(
    seed, truth, capsys, monkeypatch
):
    station = seed(truth)
    keys = station.keys()
    flaky_storj(monkeypatch, station.s3, read=ARCHIVE_KEY)

    with pytest.raises(storj.StorageReadError):
        run(["--write"], capsys)

    assert station.keys() == keys


def test_a_failed_reference_read_stops_the_repair_before_any_save(
    seed, truth, capsys, monkeypatch
):
    # A failed read is never compared as an empty reference (#185).
    station = seed(truth)
    keys = station.keys()
    flaky_storj(monkeypatch, station.s3, read=REFERENCE_KEY)

    with pytest.raises(storj.StorageReadError):
        run(["--write"], capsys)

    assert station.keys() == keys


def test_an_empty_reference_aborts_the_write(seed, truth, capsys):
    station = seed(truth, reference=truth.iloc[0:0])
    keys = station.keys()

    status, out = run(["--write"], capsys)

    assert status == 1
    assert "is empty" in out
    assert "missing" not in out
    assert "Nothing written" in out
    assert station.keys() == keys


class TestCrossCheck:
    """`cross_check`: refilled rows against the backup, at 2 decimals."""

    def frame(self, values) -> pd.DataFrame:
        return pd.DataFrame({"dateutc": [1, 2, 3], **{c: values for c in COUNTERS}})

    def test_three_decimal_backup_values_match_their_two_decimal_rounding(self):
        reference = self.frame([0.125, 0.114, 1.005])
        refilled = self.frame([0.13, 0.11, 1.01])

        check = repair_rain.cross_check(refilled, reference, 0.001)

        assert (check.compared, check.over) == (3, 0)

    def test_a_nan_on_one_side_is_a_mismatch(self):
        reference = self.frame([0.1, 0.2, 0.3])
        refilled = self.frame([0.1, np.nan, 0.3])

        check = repair_rain.cross_check(refilled, reference, 0.001)

        assert (check.compared, check.over) == (3, 1)

    def test_a_column_the_backup_lacks_is_not_compared(self):
        reference = self.frame([0.1, 0.2, 0.3]).drop(columns=["hourlyrainin"])
        refilled = self.frame([0.1, 0.2, 0.3])

        check = repair_rain.cross_check(refilled, reference, 0.001)

        assert check.over == 0  # only shared columns are compared


class TestVerifyRestored:
    """`verify_restored`: restored rows must equal the backup exactly."""

    def test_a_restored_value_off_by_a_thousandth_differs(self):
        reference = pd.DataFrame({"dateutc": [1, 2], "dailyrainin": [0.1, 0.2]})
        result = pd.DataFrame({"dateutc": [1, 2], "dailyrainin": [0.1, 0.201]})

        assert repair_rain.verify_restored(result, reference, [1, 2]) == (2, 1)

    def test_nan_restored_from_a_nan_matches(self):
        reference = pd.DataFrame({"dateutc": [1], "dailyrainin": [np.nan]})

        assert repair_rain.verify_restored(reference, reference, [1]) == (1, 0)


# ---------- one precedence rule (#431) ----------
# A reading repeated in the reference or in what Ambient serves resolves as
# the archive merge resolves it (`dateutc.normalize`: the last copy wins),
# never by a rule of the repair's own.


def test_a_reading_the_reference_repeats_restores_from_the_copy_the_merge_keeps(
    seed, truth, capsys, frozen_today
):
    x = START_MS + 6 * 3_600_000
    later = truth[truth["dateutc"] == x].assign(
        yearlyrainin=lambda df: df["yearlyrainin"] + 0.25
    )
    reference = pd.concat([truth, later], ignore_index=True)
    station = seed(truth, reference=reference)

    status, out = run(["--write"], capsys)

    assert status == 0, out
    rows = int(in_window(truth).sum())
    assert f"restored {rows} from the backup" in out
    assert f"{rows} restored rows checked, 0 differ from the backup" in out
    stored_reference = storj.read_df_from_s3(BUCKET, REFERENCE_KEY, "parquet")
    copies = stored_reference.loc[stored_reference["dateutc"] == x, "yearlyrainin"]
    assert copies.nunique() == 2
    kept = by_time(normalize(stored_reference)).loc[x, "yearlyrainin"]
    assert by_time(station.stored()).loc[x, "yearlyrainin"] == kept


def test_a_reading_ambient_serves_twice_is_refilled_once(
    seed, truth, capsys, frozen_today
):
    archive, reference, served = _hole_and_two_strays(truth, 0.0)
    x = START_MS + 6 * 3_600_000 + FIVE_MIN
    served = pd.concat([served, served[served["dateutc"] == x]], ignore_index=True)
    station = seed(truth, served=served, reference=reference, archive=archive)

    status, out = run(["--write"], capsys)

    assert status == 0, out
    assert "refilled 3 from Ambient" in out
    assert station.stored()["dateutc"].is_unique


# ---------- the archive-writer lock (#244) ----------


def test_a_held_lock_refuses_write_naming_the_lock_file(
    seed, truth, capsys, archive_lock_file
):
    station = seed(truth)
    keys = station.keys()

    with hold_archive_lock():
        status, out = run(["--write"], capsys)

    assert status == 1
    assert str(archive_lock_file) in out
    assert station.replay.calls == []
    assert station.keys() == keys


def test_a_dry_run_runs_while_the_lock_is_held(seed, truth, capsys):
    seed(truth)

    with hold_archive_lock():
        status, out = run([], capsys)

    assert status == 0
    assert "Dry run" in out


def test_a_write_leaves_the_lock_free_when_it_ends(seed, truth, capsys):
    seed(truth)

    status, _ = run(["--write"], capsys)

    assert status == 0
    with hold_archive_lock():
        pass


# ---------- --station ----------


def window_of(df: pd.DataFrame) -> pd.DataFrame:
    """The rows inside the test window, indexed on `dateutc`."""
    return by_time(df)[in_window(df.sort_values("dateutc")).to_numpy()]


def seed_second(
    station: CatchupStation, truth: pd.DataFrame, reference: bool = True
) -> str:
    """
    Add Salmon Run: on the account, in the inventory, damaged like the first,
    with its own raw reference unless `reference` is False.

    :param reference: Whether to store the second station's raw reference.
    :return: The key of its archive.
    """
    with_second_station(station)
    key = f"{SECOND_MAC}.parquet"
    storj.save_df_to_s3(damaged(truth), BUCKET, key)
    if reference:
        storj.save_df_to_s3(
            truth, BUCKET, repair_rain.REFERENCE_KEY.format(mac=SECOND_MAC)
        )
    return key


def test_without_station_the_default_station_is_repaired(seed, truth, capsys, caplog):
    station = seed(truth)
    second_key = seed_second(station, truth)
    second_before = storj.get_df_from_s3(BUCKET, second_key, file_type="parquet")

    with caplog.at_level("INFO"):
        status, _out = run(["--write"], capsys)

    assert status == 0
    assert f".backups/{FAKE_MAC}/" in caplog.text
    assert_counters_equal(window_of(station.stored()), by_time(truth))
    pd.testing.assert_frame_equal(
        storj.get_df_from_s3(BUCKET, second_key, file_type="parquet"), second_before
    )


def test_station_repairs_the_named_station_only(seed, truth, capsys, caplog):
    station = seed(truth)
    second_key = seed_second(station, truth)
    first_before = station.stored()

    with caplog.at_level("INFO"):
        status, _out = run(["--write", "--station", SECOND_MAC.lower()], capsys)

    assert status == 0
    assert f".backups/{SECOND_MAC}/" in caplog.text
    repaired = storj.get_df_from_s3(BUCKET, second_key, file_type="parquet")
    assert_counters_equal(window_of(repaired), by_time(truth))
    pd.testing.assert_frame_equal(station.stored(), first_before)


@pytest.mark.parametrize("flags", [[], ["--write"]])
def test_a_station_outside_the_inventory_is_refused_before_ambient(
    seed, truth, capsys, flags
):
    station = seed(truth)
    seed_second(station, truth)
    keys = station.keys()
    station.replay.calls.clear()

    status, out = run([*flags, "--station", "00:00:00:00:00:99"], capsys)

    assert status == 1
    assert "Refused: Station 00:00:00:00:00:99 is not in the station" in out
    assert station.replay.calls == []
    assert station.keys() == keys


@pytest.mark.parametrize("flags", [[], ["--write"]])
def test_a_station_with_no_raw_reference_needs_no_repair(seed, truth, capsys, flags):
    station = seed(truth)
    second_key = seed_second(station, truth, reference=False)
    keys = station.keys()
    second_before = storj.get_df_from_s3(BUCKET, second_key, file_type="parquet")

    status, out = run([*flags, "--station", SECOND_MAC], capsys)

    assert status == 0
    assert "no raw reference from before #95" in out
    assert "does not apply" in out
    assert "Problem" not in out
    assert station.keys() == keys
    pd.testing.assert_frame_equal(
        storj.get_df_from_s3(BUCKET, second_key, file_type="parquet"), second_before
    )


# The repair backup and the raw read-back (#169, #170).
REPAIR_KEY = (
    f".backups/{FAKE_MAC}/{FROZEN_DAY}-120000-repair-rain-carryover/{ARCHIVE_KEY}"
)


def test_a_repair_backs_up_even_when_todays_backup_exists(
    seed, truth, capsys, frozen_today, caplog
):
    station = seed(truth)
    station.s3.put_object(Bucket=BUCKET, Key=BACKUP_KEY, Body=b"daily")
    before = station.stored()

    with caplog.at_level("INFO"):
        status, out = run(["--write"], capsys)

    assert status == 0, out
    assert station.s3.get_object(Bucket=BUCKET, Key=BACKUP_KEY)["Body"].read() == (
        b"daily"
    )
    assert REPAIR_KEY in station.keys()
    pd.testing.assert_frame_equal(station.stored(REPAIR_KEY), before)
    assert f"rollback point s3://{BUCKET}/{REPAIR_KEY}" in caplog.text


def test_a_write_prints_the_rollback_key_beside_its_success(
    seed, truth, capsys, frozen_today
):
    seed(truth)

    status, out = run(["--write"], capsys)

    assert status == 0, out
    lines = out.splitlines()
    saved = lines.index("Repaired archive saved; every read-back check passed.")
    assert lines[saved + 1] == f"Rollback point: s3://{BUCKET}/{REPAIR_KEY}"


def test_a_dry_run_prints_no_rollback_key(seed, truth, capsys, frozen_today):
    station = seed(truth)
    keys = station.keys()

    status, out = run([], capsys)

    assert status == 0, out
    assert "Rollback point" not in out
    assert ".backups/" not in out
    assert station.keys() == keys


def test_a_failed_repair_backup_saves_nothing(
    seed, truth, capsys, monkeypatch, frozen_today
):
    station = seed(truth)
    keys = station.keys()
    before = station.stored()
    flaky_storj(monkeypatch, station.s3, copy=True)

    with pytest.raises(storj.BackupError):
        run(["--write"], capsys)

    assert station.keys() == keys
    pd.testing.assert_frame_equal(station.stored(), before)


def test_a_correct_write_passes_every_readback_check(
    seed, truth, capsys, frozen_today, caplog
):
    seed(truth)

    with caplog.at_level("INFO"):
        status, out = run(["--write"], capsys)

    assert status == 0, out
    lines = readback_lines(caplog)
    assert len(lines) == 4
    assert all(line.startswith("Read-back check passed") for line in lines)
    assert "passed: restored rows equal the raw reference" in lines[-1]


def _save_then(monkeypatch, damage: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """After the repair's save, replace the stored archive with `damage` of it."""
    save = storj.save_archive

    def save_and_damage(df, bucket, mac, **kwargs):
        save(df, bucket, mac, **kwargs)
        stored = storj.read_df_from_s3(bucket, f"{mac}.parquet", file_type="parquet")
        storj.save_df_to_s3(damage(stored), bucket, f"{mac}.parquet")

    monkeypatch.setattr(storj, "save_archive", save_and_damage)


def oldest_first(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[::-1].reset_index(drop=True)


def missing_a_row(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(df.index[len(df) // 2]).reset_index(drop=True)


def changed_outside(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    row = ~in_window(out)
    out.loc[row[row].index[0], "tempf"] += 1.0
    return out


@pytest.mark.parametrize(
    "damage, failed",
    [
        (oldest_first, "dateutc strictly decreasing"),
        (missing_a_row, "row count"),
        (changed_outside, "rows outside the windows unchanged"),
    ],
)
def test_a_bad_stored_archive_fails_its_readback_check(
    seed, truth, capsys, monkeypatch, frozen_today, caplog, damage, failed
):
    seed(truth)
    _save_then(monkeypatch, damage)

    with caplog.at_level("INFO"):
        status, out = run(["--write"], capsys)

    assert status == 1
    line = next(ln for ln in readback_lines(caplog) if f"FAILED: {failed}" in ln)
    assert REPAIR_KEY in line
    assert f"roll back from s3://{BUCKET}/{REPAIR_KEY}" in out


def duplicated(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.head(1)], ignore_index=True).sort_values(
        "dateutc", ascending=False, kind="stable", ignore_index=True
    )


def test_a_duplicate_dateutc_fails_the_order_check(
    seed, truth, capsys, monkeypatch, frozen_today, caplog
):
    seed(truth)
    _save_then(monkeypatch, duplicated)

    with caplog.at_level("INFO"):
        status, _out = run(["--write"], capsys)

    assert status == 1
    failed = "FAILED: dateutc strictly decreasing"
    assert any(failed in line for line in readback_lines(caplog))


def test_an_unreadable_readback_names_the_rollback_key(
    seed, truth, capsys, monkeypatch, frozen_today
):
    station = seed(truth)
    save = storj.save_archive

    def save_then_delete(df, bucket, mac, **kwargs):
        save(df, bucket, mac, **kwargs)
        station.s3.delete_object(Bucket=BUCKET, Key=ARCHIVE_KEY)

    monkeypatch.setattr(storj, "save_archive", save_then_delete)

    status, out = run(["--write"], capsys)

    assert status == 1
    assert f"roll back from s3://{BUCKET}/{REPAIR_KEY}" in out
