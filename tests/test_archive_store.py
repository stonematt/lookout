"""
The archive store (#428, spec #423): `lookout.storage.archive_store`.

Tests drive the archive store's interface, `open` and the session's
`merge`, with Storj on moto, and check what ends up in the bucket (the
archive, its backups) and what the caller sees (the merge report, the
exception). The archive-writer lock's "held by this process" check is
tested only here, through a save that refuses without it.
"""

from io import BytesIO
from typing import Iterator, List, Tuple

import pandas as pd
import pytest

from lookout.core import archive_merge
from lookout.storage import archive_store, storj
from lookout.storage.archive_readback import Check, ReadbackFailed
from lookout.utils.archive_lock import ArchiveLockNotHeld, hold_archive_lock
from tests.fixtures.archive import FAKE_MAC, text_as_values
from tests.fixtures.real import load_real_slice
from tests.fixtures.storage import (
    BUCKET,
    FROZEN_DAY,
    FlakyStorj,
    bucket_keys,
    flaky_storj,
)

ARCHIVE_KEY = f"{FAKE_MAC}.parquet"
FIVE_MINUTES_MS = 5 * 60 * 1000


@pytest.fixture
def stored(fake_s3: object, frozen_today: str) -> pd.DataFrame:
    """
    The real slice, stored as the station's archive.

    :return: The archive as seeded, newest first.
    """
    archive = load_real_slice()
    storj.save_df_to_s3(archive, BUCKET, ARCHIVE_KEY)
    return archive


@pytest.fixture
def locked() -> Iterator[None]:
    """Hold the archive-writer lock for the test, as an archive writer does."""
    with hold_archive_lock():
        yield


def backup_key(day: str) -> str:
    """The daily backup's key for `day` (`YYYYMMDD`)."""
    return f".backups/{FAKE_MAC}/{day}/{ARCHIVE_KEY}"


def stored_bytes(s3: object, key: str = ARCHIVE_KEY) -> bytes:
    """The object's bytes, as stored."""
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def refetched(archive: pd.DataFrame, rows: int = 3) -> pd.DataFrame:
    """The newest stored readings, as Ambient sends them again each run."""
    newest = archive.sort_values("dateutc", ascending=False).head(rows)
    return text_as_values(newest).reset_index(drop=True)


def new_readings(archive: pd.DataFrame, count: int = 1) -> pd.DataFrame:
    """
    Readings after the archive's newest, five minutes apart, as Ambient sends.

    :param archive: The archive.
    :param count: How many.
    :return: Copies of the newest reading, moved forward.
    """
    newest = refetched(archive, rows=1)
    rows = []
    for i in range(1, count + 1):
        row = newest.copy()
        row["dateutc"] += i * FIVE_MINUTES_MS
        row["date"] = row["date"] + pd.to_timedelta(i * FIVE_MINUTES_MS, unit="ms")
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


class TestOpen:
    def test_opens_the_station_archive_normalized(self, stored):
        session = archive_store.open(BUCKET, FAKE_MAC)

        assert len(session.archive) == len(stored)
        assert session.archive["dateutc"].is_monotonic_increasing

    def test_a_failed_read_raises(self, stored, fake_s3, monkeypatch):
        flaky_storj(monkeypatch, fake_s3, read=ARCHIVE_KEY)

        with pytest.raises(storj.StorageReadError):
            archive_store.open(BUCKET, FAKE_MAC)

    def test_a_missing_archive_raises(self, fake_s3):
        with pytest.raises(storj.StorageReadError, match="missing"):
            archive_store.open(BUCKET, FAKE_MAC)

    def test_a_missing_archive_is_told_apart_from_a_failed_read(
        self, stored, fake_s3, monkeypatch
    ):
        # `stations add` builds only on missing, never on failed (ADR-0004).
        with pytest.raises(storj.ArchiveMissing):
            archive_store.open(BUCKET, "00:00:00:00:00:00")

        flaky_storj(monkeypatch, fake_s3, read=ARCHIVE_KEY)
        with pytest.raises(storj.StorageReadError) as failed:
            archive_store.open(BUCKET, FAKE_MAC)

        assert not isinstance(failed.value, storj.ArchiveMissing)


class TestMerge:
    def test_a_merge_that_changes_nothing_writes_nothing_and_takes_no_backup(
        self, stored, fake_s3, locked
    ):
        before = stored_bytes(fake_s3)
        session = archive_store.open(BUCKET, FAKE_MAC)

        report = session.merge(refetched(stored))

        assert not report.changed
        assert bucket_keys(fake_s3) == [ARCHIVE_KEY]
        assert stored_bytes(fake_s3) == before

    def test_a_changing_merge_backs_up_then_saves(
        self, stored, fake_s3, frozen_today, locked
    ):
        before = stored_bytes(fake_s3)
        session = archive_store.open(BUCKET, FAKE_MAC)

        report = session.merge(new_readings(stored, 2))

        assert report.added == 2
        assert bucket_keys(fake_s3) == sorted([ARCHIVE_KEY, backup_key(frozen_today)])
        assert stored_bytes(fake_s3, backup_key(frozen_today)) == before
        saved = storj.load_archive(BUCKET, FAKE_MAC)
        assert len(saved) == len(stored) + 2

    def test_the_daily_backup_is_taken_once_a_day(
        self, stored, fake_s3, frozen_today, locked
    ):
        before = stored_bytes(fake_s3)
        archive_store.open(BUCKET, FAKE_MAC).merge(new_readings(stored, 1))

        archive_store.open(BUCKET, FAKE_MAC).merge(new_readings(stored, 2))

        # Today's backup still holds the archive before the day's first save.
        assert stored_bytes(fake_s3, backup_key(frozen_today)) == before
        assert len(storj.load_archive(BUCKET, FAKE_MAC)) == len(stored) + 2

    def test_the_session_holds_the_merged_archive_with_its_text_as_category(
        self, stored, locked
    ):
        # #285: what every later step of a writer reads holds its text as
        # category, though Ambient's fetch sends plain text.
        fetched = new_readings(stored, 1)
        assert fetched["battrain"].dtype == object
        session = archive_store.open(BUCKET, FAKE_MAC)

        session.merge(fetched)

        assert len(session.archive) == len(stored) + 1
        assert session.archive["battrain"].dtype == "category"
        assert session.archive["battrain"].isna().sum() == 0

    def test_a_merge_that_would_drop_a_stored_reading_raises(
        self, stored, fake_s3, monkeypatch, locked
    ):
        before = stored_bytes(fake_s3)
        # A merge never drops a stored reading, so stand in for one that did.
        monkeypatch.setattr(
            archive_merge,
            "merge",
            lambda archive, _fetched: archive_merge.Merged(
                archive.iloc[1:], archive_merge.MergeReport(added=1)
            ),
        )
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(storj.ArchiveLossError, match="drop 1 stored reading"):
            session.merge(new_readings(stored, 1))

        assert stored_bytes(fake_s3) == before

    def test_a_failed_backup_saves_nothing(self, stored, fake_s3, monkeypatch, locked):
        before = stored_bytes(fake_s3)
        flaky_storj(monkeypatch, fake_s3, copy=True)
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(storj.BackupError):
            session.merge(new_readings(stored, 1))

        assert stored_bytes(fake_s3) == before


def repair_backup_key(reason: str) -> str:
    """The repair backup's key at the frozen moment, for `reason`."""
    return f".backups/{FAKE_MAC}/{FROZEN_DAY}-120000-repair-{reason}/{ARCHIVE_KEY}"


def replaced(archive: pd.DataFrame) -> pd.DataFrame:
    """
    A repaired archive: the stored one less its oldest reading, with the
    newest reading's temperature corrected.

    :param archive: The archive as opened.
    :return: The repaired frame, oldest first.
    """
    repaired = archive.sort_values("dateutc").iloc[1:].reset_index(drop=True)
    repaired.loc[repaired.index[-1], "tempf"] += 1.0
    return repaired


class TestReplace:
    def test_a_replace_backs_up_the_archive_it_replaces_after_the_daily_backup(
        self, stored, fake_s3, frozen_today, locked
    ):
        before = stored_bytes(fake_s3)
        # The day's first routine save already took the daily backup.
        fake_s3.put_object(
            Bucket=BUCKET, Key=backup_key(frozen_today), Body=b"this morning"
        )
        session = archive_store.open(BUCKET, FAKE_MAC)

        session.replace(replaced(session.archive), "normalize")

        assert stored_bytes(fake_s3, repair_backup_key("normalize")) == before
        assert stored_bytes(fake_s3, backup_key(frozen_today)) == b"this morning"
        assert stored_bytes(fake_s3) != before

    def test_a_replace_saves_with_the_loss_check_lifted(self, stored, locked):
        session = archive_store.open(BUCKET, FAKE_MAC)
        repaired = replaced(session.archive)

        session.replace(repaired, "normalize")

        saved = storj.load_archive(BUCKET, FAKE_MAC)
        assert len(saved) == len(stored) - 1
        assert saved["dateutc"].min() == repaired["dateutc"].min()
        assert saved["tempf"].iloc[-1] == repaired["tempf"].iloc[-1]

    def test_a_replace_returns_the_merge_report(self, stored, locked):
        session = archive_store.open(BUCKET, FAKE_MAC)

        report = session.replace(replaced(session.archive), "normalize")

        assert (report.added, report.filled, report.updated) == (0, 0, 1)

    def test_the_reason_is_in_the_log(self, stored, locked, caplog):
        session = archive_store.open(BUCKET, FAKE_MAC)

        with caplog.at_level("INFO", logger="lookout.storage.archive_store"):
            session.replace(replaced(session.archive), "rain-carryover")

        assert (
            "Archive replace (rain-carryover): rollback point "
            f"s3://{BUCKET}/{repair_backup_key('rain-carryover')}" in caplog.messages
        )

    def test_a_failed_backup_raises_and_leaves_the_archive(
        self, stored, fake_s3, monkeypatch, locked
    ):
        before = stored_bytes(fake_s3)
        flaky_storj(monkeypatch, fake_s3, copy=True)
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(storj.BackupError):
            session.replace(replaced(session.archive), "normalize")

        assert bucket_keys(fake_s3) == [ARCHIVE_KEY]
        assert stored_bytes(fake_s3) == before

    def test_an_archive_gone_before_the_backup_raises_and_saves_nothing(
        self, stored, fake_s3, locked
    ):
        session = archive_store.open(BUCKET, FAKE_MAC)
        fake_s3.delete_object(Bucket=BUCKET, Key=ARCHIVE_KEY)

        with pytest.raises(storj.BackupError, match="to back up"):
            session.replace(replaced(session.archive), "normalize")

        assert bucket_keys(fake_s3) == []

    @pytest.mark.parametrize("reason", ["", "Rain Carryover", "../daily", "rain-"])
    def test_a_reason_that_is_not_hyphenated_words_refuses_and_writes_nothing(
        self, stored, fake_s3, locked, reason
    ):
        before = stored_bytes(fake_s3)
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(ValueError, match="repair reason"):
            session.replace(replaced(session.archive), reason)

        assert bucket_keys(fake_s3) == [ARCHIVE_KEY]
        assert stored_bytes(fake_s3) == before


class ReadBackFails(FlakyStorj):
    """Storage whose archive read fails once the archive has been saved."""

    def upload_fileobj(self, **kwargs: object) -> None:
        super().upload_fileobj(**kwargs)
        if kwargs["Key"] == ARCHIVE_KEY:
            self.read = ARCHIVE_KEY


class DropsARow(FlakyStorj):
    """Storage that saves the archive less its first row: a corrupt write."""

    def upload_fileobj(self, **kwargs: object) -> None:
        if kwargs["Key"] == ARCHIVE_KEY:
            frame = pd.read_parquet(kwargs["Fileobj"]).iloc[1:]
            kwargs["Fileobj"] = BytesIO(frame.to_parquet())
        super().upload_fileobj(**kwargs)


def readback_lines(caplog: pytest.LogCaptureFixture) -> List[str]:
    """The read-back check lines logged."""
    return [m for m in caplog.messages if m.startswith("Read-back check")]


class TestReplaceReadsBack:
    def test_a_replace_reads_the_archive_back_and_logs_each_check(
        self, stored, locked, caplog
    ):
        session = archive_store.open(BUCKET, FAKE_MAC)
        rows = len(stored) - 1

        with caplog.at_level("INFO"):
            session.replace(replaced(session.archive), "normalize")

        assert readback_lines(caplog) == [
            f"Read-back check passed: row count ({rows} stored, {rows} predicted)",
            "Read-back check passed: rows outside the windows unchanged "
            f"({rows} stored, {rows} before)",
            "Read-back check passed: dateutc strictly decreasing (0 duplicates)",
        ]

    def test_a_corrupt_read_back_fails(self, stored, fake_s3, monkeypatch, locked):
        flaky_storj_as(monkeypatch, DropsARow(fake_s3))
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(ReadbackFailed, match=repair_backup_key("normalize")):
            session.replace(replaced(session.archive), "normalize")

    def test_an_unreadable_read_back_is_a_failed_check(
        self, stored, fake_s3, monkeypatch, locked, caplog
    ):
        flaky_storj_as(monkeypatch, ReadBackFails(fake_s3))
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(ReadbackFailed):
            session.replace(replaced(session.archive), "normalize")

        (failed,) = readback_lines(caplog)
        assert failed.startswith(
            "Read-back check FAILED: the saved archive is unreadable"
        )
        assert failed.endswith(f"roll back from {repair_backup_key('normalize')}")

    def test_a_windowed_replace_passes_when_only_its_windows_changed(
        self, stored, locked
    ):
        session = archive_store.open(BUCKET, FAKE_MAC)
        repaired = newest_corrected(session.archive)

        session.replace(repaired, "rain-carryover", windows=[newest_window(stored)])

        assert len(storj.load_archive(BUCKET, FAKE_MAC)) == len(stored)

    def test_a_windowed_replace_that_changed_a_reading_outside_fails(
        self, stored, locked, caplog
    ):
        session = archive_store.open(BUCKET, FAKE_MAC)
        # Drops the oldest reading, outside the window.
        repaired = replaced(session.archive)

        with pytest.raises(ReadbackFailed):
            session.replace(repaired, "rain-carryover", windows=[newest_window(stored)])

        assert any(
            "FAILED: rows outside the windows unchanged" in line
            for line in readback_lines(caplog)
        )

    def test_a_check_of_the_callers_own_runs_on_the_read_back(self, stored, locked):
        session = archive_store.open(BUCKET, FAKE_MAC)
        seen = []

        def restored_rows(read_back: pd.DataFrame) -> List[Check]:
            seen.append(len(read_back))
            return [Check("restored rows equal the raw reference", False, "1 differ")]

        with pytest.raises(ReadbackFailed):
            session.replace(
                replaced(session.archive), "rain-carryover", checks=restored_rows
            )

        assert seen == [len(stored) - 1]


def flaky_storj_as(monkeypatch: pytest.MonkeyPatch, client: FlakyStorj) -> None:
    """Route every `storj` client through `client`."""
    monkeypatch.setattr(storj, "get_s3_client", lambda: client)


def newest_corrected(archive: pd.DataFrame) -> pd.DataFrame:
    """The archive with its newest reading's temperature corrected."""
    repaired = archive.sort_values("dateutc").reset_index(drop=True)
    repaired.loc[repaired.index[-1], "tempf"] += 1.0
    return repaired


def newest_window(archive: pd.DataFrame) -> Tuple[int, int]:
    """A window holding only the archive's newest reading."""
    newest = int(archive["dateutc"].max())
    return newest, newest + 1


class TestTheLock:
    def test_a_merge_without_the_lock_refuses_and_writes_nothing(self, stored, fake_s3):
        before = stored_bytes(fake_s3)
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(ArchiveLockNotHeld):
            session.merge(new_readings(stored, 1))

        assert bucket_keys(fake_s3) == [ARCHIVE_KEY]
        assert stored_bytes(fake_s3) == before

    def test_a_merge_after_the_lock_is_released_refuses(self, stored, fake_s3):
        with hold_archive_lock():
            session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(ArchiveLockNotHeld):
            session.merge(new_readings(stored, 1))

        assert bucket_keys(fake_s3) == [ARCHIVE_KEY]

    def test_a_replace_without_the_lock_refuses_and_writes_nothing(
        self, stored, fake_s3
    ):
        before = stored_bytes(fake_s3)
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(ArchiveLockNotHeld):
            session.replace(replaced(session.archive), "normalize")

        assert bucket_keys(fake_s3) == [ARCHIVE_KEY]
        assert stored_bytes(fake_s3) == before

    def test_a_quiet_merge_without_the_lock_refuses_too(self, stored):
        # A writer that forgets the lock fails on its first run, not on the
        # first run that happens to have something to save.
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(ArchiveLockNotHeld):
            session.merge(refetched(stored))


class TestCreate:
    def test_builds_the_first_archive_newest_first_with_no_backup(
        self, fake_s3, locked
    ):
        fetched = text_as_values(load_real_slice())

        session = archive_store.create(BUCKET, FAKE_MAC, fetched)

        assert bucket_keys(fake_s3) == [ARCHIVE_KEY]  # nothing to back up
        raw = storj.read_df_from_s3(BUCKET, ARCHIVE_KEY, file_type="parquet")
        assert raw["dateutc"].is_monotonic_decreasing
        saved = storj.load_archive(BUCKET, FAKE_MAC)
        assert sorted(saved["dateutc"]) == sorted(fetched["dateutc"])
        assert len(session.archive) == len(fetched)

    def test_refuses_a_found_archive_and_leaves_it_as_it_was(
        self, stored, fake_s3, locked
    ):
        before = stored_bytes(fake_s3)

        with pytest.raises(archive_store.ArchiveExists):
            archive_store.create(BUCKET, FAKE_MAC, new_readings(stored, 1))

        assert bucket_keys(fake_s3) == [ARCHIVE_KEY]
        assert stored_bytes(fake_s3) == before

    def test_refuses_a_found_archive_with_no_valid_reading(self, fake_s3, locked):
        # Whatever the stored object holds, a first build never overwrites it.
        stale = load_real_slice().assign(dateutc=0)
        storj.save_df_to_s3(stale, BUCKET, ARCHIVE_KEY)
        before = stored_bytes(fake_s3)

        with pytest.raises(archive_store.ArchiveExists):
            archive_store.create(BUCKET, FAKE_MAC, text_as_values(load_real_slice()))

        assert stored_bytes(fake_s3) == before

    def test_a_failed_read_raises_and_writes_nothing(
        self, fake_s3, monkeypatch, locked
    ):
        # A read that failed is not "no archive" (ADR-0004).
        flaky_storj(monkeypatch, fake_s3, read=ARCHIVE_KEY)

        with pytest.raises(storj.StorageReadError):
            archive_store.create(BUCKET, FAKE_MAC, text_as_values(load_real_slice()))

        assert bucket_keys(fake_s3) == []

    def test_no_valid_reading_raises_and_writes_nothing(self, fake_s3, locked):
        with pytest.raises(archive_store.NoReadings):
            archive_store.create(BUCKET, FAKE_MAC, pd.DataFrame())

        assert bucket_keys(fake_s3) == []

    def test_without_the_lock_it_refuses_and_writes_nothing(self, fake_s3):
        with pytest.raises(ArchiveLockNotHeld):
            archive_store.create(BUCKET, FAKE_MAC, text_as_values(load_real_slice()))

        assert bucket_keys(fake_s3) == []
