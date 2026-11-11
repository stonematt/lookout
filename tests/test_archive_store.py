"""
The archive store (#428, spec #423): `lookout.storage.archive_store`.

Tests drive the archive store's interface, `open` and the session's
`merge`, with Storj on moto, and check what ends up in the bucket (the
archive, its backups) and what the caller sees (the merge report, the
exception). The archive-writer lock's "held by this process" check is
tested only here, through a save that refuses without it.
"""

from typing import Iterator

import pandas as pd
import pytest

from lookout.core import archive_merge
from lookout.storage import archive_store, storj
from lookout.utils.archive_lock import ArchiveLockNotHeld, hold_archive_lock
from tests.fixtures.archive import FAKE_MAC, text_as_values
from tests.fixtures.real import load_real_slice
from tests.fixtures.storage import BUCKET, bucket_keys, flaky_storj

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

    def test_a_quiet_merge_without_the_lock_refuses_too(self, stored):
        # A writer that forgets the lock fails on its first run, not on the
        # first run that happens to have something to save.
        session = archive_store.open(BUCKET, FAKE_MAC)

        with pytest.raises(ArchiveLockNotHeld):
            session.merge(refetched(stored))
