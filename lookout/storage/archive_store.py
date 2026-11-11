"""
archive_store.py: The archive store, the one write path for a station's
archive (spec #423).

An archive writer takes the archive-writer lock at its entry point
(`lookout.utils.archive_lock`), opens the station's archive here, and names
the change it wants. The archive store applies the backup that change
calls for, skips a save that changes nothing, and saves with the loss check
(`storj.save_archive`). Writers never pass `stored=` or `repair=` and never
call the backup functions themselves.

- `open(bucket, mac)` loads the archive through `storj.load_archive`
  (normalized; a failed read raises `StorageReadError`, ADR-0004, and so
  does a missing archive, as `storj.ArchiveMissing`: adding a station is a
  human act) and returns an `ArchiveSession`.
- `create(bucket, mac, fetched)` builds a station's first archive, for
  `stations add`. It refuses when an archive is found (`ArchiveExists`),
  raises when the read failed (an unknown archive state never counts as
  "no archive", ADR-0004), takes no backup, and saves through the archive
  save.
- `ArchiveSession.merge(fetched)` merges fetched readings by the archive
  merge (Ambient wins, `lookout.core.archive_merge`). When nothing changed
  it writes nothing and takes no backup. Otherwise it takes the daily
  backup (`storj.backup_archive`), then saves; a save that would drop or
  blank a stored reading raises `ArchiveLossError`. It returns the merge
  report. Plain merges do not read the archive back: a read-back is a full
  download, and the refresh run merges every 5 minutes.

Every verb that may save refuses, before it reads or writes anything, when
this process does not hold the archive-writer lock (`ArchiveLockNotHeld`),
whether or not this call would have saved: a writer that forgets the lock
fails on its first run.

Offers `open`, `create`, `ArchiveSession`, `ArchiveExists` and
`NoReadings`.
"""

from datetime import datetime, timezone

import pandas as pd

from lookout.config import HISTORY_FILE_TYPE
from lookout.core import archive_merge
from lookout.storage import storj
from lookout.utils import archive_lock
from lookout.utils.log_util import app_logger
from lookout.utils.memory_utils import log_peak_rss

logger = app_logger(__name__)


class ArchiveExists(RuntimeError):
    """`create` found an archive stored for the station; it wrote nothing."""


class NoReadings(ValueError):
    """`create` was given no valid reading to build an archive from."""


class ArchiveSession:
    """
    A station's archive, opened for an archive writer; its verbs save it.

    :param bucket: The bucket.
    :param mac: The station's MAC address.
    :param archive: The archive as loaded (`storj.load_archive`).
    """

    def __init__(self, bucket: str, mac: str, archive: pd.DataFrame) -> None:
        self.bucket = bucket
        self.mac = mac
        self._archive = archive

    @property
    def archive(self) -> pd.DataFrame:
        """
        The archive as it stands: as loaded, then as each verb left it.

        Raw, as stored: rain corrections are applied on read by the caller,
        never here (#95).

        :return: The frame. Row order is not part of the contract (as
            loaded it is oldest first, after a merge newest first): take the
            newest reading with `max()`, never by position.
        """
        return self._archive

    def merge(self, fetched: pd.DataFrame) -> archive_merge.MergeReport:
        """
        Merge fetched readings in, and save when anything changed.

        A merge that changes nothing writes nothing and takes no backup
        (#242). One that changes something takes the daily backup, then
        saves, refusing any loss of a stored reading or value.

        :param fetched: The readings Ambient sent, raw; may be empty.
        :return: What the merge added, filled and updated.
        :raises ArchiveLockNotHeld: If this process does not hold the
            archive-writer lock; nothing is merged or written.
        :raises BackupError: If the backup fails; nothing is saved.
        :raises ArchiveLossError: If the merged archive would drop a stored
            reading or blank a stored value; nothing is saved.
        :raises StorageWriteError: If the save fails.
        """
        _require_lock()
        merged, report = archive_merge.merge(self._archive, fetched)
        logger.info(f"Archive merge: {report.describe()}")
        log_peak_rss("archive_merge", logger)
        if report.changed:
            storj.backup_archive(bucket=self.bucket, mac=self.mac)
            storj.save_archive(merged, self.bucket, self.mac, stored=self._archive)
            logger.info(f"Archive saved: {_describe(merged)}")
        else:
            # Skip the whole-archive rewrite, and its window for a partial
            # write, on the quiet runs (#242).
            logger.info("Archive save skipped: the merge changed nothing.")
        log_peak_rss("archive_save", logger)
        self._archive = merged
        return report


def open(bucket: str, mac: str) -> ArchiveSession:
    """
    Open a station's archive for an archive writer.

    The one archive load on the write path (#329 lands here). Opening reads
    only, so it needs no lock; the verbs that save check it.

    :param bucket: The bucket.
    :param mac: The station's MAC address.
    :return: The session to merge through.
    :raises StorageReadError: If the archive is missing (`ArchiveMissing`),
        unreadable, or has no valid rows (ADR-0004).
    """
    archive = storj.load_archive(bucket, mac)
    logger.info(f"Archive loaded: {_describe(archive)}")
    log_peak_rss("archive_load", logger)
    return ArchiveSession(bucket, mac, archive)


def create(bucket: str, mac: str, fetched: pd.DataFrame) -> ArchiveSession:
    """
    Build a station's first archive from fetched readings, and save it.

    For `stations add`, when no archive is stored for the station. It reads
    the archive's key first: a found archive is refused, never overwritten,
    and a failed read raises, since an unknown archive state never counts as
    "no archive" (ADR-0004). There is nothing to back up, so it takes no
    backup. The readings join through the archive merge, as every fetch
    does, and save through the archive save.

    :param bucket: The bucket.
    :param mac: The station's MAC address.
    :param fetched: The readings Ambient sent, raw.
    :return: The session, holding the archive as saved.
    :raises ArchiveLockNotHeld: If this process does not hold the
        archive-writer lock; nothing is read or written.
    :raises StorageReadError: If the archive read failed; nothing is written.
    :raises ArchiveExists: If an archive is stored for the station; nothing
        is written.
    :raises NoReadings: If `fetched` holds no valid reading; nothing is
        written.
    :raises StorageWriteError: If the save fails.
    """
    _require_lock()
    key = f"{mac}.{HISTORY_FILE_TYPE}"
    # Only `dateutc` is parsed: the read only has to end found, missing or
    # failed, and a found archive is refused whatever it holds.
    found = storj.read_df_from_s3(
        bucket, key, file_type=HISTORY_FILE_TYPE, columns=["dateutc"]
    )
    if found is not None:
        raise ArchiveExists(f"s3://{bucket}/{key} is stored; not creating it")
    archive, report = archive_merge.merge(pd.DataFrame(), fetched)
    logger.info(f"First archive merge: {report.describe()}")
    if archive.empty:
        raise NoReadings(f"no valid reading for {mac}; nothing to archive")
    storj.save_archive(archive, bucket, mac, stored=pd.DataFrame())
    logger.info(f"Archive created: {_describe(archive)}")
    return ArchiveSession(bucket, mac, archive)


def _require_lock() -> None:
    """
    Refuse unless this process holds the archive-writer lock.

    :raises ArchiveLockNotHeld: If it does not.
    """
    if not archive_lock.held_by_this_process():
        raise archive_lock.ArchiveLockNotHeld(archive_lock.lock_file())


def _describe(archive: pd.DataFrame) -> str:
    """
    The archive's size and span, for the log.

    :param archive: A non-empty archive.
    :return: E.g. "105,120 readings, 2025-01-01 08:00Z to 2026-01-01 07:55Z".
    """

    def utc(ms: int) -> str:
        """:param ms: Epoch ms. :return: The UTC minute it falls in."""
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%MZ"
        )

    times = archive["dateutc"]
    return f"{len(archive):,} readings, {utc(times.min())} to {utc(times.max())}"
