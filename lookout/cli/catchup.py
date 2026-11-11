"""
catchup.py: Synchronize new Ambient Weather data with archived records in S3.

This module pulls data since the last known timestamp in the archive and updates
S3 storage with any newly retrieved data. Supports a dry-run mode for simulation
and offline testing. Also supports archive repair (schema normalization) with
--repair-archive and optional --dry-run; the repair also strips the derived
columns stored (#191). A repair that writes backs the archive up to a repair
backup of its own right before it saves, every time, and reads the saved
archive back raw (#422); a dry run does neither. A run that writes takes the
archive-writer lock first (#244); `--dry-run` alone takes none.

Usage:
    python catchup.py --bucket lookout [--dry-run] [--pages 20] [--repair-archive]
"""

import argparse
import sys
from typing import Dict, Tuple

import pandas as pd

import lookout.api.awn_controller as awn
import lookout.storage.storj as sj
from lookout.api.ambient_client import get_station_device
from lookout.core import archive_merge
from lookout.storage import archive_readback
from lookout.storage.station_inventory import resolve_station
from lookout.utils.archive_lock import ArchiveLockHeld, writer_lock
from lookout.utils.dateutc import normalize
from lookout.utils.log_util import app_logger
from lookout.utils.memory_utils import log_peak_rss

logger = app_logger(__name__, log_file="catchup.log")


# ---------- archive utilities ----------


def _repair_archive_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    CLI-shaped wrapper around `dateutc.normalize` that produces an
    operator-readable metrics dict for the `--repair-archive` report.

    Returns (fixed_df, metrics dict). The actual validation/cast/dedupe/
    sort work is delegated to `normalize`; this function only adapts
    the result into a metrics shape. The fixed frame also drops the derived
    columns stored (`archive_merge.DERIVED_COLUMNS`, #191), named in
    `metrics["derived_columns"]`.
    """
    metrics: Dict = {}
    if df is None or df.empty:
        return pd.DataFrame(), {"status": "empty_input"}

    if "dateutc" not in df.columns:
        # try common fallback column names; if found, map to dateutc for normalization
        candidates = [c for c in ("date", "dt_utc", "timestamp") if c in df.columns]
        if candidates:
            df = df.rename(columns={candidates[0]: "dateutc"})
        else:
            return pd.DataFrame(), {"status": "missing_dateutc"}

    metrics["before_len"] = len(df)
    metrics["before_dtype"] = str(df["dateutc"].dtype)

    normalized = normalize(df)
    fixed = archive_merge.strip_derived(normalized)
    metrics["derived_columns"] = [
        c for c in normalized.columns if c not in fixed.columns
    ]

    after = len(fixed)
    metrics["after_len"] = after
    metrics["dropped"] = metrics["before_len"] - after
    metrics["after_dtype"] = str(fixed["dateutc"].dtype) if not fixed.empty else "empty"

    if not fixed.empty:
        mn = int(fixed["dateutc"].iloc[0])
        mx = int(fixed["dateutc"].iloc[-1])
        metrics["min"] = pd.to_datetime(mn, unit="ms", utc=True)
        metrics["max"] = pd.to_datetime(mx, unit="ms", utc=True)

    return fixed, metrics


def _check_repair_readback(
    bucket_name: str, mac: str, repaired: pd.DataFrame, backup_key: str
) -> None:
    """
    Read the repaired archive back raw and check it (#170, #422).

    The repair rewrites the whole archive, so the read-back is checked against
    the repaired archive itself, with no windows: the stored rows must be the
    repaired rows, as many, newest first with no duplicate. One log line per
    check; an unreadable read-back is a failed check.

    :param bucket_name: The bucket.
    :param mac: The station's MAC address.
    :param repaired: The archive the repair saved, in memory.
    :param backup_key: The repair backup, named on any failure.
    :raises ReadbackFailed: If any check failed.
    """
    try:
        stored = archive_readback.read_back(bucket_name, mac)
    except sj.StorageReadError as e:
        checks = [
            archive_readback.Check("the saved archive is unreadable", False, str(e))
        ]
    else:
        checks = archive_readback.check_readback(stored, repaired, len(repaired), [])
    for check in checks:
        if check.ok:
            logger.info(check.line(backup_key))
        else:
            logger.error(check.line(backup_key))
    if not all(check.ok for check in checks):
        raise archive_readback.ReadbackFailed(
            f"The repaired archive failed its read-back; roll back from {backup_key}"
        )


def _log_archive_range(df: pd.DataFrame, label: str) -> None:
    """
    Log human-readable range. Works whether df has `date` (tz-aware) or just `dateutc`.
    """
    if df is None or df.empty:
        logger.info(f"{label}: <empty>")
        return

    if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
        lo = df["date"].min()
        hi = df["date"].max()
    else:
        lo_ms = int(df["dateutc"].min())
        hi_ms = int(df["dateutc"].max())
        lo = pd.to_datetime(lo_ms, unit="ms", utc=True)
        hi = pd.to_datetime(hi_ms, unit="ms", utc=True)

    logger.info(
        f"{label}: Range: ({lo.strftime('%y-%m-%d %H:%M')}) - ({hi.strftime('%y-%m-%d %H:%M')})"
    )


# ---------- main flow ----------


def update_archive(device: Dict, bucket_name: str, pages: int) -> pd.DataFrame:
    """
    One catchup: back up the archive, merge in what Ambient has since, save.

    Fetched readings join the archive through the archive merge: new
    readings are added, and for the readings re-fetched, Ambient's value wins
    for each value it sends while values it omits keep their stored value.
    The merge report (added, filled, updated, one-sided columns) is logged.

    The catchup CLI runs this. The refresh run does the same work in two
    phases, so all its Ambient calls come before any station's archive is
    loaded (#271): `fetch_delta` first, then `save_fetched`. Every failure
    raises, so the run exits non-zero (#160): a failed backup before
    anything is read, a failed or empty archive read before anything is
    saved, a save that would drop a stored reading or blank a stored value,
    and a failed save.

    :param device: The station's `/devices` entry.
    :param bucket_name: S3 bucket to read/write data.
    :param pages: Maximum number of pages to retrieve.
    :return: The archive as stored after the run. When the merge changed
        nothing the save is skipped (#242); the content is the archive as read.
    :raises BackupError: If the backup fails.
    :raises StorageReadError: If the archive cannot be read or holds no rows.
    :raises CatchupStalled: If Ambient holds readings past a gap the run
        could not cross.
    :raises StorageWriteError: If the save fails, or (`ArchiveLossError`)
        would drop a stored reading or blank a stored value.
    """
    # Always back up before mutating the current archive
    sj.backup_archive(bucket=bucket_name, mac=device["macAddress"])

    archive_df = _load_archive(device, bucket_name)
    log_peak_rss("archive_load", logger)

    fetched = fetch_delta(device, int(archive_df["dateutc"].iloc[-1]), pages)
    return _merge_and_save(device, bucket_name, archive_df, fetched)


def fetch_delta(device: Dict, newest_ms: int, pages: int) -> pd.DataFrame:
    """
    Fetch what Ambient holds after the archive's newest reading.

    :param device: The station's `/devices` entry.
    :param newest_ms: The archive's newest `dateutc`, epoch ms.
    :param pages: Maximum number of pages to retrieve.
    :return: The fetched readings, raw; empty when there are none.
    :raises CatchupStalled: If Ambient holds readings past a gap the fetch
        could not cross.
    """
    newest_only = pd.DataFrame({"dateutc": [newest_ms]})
    fetched = awn.fetch_since_last_archive(device, newest_only, sleep=True, pages=pages)
    log_peak_rss("ambient_fetch", logger)
    return fetched


def save_fetched(device: Dict, bucket_name: str, fetched: pd.DataFrame) -> pd.DataFrame:
    """
    Back up the archive, merge in readings already fetched, save.

    The refresh run's archive step (#271): `fetch_delta` ran earlier, in the
    run's Ambient phase. Every failure raises, as `update_archive`'s do.

    :param device: The station's `/devices` entry.
    :param bucket_name: S3 bucket to read/write data.
    :param fetched: What `fetch_delta` returned for this station.
    :return: The archive as stored after the run (the archive as read when
        the merge changed nothing and the save was skipped, #242).
    :raises BackupError: If the backup fails.
    :raises StorageReadError: If the archive cannot be read or holds no rows.
    :raises StorageWriteError: If the save fails, or (`ArchiveLossError`)
        would drop a stored reading or blank a stored value.
    """
    sj.backup_archive(bucket=bucket_name, mac=device["macAddress"])

    archive_df = _load_archive(device, bucket_name)
    log_peak_rss("archive_load", logger)

    return _merge_and_save(device, bucket_name, archive_df, fetched)


def _merge_and_save(
    device: Dict, bucket_name: str, archive_df: pd.DataFrame, fetched: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge fetched readings into the loaded archive and save it when changed.

    :param device: The station's `/devices` entry.
    :param bucket_name: S3 bucket to write to.
    :param archive_df: The archive as loaded; the save's `stored` guard.
    :param fetched: The fetched readings, raw.
    :return: The merged archive.
    :raises StorageWriteError: If the save fails, or (`ArchiveLossError`)
        would drop a stored reading or blank a stored value.
    """
    # Ambient wins for each value it sends; the routine re-fetch of the last
    # readings shows up as updated values, never as a failure.
    new_archive_df, report = archive_merge.merge(archive_df, fetched)
    logger.info(f"Archive merge: {report.describe()}")
    _log_archive_range(new_archive_df, "Updated archive")
    logger.info(f"Total records after update: {len(new_archive_df)}")
    log_peak_rss("archive_merge", logger)

    if not report.changed:
        # Nothing to write: skip the 13 MB rewrite, and its window for a
        # partial write, every 5 minutes (#242). The backup above still ran.
        logger.info("Archive save skipped: the merge changed nothing.")
        log_peak_rss("archive_save", logger)
        return new_archive_df

    # Save updated archive to S3, newest first whatever came back (#133)
    sj.save_archive(
        new_archive_df, bucket_name, device["macAddress"], stored=archive_df
    )
    logger.info("Updated archive saved.")
    log_peak_rss("archive_save", logger)
    return new_archive_df


def _load_archive(device: Dict, bucket_name: str) -> pd.DataFrame:
    """
    Load the station's archive from storage and normalize it.

    Strict (#160): a run that saves after this must never start from a
    failed read, so it reads storage directly (no dev-mode local cache) and
    raises rather than returning an empty frame. The archive returned is
    after `normalize` drops invalid and duplicate rows (`sj.load_archive`);
    it is the archive the save is guarded by (`save_archive`'s `stored`).

    :param device: The station's `/devices` entry.
    :param bucket_name: S3 bucket to read from.
    :return: The normalized archive, at least one row.
    :raises StorageReadError: If the archive is missing, unreadable, or has
        no valid rows.
    """
    archive_df = sj.load_archive(bucket_name, device["macAddress"])

    _log_archive_range(archive_df, "Loaded archive")
    logger.info(f"Total records in archive: {len(archive_df)}")
    return archive_df


def main(bucket_name: str, dry_run: bool, pages: int, repair_archive: bool) -> None:
    """
    Update archived weather data or repair archive schema.

    :param bucket_name: S3 bucket to read/write data.
    :param dry_run: If True, simulate actions without modifying data.
    :param pages: Maximum number of pages to retrieve during update.
    :param repair_archive: If True, validate/fix archive only (optionally dry-run).
    :raises ArchiveLockHeld: If a run that writes (any but `--dry-run`)
        finds another archive writer holding the lock (#244); raised before
        anything is read or written.
    :raises StorageReadError: If the station inventory is missing or cannot
        be read; raised before anything is written.
    :raises StationNotFoundError: If the default station (the station
        inventory's first) is not on the Ambient account; raised before
        anything is written.
    :raises BackupError: If a repair cannot take its repair backup; the
        archive is not saved then.
    :raises ReadbackFailed: If a repair's read-back of the saved archive
        fails a check or cannot be read; the log names the repair backup.
    """
    with writer_lock(saves=not dry_run):
        device = get_station_device(resolve_station(bucket_name))
        mac = device.get("macAddress")
        name = device.get("info", {}).get("name", "Unnamed Device")
        logger.info(f"Selected device: {name} ({mac})")

        if dry_run and not repair_archive:
            logger.info(
                "Running in dry-run mode (no write). "
                "Fetching latest page only for visibility."
            )
            df = awn.get_device_history_to_date(device)
            # Normalize for consistent printing
            df = normalize(df)
            _log_archive_range(df, "Dry-run latest page")
            df.info()
            return

        if not repair_archive:
            update_archive(device, bucket_name, pages)
            return

        archive_df = _load_archive(device, bucket_name)

        fixed, m = _repair_archive_df(archive_df)
        logger.info(
            "Archive repair: %s -> %s rows (dropped=%s), dtype %s -> %s, "
            "range %s .. %s",
            m.get("before_len"),
            m.get("after_len"),
            m.get("dropped"),
            m.get("before_dtype"),
            m.get("after_dtype"),
            m.get("min"),
            m.get("max"),
        )
        derived = m.get("derived_columns")
        if derived:
            verb = "would strip" if dry_run else "strips"
            logger.info(f"Archive repair: {verb} derived columns {', '.join(derived)}")
        else:
            logger.info("Archive repair: no derived columns stored")
        if dry_run:
            logger.info("Dry-run: NOT saving repaired archive.")
            return

        # Persist the repaired archive, newest first (#133), and exit. An
        # explicit repair: the one catchup save that may drop or blank (#190).
        # It backs the live archive up first, every time, to a repair backup
        # of its own (#169, #422), then reads the saved archive back raw.
        backup_key = sj.backup_and_save_history(
            fixed, device, bucket_name, stored=archive_df, repair=True
        )
        logger.info(f"Rollback point: s3://{bucket_name}/{backup_key}")
        logger.info("Repaired archive saved.")
        _check_repair_readback(bucket_name, mac, fixed, backup_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update weather data archive.")
    parser.add_argument("--bucket", type=str, default="lookout", help="S3 bucket name")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without making any changes"
    )
    parser.add_argument(
        "--pages", type=int, default=20, help="Max number of pages (288 records each)"
    )
    parser.add_argument(
        "--repair-archive",
        action="store_true",
        help=(
            "Validate/fix archive (dateutc), de-dupe, sort, strip derived "
            "columns. Can be used with --dry-run."
        ),
    )
    args = parser.parse_args()

    try:
        main(args.bucket, args.dry_run, args.pages, args.repair_archive)
    except ArchiveLockHeld as e:
        logger.error(f"{e}; catchup did nothing.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Unhandled exception in catchup: {e}")
        sys.exit(1)
