"""
catchup.py: Synchronize new Ambient Weather data with archived records in S3.

This module pulls data since the last known timestamp in the archive and updates
S3 storage with any newly retrieved data. Supports a dry-run mode for simulation
and offline testing. Also supports archive repair (schema normalization) with
--repair-archive and optional --dry-run; the repair also strips the derived
columns stored (#191). A repair that writes saves through the archive
store's replace (#430), which backs the archive up to a repair backup of its
own right before it saves, every time, and reads the saved archive back raw;
a dry run does neither. A run that writes takes the archive-writer lock
first (#244); `--dry-run` alone takes none.

Usage:
    python catchup.py --bucket lookout [--dry-run] [--pages 20] [--repair-archive]
"""

import argparse
import sys
from typing import Dict, Tuple

import pandas as pd

import lookout.api.awn_controller as awn
from lookout.api.ambient_client import get_station_device
from lookout.core import archive_merge
from lookout.storage import archive_store
from lookout.storage.station_inventory import resolve_station
from lookout.utils.archive_lock import ArchiveLockHeld, writer_lock
from lookout.utils.dateutc import normalize
from lookout.utils.log_util import app_logger
from lookout.utils.memory_utils import log_peak_rss

logger = app_logger(__name__, log_file="catchup.log")

# The reason a `--repair-archive` replace names, in its repair backup's key.
REPAIR_REASON = "normalize"


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

    span = f"({lo.strftime('%y-%m-%d %H:%M')}) - ({hi.strftime('%y-%m-%d %H:%M')})"
    logger.info(f"{label}: Range: {span}")


# ---------- main flow ----------


def update_archive(device: Dict, bucket_name: str, pages: int) -> None:
    """
    One catchup: open the archive, fetch what Ambient has since, merge it in.

    The merge goes through the archive store (`archive_store`), which takes
    the daily backup and saves only when the merge changed something (#242),
    and refuses a save that would drop a stored reading or blank a stored
    value. The refresh run does the same in two phases, so all its Ambient
    calls come before any station's archive is loaded (#271): `fetch_delta`
    first, then the archive store's merge. Every failure raises, so the run
    exits non-zero (#160).

    :param device: The station's `/devices` entry.
    :param bucket_name: S3 bucket to read/write data.
    :param pages: Maximum number of pages to retrieve.
    :raises StorageReadError: If the archive cannot be read or holds no rows.
    :raises CatchupStalled: If Ambient holds readings past a gap the run
        could not cross.
    :raises BackupError: If the backup before a save fails.
    :raises StorageWriteError: If the save fails, or (`ArchiveLossError`)
        would drop a stored reading or blank a stored value.
    """
    session = archive_store.open(bucket_name, device["macAddress"])
    fetched = fetch_delta(device, int(session.archive["dateutc"].max()), pages)
    session.merge(fetched)


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
    :raises StorageReadError: If the station inventory or the archive is
        missing or cannot be read; raised before anything is written.
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

        session = archive_store.open(bucket_name, mac)
        _log_archive_range(session.archive, "Loaded archive")
        logger.info(f"Total records in archive: {len(session.archive)}")

        fixed, m = _repair_archive_df(session.archive)
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

        # An explicit repair: the one catchup save that may drop or blank
        # (#190). The archive store backs the live archive up to a repair
        # backup first, every time, then reads the saved archive back raw
        # (#430). It rewrites the whole archive, so it names no windows.
        session.replace(fixed, REPAIR_REASON)
        logger.info("Repaired archive saved.")


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
