"""
catchup.py: Synchronize new Ambient Weather data with archived records in S3.

This module pulls data since the last known timestamp in the archive and updates
S3 storage with any newly retrieved data. Supports a dry-run mode for simulation
and offline testing. Also supports archive repair (schema normalization) with
--repair-archive and optional --dry-run.

Usage:
    python catchup.py --bucket lookout [--dry-run] [--pages 20] [--repair-archive]
"""

import argparse
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

import lookout.api.awn_controller as awn
import lookout.storage.storj as sj
from lookout.api.ambient_client import get_devices
from lookout.utils.dateutc import normalize
from lookout.utils.log_util import app_logger

logger = app_logger(__name__, log_file="catchup.log")

AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]


# ---------- archive utilities ----------


def _repair_archive_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    CLI-shaped wrapper around `dateutc.normalize` that produces an
    operator-readable metrics dict for the `--repair-archive` report.

    Returns (fixed_df, metrics dict). The actual validation/cast/dedupe/
    sort work is delegated to `normalize`; this function only adapts
    the result into a metrics shape.
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

    fixed = normalize(df)

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

    logger.info(
        f"{label}: Range: ({lo.strftime('%y-%m-%d %H:%M')}) - ({hi.strftime('%y-%m-%d %H:%M')})"
    )


# ---------- main flow ----------


def main(bucket_name: str, dry_run: bool, pages: int, repair_archive: bool) -> None:
    """
    Update archived weather data or repair archive schema.

    :param bucket_name: S3 bucket to read/write data.
    :param dry_run: If True, simulate actions without modifying data.
    :param pages: Maximum number of pages to retrieve during update.
    :param repair_archive: If True, validate/fix archive only (optionally dry-run).
    """
    devices = get_devices()
    if not devices:
        logger.error("No devices found.")
        return

    device = devices[0]
    mac = device.get("macAddress")
    name = device.get("info", {}).get("name", "Unnamed Device")
    logger.info(f"Selected device: {name} ({mac})")

    if dry_run and not repair_archive:
        logger.info(
            "Running in dry-run mode (no write). Fetching latest page only for visibility."
        )
        df = awn.get_device_history_to_date(device)
        # Normalize for consistent printing
        df = normalize(df)
        _log_archive_range(df, "Dry-run latest page")
        df.info()
        return

    # Always back up before mutating the current archive (skip in dry-run)
    sj.backup_data(bucket=bucket_name, prefix=mac, dry_run=dry_run)

    # Load existing archive
    archive_df = awn.load_archive_for_device(device, bucket=bucket_name)
    archive_df = normalize(archive_df)

    _log_archive_range(archive_df, "Loaded archive")
    logger.info(f"Total records in archive: {len(archive_df)}")

    if repair_archive:
        fixed, m = _repair_archive_df(archive_df)
        logger.info(
            "Archive repair: %s -> %s rows (dropped=%s), dtype %s -> %s, range %s .. %s",
            m.get("before_len"),
            m.get("after_len"),
            m.get("dropped"),
            m.get("before_dtype"),
            m.get("after_dtype"),
            m.get("min"),
            m.get("max"),
        )
        if dry_run:
            logger.info("Dry-run: NOT saving repaired archive.")
            return

        # Persist the repaired archive and exit
        key = f"{mac}.parquet"
        sj.save_df_to_s3(fixed, bucket_name, key)
        logger.info("Repaired archive saved.")
        return

    # Normal catchup/update flow
    new_archive_df = awn.get_history_since_last_archive(
        device, archive_df, sleep=True, pages=pages
    )
    _log_archive_range(new_archive_df, "Updated archive")
    logger.info(f"Total records after update: {len(new_archive_df)}")

    if dry_run:
        logger.info("Dry-run: NOT saving updated archive.")
        return

    # Save updated archive to S3
    key = f"{mac}.parquet"
    sj.save_df_to_s3(new_archive_df, bucket_name, key)
    logger.info("Updated archive saved.")


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
        help="Validate/fix archive (dateutc), de-dupe, sort. Can be used with --dry-run.",
    )
    args = parser.parse_args()

    try:
        main(args.bucket, args.dry_run, args.pages, args.repair_archive)
    except Exception as e:
        logger.exception(f"❌ Unhandled exception in catchup: {e}")
