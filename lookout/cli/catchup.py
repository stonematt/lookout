"""
catchup.py: Synchronize new Ambient Weather data with archived records in S3.

This module pulls data since the last known timestamp in the archive and updates
S3 storage with any newly retrieved data. Supports a dry-run mode for simulation
and offline testing.

Usage:
    python catchup.py --bucket lookout [--dry-run]
"""

import argparse

import streamlit as st

import lookout.api.awn_controller as awn
import lookout.storage.storj as sj
from lookout.api.ambient_client import get_devices
from lookout.utils.log_util import app_logger

logger = app_logger(__name__, log_file="catchup.log")

AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]


def main(bucket_name: str, dry_run: bool, pages: int) -> None:
    """
    Main execution function for updating archived weather data.

    :param bucket_name: S3 bucket to read/write data.
    :param dry_run: If True, simulate actions without modifying data.
    :param pages: Maximum number of pages (288 records each) to retrieve.
    """

    devices = get_devices()
    if not devices:
        logger.error("No devices found.")
        return

    device = devices[0]
    mac = device.get("macAddress")
    name = device.get("info", {}).get("name", "Unnamed Device")
    logger.info(f"Selected device: {name} ({mac})")

    if dry_run:
        logger.info("Running in dry-run mode.")
        df = awn.get_device_history_to_date(device)
        df.info()
        return

    # Backup current archive before changes
    sj.backup_data(bucket=bucket_name, prefix=mac, dry_run=dry_run)

    # Load existing archive
    archive_df = awn.load_archive_for_device(device, bucket=bucket_name)

    logger.info(
        f"Range: ({archive_df['date'].min().strftime('%y-%m-%d %H:%M')}) - "
        f"({archive_df['date'].max().strftime('%y-%m-%d %H:%M')})"
    )
    logger.info(f"Total records in archive: {len(archive_df)}")

    # Fetch new records since last date
    new_archive_df = awn.get_history_since_last_archive(
        device, archive_df, sleep=True, pages=pages
    )
    logger.info(f"Total records after update: {len(new_archive_df)}")

    # Save updated archive to S3
    key = f"{mac}.parquet"
    sj.save_df_to_s3(new_archive_df, bucket_name, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update weather data archive.")
    parser.add_argument("--bucket", type=str, default="lookout", help="S3 bucket name")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without making any changes"
    )
    parser.add_argument(
        "--pages", type=int, default=20, help="Max number of pages (288 records each)"
    )

    args = parser.parse_args()

    try:
        main(args.bucket, args.dry_run, args.pages)

    except Exception as e:
        logger.exception(f"❌ Unhandled exception in catchup: {e}")
