import argparse
from ambient_api.ambientapi import AmbientAPI
import streamlit as st
import storj_df_s3 as sj
import awn_controller as awn
from log_util import app_logger

logger = app_logger(__name__, log_file="catchup.log")

AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

# Initialize Ambient API
api = AmbientAPI(
    # log_level="INFO",
    AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
    AMBIENT_API_KEY=AMBIENT_API_KEY,
    AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
)


def main(bucket_name, dry_run):

    devices = api.get_devices()

    if not devices:
        logger.error("No devices found.")
        return

    device = devices[0]
    logger.info(f"Selected device: {device.mac_address}")

    if dry_run:
        logger.info("Running in dry-run mode.")
        # Here, you would add logic to simulate actions without making changes
    else:
        # Backup process
        sj.backup_data(bucket=bucket_name, prefix=device.mac_address, dry_run=dry_run)

        # Get historical archive from S3
        archive_df = awn.load_archive_for_device(device, bucket_name)

        logger.info(
            f"Range: ({archive_df['date'].min().strftime('%y-%m-%d %H:%M')}) - "
            f"({archive_df['date'].max().strftime('%y-%m-%d %H:%M')})"
        )
        logger.info(f"Total records in archive: {len(archive_df)}")

        # Update process
        new_archive_df = awn.get_history_since_last_archive(
            device, archive_df, sleep=True, pages=20
        )
        logger.info(f"Total records after update: {len(new_archive_df)}")

        # Save back to S3
        key = f"{device.mac_address}.parquet"
        sj.save_df_to_s3(new_archive_df, bucket_name, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update weather data archive.")
    parser.add_argument("--bucket", type=str, default="lookout", help="S3 bucket name")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without making any changes"
    )

    args = parser.parse_args()

    main(args.bucket, args.dry_run)
