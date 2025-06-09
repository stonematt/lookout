"""
awn_controller.py: Manages retrieval and archival of historical weather data
from Ambient Weather devices.

Integrates with ambient_client for API access and storj_df_s3 for archive storage.
Supports paging through historical data, deduplication, and detailed logging.

Functions:
- get_devices(): Discover devices from Ambient API.
- get_device_history_to_date(): Fetch recent records for preview or dry-run.
- fetch_device_data(): Request one page of device data before a cutoff date.
- load_archive_for_device(): Load existing archive from S3.
- get_history_since_last_archive(): Extend archive with newly available data.
- main(): Optional diagnostic/test runner for direct use.
"""

import time
from datetime import datetime

import pandas as pd

import storj_df_s3 as sj
from ambient_client import get_device_history, get_devices
from log_util import app_logger

logger = app_logger(__name__)


def get_device_history_to_date(device: dict) -> pd.DataFrame:
    """
    Fetch up to 10 of the most recent records for a device.

    :param device: Ambient device dictionary with 'macAddress'.
    :return: DataFrame containing up to 10 recent weather readings.
    """
    mac = device.get("macAddress")
    if not isinstance(mac, str):
        logger.warning("Device missing valid MAC address.")
        return pd.DataFrame()
    return get_device_history(mac, limit=10)


def fetch_device_data(
    device: dict, last_date: datetime, limit: int = 288
) -> tuple[pd.DataFrame, bool]:
    """
    Fetch a page of historical data ending at a specified datetime.

    :param device: Ambient device dictionary.
    :param last_date: Timestamp to fetch data before (inclusive).
    :param limit: Number of records to request (max 288).
    :return: Tuple of (dataframe, success flag).
    """
    mac = device.get("macAddress")
    if not isinstance(mac, str):
        logger.error("Device has no valid MAC address.")
        return pd.DataFrame(), False

    end_timestamp = int(last_date.timestamp() * 1000)
    logger.info(
        f"Fetch history: {mac}, Params: {{'limit': {limit}, 'end_date': {end_timestamp}}}"
    )
    df = get_device_history(mac, limit=limit, end_date=end_timestamp)
    return df, not df.empty


def log_interim_progress(page: int, pages: int, df: pd.DataFrame) -> None:
    """
    Log the current page status and date range of the interim DataFrame.

    :param page: Zero-based page number.
    :param pages: Total number of pages expected.
    :param df: Interim DataFrame containing accumulated data.
    """
    logger.info(
        f"Interim Page: {page + 1}/{pages} "
        f"Range: ({df['date'].min().strftime('%y-%m-%d %H:%M')}) - "
        f"({df['date'].max().strftime('%y-%m-%d %H:%M')})"
    )


def load_archive_for_device(device: dict, bucket: str) -> pd.DataFrame:
    """
    Load historical data archive for a device from S3.

    :param device: Ambient device dictionary.
    :param bucket: S3 bucket containing the archive file.
    :return: DataFrame containing the loaded archive data.
    """
    mac = device.get("macAddress")
    if not isinstance(mac, str):
        logger.error("Cannot load archive: device missing valid MAC address.")
        return pd.DataFrame()
    key = f"{mac}.parquet"
    logger.info(f"Load from S3: {bucket}/{key}")
    return sj.get_df_from_s3(bucket, key, file_type="parquet")


def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize types for date fields in the weather history DataFrame.

    :param df: DataFrame containing historical weather records.
    :return: DataFrame with standardized datetime fields.
    """
    df["dateutc"] = pd.to_numeric(df["dateutc"], errors="coerce")

    if "lastRain" in df.columns:
        # Always coerce to UTC first, then convert to archive TZ
        df["lastRain"] = pd.to_datetime(
            df["lastRain"], errors="coerce", utc=True
        ).dt.tz_convert("America/Los_Angeles")

    df["date"] = pd.to_datetime(
        df["dateutc"], unit="ms", errors="coerce", utc=True
    ).dt.tz_convert("America/Los_Angeles")

    return df


def get_history_since_last_archive(
    device: dict, archive_df: pd.DataFrame, pages: int = 20, sleep: bool = True
) -> pd.DataFrame:
    """
    Fetch and merge data since the last recorded timestamp in the archive.

    :param device: Ambient device dictionary.
    :param archive_df: Previously archived weather data.
    :param pages: Number of 288-record pages to attempt.
    :param sleep: Whether to wait 1 second between API calls.
    :return: Updated DataFrame including newly fetched records.
    """
    interim_df = archive_df.copy()
    gap_attempts = 0

    for page in range(pages):
        if sleep:
            time.sleep(1)

        last_date = pd.to_datetime(
            interim_df["dateutc"].max(), unit="ms"
        ).to_pydatetime()
        new_data, ok = fetch_device_data(device, last_date)

        if not ok:
            gap_attempts += 1
            if gap_attempts > 3:
                logger.warning("Too many empty responses. Stopping early.")
                break
            continue

        interim_df = pd.concat(
            [interim_df, new_data], ignore_index=True
        ).drop_duplicates()
        interim_df = normalize_history_df(interim_df)
        log_interim_progress(page, pages, interim_df)

    interim_df.sort_values(by="dateutc", inplace=True)
    interim_df.reset_index(drop=True, inplace=True)

    logger.info(
        f"Full History Range: ({interim_df['date'].min().strftime('%y-%m-%d %H:%M')}) - "
        f"({interim_df['date'].max().strftime('%y-%m-%d %H:%M')})"
    )

    return interim_df


def main() -> None:
    """
    Run a diagnostic device query and fetch the most recent data slice.
    """
    logger = app_logger("awn_main")

    try:
        devices = get_devices()
        if not devices:
            logger.error("❌ No devices found or no connection to Ambient Network.")
            return

        logger.info(f"✅ Connected. Found {len(devices)} device(s).")

        for device in devices:
            name = device.get("info", {}).get("name", "Unnamed Device")
            mac = device.get("macAddress")

            if not isinstance(mac, str):
                logger.warning(f"⚠️  Skipping device '{name}' — missing MAC address.")
                continue

            logger.info(f"📡 Device: {name} ({mac})")

            df = get_device_history(mac, limit=10)
            if df.empty:
                logger.warning("  ⚠️  No recent data retrieved.")
            else:
                latest = df["dateutc"].max()
                logger.info(
                    f"  ✅ Retrieved {len(df)} records. Latest timestamp: {latest}"
                )

            time.sleep(1)

    except Exception as e:
        logger.exception(f"❌ Exception during diagnostic: {e}")


if __name__ == "__main__":
    main()
