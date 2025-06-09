"""
awn_controller.py: Integrates weather data from the Ambient Weather Network (AWN)
with S3-compatible storage. Utilizes ambient_client for data retrieval and
storj_df_s3.py for data persistence.

This module focuses on retrieving, updating, and maintaining historical weather data,
ensuring seamless data flow and consistency across storage solutions—critical for
backend operations in the Streamlit-based weather dashboard.

Functions:
- get_archive()
- load_archive_for_device()
- get_device_history_to_date()
- get_device_history_from_date()
- get_history_since_last_archive()
- combine_df()
- main()
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import storj_df_s3 as sj
from ambient_client import get_device_history, get_devices
from log_util import app_logger

logger = app_logger(__name__)

# Secrets for API access
AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

sec_in_hour = 3600 * 1000


def get_archive(archive_file: str) -> pd.DataFrame:
    """
    Load archived weather data from a Parquet file.

    :param archive_file: Path to the archive Parquet file.
    :return: Loaded DataFrame or empty if not found or errored.
    """
    logger.info(f"Load archive: {archive_file}")
    try:
        return pd.read_parquet(archive_file)
    except FileNotFoundError:
        logger.error(f"File not found: {archive_file}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Read error: {archive_file}, {e}")
        return pd.DataFrame()


def load_archive_for_device(
    device: dict, bucket: str, file_type: str = "parquet"
) -> pd.DataFrame:
    """
    Load a device's archived data from S3 storage.

    :param device: Device dictionary.
    :param bucket: S3 bucket name.
    :param file_type: File format ('parquet' or 'json').
    :return: DataFrame containing archived data.
    """
    key = f"{device['macAddress']}.{file_type}"
    logger.info(f"Load from S3: {bucket}/{key}")
    try:
        return sj.get_df_from_s3(bucket, key, file_type=file_type)
    except Exception as e:
        logger.error(f"S3 load error: {bucket}/{key}, {e}")
        return pd.DataFrame()


def get_device_history_to_date(device: dict, end_date=None, limit=288) -> pd.DataFrame:
    """
    Fetch historical data for a device up to a specified date.

    :param device: Device dictionary.
    :param end_date: Optional end timestamp (milliseconds).
    :param limit: Max records to retrieve.
    :return: DataFrame of weather history.
    """
    try:
        params = {"limit": limit}
        if end_date:
            params["end_date"] = end_date

        logger.info(f"Fetch history: {device['macAddress']}, Params: {params}")
        df = get_device_history(device["macAddress"], **params)

        if df.empty:
            logger.debug("Empty response, no new data")
            return pd.DataFrame()

        df.sort_values(by="dateutc", inplace=True)

        # Convert relevant time columns
        for col in ["date", "lastRain"]:
            _df_column_to_datetime(df, col, device.get("lastData", {}).get("tz", "UTC"))

        return df
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return pd.DataFrame()


def get_device_history_from_date(
    device: dict, start_date: datetime, limit=288
) -> pd.DataFrame:
    """
    Fetch a slice of historical data for a device starting from a given datetime.

    :param device: Device dictionary.
    :param start_date: Start time for data collection.
    :param limit: Number of records to attempt.
    :return: DataFrame of fetched data.
    """
    current_time = datetime.now()
    end_date = min(start_date + timedelta(minutes=(limit - 3) * 5), current_time)
    end_date_timestamp = int(end_date.timestamp() * 1000)
    return get_device_history_to_date(device, end_date=end_date_timestamp, limit=limit)


def get_history_since_last_archive(
    device: dict, archive_df: pd.DataFrame, limit=250, pages=10, sleep=False
) -> pd.DataFrame:
    """
    Walk forward in time from the latest archive and gather new data in pages.

    :param device: Device dictionary.
    :param archive_df: Existing archive DataFrame.
    :param limit: Number of records per request.
    :param pages: Number of fetch iterations.
    :param sleep: Optional delay between fetches.
    :return: Combined DataFrame with new records appended.
    """
    if archive_df.empty or "dateutc" not in archive_df.columns:
        logger.error("archive_df is empty or missing 'dateutc'.")
        return archive_df

    interim_df = pd.DataFrame()
    last_date = pd.to_datetime(archive_df["dateutc"].max(), unit="ms").to_pydatetime()
    gap_attempts = 0

    for page in range(pages):
        if sleep:
            time.sleep(1)

        new_data, ok = fetch_device_data(device, last_date, limit)
        if not ok:
            break

        if not validate_new_data(new_data, interim_df, gap_attempts):
            gap_attempts += 1
            if gap_attempts >= 3:
                logger.info("Maximum gap attempts reached. Exiting.")
                break
            continue

        gap_attempts = 0
        interim_df = combine_df(interim_df, new_data)
        last_date = pd.to_datetime(new_data["dateutc"].max(), unit="ms").to_pydatetime()
        log_interim_progress(page, pages, interim_df)

    return combine_full_history(archive_df, interim_df)


def fetch_device_data(device: dict, last_date: datetime, limit: int):
    """
    Wrapper to retrieve new device data from a starting timestamp.

    :param device: Device dictionary.
    :param last_date: Starting datetime for next fetch.
    :param limit: Number of records to request.
    :return: Tuple of (dataframe, success flag).
    """
    try:
        new_data = get_device_history_from_date(device, last_date, limit)
        if new_data.empty:
            logger.debug("No new data fetched.")
            return pd.DataFrame(), False
        return new_data, True
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame(), False


def validate_new_data(
    new_data: pd.DataFrame, interim_df: pd.DataFrame, gap_attempts: int
) -> bool:
    """
    Check if new data is valid and genuinely newer than current interim data.

    :param new_data: New data just fetched.
    :param interim_df: Interim data so far.
    :param gap_attempts: Count of consecutive missing fetches.
    :return: True if the data should be accepted.
    """
    if "dateutc" not in new_data.columns:
        logger.error("New data is missing 'dateutc'.")
        return False
    if not _is_data_new(interim_df, new_data):
        logger.info(f"Seeking ahead: {gap_attempts + 1}/3")
        return False
    return True


def combine_full_history(
    archive_df: pd.DataFrame, interim_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine archived and newly retrieved data, logging the full time range.

    :param archive_df: Original data from archive.
    :param interim_df: New data just retrieved.
    :return: Combined and deduplicated DataFrame.
    """
    full = combine_df(archive_df, interim_df)
    logger.info(
        f"Full History Range: "
        f"({full['date'].min().strftime('%y-%m-%d %H:%M')}) - "
        f"({full['date'].max().strftime('%y-%m-%d %H:%M')})"
    )
    return full


def combine_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate two DataFrames and remove duplicates, sorted by date.

    :param df1: First DataFrame.
    :param df2: Second DataFrame.
    :return: Combined DataFrame.
    """
    try:
        return (
            pd.concat([df1, df2], ignore_index=True)
            .drop_duplicates()
            .sort_values(by="dateutc", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        logger.error(f"Error combining DataFrames: {e}")
        raise


def _is_data_new(interim_df: pd.DataFrame, new_data: pd.DataFrame) -> bool:
    """
    Determine if the incoming data includes timestamps newer than interim.

    :param interim_df: Data fetched so far.
    :param new_data: Newly fetched data.
    :return: Boolean indicating novelty.
    """
    if interim_df.empty:
        return True
    return new_data["dateutc"].max() > interim_df["dateutc"].max()


def _df_column_to_datetime(df: pd.DataFrame, column: str, tz: str) -> None:
    """
    Convert a DataFrame column to timezone-aware datetime.

    :param df: The DataFrame.
    :param column: The column name to convert.
    :param tz: Target timezone string.
    """
    try:
        df[column] = pd.to_datetime(df[column]).dt.tz_localize("UTC").dt.tz_convert(tz)
        logger.debug(f"Converted '{column}' to '{tz}'")
    except KeyError:
        logger.warning(f"Column not found: {column}")
    except Exception as e:
        logger.error(f"Conversion error in column '{column}': {e}")


def log_interim_progress(page: int, pages: int, df: pd.DataFrame) -> None:
    """
    Log page progress while fetching device history in steps.

    :param page: Current page number.
    :param pages: Total pages configured.
    :param df: Interim data frame at this step.
    """
    logger.info(
        f"Interim Page: {page}/{pages} "
        f"Range: ({df['date'].min().strftime('%y-%m-%d %H:%M')}) - "
        f"({df['date'].max().strftime('%y-%m-%d %H:%M')})"
    )


def main() -> None:
    """
    Diagnostic routine for device discovery and basic data fetch validation.

    Useful for debugging API credentials and testing connectivity before a full run.
    """
    logger = app_logger("awn_main")

    try:
        devices = get_devices()
        if not devices:
            logger.error("❌ No devices found or no connection to Ambient Network.")
            return

        logger.info(f"✅ Connected. Found {len(devices)} device(s).")

        for device in devices:
            mac = device.get("macAddress")
            name = device.get("info", {}).get("name", "Unnamed Device")

            if not mac:
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

    except Exception as e:
        logger.exception(f"❌ Exception during diagnostic: {e}")


if __name__ == "__main__":
    main()
