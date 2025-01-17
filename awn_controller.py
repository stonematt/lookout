"""
awn_controller.py: Integrates weather data from the Ambient Weather Network (AWN)
with S3-compatible storage. Utilizes AmbientAPI for data retrieval and storj_df_s3.py
for data persistence. This module focuses on retrieving, updating, and maintaining
historical weather data, ensuring seamless data flow and consistency across storage
solutions, crucial for the backend operations of a Streamlit-based weather application.

Functions:
- get_archive: Retrieves weather data archives from the local filesystem.
- load_archive_for_device: Loads device-specific archived data from S3 storage.
- get_device_history_to_date: Fetches historical data for a device with end_date.
- get_device_history_from_date: Fetches historical data for a device with start_date.
- get_history_since_last_archive: Add new data to archive forward by n pages.
- combine_df: Merges two DataFrames, removing duplicates and sorting by 'dateutc'.

Helper Functions:
- validate_archive: Validates if the archive DataFrame is usable.
- fetch_device_data: Fetches historical data for a device.
- validate_new_data: Validates new data fetched for completeness and freshness.
- combine_interim_data: Combines interim DataFrame with new data.
- update_last_date: Updates the `last_date` for the next fetch.
- log_interim_progress: Logs progress during data fetching.
"""

from ambient_api.ambientapi import AmbientAPI
from datetime import datetime, timedelta
import pandas as pd
import time
import streamlit as st
import storj_df_s3 as sj
from log_util import app_logger

logger = app_logger(__name__)

# Constants
AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]
sec_in_hour = 3600 * 1000

# Core Functions


def get_archive(archive_file: str) -> pd.DataFrame:
    """
    Retrieves a DataFrame from a Parquet file stored in the local filesystem.

    :param archive_file: str - The file path of the Parquet file to be read.
    :return: DataFrame - The DataFrame containing the archived weather data.
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
    device, bucket: str, file_type: str = "parquet"
) -> pd.DataFrame:
    """
    Loads device-specific weather data from S3 into a DataFrame.

    :param device: Device object for data identification.
    :param bucket: str - Name of the S3 bucket.
    :param file_type: str - File type ('json' or 'parquet').
    :return: DataFrame containing the device's archived data.
    """
    key = f"{device.mac_address}.{file_type}"
    logger.info(f"Load from S3: {bucket}/{key}")
    try:
        return sj.get_df_from_s3(bucket, key, file_type=file_type)
    except Exception as e:
        logger.error(f"S3 load error: {bucket}/{key}, {e}")
        return pd.DataFrame()


def get_device_history_to_date(device, end_date=None, limit=288) -> pd.DataFrame:
    """
    Fetches historical data for a device up to a specified date.

    :param device: The device to fetch data for.
    :param end_date: End date for data retrieval, defaults to None.
    :param limit: Max records to retrieve, defaults to 288.
    :return: DataFrame of device history data.
    """
    try:
        params = (
            {"limit": limit, "end_date": end_date} if end_date else {"limit": limit}
        )
        logger.info(f"Fetch history: {device.mac_address}, Params: {params}")
        df = pd.json_normalize(device.get_data(**params))

        if df.empty:
            logger.debug("Empty response, no new data")
            return pd.DataFrame()

        # Sort the data by 'dateutc'
        df.sort_values(by="dateutc", inplace=True)

        # Convert 'date' column to local time
        dt_columns = ["date", "lastRain"]
        for c in dt_columns:
            _df_column_to_datetime(df, c, device.last_data.get("tz"))

        return df
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return pd.DataFrame()


def get_device_history_from_date(device, start_date, limit=288):
    """
    Fetches a page of data for the device starting from the specified date.

    :param device: The device object to fetch data for.
    :param start_date: The datetime to start fetching data from.
    :param limit: The number of records to fetch.
    :return: A DataFrame with the fetched data.
    """
    current_time = datetime.now()
    end_date = start_date + timedelta(minutes=(limit - 3) * 5)

    if end_date > current_time:
        end_date = current_time

    end_date_timestamp = int(end_date.timestamp() * 1000)

    return get_device_history_to_date(device, end_date=end_date_timestamp, limit=limit)


def get_history_since_last_archive(
    device, archive_df, limit=250, pages=10, sleep=False
):
    """
    Sequentially retrieves device history from the last archive date forward in time,
    handling data gaps and ensuring only new data is added.

    :param device: Device object for fetching data.
    :param archive_df: DataFrame of archived data.
    :param limit: int - Max records to fetch per call.
    :param pages: int - Number of pages to fetch moving forward in time.
    :param sleep: bool - Enables delay between API calls if True, default False.
    :return: DataFrame with updated device history.
    """
    if not validate_archive(archive_df):
        return archive_df

    interim_df = pd.DataFrame()
    last_date = pd.to_datetime(archive_df["dateutc"].max(), unit="ms")
    gap_attempts = 0

    for page in range(pages):
        if sleep:
            time.sleep(1)

        new_data, fetch_successful = fetch_device_data(device, last_date, limit)
        if not fetch_successful:
            break

        if not validate_new_data(new_data, interim_df, gap_attempts, last_date, limit):
            gap_attempts += 1
            if gap_attempts >= 3:
                logger.info("Maximum gap attempts reached. Exiting.")
                break
            continue

        gap_attempts = 0
        interim_df = combine_interim_data(interim_df, new_data)
        last_date = update_last_date(new_data)

        log_interim_progress(page, pages, interim_df)

    return combine_full_history(archive_df, interim_df)


# Helper Functions


def validate_archive(archive_df):
    """Validate that the archive DataFrame is usable."""
    if archive_df.empty or "dateutc" not in archive_df.columns:
        logger.error("archive_df is empty or missing 'dateutc'.")
        return False
    return True


def fetch_device_data(device, last_date, limit):
    """Fetch historical data for a device starting from a given date."""
    try:
        new_data = get_device_history_from_date(device, last_date, limit)
        if new_data.empty:
            logger.debug("No new data fetched.")
            return pd.DataFrame(), False
        return new_data, True
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame(), False


def validate_new_data(new_data, interim_df, gap_attempts, last_date, limit):
    """Validate new data fetched for completeness and freshness."""
    if "dateutc" not in new_data.columns:
        logger.error("New data is missing 'dateutc'.")
        return False

    if not _is_data_new(interim_df, new_data):
        logger.info(f"Seeking ahead: {gap_attempts + 1}/3")
        last_date = _calculate_next_start_date(last_date, gap_attempts, limit)
        return False
    return True


def combine_interim_data(interim_df, new_data):
    """Combine interim data with newly fetched data."""
    return combine_df(interim_df, new_data)


def update_last_date(new_data):
    """Update the last_date for the next fetch."""
    return pd.to_datetime(new_data["dateutc"].max(), unit="ms")


def log_interim_progress(page, pages, interim_df):
    """Log progress during interim data fetching."""
    logger.info(
        f"Interim Page: {page}/{pages} "
        f"Range: ({interim_df['date'].min().strftime('%y-%m-%d %H:%M')}) - "
        f"({interim_df['date'].max().strftime('%y-%m-%d %H:%M')})"
    )


def combine_full_history(archive_df, interim_df):
    """Combine the archive with interim data and log the range."""
    full_history_df = combine_df(archive_df, interim_df)
    logger.info(
        f"Full History Range: "
        f"({full_history_df['date'].min().strftime('%y-%m-%d %H:%M')}) - "
        f"({full_history_df['date'].max().strftime('%y-%m-%d %H:%M')})"
    )
    return full_history_df


def combine_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merges two DataFrames, removes duplicates, and sorts by 'dateutc'.

    :param df1: First DataFrame to merge.
    :param df2: Second DataFrame to merge.
    :return: Merged and sorted DataFrame.
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
        raise Exception(f"Failed to combine DataFrames: {e}")


def _calculate_next_start_date(current_max_date, gap_attempts, limit):
    """
    Internally calculates the next start date for fetching, considering gap attempts.

    :param current_max_date: The maximum date in the current dataset.
    :param gap_attempts: The number of consecutive gap attempts made.
    :param limit: The number of records per fetch.
    :return: The next start date for data fetching.
    """
    return current_max_date + timedelta(minutes=5 * limit * gap_attempts)


def _is_data_new(interim_df, new_data):
    """
    Internally determines if the fetched data contains new information.

    :param interim_df: The current interim DataFrame with previously fetched data.
    :param new_data: The newly fetched data to compare.
    :return: Boolean indicating whether the new data contains new information.
    """
    if interim_df.empty:
        return True
    return new_data["dateutc"].max() > interim_df["dateutc"].max()


def _df_column_to_datetime(df: pd.DataFrame, column: str, tz: str) -> None:
    """
    Converts and adjusts a DataFrame column to a specified timezone.

    :param df: DataFrame with the column.
    :param column: Column name for datetime conversion.
    :param tz: Target timezone for conversion.
    """
    try:
        df[column] = pd.to_datetime(df[column]).dt.tz_convert(tz)
        logger.debug(f"Converted '{column}' to '{tz}'")
    except KeyError:
        logger.error(f"Column not found: '{column}'")
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        raise e


def main():
    api = AmbientAPI(
        log_level="WARN",
        AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
        AMBIENT_API_KEY=AMBIENT_API_KEY,
        AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
    )

    devices = api.get_devices()
    device = devices[0]

    df = load_archive_for_device(device, "lookout", "parquet")
    print(df.date.min())
    print(df.date.max())

    df = get_history_since_last_archive(device, df)
    print(df.date.min())
    print(df.date.max())


if __name__ == "__main__":
    main()
