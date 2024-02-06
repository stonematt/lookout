"""
awn_controller.py: Manages the integration of weather data from the Ambient Weather
Network (AWN) with S3-compatible storage, using AmbientAPI for data retrieval and
storj_df_s3.py for data persistence. This module focuses on retrieving, updating,
and maintaining historical weather data. It ensures seamless data flow and consistency
across local and cloud storage, crucial for the backend operations of a Streamlit-based
weather application.

Functions:
- get_archive: Retrieves weather data archives from the local filesystem.
- load_archive_for_device: Loads archived weather data from S3 storage.
- get_device_history_to_date: Fetches historical data up to a specified date.
- get_history_since_last_archive: Combines archived with recent to provide a full history.
- combine_df: Merges two dataframes, removing duplicates and sorting by date.

Example Usage:
In a Streamlit app, this module can be used to load historical weather data for a
specific device, update it with the latest data from AWN, and then store the updated
data back to S3 storage.
"""

# %%
from ambient_api.ambientapi import AmbientAPI
import time
import pandas as pd
import streamlit as st
import logging
import storj_df_s3 as sj
from datetime import datetime, timedelta


# %%
AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

api = AmbientAPI(
    # log_level="INFO",
    AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
    AMBIENT_API_KEY=AMBIENT_API_KEY,
    AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
)

sec_in_hour = 3600 * 1000


# Module-level logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_archive(archive_file: str) -> pd.DataFrame:
    """
    Retrieves a DataFrame from a Parquet file stored in the local filesystem.

    :param archive_file: str - The file path of the Parquet file to be read.
    :return: DataFrame - The DataFrame containing the archived weather data.
    """
    logging.info(f"Load archive: {archive_file}")
    try:
        return pd.read_parquet(archive_file)
    except FileNotFoundError:
        logging.error(f"File not found: {archive_file}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Read error: {archive_file}, {e}")
        return pd.DataFrame()


# todo: keep this one
# %% Load archive from s3
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
    logging.info(f"Load from S3: {bucket}/{key}")
    try:
        return sj.get_df_from_s3(bucket, key, file_type=file_type)
    except Exception as e:
        logging.error(f"S3 load error: {bucket}/{key}, {e}")
        return pd.DataFrame()


def _df_column_to_datetime(df: pd.DataFrame, column: str, tz: str) -> None:
    """
    Converts and adjusts a DataFrame column to a specified timezone.

    :param df: DataFrame with the column.
    :param column: Column name for datetime conversion.
    :param tz: Target timezone for conversion.
    """
    try:
        df[column] = pd.to_datetime(df[column]).dt.tz_convert(tz)
        logging.debug(f"Converted '{column}' to '{tz}'")
    except KeyError:
        logging.error(f"Column not found: '{column}'")
    except Exception as e:
        logging.error(f"Conversion error: {e}")
        raise e


# todo: keep this one
# %%
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
        logging.info(f"Fetch history: {device.mac_address}, Params: {params}")
        df = pd.json_normalize(device.get_data(**params))
        df.sort_values(by="dateutc", inplace=True)

        # Convert 'date' column to local time
        dt_columns = ["date", "lastRain"]
        for c in dt_columns:
            _df_column_to_datetime(df, c, device.last_data.get("tz"))

        return df
    except Exception as e:
        logging.error(f"Fetch error: {e}")
        return pd.DataFrame()


# %%
# todo: keep this one (forward)
def get_device_history_from_date(device, start_date, limit=288):
    """
    Fetches a page of data for the device starting from the specified date.

    :param device: The device object to fetch data for.
    :param start_date: The datetime to start fetching data from.
    :param limit: The number of records to fetch.
    :return: A DataFrame with the fetched data.
    """
    # Calculate end_date considering the service returns data points before this date
    # and ensuring overlap by subtracting 15 minutes from the calculated end date
    # to get (limit-3) new data points and 3 overlapping points.
    end_date = start_date + timedelta(minutes=(limit - 3) * 5)
    end_date_timestamp = int(end_date.timestamp() * 1000)  # Convert to milliseconds

    new_data = get_device_history_to_date(
        device, end_date=end_date_timestamp, limit=limit
    )

    if new_data.empty:
        logging.info("No more data to fetch.")
    else:
        logging.info(
            f"Retrieved {len(new_data)} records. "
            f"Range: {new_data['dateutc'].min()} - {new_data['dateutc'].max()}"
        )

    return new_data


def get_history_since_last_archive(
    device, archive_df, limit=250, pages=10, sleep=False
):
    """
    Sequentially retrieves device history from the last archive date forward in time,
    stopping after a specified number of pages or when no more data is available.

    :param device: Device object for fetching data.
    :param archive_df: DataFrame of archived data.
    :param limit: int - Max records to fetch per call.
    :param pages: int - Number of pages to fetch moving forward in time.
    :param sleep: bool - Enables delay between API calls if True, default False.
    :return: DataFrame with updated device history.
    """
    try:
        interim_df = pd.DataFrame()

        # Initialize the start date from the last date in the archive
        last_date = pd.to_datetime(archive_df["dateutc"].max(), unit="ms")

        for page in range(pages):
            if sleep:
                time.sleep(1)  # Optional sleep between fetches if enabled

            # Fetch data using the helper function
            new_data = get_device_history_from_date(device, last_date, limit)

            if new_data.empty:
                logging.info(f"No more data to fetch after page {page + 1}.")
                break  # Exit if no data was returned

            # Update interim_df with new_data
            interim_df = combine_df(interim_df, new_data)
            logging.info(
                f"Page: {page + 1} "
                f"Range: ({interim_df['date'].min().strftime('%y-%m-%d %H:%M')}) - "
                f"({interim_df['date'].max().strftime('%y-%m-%d %H:%M')})"
            )

            # Update last_date for the next fetch
            last_date = pd.to_datetime(new_data["dateutc"].max(), unit="ms")

        # Combine the archived data with the newly fetched data
        full_history_df = combine_df(archive_df, interim_df)
        return full_history_df

    except Exception as e:
        logging.error(f"Error in get_history_since_last_archive_modified: {e}")
        return archive_df


# todo: keep this one
# %% combine DFs
def combine_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merges two DataFrames, removes duplicates, and sorts by 'dateutc'.
    Raises an exception if the merge fails.

    :param df1: First DataFrame to merge.
    :param df2: Second DataFrame to merge.
    :return: Merged and sorted DataFrame.
    :raises: Exception if DataFrames cannot be merged.
    """
    try:
        result_df = (
            pd.concat([df1, df2], ignore_index=True)
            .drop_duplicates()
            .sort_values(by="dateutc", ascending=False)
            .reset_index(drop=True)
        )
        logging.info(f"DataFrames combined: {result_df.shape[0]} records")
        return result_df
    except Exception as e:
        logging.error(f"Error combining DataFrames: {e}")
        raise Exception(f"Failed to combine DataFrames: {e}")


# %%
def main():
    api = AmbientAPI(
        log_level="WARN",
        AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
        AMBIENT_API_KEY=AMBIENT_API_KEY,
        AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
    )
    # %%

    devices = api.get_devices()

    device = devices[0]

    df = load_archive_for_device(device, "lookout", "parquet")
    print(df.date.min())
    print(df.date.max())

    # df = get_history_since_last_archive(device, df)
    print(df.date.min())
    print(df.date.max())

    return True


if __name__ == "__main__":
    main()
# %%
