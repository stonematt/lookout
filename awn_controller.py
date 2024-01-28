"""
awn_controller.py: Manages the integration of weather data from the Ambient Weather
Network (AWN) with S3-compatible storage, using AmbientAPI for data retrieval and
storj_df_s3.py for data persistence. This module focuses on retrieving, updating,
and maintaining historical weather data. It ensures seamless data flow and consistency
across local and cloud storage, crucial for the backend operations of a Streamlit-based
weather application.

Functions:
- get_archive: Retrieves weather data archives from the local filesystem.
- load_archive_for_device: Loads archived weather data for a  device from S3 storage.
- get_device_history_to_date: Fetches historical data for a device up to a specified date.
- get_history_since_last_archive: Combines archived data with recent data to provide a full history.
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
from datetime import datetime


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
# todo: keep this one
def get_history_since_last_archive(device, archive_df, limit=288, sleep=False):
    """
    Retrieves full device history by combining archived and recent data.

    :param device: Device object for fetching data.
    :param archive_df: DataFrame of archived data.
    :param limit: int - Max records to fetch per call, defaults to 288.
    :param sleep: bool - Enables delay between API calls if True.
    :return: DataFrame with combined device history.
    """
    try:
        update_message = st.empty()
        update_message.text("Updating and refreshing...")
        progress_message = st.empty()

        # approach:
        # - Make an interim_df of data between last archive and now
        # - add interim_df to archive and return full history df

        max_archive_date = archive_df["dateutc"].max()
        interim_df = pd.DataFrame()
        page = 0

        while True:
            progress_message.text(f"Getting page {page}")
            if sleep:
                time.sleep(1)

            end_date = (
                interim_df["dateutc"].min()
                if not interim_df.empty
                else int(datetime.now().timestamp() * 1000)
            )
            new_data = get_device_history_to_date(
                device, end_date=end_date, limit=limit
            )

            logging.info(f"Page {page}: {len(new_data)} new records")
            interim_df = combine_df(interim_df, new_data)
            page += 1

            if new_data.empty or new_data["dateutc"].min() < max_archive_date:
                break

        full_history_df = combine_df(archive_df, interim_df)
        return full_history_df

    except Exception as e:
        logging.error(f"Error in get_history_since_last_archive: {e}")
        return archive_df
    finally:
        update_message.empty()
        progress_message.empty()


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

    df = get_history_since_last_archive(device, df)
    print(df.date.min())
    print(df.date.max())

    return True


if __name__ == "__main__":
    main()
# %%
