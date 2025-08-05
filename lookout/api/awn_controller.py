"""
awn_controller.py: Orchestrates historical weather data integration from the Ambient Weather Network (AWN)
with S3-compatible object storage. Replaces legacy AmbientAPI with ambient_client for improved modularity.

Relies on:
- `ambient_client` for fetching AWN data via HTTP.
- `storj.py` for S3-compatible persistence.
- Streamlit for secrets management.
- Logging for operational observability.

This module ensures reliable, incremental retrieval and synchronization of weather data archives,
including deduplication, paging, and gap detection—supporting consistent backend storage for
a Streamlit-based weather dashboard.

Functions:
- get_archive: Load local Parquet archive.
- load_archive_for_device: Load archived device data from S3.
- get_device_history_to_date: Fetch records ending at a given timestamp.
- get_device_history_from_date: Fetch records starting from a timestamp.
- get_history_since_last_archive: Incrementally extend archive forward.
- combine_df: Merge and deduplicate DataFrames by 'dateutc'.

Helper Functions:
- validate_archive: Ensure archive is non-empty and valid.
- fetch_device_data: Page data from AWN from a starting point.
- is_fresh_data: Check if new data is structurally sound and non-overlapping.
- combine_interim_data: Accumulate paged results.
- update_last_date: Move forward in time for next fetch.
- log_interim_progress: Log fetch progress in paged loop.
- combine_full_history: Final merge of archive and updates.
"""

import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

import lookout.storage.storj as sj
from lookout.api.ambient_client import get_device_history, get_devices
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Constants
AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]
sec_in_hour = 3600 * 1000

# Core Functions


def update_session_data(device, hist_df=None, limit=250, pages=10):
    """
    Update session with latest historical data and reset session counter.

    :param device: Object representing the device.
    :param hist_df: DataFrame of current historical data, defaults to session state history.
    :param limit: int - Max records to fetch per call, default 250.
    :param pages: int - Number of pages to fetch, default 10.
    :return: None.
    """
    try:
        # Use provided or session state history
        current_df = (
            hist_df
            if hist_df is not None
            else st.session_state.get("history_df", pd.DataFrame())
        )

        # Fetch updated history
        updated_df = get_history_since_last_archive(
            device, current_df, limit=limit, pages=pages
        )

        # Update session state
        st.session_state["history_df"] = updated_df
        st.session_state["history_max_dateutc"] = int(
            st.session_state["history_df"]["dateutc"].max().timestamp() * 1000
        )

        logger.info("Session data updated successfully.")
    except Exception as e:
        logger.error(f"Failed to update session data: {e}")
        st.error("An error occurred while updating session data. Please try again.")


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
    device: dict, bucket: str, file_type: str = "parquet"
) -> pd.DataFrame:
    """
    Loads device-specific weather data from S3 into a DataFrame.

    :param device: Device dictionary with MAC address and metadata.
    :param bucket: Name of the S3 bucket.
    :param file_type: File type ('json' or 'parquet').
    :return: DataFrame containing the device's archived data.
    """
    mac = device.get("macAddress")
    key = f"{mac}.{file_type}"
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
    mac = device.get("macAddress")
    if not isinstance(mac, str):
        logger.error("Device is missing a valid 'macAddress'")
        return pd.DataFrame()

    try:
        params = {"limit": limit}
        if end_date:
            params["end_date"] = end_date

        logger.info(f"Fetch history: {mac}, Params: {params}")
        df = get_device_history(mac, **params)

        if df.empty:
            logger.debug("Empty response, no new data")
            return pd.DataFrame()

        df.sort_values(by="dateutc", inplace=True)

        for col in ["date", "lastRain"]:
            _df_column_to_datetime(df, col, device.get("lastData", {}).get("tz"))

        return df

    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return pd.DataFrame()


def get_device_history_from_date(device, start_date, limit=288) -> pd.DataFrame:
    """
    Fetches a page of data for the device starting from the specified date.

    :param device: The device dictionary to fetch data for.
    :param start_date: The datetime to start fetching data from.
    :param limit: The number of records to fetch.
    :return: A DataFrame with the fetched data.
    """
    mac = device.get("macAddress")
    if not isinstance(mac, str):
        logger.error("Device is missing a valid 'macAddress'")
        return pd.DataFrame()

    current_time = datetime.now()
    end_date = start_date + timedelta(minutes=(limit - 3) * 5)

    if end_date > current_time:
        end_date = current_time

    end_date_ts = int(end_date.timestamp() * 1000)
    return get_device_history_to_date(device, end_date=end_date_ts, limit=limit)


def seek_over_time_gap(
    last_date: datetime, gap_attempts: int, limit: int, max_attempts: int = 3
) -> tuple[datetime, int, bool]:
    """
    Handles logic for gap-based time seeking during history fetch.

    :param last_date: datetime - Current max date seen.
    :param gap_attempts: int - Current number of failed fetch/validation attempts.
    :param limit: int - Records per page used for offsetting.
    :param max_attempts: int - Maximum retries before exiting.
    :return: (next_date, new_gap_attempts, should_exit)
    """
    gap_attempts += 1
    if gap_attempts >= max_attempts:
        logger.info("Maximum gap attempts reached. Exiting.")
        return last_date, gap_attempts, True

    logger.info(f"Seeking ahead: {gap_attempts}/{max_attempts}")
    next_date = _calculate_next_start_date(last_date, gap_attempts, limit)
    return next_date, gap_attempts, False


def get_history_since_last_archive(
    device: dict,
    archive_df: pd.DataFrame,
    limit: int = 250,
    pages: int = 10,
    sleep: bool = False,
) -> pd.DataFrame:
    """
    Retrieves device history from the last archive forward, avoiding duplicate fetches,
    handling gaps, and skipping future timestamps.

    :param device: dict - Device metadata for fetch.
    :param archive_df: pd.DataFrame - Historical archive to append to.
    :param limit: int - Records per fetch.
    :param pages: int - Max pages to pull.
    :param sleep: bool - If True, sleep 1s between calls.
    :return: pd.DataFrame - Combined new and archived data.
    """
    if not validate_archive(archive_df):
        return archive_df

    interim_df = pd.DataFrame()
    last_date = pd.to_datetime(archive_df["dateutc"].to_numpy().max(), unit="ms")
    gap_attempts = 0

    for page in range(pages):
        if sleep:
            time.sleep(1)

        # Stop if we're trying to fetch data from the future
        if last_date.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc):
            logger.info("Next fetch timestamp is in the future. Stopping early.")
            break

        # Attempt to fetch new records since last known timestamp
        new_data, fetch_successful = fetch_device_data(device, last_date, limit)

        # If fetch fails or returns no data
        if not fetch_successful or new_data.empty:
            logger.debug("No new data fetched.")
            last_date, gap_attempts, exit_flag = seek_over_time_gap(
                last_date, gap_attempts, limit
            )
            if exit_flag:
                break
            continue

        # Check if the new data is valid and advances the timeline
        is_fresh = is_fresh_data(new_data, interim_df)
        if not is_fresh:
            last_date, gap_attempts, exit_flag = seek_over_time_gap(
                last_date, gap_attempts, limit
            )
            if exit_flag:
                break
            continue

        # If valid data is found, append it and update reference timestamp
        gap_attempts = 0
        interim_df = combine_interim_data(interim_df, new_data)
        last_date = update_last_date(new_data)
        log_interim_progress(page + 1, pages, interim_df)

    # Merge the new records with the existing archive
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
            return pd.DataFrame(), False
        return new_data, True
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame(), False


def is_fresh_data(
    new_data: pd.DataFrame,
    interim_df: pd.DataFrame,
) -> bool:
    """
    Validates the new data is structurally complete and non-overlapping.

    :param new_data: Newly fetched DataFrame.
    :param interim_df: Accumulated interim data.
    :return: True if data is fresh and non-duplicate; False otherwise.
    """
    if "dateutc" not in new_data.columns:
        logger.error("New data is missing 'dateutc'.")
        return False

    if not _is_data_new(interim_df, new_data):
        logger.debug("Fetched data is not new or overlaps with interim.")
        return False

    return True


def combine_interim_data(interim_df, new_data):
    """Combine interim data with newly fetched data."""
    return combine_df(interim_df, new_data)


def update_last_date(new_data: pd.DataFrame) -> datetime:
    """
    Update the last_date for the next fetch.

    :param new_data: DataFrame containing the new data.
    :return: The max 'dateutc' as a datetime object.
    """
    return pd.to_datetime(new_data["dateutc"].max(), unit="ms").to_pydatetime()


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
    Merge two DataFrames on 'dateutc', keeping the last entry for each timestamp.
    """
    try:
        df = pd.concat([df1, df2], ignore_index=True)
        df["dateutc"] = pd.to_datetime(df["dateutc"], unit="ms", errors="coerce")
        df = df.dropna(subset=["dateutc"])
        df.sort_values("dateutc", ascending=True, inplace=True)
        return (
            df.drop_duplicates(subset="dateutc", keep="last")
            .sort_values("dateutc", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        logger.error(f"Error combining DataFrames: {e}")
        raise


def _calculate_next_start_date(
    current_max_date: datetime, gap_attempts: int, limit: int
) -> datetime:
    """
    Calculates the next start date for fetching, offset by paging attempt.

    :param current_max_date: The last known good datetime to start from.
    :param gap_attempts: Number of consecutive non-advancing fetches.
    :param limit: Number of records to fetch per page.
    :return: New datetime for the next fetch attempt.
    """
    return current_max_date + timedelta(minutes=5 * limit * gap_attempts)


def _is_data_new(interim_df, new_data):
    """
    Determines if the newly fetched data is truly new.

    :param interim_df: Existing fetched data (interim).
    :param new_data: Just-fetched data to check.
    :return: bool - Whether the new data is considered "new".
    """
    logger.debug("=== _IS_DATA_NEW ===")

    if interim_df.empty:
        logger.debug("Interim is empty. Data considered new.")
        return True

    if new_data.empty:
        logger.debug("New data is empty. Not considered new.")
        return False

    if "dateutc" not in new_data.columns or "dateutc" not in interim_df.columns:
        logger.error("Missing 'dateutc' column in data.")
        return False

    try:
        new_max = pd.to_datetime(new_data["dateutc"].values.max(), unit="ms", utc=True)
        interim_max = pd.to_datetime(
            interim_df["dateutc"].values.max(), unit="ms", utc=True
        )

        logger.debug(
            f"new_max={new_max}, interim_max={interim_max}, delta={new_max - interim_max}"
        )
        is_newer = new_max > interim_max

        if not is_newer:
            overlap = set(new_data["dateutc"]).intersection(interim_df["dateutc"])
            logger.debug(f"Overlapping timestamps: {len(overlap)}")

        return is_newer

    except Exception as e:
        logger.error(f"Error in _is_data_new: {e}")
        return False


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


def main() -> None:
    """
    Diagnostic routine:
    - Verifies connectivity to Ambient Weather Network.
    - Lists available devices and their names.
    - Loads existing archive from S3.
    - Prints archive date range (if available).
    - Fetches a 10-record sample using ambient_client.
    - Prints one raw sample record directly from device JSON.
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

            archive_df = load_archive_for_device(device, "lookout", "parquet")
            if archive_df.empty:
                logger.info("  📂 Archive: empty")
            else:
                logger.info(
                    f"  📂 Archive Range: {archive_df.date.min()} to {archive_df.date.max()}"
                )

            df = get_device_history(mac, limit=10)
            if df.empty:
                logger.warning("  ⚠️  No recent data retrieved.")
            else:
                latest = df["dateutc"].max()
                logger.info(
                    f"  ✅ Retrieved {len(df)} records. Latest timestamp: {latest}"
                )

            # Direct raw access
            raw = device.get("data", [])[:1]
            if raw:
                print("  🧾 Raw sample:\n", json.dumps(raw[0], indent=2))

            break  # Remove to check all devices

    except Exception as e:
        logger.exception(f"❌ Exception during diagnostic: {e}")


if __name__ == "__main__":
    main()
