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
from typing import Optional

import pandas as pd
import streamlit as st

from lookout.utils.trend_utils import (
    calculate_temperature_trend,
    calculate_barometer_trend,
)

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

        # Calculate trends for latest data point
        if not updated_df.empty:
            latest_row = updated_df.iloc[0]  # Most recent (archive is reverse sorted)
            temp_trend = calculate_temperature_trend(updated_df, latest_row["tempf"])
            barom_trend = calculate_barometer_trend(
                updated_df, latest_row["baromrelin"]
            )

            # Store trends in session state
            st.session_state["temp_trend"] = temp_trend
            st.session_state["barom_trend"] = barom_trend

        # Only update session state if data actually changed
        if not updated_df.equals(current_df):
            st.session_state["history_df"] = updated_df
            st.session_state["history_max_dateutc"] = st.session_state["history_df"][
                "dateutc"
            ].max()

            # Force garbage collection to free memory from old DataFrame
            import gc

            gc.collect()

            logger.info("Session data updated successfully.")
        else:
            logger.debug("No data changes detected, skipping session state update")
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
        human = None
        if "end_date" in params and isinstance(params["end_date"], (int, float)):
            try:
                human = pd.to_datetime(int(params["end_date"]), unit="ms", utc=True)
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid end_date value: {params['end_date']} ({e})")
                human = None
        logger.debug(f"to_date:params={params}, human_end={human}")
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


def get_device_history_from_date(
    device, start_date, limit=288, end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Fetches a page of data for the device starting from the specified date.

    :param device: The device dictionary to fetch data for.
    :param start_date: The datetime to start fetching data from.
    :param limit: The number of records to fetch.
    :param end_date: Optional datetime to stop fetching at (exclusive).
    :return: A DataFrame with the fetched data.
    """
    mac = device["macAddress"]
    if not isinstance(mac, str):
        logger.error("Device is missing a valid 'macAddress'")
        return pd.DataFrame()

    # Ensure start is timezone-aware
    if getattr(start_date, "tzinfo", None) is None:
        start_date = start_date.replace(tzinfo=timezone.utc)

    current_time = datetime.now(timezone.utc)

    # Resolve end_date with fallback and clip to current time
    end_date = (
        pd.to_datetime(end_date, utc=True)
        if end_date
        else start_date + timedelta(minutes=(limit - 3) * 5)
    )
    end_date = min(end_date, current_time)
    end_ms = int(end_date.timestamp() * 1000)

    logger.debug(
        "from_date: start=%s end=%s (tz=%s) now=%s (tz=%s)",
        start_date,
        end_date,
        start_date.tzinfo,
        current_time,
        current_time.tzinfo,
    )

    return get_device_history_to_date(device, end_date=end_ms, limit=limit)


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


def _utc_now():
    return datetime.now(timezone.utc)


def _to_utc(dt: datetime) -> datetime:
    """Return an aware UTC datetime for comparison."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _advance_cursor_from_page(new_data: pd.DataFrame, now_utc: datetime) -> datetime:
    """
    Advance the cursor using page's max dateutc (UTC) + 5 minutes, capped at now.
    """
    page_max_utc = pd.to_datetime(
        new_data["dateutc"].max(), unit="ms", utc=True
    ).to_pydatetime()
    return min(page_max_utc + timedelta(minutes=5), now_utc)


def _fetch_and_validate_page(
    device: dict, cursor_dt: datetime, limit: int, interim_df: pd.DataFrame
):
    """
    Fetch one page and decide whether it progresses the timeline.
    Returns: (new_data_df, progressed_bool)
    """
    new_data, ok = fetch_device_data(device, cursor_dt, limit)
    if not ok or new_data.empty:
        logger.debug("No new data fetched.")
        return pd.DataFrame(), False

    if not is_fresh_data(new_data, interim_df):
        logger.debug("Fetched data did not advance timeline (not fresh).")
        return pd.DataFrame(), False

    return new_data, True


def _first_page_catchup_to_now(device: dict, limit: int) -> pd.DataFrame:
    """
    If we hit the 'future' guard before getting any interim data,
    pull one page ending at 'now' so we at least land on the latest snapshot.
    """
    end_ms = int(_utc_now().timestamp() * 1000)
    logger.info(
        f"First-page future guard tripped; pulling one page ending at now ({end_ms})."
    )
    df = get_device_history_to_date(device, end_date=end_ms, limit=limit)
    if df.empty:
        logger.info("First-page catch-up returned no data.")
    return df


def get_history_since_last_archive(
    device: dict,
    archive_df: pd.DataFrame,
    limit: int = 250,
    pages: int = 10,
    sleep: bool = False,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Retrieves device history from the last archive forward, avoiding duplicate fetches,
    handling gaps, and skipping future timestamps.
    """
    if not validate_archive(archive_df):
        return archive_df

    interim_df = pd.DataFrame()

    # Always work in UTC for consistency
    last_date = pd.to_datetime(
        archive_df["dateutc"].to_numpy().max(), unit="ms", utc=True
    )

    # Define a fixed loop cap based on end_date (if provided) or now
    if end_date:
        if isinstance(end_date, (int, float)):
            end_date = pd.to_datetime(end_date, unit="ms", utc=True)
        else:
            end_date = _to_utc(pd.to_datetime(end_date, utc=True))
        now_utc = end_date
    else:
        now_utc = _utc_now()

    gap_attempts = 0

    for page in range(pages):
        if sleep:
            time.sleep(1)

        last_date_utc = _to_utc(last_date)

        # Stop if we've hit the future or past end bound
        if _should_stop_for_future(last_date_utc, now_utc):
            if interim_df.empty:
                first_page = _first_page_catchup_to_now(device, limit)
                if not first_page.empty:
                    interim_df = combine_interim_data(interim_df, first_page)
                    log_page_quality(
                        interim_df, f"Interim after page {page+1} (catch-up)"
                    )
            break

        page_label = f"Page {page+1}/{pages} fetched"
        new_data, progressed = fetch_device_data(
            device, last_date_utc, limit, page_label
        )

        if not progressed:
            last_date, gap_attempts, exit_flag = seek_over_time_gap(
                last_date, gap_attempts, limit
            )
            if exit_flag:
                break
            continue

        # Progressing page: merge, log, update cursor
        gap_attempts = 0
        interim_df = combine_interim_data(interim_df, new_data)
        log_page_quality(interim_df, f"Interim after page {page+1}")
        last_date = _advance_cursor_from_page(new_data, now_utc)
        logger.debug(f"cursor advance: page_max->next_cursor={last_date} now={now_utc}")
        log_interim_progress(page + 1, pages, interim_df)

    return combine_full_history(archive_df, interim_df)


# Helper Functions


def _should_stop_for_future(next_dt, now_utc) -> bool:
    """
    True if the next fetch timestamp would be in the future.
    Normalizes naive datetimes to UTC for a fair comparison.
    """
    next_utc = (
        next_dt
        if getattr(next_dt, "tzinfo", None)
        else next_dt.replace(tzinfo=timezone.utc)
    )
    if next_utc >= now_utc:
        logger.info(
            f"Next fetch timestamp {next_utc} is in the future vs now {now_utc}. Stopping early."
        )
        return True
    return False


def fill_archive_gap(device, history_df, start, end):
    """Fetch missing data for a specified gap and append it to the in-memory history."""
    logger.info(f"*** Attempting to fill gap from {start} to {end} ***")

    # Normalize input
    working_history_df = history_df.copy()
    working_history_df["dateutc"] = pd.to_datetime(
        history_df["dateutc"], unit="ms", utc=True
    )
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)

    if start >= end:
        logger.warning("Invalid gap: start >= end")
        return history_df

    # Trim to before the gap
    archive_df = working_history_df[working_history_df["dateutc"] < start]

    if archive_df.empty:
        logger.warning("No archive data before gap start; can't determine baseline.")
        return history_df

    # Fetch new records to fill the gap
    new_data_df = get_history_since_last_archive(
        device, archive_df, end_date=end, pages=20
    )

    # Merge new data back into full archive
    full_combined = combine_full_history(history_df, new_data_df)

    if len(full_combined) == len(history_df):
        logger.info("No new data found. Marking gap as skipped.")
        if "skipped_gaps" not in st.session_state:
            st.session_state["skipped_gaps"] = []
        st.session_state["skipped_gaps"].append({"start": start, "end": end})
        return history_df

    return full_combined


def validate_archive(archive_df):
    """Validate that the archive DataFrame is usable."""
    if archive_df.empty or "dateutc" not in archive_df.columns:
        logger.error("archive_df is empty or missing 'dateutc'.")
        return False
    return True


def fetch_device_data(device, last_date, limit, label="Fetched data"):
    """Fetch historical data for a device starting from a given date."""
    try:
        new_data = get_device_history_from_date(device, last_date, limit)
        # after a successful fetch (before freshness check)
        log_page_quality(new_data, label)

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


def combine_interim_data(
    interim_df: pd.DataFrame, new_data: pd.DataFrame
) -> pd.DataFrame:
    before = len(interim_df)
    combined = combine_df(interim_df, new_data)
    added = len(combined) - before
    logger.info(f"combine: +{added} rows (interim total={len(combined)})")
    return combined


def update_last_date(new_data: pd.DataFrame) -> datetime:
    """
    Update the last_date for the next fetch.

    :param new_data: DataFrame containing the new data.
    :return: The max 'dateutc' as a datetime object (UTC aware).
    """
    max_ts = int(new_data["dateutc"].max())
    max_dt = pd.to_datetime(max_ts, unit="ms", utc=True).to_pydatetime()

    logger.debug(
        f"update_last_date: max_ts={max_ts}, max_dt={max_dt} (tz={max_dt.tzinfo})"
    )

    return max_dt


def log_interim_progress(page, pages, interim_df):
    """Log progress during interim data fetching."""
    logger.info(
        f"Interim Page: {page}/{pages} "
        f"Range: ({interim_df['date'].min().strftime('%y-%m-%d %H:%M')}) - "
        f"({interim_df['date'].max().strftime('%y-%m-%d %H:%M')})"
    )


def combine_full_history(
    archive_df: pd.DataFrame, interim_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine the archive with interim data, log added rows and final UTC range.
    """
    before = len(archive_df)

    if interim_df is None or interim_df.empty:
        combined = archive_df
        added = 0
    else:
        combined = combine_df(archive_df, interim_df)
        added = len(combined) - before

    # Compute range from dateutc (ms) in UTC
    # (do not assume presence/correctness of any local 'date' column)
    dt = pd.to_datetime(combined["dateutc"], unit="ms", utc=True, errors="coerce")
    dt = dt.dropna()
    if dt.empty:
        logger.info(
            f"Full History Range: (n/a) - (n/a); +{added} rows (total={len(combined)})"
        )
        return combined

    start = dt.min().strftime("%y-%m-%d %H:%M")
    end = dt.max().strftime("%y-%m-%d %H:%M")

    logger.info(
        f"Full History Range: ({start}) - ({end}); +{added} rows (total={len(combined)})"
    )

    return combined


def combine_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two DataFrames on 'dateutc', keeping the last entry per timestamp.
    Operates in UTC datetimes for correctness, then returns dateutc as int64 ms.
    """
    try:
        df = pd.concat([df1, df2], ignore_index=True)

        # Normalize to UTC datetimes for safe dedupe
        df["dateutc"] = pd.to_datetime(
            df["dateutc"], unit="ms", utc=True, errors="coerce"
        )
        df = df.dropna(subset=["dateutc"])

        # Dedupe and sort (desc to match existing callers)
        df = (
            df.drop_duplicates(subset="dateutc", keep="last")
            .sort_values("dateutc", ascending=False)
            .reset_index(drop=True)
        )

        # Convert back to int64 milliseconds for storage schema (avoid .view() warnings)
        df["dateutc"] = (df["dateutc"].astype("int64") // 1_000_000).astype("int64")

        return df
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


def log_page_quality(df: pd.DataFrame, label: str) -> None:
    """
    Emit quick stats for a fetched/combined page to spot gaps or drops.
    """
    if df is None or df.empty:
        logger.info(f"{label}: empty page")
        return

    try:
        ts = pd.to_datetime(df["dateutc"], unit="ms", utc=True, errors="coerce")
        ts = ts.dropna()
        if ts.empty:
            logger.info(f"{label}: no valid timestamps")
            return

        n = len(df)
        n_dup = df["dateutc"].duplicated().sum()
        first_ts = ts.min()
        last_ts = ts.max()
        span = last_ts - first_ts

        # check expected 5‑min cadence (not strict—just a hint)
        cadence = ts.sort_values().diff().dropna().value_counts().head(3).to_dict()

        logger.info(
            f"{label}: n={n}, dup={n_dup}, range=({first_ts})–({last_ts}), span={span}, top_steps={cadence}"
        )
    except Exception as e:
        logger.debug(f"{label}: page-quality log failed: {e}")


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
