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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from lookout.config import DEFAULT_BUCKET
from lookout.secrets_loader import get_setting
from lookout.utils.dateutc import normalize

from lookout.utils.trend_utils import (
    calculate_temperature_trend,
    calculate_barometer_trend,
    newest_row,
)

import lookout.storage.storj as sj
from lookout.api.ambient_client import get_device_history
from lookout.core.rain_corrections import apply_corrections
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

# Constants
sec_in_hour = 3600 * 1000

# Local cache configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CACHE_FRESHNESS_HOURS = 48  # Refresh cache if older than 2 days

# Cache Management Functions


def get_local_archive_path(mac_address: str) -> Path:
    """
    Get the local parquet file path for a device MAC address.

    :param mac_address: Device MAC address
    :return: Path object for local parquet file
    """
    filename = f"{mac_address.replace(':', '-')}.parquet"
    return DATA_DIR / filename


def is_local_cache_fresh(file_path: Path) -> bool:
    """
    Check if local cache file is fresh (less than 2 days old).

    :param file_path: Path to local cache file
    :return: True if file exists and is fresh, False otherwise
    """
    if not file_path.exists():
        return False

    file_age_hours = (
        datetime.now(timezone.utc)
        - datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    ).total_seconds() / 3600

    return file_age_hours < CACHE_FRESHNESS_HOURS


def load_local_archive(mac_address: str) -> pd.DataFrame:
    """
    Load archive data from local parquet file.

    :param mac_address: Device MAC address
    :return: DataFrame containing local archive data
    """
    local_path = get_local_archive_path(mac_address)
    logger.info(f"Loading local archive: {local_path}")

    try:
        df = pd.read_parquet(local_path)
        logger.info(f"✅ Local cache loaded: {len(df)} records")
        return df
    except FileNotFoundError:
        logger.error(f"Local cache file not found: {local_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading local cache: {local_path}, {e}")
        return pd.DataFrame()


def update_local_cache(df: pd.DataFrame, mac_address: str) -> None:
    """
    Update local cache with fresh data from S3.

    :param df: DataFrame to cache locally
    :param mac_address: Device MAC address
    """
    if df.empty:
        logger.debug("Skipping cache update: empty DataFrame")
        return

    local_path = get_local_archive_path(mac_address)

    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    try:
        df.to_parquet(local_path, index=False)
        logger.info(f"✅ Local cache updated: {local_path} ({len(df)} records)")
    except Exception as e:
        logger.error(f"Failed to update local cache: {local_path}, {e}")


def is_dev_mode() -> bool:
    """
    Check if application is running in development mode.

    Read from the `DEV_MODE` setting (env var, the secrets file, then
    Streamlit's `[environment] dev`), so a run with no Streamlit secrets is
    simply not in dev mode.

    :return: True if dev mode is enabled
    """
    value = get_setting("DEV_MODE", "") or ""
    return value.strip().lower() in ("1", "true", "yes", "on")


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
    # The Streamlit UI's entry point; imported here so the refresh run, which
    # never calls it, never loads Streamlit.
    import streamlit as st

    try:
        # Use provided or session state history
        current_df = (
            hist_df
            if hist_df is not None
            else st.session_state.get("history_df", pd.DataFrame())
        )

        # Fetch updated history; the session history is already corrected,
        # so only the fetched rows get the rain corrections (#95)
        updated_df = extend_corrected(
            current_df,
            get_history_since_last_archive(
                device, current_df, limit=limit, pages=pages
            ),
            device["macAddress"],
        )

        # Calculate trends for latest data point
        if not updated_df.empty:
            latest_row = newest_row(updated_df)  # any row order (#133)
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
    In dev mode, uses local cache first if available and fresh.

    :param device: Device dictionary with MAC address and metadata.
    :param bucket: Name of the S3 bucket.
    :param file_type: File type ('json' or 'parquet').
    :return: DataFrame containing the device's archived data.
    """
    mac = device.get("macAddress")
    if not mac:
        logger.error("Device missing MAC address")
        return pd.DataFrame()

    # Type assertion for mypy
    mac_address = str(mac)
    key = f"{mac_address}.{file_type}"

    # Dev mode: try local cache first
    if is_dev_mode():
        local_path = get_local_archive_path(mac_address)

        if local_path.exists() and is_local_cache_fresh(local_path):
            logger.info(f"🏎️  Using fresh local cache: {local_path}")
            return load_local_archive(mac_address)
        elif local_path.exists():
            logger.info(
                f"📅 Local cache stale ({CACHE_FRESHNESS_HOURS}h threshold), will refresh"
            )
        else:
            logger.info(f"📂 No local cache found, will create")

    # Load from S3 (production or dev fallback)
    logger.info(f"Load from S3: {bucket}/{key}")
    try:
        df = sj.get_df_from_s3(bucket, key, file_type=file_type)

        # Update local cache in dev mode if we got fresh data
        if is_dev_mode() and not df.empty:
            update_local_cache(df, mac_address)
            logger.info(f"🔄 Dev mode: Updated local cache from S3")

        return df

    except Exception as e:
        logger.error(f"S3 load error: {bucket}/{key}, {e}")

        # Dev mode: try to use stale local cache as last resort
        if is_dev_mode():
            local_path = get_local_archive_path(mac_address)
            if local_path.exists():
                logger.warning(f"🆘 S3 failed, using stale local cache: {local_path}")
                return load_local_archive(mac_address)

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
    last_date: datetime, gap_attempts: int, limit: int, max_attempts: int = 10
) -> tuple[datetime, int, bool]:
    """
    Move the cursor past a page that brought nothing new, one page at a time.

    Each empty probe advances the cursor by exactly one page of 5-minute
    readings, whatever the attempt count, so the windows requested across
    an outage touch and no stretch goes unrequested (#132). The walk's
    reach is therefore `max_attempts` pages from where the gap began.

    :param last_date: datetime - The cursor the empty probe was made from.
    :param gap_attempts: int - Consecutive empty probes before this one.
    :param limit: int - Records per page; the step is `limit` readings.
    :param max_attempts: int - Empty probes after which the walk gives up.
    :return: (next_date, new_gap_attempts, should_exit)
    """
    gap_attempts += 1
    if gap_attempts >= max_attempts:
        logger.info("Maximum gap attempts reached. Exiting.")
        return last_date, gap_attempts, True

    logger.info(f"Seeking ahead: {gap_attempts}/{max_attempts}")
    next_date = _calculate_next_start_date(last_date, limit)
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

    The result is raw: the archive rows as handed, the fetched rows as Ambient
    reported them. Rain corrections are not applied; a reader applies them
    once, after loading (#95), so the archive written to storage stays raw.
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
        new_data, progressed = _fetch_and_validate_page(
            device, last_date_utc, limit, interim_df
        )
        if progressed:
            log_page_quality(new_data, page_label)

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

    # Raw, as Ambient reported it: rain corrections are applied on read,
    # never before a save (#95).
    return combine_full_history(archive_df, interim_df)


def extend_corrected(
    corrected: pd.DataFrame,
    extended: pd.DataFrame,
    mac_address: str,
    bucket: str = DEFAULT_BUCKET,
) -> pd.DataFrame:
    """
    Add the rows of `extended` that `corrected` lacks, rain corrected once.

    For a frame whose rain counters are already corrected (the Streamlit
    session history) and a raw extension of it (what
    `get_history_since_last_archive` or `fill_archive_gap` returns for it).
    Corrections are per row, by date, so correcting only the new rows gives
    the same counters as correcting a raw archive whole, and no row is
    corrected twice. Rows `corrected` already holds are kept as they are.
    Only for frames that are read, never saved: storage stays raw.

    :param corrected: The rain-corrected frame, `dateutc` in epoch ms.
    :param extended: A raw frame holding `corrected`'s rows and new ones.
    :param mac_address: The station, whose corrections catalog applies.
    :param bucket: The bucket holding the catalog.
    :return: `corrected` itself when nothing is new; else the union, newest
        first.
    """
    if extended.empty or "dateutc" not in extended.columns:
        return corrected
    if corrected.empty or "dateutc" not in corrected.columns:
        return apply_corrections(extended, mac_address, bucket)
    new_rows = extended[~extended["dateutc"].isin(corrected["dateutc"])]
    if new_rows.empty:
        return corrected
    return combine_df(corrected, apply_corrections(new_rows, mac_address, bucket))


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


@dataclass
class GapFill:
    """
    The outcome of `fill_archive_gap`.

    :param history_df: The archive after the fill; the very frame passed in
        when nothing was added.
    :param start: The gap's start, UTC (the argument as parsed).
    :param end: The gap's end, UTC.
    :param status: "filled" (rows were added), "no_new_data" (Ambient had
        nothing for the gap) or "refused" (inverted gap, or no archive data
        before it).
    :param rows_added: How many rows the fill added.
    """

    history_df: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    status: str
    rows_added: int = 0

    @property
    def skipped(self) -> bool:
        """True when the gap was tried and Ambient had no data for it."""
        return self.status == "no_new_data"


def fill_archive_gap(
    device: dict,
    history_df: pd.DataFrame,
    start,
    end,
    pages: int = 20,
    sleep: bool = False,
) -> GapFill:
    """
    Fetch missing data for a specified gap and merge it into the history, raw.

    Touches no session state: a caller that keeps a list of skipped gaps
    (the Diagnostics tab) records the ones that come back `skipped`.

    :param device: The station's `/devices` entry.
    :param history_df: The archive, raw (as stored), `dateutc` in epoch ms.
        The fill applies no rain corrections, so a raw archive saves raw.
    :param start: The gap's start; anything `pd.to_datetime` reads, as UTC.
    :param end: The gap's end.
    :param pages: Most Ambient pages to fetch (about 246 new readings
        each); a long window, like the rain repair's, needs more than the
        default.
    :param sleep: Wait a second before each page, to stay under Ambient's
        rate limit on a long fill.
    :return: A `GapFill`; its `history_df` is the merged archive on "filled".
    """
    logger.info(f"*** Attempting to fill gap from {start} to {end} ***")

    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)

    if start >= end:
        logger.warning("Invalid gap: start >= end")
        return GapFill(history_df, start, end, "refused")

    # Trim to before the gap, on the int ms column: a slice with Timestamp
    # `dateutc` would meet the fetched int ms rows in one object column, and
    # `normalize` would drop every archive row from the merge.
    start_ms = int(start.value // 1_000_000)
    archive_df = history_df[history_df["dateutc"] < start_ms]

    if archive_df.empty:
        logger.warning("No archive data before gap start; can't determine baseline.")
        return GapFill(history_df, start, end, "refused")

    # Fetch new records to fill the gap
    new_data_df = get_history_since_last_archive(
        device, archive_df, end_date=end, pages=pages, sleep=sleep
    )

    # Merge new data back into full archive; raw, like the fetched rows
    full_combined = combine_full_history(history_df, new_data_df)

    if len(full_combined) == len(history_df):
        logger.info("No new data found. Gap skipped.")
        return GapFill(history_df, start, end, "no_new_data")

    return GapFill(
        full_combined, start, end, "filled", len(full_combined) - len(history_df)
    )


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
    logger.debug(f"combine: +{added} rows (interim total={len(combined)})")
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
    logger.debug(
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

    Validation, cast to int64 ms, drop-invalid, and dedupe are delegated to
    `lookout.utils.dateutc.normalize`. The result is re-sorted **descending**
    to preserve the existing caller-visible order (an ergonomic affordance
    for `df.head()` debugging).

    Behavioral note: as of the dateutc consolidation this path also drops
    rows below the post-2020 floor (see ADR-0001), closing the same class
    of corruption that the load/repair paths already guarded.
    """
    try:
        df = pd.concat([df1, df2], ignore_index=True)
        df = normalize(df)
        return df.sort_values("dateutc", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error combining DataFrames: {e}")
        raise


def _calculate_next_start_date(current_cursor: datetime, limit: int) -> datetime:
    """
    The cursor one page of 5-minute readings after `current_cursor`.

    A probe from a cursor requests the `limit` readings before
    `cursor + (limit - 3) * 5` minutes, so the next probe's window starts
    where this one's ended.

    :param current_cursor: The cursor the empty probe was made from.
    :param limit: Number of records to fetch per page.
    :return: The cursor for the next probe.
    """
    return current_cursor + timedelta(minutes=5 * limit)


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
        logger.debug(f"{label}: empty page")
        return

    try:
        ts = pd.to_datetime(df["dateutc"], unit="ms", utc=True, errors="coerce")
        ts = ts.dropna()
        if ts.empty:
            logger.debug(f"{label}: no valid timestamps")
            return

        n = len(df)
        n_dup = df["dateutc"].duplicated().sum()
        first_ts = ts.min()
        last_ts = ts.max()
        span = last_ts - first_ts

        # check expected 5‑min cadence (not strict—just a hint)
        cadence = ts.sort_values().diff().dropna().value_counts().head(3).to_dict()

        logger.debug(
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
