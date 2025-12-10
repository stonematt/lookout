"""data_processing.py
Collection of functions to manipulate data frames for a streamlit dashboard
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import sys

import lookout.api.awn_controller as awn
from lookout.utils.trend_utils import (
    calculate_temperature_trend,
    calculate_barometer_trend,
)
from lookout.utils.log_util import app_logger
from lookout.utils.memory_utils import (
    get_memory_usage,
    log_memory_usage,
    get_object_memory_usage,
    BYTES_TO_MB,
)

logger = app_logger(__name__)


# ========================================
# Entry points
# ========================================
def load_or_update_data(
    device, bucket, file_type, auto_update, short_minutes, long_minutes
):
    """
    Wrapper function to load or update historical data with status messages.

    :param device: Object representing the device.
    :param bucket: str - The S3 bucket name for archive storage.
    :param file_type: str - The file type for archive storage (e.g., 'parquet').
    :param auto_update: bool - Whether to perform updates automatically.
    :param short_minutes: int - Minimum age threshold for updates (in minutes).
    :param long_minutes: int - Maximum age threshold for updates (in minutes).
    :return: None - Updates Streamlit session state directly.
    """
    update_message = st.empty()

    # Initial load
    if "history_df" not in st.session_state:
        update_message.text("Getting archive data...")
        st.session_state["history_df"] = awn.load_archive_for_device(
            device, bucket, file_type
        )

        # Calculate trends for initial load
        if not st.session_state["history_df"].empty:
            latest_row = st.session_state["history_df"].iloc[0]
            temp_trend = calculate_temperature_trend(
                st.session_state["history_df"], latest_row["tempf"]
            )
            barom_trend = calculate_barometer_trend(
                st.session_state["history_df"], latest_row["baromrelin"]
            )

            # Store trends in session state
            st.session_state["temp_trend"] = temp_trend
            st.session_state["barom_trend"] = barom_trend

        # Initialize session state variables
        max_ms = st.session_state["history_df"]["dateutc"].max()
        st.session_state["history_max_dateutc"] = max_ms
        st.session_state["cloud_max_dateutc"] = max_ms
        st.session_state["session_counter"] = 0

        # Log initial memory usage
        df_size_mb = get_object_memory_usage(st.session_state["history_df"])
        logger.debug(
            f"INITIAL LOAD: {len(st.session_state['history_df'])} rows, {df_size_mb:.1f}MB"
        )

        logger.info("Initial archive load completed.")
        update_message.empty()

    # Fetch interim data if conditions are met
    history_df = st.session_state["history_df"]
    history_max_dateutc = st.session_state["history_max_dateutc"]

    if should_update_history(
        device_last_dateutc=device["lastData"]["dateutc"],
        history_max_dateutc=history_max_dateutc,
        short_minutes=short_minutes,
        long_minutes=long_minutes,
        auto_update=auto_update,
    ):
        update_message.text("Updating historical data...")

        # Track memory before update
        before_size = get_object_memory_usage(st.session_state["history_df"])
        before_rows = len(st.session_state["history_df"])

        awn.update_session_data(device, history_df)
        st.session_state["history_max_dateutc"] = st.session_state["history_df"][
            "dateutc"
        ].max()

        # Track memory after update - only increment counter if data actually changed
        after_size = get_object_memory_usage(st.session_state["history_df"])
        after_rows = len(st.session_state["history_df"])

        # Only increment counter and trigger updates if data actually changed
        if before_rows != after_rows or abs(before_size - after_size) > 0.1:
            st.session_state["session_counter"] = (
                st.session_state.get("session_counter", 0) + 1
            )

        # Force garbage collection to free memory from old data
        import gc

        gc.collect()

        logger.debug(
            f"UPDATE #{st.session_state['session_counter']}: "
            f"{before_rows}→{after_rows} rows, "
            f"{before_size:.1f}→{after_size:.1f}MB "
            f"(+{after_size-before_size:.1f}MB)"
        )

        # Log process memory after update
        process_memory = get_memory_usage()
        if process_memory > 0:
            logger.debug(f"PROCESS_MEMORY after update: {process_memory:.1f}MB")

        logger.info("Historical data updated successfully.")
        update_message.empty()

    # Load rain event catalog after archive data is available
    if "rain_events_catalog" not in st.session_state:
        try:
            from lookout.core.rain_events import RainEventCatalog

            catalog = RainEventCatalog(device["macAddress"], file_type)

            if catalog.catalog_exists():
                events_df = catalog.load_catalog()
                if not events_df.empty:
                    # Update catalog with new data from archive
                    updated_events = catalog.update_catalog_with_new_data(
                        st.session_state["history_df"], events_df
                    )
                    st.session_state["rain_events_catalog"] = updated_events
                    logger.info(
                        f"Rain event catalog loaded: {len(updated_events)} events"
                    )
                else:
                    st.session_state["rain_events_catalog"] = pd.DataFrame()
                    logger.info("Rain event catalog exists but is empty")
            else:
                # Generate catalog if doesn't exist
                events_df = catalog.detect_and_catalog_events(
                    st.session_state["history_df"], auto_save=False
                )
                st.session_state["rain_events_catalog"] = events_df
                logger.info(f"Rain event catalog generated: {len(events_df)} events")

        except Exception as e:
            logger.warning(f"Failed to load rain event catalog: {e}")
            st.session_state["rain_events_catalog"] = pd.DataFrame()

    # Load energy catalog after archive data is available
    if "energy_catalog" not in st.session_state:
        try:
            update_message.text("Getting solar energy catalog...")
            from lookout.core.energy_catalog import EnergyCatalog

            catalog = EnergyCatalog(device["macAddress"], file_type)

            if catalog.catalog_exists():
                periods_df = catalog.load_catalog()
                if not periods_df.empty:
                    # Update catalog with new data from archive
                    update_message.text("Updating solar energy catalog...")
                    updated_periods = catalog.update_catalog_with_new_data(
                        st.session_state["history_df"], periods_df
                    )
                    st.session_state["energy_catalog"] = updated_periods

                    # Auto-save if catalog is >2 days old
                    if catalog.get_catalog_age() > pd.Timedelta(days=2):
                        update_message.text("Saving solar energy catalog...")
                        catalog.save_catalog(updated_periods)
                        logger.info("Energy catalog auto-saved (age > 2 days)")
                    update_message.text("Solar energy catalog loaded successfully")
                    logger.info(f"Energy catalog loaded: {len(updated_periods)} periods")
                else:
                    st.session_state["energy_catalog"] = pd.DataFrame()
                    update_message.text("Solar energy catalog loaded successfully")
                    logger.info("Energy catalog exists but is empty")
            else:
                # Generate catalog if doesn't exist
                update_message.text("Generating solar energy periods...")
                periods_df = catalog.detect_and_calculate_periods(
                    st.session_state["history_df"], auto_save=False
                )
                st.session_state["energy_catalog"] = periods_df
                update_message.text("Solar energy catalog loaded successfully")
                logger.info(f"Energy catalog generated: {len(periods_df)} periods")

        except Exception as e:
            logger.warning(f"Failed to load energy catalog: {e}")
            st.session_state["energy_catalog"] = pd.DataFrame()
            update_message.text("Solar energy catalog loading failed")

    # Update header now that catalogs are loaded
    if "header_placeholder" in st.session_state and "device" in st.session_state:
        try:
            device_name = (
                st.session_state["device"].get("info", {}).get("name", "Unknown")
            )
            with st.session_state.header_placeholder.container():
                from lookout.ui import header

                header.render_weather_header(device_name)
            logger.debug("Header updated with active event information")
        except Exception as e:
            logger.warning(f"Failed to update header after catalog load: {e}")


def should_update_history(
    device_last_dateutc, history_max_dateutc, short_minutes, long_minutes, auto_update
):
    """
    Determines if the historical data should be updated based on age thresholds.

    :param device_last_dateutc: int - Last data timestamp from the device (in milliseconds).
    :param history_max_dateutc: int - Maximum data timestamp in the archive (in milliseconds).
    :param short_minutes: int - Minimum age (in minutes) required to trigger an update.
    :param long_minutes: int - Maximum age (in minutes) for which updates are valid.
    :param auto_update: bool - Whether auto-update is enabled.
    :return: bool - True if history should be updated, False otherwise.
    """
    if not auto_update:
        return False

    delta_ms = device_last_dateutc - history_max_dateutc
    short_ms = short_minutes * 60 * 1000  # Convert minutes to milliseconds
    long_ms = long_minutes * 60 * 1000  # Convert minutes to milliseconds

    return short_ms <= delta_ms < long_ms


def get_human_readable_duration(recent_dateutc, history_dateutc):
    """
    Returns a human-centric duration in minutes, hours, or days.

    Parameters:
    recent_dateutc (int): The last date in UTC from the device.
    history_dateutc (int): The maximum date in UTC from the history.

    Returns:
    str: A human-readable duration.
    """

    history_age_minutes = (recent_dateutc - history_dateutc) / 60000

    if history_age_minutes < 60:
        return f"{history_age_minutes:.0f} minutes"
    elif history_age_minutes < 1440:
        return f"{history_age_minutes / 60:.1f} hours"
    else:
        return f"{history_age_minutes / 1440:.1f} days"


def get_history_min_max(df, date_column="date", data_column="tempf", data_label="temp"):
    """
    Calculate the minimum and maximum values of a data column for specific time periods.

    :param df: pd.DataFrame - Input dataframe with weather data.
    :param date_column: str - The column representing dates that are tz-aware.
    :param data_column: str - The column representing data.
    :param data_label: str - The label for the data column.
    :return: dict - Dictionary with min and max values for specific time periods.
    """
    # Ensure the date column is datetime and timezone-aware
    df[date_column] = pd.to_datetime(df[date_column])
    tz = df[date_column].dt.tz

    # Get the current timestamp in the same timezone
    now = pd.Timestamp.now(tz=tz)
    today_start = now.normalize()  # Start of the current day

    # Define date ranges
    date_ranges = {
        "today": (today_start, now),
        "yesterday": (today_start - pd.Timedelta(days=1), today_start),
        "last 7d": (now - pd.Timedelta(days=7), now),
        "last 30d": (now - pd.Timedelta(days=30), now),
        "last 90d": (now - pd.Timedelta(days=90), now),
        "last 365d": (now - pd.Timedelta(days=365), now),
    }

    results = {}

    # Current value is the data_column for the max(date_column)
    current_data = df.loc[df[date_column].idxmax(), data_column]

    for label, (start, end) in date_ranges.items():
        # Filter data for the specific time period
        period_data = df[(df[date_column] >= start) & (df[date_column] < end)]

        # Calculate min and max values
        results[label] = {
            "min": period_data[data_column].min(),
            "max": period_data[data_column].max(),
            "current": current_data,
        }

    return results


# ========================================
# Polar chart support
# ========================================
def prepare_polar_chart_data(
    df,
    value_col,
    direction_col,
    num_bins=5,
    sector_size=30,
    value_bin_col="value_bin",
    direction_bin_col="direction_bin",
    max_percentile=0.9,
):
    """
    Prepares data for a polar chart by binning values, binning directions, and
    calculating percentages.

    :param df: pd.DataFrame - Input DataFrame containing the raw data.
    :param value_col: str - Column name for the continuous values (e.g., wind speed).
    :param direction_col: str - Column name for the directional data (e.g., wind direction).
    :param num_bins: int - Number of bins for the value column. Defaults to 5.
    :param sector_size: int - Size of directional sectors (degrees). Defaults to 30.
    :param value_bin_col: str - Column name for the binned values. Defaults to "value_bin".
    :param direction_bin_col: str - Column name for the binned directions. Defaults to "direction_bin".
    :param max_percentile: float - Percentile to set the maximum binning value. Defaults to 0.9 (90th percentile).
    :return: pd.DataFrame, list - Grouped data for the polar chart, and value labels.
    """
    # Step 1: Bin continuous values
    df, value_labels = bin_values(
        df,
        value_col,
        num_bins=num_bins,
        bin_col_name=value_bin_col,
        max_percentile=max_percentile,
    )

    # Step 2: Bin directional values
    df, direction_labels = bin_directions(
        df, direction_col, sector_size, bin_col_name=direction_bin_col
    )

    # Step 3: Calculate percentages
    grouped_data = calculate_percentages(df, [value_bin_col, direction_bin_col])

    return grouped_data, value_labels, direction_labels


def bin_values(
    df, value_col, bin_col_name="value_bin", num_bins=5, max_percentile=0.75
):
    """
    Bins continuous numeric values into categories, setting max_value to the 75th percentile
    and allowing the number of bins to be specified.

    :param df: pd.DataFrame - DataFrame containing the data.
    :param value_col: str - Column name for the values to bin.
    :param num_bins: int, optional - Number of bins to create. Defaults to 5.
    :param bin_col_name: str - Name of the column to store the binned values.
    :return: pd.DataFrame, list - DataFrame with binned values in `bin_col_name`, and bin labels.
    """
    # Determine min and max from the data
    min_value = df[value_col].min()
    max_value = df[value_col].quantile(max_percentile)

    # Avoid division by zero or negative steps
    if max_value <= min_value:
        raise ValueError("Max value must be greater than min value to calculate bins.")

    # Determine step size based on number of bins
    step = (max_value - min_value) / num_bins
    step = max(step, 1)  # Ensure step is at least 1

    # Create bins
    value_bins = list(range(int(min_value), int(max_value) + int(step), int(step))) + [
        float("inf")
    ]
    value_labels = [
        f"{value_bins[i]}-{value_bins[i+1]}" for i in range(len(value_bins) - 2)
    ] + [f"{int(max_value)}+"]

    # Bin values into categories
    df[bin_col_name] = pd.cut(
        df[value_col], bins=value_bins, labels=value_labels, right=False
    )

    return df, value_labels


def bin_directions(df, direction_col, sector_size=30, bin_col_name="direction_bin"):
    """
    Groups directional data into equal-sized sectors.

    :param df: pd.DataFrame - DataFrame containing the directional data to process.
    :param direction_col: str - Column name for directional data (degrees).
    :param sector_size: int - Size of directional sectors (e.g., 30°).
    :param bin_col_name: str - Name of the column to store the binned directions.
    :return: pd.DataFrame, list - DataFrame with binned directions in `bin_col_name`,
             and sector labels for visualization.
    """
    direction_bins = np.arange(0, 361, sector_size)
    direction_labels = [
        f"{direction_bins[i]}-{direction_bins[i+1]}"
        for i in range(len(direction_bins) - 1)
    ]
    df[bin_col_name] = pd.cut(
        df[direction_col],
        bins=direction_bins,
        labels=direction_labels,
        right=False,
        include_lowest=True,
    )
    return df, direction_labels


def calculate_percentages(df, group_cols):
    """
    Calculates percentage distribution within grouped categories.

    :param df: pd.DataFrame - DataFrame containing grouped data.
    :param group_cols: list - Columns to group by (e.g., 'value_bin', 'direction_bin').
    :return: pd.DataFrame - DataFrame with counts and percentage distributions for each group.
    """
    total_count = len(df)
    grouped = df.groupby(group_cols, observed=False).size().reset_index(name="count")
    grouped["percentage"] = (grouped["count"] / total_count) * 100
    return grouped


# ========================================
# Gap and Quality Analysis
# ========================================
def detect_gaps(
    df: pd.DataFrame, timestamp_col: str = "dateutc", threshold_minutes: int = 10
) -> pd.DataFrame:
    """
    Detects time gaps in a DataFrame based on a minimum threshold.

    :param df: pd.DataFrame - Input DataFrame with a datetime or integer timestamp column.
    :param timestamp_col: str - Column name containing timestamps (e.g., "dateutc").
    :param threshold_minutes: int - Minimum gap size to consider (in minutes).
    :return: pd.DataFrame - With ['start', 'end', 'duration_minutes'] columns for each gap.
    """
    if df.empty or timestamp_col not in df.columns:
        return pd.DataFrame(
            {
                "start": pd.Series(dtype="datetime64[ns]"),
                "end": pd.Series(dtype="datetime64[ns]"),
                "duration_minutes": pd.Series(dtype="float"),
            }
        )

    df_sorted = df.sort_values(timestamp_col).copy()

    if not pd.api.types.is_datetime64_any_dtype(df_sorted[timestamp_col]):
        df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], unit="ms")

    time_diffs = (
        df_sorted[timestamp_col].diff().dt.total_seconds().div(60).astype(float)
    )
    gap_mask = time_diffs.gt(float(threshold_minutes))

    starts = df_sorted[timestamp_col].shift(1)[gap_mask].reset_index(drop=True)
    ends = df_sorted[timestamp_col][gap_mask].reset_index(drop=True)
    durations = time_diffs[gap_mask].reset_index(drop=True)

    return pd.DataFrame(
        {
            "start": starts,
            "end": ends,
            "duration_minutes": durations,
        }
    )


def prepare_box_plot_data(
    history_df: pd.DataFrame, selected_metrics: List[Dict], box_width_option: str
) -> Tuple[pd.DataFrame, str]:
    """
    Prepare data for box plot visualization.

    :param history_df: Historical weather data
    :param selected_metrics: List of selected metrics with titles and column names
    :param box_width_option: Time grouping option (hour, day, week, month)
    :return: Tuple of (processed_df, group_column_name)
    """
    if not selected_metrics or "date" not in history_df.columns:
        return history_df, ""

    # Convert 'date' column to datetime if it's not already
    df_local = history_df.copy()
    df_local["date"] = pd.to_datetime(df_local["date"])

    group_column = ""

    # Group by the selected box width option
    if box_width_option == "hour":
        df_local["hour"] = df_local["date"].dt.hour
        group_column = "hour"
    elif box_width_option == "day":
        df_local["day"] = df_local["date"].dt.dayofyear
        group_column = "day"
    elif box_width_option == "week":
        df_local["week"] = df_local["date"].dt.isocalendar().week
        group_column = "week"
    elif box_width_option == "month":
        df_local["month"] = df_local["date"].dt.month
        group_column = "month"

    return df_local, group_column
