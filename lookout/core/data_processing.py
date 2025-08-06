"""data_processing.py
Collection of functions to manipulate data frames for a streamlit dashboard
"""

import numpy as np
import pandas as pd
import streamlit as st

import lookout.api.awn_controller as awn
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


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

        # Initialize session state variables
        st.session_state["history_max_dateutc"] = int(
            st.session_state["history_df"]["dateutc"].max().timestamp() * 1000
        )
        st.session_state["cloud_max_dateutc"] = int(
            st.session_state["history_df"]["dateutc"].max().timestamp() * 1000
        )
        st.session_state["session_counter"] = 0

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
        awn.update_session_data(device, history_df)
        st.session_state["history_max_dateutc"] = int(
            st.session_state["history_df"]["dateutc"].max().timestamp() * 1000
        )

        logger.info("Historical data updated successfully.")
        update_message.empty()


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
