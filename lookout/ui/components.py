"""
Reusable UI components for Lookout weather station dashboard.

This module provides shared UI components that can be used across different
tabs and modules to ensure consistent behavior and styling.
"""

import datetime
import pandas as pd
import streamlit as st


def create_date_range_slider(
    data_df: pd.DataFrame,
    date_column: str = "timestamp",
    key_prefix: str = "date_range",
) -> tuple:
    """
    Create a reusable date range slider component.

    :param data_df: DataFrame containing date data
    :param date_column: Name of the column containing dates/timestamps
    :param key_prefix: Prefix for streamlit widget keys to avoid conflicts
    :return: Tuple of (start_timestamp, end_timestamp) in UTC, or (None, None) if invalid
    """
    if data_df.empty or date_column not in data_df.columns:
        st.warning("No date data available for filtering.")
        return None, None

    # Get date range from data
    min_date = data_df[date_column].min().date()
    max_date = data_df[date_column].max().date()

    st.write("**Date Range:**")

    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MMM DD, YYYY",
        label_visibility="collapsed",
        key=f"{key_prefix}_slider",
    )

    if len(date_range) == 2:
        start_date, end_date = date_range

        # Ensure we have proper timestamp objects for consistent comparisons
        if isinstance(start_date, (datetime.date, str)):
            start_ts = (
                pd.Timestamp(start_date)
                .tz_localize("America/Los_Angeles")
                .tz_convert("UTC")
            )
        else:
            start_ts = start_date  # Already a timestamp

        if isinstance(end_date, (datetime.date, str)):
            end_ts = (
                (pd.Timestamp(end_date) + pd.Timedelta(days=1))
                .tz_localize("America/Los_Angeles")
                .tz_convert("UTC")
            )
        else:
            end_ts = end_date  # Already a timestamp

        return start_ts, end_ts

    return None, None


def filter_dataframe_by_date_range(
    df: pd.DataFrame, date_column: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp
) -> pd.DataFrame:
    """
    Filter a DataFrame by date range using UTC timestamps.

    :param df: DataFrame to filter
    :param date_column: Column name containing timestamps
    :param start_ts: Start timestamp (UTC)
    :param end_ts: End timestamp (UTC)
    :return: Filtered DataFrame (view, not copy for memory efficiency)
    """
    if start_ts is None or end_ts is None:
        return df

    # Validate timestamp types to prevent comparison errors
    if not isinstance(start_ts, pd.Timestamp) or not isinstance(end_ts, pd.Timestamp):
        raise ValueError("start_ts and end_ts must be pandas Timestamps")

    # Use .loc[] to return a view instead of copy for memory efficiency
    mask = (df[date_column] >= start_ts) & (df[date_column] < end_ts)
    return df.loc[mask]
