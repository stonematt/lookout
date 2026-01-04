"""
Reusable UI components for Lookout weather station dashboard.

This module provides shared UI components that can be used across different
tabs and modules to ensure consistent behavior and styling.
"""

import datetime
from typing import Optional, Union, List, Dict, Tuple
import pandas as pd
import streamlit as st


def create_date_range_slider(
    data_df: pd.DataFrame,
    date_column: str = "timestamp",
    key_prefix: str = "date_range",
    default_days: Union[int, None] = None,
) -> tuple:
    """
    Create a reusable date range slider component.

    :param data_df: DataFrame containing date data
    :param date_column: Name of the column containing dates/timestamps
    :param key_prefix: Prefix for streamlit widget keys to avoid conflicts
    :param default_days: Optional number of days to show by default (pre-selects last N days).
                        If None, shows full available range selected.
    :return: Tuple of (start_timestamp, end_timestamp) in UTC, or (None, None) if invalid
    """
    if data_df.empty or date_column not in data_df.columns:
        st.warning("No date data available for filtering.")
        return None, None

    # Get date range from data
    min_date = data_df[date_column].min().date()
    max_date = data_df[date_column].max().date()

    st.write("**Date Range:**")

    # Calculate default selected range
    if default_days is not None:
        # Pre-select last N days, but don't go before available data
        default_start = max(min_date, max_date - pd.Timedelta(days=default_days))
        selected_value = (default_start, max_date)
    else:
        # Default behavior: select full available range
        selected_value = (min_date, max_date)

    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=selected_value,
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


def render_solar_metric_card(title: str, value: float, unit: str,
                           sparkline_data: List[float], period_type: str,
                           y_axis_range: Tuple[float, float],
                           delta: Optional[Dict] = None):
    """
    Render individual solar radiation metric card with step-chart sparkline.

    Card layout:
    - Title row (muted text)
    - Value row (large numeric + unit)
    - Sparkline row (40px bar chart)
    - Delta row (optional up/down badge)

    :param title: Period title (e.g., "Today", "Last 7 Days")
    :param value: Numeric value to display
    :param unit: Unit string ("kWh/m²" or "Wh/m²")
    :param sparkline_data: List of values for sparkline (NaN for missing)
    :param period_type: Period type for sparkline configuration
    :param y_axis_range: Fixed y-axis range tuple (min, max)
    :param delta: Optional delta dict with 'value' and 'direction' keys
    """
    from lookout.core.solar_cards import create_solar_sparkline

    # Title row - muted text
    st.caption(title)

    # Value row - large numeric with unit
    col_value, col_unit = st.columns([3, 1])
    with col_value:
        st.metric("", f"{value:.1f}", label_visibility="collapsed")
    with col_unit:
        st.caption(unit)

    # Sparkline row - step chart with no margins
    sparkline_fig = create_solar_sparkline(sparkline_data, period_type, y_axis_range)
    st.plotly_chart(sparkline_fig, width="stretch", config={"displayModeBar": False})

    # Delta row - optional up/down badge
    if delta:
        delta_symbol = "↑" if delta["direction"] == "up" else "↓"
        delta_text = f"{delta_symbol} {delta['value']:.1f}"
        st.caption(delta_text)
