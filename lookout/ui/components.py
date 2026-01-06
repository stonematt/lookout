"""
Reusable UI components for Lookout weather station dashboard.

This module provides shared UI components that can be used across different
tabs and modules to ensure consistent behavior and styling.
"""

import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
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





def render_solar_tile(
    title: str,
    total_kwh: float,
    period_type: str,
    sparkline_data: List[float],
    y_axis_range: Tuple[float, float],
    delta_value: Optional[float] = None,
    hover_labels: Optional[List[str]] = None,
    current_period_index: Optional[int] = None,
):
    """
    Render compact solar tile with metric + sparkline.

    Tile layout:
    - Row 1: Streamlit metric with title as label + delta (integrated)
    - Row 2: Simple sparkline (40px, dense)

    :param title: Metric label ("Last 24h", "Last 7d", etc.)
    :param total_kwh: Sum of period energy
    :param period_type: "last_24h", "last_7d", etc. (unused but for consistency)
    :param sparkline_data: List of hourly/daily values
    :param y_axis_range: Fixed axis for comparison across tiles
    :param delta_value: Delta value (positive/negative/None)
    :param hover_labels: Hover text with time/date + energy
    :param current_period_index: Index to highlight with border
    """
    # Metric row with integrated delta (title becomes metric label)
    if delta_value is not None:
        delta_display = f"{delta_value:+.2f}"
    else:
        delta_display = None
    st.metric(title, f"{total_kwh:.2f}", delta=delta_display)

    # Sparkline row - simple column chart
    fig = _create_simple_sparkline(
        sparkline_data, hover_labels or [], current_period_index, y_axis_range
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _create_simple_sparkline(
    values: List[float],
    hover_labels: Optional[List[str]] = None,
    current_period_index: Optional[int] = None,
    y_axis_range: Tuple[float, float] = (0, 1.0),
) -> go.Figure:
    """
    Create simple column chart sparkline (15-line replacement for 135-line function).

    Features:
    - Step chart style (full-width bars, no spacing)
    - Gray bars for NaN values (full height)
    - Black border for current period highlight
    - Minimal styling (no axes/labels/toolbar)

    :param values: List of numeric values (NaN for missing data)
    :param hover_labels: Hover text with time/date + energy values
    :param current_period_index: Index to highlight with black border
    :param y_axis_range: Fixed y-axis range for comparison
    :return: Plotly figure configured as minimal sparkline
    """
    from lookout.core.chart_config import get_standard_colors

    colors = get_standard_colors()
    fig = go.Figure()

    if not values:
        # Empty sparkline
        fig.update_layout(height=40, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, range=y_axis_range)
        return fig

    # Calculate max value for auto-scaling gray bar height
    if y_axis_range is None:
        non_nan_values = [v for v in values if not pd.isna(v)]
        max_actual_value = max(non_nan_values) if non_nan_values else 1.0
        gray_bar_height = max_actual_value  # Auto-scale: use actual max
    else:
        gray_bar_height = y_axis_range[1]  # Fixed range: use max from range

    # Prepare colors and borders
    bar_colors = []
    border_widths = []

    for i, value in enumerate(values):
        if pd.isna(value):
            bar_colors.append(colors["gap_fill"])  # Gray for missing data
            border_widths.append(0)
        else:
            bar_colors.append(colors["solar_medium"])  # Golden orange
            # Black border for current period
            if i == current_period_index:
                border_widths.append(1)
            else:
                border_widths.append(0)

    # Single trace with conditional styling
    fig.add_trace(
        go.Bar(
            x=list(range(len(values))),
            y=[v if not pd.isna(v) else gray_bar_height for v in values],
            marker_color=bar_colors,
            marker_line_width=border_widths,
            marker_line_color="black",
            hovertext=hover_labels
            or [f"Value: {v:.2f}" if not pd.isna(v) else "No data" for v in values],
            hoverinfo="text",
            width=1,  # Full width, no spacing
            showlegend=False,
        )
    )

    # Minimal layout
    fig.update_layout(
        height=40,  # Compact sparkline height
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        hovermode="x",
    )

    # Hidden axes
    fig.update_xaxes(visible=False)

    # Auto-scale if no range provided, otherwise use fixed range
    if y_axis_range is not None:
        fig.update_yaxes(visible=False, range=y_axis_range)
    else:
        fig.update_yaxes(visible=False)  # Auto-scale

    return fig
