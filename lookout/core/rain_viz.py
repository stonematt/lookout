"""
rain_viz.py
Rainfall-specific visualization functions for Lookout weather station dashboard.

This module contains specialized visualization functions for rainfall analysis,
including event charts, accumulation heatmaps, violin plots, and year-over-year
comparisons. Functions are extracted from the main visualization module to
improve code organization and maintainability.

Functions:
- create_rainfall_violin_plot: Single violin plot for rainfall distribution
- create_dual_violin_plot: Side-by-side violin plot comparison
- create_event_accumulation_chart: Cumulative rainfall area chart for events
- create_event_rate_chart: Rainfall intensity bar chart for events
- create_rainfall_summary_violin: Box plot showing current vs historical rainfall
- prepare_rain_accumulation_heatmap_data: Prepare data for accumulation heatmaps
- create_rain_accumulation_heatmap: Create rainfall accumulation heatmap
- create_year_over_year_accumulation_chart: Year-over-year cumulative rainfall
- create_event_histogram: Histogram showing event count over time with range highlighting
- extract_event_data: Extract and filter event data from weather archive
- format_event_header: Create formatted event header string with metadata
- render_event_visualization_core: Unified core event visualization logic
- _create_event_headline: Format event headline string (internal)
- create_event_detail_charts: Create both accumulation and rate charts for events
"""

import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lookout.core.chart_config import (
    get_standard_colors,
    apply_time_series_layout,
    apply_standard_axes,
    create_standard_annotation,
    apply_violin_layout,
)
from lookout.utils.log_util import app_logger
from lookout.utils.memory_utils import force_garbage_collection

logger = app_logger(__name__)


def create_rainfall_violin_plot(
    window: str,
    violin_data: dict,
    unit: str = "in",
    title: Optional[str] = None,
) -> None:
    """
    Create single violin plot showing rainfall distribution for specified window.

    :param window: Window size (e.g., "7d", "30d")
    :param violin_data: Data from prepare_violin_plot_data()
    :param unit: Unit for display (e.g., "in")
    :param title: Chart title (auto-generated if None)
    """
    if window not in violin_data or len(violin_data[window]["values"]) == 0:
        st.warning(f"No historical data available for {window} period.")
        return

    data = violin_data[window]
    values = data["values"]
    current = data["current"]
    percentile = data["percentile"]

    # Get standard colors from chart_config
    colors = get_standard_colors()

    fig = go.Figure()

    fig.add_trace(
        go.Violin(
            y=values,
            name=f"Historical {window}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors["muted_marker"],
            line_color=colors["muted_line"],
            x0=f"{window} Periods",
        )
    )

    if not np.isnan(current):
        fig.add_trace(
            go.Scatter(
                x=[f"{window} Periods"],
                y=[current],
                mode="markers",
                marker=dict(
                    symbol="diamond-tall",
                    size=16,
                    color="red",
                    line=dict(width=2, color="darkred"),
                ),
                name=f"Current ({current:.2f}{unit})",
                text=[
                    (
                        f"Current: {current:.2f}{unit}<br>Percentile: {percentile:.1f}th"
                        if not np.isnan(percentile)
                        else f"Current: {current:.2f}{unit}"
                    )
                ],
                hoverinfo="text",
            )
        )

    # Apply violin layout configuration
    chart_title = title or f"Rainfall Distribution: {window} Rolling Periods"
    fig = apply_violin_layout(
        fig,
        title=chart_title,
        height=500,
        yaxis_title=f"Rainfall ({unit})",
        showlegend=True
    )

    st.plotly_chart(fig, width="stretch")

    if not np.isnan(percentile):
        if percentile >= 90:
            status = "🔴 **Extremely wet**"
        elif percentile >= 75:
            status = "🟡 **Above normal**"
        elif percentile >= 25:
            status = "🟢 **Normal range**"
        else:
            status = "🔵 **Below normal**"

        st.markdown(
            f"{status} - Current {window} total ({current:.2f}{unit}) ranks at **{percentile:.1f}th percentile** of {len(values):,} historical periods."
        )


def create_dual_violin_plot(
    left_window: str,
    right_window: str,
    violin_data: dict,
    unit: str = "in",
    title: Optional[str] = None,
) -> None:
    """
    Create dual violin plot comparing two different rainfall windows side-by-side.

    :param left_window: Left window size (e.g., "1d")
    :param right_window: Right window size (e.g., "7d")
    :param violin_data: Data from prepare_violin_plot_data()
    :param unit: Unit for display (e.g., "in")
    :param title: Chart title (auto-generated if None)
    """

    missing_data = []
    for window in [left_window, right_window]:
        if window not in violin_data or len(violin_data[window]["values"]) == 0:  # type: ignore
            missing_data.append(window)

    if missing_data:
        st.warning(f"No historical data available for: {', '.join(missing_data)}")
        return

    left_data = violin_data[left_window]
    right_data = violin_data[right_window]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{left_window} Periods", f"{right_window} Periods"],
        shared_yaxes=True,
    )

    # Get standard colors from chart_config
    colors = get_standard_colors()
    
    # Use contrasting colors for dual violin comparison
    violin_colors = {
        left_window: colors["muted_marker"],
        right_window: colors["muted_current"],
    }
    line_colors = {
        left_window: colors["muted_line"],
        right_window: colors["muted_line"],
    }

    fig.add_trace(
        go.Violin(
            y=left_data["values"],
            name=f"Historical {left_window}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=violin_colors[left_window],
            line_color=line_colors[left_window],
            x0=f"{left_window}",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Violin(
            y=right_data["values"],
            name=f"Historical {right_window}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=violin_colors[right_window],
            line_color=line_colors[right_window],
            x0=f"{right_window}",
        ),
        row=1,
        col=2,
    )

    for i, (window, data) in enumerate(
        [(left_window, left_data), (right_window, right_data)], 1
    ):
        current = data["current"]
        percentile = data["percentile"]

        if not np.isnan(current):
            fig.add_trace(
                go.Scatter(
                    x=[window],
                    y=[current],
                    mode="markers",
                    marker=dict(
                        symbol="diamond-tall",
                        size=16,
                        color="red",
                        line=dict(width=2, color="darkred"),
                    ),
                    name=f"Current {window} ({current:.2f}{unit})",
                    text=[
                        (
                            f"Current: {current:.2f}{unit}<br>Percentile: {percentile:.1f}th"
                            if not np.isnan(percentile)
                            else f"Current: {current:.2f}{unit}"
                        )
                    ],
                    hoverinfo="text",
                    showlegend=True,
                ),
                row=1,
                col=i,
            )

    # Apply violin layout configuration
    chart_title = (
        title or f"Rainfall Distribution Comparison: {left_window} vs {right_window}"
    )
    fig = apply_violin_layout(
        fig,
        title=chart_title,
        height=600,
        yaxis_title=f"Rainfall ({unit})",
        showlegend=True
    )

    fig.update_yaxes(title_text=f"Rainfall ({unit})", row=1, col=1)

    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        left_current = left_data["current"]
        left_percentile = left_data["percentile"]
        if not np.isnan(left_percentile):
            st.markdown(
                f"**{left_window} Period**: {left_current:.2f}{unit} ({left_percentile:.1f}th percentile)"
            )

    with col2:
        right_current = right_data["current"]
        right_percentile = right_data["percentile"]
        if not np.isnan(right_percentile):
            st.markdown(
                f"**{right_window} Period**: {right_current:.2f}{unit} ({right_percentile:.1f}th percentile)"
            )


def create_event_accumulation_chart(
    event_data: pd.DataFrame, event_info: dict
) -> go.Figure:
    """
    Create area chart showing cumulative rainfall for a rain event.

    :param event_data: DataFrame with dateutc and eventrainin columns (sorted by time)
    :param event_info: Dict with total_rainfall, duration_minutes, start_time, end_time
    :return: Plotly figure
    """
    df = event_data.copy()
    df["timestamp"] = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    df["time_pst"] = df["timestamp"].dt.tz_convert("America/Los_Angeles")

    # Get standard colors from chart_config
    colors = get_standard_colors()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["time_pst"],
            y=df["eventrainin"],
            mode="lines",
            fill="tozeroy",
            line=dict(color=colors["rainfall_line"], width=2),
            fillcolor=colors["rainfall_fill"],
            hovertemplate='%{x|%b %d %I:%M %p}<br>%{y:.3f}"<extra></extra>',
            name="Rainfall",
        )
    )

    # Apply time series layout configuration
    fig = apply_time_series_layout(
        fig,
        height=300,
        showlegend=False,
        hovermode="x unified"
    )

    # Apply standard axes configuration
    fig = apply_standard_axes(
        fig,
        xaxis_title="",
        yaxis_title="Rainfall (in)",
        showgrid_x=False,
        showgrid_y=True
    )

    # Add total rainfall annotation using chart_config helper
    total_annotation = create_standard_annotation(
        text=f"Total: {event_info['total_rainfall']:.3f}\"",
        position="top_right"
    )
    fig.add_annotation(total_annotation)

    return fig


def create_event_rate_chart(event_data: pd.DataFrame) -> go.Figure:
    """
    Create bar chart showing rainfall intensity with time-aware rate calculation.

    Rates calculated from actual time intervals. Data gaps (>10 min) are filled with
    synthetic 5-min interval bars showing average rate over the gap period, colored
    gray to distinguish from instantaneous measurements.

    :param event_data: DataFrame with dateutc and dailyrainin columns (sorted by time)
    :return: Plotly figure
    """
    df = event_data.copy()
    df["timestamp"] = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    df["time_pst"] = df["timestamp"].dt.tz_convert("America/Los_Angeles")

    df["time_diff_min"] = df["timestamp"].diff().dt.total_seconds() / 60
    df.loc[df.index[0], "time_diff_min"] = 5

    df["interval_rain"] = df["dailyrainin"].diff().clip(lower=0)
    df.loc[df.index[0], "interval_rain"] = 0

    df["rate"] = df["interval_rain"] / (df["time_diff_min"] / 60)

    # Get standard colors from chart_config
    colors = get_standard_colors()

    times = []
    rates = []
    bar_colors = []
    customdata = []

    for idx in df.index:
        row = df.loc[idx]
        time_gap = row["time_diff_min"]
        interval_rain = row["interval_rain"]
        rate = row["rate"]

        if time_gap > 10:
            num_intervals = max(1, int(time_gap / 5))
            avg_rate = interval_rain / (time_gap / 60)

            prev_idx = df.index[df.index.get_loc(idx) - 1]
            prev_time = df.loc[prev_idx, "time_pst"]
            curr_time = row["time_pst"]

            for i in range(num_intervals):
                synthetic_time = prev_time + pd.Timedelta(minutes=5 * (i + 1))
                if synthetic_time <= curr_time:
                    times.append(synthetic_time)
                    rates.append(avg_rate)
                    bar_colors.append(colors["gap_fill"])
                    customdata.append(f"({int(time_gap)}min avg)")
        else:
            times.append(row["time_pst"])
            rates.append(rate)

            if rate < 0.1:
                bar_colors.append(colors["rate_low"])
            elif rate < 0.3:
                bar_colors.append(colors["rate_medium"])
            else:
                bar_colors.append(colors["rate_high"])
            customdata.append("")

    hover_template = (
        "%{x|%b %d %I:%M %p}<br>" "%{y:.3f} in/hr<br>" "%{customdata}<extra></extra>"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=times,
            y=rates,
            marker_color=bar_colors,
            hovertemplate=hover_template,
            customdata=customdata,
        )
    )

    # Apply time series layout configuration
    fig = apply_time_series_layout(
        fig,
        height=150,
        showlegend=False,
        hovermode="closest"  # Better for bar charts
    )

    # Apply standard axes configuration
    fig = apply_standard_axes(
        fig,
        xaxis_title="",
        yaxis_title="Rate (in/hr)",
        showgrid_x=False,
        showgrid_y=True
    )

    # Set bar gap (specific to bar charts)
    fig.update_layout(bargap=0)

    return fig


def create_rainfall_summary_violin(
    daily_rain_df: pd.DataFrame,
    current_values: dict,
    rolling_context_df: pd.DataFrame,
    end_date: pd.Timestamp,
    windows: list = None,
    title: str = None,
) -> go.Figure:
    """
    Create box plot showing current rainfall vs historical distributions.

    Shows specified windows with box plots (via violin with hidden shape) and
    current values as colored diamond markers.

    :param daily_rain_df: DataFrame with daily rainfall totals
    :param current_values: Dict with today, yesterday, 7d, 30d, 90d, 365d values
    :param rolling_context_df: DataFrame from compute_rolling_rain_context
    :param end_date: Current date for analysis
    :param windows: List of window keys to display (default: all 6)
    :return: Plotly figure
    """

    if windows is None:
        windows = ["Today", "Yesterday", "7d", "30d", "90d", "365d"]

    fig = go.Figure()

    # Convert dates without copying the entire DataFrame
    dates = pd.to_datetime(daily_rain_df["date"])
    all_single_days = daily_rain_df[dates.dt.year != end_date.year]["rainfall"].values

    annotations_data = []

    for window in windows:
        if window in ["Today", "Yesterday"]:
            category = window
            current_val = current_values.get(window.lower(), 0)
            distribution = all_single_days
        else:
            category = window
            window_days = int(window.rstrip("d"))

            window_row = rolling_context_df[
                rolling_context_df["window_days"] == window_days
            ]

            if len(window_row) == 0:
                continue

            row = window_row.iloc[0]
            current_val = current_values.get(window, row.get("total", 0))

            # Convert date column to datetime and set as index for proper year comparison
            df_temp = daily_rain_df.copy()
            df_temp["date"] = pd.to_datetime(df_temp["date"])
            s = df_temp.set_index("date")["rainfall"].sort_index()
            historical_data = s[s.index.year != end_date.year]

            if len(historical_data) < window_days:
                continue

            all_periods = (
                historical_data.rolling(window=window_days).sum().dropna().values
            )
            distribution = all_periods[np.isfinite(all_periods)]

        if len(distribution) > 0:
            q25, q75 = np.percentile(distribution, [25, 75])

            fig.add_trace(
                go.Box(
                    y=distribution,
                    name=category,
                    boxpoints="outliers",
                    marker=dict(color="lightblue", size=3),
                    line=dict(color="steelblue"),
                    fillcolor="lightblue",
                    showlegend=False,
                )
            )

            percentile = (
                (distribution < current_val).sum() / len(distribution) * 100
                if len(distribution) > 0
                else 50
            )

            marker_color = (
                "red"
                if current_val > q75
                else "green" if current_val < q25 else "orange"
            )

            fig.add_trace(
                go.Scatter(
                    x=[category],
                    y=[current_val],
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=12,
                        color=marker_color,
                        line=dict(width=2, color="black"),
                    ),
                    showlegend=False,
                    hovertemplate=f'{current_val:.2f}" ({percentile:.0f}th percentile)<extra></extra>',
                )
            )

            annotations_data.append((category, current_val, percentile))

    fig.update_layout(
        height=400,
        yaxis_title="Rainfall (inches)",
        yaxis=dict(rangemode="tozero", showgrid=True, gridcolor="lightgray"),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=50, r=20, t=30, b=100),
        hovermode="x unified",
        title=title,
    )

    for cat, val, pct in annotations_data:
        fig.add_annotation(
            x=cat,
            y=0,
            yshift=-40,
            text=f'{val:.2f}"<br>{pct:.0f}th',
            showarrow=False,
            yref="paper",
            yanchor="top",
            font=dict(size=10),
        )

    # Memory cleanup for large DataFrames created during visualization
    try:
        del pivot, full_data, indexed
        force_garbage_collection()
    except:
        pass

    return fig


def prepare_rain_accumulation_heatmap_data(
    archive_df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    timezone: str = "America/Los_Angeles",
    num_days: Optional[int] = None,
    include_gaps: bool = False,
    row_mode: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare rainfall accumulation data for heatmap with simplified aggregation.

    All rainfall increments are included to ensure accurate totals.
    Each dailyrainin.diff() represents real rain that fell, captured
    at the time of the reading regardless of data gaps.

    Row modes: 'day', 'week', 'month', 'year_month', 'auto'
    - day: Daily rows with Hour of Day columns
    - week: Weekly rows with Day of Week columns
    - month: Monthly rows with Day of Month columns
    - year_month: YY-MM rows with Day of Month columns
    - auto: Automatically select based on period length

    :param archive_df: Archive with dateutc and dailyrainin columns
    :param start_date: Filter start date (timezone-aware or naive UTC)
    :param end_date: Filter end date (timezone-aware or naive UTC)
    :param timezone: Timezone for hour bucketing
    :param num_days: Number of days in period (helps determine auto mode)
    :param include_gaps: Deprecated parameter (has no effect, all data included)
    :param row_mode: Row aggregation mode ('day', 'week', 'month', 'year_month', 'auto')
    :return: DataFrame with (date, hour, accumulation) columns
    """
    if archive_df.empty or "dailyrainin" not in archive_df.columns:
        logger.warning("No dailyrainin data available for accumulation heatmap")
        return pd.DataFrame(columns=["date", "hour", "accumulation"])

    df = archive_df.copy()

    # Convert to datetime and timezone
    df["timestamp"] = pd.to_datetime(df["dateutc"], unit="ms", utc=True)
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(timezone)

    # Filter by date range if specified
    if start_date:
        if start_date.tz is None:
            start_date = start_date.tz_localize("UTC")
        df = df[df["timestamp"] >= start_date]

    if end_date:
        if end_date.tz is None:
            end_date = end_date.tz_localize("UTC")
        df = df[df["timestamp"] <= end_date]

    if df.empty:
        logger.warning("No data in specified date range")
        return pd.DataFrame(columns=["date", "hour", "accumulation"])

    # Sort by timestamp ascending (archive may be DESC)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Calculate interval accumulation
    df["time_diff_min"] = df["timestamp"].diff().dt.total_seconds() / 60
    df["interval_rain"] = df["dailyrainin"].diff().clip(lower=0)

    # Handle first row
    first_idx = df.index[0]
    df.loc[first_idx, "time_diff_min"] = 5
    df.loc[first_idx, "interval_rain"] = 0

    # Extract date and hour from local time
    df["date"] = df["timestamp_local"].dt.date
    df["hour"] = df["timestamp_local"].dt.hour

    # NOTE: Gap filtering removed - all accumulation data is included
    # to ensure accurate totals and proper hourly distribution

    # Aggregate by (date, hour)
    hourly_accum = df.groupby(["date", "hour"])["interval_rain"].sum().reset_index()
    hourly_accum.columns = ["date", "hour", "accumulation"]

    # Determine aggregation mode
    if row_mode is None or row_mode == "auto":
        if num_days and num_days > 730:  # 2 years
            row_mode = "year_month"
        elif num_days and num_days > 180:
            row_mode = "week"
        else:
            row_mode = "day"

    # Add timestamp for date operations
    hourly_accum["date_ts"] = pd.to_datetime(hourly_accum["date"])

    # Apply aggregation based on row mode (column type is determined by row type)
    if row_mode == "month":
        logger.info(f"Aggregating by month/day-of-month")

        # Add month and day columns
        hourly_accum["month"] = hourly_accum["date_ts"].dt.month
        hourly_accum["day_of_month"] = hourly_accum["date_ts"].dt.day

        # Aggregate by (month, day_of_month)
        monthly_accum = (
            hourly_accum.groupby(["month", "day_of_month"])["accumulation"]
            .sum()
            .reset_index()
        )
        monthly_accum.columns = ["date", "hour", "accumulation"]  # Reuse column names

        logger.info(
            f"Prepared monthly/day heatmap data: {len(monthly_accum)} cells "
            f"across all months"
        )

        return monthly_accum

    elif row_mode == "year_month":
        logger.info(f"Aggregating by year-month/day-of-month")

        # Add year-month and day columns
        hourly_accum["year_month"] = (
            hourly_accum["date_ts"].dt.to_period("M").dt.strftime("%Y-%m")
        )
        hourly_accum["day_of_month"] = hourly_accum["date_ts"].dt.day

        # Aggregate by (year_month, day_of_month)
        year_month_accum = (
            hourly_accum.groupby(["year_month", "day_of_month"])["accumulation"]
            .sum()
            .reset_index()
        )
        year_month_accum.columns = [
            "date",
            "hour",
            "accumulation",
        ]  # Reuse column names

        logger.info(
            f"Prepared year-month/day heatmap data: {len(year_month_accum)} cells "
            f"from {year_month_accum['date'].min()} to {year_month_accum['date'].max()}"
        )

        return year_month_accum

    elif row_mode == "week":
        logger.info(f"Aggregating by week/day-of-week")

        # Add week and day-of-week columns
        hourly_accum["week_start"] = (
            hourly_accum["date_ts"].dt.to_period("W").dt.start_time.dt.date
        )
        hourly_accum["day_of_week"] = hourly_accum["date_ts"].dt.dayofweek  # 0=Monday

        # Aggregate by (week, day_of_week)
        weekly_accum = (
            hourly_accum.groupby(["week_start", "day_of_week"])["accumulation"]
            .sum()
            .reset_index()
        )
        weekly_accum.columns = ["date", "hour", "accumulation"]  # Reuse column names

        logger.info(
            f"Prepared weekly heatmap data: {len(weekly_accum)} weekly cells "
            f"from {weekly_accum['date'].min()} to {weekly_accum['date'].max()}"
        )

        return weekly_accum

    logger.info(
        f"Prepared accumulation heatmap data: {len(hourly_accum)} hourly cells "
        f"from {hourly_accum['date'].min()} to {hourly_accum['date'].max()}"
    )

    return hourly_accum


def create_rain_accumulation_heatmap(
    accumulation_df: pd.DataFrame,
    height: int = 600,
    max_accumulation: Optional[float] = None,
    num_days: Optional[int] = None,
    row_mode: Optional[str] = None,
    compact: bool = False,
) -> go.Figure:
    """
    Create heatmap showing rainfall accumulation with simplified grid options.

    Row modes: 'day', 'week', 'month', 'year_month', 'auto'
    - day: Daily rows with Hour of Day columns
    - week: Weekly rows with Day of Week columns
    - month: Monthly rows with Day of Month columns
    - year_month: YY-MM rows with Day of Month columns
    - auto: Automatically select based on period length

    Auto behavior:
    - ≤180 days: daily × hourly
    - 180-730 days: weekly × day-of-week
    - >730 days: year_month × day-of-month

    :param accumulation_df: DataFrame with (date, hour, accumulation)
    :param height: Chart height in pixels (auto-calculated if using default)
    :param max_accumulation: Cap color scale (auto-calculated at 90th percentile if None)
    :param num_days: Number of days in period (helps determine auto mode)
    :param row_mode: Row aggregation mode ('day', 'week', 'month', 'year_month', 'auto')
    :param compact: If True, removes legend and axis labels for overview display
    :return: Plotly figure
    """
    if accumulation_df.empty:
        logger.warning("Empty accumulation data for heatmap")
        return go.Figure()

    # Determine display mode
    if row_mode is None or row_mode == "auto":
        if num_days and num_days > 730:  # 2 years
            row_mode = "year_month"
        elif num_days and num_days > 180:
            row_mode = "week"
        else:
            row_mode = "day"

    # Setup grid based on row mode (column type is determined by row type)
    if row_mode == "month":
        # Month/Day grid: 12 rows x 31 columns
        all_months = list(range(1, 13))  # 1-12
        all_days = list(range(1, 32))  # 1-31

        full_index = pd.MultiIndex.from_product(
            [all_months, all_days], names=["date", "hour"]
        )
        x_labels = [str(d) for d in all_days]
        x_title = "Day of Month"
        y_title = "Month"
        chart_title = "Monthly Rainfall Patterns"
        height = 400  # Fixed height for 12 rows
        grid_gap = 0

    elif row_mode == "year_month":
        # YY-MM/Day grid: variable rows x 31 columns
        # Get unique year-month values from data
        year_months = sorted(accumulation_df["date"].unique())
        all_days = list(range(1, 32))  # 1-31

        full_index = pd.MultiIndex.from_product(
            [year_months, all_days], names=["date", "hour"]
        )
        x_labels = [str(d) for d in all_days]
        x_title = "Day of Month"
        y_title = "Year-Month"
        chart_title = "Monthly Timeline Rainfall Patterns"
        height = min(800, max(400, len(year_months) * 25))  # Dynamic height
        grid_gap = 0

    elif row_mode == "week":
        # Weekly mode: rows are weeks, columns are days of week
        all_weeks = pd.date_range(
            accumulation_df["date"].min(), accumulation_df["date"].max(), freq="W-MON"
        ).date
        all_days = list(range(7))  # 0=Monday to 6=Sunday

        full_index = pd.MultiIndex.from_product(
            [all_weeks, all_days], names=["date", "hour"]
        )
        x_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        x_title = "Day of Week"
        y_title = "Week Starting"
        chart_title = "Weekly Rainfall Patterns"
        height = min(800, max(400, len(all_weeks) * 15))  # Dynamic height
        grid_gap = 0

    else:  # day
        # Default hourly mode: rows are dates, columns are hours
        all_dates = pd.date_range(
            accumulation_df["date"].min(), accumulation_df["date"].max(), freq="D"
        ).date
        all_hours = list(range(24))

        full_index = pd.MultiIndex.from_product(
            [all_dates, all_hours], names=["date", "hour"]
        )
        x_labels = [f"{h:02d}:00" for h in range(24)]
        x_title = "Hour of Day"
        y_title = "Date"
        chart_title = "Hourly Rainfall Accumulation"
        height = min(1200, max(600, len(all_dates) * 10))  # Dynamic height
        grid_gap = 1  # Small gaps for daily view

    # Reindex and fill missing with NaN
    indexed = accumulation_df.set_index(["date", "hour"])
    full_data = indexed.reindex(full_index, fill_value=np.nan).reset_index()

    # Pivot: rows=dates/weeks/months, columns=hours/days
    pivot = full_data.pivot(index="date", columns="hour", values="accumulation")

    # Auto-scale colorbar at 90th percentile of non-zero values if not specified
    if max_accumulation is None:
        valid_values = pivot.values[~np.isnan(pivot.values)]
        non_zero_values = valid_values[valid_values > 0]

        if len(non_zero_values) > 0:
            max_accumulation = float(np.percentile(non_zero_values, 90))
            max_accumulation = max(max_accumulation, 0.05)  # Minimum 0.05"
        else:
            max_accumulation = 0.05

    # Create heatmap
    hover_template = (
        "<b>%{y}</b><br>"
        f"{x_title}: %{{x}}<br>"
        'Accumulation: %{z:.3f}"'
        "<extra></extra>"
    )

    # Custom colorscale: white at 0, then blue gradient for positive values
    colorscale = [
        [0.0, "white"],  # 0 maps to white
        [0.001, "#f7fbff"],  # Very light blue
        [0.25, "#deebf7"],  # Light blue
        [0.5, "#9ecae1"],  # Medium blue
        [0.75, "#4292c6"],  # Darker blue
        [1.0, "#08519c"],  # Darkest blue at 90th percentile
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=x_labels,
            y=pivot.index.astype(str),
            colorscale=colorscale,
            zmin=0,
            zmax=max_accumulation,
            colorbar=dict(title="Rain (in)"),
            hovertemplate=hover_template,
            zsmooth=False,
            hoverongaps=False,
        )
    )

    fig.update_traces(xgap=grid_gap, ygap=grid_gap)

    # Apply compact styling if requested
    if compact:
        margin = dict(l=30, r=20, t=30, b=40)
        showlegend = False
        xaxis_showticklabels = False
        colorbar_title = ""
        colorbar_dtick = 0.1
    else:
        margin = dict(l=80, r=20, t=60, b=60)
        showlegend = None  # Use default
        xaxis_showticklabels = None  # Use default
        colorbar_title = "Rain (in)"
        colorbar_dtick = None  # Use default

    fig.update_layout(
        title=chart_title,
        xaxis=dict(
            title=x_title,
            tickmode="linear",
            dtick=(
                1 if row_mode in ["week", "month", "year_month"] else 2
            ),  # Show all for aggregated views, every 2 for hourly
            type="category",
            showgrid=True,
            gridcolor="lightgrey",
            showticklabels=xaxis_showticklabels,
        ),
        yaxis=dict(
            title=y_title,
            type="category",
            autorange="reversed",  # Most recent at top
            showgrid=True,
            gridcolor="lightgrey",
        ),
        height=height,
        margin=margin,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=showlegend,
        coloraxis_colorbar=dict(
            title=colorbar_title,
            tickmode="linear",
            tick0=0,
            dtick=colorbar_dtick,
        ),
    )

    return fig


def create_year_over_year_accumulation_chart(
    yoy_data: pd.DataFrame, start_day: int = 1, end_day: int = 365
) -> go.Figure:
    """
    Create year-over-year cumulative rainfall line chart.

    Displays multiple lines, one for each year, showing cumulative rainfall
    progression through the specified day range. Enables visual comparison of
    rainfall patterns across different years for specific time periods.

    :param yoy_data: DataFrame with day_of_year, year, cumulative_rainfall columns.
    :param start_day: Start day of year displayed (for axis labeling).
    :param end_day: End day of year displayed (for axis labeling).
    :return: Plotly figure with line chart.
    """
    if yoy_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No year-over-year data available",
            height=400,
            margin=dict(l=50, r=20, t=50, b=40),
        )
        return fig

    fig = go.Figure()

    # Color palette for different years
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Get unique years and sort them
    years = sorted(yoy_data["year"].unique())

    # Add a line for each year
    for i, year in enumerate(years):
        year_data = yoy_data[yoy_data["year"] == year].copy()
        year_data = year_data.sort_values("day_of_year")

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=year_data["day_of_year"],
                y=year_data["cumulative_rainfall"],
                mode="lines",
                line=dict(color=color, width=2),
                name=str(year),
                hovertemplate=(
                    "Day %{x}<br>"
                    "Year: " + str(year) + "<br>"
                    'Cumulative: %{y:.2f}"<extra></extra>'
                ),
            )
        )

    # Update layout
    fig.update_layout(
        title="Year-over-Year Rainfall Accumulation",
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis_title="Day of Year",
        yaxis_title="Cumulative Rainfall (inches)",
        showlegend=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgray",
        range=[start_day, end_day],
        tickmode="array",
        tickvals=[1, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
        ticktext=[
            "Jan 1",
            "Feb 1",
            "Mar 1",
            "Apr 1",
            "May 1",
            "Jun 1",
            "Jul 1",
            "Aug 1",
            "Sep 1",
            "Oct 1",
            "Nov 1",
            "Dec 1",
            "Dec 31",
        ],
    )

    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")

    return fig


def extract_event_data(
    archive_df: pd.DataFrame, selected_event: pd.Series
) -> pd.DataFrame:
    """
    Extract and filter event data from weather archive.

    :param archive_df: Full weather archive (unsorted)
    :param selected_event: Event row from catalog DataFrame
    :return: Filtered event data sorted by timestamp
    """
    archive_df = archive_df.copy()
    archive_df["timestamp"] = pd.to_datetime(archive_df["dateutc"], unit="ms", utc=True)
    start_time = pd.to_datetime(selected_event["start_time"], utc=True)
    end_time = pd.to_datetime(selected_event["end_time"], utc=True)

    mask = (archive_df["timestamp"] >= start_time) & (
        archive_df["timestamp"] <= end_time
    )
    event_data = archive_df[mask].sort_values("timestamp").copy()

    return event_data


def format_event_header(selected_event: pd.Series) -> str:
    """
    Create formatted event header string with duration, rainfall, and quality info.

    :param selected_event: Event row from catalog DataFrame
    :return: Formatted header string
    """
    start_pst = pd.to_datetime(selected_event["start_time"]).tz_convert(
        "America/Los_Angeles"
    )
    end_pst = pd.to_datetime(selected_event["end_time"]).tz_convert(
        "America/Los_Angeles"
    )

    start_str = start_pst.strftime("%b %-d")
    if end_pst.date() != start_pst.date():
        end_str = end_pst.strftime("%-d, %Y")
    else:
        end_str = end_pst.strftime("%-I:%M %p").lower().lstrip("0")

    duration_h = selected_event["duration_minutes"] / 60
    total_rain = selected_event["total_rainfall"]
    peak_rate = selected_event["max_hourly_rate"]
    quality = selected_event["quality_rating"].title()

    flag_str = ""
    if selected_event.get("flags"):
        flags = (
            selected_event["flags"]
            if isinstance(selected_event["flags"], dict)
            else json.loads(selected_event["flags"])
        )
        if flags.get("ongoing"):
            flag_str = " • 🔄"
        elif flags.get("interrupted"):
            flag_str = " • ⚠️"

    if duration_h >= 48:
        duration_str = f"{duration_h/24:.1f}d"
    else:
        duration_str = f"{duration_h:.1f}h"

    header = f'{start_str}-{end_str} • {duration_str} • {total_rain:.3f}" • {peak_rate:.3f} in/hr • {quality}{flag_str}'
    return header


def render_event_visualization_core(
    selected_event: pd.Series, archive_df: pd.DataFrame
) -> tuple:
    """
    Core event visualization logic - data processing, header formatting, and chart creation.

    :param selected_event: Event row from catalog DataFrame
    :param archive_df: Full weather archive (unsorted)
    :return: Tuple of (event_data, header, acc_fig, rate_fig)
    """
    # Extract and filter event data
    event_data = extract_event_data(archive_df, selected_event)

    if len(event_data) == 0:
        return None, None, None, None

    # Create formatted header
    header = format_event_header(selected_event)

    # Create event info for charts
    event_info = {
        "total_rainfall": selected_event["total_rainfall"],
        "duration_minutes": selected_event["duration_minutes"],
        "start_time": pd.to_datetime(selected_event["start_time"], utc=True),
        "end_time": pd.to_datetime(selected_event["end_time"], utc=True),
    }

    # Create charts
    acc_fig = create_event_accumulation_chart(event_data, event_info)
    rate_fig = create_event_rate_chart(event_data)

    return event_data, header, acc_fig, rate_fig

    fig = go.Figure()

    # Color palette for different years
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Get unique years and sort them
    years = sorted(yoy_data["year"].unique())

    # Add a line for each year
    for i, year in enumerate(years):
        year_data = yoy_data[yoy_data["year"] == year].copy()
        year_data = year_data.sort_values("day_of_year")

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=year_data["day_of_year"],
                y=year_data["cumulative_rainfall"],
                mode="lines",
                line=dict(color=color, width=2),
                name=str(year),
                hovertemplate=(
                    "Day %{x}<br>"
                    "Year: " + str(year) + "<br>"
                    'Cumulative: %{y:.2f}"<extra></extra>'
                ),
            )
        )

    # Update layout
    fig.update_layout(
        title="Year-over-Year Rainfall Accumulation",
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis_title="Day of Year",
        yaxis_title="Cumulative Rainfall (inches)",
        showlegend=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgray",
        range=[start_day, end_day],
        tickmode="array",
        tickvals=[1, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
        ticktext=[
            "Jan 1",
            "Feb 1",
            "Mar 1",
            "Apr 1",
            "May 1",
            "Jun 1",
            "Jul 1",
            "Aug 1",
            "Sep 1",
            "Oct 1",
            "Nov 1",
            "Dec 1",
            "Dec 31",
        ],
    )

    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")

    return fig


def _create_event_headline(current_event):
    """
    Create formatted headline for rain event display.

    :param current_event: Event dictionary from catalog
    :return: Formatted headline string
    """
    start_time = pd.to_datetime(current_event["start_time"], utc=True)
    end_time = pd.to_datetime(current_event["end_time"], utc=True)

    # Convert to Pacific time
    start_pst = start_time.tz_convert("America/Los_Angeles")
    end_pst = end_time.tz_convert("America/Los_Angeles")

    # Format date strings
    start_str = start_pst.strftime("%b %-d")
    if end_pst.date() != start_pst.date():
        end_str = end_pst.strftime("%-d, %Y")
    else:
        end_str = end_pst.strftime("%-I:%M %p").lower().lstrip("0")

    # Duration formatting
    duration_h = current_event["duration_minutes"] / 60
    if duration_h >= 48:
        duration_str = f"{duration_h/24:.1f}d"
    else:
        duration_str = f"{duration_h:.1f}h"

    # Extract values
    total_rain = current_event["total_rainfall"]
    peak_rate = current_event["max_hourly_rate"]

    # Create headline without quality and flags
    headline = f'Rain Event: {start_str}-{end_str} • {duration_str} • {total_rain:.3f}" • {peak_rate:.3f} in/hr'

    return headline


def create_event_detail_charts(history_df, current_event, event_key="event"):
    """
    Create both accumulation and rate charts for rain event detail.

    :param history_df: Full weather history DataFrame
    :param current_event: Current event dictionary from catalog
    :param event_key: Key prefix for chart uniqueness
    :return: Tuple of (accumulation_fig, rate_fig)
    """
    # Extract event data from history
    history_df = history_df.copy()
    history_df["timestamp"] = pd.to_datetime(history_df["dateutc"], unit="ms", utc=True)

    start_time = pd.to_datetime(current_event["start_time"], utc=True)
    end_time = pd.to_datetime(current_event["end_time"], utc=True)

    mask = (history_df["timestamp"] >= start_time) & (
        history_df["timestamp"] <= end_time
    )
    event_data = history_df[mask].sort_values("timestamp").copy()

    if len(event_data) == 0:
        return None, None, None

    # Create event info for accumulation chart
    event_info = {
        "total_rainfall": current_event["total_rainfall"],
        "duration_minutes": current_event["duration_minutes"],
        "start_time": start_time,
        "end_time": end_time,
    }

    # Create charts using existing functions
    acc_fig = create_event_accumulation_chart(event_data, event_info)
    rate_fig = create_event_rate_chart(event_data)

    # Create headline for overview display
    headline = _create_event_headline(current_event)

    return acc_fig, rate_fig, headline


def create_event_histogram(events_df: pd.DataFrame, selected_range: tuple) -> go.Figure:
    """
    Create histogram showing event count over time with selected range highlighted.

    :param events_df: DataFrame with event data
    :param selected_range: Tuple of (start_date, end_date) for highlighting
    :return: Plotly figure
    """
    events_by_week = (
        events_df.set_index("start_time").resample("W").size().reset_index(name="count")
    )

    fig = go.Figure()

    start_date, end_date = selected_range
    start_ts = (
        pd.Timestamp(start_date).tz_localize("America/Los_Angeles").tz_convert("UTC")
    )
    end_ts = (
        (pd.Timestamp(end_date) + pd.Timedelta(days=1))
        .tz_localize("America/Los_Angeles")
        .tz_convert("UTC")
    )

    def get_bar_color(date):
        date_ts = (
            pd.Timestamp(date).tz_localize("UTC")
            if pd.Timestamp(date).tz is None
            else pd.Timestamp(date)
        )
        return "steelblue" if start_ts <= date_ts < end_ts else "lightgray"

    colors = [get_bar_color(date) for date in events_by_week["start_time"]]

    fig.add_trace(
        go.Bar(
            x=events_by_week["start_time"],
            y=events_by_week["count"],
            marker_color=colors,
            hovertemplate="<b>Week of %{x|%Y-%m-%d}</b><br>Events: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Events by Week",
        xaxis_title="",
        yaxis_title="Event Count",
        height=200,
        margin=dict(l=40, r=20, t=40, b=20),
        showlegend=False,
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    return fig

