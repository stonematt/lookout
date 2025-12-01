"""
solar_viz.py
Solar radiation-specific visualization functions for Lookout weather station dashboard.

This module contains specialized visualization functions for solar analysis,
including radiation time series, energy accumulation charts, seasonal comparisons,
hourly patterns, and solar heatmaps. Functions are extracted from CLI tools to
enable Streamlit usage with consistent styling.

Functions:
- create_solar_time_series_chart: Time series of solar radiation intensity
- create_daily_energy_chart: Daily energy accumulation over time
- create_seasonal_comparison_chart: Seasonal energy comparison
- create_hourly_pattern_chart: Average radiation by hour of day
- create_solar_heatmap: Solar radiation heatmap by hour/month
- prepare_solar_heatmap_data: Data preparation for solar heatmaps
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any

from lookout.core.chart_config import (
    get_standard_colors,
    apply_time_series_layout,
    apply_standard_axes,
    create_standard_annotation,
    apply_heatmap_layout,
)
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def create_solar_time_series_chart(
    df: pd.DataFrame,
    height: int = 450,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create time series chart showing solar radiation intensity over time.

    :param df: DataFrame with datetime, solarradiation, and daylight_period columns
    :param height: Chart height in pixels
    :param title: Chart title (auto-generated if None)
    :return: Plotly figure
    """
    if df.empty:
        logger.warning("Empty data for solar time series chart")
        return go.Figure()

    # Filter for daytime data with valid solar readings
    daytime_df = df[(df['daylight_period'] == 'day') &
                   (df['solarradiation'].notna())].copy()

    if daytime_df.empty:
        logger.warning("No valid daytime solar data for time series chart")
        return go.Figure()

    # Ensure datetime column is properly typed
    daytime_df = daytime_df.copy()
    daytime_df['datetime'] = pd.to_datetime(daytime_df['datetime'], utc=True)
    daytime_df['datetime_local'] = daytime_df['datetime'].dt.tz_convert('America/Los_Angeles')

    # Get standard colors
    colors = get_standard_colors()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daytime_df['datetime_local'],
            y=daytime_df['solarradiation'],
            mode='lines',
            line=dict(color=colors['solar_line'], width=2),
            fill='tozeroy',
            fillcolor=colors['solar_fill'],
            hovertemplate='%{x|%b %d %I:%M %p}<br>%{y:.1f} W/m²<extra></extra>',
            name='Solar Radiation',
        )
    )

    # Apply time series layout
    chart_title = title or 'Solar Radiation Intensity'
    fig = apply_time_series_layout(
        fig,
        height=height,
        showlegend=False,
        title=chart_title,
        hovermode='x unified'
    )

    # Apply standard axes
    fig = apply_standard_axes(
        fig,
        xaxis_title='',
        yaxis_title='Solar Radiation (W/m²)',
        showgrid_x=False,
        showgrid_y=True,
    )

    # Add peak radiation annotation
    peak_radiation = daytime_df['solarradiation'].max()
    peak_annotation = create_standard_annotation(
        text=f"Peak: {peak_radiation:.0f} W/m²",
        position="top_right"
    )
    fig.add_annotation(peak_annotation)

    return fig


def create_daily_energy_chart(
    daily_energy: pd.Series,
    height: int = 450,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create time series chart showing daily solar energy accumulation.

    :param daily_energy: Series with daily kWh/m² values indexed by date
    :param height: Chart height in pixels
    :param title: Chart title (auto-generated if None)
    :return: Plotly figure
    """
    if daily_energy.empty:
        logger.warning("Empty daily energy data for chart")
        return go.Figure()

    # Get standard colors
    colors = get_standard_colors()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily_energy.index,
            y=daily_energy.values,
            mode='lines+markers',
            line=dict(color=colors['solar_line'], width=2),
            marker=dict(
                color=colors['solar_medium'],
                size=6,
                line=dict(width=1, color=colors['solar_high'])
            ),
            hovertemplate='%{x|%b %d, %Y}<br>%{y:.2f} kWh/m²<extra></extra>',
            name='Daily Energy',
        )
    )

    # Apply time series layout
    chart_title = title or 'Daily Solar Energy Accumulation'
    fig = apply_time_series_layout(
        fig,
        height=height,
        showlegend=False,
        title=chart_title,
        hovermode='x unified'
    )

    # Apply standard axes
    fig = apply_standard_axes(
        fig,
        xaxis_title='',
        yaxis_title='Daily Energy (kWh/m²)',
        showgrid_x=False,
        showgrid_y=True,
    )

    # Add average energy annotation
    avg_energy = daily_energy.mean()
    avg_annotation = create_standard_annotation(
        text=f"Avg: {avg_energy:.2f} kWh/m²/day",
        position="top_right"
    )
    fig.add_annotation(avg_annotation)

    return fig


def create_seasonal_comparison_chart(
    seasonal_data: Dict[str, Dict[str, float]],
    height: int = 450,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create bar chart comparing seasonal solar energy statistics.

    :param seasonal_data: Dictionary with seasonal statistics from get_seasonal_breakdown()
    :param height: Chart height in pixels
    :param title: Chart title (auto-generated if None)
    :return: Plotly figure
    """
    if not seasonal_data:
        logger.warning("Empty seasonal data for comparison chart")
        return go.Figure()

    # Extract data for plotting
    seasons = []
    means = []
    maxes = []

    season_order = ['winter', 'spring', 'summer', 'fall']
    season_labels = ['Winter', 'Spring', 'Summer', 'Fall']

    for season_key, label in zip(season_order, season_labels):
        if season_key in seasonal_data:
            seasons.append(label)
            means.append(seasonal_data[season_key]['mean_kwh_per_m2'])
            maxes.append(seasonal_data[season_key]['max_kwh_per_m2'])

    if not seasons:
        logger.warning("No seasonal data available for chart")
        return go.Figure()

    # Get standard colors
    colors = get_standard_colors()

    fig = go.Figure()

    # Add mean bars
    fig.add_trace(
        go.Bar(
            x=seasons,
            y=means,
            name='Average Daily Energy',
            marker_color=colors['solar_medium'],
            hovertemplate='%{x}<br>Average: %{y:.2f} kWh/m²/day<extra></extra>',
        )
    )

    # Add max markers
    fig.add_trace(
        go.Scatter(
            x=seasons,
            y=maxes,
            mode='markers',
            name='Peak Daily Energy',
            marker=dict(
                color=colors['solar_high'],
                size=10,
                symbol='diamond',
                line=dict(width=2, color=colors['solar_line'])
            ),
            hovertemplate='%{x}<br>Peak: %{y:.2f} kWh/m²/day<extra></extra>',
        )
    )

    # Apply time series layout (works for bar charts too)
    chart_title = title or 'Seasonal Solar Energy Comparison'
    fig = apply_time_series_layout(
        fig,
        height=height,
        showlegend=True,
        title=chart_title,
        hovermode='x unified'
    )

    # Apply standard axes
    fig = apply_standard_axes(
        fig,
        xaxis_title='Season',
        yaxis_title='Energy (kWh/m²/day)',
        showgrid_x=False,
        showgrid_y=True,
        type_x='category',
    )

    # Add seasonal variation annotation if available
    if 'seasonal_variation_ratio' in seasonal_data:
        variation = seasonal_data['seasonal_variation_ratio']
        variation_annotation = create_standard_annotation(
            text=f"Seasonal Variation: {variation:.1f}x",
            position="top_left"
        )
        fig.add_annotation(variation_annotation)

    return fig


def create_hourly_pattern_chart(
    hourly_patterns: pd.Series,
    height: int = 450,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create line chart showing average solar radiation by hour of day.

    :param hourly_patterns: Series with average radiation by hour (0-23)
    :param height: Chart height in pixels
    :param title: Chart title (auto-generated if None)
    :return: Plotly figure
    """
    if hourly_patterns.empty:
        logger.warning("Empty hourly pattern data for chart")
        return go.Figure()

    # Get standard colors
    colors = get_standard_colors()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hourly_patterns.index,
            y=hourly_patterns.values,
            mode='lines+markers',
            line=dict(color=colors['solar_line'], width=3),
            marker=dict(
                color=colors['solar_medium'],
                size=8,
                line=dict(width=2, color=colors['solar_high'])
            ),
            hovertemplate='Hour %{x}:00<br>%{y:.1f} W/m² average<extra></extra>',
            name='Average Radiation',
        )
    )

    # Apply time series layout
    chart_title = title or 'Average Solar Radiation by Hour'
    fig = apply_time_series_layout(
        fig,
        height=height,
        showlegend=False,
        title=chart_title,
        hovermode='x unified'
    )

    # Apply standard axes
    fig = apply_standard_axes(
        fig,
        xaxis_title='Hour of Day (Pacific Time)',
        yaxis_title='Solar Radiation (W/m²)',
        showgrid_x=True,
        showgrid_y=True,
        type_x='category',
    )

    # Add peak hour annotation
    peak_hour = hourly_patterns.idxmax()
    peak_value = hourly_patterns.max()
    peak_annotation = create_standard_annotation(
        text=f"Peak: {peak_value:.1f} W/m² at {peak_hour}:00",
        position="top_right"
    )
    fig.add_annotation(peak_annotation)

    return fig


def prepare_solar_heatmap_data(
    df: pd.DataFrame,
    row_mode: str = 'day'
) -> pd.DataFrame:
    """
    Prepare solar radiation data for heatmap visualization.

    :param df: DataFrame with datetime, solarradiation, and daylight_period columns
    :param row_mode: Row aggregation mode ('day', 'week', 'month')
    :return: DataFrame with (date, hour, radiation) columns ready for heatmap
    """
    if df.empty:
        return pd.DataFrame()

    # Filter for daytime data with valid solar readings
    daytime_df = df[(df['daylight_period'] == 'day') &
                   (df['solarradiation'].notna())].copy()

    if daytime_df.empty:
        return pd.DataFrame()

    # Convert to local time
    daytime_df = daytime_df.copy()
    daytime_df['datetime'] = pd.to_datetime(daytime_df['datetime'], utc=True)
    daytime_df['datetime_local'] = daytime_df['datetime'].dt.tz_convert('America/Los_Angeles')
    daytime_df['date'] = daytime_df['datetime_local'].dt.date
    daytime_df['hour'] = daytime_df['datetime_local'].dt.hour

    # Aggregate based on row mode
    if row_mode == 'week':
        daytime_df['week_start'] = daytime_df['datetime_local'].dt.to_period('W').dt.start_time.dt.date
        heatmap_df = daytime_df.groupby(['week_start', daytime_df['datetime_local'].dt.dayofweek])['solarradiation'].mean().reset_index()
        heatmap_df.columns = ['date', 'hour', 'radiation']
    elif row_mode == 'month':
        daytime_df['month'] = daytime_df['datetime_local'].dt.month
        heatmap_df = daytime_df.groupby(['month', 'hour'])['solarradiation'].mean().reset_index()
        heatmap_df.columns = ['date', 'hour', 'radiation']
    else:  # day
        heatmap_df = daytime_df.groupby(['date', 'hour'])['solarradiation'].mean().reset_index()
        heatmap_df.columns = ['date', 'hour', 'radiation']

    return heatmap_df


def create_solar_heatmap(
    heatmap_df: pd.DataFrame,
    height: int = 600,
    row_mode: str = 'day',
    title: Optional[str] = None,
) -> go.Figure:
    """
    Create heatmap showing solar radiation patterns by hour and date/week/month.

    :param heatmap_df: DataFrame from prepare_solar_heatmap_data()
    :param height: Chart height in pixels
    :param row_mode: Row mode ('day', 'week', 'month')
    :param title: Chart title (auto-generated if None)
    :return: Plotly figure
    """
    if heatmap_df.empty:
        logger.warning("Empty heatmap data")
        return go.Figure()

    # Create pivot table for heatmap
    heatmap_data = heatmap_df.pivot_table(
        values='radiation',
        index='date',
        columns='hour',
        aggfunc='mean',
        fill_value=0
    )

    # Get standard colors
    colors = get_standard_colors()

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[
                [0.0, colors['solar_low']],
                [0.5, colors['solar_medium']],
                [1.0, colors['solar_high']]
            ],
            hovertemplate='%{y} at %{x}:00<br>%{z:.1f} W/m²<extra></extra>',
            name='Solar Radiation',
        )
    )

    # Determine axis titles based on row mode
    if row_mode == 'week':
        y_title = 'Week Starting'
        x_title = 'Day of Week'
        chart_title = title or 'Weekly Solar Radiation Patterns'
    elif row_mode == 'month':
        y_title = 'Month'
        x_title = 'Hour of Day'
        chart_title = title or 'Monthly Solar Radiation Patterns'
    else:
        y_title = 'Date'
        x_title = 'Hour of Day (Pacific Time)'
        chart_title = title or 'Daily Solar Radiation Patterns'

    # Apply heatmap layout
    fig = apply_heatmap_layout(
        fig,
        title=chart_title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        colorbar_title='Solar Radiation (W/m²)',
        compact=False,
    )

    return fig