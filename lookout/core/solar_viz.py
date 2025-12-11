"""
Solar Energy Visualizations
Plotly-based charts and heatmaps for solar production data.
"""

import pandas as pd
import plotly.graph_objects as go
from lookout.core.solar_energy_periods import aggregate_to_daily, aggregate_to_hourly
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


# Color constants for all solar visualizations
SOLAR_COLORSCALE = [
    [0.0, "#FFF9E6"],  # Dawn/dusk - pale yellow
    [0.3, "#FFE680"],  # Morning - light yellow
    [0.6, "#FFB732"],  # Midday - golden orange
    [1.0, "#FF8C00"],  # Peak sun - deep orange
]

SOLAR_BAR_COLOR = "#FFB732"  # Golden orange for bar charts


def create_month_day_heatmap(periods_df: pd.DataFrame) -> go.Figure:
    """
    Create month/day heatmap showing daily energy production.

    Design (Sunlight-inspired):
    - X-axis: Days 1-31
    - Y-axis: Months (reverse chrono, newest top)
    - Color: SOLAR_COLORSCALE (yellow-to-orange gradient)
    - Cells: Show kWh on hover (2 decimals)
    - Zero energy days (0.0 kWh): Show as pale yellow (bottom of colorscale) - these are valid cloudy days
    - Missing days (NaN - no data at all): Light gray (#F0F0F0)
    - Click: Enable click events for interactivity
    - Height: ~500px
    """
    import numpy as np

    logger.debug("Creating month/day solar energy heatmap")

    # Aggregate to daily totals (keeps ALL days including zeros)
    daily_df = aggregate_to_daily(periods_df)

    if daily_df.empty:
        logger.info("No daily solar data available for heatmap")
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No Solar Data Available",
            height=500
        )
        return fig

    # Extract month and day for pivot table
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["month"] = daily_df["date"].dt.strftime("%Y-%m")  # YYYY-MM format
    daily_df["day"] = daily_df["date"].dt.day

    # Create pivot table: months x days (1-31)
    pivot_df = daily_df.pivot_table(
        values="daily_kwh",
        index="month",
        columns="day",
        aggfunc="sum"  # Should be single value per month/day
    )

    # Ensure all days 1-31 are present as columns (fill missing with NaN)
    all_days = list(range(1, 32))
    for day in all_days:
        if day not in pivot_df.columns:
            pivot_df[day] = np.nan

    # Sort columns by day
    pivot_df = pivot_df[all_days]

    # Sort months chronologically (newest first for reverse y-axis)
    pivot_df = pivot_df.sort_index(ascending=False)

    # Prepare data for heatmap
    z_values = pivot_df.values
    x_labels = [str(i) for i in range(1, 32)]  # Days 1-31
    y_labels = pivot_df.index.tolist()

    # Create custom hover text that handles missing data properly
    hover_text = []
    for i in range(len(y_labels)):
        row_text = []
        for j in range(len(x_labels)):
            val = z_values[i, j]
            if np.isnan(val):
                row_text.append(f"<b>{x_labels[j]}/{y_labels[i]}</b><br>No data")
            else:
                row_text.append(f"<b>{x_labels[j]}/{y_labels[i]}</b><br>{val:.2f} kWh")
        hover_text.append(row_text)

    # Create heatmap with NaN values (Plotly handles NaN as transparent/missing)
    fig = go.Figure(data=go.Heatmap(
        z=z_values,  # Keep NaN values as-is
        x=x_labels,
        y=y_labels,
        colorscale=SOLAR_COLORSCALE,
        zmin=0,  # Force scale to start at 0
        zmax=z_values[~np.isnan(z_values)].max() if np.any(~np.isnan(z_values)) else 1,
        text=hover_text,
        hoverinfo="text",
        showscale=True,
        colorbar=dict(
            title="kWh/day",
            ticksuffix=" kWh"
        )
    ))

    # Update layout
    fig.update_layout(
        height=500,
        xaxis_title="Day of Month",
        yaxis_title="Month",
        yaxis_autorange="reversed"  # Newest months at top
    )

    logger.info(f"Created solar energy heatmap with {len(y_labels)} months × {len(x_labels)} days")
    return fig


def create_day_column_chart(periods_df: pd.DataFrame, selected_date: str) -> go.Figure:
    """
    Create hourly column chart for a specific day.

    Design:
    - X-axis: Hours (0-23)
    - Y-axis: Energy (kWh)
    - Bars: SOLAR_BAR_COLOR (#FFB732 - golden orange)
    - Height: ~400px
    """
    logger.debug(f"Creating hourly column chart for date: {selected_date}")

    # Aggregate to hourly totals for the selected date
    hourly_df = aggregate_to_hourly(periods_df, selected_date)

    if hourly_df.empty:
        logger.info(f"No hourly solar data available for {selected_date}")
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title=f"No Solar Data Available - {selected_date}",
            height=400
        )
        return fig

    # Create bar chart
    fig = go.Figure(data=go.Bar(
        x=hourly_df["hour"],
        y=hourly_df["hourly_kwh"],
        marker_color=SOLAR_BAR_COLOR,
        hovertemplate="<b>%{x}:00</b><br>%{y:.2f} kWh<extra></extra>"
    ))

    # Update layout
    fig.update_layout(
        title=f"Hourly Production - {selected_date}",
        xaxis_title="Hour of Day",
        yaxis_title="Energy (kWh)",
        height=400,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, 23.5]  # Show all hours 0-23
        )
    )

    logger.info(f"Created hourly column chart for {selected_date} with {len(hourly_df)} hours")
    return fig


def create_day_15min_heatmap(periods_df: pd.DataFrame) -> go.Figure:
    """
    Create day/15min heatmap showing granular production patterns.

    TODO: Implement in Phase 2 - Epic 2.3
    """
    raise NotImplementedError("Phase 2 - Epic 2.3 in progress")


def create_15min_bar_chart(periods_df: pd.DataFrame, selected_date: str) -> go.Figure:
    """
    Create 15-minute bar chart for a specific day.

    TODO: Implement in Phase 2 - Epic 2.4
    """
    raise NotImplementedError("Phase 2 - Epic 2.4 in progress")
